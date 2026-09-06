"""Independent summary generator for the DaT Stage 2 grid.

This intentionally does not import or modify ``runs/summary/generate_summary.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = ROOT / "runs" / "dat_parkinsons" / "resnet18_3d"
SUMMARY_DIR = ROOT / "runs" / "dat_parkinsons" / "summary"


def _read_runs() -> list[tuple[dict, pd.DataFrame, Path]]:
    rows = []
    for metrics_path in sorted(RUN_ROOT.rglob("metrics.csv")):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(metrics_path)
        required = {"epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy", "val_log_loss", "val_auroc", "val_brier_score"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{metrics_path} is missing required columns: {sorted(missing)}")
        rows.append((config, frame, run_dir))
    return rows


def _oof_logloss(config: dict, calibration: dict) -> tuple[float, float]:
    """Return raw and fixed-calibration OOF log loss when a run saved OOF arrays."""
    artifact = config.get("oof_artifact")
    if not artifact or not Path(artifact).is_file():
        return float("nan"), float("nan")
    payload = np.load(artifact)
    logits = np.asarray(payload["logits"], dtype=np.float64)
    targets = np.asarray(payload["targets"], dtype=np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    raw = np.clip(probabilities[np.arange(targets.size), targets], 1e-7, 1.0)
    temperature = float(calibration.get("temperature", 1.0)) if str(calibration.get("method", "raw")) == "temperature" else 1.0
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Summary calibration temperature must be finite and positive.")
    calibrated_logits = logits / temperature
    shifted = calibrated_logits - calibrated_logits.max(axis=1, keepdims=True)
    calibrated_probabilities = np.exp(shifted)
    calibrated_probabilities /= calibrated_probabilities.sum(axis=1, keepdims=True)
    calibrated = np.clip(calibrated_probabilities[np.arange(targets.size), targets], 1e-7, 1.0)
    return float(-np.log(raw).mean()), float(-np.log(calibrated).mean())


def _run_summary(config: dict, frame: pd.DataFrame, run_dir: Path, calibration: dict) -> dict:
    best_index = frame["val_log_loss"].astype(float).idxmin()
    best = frame.loc[best_index]
    tail = frame.tail(min(20, len(frame)))
    running_best = frame["val_log_loss"].cummin()
    oof_raw, oof_calibrated = _oof_logloss(config, calibration)
    return {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "fold": int(config.get("fold", -1)),
        "condition": str(config.get("condition", config.get("cutout_mode", "none"))),
        "M": int(config.get("cutout_m", 0)),
        "fraction": float(config.get("cutout_fraction", 0.0) or 0.0),
        "epochs_completed": int(frame["epoch"].max()),
        "minimum_validation_log_loss": float(best["val_log_loss"]),
        "epoch_of_minimum_validation_log_loss": int(best["epoch"]),
        "final_validation_log_loss": float(frame.iloc[-1]["val_log_loss"]),
        "final20_logloss_mean": float(tail["val_log_loss"].mean()),
        "final20_logloss_std": float(tail["val_log_loss"].std(ddof=1)) if len(tail) > 1 else 0.0,
        "best_validation_accuracy": float(frame["val_accuracy"].max()),
        "final_validation_accuracy": float(frame.iloc[-1]["val_accuracy"]),
        "auroc": float(best["val_auroc"]),
        "brier_score": float(best["val_brier_score"]),
        "oof_raw_log_loss": oof_raw,
        "oof_calibrated_log_loss": oof_calibrated,
        "train_validation_accuracy_gap": float(frame.iloc[-1]["train_accuracy"] - frame.iloc[-1]["val_accuracy"]),
        "maximum_validation_logloss_drawdown": float((frame["val_log_loss"] - running_best).max()),
    }


def _aggregate(per_run: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["condition", "M", "fraction"]
    metric_columns = [column for column in per_run.columns if column not in group_columns + ["run_dir", "fold"] and pd.api.types.is_numeric_dtype(per_run[column])]
    rows = []
    for keys, group in per_run.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean()) if values.size else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            row[f"{metric}_sem"] = float(row[f"{metric}_std"] / math.sqrt(values.size)) if values.size else float("nan")
            row[f"{metric}_ci95_low"] = float(row[f"{metric}_mean"] - 1.96 * row[f"{metric}_sem"]) if values.size else float("nan")
            row[f"{metric}_ci95_high"] = float(row[f"{metric}_mean"] + 1.96 * row[f"{metric}_sem"]) if values.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _paired(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fraction in sorted(per_run["fraction"].unique()):
        for m_value in sorted(per_run["M"].unique()):
            subset = per_run[(per_run["fraction"] == fraction) & (per_run["M"] == m_value)]
            by_key = {(int(row.fold), str(row.condition)): row for row in subset.itertuples()}
            random_rows = {fold: row for (fold, condition), row in by_key.items() if condition == "random"}
            baseline_rows = {
                int(row.fold): row
                for row in per_run[(per_run["fraction"] == fraction) & (per_run["condition"] == "none")].itertuples()
            }
            for treatment in ("cam_low", "cam_high"):
                for fold, random_row in random_rows.items():
                    candidate = by_key.get((fold, treatment))
                    if candidate is None:
                        continue
                    rows.append({
                        "comparison": f"{treatment}_versus_random",
                        "fold": fold, "M": m_value, "fraction": fraction,
                        "accuracy_difference_treatment_minus_random": candidate.final_validation_accuracy - random_row.final_validation_accuracy,
                        "logloss_improvement": random_row.final_validation_log_loss - candidate.final_validation_log_loss,
                    })
            for fold, random_row in random_rows.items():
                baseline = baseline_rows.get(fold)
                if baseline is not None:
                    rows.append({
                        "comparison": "random_versus_none", "fold": fold, "M": m_value, "fraction": fraction,
                        "accuracy_difference_treatment_minus_random": random_row.final_validation_accuracy - baseline.final_validation_accuracy,
                        "logloss_improvement": baseline.final_validation_log_loss - random_row.final_validation_log_loss,
                    })
    for fraction in sorted(per_run["fraction"].unique()):
        for condition in sorted(set(per_run["condition"]) - {"none"}):
            subset = per_run[(per_run["fraction"] == fraction) & (per_run["condition"] == condition)]
            for fold in sorted(subset["fold"].unique()):
                rows_for_fold = subset[subset["fold"] == fold].set_index("M")
                if 4 in rows_for_fold.index and 8 in rows_for_fold.index:
                    m4, m8 = rows_for_fold.loc[4], rows_for_fold.loc[8]
                    rows.append({
                        "comparison": f"{condition}_M8_versus_M4", "fold": fold, "M": 8, "fraction": fraction,
                        "accuracy_difference_treatment_minus_random": m8.final_validation_accuracy - m4.final_validation_accuracy,
                        "logloss_improvement": m4.final_validation_log_loss - m8.final_validation_log_loss,
                    })
    return pd.DataFrame(rows)


def _save_plot(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plots(per_run: pd.DataFrame, aggregate: pd.DataFrame, paired: pd.DataFrame) -> list[Path]:
    plot_dir = SUMMARY_DIR / "plots"
    outputs = []
    for metric, title, filename in [
        ("minimum_validation_log_loss_mean", "Mean validation log loss by fraction", "mean_log_loss_by_fraction.png"),
        ("oof_calibrated_log_loss_mean", "Mean calibrated OOF log loss by fraction", "calibrated_log_loss_by_fraction.png"),
        ("final_validation_accuracy_mean", "Validation accuracy by fraction", "accuracy_by_fraction.png"),
        ("auroc_mean", "AUROC by fraction", "auroc_by_fraction.png"),
    ]:
        if metric not in aggregate.columns or not np.isfinite(aggregate[metric].to_numpy(dtype=float)).any():
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for condition, group in aggregate.groupby("condition"):
            group = group.sort_values("fraction")
            ax.plot(group["fraction"], group[metric], "o-", label=condition)
        ax.set_title(title); ax.set_xlabel("Cutout fraction"); ax.set_ylabel(metric.replace("_mean", "")); ax.grid(alpha=0.25); ax.legend()
        path = plot_dir / filename; _save_plot(fig, path); outputs.append(path)

    if not paired.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        effect = paired[paired["comparison"].str.contains("cam_.*_versus_random", regex=True)]
        if not effect.empty:
            effect.boxplot(column="logloss_improvement", by="comparison", ax=ax)
        ax.set_title("Paired CAM log-loss improvement versus random"); ax.set_xlabel(""); ax.set_ylabel("random log loss - treatment log loss")
        plt.suptitle(""); path = plot_dir / "paired_cam_effects_vs_random.png"; _save_plot(fig, path); outputs.append(path)

        effect = paired[paired["comparison"].str.contains("M8_versus_M4")]
        if not effect.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5)); effect.boxplot(column="logloss_improvement", by="comparison", ax=ax); ax.set_title("M8 versus M4 log-loss improvement"); ax.set_xlabel(""); ax.set_ylabel("M4 log loss - M8 log loss"); plt.suptitle(""); path = plot_dir / "m8_minus_m4.png"; _save_plot(fig, path); outputs.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (condition, m_value, fraction), group in per_run.groupby(["condition", "M", "fraction"]):
        for _, row in group.iterrows():
            metrics_path = ROOT / row["run_dir"] / "metrics.csv"
            frame = pd.read_csv(metrics_path)
            ax.plot(frame["epoch"], frame["val_log_loss"], alpha=0.35, label=f"{condition} M{m_value} f{fraction:.2f}")
    ax.set_title("DaT learning curves"); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation log loss"); ax.grid(alpha=0.25)
    path = plot_dir / "learning_curves.png"; _save_plot(fig, path); outputs.append(path)

    for value_column, title, filename in [
        ("minimum_validation_log_loss_mean", "Mean log loss heatmap", "mean_heatmap.png"),
        ("minimum_validation_log_loss_std", "Log-loss variability heatmap", "variability_heatmap.png"),
    ]:
        pivot = aggregate.pivot_table(index="condition", columns="fraction", values=value_column, aggfunc="mean")
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
            ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
            ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([f"{float(v):.2f}" for v in pivot.columns])
            ax.set_xlabel("Cutout fraction"); ax.set_title(title); fig.colorbar(image, ax=ax)
            path = plot_dir / filename; _save_plot(fig, path); outputs.append(path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(aggregate["minimum_validation_log_loss_mean"], aggregate["minimum_validation_log_loss_std"], alpha=0.8)
    for _, row in aggregate.iterrows():
        ax.annotate(str(row["condition"]), (row["minimum_validation_log_loss_mean"], row["minimum_validation_log_loss_std"]), fontsize=8)
    ax.set_xlabel("Mean best validation log loss"); ax.set_ylabel("Between-fold standard deviation"); ax.set_title("Mean versus variability"); ax.grid(alpha=0.25)
    path = plot_dir / "mean_vs_variability.png"; _save_plot(fig, path); outputs.append(path)

    stability = aggregate.sort_values("fraction")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for condition, group in stability.groupby("condition"):
        ax.plot(group["fraction"], group["final20_logloss_std_mean"], "o-", label=condition)
    ax.set_xlabel("Cutout fraction"); ax.set_ylabel("Final-20 log-loss standard deviation"); ax.set_title("Training stability"); ax.grid(alpha=0.25); ax.legend()
    path = plot_dir / "stability_by_fraction.png"; _save_plot(fig, path); outputs.append(path)
    return outputs


def main() -> None:
    global RUN_ROOT
    parser = argparse.ArgumentParser(description="Summarize DaT Stage 2 runs.")
    parser.add_argument("--run_root", default=str(RUN_ROOT))
    parser.add_argument("--best_config", default="")
    parser.add_argument("--calibration", default="artifacts/dat_parkinsons/optimization/calibration.json")
    parser.add_argument("--expected_folds", type=int, default=5)
    parser.add_argument("--expected_epochs", type=int, default=100)
    args = parser.parse_args()
    RUN_ROOT = Path(args.run_root)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    calibration_path = Path(args.calibration)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.is_file() else {"method": "raw", "temperature": 1.0}
    runs = _read_runs()
    if not runs:
        raise FileNotFoundError(f"No DaT metrics.csv files found under {RUN_ROOT}.")
    summaries = [_run_summary(config, frame, path, calibration) for config, frame, path in runs]
    per_run = pd.DataFrame(summaries)
    expected_conditions = {"none", "random", "cam_low", "cam_high"}
    present_conditions = set(per_run["condition"])
    if not present_conditions.issubset(expected_conditions):
        raise ValueError("Unexpected DaT masking condition found.")
    if present_conditions != expected_conditions:
        raise ValueError("DaT summary is missing one or more expected masking conditions.")
    if args.expected_folds > 0 and set(per_run["fold"]) != set(range(args.expected_folds)):
        raise ValueError("DaT summary is missing one or more expected CV folds.")
    non_baseline = per_run[per_run["condition"] != "none"]
    if not non_baseline.empty:
        if not {4, 8}.issubset(set(int(value) for value in non_baseline["M"])):
            raise ValueError("DaT summary is missing M4 or M8 runs.")
        if not {0.05, 0.10, 0.20, 0.30}.issubset(set(round(float(value), 2) for value in non_baseline["fraction"])):
            raise ValueError("DaT summary is missing one or more expected cutout fractions.")
    if args.expected_epochs > 0 and (per_run["epochs_completed"] > args.expected_epochs).any():
        raise ValueError("A DaT run exceeds the configured epoch budget.")
    for config, _frame, _path in runs:
        if args.expected_epochs > 0 and int(config.get("epochs", args.expected_epochs)) != args.expected_epochs:
            raise ValueError("A DaT run does not match the expected frozen epoch budget.")
        if config.get("validation_augmentation", False) or config.get("test_augmentation", False):
            raise ValueError("Validation/test augmentation is not allowed in the DaT summary.")
    if args.best_config:
        frozen = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
        frozen_keys = ("model", "optimizer", "learning_rate", "weight_decay", "batch_size", "preprocessing")
        for config, _frame, path in runs:
            for key in frozen_keys:
                if key in frozen and key in config and config[key] != frozen[key]:
                    raise ValueError(f"{path} does not match frozen Stage 1 setting '{key}'.")
    aggregate = _aggregate(per_run)
    paired = _paired(per_run)
    table_dir = SUMMARY_DIR / "tables"; table_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(table_dir / "per_run_metrics.csv", index=False)
    aggregate.to_csv(table_dir / "aggregate_metrics.csv", index=False)
    paired.to_csv(table_dir / "paired_effects.csv", index=False)
    plots = _plots(per_run, aggregate, paired)
    report = {
        "run_count": len(per_run),
        "conditions": sorted(per_run["condition"].unique()),
        "folds": sorted(int(v) for v in per_run["fold"].unique()),
        "plots": [str(path.relative_to(ROOT)) for path in plots],
        "logloss_effect_definition": "logloss_improvement = random_logloss - treatment_logloss; positive is better treatment log loss",
        "oof_logloss_definition": "oof_raw_log_loss and oof_calibrated_log_loss are computed from saved fold OOF logits when available.",
        "calibration_source": str(calibration_path),
        "privacy": "No competition data, UIDs, or patient-level predictions are written by this generator.",
    }
    (SUMMARY_DIR / "summary_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"runs": len(per_run), "plots": len(plots)}, indent=2))


if __name__ == "__main__":
    main()
