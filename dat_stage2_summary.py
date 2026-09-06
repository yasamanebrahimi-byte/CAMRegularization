"""Commit-safe DaT Stage 2 summary tables and research figures."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dat_provenance import REPO_ROOT, portable_path
from dat_select_model import integrity_check


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _resolve_artifact(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _portable_or_key(path: str | Path) -> str:
    try:
        return portable_path(path)
    except ValueError:
        return f"external/{Path(path).name}"


def generate_summary(
    runs_dir: str | Path = "runs/dat_parkinsons/resnet18_3d",
    summary_dir: str | Path = "runs/dat_parkinsons/summary",
    *, expected_folds: int = 5, selection_path: str | Path | None = None,
    conditions=("none", "random", "cam_low", "cam_high"), m_values=(4, 8), fractions=(0.05, 0.10, 0.20, 0.30),
    frozen_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runs_root = Path(runs_dir)
    output = Path(summary_dir)
    report = integrity_check(runs_root, expected_folds=expected_folds, conditions=conditions, m_values=m_values, fractions=fractions, frozen_config=frozen_config)
    output.mkdir(parents=True, exist_ok=True)
    (output / "integrity_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not report["passed"]:
        raise ValueError("Stage 2 integrity check failed; summary cannot proceed.")
    selection = {}
    if selection_path and Path(selection_path).is_file():
        selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if selection:
        (output / "selected_best_masked.json").write_text(
            json.dumps(selection.get("best_masked", {}), indent=2, sort_keys=True), encoding="utf-8"
        )
    selected_by_key = {(row["condition"], int(row["M"]), float(row["fraction"])): row
                      for row in selection.get("candidates", [])}
    summaries = []
    for config_path in sorted(runs_root.rglob("config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("stage") != 2 or config_path.parent.name == "teachers":
            continue
        metrics_path = config_path.parent / "metrics.csv"
        frame = pd.read_csv(metrics_path)
        best = frame.loc[frame["val_log_loss"].astype(float).idxmin()]
        tail = frame.tail(min(20, len(frame)))
        artifact = _resolve_artifact(config["oof_artifact"])
        payload = np.load(artifact)
        logits = np.asarray(payload["logits"], dtype=np.float64)
        targets = np.asarray(payload["targets"], dtype=np.int64)
        from dat_metrics import compute_binary_metrics
        candidate = selected_by_key.get((str(config["condition"]), int(config.get("cutout_m", 0)), float(config.get("cutout_fraction", 0.0) or 0.0)), {})
        summaries.append({
            "run_dir": _portable_or_key(config_path.parent),
            "fold": int(config["fold"]), "condition": str(config["condition"]), "M": int(config.get("cutout_m", 0)),
            "fraction": float(config.get("cutout_fraction", 0.0) or 0.0),
            "epochs_completed": int(frame["epoch"].max()), "minimum_validation_log_loss": float(best["val_log_loss"]),
            "best_validation_accuracy": float(frame.loc[frame["val_accuracy"].astype(float).idxmax(), "val_accuracy"]),
            "best_validation_auroc": float(best["val_auroc"]),
            "best_validation_brier_score": float(best["val_brier_score"]),
            "final_validation_log_loss": float(frame.iloc[-1]["val_log_loss"]),
            "final_validation_accuracy": float(frame.iloc[-1]["val_accuracy"]),
            "final20_logloss_mean": float(tail["val_log_loss"].mean()),
            "final20_logloss_std": float(tail["val_log_loss"].std(ddof=1)) if len(tail) > 1 else 0.0,
            "oof_raw_log_loss": float(compute_binary_metrics(targets, logits=logits)["log_loss"]),
            "oof_cross_fitted_calibrated_log_loss": float(candidate.get("cross_fitted_calibrated_oof_log_loss", float("nan"))),
            "validation_trajectory_source": "outer_fold_metrics.csv",
            "oof_raw_source": "candidate_fold_OOF_logits",
            "oof_calibrated_source": "candidate_specific_cross_fitted_calibration",
        })
    per_run = pd.DataFrame(summaries)
    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(table_dir / "per_run_metrics.csv", index=False)
    numeric = [c for c in per_run.columns if c not in {"run_dir", "condition", "fold", "validation_trajectory_source", "oof_raw_source", "oof_calibrated_source"} and pd.api.types.is_numeric_dtype(per_run[c])]
    aggregate_rows = []
    for keys, group in per_run.groupby(["condition", "M", "fraction"]):
        row = dict(zip(["condition", "M", "fraction"], keys))
        for column in numeric:
            values = group[column].dropna().to_numpy(dtype=float)
            row[f"{column}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(table_dir / "aggregate_metrics.csv", index=False)
    pd.DataFrame(selection.get("candidates", [])).drop(columns=["calibration"], errors="ignore").to_csv(
        table_dir / "candidate_oof_metrics.csv", index=False
    )
    baseline = {(int(row.fold),): row for row in per_run[per_run.condition == "none"].itertuples()}
    paired_rows = []
    for row in per_run[per_run.condition != "none"].itertuples():
        base = baseline.get((int(row.fold),))
        if base is not None:
            paired_rows.append({"comparison": f"{row.condition}_versus_none", "fold": row.fold, "M": row.M, "fraction": row.fraction,
                                "validation_logloss_improvement": base.final_validation_log_loss - row.final_validation_log_loss})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(table_dir / "paired_effects.csv", index=False)
    plots = output / "plots"
    # Matched CAM-vs-random effects use the same fold, M, and fraction.
    effect_rows = []
    indexed = {(int(row.fold), str(row.condition), int(row.M), float(row.fraction)): row
               for row in per_run.itertuples()}
    for (fold, condition, m_value, fraction), row in indexed.items():
        if condition not in {"cam_low", "cam_high"}:
            continue
        random_row = indexed.get((fold, "random", m_value, fraction))
        if random_row is not None:
            effect_rows.append({"condition": condition, "fold": fold, "M": m_value, "fraction": fraction,
                                "logloss_improvement": random_row.final_validation_log_loss - row.final_validation_log_loss})
    effects = pd.DataFrame(effect_rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not effects.empty:
        effects.boxplot(column="logloss_improvement", by="condition", ax=ax)
    ax.set_title("CAM versus random paired validation effect"); ax.set_xlabel(""); ax.set_ylabel("Random log loss - CAM log loss"); plt.suptitle("")
    _save(fig, plots / "paired_cam_effects_vs_random.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (condition, m_value, fraction), group in per_run.groupby(["condition", "M", "fraction"]):
        for _, row in group.iterrows():
            run_ref = _resolve_artifact(row.run_dir)
            if not (run_ref / "metrics.csv").is_file():
                continue
            frame = pd.read_csv(run_ref / "metrics.csv")
            ax.plot(frame["epoch"], frame["val_log_loss"], alpha=0.35, label=f"{condition} M{m_value} f{fraction:.2f}")
    ax.set_title("DaT Stage 2 validation learning curves"); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation log loss"); ax.grid(alpha=.25)
    _save(fig, plots / "learning_curves.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not aggregate.empty:
        for condition, group in aggregate.groupby("condition"):
            ax.plot(group["fraction"], group["oof_cross_fitted_calibrated_log_loss_mean"], "o-", label=condition)
    ax.set_title("Candidate-specific cross-fitted calibrated OOF log loss"); ax.set_xlabel("Fraction"); ax.set_ylabel("OOF log loss"); ax.grid(alpha=.25); ax.legend()
    _save(fig, plots / "candidate_calibrated_oof.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not aggregate.empty:
        ax.scatter(aggregate["minimum_validation_log_loss_mean"], aggregate["minimum_validation_log_loss_std"])
        for _, row in aggregate.iterrows(): ax.annotate(str(row["condition"]), (row["minimum_validation_log_loss_mean"], row["minimum_validation_log_loss_std"]))
    ax.set_title("Mean versus variability"); ax.set_xlabel("Mean best validation log loss"); ax.set_ylabel("Between-fold std"); ax.grid(alpha=.25)
    _save(fig, plots / "mean_vs_variability.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not aggregate.empty:
        for condition, group in aggregate.groupby("condition"):
            ax.plot(group["fraction"], group["final20_logloss_std_mean"], "o-", label=condition)
    ax.set_title("Training stability"); ax.set_xlabel("Fraction"); ax.set_ylabel("Final-20 validation log-loss std"); ax.grid(alpha=.25); ax.legend()
    _save(fig, plots / "stability_by_fraction.png")
    summary = {
        "run_count": int(len(per_run)), "expected_run_count": int(report["expected_cell_count"]),
        "validation_trajectory_metrics": ["minimum_validation_log_loss", "final_validation_log_loss", "final_validation_accuracy"],
        "oof_raw_metrics": ["oof_raw_log_loss"],
        "oof_cross_fitted_calibrated_metrics": ["oof_cross_fitted_calibrated_log_loss"],
        "stage1_lineage": selection.get("best_masked", {}).get("selected_stage1_config_fingerprint"),
        "integrity_report": "integrity_report.json",
        "privacy": "No UIDs, patient-level predictions, OOF arrays, or local absolute paths are written.",
    }
    (output / "summary_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"per_run": per_run, "aggregate": aggregate, "paired": paired, "report": report}
