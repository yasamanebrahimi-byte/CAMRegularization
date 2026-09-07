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
        summaries.append({
            "run_dir": _portable_or_key(config_path.parent),
            "fold": int(config["fold"]), "condition": str(config["condition"]), "M": int(config.get("cutout_m", 0)),
            "fraction": float(config.get("cutout_fraction", 0.0) or 0.0),
            "epochs_completed": int(frame["epoch"].max()),
            "minimum_validation_log_loss": float(best["val_log_loss"]),
            "epoch_at_minimum_validation_log_loss": int(best["epoch"]),
            "accuracy_at_minimum_validation_log_loss": float(best["val_accuracy"]),
            "auroc_at_minimum_validation_log_loss": float(best["val_auroc"]),
            "brier_at_minimum_validation_log_loss": float(best["val_brier_score"]),
            "ece_at_minimum_validation_log_loss": float(best["val_ece"]),
            "maximum_validation_auroc": float(frame["val_auroc"].astype(float).max()),
            "final_validation_log_loss": float(frame.iloc[-1]["val_log_loss"]),
            "final_validation_accuracy": float(frame.iloc[-1]["val_accuracy"]),
            "final20_logloss_mean": float(tail["val_log_loss"].mean()),
            "final20_logloss_std": float(tail["val_log_loss"].std(ddof=1)) if len(tail) > 1 else 0.0,
            "oof_raw_log_loss": float(compute_binary_metrics(targets, logits=logits)["log_loss"]),
            "validation_trajectory_source": "outer_fold_metrics.csv",
            "oof_raw_source": "candidate_fold_OOF_logits",
            "oof_calibrated_source": "candidate_oof_metrics.csv_only",
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
    candidate_rows = []
    for candidate in selection.get("candidates", []):
        candidate_rows.append({
            "condition": candidate["condition"], "M": candidate["M"], "fraction": candidate["fraction"],
            "cam_layer": candidate.get("cam_layer"),
            "saliency_candidate_percent": candidate.get("saliency_candidate_percent"),
            "min_foreground_fraction": candidate.get("min_foreground_fraction"),
            "student_max_cv_epochs": candidate.get("student_max_cv_epochs"),
            "early_stopping_patience": candidate.get("early_stopping_patience"),
            "raw_oof_log_loss": candidate.get("raw_oof_log_loss"),
            "cross_fitted_calibrated_oof_log_loss": candidate.get("cross_fitted_calibrated_oof_log_loss"),
            "final_stage2_training_epochs": candidate.get("final_stage2_training_epochs"),
            "selected_candidate_fold_best_epochs": json.dumps(candidate.get("selected_candidate_fold_best_epochs", [])),
            "calibration_provenance": candidate.get("calibration_provenance"),
        })
    pd.DataFrame(candidate_rows).to_csv(table_dir / "candidate_oof_metrics.csv", index=False)
    fold_metrics = pd.DataFrame()
    if selection.get("candidates"):
        fold_rows = []
        for candidate in selection["candidates"]:
            for row in candidate.get("candidate_fold_metrics", []):
                fold_rows.append({"condition": candidate["condition"], "M": candidate["M"], "fraction": candidate["fraction"], **row})
        fold_metrics = pd.DataFrame(fold_rows)
    if not fold_metrics.empty:
        fold_metrics.to_csv(table_dir / "candidate_fold_metrics.csv", index=False)

    indexed = {(int(row.fold), str(row.condition), int(row.M), float(row.fraction)): row
               for row in per_run.itertuples()}
    none_by_fold = {int(row.fold): row for row in per_run[per_run.condition == "none"].itertuples()}
    paired_rows = []

    def add_pair(comparison, treatment, reference, treatment_m, reference_m, fold, fraction, treatment_row, reference_row):
        paired_rows.append({
            "comparison": comparison, "fold": int(fold), "condition": treatment,
            "reference_condition": reference, "M": int(treatment_m),
            "reference_M": int(reference_m), "fraction": float(fraction),
            "reference_logloss": float(reference_row.minimum_validation_log_loss),
            "treatment_logloss": float(treatment_row.minimum_validation_log_loss),
            "logloss_improvement": float(reference_row.minimum_validation_log_loss - treatment_row.minimum_validation_log_loss),
            "reference_accuracy": float(reference_row.accuracy_at_minimum_validation_log_loss),
            "treatment_accuracy": float(treatment_row.accuracy_at_minimum_validation_log_loss),
            "accuracy_difference": float(treatment_row.accuracy_at_minimum_validation_log_loss - reference_row.accuracy_at_minimum_validation_log_loss),
            "effect_source": "paired_outer_fold_validation_at_minimum_validation_log_loss",
        })

    for key, treatment_row in indexed.items():
        fold, condition, m_value, fraction = key
        if condition == "none":
            continue
        if condition == "random":
            reference_row = none_by_fold.get(fold)
            if reference_row is not None:
                add_pair("random_vs_none", condition, "none", m_value, 0, fold, fraction, treatment_row, reference_row)
        elif condition in {"cam_low", "cam_high"}:
            reference_row = indexed.get((fold, "random", m_value, fraction))
            if reference_row is not None:
                add_pair(f"{condition}_vs_random", condition, "random", m_value, m_value, fold, fraction, treatment_row, reference_row)

    for (fold, condition, m_value, fraction), treatment_row in indexed.items():
        if m_value != 8 or condition == "none":
            continue
        reference_row = indexed.get((fold, condition, 4, fraction))
        if reference_row is not None:
            add_pair("M8_vs_M4", condition, condition, 8, 4, fold, fraction, treatment_row, reference_row)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(table_dir / "paired_effects.csv", index=False)
    plots = output / "plots"
    # Matched CAM-vs-random effects use the already paired, checkpoint-aligned
    # validation values written to paired_effects.csv.
    effects = (paired[paired["comparison"].isin(["cam_low_vs_random", "cam_high_vs_random"])].copy()
               if "comparison" in paired.columns else pd.DataFrame())
    if not effects.empty:
        effects["condition"] = effects["comparison"].str.replace("_vs_random", "", regex=False)
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
    candidate_frame = pd.DataFrame(candidate_rows)
    if not candidate_frame.empty:
        for condition, group in candidate_frame.groupby("condition"):
            ax.plot(group["fraction"], group["cross_fitted_calibrated_oof_log_loss"], "o-", label=condition)
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
        "selection_basis": "cross_fitted_calibrated_oof_log_loss",
        "selection_lower_is_better": True,
        "best_overall": selection.get("best_overall", {}),
        "best_masked": selection.get("best_masked", {}),
        "validation_trajectory_metrics": {
            "source": "each outer fold metrics.csv",
            "fields": ["minimum_validation_log_loss", "epoch_at_minimum_validation_log_loss", "accuracy_at_minimum_validation_log_loss", "auroc_at_minimum_validation_log_loss", "brier_at_minimum_validation_log_loss", "ece_at_minimum_validation_log_loss"],
        },
        "oof_raw_metrics": {"source": "concatenated fold-best-checkpoint OOF logits", "fields": ["oof_raw_log_loss"]},
        "oof_cross_fitted_calibrated_metrics": {"source": "candidate_oof_metrics.csv; candidate-specific fold-aware calibration", "fields": ["cross_fitted_calibrated_oof_log_loss", "cross_fitted_calibrated_oof_brier_score", "cross_fitted_calibrated_oof_ece"]},
        "final_fitted_calibration": {"source": "all candidate OOF logits for the selected candidate; packaged with final model", "field": "calibration"},
        "stage1_lineage": selection.get("best_masked", {}).get("selected_stage1_config_fingerprint"),
        "selected_stage2_fold_best_epochs": selection.get("best_masked", {}).get("selected_candidate_fold_best_epochs"),
        "final_stage2_training_epoch_rule": selection.get("best_masked", {}).get("final_stage2_training_epoch_rule"),
        "final_stage2_training_epochs": selection.get("best_masked", {}).get("final_stage2_training_epochs"),
        "paired_effects": {"source": "paired_effects.csv", "logloss_definition": "reference_logloss - treatment_logloss", "accuracy_definition": "treatment_accuracy - reference_accuracy"},
        "integrity_report": "integrity_report.json",
        "privacy": "No UIDs, patient-level predictions, OOF arrays, or local absolute paths are written.",
    }
    (output / "summary_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"per_run": per_run, "aggregate": aggregate, "paired": paired, "report": report}
