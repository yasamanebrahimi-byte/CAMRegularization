"""Integrity-check and select DaT Stage 2 candidates from candidate OOF data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dat_calibration import fit_candidate_calibration
from dat_masking_experiments import CONDITIONS, DEFAULT_FRACTIONS, DEFAULT_M, _run_is_valid, cell_key, expected_grid
from dat_provenance import REPO_ROOT, fingerprint, median_round_half_up, portable_path


# These fields define the treatment recipe and the frozen Stage 1 training
# recipe.  They are intentionally explicit so a partial/defaulted candidate
# cannot be combined across folds or silently changed at final training time.
CANDIDATE_RECIPE_FIELDS = (
    "condition", "cutout_m", "cutout_fraction", "cam_layer",
    "saliency_candidate_percent", "min_foreground_fraction",
    "student_seed_policy", "student_model",
    "selected_stage1_config_fingerprint", "preprocessing_fingerprint",
    "fold_assignment_fingerprint", "student_max_cv_epochs", "epochs",
    "early_stopping_patience", "patience",
)

FROZEN_STAGE1_FIELDS = (
    "model", "n_input_channels", "num_classes", "base_channels", "dropout",
    "spatial_augmentation", "optimizer", "learning_rate", "weight_decay",
    "scheduler", "min_lr", "momentum", "nesterov", "adamw_betas",
    "label_smoothing", "batch_size", "amp", "epochs", "patience",
    "preprocessing_fingerprint", "selected_stage1_config_fingerprint",
    "fold_assignment_fingerprint", "final_training_epochs",
)

CAM_RECIPE_FIELDS = ("cam_layer", "saliency_candidate_percent", "min_foreground_fraction")


def _config_value(config: dict[str, Any], field: str) -> Any:
    if field == "selected_stage1_config_fingerprint":
        return config.get("selected_stage1_config_fingerprint", config.get("config_fingerprint"))
    if field == "learning_rate":
        return config.get("learning_rate", config.get("lr"))
    if field == "final_training_epochs":
        return config.get("stage1_final_training_epochs", config.get(field))
    return config.get(field)


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))
    return left == right


def _consistency_issues(entries: list[tuple[dict[str, Any], Path]], fields: tuple[str, ...], *, require_present: bool = True) -> list[dict[str, Any]]:
    issues = []
    if not entries:
        return issues
    for field in fields:
        values = [_config_value(config, field) for config, _ in entries]
        if (require_present and values[0] is None) or any(
            (require_present and value is None) or not _same_value(value, values[0]) for value in values[1:]
        ):
            issues.append({
                "field": field,
                "values": [value for value in values],
                "run_dirs": [str(path) for _, path in entries],
            })
    return issues


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _portable_or_key(path: str | Path) -> str:
    try:
        return portable_path(path)
    except ValueError:
        return f"external/{Path(path).name}"


def _is_student_config(path: Path, config: dict[str, Any]) -> bool:
    if config.get("stage") != 2:
        return False
    if str(config.get("condition", "")) not in CONDITIONS:
        return False
    return "teachers" not in {part.lower() for part in path.parts}


def integrity_check(
    runs_dir: str | Path,
    *,
    expected_folds: int,
    conditions=CONDITIONS,
    m_values=DEFAULT_M,
    fractions=DEFAULT_FRACTIONS,
    frozen_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(runs_dir)
    expected = expected_grid(expected_folds, conditions, m_values, fractions)
    expected_by_key = {cell["cell_key"]: cell for cell in expected}
    discovered: dict[str, list[Path]] = {}
    invalid: list[dict[str, Any]] = []
    debug_runs: list[str] = []
    teacher_issues: list[dict[str, Any]] = []
    oof_issues: list[dict[str, Any]] = []
    candidate_configs: dict[tuple[str, int, float], list[tuple[dict[str, Any], Path]]] = {}
    frozen_control_issues: list[dict[str, Any]] = []
    for config_path in (sorted(root.rglob("config.json")) if root.is_dir() else []):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            invalid.append({"run_dir": str(config_path.parent), "issues": [f"invalid_config:{exc}"]})
            continue
        if not _is_student_config(config_path, config):
            continue
        key = cell_key(int(config.get("fold", -1)), str(config.get("condition")), int(config.get("cutout_m", 0)), float(config.get("cutout_fraction", 0.0) or 0.0))
        discovered.setdefault(key, []).append(config_path.parent)
        candidate_key = (
            str(config.get("condition")), int(config.get("cutout_m", 0)),
            float(config.get("cutout_fraction", 0.0) or 0.0),
        )
        candidate_configs.setdefault(candidate_key, []).append((config, config_path.parent))
        if frozen_config:
            expected_lineage = {
                "selected_stage1_config_fingerprint": frozen_config.get("config_fingerprint"),
                "preprocessing_fingerprint": frozen_config.get("preprocessing_fingerprint"),
                "fold_assignment_fingerprint": frozen_config.get("fold_assignment_fingerprint"),
                "final_training_epochs": frozen_config.get("final_training_epochs"),
                "epochs": frozen_config.get("epochs"),
                "student_max_cv_epochs": frozen_config.get("epochs"),
                "patience": frozen_config.get("patience"),
                "early_stopping_patience": frozen_config.get("patience"),
            }
            for field, value in expected_lineage.items():
                if value is not None and config.get(field) != value:
                    invalid.append({"run_dir": str(config_path.parent), "cell_key": key,
                                    "issues": [f"config_mismatch:{field}"]})
        valid, issues = _run_is_valid(config_path.parent)
        if not valid:
            invalid.append({"run_dir": str(config_path.parent), "cell_key": key, "issues": issues})
            if "debug_or_truncated" in issues:
                debug_runs.append(str(config_path.parent))
        if str(config.get("condition", "")).startswith("cam_") and not config.get("teacher_checkpoint_sha256"):
            teacher_issues.append({"run_dir": str(config_path.parent), "cell_key": key, "issue": "missing_teacher_fingerprint"})
        artifact = config.get("oof_artifact")
        artifact_path = _resolve(artifact) if artifact else None
        run_is_repo_local = False
        try:
            config_path.parent.resolve().relative_to(REPO_ROOT.resolve())
            run_is_repo_local = True
        except ValueError:
            pass
        if artifact and Path(artifact).is_absolute() and run_is_repo_local:
            oof_issues.append({"run_dir": str(config_path.parent), "cell_key": key, "issue": "nonportable_absolute_oof_artifact"})
        if artifact_path is None or not artifact_path.is_file():
            oof_issues.append({"run_dir": str(config_path.parent), "cell_key": key, "issue": "missing_oof_artifact"})
        else:
            try:
                payload = np.load(artifact_path)
                if not np.isfinite(payload["logits"]).all() or len(payload["logits"]) != len(payload["targets"]):
                    oof_issues.append({"run_dir": str(config_path.parent), "cell_key": key, "issue": "invalid_oof_artifact"})
                if "fold" in payload and not np.all(np.asarray(payload["fold"]) == int(config.get("fold", -1))):
                    oof_issues.append({"run_dir": str(config_path.parent), "cell_key": key, "issue": "fold_assignment_mismatch"})
            except Exception:
                oof_issues.append({"run_dir": str(config_path.parent), "cell_key": key, "issue": "invalid_oof_artifact"})
    candidate_recipe_issues: list[dict[str, Any]] = []
    for candidate_key, entries in candidate_configs.items():
        for issue in _consistency_issues(entries, CANDIDATE_RECIPE_FIELDS):
            item = {"candidate": candidate_key, **issue}
            candidate_recipe_issues.append(item)
            invalid.append({
                "run_dir": issue["run_dirs"][0],
                "cell_key": cell_key(int(entries[0][0].get("fold", -1)), candidate_key[0], candidate_key[1], candidate_key[2]),
                "issues": [f"candidate_recipe_mismatch:{issue['field']}"] ,
            })

    # All candidates share the scientific Stage 1 recipe. Operational worker
    # counts are deliberately excluded because CAM candidates may be forced to
    # zero workers while none/random retain the requested count.
    all_entries = [entry for entries in candidate_configs.values() for entry in entries]
    for issue in _consistency_issues(all_entries, FROZEN_STAGE1_FIELDS, require_present=False):
        frozen_control_issues.append({"scope": "all_stage2_candidates", **issue})
    if frozen_config:
        for field in FROZEN_STAGE1_FIELDS:
            expected_value = _config_value(frozen_config, field)
            if expected_value is None:
                continue
            actual_values = [_config_value(config, field) for config, _ in all_entries]
            if any(value is None or not _same_value(value, expected_value) for value in actual_values):
                frozen_control_issues.append({
                    "scope": "frozen_stage1_config", "field": field,
                    "expected": expected_value, "values": actual_values,
                    "run_dirs": [str(path) for _, path in all_entries],
                })

    # CAM placement controls are common across CAM conditions in one research
    # grid. A differing fold or condition is a hard integrity error.
    cam_entries = [entry for candidate, entries in candidate_configs.items() if candidate[0].startswith("cam_") for entry in entries]
    cam_control_issues = _consistency_issues(cam_entries, CAM_RECIPE_FIELDS)
    for issue in cam_control_issues:
        frozen_control_issues.append({"scope": "all_cam_candidates", **issue})

    for issue in candidate_recipe_issues + frozen_control_issues:
        invalid.append({
            "run_dir": issue["run_dirs"][0] if issue.get("run_dirs") else str(root),
            "issues": [f"stage2_control_mismatch:{issue['field']}"] ,
        })
    duplicate_cells = [{"cell_key": key, "run_dirs": [str(p) for p in paths]} for key, paths in discovered.items() if len(paths) > 1]
    missing_cells = [cell for key, cell in expected_by_key.items() if key not in discovered]
    unexpected_cells = [key for key in discovered if key not in expected_by_key]
    valid_count = 0
    for key, paths in discovered.items():
        if key in expected_by_key and len(paths) == 1 and _run_is_valid(paths[0])[0]:
            valid_count += 1
    report = {
        "expected_cell_count": len(expected), "discovered_cell_count": sum(len(v) for v in discovered.values()),
        "unique_discovered_cell_count": len(discovered), "valid_cell_count": valid_count,
        "missing_cells": missing_cells, "duplicate_cells": duplicate_cells,
        "unexpected_cells": unexpected_cells, "invalid_cells": invalid,
        "debug_truncated_runs": debug_runs, "config_mismatches": [item for item in invalid if any("config_mismatch" in issue for issue in item["issues"])],
        "teacher_fingerprint_issues": teacher_issues, "oof_artifact_issues": oof_issues,
        "candidate_recipe_issues": candidate_recipe_issues,
        "frozen_control_issues": frozen_control_issues,
        "passed": not (missing_cells or duplicate_cells or unexpected_cells or invalid or teacher_issues or oof_issues),
    }
    return report


def _candidate_runs(runs_dir: Path, report: dict[str, Any]) -> dict[tuple[str, int, float], list[tuple[dict, Path]]]:
    invalid_keys = {item.get("cell_key") for item in report["invalid_cells"]}
    result: dict[tuple[str, int, float], list[tuple[dict, Path]]] = {}
    for config_path in sorted(runs_dir.rglob("config.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not _is_student_config(config_path, config):
            continue
        key = cell_key(int(config.get("fold", -1)), str(config.get("condition")), int(config.get("cutout_m", 0)), float(config.get("cutout_fraction", 0.0) or 0.0))
        if key in invalid_keys:
            continue
        artifact = config.get("oof_artifact")
        if not artifact:
            continue
        path = _resolve(artifact)
        if not path.is_file():
            continue
        candidate_key = (str(config["condition"]), int(config.get("cutout_m", 0)), float(config.get("cutout_fraction", 0.0) or 0.0))
        result.setdefault(candidate_key, []).append((config, config_path.parent))
    return result


def _selected_fold_metrics(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Extract only fold-level, minimum-validation-log-loss metrics."""
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise ValueError(f"Missing Stage 2 metrics for {run_dir}.")
    frame = pd.read_csv(metrics_path)
    if frame.empty or "val_log_loss" not in frame:
        raise ValueError(f"Stage 2 metrics have no validation trajectory: {run_dir}.")
    row = frame.loc[frame["val_log_loss"].astype(float).idxmin()]
    best_epoch = int(config.get("best_epoch", row["epoch"]))
    return {
        "fold": int(config["fold"]),
        "best_epoch": best_epoch,
        "minimum_validation_log_loss": float(row["val_log_loss"]),
        "epoch_at_minimum_validation_log_loss": int(row["epoch"]),
        "accuracy_at_minimum_validation_log_loss": float(row["val_accuracy"]),
        "auroc_at_minimum_validation_log_loss": float(row["val_auroc"]),
        "brier_at_minimum_validation_log_loss": float(row["val_brier_score"]),
        "ece_at_minimum_validation_log_loss": float(row["val_ece"]),
        "teacher_checkpoint_sha256": config.get("teacher_checkpoint_sha256"),
        "run_dir": _portable_or_key(run_dir),
    }


def _candidate_recipe(entries: list[tuple[dict[str, Any], Path]]) -> dict[str, Any]:
    config = entries[0][0]
    return {
        "condition": str(config["condition"]),
        "M": int(config.get("cutout_m", 0)),
        "fraction": float(config.get("cutout_fraction", 0.0) or 0.0),
        "cam_layer": config.get("cam_layer"),
        "saliency_candidate_percent": float(config["saliency_candidate_percent"]),
        "min_foreground_fraction": float(config["min_foreground_fraction"]),
        "student_seed_policy": config.get("student_seed_policy"),
        "student_model": config.get("student_model", config.get("model")),
        "selected_stage1_config_fingerprint": config.get("selected_stage1_config_fingerprint"),
        "preprocessing_fingerprint": config.get("preprocessing_fingerprint"),
        "fold_assignment_fingerprint": config.get("fold_assignment_fingerprint"),
        "student_max_cv_epochs": int(config.get("student_max_cv_epochs", config.get("epochs"))),
        "early_stopping_patience": int(config.get("early_stopping_patience", config.get("patience", 15))),
    }


def select_best_overall_and_masked(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep the research winner and masked competition winner distinct."""
    if not candidates:
        raise ValueError("No valid Stage 2 candidates were found.")
    best_overall = min(candidates, key=lambda row: (row["selection_score"], row["condition"], row["M"], row["fraction"]))
    masked = [row for row in candidates if row["condition"] in {"random", "cam_low", "cam_high"}]
    if not masked:
        raise ValueError("Stage 2 requires at least one masked candidate for Submission #2.")
    best_masked = min(masked, key=lambda row: (row["selection_score"], row["condition"], row["M"], row["fraction"]))
    return best_overall, best_masked


def select_candidates(
    runs_dir: str | Path,
    *,
    expected_folds: int,
    output_path: str | Path,
    conditions=CONDITIONS,
    m_values=DEFAULT_M,
    fractions=DEFAULT_FRACTIONS,
    calibration_method: str = "temperature",
    frozen_config: dict[str, Any] | None = None,
    summary_dir: str | Path | None = None,
) -> dict[str, Any]:
    report = integrity_check(runs_dir, expected_folds=expected_folds, conditions=conditions, m_values=m_values, fractions=fractions, frozen_config=frozen_config)
    summary_dir = Path(summary_dir) if summary_dir is not None else Path(runs_dir).parent / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "integrity_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not report["passed"]:
        raise ValueError("Stage 2 integrity check failed; see summary/integrity_report.json.")
    grouped = _candidate_runs(Path(runs_dir), report)
    candidates = []
    candidate_fold_rows: list[dict[str, Any]] = []
    for (condition, m_value, fraction), entries in sorted(grouped.items()):
        entries = sorted(entries, key=lambda item: int(item[0].get("fold", -1)))
        recipe = _candidate_recipe(entries)
        fold_metrics = [_selected_fold_metrics(config, run_dir) for config, run_dir in entries]
        fold_metrics = sorted(fold_metrics, key=lambda row: row["fold"])
        selected_fold_best_epochs = [int(row["best_epoch"]) for row in fold_metrics]
        final_stage2_training_epochs = median_round_half_up(selected_fold_best_epochs)
        for row in fold_metrics:
            candidate_fold_rows.append({
                "condition": condition, "M": int(m_value), "fraction": float(fraction),
                **{key: value for key, value in row.items() if key != "run_dir"},
            })
        logits_parts, target_parts, fold_parts = [], [], []
        for config, _run_dir in entries:
            payload = np.load(_resolve(config["oof_artifact"]))
            logits_parts.append(np.asarray(payload["logits"], dtype=np.float64))
            target_parts.append(np.asarray(payload["targets"], dtype=np.int64))
            if "fold" in payload:
                fold_parts.append(np.asarray(payload["fold"], dtype=np.int64))
            else:
                fold_parts.append(np.full(len(payload["targets"]), int(config["fold"]), dtype=np.int64))
        logits = np.concatenate(logits_parts, axis=0)
        targets = np.concatenate(target_parts, axis=0)
        fold_ids = np.concatenate(fold_parts, axis=0)
        if set(int(v) for v in fold_ids.tolist()) != set(range(int(expected_folds))):
            raise ValueError(f"Candidate {condition} M{m_value} fraction {fraction:.2f} lacks an OOF partition for every fold.")
        result = fit_candidate_calibration(logits, targets, fold_ids, method=calibration_method)
        calibration = dict(result["final_calibration"])
        calibration_filename = f"{condition}_M{m_value}_fraction{fraction:.2f}".replace(".", "p") + ".json"
        calibration_path = summary_dir / "calibration" / calibration_filename
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path.write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")
        candidate = {
            **recipe,
            "selected_stage1_config_fingerprint": entries[0][0].get("selected_stage1_config_fingerprint"),
            "preprocessing_fingerprint": entries[0][0].get("preprocessing_fingerprint"),
            "candidate_fold_metrics": fold_metrics,
            "selected_candidate_fold_best_epochs": selected_fold_best_epochs,
            "final_stage2_training_epoch_rule": "median_selected_stage2_fold_best_epoch_round_half_up",
            "final_stage2_training_epochs": final_stage2_training_epochs,
            "stage1_final_training_epochs": int(entries[0][0].get("stage1_final_training_epochs", entries[0][0].get("final_training_epochs"))),
            "fold_assignment_fingerprint": entries[0][0].get("fold_assignment_fingerprint"),
            "raw_oof_log_loss": float(result["raw_metrics"]["log_loss"]),
            "raw_oof_auroc": float(result["raw_metrics"]["auroc"]),
            "raw_oof_brier_score": float(result["raw_metrics"]["brier_score"]),
            "raw_oof_accuracy": float(result["raw_metrics"]["accuracy"]),
            "cross_fitted_calibrated_oof_log_loss": float(result["cross_fitted_metrics"]["log_loss"]),
            "cross_fitted_calibrated_oof_auroc": float(result["cross_fitted_metrics"]["auroc"]),
            "cross_fitted_calibrated_oof_brier_score": float(result["cross_fitted_metrics"]["brier_score"]),
            "cross_fitted_calibrated_oof_accuracy": float(result["cross_fitted_metrics"]["accuracy"]),
            "cross_fitted_calibrated_oof_ece": float(result["cross_fitted_metrics"]["ece"]),
            "raw_cv_log_loss": float(result["raw_metrics"]["log_loss"]),
            "calibrated_cv_log_loss": float(result["cross_fitted_metrics"]["log_loss"]),
            "final_fitted_calibration_method": calibration.get("method", "raw"),
            "final_fitted_temperature": float(calibration.get("temperature", 1.0)),
            "calibration": calibration,
            "calibration_path": _portable_or_key(calibration_path),
            "calibration_provenance": "candidate_own_fold_OOF_logits_only",
            "n_oof_samples": int(len(targets)), "n_cv_folds": int(len(set(fold_ids.tolist()))),
            "fold_ids": sorted(set(int(v) for v in fold_ids.tolist())),
            "selection_score": float(result["cross_fitted_metrics"]["log_loss"]),
        }
        candidates.append(candidate)
    best_overall, best_masked = select_best_overall_and_masked(candidates)
    payload = {
        "selection_basis": "cross_fitted_calibrated_oof_log_loss",
        "best_overall": best_overall, "best_masked": best_masked,
        "selected_stage2_fold_best_epochs": best_masked["selected_candidate_fold_best_epochs"],
        "final_stage2_training_epoch_rule": best_masked["final_stage2_training_epoch_rule"],
        "final_stage2_training_epochs": best_masked["final_stage2_training_epochs"],
        # Compatibility for older callers: selected is always the masked
        # competition candidate, never the no-cutout baseline.
        "selected": best_masked, "candidates": candidates,
        "integrity_report": _portable_or_key(summary_dir / "integrity_report.json"),
    }
    pd.DataFrame([{
        "condition": row["condition"], "M": row["M"], "fraction": row["fraction"],
        "cam_layer": row["cam_layer"],
        "saliency_candidate_percent": row["saliency_candidate_percent"],
        "min_foreground_fraction": row["min_foreground_fraction"],
        "student_model": row["student_model"],
        "student_max_cv_epochs": row["student_max_cv_epochs"],
        "early_stopping_patience": row["early_stopping_patience"],
        "final_stage2_training_epochs": row["final_stage2_training_epochs"],
        "raw_oof_log_loss": row["raw_oof_log_loss"],
        "raw_oof_auroc": row["raw_oof_auroc"],
        "raw_oof_brier_score": row["raw_oof_brier_score"],
        "raw_oof_accuracy": row["raw_oof_accuracy"],
        "cross_fitted_calibrated_oof_log_loss": row["cross_fitted_calibrated_oof_log_loss"],
        "cross_fitted_calibrated_oof_auroc": row["cross_fitted_calibrated_oof_auroc"],
        "cross_fitted_calibrated_oof_brier_score": row["cross_fitted_calibrated_oof_brier_score"],
        "cross_fitted_calibrated_oof_accuracy": row["cross_fitted_calibrated_oof_accuracy"],
        "cross_fitted_calibrated_oof_ece": row["cross_fitted_calibrated_oof_ece"],
        "final_fitted_calibration_method": row["final_fitted_calibration_method"],
        "final_fitted_temperature": row["final_fitted_temperature"],
        "n_oof_samples": row["n_oof_samples"], "n_cv_folds": row["n_cv_folds"],
    } for row in candidates]).to_csv(summary_dir / "candidate_oof_metrics.csv", index=False)
    pd.DataFrame(candidate_fold_rows).to_csv(summary_dir / "candidate_fold_metrics.csv", index=False)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best masked DaT candidate without leaderboard data.")
    parser.add_argument("--runs_dir", default="runs/dat_parkinsons/resnet18_3d")
    parser.add_argument("--summary_table", default="")
    parser.add_argument("--best_config", default="")
    parser.add_argument("--expected_folds", type=int, default=5)
    parser.add_argument("--output", default="artifacts/dat_parkinsons/selected_model.json")
    parser.add_argument("--calibration", default="", help="Deprecated: Stage 1 calibration is not used for Stage 2 selection.")
    args = parser.parse_args()
    frozen = json.loads(Path(args.best_config).read_text(encoding="utf-8")) if args.best_config and Path(args.best_config).is_file() else None
    result = select_candidates(args.runs_dir, expected_folds=args.expected_folds, output_path=args.output, frozen_config=frozen)
    print(json.dumps({"best_overall": result["best_overall"], "best_masked": result["best_masked"]}, indent=2))


if __name__ == "__main__":
    main()
