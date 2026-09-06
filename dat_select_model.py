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
from dat_provenance import REPO_ROOT, fingerprint, portable_path


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
        if frozen_config:
            expected_lineage = {
                "selected_stage1_config_fingerprint": frozen_config.get("config_fingerprint"),
                "preprocessing_fingerprint": frozen_config.get("preprocessing_fingerprint"),
                "fold_assignment_fingerprint": frozen_config.get("fold_assignment_fingerprint"),
                "final_training_epochs": frozen_config.get("final_training_epochs"),
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
    for (condition, m_value, fraction), entries in sorted(grouped.items()):
        entries = sorted(entries, key=lambda item: int(item[0].get("fold", -1)))
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
            "condition": condition, "M": m_value, "fraction": fraction,
            "selected_stage1_config_fingerprint": entries[0][0].get("selected_stage1_config_fingerprint"),
            "preprocessing_fingerprint": entries[0][0].get("preprocessing_fingerprint"),
            "raw_oof_log_loss": float(result["raw_metrics"]["log_loss"]),
            "cross_fitted_calibrated_oof_log_loss": float(result["cross_fitted_metrics"]["log_loss"]),
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
    if not candidates:
        raise ValueError("No valid Stage 2 candidates were found.")
    best_overall = min(candidates, key=lambda row: (row["selection_score"], row["condition"], row["M"], row["fraction"]))
    masked = [row for row in candidates if row["condition"] in {"random", "cam_low", "cam_high"}]
    if not masked:
        raise ValueError("Stage 2 requires at least one masked candidate for Submission #2.")
    best_masked = min(masked, key=lambda row: (row["selection_score"], row["condition"], row["M"], row["fraction"]))
    payload = {
        "selection_basis": "cross_fitted_calibrated_oof_log_loss",
        "best_overall": best_overall, "best_masked": best_masked,
        # Compatibility for older callers: selected is always the masked
        # competition candidate, never the no-cutout baseline.
        "selected": best_masked, "candidates": candidates,
        "integrity_report": _portable_or_key(summary_dir / "integrity_report.json"),
    }
    pd.DataFrame([{
        "condition": row["condition"], "M": row["M"], "fraction": row["fraction"],
        "raw_oof_log_loss": row["raw_oof_log_loss"],
        "cross_fitted_calibrated_oof_log_loss": row["cross_fitted_calibrated_oof_log_loss"],
        "final_fitted_calibration_method": row["final_fitted_calibration_method"],
        "final_fitted_temperature": row["final_fitted_temperature"],
        "n_oof_samples": row["n_oof_samples"], "n_cv_folds": row["n_cv_folds"],
    } for row in candidates]).to_csv(summary_dir / "candidate_oof_metrics.csv", index=False)
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
