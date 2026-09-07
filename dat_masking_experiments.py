"""Leakage-safe, resumable DaT Stage 2 masking experiments."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from cutout import CutoutAugmentedDataset
from dat_cv import load_fold_assignments, make_stratified_folds, save_fold_assignments
from dat_preprocessing import DatDataset, load_dat_records
from dat_provenance import (
    REPO_ROOT,
    current_git_commit,
    fingerprint,
    portable_path,
    research_valid,
    sha256_file,
)
from dat_training import build_dat_model, fit_dat_model, fit_dat_model_fixed_epochs


CONDITIONS = ("none", "random", "cam_low", "cam_high")
MASKED_CONDITIONS = ("random", "cam_low", "cam_high")
DEFAULT_FRACTIONS = (0.05, 0.10, 0.20, 0.30)
DEFAULT_M = (4, 8)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen DaT Stage 2 cutout comparison.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--best_config", required=True)
    parser.add_argument("--output_dir", default="runs/dat_parkinsons/resnet18_3d")
    parser.add_argument("--fold_assignments", default="")
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--m_values", nargs="+", type=int, default=list(DEFAULT_M))
    parser.add_argument("--fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cam_layer", default="auto")
    parser.add_argument("--saliency_candidate_percent", type=float, default=10.0)
    parser.add_argument("--min_foreground_fraction", type=float, default=0.75)
    parser.add_argument("--cam_cache_dir", default="artifacts/dat_parkinsons/cam_cache")
    parser.add_argument("--preprocessed_cache_dir", default="artifacts/dat_parkinsons/cache/preprocessed")
    return parser


def cell_key(fold: int, condition: str, m_value: int, fraction: float) -> str:
    return f"fold={int(fold)}|condition={condition}|M={int(m_value)}|fraction={float(fraction):.4f}"


def expected_grid(
    n_folds: int, conditions: Iterable[str] = CONDITIONS,
    m_values: Iterable[int] = DEFAULT_M, fractions: Iterable[float] = DEFAULT_FRACTIONS,
) -> list[dict[str, Any]]:
    """Return the exact student grid; no-cutout is deliberately one cell/fold."""
    conditions = tuple(conditions)
    fractions = tuple(float(v) for v in fractions)
    m_values = tuple(int(v) for v in m_values)
    cells = []
    for fold in range(int(n_folds)):
        if "none" in conditions:
            cells.append({"fold": fold, "condition": "none", "M": 0, "fraction": 0.0,
                          "cell_key": cell_key(fold, "none", 0, 0.0)})
        for condition in MASKED_CONDITIONS:
            if condition not in conditions:
                continue
            for m_value in m_values:
                for fraction in fractions:
                    cells.append({"fold": fold, "condition": condition, "M": m_value,
                                  "fraction": fraction,
                                  "cell_key": cell_key(fold, condition, m_value, fraction)})
    return cells


def _checkpoint_hash(path: Path) -> str:
    return sha256_file(path)


def _portable_or_key(path: str | Path) -> str:
    try:
        return portable_path(path)
    except ValueError:
        return f"external/{Path(path).name}"


def _effective_num_workers(condition: str, args) -> int:
    """Resolve Stage 2 workers without allowing CAM saliency in workers.

    CAM saliency/window creation is deliberately kept in the main process. A
    precomputed cache can be added later, but the default research path must
    not depend on forked workers running teacher operations.
    """
    requested = max(0, int(getattr(args, "num_workers", 0) or 0))
    if str(condition).startswith("cam_") and requested > 0:
        print(
            "[Stage 2] CAM cutout training uses num_workers=0 because saliency "
            "and window caches are not proven complete."
        )
        return 0
    return requested


def _run_is_valid(run_root: str | Path, expected: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    """Validate completion evidence instead of treating directory existence as success."""
    root = Path(run_root)
    problems = []
    config_path = root / "config.json"
    metrics_path = root / "metrics.csv"
    if not config_path.is_file():
        problems.append("missing_config")
        return False, problems
    if not metrics_path.is_file():
        problems.append("missing_metrics")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        problems.append("invalid_config")
        return False, problems
    if expected:
        for key, value in expected.items():
            if key in {"oof_artifact", "completed"}:
                continue
            if config.get(key) != value:
                problems.append(f"config_mismatch:{key}")
    artifact = config.get("oof_artifact")
    artifact_path = (REPO_ROOT / artifact) if artifact and not Path(artifact).is_absolute() else Path(artifact) if artifact else None
    if artifact_path is None or not artifact_path.is_file():
        problems.append("missing_oof_artifact")
    else:
        try:
            payload = np.load(artifact_path)
            if not np.isfinite(payload["logits"]).all() or len(payload["logits"]) != len(payload["targets"]):
                problems.append("invalid_oof_artifact")
        except Exception:
            problems.append("invalid_oof_artifact")
    if config.get("completed") is not True:
        problems.append("incomplete_training_state")
    if config.get("checkpoint_selection") not in {"minimum_validation_log_loss", "final_scheduled_epoch"}:
        problems.append("missing_checkpoint_state")
    if not bool(config.get("research_valid", False)):
        problems.append("debug_or_truncated")
    if metrics_path.is_file():
        try:
            frame = pd.read_csv(metrics_path)
            numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
            if frame.empty or not np.isfinite(numeric).all():
                # Final models intentionally contain NaN validation columns,
                # but Stage 2 student runs must have finite trajectories.
                if config.get("stage") == 2:
                    problems.append("nonfinite_metrics")
            if config.get("stage") == 2:
                if "best_epoch" not in config or "epochs_completed" not in config:
                    problems.append("missing_epoch_completion_metadata")
                elif int(config["epochs_completed"]) != int(frame["epoch"].max()):
                    problems.append("truncated_or_inconsistent_epoch_metadata")
        except Exception:
            problems.append("invalid_metrics")
    return not problems, problems


def _train_teacher(records, train_indices, preprocessing, config, fold, args, fold_hash=None):
    teacher_root = Path(args.cam_cache_dir).parent / "teachers" / f"fold_{fold}"
    teacher_root.mkdir(parents=True, exist_ok=True)
    selected_stage1_fingerprint = config.get("config_fingerprint", fingerprint(config))
    preprocessing_fingerprint = config.get("preprocessing_fingerprint", fingerprint(preprocessing))
    fold_assignment_fingerprint = fold_hash or config.get("fold_assignment_fingerprint")
    if not fold_assignment_fingerprint:
        fold_assignment_fingerprint = fingerprint({"fold": int(fold)})
    teacher_config = deepcopy(config)
    epoch_budget = int(config.get("final_training_epochs", config.get("epochs", 100)))
    teacher_config.update({
        "cutout_mode": "none", "cutout_m": 0, "stage": "stage2_teacher", "fold": int(fold),
        "epochs": epoch_budget, "final_training_epochs": epoch_budget,
        "stage1_final_training_epochs": epoch_budget,
        "selected_stage1_config_fingerprint": selected_stage1_fingerprint,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "fold_assignment_fingerprint": fold_assignment_fingerprint,
        "teacher_recipe_provenance": "stage1_selected_unmasked_recipe_fixed_epoch_teacher",
        "teacher_lineage": "selected_stage1_config",
        "num_workers": args.num_workers, "max_train_batches": int(args.max_train_batches or 0),
        "max_val_batches": 0, "debug": bool(getattr(args, "debug", False) or args.max_train_batches or args.max_val_batches),
        "research_valid": research_valid(max_train_batches=args.max_train_batches or 0,
                                          max_val_batches=0, debug=getattr(args, "debug", False)),
        "completed": False,
        "checkpoint_selection": "final_scheduled_epoch",
    })
    checkpoint = teacher_root / "final_model.pt"
    existing_config_path = teacher_root / "config.json"
    if checkpoint.is_file() and existing_config_path.is_file():
        try:
            existing = json.loads(existing_config_path.read_text(encoding="utf-8"))
            expected = {
                "stage": "stage2_teacher", "fold": int(fold), "epochs": epoch_budget,
                "final_training_epochs": epoch_budget,
                "stage1_final_training_epochs": epoch_budget,
                "selected_stage1_config_fingerprint": selected_stage1_fingerprint,
                "preprocessing_fingerprint": preprocessing_fingerprint,
                "fold_assignment_fingerprint": fold_assignment_fingerprint,
                "research_valid": teacher_config["research_valid"],
                "checkpoint_selection": "final_scheduled_epoch",
                "teacher_recipe_provenance": "stage1_selected_unmasked_recipe_fixed_epoch_teacher",
                "num_workers": int(args.num_workers or 0),
                "max_train_batches": int(args.max_train_batches or 0),
                "max_val_batches": 0,
                "debug": bool(getattr(args, "debug", False) or args.max_train_batches or args.max_val_batches),
            }
            matches = all(existing.get(key) == value for key, value in expected.items())
            matches = matches and existing.get("completed") is True
            if matches:
                teacher_hash = _checkpoint_hash(checkpoint)
                matches = existing.get("teacher_checkpoint_sha256") == teacher_hash
            if matches:
                teacher = build_dat_model(config)
                payload = torch.load(checkpoint, map_location="cpu")
                state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
                teacher.load_state_dict({str(k).removeprefix("module."): v for k, v in state.items()}, strict=True)
                active_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                teacher.to(active_device)
                teacher.eval()
                for parameter in teacher.parameters():
                    parameter.requires_grad_(False)
                print(f"[Stage 2] resume: reusing valid fold {fold} CAM teacher ({teacher_hash[:12]})")
                return teacher, checkpoint, teacher_hash
        except Exception:
            pass
    teacher_dataset = DatDataset(
        [records[index] for index in train_indices], preprocessing,
        train=True, augment=bool(config.get("spatial_augmentation", False)),
        seed=args.seed + fold, cache_dir=args.preprocessed_cache_dir,
    )
    # A Stage 2 teacher never receives the outer validation dataset.  It is
    # also never evaluated on its training samples for checkpoint selection.
    result = fit_dat_model_fixed_epochs(
        teacher_dataset, teacher_config, seed=args.seed + 10000 + fold,
        run_dir=teacher_root, epochs=epoch_budget,
        max_train_batches=args.max_train_batches or None,
    )
    checkpoint = teacher_root / "final_model.pt"
    teacher_hash = _checkpoint_hash(checkpoint)
    persisted = json.loads((teacher_root / "config.json").read_text(encoding="utf-8"))
    persisted.update({
        "teacher_checkpoint_sha256": teacher_hash, "outer_train_record_count": len(train_indices),
        "outer_validation_used_for_teacher": False, "teacher_checkpoint_selection": "frozen_stage1_epoch_budget",
        "selected_stage1_config_fingerprint": selected_stage1_fingerprint,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "fold_assignment_fingerprint": fold_assignment_fingerprint,
        "fold": int(fold), "final_training_epochs": epoch_budget,
        "stage1_final_training_epochs": epoch_budget,
        "teacher_recipe_provenance": "stage1_selected_unmasked_recipe_fixed_epoch_teacher",
        "research_valid": teacher_config["research_valid"], "completed": True,
        "checkpoint_selection": "final_scheduled_epoch",
    })
    (teacher_root / "config.json").write_text(json.dumps(persisted, indent=2, sort_keys=True), encoding="utf-8")
    teacher = result["model"]
    active_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(active_device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher, checkpoint, teacher_hash


def _student_expected_config(config, fold, condition, m_value, fraction, args, teacher_hash, fold_hash):
    student_epoch_budget = int(config.get("epochs", 100))
    stage1_final_epoch_budget = int(config.get("final_training_epochs", student_epoch_budget))
    effective_workers = _effective_num_workers(condition, args)
    return {
        "stage": 2, "fold": int(fold), "condition": condition,
        "cutout_mode": condition, "cutout_m": int(m_value) if condition != "none" else 0,
        "cutout_fraction": float(fraction), "teacher_checkpoint_sha256": teacher_hash if condition.startswith("cam_") else None,
        "selected_stage1_config_fingerprint": config.get("config_fingerprint", fingerprint(config)),
        "preprocessing_fingerprint": config.get("preprocessing_fingerprint", fingerprint(config["preprocessing"])),
        "fold_assignment_fingerprint": fold_hash, "cam_layer": args.cam_layer,
        "saliency_candidate_percent": float(args.saliency_candidate_percent),
        "min_foreground_fraction": float(args.min_foreground_fraction),
        "epochs": student_epoch_budget, "student_max_cv_epochs": student_epoch_budget,
        "stage1_final_training_epochs": stage1_final_epoch_budget,
        "final_training_epochs": stage1_final_epoch_budget,
        "early_stopping_patience": int(config.get("patience", 15)),
        "student_seed_policy": "base_seed_plus_fold",
        "student_model": str(config.get("model", "resnet18_3d")),
        "seed": int(args.seed + fold), "num_workers": effective_workers,
        "requested_num_workers": int(getattr(args, "num_workers", 0) or 0),
        "max_train_batches": int(args.max_train_batches or 0),
        "max_val_batches": int(args.max_val_batches or 0), "debug": bool(getattr(args, "debug", False)),
        "research_valid": research_valid(max_train_batches=args.max_train_batches or 0,
                                          max_val_batches=args.max_val_batches or 0, debug=getattr(args, "debug", False)),
        "git_commit": current_git_commit(),
    }


def _run_student(records, train_indices, validation_indices, preprocessing, config, fold, condition, m_value, fraction, args, teacher, teacher_hash, fold_hash=None):
    fold_hash = fold_hash or config.get("fold_assignment_fingerprint", fingerprint({"fold": int(fold)}))
    expected = _student_expected_config(config, fold, condition, m_value, fraction, args, teacher_hash, fold_hash)
    if condition == "none":
        run_root = Path(args.output_dir) / f"fold_{fold}" / "fraction_0.00" / f"resnet18_3d_fold{fold}_none"
    else:
        run_root = Path(args.output_dir) / f"fold_{fold}" / f"fraction_{fraction:.2f}" / f"resnet18_3d_fold{fold}_{condition}_M{m_value}_fraction{fraction:.2f}"
    valid, problems = _run_is_valid(run_root, expected)
    if valid:
        print(f"[Stage 2] resume: skipping valid {expected['condition']} fold {fold} M{m_value} fraction {fraction:.2f}")
        return {"run_dir": run_root, "skipped": True}
    if problems and run_root.exists():
        print(f"[Stage 2] resume: rerunning fold {fold} {condition} M{m_value} fraction {fraction:.2f} ({', '.join(problems)})")
    train_records = [records[index] for index in train_indices]
    val_records = [records[index] for index in validation_indices]
    base_train = DatDataset(
        train_records, preprocessing, train=True,
        augment=bool(config.get("spatial_augmentation", False)), seed=args.seed + fold,
        cache_dir=args.preprocessed_cache_dir,
    )
    val_dataset = DatDataset(val_records, preprocessing, train=False, augment=False,
                             seed=args.seed + fold, cache_dir=args.preprocessed_cache_dir)
    student_config = deepcopy(config)
    student_config.update(expected)
    if condition == "none":
        train_dataset = base_train
    else:
        cache_settings = {
            "dataset": "dat_parkinsons", "student_model": "resnet18_3d", "teacher_model": "resnet18_3d",
            "teacher_checkpoint_sha256": teacher_hash, "cam_layer": args.cam_layer,
            "spatial_dims": 3, "preprocessing": preprocessing,
            "min_foreground_fraction": float(args.min_foreground_fraction),
        }
        train_dataset = CutoutAugmentedDataset(
            base_dataset=base_train, cutout_mode=condition, cutout_m=int(m_value), cutout_size=None,
            cutout_area=float(fraction), mean=(0.0,), std=(1.0,), seed=args.seed + fold,
            saliency_candidate_percent=args.saliency_candidate_percent,
            teacher_model=teacher if condition.startswith("cam_") else None,
            cam_layer=args.cam_layer, cam_cache_dir=args.cam_cache_dir if condition.startswith("cam_") else None,
            cam_cache_settings=cache_settings if condition.startswith("cam_") else None,
            min_foreground_fraction=args.min_foreground_fraction,
        )
    result = fit_dat_model(
        train_dataset, val_dataset, student_config, seed=args.seed + fold, run_dir=run_root,
        max_train_batches=args.max_train_batches or None,
        # OOF selection always evaluates the complete outer validation fold.
        max_val_batches=None,
    )
    artifact_id = fingerprint({"cell": cell_key(fold, condition, m_value, fraction),
                               "stage1": expected["selected_stage1_config_fingerprint"], "seed": args.seed + fold})
    oof_path = REPO_ROOT / "artifacts" / "dat_parkinsons" / "oof" / f"stage2_{artifact_id}.npz"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(oof_path, logits=result["best_logits"], targets=result["best_targets"], fold=np.full(len(result["best_targets"]), fold, dtype=np.int64))
    persisted = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    persisted.update({
        "oof_artifact": _portable_or_key(oof_path), "completed": True,
        "epochs_completed": int(result["epochs_completed"]), "best_epoch": int(result["best_epoch"]),
        "checkpoint_selection": "minimum_validation_log_loss",
        "minimum_validation_log_loss": float(result["best_metrics"]["log_loss"]),
        "accuracy_at_minimum_validation_log_loss": float(result["best_metrics"]["accuracy"]),
        "auroc_at_minimum_validation_log_loss": float(result["best_metrics"]["auroc"]),
        "brier_at_minimum_validation_log_loss": float(result["best_metrics"]["brier_score"]),
        "ece_at_minimum_validation_log_loss": float(result["best_metrics"]["ece"]),
    })
    (run_root / "config.json").write_text(json.dumps(persisted, indent=2, sort_keys=True), encoding="utf-8")
    return {"run_dir": run_root, "skipped": False, "result": result}


def run(args: argparse.Namespace) -> dict[str, Any]:
    best_config = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
    records = load_dat_records(args.data_dir)
    preprocessing = best_config["preprocessing"]
    fold_path = Path(args.fold_assignments or "")
    if not fold_path.is_file():
        configured = best_config.get("fold_assignments", "")
        fold_path = (REPO_ROOT / configured) if configured and not Path(configured).is_absolute() else Path(configured)
    if fold_path.is_file():
        folds = load_fold_assignments(fold_path, records)
    else:
        folds = make_stratified_folds(records, n_splits=int(best_config.get("cv_folds", 5)), seed=args.seed)
        fold_path = Path(args.output_dir) / "fold_assignments.json"
        save_fold_assignments(fold_path, records, folds, seed=args.seed)
    fold_hash = best_config.get("fold_assignment_fingerprint", fingerprint([[list(a), list(b)] for a, b in folds]))
    selected_folds = range(len(folds)) if args.fold < 0 else [args.fold]
    results = []
    for fold in selected_folds:
        train_indices, validation_indices = folds[fold]
        teacher = None
        teacher_hash = None
        if any(condition.startswith("cam_") for condition in args.conditions):
            teacher, _teacher_checkpoint, teacher_hash = _train_teacher(
                records, train_indices, preprocessing, best_config, fold, args, fold_hash=fold_hash
            )
        if "none" in args.conditions:
            # Fraction is not a no-cutout hyperparameter.  One baseline is
            # reused logically in summaries for each masked fraction.
            results.append(_run_student(records, train_indices, validation_indices, preprocessing, best_config, fold, "none", 0, 0.0, args, teacher, teacher_hash, fold_hash))
        for condition in MASKED_CONDITIONS:
            if condition not in args.conditions:
                continue
            for m_value in args.m_values:
                for fraction in args.fractions:
                    results.append(_run_student(records, train_indices, validation_indices, preprocessing, best_config, fold, condition, m_value, fraction, args, teacher, teacher_hash, fold_hash))
    return {"folds": len(folds), "results": results, "expected_cells": expected_grid(len(folds), args.conditions, args.m_values, args.fractions)}


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
