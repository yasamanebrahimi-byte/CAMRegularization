"""Stage 2 DaT masking grid with fold-specific no-cutout teachers."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from cutout import CutoutAugmentedDataset
from dat_cv import load_fold_assignments, make_stratified_folds, save_fold_assignments
from dat_model import build_resnet18_3d
from dat_preprocessing import DatDataset, load_dat_records
from dat_training import fit_dat_model


CONDITIONS = ("none", "random", "cam_low", "cam_high")
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
    parser.add_argument("--cam_layer", default="auto")
    parser.add_argument("--saliency_candidate_percent", type=float, default=10.0)
    parser.add_argument("--min_foreground_fraction", type=float, default=0.75)
    parser.add_argument("--cam_cache_dir", default="artifacts/dat_parkinsons/cam_cache")
    parser.add_argument("--preprocessed_cache_dir", default="artifacts/dat_parkinsons/cache/preprocessed")
    return parser


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _train_teacher(records, train_indices, preprocessing, config, fold, args):
    teacher_root = Path(args.cam_cache_dir).parent / "teachers" / f"fold_{fold}"
    teacher_root.mkdir(parents=True, exist_ok=True)
    teacher_dataset = DatDataset(
        [records[index] for index in train_indices], preprocessing,
        train=True, augment=bool(config.get("spatial_augmentation", False)),
        seed=args.seed + fold, cache_dir=args.preprocessed_cache_dir,
    )
    train_eval_dataset = DatDataset(
        [records[index] for index in train_indices], preprocessing,
        train=False, augment=False, seed=args.seed + fold, cache_dir=args.preprocessed_cache_dir,
    )
    teacher_config = deepcopy(config)
    teacher_config.update({"cutout_mode": "none", "cutout_m": 0, "stage": "stage2_teacher", "num_workers": args.num_workers})
    result = fit_dat_model(
        teacher_dataset, train_eval_dataset, teacher_config,
        seed=args.seed + 10000 + fold, run_dir=teacher_root,
        max_train_batches=args.max_train_batches or None,
        max_val_batches=args.max_val_batches or None,
    )
    checkpoint = teacher_root / "best_model.pt"
    return result["model"], checkpoint, _checkpoint_hash(checkpoint)


def _run_student(records, train_indices, validation_indices, preprocessing, config, fold, condition, m_value, fraction, args, teacher, teacher_hash):
    train_records = [records[index] for index in train_indices]
    val_records = [records[index] for index in validation_indices]
    base_train = DatDataset(
        train_records, preprocessing, train=True,
        augment=bool(config.get("spatial_augmentation", False)),
        seed=args.seed + fold, cache_dir=args.preprocessed_cache_dir,
    )
    val_dataset = DatDataset(
        val_records, preprocessing, train=False, augment=False,
        seed=args.seed + fold, cache_dir=args.preprocessed_cache_dir,
    )
    student_config = deepcopy(config)
    student_config.update({
        "stage": 2,
        "fold": int(fold),
        "condition": condition,
        "cutout_mode": condition,
        "cutout_m": int(m_value) if condition != "none" else 0,
        # Keep no-cutout baselines under the matched fraction key for paired
        # summaries; the fraction has no effect when cutout_m is zero.
        "cutout_fraction": float(fraction),
        "min_foreground_fraction": float(args.min_foreground_fraction),
        "saliency_candidate_percent": float(args.saliency_candidate_percent),
        "cam_layer": args.cam_layer,
        "teacher_checkpoint_sha256": teacher_hash if condition.startswith("cam_") else None,
        "num_workers": args.num_workers,
    })
    if condition == "none":
        train_dataset = base_train
    else:
        cache_settings = {
            "dataset": "dat_parkinsons",
            "student_model": "resnet18_3d",
            "teacher_model": "resnet18_3d",
            "teacher_checkpoint_sha256": teacher_hash,
            "cam_layer": args.cam_layer,
            "spatial_dims": 3,
            "preprocessing": preprocessing,
            "target_spacing": preprocessing["target_spacing"],
            "target_shape": preprocessing["target_shape"],
            "min_foreground_fraction": float(args.min_foreground_fraction),
        }
        train_dataset = CutoutAugmentedDataset(
            base_dataset=base_train,
            cutout_mode=condition,
            cutout_m=int(m_value),
            cutout_size=None,
            cutout_area=float(fraction),
            mean=(0.0,),
            std=(1.0,),
            seed=args.seed + fold,
            saliency_candidate_percent=args.saliency_candidate_percent,
            teacher_model=teacher if condition.startswith("cam_") else None,
            cam_layer=args.cam_layer,
            cam_cache_dir=args.cam_cache_dir if condition.startswith("cam_") else None,
            cam_cache_settings=cache_settings if condition.startswith("cam_") else None,
            min_foreground_fraction=args.min_foreground_fraction,
        )

    run_root = Path(args.output_dir) / f"fold_{fold}" / f"fraction_{fraction:.2f}" / (
        f"resnet18_3d_fold{fold}_{condition}" + (f"_M{m_value}_fraction{fraction:.2f}" if condition != "none" else "")
    )
    result = fit_dat_model(
        train_dataset, val_dataset, student_config,
        seed=args.seed + fold, run_dir=run_root,
        max_train_batches=args.max_train_batches or None,
        max_val_batches=args.max_val_batches or None,
    )
    # Patient-level OOF arrays stay outside runs/ so lightweight research run
    # folders remain safe to commit and no identifiers are materialized here.
    relative_key = f"fold_{fold}_{condition}_M{m_value}_fraction{fraction:.4f}"
    artifact_key = hashlib.sha256(relative_key.encode()).hexdigest()[:20]
    oof_dir = Path(args.output_dir).resolve().parents[2] / "artifacts" / "dat_parkinsons" / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_path = oof_dir / f"{artifact_key}.npz"
    np.savez_compressed(oof_path, logits=result["best_logits"], targets=result["best_targets"])
    config_path = run_root / "config.json"
    persisted_config = json.loads(config_path.read_text(encoding="utf-8"))
    persisted_config["oof_artifact"] = str(oof_path)
    config_path.write_text(json.dumps(persisted_config, indent=2, sort_keys=True), encoding="utf-8")
    return result


def run(args: argparse.Namespace) -> None:
    best_config = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
    records = load_dat_records(args.data_dir)
    preprocessing = best_config["preprocessing"]
    fold_path = Path(args.fold_assignments or best_config.get("fold_assignments", ""))
    if fold_path.is_file():
        folds = load_fold_assignments(fold_path, records)
    else:
        folds = make_stratified_folds(records, n_splits=int(best_config.get("cv_folds", 5)), seed=args.seed)
        fold_path = Path(args.output_dir) / "fold_assignments.json"
        save_fold_assignments(fold_path, records, folds, seed=args.seed)

    selected_folds = range(len(folds)) if args.fold < 0 else [args.fold]
    for fold in selected_folds:
        train_indices, validation_indices = folds[fold]
        teacher = None
        teacher_hash = None
        if any(condition.startswith("cam_") for condition in args.conditions):
            teacher, _teacher_checkpoint, teacher_hash = _train_teacher(
                records, train_indices, preprocessing, best_config, fold, args
            )
        for fraction in args.fractions:
            if "none" in args.conditions:
                _run_student(records, train_indices, validation_indices, preprocessing, best_config, fold, "none", 0, fraction, args, teacher, teacher_hash)
            for condition in args.conditions:
                if condition == "none":
                    continue
                for m_value in args.m_values:
                    _run_student(records, train_indices, validation_indices, preprocessing, best_config, fold, condition, m_value, fraction, args, teacher, teacher_hash)


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
