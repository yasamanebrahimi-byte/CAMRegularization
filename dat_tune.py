"""Stage 1 DaT optimizer: no cutout, fixed stratified CV, OOF calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from dat_calibration import cross_fitted_calibration, fit_calibration, save_calibration, calibrated_probabilities
from dat_cv import make_protocol_group_folds, make_stratified_folds, save_fold_assignments
from dat_metrics import compute_binary_metrics
from dat_preprocessing import (
    DEFAULT_TARGET_SHAPE,
    DatDataset,
    default_preprocessing_config,
    estimate_target_spacing,
    load_dat_records,
    parse_target_shape,
    parse_target_spacing,
)
from dat_training import fit_dat_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune a no-cutout 3D DaT classifier using fixed CV folds.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="artifacts/dat_parkinsons/optimization")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--fold_scheme", choices=["stratified", "protocol_group"], default="stratified", help="Primary CV scheme; protocol_group is a shape/spacing robustness diagnostic, not a hospital-center split.")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--target_spacing", nargs=3, type=float, default=None)
    parser.add_argument("--target_shape", nargs=3, type=int, default=list(DEFAULT_TARGET_SHAPE))
    parser.add_argument("--intensity_lower_percentile", type=float, default=1.0)
    parser.add_argument("--intensity_upper_percentile", type=float, default=99.0)
    parser.add_argument("--foreground_threshold", type=float, default=0.0)
    parser.add_argument("--crop_margin_mm", type=float, default=8.0)
    parser.add_argument("--calibration", choices=["raw", "temperature"], default="temperature")
    parser.add_argument("--search_space_json", default="", help="Optional JSON object of parameter -> list values.")
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--augmentation", choices=["none", "mild"], default="none")
    return parser


def _load_search_space(path: str) -> dict[str, list[Any]]:
    if not path:
        return {
            "learning_rate": [1e-3, 3e-4, 1e-4],
            "weight_decay": [1e-4, 1e-5, 1e-3],
            "dropout": [0.0, 0.1, 0.25],
            "label_smoothing": [0.0, 0.02, 0.05],
            "optimizer": ["adamw", "sgd"],
        }
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("search_space_json must contain a non-empty JSON object.")
    return {str(key): list(values) for key, values in payload.items() if isinstance(values, list) and values}


def _trial_config(args, preprocessing: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    config = {
        "dataset": "dat_parkinsons",
        "model": "resnet18_3d",
        "spatial_dims": 3,
        "n_input_channels": 1,
        "num_classes": 2,
        "base_channels": int(args.base_channels),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "scheduler": "cosine",
        "min_lr": 1e-6,
        "momentum": 0.9,
        "nesterov": True,
        "adamw_betas": [0.9, 0.999],
        "amp": bool(args.amp),
        "spatial_augmentation": args.augmentation == "mild",
        "seed": int(args.seed),
        "calibration": args.calibration,
        "cutout_mode": "none",
        "cutout_m": 0,
        "preprocessing": preprocessing,
    }
    config.update(values)
    config["preprocessing"] = dict(preprocessing)
    if "target_shape" in values:
        config["preprocessing"]["target_shape"] = list(parse_target_shape(values["target_shape"]))
    if "target_spacing" in values:
        config["preprocessing"]["target_spacing"] = list(parse_target_spacing(values["target_spacing"]))
    return config


def _config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_trials(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot([row["trial_id"] for row in rows], [row["mean_oof_log_loss"] for row in rows], "o-")
    axis.set_xlabel("Trial")
    axis.set_ylabel("Mean OOF log loss")
    axis.set_title("DaT Stage 1 optimization")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = load_dat_records(args.data_dir)
    spacing = parse_target_spacing(args.target_spacing) if args.target_spacing else estimate_target_spacing(records)
    base_preprocessing = default_preprocessing_config(
        records,
        target_spacing=spacing,
        target_shape=parse_target_shape(args.target_shape),
        lower_percentile=args.intensity_lower_percentile,
        upper_percentile=args.intensity_upper_percentile,
        foreground_threshold=args.foreground_threshold,
        crop_margin_mm=args.crop_margin_mm,
    )
    fold_scheme = str(getattr(args, "fold_scheme", "stratified"))
    grouped = fold_scheme == "protocol_group"
    folds = (
        make_protocol_group_folds(records, n_splits=args.cv_folds, seed=args.seed)
        if grouped else make_stratified_folds(records, n_splits=args.cv_folds, seed=args.seed)
    )
    fold_path = output / "fold_assignments.json"
    save_fold_assignments(fold_path, records, folds, seed=args.seed, grouped=grouped)
    search_space = _load_search_space(args.search_space_json)
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any], np.ndarray, np.ndarray] | None = None
    trial_values = {
        trial_id: {key: rng.choice(options) for key, options in search_space.items()}
        for trial_id in range(int(args.trials))
    }

    # A completed trial is resumable because its config and OOF arrays are
    # written atomically before the trial table is updated.
    completed_trials: set[int] = set()
    trial_table = output / "cv_trials.csv"
    if trial_table.exists():
        with trial_table.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                trial_id = int(row["trial_id"])
                matches = sorted((output / "trials").glob(f"trial_{trial_id:03d}_*/trial_config.json"))
                if not matches:
                    continue
                trial_dir = matches[0].parent
                oof_path = trial_dir / "oof_logits.npz"
                if not oof_path.exists():
                    continue
                payload = np.load(oof_path)
                logits = payload["logits"]
                if not np.isfinite(logits).all():
                    continue
                config = json.loads(matches[0].read_text(encoding="utf-8"))
                normalized = {key: (float(value) if key.startswith("mean_") else value) for key, value in row.items()}
                normalized["trial_id"] = trial_id
                rows.append(normalized)
                completed_trials.add(trial_id)
                score = float(row["mean_oof_log_loss"])
                if best is None or score < best[0]:
                    best = (score, config, logits, payload["targets"])

    for trial_id in range(int(args.trials)):
        if trial_id in completed_trials:
            continue
        values = trial_values[trial_id]
        config = _trial_config(args, base_preprocessing, values)
        trial_dir = output / "trials" / f"trial_{trial_id:03d}_{_config_hash(config)}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "trial_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        oof_logits = np.full((len(records), 2), np.nan, dtype=np.float32)
        oof_targets = np.asarray([record.label for record in records], dtype=np.int64)
        for fold_index, (train_indices, validation_indices) in enumerate(folds):
            train_records = [records[index] for index in train_indices]
            validation_records = [records[index] for index in validation_indices]
            cache = output.parent / "cache" / "preprocessed" / _config_hash(config)
            train_dataset = DatDataset(train_records, config["preprocessing"], train=True, augment=bool(config.get("spatial_augmentation", False)), seed=args.seed + fold_index, cache_dir=cache)
            validation_dataset = DatDataset(validation_records, config["preprocessing"], train=False, augment=False, seed=args.seed + fold_index, cache_dir=cache)
            result = fit_dat_model(
                train_dataset,
                validation_dataset,
                config,
                seed=args.seed + 1000 * trial_id + fold_index,
                run_dir=trial_dir / f"fold_{fold_index}",
                max_train_batches=args.max_train_batches or None,
                # OOF predictions must cover every validation record; never
                # truncate validation while selecting a Stage 1 candidate.
                max_val_batches=None,
            )
            oof_logits[validation_indices] = result["best_logits"]

        if not np.isfinite(oof_logits).all():
            raise RuntimeError("Stage 1 produced incomplete OOF logits.")
        oof_metrics = compute_binary_metrics(oof_targets, logits=oof_logits)
        row = {
            "trial_id": trial_id,
            "config_hash": _config_hash(config),
            "mean_oof_log_loss": float(oof_metrics["log_loss"]),
            "mean_oof_auroc": float(oof_metrics["auroc"]),
            "mean_oof_brier_score": float(oof_metrics["brier_score"]),
            "mean_oof_accuracy": float(oof_metrics["accuracy"]),
        }
        rows.append(row)
        np.savez_compressed(trial_dir / "oof_logits.npz", logits=oof_logits, targets=oof_targets)
        if best is None or row["mean_oof_log_loss"] < best[0]:
            best = (row["mean_oof_log_loss"], config, oof_logits, oof_targets)
        _write_rows(output / "cv_trials.csv", rows)
        _plot_trials(rows, output / "optimization_summary.png")

    if best is None:
        raise RuntimeError("No Stage 1 trials were completed.")
    _, best_config, best_logits, best_targets = best
    best_config = dict(best_config)
    best_config.update({
        "selected_by": "mean_cross_validated_oof_log_loss",
        "cv_folds": int(args.cv_folds),
        "fold_scheme": fold_scheme,
        "fold_assignments": str(fold_path),
        "optimization_output_dir": str(output),
        "stage": 1,
    })
    (output / "best_config.json").write_text(json.dumps(best_config, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(output / "oof_predictions.npz", logits=best_logits, targets=best_targets)

    validation_folds = [validation for _, validation in folds]
    raw_metrics = compute_binary_metrics(best_targets, logits=best_logits)
    calibrated_probs, fold_calibrations = cross_fitted_calibration(
        best_logits, best_targets, validation_folds, method=args.calibration
    )
    calibrated_metrics = compute_binary_metrics(best_targets, probabilities=calibrated_probs)
    final_calibration = fit_calibration(best_logits, best_targets, method=args.calibration)
    final_calibration.update({
        "raw_oof_log_loss": raw_metrics["log_loss"],
        "calibrated_oof_log_loss": calibrated_metrics["log_loss"],
        "cross_fitted_fold_calibrations": fold_calibrations,
        "calibration_fit_data": "labeled_training_oof_predictions_only",
    })
    save_calibration(output / "calibration.json", final_calibration)
    (output / "calibration_report.json").write_text(
        json.dumps({"raw": raw_metrics, "calibrated_cross_fitted": calibrated_metrics}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"best_config": best_config, "raw_metrics": raw_metrics, "calibrated_metrics": calibrated_metrics}


def main() -> None:
    args = _parser().parse_args()
    result = run(args)
    print(json.dumps({"raw_oof_log_loss": result["raw_metrics"]["log_loss"], "calibrated_oof_log_loss": result["calibrated_metrics"]["log_loss"]}, indent=2))


if __name__ == "__main__":
    main()
