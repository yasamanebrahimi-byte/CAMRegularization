"""Stage 1 DaT optimizer: no cutout, fixed stratified CV, OOF calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dat_calibration import cross_fitted_calibration, fit_calibration, save_calibration, calibrated_probabilities
from dat_cv import make_protocol_group_folds, make_stratified_folds, save_fold_assignments
from dat_metrics import compute_binary_metrics
from dat_provenance import current_git_commit, fingerprint, median_round_half_up, portable_path, research_valid
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
    parser.add_argument("--trials", type=int, default=10)
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
    parser.add_argument("--debug", action="store_true", help="Mark the run as smoke/debug; it cannot be used for research selection.")
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
        "max_train_batches": int(getattr(args, "max_train_batches", 0) or 0),
        "max_val_batches": int(getattr(args, "max_val_batches", 0) or 0),
        "debug": bool(getattr(args, "debug", False)),
    }
    config.update(values)
    config["preprocessing"] = dict(preprocessing)
    config["research_valid"] = research_valid(
        max_train_batches=config["max_train_batches"],
        max_val_batches=config["max_val_batches"],
        debug=config["debug"],
    )
    if "target_shape" in values:
        config["preprocessing"]["target_shape"] = list(parse_target_shape(values["target_shape"]))
    if "target_spacing" in values:
        config["preprocessing"]["target_spacing"] = list(parse_target_spacing(values["target_spacing"]))
    config["preprocessing_fingerprint"] = fingerprint(config["preprocessing"])
    return config


def _config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def _canonical_json(value: Any) -> str:
    """Return a stable representation for duplicate-search detection."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _search_config_key(values: dict[str, Any], keys: tuple[str, ...]) -> str:
    return _canonical_json({key: values.get(key) for key in keys})


def generate_unique_trial_values(
    search_space: dict[str, list[Any]],
    requested_trials: int,
    seed: int,
    *,
    reserved_values: list[dict[str, Any]] | None = None,
) -> tuple[dict[int, dict[str, Any]], int]:
    """Generate deterministic, non-repeating finite-search configurations.

    ``reserved_values`` contains configurations already completed in a resumed
    run. They are excluded from newly generated trials so a requested count
    always refers to unique configurations, including resumed work.
    """
    requested_trials = int(requested_trials)
    if requested_trials < 0:
        raise ValueError("--trials must be non-negative.")
    keys = tuple(search_space)
    if not keys or any(not values for values in search_space.values()):
        raise ValueError("The Stage 1 search space must contain non-empty value lists.")
    unique_options = []
    for key in keys:
        seen = set()
        options = []
        for value in search_space[key]:
            marker = _canonical_json(value)
            if marker not in seen:
                seen.add(marker)
                options.append(value)
        unique_options.append(options)
    combinations = [dict(zip(keys, values)) for values in itertools.product(*unique_options)]
    rng = random.Random(int(seed))
    rng.shuffle(combinations)
    reserved = {
        _search_config_key(values, keys)
        for values in (reserved_values or [])
        if all(key in values for key in keys)
    }
    available = [values for values in combinations if _search_config_key(values, keys) not in reserved]
    total_unique = len(combinations)
    if requested_trials > total_unique:
        print(
            f"[Stage 1] requested {requested_trials} trials but the finite search space has "
            f"only {total_unique} unique configurations; capping at {total_unique}."
        )
        requested_trials = total_unique
    if requested_trials > len(available):
        print(
            f"[Stage 1] only {len(available)} unique configurations remain after resumed trials; "
            f"capping new trials at {len(available)}."
        )
        requested_trials = len(available)
    result: dict[int, dict[str, Any]] = {}
    available_index = 0
    used_keys: set[str] = set()
    for trial_id in range(requested_trials):
        while available_index < len(available) and _search_config_key(available[available_index], keys) in used_keys:
            available_index += 1
        if available_index >= len(available):
            raise RuntimeError(
                "The requested Stage 1 trial count cannot be satisfied with unique configurations "
                "after accounting for resumed trials; reduce --trials or rerun Stage 1."
            )
        result[trial_id] = dict(available[available_index])
        used_keys.add(_search_config_key(result[trial_id], keys))
        available_index += 1
    return result, requested_trials


def _fold_ids_from_folds(folds: list[tuple[list[int], list[int]]], n_samples: int) -> np.ndarray:
    fold_ids = np.full(int(n_samples), -1, dtype=np.int64)
    for fold, (_train, validation) in enumerate(folds):
        validation = np.asarray(validation, dtype=np.int64)
        if validation.size == 0 or np.any(validation < 0) or np.any(validation >= n_samples):
            raise ValueError("Stage 1 validation folds contain invalid indices.")
        if np.any(fold_ids[validation] != -1):
            raise ValueError("Stage 1 validation folds overlap.")
        fold_ids[validation] = int(fold)
    if np.any(fold_ids < 0):
        raise ValueError("Stage 1 validation folds do not cover every OOF observation.")
    return fold_ids


def evaluate_stage1_oof(
    logits: np.ndarray,
    targets: np.ndarray,
    fold_ids: np.ndarray,
    *,
    calibration_method: str = "temperature",
) -> dict[str, Any]:
    """Score a Stage 1 trial with raw and leakage-safe calibrated OOF metrics."""
    logits = np.asarray(logits)
    targets = np.asarray(targets, dtype=np.int64)
    fold_ids = np.asarray(fold_ids, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[1] != 2 or len(logits) != len(targets) or len(targets) != len(fold_ids):
        raise ValueError("Stage 1 OOF logits, targets, and fold_ids must be aligned as [N,2], [N], [N].")
    if len(targets) == 0 or not np.isfinite(logits).all() or np.any(fold_ids < 0):
        raise ValueError("Stage 1 OOF artifacts must be finite, non-empty, and completely assigned to folds.")
    unique_folds = sorted(set(int(value) for value in fold_ids.tolist()))
    validation_folds = [np.flatnonzero(fold_ids == fold).tolist() for fold in unique_folds]
    raw_metrics = compute_binary_metrics(targets, logits=logits)
    cross_fitted_probs, fold_calibrations = cross_fitted_calibration(
        logits, targets, validation_folds, method=calibration_method
    )
    cross_fitted_metrics = compute_binary_metrics(targets, probabilities=cross_fitted_probs)
    final_calibration = fit_calibration(logits, targets, method=calibration_method)
    # Keep the all-OOF deployment fit separate from the cross-fitted estimate.
    final_probabilities = calibrated_probabilities(logits, final_calibration)
    final_all_oof_metrics = compute_binary_metrics(targets, probabilities=final_probabilities)
    return {
        "raw_metrics": raw_metrics,
        "cross_fitted_metrics": cross_fitted_metrics,
        "final_all_oof_metrics": final_all_oof_metrics,
        "fold_calibrations": fold_calibrations,
        "final_calibration": final_calibration,
        "selection_score": float(cross_fitted_metrics["log_loss"]),
    }


def select_best_stage1_trial(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select only by the leakage-safe calibrated OOF log-loss criterion."""
    if not rows:
        raise ValueError("No completed Stage 1 trials are available.")
    missing = [row.get("trial_id") for row in rows if row.get("cross_fitted_calibrated_oof_log_loss") in (None, "")]
    if missing:
        raise RuntimeError(
            "Completed Stage 1 trials are missing cross_fitted_calibrated_oof_log_loss "
            f"for trial(s) {missing}; rerun Stage 1 rather than comparing incompatible metrics."
        )
    return min(rows, key=lambda row: (float(row["cross_fitted_calibrated_oof_log_loss"]), int(row["trial_id"])))


def _stage1_trial_row(
    trial_id: int,
    config: dict[str, Any],
    values: dict[str, Any],
    evaluation: dict[str, Any],
    calibration_method: str,
) -> dict[str, Any]:
    raw = evaluation["raw_metrics"]
    cross_fitted = evaluation["cross_fitted_metrics"]
    row: dict[str, Any] = {
        "trial_id": int(trial_id),
        "config_hash": _config_hash(config),
        "calibration_method": str(calibration_method),
        "raw_oof_log_loss": float(raw["log_loss"]),
        "raw_oof_auroc": float(raw["auroc"]),
        "raw_oof_brier_score": float(raw["brier_score"]),
        "raw_oof_accuracy": float(raw["accuracy"]),
        "cross_fitted_calibrated_oof_log_loss": float(cross_fitted["log_loss"]),
        "cross_fitted_calibrated_oof_auroc": float(cross_fitted["auroc"]),
        "cross_fitted_calibrated_oof_brier_score": float(cross_fitted["brier_score"]),
        "cross_fitted_calibrated_oof_accuracy": float(cross_fitted["accuracy"]),
        "cross_fitted_calibrated_oof_ece": float(cross_fitted["ece"]),
        "final_all_oof_calibrated_log_loss": float(evaluation["final_all_oof_metrics"]["log_loss"]),
        "selection_score": float(evaluation["selection_score"]),
        # Compatibility aliases. They are explicitly raw/uncalibrated.
        "mean_oof_log_loss": float(raw["log_loss"]),
        "mean_oof_auroc": float(raw["auroc"]),
        "mean_oof_brier_score": float(raw["brier_score"]),
        "mean_oof_accuracy": float(raw["accuracy"]),
    }
    for key, value in values.items():
        row[key] = _canonical_json(value) if isinstance(value, (dict, list, tuple)) else value
    for key in ("fold_assignment_fingerprint", "preprocessing_fingerprint"):
        if key in config:
            row[key] = config[key]
    return row


def _load_resumable_stage1_trial(
    trial_dir: Path,
    config: dict[str, Any],
    expected_targets: np.ndarray,
    current_fold_ids: np.ndarray,
    *,
    calibration_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and re-score a completed trial using the current selection rule."""
    oof_path = trial_dir / "oof_logits.npz"
    if not oof_path.is_file():
        raise RuntimeError(
            f"Cannot safely resume Stage 1 trial in {trial_dir}: missing persisted OOF logits; rerun Stage 1."
        )
    try:
        with np.load(oof_path) as payload:
            logits = np.asarray(payload["logits"])
            targets = np.asarray(payload["targets"], dtype=np.int64)
            if "fold_ids" in payload:
                fold_ids = np.asarray(payload["fold_ids"], dtype=np.int64)
            elif "fold" in payload:
                fold_ids = np.asarray(payload["fold"], dtype=np.int64)
            else:
                # Historical Stage 1 artifacts did not persist fold IDs. It is
                # safe to reconstruct them only when the current OOF order and
                # labels match exactly; otherwise comparison is invalid.
                fold_ids = current_fold_ids.copy()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot safely resume Stage 1 trial in {trial_dir}: invalid OOF artifact ({exc}); rerun Stage 1."
        ) from exc
    if logits.shape != (len(expected_targets), 2) or targets.shape != expected_targets.shape:
        raise RuntimeError(
            f"Cannot safely resume Stage 1 trial in {trial_dir}: OOF shape is incompatible with the current "
            "dataset/folds; rerun Stage 1."
        )
    if not np.array_equal(targets, expected_targets):
        raise RuntimeError(
            f"Cannot safely resume Stage 1 trial in {trial_dir}: persisted OOF targets do not match the current "
            "training order; rerun Stage 1."
        )
    if fold_ids.shape != current_fold_ids.shape or not np.array_equal(fold_ids, current_fold_ids):
        raise RuntimeError(
            f"Cannot safely resume Stage 1 trial in {trial_dir}: persisted fold membership differs from the "
            "current fold assignment; rerun Stage 1."
        )
    persisted_method = str(config.get("calibration", calibration_method)).lower()
    if persisted_method != str(calibration_method).lower():
        raise RuntimeError(
            f"Cannot safely resume Stage 1 trial in {trial_dir}: calibration method changed from "
            f"'{persisted_method}' to '{calibration_method}'; rerun Stage 1."
        )
    if not np.isfinite(logits).all():
        raise RuntimeError(
            f"Cannot safely resume Stage 1 trial in {trial_dir}: OOF logits are non-finite; rerun Stage 1."
        )
    evaluation = evaluate_stage1_oof(logits, targets, fold_ids, calibration_method=calibration_method)
    return logits, targets, fold_ids, evaluation


def _portable_or_key(path: str | Path) -> str:
    try:
        return portable_path(path)
    except ValueError:
        return f"external/{Path(path).name}"


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_trials(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    trial_ids = [row["trial_id"] for row in rows]
    axis.plot(trial_ids, [row["raw_oof_log_loss"] for row in rows], "o-", label="Raw OOF log loss")
    axis.plot(
        trial_ids,
        [row["cross_fitted_calibrated_oof_log_loss"] for row in rows],
        "s-",
        label="Cross-fitted calibrated OOF log loss (selection)",
    )
    if rows:
        selected = select_best_stage1_trial(rows)
        axis.scatter(
            [selected["trial_id"]], [selected["cross_fitted_calibrated_oof_log_loss"]],
            marker="*", s=140, zorder=4, label="Selected trial",
        )
    axis.set_xlabel("Trial")
    axis.set_ylabel("OOF log loss (lower is better)")
    axis.set_title("DaT Stage 1 optimization: calibrated OOF loss selects the trial")
    axis.grid(alpha=0.25)
    axis.legend()
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
    fold_ids = _fold_ids_from_folds(folds, len(records))
    fold_assignment_fingerprint = fingerprint([[list(a), list(b)] for a, b in folds])
    search_space = _load_search_space(args.search_space_json)
    rows: list[dict[str, Any]] = []
    expected_targets = np.asarray([record.label for record in records], dtype=np.int64)
    completed_trials: dict[int, dict[str, Any]] = {}
    trial_table = output / "cv_trials.csv"
    if trial_table.exists():
        with trial_table.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                trial_id = int(row["trial_id"])
                matches = sorted((output / "trials").glob(f"trial_{trial_id:03d}_*/trial_config.json"))
                if not matches:
                    continue
                trial_dir = matches[0].parent
                config = json.loads(matches[0].read_text(encoding="utf-8"))
                if trial_id in completed_trials:
                    raise RuntimeError(f"Duplicate completed Stage 1 trial ID {trial_id} in {trial_table}; repair or rerun Stage 1.")
                logits, targets, persisted_fold_ids, evaluation = _load_resumable_stage1_trial(
                    trial_dir, config, expected_targets, fold_ids, calibration_method=args.calibration
                )
                values = {key: config[key] for key in search_space if key in config}
                if len(values) != len(search_space):
                    raise RuntimeError(
                        f"Cannot safely resume Stage 1 trial {trial_id}: persisted hyperparameters are incomplete; rerun Stage 1."
                    )
                normalized = _stage1_trial_row(trial_id, config, values, evaluation, args.calibration)
                completed_trials[trial_id] = {
                    "row": normalized, "config": config, "logits": logits,
                    "targets": targets, "fold_ids": persisted_fold_ids,
                    "evaluation": evaluation, "values": values,
                }

    completed_values = [item["values"] for item in completed_trials.values()]
    completed_keys = [_search_config_key(values, tuple(search_space)) for values in completed_values]
    if len(completed_keys) != len(set(completed_keys)):
        raise RuntimeError(
            "Resumed Stage 1 artifacts contain duplicate hyperparameter configurations. "
            "Rerun Stage 1 so requested trials are unique."
        )
    requested_trials = int(args.trials)
    new_trial_ids = [trial_id for trial_id in range(max(0, requested_trials)) if trial_id not in completed_trials]
    new_values, effective_new_count = generate_unique_trial_values(
        search_space, len(new_trial_ids), args.seed, reserved_values=completed_values
    )
    if requested_trials > 0 and len(new_trial_ids) != effective_new_count:
        new_trial_ids = new_trial_ids[:effective_new_count]
    trial_values = {trial_id: new_values[index] for index, trial_id in enumerate(new_trial_ids)}
    for trial_id in sorted(completed_trials):
        rows.append(completed_trials[trial_id]["row"])

    trial_records = dict(completed_trials)
    for trial_id in new_trial_ids:
        values = trial_values[trial_id]
        config = _trial_config(args, base_preprocessing, values)
        config.update({
            "cv_folds": int(args.cv_folds), "fold_scheme": fold_scheme,
            "fold_assignment_fingerprint": fold_assignment_fingerprint,
            "selection_objective": "cross_fitted_calibrated_oof_log_loss",
            "selection_lower_is_better": True,
        })
        trial_dir = output / "trials" / f"trial_{trial_id:03d}_{_config_hash(config)}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "trial_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        oof_logits = np.full((len(records), 2), np.nan, dtype=np.float32)
        oof_targets = expected_targets.copy()
        fold_rows: list[dict[str, Any]] = []
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
            fold_rows.append({
                "fold": int(fold_index),
                "best_epoch": int(result["best_epoch"]),
                "minimum_validation_log_loss": float(result["best_metrics"]["log_loss"]),
                "epoch_at_minimum_validation_log_loss": int(result["best_epoch"]),
                "accuracy_at_minimum_validation_log_loss": float(result["best_metrics"]["accuracy"]),
                "auroc_at_minimum_validation_log_loss": float(result["best_metrics"]["auroc"]),
                "brier_at_minimum_validation_log_loss": float(result["best_metrics"]["brier_score"]),
                "ece_at_minimum_validation_log_loss": float(result["best_metrics"]["ece"]),
                "epochs_completed": int(result["epochs_completed"]),
                "research_valid": bool(result["research_valid"] and not getattr(args, "max_val_batches", 0)),
            })

        if not np.isfinite(oof_logits).all():
            raise RuntimeError("Stage 1 produced incomplete OOF logits.")
        evaluation = evaluate_stage1_oof(
            oof_logits, oof_targets, fold_ids, calibration_method=args.calibration
        )
        row = _stage1_trial_row(trial_id, config, values, evaluation, args.calibration)
        rows.append(row)
        np.savez_compressed(trial_dir / "oof_logits.npz", logits=oof_logits, targets=oof_targets, fold_ids=fold_ids)
        _write_rows(trial_dir / "fold_metrics.csv", fold_rows)
        trial_records[trial_id] = {
            "row": row, "config": config, "logits": oof_logits,
            "targets": oof_targets, "fold_ids": fold_ids,
            "evaluation": evaluation, "values": values,
        }
        _write_rows(output / "cv_trials.csv", rows)
        _plot_trials(rows, output / "optimization_summary.png")

    if not trial_records:
        raise RuntimeError("No Stage 1 trials were completed.")
    rows = [trial_records[trial_id]["row"] for trial_id in sorted(trial_records)]
    _write_rows(output / "cv_trials.csv", rows)
    selected_row = select_best_stage1_trial(rows)
    best_trial_id = int(selected_row["trial_id"])
    selected_record = trial_records[best_trial_id]
    best_config = dict(selected_record["config"])
    best_logits = selected_record["logits"]
    best_targets = selected_record["targets"]
    selected_evaluation = selected_record["evaluation"]

    best_trial_dirs = sorted((output / "trials").glob(f"trial_{best_trial_id:03d}_*/"))
    if not best_trial_dirs:
        raise RuntimeError("The selected Stage 1 trial has no persisted directory.")
    best_trial_dir = best_trial_dirs[0]
    best_fold_metrics_path = best_trial_dir / "fold_metrics.csv"
    if not best_fold_metrics_path.is_file():
        raise RuntimeError("The selected Stage 1 trial has no fold metrics.")
    fold_frame = __import__("pandas").read_csv(best_fold_metrics_path)
    if len(fold_frame) != len(folds):
        raise RuntimeError("The selected Stage 1 trial has incomplete fold metrics.")
    best_epochs = [int(value) for value in fold_frame["best_epoch"].tolist()]
    final_epochs = median_round_half_up(best_epochs)
    best_config.update({
        "selected_by": "cross_fitted_calibrated_oof_log_loss",
        "selection_objective": "cross_fitted_calibrated_oof_log_loss",
        "selection_lower_is_better": True,
        "cv_folds": int(args.cv_folds),
        "fold_scheme": fold_scheme,
        "fold_assignments": _portable_or_key(fold_path),
        "optimization_output_dir": _portable_or_key(output),
        "stage": 1,
        "selected_trial_id": int(best_trial_id),
        "selected_trial_score": float(selected_row["cross_fitted_calibrated_oof_log_loss"]),
        "config_fingerprint": fingerprint(best_config),
        "fold_assignment_fingerprint": fingerprint([[list(a), list(b)] for a, b in folds]),
        "final_training_epoch_rule": "median_stage1_best_epoch_round_half_up",
        "final_training_epochs": final_epochs,
        "fold_best_epochs": best_epochs,
        "research_valid": bool(best_config.get("research_valid", True) and not getattr(args, "max_train_batches", 0) and not getattr(args, "max_val_batches", 0)),
        "git_commit": current_git_commit(),
        "raw_oof_metrics": selected_evaluation["raw_metrics"],
        "cross_fitted_calibrated_oof_metrics": selected_evaluation["cross_fitted_metrics"],
        "final_all_oof_calibrated_metrics": selected_evaluation["final_all_oof_metrics"],
        "calibration_method": args.calibration,
        "calibration_provenance": "cross_fitted_for_selection_all_winning_oof_for_deployment",
    })
    (output / "best_config.json").write_text(json.dumps(best_config, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(output / "oof_predictions.npz", logits=best_logits, targets=best_targets, fold_ids=fold_ids)
    best_config["oof_artifact"] = _portable_or_key(output / "oof_predictions.npz")
    (output / "best_config.json").write_text(json.dumps(best_config, indent=2, sort_keys=True), encoding="utf-8")

    raw_metrics = selected_evaluation["raw_metrics"]
    calibrated_metrics = selected_evaluation["cross_fitted_metrics"]
    fold_calibrations = selected_evaluation["fold_calibrations"]
    final_all_oof_metrics = selected_evaluation["final_all_oof_metrics"]
    final_calibration = dict(selected_evaluation["final_calibration"])
    final_calibration.update({
        "selection_objective": "cross_fitted_calibrated_oof_log_loss",
        "raw_oof_metrics": raw_metrics,
        "cross_fitted_calibrated_oof_metrics": calibrated_metrics,
        "final_all_oof_calibrated_metrics": final_all_oof_metrics,
        "raw_oof_log_loss": raw_metrics["log_loss"],
        "cross_fitted_calibrated_oof_log_loss": calibrated_metrics["log_loss"],
        "final_all_oof_calibrated_log_loss": final_all_oof_metrics["log_loss"],
        "cross_fitted_fold_calibrations": fold_calibrations,
        "calibration_fit_data": "all_winning_trial_labeled_oof_predictions_only_for_deployment",
        "calibration_selection_data": "other_folds_only_for_each_cross_fitted_validation_fold",
    })
    save_calibration(output / "calibration.json", final_calibration)
    (output / "calibration_report.json").write_text(
        json.dumps({
            "selection_objective": "cross_fitted_calibrated_oof_log_loss",
            "raw": raw_metrics,
            "cross_fitted_calibrated": calibrated_metrics,
            "final_all_oof_calibrated": final_all_oof_metrics,
            "calibration_method": args.calibration,
            "deployment_fit": "all_winning_trial_oof_logits_and_targets",
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    # Commit-safe, UID-free research outputs are kept under runs/ as well as
    # the historical local optimization directory under artifacts/.
    research_dir = Path(getattr(args, "research_output_dir", "runs/dat_parkinsons/optimization"))
    research_dir.mkdir(parents=True, exist_ok=True)
    if research_dir.resolve() != output.resolve():
        for name in ("cv_trials.csv", "calibration_report.json"):
            source = output / name
            if source.is_file():
                (research_dir / name).write_bytes(source.read_bytes())
    _write_rows(research_dir / "selected_fold_metrics.csv", fold_frame.to_dict("records"))
    (research_dir / "selected_config.json").write_text(json.dumps(best_config, indent=2, sort_keys=True), encoding="utf-8")
    (research_dir / "calibration_report.json").write_text(
        json.dumps({"selection_objective": "cross_fitted_calibrated_oof_log_loss",
                    "raw": raw_metrics, "calibrated_cross_fitted": calibrated_metrics,
                    "final_all_oof_calibrated": final_all_oof_metrics,
                    "calibration_method": args.calibration,
                    "calibration_provenance": "cross_fitted_other_folds_for_selection_all_winning_oof_for_deployment",
                    "research_valid": bool(best_config["research_valid"])}, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "stage": 1, "research_valid": bool(best_config["research_valid"]),
        "selection_basis": "cross_fitted_calibrated_oof_log_loss",
        "selection_lower_is_better": True,
        "calibration_method": args.calibration,
        "selected_trial_id": int(best_trial_id), "selected_config_fingerprint": best_config["config_fingerprint"],
        "selected_trial_score": float(selected_row["cross_fitted_calibrated_oof_log_loss"]),
        "fold_count": len(folds), "fold_best_epochs": best_epochs,
        "final_training_epoch_rule": best_config["final_training_epoch_rule"],
        "final_training_epochs": final_epochs, "raw_oof_metrics": raw_metrics,
        "cross_fitted_calibrated_oof_metrics": calibrated_metrics,
        "final_all_oof_calibrated_metrics": final_all_oof_metrics,
        "calibration_provenance": "cross_fitted_other_folds_for_selection_all_winning_oof_for_deployment",
        "privacy": "Aggregate metrics only; no UIDs, patient predictions, arrays, or machine paths.",
    }
    (research_dir / "stage1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _plot_trials(rows, research_dir / "optimization_summary.png")
    _plot_trials(rows, research_dir / "optimization_summary.pdf")
    return {"best_config": best_config, "raw_metrics": raw_metrics, "calibrated_metrics": calibrated_metrics,
            "fold_metrics": fold_frame.to_dict("records"), "final_training_epochs": final_epochs,
            "research_output_dir": str(research_dir)}


def main() -> None:
    args = _parser().parse_args()
    result = run(args)
    print(json.dumps({
        "raw_oof_log_loss": result["raw_metrics"]["log_loss"],
        "cross_fitted_calibrated_oof_log_loss": result["calibrated_metrics"]["log_loss"],
    }, indent=2))


if __name__ == "__main__":
    main()
