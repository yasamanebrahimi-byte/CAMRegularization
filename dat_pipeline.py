"""The complete DaT Parkinson's experiment pipeline.

This module intentionally owns orchestration rather than low-level image or
model code.  The sections below preserve the existing Stage 1/Stage 2
artifacts, integrity checks, calibration rules, and teacher lineage while
keeping the workflow in one discoverable place.
"""

from __future__ import annotations

import csv
import itertools
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from cutout import CutoutAugmentedDataset
from dat_preprocessing import (
    DEFAULT_TARGET_SHAPE,
    DatDataset,
    check_dat_dataset,
    default_preprocessing_config,
    estimate_target_spacing,
    load_dat_records,
    parse_target_shape,
    parse_target_spacing,
)
from dat_training import (
    build_dat_model,
    compute_binary_metrics,
    fit_dat_model,
    fit_dat_model_fixed_epochs,
    probabilities_from_logits,
)
from graphics import plot_stage1_trials, save_research_figure
from model_registry import build_resnet18_3d
from utils import (
    REPO_ROOT,
    current_git_commit,
    fingerprint,
    median_round_half_up,
    portable_path,
    research_valid,
    sha256_file,
)


# ================================================================
# Cross-validation
# ================================================================


def make_stratified_folds(records: Sequence, n_splits: int = 5, seed: int = 42) -> list[tuple[list[int], list[int]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if len(records) < n_splits:
        raise ValueError("There must be at least one labeled record per fold.")
    rng = np.random.default_rng(int(seed))
    buckets = [[] for _ in range(int(n_splits))]
    for label in sorted(set(record.label for record in records)):
        indices = np.asarray([i for i, record in enumerate(records) if record.label == label], dtype=np.int64)
        rng.shuffle(indices)
        for position, index in enumerate(indices.tolist()):
            buckets[position % n_splits].append(int(index))
    all_indices = set(range(len(records)))
    folds = []
    for validation in buckets:
        validation = sorted(validation)
        training = sorted(all_indices.difference(validation))
        if not validation or not training:
            raise ValueError("A stratified fold is empty; reduce n_splits or provide more labeled records.")
        folds.append((training, validation))
    return folds


def protocol_signature(record, decimals: int = 1) -> tuple:
    from dat_preprocessing import _canonical_array_and_spacing

    volume, spacing = _canonical_array_and_spacing(record.path)
    return tuple(int(v) for v in volume.shape), tuple(round(float(v), decimals) for v in spacing)


def make_protocol_group_folds(records: Sequence, n_splits: int = 5, seed: int = 42) -> list[tuple[list[int], list[int]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    groups: dict[tuple, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(protocol_signature(record), []).append(index)
    ordered_groups = list(groups.items())
    rng = np.random.default_rng(int(seed))
    rng.shuffle(ordered_groups)
    fold_groups = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for _signature, indices in sorted(ordered_groups, key=lambda item: -len(item[1])):
        target = min(range(n_splits), key=lambda fold: fold_sizes[fold])
        fold_groups[target].extend(indices)
        fold_sizes[target] += len(indices)
    all_indices = set(range(len(records)))
    result = []
    for validation in fold_groups:
        validation = sorted(validation)
        training = sorted(all_indices.difference(validation))
        if not validation or not training:
            raise ValueError("A protocol-group fold is empty; reduce n_splits or provide more protocol groups.")
        result.append((training, validation))
    return result


def save_fold_assignments(path: str | Path, records: Sequence, folds: Sequence[tuple[Sequence[int], Sequence[int]]], *, seed: int, grouped: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "version": 1, "seed": int(seed), "n_splits": len(folds),
        "grouped_protocol_diagnostic": bool(grouped),
        "folds": [{
            "train_uids": [records[int(i)].uid for i in train],
            "validation_uids": [records[int(i)].uid for i in validation],
        } for train, validation in folds],
    }, indent=2, sort_keys=True), encoding="utf-8")


def load_fold_assignments(path: str | Path, records: Sequence) -> list[tuple[list[int], list[int]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_uid = {record.uid: index for index, record in enumerate(records)}
    folds = []
    for fold in payload.get("folds", []):
        try:
            train = [by_uid[uid] for uid in fold["train_uids"]]
            validation = [by_uid[uid] for uid in fold["validation_uids"]]
        except KeyError as exc:
            raise ValueError("Fold assignment file does not match the current labeled dataset.") from exc
        folds.append((train, validation))
    if not folds:
        raise ValueError("Fold assignment file contains no folds.")
    return folds


def _fold_ids_from_folds(folds: list[tuple[list[int], list[int]]], n_samples: int) -> np.ndarray:
    fold_ids = np.full(int(n_samples), -1, dtype=np.int64)
    for fold, (_train, validation) in enumerate(folds):
        validation = np.asarray(validation, dtype=np.int64)
        if validation.size == 0 or np.any(validation < 0) or np.any(validation >= n_samples):
            raise ValueError("Validation folds contain invalid indices.")
        if np.any(fold_ids[validation] != -1):
            raise ValueError("Validation folds overlap.")
        fold_ids[validation] = int(fold)
    if np.any(fold_ids < 0):
        raise ValueError("Validation folds do not cover every OOF observation.")
    return fold_ids


# ================================================================
# Calibration and binary probability metrics
# ================================================================


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    temperature = float(temperature)
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Temperature must be finite and positive.")
    return probabilities_from_logits(np.asarray(logits, dtype=np.float64) / temperature)


def fit_temperature(logits: np.ndarray, targets: np.ndarray, max_iter: int = 100) -> float:
    logits_tensor = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    targets_tensor = torch.as_tensor(np.asarray(targets), dtype=torch.long)
    if logits_tensor.ndim != 2 or logits_tensor.shape[1] != 2:
        raise ValueError("Temperature scaling expects logits with shape [N,2].")
    if logits_tensor.shape[0] != targets_tensor.numel() or targets_tensor.numel() == 0:
        raise ValueError("Logits and targets must have equal non-zero length.")
    parameter = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([parameter], lr=0.1, max_iter=int(max_iter), line_search_fn="strong_wolfe")
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temperature = parameter.exp().clamp(1e-3, 1e3)
        loss = criterion(logits_tensor / temperature, targets_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(parameter.detach().exp().clamp(1e-3, 1e3).item())


def fit_calibration(logits: np.ndarray, targets: np.ndarray, method: str = "temperature") -> dict[str, Any]:
    method = str(method or "raw").lower()
    if method in {"none", "raw", "identity"}:
        return {"method": "raw", "temperature": 1.0}
    if method != "temperature":
        raise ValueError(f"Unsupported calibration method '{method}'.")
    return {"method": "temperature", "temperature": fit_temperature(logits, targets)}


def calibrated_probabilities(logits: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    method = str(calibration.get("method", "raw")).lower()
    if method == "raw":
        return probabilities_from_logits(logits)
    if method == "temperature":
        return apply_temperature(logits, float(calibration["temperature"]))
    raise ValueError(f"Unsupported calibration method '{method}'.")


def cross_fitted_calibration(logits: np.ndarray, targets: np.ndarray, validation_folds: Sequence[Sequence[int]], method: str = "temperature") -> tuple[np.ndarray, list[dict[str, Any]]]:
    logits = np.asarray(logits)
    targets = np.asarray(targets)
    if logits.ndim != 2 or logits.shape[0] != len(targets) or len(targets) == 0:
        raise ValueError("OOF logits and targets must be non-empty and aligned.")
    calibrated = np.empty(len(targets), dtype=np.float64)
    all_indices = np.arange(len(targets))
    fold_calibrations = []
    seen = []
    for validation in validation_folds:
        validation = np.asarray(list(validation), dtype=np.int64)
        if validation.size == 0 or np.any(validation < 0) or np.any(validation >= len(targets)):
            raise ValueError("Calibration validation folds contain invalid indices.")
        training = np.setdiff1d(all_indices, validation, assume_unique=False)
        if training.size == 0:
            raise ValueError("Cross-fitted calibration requires samples outside each validation fold.")
        calibration = fit_calibration(logits[training], targets[training], method=method)
        calibrated[validation] = calibrated_probabilities(logits[validation], calibration)
        fold_calibrations.append(calibration)
        seen.extend(validation.tolist())
    if sorted(seen) != list(range(len(targets))):
        raise ValueError("Calibration folds must partition every OOF sample exactly once.")
    return calibrated, fold_calibrations


def fit_candidate_calibration(logits: np.ndarray, targets: np.ndarray, fold_ids: np.ndarray, method: str = "temperature") -> dict[str, Any]:
    logits, targets, fold_ids = np.asarray(logits), np.asarray(targets), np.asarray(fold_ids)
    if logits.ndim != 2 or logits.shape[1] != 2 or len(targets) != len(logits) or len(fold_ids) != len(targets):
        raise ValueError("Candidate OOF logits, targets, and fold_ids must be aligned [N,2], [N], [N].")
    if len(targets) == 0 or not np.isfinite(logits).all():
        raise ValueError("Candidate OOF logits must be finite and non-empty.")
    unique_folds = sorted(set(int(value) for value in fold_ids.tolist()))
    validation_folds = [np.flatnonzero(fold_ids == fold).tolist() for fold in unique_folds]
    cross_probs, fold_calibrations = cross_fitted_calibration(logits, targets, validation_folds, method=method)
    raw_metrics = compute_binary_metrics(targets, logits=logits)
    cross_metrics = compute_binary_metrics(targets, probabilities=cross_probs)
    final = fit_calibration(logits, targets, method=method)
    final_metrics = compute_binary_metrics(targets, probabilities=calibrated_probabilities(logits, final))
    final.update({
        "provenance": "candidate_oof_logits_only", "n_oof_samples": int(len(targets)),
        "n_cv_folds": int(len(unique_folds)), "fold_calibrations": fold_calibrations,
        "raw_oof_log_loss": float(raw_metrics["log_loss"]),
        "cross_fitted_calibrated_oof_log_loss": float(cross_metrics["log_loss"]),
        "final_all_oof_calibrated_log_loss": float(final_metrics["log_loss"]),
    })
    return {
        "raw_metrics": raw_metrics, "cross_fitted_metrics": cross_metrics,
        "final_calibration": final, "fold_calibrations": fold_calibrations,
        "fold_ids": unique_folds,
    }


def save_calibration(path: str | Path, calibration: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")


def load_calibration(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Calibration file must contain a JSON object.")
    return value


# ================================================================
# Stage 1: unmasked optimization and deployment model
# ================================================================


def _load_search_space(path: str) -> dict[str, list[Any]]:
    if not path:
        return {
            "learning_rate": [1e-3, 3e-4, 1e-4], "weight_decay": [1e-4, 1e-5, 1e-3],
            "dropout": [0.0, 0.1, 0.25], "label_smoothing": [0.0, 0.02, 0.05],
            "optimizer": ["adamw", "sgd"],
        }
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("search_space_json must contain a non-empty JSON object.")
    return {str(key): list(values) for key, values in payload.items() if isinstance(values, list) and values}


def _trial_config(args, preprocessing: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    config = {
        "dataset": "dat_parkinsons", "model": "resnet18_3d", "spatial_dims": 3,
        "n_input_channels": 1, "num_classes": 2, "base_channels": int(args.base_channels),
        "epochs": int(args.epochs), "patience": int(args.patience), "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers), "scheduler": "cosine", "min_lr": 1e-6,
        "momentum": 0.9, "nesterov": True, "adamw_betas": [0.9, 0.999], "amp": bool(args.amp),
        "spatial_augmentation": args.augmentation == "mild", "seed": int(args.seed),
        "calibration": args.calibration, "cutout_mode": "none", "cutout_m": 0,
        "preprocessing": preprocessing, "max_train_batches": int(args.max_train_batches or 0),
        "max_val_batches": int(args.max_val_batches or 0), "debug": bool(args.debug),
    }
    config.update(values)
    config["preprocessing"] = dict(preprocessing)
    config["research_valid"] = research_valid(
        max_train_batches=config["max_train_batches"], max_val_batches=config["max_val_batches"], debug=config["debug"]
    )
    if "target_shape" in values:
        config["preprocessing"]["target_shape"] = list(parse_target_shape(values["target_shape"]))
    if "target_spacing" in values:
        config["preprocessing"]["target_spacing"] = list(parse_target_spacing(values["target_spacing"]))
    config["preprocessing_fingerprint"] = fingerprint(config["preprocessing"])
    return config


def _config_hash(config: dict[str, Any]) -> str:
    return fingerprint(config, length=12)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _search_config_key(values: dict[str, Any], keys: tuple[str, ...]) -> str:
    return _canonical_json({key: values.get(key) for key in keys})


def generate_unique_trial_values(search_space: dict[str, list[Any]], requested_trials: int, seed: int, *, reserved_values: list[dict[str, Any]] | None = None) -> tuple[dict[int, dict[str, Any]], int]:
    requested_trials = int(requested_trials)
    if requested_trials < 0:
        raise ValueError("--trials must be non-negative.")
    keys = tuple(search_space)
    if not keys or any(not values for values in search_space.values()):
        raise ValueError("The Stage 1 search space must contain non-empty value lists.")
    options = []
    for key in keys:
        seen, unique = set(), []
        for value in search_space[key]:
            marker = _canonical_json(value)
            if marker not in seen:
                seen.add(marker)
                unique.append(value)
        options.append(unique)
    combinations = [dict(zip(keys, values)) for values in itertools.product(*options)]
    random.Random(int(seed)).shuffle(combinations)
    reserved = {_search_config_key(values, keys) for values in (reserved_values or []) if all(key in values for key in keys)}
    available = [values for values in combinations if _search_config_key(values, keys) not in reserved]
    requested_trials = min(requested_trials, len(combinations), len(available))
    return {trial_id: dict(available[trial_id]) for trial_id in range(requested_trials)}, requested_trials


def evaluate_stage1_oof(logits: np.ndarray, targets: np.ndarray, fold_ids: np.ndarray, *, calibration_method: str = "temperature") -> dict[str, Any]:
    logits, targets, fold_ids = np.asarray(logits), np.asarray(targets, dtype=np.int64), np.asarray(fold_ids, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[1] != 2 or len(logits) != len(targets) or len(targets) != len(fold_ids):
        raise ValueError("Stage 1 OOF logits, targets, and fold_ids must be aligned as [N,2], [N], [N].")
    if len(targets) == 0 or not np.isfinite(logits).all() or np.any(fold_ids < 0):
        raise ValueError("Stage 1 OOF artifacts must be finite, non-empty, and completely assigned to folds.")
    validation_folds = [np.flatnonzero(fold_ids == fold).tolist() for fold in sorted(set(fold_ids.tolist()))]
    raw = compute_binary_metrics(targets, logits=logits)
    cross_probs, fold_calibrations = cross_fitted_calibration(logits, targets, validation_folds, method=calibration_method)
    cross = compute_binary_metrics(targets, probabilities=cross_probs)
    final_calibration = fit_calibration(logits, targets, method=calibration_method)
    final = compute_binary_metrics(targets, probabilities=calibrated_probabilities(logits, final_calibration))
    return {
        "raw_metrics": raw, "cross_fitted_metrics": cross, "final_all_oof_metrics": final,
        "fold_calibrations": fold_calibrations, "final_calibration": final_calibration,
        "selection_score": float(cross["log_loss"]),
    }


def select_best_stage1_trial(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No completed Stage 1 trials are available.")
    missing = [row.get("trial_id") for row in rows if row.get("cross_fitted_calibrated_oof_log_loss") in (None, "")]
    if missing:
        raise RuntimeError(f"Completed Stage 1 trials are missing calibrated OOF log loss: {missing}")
    return min(rows, key=lambda row: (float(row["cross_fitted_calibrated_oof_log_loss"]), int(row["trial_id"])))


def _stage1_trial_row(trial_id: int, config: dict[str, Any], values: dict[str, Any], evaluation: dict[str, Any], calibration_method: str) -> dict[str, Any]:
    raw, cross = evaluation["raw_metrics"], evaluation["cross_fitted_metrics"]
    row = {
        "trial_id": int(trial_id), "config_hash": _config_hash(config), "calibration_method": str(calibration_method),
        "raw_oof_log_loss": float(raw["log_loss"]), "raw_oof_auroc": float(raw["auroc"]),
        "raw_oof_brier_score": float(raw["brier_score"]), "raw_oof_accuracy": float(raw["accuracy"]),
        "cross_fitted_calibrated_oof_log_loss": float(cross["log_loss"]),
        "cross_fitted_calibrated_oof_auroc": float(cross["auroc"]),
        "cross_fitted_calibrated_oof_brier_score": float(cross["brier_score"]),
        "cross_fitted_calibrated_oof_accuracy": float(cross["accuracy"]),
        "cross_fitted_calibrated_oof_ece": float(cross["ece"]),
        "final_all_oof_calibrated_log_loss": float(evaluation["final_all_oof_metrics"]["log_loss"]),
        "selection_score": float(evaluation["selection_score"]),
        "mean_oof_log_loss": float(raw["log_loss"]), "mean_oof_auroc": float(raw["auroc"]),
        "mean_oof_brier_score": float(raw["brier_score"]), "mean_oof_accuracy": float(raw["accuracy"]),
    }
    for key, value in values.items():
        row[key] = _canonical_json(value) if isinstance(value, (dict, list, tuple)) else value
    for key in ("fold_assignment_fingerprint", "preprocessing_fingerprint"):
        if key in config:
            row[key] = config[key]
    return row


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _portable_or_key(path: str | Path) -> str:
    try:
        return portable_path(path)
    except ValueError:
        return f"external/{Path(path).name}"


def _load_resumable_stage1_trial(trial_dir: Path, config: dict[str, Any], expected_targets: np.ndarray, current_fold_ids: np.ndarray, *, calibration_method: str):
    oof_path = trial_dir / "oof_logits.npz"
    if not oof_path.is_file():
        raise RuntimeError(f"Cannot safely resume Stage 1 trial in {trial_dir}: missing persisted OOF logits; rerun Stage 1.")
    with np.load(oof_path) as payload:
        logits = np.asarray(payload["logits"])
        targets = np.asarray(payload["targets"], dtype=np.int64)
        if "fold_ids" in payload:
            fold_ids = np.asarray(payload["fold_ids"], dtype=np.int64)
        elif "fold" in payload:
            fold_ids = np.asarray(payload["fold"], dtype=np.int64)
        else:
            fold_ids = current_fold_ids.copy()
    if logits.shape != (len(expected_targets), 2) or targets.shape != expected_targets.shape or not np.array_equal(targets, expected_targets):
        raise RuntimeError(f"Cannot safely resume Stage 1 trial in {trial_dir}: persisted OOF targets/shapes do not match.")
    if fold_ids.shape != current_fold_ids.shape or not np.array_equal(fold_ids, current_fold_ids):
        raise RuntimeError(f"Cannot safely resume Stage 1 trial in {trial_dir}: fold membership differs from current assignment.")
    if str(config.get("calibration", calibration_method)).lower() != str(calibration_method).lower():
        raise RuntimeError(f"Cannot safely resume Stage 1 trial in {trial_dir}: calibration method changed.")
    if not np.isfinite(logits).all():
        raise RuntimeError(f"Cannot safely resume Stage 1 trial in {trial_dir}: OOF logits are non-finite.")
    return logits, targets, fold_ids, evaluate_stage1_oof(logits, targets, fold_ids, calibration_method=calibration_method)


def run_stage1(args) -> dict[str, Any]:
    """Tune, calibrate, train, and package the unmasked Stage 1 model."""
    # Stage 1 owns the directory containing the authoritative handoff.  Stage 2
    # uses args.output_dir for its experiment grid, so the two namespaces stay
    # independent while custom --best_config paths remain supported.
    output = Path(args.best_config).parent
    output.mkdir(parents=True, exist_ok=True)
    records = load_dat_records(args.data_dir)
    spacing = parse_target_spacing(args.target_spacing) if args.target_spacing else estimate_target_spacing(records)
    preprocessing = default_preprocessing_config(
        records, target_spacing=spacing, target_shape=parse_target_shape(args.target_shape),
        lower_percentile=args.intensity_lower_percentile, upper_percentile=args.intensity_upper_percentile,
        foreground_threshold=args.foreground_threshold, crop_margin_mm=args.crop_margin_mm,
    )
    grouped = str(args.fold_scheme) == "protocol_group"
    folds = make_protocol_group_folds(records, args.cv_folds, args.seed) if grouped else make_stratified_folds(records, args.cv_folds, args.seed)
    fold_path = output / "fold_assignments.json"
    save_fold_assignments(fold_path, records, folds, seed=args.seed, grouped=grouped)
    fold_ids = _fold_ids_from_folds(folds, len(records))
    fold_hash = fingerprint([[list(a), list(b)] for a, b in folds])
    search_space = _load_search_space(args.search_space_json)
    expected_targets = np.asarray([record.label for record in records], dtype=np.int64)
    trial_table = output / "cv_trials.csv"
    trial_records: dict[int, dict[str, Any]] = {}
    if trial_table.is_file():
        for row in csv.DictReader(trial_table.open("r", newline="", encoding="utf-8")):
            trial_id = int(row["trial_id"])
            matches = sorted((output / "trials").glob(f"trial_{trial_id:03d}_*/trial_config.json"))
            if not matches:
                continue
            trial_dir = matches[0].parent
            config = json.loads(matches[0].read_text(encoding="utf-8"))
            logits, targets, persisted_fold_ids, evaluation = _load_resumable_stage1_trial(
                trial_dir, config, expected_targets, fold_ids, calibration_method=args.calibration
            )
            values = {key: config[key] for key in search_space}
            trial_records[trial_id] = {
                "row": _stage1_trial_row(trial_id, config, values, evaluation, args.calibration),
                "config": config, "logits": logits, "targets": targets,
                "fold_ids": persisted_fold_ids, "evaluation": evaluation, "values": values,
            }
    completed_values = [item["values"] for item in trial_records.values()]
    new_ids = [trial_id for trial_id in range(max(0, int(args.trials))) if trial_id not in trial_records]
    new_values, count = generate_unique_trial_values(search_space, len(new_ids), args.seed, reserved_values=completed_values)
    new_ids = new_ids[:count]
    for trial_id in new_ids:
        values = new_values[len([i for i in new_ids if i < trial_id])]
        config = _trial_config(args, preprocessing, values)
        config.update({
            "cv_folds": int(args.cv_folds), "fold_scheme": str(args.fold_scheme),
            "fold_assignment_fingerprint": fold_hash,
            "selection_objective": "cross_fitted_calibrated_oof_log_loss", "selection_lower_is_better": True,
        })
        trial_dir = output / "trials" / f"trial_{trial_id:03d}_{_config_hash(config)}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "trial_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        oof_logits = np.full((len(records), 2), np.nan, dtype=np.float32)
        fold_rows = []
        for fold_index, (train_indices, validation_indices) in enumerate(folds):
            train_ds = DatDataset([records[i] for i in train_indices], config["preprocessing"], train=True,
                                  augment=bool(config.get("spatial_augmentation", False)), seed=args.seed + fold_index,
                                  cache_dir=output.parent / "cache" / "preprocessed" / _config_hash(config))
            val_ds = DatDataset([records[i] for i in validation_indices], config["preprocessing"], train=False,
                                augment=False, seed=args.seed + fold_index,
                                cache_dir=output.parent / "cache" / "preprocessed" / _config_hash(config))
            result = fit_dat_model(train_ds, val_ds, config, seed=args.seed + 1000 * trial_id + fold_index,
                                   run_dir=trial_dir / f"fold_{fold_index}",
                                   max_train_batches=args.max_train_batches or None, max_val_batches=None)
            oof_logits[validation_indices] = result["best_logits"]
            fold_rows.append({
                "fold": fold_index, "best_epoch": int(result["best_epoch"]),
                "minimum_validation_log_loss": float(result["best_metrics"]["log_loss"]),
                "epoch_at_minimum_validation_log_loss": int(result["best_epoch"]),
                "accuracy_at_minimum_validation_log_loss": float(result["best_metrics"]["accuracy"]),
                "auroc_at_minimum_validation_log_loss": float(result["best_metrics"]["auroc"]),
                "brier_at_minimum_validation_log_loss": float(result["best_metrics"]["brier_score"]),
                "ece_at_minimum_validation_log_loss": float(result["best_metrics"]["ece"]),
                "epochs_completed": int(result["epochs_completed"]),
                "research_valid": bool(result["research_valid"] and not args.max_val_batches),
            })
        if not np.isfinite(oof_logits).all():
            raise RuntimeError("Stage 1 produced incomplete OOF logits.")
        evaluation = evaluate_stage1_oof(oof_logits, expected_targets, fold_ids, calibration_method=args.calibration)
        row = _stage1_trial_row(trial_id, config, values, evaluation, args.calibration)
        np.savez_compressed(trial_dir / "oof_logits.npz", logits=oof_logits, targets=expected_targets, fold_ids=fold_ids)
        _write_rows(trial_dir / "fold_metrics.csv", fold_rows)
        trial_records[trial_id] = {
            "row": row, "config": config, "logits": oof_logits, "targets": expected_targets,
            "fold_ids": fold_ids, "evaluation": evaluation, "values": values,
        }
        rows_now = [trial_records[i]["row"] for i in sorted(trial_records)]
        _write_rows(trial_table, rows_now)
        plot_stage1_trials(rows_now, output / "optimization_summary.png", select_best_stage1_trial(rows_now))
    if not trial_records:
        raise RuntimeError("No Stage 1 trials were completed.")
    rows = [trial_records[i]["row"] for i in sorted(trial_records)]
    _write_rows(trial_table, rows)
    selected_row = select_best_stage1_trial(rows)
    selected = trial_records[int(selected_row["trial_id"])]
    best_config = dict(selected["config"])
    best_trial_dir = sorted((output / "trials").glob(f"trial_{int(selected_row['trial_id']):03d}_*/"))[0]
    fold_frame = pd.read_csv(best_trial_dir / "fold_metrics.csv")
    if len(fold_frame) != len(folds):
        raise RuntimeError("The selected Stage 1 trial has incomplete fold metrics.")
    final_epochs = median_round_half_up([int(value) for value in fold_frame["best_epoch"].tolist()])
    evaluation = selected["evaluation"]
    best_config.update({
        "selected_by": "cross_fitted_calibrated_oof_log_loss", "selection_objective": "cross_fitted_calibrated_oof_log_loss",
        "selection_lower_is_better": True, "cv_folds": int(args.cv_folds), "fold_scheme": str(args.fold_scheme),
        "fold_assignments": _portable_or_key(fold_path), "optimization_output_dir": _portable_or_key(output),
        "stage": 1, "selected_trial_id": int(selected_row["trial_id"]),
        "selected_trial_score": float(selected_row["cross_fitted_calibrated_oof_log_loss"]),
        "fold_assignment_fingerprint": fold_hash,
        "final_training_epoch_rule": "median_stage1_best_epoch_round_half_up", "final_training_epochs": final_epochs,
        "fold_best_epochs": [int(value) for value in fold_frame["best_epoch"].tolist()],
        "research_valid": bool(best_config.get("research_valid", True) and not args.max_train_batches and not args.max_val_batches),
        "git_commit": current_git_commit(), "raw_oof_metrics": evaluation["raw_metrics"],
        "cross_fitted_calibrated_oof_metrics": evaluation["cross_fitted_metrics"],
        "final_all_oof_calibrated_metrics": evaluation["final_all_oof_metrics"],
        "calibration_method": args.calibration,
        "calibration_provenance": "cross_fitted_for_selection_all_winning_oof_for_deployment",
    })
    # Fingerprint the complete selected recipe.  The later OOF artifact path is
    # deliberately excluded by load_valid_stage1_config for portability.
    best_config["config_fingerprint"] = fingerprint(best_config)
    (output / "best_config.json").write_text(json.dumps(best_config, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(output / "oof_predictions.npz", logits=selected["logits"], targets=selected["targets"], fold_ids=fold_ids)
    best_config["oof_artifact"] = _portable_or_key(output / "oof_predictions.npz")
    (output / "best_config.json").write_text(json.dumps(best_config, indent=2, sort_keys=True), encoding="utf-8")
    final_calibration = dict(evaluation["final_calibration"])
    final_calibration.update({
        "selection_objective": "cross_fitted_calibrated_oof_log_loss", "raw_oof_metrics": evaluation["raw_metrics"],
        "cross_fitted_calibrated_oof_metrics": evaluation["cross_fitted_metrics"],
        "final_all_oof_calibrated_metrics": evaluation["final_all_oof_metrics"],
        "raw_oof_log_loss": evaluation["raw_metrics"]["log_loss"],
        "cross_fitted_calibrated_oof_log_loss": evaluation["cross_fitted_metrics"]["log_loss"],
        "final_all_oof_calibrated_log_loss": evaluation["final_all_oof_metrics"]["log_loss"],
        "cross_fitted_fold_calibrations": evaluation["fold_calibrations"],
        "calibration_fit_data": "all_winning_trial_labeled_oof_predictions_only_for_deployment",
        "calibration_selection_data": "other_folds_only_for_each_cross_fitted_validation_fold",
    })
    save_calibration(output / "calibration.json", final_calibration)
    (output / "calibration_report.json").write_text(json.dumps({
        "selection_objective": "cross_fitted_calibrated_oof_log_loss", "raw": evaluation["raw_metrics"],
        "cross_fitted_calibrated": evaluation["cross_fitted_metrics"],
        "final_all_oof_calibrated": evaluation["final_all_oof_metrics"], "calibration_method": args.calibration,
        "deployment_fit": "all_winning_trial_oof_logits_and_targets",
    }, indent=2, sort_keys=True), encoding="utf-8")
    research_dir = Path(args.research_output_dir)
    research_dir.mkdir(parents=True, exist_ok=True)
    for name in ("cv_trials.csv", "calibration_report.json"):
        if (output / name).is_file() and output.resolve() != research_dir.resolve():
            (research_dir / name).write_bytes((output / name).read_bytes())
    _write_rows(research_dir / "selected_fold_metrics.csv", fold_frame.to_dict("records"))
    (research_dir / "selected_config.json").write_text(json.dumps(best_config, indent=2, sort_keys=True), encoding="utf-8")
    (research_dir / "calibration_report.json").write_text(json.dumps({
        "selection_objective": "cross_fitted_calibrated_oof_log_loss",
        "raw": evaluation["raw_metrics"], "calibrated_cross_fitted": evaluation["cross_fitted_metrics"],
        "final_all_oof_calibrated": evaluation["final_all_oof_metrics"], "calibration_method": args.calibration,
        "calibration_provenance": "cross_fitted_other_folds_for_selection_all_winning_oof_for_deployment",
        "research_valid": bool(best_config["research_valid"]),
    }, indent=2, sort_keys=True), encoding="utf-8")
    (research_dir / "stage1_summary.json").write_text(json.dumps({
        "stage": 1, "research_valid": bool(best_config["research_valid"]),
        "selection_basis": "cross_fitted_calibrated_oof_log_loss", "selection_lower_is_better": True,
        "calibration_method": args.calibration, "selected_trial_id": int(selected_row["trial_id"]),
        "selected_config_fingerprint": best_config["config_fingerprint"],
        "selected_trial_score": float(selected_row["cross_fitted_calibrated_oof_log_loss"]),
        "fold_count": len(folds), "fold_best_epochs": best_config["fold_best_epochs"],
        "final_training_epoch_rule": best_config["final_training_epoch_rule"], "final_training_epochs": final_epochs,
        "raw_oof_metrics": evaluation["raw_metrics"], "cross_fitted_calibrated_oof_metrics": evaluation["cross_fitted_metrics"],
        "final_all_oof_calibrated_metrics": evaluation["final_all_oof_metrics"],
        "calibration_provenance": "cross_fitted_other_folds_for_selection_all_winning_oof_for_deployment",
        "privacy": "Aggregate metrics only; no UIDs, patient predictions, arrays, or machine paths.",
    }, indent=2, sort_keys=True), encoding="utf-8")
    plot_stage1_trials(rows, research_dir / "optimization_summary.png", selected_row)
    plot_stage1_trials(rows, research_dir / "optimization_summary.pdf", selected_row)
    if not best_config["research_valid"]:
        raise ValueError("Stage 1 smoke/debug or truncated runs cannot produce research or competition models.")
    final_model = train_final_stage1_model(
        args.data_dir, output / "best_config.json", args.stage1_model_dir, output / "calibration.json",
        seed=args.seed, num_workers=args.num_workers, max_train_batches=args.max_train_batches,
    )
    from dat_submission import build_submission

    archive = build_submission(args.stage1_model_dir, args.stage1_submission_zip)
    return {
        "best_config": best_config, "raw_metrics": evaluation["raw_metrics"],
        "calibrated_metrics": evaluation["cross_fitted_metrics"], "fold_metrics": fold_frame.to_dict("records"),
        "final_training_epochs": final_epochs, "final_model_dir": str(final_model["output_dir"]),
        "submission_zip": str(archive), "research_output_dir": str(research_dir),
    }


# ================================================================
# Stage 2 masking experiments
# ================================================================


CONDITIONS = ("none", "random", "cam_low", "cam_high")
MASKED_CONDITIONS = ("random", "cam_low", "cam_high")
DEFAULT_FRACTIONS = (0.05, 0.10, 0.20, 0.30)
DEFAULT_M = (4, 8)


def cell_key(fold: int, condition: str, m_value: int, fraction: float) -> str:
    return f"fold={int(fold)}|condition={condition}|M={int(m_value)}|fraction={float(fraction):.4f}"


def expected_grid(n_folds: int, conditions: Iterable[str] = CONDITIONS, m_values: Iterable[int] = DEFAULT_M, fractions: Iterable[float] = DEFAULT_FRACTIONS) -> list[dict[str, Any]]:
    cells = []
    conditions, m_values, fractions = tuple(conditions), tuple(int(v) for v in m_values), tuple(float(v) for v in fractions)
    for fold in range(int(n_folds)):
        if "none" in conditions:
            cells.append({"fold": fold, "condition": "none", "M": 0, "fraction": 0.0, "cell_key": cell_key(fold, "none", 0, 0.0)})
        for condition in MASKED_CONDITIONS:
            if condition not in conditions:
                continue
            for m_value in m_values:
                for fraction in fractions:
                    cells.append({"fold": fold, "condition": condition, "M": m_value, "fraction": fraction, "cell_key": cell_key(fold, condition, m_value, fraction)})
    return cells


def _effective_num_workers(condition: str, args) -> int:
    requested = max(0, int(getattr(args, "num_workers", 0) or 0))
    if str(condition).startswith("cam_") and requested > 0:
        print("[Stage 2] CAM cutout training uses num_workers=0 because saliency/window caches are main-process only.")
        return 0
    return requested


def _run_is_valid(run_root: str | Path, expected: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    root = Path(run_root)
    problems = []
    config_path, metrics_path = root / "config.json", root / "metrics.csv"
    if not config_path.is_file():
        return False, ["missing_config"]
    if not metrics_path.is_file():
        problems.append("missing_metrics")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False, ["invalid_config"]
    for key, value in (expected or {}).items():
        if key not in {"oof_artifact", "completed"} and config.get(key) != value:
            problems.append(f"config_mismatch:{key}")
    artifact = config.get("oof_artifact")
    artifact_path = REPO_ROOT / artifact if artifact and not Path(artifact).is_absolute() else Path(artifact) if artifact else None
    if artifact_path is None or not artifact_path.is_file():
        problems.append("missing_oof_artifact")
    else:
        try:
            with np.load(artifact_path) as payload:
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
            if frame.empty:
                problems.append("invalid_metrics")
            if config.get("stage") == 2:
                if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all():
                    problems.append("nonfinite_metrics")
                if int(config.get("epochs_completed", -1)) != int(frame["epoch"].max()):
                    problems.append("truncated_or_inconsistent_epoch_metadata")
        except Exception:
            problems.append("invalid_metrics")
    return not problems, problems


def _train_teacher(records, train_indices, preprocessing, config, fold, args, fold_hash=None):
    teacher_root = Path(args.cam_cache_dir).parent / "teachers" / f"fold_{fold}"
    teacher_root.mkdir(parents=True, exist_ok=True)
    selected_stage1_fingerprint = config.get("config_fingerprint", fingerprint(config))
    preprocessing_fingerprint = config.get("preprocessing_fingerprint", fingerprint(preprocessing))
    fold_assignment_fingerprint = fold_hash or config.get("fold_assignment_fingerprint") or fingerprint({"fold": int(fold)})
    teacher_config = deepcopy(config)
    epoch_budget = int(config.get("final_training_epochs", config.get("epochs", 100)))
    teacher_config.update({
        "cutout_mode": "none", "cutout_m": 0, "stage": "stage2_teacher", "fold": int(fold),
        "epochs": epoch_budget, "final_training_epochs": epoch_budget, "stage1_final_training_epochs": epoch_budget,
        "selected_stage1_config_fingerprint": selected_stage1_fingerprint, "preprocessing_fingerprint": preprocessing_fingerprint,
        "fold_assignment_fingerprint": fold_assignment_fingerprint,
        "teacher_recipe_provenance": "stage1_selected_unmasked_recipe_fixed_epoch_teacher", "teacher_lineage": "selected_stage1_config",
        "num_workers": int(args.num_workers), "max_train_batches": int(args.max_train_batches or 0), "max_val_batches": 0,
        "debug": bool(args.debug or args.max_train_batches or args.max_val_batches),
        "research_valid": research_valid(max_train_batches=args.max_train_batches or 0, max_val_batches=0, debug=args.debug),
        "completed": False, "checkpoint_selection": "final_scheduled_epoch",
    })
    checkpoint = teacher_root / "final_model.pt"
    config_path = teacher_root / "config.json"
    if checkpoint.is_file() and config_path.is_file():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            expected = {key: teacher_config[key] for key in (
                "stage", "fold", "epochs", "final_training_epochs", "stage1_final_training_epochs",
                "selected_stage1_config_fingerprint", "preprocessing_fingerprint", "fold_assignment_fingerprint",
                "research_valid", "checkpoint_selection", "teacher_recipe_provenance", "num_workers",
                "max_train_batches", "max_val_batches", "debug",
            )}
            if all(existing.get(key) == value for key, value in expected.items()) and existing.get("completed") is True and existing.get("teacher_checkpoint_sha256") == sha256_file(checkpoint):
                teacher = build_dat_model(config)
                payload = torch.load(checkpoint, map_location="cpu")
                state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
                teacher.load_state_dict({str(k).removeprefix("module."): v for k, v in state.items()}, strict=True)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                teacher.to(device).eval()
                for parameter in teacher.parameters():
                    parameter.requires_grad_(False)
                return teacher, checkpoint, sha256_file(checkpoint)
        except Exception:
            pass
    teacher_dataset = DatDataset([records[index] for index in train_indices], preprocessing, train=True,
                                 augment=bool(config.get("spatial_augmentation", False)), seed=args.seed + fold,
                                 cache_dir=args.preprocessed_cache_dir)
    result = fit_dat_model_fixed_epochs(teacher_dataset, teacher_config, seed=args.seed + 10000 + fold,
                                        run_dir=teacher_root, epochs=epoch_budget, max_train_batches=args.max_train_batches or None)
    teacher_hash = sha256_file(checkpoint)
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    persisted.update({
        "teacher_checkpoint_sha256": teacher_hash, "outer_train_record_count": len(train_indices),
        "outer_validation_used_for_teacher": False, "teacher_checkpoint_selection": "frozen_stage1_epoch_budget",
        "selected_stage1_config_fingerprint": selected_stage1_fingerprint, "preprocessing_fingerprint": preprocessing_fingerprint,
        "fold_assignment_fingerprint": fold_assignment_fingerprint, "fold": int(fold),
        "final_training_epochs": epoch_budget, "stage1_final_training_epochs": epoch_budget,
        "teacher_recipe_provenance": "stage1_selected_unmasked_recipe_fixed_epoch_teacher",
        "research_valid": teacher_config["research_valid"], "completed": True, "checkpoint_selection": "final_scheduled_epoch",
    })
    config_path.write_text(json.dumps(persisted, indent=2, sort_keys=True), encoding="utf-8")
    teacher = result["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher, checkpoint, teacher_hash


def _student_expected_config(config, fold, condition, m_value, fraction, args, teacher_hash, fold_hash):
    student_epoch_budget = int(config.get("epochs", 100))
    stage1_epoch_budget = int(config.get("final_training_epochs", student_epoch_budget))
    effective_workers = _effective_num_workers(condition, args)
    return {
        "stage": 2, "fold": int(fold), "condition": condition, "cutout_mode": condition,
        "cutout_m": int(m_value) if condition != "none" else 0, "cutout_fraction": float(fraction),
        "teacher_checkpoint_sha256": teacher_hash if condition.startswith("cam_") else None,
        "selected_stage1_config_fingerprint": config.get("config_fingerprint", fingerprint(config)),
        "preprocessing_fingerprint": config.get("preprocessing_fingerprint", fingerprint(config["preprocessing"])),
        "fold_assignment_fingerprint": fold_hash, "cam_layer": args.cam_layer,
        "saliency_candidate_percent": float(args.saliency_candidate_percent), "min_foreground_fraction": float(args.min_foreground_fraction),
        "epochs": student_epoch_budget, "student_max_cv_epochs": student_epoch_budget,
        "stage1_final_training_epochs": stage1_epoch_budget, "final_training_epochs": stage1_epoch_budget,
        "early_stopping_patience": int(config.get("patience", 15)), "student_seed_policy": "base_seed_plus_fold",
        "student_model": str(config.get("model", "resnet18_3d")), "seed": int(args.seed + fold),
        "num_workers": effective_workers, "requested_num_workers": int(args.num_workers or 0),
        "max_train_batches": int(args.max_train_batches or 0), "max_val_batches": int(args.max_val_batches or 0),
        "debug": bool(args.debug), "research_valid": research_valid(max_train_batches=args.max_train_batches or 0, max_val_batches=args.max_val_batches or 0, debug=args.debug),
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
        return {"run_dir": run_root, "skipped": True}
    if problems and run_root.exists():
        print(f"[Stage 2] rerunning fold {fold} {condition} M{m_value} fraction {fraction:.2f}: {', '.join(problems)}")
    base_train = DatDataset([records[index] for index in train_indices], preprocessing, train=True,
                            augment=bool(config.get("spatial_augmentation", False)), seed=args.seed + fold,
                            cache_dir=args.preprocessed_cache_dir)
    val_dataset = DatDataset([records[index] for index in validation_indices], preprocessing, train=False,
                             augment=False, seed=args.seed + fold, cache_dir=args.preprocessed_cache_dir)
    student_config = deepcopy(config)
    student_config.update(expected)
    if condition == "none":
        train_dataset = base_train
    else:
        cache_settings = {
            "dataset": "dat_parkinsons", "student_model": "resnet18_3d", "teacher_model": "resnet18_3d",
            "teacher_checkpoint_sha256": teacher_hash, "cam_layer": args.cam_layer, "spatial_dims": 3,
            "preprocessing": preprocessing, "min_foreground_fraction": float(args.min_foreground_fraction),
        }
        train_dataset = CutoutAugmentedDataset(
            base_dataset=base_train, cutout_mode=condition, cutout_m=int(m_value), cutout_size=None,
            cutout_area=float(fraction), mean=(0.0,), std=(1.0,), seed=args.seed + fold,
            saliency_candidate_percent=args.saliency_candidate_percent,
            teacher_model=teacher if condition.startswith("cam_") else None, cam_layer=args.cam_layer,
            cam_cache_dir=args.cam_cache_dir if condition.startswith("cam_") else None,
            cam_cache_settings=cache_settings if condition.startswith("cam_") else None,
            min_foreground_fraction=args.min_foreground_fraction,
        )
    result = fit_dat_model(train_dataset, val_dataset, student_config, seed=args.seed + fold, run_dir=run_root,
                           max_train_batches=args.max_train_batches or None, max_val_batches=None)
    artifact_id = fingerprint({"cell": cell_key(fold, condition, m_value, fraction), "stage1": expected["selected_stage1_config_fingerprint"], "seed": args.seed + fold})
    oof_path = REPO_ROOT / "artifacts" / "dat_parkinsons" / "oof" / f"stage2_{artifact_id}.npz"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(oof_path, logits=result["best_logits"], targets=result["best_targets"], fold=np.full(len(result["best_targets"]), fold, dtype=np.int64))
    persisted = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    persisted.update({
        "oof_artifact": _portable_or_key(oof_path), "completed": True, "epochs_completed": int(result["epochs_completed"]),
        "best_epoch": int(result["best_epoch"]), "checkpoint_selection": "minimum_validation_log_loss",
        "minimum_validation_log_loss": float(result["best_metrics"]["log_loss"]),
        "accuracy_at_minimum_validation_log_loss": float(result["best_metrics"]["accuracy"]),
        "auroc_at_minimum_validation_log_loss": float(result["best_metrics"]["auroc"]),
        "brier_at_minimum_validation_log_loss": float(result["best_metrics"]["brier_score"]),
        "ece_at_minimum_validation_log_loss": float(result["best_metrics"]["ece"]),
    })
    (run_root / "config.json").write_text(json.dumps(persisted, indent=2, sort_keys=True), encoding="utf-8")
    return {"run_dir": run_root, "skipped": False, "result": result}


def run_stage2_grid(args) -> dict[str, Any]:
    best_config = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
    records = load_dat_records(args.data_dir)
    preprocessing = best_config["preprocessing"]
    configured = best_config.get("fold_assignments", "")
    fold_path = Path(args.fold_assignments) if args.fold_assignments else (REPO_ROOT / configured if configured and not Path(configured).is_absolute() else Path(configured))
    if fold_path.is_file():
        folds = load_fold_assignments(fold_path, records)
    else:
        folds = make_stratified_folds(records, int(best_config.get("cv_folds", 5)), args.seed)
        fold_path = Path(args.output_dir) / "fold_assignments.json"
        save_fold_assignments(fold_path, records, folds, seed=args.seed)
    fold_hash = best_config.get("fold_assignment_fingerprint", fingerprint([[list(a), list(b)] for a, b in folds]))
    selected_folds = range(len(folds)) if args.fold < 0 else [args.fold]
    results = []
    for fold in selected_folds:
        train_indices, validation_indices = folds[fold]
        teacher = teacher_hash = None
        if any(condition.startswith("cam_") for condition in args.conditions):
            teacher, _checkpoint, teacher_hash = _train_teacher(records, train_indices, preprocessing, best_config, fold, args, fold_hash=fold_hash)
        if "none" in args.conditions:
            results.append(_run_student(records, train_indices, validation_indices, preprocessing, best_config, fold, "none", 0, 0.0, args, teacher, teacher_hash, fold_hash))
        for condition in MASKED_CONDITIONS:
            if condition not in args.conditions:
                continue
            for m_value in args.m_values:
                for fraction in args.fractions:
                    results.append(_run_student(records, train_indices, validation_indices, preprocessing, best_config, fold, condition, m_value, fraction, args, teacher, teacher_hash, fold_hash))
    return {"folds": len(folds), "results": results, "expected_cells": expected_grid(len(folds), args.conditions, args.m_values, args.fractions)}


# ================================================================
# Candidate selection and research summary
# ================================================================


CANDIDATE_RECIPE_FIELDS = (
    "condition", "cutout_m", "cutout_fraction", "cam_layer", "saliency_candidate_percent",
    "min_foreground_fraction", "student_seed_policy", "student_model", "selected_stage1_config_fingerprint",
    "preprocessing_fingerprint", "fold_assignment_fingerprint", "student_max_cv_epochs", "epochs",
    "early_stopping_patience", "patience",
)
FROZEN_STAGE1_FIELDS = (
    "model", "n_input_channels", "num_classes", "base_channels", "dropout", "spatial_augmentation", "optimizer",
    "learning_rate", "weight_decay", "scheduler", "min_lr", "momentum", "nesterov", "adamw_betas",
    "label_smoothing", "batch_size", "amp", "epochs", "patience", "preprocessing_fingerprint",
    "selected_stage1_config_fingerprint", "fold_assignment_fingerprint", "final_training_epochs",
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


def _consistency_issues(entries, fields, *, require_present=True):
    issues = []
    if not entries:
        return issues
    for field in fields:
        values = [_config_value(config, field) for config, _ in entries]
        if (require_present and values[0] is None) or any((require_present and value is None) or not _same_value(value, values[0]) for value in values[1:]):
            issues.append({"field": field, "values": values, "run_dirs": [str(path) for _, path in entries]})
    return issues


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _is_student_config(path: Path, config: dict[str, Any]) -> bool:
    return config.get("stage") == 2 and str(config.get("condition", "")) in CONDITIONS and "teachers" not in {part.lower() for part in path.parts}


def integrity_check(runs_dir: str | Path, *, expected_folds: int, conditions=CONDITIONS, m_values=DEFAULT_M, fractions=DEFAULT_FRACTIONS, frozen_config: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(runs_dir)
    expected = expected_grid(expected_folds, conditions, m_values, fractions)
    expected_by_key = {cell["cell_key"]: cell for cell in expected}
    discovered, invalid, debug_runs, teacher_issues, oof_issues = {}, [], [], [], []
    candidate_configs: dict[tuple[str, int, float], list[tuple[dict[str, Any], Path]]] = {}
    frozen_control_issues = []
    for config_path in sorted(root.rglob("config.json")) if root.is_dir() else []:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            invalid.append({"run_dir": str(config_path.parent), "issues": [f"invalid_config:{exc}"]})
            continue
        if not _is_student_config(config_path, config):
            continue
        key = cell_key(int(config.get("fold", -1)), str(config.get("condition")), int(config.get("cutout_m", 0)), float(config.get("cutout_fraction", 0.0) or 0.0))
        discovered.setdefault(key, []).append(config_path.parent)
        candidate_key = (str(config["condition"]), int(config.get("cutout_m", 0)), float(config.get("cutout_fraction", 0.0) or 0.0))
        candidate_configs.setdefault(candidate_key, []).append((config, config_path.parent))
        if frozen_config:
            expected_lineage = {
                "selected_stage1_config_fingerprint": frozen_config.get("config_fingerprint"),
                "preprocessing_fingerprint": frozen_config.get("preprocessing_fingerprint"),
                "fold_assignment_fingerprint": frozen_config.get("fold_assignment_fingerprint"),
                "final_training_epochs": frozen_config.get("final_training_epochs"), "epochs": frozen_config.get("epochs"),
                "student_max_cv_epochs": frozen_config.get("epochs"), "patience": frozen_config.get("patience"),
                "early_stopping_patience": frozen_config.get("patience"),
            }
            for field, value in expected_lineage.items():
                if value is not None and config.get(field) != value:
                    invalid.append({"run_dir": str(config_path.parent), "cell_key": key, "issues": [f"config_mismatch:{field}"]})
        valid, issues = _run_is_valid(config_path.parent)
        if not valid:
            invalid.append({"run_dir": str(config_path.parent), "cell_key": key, "issues": issues})
            if "debug_or_truncated" in issues:
                debug_runs.append(str(config_path.parent))
        if str(config.get("condition", "")).startswith("cam_") and not config.get("teacher_checkpoint_sha256"):
            teacher_issues.append({"run_dir": str(config_path.parent), "cell_key": key, "issue": "missing_teacher_fingerprint"})
        artifact = config.get("oof_artifact")
        artifact_path = _resolve(artifact) if artifact else None
        if artifact and Path(artifact).is_absolute() and config_path.parent.resolve().is_relative_to(REPO_ROOT.resolve()):
            oof_issues.append({"run_dir": str(config_path.parent), "cell_key": key, "issue": "nonportable_absolute_oof_artifact"})
        if artifact_path is None or not artifact_path.is_file():
            oof_issues.append({"run_dir": str(config_path.parent), "cell_key": key, "issue": "missing_oof_artifact"})
        else:
            try:
                with np.load(artifact_path) as payload:
                    if not np.isfinite(payload["logits"]).all() or len(payload["logits"]) != len(payload["targets"]):
                        oof_issues.append({"run_dir": str(config_path.parent), "cell_key": key, "issue": "invalid_oof_artifact"})
                    if "fold" in payload and not np.all(np.asarray(payload["fold"]) == int(config.get("fold", -1))):
                        oof_issues.append({"run_dir": str(config_path.parent), "cell_key": key, "issue": "fold_assignment_mismatch"})
            except Exception:
                oof_issues.append({"run_dir": str(config_path.parent), "cell_key": key, "issue": "invalid_oof_artifact"})
    candidate_recipe_issues = []
    for candidate_key, entries in candidate_configs.items():
        for issue in _consistency_issues(entries, CANDIDATE_RECIPE_FIELDS):
            candidate_recipe_issues.append({"candidate": candidate_key, **issue})
    all_entries = [entry for entries in candidate_configs.values() for entry in entries]
    for issue in _consistency_issues(all_entries, FROZEN_STAGE1_FIELDS, require_present=False):
        frozen_control_issues.append({"scope": "all_stage2_candidates", **issue})
    if frozen_config:
        for field in FROZEN_STAGE1_FIELDS:
            expected_value = _config_value(frozen_config, field)
            if expected_value is not None:
                actual_values = [_config_value(config, field) for config, _ in all_entries]
                if any(value is None or not _same_value(value, expected_value) for value in actual_values):
                    frozen_control_issues.append({"scope": "frozen_stage1_config", "field": field, "expected": expected_value, "values": actual_values, "run_dirs": [str(path) for _, path in all_entries]})
    cam_entries = [entry for candidate, entries in candidate_configs.items() if candidate[0].startswith("cam_") for entry in entries]
    for issue in _consistency_issues(cam_entries, CAM_RECIPE_FIELDS):
        frozen_control_issues.append({"scope": "all_cam_candidates", **issue})
    for issue in candidate_recipe_issues + frozen_control_issues:
        invalid.append({"run_dir": issue["run_dirs"][0] if issue.get("run_dirs") else str(root), "issues": [f"stage2_control_mismatch:{issue['field']}"]})
    duplicate_cells = [{"cell_key": key, "run_dirs": [str(path) for path in paths]} for key, paths in discovered.items() if len(paths) > 1]
    missing_cells = [cell for key, cell in expected_by_key.items() if key not in discovered]
    unexpected_cells = [key for key in discovered if key not in expected_by_key]
    valid_count = sum(1 for key, paths in discovered.items() if key in expected_by_key and len(paths) == 1 and _run_is_valid(paths[0])[0])
    return {
        "expected_cell_count": len(expected), "discovered_cell_count": sum(len(v) for v in discovered.values()),
        "unique_discovered_cell_count": len(discovered), "valid_cell_count": valid_count,
        "missing_cells": missing_cells, "duplicate_cells": duplicate_cells, "unexpected_cells": unexpected_cells,
        "invalid_cells": invalid, "debug_truncated_runs": debug_runs,
        "config_mismatches": [item for item in invalid if any("config_mismatch" in issue for issue in item["issues"])],
        "teacher_fingerprint_issues": teacher_issues, "oof_artifact_issues": oof_issues,
        "candidate_recipe_issues": candidate_recipe_issues, "frozen_control_issues": frozen_control_issues,
        "passed": not (missing_cells or duplicate_cells or unexpected_cells or invalid or teacher_issues or oof_issues),
    }


def _candidate_runs(runs_dir: Path, report: dict[str, Any]):
    invalid_keys = {item.get("cell_key") for item in report["invalid_cells"]}
    result = {}
    for config_path in sorted(runs_dir.rglob("config.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not _is_student_config(config_path, config):
            continue
        key = cell_key(int(config.get("fold", -1)), str(config.get("condition")), int(config.get("cutout_m", 0)), float(config.get("cutout_fraction", 0.0) or 0.0))
        if key in invalid_keys or not config.get("oof_artifact") or not _resolve(config["oof_artifact"]).is_file():
            continue
        candidate_key = (str(config["condition"]), int(config.get("cutout_m", 0)), float(config.get("cutout_fraction", 0.0) or 0.0))
        result.setdefault(candidate_key, []).append((config, config_path.parent))
    return result


def _selected_fold_metrics(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    frame = pd.read_csv(run_dir / "metrics.csv")
    if frame.empty or "val_log_loss" not in frame:
        raise ValueError(f"Stage 2 metrics have no validation trajectory: {run_dir}.")
    row = frame.loc[frame["val_log_loss"].astype(float).idxmin()]
    return {
        "fold": int(config["fold"]), "best_epoch": int(config.get("best_epoch", row["epoch"])),
        "minimum_validation_log_loss": float(row["val_log_loss"]), "epoch_at_minimum_validation_log_loss": int(row["epoch"]),
        "accuracy_at_minimum_validation_log_loss": float(row["val_accuracy"]), "auroc_at_minimum_validation_log_loss": float(row["val_auroc"]),
        "brier_at_minimum_validation_log_loss": float(row["val_brier_score"]), "ece_at_minimum_validation_log_loss": float(row["val_ece"]),
        "teacher_checkpoint_sha256": config.get("teacher_checkpoint_sha256"), "run_dir": _portable_or_key(run_dir),
    }


def _candidate_recipe(entries):
    config = entries[0][0]
    return {
        "condition": str(config["condition"]), "M": int(config.get("cutout_m", 0)), "fraction": float(config.get("cutout_fraction", 0.0) or 0.0),
        "cam_layer": config.get("cam_layer"), "saliency_candidate_percent": float(config["saliency_candidate_percent"]),
        "min_foreground_fraction": float(config["min_foreground_fraction"]), "student_seed_policy": config.get("student_seed_policy"),
        "student_model": config.get("student_model", config.get("model")), "selected_stage1_config_fingerprint": config.get("selected_stage1_config_fingerprint"),
        "preprocessing_fingerprint": config.get("preprocessing_fingerprint"), "fold_assignment_fingerprint": config.get("fold_assignment_fingerprint"),
        "student_max_cv_epochs": int(config.get("student_max_cv_epochs", config.get("epochs"))),
        "early_stopping_patience": int(config.get("early_stopping_patience", config.get("patience", 15))),
    }


def select_best_overall_and_masked(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not candidates:
        raise ValueError("No valid Stage 2 candidates were found.")
    best_overall = min(candidates, key=lambda row: (row["selection_score"], row["condition"], row["M"], row["fraction"]))
    masked = [row for row in candidates if row["condition"] in MASKED_CONDITIONS]
    if not masked:
        raise ValueError("Stage 2 requires at least one masked candidate for Submission #2.")
    return best_overall, min(masked, key=lambda row: (row["selection_score"], row["condition"], row["M"], row["fraction"]))


def select_stage2_candidate(runs_dir: str | Path, *, expected_folds: int, output_path: str | Path, conditions=CONDITIONS, m_values=DEFAULT_M, fractions=DEFAULT_FRACTIONS, calibration_method: str = "temperature", frozen_config: dict[str, Any] | None = None, summary_dir: str | Path | None = None) -> dict[str, Any]:
    report = integrity_check(runs_dir, expected_folds=expected_folds, conditions=conditions, m_values=m_values, fractions=fractions, frozen_config=frozen_config)
    summary_dir = Path(summary_dir) if summary_dir is not None else Path(runs_dir).parent / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "integrity_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not report["passed"]:
        raise ValueError("Stage 2 integrity check failed; see summary/integrity_report.json.")
    grouped = _candidate_runs(Path(runs_dir), report)
    candidates, candidate_fold_rows = [], []
    for (condition, m_value, fraction), entries in sorted(grouped.items()):
        entries = sorted(entries, key=lambda item: int(item[0].get("fold", -1)))
        recipe = _candidate_recipe(entries)
        fold_metrics = sorted([_selected_fold_metrics(config, run_dir) for config, run_dir in entries], key=lambda row: row["fold"])
        best_epochs = [int(row["best_epoch"]) for row in fold_metrics]
        for row in fold_metrics:
            candidate_fold_rows.append({"condition": condition, "M": int(m_value), "fraction": float(fraction), **{key: value for key, value in row.items() if key != "run_dir"}})
        logits_parts, target_parts, fold_parts = [], [], []
        for config, _run_dir in entries:
            with np.load(_resolve(config["oof_artifact"])) as payload:
                logits_parts.append(np.asarray(payload["logits"], dtype=np.float64))
                target_parts.append(np.asarray(payload["targets"], dtype=np.int64))
                fold_parts.append(np.asarray(payload["fold"], dtype=np.int64) if "fold" in payload else np.full(len(payload["targets"]), int(config["fold"]), dtype=np.int64))
        logits, targets, fold_ids = np.concatenate(logits_parts), np.concatenate(target_parts), np.concatenate(fold_parts)
        if set(int(v) for v in fold_ids.tolist()) != set(range(int(expected_folds))):
            raise ValueError(f"Candidate {condition} M{m_value} fraction {fraction:.2f} lacks an OOF partition for every fold.")
        result = fit_candidate_calibration(logits, targets, fold_ids, method=calibration_method)
        calibration = dict(result["final_calibration"])
        calibration_path = summary_dir / "calibration" / (f"{condition}_M{m_value}_fraction{fraction:.2f}".replace(".", "p") + ".json")
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path.write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")
        candidate = {
            **recipe, "candidate_fold_metrics": fold_metrics, "selected_candidate_fold_best_epochs": best_epochs,
            "final_stage2_training_epoch_rule": "median_selected_stage2_fold_best_epoch_round_half_up",
            "final_stage2_training_epochs": median_round_half_up(best_epochs),
            "stage1_final_training_epochs": int(entries[0][0].get("stage1_final_training_epochs", entries[0][0].get("final_training_epochs"))),
            "raw_oof_log_loss": float(result["raw_metrics"]["log_loss"]), "raw_oof_auroc": float(result["raw_metrics"]["auroc"]),
            "raw_oof_brier_score": float(result["raw_metrics"]["brier_score"]), "raw_oof_accuracy": float(result["raw_metrics"]["accuracy"]),
            "cross_fitted_calibrated_oof_log_loss": float(result["cross_fitted_metrics"]["log_loss"]),
            "cross_fitted_calibrated_oof_auroc": float(result["cross_fitted_metrics"]["auroc"]),
            "cross_fitted_calibrated_oof_brier_score": float(result["cross_fitted_metrics"]["brier_score"]),
            "cross_fitted_calibrated_oof_accuracy": float(result["cross_fitted_metrics"]["accuracy"]),
            "cross_fitted_calibrated_oof_ece": float(result["cross_fitted_metrics"]["ece"]),
            "raw_cv_log_loss": float(result["raw_metrics"]["log_loss"]), "calibrated_cv_log_loss": float(result["cross_fitted_metrics"]["log_loss"]),
            "final_fitted_calibration_method": calibration.get("method", "raw"), "final_fitted_temperature": float(calibration.get("temperature", 1.0)),
            "calibration": calibration, "calibration_path": _portable_or_key(calibration_path),
            "calibration_provenance": "candidate_own_fold_OOF_logits_only", "n_oof_samples": int(len(targets)),
            "n_cv_folds": int(len(set(fold_ids.tolist()))), "fold_ids": sorted(set(int(v) for v in fold_ids.tolist())),
            "selection_score": float(result["cross_fitted_metrics"]["log_loss"]),
        }
        candidates.append(candidate)
    best_overall, best_masked = select_best_overall_and_masked(candidates)
    payload = {
        "selection_basis": "cross_fitted_calibrated_oof_log_loss", "best_overall": best_overall, "best_masked": best_masked,
        "selected_stage2_fold_best_epochs": best_masked["selected_candidate_fold_best_epochs"],
        "final_stage2_training_epoch_rule": best_masked["final_stage2_training_epoch_rule"],
        "final_stage2_training_epochs": best_masked["final_stage2_training_epochs"], "selected": best_masked,
        "candidates": candidates, "integrity_report": _portable_or_key(summary_dir / "integrity_report.json"),
    }
    pd.DataFrame([{
        "condition": row["condition"], "M": row["M"], "fraction": row["fraction"], "cam_layer": row["cam_layer"],
        "saliency_candidate_percent": row["saliency_candidate_percent"], "min_foreground_fraction": row["min_foreground_fraction"],
        "student_model": row["student_model"], "student_max_cv_epochs": row["student_max_cv_epochs"],
        "early_stopping_patience": row["early_stopping_patience"], "final_stage2_training_epochs": row["final_stage2_training_epochs"],
        "raw_oof_log_loss": row["raw_oof_log_loss"], "raw_oof_auroc": row["raw_oof_auroc"], "raw_oof_brier_score": row["raw_oof_brier_score"],
        "raw_oof_accuracy": row["raw_oof_accuracy"], "cross_fitted_calibrated_oof_log_loss": row["cross_fitted_calibrated_oof_log_loss"],
        "cross_fitted_calibrated_oof_auroc": row["cross_fitted_calibrated_oof_auroc"], "cross_fitted_calibrated_oof_brier_score": row["cross_fitted_calibrated_oof_brier_score"],
        "cross_fitted_calibrated_oof_accuracy": row["cross_fitted_calibrated_oof_accuracy"], "cross_fitted_calibrated_oof_ece": row["cross_fitted_calibrated_oof_ece"],
        "final_fitted_calibration_method": row["final_fitted_calibration_method"], "final_fitted_temperature": row["final_fitted_temperature"],
        "n_oof_samples": row["n_oof_samples"], "n_cv_folds": row["n_cv_folds"],
    } for row in candidates]).to_csv(summary_dir / "candidate_oof_metrics.csv", index=False)
    pd.DataFrame(candidate_fold_rows).to_csv(summary_dir / "candidate_fold_metrics.csv", index=False)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


select_candidates = select_stage2_candidate


def generate_stage2_summary(runs_dir: str | Path = "runs/dat_parkinsons/resnet18_3d", summary_dir: str | Path = "runs/dat_parkinsons/summary", *, expected_folds: int = 5, selection_path: str | Path | None = None, conditions=CONDITIONS, m_values=DEFAULT_M, fractions=DEFAULT_FRACTIONS, frozen_config: dict[str, Any] | None = None) -> dict[str, Any]:
    runs_root, output = Path(runs_dir), Path(summary_dir)
    report = integrity_check(runs_root, expected_folds=expected_folds, conditions=conditions, m_values=m_values, fractions=fractions, frozen_config=frozen_config)
    output.mkdir(parents=True, exist_ok=True)
    (output / "integrity_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not report["passed"]:
        raise ValueError("Stage 2 integrity check failed; summary cannot proceed.")
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8")) if selection_path and Path(selection_path).is_file() else {}
    if selection:
        (output / "selected_best_masked.json").write_text(json.dumps(selection.get("best_masked", {}), indent=2, sort_keys=True), encoding="utf-8")
    summaries = []
    for config_path in sorted(runs_root.rglob("config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("stage") != 2 or "teachers" in {part.lower() for part in config_path.parent.parts}:
            continue
        frame = pd.read_csv(config_path.parent / "metrics.csv")
        best = frame.loc[frame["val_log_loss"].astype(float).idxmin()]
        tail = frame.tail(min(20, len(frame)))
        with np.load(_resolve(config["oof_artifact"])) as payload:
            logits, targets = np.asarray(payload["logits"], dtype=np.float64), np.asarray(payload["targets"], dtype=np.int64)
        summaries.append({
            "run_dir": _portable_or_key(config_path.parent), "fold": int(config["fold"]), "condition": str(config["condition"]),
            "M": int(config.get("cutout_m", 0)), "fraction": float(config.get("cutout_fraction", 0.0) or 0.0),
            "epochs_completed": int(frame["epoch"].max()), "minimum_validation_log_loss": float(best["val_log_loss"]),
            "epoch_at_minimum_validation_log_loss": int(best["epoch"]), "accuracy_at_minimum_validation_log_loss": float(best["val_accuracy"]),
            "auroc_at_minimum_validation_log_loss": float(best["val_auroc"]), "brier_at_minimum_validation_log_loss": float(best["val_brier_score"]),
            "ece_at_minimum_validation_log_loss": float(best["val_ece"]), "maximum_validation_auroc": float(frame["val_auroc"].astype(float).max()),
            "final_validation_log_loss": float(frame.iloc[-1]["val_log_loss"]), "final_validation_accuracy": float(frame.iloc[-1]["val_accuracy"]),
            "final20_logloss_mean": float(tail["val_log_loss"].mean()), "final20_logloss_std": float(tail["val_log_loss"].std(ddof=1)) if len(tail) > 1 else 0.0,
            "oof_raw_log_loss": float(compute_binary_metrics(targets, logits=logits)["log_loss"]),
            "validation_trajectory_source": "outer_fold_metrics.csv", "oof_raw_source": "candidate_fold_OOF_logits",
            "oof_calibrated_source": "candidate_oof_metrics.csv_only",
        })
    per_run = pd.DataFrame(summaries)
    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(table_dir / "per_run_metrics.csv", index=False)
    excluded = {"run_dir", "condition", "fold", "validation_trajectory_source", "oof_raw_source", "oof_calibrated_source"}
    numeric = [column for column in per_run.columns if column not in excluded and pd.api.types.is_numeric_dtype(per_run[column])]
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
    candidate_rows = [{
        "condition": candidate["condition"], "M": candidate["M"], "fraction": candidate["fraction"],
        "cam_layer": candidate.get("cam_layer"), "saliency_candidate_percent": candidate.get("saliency_candidate_percent"),
        "min_foreground_fraction": candidate.get("min_foreground_fraction"), "student_max_cv_epochs": candidate.get("student_max_cv_epochs"),
        "early_stopping_patience": candidate.get("early_stopping_patience"), "raw_oof_log_loss": candidate.get("raw_oof_log_loss"),
        "cross_fitted_calibrated_oof_log_loss": candidate.get("cross_fitted_calibrated_oof_log_loss"),
        "final_stage2_training_epochs": candidate.get("final_stage2_training_epochs"),
        "selected_candidate_fold_best_epochs": json.dumps(candidate.get("selected_candidate_fold_best_epochs", [])),
        "calibration_provenance": candidate.get("calibration_provenance"),
    } for candidate in selection.get("candidates", [])]
    pd.DataFrame(candidate_rows).to_csv(table_dir / "candidate_oof_metrics.csv", index=False)
    fold_rows = []
    for candidate in selection.get("candidates", []):
        for row in candidate.get("candidate_fold_metrics", []):
            fold_rows.append({"condition": candidate["condition"], "M": candidate["M"], "fraction": candidate["fraction"], **row})
    pd.DataFrame(fold_rows).to_csv(table_dir / "candidate_fold_metrics.csv", index=False)
    indexed = {(int(row.fold), str(row.condition), int(row.M), float(row.fraction)): row for row in per_run.itertuples()}
    none_by_fold = {int(row.fold): row for row in per_run[per_run.condition == "none"].itertuples()}
    paired_rows = []

    def add_pair(comparison, treatment, reference, treatment_m, reference_m, fold, fraction, treatment_row, reference_row):
        paired_rows.append({
            "comparison": comparison, "fold": int(fold), "condition": treatment, "reference_condition": reference,
            "M": int(treatment_m), "reference_M": int(reference_m), "fraction": float(fraction),
            "reference_logloss": float(reference_row.minimum_validation_log_loss), "treatment_logloss": float(treatment_row.minimum_validation_log_loss),
            "logloss_improvement": float(reference_row.minimum_validation_log_loss - treatment_row.minimum_validation_log_loss),
            "reference_accuracy": float(reference_row.accuracy_at_minimum_validation_log_loss), "treatment_accuracy": float(treatment_row.accuracy_at_minimum_validation_log_loss),
            "accuracy_difference": float(treatment_row.accuracy_at_minimum_validation_log_loss - reference_row.accuracy_at_minimum_validation_log_loss),
            "effect_source": "paired_outer_fold_validation_at_minimum_validation_log_loss",
        })
    for (fold, condition, m_value, fraction), treatment_row in indexed.items():
        if condition == "none":
            continue
        reference_row = none_by_fold.get(fold) if condition == "random" else indexed.get((fold, "random", m_value, fraction))
        if reference_row is not None:
            add_pair("random_vs_none" if condition == "random" else f"{condition}_vs_random", condition, "none" if condition == "random" else "random", m_value, 0 if condition == "random" else m_value, fold, fraction, treatment_row, reference_row)
    for (fold, condition, m_value, fraction), treatment_row in indexed.items():
        if m_value == 8 and condition != "none" and (reference_row := indexed.get((fold, condition, 4, fraction))) is not None:
            add_pair("M8_vs_M4", condition, condition, 8, 4, fold, fraction, treatment_row, reference_row)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(table_dir / "paired_effects.csv", index=False)
    plots = output / "plots"
    effects = paired[paired["comparison"].isin(["cam_low_vs_random", "cam_high_vs_random"])].copy() if "comparison" in paired.columns else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not effects.empty:
        effects["condition"] = effects["comparison"].str.replace("_vs_random", "", regex=False)
        effects.boxplot(column="logloss_improvement", by="condition", ax=ax)
    ax.set_title("CAM versus random paired validation effect"); ax.set_xlabel(""); ax.set_ylabel("Random log loss - CAM log loss"); plt.suptitle(""); save_research_figure(fig, plots / "paired_cam_effects_vs_random.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (condition, m_value, fraction), group in per_run.groupby(["condition", "M", "fraction"]):
        for _, row in group.iterrows():
            run_ref = _resolve(row.run_dir)
            if (run_ref / "metrics.csv").is_file():
                frame = pd.read_csv(run_ref / "metrics.csv")
                ax.plot(frame["epoch"], frame["val_log_loss"], alpha=0.35, label=f"{condition} M{m_value} f{fraction:.2f}")
    ax.set_title("DaT Stage 2 validation learning curves"); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation log loss"); ax.grid(alpha=.25); save_research_figure(fig, plots / "learning_curves.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if candidate_rows:
        candidate_frame = pd.DataFrame(candidate_rows)
        for condition, group in candidate_frame.groupby("condition"):
            ax.plot(group["fraction"], group["cross_fitted_calibrated_oof_log_loss"], "o-", label=condition)
    ax.set_title("Candidate-specific cross-fitted calibrated OOF log loss"); ax.set_xlabel("Fraction"); ax.set_ylabel("OOF log loss"); ax.grid(alpha=.25); ax.legend(); save_research_figure(fig, plots / "candidate_calibrated_oof.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not aggregate.empty:
        ax.scatter(aggregate["minimum_validation_log_loss_mean"], aggregate["minimum_validation_log_loss_std"])
        for _, row in aggregate.iterrows():
            ax.annotate(str(row["condition"]), (row["minimum_validation_log_loss_mean"], row["minimum_validation_log_loss_std"]))
    ax.set_title("Mean versus variability"); ax.set_xlabel("Mean best validation log loss"); ax.set_ylabel("Between-fold std"); ax.grid(alpha=.25); save_research_figure(fig, plots / "mean_vs_variability.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not aggregate.empty:
        for condition, group in aggregate.groupby("condition"):
            ax.plot(group["fraction"], group["final20_logloss_std_mean"], "o-", label=condition)
    ax.set_title("Training stability"); ax.set_xlabel("Fraction"); ax.set_ylabel("Final-20 validation log-loss std"); ax.grid(alpha=.25); ax.legend(); save_research_figure(fig, plots / "stability_by_fraction.png")
    report_payload = {
        "run_count": int(len(per_run)), "expected_run_count": int(report["expected_cell_count"]),
        "selection_basis": "cross_fitted_calibrated_oof_log_loss", "selection_lower_is_better": True,
        "best_overall": selection.get("best_overall", {}), "best_masked": selection.get("best_masked", {}),
        "validation_trajectory_metrics": {"source": "each outer fold metrics.csv", "fields": ["minimum_validation_log_loss", "epoch_at_minimum_validation_log_loss", "accuracy_at_minimum_validation_log_loss", "auroc_at_minimum_validation_log_loss", "brier_at_minimum_validation_log_loss", "ece_at_minimum_validation_log_loss"]},
        "oof_raw_metrics": {"source": "concatenated fold-best-checkpoint OOF logits", "fields": ["oof_raw_log_loss"]},
        "oof_cross_fitted_calibrated_metrics": {"source": "candidate_oof_metrics.csv; candidate-specific fold-aware calibration", "fields": ["cross_fitted_calibrated_oof_log_loss", "cross_fitted_calibrated_oof_brier_score", "cross_fitted_calibrated_oof_ece"]},
        "final_fitted_calibration": {"source": "all candidate OOF logits for selected candidate; packaged with final model", "field": "calibration"},
        "stage1_lineage": selection.get("best_masked", {}).get("selected_stage1_config_fingerprint"),
        "selected_stage2_fold_best_epochs": selection.get("best_masked", {}).get("selected_candidate_fold_best_epochs"),
        "final_stage2_training_epoch_rule": selection.get("best_masked", {}).get("final_stage2_training_epoch_rule"),
        "final_stage2_training_epochs": selection.get("best_masked", {}).get("final_stage2_training_epochs"),
        "paired_effects": {"source": "paired_effects.csv", "logloss_definition": "reference_logloss - treatment_logloss", "accuracy_definition": "treatment_accuracy - reference_accuracy"},
        "integrity_report": "integrity_report.json", "privacy": "No UIDs, patient-level predictions, OOF arrays, or local absolute paths are written.",
    }
    (output / "summary_report.json").write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"per_run": per_run, "aggregate": aggregate, "paired": paired, "report": report}


generate_summary = generate_stage2_summary


# ================================================================
# Final model training and Stage 1/Stage 2 handoff
# ================================================================


def _load_stage1_teacher(stage1_model_dir: str | Path, config: dict[str, Any]):
    model_dir = Path(stage1_model_dir)
    checkpoint = next((model_dir / name for name in ("final_model.pt", "best_model.pt") if (model_dir / name).is_file()), None)
    if checkpoint is None:
        raise FileNotFoundError(f"Stage 1 final checkpoint not found under {model_dir}.")
    model = build_resnet18_3d(num_classes=int(config.get("num_classes", 2)), n_input_channels=int(config.get("n_input_channels", 1)), dropout=float(config.get("dropout", 0.0)), base_channels=int(config.get("base_channels", 32)))
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("Stage 1 checkpoint does not contain a model state dictionary.")
    model.load_state_dict({str(k).removeprefix("module."): v for k, v in state.items()}, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint, sha256_file(checkpoint)


def train_final_dat_model(data_dir: str | Path, best_config_path: str | Path, output_dir: str | Path, calibration_path: str | Path | None = None, *, selected: dict[str, Any] | None = None, stage1_model_dir: str | Path | None = None, calibration_payload: dict[str, Any] | None = None, seed: int = 42, num_workers: int = 0, max_train_batches: int = 0) -> dict[str, Any]:
    best_config = json.loads(Path(best_config_path).read_text(encoding="utf-8"))
    records = load_dat_records(data_dir)
    preprocessing = best_config["preprocessing"]
    selected_provided = selected is not None
    selected = dict(selected or {})
    if selected_provided and "condition" not in selected:
        raise ValueError("A Stage 2 selected candidate must explicitly contain condition.")
    condition = str(selected.get("condition", "none"))
    if selected_provided and condition == "none":
        raise ValueError("Submission #2 must use a masked Stage 2 candidate, not condition='none'.")
    m_value = int(selected["M"] if selected_provided else selected.get("M", 0))
    fraction = float(selected["fraction"] if selected_provided else selected.get("fraction", 0.0))
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported final DaT condition: {condition}")
    if condition != "none" and (m_value <= 0 or fraction <= 0):
        raise ValueError("Masked final models require positive M and fraction.")
    stage1_epochs = int(best_config.get("final_training_epochs", best_config.get("epochs", 100)))
    if calibration_payload is None and selected_provided:
        calibration_payload = selected.get("calibration")
    if selected_provided and calibration_payload is None:
        raise ValueError("Selected Stage 2 candidate must provide its candidate-specific calibration payload.")
    if selected_provided:
        required = ("cam_layer", "saliency_candidate_percent", "min_foreground_fraction", "final_stage2_training_epochs", "selected_candidate_fold_best_epochs", "calibration_provenance")
        missing = [field for field in required if field not in selected]
        if missing:
            raise ValueError("Selected Stage 2 candidate is missing required recipe fields: " + ", ".join(missing))
        epoch_budget = int(selected["final_stage2_training_epochs"])
        cam_layer = str(selected["cam_layer"]); saliency_candidate_percent = float(selected["saliency_candidate_percent"]); min_foreground_fraction = float(selected["min_foreground_fraction"])
    else:
        epoch_budget = stage1_epochs; cam_layer = str(best_config.get("cam_layer", "auto")); saliency_candidate_percent = float(best_config.get("saliency_candidate_percent", 10.0)); min_foreground_fraction = float(best_config.get("min_foreground_fraction", 0.75))
    if epoch_budget <= 0:
        raise ValueError("The final DaT model requires a positive epoch budget.")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    config = deepcopy(best_config)
    config.update({
        "stage": 1 if condition == "none" else 2, "condition": condition, "cutout_mode": condition,
        "cutout_m": m_value if condition != "none" else 0, "cutout_fraction": fraction, "epochs": epoch_budget,
        "final_training_epochs": epoch_budget, "stage1_final_training_epochs": stage1_epochs,
        "num_workers": 0 if condition.startswith("cam_") else int(num_workers), "max_train_batches": int(max_train_batches or 0),
        "max_val_batches": 0, "debug": bool(max_train_batches), "cam_layer": cam_layer,
        "saliency_candidate_percent": saliency_candidate_percent, "min_foreground_fraction": min_foreground_fraction,
    })
    if selected_provided:
        config.update({
            "student_max_cv_epochs": int(selected["student_max_cv_epochs"]), "student_seed_policy": selected["student_seed_policy"], "student_model": selected["student_model"],
            "selected_candidate_fold_best_epochs": [int(value) for value in selected["selected_candidate_fold_best_epochs"]],
            "final_stage2_training_epoch_rule": selected.get("final_stage2_training_epoch_rule", "median_selected_stage2_fold_best_epoch_round_half_up"),
            "selected_stage1_config_fingerprint": selected["selected_stage1_config_fingerprint"], "fold_assignment_fingerprint": selected["fold_assignment_fingerprint"],
            "selected_candidate_fold_metrics": selected.get("candidate_fold_metrics", []), "calibration_provenance": selected["calibration_provenance"],
            "selected_candidate_calibration": calibration_payload,
        })
    cache_dir = REPO_ROOT / "artifacts" / "dat_parkinsons" / "cache" / "preprocessed"
    base_train = DatDataset(records, preprocessing, train=True, augment=bool(best_config.get("spatial_augmentation", False)), seed=seed, cache_dir=cache_dir)
    teacher = teacher_checkpoint = teacher_hash = None
    if condition.startswith("cam_"):
        if stage1_model_dir is None:
            raise ValueError("Final CAM masking requires the exact Stage 1 final model directory.")
        teacher, teacher_checkpoint, teacher_hash = _load_stage1_teacher(stage1_model_dir, best_config)
        config.update({"teacher_checkpoint": _portable_or_key(teacher_checkpoint), "teacher_checkpoint_sha256": teacher_hash, "teacher_lineage": "stage1_final_unmasked_checkpoint"})
        train_dataset = CutoutAugmentedDataset(
            base_dataset=base_train, cutout_mode=condition, cutout_m=m_value, cutout_size=None, cutout_area=fraction,
            mean=(0.0,), std=(1.0,), seed=seed, saliency_candidate_percent=saliency_candidate_percent, teacher_model=teacher,
            cam_layer=cam_layer, cam_cache_dir=str(REPO_ROOT / "artifacts" / "dat_parkinsons" / "cam_cache" / "final_stage2"),
            cam_cache_settings={"dataset": "dat_parkinsons", "student_model": "resnet18_3d", "teacher_checkpoint_sha256": teacher_hash, "cam_layer": cam_layer, "spatial_dims": 3, "preprocessing": preprocessing, "min_foreground_fraction": min_foreground_fraction},
            min_foreground_fraction=min_foreground_fraction,
        )
    elif condition == "none":
        train_dataset = base_train
    else:
        train_dataset = CutoutAugmentedDataset(base_dataset=base_train, cutout_mode=condition, cutout_m=m_value, cutout_size=None, cutout_area=fraction, mean=(0.0,), std=(1.0,), seed=seed, min_foreground_fraction=min_foreground_fraction)
    result = fit_dat_model_fixed_epochs(train_dataset, config, seed=seed, run_dir=output, epochs=epoch_budget, max_train_batches=max_train_batches or None)
    model_config = {
        "model": "resnet18_3d", "num_classes": int(config.get("num_classes", 2)), "n_input_channels": int(config.get("n_input_channels", 1)),
        "base_channels": int(config.get("base_channels", 32)), "dropout": float(config.get("dropout", 0.0)), "training_condition": condition,
        "training_cutout_m": m_value, "training_cutout_fraction": fraction, "cam_layer": cam_layer,
        "saliency_candidate_percent": saliency_candidate_percent, "min_foreground_fraction": min_foreground_fraction,
        "final_training_epochs": epoch_budget,
    }
    if selected_provided:
        model_config.update({"student_max_cv_epochs": int(selected["student_max_cv_epochs"]), "final_stage2_training_epochs": epoch_budget, "selected_candidate_fold_best_epochs": [int(value) for value in selected["selected_candidate_fold_best_epochs"]], "calibration_provenance": selected["calibration_provenance"], "calibration_method": str((calibration_payload or {}).get("method", "raw")), "calibration_temperature": float((calibration_payload or {}).get("temperature", 1.0)), "selected_candidate_calibration": calibration_payload})
    (output / "model_config.json").write_text(json.dumps(model_config, indent=2, sort_keys=True), encoding="utf-8")
    (output / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2, sort_keys=True), encoding="utf-8")
    if calibration_payload is None:
        if calibration_path is None:
            raise ValueError("A calibration file or calibration payload is required.")
        calibration_payload = load_calibration(calibration_path)
    (output / "calibration.json").write_text(json.dumps(calibration_payload, indent=2, sort_keys=True), encoding="utf-8")
    provenance = {
        "pipeline": "stage1_unmasked" if condition == "none" else "stage2_masked", "stage": int(config["stage"]), "selected_condition": condition,
        "M": m_value, "fraction": fraction, "condition": condition, "cam_layer": cam_layer, "saliency_candidate_percent": saliency_candidate_percent, "min_foreground_fraction": min_foreground_fraction,
        "selected_stage1_config_fingerprint": selected.get("selected_stage1_config_fingerprint", best_config.get("config_fingerprint", fingerprint(best_config))),
        "stage1_selection_objective": best_config.get("selection_objective", "cross_fitted_calibrated_oof_log_loss"), "stage1_raw_oof_metrics": best_config.get("raw_oof_metrics"),
        "stage1_cross_fitted_calibrated_oof_metrics": best_config.get("cross_fitted_calibrated_oof_metrics"), "stage1_calibration_method": best_config.get("calibration_method"),
        "preprocessing_fingerprint": best_config.get("preprocessing_fingerprint", fingerprint(preprocessing)), "final_epoch_budget": epoch_budget,
        "final_stage2_training_epochs": epoch_budget if selected_provided else None, "stage1_final_training_epochs": stage1_epochs,
        "selected_stage2_fold_best_epochs": selected.get("selected_candidate_fold_best_epochs") if selected_provided else None,
        "final_stage2_training_epoch_rule": selected.get("final_stage2_training_epoch_rule") if selected_provided else None,
        "seed": int(seed), "checkpoint_selection": "final_scheduled_epoch", "research_valid": not bool(max_train_batches), "git_commit": current_git_commit(),
        "final_checkpoint_sha256": sha256_file(output / "final_model.pt"), "calibration_provenance": selected.get("calibration_provenance", "stage1_oof_logits_only" if condition == "none" else "selected_stage2_candidate_oof_logits_only"),
        "calibration_candidate": {"condition": condition, "M": m_value, "fraction": fraction},
    }
    if selected_provided:
        provenance.update({
            "stage2_selection_objective": "cross_fitted_calibrated_oof_log_loss", "stage2_selection_score": selected.get("selection_score"),
            "stage2_raw_oof_metrics": {
                "log_loss": selected.get("raw_oof_log_loss"), "auroc": selected.get("raw_oof_auroc"),
                "brier_score": selected.get("raw_oof_brier_score"), "accuracy": selected.get("raw_oof_accuracy"),
            },
            "stage2_cross_fitted_calibrated_oof_metrics": {"log_loss": selected.get("cross_fitted_calibrated_oof_log_loss"), "auroc": selected.get("cross_fitted_calibrated_oof_auroc"), "brier_score": selected.get("cross_fitted_calibrated_oof_brier_score"), "accuracy": selected.get("cross_fitted_calibrated_oof_accuracy"), "ece": selected.get("cross_fitted_calibrated_oof_ece")},
            "selected_stage2_recipe": {"condition": condition, "M": m_value, "fraction": fraction, "cam_layer": cam_layer, "saliency_candidate_percent": saliency_candidate_percent, "min_foreground_fraction": min_foreground_fraction},
            "selected_candidate_fold_best_epochs": [int(value) for value in selected["selected_candidate_fold_best_epochs"]], "selected_candidate_fold_metrics": selected.get("candidate_fold_metrics", []),
            "student_max_cv_epochs": int(selected["student_max_cv_epochs"]), "student_seed_policy": selected["student_seed_policy"], "student_model": selected["student_model"], "fold_assignment_fingerprint": selected["fold_assignment_fingerprint"],
            "final_stage2_training_epoch_rule": selected.get("final_stage2_training_epoch_rule", "median_selected_stage2_fold_best_epoch_round_half_up"),
        })
    if teacher_checkpoint is not None:
        provenance.update({"teacher_lineage": "stage1_final_unmasked_checkpoint", "stage1_teacher_checkpoint": _portable_or_key(teacher_checkpoint), "stage1_teacher_checkpoint_sha256": teacher_hash, "teacher_checkpoint_path": _portable_or_key(teacher_checkpoint), "teacher_checkpoint_sha256": teacher_hash})
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    return {"output_dir": output, "model_config": model_config, "config": config, "provenance": provenance, "result": result}


def train_final_stage1_model(data_dir, best_config_path, output_dir, calibration_path, **kwargs):
    return train_final_dat_model(data_dir, best_config_path, output_dir, calibration_path, **kwargs)


def train_final_stage2_model(data_dir, best_config_path, output_dir, selected, *, stage1_model_dir, seed=42, num_workers=0, max_train_batches=0):
    return train_final_dat_model(data_dir, best_config_path, output_dir, selected=selected, stage1_model_dir=stage1_model_dir, calibration_payload=selected.get("calibration"), seed=seed, num_workers=num_workers, max_train_batches=max_train_batches)


# ================================================================
# Single-entry-point handoff
# ================================================================


def load_valid_stage1_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Stage 1 configuration exists but is invalid:\n{path}\n{exc}") from exc
    problems = []
    if not isinstance(config, dict):
        problems.append("the JSON root must be an object")
    else:
        if config.get("dataset") != "dat_parkinsons": problems.append("dataset must be 'dat_parkinsons'")
        if config.get("model") != "resnet18_3d": problems.append("model must be 'resnet18_3d'")
        if type(config.get("stage")) is not int or config.get("stage") != 1: problems.append("stage must be 1")
        if config.get("research_valid") is not True: problems.append("research_valid must be true")
        if not isinstance(config.get("preprocessing"), dict): problems.append("preprocessing must be a JSON object")
        if config.get("preprocessing_fingerprint") is not None and isinstance(config.get("preprocessing"), dict) and config["preprocessing_fingerprint"] != fingerprint(config["preprocessing"]): problems.append("preprocessing_fingerprint does not match preprocessing")
        if config.get("config_fingerprint") is not None:
            fingerprinted = dict(config); fingerprinted.pop("config_fingerprint", None); fingerprinted.pop("oof_artifact", None)
            if config["config_fingerprint"] != fingerprint(fingerprinted): problems.append("config_fingerprint does not match selected configuration")
    if problems:
        raise ValueError(f"Stage 1 configuration exists but is invalid:\n{path}\n" + "\n".join(f"- {problem}" for problem in problems))
    return config


def ensure_stage1(args) -> Path:
    path = Path(args.best_config)
    if path.exists():
        if not path.is_file():
            raise ValueError(f"Stage 1 configuration exists but is not a file: {path}")
        load_valid_stage1_config(path)
        print(f"[Stage 2] Found valid Stage 1 configuration at {path}; skipping Stage 1.")
        return path
    print(f"[Stage 2] No Stage 1 configuration found at {path}; running Stage 1 first.")
    run_stage1(args)
    if not path.is_file():
        raise RuntimeError(f"Stage 1 completed without creating the expected best_config.json at {path}")
    load_valid_stage1_config(path)
    return path


def run_stage2(args) -> dict[str, Any]:
    best_config_path = ensure_stage1(args)
    if any(str(condition).startswith("cam_") for condition in args.conditions):
        model_dir = Path(args.stage1_model_dir)
        if not any((model_dir / name).is_file() for name in ("final_model.pt", "best_model.pt")):
            raise FileNotFoundError(f"The valid Stage 1 configuration was found, but the final Stage 1 checkpoint is missing under {model_dir}.")
    grid_result = run_stage2_grid(args)
    best_config = json.loads(best_config_path.read_text(encoding="utf-8"))
    n_folds = int(best_config.get("cv_folds", grid_result["folds"]))
    selection = select_stage2_candidate(args.output_dir, expected_folds=n_folds, output_path=args.selected_model, conditions=args.conditions, m_values=args.m_values, fractions=args.fractions, calibration_method=args.calibration_method, frozen_config=best_config, summary_dir=args.summary_dir)
    generate_stage2_summary(args.output_dir, args.summary_dir, expected_folds=n_folds, selection_path=args.selected_model, conditions=args.conditions, m_values=args.m_values, fractions=args.fractions, frozen_config=best_config)
    selected = selection["best_masked"]
    final = train_final_stage2_model(args.data_dir, best_config_path, args.final_model_dir, selected, stage1_model_dir=args.stage1_model_dir, seed=args.seed, num_workers=args.num_workers, max_train_batches=0)
    from dat_submission import build_submission

    archive = build_submission(args.final_model_dir, args.stage2_submission_zip)
    return {
        "stage2_research_output": str(args.output_dir), "summary_dir": str(args.summary_dir), "selection": str(args.selected_model),
        "selected_condition": selected["condition"], "selected_M": selected["M"], "selected_fraction": selected["fraction"],
        "final_stage2_training_epochs": selected["final_stage2_training_epochs"], "final_model_dir": str(final["output_dir"]),
        "submission_zip": str(archive), "best_overall": selection["best_overall"], "best_masked": selection["best_masked"],
    }


def check_data(data_dir: str | Path, *, target_spacing=None, target_shape=DEFAULT_TARGET_SHAPE, limit=0) -> dict[str, Any]:
    return check_dat_dataset(data_dir, target_spacing=target_spacing, target_shape=target_shape, limit=limit)
