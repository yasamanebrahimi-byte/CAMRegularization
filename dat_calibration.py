"""Training-only probability calibration for DaT logits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from dat_metrics import compute_binary_metrics, probabilities_from_logits


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    temperature = float(temperature)
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Temperature must be finite and positive.")
    return probabilities_from_logits(logits / temperature)


def fit_temperature(logits: np.ndarray, targets: np.ndarray, max_iter: int = 100) -> float:
    logits_tensor = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    targets_tensor = torch.as_tensor(np.asarray(targets), dtype=torch.long)
    if logits_tensor.ndim != 2 or logits_tensor.shape[1] != 2:
        raise ValueError("Temperature scaling expects logits with shape [N,2].")
    if logits_tensor.shape[0] != targets_tensor.numel() or targets_tensor.numel() == 0:
        raise ValueError("Logits and targets must have equal non-zero length.")
    parameter = nn.Parameter(torch.zeros((), dtype=torch.float64))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([parameter], lr=0.1, max_iter=int(max_iter), line_search_fn="strong_wolfe")

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


def cross_fitted_calibration(
    logits: np.ndarray,
    targets: np.ndarray,
    validation_folds: Sequence[Sequence[int]],
    method: str = "temperature",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
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
            raise ValueError("Cross-fitted calibration requires training samples outside each validation fold.")
        calibration = fit_calibration(logits[training], targets[training], method=method)
        calibrated[validation] = calibrated_probabilities(logits[validation], calibration)
        fold_calibrations.append(calibration)
        seen.extend(validation.tolist())
    if sorted(seen) != list(range(len(targets))):
        raise ValueError("Calibration folds must partition every OOF sample exactly once.")
    return calibrated, fold_calibrations


def fit_candidate_calibration(
    logits: np.ndarray,
    targets: np.ndarray,
    fold_ids: np.ndarray,
    method: str = "temperature",
) -> dict[str, Any]:
    """Fit and score one candidate using only that candidate's fold OOF logits.

    ``fold_ids`` preserves the outer validation assignment while allowing the
    candidate's per-fold arrays to be concatenated in any stable order.
    """
    logits = np.asarray(logits)
    targets = np.asarray(targets)
    fold_ids = np.asarray(fold_ids)
    if logits.ndim != 2 or logits.shape[1] != 2 or len(targets) != len(logits) or len(fold_ids) != len(targets):
        raise ValueError("Candidate OOF logits, targets, and fold_ids must be aligned [N,2], [N], [N].")
    if len(targets) == 0 or not np.isfinite(logits).all():
        raise ValueError("Candidate OOF logits must be finite and non-empty.")
    unique_folds = sorted(set(int(value) for value in fold_ids.tolist()))
    validation_folds = [np.flatnonzero(fold_ids == fold).tolist() for fold in unique_folds]
    cross_fitted_probs, fold_calibrations = cross_fitted_calibration(
        logits, targets, validation_folds, method=method
    )
    raw_metrics = compute_binary_metrics(targets, logits=logits)
    cross_fitted_metrics = compute_binary_metrics(targets, probabilities=cross_fitted_probs)
    final = fit_calibration(logits, targets, method=method)
    final_metrics = compute_binary_metrics(targets, probabilities=calibrated_probabilities(logits, final))
    final.update({
        "provenance": "candidate_oof_logits_only",
        "n_oof_samples": int(len(targets)),
        "n_cv_folds": int(len(unique_folds)),
        "fold_calibrations": fold_calibrations,
        "raw_oof_log_loss": float(raw_metrics["log_loss"]),
        "cross_fitted_calibrated_oof_log_loss": float(cross_fitted_metrics["log_loss"]),
        "final_all_oof_calibrated_log_loss": float(final_metrics["log_loss"]),
    })
    return {
        "raw_metrics": raw_metrics,
        "cross_fitted_metrics": cross_fitted_metrics,
        "final_calibration": final,
        "fold_calibrations": fold_calibrations,
        "fold_ids": unique_folds,
    }


def evaluate_calibration(logits: np.ndarray, targets: np.ndarray, calibration: dict[str, Any]) -> dict[str, float]:
    probabilities = calibrated_probabilities(logits, calibration)
    return compute_binary_metrics(targets, probabilities=probabilities)


def save_calibration(path: str | Path, calibration: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")


def load_calibration(path: str | Path) -> dict[str, Any]:
    calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(calibration, dict):
        raise ValueError("Calibration file must contain a JSON object.")
    return calibration
