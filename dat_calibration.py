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
    calibrated = np.empty(len(targets), dtype=np.float64)
    all_indices = np.arange(len(targets))
    fold_calibrations = []
    for validation in validation_folds:
        validation = np.asarray(list(validation), dtype=np.int64)
        training = np.setdiff1d(all_indices, validation, assume_unique=False)
        calibration = fit_calibration(logits[training], targets[training], method=method)
        calibrated[validation] = calibrated_probabilities(logits[validation], calibration)
        fold_calibrations.append(calibration)
    return calibrated, fold_calibrations


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

