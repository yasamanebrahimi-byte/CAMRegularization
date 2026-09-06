"""Inference-only fixed calibration helper for the packaged submission."""

from __future__ import annotations

import numpy as np


def apply_fixed_calibration(logits: np.ndarray, calibration: dict) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Inference calibration expects logits with shape [N,2].")
    method = str(calibration.get("method", "raw")).lower()
    if method == "temperature":
        temperature = float(calibration.get("temperature", 1.0))
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Packaged calibration temperature must be finite and positive.")
        logits = logits / temperature
    elif method != "raw":
        raise ValueError(f"Unsupported packaged calibration method '{method}'.")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True))[:, 1]

