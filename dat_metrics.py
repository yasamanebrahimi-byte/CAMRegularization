"""Probability-first metrics and plotting for the DaT binary task."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected logits with shape [N,2], got {logits.shape}.")
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    return probs[:, 1]


def safe_log_loss(targets: Iterable[int], probabilities: Iterable[float]) -> float:
    y = np.asarray(list(targets), dtype=np.int64)
    p = np.asarray(list(probabilities), dtype=np.float64)
    if y.size == 0 or p.size != y.size:
        raise ValueError("Targets and probabilities must be non-empty and have equal length.")
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1]))


def expected_calibration_error(targets: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    result = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & ((probabilities < right) if right < 1 else (probabilities <= right))
        if not np.any(mask):
            continue
        result += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(targets[mask].mean()))
    return float(result)


def compute_binary_metrics(
    targets: Iterable[int],
    *,
    logits: np.ndarray | None = None,
    probabilities: Iterable[float] | None = None,
) -> dict[str, float]:
    y = np.asarray(list(targets), dtype=np.int64)
    if logits is not None:
        logits_array = np.asarray(logits)
        p = probabilities_from_logits(logits_array)
    elif probabilities is not None:
        p = np.asarray(list(probabilities), dtype=np.float64)
    else:
        raise ValueError("Provide logits or probabilities.")
    if y.size == 0 or p.size != y.size:
        raise ValueError("Targets and predictions must be non-empty and have equal length.")
    p = np.clip(p.astype(np.float64), 1e-7, 1.0 - 1e-7)
    hard = (p >= 0.5).astype(np.int64)
    result = {
        "log_loss": safe_log_loss(y, p),
        "brier_score": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, hard)),
        "ece": expected_calibration_error(y, p),
        "sensitivity": float(np.sum((hard == 1) & (y == 1)) / max(1, np.sum(y == 1))),
        "specificity": float(np.sum((hard == 0) & (y == 0)) / max(1, np.sum(y == 0))),
    }
    try:
        result["auroc"] = float(roc_auc_score(y, p))
    except ValueError:
        result["auroc"] = float("nan")
    return result


def aggregate_fold_metrics(frames: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not frames:
        return {}
    keys = sorted({key for frame in frames for key, value in frame.items() if isinstance(value, (float, int))})
    result = {}
    for key in keys:
        values = np.asarray([float(frame[key]) for frame in frames if np.isfinite(float(frame[key]))], dtype=float)
        if values.size == 0:
            result[key] = {"mean": float("nan"), "std": float("nan"), "sem": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
            continue
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        sem = std / np.sqrt(values.size)
        result[key] = {
            "mean": float(np.mean(values)),
            "std": std,
            "sem": float(sem),
            "ci95_low": float(np.mean(values) - 1.96 * sem),
            "ci95_high": float(np.mean(values) + 1.96 * sem),
        }
    return result


def plot_dat_metrics(metrics_csv: str | Path, output_path: str | Path) -> None:
    frame = pd.read_csv(metrics_csv)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="validation")
    axes[0].set_title("Cross-entropy loss")
    axes[1].plot(frame["epoch"], frame["train_accuracy"], label="train")
    axes[1].plot(frame["epoch"], frame["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[2].plot(frame["epoch"], frame["val_log_loss"], label="validation log loss")
    axes[2].set_title("Validation probability quality")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
