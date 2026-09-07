from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path


def _setup_subplot(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def _build_plot_title(model_name=None, dataset_name=None):
    parts = ["Training & Evaluation Metrics"]
    if model_name:
        parts.append(f"Model: {model_name}")
    if dataset_name:
        parts.append(f"Dataset: {dataset_name}")
    return " | ".join(parts)


def plot_metrics(metrics_csv, run_dir, model_name=None, dataset_name=None):
    df = pd.read_csv(metrics_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(_build_plot_title(model_name=model_name, dataset_name=dataset_name), fontsize=16, fontweight="bold")
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], "b-", linewidth=2, label="Train Loss")
    ax.plot(df["epoch"], df["eval_loss"],  "r-", linewidth=2, label="Eval Loss")
    _setup_subplot(ax, "Epoch", "Loss", "Loss (Train vs Eval)")
    ax = axes[1]
    ax.plot(df["epoch"], df["train_acc1"] * 100, "b-", linewidth=2, label="Train Acc1")
    ax.plot(df["epoch"], df["eval_acc1"] * 100,  "r-", linewidth=2, label="Eval Acc1")
    _setup_subplot(ax, "Epoch", "Accuracy (%)", "Accuracy@1 (Train vs Eval)")
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "metrics_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dat_metrics(metrics_csv: str | Path, output_path: str | Path) -> None:
    """Plot the reusable DaT training trajectory without owning pipeline policy."""
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


def plot_stage1_trials(rows: list[dict], path: str | Path, selected: dict | None = None) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    trial_ids = [row["trial_id"] for row in rows]
    axis.plot(trial_ids, [row["raw_oof_log_loss"] for row in rows], "o-", label="Raw OOF log loss")
    axis.plot(trial_ids, [row["cross_fitted_calibrated_oof_log_loss"] for row in rows], "s-", label="Cross-fitted calibrated OOF log loss")
    if selected:
        axis.scatter([selected["trial_id"]], [selected["cross_fitted_calibrated_oof_log_loss"]], marker="*", s=140, zorder=4, label="Selected trial")
    axis.set_xlabel("Trial")
    axis.set_ylabel("OOF log loss (lower is better)")
    axis.set_title("DaT Stage 1 optimization")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_research_figure(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
