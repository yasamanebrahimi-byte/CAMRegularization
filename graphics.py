import matplotlib.pyplot as plt
import pandas as pd
import os


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
    plot_path = os.path.join(run_dir, f"metrics_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
