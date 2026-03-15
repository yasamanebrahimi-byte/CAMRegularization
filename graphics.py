import matplotlib.pyplot as plt
import pandas as pd
import os


def _setup_subplot(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_metrics(metrics_csv, run_dir):
    df = pd.read_csv(metrics_csv)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training & Evaluation Metrics", fontsize=16, fontweight="bold")
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], "b-", linewidth=2, label="Train Loss")
    ax.plot(df["epoch"], df["eval_loss"],  "r-", linewidth=2, label="Eval Loss")
    _setup_subplot(ax, "Epoch", "Loss", "Loss (Train vs Eval)")
    ax = axes[1]
    ax.plot(df["epoch"], df["train_acc1"] * 100, "b-", linewidth=2, label="Train Acc1")
    ax.plot(df["epoch"], df["eval_acc1"] * 100,  "r-", linewidth=2, label="Eval Acc1")
    _setup_subplot(ax, "Epoch", "Accuracy (%)", "Accuracy@1 (Train vs Eval)")
    ax = axes[2]
    if {"train_f1", "eval_f1"}.issubset(df.columns):
        ax.plot(df["epoch"], df["train_f1"] * 100, "b--", linewidth=2, label="Train F1")
        ax.plot(df["epoch"], df["eval_f1"] * 100,  "r--", linewidth=2, label="Eval F1")
        _setup_subplot(ax, "Epoch", "Score (%)", "F1 (Train vs Eval)")
    else:
        ax.axis("off")
    plt.tight_layout()
    plot_path = os.path.join(run_dir, f"metrics_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()