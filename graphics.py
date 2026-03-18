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


def _load_validation_metrics(metrics_csv):
    df = pd.read_csv(metrics_csv)
    if "eval_split" not in df.columns:
        return pd.DataFrame()
    val_df = df[df["eval_split"] == "val"].copy()
    if val_df.empty:
        return val_df
    needed = ["epoch", "eval_loss", "eval_acc1"]
    missing = [col for col in needed if col not in val_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {metrics_csv}: {missing}")
    return val_df


def plot_variant_validation_comparison(
    original_metrics_csv,
    low_saliency_metrics_csv,
    out_png,
    output_model_name,
    input_models,
    dataset_name,
):
    original_df = _load_validation_metrics(original_metrics_csv)
    low_df = _load_validation_metrics(low_saliency_metrics_csv)

    if original_df.empty or low_df.empty:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    inputs_str = ", ".join(input_models)
    fig.suptitle(
        f"Validation Comparison | Output Model: {output_model_name} | "
        f"Mask Inputs: {inputs_str} | Dataset: {dataset_name}",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0]
    ax.plot(original_df["epoch"], original_df["eval_loss"], color="#1f77b4", linewidth=2, label="Original")
    ax.plot(low_df["epoch"], low_df["eval_loss"], color="#ff7f0e", linewidth=2, label="Low Saliency")
    _setup_subplot(ax, "Epoch", "Loss", "Validation Loss")

    ax = axes[1]
    ax.plot(original_df["epoch"], original_df["eval_acc1"] * 100, color="#1f77b4", linewidth=2, label="Original")
    ax.plot(low_df["epoch"], low_df["eval_acc1"] * 100, color="#ff7f0e", linewidth=2, label="Low Saliency")
    _setup_subplot(ax, "Epoch", "Accuracy (%)", "Validation Accuracy@1")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    return True