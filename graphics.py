import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple
from logger import get_logger
import os

logger = get_logger(__name__)


def _setup_subplot(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_metrics(metrics_csv, run_dir):
    df = pd.read_csv(metrics_csv)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training & Evaluation Metrics", fontsize=16, fontweight="bold")
    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_loss"], "b-", linewidth=2, label="Train Loss")
    ax.plot(df["epoch"], df["eval_loss"],  "r-", linewidth=2, label="Eval Loss")
    _setup_subplot(ax, "Epoch", "Loss", "Loss (Train vs Eval)")
    ax = axes[0, 1]
    ax.plot(df["epoch"], df["train_acc1"] * 100, "b-", linewidth=2, label="Train Acc1")
    ax.plot(df["epoch"], df["eval_acc1"] * 100,  "r-", linewidth=2, label="Eval Acc1")
    _setup_subplot(ax, "Epoch", "Accuracy (%)", "Accuracy@1 (Train vs Eval)")
    ax = axes[1, 0]
    if {"train_f1", "eval_f1"}.issubset(df.columns):
        ax.plot(df["epoch"], df["train_f1"] * 100, "b--", linewidth=2, label="Train F1")
        ax.plot(df["epoch"], df["eval_f1"] * 100,  "r--", linewidth=2, label="Eval F1")
        _setup_subplot(ax, "Epoch", "Score (%)", "Macro F1 (Train vs Eval)")
    else:
        ax.axis("off")
    ax = axes[1, 1]
    if {"train_loss", "eval_loss", "train_acc1", "eval_acc1"}.issubset(df.columns):
        ax.plot(df["epoch"], df["eval_loss"] - df["train_loss"], linewidth=2, label="Loss Gap (Eval-Train)")
        ax.plot(df["epoch"], (df["eval_acc1"] - df["train_acc1"]) * 100, linewidth=2, label="Acc1 Gap (Eval-Train)")
        if {"train_f1", "eval_f1"}.issubset(df.columns):
            ax.plot(df["epoch"], (df["eval_f1"] - df["train_f1"]) * 100, linewidth=2, label="F1 Gap (Eval-Train)")
        _setup_subplot(ax, "Epoch", "Gap", "Generalization Gap")
    else:
        ax.axis("off")
    plt.tight_layout()
    plot_path = os.path.join(run_dir, f"metrics_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Metrics plot saved to {plot_path}")
    plt.close()


def print_summary(tuning_dir: Path, results: List[Dict[str, Any]]) -> None:
    logger.info("\n" + "=" * 70)
    logger.info("TUNING SUMMARY")
    logger.info("=" * 70)
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]
    other = [r for r in results if r.get("status") not in {"success", "failed"}]
    logger.info(f"Total runs: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Other: {len(other)}")
    best_test = max((r for r in successful if "final_test_acc1" in r), default=None, key=lambda r: r["final_test_acc1"])
    if best_test:
        acc = best_test["final_test_acc1"]
        p = best_test["params"]
        logger.info(f"\nBest test accuracy (for reference): {acc * 100:.2f}% ({best_test['run_name']})")
        logger.info(f"   lr={p['lr']}, epochs={p['epochs']}, wd={p['weight_decay']}, val_split={p['val_split']}")
    ranked: List[Tuple[Dict[str, Any], float]] = [
        (r, r["best_val_acc"]) for r in successful if "best_val_acc" in r
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    if ranked:
        logger.info("\nTop 10 runs by BEST val_acc1 (max over epochs):")
        for i, (r, best_val) in enumerate(ranked[:10], 1):
            p = r["params"]
            logger.info(f"  {i}. {r['run_name']}: best_val_acc1={best_val:.6f}")
            logger.info(
                "     "
                f"lr={p['lr']}, ep={p['epochs']}, wd={p['weight_decay']}, mom={p['momentum']}, "
                f"nest={p['nesterov']}, ls={p['label_smoothing']}, sch={p['scheduler']}, "
                f"wu={p['warmup_epochs']}, ms={p.get('milestones','')}"
            )
        rows = [
            {
                "run_name": r["run_name"],
                "best_val_acc1": best_val,
                "lr": r["params"]["lr"],
                "epochs": r["params"]["epochs"],
                "weight_decay": r["params"]["weight_decay"],
                "momentum": r["params"]["momentum"],
                "nesterov": r["params"]["nesterov"],
                "label_smoothing": r["params"]["label_smoothing"],
                "scheduler": r["params"]["scheduler"],
                "warmup_epochs": r["params"]["warmup_epochs"],
                "milestones": r["params"].get("milestones", ""),
            }
            for r, best_val in ranked
        ]
        out_csv = tuning_dir / "ranked_by_val.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        logger.info(f"\nSaved ranking to {out_csv}")