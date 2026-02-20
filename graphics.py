import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple
from logger import get_logger

logger = get_logger(__name__)

def _setup_subplot(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

def plot_metrics(metrics_csv, run_dir):
    try:
        df = pd.read_csv(metrics_csv)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
        axes[0, 0].plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Train Loss')
        _setup_subplot(axes[0, 0], 'Epoch', 'Loss', 'Training Loss')
        axes[0, 1].plot(df['epoch'], df['eval_loss'], 'r-', linewidth=2, label='Eval Loss')
        _setup_subplot(axes[0, 1], 'Epoch', 'Loss', 'Evaluation Loss')
        axes[1, 0].plot(df['epoch'], df['train_acc1'] * 100, 'b-', linewidth=2, label='Train Acc1')
        axes[1, 0].plot(df['epoch'], df['train_acc5'] * 100, 'b--', linewidth=1.5, label='Train Acc5')
        _setup_subplot(axes[1, 0], 'Epoch', 'Accuracy (%)', 'Training Accuracy')
        axes[1, 1].plot(df['epoch'], df['eval_acc1'] * 100, 'r-', linewidth=2, label='Eval Acc1')
        axes[1, 1].plot(df['epoch'], df['eval_acc5'] * 100, 'r--', linewidth=1.5, label='Eval Acc5')
        _setup_subplot(axes[1, 1], 'Epoch', 'Accuracy (%)', 'Evaluation Accuracy')
        plt.tight_layout()
        plot_path = os.path.join(run_dir, 'metrics_plot.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Metrics plot saved to {plot_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting metrics: {e}")


def plot_tuning_results(results, tuning_dir):
    try:
        successful = [r for r in results if r["status"] == "success"]
        if not successful:
            logger.info("No successful runs to plot")
            return
        run_names = []
        test_accs = []
        lr_values = []
        wd_values = []
        for r in successful:
            if "final_test_acc1" in r:
                run_names.append(r["run_name"])
                test_accs.append(r["final_test_acc1"] * 100)
                lr_values.append(r["params"]["lr"])
                wd_values.append(r["params"]["weight_decay"])
        if not test_accs:
            logger.info("No test accuracy data to plot")
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Hyperparameter Tuning Results', fontsize=14, fontweight='bold')
        x_pos = range(len(run_names))
        axes[0].bar(x_pos, test_accs, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Configuration')
        axes[0].set_ylabel('Test Accuracy (%)')
        axes[0].set_title('Final Test Accuracy by Configuration')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(range(1, len(run_names) + 1))
        axes[0].grid(True, alpha=0.3, axis='y')
        scatter = axes[1].scatter(lr_values, test_accs, c=wd_values, cmap='viridis', 
                                  s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
        axes[1].set_xlabel('Learning Rate')
        axes[1].set_ylabel('Test Accuracy (%)')
        axes[1].set_title('Test Accuracy vs Learning Rate')
        axes[1].grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes[1])
        cbar.set_label('Weight Decay')
        plt.tight_layout()
        plot_path = tuning_dir / 'tuning_results_plot.png'
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        logger.info(f"Tuning results plot saved to {plot_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting tuning results: {e}")


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
    ranked: List[Tuple[Dict[str, Any], float]] = []
    for r in successful:
        if "best_val_acc" in r:
            ranked.append((r, r["best_val_acc"]))
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
        rows = []
        for r, best_val in ranked:
            p = r["params"]
            rows.append(
                {
                    "run_name": r["run_name"],
                    "best_val_acc1": best_val,
                    "lr": p["lr"],
                    "epochs": p["epochs"],
                    "weight_decay": p["weight_decay"],
                    "momentum": p["momentum"],
                    "nesterov": p["nesterov"],
                    "label_smoothing": p["label_smoothing"],
                    "scheduler": p["scheduler"],
                    "warmup_epochs": p["warmup_epochs"],
                    "milestones": p.get("milestones", ""),
                }
            )
        out_csv = tuning_dir / "ranked_by_val.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        logger.info(f"\nSaved ranking to {out_csv}")
