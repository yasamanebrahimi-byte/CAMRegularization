import os
import random
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def accuracy_top1(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()


@torch.no_grad()
def accuracy_top5(logits, targets):
    """Compute top-5 accuracy."""
    _, top5_preds = logits.topk(5, dim=1)
    targets_expanded = targets.view(-1, 1).expand_as(top5_preds)
    return (top5_preds == targets_expanded).any(dim=1).float().mean().item()


def plot_metrics(metrics_csv, run_dir):
    """Plot training and evaluation metrics from CSV file."""
    try:
        df = pd.read_csv(metrics_csv)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
        
        # Plot 1: Training Loss
        axes[0, 0].plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Train Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Evaluation Loss
        axes[0, 1].plot(df['epoch'], df['eval_loss'], 'r-', linewidth=2, label='Eval Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Evaluation Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: Training Accuracy (top-1)
        axes[1, 0].plot(df['epoch'], df['train_acc1'] * 100, 'b-', linewidth=2, label='Train Acc1')
        axes[1, 0].plot(df['epoch'], df['train_acc5'] * 100, 'b--', linewidth=1.5, label='Train Acc5')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Evaluation Accuracy (top-1)
        axes[1, 1].plot(df['epoch'], df['eval_acc1'] * 100, 'r-', linewidth=2, label='Eval Acc1')
        axes[1, 1].plot(df['epoch'], df['eval_acc5'] * 100, 'r--', linewidth=1.5, label='Eval Acc5')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Evaluation Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save the figure
        plot_path = os.path.join(run_dir, 'metrics_plot.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Metrics plot saved to {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting metrics: {e}")

def best_val_from_metrics(metrics_path: Path):
    """Return best validation acc1 from a run's metrics.csv, or None if unavailable."""
    try:
        df = pd.read_csv(metrics_path)
        if "eval_split" not in df.columns or "eval_acc1" not in df.columns: 
            return None
        df_val = df[df["eval_split"] == "val"]
        if df_val.empty: 
            return None
        # eval_acc1 should already be numeric; coerce just in case
        best = pd.to_numeric(df_val["eval_acc1"], errors="coerce").max()
        return None if pd.isna(best) else float(best)
    except Exception:
        return None

def plot_tuning_results(results, tuning_dir):
    """Plot tuning results comparing different configurations."""
    try:
        successful = [r for r in results if r["status"] == "success"]
        
        if not successful:
            print("No successful runs to plot")
            return
        
        # Extract data for plotting
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
            print("No test accuracy data to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Hyperparameter Tuning Results', fontsize=14, fontweight='bold')
        
        # Plot 1: Test accuracy by run
        x_pos = range(len(run_names))
        axes[0].bar(x_pos, test_accs, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Configuration')
        axes[0].set_ylabel('Test Accuracy (%)')
        axes[0].set_title('Final Test Accuracy by Configuration')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(range(1, len(run_names) + 1))
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Scatter - Accuracy vs Learning Rate (colored by weight decay)
        unique_wd = sorted(set(wd_values))
        colors = plt.cm.viridis([(wd_values[i] - min(unique_wd)) / (max(unique_wd) - min(unique_wd) + 1e-6) 
                                  for i in range(len(wd_values))])
        
        scatter = axes[1].scatter(lr_values, test_accs, c=wd_values, cmap='viridis', 
                                  s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
        axes[1].set_xlabel('Learning Rate')
        axes[1].set_ylabel('Test Accuracy (%)')
        axes[1].set_title('Test Accuracy vs Learning Rate')
        axes[1].grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes[1])
        cbar.set_label('Weight Decay')
        
        plt.tight_layout()
        
        # Save the figure
        plot_path = tuning_dir / 'tuning_results_plot.png'
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        print(f"Tuning results plot saved to {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting tuning results: {e}")