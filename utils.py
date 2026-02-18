import os
import random
import torch
import matplotlib.pyplot as plt
import pandas as pd


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