import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from logger import get_logger

logger = get_logger(__name__)

"""Configure common subplot properties."""
def _setup_subplot(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

"""Plot training and evaluation metrics from CSV file."""
def plot_metrics(metrics_csv, run_dir):
    try:
        df = pd.read_csv(metrics_csv)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
        
        # Plot 1: Training Loss
        axes[0, 0].plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Train Loss')
        _setup_subplot(axes[0, 0], 'Epoch', 'Loss', 'Training Loss')
        
        # Plot 2: Evaluation Loss
        axes[0, 1].plot(df['epoch'], df['eval_loss'], 'r-', linewidth=2, label='Eval Loss')
        _setup_subplot(axes[0, 1], 'Epoch', 'Loss', 'Evaluation Loss')
        
        # Plot 3: Training Accuracy (top-1)
        axes[1, 0].plot(df['epoch'], df['train_acc1'] * 100, 'b-', linewidth=2, label='Train Acc1')
        axes[1, 0].plot(df['epoch'], df['train_acc5'] * 100, 'b--', linewidth=1.5, label='Train Acc5')
        _setup_subplot(axes[1, 0], 'Epoch', 'Accuracy (%)', 'Training Accuracy')
        
        # Plot 4: Evaluation Accuracy (top-1)
        axes[1, 1].plot(df['epoch'], df['eval_acc1'] * 100, 'r-', linewidth=2, label='Eval Acc1')
        axes[1, 1].plot(df['epoch'], df['eval_acc5'] * 100, 'r--', linewidth=1.5, label='Eval Acc5')
        _setup_subplot(axes[1, 1], 'Epoch', 'Accuracy (%)', 'Evaluation Accuracy')
        
        plt.tight_layout()
        
        # Save the figure
        plot_path = os.path.join(run_dir, 'metrics_plot.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Metrics plot saved to {plot_path}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting metrics: {e}")

"""Plot tuning results comparing different configurations."""
def plot_tuning_results(results, tuning_dir):
    try:
        successful = [r for r in results if r["status"] == "success"]
        
        if not successful:
            logger.info("No successful runs to plot")
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
            logger.info("No test accuracy data to plot")
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
        logger.info(f"Tuning results plot saved to {plot_path}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting tuning results: {e}")
