import subprocess
import itertools
import os
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def tune_hyperparameters():
    """Run grid search over hyperparameter combinations."""
    # Define hyperparameter grids
    param_grid = {
        "epochs": [150, 200],
        "lr": [0.05, 0.1, 0.2],
        "weight_decay": [5e-4, 1e-3, 2e-3],
        "momentum": [0.9, 0.95],
        "nesterov": [False, True],
        "label_smoothing": [0.0, 0.05, 0.1],
        "scheduler": ["cosine","multistep"],
        "warmup_epochs": [0, 2, 5],
        "min_lr": [0.0],
        "gamma": [0.1],
        "milestones": ["100,150"],  # only used when scheduler=multistep
        "dropout": [0.0],
        "val_split": [0.1],
    }

    fixed_params = {
        "batch_size": 128,
        "num_workers": 2,
        "seed": 42,
        "log_every": 100,
        "amp": True,
    }

    tuning_dir = Path("./runs_cifar100_resnet18/tuning_results")
    tuning_dir.mkdir(parents=True, exist_ok=True)

    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Running {len(param_combinations)} training configurations...")
    print(f"Results will be saved to {tuning_dir}\n")

    results=[]
    for idx, params in enumerate(param_combinations, 1):
        all_params = {**fixed_params, **params}
        run_name = (
            f"tune_ep{params['epochs']}_bs{fixed_params['batch_size']}_lr{params['lr']}"
            f"_wd{params['weight_decay']:.0e}_m{params['momentum']}_nest{int(params['nesterov'])}"
            f"_ls{params['label_smoothing']}_sch{params['scheduler']}_wu{params['warmup_epochs']}"
        )

        cmd=["python","train.py","--run_name",run_name]
        for key, value in all_params.items():
            if key=="amp" and value: cmd.append("--amp")
            elif key!="amp":
                if key=="nesterov" and value: cmd.append("--nesterov")
                else: cmd.extend([f"--{key}", str(value)])
        
        print(f"[{idx}/{len(param_combinations)}] Running: {run_name}")
        print(f"  Command: {' '.join(cmd)}")
        
        try:
            # Run training
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            # Extract final test accuracy from stdout first
            final_test_acc1 = None
            if "Final test:" in result.stdout:
                for line in result.stdout.split('\n'):
                    if "Final test:" in line:
                        # Parse: "Final test: loss X.XXXX acc1 X.XX%"
                        match = re.search(r'acc1\s+([\d.]+)%', line)
                        if match:
                            final_test_acc1 = float(match.group(1)) / 100.0
                        break
            
            # Check if run was successful
            if result.returncode == 0:
                if final_test_acc1 is not None:
                    print(f"Completed successfully - Test Acc: {final_test_acc1*100:.2f}%")
                else:
                    print(f"Completed successfully")
                status = "success"
            else:
                print(f"Failed with exit code {result.returncode}")
                # Print full stderr for debugging
                if result.stderr:
                    error_lines = result.stderr.split('\n')
                    for line in error_lines[-10:]:  # Show last 10 lines of error
                        if line.strip():
                            print(f"    {line}")
                status = "failed"
            
            # Store result information
            result_info = {
                "run_name": run_name,
                "params": all_params,
                "status": status,
                "exit_code": result.returncode,
                "stderr": result.stderr[-500:] if result.stderr else "",  # Store last 500 chars of stderr
            }
            
            if final_test_acc1 is not None:
                result_info["final_test_acc1"] = final_test_acc1
            
            # Try to load metrics if available
            metrics_path = Path(f"./runs_cifar100_resnet18/{run_name}/metrics.csv")
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        # Get last line (final epoch results)
                        last_line = lines[-1].strip()
                        header = lines[0].strip().split(",")
                        values = last_line.split(",")
                        metrics = dict(zip(header, values))
                        result_info["final_metrics"] = metrics
            
            results.append(result_info)
            
        except subprocess.TimeoutExpired:
            print(f"Timeout (exceeded 1 hour)")
            results.append({
                "run_name": run_name,
                "params": all_params,
                "status": "timeout",
            })
        except Exception as e:
            print(f"  ✗ Exception: {str(e)}")
            results.append({
                "run_name": run_name,
                "params": all_params,
                "status": "error",
                "error": str(e),
            })
        
        print()
    
    # Save tuning results
    results_file = tuning_dir / "tuning_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTuning complete! Results saved to {results_file}")
    print_summary(results)
    plot_tuning_results(results, tuning_dir)


def print_summary(results):
    """Print summary of tuning results."""
    print("\n" + "="*70)
    print("TUNING SUMMARY")
    print("="*70)
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    other = [r for r in results if r["status"] not in ["success", "failed"]]
    
    print(f"Total runs: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Other: {len(other)}")
    
    # Find and display best test accuracy
    best_test_acc = None
    best_test_run = None
    for r in successful:
        if "final_test_acc1" in r:
            if best_test_acc is None or r["final_test_acc1"] > best_test_acc:
                best_test_acc = r["final_test_acc1"]
                best_test_run = r
    
    if best_test_acc is not None:
        print(f"\nBest test accuracy: {best_test_acc*100:.2f}% ({best_test_run['run_name']})")
        params = best_test_run["params"]
        print(f"   lr={params['lr']}, epochs={params['epochs']}, wd={params['weight_decay']}, val_split={params['val_split']}")
    
    # Show best runs by eval_acc1
    if successful:
        print("\nTop 5 runs by final eval_acc1:")
        ranked = []
        for r in successful:
            if "final_metrics" in r and "eval_acc1" in r["final_metrics"]:
                try:
                    acc = float(r["final_metrics"]["eval_acc1"])
                    ranked.append((r, acc))
                except (ValueError, TypeError):
                    pass
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        for i, (r, acc) in enumerate(ranked[:5], 1):
            params = r["params"]
            print(f"  {i}. {r['run_name']}: {acc:.6f}")
            print(f"     lr={params['lr']}, epochs={params['epochs']}, wd={params['weight_decay']}, dropout={params['dropout']}")


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


if __name__ == "__main__":
    tune_hyperparameters()
