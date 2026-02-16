import subprocess
import itertools
import os
import json
from pathlib import Path


def tune_hyperparameters():
    """Run grid search over hyperparameter combinations."""
    
    # Define hyperparameter grids
    param_grid = {
        "epochs": [30, 50],
        "lr": [0.05, 0.1, 0.25],
        "weight_decay": [1e-4, 5e-4],
        "dropout": [0.0],
        "val_split": [0.0, 0.1],
    }
    
    # Fixed parameters (same for all runs)
    fixed_params = {
        "batch_size": 128,
        "num_workers": 2,
        "momentum": 0.9,
        "seed": 42,
        "log_every": 100,
        "amp": True,
    }
    
    # Create tuning results directory
    tuning_dir = Path("./runs_cifar100_resnet18/tuning_results")
    tuning_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all parameter combinations
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Running {len(param_combinations)} training configurations...")
    print(f"Results will be saved to {tuning_dir}\n")
    
    results = []
    
    for idx, params in enumerate(param_combinations, 1):
        # Combine fixed and variable parameters
        all_params = {**fixed_params, **params}
        
        # Create a descriptive run name
        run_name = f"tune_ep{params['epochs']}_lr{params['lr']}_wd{params['weight_decay']:.0e}_do{params['dropout']}"
        
        # Build command line arguments
        cmd = ["python", "train.py", "--run_name", run_name]
        for key, value in all_params.items():
            if key == "amp" and value:
                cmd.append("--amp")
            elif key != "amp":
                cmd.extend([f"--{key}", str(value)])
        
        print(f"[{idx}/{len(param_combinations)}] Running: {run_name}")
        print(f"  Command: {' '.join(cmd)}")
        
        try:
            # Run training
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            # Check if run was successful
            if result.returncode == 0:
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


if __name__ == "__main__":
    tune_hyperparameters()
