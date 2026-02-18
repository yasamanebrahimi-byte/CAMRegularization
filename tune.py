import subprocess
import itertools
import json
import re
from pathlib import Path
import pandas as pd

from utils import *

"""Run grid search over hyperparameter combinations."""
def tune_hyperparameters():
    # Define hyperparameter grids
    param_grid = {
        "epochs": [100],
        "lr": [0.05, 0.1, 0.2],
        "weight_decay": [5e-4, 2e-3],
        "momentum": [0.95],
        "nesterov": [False, True],
        "label_smoothing": [0.0, 0.1],
        "scheduler": ["cosine","multistep"],
        "warmup_epochs": [0, 5],
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
    filtered = []
    for c in param_combinations:
        # Example pruning rules:
        # (A) Only test warmup with cosine (optional but common)
        if c["scheduler"] == "multistep" and c["warmup_epochs"] > 0:
            continue
        # (B) If cosine, gamma/milestones don't matter (doesn't reduce count unless you vary them)
        filtered.append(c)

    param_combinations = filtered

    print(f"Running {len(param_combinations)} training configurations...")
    print(f"Results will be saved to {tuning_dir}\n")

    results=[]
    for idx, params in enumerate(param_combinations, 1):
        all_params = {**fixed_params, **params}

        # Make multistep milestones match the epoch budget (ignore for cosine)
        if all_params["scheduler"] == "multistep":
            ep = all_params["epochs"]
            if ep == 150:
                all_params["milestones"] = "90,120"
            elif ep == 200:
                all_params["milestones"] = "100,150"
            else:
                all_params["milestones"] = f"{int(0.5*ep)},{int(0.75*ep)}"

        run_name = (
            f"tune_ep{all_params['epochs']}_bs{fixed_params['batch_size']}_lr{all_params['lr']}"
            f"_wd{all_params['weight_decay']:.0e}_m{all_params['momentum']}_nest{int(all_params['nesterov'])}"
            f"_ls{all_params['label_smoothing']}_sch{all_params['scheduler']}_wu{all_params['warmup_epochs']}"
            f"_ms{all_params['milestones'] if all_params['scheduler']=='multistep' else 'na'}"
        )

        cmd=["python","train.py","--run_name",run_name]
        for key, value in all_params.items():
            if key=="amp" and value:
                cmd.append("--amp")
            elif key!="amp":
                if key=="nesterov":
                    if value: cmd.append("--nesterov")
                else:
                    cmd.extend([f"--{key}", str(value)])
        
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
            results.append(result_info)
            
        except subprocess.TimeoutExpired:
            print(f"Timeout (exceeded 1 hour)")
            results.append({
                "run_name": run_name,
                "params": all_params,
                "status": "timeout",
            })
        except Exception as e:
            print(f"Exception: {str(e)}")
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
        print(f"\nBest test accuracy (for reference): {best_test_acc*100:.2f}% ({best_test_run['run_name']})")
        params = best_test_run["params"]
        print(f"   lr={params['lr']}, epochs={params['epochs']}, wd={params['weight_decay']}, val_split={params['val_split']}")
    
    # Show best runs by best validation accuracy (max over epochs)
    if successful:
        ranked=[]
        for r in successful:
            metrics_path = Path(f"./runs_cifar100_resnet18/{r['run_name']}/metrics.csv")
            best_val = best_val_from_metrics(metrics_path)
            if best_val is not None:
                ranked.append((r, best_val))

        ranked.sort(key=lambda x: x[1], reverse=True)

        print("\nTop 10 runs by BEST val_acc1 (max over epochs):")
        for i, (r, best_val) in enumerate(ranked[:10], 1):
            p = r["params"]
            print(f"  {i}. {r['run_name']}: best_val_acc1={best_val:.6f}")
            print(f"     lr={p['lr']}, ep={p['epochs']}, wd={p['weight_decay']}, mom={p['momentum']}, nest={p['nesterov']}, ls={p['label_smoothing']}, sch={p['scheduler']}, wu={p['warmup_epochs']}, ms={p.get('milestones','')}")

        # Save full ranking table
        if ranked:
            rows=[]
            for r, best_val in ranked:
                rows.append({
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
                })
            df_rank = pd.DataFrame(rows)
            out_csv = Path("./runs_cifar100_resnet18/tuning_results/ranked_by_val.csv")
            df_rank.to_csv(out_csv, index=False)
            print(f"\nSaved ranking to {out_csv}")


if __name__ == "__main__":
    tune_hyperparameters()
