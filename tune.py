import itertools
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from utils import best_val_from_metrics, plot_tuning_results, final_test_from_metrics
from logger import get_logger
import time

# Will be set by tune_hyperparameters at startup so all functions use same file
logger = None


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class TuningConfig:
    runs_root: Path = Path("./runs_cifar100_resnet18")
    tuning_dirname: str = "tuning_results"
    train_entry: str = "train.py"
    timeout_sec: int = 3600  # 1 hour


PARAM_GRID: Dict[str, List[Any]] = {
    "epochs": [100],
    "lr": [0.05, 0.1, 0.2],
    "weight_decay": [5e-4, 2e-3],
    "momentum": [0.95],
    "nesterov": [False, True],
    "label_smoothing": [0.0, 0.1],
    "scheduler": ["cosine", "multistep"],
    "warmup_epochs": [0, 5],
    "min_lr": [0.0],
    "gamma": [0.1],
    "milestones": ["100,150"],
    "dropout": [0.0],
    "val_split": [0.1],
}

FIXED_PARAMS: Dict[str, Any] = {
    "batch_size": 128,
    "num_workers": 2,
    "seed": 42,
    "log_every": 100,
    "amp": True,
}


# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def cartesian_product(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combos.append(dict(zip(keys, values)))
    return combos


def prune_combinations(combos: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply simple pruning rules to cut unhelpful configs."""
    kept = []
    for c in combos:
        # Only test warmup with cosine (common choice)
        if c["scheduler"] == "multistep" and c["warmup_epochs"] > 0:
            continue
        kept.append(c)
    return kept


def compute_multistep_milestones(epochs: int) -> str:
    if epochs == 150:
        return "90,120"
    if epochs == 200:
        return "100,150"
    return f"{int(0.5 * epochs)},{int(0.75 * epochs)}"


def with_scheduler_dependent_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy where multistep milestones are consistent with the epoch budget."""
    out = dict(params)
    if out.get("scheduler") == "multistep":
        out["milestones"] = compute_multistep_milestones(int(out["epochs"]))
    return out


def format_run_name(all_params: Dict[str, Any], fixed_params: Dict[str, Any]) -> str:
    wd = float(all_params["weight_decay"])
    ms = all_params["milestones"] if all_params["scheduler"] == "multistep" else "na"
    return (
        f"tune_ep{all_params['epochs']}_bs{fixed_params['batch_size']}_lr{all_params['lr']}"
        f"_wd{wd:.0e}_m{all_params['momentum']}_nest{int(bool(all_params['nesterov']))}"
        f"_ls{all_params['label_smoothing']}_sch{all_params['scheduler']}_wu{all_params['warmup_epochs']}"
        f"_ms{ms}"
    )


def build_train_cmd(train_entry: str, run_name: str, params: Dict[str, Any]) -> List[str]:
    """
    Build CLI args in a consistent way:
    - boolean flags (amp/nesterov) become --flag when True
    - everything else becomes --key value
    """
    cmd = ["python", train_entry, "--run_name", run_name]

    bool_flags = {"amp", "nesterov"}
    for key, value in params.items():
        if key in bool_flags:
            if bool(value):
                cmd.append(f"--{key}")
            continue
        cmd.extend([f"--{key}", str(value)])

    return cmd


def tail(text: Optional[str], n: int = 500) -> str:
    if not text:
        return ""
    return text[-n:]


def metrics_csv_path(cfg: TuningConfig, run_name: str) -> Path:
    return cfg.runs_root / run_name / "metrics.csv"


# -----------------------------
# Main
# -----------------------------

def tune_hyperparameters(cfg: TuningConfig = TuningConfig()) -> None:
    # create tuning dir and setup a single log file for this tuning run
    tuning_dir = ensure_dir(cfg.runs_root / cfg.tuning_dirname)
    global logger
    log_root = Path.cwd() / "log"
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_root / f"tune_{timestamp}.log"
    logger = get_logger(__name__, log_file=log_file)

    combos = prune_combinations(cartesian_product(PARAM_GRID))
    logger.info(f"Running {len(combos)} training configurations...")
    logger.info(f"Results will be saved to {tuning_dir}\n")

    results: List[Dict[str, Any]] = []

    for idx, grid_params in enumerate(combos, 1):
        all_params = {**FIXED_PARAMS, **grid_params}
        all_params = with_scheduler_dependent_params(all_params)

        run_name = format_run_name(all_params, FIXED_PARAMS)
        cmd = build_train_cmd(cfg.train_entry, run_name, all_params)

        logger.info(f"[{idx}/{len(combos)}] Running: {run_name}")
        logger.info(f"  Command: {' '.join(cmd)}")

        result_info = run_single_training_run(cfg, run_name, all_params, cmd)
        
        # Read final test accuracy from metrics and log completion status
        if result_info["status"] == "success":
            csv_path = metrics_csv_path(cfg, run_name)
            final_test_acc1 = final_test_from_metrics(csv_path)
            if final_test_acc1 is not None:
                result_info["final_test_acc1"] = final_test_acc1
                logger.info(f"Completed successfully - Test Acc: {final_test_acc1 * 100:.2f}%")
            else:
                logger.info("Completed successfully")
        
        results.append(result_info)
        logger.info("")

    results_file = tuning_dir / "tuning_results.json"
    results_file.write_text(json.dumps(results, indent=2))

    logger.info(f"\nTuning complete! Results saved to {results_file}")
    print_summary(cfg, results)
    plot_tuning_results(results, tuning_dir)


def run_single_training_run(
    cfg: TuningConfig, run_name: str, params: Dict[str, Any], cmd: List[str]
) -> Dict[str, Any]:
    """Run a single training subprocess and return basic execution info."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.timeout_sec)
        status = "success" if proc.returncode == 0 else "failed"
        
        if status == "failed":
            logger.error(f"Failed with exit code {proc.returncode}")
            if proc.stderr:
                for line in [l for l in proc.stderr.splitlines() if l.strip()][-10:]:
                    logger.error(f"    {line}")

        return {
            "run_name": run_name,
            "params": params,
            "status": status,
            "exit_code": proc.returncode,
            "stderr": tail(proc.stderr, 500),
        }

    except subprocess.TimeoutExpired:
        logger.warning("Timeout (exceeded 1 hour)")
        return {"run_name": run_name, "params": params, "status": "timeout"}

    except Exception as e:
        logger.error(f"Exception: {e}")
        return {"run_name": run_name, "params": params, "status": "error", "error": str(e)}


def print_summary(cfg: TuningConfig, results: List[Dict[str, Any]]) -> None:
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
        mpath = metrics_csv_path(cfg, r["run_name"])
        best_val = best_val_from_metrics(mpath)
        if best_val is not None:
            ranked.append((r, best_val))

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

        out_csv = cfg.runs_root / cfg.tuning_dirname / "ranked_by_val.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        logger.info(f"\nSaved ranking to {out_csv}")


if __name__ == "__main__":
    tune_hyperparameters()
