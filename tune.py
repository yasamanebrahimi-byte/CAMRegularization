import itertools
import json
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse

from IOutils import build_parser, ensure_dir, make_run_dir, write_json, build_args_from_params
from utils import DEFAULT_DATASET, DEFAULT_MODEL
from graphics import plot_tuning_results, print_summary
from logger import get_logger
from train import train_with_config


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class TuningConfig:
    runs_root: Path = Path("./runs")  # Generic runs directory
    tuning_dirname: str = "tuning_results"
    dataset: str = DEFAULT_DATASET
    model: str = DEFAULT_MODEL


PARAM_GRID: Dict[str, List[Any]] = {
    "lr": [0.03, 0.05, 0.1],
    "weight_decay": [3e-4, 5e-4], 
    "label_smoothing": [0.0, 0.05], 
    "warmup_epochs": [0, 5], 
}

FIXED_PARAMS: Dict[str, Any] = {
    "epochs": 150,
    "data_dir": "./data",
    "batch_size": 128,
    "num_workers": 2,
    "seed": 42,
    "log_every": 100,
    "amp": True,
    "val_split": 0.1,
    "nesterov": True,
    "scheduler": "cosine",
    "min_lr": 1e-5,
    "momentum": 0.9
}

OPTIMAL_CONFIG_PATH = Path("data") / "optimal_config.json"


# -----------------------------
# Helpers
# -----------------------------
def cartesian_product(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combos.append(dict(zip(keys, values)))
    return combos


def format_run_name(all_params: Dict[str, Any], fixed_params: Dict[str, Any], dataset: str, model: str) -> str:
    wd = float(all_params["weight_decay"])
    return (
        f"tune_{model}_{dataset}_ep{all_params['epochs']}_bs{fixed_params['batch_size']}_lr{all_params['lr']}"
        f"_wd{wd:.0e}_m{all_params['momentum']}_nest{int(bool(all_params['nesterov']))}"
        f"_ls{all_params['label_smoothing']}_sch{all_params['scheduler']}_wu{all_params['warmup_epochs']}"
    )


def load_optimal_config_params(
    cfg: TuningConfig, logger=None
) -> Optional[Dict[str, Any]]:
    """
    Load precomputed best hyperparameters from data/optimal_config.json when available.
    Returns None if file does not exist or cannot be parsed.
    """
    optimal_path = Path.cwd() / OPTIMAL_CONFIG_PATH
    if not optimal_path.exists():
        return None

    try:
        loaded = json.loads(optimal_path.read_text())
    except Exception as e:
        if logger is not None:
            logger.warning(f"Failed to load optimal config from {optimal_path}: {e}")
        return None

    if not isinstance(loaded, dict):
        if logger is not None:
            logger.warning(f"Optimal config at {optimal_path} is not a JSON object; ignoring file")
        return None

    parser = build_parser()
    valid_arg_keys = set(vars(parser.parse_args([])).keys())

    params = {k: v for k, v in loaded.items() if k in valid_arg_keys}
    params = {**FIXED_PARAMS, **params}
    params["dataset"] = cfg.dataset
    params["model"] = cfg.model

    params.pop("run_name", None)
    params.pop("out_dir", None)

    return params


# -----------------------------
# Main
# -----------------------------

def tune_hyperparameters(cfg: TuningConfig = TuningConfig()) -> Optional[Dict[str, Any]]:
    # create tuning dir and setup a single log file for this tuning run
    tuning_dir = ensure_dir(cfg.runs_root / f"{cfg.model}_{cfg.dataset}" / cfg.tuning_dirname)
    logger = get_logger(__name__, console=False)

    results: List[Dict[str, Any]] = []

    optimal_params = load_optimal_config_params(cfg, logger=logger)
    if optimal_params is not None:
        run_name = format_run_name(optimal_params, FIXED_PARAMS, cfg.dataset, cfg.model)
        logger.info(
            f"Found {OPTIMAL_CONFIG_PATH}; running a single training configuration for {cfg.model} on {cfg.dataset}"
        )
        logger.info(f"Results will be saved to {tuning_dir}\n")
        logger.info(f"[1/1] Running: {run_name}")

        result_info = run_single_training_run(cfg, run_name, optimal_params, logger)
        if result_info["status"] == "success" and "final_test_acc1" in result_info:
            logger.info(f"Completed successfully - Test Acc: {result_info['final_test_acc1'] * 100:.2f}%")
        else:
            logger.info("Completed successfully" if result_info["status"] == "success" else "Failed")

        results.append(result_info)
        logger.info("")
    else:
        combos = cartesian_product(PARAM_GRID)
        logger.info(f"Running {len(combos)} training configurations for {cfg.model} on {cfg.dataset}...")
        logger.info(f"Results will be saved to {tuning_dir}\n")

        for idx, grid_params in enumerate(combos, 1):
            all_params = {**FIXED_PARAMS, **grid_params}

            all_params["dataset"] = cfg.dataset
            all_params["model"] = cfg.model

            run_name = format_run_name(all_params, FIXED_PARAMS, cfg.dataset, cfg.model)

            logger.info(f"[{idx}/{len(combos)}] Running: {run_name}")

            result_info = run_single_training_run(cfg, run_name, all_params, logger)

            if result_info["status"] == "success" and "final_test_acc1" in result_info:
                logger.info(f"Completed successfully - Test Acc: {result_info['final_test_acc1'] * 100:.2f}%")
            else:
                logger.info("Completed successfully" if result_info["status"] == "success" else "Failed")

            results.append(result_info)
            logger.info("")

    results_file = tuning_dir / "tuning_results.json"
    results_file.write_text(json.dumps(results, indent=2))

    logger.info(f"\nTuning complete! Results saved to {results_file}")
    print_summary(tuning_dir, results)
    plot_tuning_results(results, tuning_dir)

    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        logger.warning("No successful runs found; returning None for best hyperparameters")
        return None

    if any("best_val_acc" in r for r in successful):
        best_run = max(successful, key=lambda r: r.get("best_val_acc", float("-inf")))
    else:
        best_run = max(successful, key=lambda r: r.get("final_test_acc1", float("-inf")))

    best_params = dict(best_run.get("params", {}))
    logger.info(
        f"Selected best run for downstream tuning: {best_run.get('run_name', 'unknown')} | "
        f"best_val_acc={best_run.get('best_val_acc', 'n/a')} | "
        f"final_test_acc1={best_run.get('final_test_acc1', 'n/a')}"
    )
    return best_params


def run_single_training_run(
    cfg: TuningConfig, run_name: str, params: Dict[str, Any], logger
) -> Dict[str, Any]:
    """Run a single training and return results."""
    try:
        # Build args from params
        args = build_args_from_params(params)
        args.run_name = run_name
        
        # Ensure dataset and model are set in args
        args.dataset = params.get("dataset", cfg.dataset)
        args.model = params.get("model", cfg.model)
        
        # Set out_dir to the model/dataset subdirectory under cfg.runs_root
        args.out_dir = str(cfg.runs_root / args.model / args.dataset)
        
        # Create run directory
        run_dir = make_run_dir(args.out_dir, args.run_name)
        write_json(os.path.join(run_dir, "config.json"), vars(args))
        run_logger = get_logger(__name__, log_file=Path(run_dir) / "train.log", console=False)
        run_logger.info(f"Resolved training args for {run_name}: {json.dumps(vars(args), sort_keys=True)}")
        
        # Train and get metrics
        metrics = train_with_config(args, run_dir=run_dir, logger=run_logger)
        
        return {
            "run_name": run_name,
            "params": params,
            "status": "success",
            "final_test_acc1": metrics["final_test_acc1"],
            "best_val_acc": metrics["best_val_acc"],
            "final_test_loss": metrics["final_test_loss"],
        }

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error(traceback.format_exc())
        return {"run_name": run_name, "params": params, "status": "error", "error": str(e)}




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameter Tuning")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, 
                        help=f"Dataset name (default: {DEFAULT_DATASET})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--runs_root", type=str, default="./runs",
                        help="Root directory for runs")
    args = parser.parse_args()
    
    cfg = TuningConfig(
        runs_root=Path(args.runs_root),
        dataset=args.dataset,
        model=args.model
    )
    tune_hyperparameters(cfg)
