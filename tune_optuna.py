import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import argparse
import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from utils import *
from IOutils import *
from graphics import plot_tuning_results, print_summary
from logger import get_logger
from train import train_with_config

# Will be set by tune_hyperparameters_optuna at startup so all functions use same file
logger = None

# Default configuration - can be overridden when calling tune_hyperparameters_optuna()
DEFAULT_DATASET = "cifar100"
DEFAULT_MODEL = "resnet18"


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class OptunaTuningConfig:
    runs_root: Path = Path("./runs")  # Generic runs directory
    tuning_dirname: str = "tuning_results_optuna"
    dataset: str = DEFAULT_DATASET
    model: str = DEFAULT_MODEL
    n_trials: int = 50
    n_jobs: int = 1


FIXED_PARAMS: Dict[str, Any] = {
    "data_dir": "./data",
    "batch_size": 128,
    "num_workers": 2,
    "seed": 42,
    "log_every": 100,
    "amp": True,
    "epochs": 100,
    "min_lr": 0.0,
    "gamma": 0.1,
    "val_split": 0.1,
}


# Hyperparameter search space
OPTUNA_SPACE = {
    "lr": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-5, "high": 5e-3, "log": True},
    "momentum": {"type": "float", "low": 0.8, "high": 0.99},
    "nesterov": {"type": "categorical", "choices": [False, True]},
    "label_smoothing": {"type": "float", "low": 0.0, "high": 0.2},
    "scheduler": {"type": "categorical", "choices": ["cosine", "multistep"]},
    "warmup_epochs": {"type": "int", "low": 0, "high": 10},
}


# -----------------------------
# Helpers
# -----------------------------

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
    else:
        out["milestones"] = "na"
    return out


def format_run_name(all_params: Dict[str, Any], fixed_params: Dict[str, Any], dataset: str, model: str, trial_id: int) -> str:
    wd = float(all_params["weight_decay"])
    ms = all_params["milestones"] if all_params["scheduler"] == "multistep" else "na"
    return (
        f"optuna_trial{trial_id:03d}_{model}_{dataset}_ep{all_params['epochs']}_bs{fixed_params['batch_size']}_lr{all_params['lr']:.4f}"
        f"_wd{wd:.0e}_m{all_params['momentum']:.2f}_nest{int(bool(all_params['nesterov']))}"
        f"_ls{all_params['label_smoothing']:.1f}_sch{all_params['scheduler']}_wu{all_params['warmup_epochs']}"
        f"_ms{ms}"
    )


def build_args_from_params(params: Dict[str, Any]) -> argparse.Namespace:
    """
    Convert a params dict to an argparse.Namespace object that train_with_config expects.
    """
    parser = build_parser()
    args = parser.parse_args([])  # Parse empty args to get defaults
    
    # Override with provided params
    for key, value in params.items():
        setattr(args, key, value)
    
    return args


# Pruning callback for intermediate reporting
class TrialPruningCallback:
    def __init__(self, trial: optuna.Trial):
        self.trial = trial
    
    def __call__(self, epoch: int, val_acc: float) -> None:
        """Called during training to report intermediate value."""
        self.trial.report(val_acc, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# -----------------------------
# Objective and Trial
# -----------------------------

def objective(trial: optuna.Trial, cfg: OptunaTuningConfig) -> float:
    """
    Objective function for Optuna optimization.
    Suggests hyperparameters and trains a model, returning best validation accuracy.
    """
    # Suggest hyperparameters from search space
    suggested_params = {}
    for param_name, param_spec in OPTUNA_SPACE.items():
        if param_spec["type"] == "float":
            suggested_params[param_name] = trial.suggest_float(
                param_name,
                low=param_spec["low"],
                high=param_spec["high"],
                log=param_spec.get("log", False)
            )
        elif param_spec["type"] == "int":
            suggested_params[param_name] = trial.suggest_int(
                param_name,
                low=param_spec["low"],
                high=param_spec["high"]
            )
        elif param_spec["type"] == "categorical":
            suggested_params[param_name] = trial.suggest_categorical(
                param_name,
                choices=param_spec["choices"]
            )
    
    # Combine with fixed params
    all_params = {**FIXED_PARAMS, **suggested_params}
    all_params = with_scheduler_dependent_params(all_params)
    
    # Add dataset and model
    all_params["dataset"] = cfg.dataset
    all_params["model"] = cfg.model
    
    run_name = format_run_name(all_params, FIXED_PARAMS, cfg.dataset, cfg.model, trial.number)
    
    logger.info(f"[Trial {trial.number}] Running: {run_name}")
    
    try:
        # Run training
        result_info = run_single_training_run(cfg, run_name, all_params, trial)
        
        if result_info["status"] == "success" and "best_val_acc" in result_info:
            best_val_acc = result_info["best_val_acc"]
            test_acc = result_info.get("final_test_acc1", 0.0)
            logger.info(f"[Trial {trial.number}] Best Val Acc: {best_val_acc:.6f}, Test Acc: {test_acc:.4f}")
            
            # Store trial metadata for later retrieval
            trial.set_user_attr("run_name", run_name)
            trial.set_user_attr("final_test_acc1", test_acc)
            trial.set_user_attr("final_test_loss", result_info.get("final_test_loss", 0.0))
            
            return best_val_acc
        else:
            logger.warning(f"[Trial {trial.number}] Training failed with status: {result_info['status']}")
            return 0.0
    
    except optuna.TrialPruned:
        logger.info(f"[Trial {trial.number}] Pruned by callback")
        raise
    except Exception as e:
        logger.error(f"[Trial {trial.number}] Failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0.0


def run_single_training_run(
    cfg: OptunaTuningConfig, run_name: str, params: Dict[str, Any], trial: optuna.Trial = None
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
        
        # Train and get metrics
        metrics = train_with_config(args, run_dir=run_dir, logger=logger)
        
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
        import traceback
        logger.error(traceback.format_exc())
        return {"run_name": run_name, "params": params, "status": "error", "error": str(e)}


# -----------------------------
# Main
# -----------------------------

def tune_hyperparameters_optuna(cfg: OptunaTuningConfig = OptunaTuningConfig()) -> None:
    """Run hyperparameter tuning using Optuna."""
    # Create tuning dir and setup logging
    tuning_dir = ensure_dir(cfg.runs_root / f"{cfg.model}_{cfg.dataset}" / cfg.tuning_dirname)
    global logger
    log_root = Path.cwd() / "log"
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_root / f"tune_optuna_{cfg.model}_{cfg.dataset}_{timestamp}.log"
    logger = get_logger(__name__, log_file=log_file)

    logger.info(f"Starting Optuna hyperparameter tuning for {cfg.model} on {cfg.dataset}")
    logger.info(f"Number of trials: {cfg.n_trials}")
    logger.info(f"Number of jobs: {cfg.n_jobs}")
    logger.info(f"Results will be saved to {tuning_dir}\n")

    # Create study with TPE sampler and median pruner
    sampler = TPESampler(seed=42, n_startup_trials=5)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_trials=0)
    
    study = optuna.create_study(
        study_name=f"tune_{cfg.model}_{cfg.dataset}",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=False
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, cfg),
        n_trials=cfg.n_trials,
        n_jobs=cfg.n_jobs,
        show_progress_bar=True
    )

    # Collect results from completed trials
    results: List[Dict[str, Any]] = []
    for trial in study.trials:
        if trial.state == optuna.TrialState.COMPLETE:
            result_info = {
                "run_name": trial.user_attrs.get("run_name", f"trial_{trial.number}"),
                "trial_id": trial.number,
                "params": trial.params,
                "status": "success",
                "final_test_acc1": trial.user_attrs.get("final_test_acc1", 0.0),
                "best_val_acc": trial.value,  # best_val_acc is the objective value
                "final_test_loss": trial.user_attrs.get("final_test_loss", 0.0),
            }
            results.append(result_info)
        elif trial.state == optuna.TrialState.PRUNED:
            results.append({
                "run_name": f"trial_{trial.number}_pruned",
                "trial_id": trial.number,
                "params": trial.params,
                "status": "pruned",
            })
        elif trial.state == optuna.TrialState.FAIL:
            results.append({
                "run_name": f"trial_{trial.number}_failed",
                "trial_id": trial.number,
                "params": trial.params,
                "status": "failed",
            })

    # Save results
    results_file = tuning_dir / "tuning_results_optuna.json"
    results_file.write_text(json.dumps(results, indent=2))

    logger.info(f"\nOptuna tuning complete! Results saved to {results_file}")
    logger.info(f"Best trial value: {study.best_value:.6f}")
    logger.info(f"Best trial params: {study.best_params}")
    
    # Print summary and plot results
    print_summary(tuning_dir, results)
    plot_tuning_results(results, tuning_dir)
    
    # Save study visualization
    try:
        import optuna.visualization as vis
        fig = vis.plot_param_importances(study).to_html()
        importance_file = tuning_dir / "param_importances.html"
        with open(importance_file, "w") as f:
            f.write(fig)
        logger.info(f"Parameter importance plot saved to {importance_file}")
    except Exception as e:
        logger.warning(f"Could not generate parameter importance plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameter Tuning with Optuna")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, 
                        help=f"Dataset name (default: {DEFAULT_DATASET})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--runs_root", type=str, default="./runs",
                        help="Root directory for runs")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of trials to run (default: 50)")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel jobs (default: 1)")
    args = parser.parse_args()
    
    cfg = OptunaTuningConfig(
        runs_root=Path(args.runs_root),
        dataset=args.dataset,
        model=args.model,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs
    )
    tune_hyperparameters_optuna(cfg)
