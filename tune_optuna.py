import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
import optuna.visualization as vis
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from IOutils import ensure_dir, make_run_dir, write_json, build_args_from_params
from graphics import plot_tuning_results, print_summary
from logger import get_logger
from train import train_with_config
from tune import OPTIMAL_CONFIG_PATH, TuningConfig, load_optimal_config_params, tune_hyperparameters
from utils import DEFAULT_DATASET, DEFAULT_MODEL


logger = None


@dataclass(frozen=True)
class OptunaMaskTuningConfig:
    runs_root: Path = Path("./runs")
    dataset: str = DEFAULT_DATASET
    model: str = DEFAULT_MODEL
    base_runs_subdir: str = "_base_pre_tuning"
    base_tuning_dirname: str = "tuning_results"
    mask_tuning_dirname: str = "mask_tuning_results_optuna"
    n_jobs: int = 1
    min_resource_epochs: int = 15
    max_resource_epochs: int = 100
    reduction_factor: int = 2


MASK_PARAM_SPACE: Dict[str, List[Any]] = {
    "masking": ["random", "cam_high", "cam_low"],
    "mask_warmup_epochs": [0, 15, 30],
    "mask_prob": [0.5, 0.75, 1.0],
    "mask_area": [0.2, 0.3, 0.4],
    "mask_block": [4, 6, 8],
    "cam_layer": ["layer2", "layer3", "layer4"],
}

RANDOM_MASK_CAM_LAYER = "layer2"
ALL_MASKING_TYPES = "all"


def _normalize_masking_type(masking_type: Optional[str]) -> Optional[str]:
    if masking_type in {None, ALL_MASKING_TYPES}:
        return None

    if masking_type not in MASK_PARAM_SPACE["masking"]:
        valid = ", ".join([ALL_MASKING_TYPES, *MASK_PARAM_SPACE["masking"]])
        raise ValueError(f"Invalid masking_type '{masking_type}'. Expected one of: {valid}")

    return masking_type


def _build_stage_budgets(min_epochs: int, max_epochs: int, reduction_factor: int) -> List[int]:
    budgets: List[int] = []
    resource = max(1, int(min_epochs))
    max_resource = max(1, int(max_epochs))
    factor = max(2, int(reduction_factor))

    while resource < max_resource:
        budgets.append(resource)
        next_resource = int(resource * factor)
        resource = next_resource if next_resource > resource else resource + 1

    if not budgets or budgets[-1] != max_resource:
        budgets.append(max_resource)

    return budgets


def _sample_mask_params(trial: optuna.Trial, selected_masking_type: Optional[str]) -> Dict[str, Any]:
    masking = (
        selected_masking_type
        if selected_masking_type is not None
        else trial.suggest_categorical("masking", MASK_PARAM_SPACE["masking"])
    )

    sampled: Dict[str, Any] = {
        "masking": masking,
        "mask_warmup_epochs": trial.suggest_categorical("mask_warmup_epochs", MASK_PARAM_SPACE["mask_warmup_epochs"]),
        "mask_prob": trial.suggest_categorical("mask_prob", MASK_PARAM_SPACE["mask_prob"]),
        "mask_area": trial.suggest_categorical("mask_area", MASK_PARAM_SPACE["mask_area"]),
        "mask_block": trial.suggest_categorical("mask_block", MASK_PARAM_SPACE["mask_block"]),
    }

    if masking in {"cam_high", "cam_low"}:
        sampled["cam_layer"] = trial.suggest_categorical("cam_layer", MASK_PARAM_SPACE["cam_layer"])
    else:
        sampled["cam_layer"] = RANDOM_MASK_CAM_LAYER

    return sampled


def _resolve_n_trials(selected_masking_type: Optional[str]) -> int:
    if selected_masking_type is None:
        return 100
    if selected_masking_type in {"cam_high", "cam_low"}:
        return 64
    if selected_masking_type == "random":
        return 32
    raise ValueError(f"Unsupported masking type for trial schedule: {selected_masking_type}")


def _format_run_name(
    base_params: Dict[str, Any],
    mask_params: Dict[str, Any],
    dataset: str,
    model: str,
    trial_number: int,
    stage_epochs: int,
) -> str:
    return (
        f"optuna_mask_trial{trial_number:04d}_{model}_{dataset}_ep{stage_epochs}_"
        f"bs{base_params['batch_size']}_lr{base_params['lr']}_wd{float(base_params['weight_decay']):.0e}_"
        f"ls{base_params['label_smoothing']}_wu{base_params['warmup_epochs']}_"
        f"mask{mask_params['masking']}_mwu{mask_params['mask_warmup_epochs']}_"
        f"mp{mask_params['mask_prob']}_ma{mask_params['mask_area']}_"
        f"mb{mask_params['mask_block']}_cl{mask_params['cam_layer']}"
    )


def _run_single_stage(
    cfg: OptunaMaskTuningConfig,
    run_name: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        args = build_args_from_params(params)
        args.run_name = run_name
        args.dataset = params.get("dataset", cfg.dataset)
        args.model = params.get("model", cfg.model)
        args.masking = params.get("masking", args.masking)
        args.mask_warmup_epochs = params.get("mask_warmup_epochs", args.mask_warmup_epochs)
        args.mask_prob = params.get("mask_prob", args.mask_prob)
        args.mask_area = params.get("mask_area", args.mask_area)
        args.mask_block = params.get("mask_block", args.mask_block)
        args.cam_layer = params.get("cam_layer", args.cam_layer)
        args.out_dir = str(cfg.runs_root / args.model / args.dataset)

        run_dir = make_run_dir(args.out_dir, args.run_name)
        write_json(os.path.join(run_dir, "config.json"), vars(args))
        logger.info(f"Resolved training args for {run_name}: {json.dumps(vars(args), sort_keys=True)}")

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
        logger.error(traceback.format_exc())
        return {"run_name": run_name, "params": params, "status": "error", "error": str(e)}


def _objective(
    trial: optuna.Trial,
    cfg: OptunaMaskTuningConfig,
    base_params: Dict[str, Any],
    selected_masking_type: Optional[str],
    stage_budgets: List[int],
) -> float:
    mask_params = _sample_mask_params(trial, selected_masking_type)
    merged_params = {**base_params, **mask_params, "dataset": cfg.dataset, "model": cfg.model}

    best_observed_val = float("-inf")
    final_stage_result: Optional[Dict[str, Any]] = None

    logger.info(f"[Trial {trial.number}] mask params: {mask_params}")

    for stage_idx, stage_epochs in enumerate(stage_budgets):
        stage_params = dict(merged_params)
        stage_params["epochs"] = stage_epochs
        stage_run_name = _format_run_name(
            base_params,
            mask_params,
            cfg.dataset,
            cfg.model,
            trial.number,
            stage_epochs,
        )

        logger.info(
            f"[Trial {trial.number}] Stage {stage_idx + 1}/{len(stage_budgets)} | "
            f"epochs={stage_epochs} | run={stage_run_name}"
        )

        stage_result = _run_single_stage(cfg, stage_run_name, stage_params)
        if stage_result.get("status") != "success":
            trial.set_user_attr("status", "failed")
            trial.set_user_attr("failed_stage", stage_idx)
            trial.set_user_attr("mask_params", mask_params)
            trial.set_user_attr("run_name", stage_run_name)
            return 0.0

        stage_val = float(stage_result.get("best_val_acc", 0.0))
        best_observed_val = max(best_observed_val, stage_val)
        final_stage_result = stage_result

        trial.report(stage_val, step=stage_epochs)
        trial.set_user_attr(f"stage_{stage_epochs}_best_val_acc", stage_val)

        logger.info(
            f"[Trial {trial.number}] Stage result | best_val_acc={stage_val:.6f} | "
            f"best_so_far={best_observed_val:.6f}"
        )

        if trial.should_prune() and stage_idx < len(stage_budgets) - 1:
            trial.set_user_attr("status", "pruned")
            trial.set_user_attr("pruned_at_epochs", stage_epochs)
            trial.set_user_attr("mask_params", mask_params)
            trial.set_user_attr("run_name", stage_run_name)
            trial.set_user_attr("best_val_acc", best_observed_val)
            raise optuna.TrialPruned()

    if final_stage_result is None:
        trial.set_user_attr("status", "failed")
        trial.set_user_attr("mask_params", mask_params)
        return 0.0

    trial.set_user_attr("status", "success")
    trial.set_user_attr("mask_params", mask_params)
    trial.set_user_attr("run_name", final_stage_result.get("run_name", f"trial_{trial.number}"))
    trial.set_user_attr("final_test_acc1", final_stage_result.get("final_test_acc1", 0.0))
    trial.set_user_attr("final_test_loss", final_stage_result.get("final_test_loss", 0.0))
    trial.set_user_attr("best_val_acc", best_observed_val)
    return best_observed_val


def tune_hyperparameters_optuna(
    cfg: OptunaMaskTuningConfig = OptunaMaskTuningConfig(), masking_type: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    global logger
    selected_masking_type = _normalize_masking_type(masking_type)

    tuning_dir = ensure_dir(cfg.runs_root / f"{cfg.model}_{cfg.dataset}" / cfg.mask_tuning_dirname)
    log_root = Path.cwd() / "log"
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode_label = selected_masking_type if selected_masking_type else ALL_MASKING_TYPES
    log_file = log_root / f"tune_optuna_mask_{cfg.model}_{cfg.dataset}_{mode_label}_{timestamp}.log"
    logger = get_logger(__name__, log_file=log_file, console=False)

    base_tuning_cfg = TuningConfig(
        runs_root=cfg.runs_root / cfg.base_runs_subdir,
        tuning_dirname=cfg.base_tuning_dirname,
        dataset=cfg.dataset,
        model=cfg.model,
    )

    best_base_params = load_optimal_config_params(base_tuning_cfg)
    if best_base_params is not None:
        logger.info(
            f"Found {OPTIMAL_CONFIG_PATH}; reusing cached base hyperparameters and skipping base tuning runs"
        )
    else:
        logger.info(f"Starting base hyperparameter tuning for {cfg.model} on {cfg.dataset}")
        best_base_params = tune_hyperparameters(base_tuning_cfg)

    if not best_base_params:
        logger.error("Base hyperparameter tuning did not produce successful runs")
        return None

    best_base_params.pop("masking", None)
    best_base_params.pop("mask_warmup_epochs", None)
    best_base_params.pop("mask_prob", None)
    best_base_params.pop("mask_area", None)
    best_base_params.pop("mask_block", None)
    best_base_params.pop("cam_layer", None)
    best_base_params["dataset"] = cfg.dataset
    best_base_params["model"] = cfg.model

    stage_budgets = _build_stage_budgets(
        min_epochs=cfg.min_resource_epochs,
        max_epochs=cfg.max_resource_epochs,
        reduction_factor=cfg.reduction_factor,
    )
    n_trials = _resolve_n_trials(selected_masking_type)

    logger.info(f"Starting Optuna mask tuning for {cfg.model} on {cfg.dataset}")
    logger.info(f"Masking type: {mode_label}")
    logger.info(f"Base pre-tuning runs root: {base_tuning_cfg.runs_root}")
    logger.info(f"Objective: best_val_acc")
    logger.info(f"Trials: {n_trials} | Jobs: {cfg.n_jobs}")
    logger.info(f"Multi-fidelity stage budgets (epochs): {stage_budgets}")
    logger.info("Pruning: HyperbandPruner (promotes promising trials, prunes weak trials early)")
    logger.info(f"Results will be saved to {tuning_dir}\n")

    sampler = TPESampler(seed=42, n_startup_trials=min(10, max(1, n_trials // 5)))
    pruner = HyperbandPruner(
        min_resource=stage_budgets[0],
        max_resource=stage_budgets[-1],
        reduction_factor=max(2, cfg.reduction_factor),
    )

    study = optuna.create_study(
        study_name=f"mask_optuna_{cfg.model}_{cfg.dataset}_{mode_label}_{timestamp}",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=False,
    )

    study.optimize(
        lambda trial: _objective(trial, cfg, best_base_params, selected_masking_type, stage_budgets),
        n_trials=n_trials,
        n_jobs=cfg.n_jobs,
        show_progress_bar=True,
    )

    results: List[Dict[str, Any]] = []
    for trial in study.trials:
        status_attr = trial.user_attrs.get("status", "unknown")
        mask_params_attr = trial.user_attrs.get("mask_params", {})
        trial_params = dict(mask_params_attr) if mask_params_attr else dict(trial.params)

        if trial.state == optuna.TrialState.COMPLETE:
            results.append(
                {
                    "run_name": trial.user_attrs.get("run_name", f"trial_{trial.number}"),
                    "trial_id": trial.number,
                    "params": {**best_base_params, **trial_params, "dataset": cfg.dataset, "model": cfg.model},
                    "status": "success",
                    "final_test_acc1": trial.user_attrs.get("final_test_acc1", 0.0),
                    "best_val_acc": float(trial.value) if trial.value is not None else 0.0,
                    "final_test_loss": trial.user_attrs.get("final_test_loss", 0.0),
                }
            )
        elif trial.state == optuna.TrialState.PRUNED:
            results.append(
                {
                    "run_name": trial.user_attrs.get("run_name", f"trial_{trial.number}_pruned"),
                    "trial_id": trial.number,
                    "params": {**best_base_params, **trial_params, "dataset": cfg.dataset, "model": cfg.model},
                    "status": "pruned",
                    "best_val_acc": trial.user_attrs.get("best_val_acc", 0.0),
                    "pruned_at_epochs": trial.user_attrs.get("pruned_at_epochs"),
                }
            )
        else:
            mapped_status = "failed" if status_attr in {"failed", "error", "unknown"} else status_attr
            results.append(
                {
                    "run_name": trial.user_attrs.get("run_name", f"trial_{trial.number}_failed"),
                    "trial_id": trial.number,
                    "params": {**best_base_params, **trial_params, "dataset": cfg.dataset, "model": cfg.model},
                    "status": mapped_status,
                    "best_val_acc": trial.user_attrs.get("best_val_acc", 0.0),
                }
            )

    results_filename = (
        f"mask_tuning_results_optuna_{selected_masking_type}.json"
        if selected_masking_type
        else "mask_tuning_results_optuna.json"
    )
    results_file = tuning_dir / results_filename
    results_file.write_text(json.dumps(results, indent=2))

    logger.info(f"\nOptuna mask tuning complete! Results saved to {results_file}")

    successful = [r for r in results if r.get("status") == "success"]
    if successful:
        best_run = max(successful, key=lambda r: r.get("best_val_acc", float("-inf")))
        logger.info(
            f"Best run: {best_run.get('run_name', 'unknown')} | "
            f"best_val_acc={best_run.get('best_val_acc', 'n/a')} | "
            f"final_test_acc1={best_run.get('final_test_acc1', 'n/a')}"
        )
    else:
        logger.warning("No successful trials found")

    print_summary(tuning_dir, results)
    plot_tuning_results(results, tuning_dir)

    try:
        html = vis.plot_param_importances(study).to_html()
        importance_file = tuning_dir / (
            f"param_importances_{selected_masking_type}.html" if selected_masking_type else "param_importances.html"
        )
        importance_file.write_text(html, encoding="utf-8")
        logger.info(f"Parameter importance plot saved to {importance_file}")
    except Exception as e:
        logger.warning(f"Could not generate parameter importance plot: {e}")

    if not successful:
        return None
    return dict(max(successful, key=lambda r: r.get("best_val_acc", float("-inf"))).get("params", {}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Mask Hyperparameter Tuning with Optuna")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help=f"Dataset name (default: {DEFAULT_DATASET})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--runs_root", type=str, default="./runs", help="Root directory for runs")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel Optuna jobs")
    parser.add_argument(
        "--masking_type",
        type=str,
        default=ALL_MASKING_TYPES,
        choices=[ALL_MASKING_TYPES, *MASK_PARAM_SPACE["masking"]],
        help="Masking mode to optimize (default: all)",
    )
    parser.add_argument("--min_resource_epochs", type=int, default=15, help="Lowest epoch budget per trial stage")
    parser.add_argument("--max_resource_epochs", type=int, default=100, help="Highest epoch budget per trial stage")
    parser.add_argument("--reduction_factor", type=int, default=2, help="Successive-halving reduction factor")
    args = parser.parse_args()

    cfg = OptunaMaskTuningConfig(
        runs_root=Path(args.runs_root),
        dataset=args.dataset,
        model=args.model,
        n_jobs=args.n_jobs,
        min_resource_epochs=args.min_resource_epochs,
        max_resource_epochs=args.max_resource_epochs,
        reduction_factor=args.reduction_factor,
    )
    tune_hyperparameters_optuna(cfg, masking_type=args.masking_type)
