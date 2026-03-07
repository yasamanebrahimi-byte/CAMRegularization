import argparse
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from IOutils import (
    ensure_dir,
    prepare_run_from_params,
    normalize_masking_type,
    format_mask_run_name,
    add_dataset_model_args,
    add_tuning_runtime_args,
)
from utils import DEFAULT_DATASET, DEFAULT_MODEL, apply_training_context
from graphics import plot_tuning_results, print_summary
from logger import get_logger
from train import train_with_config
from tune import OPTIMAL_CONFIG_PATH, TuningConfig, load_optimal_config_params, tune_hyperparameters


@dataclass(frozen=True)
class GreedyMaskTuningConfig:
    runs_root: Path = Path("./runs")
    dataset: str = DEFAULT_DATASET
    model: str = DEFAULT_MODEL
    base_tuning_dirname: str = "tuning_results"
    mask_tuning_dirname: str = "greedy_mask_tuning_results"
    data_dir: str = "./data"
    val_split: float = 0.1
    batch_size: int = 128
    num_workers: int = 2
    epochs: int = 150


MASK_PARAM_GRID: Dict[str, List[Any]] = {
    "masking": ["random", "cam_high", "cam_low"],
    "mask_warmup_epochs": [15, 30],
    "mask_prob": [0.5, 0.75, 1.0],
    "mask_area": [0.2, 0.3, 0.4],
    "mask_block": [4, 6, 8],
    "cam_layer": ["auto"],
}

GREEDY_FACTOR_ORDER: List[str] = [
    "mask_warmup_epochs",
    "mask_prob",
    "mask_area",
    "mask_block",
    "cam_layer",
]

RANDOM_MASK_CAM_LAYER = "auto"
ALL_MASKING_TYPES = "all"


def _normalize_masking_type(masking_type: Optional[str]) -> Optional[str]:
    return normalize_masking_type(masking_type, MASK_PARAM_GRID["masking"], all_value=ALL_MASKING_TYPES)


def _sanitize_for_name(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _build_stage_run_name(
    base_params: Dict[str, Any],
    mask_params: Dict[str, Any],
    dataset: str,
    model: str,
    stage_idx: int,
    factor: str,
    factor_value: Any,
) -> str:
    base_run_name = format_mask_run_name(base_params, mask_params, dataset, model, prefix="greedy_tune")
    return f"{base_run_name}_gs{stage_idx}_{factor}_{_sanitize_for_name(factor_value)}"


def _initial_mask_params(masking: str) -> Dict[str, Any]:
    return {
        "masking": masking,
        "mask_warmup_epochs": MASK_PARAM_GRID["mask_warmup_epochs"][0],
        "mask_prob": MASK_PARAM_GRID["mask_prob"][0],
        "mask_area": MASK_PARAM_GRID["mask_area"][0],
        "mask_block": MASK_PARAM_GRID["mask_block"][0],
        "cam_layer": RANDOM_MASK_CAM_LAYER if masking == "random" else MASK_PARAM_GRID["cam_layer"][0],
    }


def _candidate_values_for_factor(masking: str, factor: str) -> List[Any]:
    if factor == "cam_layer" and masking == "random":
        return [RANDOM_MASK_CAM_LAYER]
    return MASK_PARAM_GRID[factor]


def _factors_for_masking(masking: str) -> List[str]:
    if masking == "random":
        return [
            "mask_warmup_epochs",
            "mask_prob",
            "mask_area",
            "mask_block",
        ]
    return GREEDY_FACTOR_ORDER


def run_single_mask_training_run(
    cfg: GreedyMaskTuningConfig, run_name: str, params: Dict[str, Any], logger
) -> Dict[str, Any]:
    try:
        args, run_dir = prepare_run_from_params(
            params,
            run_name=run_name,
            runs_root=cfg.runs_root,
            dataset=cfg.dataset,
            model=cfg.model,
        )

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


def _select_stage_best(stage_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    successful = [r for r in stage_results if r.get("status") == "success"]
    if not successful:
        return None

    if any("best_val_acc" in r for r in successful):
        return max(successful, key=lambda r: r.get("best_val_acc", float("-inf")))
    return max(successful, key=lambda r: r.get("final_test_acc1", float("-inf")))


def _run_greedy_for_masking(
    cfg: GreedyMaskTuningConfig,
    best_base_params: Dict[str, Any],
    masking: str,
    all_results: List[Dict[str, Any]],
    logger,
) -> Optional[Dict[str, Any]]:
    logger.info(f"\nStarting greedy factor tuning for masking='{masking}'")
    current_mask_params = _initial_mask_params(masking)
    factors = _factors_for_masking(masking)

    for stage_idx, factor in enumerate(factors, start=1):
        candidate_values = _candidate_values_for_factor(masking, factor)
        logger.info(
            f"[masking={masking}] Stage {stage_idx}/{len(factors)}: tuning {factor} over {candidate_values}"
        )

        stage_results: List[Dict[str, Any]] = []
        for candidate_value in candidate_values:
            trial_mask_params = dict(current_mask_params)
            trial_mask_params[factor] = candidate_value

            all_params = {**best_base_params, **trial_mask_params}
            all_params["dataset"] = cfg.dataset
            all_params["model"] = cfg.model

            run_name = _build_stage_run_name(
                best_base_params,
                trial_mask_params,
                cfg.dataset,
                cfg.model,
                stage_idx,
                factor,
                candidate_value,
            )

            logger.info(f"Running candidate: {run_name}")
            result_info = run_single_mask_training_run(cfg, run_name, all_params, logger)
            result_info["greedy_stage"] = stage_idx
            result_info["greedy_factor"] = factor
            result_info["greedy_candidate"] = candidate_value
            result_info["greedy_masking"] = masking
            stage_results.append(result_info)
            all_results.append(result_info)

            if result_info["status"] == "success" and "final_test_acc1" in result_info:
                logger.info(f"Completed successfully - Test Acc: {result_info['final_test_acc1'] * 100:.2f}%")
            else:
                logger.info("Completed successfully" if result_info["status"] == "success" else "Failed")

        best_stage_result = _select_stage_best(stage_results)
        if best_stage_result is None:
            logger.warning(f"No successful runs for factor '{factor}' under masking='{masking}'")
            return None

        current_mask_params[factor] = best_stage_result["params"][factor]
        logger.info(
            f"Selected {factor}={current_mask_params[factor]} for masking='{masking}' | "
            f"best_val_acc={best_stage_result.get('best_val_acc', 'n/a')} | "
            f"final_test_acc1={best_stage_result.get('final_test_acc1', 'n/a')}"
        )

    logger.info(f"Completed greedy factor tuning for masking='{masking}'")
    return current_mask_params


def tune_mask_hyperparameters_greedy(
    cfg: GreedyMaskTuningConfig = GreedyMaskTuningConfig(), masking_type: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    selected_masking_type = _normalize_masking_type(masking_type)

    mask_tuning_dir = ensure_dir(cfg.runs_root / f"{cfg.model}_{cfg.dataset}" / cfg.mask_tuning_dirname)

    log_root = Path.cwd() / "log"
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_root / f"greedy_tune_{cfg.model}_{cfg.dataset}_{timestamp}.log"
    logger = get_logger(__name__, log_file=log_file, console=False)

    base_tuning_cfg = TuningConfig(
        runs_root=cfg.runs_root,
        tuning_dirname=cfg.base_tuning_dirname,
        dataset=cfg.dataset,
        model=cfg.model,
        data_dir=cfg.data_dir,
        val_split=cfg.val_split,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        epochs=cfg.epochs,
    )

    best_base_params = load_optimal_config_params(base_tuning_cfg, logger=logger)
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

    best_base_params = apply_training_context(
        best_base_params,
        dataset=cfg.dataset,
        model=cfg.model,
        data_dir=cfg.data_dir,
        val_split=cfg.val_split,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        epochs=cfg.epochs,
    )

    masking_values = [selected_masking_type] if selected_masking_type else MASK_PARAM_GRID["masking"]
    logger.info(f"Greedy mask tuning results will be saved to {mask_tuning_dir}")
    logger.info(f"Running greedy search for masking modes: {masking_values}")

    results: List[Dict[str, Any]] = []
    best_params_per_masking: List[Dict[str, Any]] = []

    for masking in masking_values:
        best_mask_params = _run_greedy_for_masking(cfg, best_base_params, masking, results, logger)
        if best_mask_params is None:
            continue

        merged_params = {**best_base_params, **best_mask_params}
        merged_params["dataset"] = cfg.dataset
        merged_params["model"] = cfg.model
        best_params_per_masking.append(merged_params)

    results_filename = (
        f"greedy_mask_tuning_results_{selected_masking_type}.json"
        if selected_masking_type
        else "greedy_mask_tuning_results.json"
    )
    results_file = mask_tuning_dir / results_filename
    results_file.write_text(json.dumps(results, indent=2))

    logger.info(f"\nGreedy mask tuning complete! Results saved to {results_file}")
    print_summary(mask_tuning_dir, results)
    plot_tuning_results(results, mask_tuning_dir)

    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        logger.warning("No successful greedy mask tuning runs found")
        return None

    if any("best_val_acc" in r for r in successful):
        best_mask_run = max(successful, key=lambda r: r.get("best_val_acc", float("-inf")))
    else:
        best_mask_run = max(successful, key=lambda r: r.get("final_test_acc1", float("-inf")))

    logger.info(
        f"Best greedy run: {best_mask_run.get('run_name', 'unknown')} | "
        f"best_val_acc={best_mask_run.get('best_val_acc', 'n/a')} | "
        f"final_test_acc1={best_mask_run.get('final_test_acc1', 'n/a')}"
    )
    return dict(best_mask_run.get("params", {}))


def tune_single_mask_hyperparameters_greedy(
    masking_type: str, cfg: GreedyMaskTuningConfig = GreedyMaskTuningConfig()
) -> Optional[Dict[str, Any]]:
    return tune_mask_hyperparameters_greedy(cfg, masking_type=masking_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Greedy Mask Hyperparameter Tuning")
    add_dataset_model_args(parser, default_dataset=DEFAULT_DATASET, default_model=DEFAULT_MODEL)
    add_tuning_runtime_args(parser)
    parser.add_argument(
        "--masking_type",
        type=str,
        default=ALL_MASKING_TYPES,
        choices=[ALL_MASKING_TYPES, *MASK_PARAM_GRID["masking"]],
        help="Masking mode to optimize (default: all)",
    )
    args = parser.parse_args()

    cfg = GreedyMaskTuningConfig(
        runs_root=Path(args.runs_root),
        dataset=args.dataset,
        model=args.model,
        data_dir=args.data_dir,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
    )
    tune_mask_hyperparameters_greedy(cfg, masking_type=args.masking_type)
