import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from IOutils import build_parser, ensure_dir, make_run_dir, write_json
from graphics import plot_tuning_results, print_summary
from logger import get_logger
from train import train_with_config
from tune import TuningConfig, tune_hyperparameters


DEFAULT_DATASET = "cifar100"
DEFAULT_MODEL = "resnet18"


@dataclass(frozen=True)
class MaskTuningConfig:
    runs_root: Path = Path("./runs")
    dataset: str = DEFAULT_DATASET
    model: str = DEFAULT_MODEL
    base_tuning_dirname: str = "tuning_results"
    mask_tuning_dirname: str = "mask_tuning_results"


MASK_PARAM_GRID: Dict[str, List[Any]] = {
    "masking": ["random", "cam_high", "cam_low"],
    "mask_warmup_epochs": [15, 30],
    "mask_prob": [0.5, 0.75, 1.0],
    "mask_area": [0.2, 0.3, 0.4],
    "mask_block": [4, 6, 8],
    "cam_layer": ["layer2", "layer3", "layer4"],
}


logger = None


def build_args_from_params(params: Dict[str, Any]) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args([])
    for key, value in params.items():
        setattr(args, key, value)
    return args


def generate_mask_combinations() -> List[Dict[str, Any]]:
    combinations: List[Dict[str, Any]] = []

    for masking in MASK_PARAM_GRID["masking"]:
        for warmup_epochs in MASK_PARAM_GRID["mask_warmup_epochs"]:
            for mask_prob in MASK_PARAM_GRID["mask_prob"]:
                for mask_area in MASK_PARAM_GRID["mask_area"]:
                    for mask_block in MASK_PARAM_GRID["mask_block"]:
                        for cam_layer in MASK_PARAM_GRID["cam_layer"]:
                            combinations.append(
                                {
                                    "masking": masking,
                                    "mask_warmup_epochs": warmup_epochs,
                                    "mask_prob": mask_prob,
                                    "mask_area": mask_area,
                                    "mask_block": mask_block,
                                    "cam_layer": cam_layer,
                                }
                            )
    return combinations


def format_mask_run_name(base_params: Dict[str, Any], mask_params: Dict[str, Any], dataset: str, model: str) -> str:
    base_name = (
        f"mask_tune_{model}_{dataset}_ep{base_params['epochs']}_bs{base_params['batch_size']}_"
        f"lr{base_params['lr']}_wd{float(base_params['weight_decay']):.0e}_"
        f"ls{base_params['label_smoothing']}_wu{base_params['warmup_epochs']}"
    )

    return (
        f"{base_name}_mask{mask_params['masking']}_mwu{mask_params['mask_warmup_epochs']}_"
        f"mp{mask_params['mask_prob']}_ma{mask_params['mask_area']}_"
        f"mb{mask_params['mask_block']}_cl{mask_params['cam_layer']}"
    )


def run_single_mask_training_run(
    cfg: MaskTuningConfig, run_name: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        args = build_args_from_params(params)
        args.run_name = run_name
        args.dataset = params.get("dataset", cfg.dataset)
        args.model = params.get("model", cfg.model)

        args.out_dir = str(cfg.runs_root / args.model / args.dataset)

        run_dir = make_run_dir(args.out_dir, args.run_name)
        write_json(os.path.join(run_dir, "config.json"), vars(args))

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


def tune_mask_hyperparameters(cfg: MaskTuningConfig = MaskTuningConfig()) -> Optional[Dict[str, Any]]:
    global logger

    mask_tuning_dir = ensure_dir(cfg.runs_root / f"{cfg.model}_{cfg.dataset}" / cfg.mask_tuning_dirname)

    log_root = Path.cwd() / "log"
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_root / f"mask_tune_{cfg.model}_{cfg.dataset}_{timestamp}.log"
    logger = get_logger(__name__, log_file=log_file)

    logger.info(f"Starting base hyperparameter tuning for {cfg.model} on {cfg.dataset}")
    best_base_params = tune_hyperparameters(
        TuningConfig(
            runs_root=cfg.runs_root,
            tuning_dirname=cfg.base_tuning_dirname,
            dataset=cfg.dataset,
            model=cfg.model,
        )
    )

    if not best_base_params:
        logger.error("Base hyperparameter tuning did not produce successful runs")
        return None

    best_base_params["dataset"] = cfg.dataset
    best_base_params["model"] = cfg.model

    mask_combinations = generate_mask_combinations()
    logger.info(f"Running {len(mask_combinations)} mask configurations")
    logger.info(f"Mask tuning results will be saved to {mask_tuning_dir}\n")

    results: List[Dict[str, Any]] = []
    for idx, mask_params in enumerate(mask_combinations, 1):
        all_params = {**best_base_params, **mask_params}
        run_name = format_mask_run_name(best_base_params, mask_params, cfg.dataset, cfg.model)

        logger.info(f"[{idx}/{len(mask_combinations)}] Running: {run_name}")
        result_info = run_single_mask_training_run(cfg, run_name, all_params)

        if result_info["status"] == "success" and "final_test_acc1" in result_info:
            logger.info(f"Completed successfully - Test Acc: {result_info['final_test_acc1'] * 100:.2f}%")
        else:
            logger.info("Completed successfully" if result_info["status"] == "success" else "Failed")

        results.append(result_info)
        logger.info("")

    results_file = mask_tuning_dir / "mask_tuning_results.json"
    results_file.write_text(json.dumps(results, indent=2))

    logger.info(f"\nMask tuning complete! Results saved to {results_file}")
    print_summary(mask_tuning_dir, results)
    plot_tuning_results(results, mask_tuning_dir)

    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        logger.warning("No successful mask tuning runs found")
        return None

    if any("best_val_acc" in r for r in successful):
        best_mask_run = max(successful, key=lambda r: r.get("best_val_acc", float("-inf")))
    else:
        best_mask_run = max(successful, key=lambda r: r.get("final_test_acc1", float("-inf")))

    logger.info(
        f"Best mask run: {best_mask_run.get('run_name', 'unknown')} | "
        f"best_val_acc={best_mask_run.get('best_val_acc', 'n/a')} | "
        f"final_test_acc1={best_mask_run.get('final_test_acc1', 'n/a')}"
    )
    return dict(best_mask_run.get("params", {}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Mask Hyperparameter Tuning")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help=f"Dataset name (default: {DEFAULT_DATASET})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--runs_root", type=str, default="./runs", help="Root directory for runs")
    args = parser.parse_args()

    cfg = MaskTuningConfig(
        runs_root=Path(args.runs_root),
        dataset=args.dataset,
        model=args.model,
    )
    tune_mask_hyperparameters(cfg)