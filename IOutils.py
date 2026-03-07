import argparse
import json
from pathlib import Path
import time
import os
import torch
from typing import Any, Dict, List, Optional

from utils import DEFAULT_DATASET, DEFAULT_MODEL
from dataset_registry import get_available_datasets
from model_registry import get_available_models

def non_negative_float(value):
    """Validator that ensures a float is greater than or equal to 0.0"""
    f_value = float(value)
    if f_value < 0.0:
        raise argparse.ArgumentTypeError(f"val_split must be >= 0.0, got {f_value}")
    return f_value


def _dataset_model_kwargs(default_dataset: str, default_model: str) -> Dict[str, Any]:
    datasets = get_available_datasets()
    models = get_available_models()
    kwargs: Dict[str, Any] = {}

    if datasets:
        kwargs["dataset_choices"] = datasets
        kwargs["dataset_default"] = default_dataset if default_dataset in datasets else datasets[0]
    else:
        kwargs["dataset_choices"] = None
        kwargs["dataset_default"] = default_dataset

    if models:
        kwargs["model_choices"] = models
        kwargs["model_default"] = default_model if default_model in models else models[0]
    else:
        kwargs["model_choices"] = None
        kwargs["model_default"] = default_model

    return kwargs


def add_dataset_model_args(parser: argparse.ArgumentParser, default_dataset: str = DEFAULT_DATASET, default_model: str = DEFAULT_MODEL) -> argparse.ArgumentParser:
    resolved = _dataset_model_kwargs(default_dataset, default_model)
    parser.add_argument(
        "--dataset",
        type=str,
        default=resolved["dataset_default"],
        choices=resolved["dataset_choices"],
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=resolved["model_default"],
        choices=resolved["model_choices"],
        help="Model name",
    )
    return parser


def add_tuning_runtime_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--runs_root", type=str, default="./runs", help="Root directory for runs")
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training runs")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs per run")
    return parser

def build_parser():
    p = argparse.ArgumentParser("PyTorch Model Training")
    # Dataset and model selection
    add_dataset_model_args(p, default_dataset=DEFAULT_DATASET, default_model=DEFAULT_MODEL)
    p.add_argument("--data_dir",    type=str,   default="./data")
    p.add_argument("--out_dir",     type=str,   default="./runs")
    # Training hyperparameters
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--lr",          type=float, default=0.1)
    p.add_argument("--momentum",    type=float, default=0.9)
    p.add_argument("--weight_decay",type=float, default=5e-4)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--log_every",   type=int,   default=100)
    p.add_argument("--run_name",    type=str,   default="")
    p.add_argument("--val_split",   type=non_negative_float, default=0.1)
    p.add_argument("--min_lr",      type=float, default=1e-5)
    p.add_argument("--gamma",       type=float, default=0.1)
    p.add_argument("--milestones",  type=str,   default="60,80")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--scheduler",       type=str,   choices=["multistep","cosine"], default="cosine")
    p.add_argument("--warmup_epochs",   type=int,   default=5)
    p.add_argument("--nesterov",        action="store_true",    default=False)
    p.add_argument("--amp",             action="store_true",    default=False)
    # --- CAM-guided cutout ---
    p.add_argument("--masking",     type=str,   choices=["none","random","cam_high","cam_low"], default="none")
    p.add_argument("--mask_warmup_epochs",  type=int,   default=15)
    p.add_argument("--mask_prob",   type=float, default=0.75)
    p.add_argument("--mask_area",   type=float, default=0.2)
    p.add_argument("--mask_block",  type=int,   default=8)
    p.add_argument("--cam_layer",   type=str,   default="layer4")
    return p

def make_run_dir(out_dir, run_name):
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = run_name.strip() or f"run_{ts}"
    run_dir = os.path.join(out_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def write_json(path, obj):
    with open(path, "w") as f: json.dump(obj, f, indent=2, sort_keys=True)

def append_csv(path, row, header=None, mode="a"):
    exists = os.path.exists(path) and mode == "a"
    with open(path, mode) as f:
        if (not exists) and header:
            f.write(",".join(header) + "\n")
        if row:  # Only write row if it's not empty
            f.write(",".join(str(x) for x in row) + "\n")

def save_ckpt(path, model, optimizer, epoch, best_acc, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
    }
    if extra is not None:
        state.update(extra)
    torch.save(state, path)

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def build_args_from_params(params):
    """Convert a params dict to an argparse.Namespace object that train_with_config expects."""
    parser = build_parser()
    args = parser.parse_args([])  # Parse empty args to get defaults
    
    # Override with provided params
    for key, value in params.items():
        setattr(args, key, value)
    
    return args


def normalize_masking_type(
    masking_type: Optional[str],
    valid_masking_values: List[str],
    all_value: str = "all",
) -> Optional[str]:
    if masking_type in {None, all_value}:
        return None

    if masking_type not in valid_masking_values:
        valid = ", ".join([all_value, *valid_masking_values])
        raise ValueError(f"Invalid masking_type '{masking_type}'. Expected one of: {valid}")

    return masking_type


def format_mask_run_name(
    base_params: Dict[str, Any],
    mask_params: Dict[str, Any],
    dataset: str,
    model: str,
    prefix: str,
) -> str:
    base_name = (
        f"{prefix}_{model}_{dataset}_ep{base_params['epochs']}_bs{base_params['batch_size']}_"
        f"lr{base_params['lr']}_wd{float(base_params['weight_decay']):.0e}_"
        f"ls{base_params['label_smoothing']}_wu{base_params['warmup_epochs']}"
    )

    return (
        f"{base_name}_mask{mask_params['masking']}_mwu{mask_params['mask_warmup_epochs']}_"
        f"mp{mask_params['mask_prob']}_ma{mask_params['mask_area']}_"
        f"mb{mask_params['mask_block']}_cl{mask_params['cam_layer']}"
    )