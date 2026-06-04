import argparse
import json
from pathlib import Path
import time
import os
from typing import Any, Dict, Optional

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

def add_data_loading_args(
    parser: argparse.ArgumentParser,
    *,
    data_dir_default: str = "./data",
    batch_size_default: int = 128,
    num_workers_default: int = 2,
    val_split_default: float = 0.1,
) -> argparse.ArgumentParser:
    parser.add_argument("--data_dir", type=str, default=data_dir_default)
    parser.add_argument("--batch_size", type=int, default=batch_size_default)
    parser.add_argument("--num_workers", type=int, default=num_workers_default)
    parser.add_argument("--val_split", type=non_negative_float, default=val_split_default)
    return parser


def add_training_hparam_args(
    parser: argparse.ArgumentParser,
    *,
    epochs_default: int = 100,
    lr_default: float = 0.1,
    momentum_default: float = 0.9,
    weight_decay_default: float = 5e-4,
) -> argparse.ArgumentParser:
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--epochs", type=int, default=epochs_default)
    parser.add_argument("--lr", type=float, default=lr_default)
    parser.add_argument("--momentum", type=float, default=momentum_default)
    parser.add_argument("--weight_decay", type=float, default=weight_decay_default)
    parser.add_argument("--adamw_betas", nargs=2, type=float, default=[0.9, 0.999])
    parser.add_argument("--adamw_eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--milestones", type=str, default="60,80")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, choices=["multistep", "cosine"], default="cosine")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--nesterov", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False)
    return parser


def add_cutout_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--cutout_mode",
        type=str,
        choices=["none", "random", "cam_low", "cam_high"],
        default="none",
    )
    parser.add_argument("--cutout_m", type=int, default=0)
    parser.add_argument("--cutout_size", type=int, default=0)
    parser.add_argument("--cutout_area", type=float, default=None)
    parser.add_argument("--teacher_model", type=str, default="")
    parser.add_argument("--teacher_checkpoint", type=str, default="")
    parser.add_argument("--cam_layer", type=str, default="auto")
    parser.add_argument("--saliency_candidate_percent", type=float, default=10.0)
    parser.add_argument("--grayscale", action="store_true", default=False)
    parser.add_argument("--include_regex", type=str, default="")
    return parser

def build_parser():
    p = argparse.ArgumentParser("PyTorch Model Training")
    # Dataset and model selection
    add_dataset_model_args(p, default_dataset=DEFAULT_DATASET, default_model=DEFAULT_MODEL)
    add_data_loading_args(p)
    p.add_argument("--out_dir", type=str, default="./runs")
    add_training_hparam_args(p)
    add_cutout_args(p)
    return p

def make_run_dir(out_dir, run_name):
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = run_name.strip() or f"run_{ts}"
    run_dir = os.path.join(out_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def build_time_tags() -> Dict[str, str]:
    return {
        "year_month": time.strftime("%Y_%m"),
        "day": time.strftime("%d"),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }

def write_json(path, obj):
    with open(path, "w") as f: json.dump(obj, f, indent=2, sort_keys=True)

def append_csv(path, row, header=None, mode="a"):
    exists = os.path.exists(path) and mode == "a"
    with open(path, mode) as f:
        if (not exists) and header:
            f.write(",".join(header) + "\n")
        if row:  # Only write row if it's not empty
            f.write(",".join(str(x) for x in row) + "\n")

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


TRAIN_ARG_FIELDS = (
    "dataset",
    "model",
    "data_dir",
    "out_dir",
    "optimizer",
    "epochs",
    "batch_size",
    "num_workers",
    "lr",
    "momentum",
    "weight_decay",
    "adamw_betas",
    "adamw_eps",
    "seed",
    "log_every",
    "val_split",
    "min_lr",
    "gamma",
    "milestones",
    "label_smoothing",
    "scheduler",
    "warmup_epochs",
    "nesterov",
    "amp",
    "cutout_mode",
    "cutout_m",
    "cutout_size",
    "cutout_area",
    "teacher_model",
    "teacher_checkpoint",
    "cam_layer",
    "saliency_candidate_percent",
    "grayscale",
    "include_regex",
)


def namespace_to_train_params(
    source: Any,
    *,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    out_dir: Optional[str] = None,
    run_name: str = "",
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for field in TRAIN_ARG_FIELDS:
        if hasattr(source, field):
            params[field] = getattr(source, field)

    if model is not None:
        params["model"] = model
    if dataset is not None:
        params["dataset"] = dataset
    if out_dir is not None:
        params["out_dir"] = out_dir

    params["run_name"] = run_name
    return params


def init_run_dir_with_config(out_dir: str, run_name: str, config: Dict[str, Any]) -> str:
    run_dir = make_run_dir(out_dir, run_name)
    write_json(os.path.join(run_dir, "config.json"), config)
    return run_dir

def build_args_from_params(params):
    """Convert a params dict to an argparse.Namespace object that train_with_config expects."""
    parser = build_parser()
    args = parser.parse_args([])

    for key, value in params.items():
        setattr(args, key, value)

    return args








