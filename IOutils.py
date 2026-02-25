import argparse
import json
from pathlib import Path
import time
import os
import torch

def positive_float(value):
    """Validator that ensures a float is greater than 0.0"""
    f_value = float(value)
    if f_value <= 0.0:
        raise argparse.ArgumentTypeError(f"val_split must be > 0.0, got {f_value}")
    return f_value

def build_parser():
    p = argparse.ArgumentParser("PyTorch Model Training")
    # Dataset and model selection
    p.add_argument("--dataset",     type=str,   default="cifar100",  help="Dataset name (e.g., 'cifar100', 'cifar10')")
    p.add_argument("--model",       type=str,   default="resnet18",   help="Model name (e.g., 'resnet18', 'resnet50', 'vgg16')")
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
    p.add_argument("--val_split",   type=positive_float, default=0.1)
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