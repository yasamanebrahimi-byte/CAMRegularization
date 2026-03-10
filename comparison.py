"""
comparison.py — HiResCAM Masking Comparison Pipeline

Trains *i* input CNN models on a dataset, computes per-image HiResCAM heatmaps,
merges them with an element-wise-max (logical OR-style), and generates four
dataset variants:

    1) original   – unchanged images
    2) random     – one fixed random mask per image (budget-matched to CAM masks)
    3) low_saliency  – hide pixels where merged HiResCAM <= threshold
    4) high_saliency – hide pixels where merged HiResCAM >= (1 - threshold)

A set of *output* models is then trained on each variant and evaluated on the
*original* test set so the comparison is fair.  Accuracy, weighted-F1 and loss are
reported side by side at the end.

Usage example
─────────────
    python comparison.py \\
        --input_models resnet18 resnet34 \\
        --output_models resnet50 resnet32 \\
        --dataset cifar100 \\
        --epochs 100 --batch_size 128
"""

import argparse
import json
import os
import shutil
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from cam_masking import HiResCAM, resolve_cam_target_module
from dataset_registry import (
    get_dataset_loaders,
    get_default_input_size,
    get_normalization_params,
    get_num_classes,
    infer_num_classes_from_loader,
)
from IOutils import (
    build_args_from_params,
    write_json,
    namespace_to_train_params,
    init_run_dir_with_config,
    add_data_loading_args,
    add_training_hparam_args,
)
from logger import get_logger
from train import train_with_config
from utils import set_seed, denormalize_tensor, tensor_to_pil_image

VARIANTS = ["original", "random", "low_saliency", "high_saliency"]

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.json")


# ---------------------------------------------------------------------------
# Variant zip caching helpers
# ---------------------------------------------------------------------------


def _build_variant_zip_name(dataset: str, input_models: List[str]) -> str:
    """Build a canonical zip filename from dataset and input model names.

    Models are sorted alphabetically so the name is order-independent.
    Example: malimg_densenet121_efficientnet_b0.zip
    """
    sorted_models = sorted(input_models)
    return f"{dataset}_{'_'.join(sorted_models)}.zip"


def _find_cached_variant_zip(data_dir: str, dataset: str, input_models: List[str]) -> Optional[Path]:
    """Check if a cached variant zip exists in *data_dir* for the given
    dataset / input-model combination (order-independent).  Returns the
    Path to the zip if found, else None."""
    zip_name = _build_variant_zip_name(dataset, input_models)
    zip_path = Path(data_dir) / zip_name
    if zip_path.is_file():
        return zip_path
    return None


def _zip_variant_directory(variant_root: Path, data_dir: str, dataset: str,
                           input_models: List[str], logger) -> Path:
    """Compress the entire *variant_root* directory into a zip archive
    stored in *data_dir* and return its path."""
    zip_name = _build_variant_zip_name(dataset, input_models)
    zip_path = Path(data_dir) / zip_name
    logger.info(f"Zipping variant directory {variant_root} → {zip_path}")

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(variant_root.rglob("*")):
            if file.is_file():
                arcname = file.relative_to(variant_root.parent)
                zf.write(str(file), str(arcname))

    logger.info(f"Zip created: {zip_path} ({zip_path.stat().st_size / (1024**2):.1f} MB)")
    return zip_path


def _extract_variant_zip(zip_path: Path, data_dir: str, logger) -> Path:
    """Extract a cached variant zip into *data_dir* and return the
    variant_root directory (the top-level folder inside the zip)."""
    logger.info(f"Extracting cached variant zip {zip_path} → {data_dir}")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(data_dir)

    # The zip was created with arcnames relative to variant_root.parent,
    # so the top-level directory name is the variant_root folder name.
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        top = zf.namelist()[0].split("/")[0]
    variant_root = Path(data_dir) / top
    logger.info(f"Extracted variant root: {variant_root}")
    return variant_root


# ---------------------------------------------------------------------------
# Config loading and resolution
# ---------------------------------------------------------------------------


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load the JSON configuration file. Returns empty dict if file not found."""
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def _resolve_training_args(
    cli_args: argparse.Namespace,
    config: Dict[str, Any],
    model: str,
    dataset: str,
) -> argparse.Namespace:
    """Build a resolved Namespace by layering config on top of CLI args.

    Resolution order (later wins):
        CLI defaults -> config["defaults"] -> config["model_defaults"][model]
        -> config["dataset_defaults"][dataset]
        -> config["overrides"]["model::dataset"]
    """
    resolved = argparse.Namespace(**vars(cli_args))
    if not config:
        return resolved

    # Layer 1: global defaults from config
    for key, value in config.get("defaults", {}).items():
        if hasattr(resolved, key):
            setattr(resolved, key, value)

    # Layer 2: model-specific defaults
    for key, value in config.get("model_defaults", {}).get(model, {}).items():
        if hasattr(resolved, key):
            setattr(resolved, key, value)

    # Layer 3: dataset-specific defaults
    for key, value in config.get("dataset_defaults", {}).get(dataset, {}).items():
        if hasattr(resolved, key):
            setattr(resolved, key, value)

    # Layer 4: model + dataset overrides
    override_key = f"{model}::{dataset}"
    for key, value in config.get("overrides", {}).get(override_key, {}).items():
        if hasattr(resolved, key):
            setattr(resolved, key, value)

    return resolved


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HiResCAM Masking Comparison Pipeline",
    )

    # Models
    p.add_argument(
        "--input_models",
        nargs="+",
        required=True,
        help="Names of input models to train and generate HiResCAMs from (e.g. resnet18 resnet34)",
    )
    p.add_argument(
        "--output_models",
        nargs="+",
        required=True,
        help="Names of downstream output models to evaluate on the masked datasets",
    )

    # Dataset
    p.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. cifar100)")
    add_data_loading_args(
        p,
        data_dir_default="./data",
        batch_size_default=128,
        num_workers_default=2,
        val_split_default=0.1,
    )
    p.add_argument("--out_dir", type=str, default=None)

    # Training hyper-parameters (shared by input and output training)
    add_training_hparam_args(p, epochs_default=100, lr_default=0.1, momentum_default=0.9, weight_decay_default=5e-4)

    # CAM
    p.add_argument("--cam_layer", type=str, default="auto")
    p.add_argument("--threshold", type=float, default=0.3)

    # Config file
    p.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to configs.json with per-model/dataset hyperparameters",
    )

    return p


# ---------------------------------------------------------------------------
# Step 1 — train input models (returns trained nn.Module objects)
# ---------------------------------------------------------------------------


def train_input_models(
    cli_args, config: Dict[str, Any], logger
) -> List[Tuple[str, nn.Module]]:
    """Train each input model on the dataset and return (name, model) pairs."""
    trained: List[Tuple[str, nn.Module]] = []

    for model_name in cli_args.input_models:
        resolved = _resolve_training_args(cli_args, config, model_name, cli_args.dataset)
        logger.info(f"\n{'='*60}")
        logger.info(f"Training input model: {model_name}")
        logger.info(
            f"  Resolved config → lr={resolved.lr}, wd={resolved.weight_decay}, "
            f"bs={resolved.batch_size}, epochs={resolved.epochs}, "
            f"scheduler={resolved.scheduler}, warmup={resolved.warmup_epochs}"
        )
        logger.info(f"{'='*60}")

        params = namespace_to_train_params(
            resolved,
            model=model_name,
            dataset=cli_args.dataset,
            out_dir=cli_args.out_dir,
            run_name=f"input_{model_name}",
        )
        args = build_args_from_params(params)
        run_dir = init_run_dir_with_config(args.out_dir, args.run_name, vars(args))

        result, model = train_with_config(args, run_dir=run_dir, logger=logger, return_model=True)
        logger.info(
            f"Input model {model_name} done — test acc {result['final_test_acc1']*100:.2f}%, "
            f"test F1 {result['final_test_f1']*100:.2f}%"
        )
        trained.append((model_name, model))

    return trained


# ---------------------------------------------------------------------------
# Step 2 — compute merged HiResCAM heatmaps and save variant datasets
# ---------------------------------------------------------------------------


def _get_class_names(dataloader: DataLoader) -> List[str]:
    """Best-effort extraction of class names from a DataLoader's dataset."""
    ds = dataloader.dataset
    while isinstance(ds, Subset):
        ds = ds.dataset
    if hasattr(ds, "classes"):
        return list(ds.classes)
    # Fallback — numeric names
    num_classes = infer_num_classes_from_loader(dataloader) or 10
    return [str(i) for i in range(num_classes)]


def _get_eval_transform(dataset_name: str):
    """Return a deterministic (no augmentation) transform suitable for CAM generation."""
    input_size = get_default_input_size(dataset_name)
    mean, std = get_normalization_params(dataset_name)
    if input_size <= 64:
        # Small images (CIFAR, Tiny-ImageNet): just tensor + norm
        return T.Compose([T.Resize((input_size, input_size)), T.ToTensor(), T.Normalize(mean, std)])
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def _get_raw_dataset(dataloader: DataLoader, eval_transform):
    """Create a dataset with eval transforms from the underlying data source.

    Works for CIFAR (which stores data internally) and ImageFolder datasets.
    Returns a dataset that yields (tensor, label) with deterministic transforms.
    """
    ds = dataloader.dataset
    # Unwrap Subset layers
    indices = None
    while isinstance(ds, Subset):
        if indices is None:
            indices = list(ds.indices)
        else:
            indices = [ds.indices[i] for i in indices]
        ds = ds.dataset

    # Clone the dataset with the eval transform
    if hasattr(ds, "data") and hasattr(ds, "targets"):
        # CIFAR-style datasets
        class _CifarEval(Dataset):
            def __init__(self, data, targets, transform, idx=None):
                self.data = data
                self.targets = targets
                self.transform = transform
                self.indices = idx

            def __len__(self):
                return len(self.indices) if self.indices is not None else len(self.data)

            def __getitem__(self, i):
                actual_i = self.indices[i] if self.indices is not None else i
                img = Image.fromarray(self.data[actual_i])
                label = int(self.targets[actual_i])
                if self.transform:
                    img = self.transform(img)
                return img, label

        return _CifarEval(ds.data, ds.targets, eval_transform, idx=indices)

    if hasattr(ds, "samples"):
        # ImageFolder-style
        class _FolderEval(Dataset):
            def __init__(self, samples, transform, idx=None):
                self.samples = samples
                self.transform = transform
                self.indices = idx

            def __len__(self):
                return len(self.indices) if self.indices is not None else len(self.samples)

            def __getitem__(self, i):
                actual_i = self.indices[i] if self.indices is not None else i
                path, label = self.samples[actual_i]
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label

        return _FolderEval(ds.samples, eval_transform, idx=indices)

    # Fallback — just wrap the existing dataset and override its transform
    class _Wrapped(Dataset):
        def __init__(self, base_ds, transform, idx=None):
            self.base_ds = base_ds
            self.transform = transform
            self.indices = idx

        def __len__(self):
            return len(self.indices) if self.indices is not None else len(self.base_ds)

        def __getitem__(self, i):
            actual_i = self.indices[i] if self.indices is not None else i
            img, label = self.base_ds[actual_i]
            return img, label

    return _Wrapped(ds, eval_transform, idx=indices)


@torch.no_grad()
def _compute_merged_cam_batch(
    images: torch.Tensor,
    trained_models: List[Tuple[str, nn.Module]],
    device: torch.device,
    cam_layer: str,
) -> torch.Tensor:
    """Compute per-image merged HiResCAM across all input models.

    Returns merged CAM: [B, H, W] in [0, 1].
    """
    merged = None
    for _, model in trained_models:
        model.eval()
        _, target_module = resolve_cam_target_module(model, cam_layer)
        cam_runner = HiResCAM(model, target_module)

        # HiResCAM.cam needs gradients enabled
        with torch.enable_grad():
            x = images.detach().clone().requires_grad_(True).to(device)
            cam = cam_runner.cam(x)  # [B, 1, H, W]

        cam_runner.close()
        cam = cam[:, 0].detach()  # [B, H, W]

        if merged is None:
            merged = cam
        else:
            merged = torch.max(merged, cam)  # OR-style merge

    return merged  # [B, H, W] in [0, 1]


def generate_and_save_variants(
    trained_models: List[Tuple[str, nn.Module]],
    dataloader: DataLoader,
    split: str,
    variant_root: Path,
    class_names: List[str],
    mean: Tuple,
    std: Tuple,
    device: torch.device,
    cam_layer: str,
    threshold: float,
    seed: int,
    logger,
) -> None:
    """Generate and save all four dataset variants for one split (train or test)."""
    logger.info(f"Generating {split} variants → {variant_root}")

    # Create output directories
    for variant in VARIANTS:
        for cls_name in class_names:
            (variant_root / variant / split / cls_name).mkdir(parents=True, exist_ok=True)

    global_idx = 0
    rng = np.random.RandomState(seed)

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        merged_cam = _compute_merged_cam_batch(images, trained_models, device, cam_layer)

        # Denormalize for saving
        denorm = denormalize_tensor(images.cpu(), mean, std)

        B = images.size(0)
        for i in range(B):
            img_01 = denorm[i]  # [C, H, W] in ~[0,1]
            cam_i = merged_cam[i].cpu()  # [H, W] in [0,1]
            label = labels[i].item()
            cls_name = class_names[label]
            fname = f"img_{global_idx:06d}.png"

            # 1) Original
            pil_orig = tensor_to_pil_image(img_01)
            pil_orig.save(str(variant_root / "original" / split / cls_name / fname))

            # 2) Low saliency — hide pixels where cam <= threshold
            low_mask = (cam_i <= threshold).float()  # 1 where to hide
            img_low = img_01 * (1 - low_mask)
            tensor_to_pil_image(img_low).save(str(variant_root / "low_saliency" / split / cls_name / fname))

            # 3) High saliency — hide pixels where cam >= (1 - threshold)
            high_mask = (cam_i >= (1.0 - threshold)).float()
            img_high = img_01 * (1 - high_mask)
            tensor_to_pil_image(img_high).save(str(variant_root / "high_saliency" / split / cls_name / fname))

            # 4) Random — mask same percentage as threshold
            H, W = cam_i.shape
            total_pixels = H * W
            random_budget = int(round(threshold * total_pixels))
            random_budget = min(random_budget, total_pixels)
            flat_indices = rng.choice(total_pixels, size=random_budget, replace=False)
            random_mask = torch.zeros(H * W)
            random_mask[flat_indices] = 1.0
            random_mask = random_mask.view(H, W)
            img_rand = img_01 * (1 - random_mask)
            tensor_to_pil_image(img_rand).save(str(variant_root / "random" / split / cls_name / fname))

            global_idx += 1

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            logger.info(f"  [{split}] processed {global_idx} images …")

    logger.info(f"  [{split}] saved {global_idx} images for all 4 variants")


# ---------------------------------------------------------------------------
# Step 3 — train output model on each variant and collect results
# ---------------------------------------------------------------------------


def _build_variant_loaders(
    variant_dir: Path,
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Create train / val / test loaders from saved image folders."""
    input_size = get_default_input_size(dataset_name)
    mean, std = get_normalization_params(dataset_name)

    if input_size <= 64:
        train_tfms = T.Compose([
            T.RandomCrop(input_size, padding=input_size // 8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        train_tfms = T.Compose([
            T.Resize((input_size, input_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    test_tfms = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.ImageFolder(root=str(variant_dir / "train"), transform=train_tfms)
    test_ds = torchvision.datasets.ImageFolder(root=str(variant_dir / "test"), transform=test_tfms)

    # Create val split from training data
    val_ds = None
    if val_split and val_split > 0:
        n = len(train_ds)
        n_train = int(n * (1.0 - val_split))
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).tolist()
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        # Val uses test transforms (no augmentation)
        val_ds_base = torchvision.datasets.ImageFolder(root=str(variant_dir / "train"), transform=test_tfms)
        train_ds = Subset(train_ds, train_idx)
        val_ds = Subset(val_ds_base, val_idx)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    val_dl = (
        DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)
        if val_ds is not None
        else None
    )
    return train_dl, val_dl, test_dl


def train_output_on_variant(
    cli_args,
    config: Dict[str, Any],
    output_model: str,
    variant: str,
    variant_root: Path,
    results_root: Path,
    original_test_dl: DataLoader,
    num_classes: int,
    input_size: int,
    device: torch.device,
    logger,
) -> Dict[str, Any]:
    """Train output model via train_with_config on variant train data, evaluate on original test set."""
    resolved = _resolve_training_args(cli_args, config, output_model, cli_args.dataset)

    logger.info(f"\n{'='*60}")
    logger.info(f"Training output model ({output_model}) on variant: {variant}")
    logger.info(
        f"  Resolved config → lr={resolved.lr}, wd={resolved.weight_decay}, "
        f"bs={resolved.batch_size}, epochs={resolved.epochs}, "
        f"scheduler={resolved.scheduler}, warmup={resolved.warmup_epochs}"
    )
    logger.info("  Eval split during training is forced to original test for fair comparability")
    logger.info(f"{'='*60}")

    variant_dir = variant_root / variant
    train_dl, _, _ = _build_variant_loaders(
        variant_dir,
        resolved.dataset,
        resolved.batch_size,
        resolved.num_workers,
        0.0,
        resolved.seed,
    )

    params = namespace_to_train_params(
        resolved,
        model=output_model,
        dataset=resolved.dataset,
        out_dir=str(results_root / output_model / variant),
        run_name=f"output_{output_model}_{variant}",
    )
    args = build_args_from_params(params)
    run_dir = Path(results_root) / output_model / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(str(run_dir / "config.json"), vars(args))

    result = train_with_config(
        args,
        run_dir=str(run_dir),
        logger=logger,
        train_dl=train_dl,
        val_dl=None,
        test_dl=original_test_dl,
    )

    logger.info(
        f"Variant [{variant}] done — test acc {result['final_test_acc1']*100:.2f}%, "
        f"test F1 {result['final_test_f1']*100:.2f}%, test loss {result['final_test_loss']:.4f}"
    )

    return {
        "output_model": output_model,
        "variant": variant,
        "final_test_acc1": result["final_test_acc1"],
        "final_test_f1": result["final_test_f1"],
        "final_test_loss": result["final_test_loss"],
        "best_val_acc": result["best_val_acc"],
    }


# ---------------------------------------------------------------------------
# Step 4 — comparison table
# ---------------------------------------------------------------------------


def print_comparison(results: Dict[str, Dict[str, Dict[str, Any]]], logger) -> None:
    """Pretty-print per-output-model comparison tables of variant results."""
    for output_model, model_results in results.items():
        logger.info(f"\n{'='*72}")
        logger.info(f"COMPARISON RESULTS — output model: {output_model}")
        logger.info(f"{'='*72}")
        logger.info(f"{'Variant':<18} {'Accuracy':>10} {'F1':>10} {'Loss':>10}")
        logger.info(f"{'-'*18} {'-'*10} {'-'*10} {'-'*10}")
        for variant in VARIANTS:
            r = model_results.get(variant)
            if r is None:
                logger.info(f"{variant:<18} {'FAILED':>10} {'—':>10} {'—':>10}")
            else:
                acc = f"{r['final_test_acc1']*100:.2f}%"
                f1 = f"{r['final_test_f1']*100:.2f}%"
                loss = f"{r['final_test_loss']:.4f}"
                logger.info(f"{variant:<18} {acc:>10} {f1:>10} {loss:>10}")
        logger.info(f"{'='*72}")

        print(f"\n{'='*72}")
        print(f"COMPARISON RESULTS — output model: {output_model}")
        print(f"{'='*72}")
        print(f"{'Variant':<18} {'Accuracy':>10} {'F1':>10} {'Loss':>10}")
        print(f"{'-'*18} {'-'*10} {'-'*10} {'-'*10}")
        for variant in VARIANTS:
            r = model_results.get(variant)
            if r is None:
                print(f"{variant:<18} {'FAILED':>10} {'—':>10} {'—':>10}")
            else:
                acc = f"{r['final_test_acc1']*100:.2f}%"
                f1 = f"{r['final_test_f1']*100:.2f}%"
                loss = f"{r['final_test_loss']:.4f}"
                print(f"{variant:<18} {acc:>10} {f1:>10} {loss:>10}")
        print(f"{'='*72}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _build_parser().parse_args()

    # ── Assertions ────────────────────────────────────────────────────────
    assert len(args.input_models) > 0, "At least one input model is required."
    assert len(set(args.input_models)) == len(args.input_models), (
        "All input models must be distinct."
    )
    assert len(args.output_models) > 0, "At least one output model is required."
    assert len(set(args.output_models)) == len(args.output_models), (
        "All output models must be distinct."
    )
    overlap = set(args.output_models).intersection(set(args.input_models))
    assert len(overlap) == 0, (
        f"Output models {sorted(overlap)} must differ from all input models."
    )
    assert 0.0 <= args.threshold <= 1.0, "--threshold must be in [0, 1]."

    # ── Load per-model/dataset config ─────────────────────────────────────
    config = _load_config(args.config)
    if config:
        print(f"Loaded hyperparameter config from {args.config}")
    else:
        print(f"No config file found at {args.config} — using CLI defaults for all models")

    # ── Setup ─────────────────────────────────────────────────────────────
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.out_dir:
        base_out_dir = args.out_dir
    else:
        date_tag = time.strftime("%Y_%m_%d")
        base_out_dir = os.path.join("./runs", f"{args.dataset}_{date_tag}")
    run_name = args.run_name.strip() if getattr(args, "run_name", "") else ""
    if not run_name:
        run_name = f"comparison_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}"

    Path(base_out_dir).mkdir(parents=True, exist_ok=True)
    write_json(str(Path(base_out_dir) / "comparison_config.json"), vars(args))
    args.out_dir = base_out_dir

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(base_out_dir) / f"comparison_{args.dataset}_{timestamp}.log"
    logger = get_logger(__name__, log_file=log_path, console=False)
    logger.info(f"Comparison pipeline | device={device} | out_dir={base_out_dir} | args={json.dumps(vars(args), sort_keys=True)}")

    mean, std = get_normalization_params(args.dataset)
    input_size = get_default_input_size(args.dataset)
    num_classes = get_num_classes(args.dataset)

    # ── Check for cached variant zip ───────────────────────────────────────
    cached_zip = _find_cached_variant_zip(args.data_dir, args.dataset, args.input_models)
    variant_root = Path(args.data_dir) / f"comparison_{args.dataset}"

    if cached_zip is not None:
        logger.info(f"\nFound cached variant zip: {cached_zip}")
        logger.info("Skipping input model training and variant generation.")
        print(f"Found cached variant zip: {cached_zip} — skipping input training & CAM generation")
        variant_root = _extract_variant_zip(cached_zip, args.data_dir, logger)
    else:
        # ── Step 1: train input models ────────────────────────────────────
        trained_models = train_input_models(args, config, logger)

        # ── Step 2: build eval-mode dataloaders (no augmentation) for CAM ─
        logger.info("\nPreparing eval dataloaders for HiResCAM computation …")
        train_dl, _, test_dl = get_dataset_loaders(
            args.dataset, args.data_dir, args.batch_size, args.num_workers,
            val_split=args.val_split, seed=args.seed,
        )
        eval_tfm = _get_eval_transform(args.dataset)
        eval_train_ds = _get_raw_dataset(train_dl, eval_tfm)
        eval_test_ds = _get_raw_dataset(test_dl, eval_tfm)

        eval_train_dl = DataLoader(eval_train_ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=True)
        eval_test_dl = DataLoader(eval_test_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

        class_names = _get_class_names(train_dl)

        # ── Step 3: generate and save four dataset variants ───────────────
        logger.info(f"\nSaving variant datasets to {variant_root}")

        for split_name, dl in [("train", eval_train_dl), ("test", eval_test_dl)]:
            generate_and_save_variants(
                trained_models=trained_models,
                dataloader=dl,
                split=split_name,
                variant_root=variant_root,
                class_names=class_names,
                mean=mean,
                std=std,
                device=device,
                cam_layer=args.cam_layer,
                threshold=args.threshold,
                seed=args.seed,
                logger=logger,
            )

        # ── Zip the variant directory for future reuse ────────────────────
        _zip_variant_directory(variant_root, args.data_dir, args.dataset,
                               args.input_models, logger)

    # ── Step 4: build the original test loader for fair evaluation ────────
    #    (all variants are evaluated on the *same* original test set)
    orig_test_dl = DataLoader(
        torchvision.datasets.ImageFolder(
            root=str(variant_root / "original" / "test"),
            transform=T.Compose([
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]),
        ),
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Infer num_classes from actual saved data
    inferred = infer_num_classes_from_loader(orig_test_dl)
    if inferred is not None:
        num_classes = inferred

    # ── Step 5: train each output model on each variant ───────────────────
    comparison_results: Dict[str, Dict[str, Any]] = {}
    output_results_root = Path(base_out_dir)
    for output_model in args.output_models:
        comparison_results[output_model] = {}
        logger.info(f"\nStarting output model sweep: {output_model}")
        for variant in VARIANTS:
            try:
                result = train_output_on_variant(
                    cli_args=args,
                    config=config,
                    output_model=output_model,
                    variant=variant,
                    variant_root=variant_root,
                    results_root=output_results_root,
                    original_test_dl=orig_test_dl,
                    num_classes=num_classes,
                    input_size=input_size,
                    device=device,
                    logger=logger,
                )
                comparison_results[output_model][variant] = result
            except Exception as exc:
                logger.error(f"Training output model '{output_model}' on variant '{variant}' failed: {exc}")
                logger.error(traceback.format_exc())
                comparison_results[output_model][variant] = None

    # ── Step 6: save and display comparison ───────────────────────────────
    results_path = os.path.join(base_out_dir, "comparison_results.json")
    write_json(results_path, comparison_results)
    logger.info(f"\nResults saved to {results_path}")

    print_comparison(comparison_results, logger)


if __name__ == "__main__":
    main()
