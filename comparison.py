"""
comparison.py — HiResCAM Masking Comparison Pipeline

Trains *i* input CNN models on a dataset, computes per-image HiResCAM heatmaps,
merges them with an element-wise-max (logical OR-style), and generates four
dataset variants:

    1) original   – unchanged images
    2) random     – one fixed random mask per image (budget-matched to CAM masks)
    3) low_saliency  – hide pixels where merged HiResCAM <= 0.3
    4) high_saliency – hide pixels where merged HiResCAM >= 0.7

A separate *output* model is then trained on each variant and evaluated on the
*original* test set so the comparison is fair.  Accuracy, macro-F1 and loss are
reported side by side at the end.

Usage example
─────────────
    python comparison.py \\
        --input_models resnet18 resnet34 \\
        --output_model resnet50 \\
        --dataset cifar100 \\
        --epochs 100 --batch_size 128
"""

import argparse
import json
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
from engine import evaluate, train_one_epoch
from graphics import plot_metrics, denormalize_tensor, tensor_to_pil_image
from IOutils import (
    append_csv,
    build_args_from_params,
    make_run_dir,
    write_json,
    namespace_to_train_params,
    init_run_dir_with_config,
    add_data_loading_args,
    add_training_hparam_args,
)
from logger import SimpleLogger, get_logger
from model_registry import get_model
from train import build_optimizer, train_with_config
from utils import infer_input_size_from_loader, set_seed

VARIANTS = ["original", "random", "low_saliency", "high_saliency"]


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
        "--output_model",
        type=str,
        required=True,
        help="Name of the downstream output model to evaluate on the masked datasets",
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
    p.add_argument("--out_dir", type=str, default="./runs/comparison")

    # Training hyper-parameters (shared by input and output training)
    add_training_hparam_args(p, epochs_default=100, lr_default=0.1, momentum_default=0.9, weight_decay_default=5e-4)

    # CAM
    p.add_argument("--cam_layer", type=str, default="auto")
    p.add_argument("--low_threshold", type=float, default=0.3)
    p.add_argument("--high_threshold", type=float, default=0.7)

    return p


# ---------------------------------------------------------------------------
# Step 1 — train input models (returns trained nn.Module objects)
# ---------------------------------------------------------------------------


def train_input_models(
    cli_args, logger
) -> List[Tuple[str, nn.Module]]:
    """Train each input model on the dataset and return (name, model) pairs."""
    trained: List[Tuple[str, nn.Module]] = []

    for model_name in cli_args.input_models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training input model: {model_name}")
        logger.info(f"{'='*60}")

        params = namespace_to_train_params(
            cli_args,
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
    low_thr: float,
    high_thr: float,
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

            # 2) Low saliency — hide pixels where cam <= low_thr
            low_mask = (cam_i <= low_thr).float()  # 1 where to hide
            img_low = img_01 * (1 - low_mask)
            tensor_to_pil_image(img_low).save(str(variant_root / "low_saliency" / split / cls_name / fname))

            # 3) High saliency — hide pixels where cam >= high_thr
            high_mask = (cam_i >= high_thr).float()
            img_high = img_01 * (1 - high_mask)
            tensor_to_pil_image(img_high).save(str(variant_root / "high_saliency" / split / cls_name / fname))

            # 4) Random — match budget (average of low + high mask counts)
            low_count = int(low_mask.sum().item())
            high_count = int(high_mask.sum().item())
            random_budget = int(round((low_count + high_count) / 2))
            H, W = cam_i.shape
            total_pixels = H * W
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
    variant: str,
    variant_root: Path,
    original_test_dl: DataLoader,
    num_classes: int,
    input_size: int,
    device: torch.device,
    logger,
) -> Dict[str, Any]:
    """Train the output model on *variant* training data, evaluate on the original test set."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training output model ({cli_args.output_model}) on variant: {variant}")
    logger.info(f"{'='*60}")

    variant_dir = variant_root / variant
    train_dl, val_dl, _ = _build_variant_loaders(
        variant_dir,
        cli_args.dataset,
        cli_args.batch_size,
        cli_args.num_workers,
        cli_args.val_split,
        cli_args.seed,
    )

    model = get_model(cli_args.output_model, num_classes=num_classes, input_size=input_size).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cli_args.label_smoothing)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cli_args.lr,
        momentum=cli_args.momentum,
        weight_decay=cli_args.weight_decay,
        nesterov=cli_args.nesterov,
    )

    # Scheduler
    if cli_args.scheduler == "multistep":
        ms = [int(x) for x in cli_args.milestones.split(",") if x.strip()]
        main_sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms, gamma=cli_args.gamma)
    else:
        main_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cli_args.epochs - cli_args.warmup_epochs), eta_min=cli_args.min_lr
        )

    if cli_args.warmup_epochs > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=cli_args.warmup_epochs)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, main_sched], milestones=[cli_args.warmup_epochs]
        )
    else:
        scheduler = main_sched

    scaler = torch.amp.GradScaler(enabled=(cli_args.amp and device.type == "cuda"))

    run_dir = make_run_dir(cli_args.out_dir, f"output_{variant}")
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    header = ["epoch", "lr", "train_loss", "train_acc1", "train_f1",
              "eval_loss", "eval_acc1", "eval_f1", "eval_split"]
    append_csv(metrics_csv, [], header=header, mode="w")

    best = 0.0
    for epoch in range(cli_args.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_a1, tr_f1 = train_one_epoch(
            model, train_dl, criterion, optimizer,
            scaler if scaler.is_enabled() else None,
            device, log_every=cli_args.log_every,
        )

        if val_dl is not None:
            ev_loss, ev_a1, ev_f1 = evaluate(model, val_dl, criterion, device)
            split = "val"
            metric = ev_a1
        else:
            ev_loss, ev_a1, ev_f1 = evaluate(model, original_test_dl, criterion, device)
            split = "test"
            metric = ev_a1

        append_csv(metrics_csv, [
            epoch + 1, f"{lr_now:.8f}",
            f"{tr_loss:.6f}", f"{tr_a1:.6f}", f"{tr_f1:.6f}",
            f"{ev_loss:.6f}", f"{ev_a1:.6f}", f"{ev_f1:.6f}", split,
        ])

        if metric > best:
            best = metric

        scheduler.step()

    # Final evaluation on the original test set (fair comparison)
    te_loss, te_a1, te_f1 = evaluate(model, original_test_dl, criterion, device)
    logger.info(
        f"Variant [{variant}] done — test acc {te_a1*100:.2f}%, "
        f"test F1 {te_f1*100:.2f}%, test loss {te_loss:.4f}"
    )
    plot_metrics(metrics_csv, run_dir)

    return {
        "variant": variant,
        "final_test_acc1": te_a1,
        "final_test_f1": te_f1,
        "final_test_loss": te_loss,
        "best_val_acc": best,
    }


# ---------------------------------------------------------------------------
# Step 4 — comparison table
# ---------------------------------------------------------------------------


def print_comparison(results: Dict[str, Dict[str, Any]], logger) -> None:
    """Pretty-print a comparison table of variant results."""
    logger.info(f"\n{'='*72}")
    logger.info("COMPARISON RESULTS")
    logger.info(f"{'='*72}")
    logger.info(f"{'Variant':<18} {'Accuracy':>10} {'F1':>10} {'Loss':>10}")
    logger.info(f"{'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    for variant in VARIANTS:
        r = results.get(variant)
        if r is None:
            logger.info(f"{variant:<18} {'FAILED':>10} {'—':>10} {'—':>10}")
        else:
            acc = f"{r['final_test_acc1']*100:.2f}%"
            f1 = f"{r['final_test_f1']*100:.2f}%"
            loss = f"{r['final_test_loss']:.4f}"
            logger.info(f"{variant:<18} {acc:>10} {f1:>10} {loss:>10}")
    logger.info(f"{'='*72}")

    # Also print to console (logger alone may go only to file)
    print(f"\n{'='*72}")
    print("COMPARISON RESULTS")
    print(f"{'='*72}")
    print(f"{'Variant':<18} {'Accuracy':>10} {'F1':>10} {'Loss':>10}")
    print(f"{'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    for variant in VARIANTS:
        r = results.get(variant)
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
    assert args.output_model not in args.input_models, (
        f"Output model '{args.output_model}' must differ from all input models."
    )

    # ── Setup ─────────────────────────────────────────────────────────────
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_root = Path.cwd() / "log"
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_root / f"comparison_{args.dataset}_{timestamp}.log"
    logger = get_logger(__name__, log_file=log_path, console=True)
    logger.info(f"Comparison pipeline | device={device} | args={json.dumps(vars(args), sort_keys=True)}")

    os.makedirs(args.out_dir, exist_ok=True)
    mean, std = get_normalization_params(args.dataset)
    input_size = get_default_input_size(args.dataset)
    num_classes = get_num_classes(args.dataset)

    # ── Step 1: train input models ────────────────────────────────────────
    trained_models = train_input_models(args, logger)

    # ── Step 2: build eval-mode dataloaders (no augmentation) for CAM ─────
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

    # ── Step 3: generate and save four dataset variants ───────────────────
    variant_root = Path(args.data_dir) / f"comparison_{args.dataset}"
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
            low_thr=args.low_threshold,
            high_thr=args.high_threshold,
            seed=args.seed,
            logger=logger,
        )

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

    # ── Step 5: train output model on each variant ────────────────────────
    comparison_results: Dict[str, Dict[str, Any]] = {}
    for variant in VARIANTS:
        try:
            result = train_output_on_variant(
                args, variant, variant_root, orig_test_dl,
                num_classes, input_size, device, logger,
            )
            comparison_results[variant] = result
        except Exception as exc:
            logger.error(f"Training on variant '{variant}' failed: {exc}")
            logger.error(traceback.format_exc())
            comparison_results[variant] = None

    # ── Step 6: save and display comparison ───────────────────────────────
    results_path = os.path.join(args.out_dir, "comparison_results.json")
    write_json(results_path, comparison_results)
    logger.info(f"\nResults saved to {results_path}")

    print_comparison(comparison_results, logger)


if __name__ == "__main__":
    main()
