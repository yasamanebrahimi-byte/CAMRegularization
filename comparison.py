import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    infer_num_classes_from_loader,
)
from IOutils import (
    build_args_from_params,
    write_json,
    namespace_to_train_params,
    init_run_dir_with_config,
    add_data_loading_args,
    add_training_hparam_args,
    build_time_tags,
)
from graphics import plot_variant_validation_comparison
from logger import get_logger
from train import train_with_config
from utils import set_seed, denormalize_tensor, tensor_to_pil_image

VARIANTS = ["original", "low_saliency"]
HEADER_WIDTH = 60
TABLE_WIDTH = 72
METRICS_BATCH_SIZE = 256

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.json")
TrainedModel = Tuple[str, nn.Module]


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


def _log_resolved_hparams(logger, scope: str, model_name: str, resolved: argparse.Namespace) -> None:
    """Log resolved hyperparameters in a consistent format."""
    logger.info(f"\n{'=' * HEADER_WIDTH}")
    logger.info(f"Training {scope} model: {model_name}")
    logger.info(
        f"  Resolved config → lr={resolved.lr}, wd={resolved.weight_decay}, "
        f"bs={resolved.batch_size}, epochs={resolved.epochs}, "
        f"scheduler={resolved.scheduler}, warmup={resolved.warmup_epochs}"
    )
    logger.info(f"{'=' * HEADER_WIDTH}")


def train_input_models(
    cli_args, config: Dict[str, Any], logger
) -> List[TrainedModel]:
    """Train each input model on the dataset and return (name, model) pairs."""
    trained: List[TrainedModel] = []

    for model_name in cli_args.input_models:
        resolved = _resolve_training_args(cli_args, config, model_name, cli_args.dataset)
        _log_resolved_hparams(logger, scope="input", model_name=model_name, resolved=resolved)

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
            f"test loss {result['final_test_loss']:.4f}"
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
    cam_runners: List[HiResCAM],
) -> torch.Tensor:
    """Compute per-image merged HiResCAM across all input models.

    Returns merged CAM: [B, H, W] in [0, 1].
    """
    merged = None
    for cam_runner in cam_runners:
        # HiResCAM.cam needs gradients enabled
        with torch.enable_grad():
            x = images.detach().clone().requires_grad_(True)
            cam = cam_runner.cam(x)  # [B, 1, H, W]
        cam = cam[:, 0].detach()  # [B, H, W]

        if merged is None:
            merged = cam
        else:
            merged = torch.max(merged, cam)  # OR-style merge

    return merged  # [B, H, W] in [0, 1]


def _build_cam_runners(trained_models: List[TrainedModel], cam_layer: str) -> List[HiResCAM]:
    """Build HiResCAM runners once and reuse them across batches."""
    cam_runners: List[HiResCAM] = []
    for _, model in trained_models:
        model.eval()
        _, target_module = resolve_cam_target_module(model, cam_layer)
        cam_runners.append(HiResCAM(model, target_module))
    return cam_runners


def _close_cam_runners(cam_runners: List[HiResCAM]) -> None:
    """Release hooks/resources held by HiResCAM runners."""
    for cam_runner in cam_runners:
        cam_runner.close()


def generate_and_save_variants(
    trained_models: List[TrainedModel],
    dataloader: DataLoader,
    split: str,
    variant_root: Path,
    class_names: List[str],
    mean: Tuple,
    std: Tuple,
    device: torch.device,
    cam_layer: str,
    threshold: float,
    logger,
) -> None:
    """Generate and save two dataset variants for one split (train or test)."""
    logger.info(f"Generating {split} variants → {variant_root}")

    # Create output directories
    for variant in VARIANTS:
        for cls_name in class_names:
            (variant_root / variant / split / cls_name).mkdir(parents=True, exist_ok=True)

    global_idx = 0
    cam_runners = _build_cam_runners(trained_models, cam_layer)
    try:
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            merged_cam = _compute_merged_cam_batch(images, cam_runners)

            # Denormalize for saving
            denorm = denormalize_tensor(images.cpu(), mean, std)
            merged_cam_cpu = merged_cam.cpu()

            B = images.size(0)
            for i in range(B):
                img_01 = denorm[i]  # [C, H, W] in ~[0,1]
                cam_i = merged_cam_cpu[i]  # [H, W] in [0,1]
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

                global_idx += 1

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                logger.info(f"  [{split}] processed {global_idx} images …")
    finally:
        _close_cam_runners(cam_runners)

    logger.info(f"  [{split}] saved {global_idx} images for all {len(VARIANTS)} variants")


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
    test_dl = DataLoader(test_ds, batch_size=METRICS_BATCH_SIZE, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    val_dl = (
        DataLoader(val_ds, batch_size=METRICS_BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
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
    logger,
) -> Dict[str, Any]:
    """Train output model on a variant train split and evaluate on the configured test split."""
    resolved = _resolve_training_args(cli_args, config, output_model, cli_args.dataset)

    _log_resolved_hparams(logger, scope="output", model_name=f"{output_model} [{variant}]", resolved=resolved)
    logger.info(f"  Validation during training uses variant split with val_split={resolved.val_split}")
    if variant == "low_saliency":
        logger.info("  Final test uses low_saliency variant test split")
    else:
        logger.info("  Final test remains on original test split for fair comparability")

    variant_dir = variant_root / variant
    train_dl, val_dl, variant_test_dl = _build_variant_loaders(
        variant_dir,
        resolved.dataset,
        resolved.batch_size,
        resolved.num_workers,
        resolved.val_split,
        resolved.seed,
    )

    # Evaluate low_saliency on its own test split; keep original variant on the shared original test split.
    eval_test_dl = variant_test_dl if variant == "low_saliency" else original_test_dl

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
        val_dl=val_dl,
        test_dl=eval_test_dl,
    )

    logger.info(
        f"Variant [{variant}] done — test acc {result['final_test_acc1']*100:.2f}%, "
        f"test loss {result['final_test_loss']:.4f}"
    )

    return {
        "output_model": output_model,
        "variant": variant,
        "final_test_acc1": result["final_test_acc1"],
        "final_test_loss": result["final_test_loss"],
        "best_val_acc": result["best_val_acc"],
    }


# ---------------------------------------------------------------------------
# Step 4 — comparison table
# ---------------------------------------------------------------------------


def _format_variant_row(variant: str, result: Optional[Dict[str, Any]]) -> str:
    if result is None:
        return f"{variant:<18} {'FAILED':>10} {'—':>10}"
    acc = f"{result['final_test_acc1'] * 100:.2f}%"
    loss = f"{result['final_test_loss']:.4f}"
    return f"{variant:<18} {acc:>10} {loss:>10}"


def _render_comparison_table_lines(output_model: str, model_results: Dict[str, Any]) -> List[str]:
    lines = [
        f"\n{'=' * TABLE_WIDTH}",
        f"COMPARISON RESULTS — output model: {output_model}",
        f"{'=' * TABLE_WIDTH}",
        f"{'Variant':<18} {'Accuracy':>10} {'Loss':>10}",
        f"{'-' * 18} {'-' * 10} {'-' * 10}",
    ]
    for variant in VARIANTS:
        lines.append(_format_variant_row(variant, model_results.get(variant)))
    lines.append(f"{'=' * TABLE_WIDTH}")
    return lines


def print_comparison(results: Dict[str, Dict[str, Dict[str, Any]]], logger) -> None:
    """Pretty-print per-output-model comparison tables of variant results."""
    for output_model, model_results in results.items():
        lines = _render_comparison_table_lines(output_model, model_results)
        for line in lines:
            logger.info(line)
            print(line)


def _generate_validation_comparison_plot(
    output_model: str,
    results_root: Path,
    input_models: List[str],
    dataset_name: str,
    logger,
) -> None:
    original_metrics_csv = results_root / output_model / "original" / "metrics.csv"
    low_saliency_metrics_csv = results_root / output_model / "low_saliency" / "metrics.csv"
    out_png = results_root / output_model / "validation_comparison_plot.png"

    if not original_metrics_csv.exists() or not low_saliency_metrics_csv.exists():
        logger.warning(
            f"Skipping validation comparison plot for {output_model}: "
            f"missing metrics CSV ({original_metrics_csv}, {low_saliency_metrics_csv})"
        )
        return

    written = plot_variant_validation_comparison(
        original_metrics_csv=str(original_metrics_csv),
        low_saliency_metrics_csv=str(low_saliency_metrics_csv),
        out_png=str(out_png),
        output_model_name=output_model,
        input_models=input_models,
        dataset_name=dataset_name,
    )
    if written:
        logger.info(f"Saved validation comparison plot: {out_png}")
    else:
        logger.warning(
            f"Skipped validation comparison plot for {output_model}: "
            "no validation rows found in one or both metrics files"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _build_parser().parse_args()
    tags = build_time_tags()

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
        base_out_dir = os.path.join("./runs", f"{args.dataset}_{tags['year_month']}", tags["day"])
    Path(base_out_dir).mkdir(parents=True, exist_ok=True)
    write_json(str(Path(base_out_dir) / "comparison_config.json"), vars(args))
    args.out_dir = base_out_dir

    timestamp = tags["timestamp"]
    log_path = Path(base_out_dir) / f"comparison_{args.dataset}_{timestamp}.log"
    logger = get_logger(__name__, log_file=log_path, console=False)
    logger.info(f"Comparison pipeline | device={device} | out_dir={base_out_dir} | args={json.dumps(vars(args), sort_keys=True)}")

    mean, std = get_normalization_params(args.dataset)
    input_size = get_default_input_size(args.dataset)

    # ── Variant output directory ────────────────────────────────────────────
    variant_root = Path(args.data_dir) / f"comparison_{args.dataset}_{tags['year_month']}" / tags["day"]

    # ── Step 1: train input models ────────────────────────────────────────
    trained_models = train_input_models(args, config, logger)

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

    # ── Step 3: generate and save dataset variants ────────────────────────
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
        batch_size=METRICS_BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

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
                    logger=logger,
                )
                comparison_results[output_model][variant] = result
            except Exception as exc:
                logger.error(f"Training output model '{output_model}' on variant '{variant}' failed: {exc}")
                logger.error(traceback.format_exc())
                comparison_results[output_model][variant] = None

        _generate_validation_comparison_plot(
            output_model=output_model,
            results_root=output_results_root,
            input_models=args.input_models,
            dataset_name=args.dataset,
            logger=logger,
        )

    # ── Step 6: save and display comparison ───────────────────────────────
    results_path = os.path.join(base_out_dir, "comparison_results.json")
    write_json(results_path, comparison_results)
    logger.info(f"\nResults saved to {results_path}")

    print_comparison(comparison_results, logger)


if __name__ == "__main__":
    main()
