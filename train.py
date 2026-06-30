import os
import hashlib
import json
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader

from model_registry import get_model
from dataset_registry import (
    get_dataset_loaders,
    get_train_loader,
    get_num_classes,
    get_default_input_size,
    get_normalization_params,
    infer_num_classes_from_loader,
)
from engine import train_one_epoch, evaluate
from utils import set_seed, infer_input_size_from_loader
from IOutils import build_parser, append_csv, init_run_dir_with_config, build_time_tags
from graphics import plot_metrics
from logger import get_logger, SimpleLogger
from pathlib import Path
from cutout import CutoutAugmentedDataset


def _log(logger, level: str, msg: str, *args) -> None:
    log_fn = getattr(logger, level, None) or getattr(logger, "info", None)
    if log_fn is None:
        print(msg % args if args else msg)
        return
    try:
        log_fn(msg, *args)
    except TypeError:
        log_fn(msg % args if args else msg)


def build_optimizer(args, model):
    optimizer_name = str(getattr(args, "optimizer", "sgd")).lower()

    if optimizer_name == "adamw":
        betas = tuple(float(x) for x in getattr(args, "adamw_betas", [0.9, 0.999]))
        eps = float(getattr(args, "adamw_eps", 1e-8))
        return optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=betas,
            eps=eps,
            weight_decay=args.weight_decay,
        )

    if optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )

    raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
    return checkpoint



def _cam_cache_required(args) -> bool:
    cutout_mode = str(getattr(args, "cutout_mode", "none") or "none").lower()
    cutout_m = int(getattr(args, "cutout_m", 0))
    cam_precompute_only = bool(getattr(args, "cam_precompute_only", False))
    cam_precompute_windows = bool(getattr(args, "cam_precompute_windows", False))
    return cutout_mode in {"cam_low", "cam_high"} and (
        cutout_m > 0 or cam_precompute_only or cam_precompute_windows
    )


def _slug_cache_component(value) -> str:
    text = str(value or "unknown").strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _checkpoint_fingerprint(checkpoint_path: str) -> dict:
    checkpoint_path = str(checkpoint_path or "").strip()
    if not checkpoint_path:
        raise ValueError("Teacher checkpoint is required to resolve the CAM cache directory.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    digest = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    stat = os.stat(checkpoint_path)
    return {
        "path": os.path.abspath(checkpoint_path),
        "sha256": digest.hexdigest(),
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
    }


def _resolve_cam_cache_dir(args, logger=None):
    if not _cam_cache_required(args):
        return None, None

    checkpoint_info = getattr(args, "_cam_checkpoint_fingerprint", None)
    if checkpoint_info is None:
        checkpoint_info = _checkpoint_fingerprint(getattr(args, "teacher_checkpoint", ""))
        setattr(args, "_cam_checkpoint_fingerprint", checkpoint_info)
        setattr(args, "teacher_checkpoint_sha256", checkpoint_info["sha256"])

    cache_dir = str(getattr(args, "cam_cache_dir", "") or "").strip()
    if not cache_dir:
        teacher_component = _slug_cache_component(getattr(args, "teacher_model", "") or getattr(args, "model", ""))
        cache_dir = os.path.join(
            str(getattr(args, "data_dir", "./data")),
            "cam_cache",
            _slug_cache_component(getattr(args, "dataset", "dataset")),
            teacher_component,
            checkpoint_info["sha256"][:16],
        )
        setattr(args, "cam_cache_dir", cache_dir)
    else:
        setattr(args, "cam_cache_dir", cache_dir)

    if logger is not None:
        _log(logger, "info", "CAM saliency cache directory: %s", cache_dir)
    return cache_dir, checkpoint_info


def _build_cam_cache_settings(args, checkpoint_info, input_size: int) -> dict:
    return {
        "dataset": getattr(args, "dataset", ""),
        "grayscale": bool(getattr(args, "grayscale", False)),
        "student_model": getattr(args, "model", ""),
        "teacher_model": getattr(args, "teacher_model", ""),
        "teacher_checkpoint": checkpoint_info,
        "cam_layer": str(getattr(args, "cam_layer", "auto") or "auto"),
        "input_size": int(input_size),
        "deterministic_train_transforms": bool(getattr(args, "deterministic_train_transforms", False)),
    }


def _dataset_kwargs_from_args(args) -> dict:
    return {
        "val_split": args.val_split,
        "seed": args.seed,
        "grayscale": getattr(args, "grayscale", False),
        "include_regex": getattr(args, "include_regex", ""),
        "deterministic_train_transforms": getattr(args, "deterministic_train_transforms", False),
    }


def _validate_cam_precompute_args(args) -> None:
    cutout_mode = str(getattr(args, "cutout_mode", "none") or "none").lower()
    option_name = "--cam_precompute_windows" if getattr(args, "cam_precompute_windows", False) else "--cam_precompute_only"
    if cutout_mode not in {"cam_low", "cam_high"}:
        raise ValueError(f"{option_name} is only valid with --cutout_mode cam_low or cam_high.")
    if not str(getattr(args, "teacher_checkpoint", "") or "").strip():
        raise ValueError(f"{option_name} requires --teacher_checkpoint.")
    if not str(getattr(args, "teacher_model", "") or "").strip():
        raise ValueError(f"{option_name} requires --teacher_model.")
    if not bool(getattr(args, "deterministic_train_transforms", False)):
        raise ValueError(f"{option_name} requires --deterministic_train_transforms.")

    if bool(getattr(args, "cam_precompute_windows", False)):
        cutout_m = int(getattr(args, "cutout_m", 0))
        if cutout_m <= 0:
            raise ValueError("--cam_precompute_windows requires --cutout_m > 0.")
        cutout_size = int(getattr(args, "cutout_size", 0))
        cutout_area = getattr(args, "cutout_area", None)
        if (cutout_size <= 0) and (cutout_area is None or float(cutout_area) <= 0.0):
            raise ValueError("--cam_precompute_windows requires --cutout_size or --cutout_area.")


def _load_teacher_model(args, num_classes: int, input_size: int, logger, device):
    logger = logger or SimpleLogger()
    if not args.teacher_model or not args.teacher_checkpoint:
        raise ValueError("Teacher model and checkpoint are required for cam_low/cam_high cutout.")
    if not os.path.isfile(args.teacher_checkpoint):
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_checkpoint}")

    model = get_model(args.teacher_model, num_classes=num_classes, input_size=input_size)
    checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    if state_dict is None:
        raise ValueError("Teacher checkpoint did not contain a valid state_dict.")

    cleaned = {}
    for key, value in state_dict.items():
        if isinstance(key, str) and key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        _log(logger, "warning", "Teacher checkpoint load: missing=%s unexpected=%s", missing, unexpected)
    model = model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


def _validate_cam_cutout_generation(cutout_ds: CutoutAugmentedDataset, cutout_mode: str, logger, max_samples: int = 3) -> None:
    if cutout_mode not in {"cam_low", "cam_high"}:
        return
    if not isinstance(cutout_ds, CutoutAugmentedDataset):
        raise RuntimeError("CAM cutout validation requires a CutoutAugmentedDataset.")
    if cutout_ds.cutout_m <= 0:
        return

    max_checks = max(1, int(max_samples))
    checked = 0
    try:
        for base_index in range(cutout_ds._base_len):
            for aug_index in range(1, cutout_ds.cutout_m + 1):
                dataset_index = base_index * (cutout_ds.cutout_m + 1) + aug_index
                cutout_ds[dataset_index]
                checked += 1
                if checked >= max_checks:
                    break
            if checked >= max_checks:
                break
    except Exception as exc:
        _log(logger, "error", "CAM cutout validation failed.")
        raise RuntimeError(f"CAM cutout validation failed for mode {cutout_mode}: {exc}") from exc

    if checked == 0:
        _log(logger, "error", "CAM cutout validation failed.")
        raise RuntimeError(f"CAM cutout validation failed for mode {cutout_mode}: no augmented samples were generated.")

    _log(logger, "info", "Validated CAM cutout generation for mode %s", cutout_mode)


def _maybe_wrap_cutout_loader(train_dl, args, teacher_model, mean, std, logger, input_size: int):
    cutout_mode = str(getattr(args, "cutout_mode", "none") or "none").lower()
    cutout_m = int(getattr(args, "cutout_m", 0))
    if cutout_mode == "none" or cutout_m <= 0:
        return train_dl

    cutout_size = int(getattr(args, "cutout_size", 0))
    cutout_area = getattr(args, "cutout_area", None)
    if (cutout_size <= 0) and (cutout_area is None or float(cutout_area) <= 0.0):
        raise ValueError("cutout_size or cutout_area must be provided when cutout_m > 0.")

    saliency_candidate_percent = float(getattr(args, "saliency_candidate_percent", 10.0))
    cam_layer = str(getattr(args, "cam_layer", "auto") or "auto")
    cam_cache_dir = None
    cam_cache_settings = None
    dataset_teacher_model = teacher_model
    if cutout_mode in {"cam_low", "cam_high"}:
        cam_cache_dir, checkpoint_info = _resolve_cam_cache_dir(args, logger=logger)
        cam_cache_settings = _build_cam_cache_settings(args, checkpoint_info, input_size=input_size)
        if int(getattr(train_dl, "num_workers", 0) or 0) > 0:
            dataset_teacher_model = None
            _log(
                logger,
                "info",
                "CAM cache-only mode for DataLoader workers; cache misses will raise. "
                "Run the same command once with --cam_precompute_only --num_workers 0 "
                "--deterministic_train_transforms before high-worker training.",
            )

    cutout_ds = CutoutAugmentedDataset(
        base_dataset=train_dl.dataset,
        cutout_mode=cutout_mode,
        cutout_m=cutout_m,
        cutout_size=cutout_size if cutout_size > 0 else None,
        cutout_area=cutout_area,
        mean=mean,
        std=std,
        seed=int(getattr(args, "seed", 0)),
        saliency_candidate_percent=saliency_candidate_percent,
        teacher_model=dataset_teacher_model,
        cam_layer=cam_layer,
        cam_cache_dir=cam_cache_dir,
        cam_cache_settings=cam_cache_settings,
        debug_cam_timing=bool(getattr(args, "debug_cam_timing", False)),
        logger=logger,
    )

    _log(
        logger,
        "info",
        "Cutout enabled: mode=%s m=%s size=%s area=%s",
        cutout_mode,
        cutout_m,
        cutout_size if cutout_size > 0 else None,
        cutout_area,
    )

    loader_kwargs = {
        "batch_size": train_dl.batch_size,
        "shuffle": True,
        "num_workers": train_dl.num_workers,
        "pin_memory": getattr(train_dl, "pin_memory", True),
        "drop_last": getattr(train_dl, "drop_last", False),
    }
    if int(getattr(train_dl, "num_workers", 0) or 0) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = getattr(train_dl, "prefetch_factor", 2) or 2

    wrapped_loader = DataLoader(cutout_ds, **loader_kwargs)
    if cutout_mode in {"cam_low", "cam_high"}:
        _validate_cam_cutout_generation(cutout_ds, cutout_mode, logger)
    return wrapped_loader


def precompute_cam_cache_with_config(args, run_dir=None, logger=None):
    logger = logger or SimpleLogger()
    _validate_cam_precompute_args(args)

    effective_args = deepcopy(args)
    if int(getattr(effective_args, "num_workers", 0) or 0) != 0:
        _log(logger, "info", "Forcing num_workers=0 for CAM cache precomputation.")
    effective_args.num_workers = 0

    cutout_mode = str(getattr(effective_args, "cutout_mode", "none") or "none").lower()
    cutout_m = int(getattr(effective_args, "cutout_m", 0))
    precompute_windows = bool(getattr(effective_args, "cam_precompute_windows", False))
    precompute_saliency = bool(getattr(effective_args, "cam_precompute_only", False)) or not precompute_windows
    if cutout_m < 0:
        raise ValueError("cutout_m must be >= 0.")

    set_seed(effective_args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(
        logger,
        "info",
        "CAM cache precompute device: %s | cuda: %s | gpu: %s",
        device,
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )

    effective_batch_size = int(effective_args.batch_size)
    default_input_size = get_default_input_size(effective_args.dataset)
    dataset_kwargs = _dataset_kwargs_from_args(effective_args)
    dataset_kwargs["deterministic_train_transforms"] = True

    train_dl = get_train_loader(
        effective_args.dataset,
        effective_args.data_dir,
        effective_batch_size,
        0,
        **dataset_kwargs,
    )

    inferred_num_classes = infer_num_classes_from_loader(train_dl)
    num_classes = inferred_num_classes if inferred_num_classes is not None else get_num_classes(effective_args.dataset)
    input_size = infer_input_size_from_loader(train_dl, default_input_size)

    teacher_model = _load_teacher_model(
        effective_args,
        num_classes=num_classes,
        input_size=input_size,
        logger=logger,
        device=device,
    )
    cam_cache_dir, checkpoint_info = _resolve_cam_cache_dir(effective_args, logger=logger)
    cam_cache_settings = _build_cam_cache_settings(effective_args, checkpoint_info, input_size=input_size)
    mean, std = get_normalization_params(effective_args.dataset)

    cutout_size = int(getattr(effective_args, "cutout_size", 0))
    cutout_area = getattr(effective_args, "cutout_area", None)
    saliency_candidate_percent = float(getattr(effective_args, "saliency_candidate_percent", 10.0))
    cam_layer = str(getattr(effective_args, "cam_layer", "auto") or "auto")

    cutout_ds = CutoutAugmentedDataset(
        base_dataset=train_dl.dataset,
        cutout_mode=cutout_mode,
        cutout_m=max(1, cutout_m),
        cutout_size=cutout_size if cutout_size > 0 else None,
        cutout_area=cutout_area,
        mean=mean,
        std=std,
        seed=int(getattr(effective_args, "seed", 0)),
        saliency_candidate_percent=saliency_candidate_percent,
        teacher_model=teacher_model,
        cam_layer=cam_layer,
        cam_cache_dir=cam_cache_dir,
        cam_cache_settings=cam_cache_settings,
        debug_cam_timing=bool(getattr(effective_args, "debug_cam_timing", False)),
        logger=logger,
    )

    total = len(cutout_ds.base_dataset)
    _log(
        logger,
        "info",
        "Precomputing CAM %s for %s training samples into %s.",
        "saliency and window caches" if precompute_windows and precompute_saliency else (
            "window cache" if precompute_windows else "saliency cache"
        ),
        total,
        cam_cache_dir,
    )

    total_windows = total * cutout_ds.cutout_m if precompute_windows else 0
    created_windows = 0
    cached_windows = 0
    for base_index in range(total):
        image, _target = cutout_ds.base_dataset[base_index]
        if not torch.is_tensor(image):
            raise ValueError("CAM cache precompute expects the training dataset to return tensors.")

        saliency = None
        if precompute_saliency:
            saliency = cutout_ds._get_cam_saliency(base_index, image)
        if precompute_windows:
            created, cached = cutout_ds.precompute_cam_windows_for_sample(base_index, image, saliency=saliency)
            created_windows += created
            cached_windows += cached

        completed = base_index + 1
        if completed == total or completed % 500 == 0:
            if precompute_windows:
                _log(
                    logger,
                    "info",
                    "CAM cache precompute progress: %s/%s samples | windows created=%s cached=%s/%s",
                    completed,
                    total,
                    created_windows,
                    cached_windows,
                    total_windows,
                )
            else:
                _log(logger, "info", "CAM cache precompute progress: %s/%s samples", completed, total)

    if precompute_windows:
        _log(
            logger,
            "info",
            "CAM cache precomputation complete: %s samples, %s/%s windows created, %s cached in %s",
            total,
            created_windows,
            total_windows,
            cached_windows,
            cam_cache_dir,
        )
    else:
        _log(logger, "info", "CAM cache precomputation complete: %s samples cached in %s", total, cam_cache_dir)
    return {
        "cache_dir": cam_cache_dir,
        "samples": total,
        "windows_created": created_windows,
        "windows_cached": cached_windows,
        "windows_total": total_windows,
    }


def train_with_config(
    args,
    run_dir=None,
    logger=None,
    return_model=False,
    train_dl=None,
    val_dl=None,
    test_dl=None,
):
    logger = logger or SimpleLogger()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device} | cuda: {torch.cuda.is_available()} | gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")

    effective_batch_size = int(args.batch_size)
    default_input_size = get_default_input_size(args.dataset)

    # Load dataset using registry unless explicit loaders are provided
    if train_dl is None or test_dl is None:
        dataset_kwargs = _dataset_kwargs_from_args(args)

        train_dl, val_dl, test_dl = get_dataset_loaders(
            args.dataset, args.data_dir, effective_batch_size, args.num_workers,
            **dataset_kwargs,
        )

    inferred_num_classes = infer_num_classes_from_loader(train_dl)
    num_classes = inferred_num_classes if inferred_num_classes is not None else get_num_classes(args.dataset)
    input_size = infer_input_size_from_loader(train_dl, default_input_size)
    model = get_model(args.model, num_classes=num_classes, input_size=input_size).to(device)

    logger.info(
        f"Model: {args.model} | Dataset: {args.dataset} | Classes: {num_classes} | "
        f"Optimizer: {getattr(args, 'optimizer', 'sgd')}"
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(args, model)

    cutout_mode = str(getattr(args, "cutout_mode", "none") or "none").lower()
    cutout_m = int(getattr(args, "cutout_m", 0))
    if cutout_m < 0:
        raise ValueError("cutout_m must be >= 0.")
    if cutout_mode not in {"none", "random", "cam_low", "cam_high"}:
        raise ValueError(f"Unsupported cutout_mode '{cutout_mode}'.")

    teacher_model = None
    if cutout_mode in {"cam_low", "cam_high"} and cutout_m > 0:
        teacher_model = _load_teacher_model(
            args,
            num_classes=num_classes,
            input_size=input_size,
            logger=logger,
            device=device,
        )
    if cutout_mode != "none" and cutout_m > 0:
        mean, std = get_normalization_params(args.dataset)
        train_dl = _maybe_wrap_cutout_loader(train_dl, args, teacher_model, mean, std, logger, input_size=input_size)

    # scheduler
    if args.scheduler == "multistep":
        ms = [int(x) for x in args.milestones.split(",") if x.strip()]
        main_sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms, gamma=args.gamma)
    else:
        main_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.min_lr)

    if args.warmup_epochs > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=args.warmup_epochs)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, main_sched], milestones=[args.warmup_epochs])
    else:
        scheduler = main_sched

    scaler = torch.amp.GradScaler(enabled=(args.amp and device == "cuda"))

    best = 0.0
    best_state_dict = None
    metrics_csv = None
    if run_dir is not None:
        metrics_csv = os.path.join(run_dir, "metrics.csv")
        header = [
            "epoch",
            "lr",
            "train_loss",
            "train_acc1",
            "eval_loss",
            "eval_acc1",
            "eval_split",
            "val_split",
        ]
        append_csv(metrics_csv, [], header=header, mode="w")  # write mode to replace existing file

    for epoch in range(args.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(f"\nEpoch {epoch+1}/{args.epochs} | lr {lr_now:.6f}")

        tr_loss, tr_a1 = train_one_epoch(
            model, train_dl, criterion, optimizer,
            scaler if scaler.is_enabled() else None,
            device, log_every=args.log_every,
        )

        if val_dl is not None:
            ev_loss, ev_a1 = evaluate(model, val_dl, criterion, device)
            split = "val"
            metric = ev_a1
        else:
            ev_loss, ev_a1 = evaluate(model, test_dl, criterion, device)
            split = "test"
            metric = ev_a1

        logger.info(f"Train: loss {tr_loss:.4f} acc1 {tr_a1*100:.2f}%")
        logger.info(f"{split.title()}:   loss {ev_loss:.4f} acc1 {ev_a1*100:.2f}%")

        if metrics_csv is not None:
            append_csv(
                metrics_csv,
                [
                    epoch + 1,
                    f"{lr_now:.8f}",
                    f"{tr_loss:.6f}",
                    f"{tr_a1:.6f}",
                    f"{ev_loss:.6f}",
                    f"{ev_a1:.6f}",
                    split,
                    str(args.val_split),
                ],
            )

        if best_state_dict is None or metric > best:
            best = metric
            best_state_dict = deepcopy(model.state_dict())
            logger.info(f"saved best: {best*100:.2f}%")

        scheduler.step()

    # final test with best
    final_test_acc1 = None
    final_test_loss = None
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info("Loaded best checkpoint weights for final test evaluation.")
        model.to(device)
        te_loss, te_a1 = evaluate(model, test_dl, criterion, device)
        final_test_acc1 = te_a1
        final_test_loss = te_loss
        logger.info(f"\nBest tracked ({'val' if val_dl is not None else 'test'}): {best*100:.2f}%")
        logger.info(f"Final test: loss {te_loss:.4f} acc1 {te_a1*100:.2f}%")

        if run_dir is not None:
            checkpoint_path = os.path.join(run_dir, "best_model.pt")
            torch.save(
                {
                    "model_state_dict": best_state_dict,
                    "model": args.model,
                    "dataset": args.dataset,
                    "num_classes": num_classes,
                    "input_size": input_size,
                    "best_tracked_acc": best,
                    "final_test_acc1": final_test_acc1,
                    "final_test_loss": final_test_loss,
                    "config": vars(args),
                },
                checkpoint_path,
            )
            logger.info(f"Saved best model checkpoint to {checkpoint_path}")

    # Print final results to console
    print(f"\nBest tracked ({'val' if val_dl is not None else 'test'}): {best*100:.2f}%")
    print(f"Final test: loss {te_loss:.4f} acc1 {te_a1*100:.2f}%")

    # Generate plots if we saved metrics
    if metrics_csv is not None and run_dir is not None:
        plot_metrics(
            metrics_csv,
            run_dir,
            model_name=getattr(args, "model", None),
            dataset_name=getattr(args, "dataset", None),
        )

    result = {
        "final_test_acc1": final_test_acc1,
        "best_val_acc": best,
        "final_test_loss": final_test_loss,
    }
    if return_model:
        return result, model
    return result


def main():
    args = build_parser().parse_args()
    precompute_requested = bool(getattr(args, "cam_precompute_only", False)) or bool(
        getattr(args, "cam_precompute_windows", False)
    )
    if precompute_requested:
        _validate_cam_precompute_args(args)
    if getattr(args, "cam_precompute_only", False):
        args.num_workers = 0
    _resolve_cam_cache_dir(args)

    # saving results (creates run dir) and setup per-run logging
    run_dir = init_run_dir_with_config(args.out_dir, args.run_name, vars(args))

    # create a unique log file inside this run directory
    timestamp = build_time_tags()["timestamp"]
    run_name_for_log = Path(run_dir).name.replace(",", "-")
    log_path = Path(run_dir) / f"{run_name_for_log}_{timestamp}.log"
    logger = get_logger(__name__, log_file=log_path, console=False)

    logger.info(f"Run parameters: {json.dumps(vars(args), sort_keys=True)}")

    if args.cam_precompute_only:
        precompute_cam_cache_with_config(args, run_dir=run_dir, logger=logger)
    else:
        if getattr(args, "cam_precompute_windows", False):
            precompute_cam_cache_with_config(args, run_dir=run_dir, logger=logger)
        train_with_config(args, run_dir=run_dir, logger=logger)


if __name__ == "__main__":
    main()
