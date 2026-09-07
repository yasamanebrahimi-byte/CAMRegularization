"""Reusable fold training loop for DaT Stage 1 and Stage 2."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dat_metrics import compute_binary_metrics
from dat_model import build_resnet18_3d
from dat_preprocessing import DatDataset
from utils import set_seed


def build_dat_model(config: dict[str, Any]) -> nn.Module:
    return build_resnet18_3d(
        num_classes=int(config.get("num_classes", 2)),
        n_input_channels=int(config.get("n_input_channels", 1)),
        dropout=float(config.get("dropout", 0.0)),
        base_channels=int(config.get("base_channels", 32)),
    )


def _build_optimizer(model: nn.Module, config: dict[str, Any]):
    name = str(config.get("optimizer", "adamw")).lower()
    lr = float(config.get("learning_rate", config.get("lr", 1e-3)))
    weight_decay = float(config.get("weight_decay", 1e-4))
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay, nesterov=bool(config.get("nesterov", True)),
        )
    if name == "adamw":
        betas = tuple(float(value) for value in config.get("adamw_betas", [0.9, 0.999]))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    raise ValueError(f"Unsupported optimizer '{name}'.")


def _build_scheduler(optimizer, config: dict[str, Any], epochs: int):
    scheduler_name = str(config.get("scheduler", "cosine")).lower()
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=float(config.get("min_lr", 1e-6))
        )
    if scheduler_name == "multistep":
        milestones = [int(value) for value in config.get("milestones", [60, 80])]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=float(config.get("gamma", 0.1)))
    raise ValueError(f"Unsupported scheduler '{scheduler_name}'.")


def _make_loader(dataset: Dataset, config: dict[str, Any], train: bool) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    kwargs = {"pin_memory": bool(torch.cuda.is_available())}
    if workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 2 if train else 4)),
        shuffle=train,
        num_workers=workers,
        **kwargs,
    )


def _set_dataset_epoch(dataset: Dataset, epoch: int) -> None:
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    if hasattr(dataset, "dataset"):
        _set_dataset_epoch(dataset.dataset, epoch)


def _run_training_epoch(model, loader, criterion, optimizer, device, scaler, max_batches: int | None = None):
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, targets)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        batch_size = int(targets.numel())
        loss_sum += float(loss.detach().item()) * batch_size
        correct += int((logits.detach().argmax(dim=1) == targets).sum().item())
        count += batch_size
    if count == 0:
        raise ValueError("Training loader produced no samples.")
    return loss_sum / count, correct / count


@torch.no_grad()
def evaluate_dat_model(model, loader, criterion, device, max_batches: int | None = None):
    model.eval()
    loss_sum = 0.0
    count = 0
    logits_list = []
    targets_list = []
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        logits = model(images)
        loss = criterion(logits, targets)
        batch_size = int(targets.numel())
        loss_sum += float(loss.item()) * batch_size
        count += batch_size
        logits_list.append(logits.detach().cpu())
        targets_list.append(targets.detach().cpu())
    if count == 0:
        raise ValueError("Evaluation loader produced no samples.")
    logits = torch.cat(logits_list).numpy()
    targets = torch.cat(targets_list).numpy()
    metrics = compute_binary_metrics(targets, logits=logits)
    metrics["loss"] = loss_sum / count
    return metrics, logits, targets


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fit_dat_model(
    train_dataset: Dataset,
    validation_dataset: Dataset | None,
    config: dict[str, Any],
    *,
    seed: int,
    run_dir: str | os.PathLike | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    """Fit one CV fold and select its checkpoint by validation log loss.

    Passing ``validation_dataset=None`` is supported for callers that need a
    fixed-budget full-data fit; it delegates to :func:`fit_dat_model_fixed_epochs`
    and never evaluates or selects a checkpoint on the training population.
    """
    if validation_dataset is None:
        return fit_dat_model_fixed_epochs(
            train_dataset, config, seed=seed, run_dir=run_dir,
            max_train_batches=max_train_batches,
        )
    set_seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_dat_model(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))
    # Validation loss is the ordinary probability log loss used by the
    # competition. Training-only label smoothing must not change it.
    evaluation_criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, config)
    epochs = int(config.get("epochs", 100))
    scheduler = _build_scheduler(optimizer, config, epochs)
    use_amp = bool(config.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    train_loader = _make_loader(train_dataset, config, train=True)
    validation_loader = _make_loader(validation_dataset, config, train=False)

    patience = int(config.get("patience", 15))
    best_log_loss = float("inf")
    best_state = None
    best_metrics = None
    best_logits = None
    best_targets = None
    best_epoch = None
    stale = 0
    rows = []
    for epoch in range(epochs):
        _set_dataset_epoch(train_dataset, epoch)
        train_loss, train_accuracy = _run_training_epoch(
            model, train_loader, criterion, optimizer, device, scaler, max_batches=max_train_batches
        )
        val_metrics, val_logits, val_targets = evaluate_dat_model(
            model, validation_loader, evaluation_criterion, device, max_batches=max_val_batches
        )
        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_log_loss": val_metrics["log_loss"],
            "val_auroc": val_metrics["auroc"],
            "val_brier_score": val_metrics["brier_score"],
            "val_ece": val_metrics["ece"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
        }
        rows.append(row)
        if val_metrics["log_loss"] < best_log_loss:
            best_log_loss = val_metrics["log_loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = dict(val_metrics)
            best_logits = val_logits.copy()
            best_targets = val_targets.copy()
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if patience >= 0 and stale >= patience:
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("DaT training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    if run_dir is not None:
        output = Path(run_dir)
        output.mkdir(parents=True, exist_ok=True)
        _write_metrics(output / "metrics.csv", rows)
        torch.save({"model_state_dict": best_state, "config": config}, output / "best_model.pt")
        with (output / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
        from dat_metrics import plot_dat_metrics
        plot_dat_metrics(output / "metrics.csv", output / "metrics_plot.png")
    return {
        "model": model,
        "best_state_dict": best_state,
        "best_metrics": best_metrics,
        "best_logits": best_logits,
        "best_targets": best_targets,
        "best_epoch": int(best_epoch),
        "checkpoint_selection": "minimum_validation_log_loss",
        "training_truncated": bool(max_train_batches),
        "validation_truncated": bool(max_val_batches),
        "research_valid": not bool(max_train_batches or max_val_batches or config.get("debug", False)),
        "epochs_completed": len(rows),
        "metrics_rows": rows,
    }


def fit_dat_model_fixed_epochs(
    train_dataset: Dataset,
    config: dict[str, Any],
    *,
    seed: int,
    run_dir: str | os.PathLike | None = None,
    epochs: int | None = None,
    max_train_batches: int | None = None,
) -> dict[str, Any]:
    """Train on a dataset for exactly ``epochs`` and return its final state.

    This is intentionally validation-free.  It is used for the full-data
    competition models and for leakage-safe Stage 2 fold teachers, where the
    epoch budget has already been determined by Stage 1 CV.
    """
    set_seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_dat_model(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))
    optimizer = _build_optimizer(model, config)
    epochs = int(config.get("epochs", 100) if epochs is None else epochs)
    if epochs <= 0:
        raise ValueError("The fixed epoch budget must be positive.")
    scheduler = _build_scheduler(optimizer, config, epochs)
    use_amp = bool(config.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    train_loader = _make_loader(train_dataset, config, train=True)
    rows = []
    for epoch in range(epochs):
        _set_dataset_epoch(train_dataset, epoch)
        train_loss, train_accuracy = _run_training_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            max_batches=max_train_batches,
        )
        rows.append({
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            # Kept as explicit NaNs so the file cannot be mistaken for a
            # validation trajectory or used for checkpoint selection.
            "val_loss": float("nan"), "val_accuracy": float("nan"),
            "val_log_loss": float("nan"), "val_auroc": float("nan"),
            "val_brier_score": float("nan"), "val_ece": float("nan"),
            "val_sensitivity": float("nan"), "val_specificity": float("nan"),
        })
        scheduler.step()
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(final_state)
    if run_dir is not None:
        output = Path(run_dir)
        output.mkdir(parents=True, exist_ok=True)
        _write_metrics(output / "metrics.csv", rows)
        payload = {"model_state_dict": final_state, "config": config, "checkpoint_selection": "final_scheduled_epoch", "final_epoch": epochs}
        torch.save(payload, output / "final_model.pt")
        # Backward-compatible filename.  It is an alias by meaning only: it
        # contains the scheduled final epoch, never a training-set-selected
        # best checkpoint.
        torch.save(payload, output / "best_model.pt")
        with (output / "config.json").open("w", encoding="utf-8") as handle:
            json.dump({**config, "checkpoint_selection": "final_scheduled_epoch", "final_epoch": epochs,
                       "training_truncated": bool(max_train_batches),
                       "validation_truncated": False,
                       "research_valid": not bool(max_train_batches or config.get("debug", False)),
                       "completed": True}, handle, indent=2, sort_keys=True)
        from dat_metrics import plot_dat_metrics
        plot_dat_metrics(output / "metrics.csv", output / "metrics_plot.png")
    return {
        "model": model,
        "final_state_dict": final_state,
        "best_state_dict": final_state,
        "final_epoch": epochs,
        "epochs_completed": epochs,
        "metrics_rows": rows,
        "checkpoint_selection": "final_scheduled_epoch",
        "training_truncated": bool(max_train_batches),
        "validation_truncated": False,
        "research_valid": not bool(max_train_batches or config.get("debug", False)),
    }
