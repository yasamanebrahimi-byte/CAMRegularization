import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple
from logger import get_logger
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_registry import get_dataset_loaders, get_num_classes
from model_registry import get_model
from cam_masking import GradCAM, apply_cam_cutout, apply_random_cutout
from utils import set_seed

logger = get_logger(__name__)

MEAN = torch.tensor([0.5071, 0.4867, 0.4408]).view(1,3,1,1)
STD  = torch.tensor([0.2675, 0.2565, 0.2761]).view(1,3,1,1)

def unnormalize(x):
    return (x * STD.to(x.device)) + MEAN.to(x.device)

@torch.no_grad()
def to_img(x):  # [3,H,W] in 0..1
    x = x.clamp(0, 1).permute(1,2,0).cpu().numpy()
    return x


def _apply_mask(mode, x, model, cam_runner, area, block):
    if mode == "random":
        return apply_random_cutout(x, area_frac=area, block=block, fill=0.0), None
    if mode in ("cam_high", "cam_low"):
        model.eval()
        x_cam = x.detach().requires_grad_(True)
        cam = cam_runner.cam(x_cam)
        cam_mode = "high" if mode == "cam_high" else "low"
        xm = apply_cam_cutout(
            x,
            cam.detach(),
            area_frac=area,
            block=block,
            fill=0.0,
            mode=cam_mode,
        )
        return xm, cam
    return x, None


def _build_cam_masked(x0, xm, cam):
    if cam is None:
        return None
    eps = 1e-8
    mask2d = (xm - x0).abs().sum(dim=1) > eps
    cam_masked = cam[:, 0].detach().clone()
    cam_masked[mask2d] = 1.0
    return cam_masked


def _show_panel(ax, image, title, **imshow_kwargs):
    if image is None:
        ax.set_title(f"{title} (n/a)")
    else:
        ax.imshow(image, **imshow_kwargs)
        ax.set_title(title)
    ax.axis("off")

def save_one(mode, out_dir, x, model, cam_runner, area=0.3, block=8):
    os.makedirs(out_dir, exist_ok=True)

    x0 = x.clone()
    xm, cam = _apply_mask(mode, x, model, cam_runner, area, block)

    # unnormalize for viewing
    x0_vis = unnormalize(x0)
    xm_vis = unnormalize(xm)

    cam_masked = _build_cam_masked(x0, xm, cam)

    # save first 8 samples as a grid-like panel (matplotlib)
    B = min(8, x.size(0))
    
    # Determine number of columns based on mode
    num_cols = 2 if mode in ("random", "none") else 4
    fig, axes = plt.subplots(B, num_cols, figsize=(7 if num_cols == 2 else 13, 2 * B))
    if B == 1:
        axes = axes.reshape(1, -1)

    for i in range(B):
        _show_panel(axes[i, 0], to_img(x0_vis[i]), "original")
        _show_panel(axes[i, 1], to_img(xm_vis[i]), "masked")

        if num_cols == 4:
            cam_img = cam[i, 0].detach().cpu().numpy() if cam is not None else None
            cam_masked_img = cam_masked[i].detach().cpu().numpy() if cam_masked is not None else None
            _show_panel(axes[i, 2], cam_img, "CAM", vmin=0, vmax=1)
            _show_panel(axes[i, 3], cam_masked_img, "CAM w/ mask", vmin=0, vmax=1)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{mode}_panel.png")
    plt.savefig(path, dpi=150)
    logger.info(f"Saved {mode} mask visualization to {path}")
    plt.close()


def _setup_subplot(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def _build_parser():
    p = argparse.ArgumentParser("CAM Mask Visualization")
    p.add_argument("--dataset", type=str, default="cifar100")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preview_split", type=str, choices=["train", "test"], default="test")
    p.add_argument("--warmup_epochs", type=int, default=15)
    p.add_argument("--warmup_lr", type=float, default=0.1)
    p.add_argument("--warmup_momentum", type=float, default=0.9)
    p.add_argument("--warmup_weight_decay", type=float, default=5e-4)
    p.add_argument("--warmup_label_smoothing", type=float, default=0.0)
    p.add_argument("--max_batches_per_epoch", type=int, default=0,
                   help="Limit warmup batches per epoch for faster preview (0 = full epoch).")
    p.add_argument("--cam_layer", type=str, choices=["layer2", "layer3", "layer4"], default="layer3")
    p.add_argument("--out_dir", type=str, default="./mask_images")
    p.add_argument("--area", type=float, default=0.15)
    p.add_argument("--block", type=int, default=6)
    return p


def _warmup_model(model, train_dl, device, epochs, lr, momentum, weight_decay, label_smoothing, max_batches_per_epoch):
    if epochs <= 0:
        return

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=False,
    )

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1

            if max_batches_per_epoch > 0 and batch_count >= max_batches_per_epoch:
                break

        avg_loss = running_loss / max(1, batch_count)
        logger.info(f"Warmup epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f}")

def plot_metrics(metrics_csv, run_dir):
    try:
        df = pd.read_csv(metrics_csv)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Training & Evaluation Metrics", fontsize=16, fontweight="bold")
        ax = axes[0, 0]
        ax.plot(df["epoch"], df["train_loss"], "b-", linewidth=2, label="Train Loss")
        ax.plot(df["epoch"], df["eval_loss"],  "r-", linewidth=2, label="Eval Loss")
        _setup_subplot(ax, "Epoch", "Loss", "Loss (Train vs Eval)")
        ax = axes[0, 1]
        ax.plot(df["epoch"], df["train_acc1"] * 100, "b-", linewidth=2, label="Train Acc1")
        ax.plot(df["epoch"], df["eval_acc1"] * 100,  "r-", linewidth=2, label="Eval Acc1")
        _setup_subplot(ax, "Epoch", "Accuracy (%)", "Accuracy@1 (Train vs Eval)")
        ax = axes[1, 0]
        ax.plot(df["epoch"], df["train_acc5"] * 100, "b--", linewidth=2, label="Train Acc5")
        ax.plot(df["epoch"], df["eval_acc5"] * 100,  "r--", linewidth=2, label="Eval Acc5")
        _setup_subplot(ax, "Epoch", "Accuracy (%)", "Accuracy@5 (Train vs Eval)")
        ax = axes[1, 1]
        if {"train_loss", "eval_loss", "train_acc1", "eval_acc1"}.issubset(df.columns):
            ax.plot(df["epoch"], df["eval_loss"] - df["train_loss"], linewidth=2, label="Loss Gap (Eval-Train)")
            ax.plot(df["epoch"], (df["eval_acc1"] - df["train_acc1"]) * 100, linewidth=2, label="Acc1 Gap (Eval-Train)")
            _setup_subplot(ax, "Epoch", "Gap", "Generalization Gap")
        else:
            ax.axis("off")
        plt.tight_layout()
        plot_path = os.path.join(run_dir, f"metrics_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Metrics plot saved to {plot_path}")
        plt.close()

    except Exception as e:
        logger.error(f"Error plotting metrics: {e}")


def plot_tuning_results(results, tuning_dir):
    try:
        points = [
            (
                r["run_name"],
                r["final_test_acc1"] * 100,
                r["params"]["lr"],
                r["params"]["weight_decay"],
            )
            for r in results
            if r.get("status") == "success" and "final_test_acc1" in r
        ]
        if not points:
            logger.info("No successful runs to plot")
            return
        run_names, test_accs, lr_values, wd_values = map(list, zip(*points))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Hyperparameter Tuning Results', fontsize=14, fontweight='bold')
        x_pos = range(len(run_names))
        axes[0].bar(x_pos, test_accs, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Configuration')
        axes[0].set_ylabel('Test Accuracy (%)')
        axes[0].set_title('Final Test Accuracy by Configuration')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(range(1, len(run_names) + 1))
        axes[0].grid(True, alpha=0.3, axis='y')
        scatter = axes[1].scatter(lr_values, test_accs, c=wd_values, cmap='viridis', 
                                  s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
        axes[1].set_xlabel('Learning Rate')
        axes[1].set_ylabel('Test Accuracy (%)')
        axes[1].set_title('Test Accuracy vs Learning Rate')
        axes[1].grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes[1])
        cbar.set_label('Weight Decay')
        plt.tight_layout()
        plot_path = tuning_dir / 'tuning_results_plot.png'
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        logger.info(f"Tuning results plot saved to {plot_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting tuning results: {e}")


def print_summary(tuning_dir: Path, results: List[Dict[str, Any]]) -> None:
    logger.info("\n" + "=" * 70)
    logger.info("TUNING SUMMARY")
    logger.info("=" * 70)
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]
    other = [r for r in results if r.get("status") not in {"success", "failed"}]
    logger.info(f"Total runs: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Other: {len(other)}")
    best_test = max((r for r in successful if "final_test_acc1" in r), default=None, key=lambda r: r["final_test_acc1"])
    if best_test:
        acc = best_test["final_test_acc1"]
        p = best_test["params"]
        logger.info(f"\nBest test accuracy (for reference): {acc * 100:.2f}% ({best_test['run_name']})")
        logger.info(f"   lr={p['lr']}, epochs={p['epochs']}, wd={p['weight_decay']}, val_split={p['val_split']}")
    ranked: List[Tuple[Dict[str, Any], float]] = [
        (r, r["best_val_acc"]) for r in successful if "best_val_acc" in r
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    if ranked:
        logger.info("\nTop 10 runs by BEST val_acc1 (max over epochs):")
        for i, (r, best_val) in enumerate(ranked[:10], 1):
            p = r["params"]
            logger.info(f"  {i}. {r['run_name']}: best_val_acc1={best_val:.6f}")
            logger.info(
                "     "
                f"lr={p['lr']}, ep={p['epochs']}, wd={p['weight_decay']}, mom={p['momentum']}, "
                f"nest={p['nesterov']}, ls={p['label_smoothing']}, sch={p['scheduler']}, "
                f"wu={p['warmup_epochs']}, ms={p.get('milestones','')}"
            )
        rows = [
            {
                "run_name": r["run_name"],
                "best_val_acc1": best_val,
                "lr": r["params"]["lr"],
                "epochs": r["params"]["epochs"],
                "weight_decay": r["params"]["weight_decay"],
                "momentum": r["params"]["momentum"],
                "nesterov": r["params"]["nesterov"],
                "label_smoothing": r["params"]["label_smoothing"],
                "scheduler": r["params"]["scheduler"],
                "warmup_epochs": r["params"]["warmup_epochs"],
                "milestones": r["params"].get("milestones", ""),
            }
            for r, best_val in ranked
        ]
        out_csv = tuning_dir / "ranked_by_val.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        logger.info(f"\nSaved ranking to {out_csv}")


def main():
    args = _build_parser().parse_args()
    set_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dl, _, test_dl = get_dataset_loaders(
        args.dataset,
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )

    num_classes = get_num_classes(args.dataset)
    model = get_model(args.model, num_classes=num_classes).to(device)

    logger.info(
        f"Preparing mask preview with model={args.model}, dataset={args.dataset}, "
        f"warmup_epochs={args.warmup_epochs}"
    )

    _warmup_model(
        model=model,
        train_dl=train_dl,
        device=device,
        epochs=args.warmup_epochs,
        lr=args.warmup_lr,
        momentum=args.warmup_momentum,
        weight_decay=args.warmup_weight_decay,
        label_smoothing=args.warmup_label_smoothing,
        max_batches_per_epoch=args.max_batches_per_epoch,
    )

    preview_dl = train_dl if args.preview_split == "train" else test_dl
    x, _ = next(iter(preview_dl))
    x = x.to(device)

    target_module = getattr(model, args.cam_layer)
    cam_runner = GradCAM(model, target_module)

    out_dir = args.out_dir
    for mode in ["none", "random", "cam_high", "cam_low"]:
        save_one(mode, out_dir, x, model, cam_runner, area=args.area, block=args.block)


if __name__ == "__main__":
    main()