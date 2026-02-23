import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple
from logger import get_logger
import os, torch
from dataset_registry import get_dataset_loaders
from model_registry import get_model
from cam_masking import GradCAM, apply_cam_cutout, apply_random_cutout

logger = get_logger(__name__)

MEAN = torch.tensor([0.5071, 0.4867, 0.4408]).view(1,3,1,1)
STD  = torch.tensor([0.2675, 0.2565, 0.2761]).view(1,3,1,1)

def unnormalize(x):
    return (x * STD.to(x.device)) + MEAN.to(x.device)

@torch.no_grad()
def to_img(x):  # [3,H,W] in 0..1
    x = x.clamp(0, 1).permute(1,2,0).cpu().numpy()
    return x

def save_one(mode, out_dir, x, y, model, cam_runner, area=0.3, block=8):
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device

    x0 = x.clone()
    if mode == "random":
        xm = apply_random_cutout(x, area_frac=area, block=block, fill=0.0)
        cam = None
    elif mode in ("cam_high", "cam_low"):
        model.eval()
        x_cam = x.detach().requires_grad_(True)
        cam = cam_runner.cam(x_cam, y)  # [B,1,H,W] normalized
        m = "high" if mode == "cam_high" else "low"
        xm = apply_cam_cutout(x, cam.detach(), area_frac=area, block=block, fill=0.0, mode=m)
    else:
        xm, cam = x, None

    # where pixels changed (any channel) => mask map
    diff = (xm != x0).any(dim=1, keepdim=True).float()  # [B,1,H,W]

    # unnormalize for viewing
    x0_vis = unnormalize(x0)
    xm_vis = unnormalize(xm)

    # save first 8 samples as a grid-like panel (matplotlib)
    B = min(8, x.size(0))
    fig, axes = plt.subplots(B, 4, figsize=(10, 2*B))
    if B == 1: axes = axes.reshape(1, -1)

    for i in range(B):
        axes[i,0].imshow(to_img(x0_vis[i]))
        axes[i,0].set_title("original"); axes[i,0].axis("off")

        if cam is not None:
            axes[i,1].imshow(cam[i,0].detach().cpu().numpy(), vmin=0, vmax=1)
            axes[i,1].set_title("CAM"); axes[i,1].axis("off")
        else:
            axes[i,1].axis("off"); axes[i,1].set_title("CAM (n/a)")

        axes[i,2].imshow(to_img(xm_vis[i]))
        axes[i,2].set_title("masked"); axes[i,2].axis("off")

        axes[i,3].imshow(diff[i,0].detach().cpu().numpy(), vmin=0, vmax=1)
        axes[i,3].set_title("where masked"); axes[i,3].axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, f"{mode}_panel.png")
    plt.savefig(path, dpi=150)
    plt.close()

def _setup_subplot(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

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
        successful = [r for r in results if r["status"] == "success"]
        if not successful:
            logger.info("No successful runs to plot")
            return
        run_names = []
        test_accs = []
        lr_values = []
        wd_values = []
        for r in successful:
            if "final_test_acc1" in r:
                run_names.append(r["run_name"])
                test_accs.append(r["final_test_acc1"] * 100)
                lr_values.append(r["params"]["lr"])
                wd_values.append(r["params"]["weight_decay"])
        if not test_accs:
            logger.info("No test accuracy data to plot")
            return
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
    ranked: List[Tuple[Dict[str, Any], float]] = []
    for r in successful:
        if "best_val_acc" in r:
            ranked.append((r, r["best_val_acc"]))
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
        rows = []
        for r, best_val in ranked:
            p = r["params"]
            rows.append(
                {
                    "run_name": r["run_name"],
                    "best_val_acc1": best_val,
                    "lr": p["lr"],
                    "epochs": p["epochs"],
                    "weight_decay": p["weight_decay"],
                    "momentum": p["momentum"],
                    "nesterov": p["nesterov"],
                    "label_smoothing": p["label_smoothing"],
                    "scheduler": p["scheduler"],
                    "warmup_epochs": p["warmup_epochs"],
                    "milestones": p.get("milestones", ""),
                }
            )
        out_csv = tuning_dir / "ranked_by_val.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        logger.info(f"\nSaved ranking to {out_csv}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dl, _, _ = get_dataset_loaders("cifar100", "./data", batch_size=128, num_workers=2, val_split=0.15, seed=42)
    x, y = next(iter(train_dl))
    x, y = x.to(device), y.to(device)

    model = get_model("resnet18", num_classes=100).to(device)

    # match your train.py layer selection (layer2 / layer3 / layer4)
    target_module = model.layer2
    cam_runner = GradCAM(model, target_module)

    out_dir = "./mask_images"
    for mode in ["none", "random", "cam_high", "cam_low"]:
        save_one(mode, out_dir, x, y, model, cam_runner, area=0.15, block=6)