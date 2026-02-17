import os
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

from dataloader import cifar100_loaders
from models import resnet18_cifar100
from engine import train_one_epoch, evaluate
from utils import set_seed, save_ckpt

def build_parser():
    p=argparse.ArgumentParser("CIFAR-100 ResNet-18")
    p.add_argument("--data_dir",    type=str,   default="./data")
    p.add_argument("--out_dir",     type=str,   default="./runs_cifar100_resnet18")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--lr",          type=float, default=0.1)
    p.add_argument("--momentum",    type=float, default=0.9)
    p.add_argument("--weight_decay",type=float, default=5e-4)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--log_every",   type=int,   default=100)
    p.add_argument("--amp", action="store_true", default=False)
    # new configurables
    p.add_argument("--run_name",    type=str,   default="")
    p.add_argument("--val_split",   type=float, default=0.0)
    p.add_argument("--dropout",     type=float, default=0.0)
    return p

def build_optimizer(args, model):
    return optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

def make_run_dir(out_dir, run_name):
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = run_name.strip() or f"run_{ts}"
    run_dir = os.path.join(out_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def write_json(path, obj):
    with open(path, "w") as f: json.dump(obj, f, indent=2, sort_keys=True)

def append_csv(path,row,header=None):
    exists = os.path.exists(path)
    with open(path, "a") as f:
        if (not exists) and header: f.write(",".join(header) + "\n")
        if row:  # Only write row if it's not empty
            f.write(",".join(str(x) for x in row)+"\n")

def plot_metrics(metrics_csv, run_dir):
    """Plot training and evaluation metrics from CSV file."""
    try:
        df = pd.read_csv(metrics_csv)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
        
        # Plot 1: Training Loss
        axes[0, 0].plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Train Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Evaluation Loss
        axes[0, 1].plot(df['epoch'], df['eval_loss'], 'r-', linewidth=2, label='Eval Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Evaluation Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: Training Accuracy (top-1)
        axes[1, 0].plot(df['epoch'], df['train_acc1'] * 100, 'b-', linewidth=2, label='Train Acc1')
        axes[1, 0].plot(df['epoch'], df['train_acc5'] * 100, 'b--', linewidth=1.5, label='Train Acc5')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Evaluation Accuracy (top-1)
        axes[1, 1].plot(df['epoch'], df['eval_acc1'] * 100, 'r-', linewidth=2, label='Eval Acc1')
        axes[1, 1].plot(df['epoch'], df['eval_acc5'] * 100, 'r--', linewidth=1.5, label='Eval Acc5')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Evaluation Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save the figure
        plot_path = os.path.join(run_dir, 'metrics_plot.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Metrics plot saved to {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting metrics: {e}")

def main():
    args = build_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:",device,"| cuda:",torch.cuda.is_available(),"| gpu:",torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    set_seed(args.seed)
    
    # saving results
    run_dir = make_run_dir(args.out_dir, args.run_name)
    write_json(os.path.join(run_dir, "config.json"), vars(args))


    train_dl, val_dl, test_dl = cifar100_loaders(
        args.data_dir, args.batch_size, args.num_workers,
        val_split=args.val_split, seed=args.seed
    )

    model = resnet18_cifar100(dropout = args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args, model)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[15, 25], gamma=0.1
    )

    scaler = torch.amp.GradScaler(enabled=(args.amp and device == "cuda"))

    best = 0.0
    best_path = os.path.join(run_dir, "best.pt")
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    header = ["epoch", "lr", "train_loss", "train_acc1", "train_acc5", "eval_loss", "eval_acc1", "eval_acc5", "eval_split"]
    append_csv(metrics_csv,[],header=header)  # write header once

    for epoch in range(args.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch+1}/{args.epochs} | lr {lr_now:.6f}")

        tr_loss, tr_a1, tr_a5 = train_one_epoch(
            model, train_dl, criterion, optimizer,
            scaler if scaler.is_enabled() else None,
            device, log_every=args.log_every
        )

        if val_dl is not None:
            ev_loss, ev_a1, ev_a5 = evaluate(model, val_dl, criterion, device)
            split = "val"
            metric = ev_a1
        else:
            ev_loss, ev_a1, ev_a5 = evaluate(model, test_dl, criterion, device)
            split = "test"
            metric = ev_a1

        print(f"Train: loss {tr_loss:.4f} acc1 {tr_a1*100:.2f}% acc5 {tr_a5*100:.2f}%")
        print(f"{split.title()}:   loss {ev_loss:.4f} acc1 {ev_a1*100:.2f}% acc5 {ev_a5*100:.2f}%")

        append_csv(metrics_csv,[epoch+1,f"{lr_now:.8f}",f"{tr_loss:.6f}",f"{tr_a1:.6f}",f"{tr_a5:.6f}",f"{ev_loss:.6f}",f"{ev_a1:.6f}",f"{ev_a5:.6f}",split])

        if metric > best:
            best = metric
            save_ckpt(best_path, model, optimizer, epoch+1, best, extra={"config":vars(args)})
            print(f"saved best: {best*100:.2f}%")

        scheduler.step()

    # final test with best
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    te_loss, te_a1, te_a5 = evaluate(model, test_dl, criterion, device)
    print(f"\nBest tracked ({'val' if val_dl is not None else 'test'}): {best*100:.2f}%")
    print(f"Final test: loss {te_loss:.4f} acc1 {te_a1*100:.2f}% acc5 {te_a5*100:.2f}%")
    
    # Generate plots
    plot_metrics(metrics_csv, run_dir)


if __name__ == "__main__":
    main()