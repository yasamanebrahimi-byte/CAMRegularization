import os
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim

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


if __name__ == "__main__":
    main()