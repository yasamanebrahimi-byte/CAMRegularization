import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import *
from models import *
from engine import *
from utils import *
from IOutils import *

def build_optimizer(args, model):
    return optim.SGD(
        model.parameters(),
        lr = args.lr,
        momentum = args.momentum,
        weight_decay = args.weight_decay,
        nesterov = args.nesterov
    )

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

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(args, model)

    # scheduler
    if args.scheduler == "multistep":
        ms=[int(x) for x in args.milestones.split(",") if x.strip()]
        main_sched = optim.lr_scheduler.MultiStepLR(optimizer,milestones=ms,gamma=args.gamma)
    else:
        main_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,args.epochs-args.warmup_epochs),eta_min=args.min_lr)

    if args.warmup_epochs > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(optimizer,start_factor=1e-3,total_iters=args.warmup_epochs)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer,schedulers=[warmup_sched,main_sched],milestones=[args.warmup_epochs])
    else:
        scheduler = main_sched


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