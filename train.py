import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from model_registry import get_model
from dataset_registry import (
    get_dataset_loaders,
    get_num_classes,
    get_default_input_size,
    infer_num_classes_from_loader,
)
from engine import train_one_epoch, evaluate
from utils import set_seed, infer_input_size_from_loader
from IOutils import build_parser, append_csv, init_run_dir_with_config
from graphics import plot_metrics
from logger import get_logger, SimpleLogger
import time
from pathlib import Path

def build_optimizer(args, model):
    return optim.SGD(
        model.parameters(),
        lr = args.lr,
        momentum = args.momentum,
        weight_decay = args.weight_decay,
        nesterov = args.nesterov
    )

def train_with_config(args, run_dir=None, logger=None, return_model=False):
    logger = logger or SimpleLogger()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device} | cuda: {torch.cuda.is_available()} | gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")

    effective_batch_size = int(args.batch_size)
    default_input_size = get_default_input_size(args.dataset)
    
    # Load dataset using registry
    dataset_kwargs = {
        "val_split": args.val_split,
        "seed": args.seed,
    }

    train_dl, val_dl, test_dl = get_dataset_loaders(
        args.dataset, args.data_dir, effective_batch_size, args.num_workers,
        **dataset_kwargs,
    )
    
    inferred_num_classes = infer_num_classes_from_loader(train_dl)
    num_classes = inferred_num_classes if inferred_num_classes is not None else get_num_classes(args.dataset)
    input_size = infer_input_size_from_loader(train_dl, default_input_size)
    model = get_model(args.model, num_classes=num_classes, input_size=input_size).to(device)

    logger.info(f"Model: {args.model} | Dataset: {args.dataset} | Classes: {num_classes}")
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
    metrics_csv = None
    if run_dir is not None:
        metrics_csv = os.path.join(run_dir, "metrics.csv")
        header = ["epoch", "lr", "train_loss", "train_acc1", "train_f1", "eval_loss", "eval_acc1", "eval_f1", "eval_split"]
        append_csv(metrics_csv, [], header=header, mode="w")  # write mode to replace existing file

    for epoch in range(args.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(f"\nEpoch {epoch+1}/{args.epochs} | lr {lr_now:.6f}")

        tr_loss, tr_a1, tr_f1 = train_one_epoch(
            model, train_dl, criterion, optimizer,
            scaler if scaler.is_enabled() else None,
            device, log_every=args.log_every,
        )

        if val_dl is not None:
            ev_loss, ev_a1, ev_f1 = evaluate(model, val_dl, criterion, device)
            split = "val"
            metric = ev_a1
        else:
            ev_loss, ev_a1, ev_f1 = evaluate(model, test_dl, criterion, device)
            split = "test"
            metric = ev_a1

        logger.info(f"Train: loss {tr_loss:.4f} acc1 {tr_a1*100:.2f}% f1 {tr_f1*100:.2f}%")
        logger.info(f"{split.title()}:   loss {ev_loss:.4f} acc1 {ev_a1*100:.2f}% f1 {ev_f1*100:.2f}%")

        if metrics_csv is not None:
            append_csv(metrics_csv,[epoch+1,f"{lr_now:.8f}",f"{tr_loss:.6f}",f"{tr_a1:.6f}",f"{tr_f1:.6f}",f"{ev_loss:.6f}",f"{ev_a1:.6f}",f"{ev_f1:.6f}",split])

        if metric > best:
            best = metric
            logger.info(f"saved best: {best*100:.2f}%")

        scheduler.step()

    # final test with best
    final_test_acc1 = None
    final_test_f1 = None
    final_test_loss = None
    model.to(device)
    te_loss, te_a1, te_f1 = evaluate(model, test_dl, criterion, device)
    final_test_acc1 = te_a1
    final_test_f1 = te_f1
    final_test_loss = te_loss
    logger.info(f"\nBest tracked ({'val' if val_dl is not None else 'test'}): {best*100:.2f}%")
    logger.info(f"Final test: loss {te_loss:.4f} acc1 {te_a1*100:.2f}% f1 {te_f1*100:.2f}%")
    
    # Print final results to console
    print(f"\nBest tracked ({'val' if val_dl is not None else 'test'}): {best*100:.2f}%")
    print(f"Final test: loss {te_loss:.4f} acc1 {te_a1*100:.2f}% f1 {te_f1*100:.2f}%")
    
    # Generate plots if we saved metrics
    if metrics_csv is not None and run_dir is not None:
        plot_metrics(metrics_csv, run_dir)
    
    result = {
        "final_test_acc1": final_test_acc1,
        "final_test_f1": final_test_f1,
        "best_val_acc": best,
        "final_test_loss": final_test_loss
    }
    if return_model:
        return result, model
    return result


def main():
    args = build_parser().parse_args()

    # saving results (creates run dir) and setup per-run logging
    run_dir = init_run_dir_with_config(args.out_dir, args.run_name, vars(args))

    # create centralized log directory and a unique log file for this run
    log_root = Path.cwd() / "log"
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name_for_log = Path(run_dir).name.replace(",", "-")
    log_path = log_root / f"{run_name_for_log}_{timestamp}.log"
    logger = get_logger(__name__, log_file=log_path, console=False)

    logger.info(f"Run parameters: {json.dumps(vars(args), sort_keys=True)}")
    
    # Train and return metrics
    train_with_config(args, run_dir=run_dir, logger=logger)

if __name__ == "__main__":
    main()