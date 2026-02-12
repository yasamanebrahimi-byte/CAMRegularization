import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import cifar100_loaders
from models import resnet18_cifar100
from engine import train_one_epoch, evaluate
from utils import set_seed, save_ckpt


class CFG:
    data_dir = "./data"
    out_dir = "./runs_cifar100_resnet18"
    epochs = 2
    batch_size = 128
    num_workers = 4
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    seed = 42
    log_every = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = True


def main():
    cfg = CFG()
    print("device:", cfg.device, "| cuda:", torch.cuda.is_available(), "| gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    set_seed(cfg.seed)

    train_dl, test_dl = cifar100_loaders(
        cfg.data_dir, cfg.batch_size, cfg.num_workers
    )

    model = resnet18_cifar100().to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[15, 25], gamma=0.1
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and cfg.device == "cuda"))

    best_acc = 0.0
    os.makedirs(cfg.out_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_dl, criterion, optimizer,
            scaler if scaler.is_enabled() else None,
            cfg.device, cfg.log_every
        )

        test_loss, test_acc = evaluate(
            model, test_dl, criterion, cfg.device
        )

        scheduler.step()

        print(f"Train acc: {train_acc*100:.2f}%")
        print(f"Test  acc: {test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            save_ckpt(
                os.path.join(cfg.out_dir, "best.pt"),
                model, optimizer, epoch, best_acc
            )

    print(f"\nBest test accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
