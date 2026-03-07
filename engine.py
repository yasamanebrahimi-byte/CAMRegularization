import torch
import torch.nn as nn
import torch.optim as optim
from utils import accuracy_top1, macro_f1_from_confusion
from cam_masking import apply_random_cutout, apply_cam_cutout

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, log_every,
                    epoch=0, masking_cfg=None, cam_runner=None):
    model.train()
    running_loss, running_acc1 = 0.0, 0.0
    confusion = None

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        # Apply masking BEFORE random crop
        if masking_cfg is not None:
            ms = masking_cfg
            do_mask = (epoch >= ms["warmup_epochs"]) and (ms["strategy"] != "none")
            if do_mask and (torch.rand(1, device=device).item() < ms["prob"]):
                if ms["strategy"] == "random":
                    x = apply_random_cutout(x, area_frac=ms["area"], block=ms["block"], fill=0.0)
                else:
                    if cam_runner is None:
                        raise RuntimeError("CAM masking requested but cam_runner is None.")
                    was_training = model.training
                    model.eval()
                    x_cam = x.detach().requires_grad_(True)
                    cam = cam_runner.cam(x_cam)
                    model.train(was_training)
                    mode = "high" if ms["strategy"] == "cam_high" else "low"
                    x = apply_cam_cutout(x, cam.detach(), area_frac=ms["area"], block=ms["block"], fill=0.0, mode=mode)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(scaler is not None)):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        preds = logits.detach().argmax(dim=1)
        num_classes = int(logits.size(1))
        if confusion is None:
            confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
        encoded = y * num_classes + preds
        confusion += torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

        acc1 = accuracy_top1(logits.detach(), y)
        running_loss += loss.item()
        running_acc1 += acc1
        # Per-batch progress logging removed to keep console/log concise
    f1 = macro_f1_from_confusion(confusion) if confusion is not None else 0.0
    return running_loss / len(loader), running_acc1 / len(loader), f1

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, acc1_sum = 0.0, 0.0
    confusion = None
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=1)
        num_classes = int(logits.size(1))
        if confusion is None:
            confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
        encoded = y * num_classes + preds
        confusion += torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

        loss_sum += loss.item()
        acc1_sum += accuracy_top1(logits, y)
    f1 = macro_f1_from_confusion(confusion) if confusion is not None else 0.0
    return loss_sum / len(loader), acc1_sum / len(loader), f1


def warmup_model(
    model,
    train_loader,
    device,
    epochs,
    lr,
    momentum,
    weight_decay,
    label_smoothing,
    max_batches_per_epoch,
    logger=None,
):
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
        for x, y in train_loader:
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

        if logger is not None:
            avg_loss = running_loss / max(1, batch_count)
            logger.info(f"Warmup epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f}")