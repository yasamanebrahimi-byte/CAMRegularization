import time
import torch
from utils import accuracy_top1

# Train the model for one epoch.
# - model: nn.Module to train
# - loader: DataLoader yielding (input, target)
# - criterion: loss function
# - optimizer: optimizer for parameter updates
# - scaler: GradScaler or None (for mixed precision)
# - device: torch device string or torch.device
# - log_every: how many batches between printed logs
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, log_every):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    t0 = time.time()

    for i, (x, y) in enumerate(loader, start=1):
        # Move batch to device (non_blocking for pinned memory)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Use automatic mixed precision on CUDA when scaler is provided.
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(scaler is not None)):
            logits = model(x)
            loss = criterion(logits, y)

        # Backward + optimizer step (handling optional scaler)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Compute top-1 accuracy for the batch (helper returns fraction)
        acc = accuracy_top1(logits.detach(), y)
        running_loss += loss.item()
        running_acc += acc

        # Periodic logging of running averages and throughput
        if i % log_every == 0 or i == len(loader):
            dt = time.time() - t0
            print(
                f"[train] {i}/{len(loader)} "
                f"loss {running_loss/i:.4f} "
                f"acc {running_acc/i*100:.2f}% "
                f"{i/dt:.2f} it/s"
            )

    # Return average loss and accuracy over the loader
    return running_loss / len(loader), running_acc / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, acc_sum = 0.0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        loss_sum += loss.item()
        acc_sum += accuracy_top1(logits, y)

    return loss_sum / len(loader), acc_sum / len(loader)
