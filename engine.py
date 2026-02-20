import time
import torch
from utils import accuracy_top1, accuracy_top5
from logger import get_logger

logger = get_logger(__name__)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, log_every):
    model.train()
    running_loss, running_acc1, running_acc5 = 0.0, 0.0, 0.0
    t0 = time.time()

    for i, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
        acc1 = accuracy_top1(logits.detach(), y)
        acc5 = accuracy_top5(logits.detach(), y)
        running_loss += loss.item()
        running_acc1 += acc1
        running_acc5 += acc5
        if i % log_every == 0 or i == len(loader):
            dt = time.time() - t0
            logger.info(
                f"[train] {i}/{len(loader)} "
                f"loss {running_loss/i:.4f} "
                f"acc1 {running_acc1/i*100:.2f}% "
                f"acc5 {running_acc5/i*100:.2f}% "
                f"{i/dt:.2f} it/s"
            )
    return running_loss / len(loader), running_acc1 / len(loader), running_acc5 / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, acc1_sum, acc5_sum = 0.0, 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item()
        acc1_sum += accuracy_top1(logits, y)
        acc5_sum += accuracy_top5(logits, y)
    return loss_sum / len(loader), acc1_sum / len(loader), acc5_sum / len(loader)