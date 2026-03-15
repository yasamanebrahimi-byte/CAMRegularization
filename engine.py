import torch
from utils import accuracy_top1, macro_precision_recall_f1_from_confusion, update_confusion_matrix

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, log_every):
    model.train()
    running_loss, running_acc1 = 0.0, 0.0
    confusion = None

    for x, y in loader:
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

        preds = logits.detach().argmax(dim=1)
        num_classes = int(logits.size(1))
        confusion = update_confusion_matrix(confusion, y, preds, num_classes=num_classes)

        acc1 = accuracy_top1(logits.detach(), y)
        running_loss += loss.item()
        running_acc1 += acc1
    precision, recall, f1 = macro_precision_recall_f1_from_confusion(confusion) if confusion is not None else (0.0, 0.0, 0.0)
    return running_loss / len(loader), running_acc1 / len(loader), precision, recall, f1

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
        confusion = update_confusion_matrix(confusion, y, preds, num_classes=num_classes)

        loss_sum += loss.item()
        acc1_sum += accuracy_top1(logits, y)
    precision, recall, f1 = macro_precision_recall_f1_from_confusion(confusion) if confusion is not None else (0.0, 0.0, 0.0)
    return loss_sum / len(loader), acc1_sum / len(loader), precision, recall, f1