import os
import random
import torch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy_top1(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()


@torch.no_grad()
def accuracy_top5(logits, targets):
    """Compute top-5 accuracy."""
    _, top5_preds = logits.topk(5, dim=1)
    targets_expanded = targets.view(-1, 1).expand_as(top5_preds)
    return (top5_preds == targets_expanded).any(dim=1).float().mean().item()


def save_ckpt(path, model, optimizer, epoch, best_acc, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
    }
    if extra is not None:
        state.update(extra)
    torch.save(state, path)