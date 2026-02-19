import random
import torch
import pandas as pd
from pathlib import Path
from logger import get_logger

logger = get_logger(__name__)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def accuracy_top1(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()


@torch.no_grad()
def accuracy_top5(logits, targets):
    """Compute top-5 accuracy."""
    _, top5_preds = logits.topk(5, dim=1)
    targets_expanded = targets.view(-1, 1).expand_as(top5_preds)
    return (top5_preds == targets_expanded).any(dim=1).float().mean().item()

def best_val_from_metrics(metrics_path: Path):
    """Return best validation acc1 from a run's metrics.csv, or None if unavailable."""
    try:
        df = pd.read_csv(metrics_path)
        if "eval_split" not in df.columns or "eval_acc1" not in df.columns: 
            return None
        df_val = df[df["eval_split"] == "val"]
        if df_val.empty: 
            return None
        # eval_acc1 should already be numeric; coerce just in case
        best = pd.to_numeric(df_val["eval_acc1"], errors="coerce").max()
        return None if pd.isna(best) else float(best)
    except Exception:
        return None
    
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def final_test_from_metrics(metrics_path: Path):
    """Return final test acc1 from a run's metrics.csv (last row eval_acc1), or None if unavailable."""
    try:
        df = pd.read_csv(metrics_path)
        if "eval_acc1" not in df.columns or df.empty:
            return None
        # Get the last row's eval_acc1 (final evaluation accuracy)
        acc = pd.to_numeric(df.iloc[-1]["eval_acc1"], errors="coerce")
        return None if pd.isna(acc) else float(acc)
    except Exception:
        return None