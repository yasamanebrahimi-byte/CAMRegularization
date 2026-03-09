import random
import torch
from typing import Any, Dict, Optional

# Default configuration constants
DEFAULT_DATASET = "cifar100"
DEFAULT_MODEL = "resnet18"

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_training_context(
    params: Dict[str, Any],
    *,
    dataset: str,
    model: str,
    data_dir: Optional[str] = None,
    val_split: Optional[float] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    epochs: Optional[int] = None,
) -> Dict[str, Any]:
    resolved = dict(params)
    resolved["dataset"] = dataset
    resolved["model"] = model

    if data_dir is not None:
        resolved["data_dir"] = data_dir
    if val_split is not None:
        resolved["val_split"] = val_split
    if batch_size is not None:
        resolved["batch_size"] = batch_size
    if num_workers is not None:
        resolved["num_workers"] = num_workers
    if epochs is not None:
        resolved["epochs"] = epochs

    return resolved


def infer_input_size_from_loader(loader, fallback_size: int) -> int:
    try:
        sample_batch, _ = next(iter(loader))
    except Exception:
        return fallback_size

    if sample_batch.ndim < 4:
        return fallback_size

    inferred_h = int(sample_batch.shape[-2])
    inferred_w = int(sample_batch.shape[-1])
    return min(inferred_h, inferred_w)

@torch.no_grad()
def accuracy_top1(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()


@torch.no_grad()
def macro_f1_from_confusion(confusion_matrix: torch.Tensor) -> float:
    """Compute macro F1 from a [C, C] confusion matrix."""
    conf = confusion_matrix.float()
    if conf.numel() == 0:
        return 0.0

    tp = conf.diag()
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    denom = (2 * tp + fp + fn).clamp_min(1e-12)
    f1_per_class = (2 * tp) / denom
    return f1_per_class.mean().item()