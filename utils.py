import random
import torch
from PIL import Image
from typing import Optional, Tuple

# Default configuration constants
DEFAULT_DATASET = "cifar100"
DEFAULT_MODEL = "resnet18"

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def denormalize_tensor(tensor: torch.Tensor, mean: Tuple, std: Tuple) -> torch.Tensor:
    m = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype)
    s = torch.tensor(std, device=tensor.device, dtype=tensor.dtype)
    if tensor.ndim == 4:
        m = m[None, :, None, None]
        s = s[None, :, None, None]
    elif tensor.ndim == 3:
        m = m[:, None, None]
        s = s[:, None, None]
    return tensor * s + m


def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, "RGB")

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


def update_confusion_matrix(
    confusion_matrix: Optional[torch.Tensor],
    targets: torch.Tensor,
    predictions: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    if confusion_matrix is None:
        confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=targets.device)
    encoded = targets * num_classes + predictions
    confusion_matrix += torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return confusion_matrix


@torch.no_grad()
def macro_precision_recall_f1_from_confusion(confusion_matrix: torch.Tensor) -> Tuple[float, float, float]:
    """Compute macro precision, macro recall, and macro F1 from a [C, C] confusion matrix."""
    conf = confusion_matrix.float()
    if conf.numel() == 0:
        return 0.0, 0.0, 0.0

    tp = conf.diag()
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp

    precision_per_class = tp / (tp + fp).clamp_min(1e-12)
    recall_per_class = tp / (tp + fn).clamp_min(1e-12)
    f1_per_class = (2 * precision_per_class * recall_per_class) / (precision_per_class + recall_per_class).clamp_min(1e-12)

    precision = precision_per_class.mean().item()
    recall = recall_per_class.mean().item()
    f1 = f1_per_class.mean().item()
    return precision, recall, f1