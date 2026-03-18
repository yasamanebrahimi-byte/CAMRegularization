import random
import torch
from PIL import Image
from typing import Tuple

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