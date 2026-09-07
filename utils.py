from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from typing import Tuple

# Default configuration constants
DEFAULT_DATASET = "cifar100"
DEFAULT_MODEL = "resnet18"
REPO_ROOT = Path(__file__).resolve().parent

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


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[: int(length)]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_git_commit(root: str | Path = REPO_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(root), check=True,
            capture_output=True, text=True,
        )
        value = result.stdout.strip()
        return value or None
    except (OSError, subprocess.SubprocessError):
        return None


def portable_path(path: str | Path, root: str | Path = REPO_ROOT) -> str:
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path must be inside repository root: {path}") from exc


def resolve_portable_path(value: str | Path, root: str | Path = REPO_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(root).resolve() / path


def research_valid(*, max_train_batches: int | None, max_val_batches: int | None, debug: bool = False) -> bool:
    return not bool(debug) and (max_train_batches in (None, 0)) and (max_val_batches in (None, 0))


def median_round_half_up(values: list[int] | tuple[int, ...] | np.ndarray) -> int:
    if len(values) == 0:
        raise ValueError("Cannot derive a median epoch budget from no epochs.")
    return max(1, int(np.floor(np.median(np.asarray(values, dtype=np.float64)) + 0.5)))
