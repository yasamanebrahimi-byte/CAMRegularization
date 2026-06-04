import math
import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from cam_masking import compute_saliency_map
from logger import get_logger


logger = get_logger(__name__)


def _expand_stat(values, channels: int) -> Tuple[float, ...]:
    if isinstance(values, (int, float)):
        values = (float(values),)
    if len(values) == channels:
        return tuple(float(v) for v in values)
    if len(values) == 1:
        return tuple(float(values[0]) for _ in range(channels))
    if channels < len(values):
        return tuple(float(v) for v in values[:channels])
    return tuple(float(values[-1]) for _ in range(channels))


def _black_value(mean, std, channels: int, device, dtype) -> torch.Tensor:
    mean_v = torch.tensor(_expand_stat(mean, channels), device=device, dtype=dtype)
    std_v = torch.tensor(_expand_stat(std, channels), device=device, dtype=dtype)
    return (0.0 - mean_v) / std_v


def _resolve_cutout_size(height: int, width: int, cutout_size: Optional[int], cutout_area: Optional[float]) -> int:
    if cutout_size is not None and int(cutout_size) > 0:
        size = int(cutout_size)
    elif cutout_area is not None and float(cutout_area) > 0:
        area = float(cutout_area) * float(height * width)
        size = int(round(math.sqrt(max(area, 1.0))))
    else:
        size = min(height, width)
    return max(1, min(size, height, width))


def _sample_random_window(height: int, width: int, size: int, rng: random.Random) -> Tuple[int, int]:
    if height <= size:
        top = 0
    else:
        top = rng.randint(0, height - size)
    if width <= size:
        left = 0
    else:
        left = rng.randint(0, width - size)
    return top, left


def _select_cam_window(
    saliency: torch.Tensor,
    size: int,
    mode: str,
    candidate_percent: float,
    rng: random.Random,
) -> Tuple[int, int]:
    height, width = int(saliency.shape[-2]), int(saliency.shape[-1])
    if height < size or width < size:
        return 0, 0

    scores = F.avg_pool2d(saliency.unsqueeze(0).unsqueeze(0), kernel_size=size, stride=1)
    scores = scores.flatten()
    total = scores.numel()
    if total == 0:
        return 0, 0

    percent = float(candidate_percent)
    percent = max(0.0, min(100.0, percent))
    k = max(1, int(math.ceil(total * (percent / 100.0))))
    if mode == "cam_low":
        topk = torch.topk(scores, k, largest=False).indices
    else:
        topk = torch.topk(scores, k, largest=True).indices

    choice = int(topk[rng.randrange(k)].item())
    out_w = width - size + 1
    row = choice // out_w
    col = choice % out_w
    return int(row), int(col)


class CutoutAugmentedDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        cutout_mode: str,
        cutout_m: int,
        cutout_size: Optional[int],
        cutout_area: Optional[float],
        mean,
        std,
        seed: int = 0,
        saliency_candidate_percent: float = 10.0,
        teacher_model: Optional[torch.nn.Module] = None,
        cam_layer: str = "auto",
    ):
        self.base_dataset = base_dataset
        self.cutout_mode = str(cutout_mode or "none").lower()
        self.cutout_m = int(cutout_m)
        self.cutout_size = cutout_size if cutout_size is not None else None
        self.cutout_area = cutout_area if cutout_area is not None else None
        self.seed = int(seed)
        self.mean = mean
        self.std = std
        self.saliency_candidate_percent = float(saliency_candidate_percent)
        self.teacher_model = teacher_model
        self.cam_layer = cam_layer

        self._enabled = self.cutout_m > 0 and self.cutout_mode in {"random", "cam_low", "cam_high"}
        self._base_len = len(self.base_dataset)

    def __len__(self) -> int:
        if not self._enabled:
            return self._base_len
        return self._base_len * (self.cutout_m + 1)

    def __getitem__(self, index: int):
        if not self._enabled:
            return self.base_dataset[index]

        base_index = index // (self.cutout_m + 1)
        aug_index = index % (self.cutout_m + 1)
        image, target = self.base_dataset[base_index]

        if aug_index == 0:
            return image, target

        if not torch.is_tensor(image):
            raise ValueError("CutoutAugmentedDataset expects base_dataset to return tensors.")

        height, width = int(image.shape[-2]), int(image.shape[-1])
        size = _resolve_cutout_size(height, width, self.cutout_size, self.cutout_area)
        rng = random.Random(self.seed + int(index))

        if self.cutout_mode == "random":
            top, left = _sample_random_window(height, width, size, rng)
        else:
            top, left = None, None
            if self.teacher_model is not None:
                try:
                    saliency = compute_saliency_map(
                        self.teacher_model,
                        image,
                        target_class=None,
                        cam_layer=self.cam_layer,
                    )
                    top, left = _select_cam_window(
                        saliency,
                        size,
                        self.cutout_mode,
                        self.saliency_candidate_percent,
                        rng,
                    )
                except Exception as exc:
                    logger.warning("CAM cutout failed for index %s; falling back to random. Error: %s", base_index, exc)
            if top is None or left is None:
                top, left = _sample_random_window(height, width, size, rng)

        image = image.clone()
        black = _black_value(self.mean, self.std, int(image.shape[0]), image.device, image.dtype)
        image[:, top:top + size, left:left + size] = black[:, None, None]
        return image, target
