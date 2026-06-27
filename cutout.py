import hashlib
import json
import math
import os
import random
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, get_worker_info

from cam_masking import compute_saliency_map
from logger import get_logger


module_logger = get_logger(__name__)
_CAM_CACHE_VERSION = 1


def _log_info(log, msg, *args):
    try:
        log.info(msg, *args)
    except TypeError:
        log.info(msg % args if args else msg)



def _json_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_fingerprint(tensor: torch.Tensor) -> Dict[str, Any]:
    if not torch.is_tensor(tensor):
        raise RuntimeError("CAM cache expected image to be a torch.Tensor.")
    if tensor.device.type != "cpu":
        tensor = tensor.detach().cpu()
    else:
        tensor = tensor.detach()
    tensor = tensor.contiguous()

    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return {
        "shape": tuple(int(v) for v in tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": digest.hexdigest(),
    }


def _coerce_index(index) -> int:
    if hasattr(index, "item"):
        return int(index.item())
    return int(index)


def _resolve_dataset_identity(dataset: Dataset, index: int) -> Dict[str, Any]:
    current = dataset
    resolved_index = int(index)
    subset_chain = []

    while isinstance(current, Subset):
        subset_index = resolved_index
        resolved_index = _coerce_index(current.indices[resolved_index])
        subset_chain.append({"subset_index": subset_index, "parent_index": resolved_index})
        current = current.dataset

    identity: Dict[str, Any] = {
        "dataset_class": f"{type(current).__module__}.{type(current).__name__}",
        "index": resolved_index,
    }
    if subset_chain:
        identity["subset_chain"] = subset_chain

    for attr_name in ("samples", "imgs"):
        samples = getattr(current, attr_name, None)
        if samples is None:
            continue
        try:
            sample = samples[resolved_index]
        except Exception:
            continue
        source = sample[0] if isinstance(sample, (tuple, list)) and sample else sample
        if isinstance(source, (str, os.PathLike)):
            identity.update(
                {
                    "identity_kind": attr_name,
                    "path": os.path.abspath(os.fspath(source)),
                }
            )
            return identity

    for attr_name in ("root", "base_folder", "split", "train"):
        if hasattr(current, attr_name):
            value = getattr(current, attr_name)
            if isinstance(value, (str, os.PathLike)):
                value = os.path.abspath(os.fspath(value))
            identity[attr_name] = value

    identity["identity_kind"] = "dataset_index"
    return identity


def _torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _model_device_type(model: Optional[torch.nn.Module]) -> str:
    if model is None:
        return "none"
    try:
        return next(model.parameters()).device.type
    except StopIteration:
        pass
    try:
        return next(model.buffers()).device.type
    except StopIteration:
        return "cpu"


def _validate_saliency_tensor(saliency: torch.Tensor, cache_path: Optional[Path] = None) -> torch.Tensor:
    if not torch.is_tensor(saliency):
        location = f" at '{cache_path}'" if cache_path is not None else ""
        raise RuntimeError(f"CAM cache entry{location} is not a torch.Tensor.")
    saliency = saliency.detach().cpu()
    if saliency.ndim != 2:
        location = f" at '{cache_path}'" if cache_path is not None else ""
        raise RuntimeError(f"CAM saliency{location} must have shape [H, W], got {tuple(saliency.shape)}.")
    if saliency.numel() == 0:
        location = f" at '{cache_path}'" if cache_path is not None else ""
        raise RuntimeError(f"CAM saliency{location} is empty.")
    if not torch.isfinite(saliency).all():
        location = f" at '{cache_path}'" if cache_path is not None else ""
        raise RuntimeError(f"CAM saliency{location} contains NaN or Inf values.")
    return saliency


class CamSaliencyCache:
    def __init__(
        self,
        cache_dir: str,
        settings: Optional[Dict[str, Any]] = None,
        logger=None,
        log_limit: int = 5,
    ):
        self.cache_dir = Path(cache_dir).expanduser()
        self.settings = dict(settings or {})
        self.logger = logger or module_logger
        self.log_limit = max(0, int(log_limit))
        self._hit_logs = 0
        self._miss_logs = 0
        self._save_logs = 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, base_dataset: Dataset, base_index: int, image: torch.Tensor) -> Path:
        payload = {
            "version": _CAM_CACHE_VERSION,
            "settings": self.settings,
            "sample": _resolve_dataset_identity(base_dataset, base_index),
            "image": _tensor_fingerprint(image),
            "target_class": None,
        }
        digest = _json_hash(payload)
        return self.cache_dir / digest[:2] / digest[2:4] / f"{digest}.pt"

    def load_if_exists(self, cache_path: Path) -> Optional[torch.Tensor]:
        if not cache_path.is_file():
            if self._miss_logs < self.log_limit:
                _log_info(self.logger, "CAM cache miss: %s", cache_path)
                self._miss_logs += 1
            return None

        saliency = _validate_saliency_tensor(_torch_load_cpu(cache_path), cache_path)
        if self._hit_logs < self.log_limit:
            _log_info(self.logger, "CAM cache hit: %s", cache_path)
            self._hit_logs += 1
        return saliency

    def save(self, cache_path: Path, saliency: torch.Tensor) -> None:
        saliency = _validate_saliency_tensor(saliency)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        try:
            torch.save(saliency.detach().cpu(), tmp_path)
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        if self._save_logs < self.log_limit:
            _log_info(self.logger, "CAM cache saved: %s", cache_path)
            self._save_logs += 1


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
    if not torch.is_tensor(saliency):
        raise RuntimeError("CAM saliency must be a torch.Tensor.")
    if saliency.ndim != 2:
        raise RuntimeError(f"CAM saliency must have shape [H, W], got {tuple(saliency.shape)}.")
    if saliency.numel() == 0:
        raise RuntimeError("CAM saliency map is empty.")
    if not torch.isfinite(saliency).all():
        raise RuntimeError("CAM saliency map contains NaN or Inf values.")
    if mode not in {"cam_low", "cam_high"}:
        raise RuntimeError(f"Unsupported CAM cutout mode '{mode}'.")

    height, width = int(saliency.shape[-2]), int(saliency.shape[-1])
    if size <= 0:
        raise RuntimeError(f"Cutout size must be positive, got {size}.")
    if height < size or width < size:
        return 0, 0

    scores = F.avg_pool2d(saliency.unsqueeze(0).unsqueeze(0), kernel_size=size, stride=1)
    scores = scores.flatten()
    total = scores.numel()
    if total == 0:
        raise RuntimeError("CAM cutout produced no candidate windows.")

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
        cam_cache_dir: Optional[str] = None,
        cam_cache_settings: Optional[Dict[str, Any]] = None,
        debug_log_limit: int = 5,
        logger=None,
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
        self.logger = logger or module_logger
        self._cam_window_log_limit = max(0, int(debug_log_limit))
        self._cam_window_logs_emitted = 0
        self.cam_cache = None
        if self.cutout_mode in {"cam_low", "cam_high"} and cam_cache_dir:
            self.cam_cache = CamSaliencyCache(
                cache_dir=cam_cache_dir,
                settings=cam_cache_settings,
                logger=self.logger,
                log_limit=debug_log_limit,
            )
            _log_info(self.logger, "CAM saliency cache enabled: %s", self.cam_cache.cache_dir)

        self._enabled = self.cutout_m > 0 and self.cutout_mode in {"random", "cam_low", "cam_high"}
        self._base_len = len(self.base_dataset)

    def __len__(self) -> int:
        if not self._enabled:
            return self._base_len
        return self._base_len * (self.cutout_m + 1)

    def _log_cam_window(self, base_index: int, aug_index: int, top: int, left: int, size: int) -> None:
        if self.cutout_mode not in {"cam_low", "cam_high"}:
            return
        if self._cam_window_logs_emitted >= self._cam_window_log_limit:
            return
        _log_info(
            self.logger,
            "CAM cutout window: index=%s copy=%s cutout_mode=%s top=%s left=%s height=%s width=%s",
            base_index,
            aug_index,
            self.cutout_mode,
            top,
            left,
            size,
            size,
        )
        self._cam_window_logs_emitted += 1

    def _get_cam_saliency(self, base_index: int, image: torch.Tensor) -> torch.Tensor:
        cache_path = None
        if self.cam_cache is not None:
            cache_path = self.cam_cache.path_for(self.base_dataset, base_index, image)
            cached = self.cam_cache.load_if_exists(cache_path)
            if cached is not None:
                return cached

        if self.teacher_model is None:
            raise RuntimeError(
                f"CAM cutout cache miss for dataset index {base_index}, but no teacher_model is available. "
                "Populate the cache with num_workers=0 or provide a teacher_model."
            )

        worker = get_worker_info()
        if worker is not None and _model_device_type(self.teacher_model) == "cuda":
            raise RuntimeError(
                f"CAM cutout cache miss for dataset index {base_index} inside DataLoader worker {worker.id}. "
                "CUDA saliency cannot be computed in forked workers. Populate the CAM cache first with "
                "--num_workers 0, then rerun with workers."
            )

        saliency = compute_saliency_map(
            self.teacher_model,
            image,
            target_class=None,
            cam_layer=self.cam_layer,
        )
        saliency = _validate_saliency_tensor(saliency)
        if self.cam_cache is not None and cache_path is not None:
            self.cam_cache.save(cache_path, saliency)
        return saliency

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
        elif self.cutout_mode in {"cam_low", "cam_high"}:
            try:
                saliency = self._get_cam_saliency(base_index, image)
                top, left = _select_cam_window(
                    saliency,
                    size,
                    self.cutout_mode,
                    self.saliency_candidate_percent,
                    rng,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"CAM cutout failed for dataset index {base_index}, "
                    f"cutout_mode={self.cutout_mode}: {exc}"
                ) from exc
            self._log_cam_window(base_index, aug_index, top, left, size)
        else:
            raise RuntimeError(f"Unsupported cutout_mode '{self.cutout_mode}'.")

        image = image.clone()
        black = _black_value(self.mean, self.std, int(image.shape[0]), image.device, image.dtype)
        image[:, top:top + size, left:left + size] = black[:, None, None]
        return image, target
