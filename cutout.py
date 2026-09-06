import hashlib
import json
import math
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, get_worker_info

from cam_masking import compute_saliency_map
from logger import get_logger


module_logger = get_logger(__name__)
_CAM_CACHE_VERSION = 2
_CAM_WINDOW_CACHE_VERSION = 2
_CAM_WORKER_CACHE_MISS_HINT = (
    "CAM cache miss during worker training. Run the same command once with "
    "--cam_precompute_only --num_workers 0 --deterministic_train_transforms, "
    "optionally adding --cam_precompute_windows to warm the window cache, "
    "then rerun training with num_workers > 0."
)


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


def _tensor_descriptor(tensor: torch.Tensor) -> Dict[str, Any]:
    if not torch.is_tensor(tensor):
        raise RuntimeError("CAM cache expected image to be a torch.Tensor.")
    return {
        "shape": tuple(int(v) for v in tensor.shape),
        "dtype": str(tensor.dtype),
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

    for attr_name in ("filenames", "filepaths", "paths", "image_paths"):
        paths = getattr(current, attr_name, None)
        if paths is None:
            continue
        try:
            source = paths[resolved_index]
        except Exception:
            continue
        if isinstance(source, (str, os.PathLike)):
            identity.update(
                {
                    "identity_kind": attr_name,
                    "path": os.path.abspath(os.fspath(source)),
                }
            )
            return identity

    records = getattr(current, "records", None)
    if records is not None:
        try:
            record = records[resolved_index]
            uid = getattr(record, "uid", None)
            if uid is not None:
                identity.update({"identity_kind": "record", "sample_key": str(uid)})
                return identity
        except Exception:
            pass

    for attr_name in ("root", "base_folder", "split", "train"):
        if hasattr(current, attr_name):
            value = getattr(current, attr_name)
            if isinstance(value, (str, os.PathLike)):
                value = os.path.abspath(os.fspath(value))
            identity[attr_name] = value

    identity["identity_kind"] = "dataset_index"
    return identity


def _has_stable_sample_scope(sample_identity: Dict[str, Any]) -> bool:
    if sample_identity.get("path"):
        return True
    return any(key in sample_identity for key in ("root", "base_folder", "split", "train"))


def _should_hash_tensor_for_cache(settings: Dict[str, Any], sample_identity: Dict[str, Any]) -> bool:
    # Stochastic train transforms can change the tensor for the same sample index,
    # so keep the legacy tensor hash unless deterministic transforms are in use.
    deterministic = bool((settings or {}).get("deterministic_train_transforms", False))
    return (not deterministic) or (not _has_stable_sample_scope(sample_identity))


def _cache_sample_payload(
    base_dataset: Dataset,
    base_index: int,
    image: torch.Tensor,
    settings: Optional[Dict[str, Any]],
    *,
    force_tensor_fingerprint: bool = False,
) -> Dict[str, Any]:
    sample_identity = _resolve_dataset_identity(base_dataset, base_index)
    if force_tensor_fingerprint or _should_hash_tensor_for_cache(settings or {}, sample_identity):
        image_identity = _tensor_fingerprint(image)
    else:
        image_identity = _tensor_descriptor(image)
        image_identity["sha256"] = "omitted_stable_identity"

    return {
        "sample": sample_identity,
        "image": image_identity,
    }


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
    if saliency.ndim not in (2, 3):
        location = f" at '{cache_path}'" if cache_path is not None else ""
        raise RuntimeError(f"CAM saliency{location} must have shape [H,W] or [D,H,W], got {tuple(saliency.shape)}.")
    if saliency.numel() == 0:
        location = f" at '{cache_path}'" if cache_path is not None else ""
        raise RuntimeError(f"CAM saliency{location} is empty.")
    if not torch.isfinite(saliency).all():
        location = f" at '{cache_path}'" if cache_path is not None else ""
        raise RuntimeError(f"CAM saliency{location} contains NaN or Inf values.")
    if float(saliency.max() - saliency.min()) <= 1e-12:
        location = f" at '{cache_path}'" if cache_path is not None else ""
        raise RuntimeError(f"CAM saliency{location} has no dynamic range.")
    return saliency


def _accumulate_timing(timings: Optional[Dict[str, float]], key: str, started_at: Optional[float]) -> None:
    if timings is None or started_at is None:
        return
    timings[key] = timings.get(key, 0.0) + (time.perf_counter() - started_at) * 1000.0


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
        self.legacy_lookup_enabled = self._has_existing_saliency_entries()

    def _has_existing_saliency_entries(self) -> bool:
        try:
            children = list(self.cache_dir.iterdir())
        except OSError:
            return False
        for child in children:
            if child.name == "windows":
                continue
            if child.is_file() and child.suffix == ".pt":
                return True
            if child.is_dir():
                try:
                    if any(path.is_file() and path.suffix == ".pt" for path in child.rglob("*.pt")):
                        return True
                except OSError:
                    continue
        return False

    def _path_for_payload(
        self,
        base_dataset: Dataset,
        base_index: int,
        image: torch.Tensor,
        *,
        force_tensor_fingerprint: bool,
        cache_version: int = _CAM_CACHE_VERSION,
    ) -> Path:
        sample_payload = _cache_sample_payload(
            base_dataset,
            base_index,
            image,
            self.settings,
            force_tensor_fingerprint=force_tensor_fingerprint,
        )
        payload = {
            "version": int(cache_version),
            "settings": self.settings,
            "sample": sample_payload["sample"],
            "image": sample_payload["image"],
            "target_class": None,
        }
        digest = _json_hash(payload)
        return self.cache_dir / digest[:2] / digest[2:4] / f"{digest}.pt"

    def path_for(self, base_dataset: Dataset, base_index: int, image: torch.Tensor) -> Path:
        return self._path_for_payload(
            base_dataset,
            base_index,
            image,
            force_tensor_fingerprint=False,
        )

    def legacy_path_for(self, base_dataset: Dataset, base_index: int, image: torch.Tensor) -> Path:
        return self._path_for_payload(
            base_dataset,
            base_index,
            image,
            force_tensor_fingerprint=True,
            cache_version=1,
        )

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


class CamWindowCache:
    """Cache final 2D (top,left,size) or 3D (z,top,left,size) CAM windows."""

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

    def path_for(
        self,
        base_dataset: Dataset,
        base_index: int,
        image: torch.Tensor,
        *,
        aug_index: int,
        cutout_mode: str,
        cutout_m: int,
        size: int,
        cutout_size: Optional[int],
        cutout_area: Optional[float],
        saliency_candidate_percent: float,
        seed: int,
        min_foreground_fraction: float = 0.0,
        spatial_dims: int = 2,
    ) -> Path:
        dataset_index = int(base_index) * (int(cutout_m) + 1) + int(aug_index)
        sample_payload = _cache_sample_payload(base_dataset, base_index, image, self.settings)
        payload = {
            "version": _CAM_WINDOW_CACHE_VERSION,
            "saliency_cache_version": _CAM_CACHE_VERSION,
            "settings": self.settings,
            "sample": sample_payload["sample"],
            "image": sample_payload["image"],
            "window": {
                "aug_index": int(aug_index),
                "cutout_mode": str(cutout_mode),
                "cutout_m": int(cutout_m),
                "cutout_size": int(cutout_size) if cutout_size is not None else None,
                "cutout_area": float(cutout_area) if cutout_area is not None else None,
                "resolved_size": int(size),
                "saliency_candidate_percent": float(saliency_candidate_percent),
                "seed": int(seed),
                "dataset_index": dataset_index,
                "rng_seed": int(seed) + dataset_index,
                "min_foreground_fraction": float(min_foreground_fraction),
                "spatial_dims": int(spatial_dims),
            },
        }
        digest = _json_hash(payload)
        return self.cache_dir / digest[:2] / digest[2:4] / f"{digest}.json"

    def load_if_exists(self, cache_path: Path) -> Optional[Dict[str, int]]:
        if not cache_path.is_file():
            if self._miss_logs < self.log_limit:
                _log_info(self.logger, "CAM window cache miss: %s", cache_path)
                self._miss_logs += 1
            return None

        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        try:
            window = {"top": int(payload["top"]), "left": int(payload["left"]), "size": int(payload["size"])}
            if "z" in payload:
                window["z"] = int(payload["z"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid CAM window cache entry at '{cache_path}'.") from exc

        if self._hit_logs < self.log_limit:
            _log_info(self.logger, "CAM window cache hit: %s", cache_path)
            self._hit_logs += 1
        return window

    def save(self, cache_path: Path, *, top: int, left: int, size: int, z: Optional[int] = None) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        payload = {
            "top": int(top),
            "left": int(left),
            "size": int(size),
        }
        if z is not None:
            payload["z"] = int(z)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, sort_keys=True)
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        if self._save_logs < self.log_limit:
            _log_info(self.logger, "CAM window cache saved: %s", cache_path)
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


def _resolve_cutout_cube_size(
    depth: int,
    height: int,
    width: int,
    cutout_size: Optional[int],
    cutout_area: Optional[float],
) -> int:
    if cutout_size is not None and int(cutout_size) > 0:
        size = int(cutout_size)
    elif cutout_area is not None and float(cutout_area) > 0:
        volume = float(cutout_area) * float(depth * height * width)
        size = int(round(max(volume, 1.0) ** (1.0 / 3.0)))
    else:
        size = min(depth, height, width)
    return max(1, min(size, depth, height, width))


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


def _sample_random_cube(depth: int, height: int, width: int, size: int, rng: random.Random) -> Tuple[int, int, int]:
    z = rng.randint(0, max(0, depth - size))
    top = rng.randint(0, max(0, height - size))
    left = rng.randint(0, max(0, width - size))
    return z, top, left


def _valid_window_coordinates(
    foreground_mask: Optional[torch.Tensor],
    size: int,
    spatial_shape: Tuple[int, ...],
    min_foreground_fraction: float,
) -> Optional[torch.Tensor]:
    """Return flattened valid-window indices using avg_pool2d or avg_pool3d."""
    if foreground_mask is None:
        return None
    if not torch.is_tensor(foreground_mask):
        raise RuntimeError("foreground_mask must be a torch.Tensor.")
    if tuple(foreground_mask.shape) != tuple(spatial_shape):
        raise RuntimeError(
            f"foreground_mask shape {tuple(foreground_mask.shape)} does not match image spatial shape {tuple(spatial_shape)}."
        )
    if len(spatial_shape) not in (2, 3):
        raise RuntimeError("foreground_mask must be 2D or 3D.")
    fraction = float(min_foreground_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise RuntimeError("min_foreground_fraction must be in [0,1].")
    mask = foreground_mask.detach().to(dtype=torch.float32, device="cpu")
    if len(spatial_shape) == 2:
        pooled = F.avg_pool2d(mask[None, None], kernel_size=size, stride=1)
    else:
        pooled = F.avg_pool3d(mask[None, None], kernel_size=size, stride=1)
    valid = pooled[0, 0].reshape(-1) >= fraction - 1e-7
    return torch.nonzero(valid, as_tuple=True)[0]


def _select_cam_window(
    saliency: torch.Tensor,
    size: int,
    mode: str,
    candidate_percent: float,
    rng: random.Random,
    foreground_mask: Optional[torch.Tensor] = None,
    min_foreground_fraction: float = 0.0,
) -> Tuple[int, ...]:
    if not torch.is_tensor(saliency):
        raise RuntimeError("CAM saliency must be a torch.Tensor.")
    if saliency.ndim not in (2, 3):
        raise RuntimeError(f"CAM saliency must have shape [H,W] or [D,H,W], got {tuple(saliency.shape)}.")
    if saliency.numel() == 0:
        raise RuntimeError("CAM saliency map is empty.")
    if not torch.isfinite(saliency).all():
        raise RuntimeError("CAM saliency map contains NaN or Inf values.")
    if mode not in {"cam_low", "cam_high"}:
        raise RuntimeError(f"Unsupported CAM cutout mode '{mode}'.")

    spatial_shape = tuple(int(v) for v in saliency.shape)
    height, width = int(saliency.shape[-2]), int(saliency.shape[-1])
    if size <= 0:
        raise RuntimeError(f"Cutout size must be positive, got {size}.")
    if any(axis < size for axis in spatial_shape):
        return (0, 0) if saliency.ndim == 2 else (0, 0, 0)

    if saliency.ndim == 2:
        scores_map = F.avg_pool2d(saliency[None, None], kernel_size=size, stride=1)[0, 0]
    else:
        scores_map = F.avg_pool3d(saliency[None, None], kernel_size=size, stride=1)[0, 0]
    scores = scores_map.flatten()
    valid_indices = _valid_window_coordinates(foreground_mask, size, spatial_shape, min_foreground_fraction)
    if valid_indices is not None:
        if valid_indices.numel() == 0:
            raise RuntimeError("No cutout candidates meet min_foreground_fraction.")
        scores = scores[valid_indices]
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
    if valid_indices is not None:
        choice = int(valid_indices[choice].item())
    if saliency.ndim == 2:
        out_w = width - size + 1
        row = choice // out_w
        col = choice % out_w
        return int(row), int(col)
    out_h = height - size + 1
    out_w = width - size + 1
    z = choice // (out_h * out_w)
    rem = choice % (out_h * out_w)
    row = rem // out_w
    col = rem % out_w
    return int(z), int(row), int(col)


def _resolve_dataset_foreground_mask(dataset: Dataset, index: int) -> Optional[torch.Tensor]:
    current = dataset
    resolved = int(index)
    while isinstance(current, Subset):
        resolved = _coerce_index(current.indices[resolved])
        current = current.dataset
    getter = getattr(current, "get_foreground_mask", None)
    if callable(getter):
        return getter(resolved)
    masks = getattr(current, "foreground_masks", None)
    if masks is not None:
        return masks[resolved]
    return None


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
        cam_window_cache_dir: Optional[str] = None,
        foreground_mask_getter=None,
        min_foreground_fraction: float = 0.0,
        debug_cam_timing: bool = False,
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
        self.foreground_mask_getter = foreground_mask_getter
        self.min_foreground_fraction = float(min_foreground_fraction)
        if not 0.0 <= self.min_foreground_fraction <= 1.0:
            raise ValueError("min_foreground_fraction must be in [0,1].")
        self.logger = logger or module_logger
        self.debug_cam_timing = bool(debug_cam_timing)
        self._cam_window_log_limit = max(0, int(debug_log_limit))
        self._cam_window_logs_emitted = 0
        self._cam_timing_logs_emitted = 0
        self.cam_cache = None
        if self.cutout_mode in {"cam_low", "cam_high"} and cam_cache_dir:
            self.cam_cache = CamSaliencyCache(
                cache_dir=cam_cache_dir,
                settings=cam_cache_settings,
                logger=self.logger,
                log_limit=debug_log_limit,
            )
            _log_info(self.logger, "CAM saliency cache enabled: %s", self.cam_cache.cache_dir)

        self.cam_window_cache = None
        if self.cutout_mode in {"cam_low", "cam_high"}:
            resolved_window_cache_dir = cam_window_cache_dir
            if not resolved_window_cache_dir and cam_cache_dir:
                resolved_window_cache_dir = os.path.join(cam_cache_dir, "windows")
            if resolved_window_cache_dir:
                self.cam_window_cache = CamWindowCache(
                    cache_dir=resolved_window_cache_dir,
                    settings=cam_cache_settings,
                    logger=self.logger,
                    log_limit=debug_log_limit,
                )
                _log_info(self.logger, "CAM window cache enabled: %s", self.cam_window_cache.cache_dir)

        self._enabled = self.cutout_m > 0 and self.cutout_mode in {"random", "cam_low", "cam_high"}
        self._base_len = len(self.base_dataset)

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch changes so wrapped spatial augmentation stays deterministic."""
        if hasattr(self.base_dataset, "set_epoch"):
            self.base_dataset.set_epoch(int(epoch))

    def __len__(self) -> int:
        if not self._enabled:
            return self._base_len
        return self._base_len * (self.cutout_m + 1)

    def _foreground_mask_for_sample(self, base_index: int, image: torch.Tensor) -> Optional[torch.Tensor]:
        if self.foreground_mask_getter is not None:
            try:
                mask = self.foreground_mask_getter(base_index, image)
            except TypeError:
                mask = self.foreground_mask_getter(base_index)
        else:
            mask = _resolve_dataset_foreground_mask(self.base_dataset, base_index)
        if mask is None:
            return None
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
        return mask.detach().cpu().bool()

    def _log_cam_window(self, base_index: int, aug_index: int, top: int, left: int, size: int) -> None:
        if self.cutout_mode not in {"cam_low", "cam_high"}:
            return
        if self._cam_window_logs_emitted >= self._cam_window_log_limit:
            return
        _log_info(self.logger, "CAM cutout window selected (mode=%s, size=%s).", self.cutout_mode, size)
        self._cam_window_logs_emitted += 1

    def _log_cam_timing(self, base_index: int, aug_index: int, timings: Optional[Dict[str, float]]) -> None:
        if not self.debug_cam_timing or not timings:
            return
        if self._cam_timing_logs_emitted >= self._cam_window_log_limit:
            return
        timing_text = " ".join(f"{key}={value:.2f}ms" for key, value in sorted(timings.items()))
        _log_info(self.logger, "CAM timing (mode=%s): %s", self.cutout_mode, timing_text)
        self._cam_timing_logs_emitted += 1

    def _dataset_index_for_aug(self, base_index: int, aug_index: int) -> int:
        return int(base_index) * (self.cutout_m + 1) + int(aug_index)

    def _rng_for_aug(self, base_index: int, aug_index: int) -> random.Random:
        return random.Random(self.seed + self._dataset_index_for_aug(base_index, aug_index))

    def _cam_window_cache_path(
        self,
        base_index: int,
        aug_index: int,
        image: torch.Tensor,
        size: int,
        timings: Optional[Dict[str, float]] = None,
    ) -> Optional[Path]:
        if self.cam_window_cache is None:
            return None
        started_at = time.perf_counter() if timings is not None else None
        cache_path = self.cam_window_cache.path_for(
            self.base_dataset,
            base_index,
            image,
            aug_index=aug_index,
            cutout_mode=self.cutout_mode,
            cutout_m=self.cutout_m,
            size=size,
            cutout_size=self.cutout_size,
            cutout_area=self.cutout_area,
            saliency_candidate_percent=self.saliency_candidate_percent,
            seed=self.seed,
            min_foreground_fraction=self.min_foreground_fraction,
            spatial_dims=image.ndim - 1,
        )
        _accumulate_timing(timings, "cache_key_path", started_at)
        return cache_path

    def _load_cached_cam_window(
        self,
        cache_path: Optional[Path],
        size: int,
        spatial_dims: int = 2,
        timings: Optional[Dict[str, float]] = None,
    ) -> Optional[Tuple[int, ...]]:
        if self.cam_window_cache is None or cache_path is None:
            return None
        started_at = time.perf_counter() if timings is not None else None
        cached = self.cam_window_cache.load_if_exists(cache_path)
        _accumulate_timing(timings, "window_cache_load", started_at)
        if cached is None:
            return None
        if int(cached["size"]) != int(size):
            raise RuntimeError(
                f"CAM window cache size mismatch at '{cache_path}': "
                f"expected {size}, got {cached['size']}."
            )
        if int(spatial_dims) == 3:
            if "z" not in cached:
                raise RuntimeError(f"3D CAM window cache entry at '{cache_path}' is missing z.")
            return int(cached["z"]), int(cached["top"]), int(cached["left"])
        return int(cached["top"]), int(cached["left"])

    def _save_cam_window(self, cache_path: Optional[Path], top: int, left: int, size: int, z: Optional[int] = None) -> None:
        if self.cam_window_cache is None or cache_path is None:
            return
        self.cam_window_cache.save(cache_path, top=top, left=left, size=size, z=z)

    def _get_cam_saliency(
        self,
        base_index: int,
        image: torch.Tensor,
        timings: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        cache_path = None
        if self.cam_cache is not None:
            started_at = time.perf_counter() if timings is not None else None
            cache_path = self.cam_cache.path_for(self.base_dataset, base_index, image)
            _accumulate_timing(timings, "cache_key_path", started_at)

            started_at = time.perf_counter() if timings is not None else None
            cached = self.cam_cache.load_if_exists(cache_path)
            _accumulate_timing(timings, "saliency_cache_load", started_at)
            if cached is not None:
                return cached

            if self.cam_cache.legacy_lookup_enabled:
                started_at = time.perf_counter() if timings is not None else None
                legacy_cache_path = self.cam_cache.legacy_path_for(self.base_dataset, base_index, image)
                _accumulate_timing(timings, "cache_key_path", started_at)
                if legacy_cache_path != cache_path:
                    started_at = time.perf_counter() if timings is not None else None
                    cached = self.cam_cache.load_if_exists(legacy_cache_path)
                    _accumulate_timing(timings, "saliency_cache_load", started_at)
                    if cached is not None:
                        self.cam_cache.save(cache_path, cached)
                        return cached

        if self.teacher_model is None:
            raise RuntimeError(
                f"CAM cutout cache miss for dataset index {base_index}, but no teacher_model is available. "
                f"{_CAM_WORKER_CACHE_MISS_HINT}"
            )

        worker = get_worker_info()
        if worker is not None and _model_device_type(self.teacher_model) == "cuda":
            raise RuntimeError(
                f"CAM cutout cache miss for dataset index {base_index} inside DataLoader worker {worker.id}. "
                f"CUDA saliency cannot be computed in forked workers. {_CAM_WORKER_CACHE_MISS_HINT}"
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

    def _get_or_create_cam_window(
        self,
        base_index: int,
        aug_index: int,
        image: torch.Tensor,
        size: int,
        rng: random.Random,
        timings: Optional[Dict[str, float]] = None,
        saliency: Optional[torch.Tensor] = None,
    ) -> Tuple[int, ...]:
        # Saliency maps are the expensive teacher CAM tensors. Window entries
        # are the final mask coordinates derived from those maps; caching them
        # removes repeated CPU pooling/top-k work during later epochs.
        cache_path = self._cam_window_cache_path(base_index, aug_index, image, size, timings)
        spatial_dims = image.ndim - 1
        foreground_mask = self._foreground_mask_for_sample(base_index, image)
        cached = self._load_cached_cam_window(cache_path, size, spatial_dims=spatial_dims, timings=timings)
        if cached is not None:
            return cached

        if saliency is None:
            saliency = self._get_cam_saliency(base_index, image, timings=timings)
        started_at = time.perf_counter() if timings is not None else None
        window = _select_cam_window(
            saliency,
            size,
            self.cutout_mode,
            self.saliency_candidate_percent,
            rng,
            foreground_mask=foreground_mask,
            min_foreground_fraction=self.min_foreground_fraction,
        )
        _accumulate_timing(timings, "window_selection", started_at)
        if spatial_dims == 3:
            z, top, left = window
            self._save_cam_window(cache_path, top, left, size, z=z)
        else:
            top, left = window
            self._save_cam_window(cache_path, top, left, size)
        return window

    def precompute_cam_windows_for_sample(
        self,
        base_index: int,
        image: torch.Tensor,
        saliency: Optional[torch.Tensor] = None,
    ) -> Tuple[int, int]:
        if self.cutout_mode not in {"cam_low", "cam_high"}:
            return 0, 0
        if self.cutout_m <= 0:
            return 0, 0
        if not torch.is_tensor(image):
            raise ValueError("CAM window precompute expects the training dataset to return tensors.")

        if image.ndim not in (3, 4):
            raise ValueError(f"CAM window precompute expects [C,H,W] or [C,D,H,W], got {tuple(image.shape)}.")
        spatial_dims = image.ndim - 1
        spatial_shape = tuple(int(v) for v in image.shape[1:])
        if spatial_dims == 3:
            size = _resolve_cutout_cube_size(*spatial_shape, self.cutout_size, self.cutout_area)
        else:
            size = _resolve_cutout_size(*spatial_shape, self.cutout_size, self.cutout_area)
        foreground_mask = self._foreground_mask_for_sample(base_index, image)
        created = 0
        cached_count = 0
        missing = []

        for aug_index in range(1, self.cutout_m + 1):
            cache_path = self._cam_window_cache_path(base_index, aug_index, image, size)
            cached = self._load_cached_cam_window(cache_path, size, spatial_dims=spatial_dims)
            if cached is not None:
                cached_count += 1
                continue
            missing.append((aug_index, cache_path))

        if not missing:
            return created, cached_count

        if saliency is None:
            saliency = self._get_cam_saliency(base_index, image)
        for aug_index, cache_path in missing:
            rng = self._rng_for_aug(base_index, aug_index)
            window = _select_cam_window(
                saliency,
                size,
                self.cutout_mode,
                self.saliency_candidate_percent,
                rng,
                foreground_mask=foreground_mask,
                min_foreground_fraction=self.min_foreground_fraction,
            )
            if spatial_dims == 3:
                z, top, left = window
                self._save_cam_window(cache_path, top, left, size, z=z)
            else:
                top, left = window
                self._save_cam_window(cache_path, top, left, size)
            created += 1

        return created, cached_count

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

        if image.ndim not in (3, 4):
            raise ValueError(f"CutoutAugmentedDataset expects [C,H,W] or [C,D,H,W], got {tuple(image.shape)}.")
        spatial_dims = image.ndim - 1
        spatial_shape = tuple(int(v) for v in image.shape[1:])
        if spatial_dims == 3:
            size = _resolve_cutout_cube_size(*spatial_shape, self.cutout_size, self.cutout_area)
        else:
            size = _resolve_cutout_size(*spatial_shape, self.cutout_size, self.cutout_area)
        rng = self._rng_for_aug(base_index, aug_index)
        timings = {} if self.debug_cam_timing and self.cutout_mode in {"cam_low", "cam_high"} else None
        foreground_mask = self._foreground_mask_for_sample(base_index, image)

        if self.cutout_mode == "random":
            valid_indices = _valid_window_coordinates(
                foreground_mask,
                size,
                spatial_shape,
                self.min_foreground_fraction,
            )
            if valid_indices is None:
                if spatial_dims == 3:
                    window = _sample_random_cube(*spatial_shape, size, rng)
                else:
                    window = _sample_random_window(*spatial_shape, size, rng)
            else:
                if valid_indices.numel() == 0:
                    raise RuntimeError("No random cutout candidates meet min_foreground_fraction.")
                choice = int(valid_indices[rng.randrange(valid_indices.numel())].item())
                if spatial_dims == 3:
                    height, width = spatial_shape[-2], spatial_shape[-1]
                    out_h = height - size + 1
                    out_w = width - size + 1
                    z = choice // (out_h * out_w)
                    rem = choice % (out_h * out_w)
                    window = (z, rem // out_w, rem % out_w)
                else:
                    height, width = spatial_shape
                    out_w = width - size + 1
                    window = (choice // out_w, choice % out_w)
        elif self.cutout_mode in {"cam_low", "cam_high"}:
            try:
                window = self._get_or_create_cam_window(base_index, aug_index, image, size, rng, timings=timings)
            except Exception as exc:
                raise RuntimeError(
                    f"CAM cutout failed for dataset index {base_index}, "
                    f"cutout_mode={self.cutout_mode}: {exc}"
                ) from exc
        else:
            raise RuntimeError(f"Unsupported cutout_mode '{self.cutout_mode}'.")

        started_at = time.perf_counter() if timings is not None else None
        image = image.clone()
        black = _black_value(self.mean, self.std, int(image.shape[0]), image.device, image.dtype)
        if spatial_dims == 3:
            z, top, left = window
            image[:, z:z + size, top:top + size, left:left + size] = black[:, None, None, None]
        else:
            top, left = window
            image[:, top:top + size, left:left + size] = black[:, None, None]
        _accumulate_timing(timings, "masking", started_at)
        self._log_cam_timing(base_index, aug_index, timings)
        return image, target
