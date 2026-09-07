"""Deterministic preprocessing and dataset helpers for DaT NIfTI scans.

The module deliberately keeps competition data out of the repository.  It
accepts an extracted directory or an outer archive and discovers the training
layout from ``train_labels.csv`` and ``niftis/`` rather than from a private
archive name.
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader, Subset


DEFAULT_TARGET_SHAPE = (64, 96, 96)  # D, H, W
DEFAULT_PERCENTILES = (1.0, 99.0)
DEFAULT_FOREGROUND_THRESHOLD = 0.0
DEFAULT_CROP_MARGIN_MM = 8.0


@dataclass(frozen=True)
class DatRecord:
    uid: str
    label: int
    path: str


def _as_tuple(value: Any, length: int, cast):
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"Expected {length} values, got {value!r}.")
    return tuple(cast(part) for part in value)


def parse_target_spacing(value: Any) -> tuple[float, float, float]:
    return _as_tuple(value, 3, float)


def parse_target_shape(value: Any) -> tuple[int, int, int]:
    shape = _as_tuple(value, 3, int)
    if any(v <= 0 for v in shape):
        raise ValueError(f"target_shape must be positive, got {shape}.")
    return shape


def _safe_archive_member_path(root: Path, member_name: str) -> Path:
    destination = (root / member_name).resolve()
    root_resolved = root.resolve()
    if destination != root_resolved and root_resolved not in destination.parents:
        raise ValueError("Archive contains a path outside the extraction directory.")
    return destination


def _extract_archive(archive_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".extracted_ok"
    if marker.exists():
        return destination

    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                _safe_archive_member_path(destination, member.filename)
            archive.extractall(destination)
    elif archive_path.name.lower().endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                _safe_archive_member_path(destination, member.name)
            archive.extractall(destination)
    else:
        raise ValueError(f"Unsupported DaT archive format: {archive_path.name}")
    marker.touch()
    return destination


def _find_dat_root(search_root: Path, max_depth: int = 6) -> Optional[Path]:
    if not search_root.is_dir():
        return None
    candidates = [search_root / "train_labels.csv"]
    candidates.extend(search_root.rglob("train_labels.csv"))
    seen = set()
    for labels_path in candidates:
        labels_path = labels_path.resolve()
        if labels_path in seen:
            continue
        seen.add(labels_path)
        try:
            relative_depth = len(labels_path.relative_to(search_root.resolve()).parts)
        except ValueError:
            continue
        if relative_depth > max_depth + 1:
            continue
        root = labels_path.parent
        if (root / "niftis").is_dir():
            return root
        nested = root / "data"
        if (nested / "niftis").is_dir():
            return nested
    return None


def resolve_dat_root(data_dir: str | os.PathLike, archive_path: str | os.PathLike | None = None) -> Path:
    """Resolve an extracted DaT root or extract an outer archive locally."""
    configured = Path(data_dir).expanduser()
    if configured.is_file():
        archive_path = configured
        configured = configured.parent / ".dat_extracted" / configured.stem.replace(".", "_")

    if archive_path is not None:
        archive = Path(archive_path).expanduser()
        if not archive.is_file():
            raise FileNotFoundError(f"DaT archive not found: {archive}")
        extracted = _extract_archive(archive, configured)
        root = _find_dat_root(extracted)
        if root is None:
            raise FileNotFoundError("DaT archive does not contain train_labels.csv and niftis/.")
        return root

    root = _find_dat_root(configured)
    if root is not None:
        return root

    archives: list[Path] = []
    if configured.is_dir():
        for pattern in ("*.zip", "*.tar.gz", "*.tgz", "*.tar"):
            archives.extend(configured.glob(pattern))
    if archives:
        archive = sorted(archives, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        extracted = _extract_archive(archive, configured / ".dat_extracted" / archive.stem.replace(".", "_"))
        root = _find_dat_root(extracted)
        if root is not None:
            return root

    raise FileNotFoundError(
        "Could not find a DaT dataset root. Expected train_labels.csv beside niftis/ "
        "or an outer .zip/.tar.gz archive containing that layout."
    )


def load_dat_records(data_dir: str | os.PathLike, archive_path: str | os.PathLike | None = None) -> list[DatRecord]:
    root = resolve_dat_root(data_dir, archive_path=archive_path)
    labels_path = root / "train_labels.csv"
    niftis_dir = root / "niftis"
    frame = pd.read_csv(labels_path)
    required = {"uid", "is_pathologic"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"train_labels.csv is missing columns: {sorted(missing)}")

    records: list[DatRecord] = []
    for row in frame[["uid", "is_pathologic"]].itertuples(index=False):
        uid = str(row.uid)
        if not uid or Path(uid).name != uid:
            raise ValueError("DaT UIDs must be simple file stems without path separators.")
        value = float(row.is_pathologic)
        if value not in (0.0, 1.0):
            raise ValueError("is_pathologic labels must be 0.0 or 1.0.")
        nifti_path = niftis_dir / f"{uid}.nii.gz"
        if not nifti_path.is_file():
            raise FileNotFoundError(f"NIfTI file for a labeled record is missing under niftis/: {uid}.nii.gz")
        records.append(DatRecord(uid=uid, label=int(value), path=str(nifti_path)))
    if not records:
        raise ValueError("train_labels.csv contains no labeled DaT records.")
    return records


def _canonical_array_and_spacing(path: str | os.PathLike) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = nib.as_closest_canonical(nib.load(str(path)))
    array = np.asarray(image.dataobj, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("DaT NIfTI scans must be three-dimensional.")
    spacing_xyz = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    if spacing_xyz.shape != (3,) or not np.isfinite(spacing_xyz).all() or np.any(spacing_xyz <= 0):
        raise ValueError("DaT NIfTI scan has invalid voxel spacing.")
    # NIfTI is conventionally X,Y,Z; the model uses D,H,W = Z,Y,X.
    return np.transpose(array, (2, 1, 0)), tuple(float(v) for v in spacing_xyz[::-1])


def estimate_target_spacing(
    records: Sequence[DatRecord],
    rounding: float | None = 0.5,
) -> tuple[float, float, float]:
    """Estimate spacing from labeled training records only."""
    if not records:
        raise ValueError("Cannot estimate target spacing from an empty record list.")
    spacings = np.asarray([_canonical_array_and_spacing(record.path)[1] for record in records], dtype=np.float64)
    median = np.median(spacings, axis=0)
    if rounding and rounding > 0:
        median = np.round(median / float(rounding)) * float(rounding)
    median = np.maximum(median, 1e-3)
    return tuple(float(v) for v in median)


def default_preprocessing_config(
    records: Sequence[DatRecord] | None = None,
    target_spacing: Sequence[float] | None = None,
    target_shape: Sequence[int] = DEFAULT_TARGET_SHAPE,
    lower_percentile: float = DEFAULT_PERCENTILES[0],
    upper_percentile: float = DEFAULT_PERCENTILES[1],
    foreground_threshold: float = DEFAULT_FOREGROUND_THRESHOLD,
    crop_margin_mm: float = DEFAULT_CROP_MARGIN_MM,
) -> dict[str, Any]:
    spacing = tuple(target_spacing) if target_spacing is not None else estimate_target_spacing(records or [])
    spacing = parse_target_spacing(spacing)
    shape = parse_target_shape(target_shape)
    lower = float(lower_percentile)
    upper = float(upper_percentile)
    if not 0 <= lower < upper <= 100:
        raise ValueError("Intensity percentiles must satisfy 0 <= lower < upper <= 100.")
    if foreground_threshold < 0 or crop_margin_mm < 0:
        raise ValueError("foreground_threshold and crop_margin_mm must be non-negative.")
    return {
        "target_spacing": list(spacing),
        "target_shape": list(shape),
        "intensity_lower_percentile": lower,
        "intensity_upper_percentile": upper,
        "foreground_threshold": float(foreground_threshold),
        "crop_margin_mm": float(crop_margin_mm),
        "canonical_orientation": "closest_canonical_RAS",
        "array_order": "DHW",
        "normalization": "positive_foreground_percentile_clip_to_0_1",
    }


def _resize_to_shape(array: np.ndarray, shape: tuple[int, int, int], order: int) -> np.ndarray:
    if tuple(array.shape) == tuple(shape):
        return array
    factors = tuple(float(new) / float(old) for new, old in zip(shape, array.shape))
    resized = zoom(array, factors, order=order, mode="nearest", prefilter=False)
    fixed = np.zeros(shape, dtype=resized.dtype)
    common = tuple(min(a, b) for a, b in zip(resized.shape, shape))
    src_slices = tuple(slice(0, size) for size in common)
    fixed[src_slices] = resized[src_slices]
    return fixed


def _resample(array: np.ndarray, spacing: tuple[float, float, float], target_spacing: tuple[float, float, float]) -> tuple[np.ndarray, tuple[float, float, float]]:
    target_shape = tuple(max(1, int(round(dim * old / new))) for dim, old, new in zip(array.shape, spacing, target_spacing))
    return _resize_to_shape(array, target_shape, order=1), target_spacing


def _crop_pad_center(array: np.ndarray, target_shape: tuple[int, int, int], center: tuple[int, int, int], fill: float = 0.0) -> np.ndarray:
    output = np.full(target_shape, fill, dtype=array.dtype)
    src_slices = []
    dst_slices = []
    for axis, (source_size, target_size) in enumerate(zip(array.shape, target_shape)):
        start = int(round(center[axis] - target_size / 2.0))
        src_start = max(0, start)
        src_end = min(source_size, start + target_size)
        dst_start = max(0, -start)
        dst_end = dst_start + max(0, src_end - src_start)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))
    output[tuple(dst_slices)] = array[tuple(src_slices)]
    return output


def preprocess_nifti(
    path: str | os.PathLike,
    config: dict[str, Any],
    *,
    return_foreground: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply the frozen NIfTI -> [1,D,H,W] preprocessing pipeline."""
    target_spacing = parse_target_spacing(config["target_spacing"])
    target_shape = parse_target_shape(config["target_shape"])
    lower = float(config.get("intensity_lower_percentile", DEFAULT_PERCENTILES[0]))
    upper = float(config.get("intensity_upper_percentile", DEFAULT_PERCENTILES[1]))
    threshold = float(config.get("foreground_threshold", DEFAULT_FOREGROUND_THRESHOLD))
    crop_margin_mm = float(config.get("crop_margin_mm", DEFAULT_CROP_MARGIN_MM))

    volume, spacing = _canonical_array_and_spacing(path)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    volume = np.maximum(volume, 0.0)
    volume, _ = _resample(volume, spacing, target_spacing)
    foreground = volume > threshold

    if foreground.any():
        coords = np.argwhere(foreground)
        margin = tuple(int(round(crop_margin_mm / spacing_axis)) for spacing_axis in target_spacing)
        low = np.maximum(coords.min(axis=0) - np.asarray(margin), 0)
        high = np.minimum(coords.max(axis=0) + 1 + np.asarray(margin), np.asarray(volume.shape))
        cropped = volume[tuple(slice(int(a), int(b)) for a, b in zip(low, high))]
        cropped_foreground = foreground[tuple(slice(int(a), int(b)) for a, b in zip(low, high))]
    else:
        cropped = volume
        cropped_foreground = foreground

    center = tuple((np.asarray(cropped.shape) - 1) // 2)
    fixed = _crop_pad_center(cropped, target_shape, center, fill=0.0)
    fixed_foreground = _crop_pad_center(cropped_foreground.astype(np.uint8), target_shape, center, fill=0).astype(bool)

    positive = fixed[fixed_foreground & np.isfinite(fixed)]
    if positive.size == 0:
        normalized = np.zeros(target_shape, dtype=np.float32)
    else:
        lo = float(np.percentile(positive, lower))
        hi = float(np.percentile(positive, upper))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = max(float(np.max(positive)), lo + 1e-6)
        normalized = np.zeros(target_shape, dtype=np.float32)
        normalized[fixed_foreground] = np.clip((fixed[fixed_foreground] - lo) / (hi - lo), 0.0, 1.0)

    image_tensor = torch.from_numpy(normalized[None].astype(np.float32, copy=False))
    foreground_tensor = torch.from_numpy(fixed_foreground.astype(np.bool_, copy=False))
    if return_foreground:
        return image_tensor, foreground_tensor
    return image_tensor


def _config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


class DatDataset(Dataset):
    """A labeled DaT dataset with deterministic preprocessing and optional mild augmentation."""

    def __init__(
        self,
        records: Sequence[DatRecord],
        preprocessing_config: dict[str, Any],
        *,
        train: bool = False,
        augment: bool = False,
        seed: int = 42,
        cache_dir: str | os.PathLike | None = None,
    ):
        self.records = list(records)
        self.preprocessing_config = dict(preprocessing_config)
        self.train = bool(train)
        self.augment = bool(augment and train)
        self.seed = int(seed)
        self.epoch = 0
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._config_hash = _config_hash(self.preprocessing_config)

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _cache_path(self, index: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        record = self.records[index]
        digest = hashlib.sha256(f"{record.path}|{self._config_hash}".encode()).hexdigest()
        return self.cache_dir / f"{digest}.pt"

    def _load_processed(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache_path = self._cache_path(index)
        if cache_path and cache_path.is_file():
            payload = torch.load(cache_path, map_location="cpu")
            if isinstance(payload, dict) and "image" in payload and "foreground" in payload:
                return payload["image"].float(), payload["foreground"].bool()
        image, foreground = preprocess_nifti(
            self.records[index].path,
            self.preprocessing_config,
            return_foreground=True,
        )
        if cache_path:
            tmp = cache_path.with_suffix(f".tmp.{os.getpid()}")
            torch.save({"image": image.cpu(), "foreground": foreground.cpu()}, tmp)
            os.replace(tmp, cache_path)
        return image, foreground

    def _augment_pair(self, image: torch.Tensor, foreground: torch.Tensor, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.augment:
            return image, foreground
        generator = torch.Generator().manual_seed(self.seed + 1009 * self.epoch + int(index))
        for dim in (1, 2, 3):
            if bool(torch.rand((), generator=generator) < 0.5):
                image = torch.flip(image, dims=(dim,))
                foreground = torch.flip(foreground, dims=(dim - 1,))
        if bool(torch.rand((), generator=generator) < 0.25):
            factor = float(torch.empty((), dtype=torch.float32).uniform_(0.9, 1.1, generator=generator))
            image = (image * factor).clamp(0.0, 1.0)
        return image, foreground

    def get_foreground_mask(self, index: int) -> torch.Tensor:
        _, foreground = self._load_processed(int(index))
        if self.augment:
            _, foreground = self._augment_pair(*self._load_processed(int(index)), int(index))
        return foreground

    def __getitem__(self, index: int):
        image, foreground = self._load_processed(int(index))
        image, _ = self._augment_pair(image, foreground, int(index))
        return image, torch.tensor(self.records[index].label, dtype=torch.long)


def _stratified_holdout(labels: Sequence[int], val_split: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0 < float(val_split) < 1:
        return list(range(len(labels))), []
    rng = np.random.default_rng(int(seed))
    train: list[int] = []
    val: list[int] = []
    labels_array = np.asarray(labels)
    for label in sorted(set(int(v) for v in labels_array)):
        indices = np.flatnonzero(labels_array == label).tolist()
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * float(val_split)))) if len(indices) > 1 else 0
        count = min(count, max(0, len(indices) - 1))
        val.extend(indices[:count])
        train.extend(indices[count:])
    if not val and len(train) > 1:
        val.append(train.pop())
    return sorted(train), sorted(val)


def get_dat_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    *,
    val_split: float = 0.1,
    seed: int = 42,
    target_spacing: Sequence[float] | None = None,
    target_shape: Sequence[int] = DEFAULT_TARGET_SHAPE,
    intensity_lower_percentile: float = DEFAULT_PERCENTILES[0],
    intensity_upper_percentile: float = DEFAULT_PERCENTILES[1],
    foreground_threshold: float = DEFAULT_FOREGROUND_THRESHOLD,
    crop_margin_mm: float = DEFAULT_CROP_MARGIN_MM,
    cache_dir: str | os.PathLike | None = None,
    augment: bool = True,
    _train_only: bool = False,
    **kwargs,
):
    del kwargs
    records = load_dat_records(data_dir)
    preprocessing = default_preprocessing_config(
        records,
        target_spacing=target_spacing,
        target_shape=target_shape,
        lower_percentile=intensity_lower_percentile,
        upper_percentile=intensity_upper_percentile,
        foreground_threshold=foreground_threshold,
        crop_margin_mm=crop_margin_mm,
    )
    train_indices, val_indices = _stratified_holdout([record.label for record in records], val_split, seed)
    train_ds = DatDataset(
        records,
        preprocessing,
        train=True,
        augment=bool(augment),
        seed=seed,
        cache_dir=cache_dir,
    )
    eval_ds = DatDataset(records, preprocessing, train=False, augment=False, seed=seed, cache_dir=cache_dir)
    if _train_only:
        return DataLoader(Subset(train_ds, train_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_ds = Subset(eval_ds, val_indices) if val_indices else None
    test_ds = eval_ds
    worker_args = {"pin_memory": True}
    if num_workers > 0:
        worker_args.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(Subset(train_ds, train_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers, **worker_args)
    val_loader = DataLoader(val_ds, batch_size=min(256, max(1, len(val_ds))), shuffle=False, num_workers=num_workers, **worker_args) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=min(256, max(1, len(test_ds))), shuffle=False, num_workers=num_workers, **worker_args)
    return train_loader, val_loader, test_loader


def check_dat_dataset(
    data_dir: str | os.PathLike,
    *,
    target_spacing: Sequence[float] | None = None,
    target_shape: Sequence[int] = DEFAULT_TARGET_SHAPE,
    limit: int = 0,
) -> dict[str, Any]:
    """Validate labels, NIfTI discovery, preprocessing, and fixed tensor shape."""
    records = load_dat_records(data_dir)
    config = default_preprocessing_config(
        records,
        target_spacing=target_spacing or estimate_target_spacing(records),
        target_shape=target_shape,
    )
    checked = 0
    for record in records[: int(limit) or len(records)]:
        tensor = preprocess_nifti(record.path, config)
        if tuple(tensor.shape) != (1, *config["target_shape"]):
            raise RuntimeError(f"Unexpected preprocessed shape for {record.uid}: {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Preprocessing produced non-finite values for {record.uid}.")
        checked += 1
    labels = [record.label for record in records]
    return {
        "status": "ok", "records": len(records), "checked": checked,
        "class_counts": {str(label): int(labels.count(label)) for label in sorted(set(labels))},
        "target_spacing": config["target_spacing"], "target_shape": config["target_shape"],
    }
