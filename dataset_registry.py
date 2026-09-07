import os
import zipfile
import math
import re
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Callable, Dict, Any, Tuple, Optional, List
from logger import get_logger
from dat_preprocessing import get_dat_dataloaders

DatasetLoaderFunc = Callable[..., Tuple[DataLoader, Optional[DataLoader], DataLoader]]

logger = get_logger(__name__)


def _prepend_grayscale_transform(transform, grayscale: bool):
    if not grayscale:
        return transform
    grayscale_tfm = T.Grayscale(num_output_channels=3)
    if transform is None:
        return T.Compose([grayscale_tfm])
    if isinstance(transform, T.Compose):
        return T.Compose([grayscale_tfm] + list(transform.transforms))
    return T.Compose([grayscale_tfm, transform])


def _filter_dataset_by_regex(dataset: Dataset, include_regex: Optional[str]) -> Dataset:
    if not include_regex:
        return dataset
    pattern = re.compile(include_regex)

    if isinstance(dataset, Subset):
        base = dataset.dataset
        if hasattr(base, "samples"):
            indices = [idx for idx in dataset.indices if pattern.search(str(base.samples[idx][0]))]
            if not indices:
                raise ValueError(f"include_regex='{include_regex}' filtered out all samples.")
            return Subset(base, indices)
        return dataset

    if hasattr(dataset, "samples"):
        indices = [i for i, (path, _) in enumerate(dataset.samples) if pattern.search(str(path))]
        if not indices:
            raise ValueError(f"include_regex='{include_regex}' filtered out all samples.")
        return Subset(dataset, indices)

    return dataset


def _iter_dirs_with_max_depth(root: Path, max_depth: int):
    if not root.exists() or not root.is_dir():
        return

    stack: List[Tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        yield current, depth
        if depth >= max_depth:
            continue

        try:
            children = [p for p in current.iterdir() if p.is_dir() and not p.name.startswith(".")]
        except OSError:
            continue

        for child in sorted(children, key=lambda p: p.name.lower(), reverse=True):
            stack.append((child, depth + 1))


def _collect_zip_files(search_root: Path, max_depth: int = 2) -> List[Path]:
    zip_paths: List[Path] = []
    for directory, _ in _iter_dirs_with_max_depth(search_root, max_depth):
        try:
            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == ".zip":
                    zip_paths.append(file_path)
        except OSError:
            continue
    return zip_paths


def _select_latest_path(paths: List[Path]) -> Path:
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _resolve_drive_zip_archive_path(data_dir: str, **kwargs) -> Path:
    explicit_zip = (
        kwargs.get("drive_zip_path")
        or kwargs.get("zip_path")
        or kwargs.get("archive_path")
        or kwargs.get("dataset_zip_path")
        or os.environ.get("DRIVE_DATASET_ZIP")
    )
    if explicit_zip:
        explicit_zip_str = str(explicit_zip).strip()
        if explicit_zip_str.startswith("http://") or explicit_zip_str.startswith("https://"):
            if "drive.google.com/drive/home" in explicit_zip_str:
                raise ValueError(
                    "Google Drive home URLs are not direct file links. Mount Drive in Colab and pass a local ZIP path "
                    "via DRIVE_DATASET_ZIP or drive_zip_path."
                )
            raise ValueError(
                "ZIP URL inputs are unsupported here. Mount Google Drive in Colab and use a local ZIP path instead."
            )

        explicit_zip_path = Path(explicit_zip_str).expanduser()
        if explicit_zip_path.is_file():
            return explicit_zip_path
        raise FileNotFoundError(f"Configured dataset ZIP path does not exist: '{explicit_zip_path}'.")

    data_root = Path(data_dir)
    local_zip_candidates = _collect_zip_files(data_root, max_depth=2)
    if local_zip_candidates:
        if len(local_zip_candidates) > 1:
            logger.info(
                "Found %d ZIP files under '%s'; using the most recently modified: %s",
                len(local_zip_candidates),
                data_root,
                _select_latest_path(local_zip_candidates),
            )
        return _select_latest_path(local_zip_candidates)

    colab_drive_root = Path("/content/drive")
    if colab_drive_root.exists():
        shared_candidates: List[Path] = []
        for candidate_root in [Path("/content/drive/MyDrive"), Path("/content/drive/Shareddrives")]:
            if candidate_root.exists():
                shared_candidates.extend(_collect_zip_files(candidate_root, max_depth=2))

        if len(shared_candidates) == 1:
            return shared_candidates[0]
        if len(shared_candidates) > 1:
            sample_candidates = ", ".join(str(p) for p in shared_candidates[:5])
            raise ValueError(
                "Multiple ZIP files were found on mounted Google Drive. "
                f"Set DRIVE_DATASET_ZIP (or drive_zip_path) explicitly. Example candidates: {sample_candidates}"
            )

    raise FileNotFoundError(
        "No dataset ZIP archive was found. Place one under data_dir, or set DRIVE_DATASET_ZIP to the mounted path "
        "in Colab (for example /content/drive/MyDrive/<folder>/<file>.zip)."
    )


def _extract_zip_to_cache(zip_path: Path, extract_root: Path) -> Path:
    extract_root.mkdir(parents=True, exist_ok=True)
    target_root = extract_root / zip_path.stem
    ready_marker = target_root / ".extracted_ok"

    if ready_marker.exists() and target_root.is_dir():
        return target_root

    target_root.mkdir(parents=True, exist_ok=True)
    has_existing_content = any(target_root.iterdir())
    if not has_existing_content:
        logger.info("Extracting dataset ZIP '%s' into '%s'", zip_path, target_root)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_root)

    ready_marker.touch(exist_ok=True)
    return target_root


def _collapse_single_child_directory(root: Path, max_steps: int = 4) -> Path:
    current = root
    for _ in range(max_steps):
        try:
            children = [p for p in current.iterdir() if p.is_dir() and not p.name.startswith(".")]
            files = [p for p in current.iterdir() if p.is_file() and p.name != ".extracted_ok"]
        except OSError:
            break

        if len(children) == 1 and not files:
            current = children[0]
            continue
        break
    return current


def _is_valid_imagefolder_root(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    try:
        ds = torchvision.datasets.ImageFolder(root=str(root))
    except Exception:
        return False
    return len(ds) > 0 and len(ds.classes) > 0


def _find_split_layout_root(search_root: Path, max_depth: int = 4) -> Tuple[Optional[str], Optional[Path]]:
    for candidate, _ in _iter_dirs_with_max_depth(search_root, max_depth=max_depth):
        train_dir = candidate / "train"
        val_dir = candidate / "val"
        test_dir = candidate / "test"

        has_train = _is_valid_imagefolder_root(train_dir)
        has_test = _is_valid_imagefolder_root(test_dir)
        has_val = _is_valid_imagefolder_root(val_dir)

        if has_train and has_test and has_val:
            return "train_val_test", candidate
        if has_train and has_test:
            return "train_test", candidate

    return None, None


def _find_single_imagefolder_root(search_root: Path, max_depth: int = 4) -> Optional[Path]:
    for candidate, _ in _iter_dirs_with_max_depth(search_root, max_depth=max_depth):
        if (candidate / "train").is_dir() or (candidate / "test").is_dir() or (candidate / "val").is_dir():
            continue
        if _is_valid_imagefolder_root(candidate):
            return candidate
    return None


def _resolve_drive_image_layout(search_root: Path) -> Tuple[str, Path]:
    collapsed_root = _collapse_single_child_directory(search_root)

    layout, split_root = _find_split_layout_root(collapsed_root, max_depth=4)
    if layout and split_root is not None:
        return layout, split_root

    single_root = _find_single_imagefolder_root(collapsed_root, max_depth=4)
    if single_root is not None:
        return "single", single_root

    raise FileNotFoundError(
        "Unable to detect dataset layout in extracted ZIP. Supported layouts are: "
        "(1) train/val/test with class subfolders, "
        "(2) train/test with class subfolders, "
        "or (3) a single class-folder root (auto split into train/val/test)."
    )


def _stratified_split_train_val_test_indices(
    labels: List[int],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    if len(labels) < 3:
        raise ValueError("At least 3 labeled samples are required to create train/val/test splits.")
    if val_ratio <= 0.0 or test_ratio <= 0.0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError(
            f"Invalid split ratios: val_ratio={val_ratio}, test_ratio={test_ratio}. "
            "Require val_ratio>0, test_ratio>0, and val_ratio+test_ratio<1."
        )

    class_to_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        class_to_indices.setdefault(int(label), []).append(idx)

    g = torch.Generator().manual_seed(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for class_id in sorted(class_to_indices.keys()):
        class_indices = class_to_indices[class_id]
        shuffled = [class_indices[i] for i in torch.randperm(len(class_indices), generator=g).tolist()]
        n = len(shuffled)

        if n >= 3:
            n_val = max(1, int(math.floor(n * val_ratio)))
            n_test = max(1, int(math.floor(n * test_ratio)))
            while (n_val + n_test) > (n - 1):
                if n_val >= n_test and n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break
            n_train = n - n_val - n_test
        elif n == 2:
            n_train = 1
            assign_to_val = bool(torch.randint(0, 2, (1,), generator=g).item())
            n_val = 1 if assign_to_val else 0
            n_test = 1 - n_val
        else:
            n_train = 1
            n_val = 0
            n_test = 0

        train_idx.extend(shuffled[:n_train])
        val_idx.extend(shuffled[n_train:n_train + n_val])
        test_idx.extend(shuffled[n_train + n_val:n_train + n_val + n_test])

    if not val_idx and len(train_idx) > 1:
        val_idx.append(train_idx.pop())
    if not test_idx and len(train_idx) > 1:
        test_idx.append(train_idx.pop())
    if not train_idx:
        raise ValueError("Stratified split left no training samples.")
    if not val_idx or not test_idx:
        raise ValueError("Unable to create non-empty val/test splits from provided labels.")

    train_idx = [train_idx[i] for i in torch.randperm(len(train_idx), generator=g).tolist()]
    val_idx = [val_idx[i] for i in torch.randperm(len(val_idx), generator=g).tolist()]
    test_idx = [test_idx[i] for i in torch.randperm(len(test_idx), generator=g).tolist()]
    return train_idx, val_idx, test_idx


def _split_train_val(
    train_ds,
    val_source_ds,
    val_split: float,
    seed: int,
):
    if not val_split or val_split <= 0.0:
        return train_ds, None
    train_size = int(len(train_ds) * (1.0 - val_split))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_ds), generator=g).tolist()
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]
    return Subset(train_ds, train_idx), Subset(val_source_ds, val_idx)


def _build_dataloaders(
    train_ds,
    val_ds,
    test_ds,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    worker_kwargs = _dataloader_worker_kwargs(num_workers)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        **worker_kwargs,
    )
    val_dl = (
        DataLoader(
            val_ds,
            batch_size=256,
            shuffle=False,
            num_workers=num_workers,
            **worker_kwargs,
        )
        if val_ds is not None
        else None
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        **worker_kwargs,
    )
    return train_dl, val_dl, test_dl


def _dataloader_worker_kwargs(num_workers: int) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"pin_memory": True}
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def _build_train_dataloader(train_ds, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        **_dataloader_worker_kwargs(num_workers),
    )

def _cifar100_loader(data_dir: str, batch_size: int, num_workers: int,val_split: float = 0.0, 
    seed: int = 42, **kwargs) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    grayscale = bool(kwargs.get("grayscale", False))
    deterministic_train_transforms = bool(kwargs.get("deterministic_train_transforms", False))
    train_only = bool(kwargs.get("_train_only", False))
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    if deterministic_train_transforms:
        train_tfms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        train_tfms = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    test_tfms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    train_tfms = _prepend_grayscale_transform(train_tfms, grayscale)
    test_tfms = _prepend_grayscale_transform(test_tfms, grayscale)
    train_ds = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_tfms
    )
    if train_only:
        train_ds, _ = _split_train_val(train_ds, train_ds, val_split, seed)
        return _build_train_dataloader(train_ds, batch_size, num_workers)
    train_val_ds = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=test_tfms
    )
    test_ds = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_tfms
    )
    train_ds, val_ds = _split_train_val(train_ds, train_val_ds, val_split, seed)
    if val_ds is None:
        logger.info("Caution: No validation split specified")
    return _build_dataloaders(train_ds, val_ds, test_ds, batch_size, num_workers)


def _drive_zip_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.1,
    seed: int = 42,
    **kwargs,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    grayscale = bool(kwargs.get("grayscale", False))
    include_regex = kwargs.get("include_regex")
    deterministic_train_transforms = bool(kwargs.get("deterministic_train_transforms", False))
    train_only = bool(kwargs.get("_train_only", False))
    image_size = int(kwargs.get("image_size", 224))
    if image_size <= 0:
        raise ValueError(f"Invalid image_size={image_size}. Expected a positive integer.")

    resize_size = int(round(image_size / 0.875))
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if deterministic_train_transforms:
        train_tfms = T.Compose([
            T.Resize(resize_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        train_tfms = T.Compose([
            T.Resize(resize_size),
            T.RandomResizedCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    test_tfms = T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    train_tfms = _prepend_grayscale_transform(train_tfms, grayscale)
    test_tfms = _prepend_grayscale_transform(test_tfms, grayscale)

    zip_path = _resolve_drive_zip_archive_path(data_dir, **kwargs)
    extracted_root = _extract_zip_to_cache(zip_path, Path(data_dir) / "_drive_zip_extracted")
    layout, dataset_root = _resolve_drive_image_layout(extracted_root)

    logger.info("drive_zip resolved layout '%s' at '%s'", layout, dataset_root)

    if layout == "train_val_test":
        train_ds = torchvision.datasets.ImageFolder(root=str(dataset_root / "train"), transform=train_tfms)
        train_ds = _filter_dataset_by_regex(train_ds, include_regex)
        if train_only:
            return _build_train_dataloader(train_ds, batch_size, num_workers)

        val_ds = torchvision.datasets.ImageFolder(root=str(dataset_root / "val"), transform=test_tfms)
        test_ds = torchvision.datasets.ImageFolder(root=str(dataset_root / "test"), transform=test_tfms)
        val_ds = _filter_dataset_by_regex(val_ds, include_regex)
        test_ds = _filter_dataset_by_regex(test_ds, include_regex)
        if val_split and val_split > 0.0:
            logger.info("drive_zip explicit train/val/test split detected; ignoring val_split=%s.", val_split)
        return _build_dataloaders(train_ds, val_ds, test_ds, batch_size, num_workers)

    if layout == "train_test":
        train_dir = dataset_root / "train"
        test_dir = dataset_root / "test"
        train_ds = torchvision.datasets.ImageFolder(root=str(train_dir), transform=train_tfms)
        train_ds = _filter_dataset_by_regex(train_ds, include_regex)
        if train_only:
            train_ds, _ = _split_train_val(train_ds, train_ds, val_split, seed)
            return _build_train_dataloader(train_ds, batch_size, num_workers)

        train_val_ds = torchvision.datasets.ImageFolder(root=str(train_dir), transform=test_tfms)
        test_ds = torchvision.datasets.ImageFolder(root=str(test_dir), transform=test_tfms)
        train_val_ds = _filter_dataset_by_regex(train_val_ds, include_regex)
        test_ds = _filter_dataset_by_regex(test_ds, include_regex)
        train_ds, val_ds = _split_train_val(train_ds, train_val_ds, val_split, seed)
        if val_ds is None:
            logger.info("drive_zip train/test layout detected; no validation split requested.")
        return _build_dataloaders(train_ds, val_ds, test_ds, batch_size, num_workers)

    full_train_ds = torchvision.datasets.ImageFolder(root=str(dataset_root), transform=train_tfms)
    full_train_ds = _filter_dataset_by_regex(full_train_ds, include_regex)

    eval_split = val_split if (val_split and val_split > 0.0) else 0.1
    if not val_split or val_split <= 0.0:
        logger.info("No val_split provided for drive_zip single-root dataset; defaulting val_split to %.2f.", eval_split)

    test_split = float(kwargs.get("test_split", eval_split))
    if test_split <= 0.0:
        test_split = eval_split

    if isinstance(full_train_ds, Subset) and hasattr(full_train_ds.dataset, "samples"):
        labels = [int(full_train_ds.dataset.samples[i][1]) for i in full_train_ds.indices]
    elif hasattr(full_train_ds, "samples"):
        labels = [int(label) for _, label in full_train_ds.samples]
    elif hasattr(full_train_ds, "targets"):
        labels = [int(label) for label in full_train_ds.targets]
    else:
        raise ValueError("Unable to extract labels for drive_zip stratified split.")
    train_idx, val_idx, test_idx = _stratified_split_train_val_test_indices(
        labels=labels,
        val_ratio=eval_split,
        test_ratio=test_split,
        seed=seed,
    )
    logger.info(
        "drive_zip single-root layout: using stratified train/val/test split (seed=%s, val=%.3f, test=%.3f).",
        seed,
        eval_split,
        test_split,
    )

    train_ds = Subset(full_train_ds, train_idx)
    if train_only:
        return _build_train_dataloader(train_ds, batch_size, num_workers)

    full_eval_ds = torchvision.datasets.ImageFolder(root=str(dataset_root), transform=test_tfms)
    full_eval_ds = _filter_dataset_by_regex(full_eval_ds, include_regex)
    val_ds = Subset(full_eval_ds, val_idx)
    test_ds = Subset(full_eval_ds, test_idx)
    return _build_dataloaders(train_ds, val_ds, test_ds, batch_size, num_workers)


def _dat_parkinsons_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.1,
    seed: int = 42,
    **kwargs,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Research-facing registry adapter for the dedicated NIfTI pipeline."""
    return get_dat_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        seed=seed,
        target_spacing=kwargs.get("target_spacing"),
        target_shape=kwargs.get("target_shape", (64, 96, 96)),
        intensity_lower_percentile=kwargs.get("intensity_lower_percentile", 1.0),
        intensity_upper_percentile=kwargs.get("intensity_upper_percentile", 99.0),
        foreground_threshold=kwargs.get("foreground_threshold", 0.0),
        crop_margin_mm=kwargs.get("crop_margin_mm", 8.0),
        cache_dir=kwargs.get("cache_dir"),
        augment=not bool(kwargs.get("deterministic_train_transforms", False)),
        _train_only=bool(kwargs.get("_train_only", False)),
    )

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "cifar100": {
        "loader": _cifar100_loader,
        "num_classes": 100,
        "default_input_size": 32,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "drive_zip": {
        "loader": _drive_zip_loader,
        "num_classes": 1000,
        "default_input_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "dat_parkinsons": {
        "loader": _dat_parkinsons_loader,
        "num_classes": 2,
        "default_input_size": 96,
        "default_input_shape": (64, 96, 96),
        "spatial_dims": 3,
        "input_channels": 1,
        "mean": (0.0,),
        "std": (1.0,),
    },
}


def _get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name]

def get_dataset_loaders(dataset_name: str, data_dir: str, batch_size: int, 
    num_workers: int, **kwargs) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    dataset_info = _get_dataset_info(dataset_name)
    loader = dataset_info["loader"]
    return loader(data_dir, batch_size, num_workers, **kwargs)


def get_train_loader(dataset_name: str, data_dir: str, batch_size: int, 
    num_workers: int, **kwargs) -> DataLoader:
    dataset_info = _get_dataset_info(dataset_name)
    loader = dataset_info["loader"]
    train_kwargs = dict(kwargs)
    train_kwargs["_train_only"] = True
    loaded = loader(data_dir, batch_size, num_workers, **train_kwargs)
    if isinstance(loaded, tuple):
        return loaded[0]
    return loaded

def get_num_classes(dataset_name: str) -> int:
    return _get_dataset_info(dataset_name)["num_classes"]


def _infer_num_classes_from_dataset(dataset: Dataset) -> Optional[int]:
    if hasattr(dataset, "num_classes"):
        try:
            n = int(getattr(dataset, "num_classes"))
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass

    if hasattr(dataset, "classes"):
        classes = getattr(dataset, "classes")
        try:
            n = len(classes)
            if n > 0:
                return n
        except TypeError:
            pass

    if isinstance(dataset, Subset):
        parent = dataset.dataset
        parent_classes = _infer_num_classes_from_dataset(parent)
        if parent_classes is not None:
            return parent_classes

        if hasattr(parent, "targets"):
            targets = getattr(parent, "targets")
            try:
                indexed_targets = [int(targets[i]) for i in dataset.indices]
                if indexed_targets:
                    return int(max(indexed_targets)) + 1
            except Exception:
                pass

    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        try:
            if len(targets) > 0:
                return int(max(int(v) for v in targets)) + 1
        except Exception:
            pass

    if hasattr(dataset, "samples"):
        samples = getattr(dataset, "samples")
        try:
            labels = {int(label) for _, label in samples}
            if labels:
                return int(max(labels)) + 1
        except Exception:
            pass

    return None


def infer_num_classes_from_loader(loader: DataLoader) -> Optional[int]:
    dataset = loader.dataset
    return _infer_num_classes_from_dataset(dataset)


def get_default_input_size(dataset_name: str) -> int:
    return int(_get_dataset_info(dataset_name)["default_input_size"])


def get_normalization_params(dataset_name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return (mean, std) normalization parameters for the given dataset."""
    info = _get_dataset_info(dataset_name)
    mean = info.get("mean", (0.485, 0.456, 0.406))
    std = info.get("std", (0.229, 0.224, 0.225))
    return mean, std


def get_dataset_metadata(dataset_name: str) -> Dict[str, Any]:
    """Return a copy of dimension and label metadata for a registered dataset."""
    info = _get_dataset_info(dataset_name)
    return {key: value for key, value in info.items() if key != "loader"}

def get_available_datasets() -> list:
    return list(DATASET_REGISTRY.keys())

def register_dataset(
    name: str,
    loader: DatasetLoaderFunc,
    num_classes: int,
    default_input_size: int = 224,
    spatial_dims: int = 2,
    input_channels: int = 3,
) -> None:
    if name in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' is already registered")
    
    DATASET_REGISTRY[name] = {
        "loader": loader,
        "num_classes": num_classes,
        "default_input_size": default_input_size,
        "spatial_dims": spatial_dims,
        "input_channels": input_channels,
    }
