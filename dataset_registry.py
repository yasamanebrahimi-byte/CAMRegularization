import os
import shutil
import subprocess
import sys
import zipfile
import csv
import math
import re
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Callable, Dict, Any, Tuple, Optional, List
from torchvision.datasets.utils import download_and_extract_archive
from logger import get_logger

DatasetLoaderFunc = Callable[..., Tuple[DataLoader, Optional[DataLoader], DataLoader]]

logger = get_logger(__name__)
KAGGLE_API_TOKEN = "KGAT_5944f37e07ad4aef74d5a89f1a0bce45"


class _ImagePathDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None, convert_mode: str = "RGB"):
        self.samples = samples
        self.transform = transform
        self.convert_mode = convert_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, target = self.samples[index]
        with Image.open(image_path) as img:
            image = img.convert(self.convert_mode)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _prepend_grayscale_transform(transform, grayscale: bool):
    if not grayscale:
        return transform
    grayscale_tfm = T.Grayscale(num_output_channels=3)
    if transform is None:
        return T.Compose([grayscale_tfm])
    if isinstance(transform, T.Compose):
        return T.Compose([grayscale_tfm] + list(transform.transforms))
    return T.Compose([grayscale_tfm, transform])


def _filter_samples_by_regex(samples: List[Tuple[str, int]], include_regex: Optional[str]) -> List[Tuple[str, int]]:
    if not include_regex:
        return samples
    pattern = re.compile(include_regex)
    filtered = [sample for sample in samples if pattern.search(str(sample[0]))]
    if not filtered:
        raise ValueError(f"include_regex='{include_regex}' filtered out all samples.")
    return filtered


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


class _MalwareBytesDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], image_size: int = 224, transform=None):
        self.samples = samples
        self.image_size = image_size
        self.transform = transform

    @staticmethod
    def _parse_hex_bytes(bytes_path: str, max_len: int) -> bytearray:
        values = bytearray()
        with open(bytes_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) <= 1:
                    continue
                for token in parts[1:]:
                    if len(values) >= max_len:
                        return values
                    if token == "??":
                        values.append(0)
                        continue
                    try:
                        values.append(int(token, 16))
                    except ValueError:
                        values.append(0)
        return values

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        bytes_path, target = self.samples[index]
        max_pixels = self.image_size * self.image_size
        values = self._parse_hex_bytes(bytes_path, max_pixels)
        if len(values) < max_pixels:
            values.extend([0] * (max_pixels - len(values)))

        image = Image.frombytes("L", (self.image_size, self.image_size), bytes(values))
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class _MalwareFeatureCsvDataset(Dataset):
    def __init__(self, samples: List[Tuple[List[float], int]], image_size: int = 224, transform=None):
        self.image_size = image_size
        self.transform = transform
        self.max_pixels = image_size * image_size
        self.samples: List[Tuple[bytes, int]] = [
            (self._feature_row_to_bytes(feature_values), target)
            for feature_values, target in samples
        ]

    def _feature_row_to_bytes(self, feature_values: List[float]) -> bytes:
        if not feature_values:
            return bytes([0] * self.max_pixels)

        row_min = min(feature_values)
        row_max = max(feature_values)
        if row_max > row_min:
            scale = 255.0 / (row_max - row_min)
            encoded = [int((value - row_min) * scale) for value in feature_values]
        else:
            encoded = [0 for _ in feature_values]

        if len(encoded) >= self.max_pixels:
            encoded = encoded[:self.max_pixels]
        else:
            encoded.extend([0] * (self.max_pixels - len(encoded)))
        return bytes(encoded)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_bytes, target = self.samples[index]
        image = Image.frombytes("L", (self.image_size, self.image_size), image_bytes)
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class _CIFAR100CDataset(Dataset):
    def __init__(
        self,
        root: str,
        corruption: str = "gaussian_noise",
        severity: int = 5,
        transform=None,
    ):
        self.root = Path(root)
        self.corruption = corruption
        self.severity = int(severity)
        self.transform = transform

        if self.severity < 1 or self.severity > 5:
            raise ValueError(f"CIFAR-100-C severity must be in [1, 5], got {self.severity}.")

        labels_path = self.root / "labels.npy"
        if not labels_path.exists():
            raise FileNotFoundError(f"CIFAR-100-C labels not found at '{labels_path}'.")

        labels = np.load(str(labels_path))
        start = (self.severity - 1) * 10000
        end = self.severity * 10000

        if self.corruption == "all":
            corruption_files = [
                p for p in sorted(self.root.glob("*.npy"))
                if p.name != "labels.npy"
            ]
            if not corruption_files:
                raise FileNotFoundError(f"No corruption .npy files found under '{self.root}'.")

            chunks = []
            targets = []
            for cpath in corruption_files:
                carr = np.load(str(cpath))
                chunks.append(carr[start:end])
                targets.append(labels[start:end])
            self.images = np.concatenate(chunks, axis=0)
            self.targets = np.concatenate(targets, axis=0)
        else:
            cpath = self.root / f"{self.corruption}.npy"
            if not cpath.exists():
                available = [p.stem for p in sorted(self.root.glob("*.npy")) if p.name != "labels.npy"]
                raise ValueError(
                    f"Unknown CIFAR-100-C corruption '{self.corruption}'. Available: {available}"
                )
            carr = np.load(str(cpath))
            self.images = carr[start:end]
            self.targets = labels[start:end]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        target = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _resolve_existing_path(*candidates: str) -> Optional[str]:
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def _find_microsoft_malware_mirror_root(search_root: Path) -> Optional[Path]:
    if not search_root.exists() or not search_root.is_dir():
        return None

    direct_csv = search_root / "train" / "LargeTrain.csv"
    if direct_csv.is_file():
        return search_root

    common_candidates = [
        search_root / "malwaremicrosoftbig",
        search_root / "Dataset",
        search_root / "Dataset" / "Dataset",
    ]
    for candidate in common_candidates:
        if (candidate / "train" / "LargeTrain.csv").is_file():
            return candidate

    max_depth = 5
    for train_csv in search_root.rglob("LargeTrain.csv"):
        try:
            rel_parts = train_csv.relative_to(search_root).parts
        except ValueError:
            continue
        if len(rel_parts) > (max_depth + 1):
            continue
        candidate_root = train_csv.parent.parent
        if candidate_root.is_dir():
            return candidate_root

    return None


def _load_microsoft_malware_mirror_samples(csv_root: Path) -> List[Tuple[List[float], int]]:
    train_csv = csv_root / "train" / "LargeTrain.csv"
    if not train_csv.is_file():
        raise FileNotFoundError(f"Expected mirror training CSV at '{train_csv}'.")

    samples: List[Tuple[List[float], int]] = []
    with open(train_csv, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Training CSV '{train_csv}' has no header.")

        feature_names = [name for name in reader.fieldnames if name and name != "Class"]
        if not feature_names:
            raise ValueError(f"Training CSV '{train_csv}' has no feature columns.")

        for row in reader:
            class_token = (row.get("Class") or "").strip()
            if not class_token:
                continue
            try:
                target = int(float(class_token)) - 1
            except ValueError:
                continue
            if target < 0:
                continue

            feature_values: List[float] = []
            for name in feature_names:
                value = (row.get(name) or "").strip()
                if not value:
                    feature_values.append(0.0)
                    continue
                try:
                    feature_values.append(float(value))
                except ValueError:
                    feature_values.append(0.0)

            samples.append((feature_values, target))

    return samples


def _build_kaggle_auth_env(token: str) -> Dict[str, str]:
    token = token.strip()
    if not token:
        raise ValueError("KAGGLE_API_TOKEN is empty.")
    if not token.startswith("KGAT_"):
        raise ValueError("KAGGLE_API_TOKEN must start with 'KGAT_'.")

    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    return env


def _ensure_kaggle_dataset_available(
    dataset_label: str,
    data_dir: str,
    kaggle_dataset: str,
    local_dir_candidates: List[str],
    archive_filename: str,
    kaggle_api_token: str,
) -> str:
    existing = _resolve_existing_path(*local_dir_candidates)
    if existing is not None:
        return existing

    if not kaggle_dataset.strip():
        raise FileNotFoundError(
            f"{dataset_label} dataset not found locally and kaggle dataset slug is empty. "
            "Pass kaggle_dataset='owner/dataset-name' in config/kwargs."
        )

    env = _build_kaggle_auth_env(kaggle_api_token)

    try:
        import kagglehub  # type: ignore

        logger.info("%s not found locally. Downloading via kagglehub dataset '%s'...", dataset_label, kaggle_dataset)
        downloaded_root = kagglehub.dataset_download(kaggle_dataset)
        if downloaded_root and os.path.isdir(downloaded_root):
            downloaded_root_path = Path(downloaded_root)
            data_path = Path(data_dir)
            data_path.mkdir(parents=True, exist_ok=True)

            for child in downloaded_root_path.iterdir():
                target = data_path / child.name
                if child.is_dir():
                    if not target.exists():
                        shutil.copytree(child, target)
                elif child.is_file():
                    if not target.exists():
                        shutil.copy2(child, target)

            existing = _resolve_existing_path(*local_dir_candidates)
            if existing is not None:
                return existing
            return downloaded_root
    except Exception as exc:
        logger.warning("kagglehub download failed, falling back to Kaggle CLI: %s", exc)

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    archive_path = data_path / archive_filename

    logger.info("%s not found locally. Downloading from Kaggle dataset '%s'...", dataset_label, kaggle_dataset)
    primary_cmd_candidates = [
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            kaggle_dataset,
            "-p",
            str(data_path),
            "-f",
            archive_path.name,
            "--force",
        ],
        [
            sys.executable,
            "-m",
            "kaggle.cli",
            "datasets",
            "download",
            "-d",
            kaggle_dataset,
            "-p",
            str(data_path),
            "-f",
            archive_path.name,
            "--force",
        ],
    ]

    fallback_cmd_candidates = [
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            kaggle_dataset,
            "-p",
            str(data_path),
            "--force",
        ],
        [
            sys.executable,
            "-m",
            "kaggle.cli",
            "datasets",
            "download",
            "-d",
            kaggle_dataset,
            "-p",
            str(data_path),
            "--force",
        ],
    ]

    last_error = None
    for cmd in primary_cmd_candidates:
        try:
            subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            last_error = None
            break
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            last_error = exc

    if last_error is not None:
        for fallback_cmd in fallback_cmd_candidates:
            try:
                subprocess.run(fallback_cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                last_error = None
                break
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                last_error = exc

    if last_error is not None:
        raise RuntimeError(
            "Kaggle download failed. Install 'kagglehub' for token-based auth (KAGGLE_API_TOKEN), "
            "or configure Kaggle CLI credentials (KAGGLE_USERNAME/KAGGLE_KEY)."
        ) from last_error

    zip_candidates = []
    if archive_path.exists():
        zip_candidates.append(archive_path)
    zip_candidates.extend(sorted(data_path.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True))

    extracted = False
    for zip_path in zip_candidates:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_path)
            extracted = True
            break
        except zipfile.BadZipFile:
            continue

    if not extracted:
        raise FileNotFoundError(
            f"Downloaded {dataset_label} archive could not be found or extracted."
        )

    existing = _resolve_existing_path(*local_dir_candidates)
    if existing is None:
        raise FileNotFoundError(
            f"{dataset_label} download completed, but no valid class-subfolder directory was found under data/."
        )
    return existing


def _ensure_archive_dataset_available(
    dataset_label: str,
    data_dir: str,
    local_dir_candidates: List[str],
    source_url: str,
    filename: str,
) -> str:
    existing = _resolve_existing_path(*local_dir_candidates)
    if existing is not None:
        return existing

    logger.info("%s not found locally. Downloading from %s", dataset_label, source_url)
    download_and_extract_archive(
        url=source_url,
        download_root=data_dir,
        extract_root=data_dir,
        filename=filename,
        remove_finished=False,
    )

    existing = _resolve_existing_path(*local_dir_candidates)
    if existing is None:
        raise FileNotFoundError(
            f"{dataset_label} download completed, but expected folders were not found under '{data_dir}'."
        )
    return existing


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


def _build_tiny_imagenet_val_samples(
    val_dir: Path,
    class_to_idx: Dict[str, int],
) -> List[Tuple[str, int]]:
    """Build (image_path, class_idx) samples for Tiny-ImageNet val split.

    Tiny-ImageNet official val split is flat (images + val_annotations.txt),
    so we read annotations instead of relying on ImageFolder class subfolders.
    """
    val_images = val_dir / "images"
    val_annotations = val_dir / "val_annotations.txt"
    if not val_images.exists() or not val_annotations.exists():
        raise FileNotFoundError(
            f"Tiny-ImageNet val split at '{val_dir}' is missing images/ or val_annotations.txt."
        )

    samples: List[Tuple[str, int]] = []
    with open(val_annotations, "r", encoding="utf-8") as f:
        for line in f:
            row = line.strip().split("\t")
            if len(row) < 2:
                continue
            image_name, class_name = row[0], row[1]
            if class_name not in class_to_idx:
                continue
            image_path = val_images / image_name
            if image_path.exists():
                samples.append((str(image_path), class_to_idx[class_name]))

    if not samples:
        raise FileNotFoundError(
            f"Tiny-ImageNet val annotations produced no valid samples in '{val_dir}'."
        )
    return samples


def _split_train_val_test_indices(
    n_items: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    if n_items < 3:
        raise ValueError("At least 3 labeled samples are required to create train/val/test splits.")
    if val_ratio <= 0.0 or test_ratio <= 0.0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError(
            f"Invalid split ratios: val_ratio={val_ratio}, test_ratio={test_ratio}. "
            "Require val_ratio>0, test_ratio>0, and val_ratio+test_ratio<1."
        )

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_items, generator=g).tolist()

    n_val = max(1, int(math.floor(n_items * val_ratio)))
    n_test = max(1, int(math.floor(n_items * test_ratio)))
    n_train = n_items - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"Split leaves no training samples: n_items={n_items}, n_val={n_val}, n_test={n_test}."
        )

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


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
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dl = (
        DataLoader(
            val_ds,
            batch_size=256,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        if val_ds is not None
        else None
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dl, val_dl, test_dl


def _build_train_dataloader(train_ds, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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


def _tiny_imagenet_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.0,
    seed: int = 42,
    **kwargs,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    grayscale = bool(kwargs.get("grayscale", False))
    include_regex = kwargs.get("include_regex")
    root = Path(data_dir)
    local_candidates = [
        str(root / "tiny-imagenet-200"),
        str(root / "tiny_imagenet"),
        str(root / "tiny-imagenet"),
        data_dir,
    ]
    tiny_root = _resolve_existing_path(*local_candidates)
    if tiny_root is None:
        tiny_root = _ensure_archive_dataset_available(
            dataset_label="Tiny-ImageNet",
            data_dir=data_dir,
            local_dir_candidates=local_candidates,
            source_url="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
            filename="tiny-imagenet-200.zip",
        )

    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2302, 0.2265, 0.2262)
    train_tfms = T.Compose([
        T.RandomCrop(64, padding=8),
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

    train_dir = Path(tiny_root) / "train"
    val_dir = Path(tiny_root) / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Tiny-ImageNet structure invalid at '{tiny_root}'. Expected train/ and val/ directories."
        )

    base_train_ds = torchvision.datasets.ImageFolder(root=str(train_dir), transform=train_tfms)
    train_ds = base_train_ds
    train_val_ds = torchvision.datasets.ImageFolder(root=str(train_dir), transform=test_tfms)
    class_to_idx = base_train_ds.class_to_idx
    train_ds = _filter_dataset_by_regex(train_ds, include_regex)
    train_val_ds = _filter_dataset_by_regex(train_val_ds, include_regex)

    val_samples = _build_tiny_imagenet_val_samples(val_dir, class_to_idx)
    val_samples = _filter_samples_by_regex(val_samples, include_regex)
    convert_mode = "L" if grayscale else "RGB"
    test_ds = _ImagePathDataset(val_samples, transform=test_tfms, convert_mode=convert_mode)

    train_ds, val_ds = _split_train_val(train_ds, train_val_ds, val_split, seed)
    if val_ds is None:
        logger.info("Using Tiny-ImageNet official val split as test set; pass val_split>0 to create train-val.")
    return _build_dataloaders(train_ds, val_ds, test_ds, batch_size, num_workers)


def _cub200_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.0,
    seed: int = 42,
    **kwargs,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    grayscale = bool(kwargs.get("grayscale", False))
    include_regex = kwargs.get("include_regex")
    root = Path(data_dir)
    local_candidates = [
        str(root / "CUB_200_2011"),
        str(root / "cub200"),
        str(root / "CUB-200-2011"),
        data_dir,
    ]
    cub_root = _resolve_existing_path(*local_candidates)
    if cub_root is None:
        cub_root = _ensure_archive_dataset_available(
            dataset_label="CUB-200-2011",
            data_dir=data_dir,
            local_dir_candidates=local_candidates,
            source_url="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
            filename="CUB_200_2011.tgz",
        )

    cub_path = Path(cub_root)
    images_txt = cub_path / "images.txt"
    split_txt = cub_path / "train_test_split.txt"
    labels_txt = cub_path / "image_class_labels.txt"
    images_dir = cub_path / "images"
    if not (images_txt.exists() and split_txt.exists() and labels_txt.exists() and images_dir.exists()):
        raise FileNotFoundError(
            f"CUB-200-2011 structure invalid at '{cub_root}'. Missing one of images.txt, train_test_split.txt, image_class_labels.txt, or images/."
        )

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tfms = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tfms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    train_tfms = _prepend_grayscale_transform(train_tfms, grayscale)
    test_tfms = _prepend_grayscale_transform(test_tfms, grayscale)

    with open(images_txt, "r", encoding="utf-8") as f:
        image_rows = [row.strip().split(" ", 1) for row in f if row.strip()]
    with open(split_txt, "r", encoding="utf-8") as f:
        split_rows = [row.strip().split(" ") for row in f if row.strip()]
    with open(labels_txt, "r", encoding="utf-8") as f:
        label_rows = [row.strip().split(" ") for row in f if row.strip()]

    id_to_relpath = {int(idx): rel_path for idx, rel_path in image_rows}
    id_to_train = {int(idx): int(is_train) == 1 for idx, is_train in split_rows}
    id_to_label = {int(idx): int(label) - 1 for idx, label in label_rows}

    train_samples = []
    test_samples = []
    for image_id, rel_path in id_to_relpath.items():
        sample = (str(images_dir / rel_path), id_to_label[image_id])
        if id_to_train[image_id]:
            train_samples.append(sample)
        else:
            test_samples.append(sample)

    train_samples = _filter_samples_by_regex(train_samples, include_regex)
    test_samples = _filter_samples_by_regex(test_samples, include_regex)
    convert_mode = "L" if grayscale else "RGB"
    train_ds = _ImagePathDataset(train_samples, transform=train_tfms, convert_mode=convert_mode)
    train_val_ds = _ImagePathDataset(train_samples, transform=test_tfms, convert_mode=convert_mode)
    test_ds = _ImagePathDataset(test_samples, transform=test_tfms, convert_mode=convert_mode)

    train_ds, val_ds = _split_train_val(train_ds, train_val_ds, val_split, seed)
    if val_ds is None:
        logger.info("Using CUB official train/test split; pass val_split>0 to create a train-val split.")
    return _build_dataloaders(train_ds, val_ds, test_ds, batch_size, num_workers)


def _malimg_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.0,
    seed: int = 42,
    **kwargs,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    grayscale = bool(kwargs.get("grayscale", False))
    include_regex = kwargs.get("include_regex")
    deterministic_train_transforms = bool(kwargs.get("deterministic_train_transforms", False))
    train_only = bool(kwargs.get("_train_only", False))
    kaggle_dataset = "manmandes/malimg"
    root = Path(data_dir)
    malimg_root = _ensure_kaggle_dataset_available(
        dataset_label="MalImg",
        data_dir=data_dir,
        kaggle_dataset=kaggle_dataset,
        local_dir_candidates=[
            str(root / "MalImg"),
            str(root / "malimg"),
            str(root / "malimg_dataset"),
        ],
        archive_filename="malimg_kaggle.zip",
        kaggle_api_token=KAGGLE_API_TOKEN,
    )

    data_root_path = Path(data_dir)
    preferred_split_root = data_root_path / "malimg_dataset"
    split_root = None
    split_root_candidates: List[Path] = [preferred_split_root]

    malimg_root_path = Path(malimg_root)
    if malimg_root_path != preferred_split_root:
        split_root_candidates.append(malimg_root_path)
    if malimg_root_path.is_dir():
        split_root_candidates.extend([p for p in malimg_root_path.iterdir() if p.is_dir()])

    for candidate in split_root_candidates:
        train_dir = candidate / "train"
        val_dir = candidate / "val"
        test_dir = candidate / "test"
        if train_dir.is_dir() and val_dir.is_dir() and test_dir.is_dir():
            split_root = candidate
            break

    if split_root is None:
        raise FileNotFoundError(
            "MalImg requires an explicit split directory structure with train/val/test. "
            f"Expected under '{preferred_split_root}', or nested in '{malimg_root_path}'."
        )

    train_dir = split_root / "train"
    val_dir = split_root / "val"
    test_dir = split_root / "test"

    probe_ds = torchvision.datasets.ImageFolder(root=str(train_dir))
    if len(probe_ds) == 0:
        raise ValueError(f"No images found in '{train_dir}'.")

    image_size = int(kwargs.get("image_size", 224))
    if image_size <= 0:
        raise ValueError(f"Invalid image_size={image_size}. Expected a positive integer.")

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if deterministic_train_transforms:
        train_tfms = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        train_tfms = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    test_tfms = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    train_tfms = _prepend_grayscale_transform(train_tfms, grayscale)
    test_tfms = _prepend_grayscale_transform(test_tfms, grayscale)

    train_ds = torchvision.datasets.ImageFolder(root=str(train_dir), transform=train_tfms)
    train_ds = _filter_dataset_by_regex(train_ds, include_regex)
    if train_only:
        return _build_train_dataloader(train_ds, batch_size, num_workers)

    val_ds = torchvision.datasets.ImageFolder(root=str(val_dir), transform=test_tfms)
    test_ds = torchvision.datasets.ImageFolder(root=str(test_dir), transform=test_tfms)
    val_ds = _filter_dataset_by_regex(val_ds, include_regex)
    test_ds = _filter_dataset_by_regex(test_ds, include_regex)

    if val_split and val_split > 0.0:
        logger.info("MalImg explicit train/val/test split detected; ignoring val_split=%s.", val_split)
    return _build_dataloaders(train_ds, val_ds, test_ds, batch_size, num_workers)


def _malware_classification_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.1,
    seed: int = 42,
    **kwargs,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    mirror_dataset_slug = "muhammad4hmed/malwaremicrosoftbig"
    root = Path(data_dir)
    mirror_root = _find_microsoft_malware_mirror_root(root)
    if mirror_root is None:
        mirror_download_root = _ensure_kaggle_dataset_available(
            dataset_label="Microsoft Malware Classification mirror",
            data_dir=data_dir,
            kaggle_dataset=mirror_dataset_slug,
            local_dir_candidates=[
                str(root / "malwaremicrosoftbig"),
                str(root / "malwaremicrosoftbig" / "Dataset"),
                str(root / "malwaremicrosoftbig" / "Dataset" / "Dataset"),
                str(root / "Dataset" / "Dataset"),
            ],
            archive_filename="malwaremicrosoftbig.zip",
            kaggle_api_token=KAGGLE_API_TOKEN,
        )
        mirror_root = _find_microsoft_malware_mirror_root(Path(mirror_download_root))
        if mirror_root is None:
            mirror_root = _find_microsoft_malware_mirror_root(root)

    if mirror_root is None:
        has_malimg = (root / "malimg_dataset").is_dir()
        hint = (
            "Detected 'malimg_dataset' under data_dir; if that is your intended dataset, use --dataset malimg. "
            if has_malimg
            else ""
        )
        raise FileNotFoundError(
            "Unable to prepare Microsoft Malware Classification mirror dataset. "
            "Expected train/LargeTrain.csv under the extracted mirror structure. "
            f"Current --data_dir is '{root}'. {hint}"
        )

    csv_feature_samples = _load_microsoft_malware_mirror_samples(mirror_root)
    if not csv_feature_samples:
        raise FileNotFoundError(
            f"No labeled rows were loaded from mirror CSV under '{mirror_root}'."
        )

    eval_split = val_split if (val_split and val_split > 0.0) else 0.1
    if not val_split or val_split <= 0.0:
        logger.info("No val_split provided for malware_classification; defaulting val_split to %.2f.", eval_split)

    test_split = kwargs.get("test_split", eval_split)
    if test_split <= 0.0:
        test_split = eval_split

    image_size = int(kwargs.get("image_size", 224))
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tfms = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tfms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    all_samples_count = len(csv_feature_samples)

    train_idx, val_idx, test_idx = _split_train_val_test_indices(
        n_items=all_samples_count,
        val_ratio=eval_split,
        test_ratio=test_split,
        seed=seed,
    )

    full_train_ds = _MalwareFeatureCsvDataset(csv_feature_samples, image_size=image_size, transform=train_tfms)
    full_eval_ds = _MalwareFeatureCsvDataset(csv_feature_samples, image_size=image_size, transform=test_tfms)

    train_ds = Subset(full_train_ds, train_idx)
    val_ds = Subset(full_eval_ds, val_idx)
    test_ds = Subset(full_eval_ds, test_idx)
    return _build_dataloaders(train_ds, val_ds, test_ds, batch_size, num_workers)


def _imagenette_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.0,
    seed: int = 42,
    **kwargs,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    grayscale = bool(kwargs.get("grayscale", False))
    if not hasattr(torchvision.datasets, "Imagenette"):
        raise RuntimeError(
            "torchvision.datasets.Imagenette is unavailable in this torchvision version. "
            "Upgrade torchvision to use the imagenette dataset."
        )

    imagenette_size = kwargs.get("imagenette_size", "320px")
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tfms = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tfms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    train_tfms = _prepend_grayscale_transform(train_tfms, grayscale)
    test_tfms = _prepend_grayscale_transform(test_tfms, grayscale)

    train_ds = torchvision.datasets.Imagenette(
        root=data_dir,
        split="train",
        size=imagenette_size,
        download=True,
        transform=train_tfms,
    )
    train_val_ds = torchvision.datasets.Imagenette(
        root=data_dir,
        split="train",
        size=imagenette_size,
        download=True,
        transform=test_tfms,
    )
    test_ds = torchvision.datasets.Imagenette(
        root=data_dir,
        split="val",
        size=imagenette_size,
        download=True,
        transform=test_tfms,
    )

    train_ds, val_ds = _split_train_val(train_ds, train_val_ds, val_split, seed)
    return _build_dataloaders(train_ds, val_ds, test_ds, batch_size, num_workers)


def _cifar100c_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.1,
    seed: int = 42,
    **kwargs,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    grayscale = bool(kwargs.get("grayscale", False))
    corruption = kwargs.get("corruption", "gaussian_noise")
    severity = int(kwargs.get("severity", 5))

    cifar100c_root = _ensure_archive_dataset_available(
        dataset_label="CIFAR-100-C",
        data_dir=data_dir,
        local_dir_candidates=[
            str(Path(data_dir) / "CIFAR-100-C"),
            str(Path(data_dir) / "cifar-100-c"),
        ],
        source_url="https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1",
        filename="CIFAR-100-C.tar",
    )

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
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
        root=data_dir,
        train=True,
        download=True,
        transform=train_tfms,
    )
    train_val_ds = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=test_tfms,
    )
    test_ds = _CIFAR100CDataset(
        root=cifar100c_root,
        corruption=corruption,
        severity=severity,
        transform=test_tfms,
    )

    train_ds, val_ds = _split_train_val(train_ds, train_val_ds, val_split, seed)
    if val_ds is None:
        logger.info("CIFAR-100-C run uses clean train split and corruption test split only.")
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

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "cifar100": {
        "loader": _cifar100_loader,
        "num_classes": 100,
        "default_input_size": 32,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "tiny_imagenet": {
        "loader": _tiny_imagenet_loader,
        "num_classes": 200,
        "default_input_size": 64,
        "mean": (0.4802, 0.4481, 0.3975),
        "std": (0.2302, 0.2265, 0.2262),
    },
    "cub200": {
        "loader": _cub200_loader,
        "num_classes": 200,
        "default_input_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "malimg": {
        "loader": _malimg_loader,
        "num_classes": 25,
        "default_input_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "malware_classification": {
        "loader": _malware_classification_loader,
        "num_classes": 9,
        "default_input_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "big2015": {
        "loader": _malware_classification_loader,
        "num_classes": 9,
        "default_input_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "imagenette": {
        "loader": _imagenette_loader,
        "num_classes": 10,
        "default_input_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "cifar100_c": {
        "loader": _cifar100c_loader,
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

def get_available_datasets() -> list:
    return list(DATASET_REGISTRY.keys())

def register_dataset(
    name: str,
    loader: DatasetLoaderFunc,
    num_classes: int,
    default_input_size: int = 224
) -> None:
    if name in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' is already registered")
    
    DATASET_REGISTRY[name] = {
        "loader": loader,
        "num_classes": num_classes,
        "default_input_size": default_input_size,
    }
