import os
import shutil
import subprocess
import sys
import zipfile
import csv
import math
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Callable, Dict, Any, Tuple, Optional, List
from logger import get_logger

DatasetLoaderFunc = Callable[..., Tuple[DataLoader, Optional[DataLoader], DataLoader]]

logger = get_logger(__name__)
KAGGLE_API_TOKEN = "KGAT_b9cda71d7565a5dd59748f52ea14fd41"


class _ImagePathDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, target = self.samples[index]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


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


def _resolve_existing_path(*candidates: str) -> Optional[str]:
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


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


def _ensure_kaggle_competition_available(
    competition_label: str,
    data_dir: str,
    competition_slug: str,
    local_root_candidates: List[str],
) -> str:
    existing = _resolve_existing_path(*local_root_candidates)
    if existing is not None:
        return existing

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    logger.info("%s not found locally. Downloading Kaggle competition '%s'...", competition_label, competition_slug)
    cmd_candidates = [
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            competition_slug,
            "-p",
            str(data_path),
            "--force",
        ],
        [
            sys.executable,
            "-m",
            "kaggle.cli",
            "competitions",
            "download",
            "-c",
            competition_slug,
            "-p",
            str(data_path),
            "--force",
        ],
    ]

    last_error = None
    for cmd in cmd_candidates:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            last_error = None
            break
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            last_error = exc

    if last_error is not None:
        raise RuntimeError(
            "Kaggle competition download failed. Configure Kaggle CLI credentials and accept competition rules."
        ) from last_error

    zip_candidates = sorted(data_path.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    for zip_path in zip_candidates:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_path)
        except zipfile.BadZipFile:
            continue

    # Re-check explicit local candidates first
    existing = _resolve_existing_path(*local_root_candidates)
    if existing is not None:
        return existing

    # Fallback: discover competition root by required files.
    roots = [data_path]
    roots.extend([p for p in data_path.iterdir() if p.is_dir()])
    for root in roots:
        if (root / "train").is_dir() and (root / "trainLabels.csv").is_file():
            return str(root)

    raise FileNotFoundError(
        f"{competition_label} download completed, but expected train/ and trainLabels.csv were not found under '{data_path}'."
    )


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

def _cifar100_loader(data_dir: str, batch_size: int, num_workers: int,val_split: float = 0.0, 
    seed: int = 42, **kwargs) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
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
    train_ds = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_tfms
    )
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
    root = Path(data_dir)
    tiny_root = _resolve_existing_path(
        str(root / "tiny-imagenet-200"),
        str(root / "tiny_imagenet"),
        str(root / "tiny-imagenet"),
        data_dir,
    )
    if tiny_root is None:
        raise FileNotFoundError(
            "Tiny-ImageNet not found. Expected a directory like data/tiny-imagenet-200 with train/ and val/."
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

    train_dir = Path(tiny_root) / "train"
    val_dir = Path(tiny_root) / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Tiny-ImageNet structure invalid at '{tiny_root}'. Expected train/ and val/ directories."
        )

    train_ds = torchvision.datasets.ImageFolder(root=str(train_dir), transform=train_tfms)
    train_val_ds = torchvision.datasets.ImageFolder(root=str(train_dir), transform=test_tfms)
    test_ds = torchvision.datasets.ImageFolder(root=str(val_dir), transform=test_tfms)

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
    root = Path(data_dir)
    cub_root = _resolve_existing_path(
        str(root / "CUB_200_2011"),
        str(root / "cub200"),
        str(root / "CUB-200-2011"),
        data_dir,
    )
    if cub_root is None:
        raise FileNotFoundError(
            "CUB-200-2011 not found. Expected a directory like data/CUB_200_2011 with metadata txt files and images/."
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

    train_ds = _ImagePathDataset(train_samples, transform=train_tfms)
    train_val_ds = _ImagePathDataset(train_samples, transform=test_tfms)
    test_ds = _ImagePathDataset(test_samples, transform=test_tfms)

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

    first_image_path, _ = probe_ds.samples[0]
    with Image.open(first_image_path) as first_img:
        inferred_width, inferred_height = first_img.convert("RGB").size
    inferred_size = (inferred_height, inferred_width)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tfms = T.Compose([
        T.Resize(inferred_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tfms = T.Compose([
        T.Resize(inferred_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.ImageFolder(root=str(train_dir), transform=train_tfms)
    val_ds = torchvision.datasets.ImageFolder(root=str(val_dir), transform=test_tfms)
    test_ds = torchvision.datasets.ImageFolder(root=str(test_dir), transform=test_tfms)

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
    competition_slug = "malware-classification"
    root = Path(data_dir)
    comp_root = _ensure_kaggle_competition_available(
        competition_label="Microsoft Malware Classification",
        data_dir=data_dir,
        competition_slug=competition_slug,
        local_root_candidates=[
            str(root / "malware-classification"),
            str(root / "malware_classification"),
            str(root),
        ],
    )

    comp_root_path = Path(comp_root)
    labels_csv = comp_root_path / "trainLabels.csv"
    train_dir = comp_root_path / "train"
    if not labels_csv.is_file() or not train_dir.is_dir():
        raise FileNotFoundError(
            f"Invalid Microsoft Malware Classification structure at '{comp_root_path}'. "
            "Expected trainLabels.csv and train/."
        )

    labeled_samples: List[Tuple[str, int]] = []
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["Id"].strip()
            cls = int(row["Class"]) - 1
            bytes_path = train_dir / f"{sample_id}.bytes"
            if bytes_path.is_file():
                labeled_samples.append((str(bytes_path), cls))

    if not labeled_samples:
        raise FileNotFoundError(
            f"No labeled .bytes files found under '{train_dir}'."
        )

    eval_split = val_split if (val_split and val_split > 0.0) else 0.1
    if not val_split or val_split <= 0.0:
        logger.info("No val_split provided for malware_classification; defaulting val_split to %.2f.", eval_split)

    test_split = kwargs.get("test_split", eval_split)
    if test_split <= 0.0:
        test_split = eval_split

    train_idx, val_idx, test_idx = _split_train_val_test_indices(
        n_items=len(labeled_samples),
        val_ratio=eval_split,
        test_ratio=test_split,
        seed=seed,
    )

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

    full_train_ds = _MalwareBytesDataset(labeled_samples, image_size=image_size, transform=train_tfms)
    full_eval_ds = _MalwareBytesDataset(labeled_samples, image_size=image_size, transform=test_tfms)

    train_ds = Subset(full_train_ds, train_idx)
    val_ds = Subset(full_eval_ds, val_idx)
    test_ds = Subset(full_eval_ds, test_idx)
    return _build_dataloaders(train_ds, val_ds, test_ds, batch_size, num_workers)

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "cifar100": {
        "loader": _cifar100_loader,
        "num_classes": 100,
        "default_input_size": 32,
    },
    "tiny_imagenet": {
        "loader": _tiny_imagenet_loader,
        "num_classes": 200,
        "default_input_size": 64,
    },
    "cub200": {
        "loader": _cub200_loader,
        "num_classes": 200,
        "default_input_size": 224,
    },
    "malimg": {
        "loader": _malimg_loader,
        "num_classes": 25,
        "default_input_size": 224,
    },
    "malware_classification": {
        "loader": _malware_classification_loader,
        "num_classes": 9,
        "default_input_size": 224,
    },
    "big2015": {
        "loader": _malware_classification_loader,
        "num_classes": 9,
        "default_input_size": 224,
    },
}

def get_dataset_loaders(dataset_name: str, data_dir: str, batch_size: int, 
    num_workers: int, **kwargs) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    
    dataset_info = DATASET_REGISTRY[dataset_name]
    loader = dataset_info["loader"]
    return loader(data_dir, batch_size, num_workers, **kwargs)

def get_num_classes(dataset_name: str) -> int:
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name]["num_classes"]


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
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    return int(DATASET_REGISTRY[dataset_name]["default_input_size"])

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
