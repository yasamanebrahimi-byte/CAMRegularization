import os
import subprocess
import sys
import zipfile
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

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    archive_path = data_path / archive_filename

    logger.info("%s not found locally. Downloading from Kaggle dataset '%s'...", dataset_label, kaggle_dataset)
    cmd = [
        sys.executable,
        "-m",
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
    ]

    try:
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError:
        fallback_cmd = [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            kaggle_dataset,
            "-p",
            str(data_path),
            "--force",
        ]
        subprocess.run(fallback_cmd, check=True, env=env)

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
    KAGGLE_API_TOKEN = "KGAT_b9cda71d7565a5dd59748f52ea14fd41"
    root = Path(data_dir)
    malimg_root = _ensure_kaggle_dataset_available(
        dataset_label="MalImg",
        data_dir=data_dir,
        kaggle_dataset=kaggle_dataset,
        local_dir_candidates=[
            str(root / "MalImg"),
            str(root / "malimg"),
            data_dir,
        ],
        archive_filename="malimg_kaggle.zip",
        kaggle_api_token=KAGGLE_API_TOKEN,
    )

    probe_ds = torchvision.datasets.ImageFolder(root=malimg_root)
    if len(probe_ds) == 0:
        raise ValueError(f"No images found in '{malimg_root}'.")

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

    full_train_ds = torchvision.datasets.ImageFolder(root=malimg_root, transform=train_tfms)
    full_eval_ds = torchvision.datasets.ImageFolder(root=malimg_root, transform=test_tfms)

    train_ds, val_ds = _split_train_val(full_train_ds, full_eval_ds, val_split, seed)
    if val_ds is None:
        raise ValueError("MalImg requires val_split > 0.0 because it does not provide an official test split.")

    test_ds = val_ds
    logger.info("MalImg has no official split; using validation split for evaluation.")
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
