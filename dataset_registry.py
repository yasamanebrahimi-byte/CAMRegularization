import os
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
    root = Path(data_dir)
    malimg_root = _resolve_existing_path(
        str(root / "MalImg"),
        str(root / "malimg"),
        data_dir,
    )
    if malimg_root is None:
        raise FileNotFoundError(
            "MalImg not found. Expected class-subfolder images under data/MalImg (or data/malimg)."
        )

    image_size = int(kwargs.get("image_size", 224))
    test_split = float(kwargs.get("test_split", 0.2))

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
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

    full_train_ds = torchvision.datasets.ImageFolder(root=malimg_root, transform=train_tfms)
    full_eval_ds = torchvision.datasets.ImageFolder(root=malimg_root, transform=test_tfms)

    if len(full_train_ds) == 0:
        raise ValueError(f"No images found in '{malimg_root}'.")

    test_size = int(len(full_train_ds) * test_split)
    test_size = min(max(test_size, 1), len(full_train_ds) - 1)
    train_size = len(full_train_ds) - test_size

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_train_ds), generator=g).tolist()
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_ds = Subset(full_train_ds, train_indices)
    train_eval_ds = Subset(full_eval_ds, train_indices)
    test_ds = Subset(full_eval_ds, test_indices)

    train_ds, val_ds = _split_train_val(train_ds, train_eval_ds, val_split, seed)
    if val_ds is None:
        logger.info("MalImg has no official split; using random train/test split. Pass val_split>0 for train-val.")
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
