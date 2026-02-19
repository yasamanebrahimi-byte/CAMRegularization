"""
Dataset registry for easy addition of new datasets.
Each dataset is registered with a loader function and metadata.
"""
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from typing import Callable, Dict, Any, Tuple, Optional
from logger import get_logger

# Type for dataset loader functions
DatasetLoaderFunc = Callable[..., Tuple[DataLoader, Optional[DataLoader], DataLoader]]


def _cifar100_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.0,
    seed: int = 42,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Load CIFAR-100 dataset with train/val/test splits."""
    logger = get_logger(__name__)
    
    # Mean and std to normalize images to zero-mean/unit-variance
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    # Transformations for training and test/validation
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
    
    # Loading the data
    train_ds = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_tfms
    )
    train_val_ds = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=test_tfms
    )
    test_ds = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_tfms
    )
    
    # Perform validation split
    val_ds = None
    if val_split and val_split > 0.0:
        train_size = int(len(train_ds) * (1.0 - val_split))
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(train_ds), generator=g).tolist()
        train_idx = perm[:train_size]
        val_idx = perm[train_size:]
        train_ds = Subset(train_ds, train_idx)
        val_ds = Subset(train_val_ds, val_idx)
    else:
        logger.info("Caution: No validation split specified")
    
    # Wrap datasets in DataLoaders
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True
    ) if val_ds is not None else None
    
    # For test set, use a larger fixed batch size and no shuffling
    test_dl = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_dl, val_dl, test_dl


# Registry of available datasets
# Key: dataset name, Value: {"loader": loader_fn, "num_classes": ..., "default_input_size": ...}
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "cifar100": {
        "loader": _cifar100_loader,
        "num_classes": 100,
        "default_input_size": 32,
    },
}


def get_dataset_loaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Get dataloaders for a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'cifar100', 'cifar10')
        data_dir: Directory to store/load dataset
        batch_size: Batch size for training
        num_workers: Number of workers for dataloader
        **kwargs: Additional arguments passed to the loader (e.g., val_split, seed)
    
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
        val_dataloader may be None if val_split == 0
    
    Raises:
        ValueError: If dataset_name is not registered
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    
    dataset_info = DATASET_REGISTRY[dataset_name]
    loader = dataset_info["loader"]
    return loader(data_dir, batch_size, num_workers, **kwargs)


def get_num_classes(dataset_name: str) -> int:
    """
    Get the number of classes in a dataset.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        Number of classes
    
    Raises:
        ValueError: If dataset_name is not registered
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    
    return DATASET_REGISTRY[dataset_name]["num_classes"]


def get_available_datasets() -> list:
    """Return a list of available dataset names."""
    return list(DATASET_REGISTRY.keys())


def register_dataset(
    name: str,
    loader: DatasetLoaderFunc,
    num_classes: int,
    default_input_size: int = 224
) -> None:
    """
    Register a new dataset in the registry.
    
    Args:
        name: Name for the dataset
        loader: Function that returns (train_dl, val_dl, test_dl)
        num_classes: Number of classes in the dataset
        default_input_size: Expected input image size for this dataset
    """
    if name in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' is already registered")
    
    DATASET_REGISTRY[name] = {
        "loader": loader,
        "num_classes": num_classes,
        "default_input_size": default_input_size,
    }
