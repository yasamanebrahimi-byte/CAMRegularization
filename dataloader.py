import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from logger import get_logger

"""
Create CIFAR-100 DataLoaders for training, optional validation, and testing.
- data_dir: root directory where CIFAR-100 will be downloaded/stored
- batch_size: training batch size
- num_workers: number of DataLoader worker processes
- val_split: fraction of the training set reserved for validation (0.0 disables validation)
- seed: random seed used to make the train/val split reproducible

Notes:
- Training uses data augmentation (random crop + horizontal flip) plus normalization.
- Validation (if enabled) and test use only normalization (no augmentation).
- When val_split > 0, the validation set is a subset of the training images, but with eval transforms.
- Returns (train_dl, val_dl, test_dl) where val_dl is None if val_split == 0.
"""
def cifar100_loaders(data_dir, batch_size, num_workers, val_split=0.0, seed=42):
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

    # For test set, use a larger fixed batch size and no shuffling.
    test_dl = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_dl, val_dl, test_dl