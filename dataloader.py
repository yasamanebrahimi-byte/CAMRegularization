import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split


def cifar100_loaders(data_dir, batch_size, num_workers, val_split=0.0, seed=42):
    # Pre-computed dataset mean and std for CIFAR-100 (RGB).
    # These are used to normalize images to zero-mean/unit-variance.
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    # Training transformations:
    # - RandomCrop with padding: data augmentation by random crops.
    # - RandomHorizontalFlip: augment by flipping images horizontally.
    # - ToTensor: convert PIL image to float tensor in [0,1].
    # - Normalize: apply channel-wise mean/std normalization.
    train_tfms = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # Test/validation transformations: only convert to tensor and normalize.
    # No random augmentations so evaluation is deterministic.
    test_tfms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # Create dataset objects. `download=True` will fetch the data if missing.
    train_ds = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_tfms
    )

    test_ds = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_tfms
    )

    # Handle validation split if specified
    val_dl = None
    if val_split > 0.0:
        train_size = int(len(train_ds) * (1 - val_split))
        val_size = len(train_ds) - train_size
        train_ds, val_ds = random_split(
            train_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
    else:
        val_ds = None

    # Wrap datasets in DataLoaders. Common settings:
    # - `shuffle=True` in training to randomize sample order per epoch.
    # - `pin_memory=True` can slightly speed up host->GPU transfers.
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    # Validation DataLoader (if validation split is used)
    if val_dl is None and val_ds is not None:
        val_dl = DataLoader(
            val_ds, batch_size=256, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

    # For evaluation we often use a larger fixed batch size and no shuffling.
    test_dl = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_dl, val_dl, test_dl