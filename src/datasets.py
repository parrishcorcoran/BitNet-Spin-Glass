"""Dataset loading: MNIST, CIFAR-10, and random-labels MNIST.

All datasets return (train_loader, test_loader) with the configured batch size.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from src.config import DatasetName


def get_dataloaders(
    dataset: DatasetName,
    batch_size: int = 128,
    data_dir: str = "./data",
    random_label_seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for the given dataset.

    Parameters
    ----------
    dataset : DatasetName
        Which dataset to load.
    batch_size : int
        Batch size for both loaders.
    data_dir : str
        Directory to download/cache data.
    random_label_seed : int
        Seed for shuffling labels (random_labels_mnist only).
    num_workers : int
        DataLoader workers. 0 = main process (safe for CPU).
    """
    if dataset == DatasetName.MNIST:
        return _load_mnist(batch_size, data_dir, num_workers, shuffle_labels=False)
    elif dataset == DatasetName.RANDOM_LABELS_MNIST:
        return _load_mnist(
            batch_size, data_dir, num_workers,
            shuffle_labels=True, label_seed=random_label_seed,
        )
    elif dataset == DatasetName.CIFAR10:
        return _load_cifar10(batch_size, data_dir, num_workers)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _load_mnist(
    batch_size: int,
    data_dir: str,
    num_workers: int,
    shuffle_labels: bool = False,
    label_seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Load MNIST, optionally with shuffled training labels."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    if shuffle_labels:
        rng = torch.Generator().manual_seed(label_seed)
        perm = torch.randperm(len(train_ds.targets), generator=rng)
        train_ds.targets = train_ds.targets[perm]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, test_loader


def _load_cifar10(
    batch_size: int,
    data_dir: str,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 with basic normalization (no augmentation)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, test_loader
