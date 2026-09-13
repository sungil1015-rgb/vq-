from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from kobeni.config import DataConfig

_CIFAR_MEAN = {
    "cifar10": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.5071, 0.4867, 0.4408),
}
_CIFAR_STD = {
    "cifar10": (0.2470, 0.2435, 0.2616),
    "cifar100": (0.2675, 0.2565, 0.2761),
}


def _normalized_transform(config: DataConfig) -> transforms.Compose:
    mean = config.normalization_mean or _CIFAR_MEAN[config.dataset]
    std = config.normalization_std or _CIFAR_STD[config.dataset]
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def resolve_image_shape(config: DataConfig) -> tuple[int, int]:
    if config.image_shape is None:
        return (config.image_size, config.image_size)
    return tuple(config.image_shape)


def _resize_if_needed(config: DataConfig) -> list[object]:
    image_shape = resolve_image_shape(config)
    if image_shape == (32, 32):
        return []
    return [transforms.Resize(image_shape)]


def build_cifar_train_transform(config: DataConfig) -> transforms.Compose:
    image_shape = resolve_image_shape(config)
    operations: list[object] = []
    if config.random_crop:
        operations.append(transforms.RandomCrop(image_shape, padding=config.random_crop_padding))
    elif image_shape != (32, 32):
        operations.append(transforms.Resize(image_shape))
    if config.random_horizontal_flip:
        operations.append(transforms.RandomHorizontalFlip(p=config.horizontal_flip_probability))
    operations.extend(_normalized_transform(config).transforms)
    return transforms.Compose(operations)


def build_cifar_eval_transform(config: DataConfig) -> transforms.Compose:
    return transforms.Compose(
        [*_resize_if_needed(config), *_normalized_transform(config).transforms]
    )


def prepare_cifar_data(config: DataConfig) -> None:
    dataset_class = datasets.CIFAR10 if config.dataset == "cifar10" else datasets.CIFAR100
    for train in (True, False):
        dataset_class(root=config.root, train=train, download=True)


def build_cifar_loaders(config: DataConfig) -> tuple[DataLoader, DataLoader]:
    train_transform = build_cifar_train_transform(config)
    test_transform = build_cifar_eval_transform(config)
    dataset_class = datasets.CIFAR10 if config.dataset == "cifar10" else datasets.CIFAR100
    train_dataset = dataset_class(
        root=config.root, train=True, transform=train_transform, download=True
    )
    test_dataset = dataset_class(
        root=config.root, train=False, transform=test_transform, download=True
    )
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": config.persistent_workers and config.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    return train_loader, test_loader


def build_cifar_analysis_loader(config: DataConfig, split: str) -> DataLoader:
    """Build an unaugmented, ordered loader for representation analysis."""
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split}")
    transform = build_cifar_eval_transform(config)
    dataset_class = datasets.CIFAR10 if config.dataset == "cifar10" else datasets.CIFAR100
    dataset = dataset_class(
        root=config.root,
        train=split == "train",
        transform=transform,
        download=True,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
    )
