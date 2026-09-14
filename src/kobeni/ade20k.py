from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from kobeni.config import DataConfig
from kobeni.data import resolve_image_shape


class ADE20KSegmentationDataset(Dataset[tuple[Tensor, Tensor]]):
    """ADEChallengeData2016 semantic masks, mapped from 1..150 to 0..149."""

    def __init__(self, config: DataConfig, split: str, train: bool) -> None:
        if split not in {"training", "validation"}:
            raise ValueError(f"Unsupported ADE20K split: {split}")
        root = _resolve_ade_root(Path(config.root))
        self.image_dir = root / "images" / split
        self.annotation_dir = root / "annotations" / split
        self.image_paths = sorted(self.image_dir.glob("*.jpg"))
        if not self.image_paths:
            raise FileNotFoundError(f"No ADE20K images found in {self.image_dir}")
        missing = [
            image_path
            for image_path in self.image_paths
            if not (self.annotation_dir / f"{image_path.stem}.png").is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing ADE20K annotations, first missing mask: {missing[0]}")
        self.image_shape = resolve_image_shape(config)
        self.train = train
        self.flip_probability = (
            config.horizontal_flip_probability if config.random_horizontal_flip else 0.0
        )
        self.mean = config.normalization_mean or [0.485, 0.456, 0.406]
        self.std = config.normalization_std or [0.229, 0.224, 0.225]
        self.ignore_index = config.segmentation_ignore_index
        self.num_classes = 150

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        from PIL import Image

        image_path = self.image_paths[index]
        mask_path = self.annotation_dir / f"{image_path.stem}.png"
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = transform_functional.resize(
            image,
            self.image_shape,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        mask = transform_functional.resize(
            mask,
            self.image_shape,
            interpolation=InterpolationMode.NEAREST,
        )
        if self.train and torch.rand(()).item() < self.flip_probability:
            image = transform_functional.hflip(image)
            mask = transform_functional.hflip(mask)
        image_tensor = transform_functional.normalize(
            transform_functional.to_tensor(image),
            self.mean,
            self.std,
        )
        raw_mask = transform_functional.pil_to_tensor(mask).squeeze(0).long()
        target = torch.full_like(raw_mask, self.ignore_index)
        valid = (raw_mask >= 1) & (raw_mask <= self.num_classes)
        target[valid] = raw_mask[valid] - 1
        return image_tensor, target


def _resolve_ade_root(root: Path) -> Path:
    challenge_root = root / "ADEChallengeData2016"
    return challenge_root if challenge_root.is_dir() else root


def prepare_ade20k_data(config: DataConfig) -> None:
    root = _resolve_ade_root(Path(config.root))
    required = [
        root / "images" / "training",
        root / "images" / "validation",
        root / "annotations" / "training",
        root / "annotations" / "validation",
    ]
    if all(path.is_dir() for path in required):
        return
    expected = Path(config.root) / "ADEChallengeData2016"
    raise FileNotFoundError(
        "ADE20K is not installed. Register and download it from the official ADE20K site, "
        f"then extract ADEChallengeData2016 under {expected}."
    )


def build_ade20k_loaders(config: DataConfig) -> tuple[DataLoader, DataLoader]:
    prepare_ade20k_data(config)
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": config.persistent_workers and config.num_workers > 0,
    }
    train_dataset = ADE20KSegmentationDataset(config, "training", train=True)
    validation_dataset = ADE20KSegmentationDataset(config, "validation", train=False)
    return (
        DataLoader(train_dataset, shuffle=True, **loader_options),
        DataLoader(validation_dataset, shuffle=False, **loader_options),
    )
