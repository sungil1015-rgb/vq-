import pytest
from PIL import Image
from torchvision import transforms

from kobeni.config import DataConfig
from kobeni.data import build_cifar_eval_transform, build_cifar_train_transform


@pytest.mark.parametrize(
    ("random_crop", "random_horizontal_flip", "expected"),
    [
        (True, False, ["RandomCrop"]),
        (False, True, ["RandomHorizontalFlip"]),
        (False, False, []),
        (True, True, ["RandomCrop", "RandomHorizontalFlip"]),
    ],
)
def test_train_transform_matches_augmentation_config(
    random_crop: bool, random_horizontal_flip: bool, expected: list[str]
) -> None:
    transform = build_cifar_train_transform(
        DataConfig(
            random_crop=random_crop,
            random_horizontal_flip=random_horizontal_flip,
        )
    )
    geometric = [
        type(operation).__name__
        for operation in transform.transforms
        if isinstance(operation, (transforms.RandomCrop, transforms.RandomHorizontalFlip))
    ]

    assert geometric == expected
    assert isinstance(transform.transforms[-2], transforms.ToTensor)
    assert isinstance(transform.transforms[-1], transforms.Normalize)


def test_transform_uses_configured_crop_flip_and_normalization() -> None:
    transform = build_cifar_train_transform(
        DataConfig(
            image_size=28,
            random_crop_padding=2,
            horizontal_flip_probability=0.25,
            normalization_mean=[0.1, 0.2, 0.3],
            normalization_std=[0.4, 0.5, 0.6],
        )
    )

    crop = next(item for item in transform.transforms if isinstance(item, transforms.RandomCrop))
    flip = next(
        item for item in transform.transforms if isinstance(item, transforms.RandomHorizontalFlip)
    )
    normalize = next(
        item for item in transform.transforms if isinstance(item, transforms.Normalize)
    )
    assert crop.size == (28, 28)
    assert crop.padding == 2
    assert flip.p == pytest.approx(0.25)
    assert normalize.mean == [0.1, 0.2, 0.3]
    assert normalize.std == [0.4, 0.5, 0.6]


def test_train_and_eval_transform_share_configured_image_size() -> None:
    config = DataConfig(image_size=28, random_crop=False, random_horizontal_flip=False)
    image = Image.new("RGB", (32, 32))

    assert build_cifar_train_transform(config)(image).shape[-2:] == (28, 28)
    assert build_cifar_eval_transform(config)(image).shape[-2:] == (28, 28)


def test_train_and_eval_transform_support_rectangular_image_shape() -> None:
    config = DataConfig(
        image_shape=[40, 28],
        random_crop=False,
        random_horizontal_flip=False,
    )
    image = Image.new("RGB", (32, 32))

    assert build_cifar_train_transform(config)(image).shape[-2:] == (40, 28)
    assert build_cifar_eval_transform(config)(image).shape[-2:] == (40, 28)
