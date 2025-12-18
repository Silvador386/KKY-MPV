from typing import Literal, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .const import IMAGENET_MEAN, IMAGENET_STD


def _normalize_to_tensor(mean, std):
    """Return normalization and tensor conversion transforms."""
    return [A.Normalize(mean=mean, std=std), ToTensorV2()]


def _resize_and_normalize(image_size, mean, std):
    """Return a simple resize + normalize pipeline (for validation/test)."""
    return A.Compose(
        [
            A.PadIfNeeded(*image_size),
            A.Resize(*image_size),
            *_normalize_to_tensor(mean, std),
        ]
    )


def _train_augments(
    level: Literal["light", "heavy"],
    image_size: Tuple[int, int],
    random_crop: bool = False,
    resize_factor: float = 1.0 / 0.7,
):
    """
    Internal: build the augmentation list for training.

    Args:
        level: "light" or "heavy" augmentation strength.
        image_size: (height, width) of the final image.
        random_crop: If True, add padded resize and random crop before augmentation.
        resize_factor: Scale factor applied before cropping (e.g., 1/0.7 enlarges by ~1.43×).

    Returns:
        List of Albumentations transforms.
    """
    h, w = image_size

    if level == "light":
        base = [
            A.RandomResizedCrop(h, w, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        ]
    elif level == "heavy":
        base = [
            A.RandomResizedCrop(h, w, scale=(0.7, 1.3)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
            A.GaussianBlur(blur_limit=(7, 7), p=0.5),
            A.HueSaturationValue(p=0.2),
            A.ImageCompression(50, 100, p=0.2),
            A.CoarseDropout(8, 20, 20, fill_value=128, p=0.2),
            A.ShiftScaleRotate(0.10, 0.25, 90, p=0.5),
            A.RandomGridShuffle(grid=(3, 3), p=0.1),
            A.MultiplicativeNoise(multiplier=(0.8, 1.2), elementwise=True, p=0.1),
        ]
    else:
        raise ValueError(f"Unknown augmentation level: {level}")

    if random_crop:
        resize_dims = (int(h * resize_factor), int(w * resize_factor))
        base = [
            A.PadIfNeeded(*resize_dims),
            A.Resize(*resize_dims),
            A.RandomCrop(h, w),
            *base,
        ]

    return base


def build_transforms(
    level: Literal["light", "heavy"],
    *,
    image_size: Tuple[int, int],
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    random_crop: bool = False,
    resize_factor: float = 1.0 / 0.7,
):
    """
    Build Albumentations transforms for training and validation.

    Args:
        level: "light" or "heavy" augmentation level.
        image_size: Target (height, width).
        mean: Normalization mean (default: ImageNet mean).
        std: Normalization std (default: ImageNet std).
        random_crop: Whether to apply padded resize and random crop first.
        resize_factor: Pre-crop resize factor.
            - >1.0 means enlarge before cropping (default: 1/0.7 ≈ 1.43).
            - 1.0 means no resize before cropping.

    Returns:
        (train_transforms, val_transforms)

    Example:
        >>> train_tfms, val_tfms = build_transforms(
        ...     "heavy", image_size=(224, 224), random_crop=True, resize_factor=1.3
        ... )
    """
    train_tfms = A.Compose(
        [
            *_train_augments(level, image_size, random_crop, resize_factor),
            *_normalize_to_tensor(mean, std),
        ]
    )

    if random_crop:
        resize_dims = (
            int(image_size[0] * resize_factor),
            int(image_size[1] * resize_factor),
        )
        val_tfms = A.Compose(
            [
                A.PadIfNeeded(*resize_dims),
                A.Resize(*resize_dims),
                A.CenterCrop(*image_size),
                *_normalize_to_tensor(mean, std),
            ]
        )
    else:
        val_tfms = _resize_and_normalize(image_size, mean, std)

    return train_tfms, val_tfms


def tta_transforms(
    *,
    mode: Literal["vanilla", "center", "corner"],
    image_size: Tuple[int, int],
    scale: float = 0.8,
    crop_position: Optional[
        Literal["top_left", "top_right", "bottom_left", "bottom_right"]
    ] = None,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
):
    """
    Build a parameterized Test-Time Augmentation (TTA) transform.

    Args:
        mode:
            - "vanilla": Resize + normalize (no cropping).
            - "center": Scaled resize + center crop.
            - "corner": Scaled resize + corner crop.
        image_size: Target (height, width).
        scale: Crop area relative to final size. (e.g. 0.8 → 1.25× enlarge then crop)
        crop_position: Required for mode="corner"; one of:
            {"top_left", "top_right", "bottom_left", "bottom_right"}.
        mean: Normalization mean (default: ImageNet mean).
        std: Normalization std (default: ImageNet std).

    Returns:
        Albumentations Compose object.

    Example:
        >>> tta_center = tta_transforms(mode="center", scale=0.8, image_size=(224, 224))
        >>> tta_corner = tta_transforms(
        ...     mode="corner", crop_position="bottom_right", scale=0.7, image_size=(224, 224)
        ... )
    """

    h, w = image_size

    if mode == "vanilla":
        return _resize_and_normalize(image_size, mean, std)

    new_h, new_w = int(h / scale), int(w / scale)
    transforms = [A.PadIfNeeded(new_h, new_w), A.Resize(new_h, new_w)]

    if mode == "center":
        transforms.append(A.CenterCrop(h, w))
    elif mode == "corner":
        if not crop_position:
            raise ValueError("`crop_position` must be set when mode='corner'.")
        x0 = 0 if "left" in crop_position else new_w - w
        y0 = 0 if "top" in crop_position else new_h - h
        transforms.append(A.Crop(x0, y0, x0 + h, y0 + w))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    transforms += _normalize_to_tensor(mean, std)
    return A.Compose(transforms)
