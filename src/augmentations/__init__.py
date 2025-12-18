from typing import Callable, Tuple

from PIL import ImageFile

# Albumentations-based transforms
from .albumentations import build_transforms, tta_transforms
from .const import IMAGENET_MEAN, IMAGENET_STD

# Torchvision and ViT variants
from .torchvision import light_transforms as tv_light_transforms
from .vit_torchvision import (
    vit_heavy_transforms,
    vit_light_transforms,
    vit_medium_transforms,
)

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "tta_transforms",
    "tv_light_transforms",
    "vit_light_transforms",
    "vit_medium_transforms",
    "vit_heavy_transforms",
    "get_transforms",
]

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ===========================================================
# Default transform registry
# ===========================================================
default_transforms: dict[str, Callable[..., Tuple]] = {
    # Albumentations "light" and "heavy"
    "light": lambda **kwargs: build_transforms("light", **kwargs),
    "heavy": lambda **kwargs: build_transforms("heavy", **kwargs),
    # Torchvision
    "tv_light": tv_light_transforms,
    # ViT-style RandAugment pipelines
    "vit_light": vit_light_transforms,
    "vit_medium": vit_medium_transforms,
    "vit_heavy": vit_heavy_transforms,
}


# ===========================================================
# Contrastive Learning View Generator
# ===========================================================
class ContrastiveLearningViewGenerator:
    """Generate multiple random views of the same image for contrastive training.

    Args:
        base_transform: Transform (typically Albumentations.Compose)
        n_views: Number of random views to generate per image.

    Example:
        >>> generator = ContrastiveLearningViewGenerator(train_tfms, n_views=2)
        >>> views = generator(image)
        >>> len(views)
        2
    """

    def __init__(self, base_transform, n_views: int = 2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


# ===========================================================
# Unified access point
# ===========================================================
def get_transforms(
    name: str,
    image_size: tuple,
    model_mean: tuple = IMAGENET_MEAN,
    model_std: tuple = IMAGENET_STD,
    n_views: int | None = None,
    **kwargs,
) -> Tuple[Callable, Callable]:
    """
    Retrieve train/val transforms by name.

    Args:
        name: Transform key, e.g. "light", "heavy", "tv_light", "vit_heavy".
        image_size: (height, width) target image size.
        model_mean: Normalization mean (default: ImageNet mean).
        model_std: Normalization std (default: ImageNet std).
        n_views: If provided, wraps the training transform to produce multiple
                 random crops for contrastive learning.
        **kwargs: Extra parameters passed to the transform builder
                  (e.g. random_crop=True, resize_factor=1.3).

    Returns:
        (train_transform, val_transform)

    Example:
        >>> train_tfms, val_tfms = get_transforms(
        ...     name="heavy",
        ...     image_size=(224, 224),
        ...     random_crop=True,
        ...     resize_factor=1.3
        ... )
        >>> img_out = train_tfms(image=image)[IMAGE_KEY_NAME].shape
    """
    if name not in default_transforms:
        raise ValueError(f"Unknown transform name: {name}")

    transforms_fn = default_transforms[name]
    train_tfm, val_tfm = transforms_fn(
        image_size=image_size, mean=model_mean, std=model_std, **kwargs
    )

    # Wrap in contrastive generator if requested
    if n_views is not None:
        train_tfm = ContrastiveLearningViewGenerator(train_tfm, n_views=n_views)

    return train_tfm, val_tfm
