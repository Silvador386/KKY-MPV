from typing import Callable

import torch
import torchvision.transforms.functional as F
from PIL import Image


def original(img):
    return img

def horizontal_flip(img):
    if hasattr(img, 'transpose'):  # PIL Image
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    else:  # Tensor
        return torch.flip(img, dims=[2])

def aspect_preserving_center_crop(pil_img, scale=0.8):
    """
    Returns an aspect-preserving center crop of the input PIL image.
    The crop is `scale` times the original size (e.g., 0.8 for 80% of original).
    """
    w, h = pil_img.size
    crop_w = int(scale * w)
    crop_h = int(scale * h)

    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h

    return pil_img.crop((left, top, right, bottom))


def tlcrop_no_resize(pil_img, scale=0.8):
    """
    Return four aspect-preserving corner crops, each of size
    (scale * original height, scale * original width).
    """
    w, h = pil_img.size
    crop_w = int(scale * w)
    crop_h = int(scale * h)
    box = (0, 0, crop_w, crop_h)  # Top-left
    return pil_img.crop(box)


def trcrop_no_resize(pil_img, scale=0.8):
    """
    Return four aspect-preserving corner crops, each of size
    (scale * original height, scale * original width).
    """
    w, h = pil_img.size
    crop_w = int(scale * w)
    crop_h = int(scale * h)
    box = (w - crop_w, 0, w, crop_h)  # Top-right
    return pil_img.crop(box)


def blcrop_no_resize(pil_img, scale=0.8):
    """
    Return four aspect-preserving corner crops, each of size
    (scale * original height, scale * original width).
    """
    w, h = pil_img.size
    crop_w = int(scale * w)
    crop_h = int(scale * h)
    box = (0, h - crop_h, crop_w, h)  # Bottom-left
    return pil_img.crop(box)


def brcrop_no_resize(pil_img, scale=0.8):
    """
    Return four aspect-preserving corner crops, each of size
    (scale * original height, scale * original width).
    """
    w, h = pil_img.size
    crop_w = int(scale * w)
    crop_h = int(scale * h)
    box = (w - crop_w, h - crop_h, w, h)  # Bottom-right
    return pil_img.crop(box)

def rot_90(img):
    return F.rotate(img, 90)

def rot_180(img):
    return F.rotate(img, 180)

def rot_270(img):
    return F.rotate(img, 270)

def rot_15(img):
    return img.rotate(15, resample=Image.BICUBIC, expand=False)

def rot_345(img):
    return img.rotate(345, resample=Image.BICUBIC, expand=False)

transforms = {
    "original": original,
    "horizontal_flip": horizontal_flip,
    "rot_90": rot_90,
    "rot_270": rot_270,
    "rot_15": rot_15,
    "rot_345": rot_345,
}

crop_transforms = {
    "center_crop": aspect_preserving_center_crop,
    "top_left_crop": tlcrop_no_resize,
    "top_right_crop": trcrop_no_resize,
    "bottom_left_crop": blcrop_no_resize,
    "bottom_right_crop": brcrop_no_resize,
}

view_transformations = {**transforms, **crop_transforms}


class ViewTransformGenerator:

    def __init__(self, transformations: dict[str, Callable]):
        self.transformations = transformations

    def __call__(self, x):
        return [transform(x) for name, transform in self.transformations.items()]

