import os
import argparse
import os.path as osp
import numpy as np
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm

from src.data.catalog import FungiTasticCatalog
from src.data.image_datamodule import ImageDataModule
from src.data.components.embedding_utils import save_embeddings_metadata
from src.models.components.feature_extractors import TimmFeatureExtractor, HuggingFaceFeatureExtractor, TorchHubFeatureExtractor
from src.augmentations.image_views import ViewTransformGenerator, view_transformations
from src.config import TARGET_KEY_NAME, IMAGE_KEY_NAME, IMAGE_PATH_KEY_NAME, EMBEDDING_KEY_NAME, FILENAME_KEY_NAME, TRANSFORMATION_KEY_NAME
torch.backends.cudnn.benchmark = True


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EXTRACTORS = {
    "timm": TimmFeatureExtractor,
    "hf": HuggingFaceFeatureExtractor,
    "hub": TorchHubFeatureExtractor,
}


def build_transform(mean, std, size: int):
    """
    Build a preprocessing transform as in Fine-S paper:
    - Resize shorter side to 'size' (preserve aspect ratio)
    - Take central square crop of 'size'
    - Convert to tensor and normalize
    """
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),  # shorter side -> size
        T.CenterCrop(size),                                        # central square crop
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def build_transform_views(mean, std, size, view_transformations):
    view_generator = ViewTransformGenerator(transformations=view_transformations)
    return T.Compose([
        view_generator,
        T.Lambda(lambda views: [
            T.Compose([
                T.Resize((np.array(size) * 1.14).astype(int).tolist(), interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(size),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])(v) for v in views
        ]),
        T.Lambda(lambda views: torch.stack(views, dim=0)),
    ])




def get_catalog(kind: str, dataset_root: str):
    if kind.lower() == "ft":
        return FungiTasticCatalog(dataset_root=dataset_root,
                                  dataset_variant="full",
                                  dataset_size="720p",
                                  download=False)
    elif kind.lower() == "ft-m":
        return FungiTasticCatalog(dataset_root=dataset_root,
                                  dataset_variant="mini",
                                  dataset_size="720p",
                                  download=False)
    elif kind.lower() == "ft-fewshot":
        return FungiTasticCatalog(dataset_root=dataset_root,
                                  dataset_variant="fewshot",
                                  dataset_size="720p",
                                  download=False)
    else:
        raise ValueError(f"Unknown catalog type '{kind}'")


def compute_embeddings(model: torch.nn.Module, dataloader: DataLoader, device="cuda"):
    features, targets, paths = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding"):
            # x = batch[IMAGE_KEY_NAME].to(device)
            x = batch[IMAGE_KEY_NAME].to(device, non_blocking=True)
            y = batch[TARGET_KEY_NAME]
            path = batch[IMAGE_PATH_KEY_NAME]

            with torch.autocast('cuda'):
                feats = model(x)
            feats = F.normalize(feats, p=2, dim=1)
            feats = feats.cpu().numpy()

            features.append(feats)
            targets.append(y.numpy())
            paths.append(np.array(path))

    features = np.concatenate(features, axis=0)
    targets = np.concatenate(targets, axis=0)
    paths = np.concatenate(paths, axis=0)
    filenames = np.vectorize(os.path.basename)(paths)

    return {
        EMBEDDING_KEY_NAME: features,
        TARGET_KEY_NAME: targets,
        IMAGE_PATH_KEY_NAME: paths.tolist(),
        FILENAME_KEY_NAME: filenames,
    }


def compute_embeddings_views(model: torch.nn.Module, dataloader: DataLoader, view_transformations, device="cuda"):
    features, targets, paths = [], [], []

    augmentation_names = list(view_transformations.keys())

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding"):
            x = batch[IMAGE_KEY_NAME].to(device, non_blocking=True)
            y = batch[TARGET_KEY_NAME]
            path = batch[IMAGE_PATH_KEY_NAME]

            B, V, C, H, W = x.shape
            x = x.view(B * V, C, H, W)
            x = x.cuda()
            with torch.autocast('cuda'):
                feats = model(x)

            # Mean pooling if A x A x E
            if feats.dim() == 4:              # [B, A, A, E]
                feats = feats.mean(dim=(1, 2))  # -> [B, E]
            elif feats.dim() == 3:            # [A, A, E]
                feats = feats.mean(dim=(0, 1))  # -> [E]


            feats = F.normalize(feats, p=2, dim=1)

            feats = feats.cpu().numpy()

            y = y.repeat_interleave(V).cpu().numpy()
            path = np.repeat(np.array(path), V)

            features.append(feats)
            targets.append(y)
            paths.append(path)

    features = np.concatenate(features, axis=0)
    targets = np.concatenate(targets, axis=0)
    paths = np.concatenate(paths, axis=0)
    filenames = np.vectorize(os.path.basename)(paths)

    _transforms = np.tile(augmentation_names, len(features) // len(augmentation_names))

    return {
        EMBEDDING_KEY_NAME: features,
        TRANSFORMATION_KEY_NAME: _transforms,
        TARGET_KEY_NAME: targets,
        IMAGE_PATH_KEY_NAME: paths.tolist(),
        FILENAME_KEY_NAME: filenames,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings from image dataset")

    parser.add_argument("--architecture", type=str, required=True,
                        help="HuggingFace model name, e.g. facebook/dinov3-vitb16-pretrain-lvd1689m")

    parser.add_argument("--extractor", type=str, required=True, choices=["timm", "hf", "hub"],
                        help="Feature extractor to use")

    parser.add_argument("--catalog", type=str, required=True,
                        help="Dataset catalog to use")

    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Path to dataset folder")

    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save embeddings")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--mean", nargs=3, type=float, default=IMAGENET_MEAN,
                        help="Image normalization mean (3 floats)")

    parser.add_argument("--std", nargs=3, type=float, default=IMAGENET_STD,
                        help="Image normalization std (3 floats)")

    parser.add_argument("--image-size", type=int, required=True,
                        help="Image size (square: H=W)")

    parser.add_argument("--eval-resize", type=str, default=None,
                        help="Enable resizing model to target evaluation size (e.g., 448 for patch14)")

    parser.add_argument("--use-f16", type=str, default="false",
                        help="use float 16 for weights")

    parser.add_argument("--use-views", type=str, default="false",
                        help="generate multiple transform views")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_mean = tuple(args.mean)
    model_std = tuple(args.std)
    size = (args.image_size, args.image_size)

    print(f"[+] Using model mean={model_mean}, std={model_std}, size={size}")

    if args.use_views == "true":
        transform = build_transform_views(model_mean, model_std, args.image_size, view_transformations)
    else:
        transform = build_transform(model_mean, model_std, args.image_size)

    catalog = get_catalog(args.catalog, args.dataset_root)

    data_module = ImageDataModule(
        catalog=catalog,
        transformations=(transform, transform),
        batch_size=args.batch_size,
        train_shuffle=False,
        num_workers=args.num_workers,
    )
    data_module.setup()

    dataloaders = {
        "train": data_module.train_dataloader(),
        "val": data_module.val_dataloader(),
        "test": data_module.test_dataloader(),
    }

    print(f"[+] Loading architecture: {args.architecture}")
    print(f"[+] Loading extractor: {args.extractor}")

    extractor = EXTRACTORS[args.extractor]
    use_f16 = args.use_f16 == "true"
    if args.eval_resize is not None and args.eval_resize.lower() == "true":
        model = extractor(args.architecture, dynamic_img_size=args.eval_resize, use_f16=use_f16, use_low_cpu=use_f16)
    else:
        model = extractor(args.architecture, use_f16=use_f16, use_low_cpu=use_f16)

    # model = torch.compile(model)
    model.cuda()

    inference_metadata = {
        "architecture": args.architecture,
        "dataset_root": args.dataset_root,
        "size": size,
        "model_mean": model_mean,
        "model_std": model_std,
    }

    for split, loader in dataloaders.items():
        if loader is None:
            continue

        print(f"\n[+] Processing split: {split}")
        if args.use_views == "true":
            results = compute_embeddings_views(model, loader, view_transformations=view_transformations, device="cuda")
        else:
            results = compute_embeddings(model, loader, device="cuda")

        save_embeddings_metadata(
            embeddings=results,
            split=split,
            inference_metadata=inference_metadata,
            save_dir=args.output_dir,
        )

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
