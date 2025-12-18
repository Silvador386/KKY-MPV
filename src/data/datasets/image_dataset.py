from typing import Tuple, Union, List, Dict

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from matplotlib import pyplot as plt
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image

from src.config import TARGET_KEY_NAME, IMAGE_KEY_NAME, IMAGE_PATH_KEY_NAME
from src.data.datasets.base import BaseDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(BaseDataset):

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Union[A.Compose, T.Compose, None] = None,
        call_features: List[str] = None,
    ):
        super().__init__(
            df=df,
            call_features=call_features,
        )
        self.transform = transform
        self._check_metadata_integrity()

    def __getitem__(self, idx: int) -> Dict:
        image, image_path = self.get_image(idx)
        target = self.get_target_id(idx)
        image = self.apply_transforms(image)

        item = {
            IMAGE_KEY_NAME: image,
            TARGET_KEY_NAME: target,
            IMAGE_PATH_KEY_NAME: image_path,

        }
        if self.call_features is not None:
            item.update(self.get_metadata(idx))
        return item

    def _check_metadata_integrity(self):
        """Checks if the dataset complies with the framework requirements."""
        assert (
                IMAGE_PATH_KEY_NAME in self.df.columns.values
        ), f"Dataset must contain `{IMAGE_PATH_KEY_NAME}` column!"

    def get_image(self, idx: int) -> Tuple[Image.Image, str]:
        """Get i-th image and its file path in the dataset."""
        image_path = self.df[IMAGE_PATH_KEY_NAME].iloc[idx]
        image_pil = Image.open(image_path).convert("RGB")
        return image_pil, image_path

    def apply_transforms(self, image: Image.Image) -> torch.Tensor:
        """Apply augmentation transformations on the image."""
        if self.transform is not None:
            if A is not None and isinstance(self.transform, A.Compose):
                image = self.transform(image=np.asarray(image))[IMAGE_KEY_NAME]
            elif T is not None and isinstance(self.transform, T.Compose):
                image = self.transform(image)
            else:
                raise NotImplementedError(
                    "Supports only albumentations and torchvision transforms."
                )
        return image

    def plot_dataset_grid(
        self,
        indices: List[int] = None,
        nrows: int = 2,
        ncols: int = 5,
        col_name: str = None,
        figsize: Tuple[int, int] = (10, 10),
        left_offset: float = 0.1,
        save_path: str = None,
    ):
        """
        Plot a grid where each row contains samples from the same species.
        Species label displayed ON THE RIGHT SIDE of each row.
        """
        if col_name is None:
            col_name = TARGET_KEY_NAME

        # Unique class labels
        classes = self.df[col_name].unique()

        if indices is None:
            # Random selection
            classes = np.random.choice(classes, size=nrows, replace=False)
        else:
            # Use class indices
            indices = np.asarray(indices)
            if np.any(indices >= len(classes)):
                raise ValueError("Some class indices are out of range.")
            classes = classes[indices]
            nrows = len(classes)

        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "axes.linewidth": 0.5
        })

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1:
            axes = np.array([axes])
        axes = axes.reshape(nrows, ncols)


        for row_idx in range(nrows):
            cls = classes[row_idx]

            # All rows: pick ncols random samples from this class
            class_df = self.df[self.df[col_name] == cls]
            sample_indices = np.random.choice(
                class_df.index.values, size=ncols, replace=len(class_df) < ncols
            )

            for col_idx, df_idx in enumerate(sample_indices):
                pos_idx = self.df.index.get_loc(df_idx)
                ax = axes[row_idx, col_idx]

                img, _ = self.get_image(pos_idx)
                img = self.apply_transforms(img)
                if not isinstance(img, Image.Image):
                    img = to_pil_image(img)

                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)

            if "_" in str(cls):
                formated_cls = "\n".join(str(cls).split("_"))
            else:
                formated_cls = "\n".join(str(cls).split(" "))

            fig.text(
                0.01,
                (nrows - row_idx - 0.5) / nrows,
                f"{formated_cls}",
                va="center", ha="left", fontsize=14
            )

        plt.tight_layout(rect=[left_offset, 0, 1, 1])

        if save_path:
            plt.savefig(save_path)
        plt.show()


class MultiLabelImageDataset(ImageDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Union[A.Compose, T.Compose],
        features: List[str],
        **kwargs
    ):
        assert TARGET_KEY_NAME in features, f"{TARGET_KEY_NAME} must be present in features"
        super().__init__(df, transform, **kwargs)
        self.features = features

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict, str]:
        image, image_path = self.get_image(idx)
        image = self.apply_transforms(image)
        target = {feature: self.get_feature_id(feature, idx) for feature in self.features}
        return image, target, image_path

    def get_feature_id(self, feature: str, idx: int) -> int:
        """Get genus id of i-th element in the dataset."""
        return self.df[feature].iloc[idx]