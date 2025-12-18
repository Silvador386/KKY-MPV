import os
from typing import Optional, Sequence
import pandas as pd

from src.data.datasets import ImageDataset
from src.data.catalog.base import BaseCatalog
from src.data.components.splits import DatasetSplits
from src.data.components.encoder import OrdinalEncoder

from src.config import TARGET_KEY_NAME, IMAGE_PATH_KEY_NAME, FILENAME_KEY_NAME, IMAGE_KEY_NAME

import subprocess
import zipfile
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FungiTasticDownloader:
    """
    A downloader class for the FungiTastic dataset, providing options to download and extract
    various components including images, metadata, satellite data, climatic data, and masks.

    Attributes:
        SUBSETS (list): Available dataset subsets.
        SIZES (list): Available image sizes.
        SUBSET2STR (dict): Mapping of subset abbreviations to full names.
        HAS_DNA (dict): Indicates which subsets have DNA data.
        DOWNLOAD_ROOT (str): Root URL for dataset downloads.

    Methods:
        download_file(url, tool): Downloads a file using wget or curl.
        download_and_extract(url, target_dir): Downloads and extracts a zip file.
        download_metadata(): Downloads the metadata zip file.
        download_images(subset, size): Downloads image data for a given subset and size.
        download_satellite_data(): Downloads satellite data.
        download_climatic_data(): Downloads climatic data.
        download_masks(): Downloads mask data.
        generate_img_link(subset, size, split): Generates download URL for images.
        download(subset, size): Initiates download of selected data types.
        validate_params(params): Validates input arguments for download.
    """

    SUBSETS = ["full", "fs", "m"]
    SIZES = ["300", "500", "720", "fullsize"]
    SUBSET2STR = {"fs": "FewShot", "m": "Mini"}
    HAS_DNA = {
        "full": True,
        "fs": False,
        "m": True,
    }
    DOWNLOAD_ROOT = "https://cmp.felk.cvut.cz/datagrid/FungiTastic/shared/download"

    def __init__(
            self,
            save_path: Path,
            rewrite: bool = False,
            keep_zip: bool = False,
            no_extraction: bool = False,
            metadata: bool = False,
            images: bool = False,
            satellite: bool = False,
            climatic: bool = False,
            masks: bool = False,
            download_url: str = None,
    ):
        """
        Initializes the downloader with user-specified parameters.

        Args:
            save_path (Path): Root directory to save the downloaded data.
            rewrite (bool): Whether to overwrite existing files.
            keep_zip (bool): Whether to keep the downloaded zip files.
            no_extraction (bool): If True, skip the extraction step.
            metadata (bool): If True, download metadata.
            images (bool): If True, download images.
            satellite (bool): If True, download satellite data.
            climatic (bool): If True, download climatic data.
            masks (bool): If True, download mask data.
        """
        self.save_path = save_path
        self.rewrite = rewrite
        self.keep_zip = keep_zip
        self.no_extraction = no_extraction
        self.metadata = metadata
        self.images = images
        self.satellite = satellite
        self.climatic = climatic
        self.masks = masks

        if download_url is not None:
            print(f"Using the new download url: {download_url}")
            self.DOWNLOAD_ROOT = download_url

        self.fungi_path = self.save_path / "FungiTastic"
        self.fungi_path.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, tool: str = "wget") -> Path:
        """
        Downloads a zip file from a specified URL using wget or curl.

        Args:
            url (str): The URL of the file to download.
            tool (str): The command-line tool to use for downloading ('wget' or 'curl').

        Returns:
            Path: Path to the downloaded file.

        Raises:
            RuntimeError: If the file download fails.
        """
        target_file = self.fungi_path / Path(url).name
        cmd = (
            ["wget", "-nc", "-P", str(self.fungi_path), url]
            if tool == "wget"
            else ["curl", "-O", str(target_file), url]
        )

        print(f"Downloading {url} to {target_file}")
        print(f"Command: {cmd}")
        result = subprocess.run(cmd)
        if result.returncode != 0 or not target_file.exists():
            raise RuntimeError(f"Failed to download {url}")

        return target_file

    def download_and_extract(self, url: str, target_dir: Path) -> None:
        """
        Downloads and extracts a zip file from the specified URL.

        Args:
            url (str): The URL of the zip file.
            target_dir (Path): Directory to extract the zip file to.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        zip_path = target_dir / Path(url).name

        print("Download:")
        if self.rewrite or not zip_path.exists():
            print(f"\tDownloading from {url} to {target_dir}")
            zip_file = self.download_file(url)
            print(f"\tDownload of {url} complete\n")
        else:
            zip_file = zip_path
            print(f"\tFile already downloaded to {zip_file}\n")

        print("Extract:")
        if not self.no_extraction:
            print(f"\tExtracting {zip_file}")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(target_dir)
            print(f"\tUnzipped to {target_dir}\n")
        else:
            print(f"\tSkipping extraction of {zip_file}\n")

        print("Cleanup:")
        if not (self.keep_zip or self.no_extraction):
            print(f"\tRemoving the zip file {zip_file}")
            zip_file.unlink()
            print(f"\tRemoved {zip_file}")
        else:
            print(f"\tKeeping the zip file {zip_file}\n")

    def download_metadata(self) -> None:
        """Downloads and extracts the metadata zip file."""
        metadata_url = f"{self.DOWNLOAD_ROOT}/metadata.zip"
        self.download_and_extract(metadata_url, self.fungi_path / "metadata")

    def download_images(self, subset: str, size: str) -> None:
        """
        Downloads image data for a given subset and size.

        Args:
            subset (str): The subset of images to download.
            size (str): The size of the images.
        """
        splits = ["train", "val", "test"] + (
            ["dna-test"] if self.HAS_DNA[subset] else []
        )
        for split in splits:
            img_link = self.generate_img_link(subset=subset, size=size, split=split)
            self.download_and_extract(img_link, self.fungi_path)

    def download_satellite_data(self) -> None:
        """Downloads and extracts satellite data (NIR and RGB)."""
        satellite_files = ["satellite_NIR.zip", "satellite_RGB.zip"]
        for file in satellite_files:
            satellite_link = f"{self.DOWNLOAD_ROOT}/{file}"
            self.download_and_extract(satellite_link, self.fungi_path)

    def download_climatic_data(self) -> None:
        """Downloads and extracts climatic data."""
        climatic_url = f"{self.DOWNLOAD_ROOT}/climatic.zip"
        self.download_and_extract(climatic_url, self.fungi_path)

    def download_masks(self) -> None:
        """Downloads and extracts mask data."""
        masks_url = f"{self.DOWNLOAD_ROOT}/masks.zip"
        self.download_and_extract(masks_url, self.fungi_path)

    def generate_img_link(self, subset: str, size: str, split: str) -> str:
        """
        Generates the download link for image data based on subset, size, and split.

        Args:
            subset (str): Subset of the dataset.
            size (str): Image size.
            split (str): Data split (train, val, test, dna-test).

        Returns:
            str: The generated URL for the image download.
        """
        size_str = f"{size}p" if size != "fullsize" else "fullsize"
        if subset != "full":
            return f"{self.DOWNLOAD_ROOT}/FungiTastic-{self.SUBSET2STR.get(subset, '')}-{split}-{size_str}.zip"
        else:
            return f"{self.DOWNLOAD_ROOT}/FungiTastic-{split}-{size_str}.zip"

    def download(self, subset: str = None, size: str = None) -> None:
        """
        Downloads selected components of the FungiTastic dataset.

        Args:
            subset (str): Subset of the dataset to download (optional).
            size (str): Image size to download (optional).
        """
        if self.metadata:
            self.download_metadata()

        if self.images:
            self.download_images(subset, size)

        if self.satellite:
            self.download_satellite_data()

        if self.climatic:
            self.download_climatic_data()

        if self.masks:
            self.download_masks()

    @staticmethod
    def validate_params(params) -> None:
        """
        Validates the input parameters to ensure correctness.

        Args:
            params: Parsed input arguments.

        Raises:
            FileNotFoundError: If the specified save_path does not exist.
            ValueError: If invalid subset, size, or other input arguments are provided.
        """
        save_path = Path(params.save_path)
        if not save_path.exists():
            raise FileNotFoundError(f"Data root not found: {save_path}")

        if params.images and not (params.subset and params.size):
            raise ValueError("Subset and size must be provided to download images.")

        if params.subset and params.subset not in FungiTasticDownloader.SUBSETS:
            raise ValueError(f"Invalid subset: {params.subset}")

        if params.size and params.size not in FungiTasticDownloader.SIZES:
            raise ValueError(
                f"Invalid size: {params.size}. Available sizes are: {', '.join(FungiTasticDownloader.SIZES)}"
            )

class FungiTasticCatalog(BaseCatalog):
    """
    FungiTastic Dataset

    A hierarchical image classification dataset of fungi species with multiple
    dataset variants (full, mini, fewshot) and multiple image resolutions.

    Args:
        dataset_root (str): Directory where the dataset should be stored.
        download (bool): Whether to automatically download and extract the dataset.
        download_url (str, optional): Custom URL for downloading the dataset.
        col_label (str): Taxonomic level to use as label (default: 'species').
        dataset_size (str): One of ['300p', '500p', '720p', 'fullsize'].
        dataset_variant (str): One of ['full', 'mini', 'fewshot'].
    """

    DATASET_VARIANTS = ["full", "mini", "fewshot"]
    DATASET_SIZES = ["300p", "500p", "720p", "fullsize"]
    LEVELS = ["kingdom", "phylum", "class", "order", "family", "genus", "species", "specificEpithet"]
    LEVEL_LEAF = "species"

    ARCHIVE_NAME = ""
    EXTRACTED_DIRNAME = "FungiTastic"

    def __init__(
        self,
        dataset_root: str,
        download: bool = False,
        download_url: Optional[str] = None,
        keep_zip: bool = True,
        col_label: str = "species",
        dataset_size: str = "500p",
        dataset_variant: str = "full",
        should_encode: bool = True,
    ):
        super().__init__(dataset_root=dataset_root)
        if dataset_variant not in self.DATASET_VARIANTS:
            raise ValueError(f"Invalid dataset_variant: {dataset_variant}. "
                             f"Valid options: {self.DATASET_VARIANTS}")
        if dataset_size not in self.DATASET_SIZES:
            raise ValueError(f"Invalid dataset_size: {dataset_size}. "
                             f"Valid options: {self.DATASET_SIZES}")
        if col_label not in self.LEVELS:
            raise ValueError(f"Invalid label column: {col_label}. "
                             f"Valid options: {self.LEVELS}")

        self.download_url = download_url
        self.keep_zip = keep_zip
        self.archive_path = self.root / self.ARCHIVE_NAME
        self.data_dir = self.root / self.EXTRACTED_DIRNAME

        self.col_label = col_label
        self.dataset_size = dataset_size
        self.dataset_variant = dataset_variant
        self.should_encode = should_encode
        self.encoders = None

        if download:
            self._download_and_extract()

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"FungiTastic dataset not found in {self.data_dir}. "
                f"Use download=True to fetch it."
            )

        self.df = self._load_metadata_to_df()
        self.default_splits = self._load_default_splits()

    def _download_and_extract(self):
        """Download and extract FungiTastic dataset using FungiTasticDownloader.

        https://github.com/BohemianVRA/FungiTastic/blob/main/dataset/download.py
        """
        print(f"Preparing to download FungiTastic dataset into {self.root}")
        # Map variants and sizes to downloader parameters
        subset_map = {"full": "full", "fewshot": "fs", "mini": "m"}
        subset = subset_map[self.dataset_variant]
        size = self.dataset_size.replace("p", "")  # convert e.g. '500p' -> '500'

        downloader = FungiTasticDownloader(
            save_path=self.root,
            rewrite=False,
            keep_zip=self.keep_zip,
            no_extraction=False,
            metadata=True,
            images=True,
        )

        try:
            downloader.download(subset=subset, size=size)
        except Exception as e:
            raise RuntimeError(f"Failed to download FungiTastic dataset: {e}")

        # After extraction, ensure expected directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset extraction failed. Expected directory not found: {self.data_dir}"
            )
        print("FungiTastic dataset download and extraction complete.")

    def _load_metadata_to_df(self) -> pd.DataFrame:
        """Load train, validation, and test metadata for the selected variant."""
        variant = self.dataset_variant
        size = self.dataset_size
        base_dir = self.data_dir

        meta_dir = base_dir / "metadata" / self._variant_folder(variant)
        img_dir = base_dir / self._variant_folder(variant)

        split_filenames = self._split_files(variant=variant)

        dfs = []
        for split, csv_name in split_filenames.items():
            csv_path = meta_dir / csv_name
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing metadata file: {csv_path}")

            df = pd.read_csv(csv_path)
            df["split"] = split
            df[IMAGE_PATH_KEY_NAME] = (
                    img_dir / split / size / df[FILENAME_KEY_NAME]
            ).astype(str)
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        df["hazard"] = df["poisonous"]
        df[self.LEVELS] = df[self.LEVELS].fillna("missing")
        df[TARGET_KEY_NAME] = df["category_id"].fillna(-1).astype(int)
        cols_encode = {*self.LEVELS}
        if self.should_encode:
            self._encode_features(df, cols_encode)
        return df

    def _encode_features(self, df: pd.DataFrame, features_encode: Sequence[str]):
        metadata_train = df.query("split == 'train'")
        self.encoders = {
            col: OrdinalEncoder.from_categories(metadata_train[col].values)
            for col in features_encode
        }
        for col in features_encode:
            encoder = self.encoders[col]
            data_to_encode = df[col]
            df[col] = encoder.encode(data_to_encode)

    def _load_default_splits(self):
        """Prepare train/val/test splits and hierarchical encoders."""
        return DatasetSplits(
            train_indexes=self.df.index[self.df["split"] == "train"].to_list(),
            val_indexes=self.df.index[self.df["split"] == "val"].to_list(),
            test_indexes=self.df.index[self.df["split"] == "test"].to_list(),
        )

    @staticmethod
    def _variant_folder(variant: str) -> str:
        return {
            "full": "FungiTastic",
            "mini": "FungiTastic-Mini",
            "fewshot": "FungiTastic-FewShot",
        }[variant]

    def _split_files(self, variant: str) -> dict:
        if variant == "fewshot":
            splits = {
                "train": f"{self._variant_folder(variant)}-Train.csv",
                "val": f"{self._variant_folder(variant)}-Val.csv",
                "test": f"{self._variant_folder(variant)}-Test.csv",
            }
        elif variant == "mini":
            splits = {
                "train": f"{self._variant_folder(variant)}-Train.csv",
                "val": f"{self._variant_folder(variant)}-Val.csv",
                "test": f"{self._variant_folder(variant)}-Test.csv",
            }
        else:
            splits = {
                "train": f"{self._variant_folder(variant)}-Train.csv",
                "val": f"{self._variant_folder(variant)}-ClosedSet-Val.csv",
                "test": f"{self._variant_folder(variant)}-ClosedSet-Test.csv",
            }
        return splits


if __name__ == "__main__":
    dataset_dir = os.environ.get("DATASET_ROOT", "~/datasets")

    transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    catalog = FungiTasticCatalog(
        dataset_root=dataset_dir,
        dataset_variant="fewshot",
        dataset_size="720p",
        download=False,
    )

    image_dataset = ImageDataset(
        df=catalog.get_metadata(),
        transform=transform,
    )

    print(f"Samples: {len(image_dataset)}")
    item = image_dataset[0]
    img, label = item[IMAGE_KEY_NAME], item[TARGET_KEY_NAME]
    print(f"Image shape: {img.shape}, Label: {label}")
