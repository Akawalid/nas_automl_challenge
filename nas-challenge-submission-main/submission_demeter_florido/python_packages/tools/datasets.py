import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from warnings import warn

import numpy as np
import requests
import torch
import torch.utils.data
from gromo.utils.utils import global_device
from torch.utils import data
from torchvision import datasets, transforms

from tools.augmentations import get_transforms, npy_datasets


def _npy_webdataset_url_overrides() -> dict[str, str]:
    """Optional JSON map of dataset class name -> full zip URL (``NPY_WEBDATASET_URL_OVERRIDES``)."""
    raw = os.environ.get("NPY_WEBDATASET_URL_OVERRIDES", "").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("NPY_WEBDATASET_URL_OVERRIDES must be a JSON object")
    return {str(k): str(v) for k, v in data.items()}


# Single source of truth for the ``(article_id, version)`` pair of each
# NpyWebDataset subclass. To bump a dataset to a newer Figshare version, edit
# this table only.
_FIGSHARE_ARTICLES: dict[str, tuple[str, int]] = {
    "AddNIST": ("24574354", 1),
    "MultNIST": ("24574678", 1),
    "CIFARTile": ("24551539", 1),
    "LanguageASPELL": ("24574729", 1),
    "Gutenberg": ("24574753", 1),
    "GeoClassing": ("24050256", 3),
    "Chesseract": ("24118743", 2),
    "GameOfLife": ("30000835", 1),
}


def _figshare_article_zip_url(dataset_class_name: str) -> str:
    """Return the Figshare *ndownloader* URL for a versioned article zip.

    Looks up ``(article_id, version)`` from :data:`_FIGSHARE_ARTICLES`.

    The same archives are linked from Newcastle's Data Repository
    (``data.ncl.ac.uk``), but the NCL ``ndownloader`` path is often behind AWS WAF
    (HTTP 202, empty body, ``x-amzn-waf-action: challenge``). Figshare's
    ``ndownloader.figshare.com`` serves identical zips and works with
    :mod:`requests` and automation.

    * ``NPY_WEBDATASET_NDOWNLOADER_BASE`` — default ``https://ndownloader.figshare.com``
    * ``NPY_WEBDATASET_URL_OVERRIDES`` — JSON object, e.g. ``{"MultNIST":"https://..."}``
    """
    overrides = _npy_webdataset_url_overrides()
    if dataset_class_name in overrides:
        return overrides[dataset_class_name]
    if dataset_class_name not in _FIGSHARE_ARTICLES:
        raise KeyError(
            f"No Figshare article registered for {dataset_class_name!r}. "
            f"Add an entry to tools.datasets._FIGSHARE_ARTICLES."
        )
    article_id, version = _FIGSHARE_ARTICLES[dataset_class_name]
    base = os.environ.get(
        "NPY_WEBDATASET_NDOWNLOADER_BASE", "https://ndownloader.figshare.com"
    ).rstrip("/")
    return f"{base}/articles/{article_id}/versions/{version}"


class SinDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.nb_sample = 1_000

    def __len__(self):
        return self.nb_sample

    def __getitem__(self, _):
        data = torch.rand(1) * 2 * torch.pi
        target = torch.sin(data)
        return data, target


class NpyWebDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        url: str,
        train: bool = True,
        root: str = "data/webdatasets/npy",
        name: str = "",
        download: bool = True,
        transform: Optional[Callable] = None,
        data_key: str = "_x",
        label_key: str = "_y",
    ):
        self.url = url
        self.name = name
        self.train = train
        self.root = Path(os.path.expanduser(root))
        self.download = download
        self.transform = transform
        self.data_key = data_key
        self.label_key = label_key

        self.local_zip_path = self._download_and_extract()
        self.data_files, self.label_files = self._find_data_and_labels()
        self.data, self.labels = self._load_data() if download else (None, None)

    @staticmethod
    def _validate_zip_bytes(content: bytes, url: str) -> None:
        """Ensure payload looks like a zip before writing or extracting."""
        if len(content) == 0:
            raise ValueError(
                "Downloaded archive is empty. The host may require a browser or "
                "return a WAF challenge instead of the dataset zip. "
                f"URL: {url}"
            )
        if len(content) < 4 or content[:2] != b"PK":
            preview = content[:300].decode("utf-8", errors="replace")
            raise ValueError(
                "Response is not a zip file (missing PK header). "
                "Often HTML/WAF or an error page. "
                f"URL: {url}\nPreview: {preview!r}"
            )

    @staticmethod
    def _validate_http_response_for_zip(r: requests.Response, url: str) -> bytes:
        """Raise with a clear message when bots get challenges instead of archives."""
        waf = r.headers.get("x-amzn-waf-action") or r.headers.get("X-Amzn-Waf-Action")
        if waf:
            raise RuntimeError(
                "Download blocked by upstream WAF "
                f"(x-amzn-waf-action={waf!r}). Use a browser download, mirror, or "
                f"prefetch data into `root`. URL: {url}"
            )
        r.raise_for_status()
        content = r.content
        if r.status_code == 202 and len(content) == 0:
            raise RuntimeError(
                "HTTP 202 with empty body (common for WAF/async placeholders). "
                f"No zip was returned. URL: {url}"
            )
        NpyWebDataset._validate_zip_bytes(content, url)
        return content

    @staticmethod
    def _assert_existing_zip_valid(zip_path: Path, url: str) -> None:
        if zip_path.stat().st_size == 0:
            raise ValueError(
                f"Existing file is empty (0 bytes): {zip_path}. "
                "Delete it and retry after fixing network/mirror access. "
                f"URL: {url}"
            )
        with zip_path.open("rb") as f:
            head = f.read(4)
        if len(head) < 2 or head[:2] != b"PK":
            raise ValueError(
                f"Existing file is not a zip (missing PK header): {zip_path}. "
                "Remove it and retry. "
                f"URL: {url}"
            )

    @property
    def targets(self) -> torch.Tensor | None:
        return self.labels

    def _download_and_extract(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.name == "":
            hash_name = hashlib.sha256(self.url.encode()).hexdigest()
        else:
            hash_name = self.name
        zip_path = self.root / f"{hash_name}.zip"
        extract_dir = self.root / f"{hash_name}_extracted"

        _REQUEST_HEADERS = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; experimental_grow/1.0; "
                "+https://github.com/) NpyWebDataset"
            )
        }

        if not zip_path.exists():
            r = requests.get(self.url, timeout=120, headers=_REQUEST_HEADERS)
            content = self._validate_http_response_for_zip(r, self.url)
            zip_path.write_bytes(content)
        else:
            self._assert_existing_zip_valid(zip_path, self.url)

        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

        return extract_dir

    def _find_data_and_labels(self) -> Tuple[List[Path], List[Path]]:
        files = list(self.local_zip_path.rglob("*.npy"))
        prefix = ["train", "valid"] if self.train else ["test"]

        data_files = sorted(
            [
                f
                for f in files
                if (self.data_key in f.name) and any(ix in f.name for ix in prefix)
            ]
        )
        label_files = sorted(
            [
                f
                for f in files
                if (self.label_key in f.name) and any(ix in f.name for ix in prefix)
            ]
        )

        assert len(data_files) == len(label_files), "Mismatch in data/label file counts"
        assert len(data_files) > 0, f"No matching .npy files found for prefix '{prefix}'"

        return data_files, label_files

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        data = torch.from_numpy(
            np.concatenate([np.load(f) for f in self.data_files], axis=0)
        )
        labels = torch.from_numpy(
            np.concatenate([np.load(f) for f in self.label_files], axis=0)
        ).share_memory_()
        # Reshape to (N, H, W, C) so shape[-1]==C and shape[1:-1]==(H,W),
        # matching the torchvision dataset convention
        if data.ndim == 4:       # (N, C, H, W) -> (N, H, W, C)
            data = data.permute(0, 2, 3, 1)
        elif data.ndim == 3:     # (N, H, W) grayscale -> (N, H, W, 1)
            data = data.unsqueeze(-1)
        data = data.contiguous().share_memory_()
        return data, labels

    def __getitem__(self, index):
        if self.download:
            assert self.data is not None
            assert self.labels is not None
            x = self.data[index].numpy().copy()
            y = self.labels[index].item()
        else:
            file_idx, local_idx = self._resolve_index(index)
            x = np.load(self.data_files[file_idx])[local_idx]
            if x.ndim == 3:  # (C, H, W) stored in file -> HWC for ToTensor
                x = np.moveaxis(x, 0, -1)
            x = x.copy()
            y = np.load(self.label_files[file_idx])[local_idx].item()

        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def _resolve_index(self, index) -> Tuple[int, int]:
        """Find which file and which index in file corresponds to a global index"""
        cumulative = 0
        for i, file in enumerate(self.data_files):
            n = np.load(file, mmap_mode="r").shape[0]
            if index < cumulative + n:
                return i, index - cumulative
            cumulative += n
        raise IndexError("Index out of bounds")

    def __len__(self):
        if self.download:
            assert self.data is not None
            return len(self.data)
        else:
            total = 0
            for f in self.data_files:
                total += np.load(f, mmap_mode="r").shape[0]
            return total


class AddNIST(NpyWebDataset):
    def __init__(
        self,
        train: bool = True,
        root: str = "data/webdatasets/npy",
        download: bool = True,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            url=_figshare_article_zip_url("AddNIST"),
            name="AddNIST",
            train=train,
            root=root,
            download=download,
            transform=transform,
        )



class MultNIST(NpyWebDataset):
    def __init__(
        self,
        train: bool = True,
        root: str = "data/webdatasets/npy",
        download: bool = True,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            url=_figshare_article_zip_url("MultNIST"),
            name="MultNIST",
            train=train,
            root=root,
            download=download,
            transform=transform,
        )


class CIFARTile(NpyWebDataset):
    def __init__(
        self,
        train: bool = True,
        root: str = "data/webdatasets/npy",
        download: bool = True,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            url=_figshare_article_zip_url("CIFARTile"),
            train=train,
            root=root,
            name="CIFARTile",
            download=download,
            transform=transform,
        )



class LanguageASPELL(NpyWebDataset):
    def __init__(
        self,
        train: bool = True,
        root: str = "data/webdatasets/npy",
        download: bool = True,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            url=_figshare_article_zip_url("LanguageASPELL"),
            train=train,
            root=root,
            name="LanguageASPELL",
            download=download,
            transform=transform,
        )



class Gutenberg(NpyWebDataset):
    def __init__(
        self,
        train: bool = True,
        root: str = "data/webdatasets/npy",
        download: bool = True,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            url=_figshare_article_zip_url("Gutenberg"),
            train=train,
            root=root,
            name="Gutenberg",
            download=download,
            transform=transform,
        )



class GeoClassing(NpyWebDataset):
    def __init__(
        self,
        train: bool = True,
        root: str = "data/webdatasets/npy",
        download: bool = True,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            url=_figshare_article_zip_url("GeoClassing"),
            train=train,
            root=root,
            name="GeoClassing",
            download=download,
            transform=transform,
        )



class Chesseract(NpyWebDataset):
    def __init__(
        self,
        train: bool = True,
        root: str = "data/webdatasets/npy",
        download: bool = True,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            url=_figshare_article_zip_url("Chesseract"),
            train=train,
            root=root,
            name="Chesseract",
            download=download,
            transform=transform,
        )



class GameOfLife(NpyWebDataset):
    def __init__(
        self,
        train: bool = True,
        root: str = "data/webdatasets/npy",
        download: bool = True,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            url=_figshare_article_zip_url("GameOfLife"),
            train=train,
            root=root,
            name="GameOfLife",
            download=download,
            transform=transform,
        )


known_datasets = {
    "sin": SinDataset,
    "mnist": datasets.MNIST,
    "fashion-mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "svhn": datasets.SVHN,
    "addnist": AddNIST,
    "multnist": MultNIST,
    "cifartile": CIFARTile,
    "dry-ricker": None,
    "language": LanguageASPELL,
    "gutenberg": Gutenberg,
    "geoclassing": GeoClassing,
    "chesseract": Chesseract,
    "gameoflife": GameOfLife,
    "food101": datasets.Food101,
}


def get_num_classes(dataset_name: str) -> int:
    if dataset_name not in known_datasets:
        raise ValueError(f"Unknown dataset {dataset_name}")

    num_classes_map = {
        "mnist": 10,
        "fashion-mnist": 10,
        "cifar10": 10,
        "cifar100": 100,
        "svhn": 10,
        "sin": 1,
        "addnist": 20,
        "multnist": 10,
        "cifartile": 4,
        "dry-ricker": 2,
        "language": 10,
        "gutenberg": 6,
        "geoclassing": 10,
        "chesseract": 3,
        "gameoflife": 25,
        "food101": 101,
    }

    if dataset_name not in num_classes_map:
        raise ValueError(
            f"Unsupported dataset {dataset_name}. Cannot determine the number of classes."
        )

    return num_classes_map[dataset_name]


def make_dataloader(
    dataset: data.Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    device: torch.device = global_device(),
    seed: int | None = None,
    drop_last: bool = True,
) -> data.DataLoader:
    """
    Create a DataLoader from a dataset.

    Parameters
    ----------
    dataset : data.Dataset
        The dataset to wrap in a DataLoader.
    batch_size : int, default=64
        Batch size for the dataloader.
    shuffle : bool, default=False
        Whether to shuffle the data.
    num_workers : int, default=0
        Number of worker processes for data loading. Automatically set to 0 when using CPU.
    device : torch.device, default=global_device()
        Device to use for data loading. Affects pin_memory and num_workers settings.
    seed : int | None, default=None
        Seed for random number generator for shuffling. If None, no generator will be set up.

    Returns
    -------
    data.DataLoader
        DataLoader wrapping the dataset.
    """
    pin_memory = device != torch.device("cpu")
    num_workers = num_workers if pin_memory else 0

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
        drop_last=drop_last,
    )


def get_dataloaders(
    dataset_name: str = "cifar10",
    dataset_path: str = "dataset",
    nb_class: int | None = None,
    split_train_val: float = 0.0,
    data_augmentation: list[str] | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    device: torch.device = global_device(),
    shuffle: bool = True,
    seed: int | None = None,
    separate_growth_set: bool = False,
    growth_set_split_ratio: float = 0.2,
    val_set_augmentation: bool = False,
    growth_set_augmentation: bool = True,
    drop_last: list[bool] | None = None,
) -> (
    tuple[data.DataLoader, data.DataLoader, data.DataLoader, tuple[int, int, int]]
    | tuple[
        data.DataLoader,
        data.DataLoader,
        data.DataLoader,
        data.DataLoader,
        tuple[int, int, int],
    ]
):
    """
    Create train, validation, and test dataloaders for specified dataset.

    Parameters
    ----------
    dataset_name : str, default="cifar10"
        Name of the dataset to load. Must be one of the known datasets:
        "mnist", "fashion-mnist", "cifar10", "cifar100", "svhn", "sin", "addnist", "cifartile"
    dataset_path : str, default="dataset"
        Path to the dataset directory or where to download it
    nb_class : int or None, default=None
        Number of classes to keep in the dataset. If specified, only the first
        nb_class classes will be retained. If None, all classes are kept.
    split_train_val : float, default=0.0
        Proportion of the training set to use as validation set (0.0 to 1.0).
        0.0 means no validation split, 1.0 means all training data becomes validation.
    data_augmentation : list of str or None, default=None
        List of data augmentation techniques to apply to training data.
        Available options: "horizontal_flip", "rotation", "crop", "autoaugment", "randaugment"
    batch_size : int, default=64
        Batch size for all dataloaders
    num_workers : int, default=0
        Number of worker processes for data loading. Set to 0 for single-threaded loading.
        Automatically set to 0 when using CPU device.
    device : torch.device, default=global_device()
        Device to use for data loading. Affects pin_memory and num_workers settings.
    shuffle : bool, default=True
        Whether to shuffle the training dataloader. Validation and test loaders are never shuffled.
    seed : int or None, default=None
        Seed for random number generator for shuffling. If None no generator will be set-up.
    separate_growth_set : bool, default=False
        If True, split the training set into separate growth and training datasets.
    growth_set_split_ratio : float, default=0.2
        Fraction of training data to use for growth when separate_growth_set is True.
    val_set_augmentation : bool, default=False
        If True, apply data augmentation to the validation set.
    growth_set_augmentation : bool, default=True
        If True, apply data augmentation to the growth set when separate_growth_set is True.
    drop_last : list[bool], default=None
        Whether to drop the last incomplete batch in each dataloader.
        By default, True only for training dataloader, False otherwise.

    Returns
    -------
    tuple[data.DataLoader, data.DataLoader, data.DataLoader, tuple[int]] or
    tuple[data.DataLoader, data.DataLoader, data.DataLoader, data.DataLoader, tuple[int]]
        When separate_growth_set is False (default):
        - train_dataloader: DataLoader for training data with augmentations
        - val_dataloader: DataLoader for validation data (subset of training data)
        - test_dataloader: DataLoader for test data
        - data_shape: Tuple representing the shape of a single data sample (e.g., (3, 32, 32) for CIFAR-10)

        When separate_growth_set is True:
        - growth_dataloader: DataLoader for growth data (used for computing optimal updates)
        - train_dataloader: DataLoader for training data (used for standard training)
        - val_dataloader: DataLoader for validation data
        - test_dataloader: DataLoader for test data
        - data_shape: Tuple representing the shape of a single data sample

    Examples
    --------
    Basic usage:
    >>> train_loader, val_loader, test_loader, shape = get_dataloaders("cifar10")

    With validation split and augmentation:
    >>> train_loader, val_loader, test_loader, shape = get_dataloaders(
    ...     dataset_name="cifar10",
    ...     split_train_val=0.2,
    ...     data_augmentation=["horizontal_flip", "crop"]
    ... )

    With separate growth set:
    >>> growth_loader, train_loader, val_loader, test_loader, shape = get_dataloaders(
    ...     dataset_name="cifar10",
    ...     separate_growth_set=True,
    ...     growth_set_split_ratio=0.2
    ... )

    Limit to first 5 classes only:
    >>> train_loader, val_loader, test_loader, shape = get_dataloaders(
    ...     dataset_name="cifar10",
    ...     nb_class=5
    ... )
    """

    if separate_growth_set:
        # When separate_growth_set=True:
        # 1. First split off validation: split_train_val
        # 2. From remaining (1 - split_train_val), split growth: growth_set_split_ratio
        # splits_sizes order: [growth, train, val] (before test is added)
        splits_sizes = [
            growth_set_split_ratio,  # growth from remaining
            (1 - growth_set_split_ratio - split_train_val),  # train from remaining
            split_train_val,  # validation
        ]
        # data_augmentation_split order: [growth, train, val]
        data_augmentation_split = [growth_set_augmentation, True, val_set_augmentation]
        # shuffled order: [growth, train, val, test]
        shuffled = [shuffle, shuffle, False, False]
        # seeds order: [growth, train, val, test]
        seeds = [seed, seed, None, None]
        if drop_last is None:
            drop_last = [False, True, False, False]
    else:
        # splits_sizes order: [train, val] (before test is added)
        splits_sizes = [1.0 - split_train_val, split_train_val]
        # data_augmentation_split order: [train, val]
        data_augmentation_split = [True, val_set_augmentation]
        # shuffled order: [train, val, test]
        shuffled = [shuffle, False, False]
        # seeds order: [train, val, test]
        seeds = [seed, None, None]
        if drop_last is None:
            drop_last = [True, False, False]

    # load the dataset and create the dataloaders
    dataset_splits = get_dataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        nb_class=nb_class,
        splits_sizes=splits_sizes,
        data_augmentation=data_augmentation,
        split_data_augmentation_activation=data_augmentation_split,
    )
    # dataset_splits contains: [splits..., test]
    # When separate_growth_set=True: [growth, train, val, test]
    # When separate_growth_set=False: [train, val, test]

    data_shape = dataset_splits[0][0][0].shape

    dataloaders = []
    for i, dataset in enumerate(dataset_splits[:-1]):  # Exclude test dataset
        dataloader = make_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffled[i],
            num_workers=num_workers,
            device=device,
            seed=seeds[i],
            drop_last=drop_last[i],
        )
        dataloaders.append(dataloader)

    # Add test dataloader (never shuffled)
    test_dataloader = make_dataloader(
        dataset_splits[-1],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device,
        seed=None,
        drop_last=drop_last[-1],
    )
    dataloaders.append(test_dataloader)

    return (*dataloaders, data_shape)


def get_dataset(
    dataset_name: str,
    dataset_path: str,
    nb_class: int | None = None,
    splits_sizes: list[float] | float | None = None,
    data_augmentation: list[str] | None = None,
    split_data_augmentation_activation: list[bool] | None = None,
) -> list[data.Dataset]:
    """
    Get the dataset

    Parameters
    ----------
    dataset_name: str
        The name of the dataset
    dataset_path: str
        The path to the dataset or where to download it
    nb_class: int | None
        The number of classes to keep in the dataset, for example, if nb_class=5, only the first 5 classes will be kept
    splits_sizes: list[float] | float | None
        The proportions of the training set to use as validation set and other splits (has to sum to 1.0)
        or a float indicating the proportion of validation set (between 0.0 and 1.0).
    data_augmentation: list[str] | None
        The data augmentation to apply to the training set
    split_data_augmentation_activation: list[bool] | None
        List of booleans indicating whether to apply data augmentation to each split

    Returns
    -------
    list[datasets.Dataset]
        The original train set split and the test set
    """
    # check if the dataset is known
    if dataset_name not in known_datasets:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if splits_sizes is None:
        splits_sizes = [1.0, 0.0]  # train, val
    elif isinstance(splits_sizes, float):
        splits_sizes = [1.0 - splits_sizes, splits_sizes]  # train, val
    elif isinstance(splits_sizes, (list, tuple)):
        if not (0.0 <= sum(splits_sizes) <= 1.0):
            raise ValueError("splits_sizes must sum to a value between 0.0 and 1.0")
    else:
        raise TypeError("splits_sizes must be a float or a list/tuple of floats")

    if split_data_augmentation_activation is None:
        split_data_augmentation_activation = [True] * len(splits_sizes)

    # sin dataset (special case)
    if dataset_name == "sin":
        train_data = known_datasets["sin"]()  # pyright: ignore[reportCallIssue]
        val_data = known_datasets["sin"]()  # pyright: ignore[reportCallIssue]
        test_data = known_datasets["sin"]()  # pyright: ignore[reportCallIssue]
        return [train_data, val_data, test_data]

    # get the dataset
    dataset = known_datasets[dataset_name]
    datasets_transforms, augmentation_transforms = get_transforms(
        dataset_name, data_augmentation
    )

    # load the train and test datasets
    train_test_args = {
        "root": dataset_path,
        "download": True,
        "transform": transforms.Compose(datasets_transforms),
    }
    train_split_args = (
        {"train": True} if dataset_name not in ("svhn", "food101") else {"split": "train"}
    )
    test_split_args = (
        {"train": False} if dataset_name not in ("svhn", "food101") else {"split": "test"}
    )

    train_val_data_with_aug = dataset(
        **train_test_args, **train_split_args
    )  # pyright: ignore[reportArgumentType]
    if dataset_name in npy_datasets:
        # npy data is pre-normalized float; ToTensor must come before augmentation
        full_aug_transforms = datasets_transforms + augmentation_transforms
    else:
        # PIL datasets: augment on PIL image before ToTensor
        full_aug_transforms = augmentation_transforms + datasets_transforms
    train_val_data_with_aug.transform = transforms.Compose(full_aug_transforms)

    # create a separate instance for the validation set to avoid transform issues
    train_val_data = dataset(
        **train_test_args, **train_split_args
    )  # pyright: ignore[reportArgumentType]
    test_data = dataset(
        **train_test_args, **test_split_args
    )  # pyright: ignore[reportArgumentType]

    # filter the classes
    if nb_class is not None and nb_class > get_num_classes(dataset_name):
        warn(
            f"{nb_class=} is greater than the number of classes in {dataset_name} "
            f"which is {get_num_classes(dataset_name)}. Setting nb_class to None."
        )

    initial_nb_classes = get_num_classes(dataset_name)

    if nb_class is not None and nb_class < initial_nb_classes:
        train_val_data: torch.utils.data.Dataset = filter_classes(
            train_val_data,
            nb_class,
            "labels" if dataset_name in ("svhn", "food101") else "targets",
        )
        train_val_data_with_aug: torch.utils.data.Dataset = filter_classes(
            train_val_data_with_aug,
            nb_class,
            "labels" if dataset_name in ("svhn", "food101") else "targets",
        )
        test_data: torch.utils.data.Dataset = filter_classes(
            test_data,
            nb_class,
            "labels" if dataset_name in ("svhn", "food101") else "targets",
        )

    # split the training set
    splits = torch.utils.data.random_split(train_val_data, splits_sizes)
    # assign the validation dataset to the val_data subset to avoid transform issues

    for i, split in enumerate(splits):
        if split_data_augmentation_activation[i]:
            split.dataset = train_val_data_with_aug
        else:
            split.dataset = train_val_data

    splits.append(test_data)

    return splits


def filter_classes(
    dataset: torch.utils.data.Dataset, nb_class: int | None, field_name: str
) -> torch.utils.data.Dataset:
    """
    Filter dataset to keep only the first nb_class classes.

    This function modifies the dataset in-place to retain only samples belonging
    to the first nb_class classes (indexed from 0). Useful for creating smaller
    subsets of datasets for experimentation or testing.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to filter. Must have 'data' attribute and a targets/labels
        attribute specified by field_name.
    nb_class : int or None
        Number of classes to keep. If None, no filtering is performed.
        Must be greater than 1 if specified.
    field_name : str
        Name of the attribute containing class labels/targets.
        Typically "targets" for most datasets or "labels" for SVHN.

    Returns
    -------
    torch.utils.data.Dataset
        The filtered dataset with only the first nb_class classes.

    Examples
    --------
    Keep only first 5 classes from CIFAR-10:
    >>> dataset = datasets.CIFAR10(root='data', train=True)
    >>> filtered = filter_classes(dataset, 5, 'targets')

    Notes
    -----
    - The function assumes class indices start from 0 and are consecutive
    - The dataset is modified in-place
    - Issues a warning if nb_class <= 1
    - No operation is performed if nb_class is None
    """
    if nb_class is not None and nb_class <= 1:
        warn(f"{nb_class=} should be greater than 1")
    if nb_class is not None:
        targets = getattr(dataset, field_name)
        if isinstance(targets, list):
            targets = torch.tensor(targets)
        elif isinstance(targets, np.ndarray):
            targets = torch.tensor(targets)
        assert isinstance(targets, torch.Tensor)
        assert targets.ndim == 1
        assert targets.min() == 0
        idx = targets <= (nb_class - 1)
        dataset.data = dataset.data[idx]  # pyright: ignore[reportAttributeAccessIssue]
        setattr(dataset, field_name, targets[idx])
    return dataset


def dataset_description(dataset: torch.utils.data.Dataset) -> str:
    """Return a comprehensive string description of a Dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The Dataset to describe.

    Returns
    -------
    str
        A detailed string description of the Dataset including transforms.
    """
    description = [
        "Dataset Information:",
        f"  Type: {type(dataset).__name__}",
        f"  Length: {len(dataset)}",
    ]

    # Handle different dataset types (wrapped vs direct)
    actual_dataset = dataset
    if hasattr(dataset, "dataset"):  # Subset wrapper
        actual_dataset = dataset.dataset
        description.append(f"  Wrapped dataset type: {type(actual_dataset).__name__}")
        if hasattr(dataset, "indices"):
            description.append(f"  Subset indices: {len(dataset.indices)} samples")

    # Unwrap nested Subsets to get to the root dataset for transform info
    root_dataset = actual_dataset
    while hasattr(root_dataset, "dataset"):
        root_dataset = root_dataset.dataset

    # Transform information - check root dataset
    if hasattr(root_dataset, "transform") and root_dataset.transform is not None:
        description.append("  Transforms:")
        transform = root_dataset.transform

        if hasattr(transform, "transforms"):  # Compose transform
            for i, t in enumerate(transform.transforms):
                description.append(f"    {i + 1}. {type(t).__name__}")
                # Add specific parameters for common transforms
                if hasattr(t, "__dict__"):
                    params = []
                    for key, value in t.__dict__.items():
                        if not key.startswith("_") and key != "interpolation":
                            if isinstance(value, (int, float, bool, str, tuple, list)):
                                params.append(f"{key}={value}")
                    if params:
                        description.append(f"       {', '.join(params)}")
        else:
            description.append(f"    Single transform: {type(transform).__name__}")
    else:
        description.append("  Transforms: None")

    # Target transform information - check root dataset
    if (
        hasattr(root_dataset, "target_transform")
        and root_dataset.target_transform is not None
    ):
        description.append(
            f"  Target transforms: {type(root_dataset.target_transform).__name__}"
        )
    else:
        description.append("  Target transforms: None")

    # Additional dataset-specific info - check root dataset
    if hasattr(root_dataset, "root"):
        description.append(f"  Root directory: {root_dataset.root}")
    if hasattr(root_dataset, "train"):
        description.append(f"  Training set: {root_dataset.train}")
    if hasattr(root_dataset, "split"):
        description.append(f"  Split: {root_dataset.split}")

    # Try to get a sample to show data shape
    try:
        sample_data, sample_target = dataset[0]
        description.append(f"  Sample data shape: {sample_data.shape}")
        description.append(f"  Sample data type: {sample_data.dtype}")
        description.append(f"  Sample target type: {type(sample_target)}")
        if hasattr(sample_target, "shape"):
            description.append(f"  Sample target shape: {sample_target.shape}")
    except Exception as e:
        description.append(f"  Could not access sample: {e}")

    return "\n".join(description)


def dataloader_description(data_loader: torch.utils.data.DataLoader) -> str:
    """Return a comprehensive string description of a DataLoader.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The DataLoader to describe.

    Returns
    -------
    str
        A detailed string description of the DataLoader including dataset info.
    """

    # Basic DataLoader info
    description = [
        "DataLoader Configuration:",
        f"  Batch size: {data_loader.batch_size}",
        f"  Number of batches: {len(data_loader)}",
        f"  Shuffle: {data_loader.sampler.__class__.__name__ != 'SequentialSampler'}",
        f"  Pin memory: {data_loader.pin_memory}",
        f"  Num workers: {data_loader.num_workers}",
        f"  Persistent workers: {data_loader.persistent_workers}",
        f"  Drop last: {data_loader.drop_last}",
        "",
    ]

    # Add dataset description
    dataset_desc = dataset_description(data_loader.dataset)
    description.append(dataset_desc)

    return "\n".join(description)
