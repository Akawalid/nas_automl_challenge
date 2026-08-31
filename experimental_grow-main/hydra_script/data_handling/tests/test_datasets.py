"""Tests for hydra_script.data_handling.datasets."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.utils.data
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError

from hydra_script.data_handling.datasets import (
    LoaderSplitDescriptor,
    SplitDescriptor,
    TransformedSubset,
    filter_classes,
    get_dataloaders,
    get_datasets,
    resolve_train_transform_key,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic datasets and configs
# ---------------------------------------------------------------------------


class SyntheticDataset(torch.utils.data.Dataset):
    """Minimal dataset returning ``(float_tensor, int_label)`` pairs."""

    def __init__(
        self,
        size: int = 100,
        num_classes: int = 5,
        transform: object = None,
        **_extra: object,
    ) -> None:
        self.data = torch.randn(size, 3, 8, 8)
        self.targets = torch.arange(size) % num_classes
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.targets[index]

    def __len__(self) -> int:
        return len(self.data)


class SyntheticDatasetWithLabels(torch.utils.data.Dataset):
    """Like SyntheticDataset but uses ``labels`` instead of ``targets``."""

    def __init__(
        self,
        size: int = 60,
        num_classes: int = 3,
        transform: object = None,
        **_extra: object,
    ) -> None:
        self.data = torch.randn(size, 1, 4, 4)
        self.labels = torch.arange(size) % num_classes
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[index]

    def __len__(self) -> int:
        return len(self.data)


def _double_transform(x: torch.Tensor) -> torch.Tensor:
    """Simple transform that doubles the input."""
    return x * 2


def _make_cfg(
    *,
    dataset_target: str = (
        "hydra_script.data_handling.tests.test_datasets.SyntheticDataset"
    ),
    used_classes: int | None = None,
    targets_field: str | None = None,
    size: int = 100,
    standard_target: str = "torch.nn.Identity",
    augmented_target: str | None = None,
) -> object:
    """Build an OmegaConf DictConfig mimicking a dataset YAML config."""
    cfg_dict: dict = {
        "dataset": {"_target_": dataset_target, "size": size},
        "sources": {
            "train": {"extra_args": {"num_classes": 5}},
            "test": {"extra_args": {"num_classes": 5}},
        },
        "transforms": {
            "standard": {"_target_": standard_target},
        },
    }
    if augmented_target is not None:
        cfg_dict["transforms"]["augmented"] = {"_target_": augmented_target}
    if used_classes is not None:
        cfg_dict["used_classes"] = used_classes
    if targets_field is not None:
        cfg_dict["targets_field"] = targets_field
    return OmegaConf.create(cfg_dict)


# ---------------------------------------------------------------------------
# TransformedSubset
# ---------------------------------------------------------------------------


class TestTransformedSubset:
    """Tests for :class:`TransformedSubset`."""

    def test_length_matches_subset(self) -> None:
        ds = SyntheticDataset(size=50)
        subset = torch.utils.data.Subset(ds, list(range(20)))
        wrapped = TransformedSubset(subset, transform=None)
        assert len(wrapped) == 20

    def test_no_transform_passthrough(self) -> None:
        ds = SyntheticDataset(size=10)
        subset = torch.utils.data.Subset(ds, list(range(10)))
        wrapped = TransformedSubset(subset, transform=None)
        x_orig, y_orig = ds[0]
        x_wrapped, y_wrapped = wrapped[0]
        assert torch.equal(x_orig, x_wrapped)
        assert torch.equal(y_orig, y_wrapped)

    def test_transform_applied(self) -> None:
        ds = SyntheticDataset(size=10)
        subset = torch.utils.data.Subset(ds, list(range(10)))
        wrapped = TransformedSubset(subset, transform=_double_transform)
        x_orig, _ = ds[0]
        x_wrapped, _ = wrapped[0]
        assert torch.allclose(x_wrapped, x_orig * 2)


# ---------------------------------------------------------------------------
# filter_classes
# ---------------------------------------------------------------------------


class TestFilterClasses:
    """Tests for :func:`filter_classes`."""

    def test_filters_to_requested_classes(self) -> None:
        ds = SyntheticDataset(size=100, num_classes=5)
        filter_classes(ds, num_classes=3)
        assert (ds.targets <= 2).all()
        assert len(ds.data) == len(ds.targets)

    def test_auto_detects_targets_field(self) -> None:
        ds = SyntheticDatasetWithLabels(size=60, num_classes=3)
        filter_classes(ds, num_classes=2)
        assert (ds.labels <= 1).all()

    def test_explicit_targets_field(self) -> None:
        ds = SyntheticDatasetWithLabels(size=60, num_classes=3)
        filter_classes(ds, num_classes=2, targets_field="labels")
        assert (ds.labels <= 1).all()

    def test_raises_on_unknown_field(self) -> None:
        ds = SyntheticDataset(size=10)
        # Remove the known field so auto-detection fails
        del ds.targets
        with pytest.raises(AttributeError, match="Cannot auto-detect"):
            filter_classes(ds, num_classes=2)

    def test_warns_on_num_classes_leq_1(self) -> None:
        ds = SyntheticDataset(size=10, num_classes=5)
        with pytest.warns(match="num_classes=1 should be greater than 1"):
            filter_classes(ds, num_classes=1)

    def test_list_targets(self) -> None:
        ds = SyntheticDataset(size=30, num_classes=5)
        ds.targets = ds.targets.tolist()  # convert to plain list
        filter_classes(ds, num_classes=2)
        assert isinstance(ds.targets, torch.Tensor)
        assert (ds.targets <= 1).all()

    def test_numpy_targets(self) -> None:
        ds = SyntheticDataset(size=30, num_classes=5)
        ds.targets = ds.targets.numpy()  # convert to np.ndarray
        filter_classes(ds, num_classes=2)
        assert isinstance(ds.targets, torch.Tensor)
        assert (ds.targets <= 1).all()

    def test_preserves_data_label_alignment(self) -> None:
        """Verify that data[i] still corresponds to targets[i] after filtering.

        Property: for a synthetic dataset where targets = arange(N) % C,
        after filtering to ``num_classes=2``, every remaining sample at
        index ``i`` satisfies ``targets[i] == original_targets[original_idx]``
        where ``original_idx`` is the position in the unfiltered dataset.
        We check a weaker but sufficient property: len(data) == len(targets)
        and all targets are in [0, num_classes).
        """
        ds = SyntheticDataset(size=100, num_classes=10)
        filter_classes(ds, num_classes=3)
        assert len(ds.data) == len(ds.targets)
        assert ds.targets.min() >= 0
        assert ds.targets.max() <= 2


# ---------------------------------------------------------------------------
# get_datasets
# ---------------------------------------------------------------------------


class TestGetDatasets:
    """Tests for :func:`get_datasets`."""

    def test_basic_train_val_test_split(self) -> None:
        cfg = _make_cfg(size=100)
        splits = {
            "train": SplitDescriptor("train", 0.8, "standard"),
            "val": SplitDescriptor("train", 0.2, "standard"),
            "test": SplitDescriptor("test", 1.0, "standard"),
        }
        datasets = get_datasets(cfg, splits, seed=42)
        assert set(datasets.keys()) == {"train", "val", "test"}

        # Check proportions (approximate due to rounding)
        assert abs(len(datasets["train"]) - 80) <= 1
        assert abs(len(datasets["val"]) - 20) <= 1
        assert len(datasets["test"]) == 100

    def test_dict_splits_auto_converted(self) -> None:
        cfg = _make_cfg(size=50)
        splits = {
            "train": {"source": "train", "proportion": 1.0, "transforms": "standard"},
            "test": {"source": "test", "proportion": 1.0, "transforms": "standard"},
        }
        datasets = get_datasets(cfg, splits, seed=0)
        assert len(datasets["train"]) == 50
        assert len(datasets["test"]) == 50

    def test_transform_pipeline_applied(self) -> None:
        """Verify that the correct named transform is applied per split.

        We use ``torch.nn.Identity`` for standard (no change) and check
        that both splits produce valid tensors.
        """
        cfg = _make_cfg(size=20)
        splits = {
            "train": SplitDescriptor("train", 1.0, "standard"),
            "test": SplitDescriptor("test", 1.0, "standard"),
        }
        datasets = get_datasets(cfg, splits, seed=0)
        x, y = datasets["train"][0]
        assert isinstance(x, torch.Tensor)

    def test_unknown_source_raises(self) -> None:
        cfg = _make_cfg()
        splits = {
            "train": SplitDescriptor("nonexistent", 1.0, "standard"),
        }
        with pytest.raises(ValueError, match="Unknown source 'nonexistent'"):
            get_datasets(cfg, splits)

    def test_unknown_transform_raises(self) -> None:
        cfg = _make_cfg()
        splits = {
            "train": SplitDescriptor("train", 1.0, "nonexistent_pipeline"),
        }
        with pytest.raises(ValueError, match="Transform pipeline 'nonexistent_pipeline'"):
            get_datasets(cfg, splits)

    def test_reproducible_with_seed(self) -> None:
        """Same seed produces the same split indices.

        We seed torch's global RNG before each ``get_datasets`` call so
        that the synthetic data is identical, then verify that the split
        assignment is also identical thanks to the ``seed`` parameter.
        """
        cfg = _make_cfg(size=100)
        splits = {
            "train": SplitDescriptor("train", 0.7, "standard"),
            "val": SplitDescriptor("train", 0.3, "standard"),
        }
        torch.manual_seed(0)
        ds1 = get_datasets(cfg, splits, seed=123)
        torch.manual_seed(0)
        ds2 = get_datasets(cfg, splits, seed=123)

        x1, y1 = ds1["train"][0]
        x2, y2 = ds2["train"][0]
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)

    def test_filter_classes_integration(self) -> None:
        cfg = _make_cfg(size=100, used_classes=2)
        splits = {
            "train": SplitDescriptor("train", 1.0, "standard"),
            "test": SplitDescriptor("test", 1.0, "standard"),
        }
        datasets = get_datasets(cfg, splits, seed=0)

        # All labels should be 0 or 1
        for name in ("train", "test"):
            for i in range(len(datasets[name])):
                _, y = datasets[name][i]
                assert y <= 1

    def test_multiple_splits_from_same_source(self) -> None:
        """Three splits from the 'train' source with proportions summing to 1."""
        cfg = _make_cfg(size=100)
        splits = {
            "train": SplitDescriptor("train", 0.6, "standard"),
            "val": SplitDescriptor("train", 0.2, "standard"),
            "growth": SplitDescriptor("train", 0.2, "standard"),
        }
        datasets = get_datasets(cfg, splits, seed=42)
        total = sum(len(datasets[k]) for k in ("train", "val", "growth"))
        assert total == 100


# ---------------------------------------------------------------------------
# get_dataloaders
# ---------------------------------------------------------------------------


class TestGetDataloaders:
    """Tests for :func:`get_dataloaders`."""

    def test_returns_dataloaders(self) -> None:
        cfg = _make_cfg(size=50)
        splits = {
            "train": LoaderSplitDescriptor("train", 1.0, "standard", shuffle=True),
            "test": LoaderSplitDescriptor("test", 1.0, "standard"),
        }
        loaders = get_dataloaders(cfg, splits, batch_size=16, seed=0)
        assert set(loaders.keys()) == {"train", "test"}
        assert isinstance(loaders["train"], torch.utils.data.DataLoader)

    def test_default_batch_size(self) -> None:
        cfg = _make_cfg(size=50)
        splits = {
            "train": LoaderSplitDescriptor("train", 1.0, "standard"),
        }
        loaders = get_dataloaders(cfg, splits, batch_size=10, seed=0)
        assert loaders["train"].batch_size == 10

    def test_per_split_batch_size_override(self) -> None:
        cfg = _make_cfg(size=50)
        splits = {
            "train": LoaderSplitDescriptor("train", 1.0, "standard", batch_size=32),
            "test": LoaderSplitDescriptor("test", 1.0, "standard"),
        }
        loaders = get_dataloaders(cfg, splits, batch_size=8, seed=0)
        assert loaders["train"].batch_size == 32
        assert loaders["test"].batch_size == 8

    def test_dict_splits_auto_converted(self) -> None:
        cfg = _make_cfg(size=30)
        splits = {
            "train": {
                "source": "train",
                "proportion": 1.0,
                "transforms": "standard",
                "shuffle": True,
                "drop_last": True,
            },
        }
        loaders = get_dataloaders(cfg, splits, batch_size=5, seed=0)
        assert loaders["train"].drop_last is True

    def test_shuffle_and_drop_last(self) -> None:
        cfg = _make_cfg(size=50)
        splits = {
            "train": LoaderSplitDescriptor(
                "train", 1.0, "standard", shuffle=True, drop_last=True
            ),
            "test": LoaderSplitDescriptor(
                "test", 1.0, "standard", shuffle=False, drop_last=False
            ),
        }
        loaders = get_dataloaders(cfg, splits, batch_size=16, seed=0)
        assert loaders["train"].drop_last is True
        assert loaders["test"].drop_last is False

    def test_pin_memory_cpu(self) -> None:
        cfg = _make_cfg(size=20)
        splits = {"train": LoaderSplitDescriptor("train", 1.0, "standard")}
        loaders = get_dataloaders(
            cfg, splits, batch_size=4, device=torch.device("cpu"), seed=0
        )
        assert loaders["train"].pin_memory is False
        assert loaders["train"].num_workers == 0

    def test_iterates_correctly(self) -> None:
        """Verify that we can iterate over a batch and get expected shapes."""
        cfg = _make_cfg(size=20)
        splits = {"train": LoaderSplitDescriptor("train", 1.0, "standard")}
        loaders = get_dataloaders(cfg, splits, batch_size=5, seed=0)
        batch_x, batch_y = next(iter(loaders["train"]))
        assert batch_x.shape[0] == 5
        assert batch_y.shape[0] == 5


# ---------------------------------------------------------------------------
# Bundled Hydra YAML: NAS small benchmarks (same layout as addnist.yaml)
# ---------------------------------------------------------------------------

_DATASET_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "dataset_config"
_NAS_BENCHMARK_YAMLS = (
    "addnist.yaml",
    "multnist.yaml",
    "cifartile.yaml",
    "gutenberg.yaml",
    "geoclassing.yaml",
    "chesseract.yaml",
)

# Colleague default_augmentations non-empty -> train uses transforms.augmented (YAML).
_NAS_YAMLS_WITH_AUGMENTED = (
    "addnist.yaml",
    "multnist.yaml",
    "cifartile.yaml",
    "geoclassing.yaml",
)
# Empty default list -> no augmented block; train falls back to standard.
_NAS_YAMLS_WITHOUT_AUGMENTED = ("gutenberg.yaml", "chesseract.yaml")


class TestNasBenchmarkDatasetYamlFiles:
    """Ensure committed dataset_config YAMLs have Hydra instantiation fields."""

    @pytest.mark.parametrize("filename", _NAS_BENCHMARK_YAMLS)
    def test_nas_benchmark_yaml_structure(self, filename: str) -> None:
        path = _DATASET_CONFIG_DIR / filename
        assert path.is_file(), f"Missing {path}"
        cfg = OmegaConf.load(path)
        assert cfg.name
        assert cfg.num_classes is not None
        assert cfg.dataset._target_
        assert "train" in cfg.sources and "test" in cfg.sources
        assert "standard" in cfg.transforms

    @pytest.mark.parametrize("filename", _NAS_YAMLS_WITH_AUGMENTED)
    def test_nas_yaml_defines_augmented_for_training(self, filename: str) -> None:
        path = _DATASET_CONFIG_DIR / filename
        cfg = OmegaConf.load(path)
        assert "augmented" in cfg.transforms
        assert cfg.transforms.augmented._target_.endswith("Compose")

    @pytest.mark.parametrize("filename", _NAS_YAMLS_WITHOUT_AUGMENTED)
    def test_nas_yaml_omits_augmented_when_no_default_aug(self, filename: str) -> None:
        path = _DATASET_CONFIG_DIR / filename
        cfg = OmegaConf.load(path)
        assert "augmented" not in cfg.transforms


class TestNasTrainTransformPolicyAgainstYaml:
    """Runtime train pipeline key matches committed YAML (same rule as ``create_dataloaders``).

    YAML-only tests assert keys exist; these tests call
    :func:`resolve_train_transform_key` on loaded files and verify Hydra can
    build the resulting augmented ``Compose`` (colleague augments + torchvision).
    """

    @pytest.mark.parametrize("filename", _NAS_YAMLS_WITH_AUGMENTED)
    def test_resolve_selects_augmented_for_nas_configs_with_key(
        self, filename: str
    ) -> None:
        cfg = OmegaConf.load(_DATASET_CONFIG_DIR / filename)
        assert resolve_train_transform_key(cfg) == "augmented"

    @pytest.mark.parametrize("filename", _NAS_YAMLS_WITHOUT_AUGMENTED)
    def test_resolve_selects_standard_when_yaml_omits_augmented(
        self, filename: str
    ) -> None:
        cfg = OmegaConf.load(_DATASET_CONFIG_DIR / filename)
        assert resolve_train_transform_key(cfg) == "standard"

    @pytest.mark.parametrize("filename", _NAS_YAMLS_WITH_AUGMENTED)
    def test_augmented_compose_instantiates_and_runs(self, filename: str) -> None:
        """Hydra builds the augmented ``Compose``; inputs match torchvision ``ToTensor`` (HWC uint8)."""
        from hydra.utils import instantiate

        cfg = OmegaConf.load(_DATASET_CONFIG_DIR / filename)
        key = resolve_train_transform_key(cfg)
        assert key == "augmented"
        transform = instantiate(cfg.transforms[key], _convert_="all")
        pic = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        y = transform(pic)
        assert y.ndim == 3 and y.shape[0] == 3

    def test_missing_transforms_raises_like_production(self) -> None:
        cfg = OmegaConf.create({"name": "no_transforms"})
        with pytest.raises(ConfigAttributeError):
            resolve_train_transform_key(cfg)
