"""Dataset creation and splitting with Hydra-configurable transforms.

This module provides a flexible API for loading datasets, splitting them
into named subsets, and applying per-split transform pipelines defined
in Hydra YAML configs.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal
from warnings import warn

import numpy as np
import torch
import torch.utils.data
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

_COMMON_TARGET_FIELDS = ("targets", "labels", "y", "_labels")


@dataclass
class SplitDescriptor:
    """Description of a single dataset split.

    Attributes
    ----------
    source : str
        Name of the source to draw from (must match a key in ``cfg.sources``).
    proportion : float
        Fraction of the source dataset for this split. Proportions for splits
        sharing the same source must sum to 1.0.
    transforms : str
        Name of the transform pipeline to apply (must match a key in
        ``cfg.transforms``).
    """

    source: str
    proportion: float
    transforms: str = "standard"


TrainTransformKey = Literal["standard", "augmented"]


def resolve_train_transform_key(dataset_config: Any) -> TrainTransformKey:
    """Return the transform pipeline key used for training splits in the Hydra API.

    **Rule:** Use ``"augmented"`` if the key ``"augmented"`` exists under
    ``dataset_config.transforms``, else ``"standard"``. This uses **key
    membership** (``in``), not truthiness: ``augmented: null`` still selects
    ``"augmented"`` when the key is present. To use ``standard`` for training,
    **omit** the ``augmented`` key (do not rely on null to disable).

    **Precondition:** ``dataset_config.transforms`` exists and is mapping-like.
    List-shaped ``transforms`` would change ``in`` semantics.

    Missing ``transforms`` is unsupported and typically raises when accessed,
    matching production (e.g. :class:`omegaconf.errors.ConfigAttributeError`).
    """
    return "augmented" if "augmented" in dataset_config.transforms else "standard"


class TransformedSubset(torch.utils.data.Dataset):
    """A dataset subset with its own transform pipeline.

    Wraps a :class:`torch.utils.data.Subset` and applies ``transform`` to the
    first element (input) of each sample, independently of any transform set
    on the underlying source dataset.

    Parameters
    ----------
    subset : torch.utils.data.Subset
        The subset to wrap.
    transform : torch.nn.Module | None
        Transform to apply to the input element of each ``(x, y)`` sample.
    """

    def __init__(
        self,
        subset: torch.utils.data.Subset,
        transform: torch.nn.Module | None = None,
    ) -> None:
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> tuple:
        """Return the transformed sample at ``index``.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        tuple
            ``(transformed_x, y)`` pair.
        """
        x, y = self.subset[index]  # type: ignore[misc]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        """Return the number of samples.

        Returns
        -------
        int
            Length of the underlying subset.
        """
        return len(self.subset)


def filter_classes(
    dataset: torch.utils.data.Dataset,
    num_classes: int,
    targets_field: str | None = None,
) -> torch.utils.data.Dataset:
    """Filter a dataset in-place to keep only the first ``num_classes`` classes.

    Classes are assumed to be labelled ``0, 1, ..., num_classes - 1``.
    The dataset is mutated: its ``data`` attribute and the attribute named by
    ``targets_field`` are both replaced with filtered versions.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to filter. Must have a ``data`` attribute and a targets
        attribute whose name is either provided or auto-detected.
    num_classes : int
        Number of classes to retain (classes ``0`` through ``num_classes - 1``).
    targets_field : str | None
        Name of the attribute holding class labels. If ``None``, the field is
        auto-detected by checking common names (``targets``, ``labels``,
        ``y``, ``_labels``).

    Returns
    -------
    torch.utils.data.Dataset
        The same dataset object, modified in-place.

    Raises
    ------
    AttributeError
        If ``targets_field`` is ``None`` and no common targets attribute is
        found on the dataset.
    """
    if num_classes <= 1:
        warn(f"{num_classes=} should be greater than 1", stacklevel=2)

    if targets_field is None:
        for field in _COMMON_TARGET_FIELDS:
            if hasattr(dataset, field):
                targets_field = field
                break
        else:
            raise AttributeError(
                f"Cannot auto-detect targets field on {type(dataset).__name__}. "
                "Set 'targets_field' in the dataset YAML config."
            )

    targets = getattr(dataset, targets_field)
    if isinstance(targets, list):
        targets = torch.tensor(targets)
    elif isinstance(targets, np.ndarray):
        targets = torch.tensor(targets)

    mask = targets <= (num_classes - 1)
    dataset.data = dataset.data[mask]  # type: ignore[attr-defined]
    setattr(dataset, targets_field, targets[mask])
    return dataset


def get_datasets(
    cfg: DictConfig,
    splits: Mapping[str, SplitDescriptor | dict],
    seed: int | None = None,
) -> dict[str, TransformedSubset]:
    """Create datasets for the requested splits.

    Loads source datasets via ``hydra.utils.instantiate`` using the
    ``dataset`` and ``sources`` entries from the Hydra config, optionally
    filters classes, splits each source according to the provided
    proportions, and wraps each split in a :class:`TransformedSubset` with
    the corresponding named transform pipeline.

    Parameters
    ----------
    cfg : DictConfig
        Dataset configuration loaded from YAML. Expected keys:

        - ``dataset``: Hydra ``_target_`` config for the dataset class.
        - ``sources``: mapping of source names to ``extra_args`` dicts.
        - ``transforms``: mapping of pipeline names to Hydra ``_target_``
          configs (e.g. ``torchvision.transforms.Compose``).
        - ``num_classes``: true number of classes in the dataset
          (informational, not used by this function).
        - ``used_classes`` (optional): if present, filter to this many
          classes (keep classes ``0`` through ``used_classes - 1``).
        - ``targets_field`` (optional): attribute name for class labels
          (used when ``used_classes`` triggers filtering).
    splits : Mapping[str, SplitDescriptor | dict]
        Mapping from split name to its descriptor. Dict values are
        auto-converted to :class:`SplitDescriptor`. Proportions for splits
        sharing the same source must sum to 1.0.
    seed : int | None
        Random seed for reproducible splitting.

    Returns
    -------
    dict[str, TransformedSubset]
        Mapping from split name to dataset.

    Raises
    ------
    ValueError
        If a split references an unknown source or an unknown transform
        pipeline.

    Examples
    --------
    >>> datasets = get_datasets(
    ...     cfg,
    ...     splits={
    ...         "train": {"source": "train", "proportion": 0.95,
    ...                   "transforms": "augmented"},
    ...         "val":   {"source": "train", "proportion": 0.05,
    ...                   "transforms": "standard"},
    ...         "test":  {"source": "test",  "proportion": 1.0,
    ...                   "transforms": "standard"},
    ...     },
    ...     seed=42,
    ... )
    """
    normalized_splits: dict[str, SplitDescriptor] = {
        name: SplitDescriptor(**desc) if isinstance(desc, dict) else desc
        for name, desc in splits.items()
    }

    # 1. Load each source dataset (once per source, no transforms)
    source_datasets: dict[str, torch.utils.data.Dataset] = {}
    for source_name, source_cfg in cfg.sources.items():
        extra_args: dict = OmegaConf.to_container(  # type: ignore[assignment]
            source_cfg.get("extra_args", {}), resolve=True
        )
        source_datasets[source_name] = instantiate(
            cfg.dataset, **extra_args, transform=None, _convert_="all"
        )

    # 2. Optionally filter classes
    used_classes = cfg.get("used_classes", None)
    if used_classes is not None:
        targets_field = cfg.get("targets_field", None)
        for name in source_datasets:
            filter_classes(
                source_datasets[name],
                used_classes,
                targets_field=targets_field,
            )

    # 3. Group requested splits by source
    splits_by_source: dict[str, list[tuple[str, SplitDescriptor]]] = {}
    for split_name, desc in normalized_splits.items():
        if desc.source not in source_datasets:
            raise ValueError(
                f"Unknown source '{desc.source}' for split '{split_name}'. "
                f"Available sources: {list(source_datasets.keys())}"
            )
        splits_by_source.setdefault(desc.source, []).append((split_name, desc))

    # 4. Split each source and wrap with per-split transforms
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    result: dict[str, TransformedSubset] = {}

    for source_name, split_list in splits_by_source.items():
        ratios = [s[1].proportion for s in split_list]
        subsets = torch.utils.data.random_split(
            source_datasets[source_name], ratios, generator=generator
        )

        for (split_name, desc), subset in zip(split_list, subsets):
            if desc.transforms not in cfg.transforms:
                raise ValueError(
                    f"Transform pipeline '{desc.transforms}' not found for "
                    f"split '{split_name}'. "
                    f"Available: {list(cfg.transforms.keys())}"
                )
            transform = instantiate(cfg.transforms[desc.transforms], _convert_="all")
            result[split_name] = TransformedSubset(subset, transform=transform)

    return result


@dataclass
class LoaderSplitDescriptor(SplitDescriptor):
    """Description of a dataset split with dataloader parameters.

    Extends :class:`SplitDescriptor` with settings that control how the
    split is wrapped into a :class:`torch.utils.data.DataLoader`.
    See :class:`SplitDescriptor` for inherited attributes (``source``,
    ``proportion``, ``transforms``).

    Attributes
    ----------
    batch_size : int | None
        Batch size for this split. If ``None``, the shared ``batch_size``
        parameter of :func:`get_dataloaders` is used.
    shuffle : bool
        Whether to shuffle the dataloader.
    drop_last : bool
        Whether to drop the last incomplete batch.
    seed : int | None
        Per-split seed for the dataloader's random number generator (used
        when ``shuffle=True``). If ``None``, no generator is set.
    """

    batch_size: int | None = None
    shuffle: bool = False
    drop_last: bool = False
    seed: int | None = None


def get_dataloaders(
    cfg: DictConfig,
    splits: Mapping[str, LoaderSplitDescriptor | dict[str, Any]],
    batch_size: int,
    num_workers: int = 0,
    device: torch.device = torch.device("cpu"),
    seed: int | None = None,
) -> dict[str, torch.utils.data.DataLoader]:
    """Create dataloaders for the requested splits.

    This is a convenience wrapper around :func:`get_datasets`. It first
    creates all datasets using the ``source``, ``proportion``, and
    ``transforms`` fields of each split descriptor, then wraps each one
    in a :class:`torch.utils.data.DataLoader` using the per-split
    ``shuffle``, ``drop_last``, ``seed``, and optional per-split
    ``batch_size`` fields together with the shared ``batch_size``,
    ``num_workers``, and ``device`` parameters.

    When the ``device`` is not CPU, ``pin_memory`` is enabled and
    ``num_workers`` is used as provided. On CPU, ``num_workers`` is forced
    to ``0`` and ``pin_memory`` is disabled.

    Parameters
    ----------
    cfg : DictConfig
        Dataset configuration loaded from YAML (same format as
        :func:`get_datasets`).
    splits : dict[str, LoaderSplitDescriptor | dict[str, Any]]
        Mapping from split name to its descriptor. Dict values are
        auto-converted to :class:`LoaderSplitDescriptor`. Proportions for
        splits sharing the same source must sum to 1.0.
    batch_size : int
        Default batch size, used for splits that do not specify their own
        ``batch_size`` in the descriptor.
    num_workers : int
        Number of worker processes for data loading. Forced to ``0`` when
        ``device`` is CPU.
    device : torch.device
        Target device. Determines ``pin_memory`` and effective
        ``num_workers``.
    seed : int | None
        Random seed for reproducible dataset splitting (passed to
        :func:`get_datasets`). This is independent of the per-split
        ``seed`` used for dataloader shuffling.

    Returns
    -------
    dict[str, torch.utils.data.DataLoader]
        Mapping from split name to dataloader.

    Examples
    --------
    >>> loaders = get_dataloaders(
    ...     cfg,
    ...     splits={
    ...         "train": {"source": "train", "proportion": 0.95,
    ...                   "transforms": "augmented",
    ...                   "shuffle": True, "drop_last": True, "seed": 42},
    ...         "val":   {"source": "train", "proportion": 0.05,
    ...                   "transforms": "standard"},
    ...         "test":  {"source": "test", "proportion": 1.0,
    ...                   "transforms": "standard"},
    ...     },
    ...     batch_size=64,
    ...     num_workers=4,
    ...     device=torch.device("cuda"),
    ...     seed=0,
    ... )
    """
    # Normalize dicts to LoaderSplitDescriptor
    normalized: dict[str, LoaderSplitDescriptor] = {
        name: LoaderSplitDescriptor(**desc) if isinstance(desc, dict) else desc
        for name, desc in splits.items()
    }

    # Create datasets (delegates source loading, filtering, splitting, transforms)
    datasets = get_datasets(cfg, normalized, seed=seed)

    # Device-dependent settings
    pin_memory = device != torch.device("cpu")
    effective_num_workers = num_workers if pin_memory else 0

    # Wrap each dataset in a DataLoader
    result: dict[str, torch.utils.data.DataLoader] = {}
    for split_name, dataset in datasets.items():
        desc = normalized[split_name]
        split_batch_size = desc.batch_size if desc.batch_size is not None else batch_size
        generator = (
            torch.Generator().manual_seed(desc.seed) if desc.seed is not None else None
        )
        result[split_name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=split_batch_size,
            shuffle=desc.shuffle,
            pin_memory=pin_memory,
            num_workers=effective_num_workers,
            persistent_workers=effective_num_workers > 0,
            generator=generator,
            drop_last=desc.drop_last,
        )

    return result
