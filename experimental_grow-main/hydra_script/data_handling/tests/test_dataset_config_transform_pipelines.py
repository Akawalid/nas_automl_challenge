"""Hydra ``dataset_config`` **transform pipeline** contracts (``Compose`` graphs).

The training stack resolves which named pipeline trains on via **key membership** in
YAML (``"augmented" in dataset_config.transforms``), not a loose string label. That
rule applies to **any** dataset config under ``hydra_script/configs/dataset_config/``.

This module instantiates ``transforms.standard`` and optional ``transforms.augmented``
from committed YAML and checks resolution, structure, and tensor behavior. **Fixtures**
are the NAS small-benchmark YAMLs (``multnist.yaml``, …) reused from
``test_datasets.py``—they are representative committed configs; add rows to the fixture
table when new benchmarks follow the same schema.

**What is exercised**

- **Key resolution** vs :data:`_TRANSFORM_CONTRACT_FIXTURE_TABLE` (kept in sync with
  the NAS filename groupings in ``test_datasets.py``).
- **Structural difference** between eval-style ``standard`` and the train pipeline.
- **Tensor behavior** on synthetic HWC uint8: seeded reproducibility, then inequality
  between policies when ``augmented`` is defined.

**Relationship to** ``test_augmentations.py``

- ``test_augmentations.py`` unit-tests **implementations** in ``tools.augmentations``
  (e.g. shape invariance of ``PerChannelRandomAffine``) in isolation.
- This file tests **YAML + Hydra +** :func:`~hydra_script.data_handling.datasets.resolve_train_transform_key`
  **integration** on full dataset configs. Keep both files: different layers.

**Out of scope**

- No real dataset files, no ``DataLoader``, no end-to-end ``train_and_grow`` run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from hydra_script.data_handling.datasets import resolve_train_transform_key

from .test_datasets import (
    _DATASET_CONFIG_DIR,
    _NAS_YAMLS_WITH_AUGMENTED,
    _NAS_YAMLS_WITHOUT_AUGMENTED,
)

# Shared torch seed for any pipeline that may use RNG (standard or augmented).
_TENSOR_PIPELINE_SEED = 0


@dataclass(frozen=True, slots=True)
class _TransformContractRow:
    """Expected train-transform policy for one committed YAML fixture."""

    filename: str
    """Basename under ``hydra_script/configs/dataset_config/``."""
    train_uses_augmented_named_pipeline: bool
    """If True, YAML defines ``transforms.augmented`` and train resolves to it; else train uses ``standard`` (see :func:`~hydra_script.data_handling.datasets.resolve_train_transform_key`)."""


_TRANSFORM_CONTRACT_FIXTURE_TABLE: tuple[_TransformContractRow, ...] = (
    _TransformContractRow("addnist.yaml", True),
    _TransformContractRow("multnist.yaml", True),
    _TransformContractRow("cifartile.yaml", True),
    _TransformContractRow("geoclassing.yaml", True),
    _TransformContractRow("gutenberg.yaml", False),
    _TransformContractRow("chesseract.yaml", False),
)


def _assert_fixture_table_matches_test_datasets_lists() -> None:
    """Fail at import time if the fixture table drifts from ``test_datasets`` NAS groupings."""
    with_aug = {
        r.filename
        for r in _TRANSFORM_CONTRACT_FIXTURE_TABLE
        if r.train_uses_augmented_named_pipeline
    }
    without = {
        r.filename
        for r in _TRANSFORM_CONTRACT_FIXTURE_TABLE
        if not r.train_uses_augmented_named_pipeline
    }
    assert with_aug == set(_NAS_YAMLS_WITH_AUGMENTED)
    assert without == set(_NAS_YAMLS_WITHOUT_AUGMENTED)


_assert_fixture_table_matches_test_datasets_lists()


def _load(name: str) -> object:
    """Load a dataset config from disk exactly as Hydra would merge it for tests."""
    path = _DATASET_CONFIG_DIR / name
    assert path.is_file(), path
    return OmegaConf.load(path)


def _compose_child_type_names(module: object) -> tuple[str, ...]:
    """Cheap structural fingerprint: ordered ``type().__name__`` of each child of ``Compose``.

    Not a substitute for full config equality, but catches swapped or duplicated stages
    in YAML-driven pipelines without comparing every hyperparameter.
    """
    if hasattr(module, "transforms"):
        return tuple(type(t).__name__ for t in module.transforms)
    return (type(module).__name__,)


class TestDatasetConfigTransformPipelines:
    """Contracts for :data:`_TRANSFORM_CONTRACT_FIXTURE_TABLE` YAMLs."""

    @pytest.mark.parametrize(
        "row", _TRANSFORM_CONTRACT_FIXTURE_TABLE, ids=lambda r: r.filename
    )
    def test_resolve_train_key_matches_table(self, row: _TransformContractRow) -> None:
        """``resolve_train_transform_key`` matches the table for this config.

        **Intent:** Guard the membership rule (presence of ``transforms.augmented`` key)
        against accidental edits to YAML or to :func:`~hydra_script.data_handling.datasets.resolve_train_transform_key`.

        **Mechanics:** Load ``row.filename``, compare the resolver output to
        ``\"augmented\"`` if :attr:`_TransformContractRow.train_uses_augmented_named_pipeline`
        else ``\"standard\"``.

        **If this fails:** A YAML gained or lost an ``augmented`` block without updating
        the table, or resolver semantics changed.
        """
        cfg = _load(row.filename)
        got = resolve_train_transform_key(cfg)
        want = "augmented" if row.train_uses_augmented_named_pipeline else "standard"
        assert got == want

    @pytest.mark.parametrize(
        "row", _TRANSFORM_CONTRACT_FIXTURE_TABLE, ids=lambda r: r.filename
    )
    def test_train_vs_standard_compose_graph(self, row: _TransformContractRow) -> None:
        """Instantiated train pipeline matches ``standard`` Compose graph iff YAML has no extra aug block.

        **Intent:** Training should not silently use the same transforms as validation when
        the config defines a separate ``augmented`` pipeline, and should use the exact
        same graph as ``standard`` when the ``augmented`` key is omitted.

        **Mechanics:** Instantiate ``cfg.transforms.standard`` and the train pipeline
        ``cfg.transforms[resolve_train_transform_key(cfg)]``, then compare
        :func:`_compose_child_type_names` tuples for equality or inequality according to
        ``row.train_uses_augmented_named_pipeline``.

        **If this fails:** YAML lists the wrong stages under ``augmented`` / ``standard``,
        or train resolution no longer points at the intended named pipeline.
        """
        cfg = _load(row.filename)
        std = instantiate(cfg.transforms.standard, _convert_="all")
        key = resolve_train_transform_key(cfg)
        train = instantiate(cfg.transforms[key], _convert_="all")
        fp_std = _compose_child_type_names(std)
        fp_train = _compose_child_type_names(train)
        if row.train_uses_augmented_named_pipeline:
            assert fp_std != fp_train
        else:
            assert fp_std == fp_train

    @pytest.mark.parametrize(
        "filename",
        _NAS_YAMLS_WITH_AUGMENTED,
    )
    def test_augmented_pipeline_changes_tensor_vs_standard(self, filename: str) -> None:
        """For configs that define ``augmented``, prove different graphs *and* different outputs under control.

        **Intent:** "Augmented" must be **effective** (not only a second name in YAML):
        outputs on a fixed synthetic image must differ from ``standard`` once RNG is
        controlled. Also prove that neither pipeline is flaky: two forwards with the same
        :data:`_TENSOR_PIPELINE_SEED` before each call must agree for that pipeline alone.

        **Why the steps:** Comparing ``std(pic)`` and ``aug(pic)`` without (1) structural
        inequality or (2) per-pipeline reproducibility can pass for bad reasons—e.g. two
        draws from identical random ops, or an accidental copy-paste making both pipelines
        identical while tensors still differ by chance.

        **Mechanics (order matters):**

        0. **Structure:** ``_compose_child_type_names(standard) !=
           _compose_child_type_names(augmented)`` so we never assert tensor inequality when
           the instantiated graphs could be the same.
        1. **Standard repeatability:** ``torch.manual_seed(_TENSOR_PIPELINE_SEED)`` then
           ``std(pic)``, twice independently—outputs must ``allclose``. Covers future
           stochastic ``standard`` policies.
        2. **Augmented repeatability:** same pattern for ``aug(pic)``.
        3. **Cross-policy:** reset seed, ``out_std = std(pic)``; reset seed again,
           ``out_aug = aug(pic)``; assert **not** ``allclose`` (same shapes).

        **If this fails:** Miswired YAML, dead augmentation, or (step 1/2) non-reproducible
        transforms that ignore ``torch.manual_seed``.
        """
        cfg = _load(filename)
        std = instantiate(cfg.transforms.standard, _convert_="all")
        aug = instantiate(cfg.transforms.augmented, _convert_="all")
        fp_std = _compose_child_type_names(std)
        fp_aug = _compose_child_type_names(aug)
        assert fp_std != fp_aug, (
            f"{filename}: standard and augmented pipelines must differ "
            f"(fingerprints {fp_std!r} vs {fp_aug!r}) before tensor inequality is asserted."
        )

        pic = np.random.default_rng(42).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        seed = _TENSOR_PIPELINE_SEED

        torch.manual_seed(seed)
        out_std_a = std(pic)
        torch.manual_seed(seed)
        out_std_b = std(pic)
        assert torch.allclose(out_std_a, out_std_b, rtol=0.0, atol=1e-6)

        torch.manual_seed(seed)
        out_aug_a = aug(pic)
        torch.manual_seed(seed)
        out_aug_b = aug(pic)
        assert torch.allclose(out_aug_a, out_aug_b, rtol=0.0, atol=1e-6)

        torch.manual_seed(seed)
        out_std = std(pic)
        torch.manual_seed(seed)
        out_aug = aug(pic)

        assert isinstance(out_std, torch.Tensor) and isinstance(out_aug, torch.Tensor)
        assert out_std.shape == out_aug.shape
        assert not torch.allclose(out_std, out_aug, rtol=0.0, atol=1e-6)

    @pytest.mark.parametrize(
        "filename",
        _NAS_YAMLS_WITHOUT_AUGMENTED,
    )
    def test_standard_only_train_matches_standard_on_fixed_sample(
        self, filename: str
    ) -> None:
        """When YAML omits ``transforms.augmented``, train and ``standard`` must be identical in effect.

        **Intent:** Training must use the eval-style ``standard`` pipeline only when the
        ``augmented`` key is omitted. On the same input path,
        ``instantiate(cfg.transforms[resolve_train_transform_key(cfg)])`` must match
        ``instantiate(cfg.transforms.standard)``.

        **Mechanics:** Fixed ``pic`` (uint8 HWC); compare ``train(pic)`` and ``std(pic)``
        with ``torch.allclose``. No seeding required today because both pipelines are
        deterministic ``ToTensor``-only; if ``standard`` gains RNG later, mirror the
        seeded pattern from :meth:`test_augmented_pipeline_changes_tensor_vs_standard`.

        **If this fails:** An ``augmented`` key was added without updating the fixture lists
        in ``test_datasets.py``, or train resolution no longer aliases ``standard`` for these files.
        """
        cfg = _load(filename)
        std = instantiate(cfg.transforms.standard, _convert_="all")
        train = instantiate(
            cfg.transforms[resolve_train_transform_key(cfg)], _convert_="all"
        )
        pic = np.random.default_rng(7).integers(0, 255, (32, 32, 3), dtype=np.uint8)
        assert torch.allclose(train(pic), std(pic))
