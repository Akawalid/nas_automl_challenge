"""Regression test: legacy ``dataset_config`` path emits ``FutureWarning``.

The PR introducing NAS-small Hydra dataset YAMLs deprecated the old
``dataset_config`` layout (one without ``dataset`` / ``sources`` / ``transforms``
blocks). Removing the warning silently would regress user migration guidance,
so lock in the deprecation contract with a focused test.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from hydra_script import aux_train_and_grow


def test_legacy_dataset_config_emits_future_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``dataset_config`` without a ``dataset`` block must warn before loading."""

    def _stub_get_dataloaders(**_kwargs):
        return None, None, None, None

    monkeypatch.setattr(
        aux_train_and_grow, "get_dataloaders", _stub_get_dataloaders, raising=True
    )
    monkeypatch.setattr(
        aux_train_and_grow, "dataloader_description", lambda _loader: "", raising=True
    )

    cfg = OmegaConf.create(
        {
            "dataset_config": {
                "name": "legacy-fake",
                "path": "/tmp",
                "num_classes": 2,
                "split_train_val": 0.1,
            },
            "general": {"batch_size": 4, "num_workers": 0, "seed": 0},
            "growing": {},
        }
    )

    with pytest.warns(FutureWarning, match="dataset_config.*deprecated"):
        aux_train_and_grow.create_dataloaders(cfg, torch.device("cpu"))
