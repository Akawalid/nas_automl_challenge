"""Unit tests for ``tools.augmentations`` modules (tensor-space, post-``ToTensor``).

For **Hydra** ``dataset_config`` YAML wiring (``standard`` / ``augmented`` pipelines,
:func:`~hydra_script.data_handling.datasets.resolve_train_transform_key`), see
``test_dataset_config_transform_pipelines.py``—different layer (integration vs isolated
transform classes).
"""

import torch

from tools.augmentations import PerChannelRandomAffine, RandomRot90


def test_per_channel_random_affine_preserves_shape() -> None:
    torch.manual_seed(0)
    m = PerChannelRandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))
    x = torch.rand(3, 32, 32)
    y = m(x)
    assert y.shape == x.shape


def test_random_rot90_preserves_shape() -> None:
    torch.manual_seed(0)
    m = RandomRot90()
    x = torch.rand(3, 64, 64)
    y = m(x)
    assert y.shape == x.shape


def test_per_channel_random_affine_identity_is_noop() -> None:
    """Degenerate params (no rotation/translation/unit scale) must reproduce x.

    Guards against regressions in affine matrix construction: any off-by-one in
    the 2x3 theta layout would break this on at least one channel.
    """
    torch.manual_seed(0)
    m = PerChannelRandomAffine(degrees=0, translate=(0.0, 0.0), scale=(1.0, 1.0))
    x = torch.rand(3, 32, 32)
    y = m(x)
    assert torch.allclose(y, x, atol=1e-5)
