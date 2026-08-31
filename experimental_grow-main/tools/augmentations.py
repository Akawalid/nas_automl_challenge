"""
tools/augmentations.py — shared augmentation utilities
=======================================================
Single source of truth for dataset augmentations across projects.

Install experimental_grow once and import from anywhere:
    pip install -e /path/to/experimental_grow
    from tools.augmentations import default_augmentations, get_transforms

Exports
-------
npy_datasets            : set[str]  — datasets backed by raw numpy arrays
default_augmentations   : dict[str, list[str]]  — recommended augmentations per dataset
get_transforms(dataset_name, data_augmentation) -> (base_transforms, aug_transforms)
"""

import math
from warnings import warn

import torch
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2

# ──────────────────────────────────────────────── dataset sets ──── #

# Datasets whose samples are raw numpy arrays (not PIL images).
# In get_transforms these datasets use transforms.ToTensor() as base transform.
# In the training pipeline, augmentation is applied after ToTensor (on tensors).
npy_datasets: set = {
    "addnist",
    "multnist",
    "cifartile",
    "language",
    "gutenberg",
    "geoclassing",
    "chesseract",
    "gameoflife",
}

# ──────────────────────────────────────── recommended augmentations ──── #

# Recommended augmentations per dataset, based on data modality and
# symmetry constraints.  Pass any subset of these (or a custom list) as
# the `data_augmentation` argument to get_transforms.
#
# Per-dataset notes:
#   cifar10/svhn     : AutoAugment policy
#   cifar100         : horizontal flip + random crop (32px, padding=4)
#   addnist/multnist : tiny affine applied independently per channel
#   geoclassing      : spatial symmetries only, no photometric ops
#   gameoflife       : flip + 90° rotation (binary grid)
#   cifartile        : horizontal flip only (no cross-tile operations)
#   language/gutenberg/chesseract: no augmentation
default_augmentations: dict = {
    "mnist": ["rotation", "crop"],
    "fashion-mnist": ["horizontal_flip", "rotation", "crop"],
    "cifar10": ["autoaugment"],
    "cifar100": ["horizontal_flip", "crop"],
    "svhn": ["autoaugment"],
    "addnist": ["per_channel_affine"],
    "multnist": ["per_channel_affine"],
    "cifartile": ["horizontal_flip"],
    "language": [],
    "gutenberg": [],
    "geoclassing": [
        "horizontal_flip",
        "vertical_flip",
        "random_rot90",
        "cutout",
    ],
    "chesseract": [],
    "gameoflife": ["horizontal_flip", "vertical_flip", "random_rot90"],
}

# ─────────────────────────────────────────────── get_transforms ──── #


def get_transforms(
    dataset_name: str, data_augmentation: list | None = None
) -> tuple[list, list]:
    """
    Return (base_transforms, augmentation_transforms) for a dataset.

    Parameters
    ----------
    dataset_name      : str
        Name of the dataset (must be a key in the base-transforms table below).
    data_augmentation : list[str] | None
        Augmentation names to build.  If None, no augmentation transforms are
        returned (pass default_augmentations[dataset_name] to use defaults).

    Returns
    -------
    base_transforms : list
        ToTensor + Normalize (and nothing else — no spatial augmentation).
        For PIL datasets these operate on PIL images → tensors.
        For npy datasets only ToTensor is included (data is already float).
    aug_transforms : list
        Augmentation transforms built from data_augmentation.

    Ordering in the final pipeline
    -------------------------------
    PIL datasets (augment before ToTensor):
        transforms.Compose(aug_transforms + base_transforms)
    npy datasets (augment after ToTensor, on tensors):
        transforms.Compose(base_transforms + aug_transforms)
    """
    _base: dict = {
        "mnist": [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ],
        "fashion-mnist": [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,)),
        ],
        "cifar10": [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.49139968, 0.48215827, 0.44653124),
                std=(0.24703233, 0.24348505, 0.26158768),
            ),
        ],
        "cifar100": [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)
            ),
        ],
        "svhn": [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970)
            ),
        ],
        "food101": [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ],
        "addnist": [transforms.ToTensor()],
        "multnist": [transforms.ToTensor()],
        "cifartile": [transforms.ToTensor()],
        "language": [transforms.ToTensor()],
        "gutenberg": [transforms.ToTensor()],
        "geoclassing": [transforms.ToTensor()],
        "chesseract": [transforms.ToTensor()],
        "gameoflife": [transforms.ToTensor()],
    }

    if dataset_name not in _base:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. " f"Supported: {sorted(_base)}"
        )

    aug_transforms = []
    if data_augmentation:
        for aug in data_augmentation:
            if aug == "horizontal_flip":
                aug_transforms.append(transforms.RandomHorizontalFlip())
            elif aug == "vertical_flip":
                aug_transforms.append(transforms.RandomVerticalFlip())
            elif aug == "rotation":
                aug_transforms.append(transforms.RandomRotation(degrees=10))
            elif aug == "rotation_180":
                aug_transforms.append(transforms.RandomRotation(degrees=180))
            elif aug == "rotation_90":
                aug_transforms.append(RandomApplyRot90(p=0.5))
            elif aug == "random_rot90":
                aug_transforms.append(RandomRot90())
            elif aug == "crop":
                aug_transforms.append(
                    transforms.RandomCrop(size=32, padding=4, padding_mode="reflect")
                )
            elif aug == "crop_112":
                aug_transforms.append(transforms.RandomCrop(size=112, padding=8))
            elif aug == "random_resized_crop":
                aug_transforms.append(
                    transforms.RandomResizedCrop(size=128, scale=(0.7, 1.0))
                )
            elif aug == "cutout":
                aug_transforms.append(
                    transforms.RandomErasing(
                        p=0.5,
                        scale=(0.02, 0.2),
                        ratio=(1.0, 1.0),
                        value=0,
                    )
                )
            elif aug == "gaussian_blur":
                aug_transforms.append(
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3
                    )
                )
            elif aug == "color_jitter":
                aug_transforms.append(
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0
                    )
                )
            elif aug == "per_channel_affine":
                aug_transforms.append(
                    PerChannelRandomAffine(
                        degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)
                    )
                )
            elif aug == "gaussian_noise":
                aug_transforms.append(
                    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01)
                )
            elif aug == "autoaugment":
                if dataset_name in ("cifar10", "cifar100"):
                    policy = transforms.AutoAugmentPolicy.CIFAR10
                elif dataset_name == "svhn":
                    policy = transforms.AutoAugmentPolicy.SVHN
                else:
                    raise ValueError(f"AutoAugment not available for {dataset_name}")
                aug_transforms.append(transforms.AutoAugment(policy=policy))
            elif aug == "randaugment":
                if dataset_name in npy_datasets:
                    aug_transforms.append(transforms_v2.RandAugment())
                else:
                    aug_transforms.append(transforms.RandAugment())
            else:
                warn(f"Unknown augmentation '{aug}' — skipped")

    return _base[dataset_name], aug_transforms


def cct_rand_augment_transform(
    config_str: str,
    img_mean,
    translate_const: int | None = None,
):
    """Build timm's RandAugment for the CCT / Compact-Transformers recipe.

    Thin wrapper around :func:`timm.data.auto_augment.rand_augment_transform`
    that exists for one reason: ``img_mean`` (the fill colour used by geometric
    ops such as shear/translate) must be a **tuple**, but values coming from a
    Hydra/OmegaConf YAML arrive as a **list**. PIL's ``Image.new`` rejects a list
    (``TypeError: color must be int or tuple``), so this coerces ``img_mean`` to a
    tuple of ints before handing it to timm. ``timm`` is imported lazily so this
    module stays importable in environments without it.

    Parameters
    ----------
    config_str : str
        timm RandAugment config string, e.g. ``"rand-m9-mstd0.5-inc1"``.
    img_mean : sequence of int
        Per-channel fill colour, typically ``round(255 * dataset_mean)``.
    translate_const : int | None
        Absolute translation magnitude in pixels (timm uses
        ``int(min(img_size) * 0.45)``). If ``None``, timm's default is used.

    Returns
    -------
    Callable
        A timm RandAugment transform operating on PIL images.
    """
    from timm.data.auto_augment import rand_augment_transform

    hparams: dict = {"img_mean": tuple(int(c) for c in img_mean)}
    if translate_const is not None:
        hparams["translate_const"] = int(translate_const)
    return rand_augment_transform(config_str, hparams)


class RandomApplyRot90(torch.nn.Module):
    """Randomly applies a 90-degree counter-clockwise rotation with probability p.

    Operates on tensors of shape (C, H, W). Suitable for datasets with 4-fold
    rotational symmetry (e.g. top-down satellite imagery, game-of-life grids).
    Unlike RandomRotation, this only ever rotates by exactly 90° so there is
    no interpolation artefact.

    Parameters
    ----------
    p : float
        Probability of applying the rotation. Default: 0.5.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() < self.p:
            return torch.rot90(x, k=1, dims=[-2, -1])
        return x


class RandomRot90(torch.nn.Module):
    """Uniformly samples a rotation from {0°, 90°, 180°, 270°} and applies it.

    Operates on tensors of shape (C, H, W). Provides stronger rotational
    augmentation than RandomApplyRot90 by covering all four cardinal orientations
    with equal probability. No interpolation is performed.
    """

    def forward(self, x):
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(x, k=k, dims=[-2, -1]) if k > 0 else x


class PerChannelRandomAffine(torch.nn.Module):
    """Applies an independent random affine transform to each channel of a tensor.

    Standard torchvision affine augmentation applies one transform to all
    channels simultaneously. This class draws separate rotation, scale, and
    translation parameters per channel, which is appropriate for multi-digit
    composite datasets (e.g. AddNIST / MultNIST) where each channel represents
    a distinct digit that may have shifted independently.

    Parameters
    ----------
    degrees : float
        Maximum rotation magnitude in degrees. Each channel is rotated by a
        value sampled uniformly from [-degrees, +degrees].
    translate : tuple[float, float]
        Maximum translation as a fraction of image size (tx_max, ty_max).
        Each channel is translated independently within this range.
    scale : tuple[float, float]
        Scale range (min, max). Each channel is scaled by a value sampled
        uniformly from [scale[0], scale[1]].
    """

    def __init__(self, degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)):
        super().__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scale

    def forward(self, x):
        # x: (C, H, W)
        C, H, W = x.shape

        # Generate random params for each channel
        angles = (torch.rand(C) * 2 - 1) * self.degrees  # [-deg, +deg]
        scales = torch.rand(C) * (self.scale[1] - self.scale[0]) + self.scale[0]
        tx = (torch.rand(C) * 2 - 1) * self.translate[0]
        ty = (torch.rand(C) * 2 - 1) * self.translate[1]

        # Build affine matrices (C, 2, 3)
        theta = torch.zeros(C, 2, 3)
        cos_a = torch.cos(angles * math.pi / 180)
        sin_a = torch.sin(angles * math.pi / 180)

        theta[:, 0, 0] = scales * cos_a
        theta[:, 0, 1] = -scales * sin_a
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = scales * sin_a
        theta[:, 1, 1] = scales * cos_a
        theta[:, 1, 2] = ty

        # Reshape: (C, H, W) -> (C, 1, H, W) for grid_sample
        x_batch = x.unsqueeze(1)

        # Generate grid and apply
        grid = torch.nn.functional.affine_grid(theta, x_batch.shape, align_corners=False)
        out = torch.nn.functional.grid_sample(
            x_batch, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        return out.squeeze(1)  # (C, H, W)
