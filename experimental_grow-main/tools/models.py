from typing import Callable

import torch
from gromo.containers.growing_container import GrowingContainer
from gromo.containers.growing_mlp import GrowingMLP, Perceptron
from gromo.containers.growing_mlp_mixer import GrowingMLPMixer
from gromo.containers.growing_residual_mlp import GrowingResidualMLP
from gromo.containers.vgg import init_full_vgg_structure
from gromo.containers.growing_vision_transformer import (
    GrowingCCT,
    GrowingCVT,
    GrowingTransformer,
    GrowingViTLite,
)

try:
    from gromo.containers.resnet import init_full_resnet_structure
except ImportError:
    # resnet_basic_block is not merged yet in gromo
    # so we set init_full_resnet_structure to None in the meantime
    init_full_resnet_structure = None
from gromo.utils.utils import activation_fn
from torchvision.models import resnet18, resnet34
from torchvision.models.resnet import BasicBlock, _resnet


def resnet10(**kwargs) -> torch.nn.Module:
    return _resnet(BasicBlock, [1, 1, 1, 1], **kwargs)


def get_resnet(
    input_shape: int | tuple[int, int, int],
    out_features: int,
    device: torch.device,
    resnet_name: str = "resnet18",
) -> torch.nn.Module:
    known_resnets = {
        "resnet10": resnet10,
        "resnet18": resnet18,
        "resnet34": resnet34,
    }
    if resnet_name not in known_resnets:
        raise ValueError(f"Unknown ResNet model: {resnet_name}")
    model = known_resnets[resnet_name](weights=None, num_classes=out_features)
    if isinstance(input_shape, tuple) and (input_shape[0] <= 32 or input_shape[1] <= 32):
        # Modify the first convolutional layer for smaller input sizes
        model.conv1 = torch.nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            device=device,
        )

        model.maxpool = torch.nn.Identity()
    model.to(device)
    return model


def build_growing_transformer(
    input_shape: int | tuple[int, int, int],
    out_features: int,
    device: torch.device,
    **kwargs,
) -> GrowingContainer:
    return GrowingTransformer(
        in_features=input_shape,
        out_features=out_features,
        device=device,
        **kwargs,
    )


def _image_transformer_kwargs(
    input_shape: int | tuple[int, int, int],
    out_features: int,
    device: torch.device,
    **kwargs,
) -> dict:
    if isinstance(input_shape, int) or len(input_shape) != 3:
        raise ValueError(
            "Growing CCT/CVT/ViT-Lite models expect image input_shape "
            "(channels, height, width)."
        )
    channels, height, width = input_shape
    model_kwargs = {
        "img_size": (height, width),
        "n_input_channels": channels,
        "num_classes": out_features,
        "device": device,
    }
    model_kwargs.update(kwargs)
    model_kwargs["num_classes"] = out_features
    model_kwargs["device"] = device
    return model_kwargs


def build_growing_cct(
    input_shape: int | tuple[int, int, int],
    out_features: int,
    device: torch.device,
    **kwargs,
) -> GrowingContainer:
    return GrowingCCT(
        **_image_transformer_kwargs(
            input_shape=input_shape,
            out_features=out_features,
            device=device,
            **kwargs,
        )
    )


def build_growing_cvt(
    input_shape: int | tuple[int, int, int],
    out_features: int,
    device: torch.device,
    **kwargs,
) -> GrowingContainer:
    return GrowingCVT(
        **_image_transformer_kwargs(
            input_shape=input_shape,
            out_features=out_features,
            device=device,
            **kwargs,
        )
    )


def build_growing_vit_lite(
    input_shape: int | tuple[int, int, int],
    out_features: int,
    device: torch.device,
    **kwargs,
) -> GrowingContainer:
    return GrowingViTLite(
        **_image_transformer_kwargs(
            input_shape=input_shape,
            out_features=out_features,
            device=device,
            **kwargs,
        )
    )


known_architectures: dict[str, Callable[..., GrowingContainer]] = {
    "perceptron": Perceptron,
    "mlp": GrowingMLP,
    "residual_mlp": GrowingResidualMLP,
    "mlp_mixer": GrowingMLPMixer,
    "growing_transformer": build_growing_transformer,
    "growing_cct": build_growing_cct,
    "growing_cvt": build_growing_cvt,
    "growing_vit_lite": build_growing_vit_lite,
    "resnet": init_full_resnet_structure,
    "true_resnet": get_resnet,
    "vgg": init_full_vgg_structure,
}


def get_model_from_config(
    input_shape: int | tuple[int, int, int],
    out_features: int,
    device: torch.device,
    config: dict,
) -> GrowingContainer:
    # Access the configuration values
    model_name = config.pop("model")
    if "activation" in config:
        config["activation"] = activation_fn(config["activation"])
    model_kwargs = {
        "input_shape": input_shape,
        "out_features": out_features,
        "device": device,
    }
    model_kwargs.update(config)
    if model_name in known_architectures:
        return known_architectures[model_name](**model_kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
