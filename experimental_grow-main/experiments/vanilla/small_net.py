import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

if "/home/sdouka/Documents/Projects/InriaGitlab/experimental_grow/" not in sys.path:
    sys.path.append("/home/sdouka/Documents/Projects/InriaGitlab/experimental_grow/")
if "/home/tau/sdouka/codebase/experimental_grow" not in sys.path:
    sys.path.append("/home/tau/sdouka/codebase/experimental_grow")

from tools.datasets import get_dataloaders
from tools.augmentations import default_augmentations
from tools.logger import Logger


# ── 1. Model Registry ─────────────────────────────────────────────────────────
def get_model(name: str, num_classes: int = 1000, in_channels: int = 3, pretrained: bool = False):
    weights = "DEFAULT" if pretrained else None
    registry = {
        # MobileNet family
        "mobilenet_v2": lambda: models.mobilenet_v2(weights=weights),
        "mobilenet_v3_small": lambda: models.mobilenet_v3_small(weights=weights),
        "mobilenet_v3_large": lambda: models.mobilenet_v3_large(weights=weights),
        # EfficientNet family
        "efficientnet_b0": lambda: models.efficientnet_b0(weights=weights),
        "efficientnet_b1": lambda: models.efficientnet_b1(weights=weights),
        "efficientnet_b2": lambda: models.efficientnet_b2(weights=weights),
        "efficientnet_b3": lambda: models.efficientnet_b3(weights=weights),
        # ShuffleNet family
        "shufflenet_v2_x1_0": lambda: models.shufflenet_v2_x1_0(weights=weights),
        "shufflenet_v2_x1_5": lambda: models.shufflenet_v2_x1_5(weights=weights),
        # MNASNet family
        "mnasnet1_0": lambda: models.mnasnet1_0(weights=weights),
        # ResNet family (small variants)
        "resnet18": lambda: models.resnet18(weights=weights),
        "resnet34": lambda: models.resnet34(weights=weights),
        "resnet50": lambda: models.resnet50(weights=weights),
        # DenseNet
        "densenet121": lambda: models.densenet121(weights=weights),
        # ConvNeXt-Tiny
        "convnext_tiny": lambda: models.convnext_tiny(weights=weights),
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(registry)}")
    model = registry[name]()

    # Swap first conv layer if in_channels differs from the default (3)
    if in_channels != 3:
        if hasattr(model, "conv1") and isinstance(model.conv1, nn.Conv2d):
            # ResNet
            old = model.conv1
            model.conv1 = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
        elif hasattr(model, "features") and hasattr(model.features, "conv0"):
            # DenseNet
            old = model.features.conv0
            model.features.conv0 = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
        elif hasattr(model, "conv1") and isinstance(model.conv1, nn.Sequential):
            # ShuffleNet
            old = model.conv1[0]
            model.conv1[0] = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
        elif hasattr(model, "layers") and isinstance(model.layers[0], nn.Conv2d):
            # MNASNet
            old = model.layers[0]
            model.layers[0] = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
        elif hasattr(model, "features") and isinstance(model.features[0][0], nn.Conv2d):
            # MobileNet, EfficientNet, MobileNetV3: features[0][0]
            old = model.features[0][0]
            model.features[0][0] = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)

    # Swap classifier head if needed
    if num_classes != 1000:
        if hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    print(model)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── 2. Data Pipeline ──────────────────────────────────────────────────────────
def get_augmentation(
    img_size: int = 224,
    dataset_type: str = "unknown",
    mean: list = None,
    std: list = None,
):
    # ── Augmentation policies ──────────────────────────────────────────────────
    if dataset_type == "natural":
        # Full recipe: safe to flip and crop, content is not orientation-sensitive
        train_aug = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.25),
            ]
        )
    elif dataset_type == "text":
        # Text is sensitive to flips and perspective distortion.
        # Allow mild brightness/contrast shifts (lighting variation in scans)
        # but avoid anything that alters character shape or orientation.
        train_aug = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif dataset_type == "digits":
        # Digits are highly sensitive: no flips (6 vs 9), no rotations,
        # no color ops if grayscale. Only safe spatial jitter.
        train_aug = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomAffine(
                    degrees=5,  # tiny rotation tolerance
                    translate=(0.05, 0.05),  # small shift only
                    scale=(0.95, 1.05),  # subtle zoom
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:  # "unknown" - conservative safe default
        # RandAugment without flips or crops.
        # No geometric distortions that could break orientation-sensitive content.
        # RandomErasing is skipped: could erase a digit/character.
        train_aug = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandAugment(
                    num_ops=2,
                    magnitude=7,  # slightly lower than natural-image default
                    # interpolation is bilinear - safe for all content types
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                # No RandomErasing here - too risky for text/digit content
            ]
        )

    return train_aug


def get_loaders(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 128,
    dataset_type: str = "unknown",  # "unknown" | "text" | "digits" | "natural"
    mean: list = None,
    std: list = None,
):
    """
    dataset_type:
        "natural"  - standard photos, no text/digits = full augmentation
        "text"     - document images, OCR-style      = geometry-safe only
        "digits"   - MNIST-style, digit recognition  = minimal, no flips/rotations
        "unknown"  - conservative safe set (default) = RandAugment, no flips/crops
    """
    # Fall back to ImageNet stats if dataset mean/std are unknown
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]

    normalize = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),  # safe: no cropping, fixed size
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_aug = get_augmentation(
        img_size=img_size, dataset_type=dataset_type, mean=mean, std=std
    )

    # ── Loaders ────────────────────────────────────────────────────────────────
    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_aug)
    val_ds = datasets.ImageFolder(f"{data_dir}/val", transform=normalize)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=pin,
        pin_memory_device="cuda" if pin else "",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=pin,
        pin_memory_device="cuda" if pin else "",
    )
    return train_loader, val_loader


# ── 3. Training Loop ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        n += imgs.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        total_loss += criterion(out, labels).item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        n += imgs.size(0)
    return total_loss / n, correct / n


# ── 4. Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser("Vanilla small models evaluation")

    # Data
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--classes", type=int)
    parser.add_argument(
        "--dataset_mean", type=float, nargs=3, default=None, metavar=("R", "G", "B")
    )
    parser.add_argument(
        "--dataset_std", type=float, nargs=3, default=None, metavar=("R", "G", "B")
    )
    # parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--augment", action="store_true")

    # Model
    parser.add_argument("--model", type=str)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lrate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=256)

    # Logger
    parser.add_argument("--no-logger", action="store_true")
    parser.add_argument("--api", type=str, default="wandb")
    parser.add_argument("--exp_name", type=str, default="SmallBenchmarks")
    parser.add_argument("--port", type=int, default=27028)
    parser.add_argument(
        "--log_path",
        type=str,
        default="/data/iceberg_1/titanic_1/experimentslogs_shared/tau_frugal/stella/",
    )
    parser.add_argument("--tmpdir", type=str, default="temp")

    args = parser.parse_args()


    print(f"{torch.cuda.is_available()=}")
    print(f"{torch.cuda.device_count()=}")
    print(f"{torch.cuda.current_device()=}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"{device=}")

    train_aug = default_augmentations[args.dataset] if args.augment else None
    train_loader, _, test_loader, data_shape = get_dataloaders(
        args.dataset,
        args.data_dir,
        batch_size=args.batch_size,
        data_augmentation=train_aug,
    )

    assert isinstance(data_shape, tuple)
    in_channels = data_shape[0]
    print(f"{data_shape=}")
    model = get_model(args.model, num_classes=args.classes, in_channels=in_channels).to(device)


    logger = Logger(
        experiment_name=args.exp_name, port=args.port, api=args.api, enabled=not args.no_logger
    )
    logger.setup_tracking(file_path=args.log_path)

    with logger():
        for param, value in args._get_kwargs():
            if param in ("logger", "api", "exp_name", "port", "log_path", "tmpdir"):
                continue
            logger.log_parameter(f"{param}", value)

        logger.watch_pytorch_model(model)
        count = count_parameters(model)
        logger.log_metric("training/nb of parameters", count, 0)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=args.lrate, momentum=0.9, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            logger.log_metric("training/train accuracy", train_acc, epoch, "epoch")
            logger.log_metric("training/train loss", train_loss, epoch, "epoch")

            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            logger.log_metric("training/test accuracy", test_acc, epoch, "epoch")
            logger.log_metric("training/test loss", test_loss, epoch, "epoch")

            scheduler.step()
            print(
                f"[{epoch:3d}/{args.epochs}] "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {test_loss:.4f} acc {test_acc:.4f}"
            )

        logger.log_pytorch_model(
            model=model,
            name=f"{args.model}_{args.dataset}_final",
            x=None,
            path=args.tmpdir,
            run_id=False,
        )


if __name__ == "__main__":
    main()
