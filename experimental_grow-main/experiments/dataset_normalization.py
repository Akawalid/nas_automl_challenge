"""
This script is used to calculate the mean and standard deviation of a dataset.
"""

import torch
from torchvision import datasets, transforms


def calculate_mean(dataset):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=False, num_workers=0
    )
    mean = 0.0
    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean /= len(loader.dataset)
    return mean


def calculate_std(dataset, mean):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=False, num_workers=0
    )
    std = 0.0
    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        std += ((images - mean.view(1, -1, 1)) ** 2).mean(2).sum(0)
    std = torch.sqrt(std / len(loader.dataset))
    return std


def get_dataset_normalization(dataset_name):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(
            root="./dataset", train=True, download=True, transform=transform
        )
    elif dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(
            root="./dataset", train=True, download=True, transform=transform
        )
    elif dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            root="./dataset", train=True, download=True, transform=transform
        )
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(
            root="./dataset", train=True, download=True, transform=transform
        )
    elif dataset_name == "SVHN":
        train_dataset = datasets.SVHN(
            root="./dataset", split="train", download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    mean = calculate_mean(train_dataset)
    std = calculate_std(train_dataset, mean)
    return mean, std


if __name__ == "__main__":
    dataset_name = "SVHN"
    mean, std = get_dataset_normalization(dataset_name)
    print(f"Dataset: {dataset_name}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
