import torch
import torchvision.transforms as transforms
import numpy as np
import argparse
import math
from typing import Callable, Optional
from warnings import warn
from time import time

from gromo.utils.utils import global_device
from torchvision import datasets
from torchvision.transforms import v2 as Tv2  # this is the new system

import torch.nn as nn
from gromo.containers.growing_container import GrowingContainer
from gromo.containers.growing_mlp import GrowingMLP, Perceptron
from gromo.containers.growing_mlp_mixer import GrowingMLPMixer
from gromo.containers.growing_residual_mlp import GrowingResidualMLP


#Transforms for different datasets

def get_transforms(
    dataset_name: str, data_augmentation: list[str] | None = None
) -> tuple[list, list]:
    datasets_transforms = {
        "mnist": [
            Tv2.ToDtype(torch.float32, scale=True),
            Tv2.Normalize(mean=(0.1307,), std=(0.3081,)),
        ],
        "fashion-mnist": [
            Tv2.ToDtype(torch.float32, scale=True),
            Tv2.Normalize(mean=(0.2860,), std=(0.3530,)),
        ],
        "cifar10": [
            Tv2.ToDtype(torch.float32, scale=True),
            Tv2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ],
        "cifar100": [
            Tv2.ToDtype(torch.float32, scale=True),
            Tv2.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)
            ),
        ],
        "svhn": [
            Tv2.ToDtype(torch.float32, scale=True),
            Tv2.Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970)),
        ],
        "addnist": [
            Tv2.ToDtype(torch.float32, scale=True),
        ],
        "cifartile": [
            Tv2.ToDtype(torch.float32, scale=True),
        ],
    }

    augmentation_transforms = []
    if data_augmentation:
        for aug in data_augmentation:
            if aug == "horizontal_flip":
                augmentation_transforms.append(Tv2.RandomHorizontalFlip())
            elif aug == "rotation":
                augmentation_transforms.append(Tv2.RandomRotation(10))
            elif aug == "crop":
                augmentation_transforms.append(
                                    Tv2.RandomCrop(32, padding=4, padding_mode="reflect")
                                )                
            elif aug == "autoaugment":
                if dataset_name in ["cifar10", "cifar100"]:
                    policy = transforms.AutoAugmentPolicy.CIFAR10
                elif dataset_name == "svhn":
                    policy = transforms.AutoAugmentPolicy.SVHN
                else:
                    raise ValueError(f"AutoAugment not available for {dataset_name}")
                augmentation_transforms.append(Tv2.AutoAugment(policy=policy))
            elif aug == "randaugment":
                augmentation_transforms.append(Tv2.RandAugment())
            else:
                warn(f"Unknown augmentation {aug}")
            # Add more augmentations as needed
    return datasets_transforms[dataset_name], augmentation_transforms



# Get model from configuration 

known_activations = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "selu": nn.SELU(),
    "silu": nn.SiLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "identity": nn.Identity(),
}

def get_model_from_config(in_features, out_features, config) -> GrowingContainer:
    # Access the configuration values
    model_name = config.pop("model")
    if "activation" in config:
        config["activation"] = known_activations[config["activation"]]
    model_kwargs = {
        "in_features": in_features,
        "out_features": out_features,
    }
    model_kwargs.update(config)
    if model_name == "perceptron":
        return Perceptron(**model_kwargs)
    elif model_name == "mlp":
        return GrowingMLP(**model_kwargs)
    elif model_name == "residual_mlp":
        return GrowingResidualMLP(**model_kwargs)
    elif model_name == "mlp_mixer":
        return GrowingMLPMixer(**model_kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")



# Auxiliary functions for training

def compute_statistics(
    growing_model: GrowingContainer,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module = nn.MSELoss(reduction="sum"),
    aux_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    batch_limit: int = 1_000_000,
    device: torch.device | None = None,
    show: bool = False,
) -> tuple[float, float]:
    """
    Compute the tensor of statistics of the model on the dataloader
    with a limit of batch_limit batches.

    Parameters
    ----------
    growing_model: GrowingMLP
        The model to evaluate
    loss_function: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        The loss function to use.
        /!/ The loss function should not be averaged over the batch
    aux_loss_function: nn.Module | None
        The auxiliary loss function to use.
    dataloader: DataLoader
        The dataloader to use
    batch_limit: int
        The maximum number of batches to use
    device: torch.device
        The device to use
    show: bool
        If True, display a progress bar
    """
    assert (
        loss_function.reduction == "sum"
    ), "The loss function should not be averaged over the batch"

    if device is None:
        device = global_device()

    growing_model.init_computation()
    loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()

    if show:
        dataloader = tqdm(dataloader)

    for i, (x, y) in enumerate(dataloader):
        if 0 <= batch_limit <= i:
            break
        growing_model.zero_grad()
        x, y = x.to(device), y.to(device)
        y_pred = growing_model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        growing_model.update_computation()
        loss_meter.update(loss.item() / x.size(0), x.size(0))
        if aux_loss_function is not None:
            aux_loss = aux_loss_function(y_pred, y)
            aux_loss_meter.update(aux_loss.item(), x.size(0))
    return loss_meter.avg, aux_loss_meter.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val != np.nan and val != np.inf:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    aux_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    batch_limit: int = -1,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """
    /!/ The loss function should not be averaged over the batch
    """
    assert (
        loss_function.reduction == "mean"
    ), "The loss function should be averaged over the batch"

    if device is None:
        device = global_device()

    # metrics meters
    loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if 0 <= batch_limit <= i:
                break
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            loss_meter.update(loss.item(), x.size(0))
            if aux_loss_function is not None:
                aux_loss_meter.update(aux_loss_function(y_pred, y).item(), x.size(0))
    return loss_meter.avg, aux_loss_meter.avg


def extended_evaluate_model(
    growing_model: GrowingContainer,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    aux_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    batch_limit: int = -1,
    device: torch.device | None = None,
) -> tuple[float, float]:
    assert (
        loss_function.reduction == "sum"
    ), "The loss function should not be averaged over the batch"

    if device is None:
        device = global_device()

    growing_model.eval()
    loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if 0 <= batch_limit <= i:
                break
            x, y = x.to(device), y.to(device)
            y_pred = growing_model.extended_forward(x)
            loss = loss_function(y_pred, y)
            loss_meter.update(loss.item() / x.size(0), x.size(0))
            if aux_loss_function is not None:
                aux_loss_meter.update(aux_loss_function(y_pred, y).item(), x.size(0))
    return loss_meter.avg, aux_loss_meter.avg


def line_search(
    model: GrowingContainer,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module = nn.MSELoss(reduction="sum"),
    aux_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    batch_limit: int = -1,
    initial_loss: float | None = None,
    first_order_improvement: float = 1,
    alpha: float = 0.1,
    beta: float = 0.5,
    t0: float | None = None,
    extended_search: bool = True,
    max_iter: int = 100,
    epsilon: float = 1e-7,
    verbose: bool = False,
    device: torch.device | None = None,
) -> tuple[float, float, list[float], list[float]]:
    assert model.currently_updated_layer is not None, "No currently updated layer"

    if device is None:
        device = global_device()

    gammas = []
    losses = []
    aux_losses = []
    beta = np.sqrt(beta)
    epsilon = np.sqrt(epsilon)
    if isinstance(first_order_improvement, torch.Tensor):
        first_order_improvement = first_order_improvement.item()
    if isinstance(initial_loss, torch.Tensor):
        initial_loss = initial_loss.item()

    def test_gamma(sqrt_gamma):
        model.currently_updated_layer.scaling_factor = sqrt_gamma
        loss, aux_loss = extended_evaluate_model(
            growing_model=model,
            dataloader=dataloader,
            loss_function=loss_function,
            aux_loss_function=aux_loss_function,
            batch_limit=batch_limit,
            device=device,
        )
        gammas.append(sqrt_gamma**2)
        losses.append(loss)
        aux_losses.append(aux_loss)
        if verbose:
            print(f"gamma n° {len(gammas)}: {sqrt_gamma ** 2:.3e} -> Loss: {loss:.3e}")
        return loss, aux_loss

    def under_bound(sqrt_gamma: float, loss: float):
        return loss < initial_loss - alpha * sqrt_gamma**2 * first_order_improvement

    if initial_loss is None:
        warn("Initial loss is not provided, computing it")
        initial_loss, _ = test_gamma(0.0)
        print(f"Initial loss: {initial_loss:.3e}")
    else:
        gammas.append(0.0)
        losses.append(initial_loss)
        aux_losses.append(0.0)

    # gamma = t ** 2
    if t0 is None:
        t = np.sqrt(2 * (initial_loss / first_order_improvement))
    else:
        t = np.sqrt(t0)
    l0, l0_aux = test_gamma(t)
    l1, l1_aux = l0, l0_aux
    i = 0
    if under_bound(t, l0):
        if extended_search:
            go = True
            while go:
                l0, l0_aux = l1, l1_aux
                t /= beta
                l1, l1_aux = test_gamma(t)
                go = l1 < l0 and i < max_iter
                i += 1
            t *= beta
        model.currently_updated_layer.scaling_factor = t
    else:
        go = True
        while go:
            l0, l0_aux = l1, l1_aux
            t *= beta
            l1, l1_aux = test_gamma(t)
            go = (
                ((not under_bound(t, l1)) or (l1 < l0 and extended_search))
                and i < max_iter
                and t > epsilon
            )
            i += 1
        t /= beta
        model.currently_updated_layer.scaling_factor = t

    return t**2, l0, l0_aux, gammas, losses, aux_losses


def topk_accuracy(y_pred, y, k=1):
    result = y_pred.topk(k, dim=1).indices == y.unsqueeze(1)
    return result.sum() / y.size(0)


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    aux_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    scheduler=None,
    device: torch.device | None = None,
    show: bool = False,
    cutmix_beta: float = 1.0,
    cutmix_prob: float = 0.0,
    batch_limit: int = -1,
):
    """
    Train the model on the train_dataloader
    """
    assert (
        loss_function.reduction == "mean"
    ), "The loss function should be averaged over the batch"

    if device is None:
        device = global_device()

    # metrics meters
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    # time meters
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    transfer_time_meter = AverageMeter()

    model.train()
    start_time = time()
    for i, (x, y) in enumerate(train_dataloader):
        if 0 <= batch_limit <= i:
            break
        data_time_meter.update(time() - start_time)

        x = x.to(device)
        y = y.to(device)
        transfer_time_meter.update(time() - start_time)

        optimizer.zero_grad()

        # Apply CutMix
        x, y, y_shuffled, lam = cutmix_data(
            x, y, beta=cutmix_beta, cutmix_prob=cutmix_prob
        )

        y_pred = model(x)
        loss = lam * loss_function(y_pred, y) + (1 - lam) * loss_function(
            y_pred, y_shuffled
        )
        assert (
            loss.isnan().sum() == 0
        ), f"During training of {model}, loss is NaN: {loss}, sample index: {i/len(train_dataloader)}"

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # update metrics
        loss_meter.update(loss.item(), x.size(0))
        if aux_loss_function:
            accuracy_meter.update(aux_loss_function(y_pred, y).item(), x.size(0))

        batch_time_meter.update(time() - start_time)
        start_time = time()

    if scheduler is not None:
        scheduler.epoch_step()

    if show:
        print(
            f"Train: loss={loss_meter.avg:.3e}, accuracy={accuracy_meter.avg:.2f}, time={batch_time_meter.avg:.2f}s"
        )
    return loss_meter.avg, accuracy_meter.avg


# Scheduler functions

def get_scheduler(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    num_batches_per_epoch: int,
    base_lr: float,
    warmup_epochs: int,
):
    if scheduler_name == "step":
        return StepScheduler(
            optimizer,
            step_size=num_epochs // 3,
            gamma=0.1,
            lr_init=base_lr,
            num_batches_per_epoch=num_batches_per_epoch,
            warmup_epochs=warmup_epochs,
        )
    elif scheduler_name == "multistep":
        return MultistepScheduler(
            optimizer,
            milestones=[num_epochs // 2, 3 * num_epochs // 4],
            gamma=0.1,
            lr_init=base_lr,
            num_batches_per_epoch=num_batches_per_epoch,
            warmup_epochs=warmup_epochs,
        )
    elif scheduler_name == "cosine":
        return CosineScheduler(
            optimizer,
            lr_init=base_lr,
            warmup_epochs=warmup_epochs,
            total_epochs=num_epochs,
            num_batches_per_epoch=num_batches_per_epoch,
            min_lr=1e-6,
        )
    elif scheduler_name == "none":
        return ConstantScheduler(optimizer, base_lr)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def warm_up_lr(iter, total_iters, lr_final, lr_init=0.0):
    return lr_init + (lr_final - lr_init) * iter / total_iters

class StepScheduler:
    def __init__(
        self, optimizer, step_size, gamma, lr_init, num_batches_per_epoch, warmup_epochs
    ):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.lr = lr_init
        self.current_epoch = 0
        self.current_step = 0
        self.num_batches_per_epoch = num_batches_per_epoch
        self.warmup_epochs = warmup_epochs

    def step(self):
        self.current_step += 1
        if self.current_epoch < self.warmup_epochs:
            current_step = (
                self.current_epoch * self.num_batches_per_epoch + self.current_step
            )
            lr = warm_up_lr(
                current_step, self.warmup_epochs * self.num_batches_per_epoch, self.lr
            )
        else:
            lr = self.lr * (self.gamma ** (self.current_epoch // self.step_size))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def epoch_step(self):
        self.current_epoch += 1
        self.current_step = 0


class MultistepScheduler:
    def __init__(
        self, optimizer, milestones, gamma, lr_init, num_batches_per_epoch, warmup_epochs
    ):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.lr = lr_init
        self.current_epoch = 0
        self.current_step = 0
        self.num_batches_per_epoch = num_batches_per_epoch
        self.warmup_epochs = warmup_epochs

    def step(self):
        self.current_step += 1
        if self.current_epoch < self.warmup_epochs:
            current_step = (
                self.current_epoch * self.num_batches_per_epoch + self.current_step
            )
            lr = warm_up_lr(
                current_step, self.warmup_epochs * self.num_batches_per_epoch, self.lr
            )
        else:
            lr = self.lr
            for milestone in self.milestones:
                if self.current_epoch >= milestone:
                    lr *= self.gamma
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def epoch_step(self):
        self.current_epoch += 1
        self.current_step = 0


class ConstantScheduler:
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

    def epoch_step(self):
        pass


# Learning rate scheduler
class CosineScheduler:
    def __init__(
        self,
        optimizer,
        lr_init,
        warmup_epochs,
        total_epochs,
        num_batches_per_epoch,
        min_lr=1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        self.min_lr = min_lr
        self.base_lr = lr_init
        self.current_epoch = 0
        self.current_step = 0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"CosineScheduler(base_lr={self.base_lr}, min_lr={self.min_lr}, warmup_epochs={self.warmup_epochs}, total_epochs={self.total_epochs})"

    def step(self):
        self.current_step += 1
        minibatch_step = (
            self.current_step + self.current_epoch * self.num_batches_per_epoch
        )
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (
                minibatch_step / (self.warmup_epochs * self.num_batches_per_epoch)
            )
        else:
            progress = (
                minibatch_step - self.warmup_epochs * self.num_batches_per_epoch
            ) / ((self.total_epochs - self.warmup_epochs) * self.num_batches_per_epoch)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def epoch_step(self):
        self.current_epoch += 1
        self.current_step = 0

known_schedulers = {
    "step": StepScheduler,
    "multistep": MultistepScheduler,
    "cosine": CosineScheduler,
    "none": ConstantScheduler,
}

# Parser and arguments definition

selection_methods = [
    "none",
    "fo",
    "scaled_fo",
    "one_step_fo",
]

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLP training")

    # general arguments
    general_group = parser.add_argument_group("general")
    general_group.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    general_group.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="number of workers for the dataloader (default: 4)",
    )

    logging_group = parser.add_argument_group("logging")
    logging_group.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="directory to save logs (default: logs)",
    )
    logging_group.add_argument(
        "--log-file-name",
        type=str,
        default=None,
        help="name of the log file (default: log)",
    )
    logging_group.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="name of the experiment (default: None)",
    )
    logging_group.add_argument(
        "--tags",
        type=str,
        default=None,
        help="tags to add to the experiment (default: None)",
    )
    logging_group.add_argument(
        "--log-system-metrics",
        action="store_true",
        default=False,
        help="log system metrics (default: False)",
    )

    # dataset arguments
    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="dataset to use (default: mnist)",
        choices=known_datasets.keys(),
    )
    dataset_group.add_argument(
        "--nb-class", type=int, default=None, help="number of classes (default: None)"
    )
    dataset_group.add_argument(
        "--split-train-val",
        type=float,
        default=0.2,
        help="proportion of the training set to use as validation set (default: 0.3)",
    )
    dataset_group.add_argument(
        "--dataset-path",
        type=str,
        default="dataset",
        help="path to the dataset (default: dataset)",
    )
    dataset_group.add_argument(
        "--data-augmentation",
        nargs="+",
        default=None,
        help="data augmentation to use (default: None)",
    )

    # model arguments
    model_group = parser.add_argument_group("architecture")
    model_group.add_argument(
        "--config-path",
        type=str,
        default="models/configs/mlp.yml",
        help="path to the configuration file (default: None)",
    )
    # define additional arguments to override the configuration

    # classical training arguments
    training_group = parser.add_argument_group("training")
    training_group.add_argument(
        "--seed", type=int, default=None, help="random seed (default: 0)"
    )
    training_group.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    general_group.add_argument(
        "--nb-step", type=int, default=10, help="number of cycles (default: 10)"
    )
    training_group.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="optimizer to use (default: sgd)",
        choices=known_optimizers.keys(),
    )
    training_group.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.01)"
    )
    training_group.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        help="weight decay (default: 0)",
    )
    training_group.add_argument(
        "--training-threshold",
        type=float,
        default=None,
        help="training is stopped when the loss is below this threshold (default: None)",
    )

    # scheduler arguments
    scheduler_group = parser.add_argument_group("scheduler")
    scheduler_group.add_argument(
        "--scheduler",
        type=str,
        default="none",
        help="scheduler to use (default: step)",
        choices=known_schedulers.keys(),
    )
    scheduler_group.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="number of warmup iterations (default: 0)",
    )

    # growing arguments
    growing_group = parser.add_argument_group("growing")
    growing_group.add_argument(
        "--epochs-per-growth",
        type=int,
        default=-1,
        help="number of epochs before growing the model (default: -1)",
    )
    growing_group.add_argument(
        "--selection-method",
        type=str,
        default="none",
        help="selection method to use (default: none)",
        choices=selection_methods,
    )
    growing_group.add_argument(
        "--growing-batch-limit",
        type=int,
        default=-1,
        help="maximum number of batches to use (default: -1)",
    )
    growing_group.add_argument(
        "--growing-part",
        type=str,
        default="all",
        help="part of the model to grow (default: all)",
        choices=["all", "parameter", "neuron"],
    )
    growing_group.add_argument(
        "--growing-numerical-threshold",
        type=float,
        default=1e-5,
        help="numerical threshold for growing (default: 1e-5)",
    )
    growing_group.add_argument(
        "--growing-statistical-threshold",
        type=float,
        default=1e-3,
        help="statistical threshold for growing (default: 1e-3)",
    )
    growing_group.add_argument(
        "--growing-maximum-added-neurons",
        type=int,
        default=10,
        help="maximum number of neurons to add (default: None)",
    )
    growing_group.add_argument(
        "--growing-computation-dtype",
        type=str,
        default="float32",
        help="dtype to use for the computation (default: float32)",
        choices=["float32", "float64"],
    )
    growing_group.add_argument(
        "--normalize-weights",
        action="store_true",
        default=False,
        help="normalize the weights after growing (default: False)",
    )
    growing_group.add_argument(
        "--init-new-neurons-with-random-in-and-zero-out",
        action="store_true",
        default=False,
        help="initialize the new neurons with random fan-in weights "
        "and zero fan-out weights (default: False)",
    )

    # line search arguments
    line_search_group = parser.add_argument_group("line search")
    line_search_group.add_argument(
        "--line-search-alpha",
        type=float,
        default=0.1,
        help="line search alpha (default: 0.1)",
    )
    line_search_group.add_argument(
        "--line-search-beta",
        type=float,
        default=0.5,
        help="line search beta (default: 0.5)",
    )
    line_search_group.add_argument(
        "--line-search-max-iter",
        type=int,
        default=20,
        help="line search max iteration (default: 100)",
    )
    line_search_group.add_argument(
        "--line-search-epsilon",
        type=float,
        default=1e-7,
        help="line search epsilon (default: 1e-7)",
    )
    line_search_group.add_argument(
        "--line-search-batch-limit",
        type=int,
        default=-1,
        help="maximum number of batches to use (default: -1)",
    )
    return parser


def update_config_from_args(config: dict, args: argparse.Namespace) -> dict:
    """
    Override the configuration with the arguments from the command line arguments.

    Parameters
    ----------
    config : dict
        configuration
    args : argparse.Namespace
        arguments

    Returns
    -------
    dict
        updated configuration
    """
    for key in config.keys():
        if key in args:
            config[key] = getattr(args, key)
    return config


# Datasets classes

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
        label_key: str = "_y"
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

class AddNIST(NpyWebDataset):
    def __init__(
        self,
        train: bool = True,
        root: str = "data/webdatasets/npy",
        download: bool = True,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            url="https://data.ncl.ac.uk/ndownloader/articles/24574354/versions/1",
            name="AddNIST",
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
            url="https://data.ncl.ac.uk/ndownloader/articles/24551539/versions/1",
            train=train,
            root=root,
            name="CIFARTile",
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
    "cifartile": CIFARTile,
}


known_optimizers = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}

# show time function given in example

# given a number of seconds, these two functions combine to print it out in a human-readable format
# I use these to print out status updates during my training loop
def div_remainder(n, interval):
    # finds divisor and remainder given some n/interval
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder

def show_time(seconds):
    # show amount of time as human readable
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < (60 * 60):
        minutes, seconds = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{}h,{}m,{}s".format(hours, minutes, seconds)