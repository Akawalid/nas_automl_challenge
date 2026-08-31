import numpy as np
import torch


def warm_up_lr(iter, total_iters, lr_final, lr_init=0.0):
    return lr_init + (lr_final - lr_init) * iter / total_iters


class StepScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int,
        gamma: float,
        lr_init: float,
        num_batches_per_epoch: int,
        warmup_epochs: int = 0,
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
        self,
        optimizer: torch.optim.Optimizer,
        milestones: list,
        gamma: float,
        lr_init: float,
        num_batches_per_epoch: int,
        warmup_epochs: int = 0,
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
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr: float,
        num_batches_per_epoch: int,
        warmup_epochs: int = 0,
    ):
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
        optimizer: torch.optim.Optimizer,
        lr_init,
        warmup_epochs,
        total_epochs,
        num_batches_per_epoch: int,
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


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler.LRScheduler):
    """Epoch-wise linear warmup followed by cosine annealing.

    This is intentionally small so it can be instantiated from Hydra like a
    standard PyTorch scheduler while matching the CCT baseline's warmup/cosine
    training recipe.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-5,
        eta_min: float = 1e-5,
        warmup_prefix: bool = True,
        last_epoch: int = -1,
    ) -> None:
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if warmup_epochs >= max_epochs:
            raise ValueError("warmup_epochs must be smaller than max_epochs")
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        # ``warmup_prefix`` controls where the cosine clock starts after warmup:
        # - True  (default): cosine spans ``[warmup_epochs, max_epochs]`` (the
        #   original behaviour; lr equals ``base_lr`` exactly at the warmup end).
        # - False: cosine clock counts from epoch 0 over ``max_epochs`` and is NOT
        #   reset after warmup. This matches timm's ``CosineLRScheduler`` default
        #   (``warmup_prefix=False``) used by Compact-Transformers.
        # In both cases the cosine is clamped at ``max_epochs`` so any epochs run
        # beyond it stay at ``eta_min`` (i.e. timm's cooldown-at-min behaviour).
        self.warmup_prefix = warmup_prefix
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        epoch = self.last_epoch
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            alpha = epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]

        if self.warmup_prefix:
            cosine_epochs = max(1, self.max_epochs - self.warmup_epochs)
            progress = min(1.0, (epoch - self.warmup_epochs) / cosine_epochs)
        else:
            progress = min(1.0, epoch / self.max_epochs)
        cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor
            for base_lr in self.base_lrs
        ]
