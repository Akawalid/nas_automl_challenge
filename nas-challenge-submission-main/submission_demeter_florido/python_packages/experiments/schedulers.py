from dataclasses import dataclass

import torch

try:
    from tools.lr_scheduler import (
        ConstantScheduler,
        CosineScheduler,
        MultistepScheduler,
        StepScheduler,
        WarmupCosineAnnealingLR,
        warm_up_lr,
    )
except ImportError:
    from tools.lr_scheduler import (
        ConstantScheduler,
        CosineScheduler,
        MultistepScheduler,
        StepScheduler,
        WarmupCosineAnnealingLR,
    )


@dataclass(frozen=True)
class PipelineSchedulerAblations:
    linear_warmup_enabled: bool = False
    linear_warmup_epochs: int | None = None
    global_scheduler_enabled: bool = False
    global_scheduler_total_epochs: int | None = None
    global_rebind_linear_warmup_enabled: bool = False
    global_rebind_linear_warmup_epochs: int | None = None
    post_growth_scheduler_enabled: bool = False
    growth_linear_warmup_enabled: bool = False
    growth_linear_warmup_epochs: int | None = None


def get_pipeline_scheduler_ablation_settings(
    training_config: dict,
) -> PipelineSchedulerAblations:
    """Read scheduler ablations without validating disabled parameters."""
    ablations = training_config.get("ablations", {})
    if not isinstance(ablations, dict):
        raise ValueError("training.ablations must be a mapping")

    linear_warmup = ablations.get("linear_warmup", {})
    global_scheduler = ablations.get("global_scheduler", {})
    post_growth_scheduler = ablations.get("post_growth_scheduler", {})
    if not isinstance(linear_warmup, dict):
        raise ValueError("training.ablations.linear_warmup must be a mapping")
    if not isinstance(global_scheduler, dict):
        raise ValueError("training.ablations.global_scheduler must be a mapping")
    if not isinstance(post_growth_scheduler, dict):
        raise ValueError(
            "training.ablations.post_growth_scheduler must be a mapping"
        )

    linear_warmup_enabled = linear_warmup.get("enabled", False)
    global_scheduler_enabled = global_scheduler.get("enabled", False)
    post_growth_scheduler_enabled = post_growth_scheduler.get("enabled", False)
    global_rebind_linear_warmup = global_scheduler.get("rebind_linear_warmup", {})
    if not isinstance(global_rebind_linear_warmup, dict):
        raise ValueError(
            "training.ablations.global_scheduler.rebind_linear_warmup "
            "must be a mapping"
        )
    global_rebind_linear_warmup_enabled = global_rebind_linear_warmup.get(
        "enabled", False
    )
    growth_linear_warmup = post_growth_scheduler.get("growth_linear_warmup", {})
    if not isinstance(growth_linear_warmup, dict):
        raise ValueError(
            "training.ablations.post_growth_scheduler.growth_linear_warmup "
            "must be a mapping"
        )
    growth_linear_warmup_enabled = growth_linear_warmup.get(
        "enabled", False
    )
    if not isinstance(linear_warmup_enabled, bool):
        raise ValueError(
            "training.ablations.linear_warmup.enabled must be a boolean"
        )
    if not isinstance(global_scheduler_enabled, bool):
        raise ValueError(
            "training.ablations.global_scheduler.enabled must be a boolean"
        )
    if not isinstance(global_rebind_linear_warmup_enabled, bool):
        raise ValueError(
            "training.ablations.global_scheduler.rebind_linear_warmup.enabled "
            "must be a boolean"
        )
    if not isinstance(post_growth_scheduler_enabled, bool):
        raise ValueError(
            "training.ablations.post_growth_scheduler.enabled must be a boolean"
        )
    if not isinstance(growth_linear_warmup_enabled, bool):
        raise ValueError(
            "training.ablations.post_growth_scheduler.growth_linear_warmup.enabled "
            "must be a boolean"
        )
    if post_growth_scheduler_enabled and linear_warmup_enabled:
        raise ValueError(
            "training.ablations.post_growth_scheduler cannot be combined with "
            "the top-level linear_warmup ablation. Use "
            "training.ablations.post_growth_scheduler.growth_linear_warmup "
            "instead."
        )
    if growth_linear_warmup_enabled and not post_growth_scheduler_enabled:
        raise ValueError(
            "training.ablations.post_growth_scheduler.growth_linear_warmup requires "
            "training.ablations.post_growth_scheduler.enabled"
        )
    if global_rebind_linear_warmup_enabled and not global_scheduler_enabled:
        raise ValueError(
            "training.ablations.global_scheduler.rebind_linear_warmup requires "
            "training.ablations.global_scheduler.enabled"
        )

    global_scheduler_total_epochs = global_scheduler.get("total_epochs")
    if global_scheduler_enabled and (
        isinstance(global_scheduler_total_epochs, bool)
        or not isinstance(global_scheduler_total_epochs, int)
        or global_scheduler_total_epochs <= 0
    ):
        raise ValueError(
            "training.ablations.global_scheduler.total_epochs must be a "
            "positive integer when the global scheduler is enabled"
        )
    global_rebind_linear_warmup_epochs = global_rebind_linear_warmup.get("epochs")
    if global_rebind_linear_warmup_enabled and (
        isinstance(global_rebind_linear_warmup_epochs, bool)
        or not isinstance(global_rebind_linear_warmup_epochs, int)
        or global_rebind_linear_warmup_epochs <= 0
    ):
        raise ValueError(
            "training.ablations.global_scheduler.rebind_linear_warmup.epochs "
            "must be a positive integer when global rebind linear warmup is enabled"
        )
    growth_linear_warmup_epochs = growth_linear_warmup.get("epochs")
    if growth_linear_warmup_enabled and (
        isinstance(growth_linear_warmup_epochs, bool)
        or not isinstance(growth_linear_warmup_epochs, int)
        or growth_linear_warmup_epochs <= 0
    ):
        raise ValueError(
            "training.ablations.post_growth_scheduler.growth_linear_warmup.epochs "
            "must be a positive integer when growth linear warmup is enabled"
        )

    return PipelineSchedulerAblations(
        linear_warmup_enabled=linear_warmup_enabled,
        linear_warmup_epochs=linear_warmup.get("epochs"),
        global_scheduler_enabled=global_scheduler_enabled,
        global_scheduler_total_epochs=global_scheduler_total_epochs,
        global_rebind_linear_warmup_enabled=global_rebind_linear_warmup_enabled,
        global_rebind_linear_warmup_epochs=global_rebind_linear_warmup_epochs,
        post_growth_scheduler_enabled=post_growth_scheduler_enabled,
        growth_linear_warmup_enabled=growth_linear_warmup_enabled,
        growth_linear_warmup_epochs=growth_linear_warmup_epochs,
    )


def get_pipeline_linear_warmup_settings(
    settings: PipelineSchedulerAblations,
    growth_complete: bool,
) -> tuple[bool, int | None]:
    """Return the warmup settings that apply to the current train phase."""
    if settings.post_growth_scheduler_enabled and not growth_complete:
        return (
            settings.growth_linear_warmup_enabled,
            settings.growth_linear_warmup_epochs,
        )
    if settings.post_growth_scheduler_enabled:
        return False, None
    return settings.linear_warmup_enabled, settings.linear_warmup_epochs


def is_pipeline_linear_warmup_active(
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    settings: PipelineSchedulerAblations,
    growth_complete: bool = False,
) -> bool:
    """Return whether the current training epoch uses the warmup schedule."""
    if scheduler is not None and hasattr(scheduler, "is_pipeline_warmup_active"):
        return bool(scheduler.is_pipeline_warmup_active())
    linear_warmup_enabled, linear_warmup_epochs = get_pipeline_linear_warmup_settings(
        settings,
        growth_complete=growth_complete,
    )
    if not linear_warmup_enabled or scheduler is None:
        return False
    return scheduler.last_epoch < linear_warmup_epochs


def should_defer_pipeline_scheduler(
    settings: PipelineSchedulerAblations,
    growth_complete: bool,
) -> bool:
    """Return whether scheduler creation should wait until growth finishes."""
    if not settings.post_growth_scheduler_enabled:
        return False
    return not growth_complete


def get_scheduler(
    scheduler_name: str | None,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    num_batches_per_epoch: int,
    base_lr: float,
    warmup_epochs: int,
):
    if scheduler_name is None or scheduler_name.lower() == "none":
        return ConstantScheduler(optimizer, base_lr)
    elif scheduler_name == "step":
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
    elif scheduler_name == "multistep-cutout":
        return MultistepScheduler(
            optimizer,
            milestones=[60, 120, 160],
            gamma=0.2,
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
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def get_pipeline_scheduler(
    scheduler_class,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    eta_min: float,
    linear_warmup: bool = False,
    warmup_epochs: int | None = None,
    clamp_after_end: bool = False,
):
    """Build the pipeline scheduler, optionally enabling the warmup ablation."""
    if not isinstance(linear_warmup, bool):
        raise ValueError(
            "training.ablations.linear_warmup.enabled must be a boolean"
        )
    if not isinstance(clamp_after_end, bool):
        raise ValueError("clamp_after_end must be a boolean")

    if not linear_warmup and not clamp_after_end:
        return scheduler_class(
            optimizer=optimizer,
            T_max=num_epochs,
            eta_min=eta_min,
        )

    if scheduler_class is not torch.optim.lr_scheduler.CosineAnnealingLR:
        raise ValueError(
            "The scheduler ablations are only supported with "
            "torch.optim.lr_scheduler.CosineAnnealingLR"
        )
    if linear_warmup:
        if isinstance(warmup_epochs, bool) or not isinstance(warmup_epochs, int):
            raise ValueError(
                "training.ablations.linear_warmup.epochs must be an integer "
                "when linear warmup is enabled"
            )
        if warmup_epochs <= 0:
            raise ValueError(
                "training.ablations.linear_warmup.epochs must be positive "
                "when linear warmup is enabled"
            )
        if warmup_epochs >= num_epochs:
            raise ValueError(
                "training.ablations.linear_warmup.epochs must be smaller than "
                "the scheduler duration"
            )
    else:
        warmup_epochs = 0

    return WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=num_epochs,
        warmup_start_lr=eta_min,
        eta_min=eta_min,
    )


def get_pipeline_warmup_constant_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int | None,
    warmup_start_lr: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build an epoch-wise linear warmup followed by a constant base LR."""
    if isinstance(warmup_epochs, bool) or not isinstance(warmup_epochs, int):
        raise ValueError(
            "training.ablations.post_growth_scheduler.growth_linear_warmup.epochs "
            "must be an integer when growth linear warmup is enabled"
        )
    if warmup_epochs <= 0:
        raise ValueError(
            "training.ablations.post_growth_scheduler.growth_linear_warmup.epochs "
            "must be positive when growth linear warmup is enabled"
        )

    lr_lambdas = []
    for param_group in optimizer.param_groups:
        base_lr = float(param_group["lr"])
        start_factor = 1.0 if base_lr == 0 else warmup_start_lr / base_lr

        def lr_lambda(epoch: int, start_factor: float = start_factor) -> float:
            if epoch >= warmup_epochs:
                return 1.0
            alpha = epoch / warmup_epochs
            return start_factor + alpha * (1.0 - start_factor)

        lr_lambdas.append(lr_lambda)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas)


class WarmupThenScheduler:
    """Epoch-wise warmup to the current LR, then delegate to a scheduler."""

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        optimizer: torch.optim.Optimizer,
        target_lrs: list[float],
        warmup_epochs: int,
        warmup_start_lr: float,
    ) -> None:
        if warmup_epochs <= 0:
            raise ValueError("warmup_epochs must be positive")
        if len(optimizer.param_groups) != len(target_lrs):
            raise ValueError(
                "Cannot warm up scheduler rebind with different numbers of "
                "parameter groups"
            )

        self.scheduler = scheduler
        self.optimizer = optimizer
        self.target_lrs = [float(lr) for lr in target_lrs]
        self.warmup_epochs = warmup_epochs
        self.start_lrs = [
            min(float(warmup_start_lr), target_lr) for target_lr in self.target_lrs
        ]
        self.last_epoch = 0
        self.base_lrs = scheduler.base_lrs

        for param_group, start_lr in zip(self.optimizer.param_groups, self.start_lrs):
            param_group["lr"] = start_lr

    def is_pipeline_warmup_active(self) -> bool:
        return self.last_epoch < self.warmup_epochs

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def step(self) -> None:
        if self.is_pipeline_warmup_active():
            self.last_epoch += 1
            alpha = min(1.0, self.last_epoch / self.warmup_epochs)
            for param_group, start_lr, target_lr in zip(
                self.optimizer.param_groups,
                self.start_lrs,
                self.target_lrs,
            ):
                param_group["lr"] = start_lr + alpha * (target_lr - start_lr)
            return

        self.last_epoch += 1
        self.scheduler.step()


def get_persistent_pipeline_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Return the scheduler object that must persist across train phases."""
    return getattr(scheduler, "scheduler", scheduler)


def rebind_pipeline_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Attach a persistent scheduler to a freshly created optimizer."""
    last_lrs = scheduler.get_last_lr()
    if len(optimizer.param_groups) != len(last_lrs):
        raise ValueError(
            "Cannot preserve the scheduler across optimizers with different "
            "numbers of parameter groups"
        )

    for param_group, base_lr, current_lr in zip(
        optimizer.param_groups,
        scheduler.base_lrs,
        last_lrs,
    ):
        param_group["initial_lr"] = base_lr
        param_group["lr"] = current_lr

    scheduler.optimizer = optimizer
    return scheduler


def rebind_pipeline_scheduler_with_warmup(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int | None,
    warmup_start_lr: float,
) -> WarmupThenScheduler:
    """Attach a persistent scheduler to a new optimizer via linear warmup."""
    if isinstance(warmup_epochs, bool) or not isinstance(warmup_epochs, int):
        raise ValueError(
            "training.ablations.global_scheduler.rebind_linear_warmup.epochs "
            "must be an integer when global rebind linear warmup is enabled"
        )
    if warmup_epochs <= 0:
        raise ValueError(
            "training.ablations.global_scheduler.rebind_linear_warmup.epochs "
            "must be positive when global rebind linear warmup is enabled"
        )

    target_lrs = scheduler.get_last_lr()
    rebind_pipeline_scheduler(scheduler, optimizer)
    return WarmupThenScheduler(
        scheduler=scheduler,
        optimizer=optimizer,
        target_lrs=target_lrs,
        warmup_epochs=warmup_epochs,
        warmup_start_lr=warmup_start_lr,
    )


known_schedulers = {
    "step": StepScheduler,
    "multistep": MultistepScheduler,
    "cosine": CosineScheduler,
    "none": ConstantScheduler,
}
