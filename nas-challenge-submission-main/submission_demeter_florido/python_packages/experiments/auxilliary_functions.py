import logging
from time import time
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from deprecated import deprecated
from gromo.containers.growing_container import GrowingContainer
from gromo.utils.training_utils import evaluate_model as gromo_evaluate_model
from gromo.utils.utils import global_device
from torchmetrics import Metric
from tqdm.notebook import tqdm

from tools.metrics import MetricTracker


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, reduction="mean"):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        log_prob = nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum: float | int = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self()

    def update(self, val: float | int, n=1):
        if val != torch.nan and val != torch.inf:
            self.val = val
            self.sum += val * n
            self.count += n

    def __call__(self) -> float:
        if self.count == 0:
            return 0.0
            # raise ValueError("AverageMeter has no values to compute average")
        else:
            if isinstance(self.sum, torch.Tensor):
                return (self.sum / self.count).item()  # type: ignore
            elif isinstance(self.sum, (float, int)):
                return self.sum / self.count
            else:
                raise TypeError(f"Unsupported type for sum: {type(self.sum)}")


# CutMix function
def cutmix_data(x, y, beta=1.0, cutmix_prob=0.5):
    if np.random.rand() > cutmix_prob:
        return x, y, y, 1.0  # No CutMix applied

    indices = torch.randperm(x.size(0))
    shuffled_x = x[indices]
    shuffled_y = y[indices]

    lam = np.random.beta(beta, beta)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = shuffled_x[:, :, bbx1:bbx2, bby1:bby2]

    return x, y, shuffled_y, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


@deprecated("This functionality is already in gromo module (without cutmix).")
def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    aux_loss_function: MetricTracker | None = None,
    scheduler=None,
    device: torch.device | None = None,
    show: bool = False,
    cutmix_beta: float = 1.0,
    cutmix_prob: float = 0.0,
    batch_limit: int = -1,
) -> tuple[float, float]:
    """
    Train the model on the train_dataloader

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_dataloader : torch.utils.data.DataLoader
        The dataloader for training data.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    loss_function : nn.Module
        The loss function to use. Must have reduction="mean".
    aux_loss_function : MetricTracker | None, optional
        A MetricTracker instance to track auxiliary metrics (e.g., accuracy).
        Will be reset at the start and updated each batch. Default is None.
    scheduler : optional
        Learning rate scheduler. Default is None.
    device : torch.device | None, optional
        Device to use. Default is None (uses global_device()).
    show : bool, optional
        Whether to print training progress. Default is False.
    cutmix_beta : float, optional
        Beta parameter for CutMix. Default is 1.0.
    cutmix_prob : float, optional
        Probability of applying CutMix. Default is 0.0.
    batch_limit : int, optional
        Maximum number of batches to train. Use -1 for no limit. Default is -1.

    Returns
    -------
    tuple[float, float]
        A tuple containing (average_loss, aux_loss_function_value).
    """
    assert (
        loss_function.reduction == "mean"
    ), "The loss function should be averaged over the batch"

    if device is None:
        device = global_device()

    # metrics meters
    loss_meter = AverageMeter()
    if aux_loss_function is not None:
        aux_loss_function.reset()
    # time meters
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    transfer_time_meter = AverageMeter()

    model.train()
    for i, (x, y) in enumerate(train_dataloader):
        start_time = time()
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
        ), f"During training of {model}, loss is NaN: {loss}, sample index: {i / len(train_dataloader)}"

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # update metrics
        loss_meter.update(loss.item(), x.size(0))
        if aux_loss_function is not None:
            aux_loss_function.update(y_pred.detach(), y)

        batch_time_meter.update(time() - start_time)

    if scheduler is not None:
        scheduler.epoch_step()

    aux_loss_function_value = (
        aux_loss_function() if aux_loss_function is not None else 0.0
    )
    if show:
        print(
            f"Train: loss={loss_meter.avg:.3e}, aux_loss_function={aux_loss_function_value:.2f}, time={batch_time_meter.avg:.2f}s"
        )
    return loss_meter.avg, aux_loss_function_value


def mixup_cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mixup_prob: float = 1.0,
    switch_prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup or CutMix to a batch, timm/Compact-Transformers style.

    Mirrors the batch-level behaviour of ``timm.data.Mixup`` as used by the
    Compact-Transformers ``cct_7_3x1_32`` recipe: with probability
    ``mixup_prob`` a mixing op is applied to the whole batch; when both Mixup
    and CutMix are enabled, CutMix is chosen with probability ``switch_prob``.
    The mixing coefficient ``lam`` is meant to be consumed by a two-term loss
    ``lam * loss(pred, y_a) + (1 - lam) * loss(pred, y_b)``, which is equivalent
    to ``SoftTargetCrossEntropy`` on the mixed (optionally label-smoothed)
    targets.

    Parameters
    ----------
    x : torch.Tensor
        Input batch of shape ``(B, C, H, W)``. Modified in place for CutMix.
    y : torch.Tensor
        Target labels of shape ``(B,)``.
    mixup_alpha : float, optional
        Beta distribution parameter for Mixup. ``0`` disables Mixup. Default 0.
    cutmix_alpha : float, optional
        Beta distribution parameter for CutMix. ``0`` disables CutMix. Default 0.
    mixup_prob : float, optional
        Probability of applying any mixing to the batch. Default 1.0.
    switch_prob : float, optional
        Probability of choosing CutMix over Mixup when both are enabled.
        Default 0.5.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
        ``(mixed_x, y_a, y_b, lam)``. When no mixing is applied, ``y_a == y_b``
        and ``lam == 1.0``.
    """
    mixup_on = mixup_alpha > 0.0
    cutmix_on = cutmix_alpha > 0.0
    if (not mixup_on and not cutmix_on) or np.random.rand() >= mixup_prob:
        return x, y, y, 1.0

    if mixup_on and cutmix_on:
        use_cutmix = np.random.rand() < switch_prob
    else:
        use_cutmix = cutmix_on

    indices = torch.randperm(x.size(0), device=x.device)
    y_shuffled = y[indices]

    if use_cutmix:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[indices, :, bbx1:bbx2, bby1:bby2]
        # Recompute lambda to match the exact replaced pixel ratio (as in timm).
        lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(2) * x.size(3)))
    else:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        x = lam * x + (1.0 - lam) * x[indices]

    return x, y, y_shuffled, lam


def gradient_descent_mixup(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_function: nn.Module,
    metrics: Metric | None = None,
    batch_limit: int | None = None,
    device: torch.device | None = None,
    scheduler_step_granularity: str = "epoch",
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mixup_prob: float = 1.0,
    mixup_switch_prob: float = 0.5,
    mixup_active: bool = True,
    amp_scaler: "torch.cuda.amp.GradScaler | None" = None,
    amp_enabled: bool = False,
) -> tuple[float, float]:
    """Train one epoch with optional Mixup/CutMix and mixed-precision (AMP).

    This mirrors :func:`gromo.utils.training_utils.gradient_descent` (same
    scheduler-stepping convention, ``batch_limit`` handling and return values)
    but adds the Mixup/CutMix and AMP features required to reproduce the
    Compact-Transformers ``cct_7_3x1_32`` training recipe. The gromo module is
    intentionally left untouched.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_dataloader : torch.utils.data.DataLoader
        The dataloader for training data.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    scheduler : torch.optim.lr_scheduler.LRScheduler | None
        Standard PyTorch scheduler, stepped according to
        ``scheduler_step_granularity``.
    loss_function : nn.Module
        Loss function with ``reduction="mean"`` (e.g. label-smoothing
        ``CrossEntropyLoss``).
    metrics : Metric | None, optional
        torchmetrics metric tracking accuracy against the dominant targets.
    batch_limit : int | None, optional
        Maximum number of batches to train. ``None`` for the full epoch.
    device : torch.device | None, optional
        Device to use. Default is ``global_device()``.
    scheduler_step_granularity : str, optional
        ``"epoch"`` (default) or ``"batch"``.
    mixup_alpha, cutmix_alpha : float, optional
        Beta parameters for Mixup/CutMix. ``0`` disables the corresponding op.
    mixup_prob : float, optional
        Probability of applying mixing to a batch. Default 1.0.
    mixup_switch_prob : float, optional
        Probability of CutMix over Mixup when both are enabled. Default 0.5.
    mixup_active : bool, optional
        When ``False``, no mixing is applied (e.g. after ``mixup_off_epoch``),
        while AMP still applies. Default True.
    amp_scaler : torch.cuda.amp.GradScaler | None, optional
        Gradient scaler used when ``amp_enabled`` is True. Created internally
        if needed.
    amp_enabled : bool, optional
        Whether to run the forward/backward pass under ``torch.autocast``.
        Default False.

    Returns
    -------
    tuple[float, float]
        ``(average_loss, metric_value)``.
    """
    assert (
        loss_function.reduction == "mean"
    ), "The loss function should be averaged over the batch"

    if device is None:
        device = global_device()

    use_mixup = mixup_active and (mixup_alpha > 0.0 or cutmix_alpha > 0.0)
    amp_device_type = device.type if device.type in ("cuda", "cpu") else "cpu"
    if amp_scaler is None:
        amp_scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled and device.type == "cuda"
        )

    loss_meter = AverageMeter()
    if metrics is not None:
        metrics.reset()
        metrics = metrics.to(device)

    model.train()
    for i, (x, y) in enumerate(train_dataloader):
        if batch_limit is not None and 0 <= batch_limit <= i:
            break

        x = x.to(device)
        y = y.to(device)

        if use_mixup:
            x, y_a, y_b, lam = mixup_cutmix_data(
                x,
                y,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                mixup_prob=mixup_prob,
                switch_prob=mixup_switch_prob,
            )
        else:
            y_a, y_b, lam = y, y, 1.0

        optimizer.zero_grad()

        with torch.autocast(
            device_type=amp_device_type,
            enabled=amp_enabled and device.type == "cuda",
        ):
            y_pred = model(x)
            if lam == 1.0:
                loss = loss_function(y_pred, y_a)
            else:
                loss = lam * loss_function(y_pred, y_a) + (1.0 - lam) * loss_function(
                    y_pred, y_b
                )
        assert (
            loss.isnan().sum() == 0
        ), f"During training of {model}, loss is NaN: {loss}, sample index: {i / len(train_dataloader)}"

        amp_scaler.scale(loss).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()
        if scheduler is not None and scheduler_step_granularity == "batch":
            scheduler.step()

        loss_meter.update(loss.detach(), x.size(0))
        if metrics is not None:
            metrics.update(y_pred.detach(), y_a)

    if scheduler is not None and scheduler_step_granularity == "epoch":
        scheduler.step()

    metric_value = metrics.compute().item() if metrics is not None else 0.0
    return loss_meter.avg, metric_value


@deprecated("This functionality is already in gromo module")
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    aux_loss_function: MetricTracker | None = None,
    batch_limit: int = -1,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """
    Evaluate the model on a dataloader.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    dataloader : torch.utils.data.DataLoader
        The dataloader for evaluation data.
    loss_function : nn.Module
        The loss function to use. Must have reduction="mean".
    aux_loss_function : MetricTracker | None, optional
        A MetricTracker instance to track auxiliary metrics (e.g., accuracy).
        Will be reset at the start and updated each batch. Default is None.
    batch_limit : int, optional
        Maximum number of batches to evaluate. Use -1 for no limit. Default is -1.
    device : torch.device | None, optional
        Device to use. Default is None (uses global_device()).

    Returns
    -------
    tuple[float, float]
        A tuple containing (average_loss, aux_loss_function_value).
    """
    assert (
        loss_function.reduction == "mean"
    ), "The loss function should be averaged over the batch"

    if device is None:
        device = global_device()

    # metrics meters
    loss_meter = AverageMeter()
    if aux_loss_function is not None:
        aux_loss_function.reset()

    model.eval()
    for i, (x, y) in enumerate(dataloader):
        if 0 <= batch_limit <= i:
            break
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        loss_meter.update(loss.item(), x.size(0))
        if aux_loss_function is not None:
            aux_loss_function.update(y_pred, y)

    aux_loss_function_value = (
        aux_loss_function() if aux_loss_function is not None else 0.0
    )
    return loss_meter.avg, aux_loss_function_value


@deprecated("This functionality is already in gromo module")
@torch.no_grad()
def extended_evaluate_model(
    growing_model: GrowingContainer,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    aux_loss_function: MetricTracker | None = None,
    dataloader_seed: int | None = None,
    batch_limit: int = -1,
    mask: dict = {},
    device: torch.device | None = None,
) -> tuple[float, float]:
    """
    Evaluate a growing model on a dataloader with optional masking and reseeding.

    This function evaluates a GrowingContainer model on a provided dataloader,
    optionally using a mask and reseeding the dataloader for reproducibility.
    The loss function must use reduction="sum" since this function normalizes
    the loss by batch size.

    Parameters
    ----------
    growing_model : GrowingContainer
        The growing model to evaluate.
    dataloader : torch.utils.data.DataLoader
        DataLoader containing the evaluation data.
    loss_function : nn.Module
        Loss function to use for evaluation. Must have reduction="sum".
    aux_loss_function : MetricTracker | None, optional
        A MetricTracker instance to track auxiliary metrics (e.g., accuracy).
        Will be reset at the start and updated each batch. Default is None.
    dataloader_seed : int | None, optional
        Seed for reseeding the dataloader generator for reproducibility. If None, no reseeding is done.
        Default is None.
    batch_limit : int, optional
        Maximum number of batches to evaluate. Use -1 for no limit.
        Default is -1.
    mask : dict, optional
        Retained for API compatibility. Ignored to preserve the historical
        pipeline evaluation path based on ``extended_forward(x)``.
        Default is {}.
    device : torch.device | None, optional
        Device to perform computations on. If None, uses global_device().
        Default is None.
    Returns
    -------
    tuple[float, float]
        A tuple containing:
        - loss (float): Average loss across all evaluated samples.
        - aux_loss_function (float): The computed auxiliary metric value.

    Raises
    ------
    AssertionError
        If loss_function.reduction is not "sum".
        If dataloader_seed is provided but dataloader has no generator.
        If dataloader.generator is not a torch.Generator.

    Notes
    -----
    - The loss is normalized by dividing the sum by the batch size.
    - The historical evaluation path calls ``extended_forward(x)`` without
      forwarding ``mask`` to preserve the original growth dynamics.
    - If dataloader_seed is provided, the dataloader's generator is reseeded
      for consistent evaluation across multiple runs.

    Examples
    --------
    >>> from tools.metrics import TopKAccuracy
    >>> loss, accuracy = extended_evaluate_model(
    ...     growing_model=model,
    ...     dataloader=val_loader,
    ...     loss_function=nn.CrossEntropyLoss(reduction="sum"),
    ...     aux_loss_function=TopKAccuracy(k=1),
    ...     batch_limit=100
    ... )
    """
    assert (
        loss_function.reduction == "sum"
    ), "The loss function should not be averaged over the batch"

    if device is None:
        device = global_device()

    if dataloader_seed is not None:
        # Reseed dataloader for reproducibility
        assert hasattr(dataloader, "generator"), "Dataloader has no generator to reseed"
        assert isinstance(
            dataloader.generator, torch.Generator
        ), "Dataloader generator is not a torch.Generator"
        dataloader.generator.manual_seed(dataloader_seed)

    growing_model.eval()
    loss_meter = AverageMeter()
    if aux_loss_function is not None:
        aux_loss_function.reset()

    for i, (x, y) in enumerate(dataloader):
        if 0 <= batch_limit <= i:
            break
        x, y = x.to(device), y.to(device)
        prediction = growing_model.extended_forward(x)  # type: ignore[misc]
        if isinstance(prediction, tuple):
            y_pred = prediction[0]
        else:
            y_pred = prediction
        assert isinstance(y_pred, torch.Tensor)
        loss = loss_function(y_pred, y)
        loss_meter.update(loss.item() / x.size(0), x.size(0))
        if aux_loss_function is not None:
            aux_loss_function.update(y_pred, y)

    aux_loss_function_value = (
        aux_loss_function() if aux_loss_function is not None else 0.0
    )
    return loss_meter.avg, aux_loss_function_value


@deprecated("This functionality is already in gromo module")
def compute_statistics(
    growing_model: GrowingContainer,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module = nn.MSELoss(reduction="sum"),
    aux_loss_function: MetricTracker | None = None,
    batch_limit: int = 1_000_000,
    device: torch.device | None = None,
    show: bool = False,
) -> tuple[float, float]:
    """
    Compute the tensor of statistics of the model on the dataloader
    with a limit of batch_limit batches.

    Parameters
    ----------
    growing_model : GrowingContainer
        The model to evaluate.
    dataloader : torch.utils.data.DataLoader
        The dataloader to use.
    loss_function : nn.Module
        The loss function to use. Must have reduction="sum".
    aux_loss_function : MetricTracker | None, optional
        A MetricTracker instance to track auxiliary metrics (e.g., accuracy).
        Will be reset at the start and updated each batch. Default is None.
    batch_limit : int, optional
        The maximum number of batches to use. Default is 1_000_000.
    device : torch.device | None, optional
        The device to use. Default is None (uses global_device()).
    show : bool, optional
        If True, display a progress bar. Default is False.

    Returns
    -------
    tuple[float, float]
        A tuple containing (average_loss, aux_loss_function_value).
    """
    assert (
        loss_function.reduction == "sum"
    ), "The loss function should not be averaged over the batch"

    if device is None:
        device = global_device()

    growing_model.init_computation()
    growing_model.eval()
    loss_meter = AverageMeter()
    if aux_loss_function is not None:
        aux_loss_function.reset()

    if show:
        warn(
            DeprecationWarning(
                "The 'show' parameter is deprecated and will be"
                " removed in a future version."
            )
        )
        dataloader = tqdm(dataloader)  # type: ignore

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
            aux_loss_function.update(y_pred.detach(), y)

    aux_loss_function_value = (
        aux_loss_function() if aux_loss_function is not None else 0.0
    )
    return loss_meter.avg, aux_loss_function_value


def line_search(
    model: GrowingContainer,
    dataloader: torch.utils.data.DataLoader,
    dataloader_seed: int | None = None,
    loss_function: nn.Module = nn.MSELoss(reduction="mean"),
    aux_loss_function: Metric | None = None,
    batch_limit: int = -1,
    mask: dict = {},
    initial_loss: float | None = None,
    initial_aux_loss: float | None = None,
    first_order_improvement: float | torch.Tensor = 1,
    alpha: float = 0.1,
    beta: float = 0.5,
    t0: float | None = None,
    max_gamma: float | None = None,
    extended_search: bool = True,
    max_iter: int = 100,
    epsilon: float = 1e-7,
    verbose: bool = False,
    device: torch.device | None = None,
) -> tuple[float, float, float, list[float], list[float], list[float]]:
    """
    Perform line search to find optimal scaling factor for the currently updated layer.

    This function performs a backtracking line search with Armijo condition to find
    the optimal scaling factor (gamma) for the newly added neurons in the currently
    updated layer. The search operates on the square root of gamma (t) for numerical
    stability, where gamma = t^2.

    Parameters
    ----------
    model : GrowingContainer
        The growing model with a currently updated layer that has a scaling_factor attribute.
    dataloader : torch.utils.data.DataLoader
        DataLoader to evaluate the model on during the line search.
    dataloader_seed : int | None, optional
        Seed for shuffling the dataloader during evaluation. If None, no reseeding is done.
        Default is None.
    loss_function : nn.Module, optional
        Loss function to minimize. Must use reduction="mean": evaluation goes
        through gromo's evaluate_model, whose AverageMeter does not normalize by
        batch size and asserts reduction == "mean". A mean loss keeps the same
        per-sample-mean scale as initial_loss and first_order_improvement.
        Default is nn.MSELoss(reduction="mean").
    aux_loss_function : Metric | None, optional
        A Metric instance to track auxiliary metrics (e.g., accuracy).
        Default is None.
    batch_limit : int, optional
        Maximum number of batches to use for evaluation. Use -1 for no limit.
        Default is -1.
    initial_loss : float | None, optional
        Initial loss at gamma=0. If None, it will be computed.
        Default is None.
    initial_aux_loss : float | None, optional
        Initial auxiliary loss at gamma=0. If None, will be replaced by 0.
        Default is None.
    first_order_improvement : float | torch.Tensor, optional
        Expected first-order improvement in loss. Used to initialize the search
        and for the Armijo condition.
        Default is 1.
    alpha : float, optional
        Armijo condition parameter. The sufficient decrease condition is:
        loss < initial_loss - alpha * gamma * first_order_improvement.
        Default is 0.1.
    beta : float, optional
        Step size reduction factor. The actual factor applied is sqrt(beta) since
        we work with t = sqrt(gamma). Typical values: 0.5 for aggressive, 0.8 for conservative.
        Default is 0.5.
    t0 : float | None, optional
        Initial value for t (sqrt of initial gamma). If None, computed as:
        t0 = sqrt(2 * initial_loss / first_order_improvement).
        Default is None.
    max_gamma : float | None, optional
        Maximum allowable gamma value. If None, no maximum is enforced.
    extended_search : bool, optional
        If True, extends the search in both directions (increase and decrease gamma)
        to find a local minimum. If False, only performs backtracking.
        Default is True.
    max_iter : int, optional
        Maximum number of iterations for the line search.
        Default is 100.
    epsilon : float, optional
        Minimum value for t (sqrt of gamma) below which search stops.
        Actual threshold is sqrt(epsilon).
        Default is 1e-7.
    verbose : bool, optional
        If True, prints detailed information about each tested gamma value.
        Default is False.
    device : torch.device | None, optional
        Device to perform computations on. If None, uses global_device().
        Default is None.
    Returns
    -------
    tuple[float, float, float, list[float], list[float], list[float]]
        A tuple containing:
        - gamma (float): The optimal scaling factor (t^2).
        - loss (float): The loss at the optimal gamma.
        - aux_loss (float): The auxiliary loss at the optimal gamma.
        - gammas (list[float]): List of all tested gamma values.
        - losses (list[float]): List of losses corresponding to tested gammas.
        - aux_losses (list[float]): List of auxiliary losses corresponding to tested gammas.

    Raises
    ------
    AssertionError
        If model.currently_updated_layer is None.

    Notes
    -----
    - The function works with t = sqrt(gamma) for numerical stability.
    - The Armijo condition ensures sufficient decrease in the loss function.
    - The extended_search option allows finding better local minima by exploring
      beyond the first acceptable point.
    - The scaling_factor of the currently_updated_layer is set to the optimal
      value (t, not gamma) at the end of the function.

    Examples
    --------
    >>> from tools.metrics import TopKAccuracy
    >>> gamma, loss, aux_loss, gammas, losses, aux_losses = line_search(
    ...     model=growing_model,
    ...     dataloader=train_loader,
    ...     loss_function=nn.CrossEntropyLoss(reduction="mean"),
    ...     aux_loss_function=TopKAccuracy(k=1),
    ...     initial_loss=100.0,
    ...     first_order_improvement=5.0,
    ...     alpha=0.1,
    ...     beta=0.5,
    ...     verbose=True
    ... )
    """
    logger = logging.getLogger(__name__)
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
        model.currently_updated_layer.set_scaling_factor(sqrt_gamma)
        loss, aux_loss = gromo_evaluate_model(
            model=model,
            dataloader=dataloader,
            use_extended_model=True,
            dataloader_seed=dataloader_seed,
            loss_function=loss_function,
            metrics=aux_loss_function,
            batch_limit=batch_limit,
            mask=mask,
            device=device,
        )
        gammas.append(sqrt_gamma**2)
        losses.append(loss)
        aux_losses.append(aux_loss)
        if verbose:
            logger.info(
                f"gamma n° {len(gammas)}: {sqrt_gamma**2:.3e} -> Loss: {loss:.6e} (aux_loss: {aux_loss * 100:.2f}%)"
            )
        return loss, aux_loss

    if initial_loss is None:
        logger.warning("Initial loss is not provided, computing it")
        initial_loss, initial_aux_loss = test_gamma(0.0)
        logger.info(
            f"Initial loss: {initial_loss:.3e} (aux_loss: {initial_aux_loss * 100:.2f}%)"
        )
    gammas.append(0.0)
    losses.append(initial_loss)
    initial_aux_loss = initial_aux_loss if initial_aux_loss is not None else 0.0
    aux_losses.append(initial_aux_loss)
    if verbose:
        logger.info(
            f"gamma n° {len(gammas)}: {0.0:.3e} -> Loss: {initial_loss:.3e} (aux_loss: {initial_aux_loss * 100:.2f}%)"
        )

    def under_bound(sqrt_gamma: float, loss: float):
        return loss < initial_loss - alpha * sqrt_gamma**2 * first_order_improvement

    # gamma = t ** 2
    if t0 is None:
        t = np.sqrt(2 * (initial_loss / first_order_improvement))
    else:
        t = np.sqrt(t0)
    if max_gamma is not None:
        max_t = np.sqrt(max_gamma)
        t = min(t, max_t)
    else:
        max_t = np.inf
    l0, l0_aux = test_gamma(t)
    l1, l1_aux = l0, l0_aux
    i = 0
    if under_bound(t, l0):
        if extended_search:
            go = t / beta < max_t
            while go:
                l0, l0_aux = l1, l1_aux
                t /= beta
                l1, l1_aux = test_gamma(t)
                go = l1 < l0 and i < max_iter and t < max_t
                i += 1
            t *= beta
        model.currently_updated_layer.set_scaling_factor(t)
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
        model.currently_updated_layer.set_scaling_factor(t)

    # select best gamma found
    min_loss = float("inf")
    best_idx = -1
    for idx, loss in enumerate(losses):
        if loss < min_loss:
            min_loss = loss
            best_idx = idx
    t = np.sqrt(gammas[best_idx])
    model.currently_updated_layer.set_scaling_factor(t)
    l0 = losses[best_idx]
    l0_aux = aux_losses[best_idx]

    if verbose:
        logger.info(
            f"Line search completed: optimal gamma = {t**2:.3e}, loss = {l0:.3e}, aux_loss = {l0_aux * 100:.2f}%"
        )

    # select best gamma found
    min_loss = float("inf")
    best_idx = -1
    for idx, loss in enumerate(losses):
        if loss < min_loss:
            min_loss = loss
            best_idx = idx
    t = np.sqrt(gammas[best_idx])
    model.currently_updated_layer.scaling_factor = t
    l0 = losses[best_idx]
    l0_aux = aux_losses[best_idx]

    if verbose:
        logger.info(
            f"Line search completed: optimal gamma = {t**2:.3e}, loss = {l0:.3e}, aux_loss = {l0_aux * 100:.2f}%"
        )

    return t**2, l0, l0_aux, gammas, losses, aux_losses


def full_search(
    model: GrowingContainer,
    loss: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    seed_dataloader: int | None = None,
    batch_limit: int = 1_000_000,
    initial_loss: float | None = None,
    first_order_improvement: float = 1,
    min_value: float = -100,
    max_value: float = 100,
    nb_points: int = 100,
):
    xs = np.linspace(min_value, max_value, nb_points)
    values = []
    for v in tqdm(xs):
        model.set_scaling_factor(np.sign(v) * np.sqrt(abs(v)))
        values.append(
            gromo_evaluate_model(
                model=model,
                loss_function=loss,
                dataloader=dataloader,
                dataloader_seed=seed_dataloader,
                batch_limit=batch_limit,
                use_extended_model=True,
            )
        )

    return xs, values
