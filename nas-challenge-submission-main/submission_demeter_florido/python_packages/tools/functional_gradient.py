import itertools
import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch


FunctionalDataBatches = Iterable[tuple[torch.Tensor, torch.Tensor]]


def prepare_fixed_train_probe(
    dataloader: torch.utils.data.DataLoader,
    num_batches: int = 4,
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor]],
    FunctionalDataBatches,
]:
    """Freeze a train probe without changing the first training epoch.

    The returned epoch iterable puts the prefetched batches back before the
    remainder of the same loader iterator. The optimizer therefore observes
    the exact original samples, ordering and augmentations once each.
    """
    if num_batches < 1:
        raise ValueError("A functional train probe needs at least one batch.")

    train_iterator = iter(dataloader)
    prefetched_batches = []
    fixed_probe_batches = []
    for _ in range(min(num_batches, len(dataloader))):
        try:
            inputs, targets = next(train_iterator)
        except StopIteration:
            break
        prefetched_batches.append((inputs, targets))
        fixed_probe_batches.append(
            (
                inputs.detach().cpu().clone(),
                targets.detach().cpu().clone(),
            )
        )

    if not fixed_probe_batches:
        raise ValueError(
            "Cannot build functional train diagnostics from an empty loader."
        )
    epoch_batches = itertools.chain(prefetched_batches, train_iterator)
    return fixed_probe_batches, epoch_batches


@dataclass(frozen=True)
class OutputFunctionalMetrics:
    gradient_squared_l2_norm: float
    update_squared_l2_norm: float | None
    scale_optimal_learning_rate: float | None
    approximation_l2_distance: float | None
    relative_error_approximation_denominator: float | None
    relative_error_gradient_denominator: float | None
    directional_cosine: float | None
    output_snapshot: torch.Tensor


def capture_network_outputs(
    model: torch.nn.Module,
    dataloader: FunctionalDataBatches,
    device: torch.device | str,
) -> torch.Tensor:
    """Capture network outputs on a fixed dataset without retaining a graph."""
    was_training = model.training
    outputs = []
    model.eval()
    try:
        with torch.no_grad():
            for inputs, _ in dataloader:
                outputs.append(model(inputs.to(device)).detach().cpu())
    finally:
        model.train(was_training)

    if not outputs:
        raise ValueError("Cannot capture network outputs from an empty dataloader.")
    return torch.cat(outputs)


def measure_output_functional_metrics(
    model: torch.nn.Module,
    dataloader: FunctionalDataBatches,
    loss_fn: torch.nn.Module,
    device: torch.device | str,
    reference_outputs: torch.Tensor | None = None,
) -> OutputFunctionalMetrics:
    """Measure ``(1/N) sum_i ||d loss_i / d f(x_i)||_2^2`` on fixed data.

    This is the squared empirical L2 norm of the functional gradient used in
    the paper. A summed batch loss recovers each sample's output gradient;
    normalizing once by the total sample count makes the result independent of
    how the same dataset is partitioned into batches. When outputs captured
    before training are provided, the gradient is measured at that pre-update
    function and the realized finite update ``f_before - f_after`` is measured
    at the same time. The realized approximation is calibrated as
    ``g = (f_before - f_after) / eta_star``, where
    ``eta_star = <f_before - f_after, grad L(f_before)> / ||grad L(f_before)||^2``.
    This yields the paper's relative error (with ``||g||`` in the denominator),
    its gradient-denominator counterpart, their L2 distance, and cosine
    similarity.
    """
    if getattr(loss_fn, "reduction", None) != "sum":
        raise ValueError(
            "The functional-gradient loss must use reduction='sum' so the "
            "measurement is independent of batch size."
        )

    was_training = model.training
    gradient_squared_norm = None
    update_squared_norm = None
    update_gradient_dot_product = None
    sample_count = 0
    current_outputs = []
    model.eval()
    try:
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            # The network is needed only to evaluate f(x). Detaching here
            # avoids retaining its internal graph: differentiation starts and
            # stops at the output, exactly where the functional lives.
            with torch.no_grad():
                output = model(inputs)
            current_outputs.append(output.detach().cpu())

            batch_size = output.shape[0]
            reference = None
            if reference_outputs is not None:
                reference = reference_outputs[
                    sample_count : sample_count + batch_size
                ].to(device=device, dtype=output.dtype)
                if reference.shape != output.shape:
                    raise ValueError(
                        "Reference and current network outputs have different shapes."
                    )

            # The paper compares the realized update with grad L(f_t), so when
            # a pre-training snapshot is available the gradient must be
            # evaluated at that same pre-update function f_t.
            gradient_point = reference if reference is not None else output
            with torch.enable_grad():
                gradient_point = gradient_point.detach().requires_grad_(True)
                loss = loss_fn(gradient_point, targets)
                (functional_gradient,) = torch.autograd.grad(loss, gradient_point)
            batch_squared_norm = (
                functional_gradient.detach().to(dtype=torch.float64).square().sum()
            )
            if gradient_squared_norm is None:
                gradient_squared_norm = batch_squared_norm
            else:
                gradient_squared_norm += batch_squared_norm

            if reference is not None:
                functional_update = reference - output.detach()
                batch_update_squared_norm = (
                    functional_update.to(dtype=torch.float64).square().sum()
                )
                batch_dot_product = torch.sum(
                    functional_update.to(dtype=torch.float64)
                    * functional_gradient.detach().to(dtype=torch.float64)
                )
                if update_squared_norm is None:
                    update_squared_norm = batch_update_squared_norm
                    update_gradient_dot_product = batch_dot_product
                else:
                    update_squared_norm += batch_update_squared_norm
                    assert update_gradient_dot_product is not None
                    update_gradient_dot_product += batch_dot_product
            sample_count += output.shape[0]
    finally:
        model.train(was_training)

    if gradient_squared_norm is None or sample_count == 0:
        raise ValueError(
            "Cannot measure the output functional gradient on an empty dataloader."
        )
    if reference_outputs is not None and len(reference_outputs) != sample_count:
        raise ValueError(
            "Reference outputs and functional-gradient dataset have different sizes."
        )

    gradient_squared_l2_norm = gradient_squared_norm.item() / sample_count
    update_squared_l2_norm = (
        update_squared_norm.item() / sample_count
        if update_squared_norm is not None
        else None
    )
    normalized_dot_product = (
        update_gradient_dot_product.item() / sample_count
        if update_gradient_dot_product is not None
        else None
    )

    eps = torch.finfo(torch.float64).eps
    directional_cosine = None
    scale_optimal_learning_rate = None
    approximation_l2_distance = None
    relative_error_approximation_denominator = None
    relative_error_gradient_denominator = None
    if (
        update_squared_l2_norm is not None
        and normalized_dot_product is not None
        and update_squared_l2_norm > eps
        and gradient_squared_l2_norm > eps
    ):
        directional_cosine = normalized_dot_product / math.sqrt(
            update_squared_l2_norm * gradient_squared_l2_norm
        )
        directional_cosine = max(-1.0, min(1.0, directional_cosine))

        candidate_learning_rate = (
            normalized_dot_product / gradient_squared_l2_norm
        )
        # Keep the signed projection scale observable even when it violates
        # the paper's positive-learning-rate assumption. This distinguishes
        # a tiny descent step from a zero or anti-aligned step in the logs.
        scale_optimal_learning_rate = candidate_learning_rate
        # The paper assumes a positive functional learning rate. A zero or
        # negative value means the realized optimizer step is not a valid
        # descent-oriented functional-gradient approximation.
        if candidate_learning_rate > eps:
            approximation_squared_l2_norm = (
                update_squared_l2_norm / candidate_learning_rate**2
            )
            approximation_error_squared_l2_norm = max(
                0.0,
                approximation_squared_l2_norm
                - 2.0 * normalized_dot_product / candidate_learning_rate
                + gradient_squared_l2_norm,
            )
            approximation_l2_distance = math.sqrt(
                approximation_error_squared_l2_norm
            )
            relative_error_approximation_denominator = (
                approximation_l2_distance
                / math.sqrt(approximation_squared_l2_norm)
            )
            relative_error_gradient_denominator = (
                approximation_l2_distance
                / math.sqrt(gradient_squared_l2_norm)
            )

    return OutputFunctionalMetrics(
        gradient_squared_l2_norm=gradient_squared_l2_norm,
        update_squared_l2_norm=update_squared_l2_norm,
        scale_optimal_learning_rate=scale_optimal_learning_rate,
        approximation_l2_distance=approximation_l2_distance,
        relative_error_approximation_denominator=(
            relative_error_approximation_denominator
        ),
        relative_error_gradient_denominator=relative_error_gradient_denominator,
        directional_cosine=directional_cosine,
        output_snapshot=torch.cat(current_outputs),
    )
