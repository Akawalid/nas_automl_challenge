import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from gromo.containers.growing_block import GrowingBlock
from gromo.containers.growing_container import GrowingContainer
from gromo.modules.growing_module import GrowingModule
from gromo.utils.training_utils import evaluate_model
from gromo.utils.utils import global_device
from torchmetrics import Metric

SetterT = Callable[[GrowingContainer, float], None]


def _target_module(currently_updated_layer) -> GrowingModule:
    """
    Resolve the GrowingModule that owns the per-component scaling factors.

    For a `GrowingBlock` the scaling factors live on its `second_layer`; for a
    bare `GrowingModule` the layer itself is the target.
    """
    if isinstance(currently_updated_layer, GrowingBlock):
        return currently_updated_layer.second_layer
    return currently_updated_layer


def default_setter(model: GrowingContainer, sqrt_gamma: float) -> None:
    """
    Legacy joint setter used by standard FoGro.

    Drives the three sub-factors via the historical coupling
    (``optimal_delta_scaling = sqrt_gamma**2``,
    ``input_extension_scaling = sqrt_gamma``,
    ``previous_module.output_extension_scaling = sqrt_gamma``).
    """
    model.currently_updated_layer.set_scaling_factor(sqrt_gamma)


def set_extensions_only(model: GrowingContainer, sqrt_gamma: float) -> None:
    """
    Drive both extension scalings together while leaving the delta untouched.

    Use this when the caller has pre-set ``optimal_delta_scaling`` (e.g. to a
    fixed multiple of the learning rate) and wants the line search to vary
    only the input/output extension product.

    Sets ``input_extension_scaling = sqrt_gamma`` and
    ``output_extension_scaling = sqrt_gamma`` so the *effective extension
    product* equals ``gamma = sqrt_gamma ** 2`` — matching the
    Armijo / quadratic-model assumptions of `line_search`.
    """
    layer = _target_module(model.currently_updated_layer)
    layer.input_extension_scaling = float(sqrt_gamma)
    if isinstance(layer.previous_module, GrowingModule):
        layer.previous_module.output_extension_scaling = float(sqrt_gamma)


def set_input_extension_only(model: GrowingContainer, sqrt_gamma: float) -> None:
    """
    Drive the input extension only — output extension and delta are left
    at whatever values the caller pre-set.

    Sets ``input_extension_scaling = sqrt_gamma ** 2`` so that when the
    caller has pinned ``previous_module.output_extension_scaling = 1`` the
    effective extension product is ``gamma = sqrt_gamma ** 2``, again
    matching the Armijo math.
    """
    layer = _target_module(model.currently_updated_layer)
    layer.input_extension_scaling = float(sqrt_gamma) ** 2


def line_search(
    model: GrowingContainer,
    dataloader: torch.utils.data.DataLoader,
    dataloader_seed: int | None = None,
    loss_function: nn.Module = nn.MSELoss(reduction="sum"),
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
    setter: SetterT = default_setter,
) -> tuple[float, float, float, list[float], list[float], list[float]]:
    """
    Perform line search to find an optimal scaling factor for the currently
    updated layer.

    Backtracking line search with the Armijo condition. The search variable
    is ``t`` with ``gamma = t ** 2``; the meaning of ``gamma`` is decided by
    *setter*, which is called as ``setter(model, t)`` to install the current
    test value before each evaluation.

    Parameters
    ----------
    model : GrowingContainer
        The growing model whose currently updated layer is being scaled.
    dataloader : torch.utils.data.DataLoader
        DataLoader to evaluate the model on during the line search.
    dataloader_seed : int | None, optional
        Seed for shuffling the dataloader during evaluation. If None, no
        reseeding is done.
    loss_function : nn.Module, optional
        Loss function to minimize (use ``reduction="sum"``).
    aux_loss_function : Metric | None, optional
        A Metric instance to track auxiliary metrics (e.g. accuracy).
    batch_limit : int, optional
        Maximum number of batches to use for evaluation. ``-1`` for no limit.
    initial_loss : float | None, optional
        Initial loss at ``gamma=0``. If None, it is computed.
    initial_aux_loss : float | None, optional
        Initial auxiliary loss at ``gamma=0``. If None, replaced by 0.
    first_order_improvement : float | torch.Tensor, optional
        Expected first-order improvement in loss for a unit ``gamma``. The
        right value depends on which scalings *setter* drives — pick the one
        that matches ``-d(loss) / d(gamma) | gamma=0`` for the chosen setter:
        - joint (default setter): ``model.currently_updated_layer.first_order_improvement``.
        - extensions-only (`set_extensions_only`,
          `set_input_extension_only`): a quantity proportional to
          ``activation_gradient * (eigenvalues_extension ** 2).sum()``.
    alpha : float, optional
        Armijo condition parameter. Sufficient decrease is
        ``loss < initial_loss - alpha * gamma * first_order_improvement``.
    beta : float, optional
        Step size reduction factor. Internally we work in ``t``, so the
        applied factor is ``sqrt(beta)``.
    t0 : float | None, optional
        Initial value for ``t``. If None,
        ``t0 = sqrt(2 * initial_loss / first_order_improvement)``.
    max_gamma : float | None, optional
        Maximum allowable ``gamma``.
    extended_search : bool, optional
        If True, extends the search in both directions to find a local
        minimum. If False, only performs backtracking.
    max_iter : int, optional
        Maximum iterations.
    epsilon : float, optional
        Lower bound on ``gamma``; the search stops when ``t < sqrt(epsilon)``.
    verbose : bool, optional
        If True, log every tested ``gamma`` value.
    device : torch.device | None, optional
        Device to evaluate on. Defaults to ``global_device()``.
    setter : Callable[[GrowingContainer, float], None], optional
        Function called as ``setter(model, t)`` to install the current
        ``t = sqrt(gamma)`` on the model. Defaults to the legacy joint
        setter that fans ``t`` out to all three sub-factors with the
        ``**2`` coupling on the delta. Pre-defined alternatives:
        `set_extensions_only`, `set_input_extension_only`. Custom setters
        are responsible for choosing values such that the loss is locally
        quadratic in ``t`` (linear in ``gamma``) — otherwise the Armijo
        bound and ``first_order_improvement`` interpretation break.

    Returns
    -------
    tuple[float, float, float, list[float], list[float], list[float]]
        - gamma (float): optimal ``gamma = t**2``.
        - loss (float): loss at the optimal ``gamma``.
        - aux_loss (float): auxiliary loss at the optimal ``gamma``.
        - gammas (list[float]): tested ``gamma`` values.
        - losses (list[float]): losses for each tested value.
        - aux_losses (list[float]): aux losses for each tested value.

    Notes
    -----
    The convention ``gamma = t ** 2`` is part of the line-search contract
    and is shared with all setters; do not pass a setter that uses a
    different relation.

    Examples
    --------
    >>> from tools.metrics import TopKAccuracy
    >>> gamma, loss, aux_loss, gammas, losses, aux_losses = line_search(
    ...     model=growing_model,
    ...     dataloader=train_loader,
    ...     loss_function=nn.CrossEntropyLoss(reduction="sum"),
    ...     aux_loss_function=TopKAccuracy(k=1),
    ...     initial_loss=100.0,
    ...     first_order_improvement=5.0,
    ...     setter=set_extensions_only,
    ...     alpha=0.1,
    ...     beta=0.5,
    ...     verbose=True,
    ... )
    """
    logger = logging.getLogger(__name__)
    assert model.currently_updated_layer is not None, "No currently updated layer"

    if device is None:
        device = global_device()

    gammas: list[float] = []
    losses: list[float] = []
    aux_losses: list[float] = []
    beta = float(np.sqrt(beta))
    epsilon = float(np.sqrt(epsilon))
    if isinstance(first_order_improvement, torch.Tensor):
        first_order_improvement = first_order_improvement.item()
    if isinstance(initial_loss, torch.Tensor):
        initial_loss = initial_loss.item()

    def test_gamma(sqrt_gamma: float) -> tuple[float, float]:
        setter(model, sqrt_gamma)
        loss, aux_loss = evaluate_model(
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
                f"gamma n° {len(gammas)}: {sqrt_gamma**2:.3e} -> "
                f"Loss: {loss:.6e} (aux_loss: {aux_loss * 100:.2f}%)"
            )
        return loss, aux_loss

    if initial_loss is None:
        logger.warning("Initial loss is not provided, computing it")
        initial_loss, initial_aux_loss = test_gamma(0.0)
        logger.info(
            f"Initial loss: {initial_loss:.3e} (aux_loss: {initial_aux_loss * 100:.2f}%)"
        )
    else:
        gammas.append(0.0)
        losses.append(initial_loss)
        initial_aux_loss = initial_aux_loss if initial_aux_loss is not None else 0.0
        aux_losses.append(initial_aux_loss)
        if verbose:
            logger.info(
                f"gamma n° {len(gammas)}: {0.0:.3e} -> Loss: {initial_loss:.3e} (aux_loss: {initial_aux_loss * 100:.2f}%)"
            )

    def under_bound(sqrt_gamma: float, loss: float) -> bool:
        return loss < initial_loss - alpha * sqrt_gamma**2 * first_order_improvement

    # gamma = t ** 2
    if t0 is None:
        t = float(np.sqrt(2 * (initial_loss / first_order_improvement)))
    else:
        t = float(np.sqrt(t0))
    if max_gamma is not None:
        max_t = float(np.sqrt(max_gamma))
        t = min(t, max_t)
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

    # Select the best gamma observed and install it via the same setter.
    best_idx = int(np.argmin(losses))
    best_t = float(np.sqrt(gammas[best_idx]))
    setter(model, best_t)
    best_loss = losses[best_idx]
    best_aux = aux_losses[best_idx]

    if verbose:
        logger.info(
            f"Line search completed: optimal gamma = {best_t**2:.3e}, "
            f"loss = {best_loss:.3e}, aux_loss = {best_aux * 100:.2f}%"
        )

    return best_t**2, best_loss, best_aux, gammas, losses, aux_losses
