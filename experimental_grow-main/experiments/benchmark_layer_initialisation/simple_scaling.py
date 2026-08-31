"""
When solving a problem of the form
    min_gamma || gamma * f2(f1(x)) - y ||_2

we can find gamma with two evaluations of the objective function if
we already know the result for gamma=0.
"""

from typing import Any, Callable, Iterator

import torch
from torch import nn


def line_search(
    layer1: nn.Module,
    layer2: nn.Module,
    data_iterator: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]],
    zero_value: float | None = None,
    device: torch.device = torch.device("cpu"),
) -> float:
    r"""
    Find the scalar :math:`\gamma^{\star}` minimising

        J(\gamma) = \sum_b \| \gamma\,f_2(f_1(x_b)) - y_b \|_2^2 .

    The closed‑form minimiser is

        γ* = (C − B) / ( 2 · (B + C − 2·A) )

    with
        A = J(0), B = J(+1), C = J(−1).

    If ``zero_value`` is *None* we compute **A** inside the loop; otherwise the
    supplied value is used and we save those FLOPS.

    Parameters
    ----------
    layer1, layer2 : nn.Module
        Consecutive layers that define ``z = f2(f1(x))``.
    data_iterator : () -> iterator of (x, y)
        Provides mini‑batches for evaluation.
    zero_value : float | None
        Pre‑computed J(0) (sum of ‖y‖² over the data).  Pass ``None`` to let
        the routine compute it automatically.
    device : torch.device
        CPU or CUDA device for the forward passes.

    Returns
    -------
    float
        Optimal scaling factor γ*; if the quadratic is degenerate we return 0.
    """
    layer1.eval()
    layer2.eval()

    loss_plus, loss_minus = 0.0, 0.0  # B and C
    loss_zero = 0.0  # A (only if needed)

    with torch.no_grad():
        for xb, yb in data_iterator():
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            zb = layer2(layer1(xb))  # f₂(f₁(x))

            diff_plus = zb - yb  # γ = +1
            diff_minus = -zb - yb  # γ = −1

            loss_plus += torch.sum(diff_plus**2).item()
            loss_minus += torch.sum(diff_minus**2).item()

            if zero_value is None:  # need A
                loss_zero += torch.sum(yb**2).item()

    zero_value = loss_zero if zero_value is None else zero_value

    denominator = 2.0 * (loss_plus + loss_minus - 2.0 * zero_value)
    if abs(denominator) < 1e-7:  # degenerate (e.g. z ≡ 0 ⇒ B=C=A)
        return 0.0

    gamma_opt = (loss_minus - loss_plus) / denominator
    return gamma_opt


def normalisation_factors(
    net: nn.Sequential | list[nn.Module],
) -> torch.Tensor:
    """
    Computes the normalisation factors for each layer in a neural network.

    Arguments:
    ----------
    net (nn.Sequential):
        Neural network to compute normalisation factors for.

    Returns:
    --------
        torch.Tensor: Normalisation factors for each layer.
    """
    norms = []
    for layer in net:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            factor = layer.weight.data.norm().item()
            norms.append(factor)
    norms = torch.tensor(norms, dtype=torch.float32)
    factor = norms.prod().pow(1.0 / len(norms))
    return factor.repeat(len(norms)) / norms


def scale_network_weights(
    net: nn.Sequential | list[nn.Module],
    factors: float | list[float],
    normalise: bool = False,
) -> nn.Sequential:
    """
    Scales the weights of a neural network by a given factor.

    Arguments:
    ----------
    net (nn.Sequential):
        Neural network to scale.
    factors (float | list[float]):
        Scaling factor or list of scaling factors for each layer.
        If a single float is provided, it will be applied uniformly to all layers.
    normalise (bool):
        If True, the scaling factors will be normalised such that the product of
        all layer weights remains constant. Defaults to False.

    Returns:
    --------
        nn.Sequential: Scaled neural network.
    """
    if isinstance(factors, (int, float)):
        factors = [factors] * len(net)
    factors = torch.tensor(factors, dtype=torch.float32)
    if normalise:
        factors = factors * normalisation_factors(net)
    current_normalisation = 1
    for layer, factor in zip(net, factors):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer.weight.data *= factor
            current_normalisation *= factor
            if layer.bias is not None:
                layer.bias.data *= current_normalisation
    return net
