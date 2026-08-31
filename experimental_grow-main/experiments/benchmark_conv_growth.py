"""
A script to compare multiple methods that try to solve
argmin_{a, b} ||y - L_b o sigma o L_a(x)||_2
"""

from time import perf_counter
from typing import Any, Callable, Iterator

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from benchmark_layer_initialisation.auxilliary_files import (
    isotropic_iterator,
    random_network_iterator,
    score,
)
from benchmark_layer_initialisation.initialisation_methods import (
    compose_initialisation,
    create_random_layers,
    create_zero_layers,
    full_s_optimized_conv_layers,
    norm_approx_optimized_conv_layers,
    restricted_fogro_optimized_conv_layers,
    sgd_optimized_layers,
)
from gromo.utils.utils import global_device, set_device

set_device(torch.device("cpu"))
DEVICE = global_device()
print(f"Using device {DEVICE}")


def benchmark_method(
    method: Callable[
        [
            dict[str, Any],
            dict[str, Any],
            nn.Module,
            Iterator[tuple[torch.Tensor, torch.Tensor]],
            torch.device,
        ],
        tuple[nn.Conv2d, nn.Conv2d, Any],
    ],
    data_iterator: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]],
    layer_1_parameters: dict[str, Any],
    layer_2_parameters: dict[str, Any],
    device: torch.device = DEVICE,
    non_linearity: nn.Module = None,
) -> float:
    """
    Benchmarks a given method by computing the average squared Euclidean norm
    between the predicted and actual y values.

    Arguments:
    ----------
    method : Callable
        A function that returns c1, c2 when given an iterator and hyperparameters.
    data_iterator: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]
        Iterator yielding batched x, y tensors.
    layer_1_parameters : Dict[str, Any]
        Hyperparameters for the first layer.
    layer_2_parameters : Dict[str, Any]
        Hyperparameters for the second layer.
    device : torch.device
        Device to run the computations on.

    Returns:
    --------
    float : The average squared Euclidean norm.
    """
    # Create convolutional layers using the method*
    layer1, layer2, _ = method(
        layer_1_parameters, layer_2_parameters, data_iterator=data_iterator, device=device
    )

    if non_linearity is None:
        non_linearity = nn.Identity()
    net = nn.Sequential(layer1, non_linearity, layer2).to(device)

    return score(net, data_iterator, device)


# Run benchmarks
if __name__ == "__main__":
    from functools import partial

    relu = None
    seed = torch.randint(0, 1000, (1,)).item()
    # seed = 249
    full_solution = False
    print(f"Seed: {seed}")

    all_benchmarks = False

    # Create composed methods using compose_initialisation
    def create_composed_method(first_method, sgd_kwargs):
        def composed_method(
            layer_1_parameters, layer_2_parameters, data_iterator, device, **kwargs
        ):
            return compose_initialisation(
                methods=[first_method, sgd_optimized_layers],
                layer_1_parameters=layer_1_parameters,
                layer_2_parameters=layer_2_parameters,
                kwargs_list=[{}, sgd_kwargs],
                data_iterator=data_iterator,
                device=device,
                **kwargs,
            )

        return composed_method

    methods = {
        "random (baseline)": create_random_layers,
        "zeros": create_zero_layers,
        "restricted_fogro": restricted_fogro_optimized_conv_layers,
        "fogro": norm_approx_optimized_conv_layers,
        # "full_version": full_s_optimized_conv_layers,
        # "AdamW(1e-3, 20)": partial(
        #     sgd_optimized_layers, lr=0.001, epochs=20, non_linearity=relu
        # ),
        "AdamW(1e-2, 20)": partial(
            sgd_optimized_layers, lr=0.01, epochs=20, non_linearity=relu
        ),
        # "AdamW(1e-1, 20)": partial(
        #     sgd_optimized_layers, lr=0.1, epochs=20, non_linearity=relu
        # ),
        # "fogro -> AdamW(1e-3, 20)": create_composed_method(
        #     norm_approx_optimized_conv_layers,
        #     {"lr": 0.001, "epochs": 20, "non_linearity": relu},
        # ),
        # "fogro -> AdamW(1e-2, 20)": create_composed_method(
        #     norm_approx_optimized_conv_layers,
        #     {"lr": 0.01, "epochs": 20, "non_linearity": relu},
        # ),
        # "full_s -> AdamW(1e-3, 20)": create_composed_method(
        #     full_s_optimized_conv_layers,
        #     {"lr": 0.001, "epochs": 20, "non_linearity": relu},
        # ),
        # "full_s -> AdamW(1e-2, 20)": create_composed_method(
        #     full_s_optimized_conv_layers,
        #     {"lr": 0.01, "epochs": 20, "non_linearity": relu},
        # ),
        # "restricted_fogro -> AdamW(1e-3, 20)": create_composed_method(
        #     restricted_fogro_optimized_conv_layers,
        #     {"lr": 0.001, "epochs": 20, "non_linearity": relu},
        # ),
        "restricted_fogro -> AdamW(1e-2, 20)": create_composed_method(
            restricted_fogro_optimized_conv_layers,
            {"lr": 0.01, "epochs": 20, "non_linearity": relu},
        ),
        # "restricted_fogro -> AdamW(1e-1, 20)": create_composed_method(
        #     restricted_fogro_optimized_conv_layers,
        #     {"lr": 0.1, "epochs": 20, "non_linearity": relu},
        # ),
    }

    in_channels = 30
    intermediate_channels = 10
    out_channels = 40

    # Define random iterators
    iterator_random = isotropic_iterator(
        batch_size=64,
        n_samples=10,
        in_size=(in_channels, 13, 13),
        out_size=(out_channels, 13, 13),
        seed=seed,
    )

    # Define hyperparameter ranges
    c1_params = {
        "type": nn.Conv2d,
        "in_channels": in_channels,
        "out_channels": intermediate_channels,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "bias": False,
    }
    c2_params = {
        "type": nn.Conv2d,
        "in_channels": intermediate_channels,
        "out_channels": out_channels,  # To match y_iterator_random
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "bias": False,
    }

    if full_solution:
        k = min(
            c1_params["in_channels"] * c1_params["kernel_size"] ** 2,
            c2_params["out_channels"] * c2_params["kernel_size"] ** 2,
        )

        c1_params["out_channels"] = k
        c2_params["in_channels"] = k
    print(f"Number of intermediate channels: {c1_params['out_channels']}")

    if all_benchmarks:
        print("Benchmarking on purely random (X, Y):")
        for name, method in methods.items():
            # print(f"Testing {name}")
            error = benchmark_method(
                method, iterator_random, c1_params, c2_params, non_linearity=relu
            )
            print(f"{name}: {error:.3e}")

    # Generate c1, c2 for the second test
    torch.manual_seed(seed)
    c1_test = nn.Conv2d(
        in_channels,
        intermediate_channels,
        kernel_size=c1_params["kernel_size"],
        stride=1,
        padding=c1_params["padding"],
        device=DEVICE,
        bias=c1_params.get("bias", False),
    )
    c2_test = nn.Conv2d(
        intermediate_channels,
        out_channels,
        kernel_size=c2_params["kernel_size"],
        stride=1,
        padding=c2_params["padding"],
        device=DEVICE,
        bias=c2_params.get("bias", False),
    )

    def perfect(*args, **kwargs):
        return c1_test, c2_test, None

    methods["perfect"] = perfect

    # Create the target network for data generation
    if relu is not None:
        target_net = nn.Sequential(c1_test, relu, c2_test)
    else:
        target_net = nn.Sequential(c1_test, c2_test)

    # Define generated iterators using random_network_iterator
    iterator_gen = random_network_iterator(
        f=target_net,
        p=None,  # No preprocessing, x = N(0,1)
        in_size=(in_channels, 9, 13),
        batch_size=256,
        n_samples=1,
        noise_level=0.0,
        device=DEVICE,
        seed=seed,
    )

    print("\nBenchmarking on X, Y = c2(c1(X)):")
    for name, method in methods.items():
        print(f"{name}: ", end="")
        error = benchmark_method(
            method, iterator_gen, c1_params, c2_params, non_linearity=relu
        )
        print(f"{name}: {error:.3e}")

    # Define generated iterators with noise
    noise = 0.1
    iterator_gen_noisy = random_network_iterator(
        f=target_net,
        p=None,  # No preprocessing, x = N(0,1)
        in_size=(in_channels, 9, 13),
        batch_size=32,
        n_samples=100,
        noise_level=noise,
        device=DEVICE,
        seed=seed,
    )

    if all_benchmarks:
        print(f"\nBenchmarking on X, Y = c2(c1(X)) + {noise} * N(0, 1) :")
        for name, method in methods.items():
            error = benchmark_method(
                method, iterator_gen_noisy, c1_params, c2_params, non_linearity=relu
            )
            print(f"{name}: {error:.3e}")
