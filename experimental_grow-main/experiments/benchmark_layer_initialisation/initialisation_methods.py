""""
This module contains the initialisation methods for two-layer networks.
All the methods are encapsulated in functions with the same signature.
The functions take the following parameters:
- layer_1_parameters: a dictionary containing the hyperparameters for the first layer
- layer_2_parameters: a dictionary containing the hyperparameters for the second layer
- activation_function: the activation function to be used after the first layer
- data_iterator: a data iterator that yields batches of data
- device: the device to use for the computations (CPU or GPU)
- **kwargs: additional keyword arguments for the initialisation method

The functions return a tuple containing:
- layer_1: the first layer of the network
- layer_2: the second layer of the network
- Any additional information that may be needed for the initialisation method
"""

from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
from gromo.modules.conv2d_growing_module import (
    FullConv2dGrowingModule,
    RestrictedConv2dGrowingModule,
)
from gromo.modules.growing_module import GrowingModule
from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.utils.tensor_statistic import TensorStatistic

from experiments.benchmark_layer_initialisation.auxilliary_files import gpu_memory_usage
from experiments.benchmark_layer_initialisation.simple_scaling import (
    line_search,
    scale_network_weights,
)


def create_random_layers(
    layer_1_parameters: dict[str, Any],
    layer_2_parameters: dict[str, Any],
    activation_function: nn.Module | None = None,
    data_iterator: (
        Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]] | None
    ) = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[nn.Module, nn.Module, None]:
    """
    Creates two random Conv2D layers such that y ~ c2(c1(x)).

    Arguments:
    ----------
    layer_1_parameters (dict):
        Hyperparameters for the first layer.
    layer_2_parameters (dict):
        Hyperparameters for the second layer.
    data_iterator: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]
        Iterator yielding batched x, y tensors.
    device (torch.device):
        Device to run the computations on.

    Returns:
    --------
        c1 (nn.Module): First layer.
        c2 (nn.Module): Second layer.
    """
    layer_1_parameters = layer_1_parameters.copy()
    layer_2_parameters = layer_2_parameters.copy()

    if "use_bias" in layer_1_parameters:
        layer_1_parameters["bias"] = layer_1_parameters.pop("use_bias")
    if "use_bias" in layer_2_parameters:
        layer_2_parameters["bias"] = layer_2_parameters.pop("use_bias")

    assert isinstance(
        layer_1_parameters, dict
    ), f"layer_1_parameters must be a dictionary but is {type(layer_1_parameters)}"
    assert isinstance(
        layer_2_parameters, dict
    ), f"layer_2_parameters must be a dictionary but is {type(layer_2_parameters)}"

    if "type" not in layer_1_parameters:
        layer_1_parameters["type"] = nn.Conv2d
    if "type" not in layer_2_parameters:
        layer_2_parameters["type"] = nn.Conv2d

    assert layer_1_parameters["type"] in [
        nn.Conv2d,
        nn.Linear,
    ], f"layer_1_parameters['type'] must be nn.Conv2d or nn.Linear but is {layer_1_parameters['type']}"

    layer_1_type = layer_1_parameters["type"]
    del layer_1_parameters["type"]
    layer1 = layer_1_type(**layer_1_parameters, device=device)

    layer_2_type = layer_2_parameters["type"]
    del layer_2_parameters["type"]
    layer2 = layer_2_type(**layer_2_parameters, device=device)

    return layer1, layer2, None


def create_zero_layers(
    layer_1_parameters: dict[str, Any],
    layer_2_parameters: dict[str, Any],
    activation_function: nn.Module | None = None,
    data_iterator: (
        Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]] | None
    ) = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[nn.Module, nn.Module, Any]:
    """
    Creates two random Conv2D layers such that y ~ c2(c1(x)).

    Arguments:
    ----------
    layer_1_parameters (dict):
        Hyperparameters for the first layer.
    layer_2_parameters (dict):
        Hyperparameters for the second layer.
    data_iterator: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]
        Iterator yielding batched x, y tensors.
    device (torch.device):
        Device to run the computations on.

    Returns:
    --------
        c1 (nn.Module): First layer.
        c2 (nn.Module): Second layer.
    """
    layer1, layer2, _ = create_random_layers(
        layer_1_parameters, layer_2_parameters, data_iterator, device=device
    )

    layer1.weight.data.fill_(0)
    if layer1.bias is not None:
        layer1.bias.data.fill_(0)

    layer2.weight.data.fill_(0)
    if layer2.bias is not None:
        layer2.bias.data.fill_(0)

    return layer1, layer2, None


def sgd_optimized_layers(
    layer_1_parameters: dict[str, Any] | None = None,
    layer_2_parameters: dict[str, Any] | None = None,
    activation_function: nn.Module | None = None,
    data_iterator: (
        Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]] | None
    ) = None,
    layer1: nn.Module | None = None,
    layer2: nn.Module | None = None,
    device: torch.device = torch.device("cpu"),
    lr: float = 0.01,
    momentum: float = 0.0,
    optimizer: str = "Adam",
    epochs: int = 5,
) -> tuple[nn.Module, nn.Module, list[float]]:
    """
    Creates and optimizes two Conv2D layers using stochastic gradient descent.

    Arguments:
    ----------
    layer_1_parameters (dict):
        Hyperparameters for the first Conv2d layer.
    layer_2_parameters (dict):
        Hyperparameters for the second Conv2d layer.
    data_iterator: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]
        Iterator yielding batched x, y tensors.
    device (torch.device):
        Device to run the computations on.
    lr (float):
        Learning rate for SGD.
    epochs (int):
        Number of epochs for optimization.

    Returns:
    --------
        c1 (nn.Conv2d): First optimized convolutional layer.
        c2 (nn.Conv2d): Second optimized convolutional layer.
    """
    # Initialize convolutional layers
    if layer1 is None or layer2 is None:
        rd_layer1, rd_layer2, _ = create_random_layers(
            layer_1_parameters,
            layer_2_parameters,
            activation_function=None,
            data_iterator=None,
            device=device,
        )
        if layer1 is None:
            layer1 = rd_layer1
        if layer2 is None:
            layer2 = rd_layer2

    if activation_function is None:
        activation_function = nn.Identity()
    net = nn.Sequential(layer1, activation_function, layer2).to(device)

    if optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    elif optimizer.lower() == "adam":
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer} (only 'sgd' and 'adam' are supported)"
        )
    loss_function = nn.MSELoss()

    losses = []
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        n_samples = 0
        for x_batch, y_batch in data_iterator():
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = net(x_batch)
            loss = loss_function(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_samples += x_batch.size(0)
        avg_loss = total_loss / n_samples
        losses.append(avg_loss)
    return layer1, layer2, losses


def norm_approx_optimized_conv_layers(
    layer_1_parameters: dict[str, Any] | None = None,
    layer_2_parameters: dict[str, Any] | None = None,
    activation_function: nn.Module | None = None,
    data_iterator: (
        Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]] | None
    ) = None,
    device: torch.device = torch.device("cpu"),
    normalize_layers: bool = False,
) -> tuple[nn.Conv2d, nn.Conv2d, None]:
    """
    Creates and optimizes two Conv2D layers using the fogro method.

    Arguments:
    ----------
    layer_1_parameters (dict):
        Hyperparameters for the first Conv2d layer.
    layer_2_parameters (dict):
        Hyperparameters for the second Conv2d layer.
    data_iterator: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]
        Iterator yielding batched x, y tensors.
    device (torch.device):
        Device to run the computations on.
    normalize_layers (bool):
        If True, normalizes the layer weights during scaling. Defaults to False.

    Returns:
    --------
        c1 (nn.Conv2d): First optimized convolutional layer.
        c2 (nn.Conv2d): Second optimized convolutional layer.
    """
    # Initialize convolutional layers
    c1 = FullConv2dGrowingModule(
        in_channels=layer_1_parameters["in_channels"],
        out_channels=1,
        kernel_size=layer_1_parameters["kernel_size"],
        stride=layer_1_parameters["stride"],
        padding=layer_1_parameters["padding"],
        device=device,
        use_bias=layer_1_parameters.get("bias", False),
    )
    c2 = FullConv2dGrowingModule(
        in_channels=1,
        out_channels=layer_2_parameters["out_channels"],
        kernel_size=layer_2_parameters["kernel_size"],
        stride=layer_2_parameters["stride"],
        padding=layer_2_parameters["padding"],
        device=device,
        previous_module=c1,
        use_bias=layer_2_parameters.get("bias", False),
    )

    net = nn.Sequential(c1, c2)

    for i in (0, 1):
        net[i].weight.data *= 0
        if net[i].bias is not None:
            net[i].bias.data *= 0

    c2.init_computation()
    c2.tensor_m_prev.init()
    c2.tensor_s_growth.init()

    flag = True
    for x_batch, y_batch in data_iterator():
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        net.zero_grad()
        y_pred = net(x_batch)
        loss = nn.functional.mse_loss(y_pred, y_batch, reduction="sum")
        loss.backward()

        c2.update_computation()
        c2.tensor_m_prev.update()
        c2.tensor_s_growth.update()

        if flag:
            flag = False
            print(
                f"Fogro, batch size: {x_batch.size(0)}, "
                f"input shape: {x_batch.shape}, output shape: {y_pred.shape}"
            )
            print("GPU USAGE:")
            gpu_memory_usage()

    c2.compute_optimal_delta()
    c2.delta_raw *= 0

    c2.compute_optimal_added_parameters(
        statistical_threshold=1e-5,
        maximum_added_neurons=layer_1_parameters["out_channels"],
    )
    assert isinstance(
        c2.extended_input_layer, nn.Conv2d
    ), f"extended_output_layer must be a Conv2d but is {type(c2.extended_output_layer)}"

    gamma = (
        line_search(
            c1.extended_output_layer,
            c2.extended_input_layer,
            data_iterator,
            device=device,
        )
        ** 0.5
    )

    scale_network_weights(
        [c1.extended_output_layer, c2.extended_input_layer],
        factors=gamma,
        normalise=normalize_layers,
    )

    return c1.extended_output_layer, c2.extended_input_layer, None


def restricted_fogro_optimized_conv_layers(
    layer_1_parameters: dict[str, Any],
    layer_2_parameters: dict[str, Any],
    activation_function: nn.Module | None = None,
    data_iterator: (
        Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]] | None
    ) = None,
    device: torch.device = torch.device("cpu"),
    normalize_layers: bool = False,
) -> tuple[nn.Conv2d, nn.Conv2d, None]:
    """
    Creates and optimizes two Conv2D layers using the restricted fogro method.

    Arguments:
    ----------
    layer_1_parameters (dict):
        Hyperparameters for the first Conv2d layer.
    layer_2_parameters (dict):
        Hyperparameters for the second Conv2d layer.
    data_iterator: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]
        Iterator yielding batched x, y tensors.
    device (torch.device):
        Device to run the computations on.
    normalize_layers (bool):
        If True, normalizes the layer weights during scaling. Defaults to False.

    Returns:
    --------
        c1 (nn.Conv2d): First optimized convolutional layer.
        c2 (nn.Conv2d): Second optimized convolutional layer.
    """
    # Initialize convolutional layers
    gconv1 = RestrictedConv2dGrowingModule(
        in_channels=layer_1_parameters["in_channels"],
        out_channels=1,
        kernel_size=layer_1_parameters["kernel_size"],
        stride=layer_1_parameters["stride"],
        padding=layer_1_parameters["padding"],
        device=device,
        use_bias=layer_1_parameters.get("bias", False),
        name="CL-C1",
    )
    c2 = RestrictedConv2dGrowingModule(
        in_channels=1,
        out_channels=layer_2_parameters["out_channels"],
        kernel_size=layer_2_parameters["kernel_size"],
        stride=layer_2_parameters["stride"],
        padding=layer_2_parameters["padding"],
        device=device,
        previous_module=gconv1,
        use_bias=layer_2_parameters.get("bias", False),
        name="CL-C2",
    )

    net = nn.Sequential(gconv1, c2)

    for i in (0, 1):
        net[i].weight.data *= 0
        if net[i].bias is not None:
            net[i].bias.data *= 0

    c2.init_computation()
    c2.tensor_m_prev.init()
    c2.tensor_s_growth.init()

    x_batch, y_batch = None, None
    flag = True
    for x_batch, y_batch in data_iterator():
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        net.zero_grad()
        y_pred = net(x_batch)
        loss = nn.functional.mse_loss(y_pred, y_batch, reduction="sum") / 2
        loss.backward()

        # c2.update_computation()
        c2.tensor_m_prev.update()
        c2.tensor_s_growth.update()
        gconv1.clear_storage()
        c2.clear_storage()

        if flag:
            flag = False
            print(
                f"Restricted FoGro, batch size: {x_batch.size(0)}, "
                f"input shape: {x_batch.shape}, output shape: {y_pred.shape}"
            )
            print("GPU USAGE:")
            gpu_memory_usage()

    # c2.delta_raw = torch.zeros_like(c2.weight).flatten(start_dim=1).t()
    c2.compute_optimal_added_parameters(
        maximum_added_neurons=layer_1_parameters["out_channels"],
        statistical_threshold=1e-5,
        projected_bottleneck=False,
    )
    assert isinstance(
        c2.extended_input_layer, nn.Conv2d
    ), f"extended_output_layer must be a Conv2d but is {type(c2.extended_output_layer)}"

    new_c1 = gconv1.extended_output_layer
    new_c2 = c2.extended_input_layer

    # gamma = (
    #     line_search(
    #         new_c1,
    #         new_c2,
    #         data_iterator,
    #         device=device,
    #     )
    #     ** 0.5
    # )
    # assert (abs(gamma ** 2 - 1) < 1e-4), f"Gamma squared is {gamma ** 2}, expected close to 1"

    gamma = 1
    scale_network_weights([new_c1, new_c2], factors=gamma, normalise=normalize_layers)

    return new_c1, new_c2, None


def full_s_optimized_conv_layers(
    layer_1_parameters: dict[str, Any],
    layer_2_parameters: dict[str, Any],
    activation_function: nn.Module | None = None,
    data_iterator: (
        Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]] | None
    ) = None,
    device: torch.device = torch.device("cpu"),
    normalize_layers: bool = False,
) -> tuple[nn.Conv2d, nn.Conv2d, None]:
    """
    Creates and optimizes two Conv2D layers using the fogro method.

    Arguments:
    ----------
    layer_1_parameters (dict):
        Hyperparameters for the first Conv2d layer.
    layer_2_parameters (dict):
        Hyperparameters for the second Conv2d layer.
    data_iterator: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]
        Iterator yielding batched x, y tensors.
    device (torch.device):
        Device to run the computations on.
    normalize_layers (bool):
        If True, normalizes the layer weights during scaling. Defaults to False.

    Returns:
    --------
        c1 (nn.Conv2d): First optimized convolutional layer.
        c2 (nn.Conv2d): Second optimized convolutional layer.
    """
    # Initialize convolutional layers
    c1 = FullConv2dGrowingModule(
        in_channels=layer_1_parameters["in_channels"],
        out_channels=1,
        kernel_size=layer_1_parameters["kernel_size"],
        stride=layer_1_parameters["stride"],
        padding=layer_1_parameters["padding"],
        device=device,
        use_bias=layer_1_parameters.get("bias", False),
    )
    c2 = FullConv2dGrowingModule(
        in_channels=1,
        out_channels=layer_2_parameters["out_channels"],
        kernel_size=layer_2_parameters["kernel_size"],
        stride=layer_2_parameters["stride"],
        padding=layer_2_parameters["padding"],
        device=device,
        previous_module=c1,
        use_bias=layer_2_parameters.get("bias", False),
    )

    def compute_full_s():
        bt = c2.masked_unfolded_prev_input
        return torch.einsum("ijea, ijfb -> eafb", bt, bt), bt.size(0)

    net = nn.Sequential(c1, c2)

    for i in (0, 1):
        net[i].weight.data *= 0
        if net[i].bias is not None:
            net[i].bias.data *= 0

    c2.init_computation()
    c2.tensor_m_prev.init()
    c2.tensor_s_growth.init()
    full_s = TensorStatistic(
        shape=None, update_function=compute_full_s, device=device, name="Full S"
    )

    flag = True
    for x_batch, y_batch in data_iterator():
        # print("Hello")
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        net.zero_grad()
        y_pred = net(x_batch)

        loss = nn.functional.mse_loss(y_pred, y_batch, reduction="sum") / 2
        loss.backward()
        c2.update_computation()

        c2.tensor_m_prev.update()
        c2.tensor_s_growth.update()

        full_s.updated = False
        full_s.update()
        if flag:
            flag = False
            print(
                f"Full S, batch size: {x_batch.size(0)}, "
                f"input shape: {x_batch.shape}, output shape: {y_pred.shape}"
            )
            print("GPU USAGE:")
            gpu_memory_usage()

    c2.compute_optimal_delta()
    c2.delta_raw *= 0

    tensor_n = -c2.tensor_m_prev()
    # C[-1] dh dw, C[+1], dh[+1] dw[+1]
    assert (
        tensor_n.size(0) == c1.in_channels * c1.kernel_size[0] * c1.kernel_size[1]
    ), f"tensor_n size 0 must be {c1.in_channels * c1.kernel_size[0] * c1.kernel_size[1]} but is {tensor_n.size(0)}"
    assert (
        tensor_n.size(1) == c2.out_channels
    ), f"tensor_n size 1 must be {c2.out_channels} but is {tensor_n.size(1)}"
    assert (
        tensor_n.size(2) == c2.kernel_size[0] * c2.kernel_size[1]
    ), f"tensor_n size 2 must be {c2.kernel_size[0] * c2.kernel_size[1]} but is {tensor_n.size(2)}"

    full_tensor_s = full_s()
    # "dh[+1] dw[+1], C[-1] dh dw, dh[+1] dw[+1], C[-1] dh dw"
    assert (
        full_tensor_s.size(0) == c2.kernel_size[0] * c2.kernel_size[1]
    ), f"full_tensor_s size 0 must be {c2.kernel_size[0] * c2.kernel_size[1]} but is {full_tensor_s.size(0)}"
    assert (
        full_tensor_s.size(1) == c1.in_channels * c1.kernel_size[0] * c1.kernel_size[1]
    ), f"full_tensor_s size 1 must be {c1.in_channels * c1.kernel_size[0] * c1.kernel_size[1]} but is {full_tensor_s.size(1)}"
    assert (
        full_tensor_s.size(2) == c2.kernel_size[0] * c2.kernel_size[1]
    ), f"full_tensor_s size 2 must be {c2.kernel_size[0] * c2.kernel_size[1]} but is {full_tensor_s.size(2)}"
    assert (
        full_tensor_s.size(3) == c1.in_channels * c1.kernel_size[0] * c1.kernel_size[1]
    ), f"full_tensor_s size 3 must be {c1.in_channels * c1.kernel_size[0] * c1.kernel_size[1]} but is {full_tensor_s.size(3)}"

    tensor_n = tensor_n.permute(2, 0, 1).flatten(end_dim=1)

    full_tensor_s = full_tensor_s.flatten(start_dim=2).flatten(end_dim=1)

    full_solution = (torch.linalg.pinv(full_tensor_s) @ tensor_n).t()
    full_solution = full_solution.reshape(
        c2.out_channels,
        c2.kernel_size[0] * c2.kernel_size[1],
        c1.in_channels * c1.kernel_size[0] * c1.kernel_size[1],
    ).flatten(end_dim=1)
    k = layer_1_parameters["out_channels"]

    u, s, v = torch.linalg.svd(full_solution, full_matrices=False)

    s = torch.sqrt(s[:k])
    u = u[:, :k] * s
    v = s[:, None] * v[:k]

    u = u.reshape(c2.out_channels, c2.kernel_size[0], c2.kernel_size[1], k).permute(
        0, 3, 1, 2
    )
    v = v.reshape(k, c1.in_channels, c1.kernel_size[0], c1.kernel_size[1])

    # u = u.permute(3, 0, 1, 2)
    # v = v.permute(1, 0, 2, 3)

    new_c1 = c1.layer_of_tensor(v)
    new_c2 = c2.layer_of_tensor(u)

    gamma = (
        line_search(
            new_c1,
            new_c2,
            data_iterator,
            device=device,
        )
        ** 0.5
    )

    scale_network_weights([new_c1, new_c2], factors=gamma, normalise=normalize_layers)

    return new_c1, new_c2, None


def compose_initialisation(
    methods: list[Callable],
    layer_1_parameters: dict[str, Any],
    layer_2_parameters: dict[str, Any],
    activation_function: nn.Module | None = None,
    data_iterator: (
        Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]] | None
    ) = None,
    device: torch.device = torch.device("cpu"),
    kwargs_list: list[dict[str, Any]] | None = None,
    **global_kwargs,
) -> tuple[nn.Module, nn.Module, list[Any]]:
    """
    Compose multiple initialization methods sequentially.

    Each method in the sequence receives the output layers from the previous method
    as input (except the first method which creates layers from scratch).

    Parameters:
    -----------
    methods : list[Callable]
        List of initialization functions to apply sequentially. Each function should
        have the signature matching the initialization methods in this module.
    layer_1_parameters : dict[str, Any]
        Hyperparameters for the first layer.
    layer_2_parameters : dict[str, Any]
        Hyperparameters for the second layer.
    activation_function : nn.Module, optional
        Activation function to apply between layers.
    data_iterator : Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]], optional
        Iterator yielding batched (x, y) tensors.
    device : torch.device
        Device to run computations on.
    kwargs_list : list[dict[str, Any]], optional
        List of keyword arguments for each method. If provided, must have same length
        as methods list. Each dict contains kwargs specific to the corresponding method.
        If None, global_kwargs are used for all methods.
    **global_kwargs
        Global keyword arguments passed to all methods (unless overridden by kwargs_list).

    Returns:
    --------
    tuple[nn.Module, nn.Module, list[Any]]
        - First layer after all methods applied
        - Second layer after all methods applied
        - List of additional outputs from each method

    Raises:
    -------
    ValueError
        If methods list is empty or kwargs_list length doesn't match methods length.
    """
    if not methods:
        raise ValueError("At least one initialization method must be provided")

    if kwargs_list is not None:
        if len(kwargs_list) != len(methods):
            raise ValueError(
                f"kwargs_list length ({len(kwargs_list)}) must match methods length ({len(methods)})"
            )
    else:
        # Use global_kwargs for all methods if no kwargs_list provided
        kwargs_list = [global_kwargs.copy() for _ in methods]

    # Merge global_kwargs with method-specific kwargs
    # Method-specific kwargs take precedence over global ones
    merged_kwargs_list = []
    for method_kwargs in kwargs_list:
        merged_kwargs = global_kwargs.copy()
        merged_kwargs.update(method_kwargs)
        merged_kwargs_list.append(merged_kwargs)

    # Apply first method (creates layers from scratch)
    first_method = methods[0]
    first_kwargs = merged_kwargs_list[0]

    # Remove activation_function from kwargs to avoid conflict
    first_method_activation = first_kwargs.pop("activation_function", activation_function)
    first_kwargs_clean = {
        k: v for k, v in first_kwargs.items() if k != "activation_function"
    }

    layer1, layer2, first_output = first_method(
        layer_1_parameters=layer_1_parameters,
        layer_2_parameters=layer_2_parameters,
        activation_function=first_method_activation,
        data_iterator=data_iterator,
        device=device,
        **first_kwargs_clean,
    )

    outputs = [first_output]

    # Apply remaining methods sequentially
    for method, method_kwargs in zip(methods[1:], merged_kwargs_list[1:]):
        # Remove activation_function from kwargs to avoid conflict
        method_activation = method_kwargs.pop("activation_function", activation_function)
        method_kwargs_clean = {
            k: v for k, v in method_kwargs.items() if k != "activation_function"
        }

        try:
            # Try to call with layer1 and layer2 parameters
            layer1, layer2, method_output = method(
                layer_1_parameters=layer_1_parameters,
                layer_2_parameters=layer_2_parameters,
                activation_function=method_activation,
                data_iterator=data_iterator,
                device=device,
                layer1=layer1,
                layer2=layer2,
                **method_kwargs_clean,
            )
        except TypeError:
            # Fallback: call without layer1/layer2 if method doesn't support them
            layer1, layer2, method_output = method(
                layer_1_parameters=layer_1_parameters,
                layer_2_parameters=layer_2_parameters,
                activation_function=method_activation,
                data_iterator=data_iterator,
                device=device,
                **method_kwargs_clean,
            )

        outputs.append(method_output)

    return layer1, layer2, outputs


def natural_gradient_step_layers(
    layer_1_parameters: dict[str, Any],
    layer_2_parameters: dict[str, Any],
    activation_function: nn.Module | None = None,
    data_iterator: (
        Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]] | None
    ) = None,
    layer1: nn.Module | None = None,
    layer2: nn.Module | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[nn.Module, nn.Module, None]:
    """
    Creates two Conv2D layers and performs a single natural gradient step on
    the second layer.

    Arguments:
    ----------
    layer_1_parameters: dict[str, Any]
        Hyperparameters for the first Conv2d layer.
    layer_2_parameters: dict[str, Any]
        Hyperparameters for the second Conv2d layer.
    activation_function: nn.Module | None
        Activation function to apply between layers.
    data_iterator: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]] | None
        Iterator yielding batched (x, y) tensors.
    layer1: nn.Module | None
        Pre-initialized first layer, if any.
    layer2: nn.Module | None
        Pre-initialized second layer, if any.
    device: torch.device
        Device to run the computations on.

    Returns:
    --------
        c1 (nn.Module): First Conv2d layer.
        c2 (nn.Module): Second Conv2d layer after natural gradient step.
    """
    layer_2_parameters = layer_2_parameters.copy()

    if "bias" in layer_2_parameters:
        layer_2_parameters["use_bias"] = layer_2_parameters.pop("bias")

    # Initialize the growing modules
    growing_type = {
        nn.Linear: LinearGrowingModule,
        nn.Conv2d: RestrictedConv2dGrowingModule,
    }

    growing_type = growing_type[layer_2_parameters["type"]]
    del layer_2_parameters["type"]
    growing_layer_2 = growing_type(
        **layer_2_parameters,
        device=device,
    )

    growing_layer_2: GrowingModule

    if layer1 is None:
        layer1, _, _ = create_random_layers(
            layer_1_parameters,
            layer_2_parameters,
            activation_function=None,
            data_iterator=None,
            device=device,
        )
    else:
        layer1 = layer1.to(device)
    if layer2 is not None:
        if isinstance(layer2, nn.Conv2d):
            assert isinstance(
                growing_layer_2, RestrictedConv2dGrowingModule
            ), f"growing_layer_2 must be a RestrictedConv2dGrowingModule but is {type(growing_layer_2)}"
            assert layer2.in_channels == growing_layer_2.in_channels, (
                f"layer2.in_channels ({layer2.in_channels}) must match "
                f"growing_layer_2.in_channels ({growing_layer_2.in_channels})"
            )
            assert layer2.out_channels == growing_layer_2.out_channels, (
                f"layer2.out_channels ({layer2.out_channels}) must match "
                f"growing_layer_2.out_channels ({growing_layer_2.out_channels})"
            )
            assert layer2.kernel_size == growing_layer_2.kernel_size, (
                f"layer2.kernel_size ({layer2.kernel_size}) must match "
                f"growing_layer_2.kernel_size ({growing_layer_2.kernel_size})"
            )
            if layer2.bias is not None:
                assert layer2.bias.shape == growing_layer_2.bias.shape, (
                    f"layer2.bias shape {layer2.bias.shape} must match "
                    f"growing_layer_2.bias shape {growing_layer_2.bias.shape}"
                )
        growing_layer_2.weight.data.copy_(layer2.weight.data)
        if growing_layer_2.bias is not None:
            growing_layer_2.bias.data.copy_(layer2.bias.data)

    if activation_function is None:
        activation_function = nn.Identity()
    net = nn.Sequential(layer1, activation_function, growing_layer_2).to(device)

    growing_layer_2.init_computation()
    activation_function.requires_grad_(False)
    layer1.requires_grad_(False)

    for x_batch, y_batch in data_iterator():
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        net.zero_grad()
        y_pred = net(x_batch)
        loss = nn.functional.mse_loss(y_pred, y_batch, reduction="sum") / 2
        loss.backward()

        growing_layer_2.update_computation()

    # Perform a single natural gradient step
    growing_layer_2.compute_optimal_delta()
    growing_layer_2.reset_computation()

    growing_layer_2.scaling_factor = 1
    growing_layer_2.apply_change()

    return layer1, growing_layer_2.layer, None
