from typing import Callable, Iterator

import nvidia_smi
import torch
from torch import nn


def score(
    net: nn.Module,
    data_iterator: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]],
    device: torch.device = torch.device("cpu"),
    error_function: nn.Module = None,
) -> float:
    """
    Evaluate a neural network's performance on a dataset using a specified loss function.

    This function computes the average loss per sample across all batches in the dataset.
    The network is evaluated in no-gradient mode to prevent memory accumulation and
    ensure consistent evaluation behavior.

    Parameters:
    -----------
    net : nn.Module
        The neural network to evaluate. Must be callable with input tensors.
    data_iterator : Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]
        A callable that returns an iterator yielding (input, target) tensor pairs.
        Each call to the callable should return a fresh iterator over the dataset.
    device : torch.device, optional
        The device to move tensors to for computation. Default is cpu.
    error_function : nn.Module, optional
        The loss function to use for evaluation. Must have reduction='sum' behavior.
        If None, defaults to nn.MSELoss(reduction='sum').

    Returns:
    --------
    float
        The average loss per sample across the entire dataset.

    Raises:
    -------
    RuntimeError
        If the network forward pass fails or tensors cannot be moved to the device.
    ValueError
        If no samples are processed (empty iterator).

    Examples:
    ---------
    >>> _ = torch.random.manual_seed(0)
    >>> model = nn.Linear(10, 1)
    >>> def data_iter():
    ...     yield torch.randn(32, 10), torch.randn(32, 1)
    >>> loss = score(model, data_iter)
    >>> print(f"Average MSE loss: {loss:.4f}")
    Average MSE loss: 2.0261
    >>> # Using a custom loss function
    >>> mae_loss = nn.L1Loss(reduction='sum')
    >>> loss = score(model, data_iter, error_function=mae_loss)
    >>> print(f"Average MAE loss: {loss:.4f}")
    Average MAE loss: 0.7890
    """
    if error_function is None:
        error_function = nn.MSELoss(reduction="sum")

    net.eval()  # Set network to evaluation mode
    total_error = 0.0
    nb_samples = 0

    with torch.no_grad():  # Explicitly disable gradients for evaluation
        for x_batch, y_batch in data_iterator():
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Compute output
            y_pred = net(x_batch)

            # Compute loss for this batch
            batch_error = error_function(y_pred, y_batch).item()
            total_error += batch_error
            nb_samples += x_batch.size(0)

    if nb_samples == 0:
        raise ValueError("No samples processed - iterator returned no data")

    return total_error / nb_samples / y_batch[0].numel()  # Average loss per sample


def isotropic_iterator(
    in_size: tuple[int],
    out_size: tuple[int] | None = None,
    batch_size: int = 32,
    n_samples: int = 100,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
) -> Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]:
    """
    Creates an iterator that yields batched random tensors.

    Arguments:
    ----------
    in_size : tuple[int]
        Size of the input tensors.
    out_size : tuple[int] | None
        Size of the output tensors. If None, defaults to in_size.
    batch_size : int
        Number of samples per batch.
    n_samples : int
        Number of batches to generate.
    device : torch.device
        Device to allocate tensors.
    seed: int
        Seed

    Returns:
    --------
    Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]
        An iterator yielding random tensors.
    """
    if out_size is None:
        out_size = in_size

    def iterator():
        for i in range(n_samples):
            # set seed
            torch.manual_seed(seed + i)
            x = torch.randn(batch_size, *in_size, device=device, requires_grad=False)
            y = torch.randn(batch_size, *out_size, device=device, requires_grad=False)
            yield x, y

    return iterator


def random_network_iterator(
    f: nn.Module,
    p: nn.Module | None = None,
    in_size: tuple[int] = (3, 32, 32),
    batch_size: int = 32,
    n_samples: int = 100,
    noise_level: float = 0.0,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
) -> Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]:
    """
    Creates an iterator that yields batched tensors generated from networks.

    Generates couples (x, y) where:
    - x = p(N(0,1)) if p is provided, else x = N(0,1)
    - y = f(x) + epsilon, where epsilon is Gaussian noise

    Parameters:
    -----------
    f : nn.Module
        The target network that generates y from x. Must be callable with tensors
        of shape (batch_size, *in_size).
    p : nn.Module, optional
        The preprocessing network that transforms input noise. If None, uses identity
        (x = N(0,1) directly). Default is None.
    in_size : tuple[int]
        Size of the input tensors (excluding batch dimension).
        Default is (3, 32, 32) for CIFAR-like images.
    batch_size : int
        Number of samples per batch. Default is 32.
    n_samples : int
        Number of batches to generate. Default is 100.
    noise_level : float
        Standard deviation of the Gaussian noise added to y.
        If 0.0, no noise is added. Default is 0.0.
    device : torch.device
        Device to allocate tensors and run computations. Default is CPU.
    seed : int
        Random seed for reproducibility. Each batch uses seed + batch_index.
        Default is 0.

    Returns:
    --------
    Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]
        A callable that returns an iterator yielding (x, y) tensor pairs.

    Raises:
    -------
    RuntimeError
        If network forward passes fail or tensors cannot be moved to device.

    Examples:
    ---------
    >>> # Simple case: y = linear(x) + noise
    >>> f_net = nn.Linear(10, 5)
    >>> data_iter = random_network_iterator(
    ...     f=f_net,
    ...     in_size=(10,),
    ...     batch_size=16,
    ...     n_samples=50,
    ...     noise_level=0.1
    ... )
    >>> x, y = next(data_iter())
    >>> print(f"x shape: {x.shape}, y shape: {y.shape}")
    x shape: torch.Size([16, 10]), y shape: torch.Size([16, 5])

    >>> # With preprocessing: y = f(conv(N(0,1))) + noise
    >>> p_net = nn.Conv2d(3, 16, 3, padding=1)
    >>> f_net = nn.Conv2d(16, 3, 3, padding=1)
    >>> data_iter = random_network_iterator(
    ...     f=f_net,
    ...     p=p_net,
    ...     in_size=(3, 32, 32),
    ...     noise_level=0.05
    ... )
    >>> x, y = next(data_iter())
    >>> print(f"x shape: {x.shape}, y shape: {y.shape}")
    x shape: torch.Size([32, 16, 32, 32]), y shape: torch.Size([32, 3, 32, 32])

    >>> # No noise, identity preprocessing
    >>> f_net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    >>> data_iter = random_network_iterator(
    ...     f=f_net,
    ...     in_size=(1, 28, 28),
    ...     noise_level=0.0
    ... )
    """
    # Move networks to device and set to eval mode
    f = f.to(device)
    f.eval()

    if p is not None:
        p = p.to(device)
        p.eval()
    else:
        p = nn.Identity().to(device)

    def iterator():
        for i in range(n_samples):
            # Set seed for reproducibility
            torch.manual_seed(seed + i)

            #   # No gradients needed for data generation
            # Generate initial random input
            z = torch.randn(batch_size, *in_size, device=device, requires_grad=False)

            # Apply preprocessing network if provided
            with torch.no_grad():
                x = p(z)

            # Generate target using f network
            y_clean = f(x)

            # Add noise if specified
            if noise_level > 0.0:
                noise = torch.randn_like(y_clean, device=device) * noise_level
                y = y_clean + noise
            else:
                y = y_clean

            # Ensure no gradients are tracked
            x = x
            y = y

            yield x, y

    return iterator


def gpu_memory_usage(verbose=True) -> tuple[int, int, int]:
    """
    Return (and print if verbose) the GPU memory usage.
    - GPU memory
    - GPU memory allocated
    - GPU memory free

    A GPU is required.
    """
    if torch.cuda.is_available():
        nvidia_smi.nvmlInit()
        index = torch.cuda.current_device()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(index)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory = info.total
        gpu_memory_allocated = info.used
        gpu_memory_free = info.free
        if verbose:
            print(
                f"GPU memory: {gpu_memory / 10 ** 9:.2e} GB VRAM",
                f"GPU memory allocated: {gpu_memory_allocated / 1e9:.2e} GB VRAM",
                f"GPU memory free: {gpu_memory_free / 1e9:.2e} GB VRAM",
            )
        return gpu_memory, gpu_memory_allocated, gpu_memory_free
    else:
        print("No GPU available")
        return 0, 0, 0
