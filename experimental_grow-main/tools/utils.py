from typing import Any, Callable

import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from deprecated import deprecated
from pyvis.network import Network
from torch.utils.data import DataLoader

__global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@deprecated("This functionality is already in gromo module")
def set_device(device: str | torch.device) -> None:
    """Set default global device

    Parameters
    ----------
    device : str | torch.device
        device choice
    """
    global __global_device
    if isinstance(device, str):
        __global_device = torch.device(device)
    else:
        __global_device = device


@deprecated("This functionality is already in gromo module")
def reset_device() -> None:
    """Reset global device"""
    global __global_device
    __global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@deprecated("This functionality is already in gromo module")
def global_device() -> torch.device:
    """Get global device for whole codebase

    Returns
    -------
    torch.device
        global device
    """
    global __global_device
    return __global_device


@deprecated("This functionality is already in gromo module")
def torch_zeros(*size: tuple[int, int], **kwargs) -> torch.Tensor:
    """Create zero tensors on global selected device

    Parameters
    ----------
    size : tuple[int, int]
        size of tensor

    Returns
    -------
    torch.Tensor
        zero-initialized tensor of defined size on global device
    """
    global __global_device
    try:
        return torch.zeros(size=size, device=__global_device, **kwargs)  # type: ignore
    except TypeError:
        return torch.zeros(*size, device=__global_device, **kwargs)


@deprecated("This functionality is already in gromo module")
def torch_ones(*size: tuple[int, int], **kwargs) -> torch.Tensor:
    """Create one tensors on global selected device

    Parameters
    ----------
    size : tuple[int, int]
        size of tensor

    Returns
    -------
    torch.Tensor
        one-initialized tensor of defined size on global device
    """
    global __global_device
    try:
        return torch.ones(size=size, device=__global_device, **kwargs)  # type: ignore
    except TypeError:
        return torch.ones(*size, device=__global_device, **kwargs)


@deprecated("This functionality is already in gromo module")
def set_from_conf(self, name: str, default: Any = None, setter: bool = True) -> Any:
    """Standardize private argument setting from config file

    Parameters
    ----------
    name : str
        name of variable
    default : Any, optional
        default value in case config does not provide one, by default None
    setter : bool, optional
        set the retrieved value as argument in the object, by default True

    Returns
    -------
    Any
        value set to variable
    """
    # Check that config file has been found and read
    assert hasattr(self, "_config_data")
    assert isinstance(self._config_data, dict)

    value = self._config_data.get(name, default)

    if setter:
        setattr(self, f"{name}", value)

    return value


@deprecated("This functionality is already in gromo module")
def activation_fn(fn_name: str) -> nn.Module:
    """Create activation function module by name

    Parameters
    ----------
    fn_name : str
        name of activation function

    Returns
    -------
    torch.nn.Module
        activation function module
    """
    if fn_name is None:
        return nn.Identity()
    fn_name = fn_name.strip().lower()
    if fn_name == "id":
        return nn.Identity()
    elif fn_name == "selu":
        return nn.SELU()
    elif fn_name == "relu":
        return nn.ReLU()
    elif fn_name == "softmax":
        return nn.Softmax(dim=1)
    else:
        return nn.Identity()


@deprecated("This functionality is already in gromo module")
def line_search(cost_fn: Callable, verbose: bool = True) -> tuple[float, float]:
    """Line search for black-box convex function

    Parameters
    ----------
    cost_fn : Callable
        black-box convex function
    verbose : bool, optional
        create plot, by default True

    Returns
    -------
    tuple[float, float]
        return minima and min value
    """
    losses = []
    n_points = 100
    f_min = 1e-6
    f_max = 1
    f_test = np.concatenate(
        [np.zeros(1), np.logspace(np.log10(f_min), np.log10(f_max), n_points)]
    )

    decrease = True
    min_loss = np.inf
    f_full = np.array([])

    while decrease:
        for factor in f_test:
            loss = cost_fn(factor)
            losses.append(loss)

        f_full = np.concatenate([f_full, f_test])

        new_min = np.min(losses)
        decrease = new_min < min_loss
        min_loss = new_min

        f_min = f_max
        f_max = f_max * 10
        f_test = np.logspace(np.log10(f_min), np.log10(f_max), n_points)

    factor = f_full[np.argmin(losses)]
    min_loss = np.min(losses)

    if verbose:
        plt.figure()
        plt.plot(f_full, losses)
        plt.xlabel("factor $\gamma$")  # type: ignore
        plt.ylabel("loss")
        plt.title(f"Minima at {factor=} with loss={min_loss}")
        plt.show()

    return factor, min_loss


@deprecated("This functionality is already in gromo module")
def mini_batch_gradient_descent(
    model: nn.Module | Callable,
    cost_fn: Callable,
    dataloader: torch.utils.data.DataLoader,
    # lrate: float,
    max_epochs: int,
    # parameters: Iterable | None = None,
    optimizer: torch.optim.Optimizer,
    fast: bool = False,
    eval_fn: Callable | None = None,
    verbose: bool = True,
) -> tuple[list[float], list[float]]:
    """Mini-batch gradient descent implementation
    Uses AdamW with no weight decay and shuffled DataLoader

    Parameters
    ----------
    model : nn.Module
        pytorch model or forwards function
    cost_fn : Callable
        cost function
    X : torch.Tensor
        input features
    Y : torch.Tensor
        true labels
    lrate : float
        learning rate
    max_epochs : int
        maximum epochs
    batch_size : int
        batch size
    parameters: iterable | None, optional
        list of torch parameters in case the model is just a forward function, by default None
    fast : bool, optional
        fast implementation without evaluation, by default False
    eval_fn : Callable | None, optional
        evaluation function, by default None
    verbose : bool, optional
        print info, by default True

    Returns
    -------
    tuple[list[float], list[float]]
        train loss history, train accuracy history
    """
    loss_history, acc_history = [], []
    full_loss = []
    gradients = []

    # if not isinstance(model, nn.Module):
    #     if (parameters is None) or (len(parameters) < 1):  # type: ignore
    #         raise AttributeError(
    #             "When the model is just a forward function, the parameters argument must not be None or empty"
    #         )
    # else:
    #     parameters = model.parameters()
    #     saved_parameters = list(model.parameters())
    # optimizer = torch.optim.AdamW(parameters, lr=lrate, weight_decay=0)

    for epoch in range(max_epochs):
        correct, total, epoch_loss = 0, 0, 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(global_device()), y_batch.to(global_device())
            optimizer.zero_grad()

            output = model(x_batch)
            loss = cost_fn(output, y_batch)
            epoch_loss += loss.item()
            full_loss.append(loss.item())

            if not fast:
                correct += (output.argmax(axis=1) == y_batch).int().sum().item()
                total += len(output)

            loss.backward()

            # if isinstance(model, nn.Module):
            #     avg_grad_norm = 0.0
            #     for param in model.parameters():
            #         avg_grad_norm += param.grad.norm()
            #     avg_grad_norm /= len(list(model.parameters()))
            #     gradients.append(avg_grad_norm.cpu())
            optimizer.step()

        loss_history.append(epoch_loss / len(dataloader))
        if not fast:
            accuracy = correct / total
            acc_history.append(accuracy)
            if eval_fn is not None:
                eval_fn()

        if verbose and epoch % 10 == 0:
            if fast:
                print(f"Epoch {epoch}: Train loss {loss_history[-1]}")
            else:
                print(
                    f"Epoch {epoch}: Train loss {loss_history[-1]} Train Accuracy {accuracy}"
                )

    if verbose:
        # plt.figure()
        # plt.plot(gradients)
        # plt.xlabel("epochs")
        # plt.ylabel("gradients average norm")
        # plt.show()

        plt.figure()
        plt.plot(full_loss)
        plt.xlabel("epochs")
        plt.ylabel("batch loss")
        plt.show()

        plt.figure()
        plt.plot(loss_history)
        plt.xlabel("epochs")
        plt.ylabel("average epoch loss")
        plt.show()

        if not fast:
            plt.figure()
            plt.plot(acc_history)
            plt.xlabel("epochs")
            plt.ylabel("accuracy")
            plt.show()

    return loss_history, acc_history


@deprecated("This functionality is already in gromo module")
def batch_gradient_descent(
    forward_fn: Callable,
    cost_fn: Callable,
    target: torch.Tensor,
    optimizer,
    max_epochs: int = 100,
    tol: float = 1e-5,
    fast: bool = True,
    eval_fn: Callable | None = None,
    verbose: bool = True,
    loss_name: str = "loss",
    title: str = "",
) -> tuple[list[float], list[float]]:
    """Batch gradient descent implementation

    Parameters
    ----------
    forward_fn : Callable
        Forward function
    cost_fn : Callable
        _description_
    target : torch.Tensor
        target tensor
    optimizer : torch.optim.Optimizer
        optimizer
    max_epochs : int, optional
        max number of epochs, by default 100
    tol : float, optional
        tolerance, by default 1e-5
    fast : bool, optional
        fast implementation without evaluation, by default True
    eval_fn : Callable | None, optional
        evaluation function, by default None
    verbose : bool, optional
        _description_, by default True
    loss_name : str, optional
        name of the loss, by default "loss"
    title : str, optional
        title of the plot, by default ""

    Returns
    -------
    list[float]
        _description_
    """
    # print(target, target.shape)
    # temp = (target**2).sum()
    # print(temp)
    loss_history, acc_history = [], []
    min_loss = np.inf
    prev_loss = np.inf

    for _ in range(max_epochs):
        output = forward_fn()
        loss = cost_fn(output, target)
        loss_history.append(loss.item())

        if not fast:
            correct = (output.argmax(axis=1) == target).int().sum().item()
            accuracy = correct / len(output)
            if eval_fn:
                eval_fn()
            acc_history.append(accuracy)

        loss.backward(retain_graph=False)
        optimizer.step()
        optimizer.zero_grad()

        # Early stopping
        # if np.abs(prev_loss - loss.item()) <= tol:
        #     break
        if loss.item() < min_loss:
            min_loss = loss.item()
        prev_loss = loss.item()
        # target.detach_()

    if verbose:
        plt.figure()
        plt.plot(loss_history)
        plt.xlabel("epochs")
        plt.ylabel(f"{loss_name}")
        plt.title(f"{title}")
        plt.show()

        if not fast:
            labels = ["train"]
            plt.figure()
            plt.plot(acc_history, label=labels)
            plt.xlabel("epochs")
            plt.ylabel("accuracy")
            plt.title(f"{title}")
            plt.legend()
            plt.show()

    return loss_history, acc_history


def DAG_to_pyvis(dag):
    """Create pyvis graph based on GrowableDAG

    Parameters
    ----------
    dag : GrowableDAG
        growable dag object

    Returns
    -------
    _type_
        pyvis object
    """
    # nt = Network('500px', '500px', directed=True, notebook=True, cdn_resources='remote')
    nt = Network(directed=True)

    default_offset_x = 150.0
    default_offset_y = 0.0

    for node in dag.nodes:
        size = dag.nodes[node]["size"]
        attrs = {
            "x": None,
            "y": None,
            "physics": True,
            "label": node,
            "title": str(size),
            "color": size_to_color(size),
            "size": np.sqrt(size),
            "mass": 4,
        }
        if node == "start":
            attrs.update(
                {"x": -default_offset_x, "y": -default_offset_y, "physics": False}
            )
        elif node == "end":
            attrs.update({"x": default_offset_x, "y": default_offset_y, "physics": False})
        nt.add_node(node, **attrs)
    for edge in dag.edges:
        prev_node, next_node = edge
        module = dag.get_edge_module(prev_node, next_node)
        nt.add_edge(
            prev_node, next_node, title=module.name, label=str(module.weight.shape)
        )

    # nt.toggle_physics(False)
    return nt


def size_to_color(size):
    cmap = mpl_cm.Reds  # type: ignore
    norm = mpl_colors.Normalize(vmin=0, vmax=784)
    rgba = cmap(norm(size))
    return mpl_colors.rgb2hex(rgba)


@deprecated("This functionality is already in gromo module")
def calculate_true_positives(
    actual: torch.Tensor, predicted: torch.Tensor, label: int
) -> tuple[float, float, float]:
    """Calculate true positives, false positives and false negatives of a specific label

    Parameters
    ----------
    actual : torch.Tensor
        true labels
    predicted : torch.Tensor
        predicted labels
    label : int
        target label to calculate metrics

    Returns
    -------
    tuple[float, float, float]
        true positives, false positives, false negatives
    """
    true_positives = torch.sum((actual == label) & (predicted == label)).item()
    false_positives = torch.sum((actual != label) & (predicted == label)).item()
    false_negatives = torch.sum((predicted != label) & (actual == label)).item()

    return true_positives, false_positives, false_negatives


@deprecated("This functionality is already in gromo module")
def f1(actual: torch.Tensor, predicted: torch.Tensor, label: int) -> float:
    """Calculate f1 score of specific label

    Parameters
    ----------
    actual : torch.Tensor
        true labels
    predicted : torch.Tensor
        predicted labels
    label : int
        target label to calculate f1 score

    Returns
    -------
    float
        f1 score of label
    """
    # F1 = 2 * (precision * recall) / (precision + recall)
    tp, fp, fn = calculate_true_positives(actual, predicted, label)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


@deprecated("This functionality is already in gromo module")
def f1_micro(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    """Calculate f1 score with micro average

    Parameters
    ----------
    actual : torch.Tensor
        true labels
    predicted : torch.Tensor
        predicted labels

    Returns
    -------
    float
        micro-average f1 score
    """
    true_positives, false_positives, false_negatives = {}, {}, {}
    for label in np.unique(actual):
        tp, fp, fn = calculate_true_positives(actual, predicted, label)
        true_positives[label] = tp
        false_positives[label] = fp
        false_negatives[label] = fn

    all_true_positives = np.sum(list(true_positives.values()))
    all_false_positives = np.sum(list(false_positives.values()))
    all_false_negatives = np.sum(list(false_negatives.values()))

    micro_precision = all_true_positives / (all_true_positives + all_false_positives)
    micro_recall = all_true_positives / (all_true_positives + all_false_negatives)

    f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

    return f1


@deprecated("This functionality is already in gromo module")
def f1_macro(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    """Calculate f1 score with macro average

    Parameters
    ----------
    actual : torch.Tensor
        true labels
    predicted : torch.Tensor
        predicted labels

    Returns
    -------
    float
        macro-average f1 score
    """
    return float(np.mean([f1(actual, predicted, label) for label in np.unique(actual)]))


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    with_f1score: bool = False,
) -> tuple[float, float] | tuple[float, float, float]:
    """Evaluate network on batch

    Important: Assumes that the batch is already on the correct device

    Parameters
    ----------
    model : torch.nn.Module
        network to evaluate
    x : torch.Tensor
        input features tensor
    y : torch.Tensor
        true labels tensor
    loss_fn : Callable
        loss function for bottleneck calculation
    with_f1score : bool, optional
        calculate f1-score, by default False

    Returns
    -------
    tuple[float, float] | tuple[float, float, float]
        accuracy and loss, optionally f1-score
    """
    model.eval()
    with torch.no_grad():
        pred = model(x)
        loss = loss_fn(pred, y)

    if model.out_features > 1 and y.dim() == 1:
        final_pred = pred.argmax(axis=1)
        correct = (final_pred == y).int().sum()
        accuracy = (correct / pred.shape[0]).item()
    else:
        accuracy = -1

    if with_f1score:
        if model.out_features > 1 and y.dim() == 1:
            f1score = f1_micro(y.cpu(), final_pred.cpu())
        else:
            f1score = -1
        return accuracy, loss.item(), f1score

    return accuracy, loss.item()


@deprecated("This functionality is already in gromo module")
def evaluate_dataset(
    model: nn.Module, dataloader: DataLoader, loss_fn: Callable
) -> tuple[float, float]:
    """Evaluate network on dataset

    Parameters
    ----------
    model : torch.nn.Module
        network to evaluate
    dataloader : DataLoader
        dataloader containing the data
    loss_fn : Callable
        loss function for bottleneck calculation

    Returns
    -------
    tuple[float, float]
        accuracy and loss
    """
    model.eval()
    correct, total = 0, 0

    loss = []
    for x, y in dataloader:
        x = x.to(model.device)
        y = y.to(model.device)
        with torch.no_grad():
            pred = model(x)
            loss.append(loss_fn(pred, y).item())

        if model.out_features > 1 and y.dim() == 1:
            final_pred = pred.argmax(axis=1)
            count_this = final_pred == y
            count_this = count_this.sum()

            correct += count_this.item()
            total += len(pred)

    return (correct / total), np.mean(loss).item()


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def set_random_seeds(
    seed: int,
    device: torch.device,
    cudnn_deterministic: bool = False,
    cudnn_benchmark: bool = False,
) -> None:
    """
    Set all random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value
    device : torch.device
        Device being used (for CUDA-specific settings)
    cudnn_deterministic : bool, optional
        Whether to set cudnn.deterministic to True, by default False
    cudnn_benchmark : bool, optional
        Whether to set cudnn.benchmark to True, by default False
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        if cudnn_deterministic:
            torch.backends.cudnn.deterministic = True
        if cudnn_benchmark:
            torch.backends.cudnn.benchmark = True


def none_constructor(*args, **kwargs):
    """A constructor that does nothing and returns None."""
    return None
