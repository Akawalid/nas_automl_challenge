import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from experiments.auxilliary_functions import compute_statistics
from tools.models import get_model_from_config


def _save_update(layer):
    for attr in [
        "optimal_delta_layer",
        "extended_input_layer",
        "extended_output_layer",
        "eigenvalues_extension",
    ]:
        new_attr = f"{attr}_copy"
        setattr(layer, new_attr, getattr(layer, attr))

    attr = "extended_output_layer"
    new_attr = f"{attr}_copy"
    setattr(layer.previous_module, new_attr, getattr(layer.previous_module, attr))


def save_update(model):
    for layer in model._growing_layers:
        _save_update(layer)


def compare_param_step(layer):
    value = torch.cosine_similarity(
        layer.optimal_delta_layer.weight.flatten(),
        layer.optimal_delta_layer_copy.weight.flatten(),
        dim=0,
    )
    return value.item()


def compare_eigenvalues(layer):
    eigenvalues_diff = layer.eigenvalues_extension - layer.eigenvalues_extension_copy
    factor = torch.norm(layer.eigenvalues_extension_copy)
    value = torch.norm(eigenvalues_diff) / factor
    return value.item()


def compare_extended_input(layer):
    extended_input_diff = torch.cosine_similarity(
        layer.extended_input_layer.weight.flatten(),
        layer.extended_input_layer_copy.weight.flatten(),
        dim=0,
    )
    return extended_input_diff.item()


def compare_extended_output_layer(layer):
    extended_output_diff = torch.cosine_similarity(
        layer.previous_module.extended_output_layer.weight.flatten(),
        layer.previous_module.extended_output_layer_copy.weight.flatten(),
        dim=0,
    )
    return extended_output_diff.item()


def plot_correlation(layer):
    partial_weights = layer.previous_module.extended_output_layer.weight.detach().T
    full_weights = layer.previous_module.extended_output_layer_copy.weight.detach().T
    # normalize weights
    partial_weights = partial_weights / torch.norm(partial_weights, dim=0, keepdim=True)
    full_weights = full_weights / torch.norm(full_weights, dim=0, keepdim=True)

    cross_correlation = partial_weights.T @ full_weights
    print("Cross correlation matrix for layer", layer.name)
    print(cross_correlation)
    print("Partial weights norm:", torch.norm(partial_weights, dim=0))
    print("Full weights norm:", torch.norm(full_weights, dim=0))
    print(f"Eigenvalues extension: {layer.eigenvalues_extension}")
    print(f"Eigenvalues extension copy: {layer.eigenvalues_extension_copy}")

    plt.title(f"Cross correlation matrix for layer {layer.name}")
    plt.imshow(cross_correlation, aspect="auto", cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    plt.show()


def to_device(model, device):
    model.to(device=device)
    for layer in model._growing_layers:
        layer.to(device=device)
        if (
            hasattr(layer, "optimal_delta_layer_copy")
            and layer.optimal_delta_layer_copy is not None
        ):
            layer.optimal_delta_layer_copy.to(device=device)
        if (
            hasattr(layer, "extended_input_layer_copy")
            and layer.extended_input_layer_copy is not None
        ):
            layer.extended_input_layer_copy.to(device=device)
        if (
            hasattr(layer.previous_module, "extended_output_layer_copy")
            and layer.previous_module.extended_output_layer_copy is not None
        ):
            layer.previous_module.extended_output_layer_copy.to(device=device)


def compute_optimal_update(model, maximum_added_neurons):
    # switch to cpu
    original_device = global_device()
    set_device("cpu")
    to_device(model, device="cpu")

    # Compute the optimal update
    with torch.no_grad():
        model.compute_optimal_updates(
            maximum_added_neurons=maximum_added_neurons,
            dtype=torch.get_default_dtype(),
        )

    # switch back to the original device
    set_device(original_device)
    to_device(model, device=original_device)


def gather_statistics_and_updates(
    model, dataloader, loss_fn, batch_limit, maximum_added_neurons
):
    # Gathering growing statistics
    original_device = global_device()
    print(f"Original device: {original_device}")
    stat_loss, stat_acc = compute_statistics(
        model, dataloader, loss_fn, batch_limit=batch_limit
    )
    print(f"Training Loss after gathering statistics: {stat_loss: .6f}")

    # Compute the optimal update
    compute_optimal_update(model, maximum_added_neurons)

    # save the updates
    save_update(model)

    return stat_loss, stat_acc


def estimate_convergence(
    model, dataloader, loss_fn, batch_limit, maximum_added_neurons, device
):
    # compute the full update
    gather_statistics_and_updates(
        model, dataloader, loss_fn, batch_limit, maximum_added_neurons
    )

    for layer in model._growing_layers:
        layer.param_steps = []
        layer.eigenvalue_steps = []
        layer.extended_input_steps = []
        layer.extended_output_steps = []

    model.init_computation()
    for i, (x, y) in enumerate(tqdm(dataloader)):
        if batch_limit >= 0 and i >= batch_limit:
            print(f"Batch limit {batch_limit} reached with {i} batches")
            break
        model.zero_grad()
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        model.update_computation()
        compute_optimal_update(model, maximum_added_neurons)

        for layer in model._growing_layers:
            layer.param_steps.append(compare_param_step(layer))
            layer.eigenvalue_steps.append(compare_eigenvalues(layer))
            layer.extended_input_steps.append(compare_extended_input(layer))
            layer.extended_output_steps.append(compare_extended_output_layer(layer))


def plot_convergence(model):
    # plot the steps
    for i, layer in enumerate(model._growing_layers):
        param_steps = layer.param_steps
        # eigenvalue_steps = layer.eigenvalue_steps
        extended_input_steps = layer.extended_input_steps
        extended_output_steps = layer.extended_output_steps
        plt.figure(figsize=(6, 4))
        plt.plot(param_steps, label="Param step")
        # plt.plot(eigenvalue_steps, label="Eigenvalue step")
        plt.plot(extended_input_steps, label="Extended input step")
        plt.plot(extended_output_steps, label="Extended output step")
        plt.title(f"Convergence: Layer {i}")
        plt.legend()
        plt.savefig(f"convergence_layer_{i}.pdf", format="pdf", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    import os

    import yaml
    from auxilliary_functions import evaluate_model, topk_accuracy
    from gromo.utils.utils import global_device, set_device
    from torch import nn

    from tools.datasets import get_dataloaders
    from tools.models import get_model_from_config

    # Define the device
    set_device("mps")  # 'cuda', 'cpu' or 'mps'
    device = global_device()

    # Fix the random seed
    torch.manual_seed(0)

    # Set the default data type
    if device.type != "mps":
        torch.set_default_dtype(torch.float32)

    # Define the dataset
    dataset_name = "cifar10"
    dataset_path = "dataset"
    num_classes = 100 if dataset_name == "cifar100" else 10
    split_train_val = 0.0
    data_augmentation = None
    batch_size = 2048
    num_workers = 2
    train_loader, val_loader, test_loader, image_size = get_dataloaders(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        nb_class=num_classes,
        split_train_val=split_train_val,
        data_augmentation=data_augmentation,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        shuffle=False,
    )
    print(f"Number of batches: {len(train_loader)}")

    # Load the YAML configuration file
    model_name = "mlp_mixer"
    config_path = os.path.join("models", "configs", f"{model_name}.yml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Define the model
    model = get_model_from_config(
        input_shape=image_size,
        out_features=num_classes,
        config=config,
    )
    model.to(device=device)
    print(model)

    # Retrieve the checkpoint
    model_dir = os.path.join("models", dataset_name, model_name)
    epoch = 40
    checkpoint_path = os.path.join(model_dir, f"epoch_{epoch-1}.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    else:
        raise Exception("Checkpoint not found")

    # Define the losses
    loss_fn_train = nn.CrossEntropyLoss(reduction="mean")
    loss_fn_growth = nn.CrossEntropyLoss(reduction="sum")

    # Evaluate the model after training
    train_loss_after, train_acc_after = evaluate_model(
        model, train_loader, loss_fn_train, topk_accuracy, batch_limit=-1, device=device
    )
    val_loss_after, val_acc_after = evaluate_model(
        model, val_loader, loss_fn_train, topk_accuracy, batch_limit=-1, device=device
    )
    test_loss_after, test_acc_after = evaluate_model(
        model, test_loader, loss_fn_train, topk_accuracy, batch_limit=-1, device=device
    )
    print(
        f"Final train loss: {train_loss_after: .6f}, Final val loss: {val_loss_after: .6f}, Final test loss: {test_loss_after: .6f}"
    )

    # Plot the convergence
    estimate_convergence(
        model,
        train_loader,
        loss_fn_growth,
        batch_limit=-1,
        maximum_added_neurons=1,
        device=device,
    )
    plot_convergence(model)
