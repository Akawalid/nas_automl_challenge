import math

import matplotlib.pyplot as plt
import torch
from gromo.containers.growing_container import GrowingContainer

from experiments.auxilliary_functions import compute_statistics


def gather_statistics_and_update(
    model: GrowingContainer,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    batch_limit: int,
    maximum_added_neurons: int,
    zero_delta: bool,
):
    # Gathering growing statistics
    stat_loss, stat_acc = compute_statistics(
        model, dataloader, loss_fn, batch_limit=batch_limit
    )
    print(f"Training Loss after gathering statistics: {stat_loss: .6f}")

    # Compute the optimal update
    growing_dtype = torch.get_default_dtype()
    model.compute_optimal_updates(
        maximum_added_neurons=maximum_added_neurons,
        dtype=growing_dtype,
        use_projected_gradient=not zero_delta,
    )

    # Select the best update
    model.select_best_update()

    return stat_loss, stat_acc


def get_param_step_activity(layer):
    if hasattr(layer, "optimal_delta_layer") and torch.any(
        layer.optimal_delta_layer.weight
    ):
        x = layer.input
        param_step_activity = layer.optimal_delta_layer(x)
        linear_factor = layer.scaling_factor**2
        return -param_step_activity * linear_factor
    else:
        return torch.zeros_like(layer.pre_activity)


def get_linear_neuron_activity(layer):
    if (
        hasattr(layer, "extended_input_layer")
        and getattr(layer, "extended_input_layer") is not None
        and hasattr(layer.previous_module, "extended_output_layer")
        and getattr(layer.previous_module, "extended_output_layer") is not None
    ):
        previous_module = layer.previous_module
        x = previous_module.input
        alpha_activity = previous_module.extended_output_layer(x) * layer.scaling_factor
        omega_activity = layer.extended_input_layer(alpha_activity) * layer.scaling_factor
        return omega_activity
    else:
        return torch.zeros_like(layer.pre_activity)


def compare_activation_patches(a1, a2):
    rmse = (a1 - a2).norm(dim=-1).mean()
    results = {
        "norm ratio": a1.norm(dim=-1).mean() / a2.norm(dim=-1).mean(),
        "cosine similarity": torch.nn.functional.cosine_similarity(a1, a2, dim=-1).mean(),
        "rmse": rmse,
        "relative rmse": rmse / a2.norm(dim=-1).mean(),
    }
    return results


def compare_activities(function_gradient, param_step_activity, neuron_step_activity):
    full_activity = param_step_activity + neuron_step_activity
    desired_update = -function_gradient - param_step_activity
    results = {
        "norm": {
            "param": param_step_activity.norm(dim=-1).mean(),
            "neuron": neuron_step_activity.norm(dim=-1).mean(),
            "fg": function_gradient.norm(dim=-1).mean(),
            "du": desired_update.norm(dim=-1).mean(),
            "full": full_activity.norm(dim=-1).mean(),
        },
        "param_fg": compare_activation_patches(param_step_activity, -function_gradient),
        "neuron_du": compare_activation_patches(neuron_step_activity, desired_update),
        "neuron_fg": compare_activation_patches(neuron_step_activity, -function_gradient),
        "param_du": compare_activation_patches(param_step_activity, desired_update),
        "full_fg": compare_activation_patches(full_activity, -function_gradient),
        "param_neuron": compare_activation_patches(
            param_step_activity, neuron_step_activity
        ),
        "fg_du": compare_activation_patches(-function_gradient, desired_update),
    }
    return results


def create_running_results():
    results = {
        "norm": {key: [] for key in ["param", "neuron", "fg", "du", "full"]},
        **{
            key: {
                metric: []
                for metric in ["norm ratio", "cosine similarity", "rmse", "relative rmse"]
            }
            for key in [
                "param_fg",
                "neuron_du",
                "neuron_fg",
                "param_du",
                "full_fg",
                "param_neuron",
                "fg_du",
            ]
        },
    }
    return results


def update_results(running_results, results):
    for key in running_results.keys():
        if key == "norm":
            for sub_key in running_results[key].keys():
                running_results[key][sub_key].append(results[key][sub_key])
        else:
            for metric in running_results[key].keys():
                running_results[key][metric].append(results[key][metric])
    return running_results


def init_layer(layer):
    layer.store_input = True
    layer.store_pre_activity = True
    layer.previous_module.store_input = True


def reset_layer(layer):
    layer.store_input = False
    layer.store_pre_activity = False
    layer.previous_module.store_input = False


def summarize_activity_comparison(model, x, y, loss_fn):
    # Set up the model
    model.zero_grad()
    updated_layer = model.currently_updated_layer
    init_layer(updated_layer)

    # Without extension
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()

    # Get the activities and the functional gradient
    functional_gradient = updated_layer.pre_activity.grad.clone()
    param_step_activity = get_param_step_activity(updated_layer)
    neuron_activity = get_linear_neuron_activity(updated_layer)

    # Compare the activities
    results = compare_activities(
        functional_gradient, param_step_activity, neuron_activity
    )

    # Reset the model
    reset_layer(updated_layer)

    return results


def compare_activities_and_gradients(model, dataloader, loss_fn, batch_limit):
    running_results = create_running_results()
    for i, (x, y) in enumerate(dataloader):
        if i == batch_limit:
            break

        results = summarize_activity_comparison(model, x, y, loss_fn)
        running_results = update_results(running_results, results)
    return running_results


def display_results(running_results):
    # convert running results to tensors
    for key in running_results.keys():
        if key == "norm":
            for sub_key in running_results[key].keys():
                running_results[key][sub_key] = torch.tensor(
                    running_results[key][sub_key]
                )
        else:
            for metric in running_results[key].keys():
                running_results[key][metric] = torch.tensor(running_results[key][metric])

    # Display the results
    for key in running_results.keys():
        print(f"{key}:")
        if key == "norm":
            for sub_key in running_results[key].keys():
                print(f"\t{sub_key}: {running_results[key][sub_key].mean(): .6f}")
        else:
            for metric in running_results[key].keys():
                print(f"\t{metric}: {running_results[key][metric].mean(): .6f}")
        print()

    # Plot histograms for each of the metrics
    for key in running_results.keys():
        sub_keys = list(running_results[key].keys())
        num_sub_keys = len(sub_keys)
        num_cols = min(5, num_sub_keys)
        num_rows = math.ceil(num_sub_keys / num_cols)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 10))
        fig.suptitle(f"{key} metrics", fontsize=16)
        for i, sub_key in enumerate(sub_keys):
            ax = axs[i // num_cols, i % num_cols] if num_rows > 1 else axs[i]
            ax.hist(running_results[key][sub_key].numpy(), bins=30)
            ax.set_title(f"{sub_key}")
            ax.grid(True)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()


if __name__ == "__main__":
    import os
    import time

    import yaml
    from auxilliary_functions import evaluate_model, topk_accuracy, train
    from gromo.utils.utils import global_device, set_device
    from schedulers import get_scheduler
    from torch import nn

    from tools.datasets import get_dataloaders
    from tools.models import get_model_from_config

    # Define the device
    set_device("cpu")  # 'cuda', 'cpu' or 'mps'
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
    batch_size = 128
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
    model_name = "residual_mlp"
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
    epoch = 0
    checkpoint_path = os.path.join(model_dir, f"epoch_{epoch}.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    else:
        raise Exception("Checkpoint not found")

    # Define the losses
    loss_fn_train = nn.CrossEntropyLoss(reduction="mean")
    loss_fn_growth = nn.CrossEntropyLoss(reduction="sum")

    # Training loss after training
    train_loss_after, train_acc_after = evaluate_model(
        model, train_loader, loss_fn_train, topk_accuracy, batch_limit=-1, device=device
    )
    print(
        f"Training loss: {train_loss_after: .6f}, Training accuracy: {train_acc_after * 100: 2.1f}%"
    )

    # Compute the update
    print(f"Computing the parameter update")
    keep_neurons = 10
    part = "all"
    stat_loss, stat_acc = gather_statistics_and_update(
        model,
        train_loader,
        loss_fn_growth,
        batch_limit=-1,
        maximum_added_neurons=keep_neurons,
        use_projected_gradient=not (part == "neurons"),
    )
    model.reset_computation()
    if model.currently_updated_layer.eigenvalues_extension is not None:
        print(
            f"Number of added neurons: {model.currently_updated_layer.eigenvalues_extension.size(0)}"
        )
    else:
        print("No neurons added")

    # Compare activities and gradients
    print("Comparing the activities and gradients")
    model.currently_updated_layer.scaling_factor = 1.0
    running_results = compare_activities_and_gradients(
        model, train_loader, loss_fn_growth, batch_limit=-1
    )
    display_results(running_results)
