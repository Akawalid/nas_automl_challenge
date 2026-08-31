import matplotlib.pyplot as plt
import torch
from gromo.utils.utils import global_device, set_device
from tqdm import tqdm

from experiments.auxilliary_functions import AverageMeter, compute_statistics
from experiments.compare_activities_and_gradients import compare_activation_patches
from experiments.estimate_convergence import to_device
from tools.models import get_model_from_config


@torch.no_grad()
def compute_optimal_update(model, maximum_added_neurons):
    # switch to cpu
    original_device = global_device()
    set_device("cpu")
    to_device(model, device="cpu")

    # Compute the optimal update
    model.compute_optimal_updates(
        maximum_added_neurons=maximum_added_neurons,
        dtype=torch.get_default_dtype(),
    )

    # switch back to the original device
    set_device(original_device)
    to_device(model, device=original_device)


def save_update(model):
    def _save_update(growing_module):
        for attr in [
            "optimal_delta_layer",
            "extended_input_layer",
            "extended_output_layer",
            "eigenvalues_extension",
        ]:
            new_attr = f"{attr}_copy"
            setattr(growing_module, new_attr, getattr(growing_module, attr))

        attr = "extended_output_layer"
        new_attr = f"{attr}_copy"
        setattr(
            growing_module.previous_module,
            new_attr,
            getattr(growing_module.previous_module, attr),
        )

    for layer in model._growing_layers:
        _save_update(layer)


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


@torch.no_grad()
def compare_param_step(growing_module):
    input = growing_module.input.clone()
    output_partial = growing_module.optimal_delta_layer(input)
    output_full = growing_module.optimal_delta_layer_copy(input)
    return compare_activation_patches(output_partial, output_full)


@torch.no_grad()
def compare_neuron_step(growing_module):
    input = growing_module.previous_module.input.clone()
    alpha_activity = growing_module.previous_module.extended_output_layer(input)
    omega_activity = growing_module.extended_input_layer(alpha_activity)
    alpha_activity_copy = growing_module.previous_module.extended_output_layer_copy(input)
    omega_activity_copy = growing_module.extended_input_layer_copy(alpha_activity_copy)
    return compare_activation_patches(omega_activity, omega_activity_copy)


@torch.no_grad()
def compare_updates(model, dataloader, batch_limit):
    def init_model(model):
        for layer in model._growing_layers:
            layer.store_pre_activity = False

    def reset_model(model):
        for layer in model._growing_layers:
            layer.store_pre_activity = True

    init_model(model)

    param_steps = [
        {
            "norm ratio": AverageMeter(),
            "cosine similarity": AverageMeter(),
            "rmse": AverageMeter(),
            "relative rmse": AverageMeter(),
        }
        for _ in range(len(model._growing_layers))
    ]
    neuron_steps = [
        {
            "norm ratio": AverageMeter(),
            "cosine similarity": AverageMeter(),
            "rmse": AverageMeter(),
            "relative rmse": AverageMeter(),
        }
        for _ in range(len(model._growing_layers))
    ]

    for i, (x, _) in enumerate(dataloader):
        if batch_limit >= 0 and i >= batch_limit:
            break

        x = x.to(device)
        _ = model(x)

        for k, (layer, param_step, neuron_step) in enumerate(
            zip(model._growing_layers, param_steps, neuron_steps)
        ):
            param_step = compare_param_step(layer)
            neuron_step = compare_neuron_step(layer)

            param_steps[k]["norm ratio"].update(param_step["norm ratio"], n=x.shape[0])
            param_steps[k]["cosine similarity"].update(
                param_step["cosine similarity"], n=x.shape[0]
            )
            param_steps[k]["rmse"].update(param_step["rmse"], n=x.shape[0])
            param_steps[k]["relative rmse"].update(
                param_step["relative rmse"], n=x.shape[0]
            )

            neuron_steps[k]["norm ratio"].update(neuron_step["norm ratio"], n=x.shape[0])
            neuron_steps[k]["cosine similarity"].update(
                neuron_step["cosine similarity"], n=x.shape[0]
            )
            neuron_steps[k]["rmse"].update(neuron_step["rmse"], n=x.shape[0])
            neuron_steps[k]["relative rmse"].update(
                neuron_step["relative rmse"], n=x.shape[0]
            )

    for layer, param_step, neuron_step in zip(
        model._growing_layers, param_steps, neuron_steps
    ):
        layer.param_steps.append(param_step)
        layer.neuron_steps.append(neuron_step)
    reset_model(model)


def estimate_functional_convergence(
    model, dataloader, eval_loader, loss_fn, batch_limit, maximum_added_neurons, device
):
    # compute the full update
    gather_statistics_and_updates(
        model, dataloader, loss_fn, batch_limit, maximum_added_neurons
    )

    for layer in model._growing_layers:
        layer.param_steps = []
        layer.neuron_steps = []

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
        compare_updates(model, eval_loader, batch_limit)


def plot_convergence(model):
    # plot the steps
    for k, layer in enumerate(model._growing_layers):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        for i, ax in enumerate(axs.flatten()):
            attr = ["norm ratio", "cosine similarity", "rmse", "relative rmse"][i]
            if attr == "norm ratio":
                param_steps = [
                    step[attr].avg.log10().item() for step in layer.param_steps
                ]
                neuron_steps = [
                    step[attr].avg.log10().item() for step in layer.neuron_steps
                ]
            else:
                param_steps = [step[attr].avg.item() for step in layer.param_steps]
                neuron_steps = [step[attr].avg.item() for step in layer.neuron_steps]
            ax.plot(param_steps, label="Param step")
            ax.plot(neuron_steps, label="Neuron step")
            ax.set_title(f"{attr.capitalize()}: Layer {k}")
            ax.set_ylim(-0.2, 1.2)
            ax.legend()
        # plt.tight_layout()
        fig.suptitle(f"Functional Convergence: Layer {k}", fontsize=16)
        plt.savefig(
            f"functional_convergence_layer_{k}.pdf", format="pdf", bbox_inches="tight"
        )
        plt.show()


if __name__ == "__main__":
    import os

    import yaml
    from auxilliary_functions import evaluate_model, topk_accuracy
    from torch import nn

    from tools.datasets import get_dataloaders
    from tools.models import get_model_from_config

    # Define the device
    set_device("mps")  # 'cuda', 'cpu' or 'mps'
    device = global_device()

    # Fix the random seed
    torch.manual_seed(1)

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
    batch_limit = -1
    print(f"Number of batches: {len(train_loader)} (batch limit: {batch_limit})")

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
    epoch = 10
    checkpoint_path = os.path.join(model_dir, f"epoch_{epoch}.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(
            torch.load(checkpoint_path, weights_only=False)["model_state_dict"]
        )
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
    estimate_functional_convergence(
        model,
        train_loader,
        test_loader,
        loss_fn_growth,
        batch_limit=batch_limit,
        maximum_added_neurons=8,
        device=device,
    )
    plot_convergence(model)
