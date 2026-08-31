import math

import matplotlib.pyplot as plt
import torch

from experiments.auxilliary_functions import compute_statistics
from tools.models import get_model_from_config


def gather_statistics_and_update(
    model, dataloader, loss_fn, batch_limit, maximum_added_neurons, zero_delta
):
    # Gathering growing statistics
    original_device = global_device()
    print(f"Original device: {original_device}")
    stat_loss, stat_acc = compute_statistics(
        model, dataloader, loss_fn, batch_limit=batch_limit
    )
    print(f"Training Loss after gathering statistics: {stat_loss: .6f}")

    # switch to cpu
    original_device = global_device()
    set_device("cpu")
    device = global_device()
    model.to(device=device)
    print(f"Model moved to device: {device}")

    # Compute the optimal update
    growing_dtype = torch.get_default_dtype()
    model.compute_optimal_updates(
        maximum_added_neurons=maximum_added_neurons,
        dtype=growing_dtype,
        use_projected_gradient=not zero_delta,
    )

    # switch back to the original device
    set_device(original_device)
    model.to(device=original_device)

    # Select the best update
    model.select_best_update()

    return stat_loss, stat_acc


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

    # Compute the growing update
    print(f"Computing the growing update")
    keep_neurons = 10
    part = "all"
    batch_limit = int(len(val_loader) * 1.0)
    stat_loss, stat_acc = gather_statistics_and_update(
        model,
        val_loader,
        loss_fn_growth,
        batch_limit=batch_limit,
        maximum_added_neurons=keep_neurons,
        zero_delta=(part == "neurons"),
    )
    model.reset_computation()
    if model.currently_updated_layer.eigenvalues_extension is not None:
        print(
            f"Number of added neurons: {model.currently_updated_layer.eigenvalues_extension.size(0)}"
        )
    else:
        print("No neurons added")

    # Line search on the scaling factor
    print("Line search on the scaling factor")
    batch_limit = int(len(val_loader) * 1.0)
    range_scaling_factor = 5.0
    nb_points = 31
    gammas = torch.linspace(0, range_scaling_factor, nb_points)
    losses = []
    test_losses = []
    for gamma in gammas:
        start_time = time.time()
        model.currently_updated_layer.scaling_factor = gamma.unsqueeze(0).to(device)
        loss, _ = extended_evaluate_model(
            model, val_loader, loss_fn_growth, batch_limit=batch_limit, device=device
        )
        test_loss, _ = extended_evaluate_model(
            model, test_loader, loss_fn_growth, batch_limit=batch_limit, device=device
        )
        iter_time = time.time() - start_time
        print(
            f"Gamma: {gamma: .2f}, Val loss: {loss: .6f}, Test loss: {test_loss: .6f}, Time: {iter_time: .2f}s"
        )
        losses.append(loss)
        test_losses.append(test_loss)

    plt.plot(gammas, losses, marker="o", label="Validation loss")
    plt.plot(gammas, test_losses, marker="o", label="Test loss")
    plt.axhline(y=stat_loss, color="r", linestyle="--", label="Initial validation loss")
    plt.plot(
        0,
        stat_loss,
        marker="*",
        color="r",
        markersize=15,
        label="Initial validation point",
    )
    plt.xlabel("Scaling factor")
    plt.ylabel("Loss")
    plt.title("Line search on the scaling factor")
    plt.legend()
    plt.show()

    # Convert the list to a Tensor
    losses_tensor = torch.tensor(losses)

    # Use torch.argmin on the Tensor
    best_gamma = gammas[torch.argmin(losses_tensor)]
    print(f"Best scaling factor: {best_gamma: .2f}")
    model.currently_updated_layer.scaling_factor = best_gamma.unsqueeze(0).to(device)
    loss, _ = extended_evaluate_model(
        model, test_loader, loss_fn_growth, batch_limit=-1, device=device
    )
    print(f"Best loss: {loss: .6f}")
