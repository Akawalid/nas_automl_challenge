import math

import matplotlib.pyplot as plt
import torch
from gromo.containers.growing_container import GrowingContainer

from experiments.auxilliary_functions import compute_statistics

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
    batch_size = 128
    num_workers = 2
    train_loader, val_loader, in_channels, image_size = get_dataloaders(
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

    # Define the losses
    loss_fn_train = nn.CrossEntropyLoss(reduction="mean")
    loss_fn_growth = nn.CrossEntropyLoss(reduction="sum")

    # Define the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, betas=(0.9, 0.99), weight_decay=5e-5
    )

    # define the scheduler
    num_epochs = 40
    scheduler = get_scheduler(
        scheduler_name="cosine",
        optimizer=optimizer,
        base_lr=0.001,
        warmup_epochs=5,
        num_epochs=300,
        num_batches_per_epoch=len(train_loader),
    )

    # Define the path to save the model
    model_dir = os.path.join("models", dataset_name, model_name)
    os.makedirs(model_dir, exist_ok=True)

    def save_training_state(model, optimizer, epoch, training_loss):
        checkpoint_path = os.path.join(model_dir, f"epoch_{epoch}.pth")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "training_loss": training_loss,
            },
            checkpoint_path,
        )

    def retrieve_checkpoint(epoch):
        checkpoint_path = None
        for epoch_test in range(epoch, -1, -1):
            checkpoint_test = os.path.join(model_dir, f"epoch_{epoch_test}.pth")
            if os.path.exists(checkpoint_test):
                checkpoint_path = checkpoint_test
                break
        if checkpoint_path is None:
            checkpoint_test = os.path.join(model_dir, f"checkpoint.pth")
            if os.path.exists(checkpoint_test):
                checkpoint_path = checkpoint_test
        if checkpoint_path is None:
            return None

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint["epoch"] > epoch:
            return None
        else:
            return checkpoint

    # Check if the checkpoint exists
    checkpoint = retrieve_checkpoint(num_epochs - 1)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        scheduler.current_epoch = start_epoch
        train_loss_init = checkpoint["training_loss"]
        print(f"Model loaded from checkpoint, resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
        print(f"Model not found, starting from scratch")

    # Regular training
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        train_loss, train_acc = train(
            model=model,
            train_dataloader=train_loader,
            optimizer=optimizer,
            loss_function=loss_fn_train,
            aux_loss_function=topk_accuracy,
            scheduler=scheduler,
            device=device,
        )
        end_time = time.time()
        epoch_time = end_time - start_time
        save_training_state(model, optimizer, epoch, train_loss)
        print(
            f"Epoch {epoch}, Training Loss: {train_loss: .6f}, Training Accuracy: {train_acc * 100: 2.1f}%, Time: {epoch_time: .2f}"
        )

    # Training loss after training
    train_loss_after, train_acc_after = evaluate_model(
        model, train_loader, loss_fn_train, topk_accuracy, batch_limit=-1, device=device
    )
    print(
        f"Final training loss: {train_loss_after: .6f}, Final training accuracy: {train_acc_after * 100: 2.1f}%"
    )
