"""
Hydra-based Training and Growing Script
=========================================

This script implements the train-and-grow procedure using Hydra for configuration
management and MLflow for experiment tracking.

The general process follows:
1. Pre-processing: Load datasets, initialize model, set up optimizer
2. Training Loop: Alternate between growth steps and standard training steps
3. Post-processing: Log final metrics and save model

Run with:
    python train_and_grow.py
    python train_and_grow.py model=resnet18 dataset=cifar10
    python train_and_grow.py optimizer.lr=0.1 training.nb_step=100
"""

import inspect
import logging
import traceback

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from hydra_script.mlflow_tools import SafeMLflowClient, warn_if_sqlite_tracking_uri


@hydra.main(version_base=None, config_path="configs", config_name="debug")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra and MLflow."""
    logger = logging.getLogger(__name__)

    try:
        _run_experiment(cfg, logger)
    except Exception as e:
        logger.error(f"Experiment failed with exception: {e}")
        logger.error(traceback.format_exc())
        # Also print to stderr for submitit logs
        print(f"ERROR: Experiment failed with exception: {e}", flush=True)
        traceback.print_exc()
        raise e


def _run_trainning(
    cfg: DictConfig,
    logger: logging.Logger,
    device,
) -> None:
    from time import time
    from typing import Any, Literal

    import mlflow
    import torch

    # Importing torchmetrics here to ensure it's available when instantiating metrics from the config
    import torchmetrics  # noqa: F401
    from gromo.containers.sequential_growing_container import SequentialGrowingModel
    from gromo.utils.training_utils import evaluate_model
    from hydra.utils import instantiate

    import tools.metrics  # noqa: F401
    from hydra_script.aux_train_and_grow import (
        EarlyStopping,
        build_optimizer,
        create_dataloaders,
        create_scheduler,
        generate_growth_schedule_if_needed,
        initialize_model,
        log_layers_metrics,
        perform_training_step,
        resolve_batch_limit,
        save_model,
    )
    from hydra_script.perfom_growth_step import perform_growth_step
    from tools.create_growth_schedule import BatchLimit

    safe_mlflow = SafeMLflowClient(logger=logger)
    safe_log_metric = safe_mlflow.log_metric

    start_time = time()
    # Get number of classes: prefer used_classes (subset) over full dataset count
    num_classes = cfg.dataset_config.get("used_classes", None)
    if num_classes is None:
        from tools.datasets import get_num_classes

        num_classes = get_num_classes(cfg.dataset_config.name)

    # Create dataloaders
    growth_loader, train_loader, val_loader, test_loader, input_shape = (
        create_dataloaders(cfg, device)
    )

    if cfg.dataset_config.split_train_val == 0.0:
        logger.warning("Validation set size is 0. Skipping validation during training.")
        validation_set_available = False
    else:
        validation_set_available = True

    # Log dataset to MLflow
    # log_dataset_to_mlflow(cfg, train_loader, val_loader, test_loader, input_shape)
    safe_mlflow.set_tag("dataset.input_shape", str(input_shape))
    safe_mlflow.set_tag("dataset.num_classes", str(num_classes))
    safe_mlflow.set_tag("dataset.name", cfg.dataset_config.name)

    # Initialize model
    model = initialize_model(cfg, input_shape, num_classes, device)

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    safe_mlflow.log_param("total_parameters", total_params)
    safe_mlflow.log_param("trainable_parameters", trainable_params)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Initialize optimizer (optionally excluding norm/bias from weight decay)
    optimizer: torch.optim.Optimizer = build_optimizer(cfg, model)
    logger.info(f"Optimizer: {optimizer}")

    logger.info("Early stopping configuration:")
    logger.info(cfg.early_stopping.early_stopping_object)
    early_stopper: EarlyStopping = instantiate(cfg.early_stopping.early_stopping_object)

    assert mlflow.active_run() is not None, "No active MLflow run"
    # Generate growth schedule if needed
    growth_schedule = generate_growth_schedule_if_needed(
        cfg,
        model,
        run_id=mlflow.active_run().info.run_id,  # type: ignore
    )
    total_nb_steps = max(
        cfg.training.get("nb_step", 0), len(growth_schedule) if growth_schedule else 0
    )

    # Initialize scheduler
    scheduler = create_scheduler(
        cfg, optimizer, len(train_loader), nb_steps=total_nb_steps
    )

    # Initialize loss functions
    loss_function_train = instantiate(cfg.loss, reduction="mean")
    loss_function_growth = instantiate(cfg.loss, reduction="sum")

    # Mixed-precision (AMP) scaler, shared across epochs. Only active on CUDA when
    # `general.amp` is set (matches the Compact-Transformers `amp: true` recipe).
    amp_enabled = bool(cfg.general.get("amp", False))
    amp_scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled and device.type == "cuda"
    )
    if amp_enabled and device.type != "cuda":
        logger.warning("general.amp=true but device is not CUDA; AMP is disabled.")

    # Define the metrics
    metrics = dict()
    for phase in ["train", "val", "test"]:
        try:
            expected_args = inspect.getfullargspec(eval(cfg.metric._target_)).args
        except NameError as e:
            raise ImportError(
                f"Failed to evaluate metric target '{cfg.metric._target_}'. "
                f"Make sure the target is correct and all necessary imports are available. "
                f"Original error: {e}"
            ) from e
        extra_args = {}
        for arg_name, arg_values in {
            "num_classes": num_classes,
            "device": device,
        }.items():
            if arg_name in expected_args:
                extra_args[arg_name] = arg_values
        metrics[phase] = instantiate(cfg.metric, **extra_args)

    # Evaluate initial model
    logger.info("Evaluating initial model...")
    train_loss, train_accuracy = evaluate_model(
        model=model,
        loss_function=loss_function_train,
        metrics=metrics["train"],
        dataloader=train_loader,
        device=device,
    )
    if validation_set_available:
        val_loss, val_accuracy = evaluate_model(
            model=model,
            loss_function=loss_function_train,
            metrics=metrics["val"],
            dataloader=val_loader,
            device=device,
        )
        if hasattr(metrics["val"], "extra_logs"):
            metrics["val"].extra_logs(safe_log_metric, prefix="val_extra_", step=0)
    else:
        val_loss, val_accuracy = -1.0, -1.0
    test_loss, test_accuracy = evaluate_model(
        model=model,
        loss_function=loss_function_train,
        metrics=metrics["test"],
        dataloader=test_loader,
        device=device,
    )
    if hasattr(metrics["test"], "extra_logs"):
        metrics["test"].extra_logs(safe_log_metric, prefix="test_extra_", step=0)

    # Log initial metrics
    logs: dict[str, Any] = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    if hasattr(model, "number_of_parameters"):
        logs["number_of_parameters"] = model.number_of_parameters()
    if hasattr(model, "weights_statistics"):
        logs["weights_statistics"] = model.weights_statistics()
    log_layers_metrics(layer_metrics=logs, step=0)

    logger.info(
        f"Epoch 0/{total_nb_steps} | "
        f"Loss: {train_loss:.4f} ({val_loss:.4f}) ({test_loss:.4f}), "
        f"Accuracy: {train_accuracy * 100:.2f}% ({val_accuracy * 100:.2f}%) ({test_accuracy * 100:.2f}%) "
        f"[Init]"
    )

    stop_growth = False

    # Determine if a step is a growth epoch
    def get_step_type(
        step: int,
    ) -> tuple[Literal["SGD", "growth", "completion"], BatchLimit]:
        nonlocal stop_growth
        if stop_growth:
            return "SGD", -1
        elif growth_schedule is not None:
            if step - 1 < len(growth_schedule):
                schedule_entry = growth_schedule[step - 1]
                return schedule_entry[0], schedule_entry[3]
            else:
                logger.warning(
                    f"Growth schedule exhausted at step {step}. No more growth will occur."
                )
                stop_growth = True
                return "SGD", -1
        elif cfg.growing.epochs_per_growth == -1:
            return "SGD", -1
        else:
            growth_warmup_steps = int(cfg.schedule.get("growth_warmup_steps", 0))
            if growth_warmup_steps < 0:
                raise ValueError(
                    f"schedule.growth_warmup_steps must be >= 0, got {growth_warmup_steps}"
                )
            if step <= growth_warmup_steps:
                return "SGD", -1
            adjusted_step = step - growth_warmup_steps
            if adjusted_step % (cfg.growing.epochs_per_growth + 1) == 0:
                return "growth", "hyper-parameter"
            return "SGD", -1

    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info("Starting training loop...")
    logger.info("=" * 80)

    last_updated_layer = -1
    best_val_accuracy = -1
    best_test_accuracy = -1
    termination_reason = None

    for step in range(1, total_nb_steps + 1):
        step_start_time = time()
        step_type, batch_limit_raw = get_step_type(step)
        if step_type == "growth":
            # ----- GROWTH EPOCH -----
            # Use growth_loader if available, otherwise use train_loader
            growth_dataloader = (
                growth_loader if growth_loader is not None else train_loader
            )
            logs, last_updated_layer = perform_growth_step(
                cfg=cfg,
                model=model,
                step=step,
                growth_schedule_step=(
                    growth_schedule[step - 1] if growth_schedule is not None else None
                ),
                dataloader=growth_dataloader,
                loss_function_growth=loss_function_growth,
                loss_function_train=loss_function_train,
                top_1_accuracy=metrics["train"],
                optimizer=optimizer,
                device=device,
                last_updated_layer=last_updated_layer,
                metric_prefix="growth_set" if growth_loader is not None else "train",
            )
            logs["epoch_type"] = "growth"
            logs["is_growth_step"] = 1

            if cfg.early_stopping.reset_on_growth_step:
                early_stopper.reset()

            # Re-initialize optimizer after growth
            optimizer = build_optimizer(cfg, model)
            scheduler.optimizer = optimizer
        elif step_type == "completion":
            # ----- COMPLETION EPOCH -----
            logger.info(f"Completing growth for all layers at step {step}.")
            logger.info(f"Model before completion:\n{model}")
            completion_kwargs = {
                "output_extension_init": cfg.growing.output_extension_init,
                "input_extension_init": cfg.growing.input_extension_init,
                "neuron_pairing": cfg.growing.get("neuron_pairing", None),
                "rescaling": cfg.growing.get("rescaling", None),
                "noise_ratio": cfg.growing.get("noise_ratio", 0.001),
            }
            if isinstance(model, SequentialGrowingModel):
                model.complete_growth(extension_kwargs=completion_kwargs)
            else:
                logger.warning(
                    "Completion step called but model is not SequentialGrowingModel."
                    "Hence completion could crash."
                )
                model.complete_growth(extension_kwargs=completion_kwargs)  # type: ignore

            logger.info(f"Completed growth for all layers at step {step}.")
            logger.info(f"Model after completion:\n{model}")

        elif step_type == "SGD":
            # ----- TRAINING EPOCH -----
            sgd_batch_limit: int | None = resolve_batch_limit(
                batch_limit_raw, len(train_loader)
            )

            logs = perform_training_step(
                cfg=cfg,
                model=model,
                step=step,
                train_loader=train_loader,
                loss_function_train=loss_function_train,
                top_1_accuracy=metrics["train"],
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                batch_limit=sgd_batch_limit,
                amp_scaler=amp_scaler,
            )
            logs["epoch_type"] = "training"
            logs["is_growth_step"] = 0
        else:
            raise ValueError(f"Unknown step type: {step_type}")

        # Evaluate on validation set
        if validation_set_available:
            val_loss, val_accuracy = evaluate_model(
                model=model,
                loss_function=loss_function_train,
                metrics=metrics["val"],
                dataloader=val_loader,
                device=device,
            )
            if hasattr(metrics["val"], "extra_logs"):
                metrics["val"].extra_logs(safe_log_metric, prefix="val_extra_", step=step)
        else:
            val_loss, val_accuracy = -1.0, -1.0
        logs["val_loss"] = val_loss
        logs["val_accuracy"] = val_accuracy

        # Evaluate on test set
        test_loss, test_accuracy = evaluate_model(
            model=model,
            loss_function=loss_function_train,
            metrics=metrics["test"],
            dataloader=test_loader,
            device=device,
        )
        if hasattr(metrics["test"], "extra_logs"):
            metrics["test"].extra_logs(safe_log_metric, prefix="test_extra_", step=step)

        logs["test_loss"] = test_loss
        logs["test_accuracy"] = test_accuracy
        logs["step_duration"] = time() - step_start_time

        # Log layer statistics
        if hasattr(model, "weights_statistics"):
            logs["weights_statistics"] = model.weights_statistics()
        if hasattr(model, "number_of_parameters"):
            logs["number_of_parameters"] = model.number_of_parameters()
        log_layers_metrics(layer_metrics=logs, step=step)

        train_accuracy_str = (
            f"{logs['train_accuracy'] * 100:.2f}%"
            if logs.get("train_accuracy", -1.0) >= 0
            else "-N/A- "
        )

        train_loss_str = (
            f"{logs['train_loss']:.4f}" if logs.get("train_loss", -1.0) >= 0 else "-N/A- "
        )

        logger.info(
            f"Epoch {step}/{total_nb_steps} | "
            f"Loss: {train_loss_str} ({logs['val_loss']:.4f}) ({test_loss:.4f}), "
            f"Accuracy: {train_accuracy_str} ({logs['val_accuracy'] * 100:.2f}%) ({test_accuracy * 100:.2f}%) "
            f"[{logs['epoch_type']}]"
        )

        # Save best model
        if validation_set_available and logs["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = logs["val_accuracy"]
            best_test_accuracy = test_accuracy
            logger.info(
                f"New best validation accuracy: {best_val_accuracy * 100:.2f}%, best test accuracy: {test_accuracy * 100:.2f}%"
            )
            safe_log_metric("best_val_accuracy", best_val_accuracy, step=step)
            safe_log_metric("best_test_accuracy", best_test_accuracy, step=step)
            safe_log_metric("best_step", step)
            if hasattr(metrics["val"], "extra_logs"):
                metrics["val"].extra_logs(
                    safe_log_metric, prefix="best_val_extra_", step=step
                )
            if hasattr(metrics["test"], "extra_logs"):
                metrics["test"].extra_logs(
                    safe_log_metric, prefix="best_test_extra_", step=step
                )

            # Save checkpoint
            save_model(
                model=model,
                model_name="best_model",
                sample_input=train_loader.dataset[0][0].unsqueeze(0),
                device=device,
            )

        # Early stopping
        if cfg.training.get("training_threshold") is not None:
            if logs["train_loss"] < cfg.training.training_threshold:
                termination_reason = f"training_threshold_reached (loss={logs['train_loss']:.6f} < {cfg.training.training_threshold})"
                logger.info(f"Training threshold reached at step {step}")
                break

        if early_stopper.check(logs[cfg.early_stopping.metric_name]):
            termination_reason = f"early_stopping (metric={cfg.early_stopping.metric_name}, value={logs[cfg.early_stopping.metric_name]:.6f})"
            logger.info(f"Early stopping triggered at step {step}")
            break
    else:
        # Loop completed without break
        termination_reason = f"completed_all_steps ({total_nb_steps} steps)"

    # Final evaluation and logging
    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    logger.info(f"Termination reason: {termination_reason}")
    logger.info(f"Final step: {step}/{total_nb_steps}")
    logger.info("=" * 80)

    total_duration = time() - start_time
    logger.info(f"Total duration: {total_duration:.2f}s")
    logger.info(f"Final model:\n{model}")

    # Log final metrics
    safe_mlflow.set_tag("termination_reason", termination_reason)
    safe_log_metric("final_step", step)
    if "train_loss" in logs:
        safe_log_metric("final_train_loss", logs["train_loss"])
    if "train_accuracy" in logs:
        safe_log_metric("final_train_accuracy", logs["train_accuracy"])
    safe_log_metric("final_val_loss", logs["val_loss"])
    safe_log_metric("final_val_accuracy", logs["val_accuracy"])
    safe_log_metric("final_test_loss", test_loss)
    safe_log_metric("final_test_accuracy", test_accuracy)
    if not validation_set_available:
        best_val_accuracy = logs["val_accuracy"]
        best_test_accuracy = test_accuracy
    safe_log_metric("best_val_accuracy", best_val_accuracy, step=step)
    safe_log_metric("best_test_accuracy", best_test_accuracy, step=step)
    safe_log_metric("total_duration", total_duration, step=step)

    # Save final model
    save_model(
        model=model,
        model_name="final_model",
        sample_input=train_loader.dataset[0][0].unsqueeze(0),
        device=device,
    )

    logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    logger.info(f"Best test accuracy: {best_test_accuracy:.4f}")
    logger.info(f"Final test accuracy: {test_accuracy:.4f}")
    logger.info(
        f"\nMLflow run ID: {mlflow.active_run().info.run_id}"  # type: ignore
    )  # pyright: ignore[reportOptionalMemberAccess]
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logger.info("View the results with: mlflow ui")


def _run_experiment(cfg: DictConfig, logger) -> None:
    """Internal training function."""
    import os
    from pathlib import Path

    import mlflow
    import torch

    from hydra_script.aux_train_and_grow import log_git_commit, setup_device
    from tools.utils import flatten_dict, set_random_seeds

    # Print configuration
    logger.info("=" * 80)
    logger.info("Experiment configuration:")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)

    # Setup MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    warn_if_sqlite_tracking_uri(logger, str(cfg.mlflow.tracking_uri))
    safe_mlflow = SafeMLflowClient(logger=logger)

    mlflow_tags = dict()
    # Log git commit info
    log_git_commit(mlflow_tags)

    try:
        slurm_job_id = os.environ["SLURM_JOB_ID"]
    except KeyError:
        slurm_job_id = "N/A"

    mlflow_tags["slurm_job_id"] = slurm_job_id
    logger.info(f"SLURM Job ID: {slurm_job_id}")

    try:
        slurm_node_name = os.environ["SLURMD_NODENAME"]
    except KeyError:
        slurm_node_name = "N/A"
    mlflow_tags["slurm_node_name"] = slurm_node_name
    logger.info(f"SLURM Node Name: {slurm_node_name}")

    # Setup device
    device: torch.device = setup_device(cfg)
    # Get device index (actual GPU index on SLURM)
    if device.type == "cuda":
        # Try to get the actual GPU index from SLURM environment variables
        if "SLURM_JOB_GPUS" in os.environ:
            # SLURM_JOB_GPUS contains the actual GPU IDs (comma-separated if multiple)
            gpu_ids = os.environ["SLURM_JOB_GPUS"].split(",")
            device_index = int(gpu_ids[0])  # Use first GPU if multiple assigned
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            # CUDA_VISIBLE_DEVICES contains remapped GPU IDs
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            device_index = int(visible_devices[0]) if visible_devices[0] else 0
        else:
            # Fall back to torch's current device
            device_index = torch.cuda.current_device()
    else:
        device_index = -1

    # Log device info
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    mlflow_tags["device"] = device_name
    logger.info(f"Device: {device_name}")
    mlflow_tags["device_index"] = str(device_index)
    logger.info(f"Device index: {device_index}")

    # Set preferred CUDA linear algebra library
    if device.type == "cuda":
        preferred_library = cfg.general.get(
            "torch_backends_cuda_preferred_linalg_library", "default"
        ).lower()
        if preferred_library != "default":
            torch.backends.cuda.preferred_linalg_library(preferred_library)
            logger.info(
                f"Set torch.backends.cuda.preferred_linalg_library to {preferred_library}"
            )
        logger.info(
            f"Used torch.backends.cuda.preferred_linalg_library:"
            f" {torch.backends.cuda.preferred_linalg_library().name}"  # type: ignore
        )
        mlflow_tags["torch_backends_cuda_preferred_linalg_library"] = (
            torch.backends.cuda.preferred_linalg_library().name  # type: ignore
        )

    # Start MLflow run
    with mlflow.start_run(
        run_name=cfg.mlflow.get("run_name"),
        log_system_metrics=cfg.mlflow.get("log_system_metrics", True),
    ):
        logger.info(f"MLflow run {mlflow.active_run().info.run_id} started")  # type: ignore
        # Set random seeds if specified
        if cfg.general.get("seed") is not None:
            set_random_seeds(cfg.general.seed, device)
            logger.info(f"Random seeds set to {cfg.general.seed}")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Log MLflow tags
        safe_mlflow.set_tags(mlflow_tags)

        # Log all configuration parameters
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        params_to_log = flatten_dict(config_dict)
        safe_mlflow.log_params(params_to_log)

        # Save full config as artifact
        hydra_cfg = HydraConfig.get()
        run_dir = Path(hydra_cfg.runtime.output_dir)

        logger.info(f"Saving full configuration to MLflow run directory: {run_dir}")
        config_path: Path = (
            run_dir / f"hydra_config_{mlflow.active_run().info.run_id}.yaml"  # type: ignore
        )
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        logger.info(f"Full configuration saved to: {config_path}")

        safe_mlflow.log_artifact(
            local_path=str(config_path),
            operation_name="log_artifact:hydra_config",
        )
        os.remove(config_path)

        exception = NotImplementedError
        crashed = False
        try:
            _run_trainning(
                cfg=cfg,
                logger=logger,
                device=device,
            )
        except Exception as e:
            crashed = True
            exception = e
            logger.error(f"Training failed with exception: {e}")
            logger.error(traceback.format_exc())
            # Also print to stderr for submitit logs
            print(f"ERROR: Training failed with exception: {e}")
            print(traceback.format_exc())

        # Log the logger output file as MLflow artifact
        log_file_path = run_dir / "train_and_grow.log"
        if log_file_path.exists():
            safe_mlflow.log_artifact(
                local_path=str(log_file_path),
                operation_name="log_artifact:train_and_grow_log",
            )
            logger.info(f"Logged training log file as MLflow artifact: {log_file_path}")
        else:
            logger.warning(f"Log file not found at {log_file_path}")

        if crashed:
            raise exception


if __name__ == "__main__":
    main()
