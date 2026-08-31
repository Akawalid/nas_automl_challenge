import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import torch
import torch.utils.data
from gromo.containers.growing_container import GrowingContainer
from gromo.utils.training_utils import gradient_descent
from gromo.utils.utils import activation_fn
from hydra.utils import instantiate
from mlflow.models import infer_signature
from mlflow.utils.git_utils import get_git_branch, get_git_commit
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from experiments.auxilliary_functions import gradient_descent_mixup
from hydra_script.data_handling import get_dataloaders as new_get_dataloaders
from hydra_script.data_handling.datasets import resolve_train_transform_key
from hydra_script.mlflow_tools import SafeMLflowClient
from tools.create_growth_schedule import (
    BatchLimit,
    generate_schedule,
    write_schedule_file,
)
from tools.datasets import dataloader_description, get_dataloaders
from tools.models import get_model_from_config


def _growable_layer_count(model: GrowingContainer) -> int:
    """Layer count used by `layer_order` indices (same space as `generate_schedule`).

    Prefers `_growable_layers` when present; otherwise falls back to `_growing_layers`.
    """
    if hasattr(model, "_growable_layers"):
        assert isinstance(
            model._growable_layers, (list, tuple)
        ), "Expected model._growable_layers to be a list"
        return len(model._growable_layers)
    return len(model._growing_layers)


class EarlyStopping:
    """
    Class to handle early stopping.

    Methods:
    __init__ to initialize patience, min_delta and
    choose between maximize or minimize the monitored value.
    check(self, value) to check if training should stop.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        """
        Initialize the EarlyStopping object.

        Parameters
        ----------
        patience : int
            Number of epochs to wait for improvement before stopping.
        min_delta : float
            Minimum change in the monitored metric to qualify as an improvement.
            In "min" mode, patience is reset if: (metric < best_value - min_delta).
            In "max" mode, patience is reset if: (metric > best_value + min_delta).
        mode : str
            One of {"min", "max"}. In "min" mode, training will stop when the
            quantity monitored has stopped decreasing; in "max" mode it will
            stop when the quantity monitored has stopped increasing.
        """
        self.patience = patience
        self.min_delta = min_delta
        if mode not in ["min", "max"]:
            raise ValueError("mode should be 'min' or 'max'")
        if mode == "min":
            self.sign = -1
        else:
            self.sign = 1
        self.best_value = -float("inf")
        self.counter = 0

    def reset(self) -> None:
        """
        Reset the early stopping counter and best value.
        """
        self.best_value = -float("inf")
        self.counter = 0

    def check(self, value: float) -> bool:
        """
        Check if training should stop based on the monitored value.

        Parameters
        ----------
        value : float
            Current value of the monitored metric.

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        if self.sign * value > (self.best_value + self.sign * self.min_delta):
            self.best_value = max(self.sign * value, self.best_value)
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


def log_layers_metrics(
    layer_metrics: dict,
    step: int,
    prefix: str | None = None,
    logger: logging.Logger | None = None,
    mlflow_client: SafeMLflowClient | None = None,
) -> None:
    """
    Log the metrics of the layers to MLflow.

    Parameters
    ----------
    layer_metrics : dict
        layers metrics
    step : int
        step number
    prefix : str, optional
        prefix to add to the keys
    logger : logging.Logger, optional
        Logger to use for logging warnings and errors
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if mlflow_client is None:
        mlflow_client = SafeMLflowClient(logger=logger)
    if prefix is not None:
        prefix = prefix.replace("(", "-").replace(")", "-")
    for key, value in layer_metrics.items():
        if key in [
            "device",
            "device_model",
            "epoch_type",
            "gamma_history",
            "loss_history",
            "aux_loss_history",
            "eigenvalues_extension",
        ]:
            continue
        prefix_key = f"{prefix}_{key}" if prefix is not None else key
        if isinstance(value, dict):
            log_layers_metrics(
                layer_metrics=value,
                step=step,
                prefix=prefix_key,
                logger=logger,
                mlflow_client=mlflow_client,
            )
        else:
            try:
                mlflow_client.log_metric(prefix_key, value, step=step)
            except TypeError:
                logger.error(f"Cannot log {prefix_key} with value {value}")


def setup_device(cfg: DictConfig) -> torch.device:
    """
    Setup the device for training.

    Parameters
    ----------
    cfg : DictConfig
        Configuration

    Returns
    -------
    torch.device
        Device to use
    """
    from gromo.utils.utils import global_device, set_device

    set_device(torch.device(cfg.general.device))
    device = global_device()
    logger = logging.getLogger(__name__)
    if cfg.general.device == "cuda" and device.type != "cuda":
        logger.warning(
            f"Requested device 'cuda' but no CUDA device is available. Using '{device.type}' instead."
        )
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    return device


def create_dataloaders(cfg: DictConfig, device: torch.device) -> tuple[
    torch.utils.data.DataLoader | None,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    tuple,
]:
    """
    Create dataloaders for training, validation, and testing.

    Parameters
    ----------
    cfg : DictConfig
        Configuration
    device : torch.device
        Device to use
    Returns
    -------
    tuple
        growth_loader (or None), train_loader, val_loader, test_loader, input_shape
    """
    if cfg.dataset_config.name == "dry-ricker":
        # I try to reduce dependencies
        try:
            from hydra_script.eeg.get_dataloader import get_dataloader_eeg
        except ImportError:
            from eeg.get_dataloader import get_dataloader_eeg
        train, val, test = get_dataloader_eeg(
            config=cfg.dataset_config,
            device=device,
            batch_size=cfg.general.batch_size,
            num_workers=cfg.general.num_workers,
            seed=cfg.general.get("seed", torch.initial_seed()),
        )
        return (None, train, val, test, train.dataset[0][0].shape)
    elif "dataset" in cfg.dataset_config:
        # New Hydra-configurable dataset API
        split_val = cfg.dataset_config.split_train_val
        separate_growth = cfg.growing.get("separate_growth_and_train_set", False)
        growth_ratio = cfg.growing.get("growth_set_split_ratio", 0.2)
        growth_augmented = cfg.growing.get("growth_set_augmentation", True)
        seed = cfg.general.get("seed", torch.initial_seed())

        train_tf = resolve_train_transform_key(cfg.dataset_config)

        splits: dict[str, dict[str, Any]] = {
            "test": {
                "source": "test",
                "proportion": 1.0,
                "transforms": "standard",
                "shuffle": False,
                "drop_last": False,
            },
        }
        if separate_growth:
            growth_tf = train_tf if growth_augmented else "standard"
            splits["train"] = {
                "source": "train",
                "proportion": 1.0 - split_val - growth_ratio,
                "transforms": train_tf,
                "shuffle": True,
                "drop_last": True,
                "seed": seed,
            }
            splits["val"] = {
                "source": "train",
                "proportion": split_val,
                "transforms": "standard",
            }
            splits["growth"] = {
                "source": "train",
                "proportion": growth_ratio,
                "transforms": growth_tf,
                "shuffle": True,
                "seed": seed,
            }
        else:
            splits["train"] = {
                "source": "train",
                "proportion": 1.0 - split_val,
                "transforms": train_tf,
                "shuffle": True,
                "drop_last": True,
                "seed": seed,
            }
            splits["val"] = {
                "source": "train",
                "proportion": split_val,
                "transforms": "standard",
            }
        loaders = new_get_dataloaders(
            cfg.dataset_config,
            splits=splits,
            batch_size=cfg.general.batch_size,
            num_workers=cfg.general.num_workers,
            device=device,
            seed=seed,
        )

        train_loader = loaders["train"]
        val_loader = loaders["val"]
        test_loader = loaders["test"]
        growth_loader = loaders.get("growth")
        input_shape = train_loader.dataset[0][0].shape

        logger = logging.getLogger(__name__)
        if growth_loader is not None:
            logger.info("Growth loader:")
            logger.info(dataloader_description(growth_loader))
        logger.info("Train loader:")
        logger.info(dataloader_description(train_loader))
        logger.info("Val loader:")
        logger.info(dataloader_description(val_loader))
        logger.info("Test loader:")
        logger.info(dataloader_description(test_loader))

        return growth_loader, train_loader, val_loader, test_loader, input_shape
    else:
        # Legacy config path (no `dataset` / `sources` / `transforms` in dataset_config)
        warnings.warn(
            "Hydra `dataset_config` without a `dataset` block is deprecated: define "
            "`dataset`, `sources`, and `transforms` in "
            "`hydra_script/configs/dataset_config/<name>.yaml` (see `addnist.yaml`). "
            "The legacy `tools.datasets.get_dataloaders` path remains for backward "
            "compatibility but may be removed in a future release.",
            FutureWarning,
            stacklevel=2,
        )
        separate_growth_set = cfg.growing.get("separate_growth_and_train_set", False)

        result = get_dataloaders(
            dataset_name=cfg.dataset_config.name,
            dataset_path=cfg.dataset_config.path,
            nb_class=cfg.dataset_config.num_classes,
            split_train_val=cfg.dataset_config.split_train_val,
            data_augmentation=cfg.dataset_config.get("data_augmentation", None),
            batch_size=cfg.general.batch_size,
            device=device,
            num_workers=cfg.general.num_workers,
            shuffle=True,
            seed=cfg.general.get("seed", torch.initial_seed()),
            separate_growth_set=separate_growth_set,
            growth_set_split_ratio=cfg.growing.get("growth_set_split_ratio", 0.2),
            growth_set_augmentation=cfg.growing.get("growth_set_augmentation", True),
        )

        logger = logging.getLogger(__name__)
        if separate_growth_set:
            assert (
                len(result) == 5
            ), "Expected 5 elements in result when separate_growth_set is True"
            growth_loader, train_loader, val_loader, test_loader, input_shape = result
            logger.info("Growth loader:")
            logger.info(dataloader_description(growth_loader))
        else:
            assert (
                len(result) == 4
            ), "Expected 4 elements in result when separate_growth_set is False"
            train_loader, val_loader, test_loader, input_shape = result
            growth_loader = None

        logger.info("Train loader:")
        logger.info(dataloader_description(train_loader))
        logger.info("Val loader:")
        logger.info(dataloader_description(val_loader))
        logger.info("Test loader:")
        logger.info(dataloader_description(test_loader))

        return growth_loader, train_loader, val_loader, test_loader, input_shape


def initialize_model(
    cfg: DictConfig,
    input_shape,
    num_classes: int,
    device: torch.device,
):
    """
    Initialize the model from configuration.

    Parameters
    ----------
    cfg : DictConfig
        Configuration
    input_shape : tuple
        Input shape
    num_classes : int
        Number of output classes
    device : torch.device
        Device to use

    Returns
    -------
    GrowingContainer
        Initialized model
    """
    # Load model config
    if cfg.model.model == "eeg_model":
        # I try to reduce dependencies
        try:
            from hydra_script.eeg.eeg_model import EEGModel
        except ImportError:
            from eeg.eeg_model import EEGModel
        return EEGModel(
            in_features=(1, 8, 175),
            out_features=num_classes,
            n_channel_input=8,
            activation=(
                torch.nn.LeakyReLU(negative_slope=0.3)
                if cfg.model.activation == "leaky_relu"
                else activation_fn(cfg.model.activation)
            ),
            use_bias=True,
            device=device,
            initial_last_layer_size=cfg.model.initial_last_layer_size,
        )

    else:
        model_config: dict[Any, Any] = OmegaConf.to_container(
            cfg.model, resolve=True
        )  # pyright: ignore[reportAssignmentType]

        model: GrowingContainer = get_model_from_config(
            input_shape=input_shape,
            out_features=num_classes,
            device=device,
            config=model_config,
        )

        logger = logging.getLogger(__name__)
        logger.info(f"Model:\n{model}")

        return model


def param_groups_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    no_weight_decay_list: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Split parameters into decay / no-decay groups (timm ``filter_bias_and_bn``).

    Mirrors ``timm.optim.param_groups_weight_decay``: biases, 1-dimensional
    parameters (LayerNorm / norm weights and biases), and any name in
    ``no_weight_decay_list`` go to a group with ``weight_decay=0``; all other
    parameters keep ``weight_decay``. This matches the Compact-Transformers
    recipe, which does not apply weight decay to norm and bias parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters are split.
    weight_decay : float
        Weight decay applied to the decay group.
    no_weight_decay_list : tuple[str, ...]
        Parameter names to force into the no-decay group.

    Returns
    -------
    list[dict[str, Any]]
        Optimizer parameter groups. Empty groups are dropped.
    """
    no_weight_decay = set(no_weight_decay_list)
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay:
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
    return [group for group in groups if group["params"]]


def build_optimizer(
    cfg: DictConfig, model: torch.nn.Module
) -> torch.optim.Optimizer:
    """Instantiate the optimizer, optionally excluding norm/bias from weight decay.

    When ``general.filter_bias_and_norm`` is true, parameters are split into
    decay / no-decay groups via :func:`param_groups_weight_decay`, reproducing
    timm's default ``filter_bias_and_bn`` behaviour used by Compact-Transformers.
    Otherwise a single parameter group is used (weight decay on all parameters),
    preserving the previous behaviour.

    Parameters
    ----------
    cfg : DictConfig
        Configuration (reads ``cfg.optimizer`` and ``cfg.general``).
    model : torch.nn.Module
        Model whose parameters are optimized.

    Returns
    -------
    torch.optim.Optimizer
        The instantiated optimizer.
    """
    filter_bias_and_norm = bool(cfg.general.get("filter_bias_and_norm", False))
    if not filter_bias_and_norm:
        return instantiate(cfg.optimizer, params=model.parameters())

    weight_decay = float(cfg.optimizer.get("weight_decay", 0.0))
    no_weight_decay_list: tuple[str, ...] = ()
    if hasattr(model, "no_weight_decay"):
        try:
            no_weight_decay_list = tuple(model.no_weight_decay())
        except Exception:  # pragma: no cover - defensive; model API varies
            no_weight_decay_list = ()

    param_groups = param_groups_weight_decay(model, weight_decay, no_weight_decay_list)
    logger = logging.getLogger(__name__)
    logger.info(
        "Excluding norm/bias parameters from weight decay "
        f"(general.filter_bias_and_norm=true): {len(param_groups)} parameter groups."
    )
    # Build via a partial so the parameter groups are passed as a plain Python
    # object at call time. Passing them through ``instantiate(..., params=...)``
    # would convert the list-of-dicts into OmegaConf containers, and PyTorch does
    # not recognise a ``DictConfig`` as a parameter group (it is not a ``dict``
    # subclass), raising "optimizer can only optimize Tensors". ``_convert_="all"``
    # makes the bound config kwargs (e.g. ``betas``) plain Python lists/tuples.
    optimizer_factory = instantiate(cfg.optimizer, _partial_=True, _convert_="all")
    return optimizer_factory(param_groups)


def create_scheduler(
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    num_batches_per_epoch: int | None = None,
    nb_steps: int | None = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create learning rate scheduler.

    Parameters
    ----------
    cfg : DictConfig
        Configuration
    optimizer : torch.optim.Optimizer
        Optimizer
    num_batches_per_epoch : int | None
        Number of batches per epoch
    nb_steps : int | None
        Number of total steps

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler
        Scheduler
    """
    scheduler = instantiate(
        cfg.lr_scheduler.scheduler_object,
        optimizer=optimizer,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Scheduler: {scheduler}")

    return scheduler


def generate_growth_schedule_if_needed(
    cfg: DictConfig,
    model: GrowingContainer,
    run_id: str,
):
    """
    Generate growth schedule if requested.

    Parameters
    ----------
    cfg : DictConfig
        Configuration
    model : GrowingContainer
        Model

    Returns
    -------
    list or None
        Growth schedule
    """
    if not cfg.general.get("generate_schedule", True):
        return None

    logger = logging.getLogger(__name__)
    # Infer number of layers if not provided (indices match ``model._growable_layers``)
    safe_mlflow = SafeMLflowClient(logger=logger)
    num_layers = cfg.schedule.get("num_layers", None)
    if num_layers is None:
        num_layers = _growable_layer_count(model)
    elif hasattr(model, "_growable_layers"):
        assert isinstance(
            model._growable_layers, (list, tuple)
        ), "Expected model._growable_layers to be a list"
        inferred = len(model._growable_layers)
        if num_layers != inferred:
            raise ValueError(
                f"schedule.num_layers ({num_layers}) must equal "
                f"len(model._growable_layers) ({inferred})."
            )

    logger.info(f"Generating growth schedule for {num_layers} layers")

    growth_warmup_steps = int(cfg.schedule.get("growth_warmup_steps", 0))
    if growth_warmup_steps < 0:
        raise ValueError(
            f"schedule.growth_warmup_steps must be positive, got {growth_warmup_steps}"
        )

    schedule = generate_schedule(
        num_layers=num_layers,
        layer_order=cfg.schedule.get("layer_order", None),
        epochs_between_growth=cfg.schedule.epochs_between_growth,
        growth_steps_per_layer=cfg.schedule.growth_steps_per_layer,
        growth_size=cfg.schedule.get("growth_size"),
        training_steps_after_growth=cfg.schedule.get("training_steps_after_growth", 0),
        include_completion_step=cfg.schedule.get("include_completion_step", False),
        growth_warmup_steps=growth_warmup_steps,
    )

    # Save to file and log as artifact
    schedule_file = f"growth_schedule_{run_id}.txt"
    write_schedule_file(schedule, schedule_file)
    safe_mlflow.log_artifact(
        local_path=schedule_file,
        operation_name="log_artifact:growth_schedule",
    )
    logger.info(f"Generated and logged growth schedule with {len(schedule)} steps")

    # Save to additional output file if specified
    if cfg.schedule.get("output_file"):
        output_file = cfg.schedule.output_file
        write_schedule_file(schedule, output_file)
        logger.info(f"Also saved growth schedule to {output_file}")

    # Clean up temporary file
    if os.path.exists(schedule_file):
        os.remove(schedule_file)

    return schedule


def resolve_batch_limit(
    raw_value: BatchLimit,
    dataloader_length: int,
    config_fallback: int = -1,
) -> int | None:
    """Resolve a batch_limit value from the schedule into an int usable by dataloaders.

    Parameters
    ----------
    raw_value : BatchLimit
        The value from the schedule's fourth column.
        - ``int``: exact number of batches (``-1`` = full epoch).
        - ``float``: fraction of the dataset in ``(0, 1]``.
        - ``"hyper-parameter"``: defer to *config_fallback*.
    dataloader_length : int
        Number of batches in the dataloader (``len(dataloader)``).
    config_fallback : int, optional
        The config value to use when *raw_value* is ``"hyper-parameter"``.
        ``-1`` means no limit (full epoch). Default is ``-1``.

    Returns
    -------
    int | None
        The resolved batch limit. ``None`` means use all batches (full epoch).
    """
    if isinstance(raw_value, str):
        if raw_value != "hyper-parameter":
            raise ValueError(
                f"Invalid string batch_limit value: '{raw_value}'. "
                "Only 'hyper-parameter' is supported."
            )
        if config_fallback == -1:
            return None
        return config_fallback
    if isinstance(raw_value, float):
        if not (0.0 < raw_value <= 1.0):
            raise ValueError(f"Float batch_limit must be in (0, 1], got {raw_value}")
        return max(1, round(raw_value * dataloader_length))
    # int case
    if raw_value == -1:
        return None
    if raw_value <= 0:
        raise ValueError(f"Integer batch_limit must be -1 or positive, got {raw_value}")
    return raw_value


def perform_training_step(
    cfg: DictConfig,
    model: GrowingContainer,
    step: int,
    train_loader,
    loss_function_train,
    top_1_accuracy,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    batch_limit: int | None = None,
    amp_scaler: "torch.cuda.amp.GradScaler | None" = None,
) -> dict:
    """
    Perform a standard training step.

    Parameters
    ----------
    cfg : DictConfig
        Configuration
    model : GrowingContainer
        Model
    step : int
        Current step
    train_loader : DataLoader
        Training data loader
    loss_function_train : callable
        Loss function
    top_1_accuracy : callable
        Accuracy function
    optimizer : torch.optim.Optimizer
        Optimizer
    scheduler : torch.optim.lr_scheduler.LRScheduler | None
        Scheduler
    device : torch.device
        Device
    batch_limit : int or None
        If not None, limit the number of batches to use
        for training.
    amp_scaler : torch.cuda.amp.GradScaler | None
        Gradient scaler shared across epochs for mixed-precision training.

    Returns
    -------
    dict
        logs
    """
    # Mixup/CutMix + AMP recipe knobs (timm/Compact-Transformers style). They live
    # in ``dataset_config`` because the augmentation recipe is dataset-specific,
    # mirroring the existing ``cutmix_prob``/``cutmix_beta`` convention. AMP is a
    # runtime flag and lives in ``general``.
    mixup_alpha = cfg.dataset_config.get("mixup_alpha", 0.0)
    cutmix_alpha = cfg.dataset_config.get(
        "cutmix_alpha", cfg.dataset_config.get("cutmix_beta", 0.0)
    )
    # Legacy ``cutmix_prob`` schema: treat a positive value as "enable CutMix".
    legacy_cutmix_prob = cfg.dataset_config.get("cutmix_prob", 0.0)
    if cutmix_alpha <= 0.0 and legacy_cutmix_prob > 0.0:
        cutmix_alpha = cfg.dataset_config.get("cutmix_beta", 1.0)
    mixup_prob = cfg.dataset_config.get(
        "mixup_prob", legacy_cutmix_prob if legacy_cutmix_prob > 0.0 else 1.0
    )
    mixup_switch_prob = cfg.dataset_config.get("mixup_switch_prob", 0.5)
    mixup_off_epoch = cfg.dataset_config.get("mixup_off_epoch", 0)
    amp_enabled = bool(cfg.general.get("amp", False))

    recipe_uses_mixup = mixup_alpha > 0.0 or cutmix_alpha > 0.0
    # ``step`` is the 1-indexed epoch number; disable mixing from mixup_off_epoch
    # onwards (matching timm's ``epoch >= mixup_off_epoch``).
    mixup_active = recipe_uses_mixup and not (
        mixup_off_epoch and step >= mixup_off_epoch
    )

    if recipe_uses_mixup or amp_enabled:
        train_loss, train_accuracy = gradient_descent_mixup(
            model=model,
            train_dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function_train,
            metrics=top_1_accuracy,
            batch_limit=batch_limit,
            device=device,
            scheduler_step_granularity=cfg.lr_scheduler.scheduler_step_granularity,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mixup_prob=mixup_prob,
            mixup_switch_prob=mixup_switch_prob,
            mixup_active=mixup_active,
            amp_scaler=amp_scaler,
            amp_enabled=amp_enabled,
        )
    else:
        train_loss, train_accuracy = gradient_descent(
            model=model,
            train_dataloader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function_train,
            metrics=top_1_accuracy,
            batch_limit=batch_limit,
            device=device,
            scheduler=scheduler,
            scheduler_step_granularity=cfg.lr_scheduler.scheduler_step_granularity,
        )

    logs: dict[str, int | float] = dict()
    logs["train_loss"] = train_loss
    logs["train_accuracy"] = train_accuracy
    logs["learning_rate"] = (  # type: ignore
        optimizer.param_groups[0]["lr"]
        if scheduler is None
        else scheduler.get_last_lr()[0]
    )
    logs["selected_update"] = -1

    return logs


def log_git_commit_repo(
    repo_path: Path,
    repo_name: str | None = None,
    tag_dict: dict | None = None,
) -> dict[str, str]:
    """
    Log git commit and branch info for a given repository path using MLflow utilities.

    Parameters
    ----------
    repo_path : str
        Path to the git repository
    repo_name : str, optional
        Name of the repository (used for tagging) if None, end of path is used
    tag_dict : dict, optional
        Dictionary to store tags, if None a new dict is created

    Returns
    -------
    dict
        Git commit and branch information
    """
    if tag_dict is None:
        tag_dict = dict()
    if repo_name is None:
        repo_name = repo_path.name
    logger = logging.getLogger(__name__)
    try:
        git_commit = get_git_commit(str(repo_path))
        git_branch = get_git_branch(str(repo_path))

        if git_commit:
            tag_dict[f"git_commit_{repo_name}"] = git_commit
            logger.info(f"Git commit for {repo_name}: {git_commit}")
        else:
            logger.warning(f"No git commit found for repository at {repo_path}")
        if git_branch:
            tag_dict[f"git_branch_{repo_name}"] = git_branch
            logger.info(f"Git branch for {repo_name}: {git_branch}")
        else:
            logger.warning(f"No git branch found for repository at {repo_path}")
    except Exception as e:
        logger.warning(f"Could not retrieve git information for {repo_path}: {e}")

    return tag_dict


def log_git_commit(
    tag_dict: dict | None = None,
) -> dict[str, str]:
    """
    Log git commit and branch info using MLflow utilities.

    Log the main repository and the gromo repository.

    Parameters
    ----------
    tag_dict : dict, optional
        Dictionary to store tags, if None a new dict is created

    Returns
    -------
    dict
        Git commit and branch information
    """
    if tag_dict is None:
        tag_dict = dict()

    logger = logging.getLogger(__name__)
    # Log git commit and branch info using MLflow utilities
    # Main repo
    repo_path = Path(__file__).parent.parent
    log_git_commit_repo(repo_path, tag_dict=tag_dict)

    # gromo repo
    try:
        import gromo

        gromo_repo_path = Path(gromo.__file__).parent.parent.parent
        log_git_commit_repo(gromo_repo_path, repo_name="gromo", tag_dict=tag_dict)
    except ImportError:
        logger.warning("Could not import gromo to log its git information.")

    return tag_dict


def save_model(
    model: torch.nn.Module,
    model_name: str,
    sample_input: torch.Tensor,
    device: torch.device,
):
    """
    Save the model to a file.

    Parameters
    ----------
    model : torch.nn.Module
        Model to save
    model_path : str
        Path to save the model
    sample_input : torch.Tensor
        Sample input for tracing
    """
    logger = logging.getLogger(__name__)
    safe_mlflow = SafeMLflowClient(logger=logger)
    # Create sample input for signature
    logger.info(f"Creating sample input for model signature: {sample_input.shape}")
    model.eval()
    with torch.no_grad():
        sample_output = model(sample_input.to(device)).cpu().numpy()
    logger.info(f"Sample output shape for model signature: {sample_output.shape}")

    signature = infer_signature(sample_input.cpu().numpy(), sample_output)
    logger.info(f"Inferred model signature: {signature}")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_model_dir = Path(tmpdir) / model_name

        mlflow.pytorch.save_model(
            pytorch_model=model,
            path=str(local_model_dir),
            signature=signature,
            pip_requirements=[],
        )
        logger.info("Model saved.")

        safe_mlflow.log_artifacts(
            local_dir=str(local_model_dir),
            artifact_path=model_name,
            operation_name=f"log_artifacts:{model_name}",
        )

    logger.info("Model logged.")
