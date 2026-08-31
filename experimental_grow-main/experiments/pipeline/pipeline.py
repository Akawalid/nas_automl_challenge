import copy
import gc
import json
import operator
import os
import random
import re
import sys
from collections import OrderedDict
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from git.repo import Repo
from gromo.containers.growing_container import GrowingContainer
from gromo.containers.growing_graph_network import (
    ExplicitGrowthInitMode,
    ExpansionType,
    GrowingGraphNetwork,
    InterMergeExpansion,
)
from gromo.modules.growing_module import GrowingModule, MergeGrowingModule
from gromo.utils.dependence_estimator import calculate_dependency
from gromo.utils.training_utils import evaluate_dataset
from gromo.utils.utils import global_device, set_device
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.auxilliary_functions import extended_evaluate_model, line_search
from experiments.growth_ablations import (
    add_noise_to_pre_activities_grad,
    get_variance_transfer_settings,
)
from experiments.pipeline.models import cell_arch, cnn  # noqa: F401
from experiments.pipeline.models.ema import EMA
from experiments.schedulers import (
    get_persistent_pipeline_scheduler,
    get_pipeline_linear_warmup_settings,
    get_pipeline_scheduler,
    get_pipeline_scheduler_ablation_settings,
    get_pipeline_warmup_constant_scheduler,
    is_pipeline_linear_warmup_active,
    rebind_pipeline_scheduler,
    rebind_pipeline_scheduler_with_warmup,
    should_defer_pipeline_scheduler,
)
from tools.augmentations import default_augmentations
from tools.datasets import get_dataset
from tools.functional_gradient import (
    capture_network_outputs,
    measure_output_functional_metrics,
    prepare_fixed_train_probe,
)
from tools.logger import Logger
from tools.utils import DAG_to_pyvis, set_random_seeds


def setup_logger(cfg, **_):
    api = cfg["logger"]["api"]
    exp_name = cfg["logger"]["exp_name"]
    port = cfg["logger"]["port"]
    enabled = cfg["logger"]["enabled"]
    path = cfg["logger"]["path"]
    logger = Logger(experiment_name=exp_name, port=port, api=api, enabled=enabled)
    logger.setup_tracking(file_path=path)
    return {"logger": logger}


def _experiment_seed(cfg) -> int | None:
    experiment_cfg = cfg.get("experiment", {})
    if not isinstance(experiment_cfg, dict):
        return None
    return experiment_cfg.get("seed")


def _growth_initialization_strategy(cfg) -> str:
    growth_cfg = cfg.get("growth", {})
    if not isinstance(growth_cfg, dict):
        return "local"
    raw_strategy = growth_cfg.get("initialization_strategy", "local")
    strategy = str(raw_strategy).lower().strip()
    if strategy == "local":
        return "local"
    if strategy == "init_scaling_ablation":
        return "init_scaling_ablation"
    raise ValueError(
        f"Unsupported growth.initialization_strategy={raw_strategy!r}. "
        "Expected 'local' or 'init_scaling_ablation'."
    )


def _growth_explicit_init_mode(
    cfg,
    key: str,
    default: ExplicitGrowthInitMode,
) -> ExplicitGrowthInitMode:
    growth_cfg = cfg.get("growth", {})
    if not isinstance(growth_cfg, dict):
        return default
    raw_mode = growth_cfg.get(key, default)
    mode = str(raw_mode).lower().strip()
    if mode == "kaiming":
        return "kaiming"
    if mode == "zeros":
        return "zeros"
    raise ValueError(
        f"Unsupported growth.{key}={raw_mode!r}. Expected 'kaiming' or 'zeros'."
    )


def _init_scaling_ablation_enabled(cfg) -> bool:
    return _growth_initialization_strategy(cfg) == "init_scaling_ablation"


def _defer_init_scaling_ablation_to_final_choice(cfg) -> bool:
    return _init_scaling_ablation_enabled(cfg)


def _disable_optimal_delta_on_apply_change(cfg) -> bool:
    growth_cfg = cfg.get("growth", {})
    if not isinstance(growth_cfg, dict):
        return False
    return bool(growth_cfg.get("disable_optimal_delta_on_apply_change", False))


def _sgd_momentum_transfer_enabled(cfg) -> bool:
    training_cfg = cfg.get("training", {})
    if not isinstance(training_cfg, dict):
        return False
    ablations_cfg = training_cfg.get("ablations", {})
    if not isinstance(ablations_cfg, dict):
        return False
    transfer_cfg = ablations_cfg.get("sgd_momentum_transfer", {})
    if isinstance(transfer_cfg, dict):
        return bool(transfer_cfg.get("enabled", False))
    return bool(transfer_cfg)


def _optimizer_is_sgd(optimizer_name: str) -> bool:
    try:
        return eval(optimizer_name) is torch.optim.SGD
    except Exception:
        return str(optimizer_name).split(".")[-1] == "SGD"


def _copy_overlap(old: torch.Tensor, new: torch.Tensor) -> tuple[torch.Tensor, bool]:
    copied = torch.zeros_like(new, memory_format=torch.preserve_format)
    if old.ndim != new.ndim:
        return copied, False

    overlap = tuple(
        min(old_dim, new_dim) for old_dim, new_dim in zip(old.shape, new.shape)
    )
    if any(size == 0 for size in overlap):
        return copied, False

    slices = tuple(slice(0, size) for size in overlap)
    copied[slices].copy_(old.to(device=new.device, dtype=new.dtype)[slices])
    return copied, True


def _snapshot_sgd_momentum_buffers(
    optimizer: torch.optim.Optimizer,
    model: GrowingContainer,
) -> dict[str, torch.Tensor]:
    param_names = {param: name for name, param in model.named_parameters()}
    buffers: dict[str, torch.Tensor] = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            name = param_names.get(param)
            if name is None:
                continue
            momentum_buffer = optimizer.state.get(param, {}).get("momentum_buffer")
            if momentum_buffer is None:
                continue
            buffers[name] = momentum_buffer.detach().cpu().clone()
    return buffers


def _transfer_sgd_momentum_buffers(
    optimizer: torch.optim.Optimizer,
    model: GrowingContainer,
    previous_buffers: dict[str, torch.Tensor] | None,
) -> dict[str, int]:
    stats = {
        "previous_buffers": 0 if previous_buffers is None else len(previous_buffers),
        "matched_buffers": 0,
        "transferred_buffers": 0,
        "resized_buffers": 0,
        "new_buffers_zero_initialized": 0,
        "shape_mismatch_skipped": 0,
    }
    if not previous_buffers:
        return stats

    for name, param in model.named_parameters():
        state = optimizer.state[param]
        old_buffer = previous_buffers.get(name)
        if old_buffer is None:
            state["momentum_buffer"] = torch.zeros_like(
                param.detach(), memory_format=torch.preserve_format
            )
            stats["new_buffers_zero_initialized"] += 1
            continue

        stats["matched_buffers"] += 1
        new_buffer, copied = _copy_overlap(old_buffer, param.detach())
        if not copied:
            stats["shape_mismatch_skipped"] += 1
            state["momentum_buffer"] = torch.zeros_like(
                param.detach(), memory_format=torch.preserve_format
            )
            continue

        state["momentum_buffer"] = new_buffer
        stats["transferred_buffers"] += 1
        if tuple(old_buffer.shape) != tuple(param.shape):
            stats["resized_buffers"] += 1

    return stats


def _log_sgd_momentum_transfer_stats(
    logger: Logger,
    stats: dict[str, int],
    global_step: int,
) -> None:
    for name, value in stats.items():
        logger.log_metric(
            f"training/sgd momentum transfer/{name}",
            value,
            global_step,
            step_name="growth step",
        )


def _apply_growing_graph_change_without_optimal_delta(
    growing_graph: GrowingGraphNetwork,
) -> None:
    """Commit a graph growth action without applying optimal_delta_layer."""
    chosen_action = growing_graph.chosen_action
    if chosen_action is None:
        return

    init_strategy = chosen_action.metrics.get("initialization_strategy")
    for prev_node, next_node in growing_graph.dag.edges:
        factor = chosen_action.metrics["scaling_factor"]
        edge_module = growing_graph.dag.get_edge_module(prev_node, next_node)

        if (
            init_strategy == "init_scaling_ablation"
            and edge_module.extended_input_layer is not None
            and edge_module.extended_input_layer.bias is not None
        ):
            with torch.no_grad():
                edge_module.extended_input_layer.bias.zero_()

        edge_module.scaling_factor = factor
        edge_module.output_extension_scaling = factor  # type: ignore
        edge_module.apply_change(
            scaling_factor=factor,
            apply_previous=False,
            apply_delta=False,
        )
        if edge_module.extended_output_layer is not None:
            new_neurons = chosen_action.metrics["active_neurons"]
            edge_module._apply_output_changes(
                scaling_factor=factor,
                extension_size=new_neurons,
            )

    if chosen_action.type != ExpansionType.NEW_EDGE:
        if chosen_action.dag == growing_graph.dag:
            assert chosen_action.expanding_node in growing_graph.dag.nodes
            expanding_node = chosen_action.expanding_node
        elif (
            isinstance(chosen_action, InterMergeExpansion)
            and chosen_action.adjacent_expanding_node in growing_graph.dag.nodes
        ):
            expanding_node = chosen_action.adjacent_expanding_node
        else:
            expanding_node = ""
        growing_graph.update_size()
        growing_graph.dag.rename_nodes({expanding_node: expanding_node.split("_")[0]})

    growing_graph.growth_history = copy.copy(chosen_action.growth_history)
    growing_graph.growth_loss_train = chosen_action.metrics.get("loss_train")
    growing_graph.growth_loss_dev = chosen_action.metrics.get("loss_dev")
    growing_graph.growth_loss_val = chosen_action.metrics.get("loss_val")
    growing_graph.growth_acc_train = chosen_action.metrics.get("acc_train")
    growing_graph.growth_acc_dev = chosen_action.metrics.get("acc_dev")
    growing_graph.growth_acc_val = chosen_action.metrics.get("acc_val")


def _pipeline_step_names(steps) -> list[str]:
    names = []
    for step in steps:
        if isinstance(step, str):
            names.append(step)
        elif isinstance(step, dict) and "loop" in step:
            names.extend(_pipeline_step_names(step["loop"].get("steps", [])))
    return names


def _require_growth_tensor(name: str, value):
    if value is None:
        raise RuntimeError(
            f"Missing growth tensor '{name}' in pipeline context. "
            "Run update_computation/calculate_bottleneck before this step."
        )
    return value


def _release_growth_tensors() -> dict[str, None]:
    return {
        "pre_activities_grad": None,
        "input_B": None,
        "bottleneck": None,
    }


def _growth_strategy(cfg) -> str:
    step_names = set(_pipeline_step_names(cfg.get("pipeline", [])))
    if "pick_random_action" in step_names:
        return "random"
    if "restrict_growth_actions" in step_names:
        return "restricted"
    return "greedy"


def _validate_init_scaling_ablation_strategy(cfg) -> None:
    if not _init_scaling_ablation_enabled(cfg):
        return
    _growth_strategy(cfg)
    _growth_explicit_init_mode(cfg, "incoming_init", "kaiming")
    _growth_explicit_init_mode(cfg, "outgoing_init", "zeros")
    _growth_explicit_init_mode(cfg, "new_edge_init", "zeros")


def _growth_label_smoothing_enabled(cfg) -> bool:
    ablations = cfg.get("training", {}).get("ablations", {})
    if not isinstance(ablations, dict):
        raise ValueError("training.ablations must be a mapping")
    label_smoothing = ablations.get("growth_label_smoothing", {})
    if not isinstance(label_smoothing, dict):
        raise ValueError("training.ablations.growth_label_smoothing must be a mapping")
    enabled = label_smoothing.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(
            "training.ablations.growth_label_smoothing.enabled must be a boolean"
        )
    return enabled


def _active_label_smoothing(cfg) -> float:
    label_smoothing = cfg["training"].get("label_smoothing", 0.0)
    if _growth_label_smoothing_enabled(cfg) and cfg.get("growth_complete", False):
        return 0.0
    return label_smoothing


def _loss_fn(cfg, **kwargs):
    label_smoothing = _active_label_smoothing(cfg)
    return eval(cfg["training"]["loss_function"])(
        label_smoothing=label_smoothing,
        **kwargs,
    )


def setup_experiment_tags(cfg, job_id: str, node_name: str, **_):
    repo = Repo(Path(__file__).resolve(), search_parent_directories=True)
    git_commit = repo.head.object.hexsha
    git_dir = repo.git_dir
    try:
        gpu_index = torch.cuda.current_device()
    except:
        gpu_index = None
    tags = {
        "git.dir": git_dir,
        "git.commit": git_commit,
        "slurm.job_id": job_id,
        "slurm.node_name": node_name,
        "gpu_index": gpu_index,
    }
    seed = _experiment_seed(cfg)
    if seed is not None:
        tags["experiment.seed"] = seed
    experiment_cfg = cfg.get("experiment", {})
    if isinstance(experiment_cfg, dict) and experiment_cfg.get("name") is not None:
        tags["experiment.name"] = experiment_cfg["name"]
    return tags


def set_random_seed(cfg, **_):
    seed = _experiment_seed(cfg)
    if seed is None:
        return {"seed": None}

    device_name = str(cfg.get("training", {}).get("device", "cpu"))
    if device_name.startswith("cuda") and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    set_random_seeds(seed, device)
    return {"seed": seed}


def load_data(cfg, **_):
    name = cfg["dataset"]["name"]
    path = cfg["dataset"]["path"]
    val_split = cfg["dataset"]["val_split"]
    augment = cfg["dataset"].get("augment", False)
    train_set, val_set, test_set = get_dataset(
        dataset_name=name,
        dataset_path=path,
        splits_sizes=[1 - val_split, val_split],
        data_augmentation=default_augmentations[name] if augment else [],
        split_data_augmentation_activation=[augment, False],
    )
    return {"train_set": train_set, "val_set": val_set, "test_set": test_set}


def split_data(cfg, train_set: Dataset, **_):
    dev_split = cfg["dataset"]["dev_split"]
    growth_set, dev_set = torch.utils.data.random_split(
        train_set, [1 - dev_split, dev_split]
    )
    return {"growth_set": growth_set, "dev_set": dev_set}


def create_dataloaders(
    cfg,
    train_set: Dataset,
    growth_set: Dataset,
    dev_set: Dataset,
    val_set: Dataset,
    test_set: Dataset,
    train_dataloader: DataLoader | None = None,
    val_dataloader: DataLoader | None = None,
    test_dataloader: DataLoader | None = None,
    **_,
):
    """Create PyTorch dataloaders based on dataset config.

    train, val, and test dataloaders are created only once and reused across
    loop iterations. growth and dev dataloaders are recreated each time since
    they depend on the random split from split_data.
    """
    batch_size = cfg["training"]["batch_size"]
    shuffle = cfg["dataset"]["shuffle"]
    growth_batch_size = cfg["growth"]["batch_size"]

    if train_dataloader is None:
        train_dataloader = DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            shuffle=shuffle,
            multiprocessing_context="spawn",
        )
    if val_dataloader is None:
        val_dataloader = DataLoader(
            val_set,
            batch_size=batch_size,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False,
            multiprocessing_context="spawn",
        )
    if test_dataloader is None:
        test_dataloader = DataLoader(
            test_set,
            batch_size=batch_size,
            num_workers=2,
            shuffle=False,
            multiprocessing_context="spawn",
        )

    growth_dataloader = DataLoader(
        growth_set,
        batch_size=growth_batch_size,
        num_workers=2,
        shuffle=False,
        multiprocessing_context="spawn",
    )
    dev_dataloader = DataLoader(
        dev_set,
        batch_size=batch_size,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        shuffle=False,
        multiprocessing_context="spawn",
    )

    return {
        "train_dataloader": train_dataloader,
        "growth_dataloader": growth_dataloader,
        "dev_dataloader": dev_dataloader,
        "val_dataloader": val_dataloader,
        "test_dataloader": test_dataloader,
    }


def set_global_device(cfg, **_):
    device = cfg["training"]["device"]
    set_device(device)


def create_model(cfg, train_set: Dataset, logger: Logger, **_):
    input_shape = train_set.dataset.data.shape[1:-1]  # type: ignore
    in_channels = train_set.dataset.data.shape[-1]  # type: ignore
    out_features = len(np.unique(train_set.dataset.targets))  # type: ignore
    model_params = cfg["model"]

    loss_fn = _loss_fn(cfg)
    device = cfg["training"]["device"]

    neurons = cfg["growth"]["neurons"]
    neuron_epochs = cfg["growth"]["neuron_epochs"]
    neuron_lrate = cfg["growth"]["neuron_lrate"]
    neuron_batch_size = cfg["growth"]["neuron_batch_size"]

    in_features = np.prod(input_shape) * in_channels  # linear case

    model = eval(cfg["model"]["class"])(
        in_features=in_features,
        input_shape=input_shape,
        in_channels=in_channels,
        out_features=out_features,
        neurons=neurons,
        neuron_epochs=neuron_epochs,
        neuron_lrate=neuron_lrate,
        neuron_batch_size=neuron_batch_size,
        loss_fn=loss_fn,
        device=device,
        **model_params,
    )
    ema = EMA(model)
    nb_params_zero = model.number_of_parameters()
    cfg["model"]["nb_params_zero"] = nb_params_zero
    logger.log_metric(
        "complexity/nb of parameters",
        nb_params_zero,
        0,
        step_name="epoch",
    )
    logger.watch_pytorch_model(model)
    return {"model": model, "ema": ema}


def load_model(cfg, train_set: Dataset, logger: Logger, **_):
    input_shape = train_set.dataset.data.shape[1:-1]  # type: ignore
    in_channels = train_set.dataset.data.shape[-1]  # type: ignore
    out_features = len(np.unique(train_set.dataset.targets))  # type: ignore
    model_params = cfg["model"]

    loss_fn = _loss_fn(cfg)
    device = cfg["training"]["device"]

    neurons = cfg["growth"]["neurons"]
    neuron_epochs = cfg["growth"]["neuron_epochs"]
    neuron_lrate = cfg["growth"]["neuron_lrate"]
    neuron_batch_size = cfg["growth"]["neuron_batch_size"]

    model = eval(cfg["model"]["class"])(
        input_shape=input_shape,
        in_channels=in_channels,
        out_features=out_features,
        neurons=neurons,
        neuron_epochs=neuron_epochs,
        neuron_lrate=neuron_lrate,
        neuron_batch_size=neuron_batch_size,
        loss_fn=loss_fn,
        device=device,
        **model_params,
    )

    # Load artifacts with dag information
    artifact_path = cfg["model"]["artifact_path"]
    local_paths = logger.load_artifact(artifact_path)
    dag_states = []
    for i, path in enumerate(local_paths):
        with open(Path(path, f"graph_params_dag{i+1}.json"), "r") as f:
            dag_states.append(json.load(f))

    # Recreate dags in the model
    model.recreate_model(dag_states)
    model.update_size()

    # Load state dictionary
    load_weights = cfg["model"].get("load_weights", False)
    if load_weights:
        model_path = cfg["model"]["model_path"]
        local_path = logger.load_pytorch_model(model_path)
        state_dict = torch.load(local_path, map_location=device)

        candidate_edges = []  # lingering connections
        for key, value in state_dict.items():
            if ".layer.weight" in key and value.numel() == 0:
                root_key = key[:-12]
                candidate_edges.append(root_key)

        pattern = re.compile(r"_[a-z]\b")
        mod_state_dict = OrderedDict(
            (pattern.sub("", k), v)
            for k, v in state_dict.items()
            if all(cand not in k for cand in candidate_edges)
        )

        model.load_state_dict(mod_state_dict, strict=True)

    ema = EMA(model)
    nb_params_zero = model.number_of_parameters()
    cfg["model"]["nb_params_zero"] = nb_params_zero
    logger.log_metric(
        "complexity/nb of parameters",
        nb_params_zero,
        0,
        step_name="epoch",
    )
    logger.watch_pytorch_model(model)
    print(model)
    return {"model": model, "ema": ema}


def improved(current, best, mode="min", abs_delta=1e-4):
    if mode == "min":
        return current <= best - abs_delta
    else:
        return current >= best + abs_delta


def log_learning_rates(logger: Logger, optimizer: torch.optim.Optimizer, epoch: int):
    lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if not lrs:
        return
    logger.log_metric("training/learning rate", lrs[0], epoch, step_name="epoch")
    if len(lrs) > 1:
        for index, lr in enumerate(lrs):
            logger.log_metric(
                f"training/learning rate/group {index}",
                lr,
                epoch,
                step_name="epoch",
            )
        logger.log_metric(
            "training/learning rate/min", min(lrs), epoch, step_name="epoch"
        )
        logger.log_metric(
            "training/learning rate/max", max(lrs), epoch, step_name="epoch"
        )


def train(
    cfg,
    model: GrowingContainer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    logger: Logger,
    ema: EMA,
    functional_train_probe_batches: list[tuple[torch.Tensor, torch.Tensor]]
    | None = None,
    strikes: dict[str, int] = None,
    training_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    **context,
):
    global_epoch = cfg.get("global_epoch", 0)
    global_step = cfg.get("global_step", -1)

    # Stopping criterion
    if strikes is not None and all(v >= 2 for v in strikes.values()):
        print(f"Desired update norm was lower than the required threshold for all layers for more than 2 step. Skipping intermediate training.")
        return {}

    # Training hyperparameters
    device = cfg["training"]["device"]
    epochs = cfg["training"]["epochs"]
    epoch_scaling = cfg["training"]["epoch_scaling"]
    grad_clip = cfg["training"].get("grad_clip", None)
    early_stopping = cfg["training"].get("early_stopping", False)
    es_abs_delta = cfg["training"].get("es_abs_delta")
    es_patience = cfg["training"].get("es_patience", 5)
    es_counter = 0
    latest = torch.inf

    # Loss function
    loss_fn = _loss_fn(cfg)

    # Optimizer
    lrate = cfg["training"]["lrate"]
    weight_decay = cfg["training"]["weight_decay"]
    optimizer_name = cfg["training"]["optimizer"]
    transfer_sgd_momentum = _sgd_momentum_transfer_enabled(cfg)
    if transfer_sgd_momentum and not _optimizer_is_sgd(optimizer_name):
        raise ValueError(
            "training.ablations.sgd_momentum_transfer.enabled requires "
            "training.optimizer=torch.optim.SGD"
        )

    if "Adam" in optimizer_name:
        eps = cfg["training"].get("eps", 1e-8)
        optimizer = eval(optimizer_name)(
            model.parameters(),
            lr=lrate,
            weight_decay=weight_decay,
            eps=eps,
        )
    else:
        momentum = cfg["training"].get("momentum")
        optimizer = eval(optimizer_name)(
            model.parameters(),
            lr=lrate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    if transfer_sgd_momentum:
        stats = _transfer_sgd_momentum_buffers(
            optimizer=optimizer,
            model=model,
            previous_buffers=context.get("sgd_momentum_buffers"),
        )
        _log_sgd_momentum_transfer_stats(logger, stats, global_step)
        print(f"SGD momentum transfer stats: {stats}")

    # Scheduler
    scheduler_class = cfg["training"].get("scheduler")
    eta_min = cfg["training"].get("eta_min", 0.0)
    scheduler_ablations = get_pipeline_scheduler_ablation_settings(cfg["training"])
    growth_complete = cfg.get("growth_complete", False)
    defer_scheduler = should_defer_pipeline_scheduler(
        scheduler_ablations,
        growth_complete=growth_complete,
    )
    linear_warmup_enabled, linear_warmup_epochs = get_pipeline_linear_warmup_settings(
        scheduler_ablations,
        growth_complete=growth_complete,
    )
    is_growth_step_training = cfg.get("global_step", -1) >= 0 and not growth_complete
    use_growth_warmup_fixed_lr = (
        defer_scheduler and linear_warmup_enabled and is_growth_step_training
    )

    scheduler = None
    if scheduler_ablations.global_scheduler_enabled and scheduler_class is None:
        raise ValueError(
            "training.scheduler must be configured when "
            "training.ablations.global_scheduler.enabled is true"
        )
    if use_growth_warmup_fixed_lr:
        scheduler = get_pipeline_warmup_constant_scheduler(
            optimizer=optimizer,
            warmup_epochs=linear_warmup_epochs,
            warmup_start_lr=eta_min,
        )
    elif scheduler_class is not None and not defer_scheduler:
        if (
            scheduler_ablations.global_scheduler_enabled
            and training_scheduler is not None
        ):
            if (
                scheduler_ablations.global_rebind_linear_warmup_enabled
                and is_growth_step_training
            ):
                scheduler = rebind_pipeline_scheduler_with_warmup(
                    training_scheduler,
                    optimizer,
                    warmup_epochs=(
                        scheduler_ablations.global_rebind_linear_warmup_epochs
                    ),
                    warmup_start_lr=eta_min,
                )
            else:
                scheduler = rebind_pipeline_scheduler(training_scheduler, optimizer)
        else:
            scheduler_epochs = (
                scheduler_ablations.global_scheduler_total_epochs
                if scheduler_ablations.global_scheduler_enabled
                else epochs
            )
            scheduler = get_pipeline_scheduler(
                scheduler_class=eval(scheduler_class),
                optimizer=optimizer,
                num_epochs=scheduler_epochs,
                eta_min=eta_min,
                linear_warmup=linear_warmup_enabled,
                warmup_epochs=linear_warmup_epochs,
                clamp_after_end=scheduler_ablations.global_scheduler_enabled,
            )

    # Exponential Moving Average
    with_ema = cfg["model"]["ema"]

    cfg["has_grown"] = False
    if epoch_scaling is not None:
        p0 = cfg["model"].get("nb_params_zero")
        pi = model.number_of_parameters()
        epochs *= (p0 / pi) ** epoch_scaling
        epochs = max(int(epochs), 1)
        logger.log_metric("growth/epochs", epochs, global_step, step_name="growth step")
    # Stable-Tiny uses one fixed joint probe for train-side functional
    # diagnostics. Materialize its default four mini-batches from the first
    # real training iterator, then put those exact batches back at the front
    # of the epoch. The optimizer therefore sees the same samples, order and
    # augmentations it would have seen without this logging-only probe.
    first_epoch_train_batches = None
    if functional_train_probe_batches is None:
        functional_train_probe_batches, first_epoch_train_batches = (
            prepare_fixed_train_probe(train_dataloader)
        )

    validation_functional_output_snapshot = capture_network_outputs(
        model=model,
        dataloader=val_dataloader,
        device=device,
    )
    train_functional_output_snapshot = capture_network_outputs(
        model=model,
        dataloader=functional_train_probe_batches,
        device=device,
    )
    for epoch in range(global_epoch, global_epoch + epochs):
        print(f"Training at epoch {epoch}")
        log_learning_rates(logger, optimizer, epoch)
        linear_warmup_active = is_pipeline_linear_warmup_active(
            scheduler,
            scheduler_ablations,
            growth_complete=growth_complete,
        )
        correct, total, epoch_loss = 0, 0, 0
        if first_epoch_train_batches is not None:
            epoch_train_batches = first_epoch_train_batches
            first_epoch_train_batches = None
        else:
            epoch_train_batches = train_dataloader
        for x_batch, y_batch in epoch_train_batches:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ema.update(model)

            epoch_loss += loss.item()
            correct += (torch.argmax(output, 1) == y_batch).int().sum().item()
            total += len(output)
        if scheduler is not None:
            scheduler.step()

        accuracy = correct / total
        epoch_loss = epoch_loss / len(train_dataloader)
        # Measure ||grad L(f_t)||^2 and ||f_t - f_(t+1)||^2 at the network
        # output on the fixed, unaugmented validation set. The helper
        # normalizes over samples once, so these paper-facing diagnostics do
        # not depend on batch size.
        validation_output_functional_metrics = measure_output_functional_metrics(
            model=model,
            dataloader=val_dataloader,
            loss_fn=_loss_fn(cfg, reduction="sum"),
            device=device,
            reference_outputs=validation_functional_output_snapshot,
        )
        validation_functional_output_snapshot = (
            validation_output_functional_metrics.output_snapshot
        )
        train_output_functional_metrics = measure_output_functional_metrics(
            model=model,
            dataloader=functional_train_probe_batches,
            loss_fn=_loss_fn(cfg, reduction="sum"),
            device=device,
            reference_outputs=train_functional_output_snapshot,
        )
        train_functional_output_snapshot = (
            train_output_functional_metrics.output_snapshot
        )

        context.setdefault("acc_train", []).append(accuracy)
        context.setdefault("loss_train", []).append(epoch_loss)

        if with_ema:
            ema.apply_shadow(model)
        result = evaluate(
            cfg,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            **context,
        )
        if with_ema:
            ema.restore(model)
        context.update(result)

        logger.log_metric("training/train loss", epoch_loss, epoch, step_name="epoch")
        logger.log_metric("training/train accuracy", accuracy, epoch, step_name="epoch")
        logger.log_metric(
            "functional gradient/output squared L2 norm",
            validation_output_functional_metrics.gradient_squared_l2_norm,
            epoch,
            step_name="epoch",
        )
        logger.log_metric(
            "functional update/output displacement squared L2 norm",
            validation_output_functional_metrics.update_squared_l2_norm,
            epoch,
            step_name="epoch",
        )
        validation_approximation_metrics = {
            "functional approximation/relative error (g denominator)": (
                validation_output_functional_metrics.relative_error_approximation_denominator
            ),
            "functional approximation/relative error (gradient denominator)": (
                validation_output_functional_metrics.relative_error_gradient_denominator
            ),
            "functional approximation/cosine similarity": (
                validation_output_functional_metrics.directional_cosine
            ),
            "functional approximation/L2 distance to gradient": (
                validation_output_functional_metrics.approximation_l2_distance
            ),
        }
        for metric_name, metric_value in validation_approximation_metrics.items():
            if metric_value is not None:
                logger.log_metric(
                    metric_name,
                    metric_value,
                    epoch,
                    step_name="epoch",
                )
        split_functional_metrics = {
            "validation": validation_output_functional_metrics,
            "train": train_output_functional_metrics,
        }
        for split, metrics in split_functional_metrics.items():
            metric_values = {
                f"functional gradient/{split} output squared L2 norm": (
                    metrics.gradient_squared_l2_norm
                ),
                f"functional update/{split} output displacement squared L2 norm": (
                    metrics.update_squared_l2_norm
                ),
                f"functional approximation/{split} scale-optimal eta": (
                    metrics.scale_optimal_learning_rate
                ),
                f"functional approximation/{split} relative error (g denominator)": (
                    metrics.relative_error_approximation_denominator
                ),
                f"functional approximation/{split} relative error (gradient denominator)": (
                    metrics.relative_error_gradient_denominator
                ),
                f"functional approximation/{split} cosine similarity": (
                    metrics.directional_cosine
                ),
                f"functional approximation/{split} L2 distance to gradient": (
                    metrics.approximation_l2_distance
                ),
            }
            for metric_name, metric_value in metric_values.items():
                if metric_value is not None:
                    logger.log_metric(
                        metric_name,
                        metric_value,
                        epoch,
                        step_name="epoch",
                    )
        logger.log_metric(
            "training/val loss", result["loss_val"][epoch], epoch, step_name="epoch"
        )
        logger.log_metric(
            "training/val accuracy", result["acc_val"][epoch], epoch, step_name="epoch"
        )
        logger.log_metric(
            "training/test loss", result["loss_test"][epoch], epoch, step_name="epoch"
        )
        logger.log_metric(
            "training/test accuracy", result["acc_test"][epoch], epoch, step_name="epoch"
        )

        if early_stopping and not linear_warmup_active:
            if improved(result["loss_val"][-1], latest, abs_delta=es_abs_delta):
                es_counter = 0
                # best_state = copy.deepcopy(model.state_dict())
            else:
                es_counter += 1

            if es_counter >= es_patience:
                print(
                    f"Early stopping at epoch {epoch - global_epoch} with {latest - result['loss_val'][-1]:.4f} < {es_abs_delta} sensitivity"
                )
                logger.log_metric(
                    "growth/epochs",
                    epoch - global_epoch + 1,
                    global_step,
                    step_name="growth step",
                )
                break
            latest = result["loss_val"][-1]

    cfg["global_epoch"] = epoch + 1
    result["model"] = model
    result["acc_train"] = context["acc_train"]
    result["loss_train"] = context["loss_train"]
    result["functional_train_probe_batches"] = functional_train_probe_batches
    if transfer_sgd_momentum:
        result["sgd_momentum_buffers"] = _snapshot_sgd_momentum_buffers(
            optimizer,
            model,
        )
    if scheduler_ablations.global_scheduler_enabled and not defer_scheduler:
        result["training_scheduler"] = get_persistent_pipeline_scheduler(scheduler)
    return result


def evaluate(
    cfg,
    model: GrowingContainer,
    train_dataloader: DataLoader,
    growth_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    strikes: dict[str, int] = None,
    **context,
):  
    # Stopping criterion
    if strikes is not None and all(v >= 2 for v in strikes.values()):
        return {}

    has_grown = cfg.get("has_grown", False)
    loss_fn = _loss_fn(cfg)
    # global_step = cfg.get("global_step", 0)
    if has_grown:
        # Evaluation after growth
        growth_acc_growth, growth_loss_growth = evaluate_dataset(
            model, growth_dataloader, loss_fn=loss_fn
        )
        growth_acc_dev, growth_loss_dev = evaluate_dataset(
            model, dev_dataloader, loss_fn=loss_fn
        )
        growth_acc_val, growth_loss_val = evaluate_dataset(
            model, val_dataloader, loss_fn=loss_fn
        )
        growth_acc_test, growth_loss_test = evaluate_dataset(
            model, test_dataloader, loss_fn=loss_fn
        )
        results = {
            "growth_acc_growth": growth_acc_growth,
            "growth_acc_dev": growth_acc_dev,
            "growth_acc_val": growth_acc_val,
            "growth_acc_test": growth_acc_test,
            "growth_loss_growth": growth_loss_growth,
            "growth_loss_dev": growth_loss_dev,
            "growth_loss_val": growth_loss_val,
            "growth_loss_test": growth_loss_test,
        }
    else:
        # Evaluation after training
        # acc_train, loss_train = evaluate_dataset(model, train_dataloader, loss_fn)
        # acc_dev, loss_dev = evaluate_dataset(model, dev_dataloader, loss_fn)
        acc_val, loss_val = evaluate_dataset(model, val_dataloader, loss_fn)
        acc_test, loss_test = evaluate_dataset(model, test_dataloader, loss_fn)
        results = {
            # "acc_train": acc_train,
            # "acc_dev": acc_dev,
            "acc_val": acc_val,
            "acc_test": acc_test,
            # "loss_train": loss_train,
            # "loss_dev": loss_dev,
            "loss_val": loss_val,
            "loss_test": loss_test,
        }
    for key, value in results.items():
        context.setdefault(key, []).append(value)
    return {key: context[key] for key in results.keys()}


def evaluate_and_log(
    cfg,
    model: GrowingContainer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    logger: Logger,
    **context,
):
    global_epoch = cfg.get("global_epoch", 0)
    has_grown = cfg.get("has_grown", False)
    loss_fn = _loss_fn(cfg)
    if not has_grown:
        # Evaluation after training
        acc_train, loss_train = evaluate_dataset(model, train_dataloader, loss_fn)
        acc_val, loss_val = evaluate_dataset(model, val_dataloader, loss_fn)
        acc_test, loss_test = evaluate_dataset(model, test_dataloader, loss_fn)
        results = {
            "acc_train": acc_train,
            "acc_val": acc_val,
            "acc_test": acc_test,
            "loss_train": loss_train,
            "loss_val": loss_val,
            "loss_test": loss_test,
        }
        logger.log_metric(
            "training/train loss", loss_train, global_epoch, step_name="epoch"
        )
        logger.log_metric(
            "training/train accuracy", acc_train, global_epoch, step_name="epoch"
        )
        logger.log_metric("training/val loss", loss_val, global_epoch, step_name="epoch")
        logger.log_metric(
            "training/val accuracy", acc_val, global_epoch, step_name="epoch"
        )
        logger.log_metric(
            "training/test loss", loss_test, global_epoch, step_name="epoch"
        )
        logger.log_metric(
            "training/test accuracy", acc_test, global_epoch, step_name="epoch"
        )
        for key, value in results.items():
            context.setdefault(key, []).append(value)
        return {key: context[key] for key in results.keys()}


def disable_early_stopping(cfg, **_):
    cfg["training"]["early_stopping"] = False
    cfg["growth_complete"] = True
    return {"strikes": None}


def cycle_layers(cfg, model: GrowingContainer, **_):
    current_layer_idx = cfg.get("current_layer_idx", -1)
    growing_dag = None
    while not isinstance(growing_dag, GrowingGraphNetwork):
        current_layer_idx = (current_layer_idx + 1) % len(model._growing_layers)
        growing_dag = model._growing_layers[current_layer_idx]
    cfg["current_layer_idx"] = current_layer_idx
    return {"growing_dag": growing_dag}


def define_growth_actions(cfg, growing_dag: GrowingGraphNetwork, logger: Logger, **_):
    global_step = cfg["global_step"]
    expand_end = True
    # if not isinstance(growing_dag.dag.get_node_module(growing_dag.dag.end).reshape_function, torch.nn.Identity):
    #     expand_end = False
    actions = growing_dag.dag.define_next_actions(expand_end=expand_end)
    logger.log_metric(
        "complexity/nb of actions", len(actions), global_step, step_name="growth step"
    )
    return {"actions": actions}


def init_computation(cfg, model: GrowingContainer, **_):
    model.init_computation()


def update_computation(
    cfg,
    model: GrowingContainer,
    growing_dag: GrowingGraphNetwork,
    growth_dataloader: DataLoader,
    actions: list,
    logger: Logger,
    strikes: dict[str, int] = None,
    **_,
):
    loss_fn = _loss_fn(cfg)
    growth_batch_size = cfg["growth"]["batch_size"]
    max_activities_size = cfg["growth"].get("max_activities_size", np.inf)
    du_threshold = cfg["growth"].get("du_threshold", 0)
    global_step = cfg["global_step"]

    # Stopping criterion
    if strikes is None:
        strikes = {
            layer._name : 0
            for layer in model._growing_layers
            if isinstance(layer, GrowingGraphNetwork)
        }
    if strikes[growing_dag._name] >= 2:
        print(f"Desired update norm was lower than the required threshold for {growing_dag._name} for more than 2 step. Skipping growth step.")
        return {"strikes": strikes, **_release_growth_tensors()}

    # Initialize activities - ensure consistency
    all_nodes = list(growing_dag.dag.nodes)
    next_node_modules = []
    for action in actions:
        if isinstance(action, InterMergeExpansion):
            # Append next nodes
            next_node_modules.extend(action.next_nodes)
            # Append sibling nodes
            for n in action.next_nodes:
                for module in n.previous_modules:
                    prev_module = module.previous_module
                    if isinstance(prev_module.previous_modules[0], MergeGrowingModule):
                        prev_module = prev_module.previous_modules[0]

                    if (
                        prev_module not in all_nodes
                        and prev_module not in next_node_modules
                    ):
                        next_node_modules.append(prev_module)

    pre_activities_grad = {
        node: []
        for node in all_nodes + [n._name for n in next_node_modules]
        if (node != growing_dag.dag.root) and ("start" not in node)
    }
    input_B = {
        node: [] for node in all_nodes + [n._name for n in next_node_modules]
    }

    # Forward - backward loop
    root_key = growing_dag.dag.root
    accumulated_samples = 0
    for X, Y in growth_dataloader:
        X, Y = X.to(model.device), Y.to(model.device)
        model.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)  # TODO: check loss reduction
        loss.backward()
        # Update tensors
        model.update_computation()

        # Accumulate pre-activity gradients and input tensors on cpu
        if accumulated_samples > max_activities_size - growth_batch_size:
            continue
        for node_module in set(
            growing_dag.dag.get_all_node_modules() + next_node_modules
        ):
            assert node_module.activity is not None

            # Save input activity of input layers
            activity = node_module.activity.clone().detach().cpu()
            if hasattr(node_module, "reshape_function"):
                activity = node_module.reshape_function(activity)
            input_B[node_module._name].append(activity)

            if node_module._name == root_key:
                accumulated_samples += activity.size(0)

            if (node_module._name == root_key) or (
                "start" in node_module._name
            ):
                continue
            assert node_module.pre_activity is not None
            assert node_module.pre_activity.grad is not None
            # Save pre activity gradients
            pre_activities_grad[node_module._name].append(
                node_module.pre_activity.grad.clone().detach().cpu()
            )

    pre_activities_grad = {k: torch.cat(v) if v else torch.empty(0) for k, v in pre_activities_grad.items()}
    pre_activities_grad, noisy_grad_metrics = add_noise_to_pre_activities_grad(
        pre_activities_grad,
        cfg,
    )
    for metric_name, metric_value in noisy_grad_metrics.items():
        logger.log_metric(
            f"growth/noisy_pre_activities_grad/{metric_name}",
            metric_value,
            global_step,
            step_name="growth step",
        )
    input_B = {k: torch.cat(v) if v else torch.empty(0) for k, v in input_B.items()}

    for node, grad in pre_activities_grad.items():
        node = re.sub("@", ".", node)
        logger.log_metric_with_stats(
            f"growth/desired update/node {node}",
            grad,
            global_step,
            step_name="growth step",
        )

    du_end_node = pre_activities_grad[growing_dag.dag.end]
    du_norm = torch.linalg.norm(du_end_node) / np.sqrt(torch.numel(du_end_node))
    if du_norm < du_threshold:
        strikes[growing_dag._name] += 1
        print(f"Desired update norm is lower than the required threshold for {growing_dag._name} ({du_norm.item()} < {du_threshold}). Strike {strikes[growing_dag._name]}.")
    else:
        strikes[growing_dag._name] = 0

    return {
        "strikes": strikes,
        "pre_activities_grad": pre_activities_grad,
        "input_B": input_B,
        "bottleneck": None,
    }


def reset_computation(cfg, model: GrowingContainer, **_):
    model.reset_computation()


def compute_optimal_delta(
    cfg,
    model: GrowingContainer,
    growing_dag: GrowingGraphNetwork,
    strikes: dict[str, int],
    **_,
):
    # Stopping criterion
    if strikes[growing_dag._name] >= 2:
        return

    model.compute_optimal_delta()


def calculate_bottleneck(
    cfg,
    growing_dag: GrowingGraphNetwork,
    logger: Logger,
    actions: list,
    strikes: dict[str, int],
    pre_activities_grad: dict[str, torch.Tensor] | None = None,
    input_B: dict[str, torch.Tensor] | None = None,
    **_,
):
    global_step = cfg["global_step"]

    # Stopping criterion
    if strikes[growing_dag._name] >= 2:
        return _release_growth_tensors()

    pre_activities_grad = _require_growth_tensor(
        "pre_activities_grad", pre_activities_grad
    )
    input_B = _require_growth_tensor("input_B", input_B)

    # Extract next_node_modules from actions
    next_node_modules = []
    for action in actions:
        if isinstance(action, InterMergeExpansion):
            next_node_modules.extend(action.next_nodes)

    bottleneck = {}
    with torch.no_grad():
        for node_module in set(
            growing_dag.dag.get_all_node_modules() + next_node_modules
        ):
            if node_module._name == growing_dag.dag.root:
                continue

            # Check if node exists in pre_activities_grad before accessing
            if node_module._name not in pre_activities_grad:
                raise KeyError(f"Activity gradient {node_module._name} was not recorded")

            # Compute expressivity bottleneck
            v_proj = pre_activities_grad[node_module._name].clone()
            for module in node_module.previous_modules:
                # Safely check if previous module exists in input_B
                prev_module = module.previous_module
                if prev_module._name not in input_B:
                    if isinstance(prev_module.previous_modules[0], MergeGrowingModule):
                        prev_module = prev_module.previous_modules[0]

                if prev_module._name not in input_B:
                    raise KeyError(f"Input activity {prev_module._name} was not recorded")

                input_activity = input_B[prev_module._name]
                v_proj -= module.optimal_delta_layer(
                    input_activity.to(module.device)  # type: ignore
                ).cpu()

            # Normalize bottleneck
            eps = 1e-8
            v_proj_norm = torch.einsum("b...,b...->b", v_proj, v_proj).mean()
            v_proj = v_proj / torch.sqrt(v_proj_norm + eps)
            bottleneck[node_module._name] = v_proj

    for node, bott in bottleneck.items():
        node = re.sub("@", ".", node)
        logger.log_metric_with_stats(
            f"growth/bottleneck/node {node}", bott, global_step, step_name="growth step"
        )

    return {"bottleneck": bottleneck}


def delete_update(cfg, model: GrowingContainer, **_):
    for layer in model._growing_layers:
        if isinstance(layer, GrowingGraphNetwork):
            layer.delete_update()
        elif isinstance(layer, GrowingModule):
            layer.delete_update(include_previous=False, delete_output=True)


def estimate_most_important_node(
    cfg,
    growing_dag: GrowingGraphNetwork,
    strikes: dict[str, int],
    bottleneck: dict[str, torch.Tensor] | None = None,
    **_,
):
    # Stopping criterion
    if strikes[growing_dag._name] >= 2:
        return {"most_important_node": None}

    bottleneck = _require_growth_tensor("bottleneck", bottleneck)
    bott_norms = {
        key: torch.linalg.norm(val)
        for key, val in bottleneck.items()
        if key in growing_dag.dag.nodes
    }
    most_important_node = max(bott_norms.items(), key=operator.itemgetter(1))[0]
    print(
        f"Found most important node {most_important_node} with norm {bott_norms[most_important_node]}"
    )
    if not hasattr(growing_dag, "bott_norm") or growing_dag.bott_norm is None:
        growing_dag.bott_norm_prev = bott_norms[growing_dag.dag.end]
    else:
        growing_dag.bott_norm_prev = growing_dag.bott_norm
    growing_dag.bott_norm = bott_norms[growing_dag.dag.end]
    return {"most_important_node": most_important_node}


def estimate_dependencies(
    cfg,
    growing_dag: GrowingGraphNetwork,
    most_important_node: str,
    strikes: dict[str, int],
    logger: Logger,
    bottleneck: dict[str, torch.Tensor] | None = None,
    input_B: dict[str, torch.Tensor] | None = None,
    **_,
):
    global_step = cfg["global_step"]
    sample_size = cfg["growth"]["dependency_estimation_sample_size"]

    # Stopping criterion
    if strikes[growing_dag._name] >= 2:
        return {"dominant_inputs": None}

    bottleneck = _require_growth_tensor("bottleneck", bottleneck)
    bottleneck = torch.flatten(bottleneck[most_important_node], 1)
    input_B = _require_growth_tensor("input_B", input_B)
    # if len(actions) > 3:
    input_B = {
        name: torch.flatten(value, 1)
        for name, value in input_B.items()
        if name in growing_dag.dag.nodes
        and name not in nx.descendants(growing_dag.dag, most_important_node)
        and name != most_important_node
    }
    hsic = calculate_dependency(input_B, bottleneck, n_samples=sample_size)
    hsic_values = torch.tensor(
        list(hsic.values()), device=global_device(), dtype=torch.float
    )
    percentile = torch.quantile(hsic_values, 0.9)
    dominant_inputs = [name for name, value in hsic.items() if value >= percentile]

    for name, value in hsic.items():
        name = re.sub("@", ".", name)
        logger.log_metric(
            f"actions/HSIC/node {name}", value, global_step, step_name="growth step"
        )
    logger.log_metric(
        "actions/HSIC 0.9 quantile", percentile, global_step, step_name="growth step"
    )

    return {"dominant_inputs": dominant_inputs}


def restrict_growth_actions(
    cfg,
    growing_dag: GrowingGraphNetwork,
    actions: list,
    strikes: dict[str, int],
    logger: Logger,
    most_important_node: str = None,
    dominant_inputs: list[str] = None,
    **_,
):
    global_step = cfg["global_step"]

    # Stopping criterion
    if strikes[growing_dag._name] >= 2:
        actions = []

    # Restrict output node
    if most_important_node is not None:
        actions = growing_dag.restrict_action_space(
            actions, chosen_outputs=[most_important_node]
        )
    # Restrict input node
    if dominant_inputs is not None:
        actions = growing_dag.restrict_action_space(actions, chosen_inputs=dominant_inputs)
    logger.log_metric(
        "complexity/nb of tested actions",
        len(actions),
        global_step,
        step_name="growth step",
    )
    return {"actions": actions}


def pick_random_action(
        cfg,
        growing_dag: GrowingGraphNetwork,
        actions: list,
        strikes: dict[str, int],
        logger: Logger,
        **_
):
    global_step = cfg["global_step"]

    # Stopping criterion
    if strikes[growing_dag._name] >= 2:
        actions = []
    else:
        actions = random.choices(actions, k=1)

    logger.log_metric(
        "complexity/nb of tested actions",
        len(actions),
        global_step,
        step_name="growth step",
    )

    return {"actions": actions}


def execute_expansion(
    cfg,
    growing_dag: GrowingGraphNetwork,
    actions: list,
    logger: Logger,
    strikes: dict[str, int],
    bottleneck: dict[str, torch.Tensor] | None = None,
    input_B: dict[str, torch.Tensor] | None = None,
    **_,
):
    global_step = cfg.get("global_step", -1)
    growth_cfg = cfg["growth"]
    neuron_selection_threshold = growth_cfg.get("neuron_selection_threshold", -np.inf)

    neuron_scaling = growth_cfg.get("neuron_scaling", 0.0)
    if neuron_scaling != 0:
        v_proj_prev = growing_dag.bott_norm_prev
        v_proj = growing_dag.bott_norm
        lambda_vproj = v_proj / v_proj_prev
        lambda_vproj = min(lambda_vproj, 1)
        neurons = growing_dag.neurons
        neurons = min(neurons, neuron_scaling * lambda_vproj * neurons)
        neurons = max(1, neurons)
        growing_dag.neurons = int(np.ceil(neurons))
        logger.log_metric("growth/neurons", neurons, global_step, step_name="growth step")

    # Stopping criterion
    if strikes[growing_dag._name] >= 2:
        return

    bottleneck = _require_growth_tensor("bottleneck", bottleneck)
    input_B = _require_growth_tensor("input_B", input_B)

    _validate_init_scaling_ablation_strategy(cfg)

    # Execute all graph growth actions with gromo's standard local initialization.
    growing_dag.global_step = cfg["global_step"]
    growing_dag.execute_expansions(
        actions=actions,
        bottleneck=bottleneck,
        input_B=input_B,
        amplitude_factor=False,
        evaluate=False,
        neuron_selection_threshold=neuron_selection_threshold,
        verbose=False,
    )

    for expansion in actions:
        if expansion.type != ExpansionType.NEW_EDGE:
            node = expansion.expanding_node

            neuron_foi = expansion.metrics["neuron_foi"]
            logger.log_histogram(
                f"neurons/foi_bott_hist/{node}",
                neuron_foi,
                global_step,
                step_name="growth step",
            )

            try:
                for in_edge in expansion.in_edges:
                    _alpha = in_edge.extended_output_layer.weight
                    logger.log_histogram(
                        f"neurons/alpha/{node}",
                        _alpha,
                        global_step,
                        step_name="growth step",
                    )
                    if in_edge.use_bias:
                        _bias = in_edge.extended_output_layer.bias
                        logger.log_histogram(
                            f"neurons/bias/{node}",
                            _bias,
                            global_step,
                            step_name="growth step",
                        )
                for out_edge in expansion.out_edges:
                    _omega = out_edge.extended_input_layer.weight
                    logger.log_histogram(
                        f"neurons/omega/{node}",
                        _omega,
                        global_step,
                        step_name="growth step",
                    )
            except AttributeError:
                pass
        else:
            node = expansion.next_node
            for edge in expansion.in_edges:
                _alpha = edge.optimal_delta_layer.weight
                logger.log_histogram(
                    f"neurons/alpha/{node}", _alpha, global_step, step_name="growth step"
                )
                if edge.use_bias:
                    _bias = edge.optimal_delta_layer.bias
                    logger.log_histogram(
                        f"neurons/bias/{node}",
                        _bias,
                        global_step,
                        step_name="growth step",
                    )
        if expansion.metrics.get("skip", False):
            expansion.delete()

    actions = [
        expansion for expansion in actions if not expansion.metrics.get("skip", False)
    ]
    return {"actions": actions}


def calculate_amplitude_factor(
    cfg,
    model: GrowingContainer,
    growing_dag: GrowingGraphNetwork,
    actions: list,
    growth_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    val_dataloader: DataLoader,
    strikes: dict[str, int],
    logger: Logger,
    pre_activities_grad: dict[str, torch.Tensor] | None = None,
    bottleneck: dict[str, torch.Tensor] | None = None,
    **_,
):
    global_step = cfg["global_step"]
    loss_fn = eval(cfg["training"]["loss_function"])
    label_smoothing = _active_label_smoothing(cfg)
    use_line_search = cfg["growth"].get("use_line_search", True) and (
        not _init_scaling_ablation_enabled(cfg)
        or _defer_init_scaling_ablation_to_final_choice(cfg)
    )
    fixed_scaling_factor = cfg["growth"].get("fixed_scaling_factor", 1.0)

    # Stopping criterion
    if strikes[growing_dag._name] >= 2:
        return _release_growth_tensors()

    _validate_init_scaling_ablation_strategy(cfg)

    if not use_line_search:
        for expansion in actions:
            model.set_scaling_factor(fixed_scaling_factor)
            expansion.metrics["scaling_factor"] = fixed_scaling_factor
            expansion.evaluate(
                model=model,
                train_dataloader=None,
                dev_dataloader=None,
                val_dataloader=val_dataloader,
                loss_fn=loss_fn(reduction="mean", label_smoothing=label_smoothing),
            )
        return _release_growth_tensors()

    pre_activities_grad = _require_growth_tensor(
        "pre_activities_grad", pre_activities_grad
    )
    bottleneck = _require_growth_tensor("bottleneck", bottleneck)
    use_foi = cfg["growth"].get("use_first_order_improvement", True)
    default_foi = cfg["growth"].get("default_foi", 1)

    # Calculate initial loss
    model.set_scaling_factor(0.0)
    initial_loss, _ = extended_evaluate_model(
        growing_model=model,
        dataloader=dev_dataloader,
        loss_function=loss_fn(reduction="sum", label_smoothing=label_smoothing),
        mask={},
        device=model.device,
    )

    # Compute amplitude factor
    for expansion in actions:
        mask = expansion.create_mask()

        # Find layer index
        for i, layer in enumerate(model._growing_layers):
            if isinstance(layer, GrowingGraphNetwork) and expansion.dag == layer.dag:
                index = i
                break
        model.currently_updated_layer_index = index

        # Compute first order improvement
        if use_foi:
            first_order_improvement = 0
            block_update = expansion.metrics[
                "block_output"
            ]  # activity updates from new layer block

            if isinstance(expansion, InterMergeExpansion):
                next_node_modules = expansion.next_nodes
            else:
                next_node_modules = expansion.dag.get_node_modules(expansion.next_nodes)

            for node_module in next_node_modules:
                # First order improvement on the loss
                gradient = pre_activities_grad[node_module._name]  # pre_activity gradient
                if gradient.dim() == 2:
                    gradient = gradient.unsqueeze(-1).unsqueeze(-1)
                # block first order improvement
                block_foi_loss = -torch.einsum(
                    "bchw,bchw->bc", gradient, block_update[node_module._name]
                ).mean(dim=0)
                logger.log_histogram(
                    f"growth/neuron_foi_loss_hist/{node_module._name}",
                    block_foi_loss,
                    global_step,
                    "growth step",
                )
                block_foi_loss = block_foi_loss.sum()

                # First order improvement on the bottleneck
                bott = bottleneck[node_module._name]
                if bott.dim() == 2:
                    bott = bott.unsqueeze(-1).unsqueeze(-1)
                block_foi_bott = -torch.einsum(
                    "bchw,bchw->bc", bott, block_update[node_module._name]
                ).mean(dim=0)
                logger.log_histogram(
                    f"growth/neuron_foi_bott_hist/{node_module._name}",
                    block_foi_bott,
                    global_step,
                    "growth step",
                )
                block_foi_bott = block_foi_bott.sum()
                logger.log_metric(
                    f"growth/foi_bott/{node_module._name}",
                    block_foi_bott,
                    global_step,
                    "growth step",
                )

                first_order_improvement += (
                    block_foi_bott + node_module.parameter_update_decrease
                )
            logger.log_metric(
                f"growth/foi/{node_module._name}",
                first_order_improvement,
                global_step,
                "growth step",
            )
            first_order_improvement = max(first_order_improvement, 1e-7)

        # Execute line search
        factor, _, _, _, _, _ = line_search(
            model=model,
            dataloader=dev_dataloader,
            # reduction="mean": line_search evaluates through gromo's evaluate_model,
            # whose AverageMeter does NOT normalize by batch size and asserts
            # reduction == "mean". A mean loss here yields the same per-sample-mean
            # scale as initial_loss (from extended_evaluate_model) and
            # first_order_improvement, reproducing Theo's original line-search
            # dynamics (his gromo divided the summed loss by batch size internally).
            loss_function=loss_fn(reduction="mean", label_smoothing=label_smoothing),
            mask=mask,
            initial_loss=initial_loss,
            first_order_improvement=first_order_improvement if use_foi else default_foi,
            device=model.device,
        )
        model.set_scaling_factor(factor)
        expansion.metrics["scaling_factor"] = factor
        expansion.evaluate(
            model=model,
            train_dataloader=None,
            dev_dataloader=None,
            val_dataloader=val_dataloader,
            loss_fn=loss_fn(reduction="mean", label_smoothing=label_smoothing),
        )

    return _release_growth_tensors()


def choose_best_growth_action(
    cfg,
    model: GrowingContainer,
    growing_dag: GrowingGraphNetwork,
    actions: list,
    logger: Logger,
    **_,
):
    global_step = cfg["global_step"]

    # Find action that generates minimum loss
    if len(actions) > 0:
        growing_dag.choose_growth_best_action(actions, use_bic=False, verbose=True)
        if _defer_init_scaling_ablation_to_final_choice(cfg):
            factor = cfg["growth"].get("fixed_scaling_factor", 1.0)
        else:
            factor = growing_dag.chosen_action.metrics["scaling_factor"]
        logger.log_metric(
            "growth/amplitude factor", factor, global_step, step_name="growth step"
        )
        node = (
            growing_dag.chosen_action.expanding_node
            if growing_dag.chosen_action.expanding_node is not None
            else growing_dag.chosen_action.next_node
        )
        logger.log_metric(
            f"neurons/amplitude factor/{node}",
            factor,
            global_step,
            step_name="growth step",
        )

        if growing_dag.chosen_action.type != ExpansionType.NEW_EDGE:
            active_neurons = growing_dag.chosen_action.metrics["active_neurons"]
            logger.log_metric(
                "growth/neurons", active_neurons, global_step, step_name="growth step"
            )

        # Apply chosen action to all graphs
        mask = growing_dag.chosen_action.create_mask()
        for layer in model._growing_layers:
            if isinstance(layer, GrowingGraphNetwork) and layer != growing_dag:
                layer.chosen_action = growing_dag.chosen_action
                layer.clean_graph_with_chosen_action(actions)
            elif isinstance(layer, GrowingModule):
                layer.delete_update(
                    include_previous=False,
                    delete_delta=False,
                    delete_input=(
                        True
                        if layer.previous_module is None
                        else layer.previous_module._name not in mask.get("nodes", [])
                    ),
                    delete_output=(
                        True
                        if layer.next_module is None
                        else layer.next_module._name not in mask.get("nodes", [])
                    ),
                )
    else:
        logger.log_metric(
            "growth/neurons", 0, global_step, step_name="growth step"
        )
        for layer in model._growing_layers:
            if isinstance(layer, GrowingGraphNetwork):
                layer.delete_update()
                layer.chosen_action = None


def apply_change(cfg, model: GrowingContainer, growing_dag: GrowingGraphNetwork, **_):
    if growing_dag.chosen_action is not None:
        extension_size = growing_dag.chosen_action.metrics.get(
            "active_neurons", growing_dag.neurons
        )
        disable_optimal_delta = _disable_optimal_delta_on_apply_change(cfg)
        variance_transfer = get_variance_transfer_settings(cfg)
        if variance_transfer.enabled:
            growing_dag.apply_expansion_rescaling(
                expansion=growing_dag.chosen_action,
                rescaling=variance_transfer.rescaling,
                extension_size=extension_size,
            )

        if _defer_init_scaling_ablation_to_final_choice(cfg):
            incoming_init = _growth_explicit_init_mode(cfg, "incoming_init", "kaiming")
            outgoing_init = _growth_explicit_init_mode(cfg, "outgoing_init", "zeros")
            new_edge_init = _growth_explicit_init_mode(cfg, "new_edge_init", "zeros")
            growing_dag.initialise_expansion_from_modes(
                expansion=growing_dag.chosen_action,
                extension_size=extension_size,
                incoming_init=incoming_init,
                outgoing_init=outgoing_init,
                new_edge_init=new_edge_init,
                overwrite_existing=True,
            )
            growing_dag.chosen_action.metrics["scaling_factor"] = cfg["growth"].get(
                "fixed_scaling_factor", 1.0
            )
        # Apply changes
        for layer in model._growing_layers:
            if isinstance(layer, GrowingGraphNetwork):
                if disable_optimal_delta:
                    _apply_growing_graph_change_without_optimal_delta(layer)
                else:
                    layer.apply_change()
            elif isinstance(layer, GrowingModule):
                if disable_optimal_delta:
                    layer.apply_change(apply_previous=False, apply_delta=False)
                else:
                    layer.apply_change(apply_previous=False)
                if layer.extended_output_layer is not None:
                    layer._apply_output_changes(extension_size=cfg["growth"]["neurons"])
    cfg["has_grown"] = True


def update_size(cfg, model: GrowingContainer, **_):
    model.update_size()


def empty_cache(cfg, **_):
    gc.collect()
    device = cfg["training"]["device"]
    if device == "cuda":
        torch.cuda.empty_cache()
    if device == "mps":
        torch.mps.empty_cache()


def log_results(
    cfg,
    model: GrowingContainer,
    train_dataloader: DataLoader,
    logger: Logger,
    growing_dag: GrowingGraphNetwork = None,
    growth_acc_growth: list = None,
    growth_acc_dev: list = None,
    growth_acc_val: list = None,
    growth_acc_test: list = None,
    growth_loss_growth: list = None,
    growth_loss_dev: list = None,
    growth_loss_val: list = None,
    growth_loss_test: list = None,
    **_,
):
    global_epoch = cfg.get("global_epoch", 0)
    global_step = cfg["global_step"]
    tmpdir = cfg["logger"]["tmpdir"]

    if growth_loss_growth is not None:
        logger.log_metric(
            "growth/loss", growth_loss_growth[-1], global_epoch, step_name="epoch"
        )
        logger.log_metric(
            "growth/dev loss", growth_loss_dev[-1], global_epoch, step_name="epoch"
        )
        logger.log_metric(
            "growth/val loss", growth_loss_val[-1], global_epoch, step_name="epoch"
        )
        logger.log_metric(
            "growth/test loss", growth_loss_test[-1], global_epoch, step_name="epoch"
        )

        logger.log_metric(
            "growth/accuracy", growth_acc_growth[-1], global_epoch, step_name="epoch"
        )
        logger.log_metric(
            "growth/dev accuracy", growth_acc_dev[-1], global_epoch, step_name="epoch"
        )
        logger.log_metric(
            "growth/val accuracy", growth_acc_val[-1], global_epoch, step_name="epoch"
        )
        logger.log_metric(
            "growth/test accuracy", growth_acc_test[-1], global_epoch, step_name="epoch"
        )

    nb_parameters = model.number_of_parameters()
    logger.log_metric(
        "complexity/nb of parameters",
        nb_parameters,
        global_epoch,
        step_name="epoch",
    )

    if growing_dag is not None:
        # nb of parameters per edge
        for edge in growing_dag.dag.edges:  # type: ignore
            edge0 = re.sub("@", ".", edge[0])
            edge1 = re.sub("@", ".", edge[1])
            params = growing_dag.dag.count_parameters([edge])  # type: ignore
            logger.log_metric(
                f"complexity/nb of parameters at/layer {edge0}_{edge1}",
                params,
                global_epoch,
                step_name="epoch",
            )
        # in-degree and out-degree per node
        for node in growing_dag.dag.nodes:  # type: ignore
            _node = re.sub("@", ".", node)
            # logger.log_metric(
            #     f"complexity/in-degree/node {_node}",
            #     growing_dag.dag.in_degree(node), # type: ignore
            #     global_epoch,
            # )
            # logger.log_metric(
            #     f"complexity/out-degree/node {_node}",
            #     growing_dag.dag.out_degree(node), # type: ignore
            #     global_epoch,
            # )
            logger.log_metric(
                f"complexity/size/node {_node}",
                growing_dag.dag.nodes[node]["size"],  # type: ignore
                global_epoch,
                step_name="epoch",
            )

    if cfg.get("logger", {}).get("save_intermediate_models", False):
        save_model(
            cfg,
            model=model,
            ema=None,
            train_dataloader=train_dataloader,
            logger=logger,
            model_specifier=f"_after_growth_step{global_step}",
            upload_artifact=False,
        )

    # Save growth history file
    try:
        path = os.path.join(tmpdir, "gh.json")
        with open(path, "w") as f:
            json.dump(growing_dag.growth_history, f)
            f.flush()
            logger.log_artifact(
                path=path, name="growth_history", metadata={"step": global_step}
            )
    except Exception as error:
        print(f"[Growth History] {error}")

    # Save interactive graph
    for layer in model._growing_layers:
        if isinstance(layer, GrowingGraphNetwork):
            name = layer.dag._name
            paths = []
            try:
                dag_params = layer.dag.export_dag_parameters()
                dag_params_path = os.path.join(tmpdir, f"graph_params_{name}.json")
                with open(dag_params_path, "w") as f:
                    json.dump(dag_params, f)
                    f.flush()
                paths.append(dag_params_path)
            except Exception as error:
                print(f"[DAG Params {name}] {error}")

            try:
                graph = DAG_to_pyvis(layer.dag)
                pyvis_path = os.path.join(tmpdir, f"graph_{name}.html")
                graph.save_graph(pyvis_path)
                paths.append(pyvis_path)
            except Exception as error:
                print(f"[Interactive DAG {name}] {error}")

            logger.log_artifact(
                path=paths, name=f"graph_{name}", metadata={"step": global_step}
            )

    return {"nb_params": nb_parameters}


def save_model(
    cfg,
    model: GrowingContainer,
    ema: EMA | None,
    train_dataloader: DataLoader,
    logger: Logger,
    model_specifier="",
    upload_artifact: bool = True,
    **_,
):
    global_step = cfg.get("global_step", -1)
    model_class = cfg["model"]["class"]
    device = cfg["training"]["device"]
    tmpdir = cfg["logger"]["tmpdir"]

    with torch.no_grad():
        try:
            logger.log_pytorch_model(
                model=model,
                name=f"{model_class}{model_specifier}",
                x=None,
                path=tmpdir,
                metadata={"step": global_step},
                upload_artifact=upload_artifact,
            )
        except Exception as error:
            print(f"[Growing Model] {error}")

        with_ema = cfg["model"]["ema"]
        if ema is not None and with_ema:
            ema.apply_shadow(model)
            try:
                logger.log_pytorch_model(
                    model=model,
                    name=f"{model_class}{model_specifier}_EMA",
                    x=None,
                    path=tmpdir,
                    metadata={"step": global_step},
                    upload_artifact=upload_artifact,
                )
            except Exception as error:
                print(f"[Growing Model EMA] {error}")
            ema.restore(model)


def save_intermediate_model(
    cfg,
    model: GrowingContainer,
    ema: EMA | None,
    train_dataloader: DataLoader,
    logger: Logger,
    **_,
):
    if not cfg.get("logger", {}).get("save_intermediate_models", False):
        return

    global_step = cfg.get("global_step", -1)
    save_model(
        cfg,
        model=model,
        ema=ema,
        train_dataloader=train_dataloader,
        logger=logger,
        model_specifier=f"_after_train_step{global_step}",
        upload_artifact=False,
    )


if __name__ == "__main__":
    import re

    from run_pipeline import available_steps

    for key, value in available_steps.items():
        func_name = re.split("[.]", value)[-1].strip()
        print(f"def {func_name}(cfg, data=None, **_):")
        print("\tpass")
        print()
