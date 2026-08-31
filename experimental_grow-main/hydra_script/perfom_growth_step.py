import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torch.utils.data
from gromo.containers.growing_block import GrowingBlock, RestrictedConv2dGrowingBlock
from gromo.containers.growing_container import GrowingContainer
from gromo.containers.growing_mlp import GrowingMLP
from gromo.containers.sequential_growing_container import SequentialGrowingModel
from gromo.modules.conv2d_growing_module import Conv2dGrowingModule
from gromo.modules.growing_module import GrowingModule
from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.utils.training_utils import compute_statistics, evaluate_model
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from hydra_script.line_search import (
    default_setter,
    line_search,
    set_extensions_only,
    set_input_extension_only,
)
from hydra_script.aux_train_and_grow import resolve_batch_limit
from tools.create_growth_schedule import ScheduleStep


def perform_growth_step(
    cfg: DictConfig,
    model: GrowingContainer,
    step: int,
    growth_schedule_step: ScheduleStep | None,
    dataloader,
    loss_function_growth,
    loss_function_train,
    top_1_accuracy,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    last_updated_layer: int,
    metric_prefix: str = "train",
) -> tuple[dict, int]:
    """
    Perform a growth step.

    Parameters
    ----------
    cfg : DictConfig
        Configuration
    model : GrowingContainer
        Model
    step : int
        Current step
    growth_schedule_step : ScheduleStep or None
        Growth schedule step (4-tuple: step_type, layer_index,
        maximum_added_neurons, batch_limit). If None, falls back to
        config-driven behaviour.
    dataloader : DataLoader
        Data loader for growth statistics and line search
    loss_function_growth : callable
        Loss function for growth
    loss_function_train : callable
        Loss function for training
    top_1_accuracy : callable
        Accuracy function
    optimizer : torch.optim.Optimizer
        Optimizer driving the surrounding training loop. Used by the
        ``scaled_fogro`` family of methods to read the currently-effective
        learning rate.
    device : torch.device
        Device
    last_updated_layer : int
        Last updated layer index
    metric_prefix : str, default="train"
        Prefix for metric names (e.g., 'train' or 'growth_set')

    Returns
    -------
    tuple
        logs dict, updated last_updated_layer
    """
    logger = logging.getLogger(__name__)
    logs: dict[str, float | int | Any] = dict()

    # Resolve the batch limit for this growth step
    if growth_schedule_step is not None:
        schedule_batch_limit_raw = growth_schedule_step[3]
    else:
        schedule_batch_limit_raw = "hyper-parameter"
    growth_batch_limit: int | None = resolve_batch_limit(
        raw_value=schedule_batch_limit_raw,
        dataloader_length=len(dataloader),
        config_fallback=cfg.growing.get("batch_limit", -1),
    )

    # Determine layer to grow and maximum neurons to add
    if growth_schedule_step is not None:
        _, schedule_layer_index, schedule_max_neurons, _ = growth_schedule_step

        if isinstance(model, SequentialGrowingModel):
            model.set_growing_layers(
                scheduling_method="sequential", index=schedule_layer_index
            )
        selector = model.dummy_select_update

        if schedule_max_neurons != -1:
            maximum_added_neurons = schedule_max_neurons
        else:
            if isinstance(model, SequentialGrowingModel):
                maximum_added_neurons = model.number_of_neurons_to_add(
                    number_of_growth_steps=cfg.growing.planned_growing_steps
                )
            else:
                maximum_added_neurons = cfg.growing.maximum_added_neurons
    else:
        # Use selection method
        selection_method = cfg.growing.selection_method
        if selection_method == "none":
            last_updated_layer = (last_updated_layer + 1) % len(model._growing_layers)

            def selector():
                return model.select_update(layer_index=last_updated_layer)

        elif selection_method == "fo":
            selector = model.select_best_update
        else:
            raise NotImplementedError(
                f"Selection method {selection_method} not implemented"
            )

        maximum_added_neurons = (
            cfg.growing.compute_optimal_updates_kwargs.maximum_added_neurons
        )
        if isinstance(model, SequentialGrowingModel):
            maximum_added_neurons = model.number_of_neurons_to_add(
                number_of_growth_steps=cfg.growing.planned_growing_steps
            )

    # 1) Gather Statistics
    if not cfg.growing.skip_compute_statistics:
        if hasattr(dataloader, "generator"):
            growth_dataloder_seed: int | None = torch.initial_seed() + step
        elif growth_batch_limit is not None:
            growth_dataloder_seed = None
            logger.warning(
                "Dataloader does not have a generator attribute. "
                "Cannot set seed for using the same batch for line search."
            )
        initial_loss, initial_accuracy = compute_statistics(
            model=model,
            dataloader=dataloader,
            loss_function=loss_function_growth,
            metrics=top_1_accuracy,
            batch_limit=growth_batch_limit,
            dataloader_seed=growth_dataloder_seed,
            device=device,
        )

        logs[f"initial_{metric_prefix}_loss"] = initial_loss
        logs[f"initial_{metric_prefix}_accuracy"] = initial_accuracy
    logs["maximum_added_neurons"] = maximum_added_neurons

    # 2) Compute optimal updates
    if cfg.growing.method == "fogro":
        if cfg.growing.compute_optimal_updates_kwargs.get("computation_dtype", None):
            growing_dtype_str = (
                cfg.growing.compute_optimal_updates_kwargs.computation_dtype
            )
            if growing_dtype_str.startswith("float"):
                growing_dtype = getattr(torch, growing_dtype_str)
            else:
                growing_dtype = eval(growing_dtype_str)
        else:
            growing_dtype = torch.float32

        compute_optimal_updates_kwargs: dict[str, Any] = OmegaConf.to_container(  # type: ignore
            cfg.growing.compute_optimal_updates_kwargs, resolve=True
        )
        assert isinstance(compute_optimal_updates_kwargs, dict)

        compute_optimal_updates_kwargs.update(
            {
                "statistical_threshold": 0,
                "maximum_added_neurons": maximum_added_neurons,
                "dtype": growing_dtype,
            }
        )

        model.compute_optimal_updates(**compute_optimal_updates_kwargs)
        model.reset_computation()

        logs["updates_information_pre_selection"] = model.update_information()
        logger.info(
            f"Updates information before selection: {logs['updates_information_pre_selection']}"
        )

    # 3) Select the layer to update
    last_updated_layer = selector()
    logs["selected_update"] = last_updated_layer

    if cfg.growing.method == "random":
        if (
            cfg.growing.get("neuron_pairing", None) is not None
            and maximum_added_neurons % 2 != 0
        ):
            raise ValueError(
                f"neuron_pairing requires an even maximum_added_neurons, "
                f"got {maximum_added_neurons}."
            )

        # Random growth
        create_layer_extensions_kwargs = {
            "extension_size": maximum_added_neurons,
            "output_extension_init": cfg.growing.output_extension_init,
            "input_extension_init": cfg.growing.input_extension_init,
            "neuron_pairing": cfg.growing.get("neuron_pairing", None),
            "rescaling": cfg.growing.get("rescaling", None),
            "noise_ratio": cfg.growing.get("noise_ratio", 0.001),
        }
        model.currently_updated_layer.create_layer_extensions(
            **create_layer_extensions_kwargs  # pyright: ignore[reportArgumentType]
        )
        logs["added_neurons"] = maximum_added_neurons

    if growth_schedule_step is not None:
        logs["using_growth_schedule"] = True
        logs["schedule_step"] = step - 1

    model.currently_updated_layer.set_scaling_factor(1.0)

    # 4) Compute the optimal gamma (line search)
    if cfg.growing.method == "fogro":
        model.currently_updated_layer.sub_select_optimal_added_parameters(
            threshold=cfg.growing.compute_optimal_updates_kwargs.statistical_threshold,
            zeros_if_not_enough=cfg.growing.get("zeros_if_not_enough", False),
            zeros_fan_in=cfg.growing.get("zeros_fan_in", True),
            zeros_fan_out=cfg.growing.get("zeros_fan_out", False),
        )

        if model.currently_updated_layer.eigenvalues_extension is not None:
            logs["added_neurons"] = (
                model.currently_updated_layer.eigenvalues_extension.size(0)
            )
        else:
            logs["added_neurons"] = 0

        # Apply variance-transfer rescaling and neuron pairing (after sub-selection)
        fogro_rescaling = cfg.growing.get("rescaling", None)
        fogro_neuron_pairing = cfg.growing.get("neuron_pairing", None)
        if fogro_rescaling is not None:
            model.currently_updated_layer.apply_rescaling(
                rescaling=fogro_rescaling,
                neuron_pairing=fogro_neuron_pairing,
            )
        if fogro_neuron_pairing is not None:
            raise NotImplementedError
            model.currently_updated_layer.apply_neuron_pairing(
                neuron_pairing=fogro_neuron_pairing,
                noise_ratio=cfg.growing.get("noise_ratio", 0.001),
            )

        if cfg.growing.normalization.get("normalize_updates", True):
            model.currently_updated_layer.normalize_optimal_updates(
                normalization_type=cfg.growing.normalization.get(
                    "normalization_type", "weird_normalization"
                )
            )

        if model.currently_updated_layer.eigenvalues_extension is not None:
            logs["top_eigenvalue"] = model.currently_updated_layer.eigenvalues_extension[
                0
            ].item()
        else:
            logs["top_eigenvalue"] = 0.0

        assert (growth_batch_limit is None) or (
            cfg.line_search.batch_limit == -1
            or growth_batch_limit >= cfg.line_search.batch_limit
        ), "Line search batch limit must be less than or equal to statistics batch limit"

        line_search_batch_limit: int = (
            cfg.line_search.batch_limit
            if cfg.line_search.batch_limit != -1
            else (growth_batch_limit if growth_batch_limit is not None else -1)
        )

        # ---- resolve scaling configuration -----------------------------------
        # `fixed_delta_gamma` is overloaded:
        #   false / null   -> delta participates in the line search
        #   true           -> use the optimizer's current learning rate
        #   numeric        -> use that value verbatim (legacy fixed step)
        fixed_delta_gamma_cfg = cfg.growing.get("fixed_delta_gamma", False)
        if fixed_delta_gamma_cfg is True:
            delta_gamma: float | None = float(optimizer.param_groups[0]["lr"])
        elif fixed_delta_gamma_cfg in (False, None):
            delta_gamma = None  # participate in line search
        else:
            delta_gamma = float(fixed_delta_gamma_cfg)

        extension_scaling_mode: str = cfg.growing.get(
            "extension_scaling", "line_search_both"
        )

        # Only the four entries of the user table are supported. Everything
        # else is rejected up front.
        _supported_combinations = {
            (True, "lr_input_only"),  # mode (A)
            (True, "line_search_input_only"),  # mode (B)
            (True, "line_search_both"),  # mode (C)
            (False, "line_search_both"),  # mode (D)
        }
        if (
            delta_gamma is not None,
            extension_scaling_mode,
        ) not in _supported_combinations:
            raise NotImplementedError(
                f"Unsupported scaling combination: "
                f"fixed_delta_gamma={fixed_delta_gamma_cfg!r}, "
                f"extension_scaling={extension_scaling_mode!r}. "
            )

        # ---- target module / availability ------------------------------------
        assert isinstance(model.currently_updated_layer, (GrowingModule, GrowingBlock))
        # `target_module` is the GrowingModule that actually owns the
        # per-component scaling factors (for a GrowingBlock that's its
        # second_layer).
        target_module: GrowingModule = (
            model.currently_updated_layer.second_layer
            if isinstance(model.currently_updated_layer, GrowingBlock)
            else model.currently_updated_layer
        )
        assert isinstance(target_module, GrowingModule)

        has_extension = model.currently_updated_layer.eigenvalues_extension is not None
        has_delta = (
            delta_gamma is not None
            and model.currently_updated_layer.optimal_delta_layer is not None
        )

        # ---- pre-set the components that are held fixed during the search ---
        if delta_gamma is not None:
            target_module.optimal_delta_scaling = delta_gamma if has_delta else 0.0

        output_pinned_to_one = extension_scaling_mode in (
            "lr_input_only",
            "line_search_input_only",
        )
        if output_pinned_to_one and isinstance(
            target_module.previous_module, GrowingModule
        ):
            target_module.previous_module.output_extension_scaling = 1.0

        # ---- helpers shared by the dispatch ---------------------------------
        def _evaluate_for_metrics() -> tuple[float, float]:
            if hasattr(dataloader, "generator"):
                dataloader.generator.manual_seed(growth_dataloder_seed)
            elif growth_batch_limit is not None:
                logger.warning(
                    "Dataloader does not have a generator attribute. "
                    "Cannot set seed for using the same batch for line search."
                )
            return evaluate_model(
                model=model,
                dataloader=dataloader,
                loss_function=loss_function_train,
                metrics=top_1_accuracy,
                batch_limit=line_search_batch_limit,
                device=device,
            )

        def _record_no_search_metrics() -> None:
            final_loss, final_accuracy = _evaluate_for_metrics()
            logs[f"{metric_prefix}_loss"] = final_loss
            logs[f"{metric_prefix}_accuracy"] = final_accuracy
            logs["number_of_line_search_iterations"] = 0

        def _do_line_search(setter, first_order_improvement) -> None:
            (
                _gamma_value,
                final_loss_value,
                final_accuracy_value,
                gamma_history_value,
                loss_history_value,
                aux_loss_history_value,
            ) = line_search(
                model=model,
                dataloader=dataloader,
                dataloader_seed=growth_dataloder_seed,
                loss_function=loss_function_train,
                aux_loss_function=top_1_accuracy,
                batch_limit=line_search_batch_limit,
                initial_loss=initial_loss,
                initial_aux_loss=initial_accuracy,
                first_order_improvement=first_order_improvement,
                alpha=cfg.line_search.alpha,
                beta=cfg.line_search.beta,
                max_iter=cfg.line_search.max_iter,
                epsilon=cfg.line_search.epsilon,
                device=device,
                max_gamma=cfg.line_search.get("max_gamma", None),
                extended_search=cfg.line_search.extended_search,
                verbose=True,
                setter=setter,
            )
            logs["gamma_history"] = gamma_history_value
            logs["loss_history"] = loss_history_value
            logs["aux_loss_history"] = aux_loss_history_value
            logs["number_of_line_search_iterations"] = len(gamma_history_value) - 1
            logs[f"{metric_prefix}_loss"] = final_loss_value
            logs[f"{metric_prefix}_accuracy"] = final_accuracy_value

        def _extension_only_first_order_improvement() -> torch.Tensor:
            # When only the extensions vary, the first-order loss decrease is
            # driven by the extension term alone. See CLAUDE_TODO.md §5.
            return (
                model.currently_updated_layer.activation_gradient
                * (model.currently_updated_layer.eigenvalues_extension**2).sum()
            )

        # ---- variable components --------------------------------------------
        if extension_scaling_mode == "lr_input_only":
            # Mode (A): no line search.
            current_lr = float(optimizer.param_groups[0]["lr"])
            target_module.input_extension_scaling = current_lr if has_extension else 0.0
            _record_no_search_metrics()

        elif extension_scaling_mode == "line_search_input_only":
            if delta_gamma is not None:
                # Mode (B): line-search the input extension only.
                if has_extension:
                    _do_line_search(
                        setter=set_input_extension_only,
                        first_order_improvement=_extension_only_first_order_improvement(),
                    )
                else:
                    target_module.input_extension_scaling = 0.0
                    _record_no_search_metrics()
            else:
                raise NotImplementedError(
                    "line_search_input_only mode requires fixed_delta_gamma to be set"
                )

        elif extension_scaling_mode == "line_search_both":
            if delta_gamma is not None:
                # Mode (C): delta fixed, line-search the extension product.
                if has_extension:
                    _do_line_search(
                        setter=set_extensions_only,
                        first_order_improvement=_extension_only_first_order_improvement(),
                    )
                else:
                    target_module.input_extension_scaling = 0.0
                    if isinstance(target_module.previous_module, GrowingModule):
                        target_module.previous_module.output_extension_scaling = 0.0
                    _record_no_search_metrics()
            else:
                # Mode (D): joint line search (legacy FoGro).
                if (
                    has_extension
                    or model.currently_updated_layer.optimal_delta_layer is not None
                ):
                    _do_line_search(
                        setter=default_setter,
                        first_order_improvement=model.currently_updated_layer.first_order_improvement,
                    )
                else:
                    model.currently_updated_layer.set_scaling_factor(0.0)
                    _record_no_search_metrics()

        # ---- per-component logs (uniform across all four modes) -------------
        logs["delta_gamma"] = float(target_module.optimal_delta_scaling.item())
        logs["extension_gamma_input"] = float(
            target_module.input_extension_scaling.item()
        )
        if isinstance(target_module.previous_module, GrowingModule):
            logs["extension_gamma_output"] = float(
                target_module.previous_module.output_extension_scaling.item()
            )
        else:
            logs["extension_gamma_output"] = 0.0
        # Backwards-compat: keep `logs["gamma"]` mapped to the input-extension
        # scaling so existing MLflow dashboards keep working.
        logs["gamma"] = logs["extension_gamma_input"]

    # 5) Apply the change
    last_updated_layer_ref = model.currently_updated_layer

    if cfg.growing.method == "fogro":
        # Apply iff some component carries a non-trivial scaling. The legacy
        # `scaling_factor` gate doesn't reflect the new sub-factors, so check
        # them directly.
        applied_any = (
            abs(target_module.optimal_delta_scaling.item()) > 1e-5
            or abs(target_module.input_extension_scaling.item()) > 1e-5
        )
    else:
        # Random growth path: `set_scaling_factor(1.0)` above puts all
        # sub-factors at 1, so the legacy gate decides whether to apply.
        applied_any = abs(model.currently_updated_layer.scaling_factor.item()) > 1e-5

    if applied_any:
        assert isinstance(logs["added_neurons"], int)
        model.apply_change(extension_size=logs["added_neurons"])
    else:
        # Still delete the optimal updates to free memory.
        assert isinstance(model.currently_updated_layer, (GrowingModule, GrowingBlock))
        model.currently_updated_layer.delete_update()

    if isinstance(last_updated_layer_ref, RestrictedConv2dGrowingBlock):
        std_in = last_updated_layer_ref.first_layer.weight.data.std(dim=[1, 2, 3])
        std_out = last_updated_layer_ref.second_layer.weight.data.std(dim=[0, 2, 3])
        logger.info("After growth, old weights std:")
        logger.info(std_in[: -logs["added_neurons"]])
        logger.info(std_out[: -logs["added_neurons"]])
        logger.info("After growth, new weights std:")
        logger.info(std_in[-logs["added_neurons"] :])
        logger.info(std_out[-logs["added_neurons"] :])
    elif isinstance(last_updated_layer_ref, LinearGrowingModule) and isinstance(
        last_updated_layer_ref.previous_module, LinearGrowingModule
    ):
        std_in = last_updated_layer_ref.weight.data.std(dim=1)
        std_out = last_updated_layer_ref.previous_module.weight.data.std(dim=0)
        logger.info("After growth, old weights std:")
        logger.info(std_in[: -logs["added_neurons"]])
        logger.info(std_out[: -logs["added_neurons"]])
        logger.info("After growth, new weights std:")
        logger.info(std_in[-logs["added_neurons"] :])
        logger.info(std_out[-logs["added_neurons"] :])

    def mlflow_loggable_name(name: str) -> str:
        return name.replace("(", "-").replace(")", "-")

    # Log layer sizes
    if isinstance(last_updated_layer_ref, LinearGrowingModule):
        logs[f"{mlflow_loggable_name(last_updated_layer_ref.name)}_in_neurons"] = (
            last_updated_layer_ref.in_neurons
        )
        logs[f"{mlflow_loggable_name(last_updated_layer_ref.name)}_out_features"] = (
            last_updated_layer_ref.out_features
        )
    elif isinstance(last_updated_layer_ref, Conv2dGrowingModule):
        logs[f"{mlflow_loggable_name(last_updated_layer_ref.name)}_in_neurons"] = (
            last_updated_layer_ref.in_neurons
        )
        logs[f"{mlflow_loggable_name(last_updated_layer_ref.name)}_out_channels"] = (
            last_updated_layer_ref.out_channels
        )
    elif isinstance(last_updated_layer_ref, GrowingBlock):
        logs[f"{mlflow_loggable_name(last_updated_layer_ref.name)}_hidden_neurons"] = (
            last_updated_layer_ref.hidden_neurons
        )

    last_updated_layer_ref.delete_update()

    if cfg.growing.get("normalize_weights", False) and isinstance(model, GrowingMLP):
        logs["layers_statistics_pre_normalization"] = model.weights_statistics()
        model.normalise()

    return logs, last_updated_layer
