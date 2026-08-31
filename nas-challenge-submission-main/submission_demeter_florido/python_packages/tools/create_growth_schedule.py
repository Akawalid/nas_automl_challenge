#!/usr/bin/env python3
"""
Script to generate growth schedule files for train_and_grow.py

This script creates a CSV file with the growth schedule format:
step_type; layer_index; maximum_added_neurons; batch_limit

Where step_type is one of:
- "SGD": Regular training step (no growth)
- "growth": Growth step (grow the specified layer)
- "completion": Completion step (finalize growth, optional)

And batch_limit is one of:
- An integer: exact number of batches to use (-1 = full epoch)
- A float in (0, 1]: fraction of the dataset to use
- "hyper-parameter": defer to the config value (e.g. cfg.growing.batch_limit)

In `layer_order`, `-1` is a reserved token: emit one `growth` step with
`layer_index=-1` (all growable layers at once). You can use `[-1]` as the
whole order, or mix indices (e.g. `[0, -1, 2]`).
"""

import argparse
import math
import os
import logging
from collections.abc import Sequence

from typing_extensions import Literal


logger = logging.getLogger(__name__)


# Type alias for the batch_limit column in the schedule.
# - int: exact number of batches (-1 = full epoch)
# - float: fraction of the dataset (0, 1]
# - str: "hyper-parameter" to defer to config
BatchLimit = int | float | str

# ``growth_size`` as passed from Hydra or the CLI before coercion (see `_coerce_growth_size`).
GrowthSizeInput = int | Sequence[int] | None

# Type alias for a single schedule step (4-tuple).
ScheduleStep = tuple[
    Literal["SGD", "growth", "completion"],
    int,
    int,
    BatchLimit,
]


def create_parser(
    parser: argparse.ArgumentParser | argparse._ArgumentGroup,
    prefix: str = "",
    require_num_layers: bool = True,
) -> argparse.ArgumentParser | argparse._ArgumentGroup:
    """
    Complete an argument parser for the schedule generator.

    Parameters
    ----------
    parser : argparse.ArgumentParser | argparse._ArgumentGroup
        The parser or argument group to add arguments to.
    prefix : str
        Prefix to add to argument names (e.g., "schedule-" for integration). Default is "".
    require_num_layers : bool
        Whether --num-layers should be required. Set to False when integrating
        into another script that might infer this value. Default is True.

    Returns
    -------
    argparse.ArgumentParser | argparse._ArgumentGroup
        The parser or group with arguments added
    """
    # Use the provided parser/group for adding arguments
    target = parser

    # Add output-file argument
    target.add_argument(
        f"--{prefix}output-file",
        type=str,
        default=None,
        help="Output file path for the generated schedule. If not provided, performs a dry-run without creating a file.",
    )

    # Add num-layers argument
    target.add_argument(
        f"--{prefix}num-layers",
        type=int,
        required=require_num_layers,
        default=None if not require_num_layers else argparse.SUPPRESS,
        help="Number of layers in the model",
    )

    target.add_argument(
        f"--{prefix}layer-order",
        type=int,
        nargs="+",
        default=None,
        help="Order of layers to grow (e.g., 0 1 2). Use -1 for a step that grows "
        "all layers at once (e.g. '-1' only, or '0 -1 2'). If not specified, uses "
        "range(num_layers).",
    )

    target.add_argument(
        f"--{prefix}epochs-between-growth",
        type=float,
        default=1.0,
        help="Number of training epochs between each growth step. "
        "Accepts floats: integer part = full epochs, fractional part = partial epoch. "
        "E.g. 2.5 = 2 full epochs + 1 half-epoch before each growth step.",
    )

    target.add_argument(
        f"--{prefix}growth-steps-per-layer",
        type=int,
        default=1,
        help="Number of full passes over layer_order",
    )

    target.add_argument(
        f"--{prefix}growth-size",
        type=int,
        nargs="*",
        default=None,
        help="Growth size (neurons to add). Can be: "
        "single value (global), "
        "multiple values (per layer), "
        "or omitted (uses -1 for automatic calculation)",
    )

    target.add_argument(
        f"--{prefix}training-steps-after-growth",
        type=int,
        default=-1,
        help="Number of training steps after the last growth",
    )

    target.add_argument(
        f"--{prefix}growth-warmup-steps",
        type=int,
        default=0,
        help="Number of full SGD steps before the first "
        "growth step in the generated schedule.",
    )

    if not prefix:
        # Only add verbose flag for standalone script
        target.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Print detailed information about the generated schedule",
        )

    return parser


def _coerce_growth_size(growth_size: GrowthSizeInput) -> list[int] | None:
    """Turn `growth_size` from Hydra or the CLI into a form `generate_schedule` uses.

    Returns
    -------
    None
        Automatic neuron count per growth row (`maximum_added_neurons == -1` in the
        schedule). Triggered when `growth_size` is `None`, or an
        empty sequence.
    list[int] of length 1
        A single global cap applied to every growth step.
    list[int] of length `num_layers`
        Per-layer caps (only valid when there is no `-1` token in `layer_order`;
        validated elsewhere).

    Any other shape (e.g. wrong list length) is left to `validate_args` /
    `_resolve_neurons_to_add` to reject.
    """
    if growth_size is None:
        return None
    gs = list(growth_size)
    if len(gs) == 0:
        return None
    return [int(x) for x in gs]


def _resolve_neurons_to_add(
    growth_size: list[int] | None,
    layer_idx: int,
    num_layers: int,
) -> int:
    """Return maximum_added_neurons for one growth step.

    Parameters
    ----------
    growth_size : list[int] | None
        Either a singleton list for one global cap across all layers, a
        per-layer list of length ``num_layers``, or None for automatic counts.
    layer_idx : int
        Layer index for this growth step. -1 means grow all layers at once,
        in which case growth_size is ignored and -1 is returned (automatic).
    num_layers : int
        Number of growable layers in the model. Must match len(growth_size)
        if growth_size is not None.

    Returns
    -------
    int
        Maximum neurons to add for this growth step, or -1 for automatic.

    Warns
    -----
    UserWarning
        If growth_size is provided but layer_idx == -1 (grow all at once),
        since growth_size cannot be applied for that layer.

    Raises
    ------
    ValueError
        If growth_size is provided but its length does not match num_layers.
    """
    if growth_size is None:
        return -1
    if layer_idx == -1:
        logger.warning(
            "growth_size is ignored for steps where layer_index == -1 "
            "(grow all layers at once). Neuron counts will be determined "
            "automatically by compute_optimal_updates."
        )
        return -1
    if len(growth_size) == 1:
        return growth_size[0]
    if len(growth_size) != num_layers:
        raise ValueError(
            f"growth_size must have length 1 or {num_layers}, got {len(growth_size)}"
        )
    return growth_size[layer_idx]


def validate_layer_order(layer_order: Sequence[int], num_layers: int) -> None:
    """Check `layer_order` invariants (shared by CLI and `generate_schedule`)."""
    lo = list(layer_order)
    has_grow_all = -1 in lo
    if has_grow_all:
        for idx in lo:
            if idx == -1:
                continue
            if idx < 0 or idx >= num_layers:
                raise ValueError(
                    f"Layer indices must be in [0, {num_layers - 1}] or -1 "
                    f"(grow all at once), got {idx}"
                )
    else:
        if len(lo) == 0:
            raise ValueError("layer_order must not be empty")
        if min(lo) < 0 or max(lo) >= num_layers:
            raise ValueError(
                f"Layer indices must be in range [0, {num_layers - 1}]"
            )


def validate_args(args: argparse.Namespace) -> None:
    """Validate the arguments."""
    if args.num_layers <= 0:
        raise ValueError("Number of layers must be positive")

    if args.layer_order is not None:
        validate_layer_order(args.layer_order, args.num_layers)

    if args.epochs_between_growth < 0.0:
        raise ValueError("Epochs between growth must be non-negative")

    if args.growth_steps_per_layer <= 0:
        raise ValueError("Growth steps per layer must be positive")

    gs = _coerce_growth_size(args.growth_size)
    if gs is not None:
        if len(gs) == 1:
            if gs[0] <= 0:
                raise ValueError("Growth size must be positive")
        elif len(gs) == args.num_layers:
            if any(size <= 0 for size in gs):
                raise ValueError("All growth sizes must be positive")
        else:
            raise ValueError(
                f"Growth size must be either 1 value (global) or "
                f"{args.num_layers} values (per layer), got {len(gs)}"
            )

    if args.layer_order is not None and -1 in args.layer_order:
        if gs is not None and len(gs) > 1:
            raise ValueError(
                "layer_order contains -1 (grow all layers at once in one step); "
                "use a single global growth_size or omit for automatic counts — "
                "per-layer growth_size lists are not supported when -1 is present."
            )

    if args.training_steps_after_growth == -1:
        args.training_steps_after_growth = math.ceil(args.epochs_between_growth)
    if args.training_steps_after_growth < 0:
        raise ValueError("Training steps after growth must be non-negative")

    warmup = getattr(args, "growth_warmup_steps", 0)
    if warmup < 0:
        raise ValueError("growth_warmup_steps must be non-negative")


def _make_sgd_steps_for_epochs(
    epochs: float,
) -> list[ScheduleStep]:
    """Create SGD steps from a (possibly fractional) epoch count.

    The integer part produces full-epoch SGD steps (batch_limit = -1).
    If there is a fractional remainder, one additional SGD step is emitted
    with batch_limit set to that fraction (a float).

    Parameters
    ----------
    epochs : float
        Number of epochs (>= 0). May be fractional.

    Returns
    -------
    list[ScheduleStep]
        List of SGD schedule steps.
    """
    steps: list[ScheduleStep] = []
    full_epochs = int(epochs)
    fractional = epochs - full_epochs

    for _ in range(full_epochs):
        steps.append(("SGD", -1, 0, -1))

    if fractional > 0.0:
        steps.append(("SGD", -1, 0, fractional))

    return steps


def generate_schedule(
    num_layers: int,
    layer_order: list[int] | None = None,
    epochs_between_growth: float = 1.0,
    growth_steps_per_layer: int = 1,
    growth_size: GrowthSizeInput = None,
    training_steps_after_growth: int = 5,
    include_completion_step: bool = False,
    growth_warmup_steps: int = 0,
) -> list[ScheduleStep]:
    """
    Generate the growth schedule.

    Parameters
    ----------
    num_layers : int
        Number of layers in the model
    layer_order : list[int] | None, optional
        Order of layers to grow. If None, uses range(num_layers).
        Use `-1` as a reserved entry: one `growth` step with
        `layer_index=-1` (all growable layers in one step), e.g. `[-1]` or
        `[0, -1, 2]`.
    epochs_between_growth : float
        Number of training epochs between each growth step.
        Accepts floats: integer part = full epochs (batch_limit=-1),
        fractional part = one partial-epoch SGD step (batch_limit=fraction).
    growth_steps_per_layer : int
        Number of times the full ``layer_order`` sequence is repeated (each
        entry yields one growth step after its preceding SGD block).
    growth_size : GrowthSizeInput, optional
        Maximum neurons to add per growth step, per layer. None for automatic
        counts. If a list, length must match num_layers. Note: for any step
        where layer_index == -1 (grow all layers at once), growth_size is
        ignored and neuron counts are determined automatically by
        compute_optimal_updates.
    training_steps_after_growth : int
        Number of full-epoch training steps after the last growth
    include_completion_step : bool
        Whether to include a completion step after all growth is done
    growth_warmup_steps : int
        Number of full-epoch SGD steps to prepend before any
        growth step. Default 0.

    Returns
    -------
    list[ScheduleStep]
        List of (step_type, layer_index, maximum_added_neurons, batch_limit)
    """
    if layer_order is None:
        layer_order_ints = list(range(num_layers))
    else:
        layer_order_ints = [int(x) for x in layer_order]
        validate_layer_order(layer_order_ints, num_layers)

    growth_size_n = _coerce_growth_size(growth_size)

    schedule: list[ScheduleStep] = []

    if growth_warmup_steps < 0:
        raise ValueError("growth_warmup_steps must be non-negative")
    for _ in range(growth_warmup_steps):
        schedule.append(("SGD", -1, 0, -1))

    # Generate the schedule alternating between layers for each growth step
    for _growth_step in range(growth_steps_per_layer):
        for layer_idx in layer_order_ints:
            # Add training epochs before growth (full + optional fractional)
            schedule.extend(_make_sgd_steps_for_epochs(epochs_between_growth))

            neurons_to_add = _resolve_neurons_to_add(
                growth_size_n, layer_idx, num_layers
            )

            # Add growth step (batch_limit defers to config)
            schedule.append(("growth", layer_idx, neurons_to_add, "hyper-parameter"))

    # Optionally add a completion step
    if include_completion_step:
        schedule.append(("completion", -1, 0, "hyper-parameter"))

    # Add final training steps (always full epochs)
    for _ in range(training_steps_after_growth):
        schedule.append(("SGD", -1, 0, -1))

    return schedule


def write_schedule_file(schedule: Sequence[ScheduleStep], output_file: str) -> None:
    """Write the schedule to a CSV file.

    Parameters
    ----------
    schedule : Sequence[ScheduleStep]
        The schedule to write (4-tuples).
    output_file : str
        Path to the output file.
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        for step_type, layer_idx, max_neurons, batch_limit in schedule:
            f.write(f"{step_type}; {layer_idx}; {max_neurons}; {batch_limit}\n")


def _format_batch_limit(batch_limit: BatchLimit) -> str:
    """Format a batch_limit value for display."""
    if isinstance(batch_limit, str):
        return batch_limit
    if isinstance(batch_limit, float):
        return f"{batch_limit:.2%}"
    if batch_limit == -1:
        return "full"
    return str(batch_limit)


def print_schedule_summary(
    schedule: Sequence[ScheduleStep], args: argparse.Namespace
) -> None:
    """Print a summary of the generated schedule."""
    print(f"\nGenerated schedule with {len(schedule)} steps:")
    print(f"  - Number of layers: {args.num_layers}")
    print(f"  - Layer order: {args.layer_order or list(range(args.num_layers))}")
    print(f"  - Epochs between growth: {args.epochs_between_growth}")
    print(f"  - Growth steps per layer: {args.growth_steps_per_layer}")
    print(f"  - Growth size: {args.growth_size or 'automatic (-1)'}")
    print(f"  - Training steps after growth: {args.training_steps_after_growth}")
    warmup = getattr(args, "growth_warmup_steps", 0)
    if warmup > 0:
        print(f"  - Growth warmup steps (prepended): {warmup}")

    # Count different step types
    growth_steps = sum(1 for step_type, _, _, _ in schedule if step_type == "growth")
    completion_steps = sum(
        1 for step_type, _, _, _ in schedule if step_type == "completion"
    )
    training_steps = len(schedule) - growth_steps - completion_steps
    partial_training_steps = sum(
        1
        for step_type, _, _, bl in schedule
        if step_type == "SGD" and isinstance(bl, float)
    )

    print("\nSchedule statistics:")
    print(f"  - Total steps: {len(schedule)}")
    print(f"  - Growth steps: {growth_steps}")
    print(f"  - Training steps: {training_steps} ({partial_training_steps} partial)")
    if completion_steps:
        print(f"  - Completion steps: {completion_steps}")

    if args.verbose:
        print("\nDetailed schedule:")
        for i, (step_type, layer_idx, max_neurons, batch_limit) in enumerate(schedule):
            step_type_str = "GROWTH" if step_type == "growth" else step_type.upper()
            neurons_str = str(max_neurons) if max_neurons >= 0 else "auto"
            bl_str = _format_batch_limit(batch_limit)
            print(
                f"  Step {i + 1:3d}: {step_type_str:10s} | "
                f"Layer {layer_idx:2d} | "
                f"Neurons: {neurons_str:>5s} | "
                f"Batch limit: {bl_str}"
            )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate growth schedule files for neural network training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = create_parser(parser)
    assert isinstance(parser, argparse.ArgumentParser)
    args = parser.parse_args()

    try:
        validate_args(args)

        # Generate the schedule
        schedule = generate_schedule(
            num_layers=args.num_layers,
            layer_order=args.layer_order,
            epochs_between_growth=args.epochs_between_growth,
            growth_steps_per_layer=args.growth_steps_per_layer,
            growth_size=args.growth_size,
            training_steps_after_growth=args.training_steps_after_growth,
            growth_warmup_steps=args.growth_warmup_steps,
        )

        # Write to file if output file is specified, otherwise do dry-run
        if args.output_file:
            write_schedule_file(schedule, args.output_file)
            print(f"Schedule written to: {args.output_file}")
        else:
            print("Dry-run mode: No file will be created")

        # Print summary
        print_schedule_summary(schedule, args)

    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
