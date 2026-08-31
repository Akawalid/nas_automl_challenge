# Growth Schedule Feature

This document describes the growth schedule functionality added to the `train_and_grow.py` script.

## Overview

The growth schedule feature allows you to precisely control when and how the model grows during training by specifying a schedule file. This overrides the default growth behavior and provides fine-grained control over:

- Which epochs are growth epochs vs training epochs
- Which layer to grow in each growth epoch  
- How many neurons to add in each growth epoch

## Usage

### Training with a Schedule File

Add the `--growth-schedule-file` argument when running the training script:

```bash
python experiments/train_and_grow.py \
    --growth-schedule-file configs/schedule_configs/my_schedule.csv \
    --dataset mnist \
    --config-path models/configs/mlp.yml
```

### Generating Schedule Files

Use the provided script to generate schedule files:

```bash
python scripts/create_growth_schedule.py \
    --output-file configs/schedule_configs/my_schedule.csv \
    --num-layers 3 \
    --growth-size 5 10 15 \
    --epochs-between-growth 2 \
    --growth-steps-per-layer 1 \
    --training-steps-after-growth 5 \
    --verbose
```

## Schedule File Format

The schedule file should be a CSV file with semicolon separators. Each line represents one training step/epoch:

```
step_type; layer_index; maximum_added_neurons; batch_limit
```

Where:
- `step_type`: One of:
  - `SGD`: Regular training step (no growth)
  - `growth`: Growth step (grow one layer, or all layers at once; see below)
  - `completion`: Optional completion step to finalize growth
- `layer_index`: Meaning depends on `step_type`:
  - For `SGD` and `completion`: conventionally `-1` (ignored).
  - For `growth`: index of the growable layer (`0` … `num_layers - 1`) when growing **one** layer at a time; **`-1`** means **grow all growable layers in a single growth step** (see [Grow all layers at once](#grow-all-layers-at-once-grow_all_at_once)).
- `maximum_added_neurons`: Maximum number of neurons to add in this growth step, or `-1` to use the automatic calculation based on model and growth step (when growing all layers at once, the same cap applies per layer in that step).
- `batch_limit`: Batch limit for that step (integer, fraction, `"hyper-parameter"`, or `-1` for full epoch); see the Hydra `schedule` README for details.

### Example Schedule File

```csv
SGD; -1; 0; -1
growth; 0; 5; hyper-parameter
SGD; -1; 0; -1
growth; 1; 8; hyper-parameter
SGD; -1; 0; -1
growth; 2; 12; hyper-parameter
```

This schedule means:
- Step 1: Training epoch (no growth)
- Step 2: Growth epoch - grow layer 0 with up to 5 neurons
- Step 3: Training epoch (no growth)
- Step 4: Growth epoch - grow layer 1 with up to 8 neurons  
- Step 5: Training epoch (no growth)
- Step 6: Growth epoch - grow layer 2 with up to 12 neurons

## Grow all layers at once (`grow_all_at_once`)

When a row has `step_type = growth` and `layer_index = -1`, the training code grows **every** growable layer in **one** growth step (in execution order). For `SequentialGrowingModel`, this uses `set_growing_layers(scheduling_method="all")` instead of selecting a single index.

**Important:** `-1` as `layer_index` on `SGD` / `completion` rows only means “not applicable”; the special “all layers” meaning applies **only** to `growth` rows.

**Logging:** The run logs `grow_all_at_once: true`, per-layer metrics under keys like `layer_0_gamma`, `layer_0_added_neurons`, etc., a summed `added_neurons` across layers, and after the step a **global** `{metric_prefix}_loss` / `{metric_prefix}_accuracy` from a final `evaluate_model` on the grown network. The returned “last updated layer” index is `-1` to indicate that all layers were updated.

**Schedule generation:** In Hydra, set ``schedule.layer_order`` to ``[-1]`` or include ``-1`` in the list; ``generate_schedule()`` emits the corresponding ``growth`` rows with ``layer_index=-1``. If ``-1`` appears anywhere in ``layer_order``, ``growth_size`` must be omitted (automatic) or a **single** global value — not a per-layer list.

## Behavior

When a growth schedule file is provided:

1. **Overrides default settings**: The schedule overrides `--epochs-per-growth`, `--selection-method`, and `--growing-maximum-added-neurons` (except when `maximum_added_neurons` is `-1`, which uses the default logic)
2. **Sequential container support**: For `SequentialGrowingModel`, a normal growth row calls `set_growing_layers(scheduling_method="sequential", index=layer_index)`; a `growth` row with `layer_index == -1` uses `scheduling_method="all"` for that step.
3. **Precise control**: Each growth epoch uses the layer index (or “all layers”) and neuron count specified in the schedule
4. **Automatic termination**: When the schedule is exhausted, no more growth occurs (training continues normally)
5. **Logging**: Additional logging shows when the schedule is being used

## Requirements

- The model should support the `set_growing_layers` API on `SequentialGrowingModel` when using schedules with layer indices
- Layer indices should be valid for the model architecture (or `-1` on `growth` rows for grow-all-at-once)
- The schedule file should have the correct format (4 semicolon-separated values per line when using the Hydra / `write_schedule_file` format)

## Validation

The implementation includes validation for:
- Proper file format (4 semicolon-separated values per line in the Hydra pipeline)
- Valid step types (`SGD`, `growth`, `completion`)
- Valid layer indices for growth steps (including `-1` on `growth` rows for grow-all-at-once)
- Non-negative neuron counts (where applicable)

## Example Use Cases

1. **Research experiments**: Test specific growth patterns
2. **Curriculum learning**: Gradually increase model complexity
3. **Ablation studies**: Compare different growth schedules
4. **Fine-tuned control**: Optimize growth timing based on validation metrics

## Compatibility

This feature is compatible with:
- All existing growth methods (`fogro`, `random`)
- Both `GrowingContainer` and `SequentialGrowingContainer` models
- All optimizers and schedulers
- Existing logging and MLflow integration

## Schedule Generation Script

The `scripts/create_growth_schedule.py` script provides an easy way to generate schedule files with various patterns:

### Options

- `--num-layers, -n`: Number of layers in the model (required)
- `--layer-order`: Order of layers to grow (e.g. `0 1 2`). Use **`-1`** for a step that grows all layers at once (e.g. `--layer-order -1` or `--layer-order 0 -1 2`). If `-1` appears, do not pass per-layer `--growth-size` lists (use one global value or omit for automatic). If not specified, uses `range(num_layers)`
- `--epochs-between-growth`: Number of training epochs between each growth step (default: 1)
- `--growth-steps-per-layer`: Number of growth steps per layer (default: 1)  
- `--growth-size`: Growth size (neurons to add). Can be:
  - Single value (global): `--growth-size 10`
  - Multiple values (per layer): `--growth-size 5 8 12`
  - Omitted (automatic): uses -1 for automatic calculation
- `--training-steps-after-growth`: Number of training steps after the last growth (default: 5)
- `--verbose, -v`: Print detailed information about the generated schedule

### Examples

**Basic schedule (3 layers, automatic growth size):**
```bash
python scripts/create_growth_schedule.py -o schedule.csv -n 3
```

**Custom layer order and per-layer growth sizes:**
```bash
python scripts/create_growth_schedule.py \
    -o schedule.csv \
    -n 3 \
    --layer-order 2 0 1 \
    --growth-size 5 8 12 \
    --epochs-between-growth 2 \
    --growth-steps-per-layer 2 \
    --training-steps-after-growth 3
```

**Grow all layers each growth round (global size 10):**
```bash
python scripts/create_growth_schedule.py \
    -o all_at_once.csv \
    -n 4 \
    --layer-order -1 \
    --growth-size 10 \
    --epochs-between-growth 1
```

**Consecutive growth (no training between growth steps):**
```bash
python scripts/create_growth_schedule.py \
    -o schedule.csv \
    -n 4 \
    --growth-size 10 \
    --epochs-between-growth 0
```