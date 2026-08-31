# Scripts Directory

This directory contains utility scripts for the experimental grow project.

## create_growth_schedule.py

A script to generate growth schedule files for use with the `train_and_grow.py` script.

### Quick Start

Generate a basic schedule for a 3-layer model:
```bash
python scripts/create_growth_schedule.py -o my_schedule.csv -n 3 --verbose
```

### Full Usage

```bash
python scripts/create_growth_schedule.py \
    --output-file schedule.csv \
    --num-layers 4 \
    --layer-order 1 0 3 2 \
    --growth-size 5 8 6 10 \
    --epochs-between-growth 2 \
    --growth-steps-per-layer 2 \
    --training-steps-after-growth 5 \
    --verbose
```

### Parameters

- `--output-file, -o`: Output file path (required)
- `--num-layers, -n`: Number of layers in the model (required)
- `--layer-order`: Custom order to grow layers (default: 0, 1, 2, ...)
- `--epochs-between-growth`: Training epochs between growth steps (default: 1)
- `--growth-steps-per-layer`: How many times to grow each layer (default: 1)
- `--growth-size`: Neurons to add - can be global value, per-layer values, or omitted for automatic (default: automatic)
- `--training-steps-after-growth`: Training steps after last growth (default: 5)
- `--verbose, -v`: Show detailed schedule information

### Examples

**Automatic growth size (uses model's internal calculation):**
```bash
python scripts/create_growth_schedule.py -o auto.csv -n 3
```

**Global growth size (same for all layers):**
```bash
python scripts/create_growth_schedule.py -o global.csv -n 3 --growth-size 10
```

**Per-layer growth sizes:**
```bash
python scripts/create_growth_schedule.py -o per_layer.csv -n 3 --growth-size 5 8 12
```

**Custom layer order:**
```bash
python scripts/create_growth_schedule.py -o custom_order.csv -n 4 --layer-order 3 1 0 2
```

**Multiple growth steps per layer:**
```bash
python scripts/create_growth_schedule.py -o multi_growth.csv -n 2 --growth-steps-per-layer 3
```

**No training between growth (consecutive growth):**
```bash
python scripts/create_growth_schedule.py -o consecutive.csv -n 3 --epochs-between-growth 0
```

### Output Format

The generated CSV file uses semicolon separators with the format:
```
is_growth_step; layer_index; maximum_added_neurons
```

Where:
- `is_growth_step`: 1 for growth epochs, 0 for training epochs
- `layer_index`: Index of the layer to grow
- `maximum_added_neurons`: Number of neurons to add (-1 for automatic calculation)

### Using Generated Schedules

Use the generated schedule with the training script:
```bash
python experiments/train_and_grow.py \
    --growth-schedule-file schedule.csv \
    --dataset mnist \
    --config-path models/configs/mlp.yml
```

See `docs/GROWTH_SCHEDULE.md` for more information about the growth schedule feature.