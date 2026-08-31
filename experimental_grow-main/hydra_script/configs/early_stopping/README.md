# Early Stopping Configuration (`early_stopping/`)

Defines when to stop training early based on a monitored metric.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric_name` | str | `"val_accuracy"` | Key in the logs dict to monitor. Common values: `"val_accuracy"`, `"val_loss"`, `"train_loss"`. |
| `reset_on_growth_step` | bool | `true` | Reset the patience counter after each growth step. Prevents early stopping right after network growth (when metrics may temporarily degrade). |

### Early Stopping Object Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `early_stopping_object._target_` | str | `hydra_script.aux_train_and_grow.EarlyStopping` | Class to instantiate. |
| `early_stopping_object.patience` | int | `50` | Number of steps to wait for improvement before stopping. |
| `early_stopping_object.min_delta` | float | `0.0` | Minimum change in the monitored metric to qualify as an improvement. |
| `early_stopping_object.mode` | str | `"max"` | `"max"` = higher is better (accuracy), `"min"` = lower is better (loss). |

## Available Configs

| Config | Metric | Patience | Mode | Description |
|--------|--------|----------|------|-------------|
| `accuracy.yaml` | val_accuracy | 50 | max | Stop when validation accuracy plateaus |
| `loss.yaml` | val_loss | 10 | min | Stop when validation loss plateaus |
| `no_early_stopping.yaml` | val_loss | 1000 | min | Effectively disabled (patience=1000, min_delta=-10) |
| `debug_stopper.yaml` | val_accuracy | 1 | max | Stops almost immediately (for debugging) |
