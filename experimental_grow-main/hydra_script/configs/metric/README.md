# Metric Configuration (`metric/`)

Defines the evaluation metric used for tracking accuracy during training, validation, and test.

Three metric instances are created (one per phase: train, val, test). The metric constructor is called with `num_classes` and/or `device` if its signature accepts them.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `_target_` | str | **required** | Fully qualified class name of the metric. |

Additional parameters are passed as kwargs to the constructor.

## Available Configs

| Config | Metric | Description |
|--------|--------|-------------|
| `multiclass_accuracy.yaml` | `torchmetrics.classification.MulticlassAccuracy` | Standard accuracy with `average=micro`. |
| `eeg_custom.yaml` | `tools.metrics.EGGCustom` | Custom metric for EEG experiments. |
| `none.yaml` | `tools.utils.none_constructor` | No metric (returns a no-op object). |

### `multiclass_accuracy.yaml` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `average` | str | `micro` | Averaging method (`micro`, `macro`, `weighted`). |

## Extra Logging

If the metric object has an `extra_logs()` method, it is called after each evaluation to log additional per-class or custom metrics to MLflow.
