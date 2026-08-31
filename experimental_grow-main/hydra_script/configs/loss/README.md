# Loss Configuration (`loss/`)

Defines the loss function. Two loss instances are created internally:
- **Training loss**: with `reduction="mean"` (used for SGD steps and evaluation).
- **Growth loss**: with `reduction="sum"` (used for computing growth statistics).

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `_target_` | str | **required** | Fully qualified class name of the loss function. |

Any additional parameters are passed as kwargs to the loss constructor (except `reduction`, which is set automatically).

## Available Configs

| Config | Loss | Typical Use |
|--------|------|-------------|
| `cross_entropy_loss.yaml` | `torch.nn.CrossEntropyLoss` | Classification tasks |
| `mse_loss.yaml` | `torch.nn.MSELoss` | Regression tasks |

## Adding a New Loss

Create a YAML file with `_target_` pointing to any `torch.nn.Module` loss. Example:

```yaml
_target_: torch.nn.SmoothL1Loss
beta: 1.0
```

**Note:** The loss must accept a `reduction` keyword argument, as the pipeline overrides it.
