# Optimizer Configuration (`optimizer/`)

Defines the optimizer for standard training steps. Uses Hydra's `instantiate` to create a `torch.optim.Optimizer`.

The optimizer is re-initialized after every growth step to include newly added parameters.

## Parameters

These follow the standard PyTorch optimizer APIs.

### SGD (`sgd.yaml`, `sgd_nesterov.yaml`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `_target_` | str | `torch.optim.SGD` | Optimizer class. |
| `lr` | float | `0.1` | Learning rate. |
| `momentum` | float | `0.0` / `0.9` | Momentum factor. |
| `weight_decay` | float | `0.0` | L2 regularization weight. |
| `nesterov` | bool | `false` / `true` | Use Nesterov momentum. |

### Adam (`adam.yaml`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `_target_` | str | `torch.optim.Adam` | Optimizer class. |
| `lr` | float | `0.001` | Learning rate. |
| `betas` | list[float] | `[0.9, 0.999]` | Coefficients for computing running averages. |
| `weight_decay` | float | `0.0` | L2 regularization weight. |

### AdamW (`adamw.yaml`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `_target_` | str | `torch.optim.AdamW` | Optimizer class. |
| `lr` | float | `0.001` | Learning rate. |
| `betas` | list[float] | `[0.9, 0.999]` | Coefficients for computing running averages. |
| `weight_decay` | float | `0.01` | Decoupled weight decay. |

## Adding a New Optimizer

Create a new YAML file with `_target_` pointing to any `torch.optim.Optimizer` subclass and its keyword arguments. Example for RMSprop:

```yaml
_target_: torch.optim.RMSprop
lr: 0.01
alpha: 0.99
momentum: 0.0
```
