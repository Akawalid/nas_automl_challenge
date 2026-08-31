# Line Search Configuration (`line_search/`)

Controls the line search procedure used during growth steps to find the optimal scaling factor (gamma) for the newly added neurons.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | `0.1` | Line search parameter —the sufficient decrease parameter (Armijo condition). |
| `beta` | float | `0.5` | Line search parameter — backtracking factor (gamma is multiplied by beta on each failed step). |
| `max_iter` | int | `20` | Maximum number of line search iterations. |
| `epsilon` | float | `1e-3` | Minimal value of gamma. |
| `batch_limit` | int | `-1` | Maximum batches used for evaluating line search candidates. `-1` = use all batches. |
| `extended_search` | bool | `false` | Whether to perform an extended search (possibly tries larger gamma values beyond initial estimate). |
| `max_gamma` | float | `None` | Maximum value of gamma. |


## Available Configs

- `default.yaml` — Standard line search (alpha=0.1, beta=0.5, max_iter=20)

## Setter API

`hydra_script.line_search.line_search` accepts a `setter:
Callable[[GrowingContainer, float], None]` argument that decides what
`gamma = t**2` means in terms of the underlying scaling factors. The
default (`default_setter`) calls
`model.currently_updated_layer.set_scaling_factor(t)` — the legacy joint
fan-out that drives `optimal_delta_scaling = t**2`,
`input_extension_scaling = t`,
`previous_module.output_extension_scaling = t`. Two pre-defined
alternatives, used by `perform_growth_step` based on
`growing.extension_scaling`:

- `set_extensions_only(model, t)` — drives both extension scalings
  simultaneously (`input = output = t`); the caller is expected to pin
  `optimal_delta_scaling` beforehand. Effective extension product equals
  `gamma`.
- `set_input_extension_only(model, t)` — drives `input_extension_scaling
  = t**2` only; the caller is expected to pin
  `previous_module.output_extension_scaling = 1` and
  `optimal_delta_scaling`. Effective extension product still equals
  `gamma`.

Custom setters must keep the loss locally linear in `gamma = t**2`,
otherwise the Armijo bound and the supplied `first_order_improvement` no
longer match. `first_order_improvement` is caller-provided and depends on
which scalings the setter drives:

- joint search (`default_setter`):
  `currently_updated_layer.first_order_improvement`.
- extensions-only (`set_extensions_only`,
  `set_input_extension_only`):
  `activation_gradient * (eigenvalues_extension ** 2).sum()`.
