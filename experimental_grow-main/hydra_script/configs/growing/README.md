# Growing Configuration (`growing/`)

Defines how the network grows: the growth method, statistics computation, and extension initialization.

## Main Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `fogro` | Growth method. `fogro` = First-Order Growth Optimization (data-driven), `random` = random extensions, `none` = no growth. |
| `skip_compute_statistics` | bool | `false` | Skip computing growth statistics (data pass). Set to `true` for random growth where statistics are not needed. |
| `separate_growth_and_train_set` | bool | `false` | Use a separate data split for growth statistics (instead of training data). |
| `batch_limit` | int | `40` | Maximum number of batches used for computing growth statistics. `-1` = use all batches. |
| `planned_growing_steps` | int | `10` | Expected number of growth steps per layer. May be used to add $(c_{final} - c_{initial}) / \texttt{planned growing steps}$ neurons per addition. |
| `normalize_weights` | bool | `false` | Whether to normalize model weights after growth. Is only available for MLP. |
| `rescaling` | str \| null | `null` | Variance-transfer rescaling applied after sub-selection. Options: `null`, `default_vt`, `vt_constraint_old_shape`, `vt_constraint_new_shape`. |
| `neuron_pairing` | str \| null | `null` | Neuron pairing strategy. `null` = no pairing. (`vv_z_negz` is reserved but not implemented.) |
| `noise_ratio` | float | `0.001` | Symmetry-breaking noise injected by neuron pairing / random init. |

### Extension Initialization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_extension_init` | str | `copy_uniform` | Initialization for output-side (fan-out) extensions. |
| `input_extension_init` | str | `zeros` | Initialization for input-side (fan-in) extensions. |

Options: `zeros`, `random`, `copy_uniform`, `kaiming_uniform`.

### Scaling Configuration

Each growth step applies three independent scalings — one for the optimal
delta and two for the input/output extensions. They live on the
`GrowingModule` as `optimal_delta_scaling`, `input_extension_scaling`, and
`previous_module.output_extension_scaling`. Two top-level options together
decide how each is computed for `method: fogro`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fixed_delta_gamma` | bool \| float \| null | `false` | Step size for the optimal delta (`gamma_delta`). Overloaded: `false` / `null` → delta participates in the line search; `true` → use the optimizer's current learning rate; numeric → use that value verbatim (legacy fixed step). When `compute_delta=False` (no `optimal_delta_layer`), `gamma_delta` collapses to `0` regardless. |
| `extension_scaling` | str | `line_search_both` | How `gamma_extension_input` / `gamma_extension_output` are chosen. One of `lr_input_only`, `lr_both`, `line_search_input_only`, `line_search_both` (see below). |

`extension_scaling` values:

| Value | `gamma_extension_input` | `gamma_extension_output` |
|-------|-------------------------|--------------------------|
| `lr_input_only` | current LR | `1` |
| `lr_both` | current LR | current LR |
| `line_search_input_only` | line search | `1` (held fixed) |
| `line_search_both` | line search | line search (= input) |

Supported `(fixed_delta_gamma, extension_scaling)` combinations (any other
combination raises `NotImplementedError`):

| Mode | `fixed_delta_gamma` | `extension_scaling` | Behaviour |
|------|---------------------|---------------------|-----------|
| (A) | `true` | `lr_input_only` | No line search. `gamma_delta = lr`, `input = lr`, `output = 1`. Matches `scaled_fogro.yaml`. |
| (B) | `true` | `line_search_input_only` | `gamma_delta = lr` (fixed), output pinned to `1`, input found by line search. |
| (C) | `true` | `line_search_both` | `gamma_delta = lr` (fixed), both extensions found by a single line search (`input = output`). |
| (D) | `false` | `line_search_both` | Standard FoGro: a single line search drives the delta and both extensions jointly. |

When the line search drives only the extensions (modes B, C), the
caller-supplied `first_order_improvement` is the extension-only quantity
`activation_gradient * (eigenvalues_extension ** 2).sum()`; the joint case
(D) uses `currently_updated_layer.first_order_improvement`. Modes (A)/(B)
deliberately leave `gamma_extension_output = 1`: function preservation at
`lr → 0` then relies on the input scaling going to zero.

Per-step MLflow logs: `delta_gamma`, `extension_gamma_input`,
`extension_gamma_output`. The legacy key `gamma` is aliased to
`extension_gamma_input` for dashboard continuity.

### Legacy Parameters (in `fogro_zeros.yaml`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `zeros_if_not_enough` | bool | `false` | if True, will keep all the new neurons and set the non selected ones to zero (either first or last depending on zeros_fan_in and zeros_fan_out) |
| `zeros_fan_in` | bool | `true` | if True and zeros_if_not_enough is True, will set the non selected fan-in parameters to zero |
| `zeros_fan_out` | bool | `false` | if True and zeros_if_not_enough is True, will set the non selected fan-out parameters to zero |

## Sub-Configurations

### Normalization (`normalization/`)

Controls how growth updates are normalized before being applied.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalize_updates` | bool | `false` | Enable normalization of growth updates. |
| `normalization_type` | str \| null | `null` | Normalization strategy. Options: `null`, `equalize_extensions`, `equalize_second_layer`, `legacy_normalization`, `match_extending_layer`. |

| Config | Description |
|--------|-------------|
| `none.yaml` | No normalization (default) |
| `equalize_extensions.yaml` | Equalize extension magnitudes |
| `equalize_second.yaml` | Equalize second layer extensions |
| `legacy.yaml` | Legacy normalization (original implementation) |
| `weird.yaml` | Same as `legacy.yaml` — deprecated |
| `match_extending_layer.yaml` | Scale each update component to the std of the layer it extends (`extended_input_layer` → `std(self.layer.weight)`, `previous_module.extended_output_layer` → `std(previous_module.layer.weight)`, `optimal_delta_layer` → `std(self.layer.weight)`). Required when `extension_scaling` uses fixed-LR scalings (modes A/B), since each component then enters at a magnitude bounded by the target layer's own scale. Raises `ValueError` if there is no previous module. |

### Compute Optimal Updates Kwargs (`compute_optimal_updates_kwargs/`)

Algorithm-specific parameters passed to `model.compute_optimal_updates()`.

| Parameter | Type | Default (fogro) | Description |
|-----------|------|-----------------|-------------|
| `numerical_threshold` | float | `1e-6` | Threshold for numerical stability (e.g., singular value cutoff). |
| `statistical_threshold` | float | `1e-3` | Threshold for statistical significance when selecting neurons. Higher values → fewer neurons added. `0` = no statistical filtering. |
| `maximum_added_neurons` | int \| null | `null` | Hard cap on neurons added per growth step. `null` = no limit. |
| `dtype` | str | `torch.float32` | Data type for growth computations. |
| `compute_delta` | bool | `true` | Whether to compute the delta (weight correction) term. |
| `use_covariance` | bool | `true` | Whether to use covariance matrices in the growth computation. |
| `alpha_zero` | bool | `false` | When `true`, sets alpha (input-side extension) to zero. |
| `omega_zero` | bool | `false` | When `true`, sets omega (output-side extension) to zero.  |
| `use_projection` | bool | `true` | Whether to use projection in the growth computation. |
| `ignore_singular_values` | bool | `false` | Whether to ignore singular values during neuron computation i.e. use only singular vectors without the singular values. |

#### Algorithm Presets

| Config | Method | Key Differences |
|--------|--------|-----------------|
| `fogro.yaml` | Full FoGro | compute_delta=True, use_covariance=True, use_projection=True |
| `scaled_fogro.yaml` | FoGro variant | Same as `fogro.yaml` but with `ignore_singular_values=True`. Used by the top-level `scaled_fogro.yaml` preset. |
| `gradmax.yaml` | GradMax | GradMax presets. |
| `tiny.yaml` | Tiny variant | Close to TINY presets. |

## Available Top-Level Configs

| Config | Method | Description |
|--------|--------|-------------|
| `fogro.yaml` | fogro | Standard FoGro with no normalization |
| `fogro_zeros.yaml` | fogro | FoGro with legacy zero-init params |
| `scaled_fogro.yaml` | fogro | FoGro variant: `ignore_singular_values=True`, `rescaling=vt_constraint_new_shape`, `normalization=match_extending_layer`, `fixed_delta_gamma=true`, `extension_scaling=lr_input_only`. Mode (A) of the scaling table — no line search; the optimal delta and the input extension are scaled by the current learning rate, the output extension stays at `1`. |
| `random_zeros.yaml` | random | Random growth, output=copy_uniform, input=zeros |
| `zeros_random.yaml` | random | Random growth, output=zeros, input=copy_uniform |
| `none.yaml` | none | No growth (pure training baseline) |
