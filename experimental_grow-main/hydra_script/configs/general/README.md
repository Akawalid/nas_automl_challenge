# General Configuration (`general/`)

General training settings shared across all experiments.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | str | `cuda` | Device to use for training (`cuda` or `cpu`). Falls back to `cpu` if CUDA is unavailable. |
| `seed` | int \| null | `0` | Random seed for reproducibility. When set, also enables `cudnn.deterministic=True` and `cudnn.benchmark=False`. `null` disables seeding. |
| `batch_size` | int | `128` | Batch size for all dataloaders (train, val, test, growth). |
| `num_workers` | int | `4` | Number of dataloader worker processes. |
| `generate_schedule` | bool | `true` | Whether to auto-generate a growth schedule from the `schedule` config. When `false`, growth timing is controlled by `growing.epochs_per_growth`. |
| `amp` | bool | `false` | Mixed-precision (AMP) training via `torch.autocast`/`GradScaler`. Only effective on CUDA. Matches timm's `amp` flag. |
| `filter_bias_and_norm` | bool | `false` | If `true`, exclude bias and 1-D (LayerNorm/norm) parameters from weight decay by building two optimizer parameter groups. Reproduces timm's default `filter_bias_and_bn`. Toggle per run, e.g. `general.filter_bias_and_norm=true`. |
| `torch_backends_cuda_preferred_linalg_library` | str | `Default` | Preferred CUDA linear algebra library. Options: `Default`, `Magma`, `Cusolver`. Only relevant when `device=cuda`. |

## Available Configs

- `default.yaml` — Standard settings (cuda, seed=0, batch_size=128)
