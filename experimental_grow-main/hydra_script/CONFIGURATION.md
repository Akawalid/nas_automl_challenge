# Configuration Reference for `train_and_grow.py`

This document describes every configuration group and hyper-parameter used by the training and growing pipeline. The pipeline uses [Hydra](https://hydra.cc/) for configuration management and [MLflow](https://mlflow.org/) for experiment tracking.

## Table of Contents

- [Overview](#overview)
- [Top-Level Configuration](#top-level-configuration)
- [Configuration Groups](#configuration-groups)
- [How to Add a New Model](#how-to-add-a-new-model)
- [How to Add a New Dataset](#how-to-add-a-new-dataset)

---

## Overview

The entry point is `config.yaml`, which composes sub-configurations via Hydra defaults:

```
configs/
├── config.yaml                  # Main entry point
├── dataset_config/              # Dataset definitions
├── early_stopping/              # Early stopping strategies
├── general/                     # General settings (device, seed, batch size)
├── growing/                     # Network growth method and sub-configs
│   ├── normalization/           #   Growth normalization strategies
│   └── compute_optimal_updates_kwargs/  # Algorithm-specific growth params
├── line_search/                 # Line search for growth scaling
├── loss/                        # Loss functions
├── lr_scheduler/                # Learning rate schedulers
├── metric/                      # Evaluation metrics
├── mlflow/                      # MLflow tracking settings
├── model/                       # Model architectures
├── optimizer/                   # Optimizers
├── schedule/                    # Growth schedule (what grows when)
├── baseline/                    # Preset baseline configurations
└── hydra/                       # Hydra launcher / output paths
```

Each sub-folder has its own `README.md` with detailed parameter descriptions.

### Running an experiment

```bash
# Single run with defaults from config.yaml
uv run python -m hydra_script.train_and_grow

# Override sub-configs
uv run python -m hydra_script.train_and_grow model=resnet18 dataset_config=cifar10

# Override individual parameters
uv run python -m hydra_script.train_and_grow optimizer.lr=0.01 general.seed=42

# Use a composite config
uv run python -m hydra_script.train_and_grow --config-name=debug

# Multi-run sweep on SLURM
uv run python -m hydra_script.train_and_grow --config-name=sweep_baseline --multirun
```

---

## Top-Level Configuration

These parameters live directly in the main config files (e.g., `config.yaml`, `debug.yaml`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `training.nb_step` | int | (from schedule) | Total number of training + growth steps. If a growth schedule is generated, the total is `max(nb_step, len(schedule))`. |
| `training.training_threshold` | float \| null | `null` | Stop training when `train_loss` drops below this value. `null` disables this criterion. |

All other parameters are organized in the configuration groups described below.

---

## Configuration Groups

| Group | Sub-folder | Description |
|-------|-----------|-------------|
| **general** | [`general/`](configs/general/README.md) | Device, seed, batch size, workers |
| **dataset_config** | [`dataset_config/`](configs/dataset_config/README.md) | Dataset loading, transforms, splits |
| **model** | [`model/`](configs/model/README.md) | Model architecture |
| **optimizer** | [`optimizer/`](configs/optimizer/README.md) | Optimizer (SGD, Adam, etc.) |
| **lr_scheduler** | [`lr_scheduler/`](configs/lr_scheduler/README.md) | Learning rate schedule |
| **loss** | [`loss/`](configs/loss/README.md) | Loss function |
| **metric** | [`metric/`](configs/metric/README.md) | Evaluation metric |
| **early_stopping** | [`early_stopping/`](configs/early_stopping/README.md) | Early stopping strategy |
| **growing** | [`growing/`](configs/growing/README.md) | Network growth method and parameters. Includes the `fixed_delta_gamma` × `extension_scaling` table that selects between fixed-LR and line-searched scalings on the optimal delta and the input/output extensions; see `scaled_fogro.yaml` for a non-line-search preset. |
| **schedule** | [`schedule/`](configs/schedule/README.md) | Growth schedule (layers, timing, size) |
| **line_search** | [`line_search/`](configs/line_search/README.md) | Line search for growth step scaling. The `line_search` function accepts a `setter` callable so different `extension_scaling` modes can drive different sub-sets of scaling factors with one search. |
| **mlflow** | [`mlflow/`](configs/mlflow/README.md) | MLflow tracking configuration |
| **hydra** | [`hydra/`](configs/hydra/README.md) | Output paths and SLURM launchers |

---

## How to Add a New Model

### Step 1: Create a config file

Create `configs/model/<your_model>.yaml`:

```yaml
model: resnet          # Model family identifier (see supported values below)
activation: relu       # Activation function
number_of_blocks_per_stage: 2  # Architecture-specific parameter
```

The `model` field selects the model builder. Currently supported values:

| `model` value | Builder | Description |
|---------------|---------|-------------|
| `perceptron` | `Perceptron` | Single-layer perceptron |
| `mlp` | `GrowingMLP` | Multi-layer perceptron with growing support |
| `residual_mlp` | `GrowingResidualMLP` | Residual MLP with growing support |
| `mlp_mixer` | `GrowingMLPMixer` | MLP-Mixer with growing support |
| `growing_transformer` | `GrowingTransformer` | Legacy ViT-style growing transformer adapter |
| `growing_cct` | `GrowingCCT` | Compact Convolutional Transformer with growable feed-forward blocks |
| `growing_cvt` | `GrowingCVT` | Compact Vision Transformer with growable feed-forward blocks |
| `growing_vit_lite` | `GrowingViTLite` | ViT-Lite with growable feed-forward blocks |
| `resnet` | `init_full_resnet_structure` | ResNet with growing support (main architecture) |
| `true_resnet` | `get_resnet` | Standard torchvision-style ResNet |
| `eeg_model` | `EEGModel` | Specialized CNN for EEG data |

### Step 2: Architecture-specific parameters

Each model family accepts different parameters. Here are the known ones for the `resnet` family:

| Parameter | Type | Description |
|-----------|------|-------------|
| `activation` | str | Activation function (`relu`, `leaky_relu`) |
| `number_of_blocks_per_stage` | int | Blocks per residual stage (1 = ResNet10, 2 = ResNet18) |
| `nb_stages` | int | Number of residual stages |
| `inplanes` | int | Number of channels in the first conv layer |
| `reduction_factor` | float | Channel width reduction factor (1 = full width) |
| `normalization` | str | Normalization type (e.g., `group`) |

For the `eeg_model` family:

| Parameter | Type | Description |
|-----------|------|-------------|
| `initial_last_layer_size` | int | Size of the last fully-connected layer |
| `activation` | str | Activation function |

For the CCT/CVT/ViT-Lite transformer families:

| Parameter | Type | Description |
|-----------|------|-------------|
| `embedding_dim` | int | Token embedding width |
| `num_layers` | int | Transformer depth and number of growth positions |
| `num_heads` | int | Attention heads |
| `mlp_ratio` | float | Feed-forward width multiplier; this branch grows |
| `kernel_size` | int | Tokenizer convolution or patch kernel size |
| `n_conv_layers` | int | CCT tokenizer convolution depth |
| `positional_embedding` | str | `learnable`, `sine`, or `none` |

### Step 3: Use the new model

```bash
uv run python -m hydra_script.train_and_grow model=<your_model>
```

### Adding a completely new model family

To add a new model family (not just a new config for an existing one):

1. Implement your model so it is compatible with the `gromo` growing interface (i.e., it should support `compute_statistics()`, `compute_optimal_updates()`, and growth extension methods if you want growth support).
2. Register it in `tools/models.py` by adding a new entry to the model dispatch logic in `get_model_from_config()`.
3. Create a config file as described above with the appropriate `model` identifier.

---

## How to Add a New Dataset

### Step 1: Create a config file

Create `configs/dataset_config/<your_dataset>.yaml`. Use the base config for shared defaults:

```yaml
defaults:
  - base       # Inherits path and split_train_val

name: my_dataset
num_classes: 10

dataset:
  _target_: torchvision.datasets.MyDataset   # Or your custom class
  root: ${dataset_config.path}
  download: true

sources:
  train:
    extra_args:
      train: true         # Passed to the dataset constructor
  test:
    extra_args:
      train: false

transforms:
  standard:
    _target_: torchvision.transforms.Compose
    transforms:
      - _target_: torchvision.transforms.ToTensor
      - _target_: torchvision.transforms.Normalize
        mean: [0.5]
        std: [0.5]
```

### Key parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | **required** | Dataset identifier |
| `num_classes` | int | **required** | Number of output classes |
| `path` | str | `${oc.env:HOME}/datasets` | Root directory for data storage |
| `split_train_val` | float | `0.05` (from `base.yaml`) | Fraction of training data used for validation. `0.0` disables validation. |
| `cutmix_prob` | float | `0.0` | Probability of applying CutMix augmentation during training |
| `targets_field` | str | (auto) | Attribute name for labels if non-standard (e.g., `_labels` for Food101) |
| `used_classes` | int \| null | `null` | Subset of classes to use. `null` uses all classes. |

### Dataset definition block

The `dataset` block uses Hydra's `instantiate` to create the dataset object:

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset._target_` | str | Fully qualified class name of the dataset |
| `dataset.root` | str | Data root directory |
| `dataset.download` | bool | Whether to download if not present |

### Sources block

The `sources` block defines how to create train/test splits:

```yaml
sources:
  train:
    extra_args:        # kwargs passed to the dataset constructor for this split
      train: true
  test:
    extra_args:
      train: false
```

Some datasets use `split` instead of `train` (e.g., Food101 uses `split: "train"` / `split: "test"`).

### Transforms block

The `transforms` block defines preprocessing pipelines. At minimum, define `standard`. Optionally define `augmented` for training-time augmentation:

```yaml
transforms:
  standard:            # Used for validation/test and growth
    _target_: torchvision.transforms.Compose
    transforms: [...]
  augmented:           # Used for training (optional, falls back to standard)
    _target_: torchvision.transforms.Compose
    transforms: [...]
```

For **NumPy-backed** datasets (many NAS benchmarks), `augmented` should apply **`ToTensor` first**, then augmentations that operate on `(C, H, W)` tensors. Custom tensor-space augmentation classes live in [`tools/augmentations.py`](../tools/augmentations.py). PIL-backed datasets (e.g. CIFAR-10) may use **crop/flip before `ToTensor`** instead—see [`configs/dataset_config/cifar10.yaml`](configs/dataset_config/cifar10.yaml).

### Step 2: (Optional) Register custom dataset class

If your dataset isn't a standard torchvision dataset, implement it in `tools/datasets.py` or `hydra_script/data_handling/datasets.py` and reference it via the `_target_` field.

For datasets that follow the `NpyWebDataset` pattern (downloadable `.npy` archives), subclass `NpyWebDataset` in `tools/datasets.py` — see `AddNIST` for an example.

### Step 3: Use the new dataset

```bash
uv run python -m hydra_script.train_and_grow dataset_config=<your_dataset>
```
