# Dataset Configuration (`dataset_config/`)

Defines how datasets are loaded, split, and transformed.

## Shared Parameters (from `base.yaml`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | `${oc.env:HOME}/datasets` | Root directory where datasets are stored/downloaded. |
| `split_train_val` | float | `0.05` | Fraction of training data reserved for validation. Set to `0.0` to disable validation. |

## Per-Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | **required** | Dataset identifier used for logging and dispatch. |
| `num_classes` | int | **required** | Number of output classes. |
| `dataset._target_` | str | **required** | Fully qualified class name for Hydra instantiation. |
| `dataset.root` | str | `${dataset_config.path}` | Data root directory. |
| `dataset.download` | bool | `true` | Download data if missing. |
| `sources.train.extra_args` | dict | varies | Keyword arguments passed to the dataset constructor for the training split. |
| `sources.test.extra_args` | dict | varies | Keyword arguments passed to the dataset constructor for the test split. |
| `transforms.standard` | Compose | **required** | Transform pipeline for validation, test, and growth evaluation. |
| `transforms.augmented` | Compose | optional | Transform pipeline for training. Training uses `"augmented"` iff this **key** exists under `transforms` (membership), not iff the value is non-null—omit the key to use `standard` for training, rather than `augmented: null`. |
| `mixup_alpha` | float | `0.0` | Beta parameter for batch-level Mixup. `0.0` disables Mixup. |
| `cutmix_alpha` | float | `0.0` | Beta parameter for batch-level CutMix. `0.0` disables CutMix. |
| `mixup_prob` | float | `1.0` | Probability of applying any mixing (Mixup or CutMix) to a batch. |
| `mixup_switch_prob` | float | `0.5` | Probability of choosing CutMix over Mixup when both are enabled. |
| `mixup_off_epoch` | int | `0` | Disable Mixup/CutMix from this 1-indexed epoch on. `0` keeps it enabled for the whole run. |
| `cutmix_prob` | float | `0.0` | **Legacy.** Old single-knob CutMix switch. Superseded by `cutmix_alpha`/`mixup_prob`; still honored when the new keys are absent. |
| `targets_field` | str | (auto) | Attribute name for targets if non-standard (e.g., `_labels` for Food101). |

Mixup/CutMix are applied at the batch level by `gradient_descent_mixup`
(`experiments/auxilliary_functions.py`) and combined with the configured
loss via a two-term mixed objective, which is equivalent to
`SoftTargetCrossEntropy` on the (optionally label-smoothed) mixed targets.
Mixed-precision (AMP) training is controlled by `general.amp` (see
[`general/`](../general/)), not by the dataset config.
| `used_classes` | int \| null | `null` | If set, only use this many classes (subset selection). |

## EEG-Specific Parameters (`dry-ricker.yaml`)

The `dry-ricker` dataset uses a completely different loading path and has its own set of parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fmin` | int | `1` | Minimum frequency (Hz) for bandpass filtering. |
| `fmax` | int | `40` | Maximum frequency (Hz) for bandpass filtering. |
| `window_size` | float | `0.35` | Window size in seconds for signal segmentation. |
| `sample_freq` | int | `500` | Sampling frequency (Hz). |
| `fps` | int | `60` | **[UNCLEAR]** Frames per second — exact role in the EEG pipeline is not documented. |
| `timewise` | str | `time_sample` | **[UNCLEAR]** Temporal processing mode. Exact options and behavior are not documented. |
| `n_class` | int | `5` | Number of EEG classes. |
| `ncal` | int | `4` | Number of calibration trials. |
| `participant` | str | `"P1"` | Participant identifier (e.g., `P1` through `P24`). |
| `pretrain` | bool | `false` | **[UNCLEAR]** Whether to use a pre-trained model/features. Exact behavior is not documented. |
| `balanced_accuracy` | bool | `true` | Whether to use balanced accuracy as the evaluation metric. |

## Available Configs

| Config | Dataset | Classes |
|--------|---------|---------|
| `mnist.yaml` | MNIST handwritten digits | 10 |
| `fashion-mnist.yaml` | Fashion-MNIST | 10 |
| `cifar10.yaml` | CIFAR-10 | 10 |
| `cifar100.yaml` | CIFAR-100 | 100 |
| `addnist.yaml` | AddNIST (NAS small benchmark, `tools.datasets.AddNIST`) | 20 |
| `multnist.yaml` | MultNIST (NAS small benchmark, `tools.datasets.MultNIST`) | 10 |
| `cifartile.yaml` | CIFARTile (NAS small benchmark, `tools.datasets.CIFARTile`) | 4 |
| `gutenberg.yaml` | Gutenberg (NAS small benchmark, `tools.datasets.Gutenberg`) | 6 |
| `geoclassing.yaml` | GeoClassing (NAS small benchmark, `tools.datasets.GeoClassing`) | 10 |
| `chesseract.yaml` | Chesseract (NAS small benchmark, `tools.datasets.Chesseract`) | 3 |
| `food101.yaml` | Food-101 | 101 |
| `dry-ricker.yaml` | Dry-Ricker EEG | 5 |

## NAS small benchmarks: training augmentation

Several NCL / Figshare benchmarks load **NumPy** arrays (`NpyWebDataset` in `tools.datasets`). For those, training uses `transforms.augmented` when present. List **`torchvision.transforms.ToTensor` first**, then tensor-space ops (flips, `tools.augmentations.PerChannelRandomAffine`, etc.). Validation and test use `transforms.standard` only.

**Downloading:** scripted fetch uses Figshare `ndownloader` URLs in `tools.datasets` (see `_figshare_article_zip_url`) so archives land under `dataset_config.path` without relying on the NCL browser UI.

Custom tensor-space augmentation classes (e.g. `PerChannelRandomAffine`, `RandomRot90`) live in [`tools/augmentations.py`](../../../tools/augmentations.py) and are referenced from YAML via `_target_`. Configs **addnist**, **multnist**, **cifartile**, and **geoclassing** define `transforms.augmented`; **gutenberg** and **chesseract** omit it (training uses `standard`).

## Adding a New Dataset

See the [main CONFIGURATION.md](../../CONFIGURATION.md#how-to-add-a-new-dataset) for step-by-step instructions.
