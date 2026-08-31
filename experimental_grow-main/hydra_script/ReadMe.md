# ReadMe for Hydra Scripts

## Prerequisites

1. Be ready to dowload datasets at `${oc.env:HOME}/data/datasets`. 

On titanic or margaret, you can create a symlink to the desired location:
```bash
cd $HOME/data
ln -s /data/tau/iceberg_1/titanic_1/datasets datasets
```
On your local machine, you can use the same trick, we let you choose the physical location of the datasets.


2. Be ready to have logs at 
- `${oc.env:HOME}/tau_frugal/${oc.env:USER}/hydra_outputs`
a. Log with a database:
- `${oc.env:HOME}/tau_frugal/${oc.env:USER}/mlflow/mlflow.db` (MLflow tracking database)
- `${oc.env:HOME}/tau_frugal/${oc.env:USER}/mlflow/mlruns` (MLflow artifacts)
b. Log with files:
- `${oc.env:HOME}/tau_frugal/${oc.env:USER}/mlruns`

Again, you can create a symlink:

```bash
cd $HOME
ln -s /data/tau/iceberg_1/titanic_1/experimentslogs_shared/tau_frugal tau_frugal
mkdir -p tau_frugal/${USER}/hydra_outputs
mkdir -p tau_frugal/${USER}/mlruns
mkdir -p tau_frugal/${USER}/mlflow/mlruns
touch tau_frugal/${USER}/mlflow/mlflow.db
```

3. Make sure the `gromo` is downloaded and in `../gromo`.

## Usage

To run the baseline experiments on `titanic` use:

```bash
uv sync
uv run python -m hydra_script.train_and_grow \
    --config-name=sweep_baseline.yaml \
    --multirun
```

We recommand to use `tmux` or `screen` to avoid interruptions. We also recommand to redirect the output to a log file:

```bash
uv run python -m hydra_script.train_and_grow \
    --config-name=sweep_baseline.yaml \
    --multirun \
    > logs/sweep_baseline_$(date +%m_%d_%H_%M_%S).out \
    2> logs/sweep_baseline_$(date +%m_%d_%H_%M_%S).err \
    &
```

## Accessing results

Using the `mlflow` part of this ([https://gitlab.inria.fr/trudkiew/remote-notebook](https://gitlab.inria.fr/trudkiew/remote-notebook)), you can just run (where $USER is your username on titanic/margaret):

For file MLflow UI:
```bash
sh remote_mlflow.sh /home/tau/$USER/tau_frugal/$USER/mlruns
```

For SQL MLflow UI:
```bash
uv run mlflow ui --backend-store-uri sqlite:////home/tau/$USER/tau_frugal/$USER/mlflow/mlflow.db
```

## Results

Here is a summary of the results obtained with the baseline experiments for ResNet18 on CIFAR-100 averaged over 5 seeds:

| Baseline | Best Test Accuracy | Duration |
|----------|--------------------|----------|
| FoGro    | 73.11 ± 0.18 % | 1.06e+04 ± 1.56e+03 s  |
| Random Zeros | 72.79 ± 0.27 % | 1.05e+04 ± 1.70e+03 s |
| Full ResNet | 75.21 ± 0.24 % | 1.16e+04 ± 2.07e+03 s  |

On CIFAR-10, the results are:
| Baseline | Best Test Accuracy | Duration |
|----------|--------------------|----------|
| FoGro    | 92.36 ± 0.34 % | 1.00e+04 ± 1.06e+03 s  |
| Random Zeros | 92.54 ± 0.24 % | 1.13e+04 ± 5.40e+01 s |
| Full ResNet | 93.61 ± 0.38 % | 9.11e+03 ± 2.74e+03 s  |


### Code:

### For CIFAR-100 experiments:

For random and full baselines:
- gromo: theo-dev, 5d8e0319ce53bbc82515e6870a3d7cb755151279
- experimental_grow: fix-eigh, 16cb0265c43be94319964b7c38b351bb0d72453d

For FoGro (same but with some docs modified):
- gromo: theo-dev, b46a5f981333fd4cc5079a78b6fb665d97247c73
- experimental_grow: fix-eigh, 16cb0265c43be94319964b7c38b351bb0d72453d

### For CIFAR-10 experiments:

- gromo: theo-dev, 6ed444e40a5750903298d269c3865ff43389d6a1
- experimental_grow: main, 2d683c629869bd2ee263e4d175c8dc80bab34e9d


# Usage for EGG experiments

To run the experiments with and without growth on EGG, you can use the following commands:
```bash
uv sync --group eeg
uv run python -m hydra_script.train_and_grow \
    --config-name=eeg_full.yaml \
    --multirun \
    > logs/eeg_full_$(date +%m_%d_%H_%M_%S).out \
    2> logs/eeg_full_$(date +%m_%d_%H_%M_%S).err \
    &
uv run python -m hydra_script.train_and_grow \
    --config-name=eeg_growth.yaml \
    --multirun \
    > logs/eeg_growth_$(date +%m_%d_%H_%M_%S).out \
    2> logs/eeg_growth_$(date +%m_%d_%H_%M_%S).err \
    &
```
