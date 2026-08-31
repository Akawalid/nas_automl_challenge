# Base Model Benchmarks

Launchers for external baseline models on the same NAS datasets and benchmark
node used by `experiments/pipeline/launch_local_base_multi_dataset.sh`.

Current launchers:

- `launch_darts_local_base_multi_dataset.sh`: DARTS baseline from
  `$HOME/dev/DemeterBM/DARTS`.
- `launch_pcdarts_local_base_multi_dataset.sh`: PC-DARTS baseline from
  `$HOME/dev/DemeterBM/PC-DARTS`.

All launchers in this directory should keep generated files under `/scratch`.
The source repositories and `experimental_grow` are used as read-only code
locations during jobs.

Run DARTS from `experimental_grow`:

```bash
bash experiments/base_model_benchmarks/launch_darts_local_base_multi_dataset.sh
```

Run PC-DARTS from `experimental_grow`:

```bash
bash experiments/base_model_benchmarks/launch_pcdarts_local_base_multi_dataset.sh
```

Shared defaults:

- Datasets: `multnist cifartile gutenberg geoclassing chesseract`
- Seeds: `0 1 2`
- Slurm partition: `gpu`
- Benchmark node: `margpu010`
- Search epochs: `50`
- Final training epochs: `200`

Model-specific defaults:

- DARTS repo: `$HOME/dev/DemeterBM/DARTS`
- PC-DARTS repo: `$HOME/dev/DemeterBM/PC-DARTS`
- DARTS MultNIST batch size: `56`
- PC-DARTS MultNIST batch size: `128`

Override any setting with environment variables, for example:

```bash
SEEDS="0 1 2 3 4" bash experiments/base_model_benchmarks/launch_darts_local_base_multi_dataset.sh
```

```bash
SEEDS="0 1 2 3 4" bash experiments/base_model_benchmarks/launch_pcdarts_local_base_multi_dataset.sh
```
