# Hydra Configuration (`hydra/`)

Controls Hydra output directories and SLURM job launcher settings.

## Output Paths (`custom_paths.yaml`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `run.dir` | str | `${HOME}/tau_frugal/${USER}/hydra_outputs/outputs/${date}/${time}` | Output directory for single runs. |
| `sweep.dir` | str | `${HOME}/tau_frugal/${USER}/hydra_outputs/multirun/${date}/${time}` | Output directory for multi-run sweeps. |
| `sweep.subdir` | str | `${hydra.job.num}` | Subdirectory per sweep job. |

## SLURM Launchers (`launcher/`)

Used for `--multirun` sweeps on SLURM clusters via the `hydra-submitit-launcher` plugin.

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `partition` | str | varies | SLURM partition name. |
| `gpus_per_node` | int | `1` | GPUs per node. |
| `nodes` | int | `1` | Number of nodes. |
| `timeout_min` | int | `1440` | Job timeout in minutes (24 hours). |
| `cpus_per_task` | int | `5` | CPUs per task. |
| `tasks_per_node` | int | `1` | Tasks per node. |
| `gres` | str | `gpu:1` | Generic resource request. |
| `array_parallelism` | int | `256` | Maximum concurrent array jobs. |
| `signal_delay_s` | int | `120` | Delay before sending signal for timeout. |
| `exclude` | str \| null | varies | Nodes to exclude. |

### Available Launcher Configs

| Config | Cluster | Partition |
|--------|---------|-----------|
| `margaret_submitit.yaml` | Margaret | `tau` |
| `titanic_submitit.yaml` | Titanic | `besteffort` |
