# MultiNIST: cinco seeds en Jean Zay desde cero

Esta guía lanza el experimento **local baseline** de
`experiments/pipeline/experiments_config.yaml` sobre MultiNIST con las seeds
`0 1 2 3 4`. Cada seed es una tarea independiente de un array Slurm y reserva
una GPU. No lanza los benchmarks externos DARTS ni PC-DARTS.

Los comandos están preparados para alguien que todavía no tiene código,
entorno Python ni datos en Jean Zay. Los valores por defecto usan H100 porque
el experimento solicita 48 horas; también se documentan V100 y A100.

## 1. Requisitos de acceso

Antes de copiar nada necesitas:

1. una cuenta IDRIS vinculada a un proyecto con horas GPU;
2. acceso desde una IP declarada en IDRIS o desde la VPN/proxy de tu centro;
3. saber qué tipo de horas tiene el proyecto: `v100`, `a100` o `h100`.

Conéctate desde tu máquina:

```bash
ssh TU_LOGIN@jean-zay.idris.fr
```

Comprueba el proyecto activo y los espacios disponibles:

```bash
idrproj
echo "$IDRPROJ"
echo "$WORK"
echo "$SCRATCH"
idr_quota_user
```

Si perteneces a varios proyectos, activa el correcto en la sesión actual:

```bash
eval "$(idrenv -d NOMBRE_PROYECTO)"
```

No pongas el repositorio, el entorno o los datos en `$HOME`: solo tiene 3 GB.
En esta guía:

- `$WORK` contiene código, entorno y dataset persistentes;
- `$SCRATCH` contiene logs, temporales y checkpoints del entrenamiento;
- `$STORE` sirve para archivar resultados importantes, pero no es visible desde
  los nodos de cálculo.

Los ficheros no usados de `$SCRATCH` se eliminan tras 30 días y este espacio no
tiene copia de seguridad.

## 2. Descargar el código

Hazlo en la frontal, que sí tiene acceso a Internet:

```bash
mkdir -p "$WORK/demeter"
cd "$WORK/demeter"

git clone git@gitlab.inria.fr:sflorido/experimental_grow.git
git clone --branch AblationDAG https://github.com/Santiago23florido/gromo.git
```

La estructura esperada es:

```text
$WORK/demeter/
├── experimental_grow/
└── gromo/
```

Si tu acceso a GitLab usa otra URL, clona tu fork con esa URL pero conserva el
nombre de directorio `experimental_grow`. Conviene anotar los commits exactos
antes del experimento:

```bash
git -C "$WORK/demeter/experimental_grow" rev-parse HEAD
git -C "$WORK/demeter/gromo" rev-parse HEAD
```

## 3. Crear el entorno Python

IDRIS recomienda Miniforge y no hace falta instalar Miniconda. El entorno se
crea en `$WORK` para no agotar el pequeño `$HOME`:

```bash
module load miniforge/24.9.0

mkdir -p "$WORK/envs"
conda create --prefix "$WORK/envs/demeter" python=3.11 -y
conda activate "$WORK/envs/demeter"

python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir -e "$WORK/demeter/gromo"
python -m pip install --no-cache-dir -e "$WORK/demeter/experimental_grow"
```

Comprueba la instalación. En la frontal es normal que CUDA dé `False`:

```bash
cd "$WORK/demeter/experimental_grow"
python -c "import torch, gromo, wandb; print(torch.__version__); print(torch.cuda.is_available()); print(gromo.__file__)"
```

Para reproducibilidad, guarda el entorno resuelto:

```bash
conda env export --prefix "$WORK/envs/demeter" > "$WORK/demeter/conda-jean-zay.yml"
python -m pip freeze > "$WORK/demeter/requirements-jean-zay.txt"
```

## 4. Descargar MultiNIST antes del job

Los nodos GPU **no tienen Internet**. El código tiene `download=True`, por lo
que hay que poblar la caché antes de enviar el array. Desde la frontal o desde
`jean-zay-pp`:

```bash
module load miniforge/24.9.0
conda activate "$WORK/envs/demeter"
cd "$WORK/demeter/experimental_grow"

export DATA_DIR="$WORK/datasets/multnist"
mkdir -p "$DATA_DIR"

python -c 'import os; from tools.datasets import MultNIST; root=os.environ["DATA_DIR"]; train=MultNIST(root=root, train=True); print("train", len(train)); del train; test=MultNIST(root=root, train=False); print("test", len(test))'
```

Verifica que existen los dos elementos que exige el job:

```bash
test -s "$DATA_DIR/MultNIST.zip"
test -d "$DATA_DIR/MultNIST_extracted"
find "$DATA_DIR/MultNIST_extracted" -name '*.npy' | head
```

No ejecutes simultáneamente las cinco seeds antes de terminar esta descarga:
podrían intentar crear la misma caché a la vez.

## 5. Seleccionar GPU y cuenta

El launcher usa por defecto:

| `GPU_TYPE` | Cuenta | Selección | QoS | Tiempo por defecto |
|---|---|---|---|---|
| `h100` | `$IDRPROJ@h100` | `-C h100` | `qos_gpu_h100-t4` | 48 h |
| `v100` | `$IDRPROJ@v100` | `-C v100-32g` | `qos_gpu-t4` | 48 h |
| `a100` | `$IDRPROJ@a100` | `-C a100` | `qos_gpu_a100-t3` | 20 h, máximo permitido |

H100 y A100 cargan respectivamente `arch/h100` y `arch/a100` antes de
Miniforge. Si tu asignación o QoS es distinta, puedes sobreescribir `ACCOUNT`,
`PARTITION`, `QOS`, `SBATCH_TIME`, `CONSTRAINT` y `MODULES` sin editar el
script. El launcher y `run_multnist_jean_zay.slurm` son exclusivos de Jean
Zay. No modifican ni reutilizan las directivas de partición, nodos o rutas de
los scripts de Margaret.

Atención: A100 no ofrece una QoS de más de 20 horas. Si una seed no termina en
ese límite, hay que implementar reanudación desde checkpoint o usar H100/V100
con QoS larga; aumentar simplemente `SBATCH_TIME` no funcionará.

## 6. Smoke test de una sola seed

Antes de gastar cinco GPUs, envía una ejecución reducida. Para H100:

```bash
cd "$WORK/demeter/experimental_grow"

GPU_TYPE=h100 \
ARRAY_RANGE=0-0 \
SEEDS="0" \
QOS=qos_gpu_h100-dev \
SBATCH_TIME=00:30:00 \
bash experiments/pipeline/launch_jean_zay_multnist_5seeds.sh \
  --training.epochs 1 \
  --growth.steps 1 \
  --growth.neuron_epochs 1
```

Para A100 cambia `GPU_TYPE=a100` y usa `QOS=qos_gpu_a100-dev`; para V100,
`GPU_TYPE=v100` y `QOS=qos_gpu-dev`.

El comando imprime un job ID. Inspecciónalo con:

```bash
squeue --me
tail -f "$SCRATCH/demeter/local_base_multnist/slurm_logs/"*.out
```

El log debe mostrar una GPU visible, `cuda: True`, dataset `multnist`, seed `0`
y el comienzo del pipeline. Cancela un job incorrecto con `scancel JOB_ID`.

## 7. Lanzar las cinco seeds completas

Con H100:

```bash
cd "$WORK/demeter/experimental_grow"
GPU_TYPE=h100 bash experiments/pipeline/launch_jean_zay_multnist_5seeds.sh
```

Se crea un array `0-4%5`: seed `0`, `1`, `2`, `3` y `4`, hasta cinco tareas en
paralelo. Para reducir el consumo simultáneo, por ejemplo a dos GPUs:

```bash
GPU_TYPE=h100 MAX_PARALLEL=2 \
bash experiments/pipeline/launch_jean_zay_multnist_5seeds.sh
```

Para otra asignación:

```bash
GPU_TYPE=v100 bash experiments/pipeline/launch_jean_zay_multnist_5seeds.sh
```

```bash
GPU_TYPE=a100 bash experiments/pipeline/launch_jean_zay_multnist_5seeds.sh
```

Cada tarea recibe exactamente una seed porque el launcher fija:

```text
DATASETS=multnist
SEEDS="0 1 2 3 4"
INIT_STRATEGIES=local
array=0-4
```

## 8. Logs, resultados y W&B

Los resultados se escriben bajo:

```text
$SCRATCH/demeter/local_base_multnist/
├── slurm_logs/
├── tmp/
└── wandb/
```

Comandos útiles:

```bash
squeue --me
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,AllocTRES
grep -R "cuda:\|Dataset:\|Seed:\|Traceback" \
  "$SCRATCH/demeter/local_base_multnist/slurm_logs"
```

El launcher fija `WANDB_MODE=offline`, porque los nodos de cálculo no pueden
contactar W&B. Cuando terminen los jobs, sincroniza desde una frontal con
Internet y con tu API key configurada:

```bash
module load miniforge/24.9.0
conda activate "$WORK/envs/demeter"
wandb login
find "$SCRATCH/demeter/local_base_multnist/wandb" -type d -name 'offline-run-*' \
  -exec wandb sync {} \;
```

Copia los resultados que quieras conservar a `$WORK` y archívalos después en
`$STORE`; no dejes la única copia en `$SCRATCH`.

## 9. Fuentes oficiales de Jean Zay

- [Primeros pasos, acceso SSH y espacios de disco](https://www.idris.fr/static/intro/doc_nouvel_utilisateur-eng.html)
- [Particiones GPU y QoS](https://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-exec_partition_slurm-eng.html)
- [Entornos Python personales y Miniforge](https://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-python-env-eng.html)
- [Módulos y arquitecturas A100/H100](https://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-doc_module-eng.html)
- [Guía para usuarios con varios proyectos](https://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-doc_multi_projet-eng.html)
