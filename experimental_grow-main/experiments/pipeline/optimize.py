from smac import MultiFidelityFacade, Scenario, HyperparameterOptimizationFacade
from smac.multi_objective.parego import ParEGO
from smac.runhistory.runhistory import StatusType
from ConfigSpace import ConfigurationSpace, Integer, Float
from run_pipeline import main
from pareto import plot_pareto, compute_pareto_mask
import wandb

import os
import subprocess
import uuid
import json
import time
import re
import warnings
import numpy as np

def get_configspace():
    cs = ConfigurationSpace()
#    cs.add(Integer(
#        name="training.epochs",
#        bounds=(1, 50),
#        log=False,
#    ))
#    cs.add(Integer(
#        name="growth.neurons",
#        bounds=(5, 50),
#        log=False,
#    ))
    cs.add(Float(
        name="growth.neuron_selection_threshold",
        bounds=(0.0, 0.0005),
        log=False,
    ))
    cs.add(Float(
        name="training.es_abs_delta",
        bounds=(0.0, 0.005),
        log=False,
    ))
#    cs.add(Float(
#        name="training.weight_decay",
#        bounds=(0.003, 0.01),
#        log=False,
#    ))
    return cs

def target(config: dict, seed, budget: int = None):
    extra_config = {}
    extra_config["experiment.seed"] = int(seed)
    if budget:
        extra_config["growth.steps"] = int(budget)
    for key, value in config.items():
        extra_config[key] = value
    print(f"Starting smac configuration {extra_config}")
    # try:
    return submit_slurm_job(extra_config)
    # except Exception:
    #     return [np.inf]*5, StatusType.CRASHED

def get_slurm_status(job_id):
    # Check active jobs
    out = None
    try:
        out = subprocess.run(
            ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
            capture_output=True,
            text=True
        ).stdout.strip()
    except subprocess.CalledProcessError as e:
        print("SQUEUE ERROR:", e.output)
        raise
    except Exception as e:
        print("SQUEUE ERROR:", e)

    if out:
        return out  # RUNNING, PENDING, etc.

    # Otherwise it has completed / failed → use sacct
    try:
        out = subprocess.run(
            ["sacct", "-j", str(job_id), "-o", "State", "-n"],
            capture_output=True,
            text=True
        ).stdout.strip()
    except subprocess.CalledProcessError as e:
        print("SACCT ERROR:", e.output)
        raise
    except Exception as e:
        print("SACCT ERROR:", e)

    if out:
        return out.split()[0]

    return "UNKNOWN"

def submit_slurm_job(config_dict):
    job_id = str(uuid.uuid4())
    results_dir = "temp/results"
    os.makedirs(results_dir, exist_ok=True)
    
    result_path = os.path.join(results_dir, f"{job_id}.json")
    slurm_script = f"temp/slurm_job_{job_id}.sh"

    # Load template
    with open("smac_worker/smac_worker.sh", "r") as f:
        bash = f.read()
    
    # Fill in placeholders
    bash = bash.replace("{CONFIG}", json.dumps(config_dict)).replace("{RESULT}", result_path)

    # Write slurm script
    with open(slurm_script, "w") as f:
        f.write(bash)

    # Submit
    try:
        out = subprocess.check_output(
            ["sbatch", slurm_script],
            stderr=subprocess.STDOUT,
            text=True
        )
        print(f"[SMAC] {out}")
    except subprocess.CalledProcessError as e:
        print("SBATCH ERROR:", e.output)
        print("  cwd :", os.getcwd())
        print("  rc  :", e.returncode)
        raise

    # Expected output: "Submitted batch job 12345678"
    match = re.search(r"Submitted batch job (\d+)", out)
    if not match:
        raise RuntimeError(f"Could not parse job ID from sbatch output: {out}")

    job_id = int(match.group(1))

    # Wait for result
    status = None
    while not os.path.exists(result_path) and status not in ["COMPLETED", "FAILED", "CANCELLED", "UNKNOWN"]:
        time.sleep(30)
        _status = get_slurm_status(job_id)
        if status != _status:
            warnings.warn(f"[SLURM] Job id {job_id} status: {_status}", UserWarning)
        status = _status
    
    # Read cost
    with open(result_path, "r") as f:
        result = json.load(f)

    return result

def main_optimize():
    config_space = get_configspace()

    min_budget = 5
    max_budget = 10
    workers = 8
    trials = 20

    scenario = Scenario(
        configspace=config_space,
        objectives=["nb_params", "loss_train", "loss_val", "err_train", "err_val"],
        n_trials=trials,
        min_budget=min_budget,
        max_budget=max_budget,
        deterministic=True,
        n_workers=workers,
    )

    smac = MultiFidelityFacade(
        scenario=scenario,
        target_function=target,
        initial_design=MultiFidelityFacade.get_initial_design(scenario, n_configs=5),
        intensifier=MultiFidelityFacade.get_intensifier(
            scenario,
            incumbent_selection="highest_budget",
        ),
        overwrite=False,
    )

    wandb.init(project="SMAC")

    incumbent = smac.optimize()
    print("Previously finished trials:", smac.runhistory.finished)

    pareto = smac.intensifier.get_incumbents()
    print("Best config found:", pareto)

    all_hp_keys = sorted({k for cfg in pareto for k in dict(cfg).keys()})
    if hasattr(smac.scenario, "objectives"):
        objective_names = smac.scenario.objectives
    else:
        objective_names = [f"obj_{i}" for i in range(len(list(smac.runhistory.get_cost(pareto[0]))))]
    
    columns = objective_names + all_hp_keys
    rows = []
    for key, value in smac.runhistory.items():
        cfg = smac.runhistory.get_config(key.config_id)
        cost = value.cost
        print(dict(cfg), cost)
        row = list(cost) + [dict(cfg).get(k) for k in all_hp_keys]
        rows.append(row)

    table = wandb.Table(
            columns=columns,
            rows=rows,
        )
    
    wandb.log({
        "pareto_front": table
    })

    wandb.log({
        "pareto_scatter": wandb.plot.scatter(
            table=table,
            x="nb_params",
            y="err_val",
            title="Pareto Front (params vs err_val)"
        )
    })

    
    wandb.finish()

def main_optimize_parego():
    config_space = get_configspace()

    workers = 15
    trials = 20

    scenario = Scenario(
        configspace=config_space,
        objectives=["nb_params", "loss_train", "loss_val", "err_train", "err_val"],
        n_trials=trials,
        deterministic=True,
        n_workers=workers,
    )

    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=target,
        initial_design=HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5),
        multi_objective_algorithm=ParEGO(scenario),
        overwrite=True,
    )

    api_key = os.environ.get("WANDB_KEY")
    wandb.login(key=api_key)
    wandb.init(project="SMAC")

    smac.optimize()
    print("Previously finished trials:", smac.runhistory.finished)

    pareto = smac.intensifier.get_incumbents()
    print("Best config found:", pareto)

    all_hp_keys = sorted({k for cfg in pareto for k in dict(cfg).keys()})
    if hasattr(smac.scenario, "objectives"):
        objective_names = smac.scenario.objectives
    else:
        objective_names = [f"obj_{i}" for i in range(len(list(smac.runhistory.get_cost(pareto[0]))))]
    
    columns = objective_names + all_hp_keys
    rows = []
    costs = []
    for key, value in smac.runhistory.items():
        cfg = smac.runhistory.get_config(key.config_id)
        cost = value.cost
        costs.append(cost)
        print(dict(cfg), cost)
        row = list(cost) + [dict(cfg).get(k) for k in all_hp_keys]
        rows.append(row)
    
    # Get cost of default configuration
    # default_config = config_space.get_default_configuration()
    # default_cost = smac.validate(default_config)
    # print(f"Default costs: {default_cost}\n")
    # costs.append(default_cost)
    # row = list(default_cost) + [dict(default_config).get(k) for k in all_hp_keys]
    # rows.append(row)
    
    # obj_x = 0
    # obj_y = 4
    # costs = np.array(costs)
    # mask = compute_pareto_mask(costs, obj=[obj_x, obj_y])
    # pareto_costs = costs[mask]
    # plot_pareto(costs, pareto_costs, objective_names, obj_x=obj_x, obj_y=obj_y)


    table = wandb.Table(
            columns=columns,
            rows=rows,
        )
    
    wandb.log({
        "pareto_front": table
    })

    wandb.log({
        "pareto_scatter": wandb.plot.scatter(
            table=table,
            x="nb_params",
            y="err_val",
            title="Pareto Front (params vs err_val)"
        )
    })

    wandb.finish()

if __name__ == "__main__":
    main_optimize_parego()
