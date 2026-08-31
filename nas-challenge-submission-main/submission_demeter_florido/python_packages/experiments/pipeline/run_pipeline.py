import argparse
import functools
import gc
import math
import operator
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.pipeline import pipeline

available_steps: dict[str, Callable] = {
    "set_random_seed": pipeline.set_random_seed,
    "load_data": pipeline.load_data,
    "split_data": pipeline.split_data,
    "create_dataloaders": pipeline.create_dataloaders,
    "set_global_device": pipeline.set_global_device,
    "create_model": pipeline.create_model,
    "load_model": pipeline.load_model,
    "train": pipeline.train,
    "evaluate": pipeline.evaluate,
    "evaluate_and_log": pipeline.evaluate_and_log,
    "disable_early_stopping": pipeline.disable_early_stopping,
    "cycle_layers" : pipeline.cycle_layers,
    "define_growth_actions": pipeline.define_growth_actions,
    "init_computation": pipeline.init_computation,
    "update_computation": pipeline.update_computation,
    "reset_computation": pipeline.reset_computation,
    "compute_optimal_delta": pipeline.compute_optimal_delta,
    "calculate_bottleneck": pipeline.calculate_bottleneck,
    "delete_update": pipeline.delete_update,
    "estimate_most_important_node": pipeline.estimate_most_important_node,
    "estimate_dependencies": pipeline.estimate_dependencies,
    "restrict_growth_actions": pipeline.restrict_growth_actions,
    "pick_random_action": pipeline.pick_random_action,
    "execute_expansion": pipeline.execute_expansion,
    "calculate_amplitude_factor": pipeline.calculate_amplitude_factor,
    "choose_best_growth_action": pipeline.choose_best_growth_action,
    "apply_change": pipeline.apply_change,
    "update_size": pipeline.update_size,
    "empty_cache": pipeline.empty_cache,
    "log_results": pipeline.log_results,
    "save_model": pipeline.save_model,
    "save_intermediate_model": pipeline.save_intermediate_model,
}

_BENCHMARK_PREFIX = "local_base_multi_dataset"
_CIFAR_ABLATION_PREFIXES = (
    "cifar_global_scheduler_350",
    "cifar_growth_label_smoothing",
    "cifar_post_growth_scheduler",
    "cifar_noisy_pre_grad_005",
    "cifar_noisy_pre_grad_010",
    "cifar_variance_transfer",
    "cifar_linear_warmup",
    "cifar_init_ablation",
)


def infer_wandb_organization(cfg):
    run_name = cfg.get("experiment", {}).get("name")
    if not run_name:
        return {}, {}

    dataset = cfg.get("dataset", {}).get("name")
    init_strategy = cfg.get("growth", {}).get("initialization_strategy")
    if not dataset or not init_strategy:
        return {}, {}

    if run_name.startswith(f"{_BENCHMARK_PREFIX}_"):
        suite = "benchmark"
        area = _BENCHMARK_PREFIX
        ablation = "local_base"
        group = f"benchmark__{dataset}__{ablation}"
        job_type = "benchmark_local_base"
    else:
        matched_ablation = None
        for prefix in _CIFAR_ABLATION_PREFIXES:
            if run_name.startswith(f"{prefix}_"):
                matched_ablation = prefix
                break
        if matched_ablation is None:
            return {}, {}
        suite = "cifar_ablation"
        area = matched_ablation
        ablation = matched_ablation
        group = f"cifar__{dataset}__{ablation}"
        job_type = "cifar_ablation"

    tags = {
        "organization.suite": suite,
        "organization.area": area,
        "dataset.name": dataset,
        "growth.initialization_strategy": init_strategy,
        "ablation.name": ablation,
    }
    return {"group": group, "job_type": job_type}, tags


def div_remainder(n, interval):
    # finds divisor and remainder given some n/interval
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder


def show_time(seconds):
    # show amount of time as human readable
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < (60 * 60):
        minutes, seconds = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{}h,{}m,{}s".format(hours, minutes, seconds)


def resolve_expr(expr, cfg):
    """Resolve ${a.b.c} → cfg['a']['b']['c']"""
    if isinstance(expr, int):
        return range(expr)
    if not isinstance(expr, str):
        return expr
    if expr.startswith("${") and expr.endswith("}"):
        path = expr[2:-1].split(".")
        val = functools.reduce(operator.getitem, path, cfg)
        return range(val)
    return expr


def execute_pipeline(cfg, steps, context):
    for step in steps:
        if isinstance(step, dict) and "loop" in step:
            loop_cfg = step["loop"]
            var = loop_cfg["var"]
            iterable = resolve_expr(loop_cfg["over"], cfg)
            for val in iterable:
                print(f"\n=== Loop {var} = {val} ===")
                # Inject loop variable in config
                cfg[var] = val
                execute_pipeline(cfg, loop_cfg["steps"], context)
        else:
            try:
                func = available_steps[step]
                print(f"\n>>> Running step: {step}")
                start_time = time.time()
                result = func(cfg, **context)
                end_time = time.time()
                print(f"Execution time {show_time(end_time - start_time)}")
                context["logger"].log_metric(f"time/{step}", end_time - start_time, cfg.get("global_step", 0))
                if isinstance(result, dict):
                    context.update(result)
                    growth_tensor_keys = {
                        "pre_activities_grad",
                        "input_B",
                        "bottleneck",
                    }
                    if growth_tensor_keys.issubset(result) and all(
                        result[key] is None for key in growth_tensor_keys
                    ):
                        gc.collect()
            except KeyError as err:
                print(err)
                raise(err)
            except RuntimeError as err:
                print(f"Unknown error occured! {err}")
                raise(err)


def _coerce_override_value(raw_value, current_value):
    if isinstance(current_value, bool):
        lowered = raw_value.lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean override value {raw_value!r}")
    if current_value is None:
        lowered = raw_value.lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        for caster in (int, float):
            try:
                return caster(raw_value)
            except ValueError:
                pass
        return raw_value
    return type(current_value)(raw_value)


def main(
    config_path="./experiments/pipeline/experiments_config.yaml",
    extra_config: dict = {},
    argv: list[str] | None = None,
) -> dict:
    if argv is None:
        argv = sys.argv[1:]

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=config_path)
    config_args, _ = config_parser.parse_known_args(argv)
    config_path = config_args.config

    # Loading pipeline configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Update smac3 hyperparameters
    for hyper, value in extra_config.items():
        path = hyper.split(".")
        target = functools.reduce(operator.getitem, path[:-1], cfg)
        operator.setitem(target, path[-1], value)

    context = {}

    # Read cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=config_path)
    parser.add_argument("--job_id", type=str)
    parser.add_argument("--node_name", type=str)
    args, unknown = parser.parse_known_args(argv)
    context.update(args.__dict__)

    # Update cmd arguments
    for i in range(0, len(unknown), 2):
        arg, value = unknown[i], unknown[i+1]
        if arg.startswith("--"):
            try:
                path = arg[2:].split(".")
                target = functools.reduce(operator.getitem, path[:-1], cfg)
                operator.setitem(
                    target,
                    path[-1],
                    _coerce_override_value(value, target[path[-1]]),
                )
                print(f"Setting {arg} to value {value}")
            except Exception as e:
                print(f'Unrecognised argument {arg}. Skipping. {e}')

    # Create tmp directory
    Path(cfg["logger"]["tmpdir"]).mkdir(parents=True, exist_ok=True)
    tmpdir = tempfile.mkdtemp(dir=cfg["logger"]["tmpdir"])
    print("Process temp folder:", tmpdir)
    cfg["logger"]["tmpdir"] = tmpdir

    # Setup logger experiment, tags, parameters
    context.update(pipeline.setup_logger(cfg, **context))
    tags = pipeline.setup_experiment_tags(cfg, **context)
    run_kwargs = {"tags": tags}
    run_name = cfg.get("experiment", {}).get("name")
    if run_name:
        if context["logger"].api == "wandb":
            run_kwargs["name"] = run_name
            organization_kwargs, organization_tags = infer_wandb_organization(cfg)
            run_kwargs.update(organization_kwargs)
            tags.update(organization_tags)
        elif context["logger"].api == "mlflow":
            run_kwargs["run_name"] = run_name
    with context["logger"](**run_kwargs) as logger:
        for key, params in cfg.items():
            if key == "pipeline":
                logger.log_parameter(key, params)
            elif key != "logger":
                for param, value in params.items():
                    logger.log_parameter(f"{key}.{param}", value)

        # Execute pipeline
        execute_pipeline(cfg, cfg["pipeline"], context)
    
    return context


if __name__ == "__main__":
    main(argv=sys.argv[1:])
