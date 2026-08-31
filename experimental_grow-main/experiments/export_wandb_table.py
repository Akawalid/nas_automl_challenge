import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml

try:
    import wandb
except ImportError as exc:  # pragma: no cover - import-time guard
    raise SystemExit(
        "wandb is required for this script. Install it in the active environment first."
    ) from exc


@dataclass
class DatasetStats:
    accuracy: list[float]
    params: list[float]


def _parse_filter(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(
            f"Invalid filter {raw!r}. Expected format KEY=VALUE, "
            "for example growth.initialization_strategy=init_scaling_ablation."
        )
    key, raw_value = raw.split("=", 1)
    return key.strip(), yaml.safe_load(raw_value)


def _lookup(mapping: dict[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]

    current: Any = mapping
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _matches_filters(
    run,
    filters: list[tuple[str, Any]],
    experiment_name_prefix: str | None,
) -> bool:
    for key, expected in filters:
        actual = _lookup(run.config, key)
        if actual != expected:
            return False

    if experiment_name_prefix:
        experiment_name = _lookup(run.config, "experiment.name")
        if not isinstance(experiment_name, str):
            return False
        if not experiment_name.startswith(experiment_name_prefix):
            return False

    return True


def _format_mean_std(
    values: list[float],
    *,
    scale: float,
    suffix: str,
    decimals: int,
    as_percent: bool = False,
) -> str:
    if not values:
        return "-"

    array = np.asarray(values, dtype=float)
    if as_percent:
        array = array * 100.0
    array = array / scale

    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    pm = "\u00b1"
    if as_percent:
        return f"{mean:.{decimals}f}% {pm} {std:.{decimals}f}"
    return f"{mean:.{decimals}f}{suffix} {pm} {std:.{decimals}f}{suffix}"


def _markdown_row(label: str, values: list[str]) -> str:
    return "| " + " | ".join([label, *values]) + " |"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-dataset W&B tables for test accuracy and number of parameters."
        )
    )
    parser.add_argument("--entity", required=True, help="W&B entity/user/team name.")
    parser.add_argument("--project", required=True, help="W&B project name.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Ordered dataset names to appear as columns.",
    )
    parser.add_argument(
        "--dataset-key",
        default="dataset.name",
        help="Config key used to identify the dataset for each run.",
    )
    parser.add_argument(
        "--acc-key",
        default="training/test accuracy",
        help="Summary metric key for final test accuracy.",
    )
    parser.add_argument(
        "--param-key",
        default="complexity/nb of parameters",
        help="Summary metric key for final number of parameters.",
    )
    parser.add_argument(
        "--method-label",
        default="Demeter (Ours)",
        help="Label used in the printed tables.",
    )
    parser.add_argument(
        "--config-filter",
        action="append",
        default=[],
        help=(
            "Repeatable config filter KEY=VALUE, "
            "for example growth.initialization_strategy=init_scaling_ablation."
        ),
    )
    parser.add_argument(
        "--experiment-name-prefix",
        default=None,
        help="Optional prefix filter on config key experiment.name.",
    )
    parser.add_argument(
        "--state",
        default="finished",
        help="W&B run state filter passed to the API. Default: finished.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="W&B API timeout in seconds.",
    )
    parser.add_argument(
        "--accuracy-decimals",
        type=int,
        default=1,
        help="Decimals for the accuracy table.",
    )
    parser.add_argument(
        "--params-decimals",
        type=int,
        default=2,
        help="Decimals for the parameter table.",
    )
    parser.add_argument(
        "--params-scale",
        type=float,
        default=1_000_000.0,
        help="Divide number of parameters by this scale before printing.",
    )
    parser.add_argument(
        "--params-suffix",
        default="M",
        help="Suffix appended to the formatted parameter values.",
    )
    args = parser.parse_args()

    config_filters = [_parse_filter(raw) for raw in args.config_filter]
    api = wandb.Api(timeout=args.timeout)
    runs = api.runs(f"{args.entity}/{args.project}", filters={"state": args.state})

    grouped: dict[str, DatasetStats] = defaultdict(
        lambda: DatasetStats(accuracy=[], params=[])
    )

    for run in runs:
        if not _matches_filters(run, config_filters, args.experiment_name_prefix):
            continue

        dataset_name = _lookup(run.config, args.dataset_key)
        if dataset_name not in args.datasets:
            continue

        acc = run.summary.get(args.acc_key)
        params = run.summary.get(args.param_key)
        if acc is None or params is None:
            continue

        grouped[str(dataset_name)].accuracy.append(float(acc))
        grouped[str(dataset_name)].params.append(float(params))

    acc_cells = [
        _format_mean_std(
            grouped[name].accuracy,
            scale=1.0,
            suffix="",
            decimals=args.accuracy_decimals,
            as_percent=True,
        )
        for name in args.datasets
    ]
    param_cells = [
        _format_mean_std(
            grouped[name].params,
            scale=args.params_scale,
            suffix=args.params_suffix,
            decimals=args.params_decimals,
            as_percent=False,
        )
        for name in args.datasets
    ]

    header = _markdown_row("Method", args.datasets)
    separator = _markdown_row("---", ["---"] * len(args.datasets))

    print("Accuracy table")
    print(header)
    print(separator)
    print(_markdown_row(args.method_label, acc_cells))
    print()
    print("Parameter table")
    print(header)
    print(separator)
    print(_markdown_row(args.method_label, param_cells))
    print()
    print("Run counts")
    print(header)
    print(separator)
    print(
        _markdown_row(
            args.method_label,
            [
                str(
                    min(
                        len(grouped[name].accuracy),
                        len(grouped[name].params),
                    )
                )
                for name in args.datasets
            ],
        )
    )


if __name__ == "__main__":
    main()
