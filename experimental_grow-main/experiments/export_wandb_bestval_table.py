#!/usr/bin/env python3
"""Export W&B benchmark tables using the best validation epoch per run."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


DEFAULT_DATASET_LABELS = {
    "multnist": "MultNIST",
    "cifartile": "CifarTile",
    "gutenberg": "Gutenberg",
    "geoclassing": "Geoclassing",
    "chesseract": "Chesseract",
}


def _load_wandb():
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - import-time guard
        raise SystemExit(
            "wandb is required for this script. Install it in the active environment first."
        ) from exc
    return wandb


@dataclass(frozen=True)
class BestValPoint:
    run_name: str
    dataset: str
    epoch: float
    val_accuracy: float
    test_accuracy: float
    parameters: float
    gpu_hours: float
    runtime_source: str
    energy_kwh: float | None
    energy_source: str


@dataclass
class DatasetStats:
    accuracy: list[float]
    params: list[float]
    gpu_hours: list[float]
    energy_kwh: list[float]


@dataclass(frozen=True)
class RuntimeInfo:
    seconds: float
    gpu_hours: float
    source: str


@dataclass(frozen=True)
class EnergyInfo:
    kwh: float
    source: str


def _parse_filter(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(
            f"Invalid filter {raw!r}. Expected format KEY=VALUE, "
            "for example growth.initialization_strategy=local."
        )
    key, raw_value = raw.split("=", 1)
    return key.strip(), _parse_scalar(raw_value.strip())


def _parse_scalar(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        return raw_value


def _lookup(mapping: dict[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]

    current: Any = mapping
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _run_display_name(run) -> str:
    for attr in ("display_name", "name", "id"):
        value = getattr(run, attr, None)
        if value:
            return str(value)
    return "<unknown>"


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _history_rows(history: Any):
    if hasattr(history, "iterrows"):
        for _, row in history.iterrows():
            yield row.to_dict()
        return

    if isinstance(history, dict):
        keys = list(history)
        length = max((len(history[key]) for key in keys), default=0)
        for index in range(length):
            yield {
                key: history[key][index]
                for key in keys
                if index < len(history[key])
            }
        return

    yield from history


def _run_attr(run, *names: str) -> Any:
    for name in names:
        value = getattr(run, name, None)
        if value is not None:
            return value

    attrs = getattr(run, "_attrs", None)
    if isinstance(attrs, dict):
        for name in names:
            if name in attrs:
                return attrs[name]
    return None


def _parse_timestamp(value: Any) -> dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        timestamp = value
    else:
        raw = str(value).strip()
        if not raw:
            return None
        raw = raw.replace("Z", "+00:00")
        raw = re.sub(r"(\.\d{6})\d+", r"\1", raw)
        try:
            timestamp = dt.datetime.fromisoformat(raw)
        except ValueError:
            return None

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    return timestamp


def _runtime_seconds_from_timestamps(run) -> float | None:
    start = _parse_timestamp(
        _run_attr(run, "created_at", "createdAt", "started_at", "startedAt")
    )
    end = _parse_timestamp(
        _run_attr(
            run,
            "heartbeat_at",
            "heartbeatAt",
            "updated_at",
            "updatedAt",
            "ended_at",
            "endedAt",
        )
    )
    if start is None or end is None:
        return None

    seconds = (end - start).total_seconds()
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    return seconds


def _history_metric_series(
    run,
    *,
    metric_key: str,
    step_key: str,
    samples: int,
    stream: str | None = None,
) -> list[tuple[float, float]]:
    if stream:
        kwargs = {
            "samples": samples,
            "pandas": False,
            "stream": stream,
        }
    else:
        keys = [metric_key] if step_key == "_step" else [metric_key, step_key]
        kwargs = {
            "keys": keys,
            "samples": samples,
            "pandas": False,
        }
        if step_key != "_step":
            kwargs["x_axis"] = step_key

    try:
        history = run.history(**kwargs)
    except TypeError:
        kwargs.pop("x_axis", None)
        try:
            history = run.history(**kwargs)
        except TypeError:
            kwargs.pop("stream", None)
            history = run.history(**kwargs)
    except Exception:
        return []

    rows: list[tuple[float, float]] = []
    for row in _history_rows(history):
        value = _to_float(row.get(metric_key))
        step = _to_float(row.get(step_key))
        if step is None and step_key != "_timestamp":
            step = _to_float(row.get("_timestamp"))
        if step is None and step_key != "_runtime":
            step = _to_float(row.get("_runtime"))
        if value is None or step is None:
            continue
        rows.append((step, value))
    return rows


def _summary_keys(run) -> list[str]:
    summary = getattr(run, "summary", {}) or {}
    if hasattr(summary, "keys"):
        try:
            return [str(key) for key in summary.keys()]
        except Exception:
            pass
    if hasattr(summary, "_json_dict") and isinstance(summary._json_dict, dict):
        return [str(key) for key in summary._json_dict.keys()]
    return []


def _sample_history_keys(run, *, samples: int, stream: str | None = None) -> list[str]:
    kwargs = {
        "samples": samples,
        "pandas": False,
    }
    if stream:
        kwargs["stream"] = stream
    try:
        history = run.history(**kwargs)
    except TypeError:
        kwargs.pop("stream", None)
        try:
            history = run.history(**kwargs)
        except Exception:
            return []
    except Exception:
        return []

    keys: set[str] = set()
    for row in _history_rows(history):
        keys.update(str(key) for key in row.keys())
    return sorted(keys)


def _runtime_seconds_from_time_metrics(
    run,
    *,
    time_metric_prefix: str,
    samples: int,
) -> float | None:
    time_keys = sorted(
        key for key in _summary_keys(run) if key.startswith(time_metric_prefix)
    )
    if not time_keys:
        return None

    try:
        history = run.history(keys=time_keys, samples=samples, pandas=False)
    except Exception:
        return None

    total = 0.0
    found = False
    for row in _history_rows(history):
        for key in time_keys:
            value = _to_float(row.get(key))
            if value is None:
                continue
            total += value
            found = True

    if not found or not math.isfinite(total) or total <= 0:
        return None
    return total


def _looks_like_gpu_power_watts_key(key: str) -> bool:
    normalized = key.lower().replace("/", ".")
    if "gpu" not in normalized or "power" not in normalized:
        return False
    return "watt" in normalized or normalized.endswith(".w") or "(w)" in normalized


def _resolve_gpu_power_keys(
    run,
    explicit_keys: list[str],
    *,
    discovery_samples: int,
    streams: list[str],
) -> list[str]:
    if explicit_keys:
        return explicit_keys
    candidates = set(_summary_keys(run))
    for stream in streams:
        stream_arg = None if stream == "default" else stream
        candidates.update(
            _sample_history_keys(
                run,
                samples=discovery_samples,
                stream=stream_arg,
            )
        )
    return sorted(key for key in candidates if _looks_like_gpu_power_watts_key(key))


def _integrate_power_series_kwh(series: list[tuple[float, float]]) -> float | None:
    if len(series) < 2:
        return None

    by_time: dict[float, float] = {}
    for timestamp, watts in series:
        if watts < 0:
            continue
        by_time[timestamp] = watts

    ordered = sorted(by_time.items())
    if len(ordered) < 2:
        return None

    watt_seconds = 0.0
    previous_time, previous_watts = ordered[0]
    for current_time, current_watts in ordered[1:]:
        delta = current_time - previous_time
        if delta > 0:
            watt_seconds += 0.5 * (previous_watts + current_watts) * delta
        previous_time, previous_watts = current_time, current_watts

    if not math.isfinite(watt_seconds) or watt_seconds <= 0:
        return None
    return watt_seconds / 3_600_000.0


def _energy_from_gpu_power_metrics(
    run,
    *,
    explicit_power_keys: list[str],
    samples: int,
    discovery_samples: int,
    time_key: str,
    streams: list[str],
) -> EnergyInfo | None:
    power_keys = _resolve_gpu_power_keys(
        run,
        explicit_power_keys,
        discovery_samples=discovery_samples,
        streams=streams,
    )
    if not power_keys:
        return None

    total_kwh = 0.0
    used_keys: list[str] = []
    for power_key in power_keys:
        key_kwh = None
        key_source = None
        for stream in streams:
            stream_arg = None if stream == "default" else stream
            series = _history_metric_series(
                run,
                metric_key=power_key,
                step_key=time_key,
                samples=samples,
                stream=stream_arg,
            )

            if not series and time_key != "_timestamp":
                series = _history_metric_series(
                    run,
                    metric_key=power_key,
                    step_key="_timestamp",
                    samples=samples,
                    stream=stream_arg,
                )

            key_kwh = _integrate_power_series_kwh(series)
            if key_kwh is not None:
                key_source = f"{stream}:{power_key}"
                break

        if key_kwh is None or key_source is None:
            continue
        total_kwh += key_kwh
        used_keys.append(key_source)

    if not used_keys or total_kwh <= 0:
        return None
    return EnergyInfo(kwh=total_kwh, source="+".join(used_keys))


def _best_from_step_rows(
    *,
    run_name: str,
    dataset: str,
    gpu_hours: float,
    runtime_source: str,
    energy_kwh: float | None,
    energy_source: str,
    by_step: dict[float, dict[str, float]],
) -> BestValPoint | None:
    latest_params: float | None = None
    best: BestValPoint | None = None

    for step in sorted(by_step):
        row = by_step[step]
        if "params" in row:
            latest_params = row["params"]

        val_accuracy = row.get("val")
        test_accuracy = row.get("test")
        if val_accuracy is None or test_accuracy is None or latest_params is None:
            continue

        if best is None or val_accuracy > best.val_accuracy:
            best = BestValPoint(
                run_name=run_name,
                dataset=dataset,
                epoch=step,
                val_accuracy=val_accuracy,
                test_accuracy=test_accuracy,
                parameters=latest_params,
                gpu_hours=gpu_hours,
                runtime_source=runtime_source,
                energy_kwh=energy_kwh,
                energy_source=energy_source,
            )

    return best


def _best_validation_point_from_history(
    run,
    *,
    dataset: str,
    gpu_hours: float,
    runtime_source: str,
    energy_kwh: float | None,
    energy_source: str,
    val_key: str,
    test_key: str,
    param_key: str,
    step_key: str,
    samples: int,
) -> BestValPoint | None:
    by_step: dict[float, dict[str, float]] = defaultdict(dict)
    series = {
        "val": _history_metric_series(
            run,
            metric_key=val_key,
            step_key=step_key,
            samples=samples,
        ),
        "test": _history_metric_series(
            run,
            metric_key=test_key,
            step_key=step_key,
            samples=samples,
        ),
        "params": _history_metric_series(
            run,
            metric_key=param_key,
            step_key=step_key,
            samples=samples,
        ),
    }
    if not series["val"] or not series["test"] or not series["params"]:
        return None

    for name, values in series.items():
        for step, value in values:
            by_step[step][name] = value

    if not by_step:
        return None

    return _best_from_step_rows(
        run_name=_run_display_name(run),
        dataset=dataset,
        gpu_hours=gpu_hours,
        runtime_source=runtime_source,
        energy_kwh=energy_kwh,
        energy_source=energy_source,
        by_step=by_step,
    )


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
        run_name = _run_display_name(run)
        if not (
            isinstance(experiment_name, str)
            and experiment_name.startswith(experiment_name_prefix)
        ) and not run_name.startswith(experiment_name_prefix):
            return False

    return True


def _dataset_for_run(run, dataset_key: str, datasets: list[str]) -> str | None:
    dataset = _lookup(run.config, dataset_key)
    if dataset in datasets:
        return str(dataset)

    for tag in getattr(run, "tags", []) or []:
        if not isinstance(tag, str) or not tag.startswith("dataset:"):
            continue
        tagged_dataset = tag.split(":", 1)[1]
        if tagged_dataset in datasets:
            return tagged_dataset

    run_name = _run_display_name(run)
    for candidate in datasets:
        if f"_{candidate}_" in run_name or run_name.endswith(f"_{candidate}"):
            return candidate
    return None


def _scan_metric_series(
    run,
    *,
    metric_key: str,
    step_key: str,
    page_size: int,
    max_rows: int,
    fallback_samples: int,
) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    scanned = 0
    scan_keys = [metric_key] if step_key == "_step" else [metric_key, step_key]

    try:
        for row in run.scan_history(keys=scan_keys, page_size=page_size):
            scanned += 1
            if max_rows and scanned > max_rows:
                break

            value = _to_float(row.get(metric_key))
            step = _to_float(row.get(step_key))
            if value is None or step is None:
                continue
            rows.append((step, value))
    except Exception:
        return _history_metric_series(
            run,
            metric_key=metric_key,
            step_key=step_key,
            samples=fallback_samples,
        )

    if rows or step_key == "_step":
        return rows

    try:
        for row in run.scan_history(keys=[metric_key], page_size=page_size):
            scanned += 1
            if max_rows and scanned > max_rows:
                break

            value = _to_float(row.get(metric_key))
            step = _to_float(row.get("_step"))
            if value is None or step is None:
                continue
            rows.append((step, value))
    except Exception:
        return _history_metric_series(
            run,
            metric_key=metric_key,
            step_key=step_key,
            samples=fallback_samples,
        )

    return rows


def _best_validation_point(
    run,
    *,
    dataset: str,
    gpu_hours: float,
    runtime_source: str,
    energy_kwh: float | None,
    energy_source: str,
    val_key: str,
    test_key: str,
    param_key: str,
    step_key: str,
    page_size: int,
    max_rows: int,
    fallback_samples: int,
    history_backend: str,
) -> BestValPoint | None:
    if history_backend in {"history", "auto"}:
        point = _best_validation_point_from_history(
            run,
            dataset=dataset,
            gpu_hours=gpu_hours,
            runtime_source=runtime_source,
            energy_kwh=energy_kwh,
            energy_source=energy_source,
            val_key=val_key,
            test_key=test_key,
            param_key=param_key,
            step_key=step_key,
            samples=fallback_samples,
        )
        if point is not None or history_backend == "history":
            return point

    series = {
        "val": _scan_metric_series(
            run,
            metric_key=val_key,
            step_key=step_key,
            page_size=page_size,
            max_rows=max_rows,
            fallback_samples=fallback_samples,
        ),
        "test": _scan_metric_series(
            run,
            metric_key=test_key,
            step_key=step_key,
            page_size=page_size,
            max_rows=max_rows,
            fallback_samples=fallback_samples,
        ),
        "params": _scan_metric_series(
            run,
            metric_key=param_key,
            step_key=step_key,
            page_size=page_size,
            max_rows=max_rows,
            fallback_samples=fallback_samples,
        ),
    }
    if not series["val"] or not series["test"] or not series["params"]:
        return None

    by_step: dict[float, dict[str, float]] = defaultdict(dict)
    for name, values in series.items():
        for step, value in values:
            by_step[step][name] = value

    return _best_from_step_rows(
        run_name=_run_display_name(run),
        dataset=dataset,
        gpu_hours=gpu_hours,
        runtime_source=runtime_source,
        energy_kwh=energy_kwh,
        energy_source=energy_source,
        by_step=by_step,
    )


def _summary_float(run, key: str) -> float | None:
    value = None
    summary = getattr(run, "summary", {}) or {}
    if hasattr(summary, "get"):
        value = summary.get(key)
    if value is None and hasattr(summary, "_json_dict"):
        value = summary._json_dict.get(key)
    return _to_float(value)


def _runtime_info(
    run,
    *,
    runtime_source: str,
    runtime_key: str,
    time_metric_prefix: str,
    history_samples: int,
    gpus_per_run: float,
) -> RuntimeInfo | None:
    candidates: list[tuple[str, float]] = []

    if runtime_source in {"time-metrics", "auto", "max"}:
        seconds = _runtime_seconds_from_time_metrics(
            run,
            time_metric_prefix=time_metric_prefix,
            samples=history_samples,
        )
        if seconds is not None:
            candidates.append((f"time_metrics:{time_metric_prefix}", seconds))

    if runtime_source in {"wandb-runtime", "auto", "max"}:
        seconds = _summary_float(run, runtime_key)
        if seconds is None and runtime_key != "_runtime":
            seconds = _summary_float(run, "_runtime")
        if seconds is not None:
            candidates.append((runtime_key, seconds))

    if runtime_source in {"timestamps", "auto", "max"}:
        seconds = _runtime_seconds_from_timestamps(run)
        if seconds is not None:
            candidates.append(("timestamps", seconds))

    if not candidates:
        return None

    if runtime_source == "max":
        source, seconds = max(candidates, key=lambda item: item[1])
        source = f"max:{source}"
    else:
        source, seconds = candidates[0]

    return RuntimeInfo(
        seconds=seconds,
        gpu_hours=seconds * gpus_per_run / 3600.0,
        source=source,
    )


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

    array = [float(value) for value in values]
    if as_percent:
        array = [value * 100.0 for value in array]
    array = [value / scale for value in array]

    mean = statistics.mean(array)
    std = statistics.stdev(array) if len(array) > 1 else 0.0
    pm = "\u00b1"
    if as_percent:
        return f"{mean:.{decimals}f}% {pm} {std:.{decimals}f}"
    return f"{mean:.{decimals}f}{suffix} {pm} {std:.{decimals}f}{suffix}"


def _markdown_row(label: str, values: list[str]) -> str:
    return "| " + " | ".join([label, *values]) + " |"


def _dataset_headers(datasets: list[str], labels: dict[str, str]) -> list[str]:
    return [labels.get(dataset, dataset) for dataset in datasets]


def _parse_dataset_labels(raw_labels: list[str]) -> dict[str, str]:
    labels = dict(DEFAULT_DATASET_LABELS)
    for raw in raw_labels:
        if "=" not in raw:
            raise ValueError(
                f"Invalid dataset label {raw!r}. Expected format dataset=Label."
            )
        dataset, label = raw.split("=", 1)
        labels[dataset.strip()] = label.strip()
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export W&B tables by taking each run at the epoch where validation "
            "accuracy is maximal."
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
        "--dataset-label",
        action="append",
        default=[],
        help="Optional display label override, repeat as dataset=Label.",
    )
    parser.add_argument(
        "--dataset-key",
        default="dataset.name",
        help="Config key used to identify the dataset for each run.",
    )
    parser.add_argument(
        "--val-key",
        default="training/val accuracy",
        help="History metric used to select the best epoch.",
    )
    parser.add_argument(
        "--test-key",
        default="training/test accuracy",
        help="History metric reported at the best validation epoch.",
    )
    parser.add_argument(
        "--param-key",
        default="complexity/nb of parameters",
        help="History metric reported at the best validation epoch.",
    )
    parser.add_argument(
        "--step-key",
        default="epoch",
        help="History x-axis key used to align validation, test, and parameters.",
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
            "for example growth.initialization_strategy=local."
        ),
    )
    parser.add_argument(
        "--experiment-name-prefix",
        default=None,
        help="Optional prefix filter on config key experiment.name or W&B run name.",
    )
    parser.add_argument(
        "--state",
        default="finished",
        help="W&B run state filter. Use 'any' to disable state filtering.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="W&B API timeout in seconds.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="W&B history scan page size.",
    )
    parser.add_argument(
        "--max-history-rows",
        type=int,
        default=0,
        help="Optional max rows scanned per metric per run. Default: no cap.",
    )
    parser.add_argument(
        "--history-samples",
        type=int,
        default=100000,
        help=(
            "Samples requested from W&B history. "
            "Default: 100000."
        ),
    )
    parser.add_argument(
        "--history-backend",
        choices=("history", "scan", "auto"),
        default="history",
        help=(
            "Backend used to read W&B history. Default: history. "
            "Use scan only if W&B scan_history works in the project."
        ),
    )
    parser.add_argument(
        "--accuracy-decimals",
        type=int,
        default=1,
        help="Decimals for the test accuracy table.",
    )
    parser.add_argument(
        "--params-decimals",
        type=int,
        default=1,
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
        help="Suffix appended to formatted parameter values.",
    )
    parser.add_argument(
        "--runtime-key",
        default="_runtime",
        help="W&B summary key containing total run duration in seconds.",
    )
    parser.add_argument(
        "--runtime-source",
        choices=("time-metrics", "wandb-runtime", "timestamps", "auto", "max"),
        default="wandb-runtime",
        help=(
            "Runtime source for GPU-hours. Default: wandb-runtime, equivalent "
            "to ${summary:_runtime} / 3600 for one GPU."
        ),
    )
    parser.add_argument(
        "--time-metric-prefix",
        default="time/",
        help="Metric prefix summed when --runtime-source uses time-metrics.",
    )
    parser.add_argument(
        "--gpus-per-run",
        type=float,
        default=1.0,
        help="GPU count reserved by each run. Default: 1.",
    )
    parser.add_argument(
        "--runtime-decimals",
        type=int,
        default=1,
        help="Decimals for the GPU-hours table.",
    )
    parser.add_argument(
        "--energy-source",
        choices=("wandb-gpu-power", "none"),
        default="wandb-gpu-power",
        help=(
            "Source for GPU consumption in kWh. Default: wandb-gpu-power, "
            "which integrates W&B system GPU powerWatts metrics over time."
        ),
    )
    parser.add_argument(
        "--gpu-power-key",
        action="append",
        default=[],
        help=(
            "Explicit W&B history key for GPU power in watts. Repeat if needed. "
            "Default: auto-detect system GPU powerWatts keys."
        ),
    )
    parser.add_argument(
        "--energy-time-key",
        default="_runtime",
        help=(
            "History time key in seconds used to integrate GPU power. "
            "Default: _runtime, with _timestamp fallback."
        ),
    )
    parser.add_argument(
        "--energy-history-streams",
        nargs="+",
        default=["system", "events", "default"],
        help=(
            "W&B history streams searched for GPU power metrics. "
            "Default: system events default."
        ),
    )
    parser.add_argument(
        "--energy-key-discovery-samples",
        type=int,
        default=1000,
        help="History samples used to discover W&B GPU power keys. Default: 1000.",
    )
    parser.add_argument(
        "--energy-decimals",
        type=int,
        default=3,
        help="Decimals for the GPU consumption kWh table.",
    )
    parser.add_argument(
        "--list-gpu-power-keys",
        action="store_true",
        help="List detected W&B GPU power keys for matching runs and exit.",
    )
    parser.add_argument(
        "--list-history-keys",
        action="store_true",
        help="List sampled W&B history keys for matching runs and exit.",
    )
    parser.add_argument(
        "--show-runs",
        action="store_true",
        help="Print the selected best epoch for each included run.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress logs printed to stderr.",
    )
    args = parser.parse_args()

    config_filters = [_parse_filter(raw) for raw in args.config_filter]
    dataset_labels = _parse_dataset_labels(args.dataset_label)
    wandb = _load_wandb()
    api = wandb.Api(timeout=args.timeout)

    filters = None if args.state.lower() == "any" else {"state": args.state}
    runs = api.runs(f"{args.entity}/{args.project}", filters=filters)

    if args.list_gpu_power_keys or args.list_history_keys:
        for run in runs:
            run_name = _run_display_name(run)
            if not _matches_filters(run, config_filters, args.experiment_name_prefix):
                continue
            dataset = _dataset_for_run(run, args.dataset_key, args.datasets)
            if dataset is None:
                continue
            if args.list_history_keys:
                chunks = []
                for stream in args.energy_history_streams:
                    stream_arg = None if stream == "default" else stream
                    keys = _sample_history_keys(
                        run,
                        samples=args.energy_key_discovery_samples,
                        stream=stream_arg,
                    )
                    chunks.append(
                        f"[{stream}] {', '.join(keys) if keys else '-'}"
                    )
                print(f"{run_name} ({dataset}): " + " | ".join(chunks))
                continue
            else:
                keys = _resolve_gpu_power_keys(
                    run,
                    args.gpu_power_key,
                    discovery_samples=args.energy_key_discovery_samples,
                    streams=args.energy_history_streams,
                )
            print(f"{run_name} ({dataset}): {', '.join(keys) if keys else '-'}")
        return

    grouped: dict[str, DatasetStats] = defaultdict(
        lambda: DatasetStats(accuracy=[], params=[], gpu_hours=[], energy_kwh=[])
    )
    points: list[BestValPoint] = []
    skipped: list[tuple[str, str]] = []
    missing_energy: list[str] = []
    matched_runs = 0

    for run in runs:
        run_name = _run_display_name(run)
        if not _matches_filters(run, config_filters, args.experiment_name_prefix):
            continue

        dataset = _dataset_for_run(run, args.dataset_key, args.datasets)
        if dataset is None:
            skipped.append((run_name, "dataset_not_matched"))
            continue

        runtime = _runtime_info(
            run,
            runtime_source=args.runtime_source,
            runtime_key=args.runtime_key,
            time_metric_prefix=args.time_metric_prefix,
            history_samples=args.history_samples,
            gpus_per_run=args.gpus_per_run,
        )
        if runtime is None:
            skipped.append((run_name, f"missing_runtime:{args.runtime_source}"))
            continue

        energy = None
        if args.energy_source == "wandb-gpu-power":
            energy = _energy_from_gpu_power_metrics(
                run,
                explicit_power_keys=args.gpu_power_key,
                samples=args.history_samples,
                discovery_samples=args.energy_key_discovery_samples,
                time_key=args.energy_time_key,
                streams=args.energy_history_streams,
            )
            if energy is None:
                missing_energy.append(run_name)

        matched_runs += 1
        if not args.quiet:
            energy_status = (
                f", energy={energy.kwh:.4f} kWh from {energy.source}"
                if energy is not None
                else ", energy=missing"
            )
            print(
                "Processing run "
                f"{matched_runs}: {run_name} ({dataset}, "
                f"runtime={runtime.gpu_hours:.3f} GPUh from {runtime.source}"
                f"{energy_status})",
                file=sys.stderr,
                flush=True,
            )

        point = _best_validation_point(
            run,
            dataset=dataset,
            gpu_hours=runtime.gpu_hours,
            runtime_source=runtime.source,
            energy_kwh=energy.kwh if energy is not None else None,
            energy_source=energy.source if energy is not None else "missing",
            val_key=args.val_key,
            test_key=args.test_key,
            param_key=args.param_key,
            step_key=args.step_key,
            page_size=args.page_size,
            max_rows=args.max_history_rows,
            fallback_samples=args.history_samples,
            history_backend=args.history_backend,
        )
        if point is None:
            skipped.append((run_name, "missing_best_validation_history"))
            continue

        grouped[dataset].accuracy.append(point.test_accuracy)
        grouped[dataset].params.append(point.parameters)
        grouped[dataset].gpu_hours.append(point.gpu_hours)
        if point.energy_kwh is not None:
            grouped[dataset].energy_kwh.append(point.energy_kwh)
        points.append(point)

    headers = _dataset_headers(args.datasets, dataset_labels)
    header = _markdown_row("Method", headers)
    separator = _markdown_row("---", ["---"] * len(args.datasets))

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
    runtime_cells = [
        _format_mean_std(
            grouped[name].gpu_hours,
            scale=1.0,
            suffix="",
            decimals=args.runtime_decimals,
            as_percent=False,
        )
        for name in args.datasets
    ]
    energy_cells = [
        _format_mean_std(
            grouped[name].energy_kwh,
            scale=1.0,
            suffix="",
            decimals=args.energy_decimals,
            as_percent=False,
        )
        for name in args.datasets
    ]
    count_cells = [str(len(grouped[name].params)) for name in args.datasets]
    energy_count_cells = [str(len(grouped[name].energy_kwh)) for name in args.datasets]

    print("Test accuracy at maximal validation accuracy")
    print(header)
    print(separator)
    print(_markdown_row(args.method_label, acc_cells))
    print()
    print("Number of parameters at maximal validation accuracy")
    print(header)
    print(separator)
    print(_markdown_row(args.method_label, param_cells))
    print()
    print("Runtime in GPU-hours from W&B summary _runtime / 3600")
    print(header)
    print(separator)
    print(_markdown_row(args.method_label, runtime_cells))
    print()
    print("GPU consumption in kWh from integrated W&B GPU power")
    print(header)
    print(separator)
    print(_markdown_row(args.method_label, energy_cells))
    print()
    print("Included finished run counts")
    print(header)
    print(separator)
    print(_markdown_row(args.method_label, count_cells))
    print()
    print("GPU consumption run counts")
    print(header)
    print(separator)
    print(_markdown_row(args.method_label, energy_count_cells))

    if args.show_runs:
        print()
        print("Included runs")
        print(
            "| Dataset | Run | Best epoch | Val acc | Test acc | Params | "
            "GPU-hours | Runtime source | GPU kWh | Energy source |"
        )
        print("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |")
        for point in sorted(points, key=lambda item: (item.dataset, item.run_name)):
            print(
                "| "
                + " | ".join(
                    [
                        dataset_labels.get(point.dataset, point.dataset),
                        point.run_name,
                        f"{point.epoch:g}",
                        f"{point.val_accuracy:.6g}",
                        f"{point.test_accuracy:.6g}",
                        f"{point.parameters:.0f}",
                        f"{point.gpu_hours:.6g}",
                        point.runtime_source,
                        (
                            f"{point.energy_kwh:.6g}"
                            if point.energy_kwh is not None
                            else "-"
                        ),
                        point.energy_source,
                    ]
                )
                + " |"
            )

    if missing_energy:
        print()
        print("Runs without GPU power history for kWh")
        for run_name in missing_energy:
            print(f"- {run_name}")

    if skipped:
        print()
        print("Skipped matching finished runs")
        for run_name, reason in skipped:
            print(f"- {run_name}: {reason}")


if __name__ == "__main__":
    main()
