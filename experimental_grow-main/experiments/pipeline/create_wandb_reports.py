#!/usr/bin/env python3
"""Create or update W&B reports for organized Demeter pipeline runs.

The script assumes organize_wandb_runs.py has already added dataset/ablation tags
and groups to the runs. It modifies reports only when --apply is passed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from organize_wandb_runs import (
    BENCHMARK_DATASETS,
    BENCHMARK_PROJECT,
    CIFAR_ABLATIONS,
    CIFAR_DATASETS,
    CIFAR_PROJECT,
    INIT_SCALING_LAST_MULTNIST_TAG,
    INIT_SCALING_PROJECT,
    _resolve_project_entity,
)


BENCHMARK_DATASET_ORDER = [
    "multnist",
    "cifartile",
    "gutenberg",
    "geoclassing",
    "chesseract",
]
CIFAR_DATASET_ORDER = ["cifar10", "cifar100"]
CIFAR_ABLATION_ORDER = [
    "cifar_init_ablation",
    "cifar_linear_warmup",
    "cifar_post_growth_scheduler",
    "cifar_growth_label_smoothing",
    "cifar_noisy_pre_grad_005",
    "cifar_noisy_pre_grad_010",
    "cifar_variance_transfer",
    "cifar_global_scheduler_350",
]


@dataclass(frozen=True)
class DatasetSource:
    dataset: str
    project: str
    suite: str
    tags: list[str]
    display_name: str
    runset_name: str
    table_groupby: list[str]


def _tags_filter(tags: list[str]) -> dict[str, Any]:
    return {
        "$and": [{"tags": tag} for tag in tags]
    }


def _tags_filter_expr(tags: list[str]) -> str:
    return " and ".join(f"Tags() == {tag!r}" for tag in tags)


def _dataset_tags(dataset: str, suite: str) -> list[str]:
    return [
        "organized:demeter_pipeline",
        f"suite:{suite}",
        f"dataset:{dataset}",
    ]


def _dataset_filter(dataset: str, suite: str) -> dict[str, Any]:
    return _tags_filter(_dataset_tags(dataset, suite))


def _source_filter(source: DatasetSource) -> dict[str, Any]:
    return _tags_filter(source.tags)


def _suite_tags(suite: str) -> list[str]:
    return [
        "organized:demeter_pipeline",
        f"suite:{suite}",
    ]


def _suite_filter(suite: str) -> dict[str, Any]:
    return _tags_filter(_suite_tags(suite))


def _ablation_tags(ablation: str, suite: str) -> list[str]:
    return [
        "organized:demeter_pipeline",
        f"suite:{suite}",
        f"ablation:{ablation}",
    ]


def _ablation_filter(ablation: str, suite: str) -> dict[str, Any]:
    return _tags_filter(_ablation_tags(ablation, suite))


def _count_runs(api, entity: str, project: str, filters: dict[str, Any]) -> int:
    return sum(1 for _ in api.runs(f"{entity}/{project}", filters=filters))


def _block(wr, name: str, text: str):
    if name == "P":
        cls = getattr(wr, "P", None) or getattr(wr, "MarkdownBlock")
        try:
            return cls(text=text)
        except TypeError:
            return cls(text)

    cls = getattr(wr, name, None)
    if cls is None and name == "H2":
        cls = getattr(wr, "H1")
    if cls is None:
        cls = getattr(wr, "MarkdownBlock")
        prefix = (
            "# " if name == "H1"
            else "## " if name == "H2"
            else ""
        )
        return cls(text=f"{prefix}{text}")

    try:
        return cls(text=text)
    except TypeError:
        return cls(text)


def _line_plot(wr, *, title: str, x: str, y: str):
    cls = getattr(wr, "LinePlot")
    try:
        return cls(title=title, x=x, y=[y])
    except TypeError:
        return cls(title=title, x=x, y=y)


def _filter_tags_from_api_filter(filters: dict[str, Any]) -> list[str]:
    return [item["tags"] for item in filters.get("$and", []) if "tags" in item]


def _runset(
    wr,
    *,
    entity: str,
    project: str,
    name: str,
    filters: dict[str, Any],
    groupby: list[str] | None = None,
):
    runset_cls = getattr(wr, "Runset", None) or getattr(wr, "RunSet")
    if runset_cls is None:
        raise RuntimeError(
            "This wandb-workspaces version does not expose Runset/RunSet. "
            "Upgrade with: python -m pip install -U wandb-workspaces"
        )
    tags = _filter_tags_from_api_filter(filters)
    kwargs = {
        "entity": entity,
        "project": project,
        "name": name,
        "filters": _tags_filter_expr(tags),
    }
    if groupby:
        kwargs["groupby"] = groupby
    try:
        return runset_cls(**kwargs)
    except TypeError:
        kwargs.pop("groupby", None)
        return runset_cls(**kwargs)


def _panel_grid(wr, *, runset, panels):
    try:
        return wr.PanelGrid(runsets=[runset], panels=panels)
    except TypeError:
        return wr.PanelGrid(panels=panels)


def _report(wr, *, entity: str, project: str, title: str, description: str):
    try:
        return wr.Report(
            entity=entity,
            project=project,
            title=title,
            description=description,
        )
    except TypeError:
        report = wr.Report(project=project, title=title, description=description)
        if hasattr(report, "entity"):
            report.entity = entity
        return report


def _metric_panels(wr):
    return [
        _line_plot(
            wr,
            title="Test accuracy",
            x="epoch",
            y="training/test accuracy",
        ),
        _line_plot(
            wr,
            title="Validation accuracy",
            x="epoch",
            y="training/val accuracy",
        ),
        _line_plot(
            wr,
            title="Test loss",
            x="epoch",
            y="training/test loss",
        ),
        _line_plot(
            wr,
            title="Number of parameters",
            x="epoch",
            y="complexity/nb of parameters",
        ),
        _line_plot(
            wr,
            title="Growth neurons",
            x="growth step",
            y="growth/neurons",
        ),
        _line_plot(
            wr,
            title="Growth epochs",
            x="growth step",
            y="growth/epochs",
        ),
    ]


def _suite_defaults(
    kind: str,
    project_override: str | None,
    *,
    include_init_scaling_multnist: bool,
):
    if kind == "cifar":
        datasets = [name for name in CIFAR_DATASET_ORDER if name in CIFAR_DATASETS]
        if include_init_scaling_multnist:
            datasets.append("multnist")
        return {
            "project": project_override or CIFAR_PROJECT,
            "suite": "cifar_ablation",
            "title": "CIFAR Ablation Suite",
            "datasets": datasets,
            "ablations": [
                name for name in CIFAR_ABLATION_ORDER if name in set(CIFAR_ABLATIONS)
            ],
            "intro": (
                "Automatically generated report for Demeter ablation runs. "
                "Runs are grouped by dataset, ablation, initialization strategy, and seed."
            ),
        }
    if kind == "benchmark":
        return {
            "project": project_override or BENCHMARK_PROJECT,
            "suite": "benchmark",
            "title": "Benchmark Local Base Multi-Dataset",
            "datasets": [
                name for name in BENCHMARK_DATASET_ORDER if name in BENCHMARK_DATASETS
            ],
            "ablations": ["local_base"],
            "intro": (
                "Automatically generated report for local base benchmark runs. "
                "Runs are grouped by dataset and seed."
            ),
        }
    raise ValueError(f"Unsupported report kind: {kind}")


def _dataset_sources(
    *,
    kind: str,
    project: str,
    suite: str,
    datasets: list[str],
    include_init_scaling_multnist: bool,
) -> list[DatasetSource]:
    sources: list[DatasetSource] = []
    for dataset in datasets:
        if (
            kind == "cifar"
            and include_init_scaling_multnist
            and dataset == "multnist"
        ):
            sources.append(
                DatasetSource(
                    dataset=dataset,
                    project=INIT_SCALING_PROJECT,
                    suite="init_scaling_ablation",
                    tags=[
                        "organized:demeter_pipeline",
                        "dataset:multnist",
                        INIT_SCALING_LAST_MULTNIST_TAG,
                    ],
                    display_name="MultiNIST",
                    runset_name="MultiNIST init-scaling runs",
                    table_groupby=["run.group"],
                )
            )
            continue

        sources.append(
            DatasetSource(
                dataset=dataset,
                project=project,
                suite=suite,
                tags=_dataset_tags(dataset, suite),
                display_name=dataset.upper(),
                runset_name=f"{dataset.upper()} runs",
                table_groupby=["run.group"],
            )
        )
    return sources


def _build_report_blocks(
    wr,
    *,
    entity: str,
    title: str,
    intro: str,
    sources: list[DatasetSource],
) -> list[Any]:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    blocks: list[Any] = [
        _block(wr, "H1", title),
        _block(wr, "P", intro),
        _block(
            wr,
            "P",
            f"Generated from W&B entity `{entity}` at {generated_at}.",
        ),
        _block(wr, "H2", "By Dataset"),
    ]

    for source in sources:
        filters = _source_filter(source)
        blocks.extend(
            [
                _block(wr, "H2", source.display_name),
                _block(
                    wr,
                    "P",
                    (
                        f"Dataset-level plots filtered by `dataset:{source.dataset}` "
                        f"from project `{source.project}`. "
                        "The organized runs table for this dataset is grouped by "
                        "run group; each group corresponds to one ablation."
                    ),
                ),
                _panel_grid(
                    wr,
                    runset=_runset(
                        wr,
                        entity=entity,
                        project=source.project,
                        name=source.runset_name,
                        filters=filters,
                        groupby=source.table_groupby,
                    ),
                    panels=_metric_panels(wr),
                ),
            ]
        )

    return blocks


def _save_report(
    *,
    entity: str,
    project: str,
    title: str,
    intro: str,
    sources: list[DatasetSource],
    report_url: str | None,
    update_title: bool,
) -> str:
    try:
        try:
            import wandb_workspaces.reports.v2 as wr
        except ImportError:
            import wandb_workspaces.reports as wr
    except ImportError as exc:
        raise SystemExit(
            "Programmatic W&B reports require wandb-workspaces. Install it in "
            "the active environment with: python -m pip install wandb-workspaces"
        ) from exc

    if report_url:
        report = _load_existing_report(wr, report_url)
        existing_title = getattr(report, "title", None)
        final_title = title if update_title or not existing_title else existing_title
    else:
        report = None
        final_title = title

    blocks = _build_report_blocks(
        wr,
        entity=entity,
        title=final_title,
        intro=intro,
        sources=sources,
    )

    if report is None:
        try:
            report = wr.Report(
                entity=entity,
                project=project,
                title=final_title,
                description=intro,
                blocks=blocks,
            )
        except TypeError:
            report = _report(
                wr,
                entity=entity,
                project=project,
                title=final_title,
                description=intro,
            )

    if update_title and hasattr(report, "title"):
        report.title = title
    if hasattr(report, "description"):
        report.description = intro
    report.blocks = blocks
    report.save()
    return str(getattr(report, "url", report_url or ""))


def _load_existing_report(wr, report_url: str):
    report_cls = getattr(wr, "Report", None)
    if report_cls is None:
        raise RuntimeError(
            "This wandb-workspaces version does not expose Report. "
            "Upgrade with: python -m pip install -U wandb-workspaces"
        )

    load_attempts = []
    if hasattr(report_cls, "from_url"):
        load_attempts.append(lambda: report_cls.from_url(report_url))
    if hasattr(report_cls, "load"):
        load_attempts.append(lambda: report_cls.load(report_url))
    if hasattr(wr, "ReportFromURL"):
        load_attempts.append(lambda: wr.ReportFromURL(report_url))

    errors: list[str] = []
    for load in load_attempts:
        try:
            return load()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    detail = "\n".join(errors) if errors else "No known report URL loader was found."
    raise RuntimeError(
        "Could not load the existing W&B report from --report-url. "
        "Upgrade wandb-workspaces, or create a new report without --report-url.\n"
        f"{detail}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or update W&B reports for organized Demeter pipeline runs."
    )
    parser.add_argument(
        "--kind",
        choices=["cifar", "benchmark"],
        default="cifar",
        help="Report family to create.",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity/user/team. If omitted, the script searches accessible entities.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Override the default W&B project for the selected kind.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Override the report title. With --report-url, omit this to keep the current title.",
    )
    parser.add_argument(
        "--report-url",
        default=None,
        help="Existing W&B report URL to update instead of creating a new report.",
    )
    parser.add_argument(
        "--include-init-scaling-multnist",
        action="store_true",
        help=(
            "For --kind cifar, include selected MultiNIST runs from "
            "demeter-init-scaling-ablation as an extra dataset section."
        ),
    )
    parser.add_argument("--timeout", type=int, default=120, help="W&B API timeout.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create/update the report. Without this flag, only print a dry-run summary.",
    )
    args = parser.parse_args()

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is required for this script. Activate the experiment environment first."
        ) from exc

    defaults = _suite_defaults(
        args.kind,
        args.project,
        include_init_scaling_multnist=args.include_init_scaling_multnist,
    )
    project = defaults["project"]
    suite = defaults["suite"]
    title = args.title or defaults["title"]
    datasets = defaults["datasets"]
    ablations = defaults["ablations"]
    intro = defaults["intro"]

    api = wandb.Api(timeout=args.timeout)
    entity = _resolve_project_entity(api, args.entity, project)
    sources = _dataset_sources(
        kind=args.kind,
        project=project,
        suite=suite,
        datasets=datasets,
        include_init_scaling_multnist=args.include_init_scaling_multnist,
    )

    print(f"Entity: {entity}")
    print(f"Project: {project}")
    print(f"Report title: {title}")
    if args.report_url:
        print(f"Existing report URL: {args.report_url}")
    print(f"Suite: {suite}")
    print("Datasets:")
    for source in sources:
        filters = _source_filter(source)
        count = _count_runs(api, entity, source.project, filters)
        print(f"  - {source.dataset} ({source.project}): {count} run(s)")
    print("Ablations:")
    for ablation in ablations:
        filters = _ablation_filter(ablation, suite)
        count = _count_runs(api, entity, project, filters)
        print(f"  - {ablation}: {count} run(s)")

    if not args.apply:
        action = "update the existing report" if args.report_url else "create the report"
        print(f"\nDry-run only. Re-run with --apply to {action}.")
        return

    url = _save_report(
        entity=entity,
        project=project,
        title=title,
        intro=intro,
        sources=sources,
        report_url=args.report_url,
        update_title=args.title is not None,
    )
    print("\nReport updated." if args.report_url else "\nReport created.")
    if url:
        print(url)


if __name__ == "__main__":
    main()
