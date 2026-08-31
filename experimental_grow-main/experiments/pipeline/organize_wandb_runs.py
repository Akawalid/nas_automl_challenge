#!/usr/bin/env python3
"""Backfill organization metadata for existing pipeline W&B runs.

This script does not rerun experiments and does not edit metrics or artifacts.
It only adds/updates W&B run metadata such as tags, group, job_type, and notes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

BENCHMARK_PROJECT = "demeter-local-base-multidataset"
CIFAR_PROJECT = "demeter-cifar-ablation-suite"
INIT_SCALING_PROJECT = "demeter-init-scaling-ablation"
INIT_SCALING_LAST_MULTNIST_TAG = "selection:init_scaling_last18_multnist"

BENCHMARK_PREFIX = "local_base_multi_dataset"
BENCHMARK_DATASETS = {
    "multnist",
    "cifartile",
    "gutenberg",
    "geoclassing",
    "chesseract",
}

CIFAR_DATASETS = {"cifar10", "cifar100"}
CIFAR_ABLATIONS = (
    "cifar_global_scheduler_350",
    "cifar_growth_label_smoothing",
    "cifar_post_growth_scheduler",
    "cifar_noisy_pre_grad_005",
    "cifar_noisy_pre_grad_010",
    "cifar_variance_transfer",
    "cifar_linear_warmup",
    "cifar_init_ablation",
)
MULTNIST_ABLATIONS = (
    "global_scheduler_350",
    "growth_label_smoothing",
    "post_growth_scheduler",
    "noisy_pre_activities_grad",
    "variance_transfer",
    "linear_warmup",
    "init_scaling_ablation",
    "base_compare",
)

INIT_STRATEGIES = ("init_scaling_ablation", "local")


@dataclass(frozen=True)
class RunMetadata:
    suite: str
    area: str
    dataset: str
    ablation: str
    init_strategy: str
    seed: str
    group: str
    job_type: str
    report_name: str

    def tags(self) -> list[str]:
        return [
            "organized:demeter_pipeline",
            f"suite:{self.suite}",
            f"area:{self.area}",
            f"dataset:{self.dataset}",
            f"ablation:{self.ablation}",
            f"init:{self.init_strategy}",
            f"seed:{self.seed}",
        ]

    def notes_block(self) -> str:
        return "\n".join(
            [
                "Demeter pipeline organization metadata:",
                f"- suite: {self.suite}",
                f"- area: {self.area}",
                f"- dataset: {self.dataset}",
                f"- ablation: {self.ablation}",
                f"- init strategy: {self.init_strategy}",
                f"- seed: {self.seed}",
                f"- suggested report: {self.report_name}",
            ]
        )


def _split_seed(raw: str) -> tuple[str, str] | None:
    marker = "_seed"
    if marker not in raw:
        return None
    before, seed = raw.rsplit(marker, 1)
    if not seed.isdigit():
        return None
    return before, seed


def _parse_init_dataset(raw: str, datasets: set[str]) -> tuple[str, str] | None:
    for init_strategy in INIT_STRATEGIES:
        prefix = f"{init_strategy}_"
        if not raw.startswith(prefix):
            continue
        dataset = raw[len(prefix) :]
        if dataset in datasets:
            return init_strategy, dataset
    return None


def parse_benchmark_run(name: str) -> RunMetadata | None:
    prefix = f"{BENCHMARK_PREFIX}_"
    if not name.startswith(prefix):
        return None

    split = _split_seed(name[len(prefix) :])
    if split is None:
        return None

    before_seed, seed = split
    parsed = _parse_init_dataset(before_seed, BENCHMARK_DATASETS)
    if parsed is None:
        return None

    init_strategy, dataset = parsed
    ablation = "local_base"
    return RunMetadata(
        suite="benchmark",
        area="local_base_multi_dataset",
        dataset=dataset,
        ablation=ablation,
        init_strategy=init_strategy,
        seed=seed,
        group=f"benchmark__{dataset}__{ablation}",
        job_type="benchmark_local_base",
        report_name=f"Benchmark - {dataset}",
    )


def parse_cifar_run(name: str) -> RunMetadata | None:
    for ablation in CIFAR_ABLATIONS:
        prefix = f"{ablation}_"
        if not name.startswith(prefix):
            continue

        split = _split_seed(name[len(prefix) :])
        if split is None:
            return None

        before_seed, seed = split
        parsed = _parse_init_dataset(before_seed, CIFAR_DATASETS)
        if parsed is None:
            return None

        init_strategy, dataset = parsed
        return RunMetadata(
            suite="cifar_ablation",
            area=ablation,
            dataset=dataset,
            ablation=ablation,
            init_strategy=init_strategy,
            seed=seed,
            group=f"cifar__{dataset}__{ablation}",
            job_type="cifar_ablation",
            report_name=f"{dataset.upper()} - Ablation Suite",
        )

    return None


def _lookup_config(run, key: str):
    current = getattr(run, "config", {}) or {}
    if key in current:
        return current[key]
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def metadata_for_init_scaling_multnist(run) -> RunMetadata:
    run_name = _run_display_name(run)
    parsed_seed = _split_seed(run_name)
    seed = str(_lookup_config(run, "experiment.seed") or (parsed_seed[1] if parsed_seed else "unknown"))
    init_strategy = str(
        _lookup_config(run, "growth.initialization_strategy")
        or ("init_scaling_ablation" if "init_scaling_ablation" in run_name else "local")
    )
    dataset = "multnist"
    ablation = infer_multnist_ablation(run_name)
    return RunMetadata(
        suite="init_scaling_ablation",
        area=ablation,
        dataset=dataset,
        ablation=ablation,
        init_strategy=init_strategy,
        seed=seed,
        group=f"multnist__{ablation}",
        job_type="multnist_ablation",
        report_name="MultiNIST - Init Scaling Ablation",
    )


def infer_multnist_ablation(run_name: str) -> str:
    for ablation in MULTNIST_ABLATIONS:
        if run_name.startswith(f"{ablation}_"):
            return ablation
    return "init_scaling_ablation"


def parse_run(project: str, name: str) -> RunMetadata | None:
    if project == BENCHMARK_PROJECT:
        return parse_benchmark_run(name)
    if project == CIFAR_PROJECT:
        return parse_cifar_run(name)
    return parse_benchmark_run(name) or parse_cifar_run(name)


def _merge_tags(existing_tags: Iterable[str], new_tags: Iterable[str]) -> list[str]:
    managed_prefixes = (
        "suite:",
        "area:",
        "dataset:",
        "ablation:",
        "init:",
        "seed:",
        "organized:",
    )
    merged: list[str] = []
    seen: set[str] = set()
    for tag in existing_tags:
        if tag.startswith(managed_prefixes):
            continue
        if tag in seen:
            continue
        seen.add(tag)
        merged.append(tag)
    for tag in new_tags:
        if tag in seen:
            continue
        seen.add(tag)
        merged.append(tag)
    return merged


def _append_notes(existing_notes: str | None, block: str) -> str:
    if not existing_notes:
        return block
    if block in existing_notes:
        return existing_notes
    return f"{existing_notes.rstrip()}\n\n{block}"


def _run_display_name(run) -> str:
    return str(getattr(run, "name", None) or getattr(run, "display_name", ""))


def organize_run(
    run,
    metadata: RunMetadata,
    *,
    apply: bool,
    extra_tags: Iterable[str] = (),
) -> None:
    existing_tags = list(getattr(run, "tags", []) or [])
    new_tags = _merge_tags(existing_tags, [*metadata.tags(), *extra_tags])
    notes = _append_notes(getattr(run, "notes", None), metadata.notes_block())

    print(
        "MATCH",
        f"name={_run_display_name(run)!r}",
        f"group={metadata.group!r}",
        f"job_type={metadata.job_type!r}",
        f"tags={[*metadata.tags(), *extra_tags]}",
        sep=" | ",
    )

    if not apply:
        return

    run.tags = new_tags
    run.group = metadata.group
    run.job_type = metadata.job_type
    run.notes = notes
    run.config.update(
        {
            "organization.suite": metadata.suite,
            "organization.area": metadata.area,
            "dataset.name": metadata.dataset,
            "ablation.name": metadata.ablation,
            "growth.initialization_strategy": metadata.init_strategy,
            "experiment.seed": metadata.seed,
            "report.group": metadata.group,
        },
        allow_val_change=True,
    )
    run.update()


def _run_created_at(run):
    value = getattr(run, "created_at", None) or getattr(run, "createdAt", None)
    if value is None:
        return ""
    return str(value)


def organize_latest_init_scaling_multnist_runs(
    runs,
    *,
    count: int,
    apply: bool,
) -> tuple[int, int]:
    ordered_runs = sorted(list(runs), key=_run_created_at, reverse=True)
    selected = set(getattr(run, "id", _run_display_name(run)) for run in ordered_runs[:count])
    matched = 0
    skipped = 0

    for index, run in enumerate(ordered_runs):
        run_id = getattr(run, "id", _run_display_name(run))
        is_selected = run_id in selected
        existing_tags = list(getattr(run, "tags", []) or [])

        if not is_selected:
            skipped += 1
            if INIT_SCALING_LAST_MULTNIST_TAG in existing_tags:
                print(f"CLEAR | name={_run_display_name(run)!r} | tag={INIT_SCALING_LAST_MULTNIST_TAG}")
                if apply:
                    run.tags = [tag for tag in existing_tags if tag != INIT_SCALING_LAST_MULTNIST_TAG]
                    run.update()
            else:
                print(f"SKIP  | name={_run_display_name(run)!r}")
            continue

        matched += 1
        metadata = metadata_for_init_scaling_multnist(run)
        print(f"SELECT_LAST_{count} | rank={index + 1}")
        organize_run(
            run,
            metadata,
            apply=apply,
            extra_tags=[INIT_SCALING_LAST_MULTNIST_TAG],
        )

    return matched, skipped


def _team_name(team) -> str:
    return str(getattr(team, "name", team))


def _candidate_entities(api) -> list[str]:
    viewer = api.viewer
    entities: list[str] = []
    username = getattr(viewer, "username", None)
    if username:
        entities.append(str(username))
    for team in getattr(viewer, "teams", []) or []:
        entities.append(_team_name(team))
    return list(dict.fromkeys(entities))


def _project_exists(api, entity: str, project: str) -> bool:
    try:
        return any(item.name == project for item in api.projects(entity))
    except Exception:
        return False


def _resolve_project_entity(api, requested_entity: str | None, project: str) -> str:
    if requested_entity:
        return requested_entity

    for entity in _candidate_entities(api):
        if _project_exists(api, entity, project):
            return entity

    candidates = ", ".join(_candidate_entities(api)) or "none"
    raise ValueError(
        f"Could not find project {project!r} under any accessible entity. "
        f"Checked: {candidates}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize existing Demeter pipeline W&B runs without rerunning them."
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity/user/team. If omitted, the script searches accessible entities.",
    )
    parser.add_argument(
        "--project",
        action="append",
        default=[],
        help=(
            "W&B project to process. Repeat this flag. "
            "Defaults to benchmark and CIFAR projects."
        ),
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Optional W&B run state filter, for example finished or running.",
    )
    parser.add_argument("--timeout", type=int, default=120, help="W&B API timeout.")
    parser.add_argument(
        "--init-scaling-multnist-last",
        type=int,
        default=None,
        help=(
            "For demeter-init-scaling-ablation, organize only the latest N runs "
            "as MultiNIST runs for the combined ablation report."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist changes. Without this flag the script only prints a dry-run.",
    )
    args = parser.parse_args()

    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - import-time guard
        raise SystemExit(
            "wandb is required for this script. Activate the experiment environment first."
        ) from exc

    projects = args.project or [BENCHMARK_PROJECT, CIFAR_PROJECT]
    api = wandb.Api(timeout=args.timeout)
    filters = {"state": args.state} if args.state else None

    matched = 0
    skipped = 0
    for project in projects:
        print(f"\nProject: {project}")
        try:
            entity = _resolve_project_entity(api, args.entity, project)
            print(f"Entity: {entity}")
            if filters is None:
                runs = api.runs(f"{entity}/{project}")
            else:
                runs = api.runs(f"{entity}/{project}", filters=filters)
            if (
                project == INIT_SCALING_PROJECT
                and args.init_scaling_multnist_last is not None
            ):
                project_matched, project_skipped = organize_latest_init_scaling_multnist_runs(
                    runs,
                    count=args.init_scaling_multnist_last,
                    apply=args.apply,
                )
                matched += project_matched
                skipped += project_skipped
                continue
            for run in runs:
                run_name = _run_display_name(run)
                metadata = parse_run(project, run_name)
                if metadata is None:
                    skipped += 1
                    print(f"SKIP  | name={run_name!r}")
                    continue
                matched += 1
                organize_run(run, metadata, apply=args.apply)
        except ValueError as exc:
            print(f"SKIP PROJECT | {project!r} | {exc}")

    mode = "applied" if args.apply else "dry-run"
    print(f"\nDone ({mode}). Matched: {matched}. Skipped: {skipped}.")
    if not args.apply:
        print("Re-run with --apply to persist these W&B metadata changes.")


if __name__ == "__main__":
    main()
