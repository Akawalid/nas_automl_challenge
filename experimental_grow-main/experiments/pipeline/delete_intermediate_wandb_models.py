#!/usr/bin/env python3
"""Delete intermediate W&B model artifacts without deleting runs.

The pipeline used to log model artifacts inside every growth step. For each run
this script deletes:

- all model artifact versions whose collection name contains ``_after_growth_``;
- older versions of the final model collections, keeping the ``latest`` alias
  or, if no alias is present, the highest ``vN`` version as a fallback.

It defaults to dry-run mode. Pass ``--delete`` to actually delete artifacts.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


VERSION_RE = re.compile(r":v(\d+)$")


@dataclass(frozen=True)
class DeleteCandidate:
    run_path: str
    artifact_name: str
    artifact_type: str
    aliases: tuple[str, ...]
    reason: str
    size_bytes: int | None


def _load_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is required. Run this in the experiment environment, "
            "for example after `wandb login` on the cluster."
        ) from exc
    return wandb


def _artifact_base_name(artifact: Any) -> str:
    return str(artifact.name).split(":", 1)[0]


def _artifact_version(artifact: Any) -> int:
    match = VERSION_RE.search(str(artifact.name))
    return int(match.group(1)) if match else -1


def _artifact_aliases(artifact: Any) -> tuple[str, ...]:
    aliases = getattr(artifact, "aliases", None) or []
    return tuple(str(alias) for alias in aliases)


def _artifact_size(artifact: Any) -> int | None:
    try:
        size = getattr(artifact, "size", None)
    except Exception:
        return None
    return size if isinstance(size, int) else None


def _format_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unknown"

    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TiB"


def _run_path(run: Any) -> str:
    entity = getattr(run, "entity", None)
    project = getattr(run, "project", None)
    run_id = getattr(run, "id", None)
    if entity and project and run_id:
        return f"{entity}/{project}/{run_id}"
    return str(getattr(run, "path", run_id))


def _iter_runs(api: Any, project_path: str, run_ids: list[str] | None, limit: int | None):
    if run_ids:
        for run_id in run_ids:
            yield api.run(f"{project_path}/{run_id}")
            if limit is not None:
                limit -= 1
                if limit <= 0:
                    return
        return

    count = 0
    for run in api.runs(project_path):
        yield run
        count += 1
        if limit is not None and count >= limit:
            return


def _model_artifacts_by_collection(run: Any) -> dict[str, list[Any]]:
    run_id = str(run.id)
    by_collection: dict[str, list[Any]] = defaultdict(list)

    for artifact in run.logged_artifacts():
        if getattr(artifact, "type", None) != "model":
            continue

        collection_name = _artifact_base_name(artifact)
        if run_id not in collection_name:
            continue

        by_collection[collection_name].append(artifact)

    return by_collection


def _select_candidates(
    run: Any,
    *,
    keep_aliases: set[str],
    delete_after_growth: bool,
) -> list[tuple[Any, DeleteCandidate]]:
    candidates: list[tuple[Any, DeleteCandidate]] = []
    run_path = _run_path(run)

    for collection_name, artifacts in sorted(_model_artifacts_by_collection(run).items()):
        if "_after_growth_" in collection_name:
            if not delete_after_growth:
                continue
            for artifact in sorted(artifacts, key=_artifact_version):
                candidates.append(
                    (
                        artifact,
                        DeleteCandidate(
                            run_path=run_path,
                            artifact_name=str(artifact.name),
                            artifact_type=str(getattr(artifact, "type", "")),
                            aliases=_artifact_aliases(artifact),
                            reason="after_growth model artifact",
                            size_bytes=_artifact_size(artifact),
                        ),
                    )
                )
            continue

        keep_names = {
            str(artifact.name)
            for artifact in artifacts
            if set(_artifact_aliases(artifact)).intersection(keep_aliases)
        }
        if not keep_names and artifacts:
            keep_names.add(str(max(artifacts, key=_artifact_version).name))

        for artifact in sorted(artifacts, key=_artifact_version):
            if str(artifact.name) in keep_names:
                continue
            candidates.append(
                (
                    artifact,
                    DeleteCandidate(
                        run_path=run_path,
                        artifact_name=str(artifact.name),
                        artifact_type=str(getattr(artifact, "type", "")),
                        aliases=_artifact_aliases(artifact),
                        reason="older version of final model collection",
                        size_bytes=_artifact_size(artifact),
                    ),
                )
            )

    return candidates


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete intermediate model artifacts from W&B while keeping runs and "
            "the final model artifact version."
        )
    )
    parser.add_argument(
        "project_path",
        help="W&B project path in the form <entity>/<project>.",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=None,
        help="Restrict to one run id. Repeat this flag for multiple runs.",
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        default=None,
        help="Inspect only the first N runs. Useful for testing.",
    )
    parser.add_argument(
        "--keep-alias",
        action="append",
        default=["latest"],
        help=(
            "Alias to preserve for non-after_growth model collections. "
            "Default: latest. Repeat to preserve extra aliases such as best."
        ),
    )
    parser.add_argument(
        "--keep-after-growth",
        action="store_true",
        help="Do not delete *_after_growth_* model artifacts.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete artifacts. Without this flag the script only prints a dry run.",
    )
    parser.add_argument(
        "--no-delete-aliases",
        action="store_true",
        help="Do not pass delete_aliases=True to W&B artifact.delete().",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    wandb = _load_wandb()
    api = wandb.Api()

    keep_aliases = set(args.keep_alias or [])
    delete_aliases = not args.no_delete_aliases
    delete_after_growth = not args.keep_after_growth

    total_candidates = 0
    total_known_size = 0
    deleted = 0
    failed = 0

    mode = "DELETE" if args.delete else "DRY RUN"
    print(f"{mode}: scanning {args.project_path}")
    if args.run_id:
        print(f"Run ids: {', '.join(args.run_id)}")
    print(f"Keeping aliases on final-model collections: {', '.join(sorted(keep_aliases))}")
    print()

    for run in _iter_runs(api, args.project_path, args.run_id, args.limit_runs):
        candidates = _select_candidates(
            run,
            keep_aliases=keep_aliases,
            delete_after_growth=delete_after_growth,
        )
        if not candidates:
            continue

        print(f"Run {_run_path(run)}")
        for artifact, candidate in candidates:
            total_candidates += 1
            if candidate.size_bytes is not None:
                total_known_size += candidate.size_bytes
            aliases = ",".join(candidate.aliases) if candidate.aliases else "-"
            print(
                "  "
                f"{candidate.artifact_name} "
                f"size={_format_size(candidate.size_bytes)} "
                f"aliases={aliases} "
                f"reason={candidate.reason}"
            )

            if args.delete:
                try:
                    artifact.delete(delete_aliases=delete_aliases)
                    deleted += 1
                except Exception as exc:
                    failed += 1
                    print(f"    ERROR: failed to delete {candidate.artifact_name}: {exc}")
        print()

    print(
        f"Matched {total_candidates} artifact versions "
        f"({ _format_size(total_known_size) } known size)."
    )
    if args.delete:
        print(f"Deleted {deleted}; failed {failed}.")
        return 1 if failed else 0

    print("No artifacts were deleted. Re-run with --delete when the list looks correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
