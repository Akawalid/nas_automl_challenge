import logging
import time
from collections.abc import Callable
from pathlib import Path

import mlflow


def is_retryable_mlflow_error(exc: Exception) -> bool:
    """Return True when the exception is likely transient and worth retrying."""
    message = str(exc).lower()
    retryable_markers = (
        "database is locked",
        "disk i/o error",
        "temporarily unavailable",
        "sqlite3.operationalerror",
        "operationalerror",
        "timeout",
    )
    return any(marker in message for marker in retryable_markers)


def call_mlflow_with_retry(
    logger: logging.Logger,
    operation_name: str,
    operation: Callable[[], object],
    max_retries: int = 5,
    initial_delay_seconds: float = 0.25,
    max_delay_seconds: float = 5.0,
) -> object:
    """Call an MLflow operation and retry on transient backend errors."""
    delay = initial_delay_seconds
    for attempt in range(1, max_retries + 1):
        try:
            return operation()
        except Exception as exc:
            is_last_attempt = attempt >= max_retries
            if is_last_attempt or not is_retryable_mlflow_error(exc):
                raise
            sleep_seconds = min(delay, max_delay_seconds)
            logger.warning(
                "MLflow operation '%s' failed (%s). Retrying in %.2fs (%d/%d).",
                operation_name,
                exc,
                sleep_seconds,
                attempt,
                max_retries,
            )
            time.sleep(sleep_seconds)
            delay *= 2
    raise RuntimeError(f"MLflow operation '{operation_name}' failed unexpectedly")


def best_effort_mlflow_call(
    logger: logging.Logger,
    operation_name: str,
    operation: Callable[[], object],
    max_retries: int = 5,
) -> object | None:
    """Run MLflow operation with retries and avoid crashing if it still fails."""
    try:
        return call_mlflow_with_retry(
            logger=logger,
            operation_name=operation_name,
            operation=operation,
            max_retries=max_retries,
        )
    except Exception as exc:
        logger.warning(
            "Skipping MLflow operation '%s' after retries. Last error: %s",
            operation_name,
            exc,
        )
        return None


class SafeMLflowClient:
    """Best-effort MLflow wrapper with retry for transient backend failures."""

    def __init__(self, logger: logging.Logger, max_retries: int = 5) -> None:
        self.logger = logger
        self.max_retries = max_retries

    def call(self, operation_name: str, operation: Callable[[], object]) -> object | None:
        return best_effort_mlflow_call(
            logger=self.logger,
            operation_name=operation_name,
            operation=operation,
            max_retries=self.max_retries,
        )

    def set_tag(self, key: str, value: object) -> object | None:
        return self.call(f"set_tag:{key}", lambda: mlflow.set_tag(key, value))

    def set_tags(self, tags: dict[str, object]) -> object | None:
        return self.call("set_tags", lambda: mlflow.set_tags(tags))

    def log_param(self, key: str, value: object) -> object | None:
        return self.call(f"log_param:{key}", lambda: mlflow.log_param(key, value))

    def log_params(self, params: dict[str, object]) -> object | None:
        return self.call("log_params", lambda: mlflow.log_params(params))

    def log_metric(
        self,
        key: str,
        value: float | int,
        step: int | None = None,
    ) -> object | None:
        return self.call(
            f"log_metric:{key}",
            lambda: mlflow.log_metric(key, value, step=step),
        )

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None,
        operation_name: str | None = None,
    ) -> object | None:
        default_operation_name = f"log_artifact:{Path(local_path).name}"
        if artifact_path is None:
            return self.call(
                operation_name or default_operation_name,
                lambda: mlflow.log_artifact(local_path),
            )
        return self.call(
            operation_name or default_operation_name,
            lambda: mlflow.log_artifact(local_path, artifact_path=artifact_path),
        )

    def log_artifacts(
        self,
        local_dir: str,
        artifact_path: str | None = None,
        operation_name: str | None = None,
    ) -> object | None:
        default_operation_name = f"log_artifacts:{Path(local_dir).name}"
        if artifact_path is None:
            return self.call(
                operation_name or default_operation_name,
                lambda: mlflow.log_artifacts(local_dir),
            )
        return self.call(
            operation_name or default_operation_name,
            lambda: mlflow.log_artifacts(local_dir, artifact_path=artifact_path),
        )


def warn_if_sqlite_tracking_uri(logger: logging.Logger, tracking_uri: str) -> None:
    """Log a warning when SQLite is used as tracking backend in parallel contexts."""
    if tracking_uri.startswith("sqlite://"):
        logger.warning(
            "SQLite MLflow backend detected. Parallel runs may fail under write contention;"
            " prefer a PostgreSQL/MySQL backend store for reliable concurrent logging."
        )
