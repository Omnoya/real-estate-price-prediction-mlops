"""Configure the project-local MLflow tracking store and artifact location."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import yaml
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from real_estate.ml.dataset import DEFAULT_ML_CONFIG_PATH


class TrackingConfigError(RuntimeError):
    """The local MLflow configuration is missing or unsafe."""


@dataclass(frozen=True)
class TrackingSettings:
    """Project-relative locations and stable names used by MLflow."""

    tracking_database: str
    artifact_directory: str
    experiment_name: str
    model_artifact_name: str

    def __post_init__(self) -> None:
        _validate_relative_path(self.tracking_database, suffix=".db")
        _validate_relative_path(self.artifact_directory)
        if not self.experiment_name.strip():
            raise ValueError("MLflow experiment_name must not be empty.")
        if not self.model_artifact_name.strip() or "/" in self.model_artifact_name:
            raise ValueError("MLflow model_artifact_name must be one path segment.")


@dataclass(frozen=True)
class TrackingContext:
    """Configured MLflow client plus identifiers used by one workflow."""

    client: MlflowClient
    tracking_uri: str
    artifact_root: Path
    experiment_id: str
    model_artifact_name: str


def _validate_relative_path(value: str, suffix: str | None = None) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("MLflow local paths must be strings.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or (suffix and path.suffix != suffix):
        raise ValueError("MLflow paths must be safe project-relative paths.")


def load_tracking_settings(path: Path = DEFAULT_ML_CONFIG_PATH) -> TrackingSettings:
    """Load the local MLflow section from the shared ML configuration."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))["mlflow"]
        return TrackingSettings(
            tracking_database=raw["tracking_database"],
            artifact_directory=raw["artifact_directory"],
            experiment_name=raw["experiment_name"],
            model_artifact_name=raw["model_artifact_name"],
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise TrackingConfigError(f"Invalid MLflow configuration: {error}") from error


def _resolve_within_root(project_root: Path, configured_path: str) -> Path:
    root = project_root.resolve()
    path = (root / configured_path).resolve()
    if not path.is_relative_to(root):
        raise TrackingConfigError("MLflow path escapes the project root.")
    return path


def configure_tracking(
    settings: TrackingSettings,
    project_root: Path,
) -> TrackingContext:
    """Select a local SQLite store and create/reuse its local experiment."""
    database = _resolve_within_root(project_root, settings.tracking_database)
    artifact_root = _resolve_within_root(project_root, settings.artifact_directory)
    database.parent.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{database}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(settings.experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            settings.experiment_name,
            artifact_location=artifact_root.as_uri(),
        )
    elif experiment.lifecycle_stage != "active":
        raise TrackingConfigError("Configured MLflow experiment is not active.")
    else:
        experiment_id = experiment.experiment_id

    active = client.search_experiments(
        view_type=ViewType.ACTIVE_ONLY,
        filter_string=f"name = '{settings.experiment_name}'",
    )
    if len(active) != 1 or active[0].experiment_id != experiment_id:
        raise TrackingConfigError("MLflow experiment resolution is ambiguous.")
    return TrackingContext(
        client=client,
        tracking_uri=tracking_uri,
        artifact_root=artifact_root,
        experiment_id=experiment_id,
        model_artifact_name=settings.model_artifact_name,
    )
