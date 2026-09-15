"""Exercise the real local MLflow store with temporary paths."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
import yaml

from real_estate.ml import tracking


@pytest.fixture(autouse=True)
def restore_tracking_uri() -> None:
    previous = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(previous)


def write_tracking_config(root: Path, **overrides: object) -> Path:
    values = {
        "tracking_database": "tracking/mlflow.db",
        "artifact_directory": "tracking/artifacts",
        "experiment_name": "synthetic-final-v1",
        "model_artifact_name": "model",
    }
    values.update(overrides)
    path = root / "ml.yaml"
    path.write_text(yaml.safe_dump({"mlflow": values}), encoding="utf-8")
    return path


def test_real_local_tracking_store_creates_and_reuses_experiment(tmp_path: Path) -> None:
    settings = tracking.load_tracking_settings(write_tracking_config(tmp_path))
    first = tracking.configure_tracking(settings, tmp_path)
    second = tracking.configure_tracking(settings, tmp_path)

    assert first.tracking_uri == f"sqlite:///{tmp_path / 'tracking/mlflow.db'}"
    assert first.artifact_root == tmp_path / "tracking/artifacts"
    assert first.experiment_id == second.experiment_id
    assert first.client.get_experiment(first.experiment_id).name == "synthetic-final-v1"
    assert (tmp_path / "tracking/mlflow.db").is_file()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tracking_database", "/tmp/external.db"),
        ("tracking_database", "../external.db"),
        ("tracking_database", "tracking/not-a-database.txt"),
        ("artifact_directory", "../artifacts"),
        ("experiment_name", ""),
        ("model_artifact_name", "models/v1"),
    ],
)
def test_tracking_configuration_rejects_unsafe_values(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    with pytest.raises(tracking.TrackingConfigError):
        tracking.load_tracking_settings(write_tracking_config(tmp_path, **{field: value}))
