"""Test frozen final training and one-shot test evaluation on synthetic data."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import mlflow
import numpy as np
import pytest
import yaml

from real_estate.ml import dataset, final_model
from tests.ml.test_baselines import ml_data, write_year
from tests.ml.test_catboost_model import (
    CATBOOST_CONFIG,
    FakePool,
    FakeRegressor,
    write_catboost_config,
)


@pytest.fixture(autouse=True)
def restore_mlflow_and_fakes() -> None:
    previous = mlflow.get_tracking_uri()
    FakePool.instances.clear()
    FakeRegressor.instances.clear()
    yield
    mlflow.set_tracking_uri(previous)


def write_final_config(root: Path) -> Path:
    path = write_catboost_config(root)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["mlflow"] = {
        "tracking_database": "tracking/mlflow.db",
        "artifact_directory": "tracking/artifacts",
        "experiment_name": "synthetic-final-v1",
        "model_artifact_name": "model",
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def write_selection(root: Path) -> Path:
    path = root / final_model.MODEL_SELECTION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(final_model.EXPECTED_MODEL_SELECTION, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_train_years(root: Path) -> int:
    rows = {
        2021: [{"prix_m2": 100.0, "surface_reelle_bati": 50.0}],
        2022: [{"prix_m2": 200.0, "surface_reelle_bati": 60.0}],
        2023: [{"prix_m2": 400.0, "surface_reelle_bati": 70.0}],
        2024: [
            {"prix_m2": 300.0, "surface_reelle_bati": 80.0},
            {"prix_m2": 600.0, "surface_reelle_bati": 90.0, "code_type_local": 2},
        ],
    }
    for year, values in rows.items():
        write_year(root, year, values)
    return sum(map(len, rows.values()))


def write_test_year(root: Path) -> int:
    rows = [
        {"prix_m2": 350.0, "surface_reelle_bati": 75.0, "code_type_local": 1},
        {"prix_m2": 650.0, "surface_reelle_bati": 95.0, "code_type_local": 2},
    ]
    write_year(root, 2025, rows)
    return len(rows)


def test_versioned_model_selection_and_feature_contract_are_exact() -> None:
    assert final_model.validate_model_selection(Path(".")) == (
        Path.cwd() / "reports/model_selection.json"
    )
    contract = final_model.feature_contract()
    assert contract["model_version"] == "v1"
    assert contract["target"] == "log_prix_m2"
    assert tuple(contract["features"]["ordered"]) == dataset.FEATURE_COLUMNS


def test_train_opens_2021_to_2024_and_has_no_eval_or_early_stopping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_rows = write_train_years(tmp_path)
    config_path = write_final_config(tmp_path)
    write_selection(tmp_path)
    opened = []
    original = dataset.load_ml_year

    def spy(path: Path, expected_year: int) -> dataset.MLDataset:
        opened.append(expected_year)
        assert expected_year != 2025
        return original(path, expected_year)

    def log_fake_model(model: object, artifact_name: str) -> None:
        artifact = tmp_path / "synthetic.cbm"
        artifact.write_text("synthetic", encoding="utf-8")
        mlflow.log_artifact(str(artifact), artifact_path=artifact_name)

    monkeypatch.setattr(dataset, "load_ml_year", spy)
    monkeypatch.setattr(final_model, "_git_commit", lambda root: "a" * 40)
    monkeypatch.setattr(final_model, "_log_model", log_fake_model)
    monkeypatch.setattr(final_model, "EXPECTED_FINAL_TRAIN_ROWS", expected_rows)
    report = final_model.train_final_model(
        config_path,
        tmp_path,
        catboost_api=(FakeRegressor, FakePool),
    )

    assert opened == [2021, 2022, 2023, 2024]
    assert report == {
        "iterations": 3000,
        "model_version": "v1",
        "run_id": report["run_id"],
        "train_rows": expected_rows,
        "train_years": [2021, 2022, 2023, 2024],
    }
    model = FakeRegressor.instances[0]
    assert model.parameters["iterations"] == 3000
    assert model.fit_arguments == {"train": FakePool.instances[0], "verbose": False}
    assert "eval_set" not in model.fit_arguments
    assert "early_stopping_rounds" not in model.fit_arguments

    client = mlflow.MlflowClient()
    run = client.get_run(report["run_id"])
    assert run.data.tags["model_version"] == "v1"
    assert run.data.tags["test_evaluation_started"] == "false"
    assert run.data.tags["test_evaluated"] == "false"
    assert run.data.params["iterations"] == "3000"
    assert run.data.params["feature_count"] == str(len(dataset.FEATURE_COLUMNS))
    artifacts = {
        artifact.path
        for directory in client.list_artifacts(report["run_id"])
        for artifact in (
            client.list_artifacts(report["run_id"], directory.path)
            if directory.is_dir
            else [directory]
        )
    }
    assert "contracts/feature_contract.json" in artifacts
    assert "contracts/ml.yaml" in artifacts
    assert "contracts/model_selection.json" in artifacts
    assert "model/synthetic.cbm" in artifacts


def test_train_rejects_wrong_row_count_before_model_fit(
    tmp_path: Path,
) -> None:
    write_train_years(tmp_path)
    config_path = write_final_config(tmp_path)
    write_selection(tmp_path)
    with pytest.raises(final_model.FinalModelError, match="row count"):
        final_model.train_final_model(
            config_path,
            tmp_path,
            catboost_api=(FakeRegressor, FakePool),
        )
    assert not FakeRegressor.instances


class CatBoostRegressor:
    """Frozen loaded-model double that fails if evaluation attempts a fit."""

    fit_calls: ClassVar[int] = 0
    tree_count_ = 3000
    feature_names_: ClassVar[list[str]] = list(dataset.FEATURE_COLUMNS)

    def get_params(self) -> dict[str, object]:
        return {
            key: value
            for key, value in CATBOOST_CONFIG.items()
            if key != "early_stopping_rounds"
        }

    def fit(self, *_args: object, **_kwargs: object) -> None:
        self.fit_calls += 1
        raise AssertionError("Evaluation must never fit the loaded model.")

    def predict(self, pool: Any) -> np.ndarray:
        return np.log(np.full(len(pool.data), 500.0))


class EvaluationPool:
    def __init__(self, *, data: Any, cat_features: list[str]) -> None:
        self.data = data
        self.cat_features = cat_features


def fake_run(
    *,
    test_evaluation_started: str = "false",
    test_evaluated: str = "false",
    model_version: str = "v1",
) -> Any:
    settings = SimpleNamespace(model_parameters={
        key: value
        for key, value in CATBOOST_CONFIG.items()
        if key != "early_stopping_rounds"
    })
    tags = final_model._run_tags() | {
        "model_version": model_version,
        "test_evaluation_started": test_evaluation_started,
        "test_evaluated": test_evaluated,
    }
    params = {
        key: str(value)
        for key, value in final_model._run_parameters(settings, "a" * 40).items()
    }
    return SimpleNamespace(
        data=SimpleNamespace(tags=tags, params=params),
        info=SimpleNamespace(status="FINISHED", run_id="run-1"),
    )


def create_real_frozen_run(
    root: Path,
    config_path: Path,
    *,
    started: str = "false",
    evaluated: str = "false",
) -> tuple[str, mlflow.MlflowClient]:
    settings = final_model.load_catboost_settings(config_path)
    tracking = final_model.configure_tracking(
        final_model.load_tracking_settings(config_path),
        root,
    )
    tags = final_model._run_tags() | {
        "test_evaluation_started": started,
        "test_evaluated": evaluated,
    }
    with mlflow.start_run(experiment_id=tracking.experiment_id, tags=tags) as run:
        mlflow.log_params(final_model._run_parameters(settings, "c" * 40))
        run_id = run.info.run_id
    return run_id, tracking.client


@contextmanager
def fake_active_run(*_args: object, **_kwargs: object) -> Any:
    yield SimpleNamespace(info=SimpleNamespace(run_id="run-1"))


def patch_evaluation_tracking(
    monkeypatch: pytest.MonkeyPatch,
    run: Any,
    events: list[str],
) -> dict[str, object]:
    client = SimpleNamespace(get_run=lambda run_id: run)
    context = SimpleNamespace(client=client, model_artifact_name="model")
    logged: dict[str, object] = {}
    monkeypatch.setattr(final_model, "configure_tracking", lambda *_args: context)
    monkeypatch.setattr(final_model, "validate_model_selection", lambda root: root)
    monkeypatch.setattr(final_model.mlflow, "start_run", fake_active_run)
    monkeypatch.setattr(
        final_model.mlflow,
        "log_metrics",
        lambda metrics: logged.update(metrics=metrics),
    )
    monkeypatch.setattr(
        final_model.mlflow,
        "set_tag",
        lambda key, value: logged.setdefault("tags", {}).update({key: value}),
    )
    monkeypatch.setattr(
        final_model,
        "_load_model",
        lambda *_args: events.append("model_loaded") or CatBoostRegressor(),
    )
    return logged


def test_evaluate_loads_model_before_test_and_never_modifies_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_final_config(tmp_path)
    events: list[str] = []
    logged = patch_evaluation_tracking(monkeypatch, fake_run(), events)
    test = ml_data([
        {"prix_m2": 400.0, "code_type_local": 1},
        {"prix_m2": 600.0, "code_type_local": 2},
    ], 2025)
    monkeypatch.setattr(
        final_model,
        "_load_sealed_test",
        lambda *_args: events.append("test_opened") or test,
    )
    monkeypatch.setattr(final_model, "EXPECTED_TEST_ROWS", 2)

    report = final_model.evaluate_final_model(
        "run-1",
        config_path,
        tmp_path,
        pool_factory=EvaluationPool,
    )
    assert events == ["model_loaded", "test_opened"]
    assert CatBoostRegressor.fit_calls == 0
    assert report["test_year"] == 2025
    assert report["metrics"]["house"]["rows"] == 1
    assert report["metrics"]["apartment"]["rows"] == 1
    assert logged["tags"] == {
        "test_evaluation_started": "true",
        "test_evaluated": "true",
    }
    assert "test_global_rmse_log" in logged["metrics"]


def test_real_store_marks_started_before_opening_test_and_success_marks_evaluated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_final_config(tmp_path)
    write_selection(tmp_path)
    run_id, client = create_real_frozen_run(tmp_path, config_path)
    monkeypatch.setattr(final_model, "_load_model", lambda *_args: CatBoostRegressor())
    monkeypatch.setattr(final_model, "EXPECTED_TEST_ROWS", 2)
    opened = []

    def load_test(*_args: object) -> dataset.MLDataset:
        current = client.get_run(run_id)
        assert current.data.tags["test_evaluation_started"] == "true"
        assert current.data.tags["test_evaluated"] == "false"
        opened.append(2025)
        return ml_data([
            {"prix_m2": 400.0, "code_type_local": 1},
            {"prix_m2": 600.0, "code_type_local": 2},
        ], 2025)

    monkeypatch.setattr(final_model, "_load_sealed_test", load_test)
    final_model.evaluate_final_model(
        run_id,
        config_path,
        tmp_path,
        pool_factory=EvaluationPool,
    )

    completed = client.get_run(run_id)
    assert opened == [2025]
    assert completed.data.tags["test_evaluation_started"] == "true"
    assert completed.data.tags["test_evaluated"] == "true"


@pytest.mark.parametrize(
    ("started", "evaluated"),
    [("true", "false"), ("false", "true")],
)
def test_real_store_refuses_started_or_evaluated_run_before_opening_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    started: str,
    evaluated: str,
) -> None:
    config_path = write_final_config(tmp_path)
    write_selection(tmp_path)
    run_id, _ = create_real_frozen_run(
        tmp_path,
        config_path,
        started=started,
        evaluated=evaluated,
    )
    events = []
    monkeypatch.setattr(
        final_model,
        "_load_model",
        lambda *_args: events.append("model_loaded") or CatBoostRegressor(),
    )
    monkeypatch.setattr(
        final_model,
        "_load_sealed_test",
        lambda *_args: events.append("test_opened"),
    )

    with pytest.raises(final_model.TestAlreadyEvaluatedError):
        final_model.evaluate_final_model(run_id, config_path, tmp_path)
    assert events == ["model_loaded"]


def test_real_store_keeps_started_after_failure_and_refuses_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_final_config(tmp_path)
    write_selection(tmp_path)
    run_id, client = create_real_frozen_run(tmp_path, config_path)
    model_loads = []
    test_opens = []
    monkeypatch.setattr(
        final_model,
        "_load_model",
        lambda *_args: model_loads.append(run_id) or CatBoostRegressor(),
    )

    def fail_after_open(*_args: object) -> dataset.MLDataset:
        assert client.get_run(run_id).data.tags["test_evaluation_started"] == "true"
        test_opens.append(2025)
        raise RuntimeError("synthetic interruption after opening test")

    monkeypatch.setattr(final_model, "_load_sealed_test", fail_after_open)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        final_model.evaluate_final_model(run_id, config_path, tmp_path)

    interrupted = client.get_run(run_id)
    assert interrupted.data.tags["test_evaluation_started"] == "true"
    assert interrupted.data.tags["test_evaluated"] == "false"
    with pytest.raises(final_model.TestAlreadyEvaluatedError):
        final_model.evaluate_final_model(run_id, config_path, tmp_path)
    assert model_loads == [run_id, run_id]
    assert test_opens == [2025]


def test_second_evaluation_validates_model_then_refuses_before_test_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_final_config(tmp_path)
    events: list[str] = []
    patch_evaluation_tracking(monkeypatch, fake_run(test_evaluated="true"), events)
    with pytest.raises(final_model.TestAlreadyEvaluatedError):
        final_model.evaluate_final_model("run-1", config_path, tmp_path)
    assert events == ["model_loaded"]


def test_incompatible_run_is_refused_before_model_or_test_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_final_config(tmp_path)
    events: list[str] = []
    patch_evaluation_tracking(monkeypatch, fake_run(model_version="v2"), events)
    with pytest.raises(final_model.FinalModelError, match="model_version"):
        final_model.evaluate_final_model("run-1", config_path, tmp_path)
    assert events == []


def test_wrong_test_row_count_keeps_started_without_metrics_or_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_final_config(tmp_path)
    events: list[str] = []
    logged = patch_evaluation_tracking(monkeypatch, fake_run(), events)
    monkeypatch.setattr(
        final_model,
        "_load_sealed_test",
        lambda *_args: events.append("test_opened") or ml_data([{"prix_m2": 400.0}], 2025),
    )
    monkeypatch.setattr(final_model, "EXPECTED_TEST_ROWS", 2)
    with pytest.raises(final_model.FinalModelError, match="row count"):
        final_model.evaluate_final_model(
            "run-1",
            config_path,
            tmp_path,
            pool_factory=EvaluationPool,
        )
    assert events == ["model_loaded", "test_opened"]
    assert logged == {"tags": {"test_evaluation_started": "true"}}


def test_real_mlflow_catboost_roundtrip_and_one_shot_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_train_rows = write_train_years(tmp_path)
    expected_test_rows = write_test_year(tmp_path)
    config_path = write_final_config(tmp_path)
    write_selection(tmp_path)
    monkeypatch.setattr(final_model, "_git_commit", lambda root: "b" * 40)
    monkeypatch.setattr(final_model, "EXPECTED_FINAL_TRAIN_ROWS", expected_train_rows)
    monkeypatch.setattr(final_model, "EXPECTED_TEST_ROWS", expected_test_rows)

    trained = final_model.train_final_model(
        config_path,
        tmp_path,
    )
    loaded = final_model._load_model(trained["run_id"], "model")
    assert type(loaded).__name__ == "CatBoostRegressor"
    assert loaded.tree_count_ == 3000

    evaluated = final_model.evaluate_final_model(
        trained["run_id"],
        config_path,
        tmp_path,
    )
    assert evaluated["test_rows"] == expected_test_rows
    client = mlflow.MlflowClient()
    run = client.get_run(trained["run_id"])
    assert run.data.tags["test_evaluation_started"] == "true"
    assert run.data.tags["test_evaluated"] == "true"
    assert "test_global_rmse_log" in run.data.metrics
    with pytest.raises(final_model.TestAlreadyEvaluatedError):
        final_model.evaluate_final_model(
            trained["run_id"],
            config_path,
            tmp_path,
        )


def test_cli_dispatches_train_and_explicit_evaluate(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        final_model,
        "train_final_model",
        lambda *_args: calls.append("train") or {"run_id": "trained"},
    )
    monkeypatch.setattr(
        final_model,
        "evaluate_final_model",
        lambda run_id, *_args: calls.append(("evaluate", run_id)) or {"run_id": run_id},
    )
    assert final_model.main(["train"]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == "trained"
    assert final_model.main(["evaluate", "--run-id", "run-1"]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == "run-1"
    assert calls == ["train", ("evaluate", "run-1")]
