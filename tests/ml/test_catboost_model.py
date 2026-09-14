"""Test CatBoost preparation and orchestration without training CatBoost."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
import pytest
import yaml

from real_estate.ml import catboost_model, dataset
from tests.ml.test_baselines import ml_data, write_year
from tests.ml.test_dataset import write_config

if TYPE_CHECKING:
    from collections.abc import Sequence


CATBOOST_CONFIG = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "random_seed": 42,
    "thread_count": -1,
    "early_stopping_rounds": 100,
    "allow_writing_files": False,
}


class FakePool:
    """Capture native categorical declarations and labels."""

    instances: ClassVar[list[FakePool]] = []

    def __init__(
        self,
        *,
        data: pd.DataFrame,
        label: np.ndarray,
        cat_features: list[str],
    ) -> None:
        self.data = data.copy(deep=True)
        self.label = np.asarray(label).copy()
        self.cat_features = list(cat_features)
        self.instances.append(self)


class FakeRegressor:
    """Capture fit semantics and return deterministic log predictions."""

    instances: ClassVar[list[FakeRegressor]] = []
    predictions: ClassVar[Sequence[float]] = np.log([110.0, 180.0])

    def __init__(self, **parameters: object) -> None:
        self.parameters = parameters
        self.fit_arguments: dict[str, object] | None = None
        self.prediction_input: FakePool | None = None
        self.instances.append(self)

    def fit(self, train: FakePool, **arguments: object) -> FakeRegressor:
        self.fit_arguments = {"train": train, **arguments}
        return self

    def predict(self, data: FakePool) -> np.ndarray:
        self.prediction_input = data
        return self.predictions.copy()

    def get_best_iteration(self) -> int:
        return 17


@pytest.fixture(autouse=True)
def reset_fakes() -> None:
    FakePool.instances.clear()
    FakeRegressor.instances.clear()


def write_catboost_config(root: Path) -> Path:
    path = write_config(root)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["catboost"] = CATBOOST_CONFIG
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def write_development_years(root: Path) -> None:
    write_year(root, 2021, [{"prix_m2": 100.0}])
    write_year(root, 2022, [{"prix_m2": 200.0}])
    write_year(root, 2023, [{"prix_m2": 400.0, "code_type_local": 2}])
    write_year(root, 2024, [
        {"prix_m2": 100.0, "code_type_local": 1},
        {"prix_m2": 200.0, "code_type_local": 2},
    ])


def test_project_catboost_configuration_is_locked() -> None:
    settings = catboost_model.load_catboost_settings()
    assert settings.model_parameters == {
        key: value
        for key, value in CATBOOST_CONFIG.items()
        if key != "early_stopping_rounds"
    }
    assert settings.early_stopping_rounds == 100


def test_changed_catboost_configuration_is_rejected(tmp_path: Path) -> None:
    path = write_catboost_config(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["catboost"]["random_seed"] = 7
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="locked V1 canary"):
        catboost_model.load_catboost_settings(path)


def test_feature_preparation_preserves_allowlist_and_native_missing_values() -> None:
    data = ml_data([{
        "surface_terrain": np.nan,
        "canonical_commune_code": None,
        "resolved_geo_type": None,
        "region_code": None,
        "has_dependance": True,
    }])
    prepared = catboost_model.prepare_catboost_features(data)

    assert tuple(prepared.columns) == dataset.FEATURE_COLUMNS
    assert np.isnan(prepared.loc[0, "surface_terrain"])
    assert prepared.loc[0, "canonical_commune_code"] == "__MISSING__"
    assert prepared.loc[0, "resolved_geo_type"] == "__MISSING__"
    assert prepared.loc[0, "region_code"] == "__MISSING__"
    assert all(prepared[column].dtype == object for column in dataset.CATEGORICAL_FEATURES)
    assert all(str(prepared[column].dtype) == "int8" for column in catboost_model.BOOLEAN_COLUMNS)
    assert prepared.loc[0, "has_dependance"] == 1


def test_orchestration_fits_train_and_uses_validation_only_as_eval_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_development_years(tmp_path)
    config_path = write_catboost_config(tmp_path)
    opened = []
    original = dataset.load_ml_year

    def spy(path: Path, expected_year: int) -> dataset.MLDataset:
        opened.append(expected_year)
        assert expected_year != 2025
        return original(path, expected_year)

    monkeypatch.setattr(dataset, "load_ml_year", spy)
    report = catboost_model.run_catboost_validation(
        config_path,
        tmp_path,
        catboost_api=(FakeRegressor, FakePool),
    )

    assert opened == [2021, 2022, 2023, 2024]
    assert len(FakePool.instances) == 2
    train_pool, validation_pool = FakePool.instances
    assert train_pool.cat_features == list(dataset.CATEGORICAL_FEATURES)
    assert validation_pool.cat_features == list(dataset.CATEGORICAL_FEATURES)
    assert tuple(train_pool.data.columns) == dataset.FEATURE_COLUMNS
    assert train_pool.label.tolist() == pytest.approx(np.log([100.0, 200.0, 400.0]))
    assert validation_pool.label.tolist() == pytest.approx(np.log([100.0, 200.0]))

    model = FakeRegressor.instances[0]
    assert model.parameters == catboost_model.load_catboost_settings(config_path).model_parameters
    assert model.fit_arguments == {
        "train": train_pool,
        "eval_set": validation_pool,
        "early_stopping_rounds": 100,
        "use_best_model": True,
        "verbose": False,
    }
    assert model.prediction_input is validation_pool
    assert report["best_iteration"] == 17
    assert report["target"] == "log_prix_m2"
    assert report["validation_metrics"]["global"]["mae_eur_m2"] == pytest.approx(15.0)
    assert report["validation_metrics"]["house"]["rows"] == 1
    assert report["validation_metrics"]["apartment"]["rows"] == 1


def test_official_baseline_comparison_uses_lower_is_better_deltas() -> None:
    metrics: dict[str, float | int] = {
        "rows": 2,
        "rmse_log": 0.5,
        "mae_eur_m2": 1000.0,
        "median_ae_eur_m2": 700.0,
    }
    report = catboost_model.compare_with_official_baseline(metrics)
    for name, baseline in catboost_model.OFFICIAL_DEPARTMENT_TYPE_BASELINE.items():
        assert report[name]["baseline_value"] == baseline
        assert report[name]["model_value"] == metrics[name]
        assert report[name]["absolute_delta"] == pytest.approx(float(metrics[name]) - baseline)
        assert report[name]["relative_improvement_pct"] == pytest.approx(
            (baseline - float(metrics[name])) / baseline * 100.0
        )
        assert report[name]["improved"] is True


def test_cli_runs_complete_synthetic_orchestration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_development_years(tmp_path)
    config_path = write_catboost_config(tmp_path)
    monkeypatch.setattr(
        catboost_model,
        "_load_catboost_api",
        lambda: (FakeRegressor, FakePool),
    )

    result = catboost_model.main([
        "--config",
        str(config_path),
        "--project-root",
        str(tmp_path),
    ])
    report: dict[str, Any] = json.loads(capsys.readouterr().out)
    assert result == 0
    assert report["train_rows"] == 3
    assert report["validation_rows"] == 2
    assert report["model"] == "CatBoostRegressor"
    assert report["best_iteration"] == 17
    assert set(report["baseline_comparison"]) == set(
        catboost_model.OFFICIAL_DEPARTMENT_TYPE_BASELINE
    )
