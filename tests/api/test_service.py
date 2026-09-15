"""Service-level tests for the frozen CatBoost serving contract."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor, Pool
from mlflow.tracking import MlflowClient

from real_estate.api import service
from real_estate.api.schemas import PredictionRequest
from real_estate.ml import catboost_model, dataset, final_model


class CapturingModel:
    """Predictor double that records the exact prepared frame."""

    def __init__(self, prediction: object = np.array([math.log(2500.0)])) -> None:
        self.prediction = prediction
        self.features: pd.DataFrame | None = None

    def predict(self, features: pd.DataFrame) -> object:
        self.features = features.copy(deep=True)
        return self.prediction


def request_data(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "surface_reelle_bati": 80.0,
        "nombre_pieces_principales": 4,
        "nombre_lots": 1,
        "surface_terrain": 350.0,
        "has_dependance": True,
        "code_type_local": 1,
        "code_postal": "75001",
        "code_departement": "75",
        "source_code_commune": "75101",
        "canonical_commune_code": "75056",
        "resolved_geo_type": "ARM",
        "region_code": "11",
        "mutation_date": date(2025, 7, 9),
    }
    values.update(overrides)
    return values


def prediction_request(**overrides: object) -> PredictionRequest:
    return PredictionRequest.model_validate(request_data(**overrides))


def test_feature_frame_is_exactly_the_frozen_allowlist_in_order() -> None:
    frame = service.build_inference_features(prediction_request())
    assert tuple(frame.columns) == dataset.FEATURE_COLUMNS
    assert frame.shape == (1, 16)
    assert not set(frame.columns) & dataset.FORBIDDEN_FEATURE_COLUMNS
    assert frame.loc[0, "mutation_year"] == 2025
    assert frame.loc[0, "mutation_month"] == 7
    assert frame.loc[0, "surface_terrain_missing"] == 0
    assert frame.loc[0, "geography_unresolved"] == 0


def test_missing_geography_and_terrain_use_training_preparation() -> None:
    frame = service.build_inference_features(prediction_request(
        surface_terrain=None,
        canonical_commune_code=None,
        resolved_geo_type=None,
        region_code=None,
        has_dependance=False,
    ))
    assert np.isnan(frame.loc[0, "surface_terrain"])
    assert frame.loc[0, "surface_terrain_missing"] == 1
    assert frame.loc[0, "geography_unresolved"] == 1
    assert frame.loc[0, "has_dependance"] == 0
    for column in ("canonical_commune_code", "resolved_geo_type", "region_code"):
        assert frame.loc[0, column] == catboost_model.MISSING_CATEGORY
    assert all(
        frame[column].dtype == object for column in dataset.CATEGORICAL_FEATURES
    )
    assert all(
        str(frame[column].dtype) == "int8"
        for column in catboost_model.BOOLEAN_COLUMNS
    )


def test_zero_terrain_is_preserved_as_a_value() -> None:
    frame = service.build_inference_features(prediction_request(surface_terrain=0.0))
    assert frame.loc[0, "surface_terrain"] == 0.0
    assert frame.loc[0, "surface_terrain_missing"] == 0


def test_prediction_exponentiates_log_output_without_dvf_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dataset,
        "_read_source",
        lambda *_args: pytest.fail("Serving must never open a DVF source."),
    )
    model = CapturingModel()
    predictor = service.PredictionService(model)
    result = predictor.predict_price_m2(prediction_request())
    assert result == pytest.approx(2500.0)
    assert model.features is not None
    assert tuple(model.features.columns) == dataset.FEATURE_COLUMNS


@pytest.mark.parametrize(
    "prediction",
    [np.array([np.nan]), np.array([np.inf]), np.array([]), np.array([1.0, 2.0])],
)
def test_prediction_requires_one_finite_log_value(prediction: np.ndarray) -> None:
    predictor = service.PredictionService(CapturingModel(prediction))
    with pytest.raises(service.PredictionError, match="invalid log-price"):
        predictor.predict_price_m2(prediction_request())


def test_prediction_requires_finite_positive_price() -> None:
    predictor = service.PredictionService(CapturingModel(np.array([1000.0])))
    with pytest.raises(service.PredictionError, match="finite range"):
        predictor.predict_price_m2(prediction_request())

    predictor = service.PredictionService(CapturingModel(np.array([-1000.0])))
    with pytest.raises(service.PredictionError, match="invalid price"):
        predictor.predict_price_m2(prediction_request())


def test_model_exception_is_wrapped_without_its_message() -> None:
    class FailingModel:
        def predict(self, _features: pd.DataFrame) -> np.ndarray:
            raise RuntimeError("/private/path/mlflow.db")

    with pytest.raises(service.PredictionError, match="Model prediction failed") as error:
        service.PredictionService(FailingModel()).predict_price_m2(
            prediction_request()
        )
    assert "/private/path" not in str(error.value)


def test_serving_settings_use_safe_defaults_and_environment_overrides() -> None:
    defaults = service.ServingSettings.from_environment({})
    assert defaults.tracking_uri == "sqlite:///mlflow.db"
    assert defaults.run_id == "f140d75d05504aacad1ea18a09f0f4a4"

    configured = service.ServingSettings.from_environment({
        "REAL_ESTATE_MLFLOW_TRACKING_URI": "sqlite:////tmp/synthetic.db",
        "REAL_ESTATE_MODEL_RUN_ID": "synthetic-run",
    })
    assert configured.tracking_uri == "sqlite:////tmp/synthetic.db"
    assert configured.run_id == "synthetic-run"


def test_loader_validates_run_and_frozen_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = SimpleNamespace(data=SimpleNamespace(params={}))
    calls = []
    monkeypatch.setattr(service, "_validate_run_contract", lambda *_args: calls.append("run"))
    monkeypatch.setattr(
        service,
        "_feature_contract_sha256",
        lambda: service.EXPECTED_FEATURE_CONTRACT_SHA256,
    )
    run.data.params["feature_contract_sha256"] = (
        service.EXPECTED_FEATURE_CONTRACT_SHA256
    )

    service._validate_frozen_run(run, Path("configs/ml.yaml"))
    assert calls == ["run"]


def test_loader_rejects_wrong_frozen_feature_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service, "_feature_contract_sha256", lambda: "wrong")
    with pytest.raises(service.ModelUnavailableError, match="Local feature contract"):
        service._validate_frozen_run(object(), Path("configs/ml.yaml"))


def test_real_mlflow_store_loads_a_synthetic_catboost_model(
    tmp_path: Path,
) -> None:
    previous_uri = mlflow.get_tracking_uri()
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_root = tmp_path / "artifacts"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment(
        "api-synthetic",
        artifact_location=artifact_root.as_uri(),
    )
    mlflow.set_tracking_uri(tracking_uri)
    try:
        requests = [
            prediction_request(code_postal="75001", surface_reelle_bati=40.0),
            prediction_request(code_postal="75002", surface_reelle_bati=60.0),
            prediction_request(
                code_type_local=2,
                code_postal="69001",
                code_departement="69",
                source_code_commune="69381",
                canonical_commune_code="69123",
                region_code="84",
                surface_reelle_bati=80.0,
            ),
            prediction_request(
                code_type_local=2,
                code_postal="13001",
                code_departement="13",
                source_code_commune="13201",
                canonical_commune_code="13055",
                region_code="93",
                surface_reelle_bati=100.0,
            ),
        ]
        features = pd.concat(
            [service.build_inference_features(item) for item in requests],
            ignore_index=True,
        )
        settings = catboost_model.load_catboost_settings()
        model = CatBoostRegressor(**settings.model_parameters)
        model.fit(
            Pool(
                features,
                label=np.log([2000.0, 2500.0, 3000.0, 3500.0]),
                cat_features=list(dataset.CATEGORICAL_FEATURES),
            ),
            verbose=False,
        )
        with mlflow.start_run(
            experiment_id=experiment_id,
            tags=final_model._run_tags(),
        ) as run:
            mlflow.log_params(final_model._run_parameters(settings, "a" * 40))
            mlflow.catboost.log_model(model, name="model")
            run_id = run.info.run_id

        loaded = service.load_prediction_service(
            service.ServingSettings(tracking_uri=tracking_uri, run_id=run_id)
        )
        prediction = loaded.predict_price_m2(requests[0])
        assert math.isfinite(prediction)
        assert prediction > 0
        assert loaded.metadata.feature_count == 16
    finally:
        mlflow.set_tracking_uri(previous_uri)
