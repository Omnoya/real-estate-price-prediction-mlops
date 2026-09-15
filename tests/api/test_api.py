"""HTTP contract tests using FastAPI TestClient and a synthetic predictor."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from real_estate.api.main import create_app
from real_estate.api.service import ModelUnavailableError, PredictionService


class FixedModel:
    def predict(self, _features: object) -> np.ndarray:
        return np.array([math.log(3210.5)])


def valid_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "surface_reelle_bati": 80.0,
        "nombre_pieces_principales": 4,
        "nombre_lots": 1,
        "surface_terrain": None,
        "has_dependance": True,
        "code_type_local": 1,
        "code_postal": "75001",
        "code_departement": "75",
        "source_code_commune": "75101",
        "canonical_commune_code": "75056",
        "resolved_geo_type": "ARM",
        "region_code": "11",
        "mutation_date": "2025-07-09",
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app(PredictionService(FixedModel()))) as current:
        yield current


def test_health_returns_minimal_frozen_version(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "v1"}


def test_model_returns_only_public_metadata(client: TestClient) -> None:
    response = client.get("/model")
    assert response.status_code == 200
    assert response.json() == {
        "model": "CatBoostRegressor",
        "model_version": "v1",
        "target": "log_prix_m2",
        "feature_count": 16,
    }
    assert "artifact" not in response.text
    assert "mlflow.db" not in response.text


def test_predict_returns_price_per_square_metre(client: TestClient) -> None:
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    assert response.json()["predicted_price_m2"] == pytest.approx(3210.5)
    assert response.json()["model_version"] == "v1"


@pytest.mark.parametrize(
    "overrides",
    [
        {"surface_reelle_bati": 0},
        {"surface_reelle_bati": -1},
        {"nombre_pieces_principales": -1},
        {"nombre_lots": -1},
        {"surface_terrain": -1},
        {"code_type_local": 3},
        {"code_postal": ""},
        {"code_departement": "   "},
        {"source_code_commune": ""},
        {"canonical_commune_code": ""},
        {"resolved_geo_type": ""},
        {"region_code": ""},
    ],
)
def test_invalid_requests_return_standard_422(
    client: TestClient,
    overrides: dict[str, object],
) -> None:
    response = client.post("/predict", json=valid_payload(**overrides))
    assert response.status_code == 422
    assert response.json()["detail"]


def test_extra_input_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json=valid_payload(valeur_fonciere=500_000),
    )
    assert response.status_code == 422


def test_internal_prediction_error_is_clean() -> None:
    class FailingModel:
        def predict(self, _features: object) -> np.ndarray:
            raise RuntimeError("secret /home/person/mlflow.db traceback")

    with TestClient(
        create_app(PredictionService(FailingModel())),
        raise_server_exceptions=False,
    ) as client:
        response = client.post("/predict", json=valid_payload())
    assert response.status_code == 500
    assert response.json() == {"detail": "Model prediction failed."}
    assert "/home/person" not in response.text
    assert "traceback" not in response.text


def test_unavailable_model_returns_clean_503() -> None:
    def fail_loader() -> PredictionService:
        raise ModelUnavailableError("secret /home/person/mlflow.db")

    with TestClient(
        create_app(service_loader=fail_loader),
        raise_server_exceptions=False,
    ) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=valid_payload())
    for response in (health, prediction):
        assert response.status_code == 503
        assert response.json() == {"detail": "Model service unavailable."}
        assert "/home/person" not in response.text


def test_model_loader_runs_once_at_application_startup() -> None:
    calls = []

    def load_once() -> PredictionService:
        calls.append("loaded")
        return PredictionService(FixedModel())

    with TestClient(create_app(service_loader=load_once)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/model").status_code == 200
        assert client.post("/predict", json=valid_payload()).status_code == 200
    assert calls == ["loaded"]


def test_imported_app_defers_default_model_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    create_app()
    assert not (tmp_path / "mlflow.db").exists()
