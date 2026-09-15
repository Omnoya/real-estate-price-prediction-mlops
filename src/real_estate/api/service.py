"""Load, validate and run the frozen CatBoost V1 model without DVF access."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from real_estate.api.schemas import PredictionRequest
from real_estate.ml.catboost_model import (
    MODEL_NAME,
    CatBoostSettings,
    load_catboost_settings,
    prepare_catboost_feature_frame,
)
from real_estate.ml.dataset import (
    DEFAULT_ML_CONFIG_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from real_estate.ml.final_model import (
    MODEL_VERSION,
    _feature_contract_sha256,
    _load_model,
    _validate_loaded_model,
    _validate_run_contract,
)
from real_estate.ml.tracking import load_tracking_settings

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_MODEL_RUN_ID = "f140d75d05504aacad1ea18a09f0f4a4"
EXPECTED_FEATURE_CONTRACT_SHA256 = (
    "359c725765a7eee4b23798ed7620c9c2d65973e9a63a03fa1369dfac5e39b7f9"
)


class ModelServiceError(RuntimeError):
    """The frozen model cannot be loaded or safely used for prediction."""


class ModelUnavailableError(ModelServiceError):
    """The configured MLflow run or model is unavailable or incompatible."""


class PredictionError(ModelServiceError):
    """The loaded model failed to return one valid price prediction."""


@dataclass(frozen=True)
class ServingSettings:
    """Runtime location of the frozen MLflow run."""

    tracking_uri: str = DEFAULT_TRACKING_URI
    run_id: str = DEFAULT_MODEL_RUN_ID

    def __post_init__(self) -> None:
        if not self.tracking_uri.strip():
            raise ValueError("MLflow tracking URI must not be empty.")
        if not self.run_id.strip():
            raise ValueError("MLflow model run ID must not be empty.")

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> ServingSettings:
        """Read serving overrides without embedding a user-specific path."""
        values = os.environ if environ is None else environ
        return cls(
            tracking_uri=values.get(
                "REAL_ESTATE_MLFLOW_TRACKING_URI",
                DEFAULT_TRACKING_URI,
            ),
            run_id=values.get("REAL_ESTATE_MODEL_RUN_ID", DEFAULT_MODEL_RUN_ID),
        )


@dataclass(frozen=True)
class ModelMetadata:
    """Safe public metadata for a validated serving model."""

    model: str = MODEL_NAME
    model_version: str = MODEL_VERSION
    target: str = TARGET_COLUMN
    feature_count: int = len(FEATURE_COLUMNS)


def build_inference_features(request: PredictionRequest) -> pd.DataFrame:
    """Derive and prepare the exact ordered V1 features for one request."""
    feature_values = {
        "surface_reelle_bati": request.surface_reelle_bati,
        "nombre_pieces_principales": request.nombre_pieces_principales,
        "nombre_lots": request.nombre_lots,
        "surface_terrain": (
            np.nan if request.surface_terrain is None else request.surface_terrain
        ),
        "has_dependance": request.has_dependance,
        "code_type_local": request.code_type_local,
        "code_postal": request.code_postal,
        "code_departement": request.code_departement,
        "source_code_commune": request.source_code_commune,
        "canonical_commune_code": request.canonical_commune_code,
        "resolved_geo_type": request.resolved_geo_type,
        "region_code": request.region_code,
        "mutation_year": request.mutation_date.year,
        "mutation_month": request.mutation_date.month,
        "surface_terrain_missing": request.surface_terrain is None,
        "geography_unresolved": request.resolved_geo_type is None,
    }
    raw = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)
    return prepare_catboost_feature_frame(raw)


@dataclass
class PredictionService:
    """Validated model plus deterministic feature and prediction handling."""

    model: Any
    metadata: ModelMetadata = ModelMetadata()

    def predict_price_m2(self, request: PredictionRequest) -> float:
        """Return exp(model prediction), requiring one finite positive price."""
        features = build_inference_features(request)
        try:
            values = np.asarray(self.model.predict(features), dtype="float64").reshape(-1)
        except Exception as error:
            raise PredictionError("Model prediction failed.") from error
        if values.size != 1 or not np.isfinite(values[0]):
            raise PredictionError("Model returned an invalid log-price prediction.")
        try:
            price = math.exp(float(values[0]))
        except OverflowError as error:
            raise PredictionError("Model prediction is outside the finite range.") from error
        if not math.isfinite(price) or price <= 0:
            raise PredictionError("Model returned an invalid price prediction.")
        return price


def _validate_frozen_run(run: Any, config_path: Path) -> CatBoostSettings:
    settings = load_catboost_settings(config_path)
    if _feature_contract_sha256() != EXPECTED_FEATURE_CONTRACT_SHA256:
        raise ModelUnavailableError("Local feature contract differs from frozen V1.")
    _validate_run_contract(run, settings)
    if run.data.params.get("feature_contract_sha256") != (
        EXPECTED_FEATURE_CONTRACT_SHA256
    ):
        raise ModelUnavailableError("MLflow run has an incompatible feature contract.")
    return settings


def load_prediction_service(
    settings: ServingSettings | None = None,
    config_path: Path = DEFAULT_ML_CONFIG_PATH,
    *,
    client_factory: Callable[..., Any] = MlflowClient,
    model_loader: Callable[[str, str], Any] = _load_model,
) -> PredictionService:
    """Load one MLflow model and validate every frozen serving invariant."""
    try:
        runtime = settings or ServingSettings.from_environment()
        tracking = load_tracking_settings(config_path)
        mlflow.set_tracking_uri(runtime.tracking_uri)
        client = client_factory(tracking_uri=runtime.tracking_uri)
        run = client.get_run(runtime.run_id)
        model_settings = _validate_frozen_run(run, config_path)
        model = model_loader(runtime.run_id, tracking.model_artifact_name)
        _validate_loaded_model(model, model_settings)
    except ModelUnavailableError:
        raise
    except Exception as error:
        raise ModelUnavailableError(
            "Frozen model is unavailable or incompatible."
        ) from error
    return PredictionService(model=model)
