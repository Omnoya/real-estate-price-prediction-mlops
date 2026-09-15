"""Validated public request and response schemas for model serving."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PredictionRequest(BaseModel):
    """Business inputs needed to reproduce the frozen V1 feature contract."""

    model_config = ConfigDict(extra="forbid")

    surface_reelle_bati: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    nombre_pieces_principales: Annotated[int, Field(ge=0)]
    nombre_lots: Annotated[int, Field(ge=0)]
    surface_terrain: Annotated[
        float | None,
        Field(ge=0, allow_inf_nan=False),
    ] = None
    has_dependance: bool
    code_type_local: Literal[1, 2]
    code_postal: NonEmptyString
    code_departement: NonEmptyString
    source_code_commune: NonEmptyString
    canonical_commune_code: NonEmptyString | None = None
    resolved_geo_type: NonEmptyString | None = None
    region_code: NonEmptyString | None = None
    mutation_date: date


class PredictionResponse(BaseModel):
    """Public price-per-square-metre prediction."""

    predicted_price_m2: float
    model_version: str


class HealthResponse(BaseModel):
    """Minimal readiness response."""

    status: Literal["ok"]
    model_version: str


class ModelResponse(BaseModel):
    """Non-sensitive metadata for the model currently being served."""

    model: str
    model_version: str
    target: str
    feature_count: int
