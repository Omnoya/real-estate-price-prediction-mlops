"""Build the allowlisted ML V1 dataset from annual DVF geography Parquets.

Development loading is deliberately limited to train 2021--2023 and validation
2024. The sealed 2025 test year is metadata only in this module; evaluating it
will require a separate, explicit command in a later project phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Formatter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

DEFAULT_ML_CONFIG_PATH = Path("configs/ml.yaml")

TRAIN_YEARS = (2021, 2022, 2023)
VALIDATION_YEAR = 2024
TEST_YEAR = 2025
TARGET_SOURCE = "prix_m2"
TARGET_TRANSFORM = "log"
TARGET_COLUMN = "log_prix_m2"

NUMERIC_FEATURES = (
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "nombre_lots",
    "surface_terrain",
)
BOOLEAN_FEATURES = ("has_dependance",)
CATEGORICAL_FEATURES = (
    "code_type_local",
    "code_postal",
    "code_departement",
    "source_code_commune",
    "canonical_commune_code",
    "resolved_geo_type",
    "region_code",
)
DERIVED_FEATURES = (
    "mutation_year",
    "mutation_month",
    "surface_terrain_missing",
    "geography_unresolved",
)
FEATURE_COLUMNS = (
    *NUMERIC_FEATURES,
    *BOOLEAN_FEATURES,
    *CATEGORICAL_FEATURES,
    *DERIVED_FEATURES,
)
FORBIDDEN_FEATURE_COLUMNS = frozenset({
    "prix_m2",
    "valeur_fonciere",
    "id_mutation",
    "id_parcelle",
    "numero_disposition",
    "source_row_count",
    "source_year",
    "date_mutation",
    "nom_commune",
    "source_geo_label",
    "canonical_commune_label",
    "longitude",
    "latitude",
    "cog_year",
    "department_code",
})
SOURCE_COLUMNS = (
    "source_year",
    "date_mutation",
    TARGET_SOURCE,
    *NUMERIC_FEATURES,
    *BOOLEAN_FEATURES,
    *CATEGORICAL_FEATURES,
)


class MLDatasetError(RuntimeError):
    """The ML contract cannot be satisfied by the requested source data."""


class MLSchemaError(MLDatasetError):
    """An annual source has an incompatible schema or values."""


@dataclass(frozen=True)
class MLConfig:
    """Locked V1 split, target, allowlist and annual path template."""

    annual_input: str
    train_years: tuple[int, ...]
    validation_year: int
    test_year: int
    target: str
    target_transform: str
    numeric_features: tuple[str, ...]
    boolean_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    derived_features: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = (
            TRAIN_YEARS,
            VALIDATION_YEAR,
            TEST_YEAR,
            TARGET_SOURCE,
            TARGET_TRANSFORM,
            NUMERIC_FEATURES,
            BOOLEAN_FEATURES,
            CATEGORICAL_FEATURES,
            DERIVED_FEATURES,
        )
        actual = (
            self.train_years,
            self.validation_year,
            self.test_year,
            self.target,
            self.target_transform,
            self.numeric_features,
            self.boolean_features,
            self.categorical_features,
            self.derived_features,
        )
        if actual != expected:
            raise ValueError("ML configuration differs from the locked V1 contract.")
        if self.validation_year in self.train_years or self.test_year in {
            *self.train_years,
            self.validation_year,
        }:
            raise ValueError("Train, validation and test years must be disjoint.")
        _validate_template(self.annual_input)

    @property
    def features(self) -> tuple[str, ...]:
        """Return the authoritative feature order."""
        return (
            *self.numeric_features,
            *self.boolean_features,
            *self.categorical_features,
            *self.derived_features,
        )


@dataclass(frozen=True)
class MLDataset:
    """Allowlisted features with source-scale and transformed targets."""

    features: pd.DataFrame
    target: pd.Series
    target_eur_m2: pd.Series

    def __post_init__(self) -> None:
        if tuple(self.features.columns) != FEATURE_COLUMNS:
            raise ValueError("MLDataset features must exactly match the V1 allowlist.")
        if not (len(self.features) == len(self.target) == len(self.target_eur_m2)):
            raise ValueError("Features and targets must have the same row count.")

    def __len__(self) -> int:
        return len(self.features)


@dataclass(frozen=True)
class DevelopmentData:
    """Train and validation only; no test data can be attached implicitly."""

    train: MLDataset
    validation: MLDataset


def _validate_template(template: str) -> None:
    if not isinstance(template, str):
        raise TypeError("annual_input must be a string template.")
    fields = [
        (name, spec, conversion)
        for _, name, spec, conversion in Formatter().parse(template)
        if name is not None
    ]
    if fields != [("year", "", None)]:
        raise ValueError("annual_input must contain exactly one plain {year} field.")
    path = Path(template.format(year=2021))
    if path.is_absolute() or ".." in path.parts or path.suffix != ".parquet":
        raise ValueError("annual_input must be a project-relative Parquet path.")


def load_ml_config(path: Path = DEFAULT_ML_CONFIG_PATH) -> MLConfig:
    """Load and validate the exact ML V1 contract."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        split = raw["split"]
        features = raw["features"]
        return MLConfig(
            annual_input=raw["data"]["annual_input"],
            train_years=tuple(split["train_years"]),
            validation_year=split["validation_year"],
            test_year=split["test_year"],
            target=raw["target"],
            target_transform=raw["target_transform"],
            numeric_features=tuple(features["numeric"]),
            boolean_features=tuple(features["boolean"]),
            categorical_features=tuple(features["categorical"]),
            derived_features=tuple(features["derived"]),
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"Invalid ML configuration: {error}") from error


def annual_path(config: MLConfig, project_root: Path, year: int) -> Path:
    """Resolve one configured source while preventing project-root escape."""
    allowed = {*config.train_years, config.validation_year, config.test_year}
    if type(year) is not int or year not in allowed:
        raise ValueError(f"ML year {year} is not configured.")
    root = project_root.resolve()
    path = (root / config.annual_input.format(year=year)).resolve()
    if not path.is_relative_to(root):
        raise ValueError("ML source path must stay within the project root.")
    return path


def validate_source_schema(schema: pa.Schema) -> None:
    """Validate required source types while allowing unused source columns."""
    missing = sorted(set(SOURCE_COLUMNS) - set(schema.names))
    if missing:
        raise MLSchemaError(f"ML source columns are missing: {', '.join(missing)}")
    numeric = {TARGET_SOURCE, *NUMERIC_FEATURES}
    for name in numeric:
        if not (pa.types.is_integer(schema.field(name).type) or pa.types.is_floating(schema.field(name).type)):
            raise MLSchemaError(f"ML source column {name} must be numeric.")
    if not pa.types.is_integer(schema.field("source_year").type):
        raise MLSchemaError("ML source source_year must be an integer.")
    if not (pa.types.is_string(schema.field("date_mutation").type) or pa.types.is_date(schema.field("date_mutation").type)):
        raise MLSchemaError("ML source date_mutation must be a string or date.")
    if not pa.types.is_boolean(schema.field("has_dependance").type):
        raise MLSchemaError("ML source has_dependance must be boolean.")
    if not pa.types.is_integer(schema.field("code_type_local").type):
        raise MLSchemaError("ML source code_type_local must be an integer.")
    for name in set(CATEGORICAL_FEATURES) - {"code_type_local"}:
        if not pa.types.is_string(schema.field(name).type):
            raise MLSchemaError(f"ML source column {name} must be a string.")


def _read_source(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise MLDatasetError(f"ML annual source is missing: {path}")
    try:
        parquet = pq.ParquetFile(path)
        validate_source_schema(parquet.schema_arrow)
        return parquet.read(columns=list(SOURCE_COLUMNS)).to_pandas()
    except (OSError, pa.ArrowException) as error:
        raise MLDatasetError(f"Cannot read ML annual source: {path}") from error


def build_ml_dataset(source: pd.DataFrame, expected_year: int) -> MLDataset:
    """Validate one year and derive targets/features without encoding or filtering."""
    missing = sorted(set(SOURCE_COLUMNS) - set(source.columns))
    if missing:
        raise MLSchemaError(f"ML source columns are missing: {', '.join(missing)}")
    if source.empty:
        raise MLSchemaError("ML source must contain at least one row.")
    years = source["source_year"]
    if years.isna().any() or not years.eq(expected_year).all():
        raise MLSchemaError(f"source_year must uniformly equal {expected_year}.")
    try:
        dates = pd.to_datetime(source["date_mutation"], format="%Y-%m-%d", errors="raise")
    except (TypeError, ValueError) as error:
        raise MLSchemaError("date_mutation must contain valid ISO dates.") from error
    if not dates.dt.year.eq(expected_year).all():
        raise MLSchemaError("date_mutation year must equal source_year.")

    target_eur = pd.to_numeric(source[TARGET_SOURCE], errors="coerce").astype("float64")
    valid_target = target_eur.notna() & np.isfinite(target_eur) & target_eur.gt(0)
    if not valid_target.all():
        raise MLSchemaError("prix_m2 must be non-null, finite and strictly positive.")

    features = source.loc[:, [*NUMERIC_FEATURES, *BOOLEAN_FEATURES, *CATEGORICAL_FEATURES]].copy()
    features["mutation_year"] = dates.dt.year.astype("int16")
    features["mutation_month"] = dates.dt.month.astype("int8")
    features["surface_terrain_missing"] = features["surface_terrain"].isna()
    features["geography_unresolved"] = features["resolved_geo_type"].isna()
    features = features.loc[:, FEATURE_COLUMNS]
    if set(features.columns) & FORBIDDEN_FEATURE_COLUMNS:
        raise AssertionError("A forbidden column escaped the feature allowlist.")
    target = pd.Series(np.log(target_eur.to_numpy()), index=source.index, name=TARGET_COLUMN)
    return MLDataset(features.reset_index(drop=True), target.reset_index(drop=True), target_eur.rename(TARGET_SOURCE).reset_index(drop=True))


def load_ml_year(path: Path, expected_year: int) -> MLDataset:
    """Load one explicit annual path; callers decide its split role."""
    return build_ml_dataset(_read_source(path), expected_year)


def _concat(datasets: list[MLDataset]) -> MLDataset:
    if not datasets:
        raise ValueError("At least one ML dataset is required.")
    return MLDataset(
        pd.concat([item.features for item in datasets], ignore_index=True),
        pd.concat([item.target for item in datasets], ignore_index=True),
        pd.concat([item.target_eur_m2 for item in datasets], ignore_index=True),
    )


def load_training_data(config: MLConfig, project_root: Path) -> MLDataset:
    """Load only 2021--2023 in chronological order."""
    if config.test_year in config.train_years or config.validation_year in config.train_years:
        raise ValueError("Training years overlap a held-out year.")
    return _concat([
        load_ml_year(annual_path(config, project_root, year), year)
        for year in config.train_years
    ])


def load_validation_data(config: MLConfig, project_root: Path) -> MLDataset:
    """Load validation 2024 without consulting the sealed test path."""
    if config.validation_year == config.test_year:
        raise ValueError("Validation and test years must differ.")
    year = config.validation_year
    return load_ml_year(annual_path(config, project_root, year), year)


def load_development_data(config: MLConfig, project_root: Path) -> DevelopmentData:
    """Load train then validation; this API has no test-loading branch."""
    return DevelopmentData(
        train=load_training_data(config, project_root),
        validation=load_validation_data(config, project_root),
    )
