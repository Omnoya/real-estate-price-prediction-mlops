"""Validate the locked ML V1 allowlist and temporal development loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from real_estate.ml import dataset


def source_row(year: int, **overrides: object) -> dict[str, object]:
    row = {
        "source_year": year,
        "date_mutation": f"{year}-03-14",
        "prix_m2": 2_500.0,
        "surface_reelle_bati": 80.0,
        "nombre_pieces_principales": 4,
        "nombre_lots": 0,
        "surface_terrain": 300.0,
        "has_dependance": False,
        "code_type_local": 1,
        "code_postal": "01000",
        "code_departement": "01",
        "source_code_commune": "01001",
        "canonical_commune_code": "01001",
        "resolved_geo_type": "COM",
        "region_code": "84",
        "valeur_fonciere": 200_000.0,
        "id_mutation": f"{year}-1",
        "longitude": None,
    }
    row.update(overrides)
    return row


def source_schema() -> pa.Schema:
    return pa.schema([
        pa.field("source_year", pa.int32()),
        pa.field("date_mutation", pa.string()),
        pa.field("prix_m2", pa.float64()),
        pa.field("surface_reelle_bati", pa.float64()),
        pa.field("nombre_pieces_principales", pa.int64()),
        pa.field("nombre_lots", pa.int64()),
        pa.field("surface_terrain", pa.float64()),
        pa.field("has_dependance", pa.bool_()),
        pa.field("code_type_local", pa.int64()),
        pa.field("code_postal", pa.string()),
        pa.field("code_departement", pa.string()),
        pa.field("source_code_commune", pa.string()),
        pa.field("canonical_commune_code", pa.string()),
        pa.field("resolved_geo_type", pa.string()),
        pa.field("region_code", pa.string()),
        pa.field("valeur_fonciere", pa.float64()),
        pa.field("id_mutation", pa.string()),
        pa.field("longitude", pa.float64()),
    ])


def config() -> dataset.MLConfig:
    return dataset.MLConfig(
        annual_input="data/dvf_{year}.parquet",
        train_years=(2021, 2022, 2023),
        validation_year=2024,
        test_year=2025,
        target="prix_m2",
        target_transform="log",
        numeric_features=dataset.NUMERIC_FEATURES,
        boolean_features=dataset.BOOLEAN_FEATURES,
        categorical_features=dataset.CATEGORICAL_FEATURES,
        derived_features=dataset.DERIVED_FEATURES,
    )


def write_year(root: Path, year: int, rows: list[dict[str, object]] | None = None, schema: pa.Schema | None = None) -> Path:
    path = root / f"data/dvf_{year}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows or [source_row(year)], schema=schema or source_schema()), path)
    return path


def write_config(root: Path) -> Path:
    path = root / "ml.yaml"
    path.write_text(yaml.safe_dump({
        "data": {"annual_input": "data/dvf_{year}.parquet"},
        "split": {"train_years": [2021, 2022, 2023], "validation_year": 2024, "test_year": 2025},
        "target": "prix_m2",
        "target_transform": "log",
        "features": {
            "numeric": list(dataset.NUMERIC_FEATURES),
            "boolean": list(dataset.BOOLEAN_FEATURES),
            "categorical": list(dataset.CATEGORICAL_FEATURES),
            "derived": list(dataset.DERIVED_FEATURES),
        },
    }), encoding="utf-8")
    return path


def test_project_config_has_exact_locked_split_and_allowlist() -> None:
    actual = dataset.load_ml_config()
    assert actual.train_years == (2021, 2022, 2023)
    assert actual.validation_year == 2024
    assert actual.test_year == 2025
    assert actual.features == dataset.FEATURE_COLUMNS
    assert set(actual.features).isdisjoint(dataset.FORBIDDEN_FEATURE_COLUMNS)


def test_load_synthetic_config(tmp_path: Path) -> None:
    actual = dataset.load_ml_config(write_config(tmp_path))
    assert actual == config()


def test_feature_allowlist_derivations_log_and_null_geography() -> None:
    frame = pd.DataFrame([
        source_row(2023, prix_m2=np.e**2, surface_terrain=None,
                   canonical_commune_code=None, resolved_geo_type=None, region_code=None)
    ])
    actual = dataset.build_ml_dataset(frame, 2023)
    assert tuple(actual.features.columns) == dataset.FEATURE_COLUMNS
    assert set(actual.features).isdisjoint(dataset.FORBIDDEN_FEATURE_COLUMNS)
    assert actual.target.iloc[0] == pytest.approx(2.0)
    assert actual.target_eur_m2.iloc[0] == pytest.approx(np.e**2)
    assert actual.features.loc[0, "mutation_year"] == 2023
    assert actual.features.loc[0, "mutation_month"] == 3
    assert bool(actual.features.loc[0, "surface_terrain_missing"])
    assert bool(actual.features.loc[0, "geography_unresolved"])
    assert pd.isna(actual.features.loc[0, "resolved_geo_type"])


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_target_rejected(bad: float) -> None:
    with pytest.raises(dataset.MLSchemaError, match="strictly positive"):
        dataset.build_ml_dataset(pd.DataFrame([source_row(2023, prix_m2=bad)]), 2023)


def test_train_and_validation_exact_years_preserve_order(tmp_path: Path) -> None:
    cfg = config()
    for year in (2021, 2022, 2023, 2024):
        write_year(tmp_path, year, [source_row(year, prix_m2=float(year))])
    development = dataset.load_development_data(cfg, tmp_path)
    assert development.train.target_eur_m2.tolist() == [2021.0, 2022.0, 2023.0]
    assert development.train.features["mutation_year"].tolist() == [2021, 2022, 2023]
    assert development.validation.target_eur_m2.tolist() == [2024.0]
    assert development.validation.features["mutation_year"].tolist() == [2024]


def test_development_loader_never_opens_2025(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = config()
    for year in (2021, 2022, 2023, 2024):
        write_year(tmp_path, year)
    opened = []
    original = dataset.load_ml_year

    def spy(path: Path, expected_year: int) -> dataset.MLDataset:
        opened.append(expected_year)
        assert expected_year != 2025
        return original(path, expected_year)

    monkeypatch.setattr(dataset, "load_ml_year", spy)
    dataset.load_development_data(cfg, tmp_path)
    assert opened == [2021, 2022, 2023, 2024]


def test_missing_annual_file_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(dataset.MLDatasetError, match="missing"):
        dataset.load_training_data(config(), tmp_path)


def test_wrong_source_year_rejected(tmp_path: Path) -> None:
    path = write_year(tmp_path, 2023, [source_row(2022, date_mutation="2023-01-01")])
    with pytest.raises(dataset.MLSchemaError, match="source_year"):
        dataset.load_ml_year(path, 2023)


def test_date_year_must_match_source_year() -> None:
    with pytest.raises(dataset.MLSchemaError, match="date_mutation year"):
        dataset.build_ml_dataset(pd.DataFrame([source_row(2023, date_mutation="2022-12-31")]), 2023)


def test_missing_schema_column_rejected(tmp_path: Path) -> None:
    fields = [field for field in source_schema() if field.name != "surface_terrain"]
    row = source_row(2023)
    row.pop("surface_terrain")
    path = write_year(tmp_path, 2023, [row], pa.schema(fields))
    with pytest.raises(dataset.MLSchemaError, match="surface_terrain"):
        dataset.load_ml_year(path, 2023)


def test_locked_contract_rejects_changed_split() -> None:
    values = config().__dict__ | {"train_years": (2021, 2022, 2024)}
    with pytest.raises(ValueError, match="locked V1"):
        dataset.MLConfig(**values)
