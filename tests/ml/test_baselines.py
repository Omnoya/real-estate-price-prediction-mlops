"""Test train-only median baselines, metrics, slices and public orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from real_estate.ml import baselines, dataset
from tests.ml.test_dataset import source_row, source_schema, write_config


def ml_data(rows: list[dict[str, object]], year: int = 2021) -> dataset.MLDataset:
    return dataset.build_ml_dataset(pd.DataFrame([source_row(year, **row) for row in rows]), year)


def write_year(root: Path, year: int, rows: list[dict[str, object]]) -> Path:
    path = root / f"data/dvf_{year}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    full_rows = [source_row(year, **row) for row in rows]
    pq.write_table(pa.Table.from_pylist(full_rows, schema=source_schema()), path)
    return path


def test_global_median_uses_train_only() -> None:
    train = ml_data([{"prix_m2": 100.0}, {"prix_m2": 400.0}, {"prix_m2": 900.0}])
    validation = ml_data([{"prix_m2": 1.0}, {"prix_m2": 1_000_000.0}], 2024)
    model = baselines.GlobalMedianBaseline.fit(train)
    assert model.median_log == pytest.approx(np.log(400.0))
    assert model.predict_log(validation).tolist() == pytest.approx([np.log(400.0)] * 2)


def test_grouped_medians_and_unknown_fallback_are_train_only() -> None:
    train = ml_data([
        {"code_departement": "01", "code_type_local": 1, "prix_m2": 100.0},
        {"code_departement": "01", "code_type_local": 1, "prix_m2": 900.0},
        {"code_departement": "75", "code_type_local": 2, "prix_m2": 10_000.0},
    ])
    validation = ml_data([
        {"code_departement": "01", "code_type_local": 1, "prix_m2": 1.0},
        {"code_departement": "99", "code_type_local": 2, "prix_m2": 1_000_000.0},
    ], 2024)
    model = baselines.DepartmentTypeMedianBaseline.fit(train)
    expected_group = np.median(np.log([100.0, 900.0]))
    expected_global = np.median(np.log([100.0, 900.0, 10_000.0]))
    assert model.predict_log(validation).tolist() == pytest.approx([expected_group, expected_global])
    assert model.unknown_group_count(validation) == 1


def test_metrics_are_exact_on_both_scales() -> None:
    data = ml_data([{"prix_m2": 100.0}, {"prix_m2": 400.0}])
    prediction = pd.Series(np.log([200.0, 200.0]))
    actual = baselines.regression_metrics(data, prediction)
    assert actual["rows"] == 2
    assert actual["rmse_log"] == pytest.approx(np.log(2.0))
    assert actual["mae_eur_m2"] == pytest.approx(150.0)
    assert actual["median_ae_eur_m2"] == pytest.approx(150.0)
    assert "mape" not in actual


def test_house_and_apartment_slices() -> None:
    data = ml_data([
        {"code_type_local": 1, "prix_m2": 100.0},
        {"code_type_local": 1, "prix_m2": 400.0},
        {"code_type_local": 2, "prix_m2": 900.0},
    ], 2024)
    prediction = pd.Series(np.log([100.0, 200.0, 300.0]))
    report = baselines.evaluate_slices(data, prediction)
    assert list(report) == ["global", "house", "apartment"]
    assert report["global"]["rows"] == 3
    assert report["house"]["rows"] == 2
    assert report["apartment"]["rows"] == 1
    assert report["apartment"]["mae_eur_m2"] == pytest.approx(600.0)


def test_orchestration_and_cli_use_only_train_then_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = {
        2021: [{"prix_m2": 100.0, "code_departement": "01"}],
        2022: [{"prix_m2": 400.0, "code_departement": "01"}],
        2023: [{"prix_m2": 900.0, "code_departement": "75", "code_type_local": 2}],
        2024: [
            {"prix_m2": 400.0, "code_departement": "01"},
            {"prix_m2": 900.0, "code_departement": "99", "code_type_local": 2},
        ],
    }
    for year, values in rows.items():
        write_year(tmp_path, year, values)
    config_path = write_config(tmp_path)
    opened = []
    original = dataset.load_ml_year

    def spy(path: Path, expected_year: int) -> dataset.MLDataset:
        opened.append(expected_year)
        assert expected_year != 2025
        return original(path, expected_year)

    monkeypatch.setattr(dataset, "load_ml_year", spy)
    assert baselines.main(["--config", str(config_path), "--project-root", str(tmp_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert opened == [2021, 2022, 2023, 2024]
    assert output["train_rows"] == 3
    assert output["validation_rows"] == 2
    assert output["target"] == "log_prix_m2"
    assert output["global_median"]["median_log"] == pytest.approx(np.log(400.0))
    assert output["department_type_median"]["groups"] == 2
    assert output["department_type_median"]["validation_fallback_rows"] == 1
    assert output["global_median"]["metrics"]["house"]["rows"] == 1
    assert output["global_median"]["metrics"]["apartment"]["rows"] == 1


def test_fit_rejects_empty_training_data() -> None:
    empty = dataset.MLDataset(
        pd.DataFrame(columns=dataset.FEATURE_COLUMNS),
        pd.Series(dtype="float64", name=dataset.TARGET_COLUMN),
        pd.Series(dtype="float64", name=dataset.TARGET_SOURCE),
    )
    with pytest.raises(baselines.BaselineError):
        baselines.GlobalMedianBaseline.fit(empty)
    with pytest.raises(baselines.BaselineError):
        baselines.DepartmentTypeMedianBaseline.fit(empty)


def test_prediction_length_and_finiteness_are_validated() -> None:
    data = ml_data([{"prix_m2": 100.0}])
    with pytest.raises(baselines.BaselineError):
        baselines.regression_metrics(data, pd.Series([], dtype="float64"))
    with pytest.raises(baselines.BaselineError):
        baselines.regression_metrics(data, pd.Series([np.inf]))
