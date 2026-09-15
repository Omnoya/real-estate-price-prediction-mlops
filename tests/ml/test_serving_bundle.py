"""Test standalone serving bundle export and loading with synthetic models."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor, Pool

from real_estate.api.schemas import PredictionRequest
from real_estate.api.service import (
    build_inference_features,
    load_configured_prediction_service,
)
from real_estate.ml import catboost_model, dataset, final_model, serving_bundle
from tests.ml.test_final_model import write_final_config

FIXED_TIME = datetime(2026, 9, 15, 8, 30, tzinfo=UTC)


@dataclass(frozen=True)
class SyntheticBundleSource:
    root: Path
    config_path: Path
    run_id: str
    bundle_dir: Path


def _request(**overrides: object) -> PredictionRequest:
    values: dict[str, object] = {
        "surface_reelle_bati": 60.0,
        "nombre_pieces_principales": 3,
        "nombre_lots": 1,
        "surface_terrain": 100.0,
        "has_dependance": False,
        "code_type_local": 1,
        "code_postal": "75001",
        "code_departement": "75",
        "source_code_commune": "75101",
        "canonical_commune_code": "75056",
        "resolved_geo_type": "ARM",
        "region_code": "11",
        "mutation_date": date(2025, 2, 1),
    }
    values.update(overrides)
    return PredictionRequest.model_validate(values)


def _fit_model(iterations: int = 3000) -> CatBoostRegressor:
    requests = [
        _request(surface_reelle_bati=40.0, code_postal="75001"),
        _request(surface_reelle_bati=60.0, code_postal="75002"),
        _request(
            surface_reelle_bati=80.0,
            code_type_local=2,
            code_postal="69001",
            code_departement="69",
            source_code_commune="69381",
            canonical_commune_code="69123",
            region_code="84",
        ),
        _request(
            surface_reelle_bati=100.0,
            code_type_local=2,
            code_postal="13001",
            code_departement="13",
            source_code_commune="13201",
            canonical_commune_code="13055",
            region_code="93",
        ),
    ]
    features = pd.concat(
        [build_inference_features(request) for request in requests],
        ignore_index=True,
    )
    parameters = catboost_model.load_catboost_settings().model_parameters | {
        "iterations": iterations
    }
    model = CatBoostRegressor(**parameters)
    model.fit(
        Pool(
            features,
            label=np.log([1800.0, 2200.0, 2800.0, 3400.0]),
            cat_features=list(dataset.CATEGORICAL_FEATURES),
        ),
        verbose=False,
    )
    return model


@pytest.fixture(scope="module")
def synthetic_source(
    tmp_path_factory: pytest.TempPathFactory,
) -> SyntheticBundleSource:
    root = tmp_path_factory.mktemp("serving-bundle")
    config_path = write_final_config(root)
    settings = catboost_model.load_catboost_settings(config_path)
    tracking = final_model.configure_tracking(
        final_model.load_tracking_settings(config_path),
        root,
    )
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking.tracking_uri)
    try:
        model = _fit_model()
        with mlflow.start_run(
            experiment_id=tracking.experiment_id,
            tags=final_model._run_tags(),
        ) as run:
            mlflow.log_params(final_model._run_parameters(settings, "a" * 40))
            mlflow.catboost.log_model(model, name=tracking.model_artifact_name)
            run_id = run.info.run_id
        bundle_dir = root / "bundle"
        serving_bundle.export_serving_bundle(
            run_id,
            bundle_dir,
            config_path,
            root,
            clock=lambda: FIXED_TIME,
        )
        yield SyntheticBundleSource(root, config_path, run_id, bundle_dir)
    finally:
        mlflow.set_tracking_uri(previous_uri)


def _copy_bundle(source: SyntheticBundleSource, destination: Path) -> Path:
    shutil.copytree(source.bundle_dir, destination)
    return destination


def _rewrite_manifest(bundle: Path, **changes: object) -> dict[str, object]:
    path = bundle / serving_bundle.MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def test_export_creates_only_model_and_deterministic_manifest(
    synthetic_source: SyntheticBundleSource,
) -> None:
    bundle = synthetic_source.bundle_dir
    assert {path.name for path in bundle.iterdir()} == {
        "manifest.json",
        "model.cbm",
    }
    manifest = serving_bundle.read_manifest(bundle)
    assert manifest["created_at"] == "2026-09-15T08:30:00Z"
    assert manifest["mlflow_run_id"] == synthetic_source.run_id
    assert manifest["training_git_commit"] == "a" * 40
    assert manifest["feature_names"] == list(dataset.FEATURE_COLUMNS)
    assert manifest["categorical_features"] == list(dataset.CATEGORICAL_FEATURES)
    assert manifest["model_sha256"] == serving_bundle.sha256_file(
        bundle / "model.cbm"
    )
    text = (bundle / "manifest.json").read_text(encoding="utf-8")
    assert text == json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def test_loaded_bundle_makes_real_synthetic_prediction(
    synthetic_source: SyntheticBundleSource,
) -> None:
    predictor = load_configured_prediction_service({
        "REAL_ESTATE_MODEL_BUNDLE_DIR": str(synthetic_source.bundle_dir)
    })
    prediction = predictor.predict_price_m2(_request())
    assert math.isfinite(prediction)
    assert prediction > 0


def test_corrupt_model_is_rejected(
    synthetic_source: SyntheticBundleSource,
    tmp_path: Path,
) -> None:
    bundle = _copy_bundle(synthetic_source, tmp_path / "corrupt-model")
    with (bundle / "model.cbm").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(serving_bundle.BundleValidationError, match="SHA-256"):
        serving_bundle.load_serving_bundle(bundle)


def test_corrupt_manifest_is_rejected(
    synthetic_source: SyntheticBundleSource,
    tmp_path: Path,
) -> None:
    bundle = _copy_bundle(synthetic_source, tmp_path / "corrupt-manifest")
    (bundle / "manifest.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(serving_bundle.BundleValidationError, match="valid JSON"):
        serving_bundle.load_serving_bundle(bundle)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("feature_contract_sha256", "0" * 64, "feature_contract_sha256"),
        ("model_version", "v2", "model_version"),
        ("feature_names", list(reversed(dataset.FEATURE_COLUMNS)), "feature_names"),
    ],
)
def test_incompatible_manifest_contract_is_rejected(
    synthetic_source: SyntheticBundleSource,
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    bundle = _copy_bundle(synthetic_source, tmp_path / field)
    _rewrite_manifest(bundle, **{field: value})
    with pytest.raises(serving_bundle.BundleValidationError, match=message):
        serving_bundle.load_serving_bundle(bundle)


def test_model_with_wrong_tree_count_is_rejected(
    synthetic_source: SyntheticBundleSource,
    tmp_path: Path,
) -> None:
    bundle = _copy_bundle(synthetic_source, tmp_path / "wrong-trees")
    model_path = bundle / "model.cbm"
    _fit_model(iterations=10).save_model(str(model_path), format="cbm")
    _rewrite_manifest(bundle, model_sha256=serving_bundle.sha256_file(model_path))
    with pytest.raises(serving_bundle.BundleValidationError, match="3000 trees"):
        serving_bundle.load_serving_bundle(bundle)


def test_export_refuses_overwrite_and_force_replaces_complete_bundle(
    synthetic_source: SyntheticBundleSource,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "bundle"
    destination.mkdir()
    marker = destination / "old"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(serving_bundle.BundleExistsError):
        serving_bundle.export_serving_bundle(
            synthetic_source.run_id,
            destination,
            synthetic_source.config_path,
            synthetic_source.root,
        )
    assert marker.read_text(encoding="utf-8") == "preserve"

    serving_bundle.export_serving_bundle(
        synthetic_source.run_id,
        destination,
        synthetic_source.config_path,
        synthetic_source.root,
        force=True,
        clock=lambda: FIXED_TIME,
    )
    assert not marker.exists()
    serving_bundle.load_serving_bundle(destination)


def test_export_cleans_temporary_directory_after_failure(
    synthetic_source: SyntheticBundleSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "bundle"
    destination.mkdir()
    marker = destination / "old"
    marker.write_text("preserved", encoding="utf-8")

    def fail_manifest(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic write failure")

    monkeypatch.setattr(serving_bundle, "_write_manifest", fail_manifest)
    with pytest.raises(OSError, match="synthetic write failure"):
        serving_bundle.export_serving_bundle(
            synthetic_source.run_id,
            destination,
            synthetic_source.config_path,
            synthetic_source.root,
            force=True,
        )
    assert marker.read_text(encoding="utf-8") == "preserved"
    assert not list(tmp_path.glob("*.part"))
    assert not list(tmp_path.glob(".*.part"))


def test_cli_dispatches_export_without_real_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "bundle"
    calls: list[tuple[Any, ...]] = []

    def fake_export(*args: Any, **kwargs: Any) -> dict[str, object]:
        calls.append((*args, kwargs))
        return {"model_version": "v1", "output_dir": str(output), "run_id": "run"}

    monkeypatch.setattr(serving_bundle, "export_serving_bundle", fake_export)
    result = serving_bundle.main([
        "--config",
        "configs/ml.yaml",
        "--project-root",
        ".",
        "export",
        "--run-id",
        "run",
        "--output-dir",
        str(output),
        "--force",
    ])
    assert result == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == "run"
    assert calls[0][-1]["force"] is True
