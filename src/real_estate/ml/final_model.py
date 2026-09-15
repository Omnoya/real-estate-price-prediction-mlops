"""Train the frozen V1 model and evaluate its sealed test exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import catboost
import mlflow
import numpy as np
import pandas as pd

from real_estate.ml.baselines import evaluate_slices
from real_estate.ml.catboost_model import (
    MISSING_CATEGORY,
    MODEL_NAME,
    _load_catboost_api,
    load_catboost_settings,
    prepare_catboost_features,
)
from real_estate.ml.dataset import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    DEFAULT_ML_CONFIG_PATH,
    DERIVED_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TARGET_SOURCE,
    TEST_YEAR,
    TRAIN_YEARS,
    VALIDATION_YEAR,
    MLDataset,
    annual_path,
    concatenate_ml_datasets,
    load_development_data,
    load_ml_config,
    load_ml_year,
)
from real_estate.ml.tracking import (
    configure_tracking,
    load_tracking_settings,
)

MODEL_VERSION = "v1"
FINAL_TRAIN_YEARS = (*TRAIN_YEARS, VALIDATION_YEAR)
EXPECTED_FINAL_TRAIN_ROWS = 2_569_244
EXPECTED_TEST_ROWS = 591_274
MODEL_SELECTION_PATH = Path("reports/model_selection.json")
TEST_EVALUATION_STARTED_TAG = "test_evaluation_started"
TEST_EVALUATED_TAG = "test_evaluated"
NON_PERSISTED_CATBOOST_PARAMETERS = frozenset({"thread_count"})

EXPECTED_MODEL_SELECTION = {
    "baseline": {
        "metrics": {
            "mae_eur_m2": 1147.7414970737116,
            "median_ae_eur_m2": 754.010441767065,
            "rmse_log": 0.6590399945589221,
        },
        "name": "DEPARTMENT_TYPE_MEDIAN",
    },
    "freeze": {
        "additional_changes_from_2024_performance_allowed": False,
        "test_used_for_selection": False,
        "test_year": TEST_YEAR,
    },
    "model_version": MODEL_VERSION,
    "selected_model": {
        "best_iteration": 2999,
        "metrics": {
            "mae_eur_m2": 829.2568229919669,
            "median_ae_eur_m2": 494.46181031372157,
            "rmse_log": 0.5454508466588915,
        },
        "model": MODEL_NAME,
        "training_budget": 3000,
    },
    "selection_protocol": {
        "train_years": list(TRAIN_YEARS),
        "validation_year": VALIDATION_YEAR,
    },
    "status": "frozen",
}


class FinalModelError(RuntimeError):
    """The frozen training or one-shot test contract was violated."""


class TestAlreadyEvaluatedError(FinalModelError):
    """A final model run has already consumed the sealed test set."""


def feature_contract() -> dict[str, object]:
    """Return the exact ordered feature and target contract logged with the model."""
    return {
        "model_version": MODEL_VERSION,
        "source_target": TARGET_SOURCE,
        "target": TARGET_COLUMN,
        "target_transform": "log",
        "categorical_missing_value": MISSING_CATEGORY,
        "boolean_encoding": {"false": 0, "true": 1},
        "features": {
            "numeric": list(NUMERIC_FEATURES),
            "boolean": list(BOOLEAN_FEATURES),
            "categorical": list(CATEGORICAL_FEATURES),
            "derived": list(DERIVED_FEATURES),
            "ordered": list(FEATURE_COLUMNS),
        },
    }


def _feature_contract_sha256() -> str:
    payload = json.dumps(
        feature_contract(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_model_selection(project_root: Path) -> Path:
    """Require the versioned selection record to match the frozen result exactly."""
    root = project_root.resolve()
    path = (root / MODEL_SELECTION_PATH).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FinalModelError(f"Frozen model-selection artifact is missing: {path}")
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalModelError("Cannot read the frozen model-selection artifact.") from error
    if content != EXPECTED_MODEL_SELECTION:
        raise FinalModelError("Model-selection artifact differs from the frozen V1 result.")
    return path


def _git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FinalModelError("Cannot identify the Git commit for the MLflow run.") from error
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise FinalModelError("Git returned an invalid commit identifier.")
    return commit


def _run_tags() -> dict[str, str]:
    return {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "target": TARGET_COLUMN,
        "train_years": ",".join(map(str, FINAL_TRAIN_YEARS)),
        "test_year": str(TEST_YEAR),
        "protocol_frozen": "true",
        TEST_EVALUATION_STARTED_TAG: "false",
        TEST_EVALUATED_TAG: "false",
    }


def _run_parameters(settings: Any, git_commit: str) -> dict[str, object]:
    return {
        **settings.model_parameters,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "target": TARGET_COLUMN,
        "train_years": ",".join(map(str, FINAL_TRAIN_YEARS)),
        "test_year": TEST_YEAR,
        "feature_count": len(FEATURE_COLUMNS),
        "categorical_features": json.dumps(list(CATEGORICAL_FEATURES)),
        "feature_contract_sha256": _feature_contract_sha256(),
        "git_commit": git_commit,
        "catboost_version": catboost.__version__,
        "mlflow_version": mlflow.__version__,
    }


def _log_model(model: Any, artifact_name: str) -> None:
    mlflow.catboost.log_model(model, name=artifact_name)


def _load_model(run_id: str, artifact_name: str) -> Any:
    return mlflow.catboost.load_model(f"runs:/{run_id}/{artifact_name}")


def _log_contract_artifacts(
    config_path: Path,
    selection_path: Path,
) -> None:
    mlflow.log_dict(feature_contract(), "contracts/feature_contract.json")
    mlflow.log_artifact(str(config_path), artifact_path="contracts")
    mlflow.log_artifact(str(selection_path), artifact_path="contracts")


def _final_training_data(config_path: Path, project_root: Path) -> MLDataset:
    config = load_ml_config(config_path)
    development = load_development_data(config, project_root)
    return concatenate_ml_datasets([development.train, development.validation])


def train_final_model(
    config_path: Path = DEFAULT_ML_CONFIG_PATH,
    project_root: Path = Path("."),
    *,
    catboost_api: tuple[Callable[..., Any], Callable[..., Any]] | None = None,
) -> dict[str, object]:
    """Fit exactly 3,000 trees on 2021--2024 and log the frozen run."""
    selection_path = validate_model_selection(project_root)
    settings = load_catboost_settings(config_path)
    if settings.model_parameters["iterations"] != 3000:
        raise FinalModelError("The final V1 training budget must be exactly 3000.")
    training = _final_training_data(config_path, project_root)
    if len(training) != EXPECTED_FINAL_TRAIN_ROWS:
        raise FinalModelError(
            f"Final training row count is {len(training)}; "
            f"expected {EXPECTED_FINAL_TRAIN_ROWS}."
        )

    prepared = prepare_catboost_features(training)
    regressor_factory, pool_factory = catboost_api or _load_catboost_api()
    train_pool = pool_factory(
        data=prepared,
        label=training.target.to_numpy(),
        cat_features=list(CATEGORICAL_FEATURES),
    )
    tracking_settings = load_tracking_settings(config_path)
    tracking = configure_tracking(tracking_settings, project_root)
    git_commit = _git_commit(project_root)

    with mlflow.start_run(
        experiment_id=tracking.experiment_id,
        run_name="catboost-v1-final",
        tags=_run_tags(),
    ) as active_run:
        mlflow.log_params(_run_parameters(settings, git_commit))
        model = regressor_factory(**settings.model_parameters)
        model.fit(train_pool, verbose=False)
        _log_contract_artifacts(config_path, selection_path)
        _log_model(model, tracking.model_artifact_name)
        run_id = active_run.info.run_id

    return {
        "iterations": 3000,
        "model_version": MODEL_VERSION,
        "run_id": run_id,
        "train_rows": len(training),
        "train_years": list(FINAL_TRAIN_YEARS),
    }


def _validate_run_contract(run: Any, settings: Any) -> None:
    expected_tags = _run_tags()
    expected_tags.pop(TEST_EVALUATION_STARTED_TAG)
    expected_tags.pop(TEST_EVALUATED_TAG)
    for key, expected in expected_tags.items():
        if run.data.tags.get(key) != expected:
            raise FinalModelError(f"MLflow run has incompatible tag {key}.")
    for key in (TEST_EVALUATION_STARTED_TAG, TEST_EVALUATED_TAG):
        if run.data.tags.get(key) not in {"false", "true"}:
            raise FinalModelError(f"MLflow run has invalid tag {key}.")

    expected_params = _run_parameters(settings, "")
    expected_params.pop("git_commit")
    expected_params.pop("catboost_version")
    expected_params.pop("mlflow_version")
    for key, expected in expected_params.items():
        if run.data.params.get(key) != str(expected):
            raise FinalModelError(f"MLflow run has incompatible parameter {key}.")
    commit = run.data.params.get("git_commit", "")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise FinalModelError("MLflow run has an invalid git_commit parameter.")
    for key in ("catboost_version", "mlflow_version"):
        if not run.data.params.get(key, "").strip():
            raise FinalModelError(f"MLflow run is missing parameter {key}.")
    if run.info.status != "FINISHED":
        raise FinalModelError("MLflow run must be finished before test evaluation.")


def _validate_loaded_model(model: Any, settings: Any) -> None:
    if type(model).__name__ != MODEL_NAME:
        raise FinalModelError("Loaded MLflow model is not CatBoostRegressor V1.")
    parameters = model.get_params()
    for key, expected in settings.model_parameters.items():
        if key in NON_PERSISTED_CATBOOST_PARAMETERS:
            continue
        if parameters.get(key) != expected:
            raise FinalModelError(f"Loaded model has incompatible parameter {key}.")
    if model.tree_count_ != 3000:
        raise FinalModelError("Loaded model does not contain exactly 3000 trees.")
    if tuple(model.feature_names_) != FEATURE_COLUMNS:
        raise FinalModelError("Loaded model feature order differs from ML V1.")


def _load_sealed_test(config_path: Path, project_root: Path) -> MLDataset:
    """The only workflow function authorized to resolve and open test 2025."""
    config = load_ml_config(config_path)
    return load_ml_year(annual_path(config, project_root, config.test_year), config.test_year)


def _flatten_test_metrics(
    metrics: Mapping[str, Mapping[str, float | int]],
) -> dict[str, float]:
    flattened = {}
    for slice_name, values in metrics.items():
        for metric in ("rmse_log", "mae_eur_m2", "median_ae_eur_m2"):
            flattened[f"test_{slice_name}_{metric}"] = float(values[metric])
    return flattened


def evaluate_final_model(
    run_id: str,
    config_path: Path = DEFAULT_ML_CONFIG_PATH,
    project_root: Path = Path("."),
    *,
    pool_factory: Callable[..., Any] | None = None,
) -> dict[str, object]:
    """Load one frozen run, consume test 2025 once, and log final metrics."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty.")
    validate_model_selection(project_root)
    settings = load_catboost_settings(config_path)
    tracking = configure_tracking(load_tracking_settings(config_path), project_root)
    run = tracking.client.get_run(run_id)
    _validate_run_contract(run, settings)
    model = _load_model(run_id, tracking.model_artifact_name)
    _validate_loaded_model(model, settings)

    evaluation_started = run.data.tags[TEST_EVALUATION_STARTED_TAG] == "true"
    test_evaluated = run.data.tags[TEST_EVALUATED_TAG] == "true"
    if evaluation_started or test_evaluated:
        raise TestAlreadyEvaluatedError(
            f"Run {run_id} has already started or completed test 2025 evaluation."
        )

    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag(TEST_EVALUATION_STARTED_TAG, "true")

    test = _load_sealed_test(config_path, project_root)
    if len(test) != EXPECTED_TEST_ROWS:
        raise FinalModelError(
            f"Final test row count is {len(test)}; expected {EXPECTED_TEST_ROWS}."
        )

    prepared = prepare_catboost_features(test)
    if pool_factory is None:
        _, pool_factory = _load_catboost_api()
    test_pool = pool_factory(
        data=prepared,
        cat_features=list(CATEGORICAL_FEATURES),
    )
    prediction = pd.Series(
        np.asarray(model.predict(test_pool), dtype="float64"),
        name="prediction_log",
    )
    metrics = evaluate_slices(test, prediction)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(_flatten_test_metrics(metrics))
        mlflow.set_tag(TEST_EVALUATED_TAG, "true")

    return {
        "metrics": metrics,
        "model_version": MODEL_VERSION,
        "run_id": run_id,
        "test_rows": len(test),
        "test_year": TEST_YEAR,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ML_CONFIG_PATH)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="Fit and log the frozen model without test data.")
    evaluate = commands.add_parser("evaluate", help="Evaluate one frozen run on test once.")
    evaluate.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch the isolated final-training and one-shot test commands."""
    args = build_parser().parse_args(argv)
    if args.command == "train":
        report = train_final_model(args.config, args.project_root)
    else:
        report = evaluate_final_model(args.run_id, args.config, args.project_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
