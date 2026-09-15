"""Train CatBoost on 2021--2023 and evaluate it on validation 2024 only.

The 2025 test split is deliberately unreachable from this module's public
orchestration and CLI. CatBoost is imported lazily so the contract can be
tested with synthetic doubles without running a real training job.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from real_estate.ml.baselines import evaluate_slices
from real_estate.ml.dataset import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    DEFAULT_ML_CONFIG_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    MLDataset,
    load_development_data,
    load_ml_config,
)

MISSING_CATEGORY = "__MISSING__"
BOOLEAN_COLUMNS = (
    *BOOLEAN_FEATURES,
    "surface_terrain_missing",
    "geography_unresolved",
)
MODEL_NAME = "CatBoostRegressor"

OFFICIAL_DEPARTMENT_TYPE_BASELINE = {
    "rmse_log": 0.6590399945589221,
    "mae_eur_m2": 1147.7414970737116,
    "median_ae_eur_m2": 754.010441767065,
}


class CatBoostContractError(RuntimeError):
    """The CatBoost validation contract cannot be executed safely."""


class CatBoostDependencyError(CatBoostContractError):
    """The declared CatBoost dependency is unavailable at runtime."""


@dataclass(frozen=True)
class CatBoostSettings:
    """Locked parameters for the first reproducible CatBoost canary."""

    loss_function: str
    eval_metric: str
    iterations: int
    learning_rate: float
    depth: int
    random_seed: int
    thread_count: int
    early_stopping_rounds: int
    allow_writing_files: bool

    def __post_init__(self) -> None:
        expected = (
            "RMSE",
            "RMSE",
            3000,
            0.05,
            8,
            42,
            -1,
            100,
            False,
        )
        actual = (
            self.loss_function,
            self.eval_metric,
            self.iterations,
            self.learning_rate,
            self.depth,
            self.random_seed,
            self.thread_count,
            self.early_stopping_rounds,
            self.allow_writing_files,
        )
        if actual != expected:
            raise ValueError("CatBoost configuration differs from the locked V1 canary.")

    @property
    def model_parameters(self) -> dict[str, object]:
        """Return parameters passed directly to CatBoostRegressor."""
        return {
            "loss_function": self.loss_function,
            "eval_metric": self.eval_metric,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "random_seed": self.random_seed,
            "thread_count": self.thread_count,
            "allow_writing_files": self.allow_writing_files,
        }


def load_catboost_settings(path: Path = DEFAULT_ML_CONFIG_PATH) -> CatBoostSettings:
    """Load and validate the locked CatBoost section."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))["catboost"]
        return CatBoostSettings(
            loss_function=raw["loss_function"],
            eval_metric=raw["eval_metric"],
            iterations=raw["iterations"],
            learning_rate=raw["learning_rate"],
            depth=raw["depth"],
            random_seed=raw["random_seed"],
            thread_count=raw["thread_count"],
            early_stopping_rounds=raw["early_stopping_rounds"],
            allow_writing_files=raw["allow_writing_files"],
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"Invalid CatBoost configuration: {error}") from error


def prepare_catboost_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Prepare one V1 feature frame without learning preprocessing statistics."""
    if tuple(features.columns) != FEATURE_COLUMNS:
        raise CatBoostContractError("CatBoost input differs from the ML V1 allowlist.")

    prepared = features.copy(deep=True)
    for column in CATEGORICAL_FEATURES:
        values = prepared[column].astype("string").fillna(MISSING_CATEGORY)
        prepared[column] = values.astype(object)
    for column in BOOLEAN_COLUMNS:
        if prepared[column].isna().any():
            raise CatBoostContractError(f"Boolean feature {column} cannot be null.")
        prepared[column] = prepared[column].astype("int8")

    if tuple(prepared.columns) != FEATURE_COLUMNS:
        raise AssertionError("CatBoost feature preparation changed the allowlist.")
    return prepared


def prepare_catboost_features(data: MLDataset) -> pd.DataFrame:
    """Prepare an ML dataset with the canonical V1 feature-frame transformer."""
    return prepare_catboost_feature_frame(data.features)


def _load_catboost_api() -> tuple[Callable[..., Any], Callable[..., Any]]:
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as error:
        raise CatBoostDependencyError(
            "CatBoost is required to run the model; install the project dependencies."
        ) from error
    return CatBoostRegressor, Pool


def compare_with_official_baseline(
    model_global_metrics: dict[str, float | int],
) -> dict[str, dict[str, float | bool]]:
    """Compare lower-is-better model metrics with the validated 2024 baseline."""
    comparison = {}
    for metric, baseline_value in OFFICIAL_DEPARTMENT_TYPE_BASELINE.items():
        model_value = float(model_global_metrics[metric])
        comparison[metric] = {
            "baseline_value": baseline_value,
            "model_value": model_value,
            "absolute_delta": model_value - baseline_value,
            "relative_improvement_pct": (
                (baseline_value - model_value) / baseline_value * 100.0
            ),
            "improved": model_value < baseline_value,
        }
    return comparison


def run_catboost_validation(
    config_path: Path = DEFAULT_ML_CONFIG_PATH,
    project_root: Path = Path("."),
    *,
    catboost_api: tuple[Callable[..., Any], Callable[..., Any]] | None = None,
) -> dict[str, object]:
    """Fit on train and evaluate validation; the sealed test is never loaded."""
    ml_config = load_ml_config(config_path)
    settings = load_catboost_settings(config_path)
    development = load_development_data(ml_config, project_root)

    train_features = prepare_catboost_features(development.train)
    validation_features = prepare_catboost_features(development.validation)
    regressor_factory, pool_factory = catboost_api or _load_catboost_api()
    categorical = list(CATEGORICAL_FEATURES)
    train_pool = pool_factory(
        data=train_features,
        label=development.train.target.to_numpy(),
        cat_features=categorical,
    )
    validation_pool = pool_factory(
        data=validation_features,
        label=development.validation.target.to_numpy(),
        cat_features=categorical,
    )

    model = regressor_factory(**settings.model_parameters)
    model.fit(
        train_pool,
        eval_set=validation_pool,
        early_stopping_rounds=settings.early_stopping_rounds,
        use_best_model=True,
        verbose=False,
    )
    prediction = pd.Series(
        np.asarray(model.predict(validation_pool), dtype="float64"),
        name="prediction_log",
    )
    metrics = evaluate_slices(development.validation, prediction)
    best_iteration = model.get_best_iteration()
    if type(best_iteration) is not int or best_iteration < 0:
        raise CatBoostContractError("CatBoost did not report a valid best_iteration.")

    return {
        "train_rows": len(development.train),
        "validation_rows": len(development.validation),
        "target": TARGET_COLUMN,
        "model": MODEL_NAME,
        "parameters": settings.model_parameters,
        "categorical_features": categorical,
        "early_stopping_rounds": settings.early_stopping_rounds,
        "best_iteration": best_iteration,
        "validation_metrics": metrics,
        "official_baseline": {
            "name": "DEPARTMENT_TYPE_MEDIAN",
            "metrics": OFFICIAL_DEPARTMENT_TYPE_BASELINE,
        },
        "baseline_comparison": compare_with_official_baseline(metrics["global"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ML_CONFIG_PATH)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run temporal CatBoost validation without exposing the test split."""
    args = build_parser().parse_args(argv)
    report = run_catboost_validation(args.config, args.project_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
