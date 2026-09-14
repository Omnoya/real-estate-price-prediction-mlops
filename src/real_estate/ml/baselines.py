"""Fit deterministic medians on train and evaluate them on validation only."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from real_estate.ml.dataset import (
    DEFAULT_ML_CONFIG_PATH,
    TARGET_COLUMN,
    MLDataset,
    load_development_data,
    load_ml_config,
)

GROUP_COLUMNS = ("code_departement", "code_type_local")
LOCAL_TYPES = (("global", None), ("house", 1), ("apartment", 2))


class BaselineError(RuntimeError):
    """A baseline is unfitted or received invalid prediction data."""


@dataclass(frozen=True)
class GlobalMedianBaseline:
    """One train-only median prediction on the logarithmic target scale."""

    median_log: float

    @classmethod
    def fit(cls, train: MLDataset) -> GlobalMedianBaseline:
        """Fit exactly one statistic from training targets."""
        if not len(train):
            raise BaselineError("Cannot fit a baseline on an empty training dataset.")
        return cls(float(train.target.median()))

    def predict_log(self, data: MLDataset) -> pd.Series:
        """Return a constant prediction without inspecting validation targets."""
        return pd.Series(self.median_log, index=data.features.index, name="prediction_log")


@dataclass(frozen=True)
class DepartmentTypeMedianBaseline:
    """Train-only department/local-type medians with a global fallback."""

    medians: dict[tuple[str, int], float]
    global_median_log: float

    @classmethod
    def fit(cls, train: MLDataset) -> DepartmentTypeMedianBaseline:
        """Compute group and fallback medians from train only."""
        if not len(train):
            raise BaselineError("Cannot fit a baseline on an empty training dataset.")
        frame = train.features.loc[:, GROUP_COLUMNS].copy()
        frame[TARGET_COLUMN] = train.target.to_numpy()
        grouped = frame.groupby(list(GROUP_COLUMNS), dropna=False, sort=True)[TARGET_COLUMN].median()
        medians = {(str(department), int(local_type)): float(value) for (department, local_type), value in grouped.items()}
        return cls(medians, float(train.target.median()))

    def predict_log(self, data: MLDataset) -> pd.Series:
        """Use known train groups and fall back without learning from validation."""
        keys = zip(
            data.features[GROUP_COLUMNS[0]],
            data.features[GROUP_COLUMNS[1]],
            strict=True,
        )
        values = [self.medians.get((str(department), int(local_type)), self.global_median_log) for department, local_type in keys]
        return pd.Series(values, index=data.features.index, name="prediction_log", dtype="float64")

    def unknown_group_count(self, data: MLDataset) -> int:
        """Count rows that will use the train-derived global fallback."""
        keys = zip(
            data.features[GROUP_COLUMNS[0]],
            data.features[GROUP_COLUMNS[1]],
            strict=True,
        )
        return sum((str(department), int(local_type)) not in self.medians for department, local_type in keys)


def regression_metrics(data: MLDataset, prediction_log: pd.Series) -> dict[str, float | int]:
    """Calculate log RMSE and two absolute-error metrics in euros per square metre."""
    predicted = np.asarray(prediction_log, dtype="float64")
    if len(predicted) != len(data) or not np.isfinite(predicted).all():
        raise BaselineError("Predictions must be finite and match the dataset length.")
    predicted_eur = np.exp(predicted)
    if not np.isfinite(predicted_eur).all():
        raise BaselineError("Exponentiated predictions must be finite.")
    log_error = data.target.to_numpy() - predicted
    absolute_error = np.abs(data.target_eur_m2.to_numpy() - predicted_eur)
    return {
        "rows": len(data),
        "rmse_log": float(np.sqrt(np.mean(np.square(log_error)))),
        "mae_eur_m2": float(np.mean(absolute_error)),
        "median_ae_eur_m2": float(np.median(absolute_error)),
    }


def _subset(data: MLDataset, local_type: int | None) -> MLDataset:
    if local_type is None:
        return data
    mask = data.features["code_type_local"].eq(local_type).to_numpy()
    return MLDataset(
        data.features.loc[mask].reset_index(drop=True),
        data.target.loc[mask].reset_index(drop=True),
        data.target_eur_m2.loc[mask].reset_index(drop=True),
    )


def evaluate_slices(data: MLDataset, prediction_log: pd.Series) -> dict[str, dict[str, float | int]]:
    """Report global, house and apartment metrics in a deterministic order."""
    reports = {}
    for name, local_type in LOCAL_TYPES:
        subset = _subset(data, local_type)
        mask = (
            np.ones(len(data), dtype=bool)
            if local_type is None
            else data.features["code_type_local"].eq(local_type).to_numpy()
        )
        reports[name] = regression_metrics(subset, prediction_log.loc[mask].reset_index(drop=True))
    return reports


def run_baselines(config_path: Path = DEFAULT_ML_CONFIG_PATH, project_root: Path = Path(".")) -> dict[str, object]:
    """Fit on configured train years and evaluate only configured validation year."""
    config = load_ml_config(config_path)
    development = load_development_data(config, project_root)
    global_model = GlobalMedianBaseline.fit(development.train)
    grouped_model = DepartmentTypeMedianBaseline.fit(development.train)
    global_prediction = global_model.predict_log(development.validation)
    grouped_prediction = grouped_model.predict_log(development.validation)
    return {
        "train_rows": len(development.train),
        "validation_rows": len(development.validation),
        "target": TARGET_COLUMN,
        "global_median": {
            "median_log": global_model.median_log,
            "metrics": evaluate_slices(development.validation, global_prediction),
        },
        "department_type_median": {
            "groups": len(grouped_model.medians),
            "fallback_global_median_log": grouped_model.global_median_log,
            "validation_fallback_rows": grouped_model.unknown_group_count(development.validation),
            "metrics": evaluate_slices(development.validation, grouped_prediction),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ML_CONFIG_PATH)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run train/validation baselines; no test-evaluation option exists."""
    args = build_parser().parse_args(argv)
    print(json.dumps(run_baselines(args.config, args.project_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
