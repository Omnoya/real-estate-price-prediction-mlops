"""Export and load a self-contained serving bundle for frozen CatBoost V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from catboost import CatBoostRegressor

from real_estate.ml.catboost_model import MODEL_NAME
from real_estate.ml.dataset import (
    CATEGORICAL_FEATURES,
    DEFAULT_ML_CONFIG_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

MODEL_FILENAME = "model.cbm"
MANIFEST_FILENAME = "manifest.json"
MODEL_VERSION = "v1"
FEATURE_CONTRACT_SHA256 = (
    "359c725765a7eee4b23798ed7620c9c2d65973e9a63a03fa1369dfac5e39b7f9"
)
FROZEN_MODEL_PARAMETERS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 3000,
    "learning_rate": 0.05,
    "depth": 8,
    "random_seed": 42,
    "allow_writing_files": False,
}
MANIFEST_FIELDS = frozenset({
    "model",
    "model_version",
    "target",
    "feature_count",
    "feature_names",
    "categorical_features",
    "feature_contract_sha256",
    "loss_function",
    "eval_metric",
    "iterations",
    "learning_rate",
    "depth",
    "random_seed",
    "allow_writing_files",
    "training_git_commit",
    "mlflow_run_id",
    "catboost_version",
    "created_at",
    "model_sha256",
})


class ServingBundleError(RuntimeError):
    """A serving bundle could not be exported or validated safely."""


class BundleExistsError(ServingBundleError):
    """The requested output bundle already exists."""


class BundleValidationError(ServingBundleError):
    """The bundle manifest or CatBoost model violates frozen V1."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA-256 without loading a model file entirely into memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
    except OSError as error:
        raise BundleValidationError("Cannot read serving bundle model.") from error
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_created_at(value: object) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise BundleValidationError("Bundle created_at must be a UTC ISO-8601 value.")
    try:
        timestamp = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise BundleValidationError(
            "Bundle created_at must be a UTC ISO-8601 value."
        ) from error
    if timestamp.utcoffset() != UTC.utcoffset(timestamp):
        raise BundleValidationError("Bundle created_at must use UTC.")


def _validate_manifest(manifest: Mapping[str, object]) -> None:
    if set(manifest) != MANIFEST_FIELDS:
        raise BundleValidationError("Bundle manifest fields are incomplete or unknown.")
    expected = {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "target": TARGET_COLUMN,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": list(FEATURE_COLUMNS),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        **FROZEN_MODEL_PARAMETERS,
    }
    for key, value in expected.items():
        if manifest.get(key) != value or type(manifest.get(key)) is not type(value):
            raise BundleValidationError(f"Bundle manifest has incompatible {key}.")
    if not _valid_git_commit(manifest["training_git_commit"]):
        raise BundleValidationError("Bundle training_git_commit is invalid.")
    run_id = manifest["mlflow_run_id"]
    if not isinstance(run_id, str) or not run_id.strip():
        raise BundleValidationError("Bundle mlflow_run_id is invalid.")
    version = manifest["catboost_version"]
    if not isinstance(version, str) or not version.strip():
        raise BundleValidationError("Bundle catboost_version is invalid.")
    _validate_created_at(manifest["created_at"])
    if not _valid_sha256(manifest["model_sha256"]):
        raise BundleValidationError("Bundle model_sha256 is invalid.")


def read_manifest(bundle_dir: Path) -> dict[str, object]:
    """Read and validate the exact manifest contract from one bundle."""
    path = bundle_dir / MANIFEST_FILENAME
    if not path.is_file() or path.is_symlink():
        raise BundleValidationError("Serving bundle manifest is missing or invalid.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BundleValidationError("Serving bundle manifest is not valid JSON.") from error
    if not isinstance(value, dict):
        raise BundleValidationError("Serving bundle manifest must be a JSON object.")
    _validate_manifest(value)
    return value


def _validate_loaded_bundle_model(model: CatBoostRegressor) -> None:
    if type(model).__name__ != MODEL_NAME:
        raise BundleValidationError("Serving bundle model is not CatBoostRegressor V1.")
    if model.tree_count_ != FROZEN_MODEL_PARAMETERS["iterations"]:
        raise BundleValidationError("Serving bundle model must contain 3000 trees.")
    if tuple(model.feature_names_) != FEATURE_COLUMNS:
        raise BundleValidationError("Serving bundle model feature order is incompatible.")
    parameters = model.get_params()
    for name, expected in FROZEN_MODEL_PARAMETERS.items():
        if parameters.get(name) != expected:
            raise BundleValidationError(
                f"Serving bundle model has incompatible parameter {name}."
            )


def load_serving_bundle(bundle_dir: Path) -> CatBoostRegressor:
    """Verify and load a standalone bundle without consulting MLflow."""
    directory = bundle_dir.expanduser()
    if not directory.is_dir() or directory.is_symlink():
        raise BundleValidationError("Serving bundle directory is missing or invalid.")
    model_path = directory / MODEL_FILENAME
    if not model_path.is_file() or model_path.is_symlink() or model_path.stat().st_size == 0:
        raise BundleValidationError("Serving bundle model is missing or empty.")
    manifest = read_manifest(directory)
    if sha256_file(model_path) != manifest["model_sha256"]:
        raise BundleValidationError("Serving bundle model SHA-256 does not match manifest.")
    try:
        model = CatBoostRegressor()
        model.load_model(str(model_path), format="cbm")
    except Exception as error:
        raise BundleValidationError("Cannot load serving bundle CatBoost model.") from error
    _validate_loaded_bundle_model(model)
    return model


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _created_at(now: datetime) -> str:
    if now.tzinfo is None or now.utcoffset() != UTC.utcoffset(now):
        raise ValueError("Bundle clock must return a timezone-aware UTC datetime.")
    return now.isoformat(timespec="seconds").replace("+00:00", "Z")


def _build_manifest(
    run: Any,
    run_id: str,
    model_sha256: str,
    now: datetime,
) -> dict[str, object]:
    return {
        "allow_writing_files": False,
        "categorical_features": list(CATEGORICAL_FEATURES),
        "catboost_version": run.data.params["catboost_version"],
        "created_at": _created_at(now),
        "depth": 8,
        "eval_metric": "RMSE",
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": list(FEATURE_COLUMNS),
        "iterations": 3000,
        "learning_rate": 0.05,
        "loss_function": "RMSE",
        "mlflow_run_id": run_id,
        "model": MODEL_NAME,
        "model_sha256": model_sha256,
        "model_version": MODEL_VERSION,
        "random_seed": 42,
        "target": TARGET_COLUMN,
        "training_git_commit": run.data.params["git_commit"],
    }


def _write_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory(temporary: Path, output: Path, force: bool) -> None:
    if not output.exists():
        os.replace(temporary, output)
        _fsync_directory(output.parent)
        return
    if not force:
        raise BundleExistsError(f"Serving bundle already exists: {output}")
    backup = output.parent / f".{output.name}.{uuid.uuid4().hex}.backup"
    os.replace(output, backup)
    try:
        os.replace(temporary, output)
    except OSError:
        os.replace(backup, output)
        raise
    shutil.rmtree(backup)
    _fsync_directory(output.parent)


def _validate_export_run(run: Any, config_path: Path) -> Any:
    from real_estate.ml.catboost_model import load_catboost_settings
    from real_estate.ml.final_model import (
        _feature_contract_sha256,
        _validate_run_contract,
    )

    settings = load_catboost_settings(config_path)
    _validate_run_contract(run, settings)
    if _feature_contract_sha256() != FEATURE_CONTRACT_SHA256:
        raise ServingBundleError("Local feature contract differs from frozen V1.")
    if run.data.params.get("feature_contract_sha256") != FEATURE_CONTRACT_SHA256:
        raise ServingBundleError("MLflow run has an incompatible feature contract.")
    return settings


def _validate_export_model(model: Any, settings: Any) -> None:
    from real_estate.ml.final_model import _validate_loaded_model

    _validate_loaded_model(model, settings)


def export_serving_bundle(
    run_id: str,
    output_dir: Path,
    config_path: Path = DEFAULT_ML_CONFIG_PATH,
    project_root: Path = Path("."),
    *,
    force: bool = False,
    clock: Callable[[], datetime] = _utc_now,
) -> dict[str, object]:
    """Export one validated MLflow run to an atomically published CBM bundle."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty.")
    output = output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise BundleExistsError("Serving bundle output must be a regular directory.")
    if output.exists() and not force:
        raise BundleExistsError(f"Serving bundle already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.part"

    from real_estate.ml.final_model import _load_model
    from real_estate.ml.tracking import configure_tracking, load_tracking_settings

    try:
        tracking = configure_tracking(load_tracking_settings(config_path), project_root)
        run = tracking.client.get_run(run_id)
        settings = _validate_export_run(run, config_path)
        model = _load_model(run_id, tracking.model_artifact_name)
        _validate_export_model(model, settings)
        temporary.mkdir()
        model_path = temporary / MODEL_FILENAME
        model.save_model(str(model_path), format="cbm")
        _fsync_file(model_path)
        manifest = _build_manifest(run, run_id, sha256_file(model_path), clock())
        _write_manifest(temporary / MANIFEST_FILENAME, manifest)
        load_serving_bundle(temporary)
        _publish_directory(temporary, output, force)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "model_version": MODEL_VERSION,
        "output_dir": str(output_dir),
        "run_id": run_id,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ML_CONFIG_PATH)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Export a frozen MLflow run.")
    export.add_argument("--run-id", required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Export a validated serving bundle without reading DVF data."""
    args = build_parser().parse_args(argv)
    report = export_serving_bundle(
        args.run_id,
        args.output_dir,
        args.config,
        args.project_root,
        force=args.force,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
