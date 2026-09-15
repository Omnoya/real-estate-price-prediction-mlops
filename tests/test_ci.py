"""Static contract checks for the small GitHub Actions workflow."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github/workflows/ci.yml"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_ci_workflow_exists_and_has_expected_triggers_and_permissions() -> None:
    assert WORKFLOW.is_file()
    text = _workflow_text()
    assert "\non:\n" in text
    assert "  push:\n    branches:\n      - main\n" in text
    assert "  pull_request:\n" in text
    assert "permissions:\n  contents: read\n" in text
    assert "runs-on: ubuntu-latest" in text


def test_quality_job_uses_python_312_and_declared_project_dependencies() -> None:
    text = _workflow_text()
    assert "  quality:\n" in text
    assert 'python-version: "3.12"' in text
    assert "cache: pip" in text
    assert 'python -m pip install ".[dev]"' in text
    assert "python -m pip check" in text
    assert "python -m ruff check ." in text
    assert "python -m pytest -q" in text


def test_docker_job_depends_on_quality_and_checks_runtime_properties() -> None:
    text = _workflow_text()
    assert "  docker:\n    needs: quality\n" in text
    assert "docker build --tag real-estate-api:ci ." in text
    assert "docker image inspect --format='{{.Config.User}}'" in text
    assert ' = "app"' in text
    assert "importlib.util.find_spec('mlflow') is None" in text
    assert "import real_estate.api.main" in text


def test_ci_never_pushes_or_uses_real_data_model_or_tracking_artifacts() -> None:
    text = _workflow_text().lower()
    assert "docker push" not in text
    assert "docker compose" not in text
    assert "data/raw" not in text
    assert "data/processed" not in text
    assert "mlflow.db" not in text
    assert "mlartifacts" not in text
    assert "artifacts/serving" not in text
    assert "secrets." not in text
    assert "continue-on-error" not in text
    assert "|| true" not in text
