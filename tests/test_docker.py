"""Small static checks for the serving-only Docker configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_dockerfile_is_non_root_and_starts_expected_uvicorn_command() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.startswith("FROM python:3.12-slim\n")
    assert "\nUSER app\n" in dockerfile
    assert 'CMD ["python", "-m", "uvicorn", "real_estate.api.main:app"' in dockerfile
    assert "--host\", \"0.0.0.0\"" in dockerfile
    assert "--port\", \"8000\"" in dockerfile


def test_dockerfile_uses_stdlib_healthcheck_and_does_not_copy_model_or_data() -> None:
    lines = (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
    copies = [line for line in lines if line.startswith("COPY ")]
    copied_sources = [line.split()[1] for line in copies]
    assert all("/app/model" not in line for line in copies)
    assert all(
        not source.startswith(("data/", "tests/", "notebooks/", "artifacts/"))
        for source in copied_sources
    )
    assert all("final_model.py" not in source for source in copied_sources)
    assert all("tracking.py" not in source for source in copied_sources)
    dockerfile = "\n".join(lines)
    assert "urllib.request" in dockerfile
    assert "curl" not in dockerfile
    assert "REAL_ESTATE_MODEL_BUNDLE_DIR=/app/model" in dockerfile
    requirements = (ROOT / "requirements-serving.txt").read_text(encoding="utf-8")
    assert "mlflow" not in requirements.lower()


def test_dockerignore_excludes_sensitive_and_large_local_content() -> None:
    patterns = set(
        (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    )
    expected = {
        ".git/",
        ".github/",
        ".venv/",
        "**pycache**",
        ".pytest_cache/",
        ".ruff_cache/",
        "data/",
        "mlflow.db",
        "mlartifacts/",
        "artifacts/",
        "notebooks/",
        "tests/",
        "*.pyc",
        "*.pyo",
        "*.pyd",
    }
    assert expected <= patterns


def test_compose_mounts_only_the_bundle_read_only() -> None:
    path = ROOT / "compose.yaml"
    compose = yaml.safe_load(path.read_text(encoding="utf-8"))
    api = compose["services"]["api"]
    assert api["ports"] == ["8000:8000"]
    assert api["volumes"] == ["./artifacts/serving/v1:/app/model:ro"]
    assert api["environment"] == {
        "REAL_ESTATE_MODEL_BUNDLE_DIR": "/app/model"
    }
    text = path.read_text(encoding="utf-8").lower()
    assert "dvf" not in text
    assert "mlflow" not in text
