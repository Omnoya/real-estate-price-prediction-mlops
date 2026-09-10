"""Acquire configured DVF archives with local, reproducible provenance.

This module only downloads and records source bytes. It does not inspect, clean
or transform DVF data, and importing it never starts a download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "data.yaml"


class DownloadError(RuntimeError):
    """Raised when a configured source cannot be safely acquired."""


class ExistingDownloadMismatchError(DownloadError):
    """Raised when an existing file does not match its manifest entry."""


class DownloadValidationError(DownloadError):
    """Raised when downloaded bytes fail integrity checks."""


@dataclass(frozen=True)
class DvfDataConfig:
    """The explicit, project-relative configuration required to acquire DVF."""

    source_name: str
    source_page: str
    years: tuple[int, ...]
    raw_directory: Path
    manifest_filename: str
    connect_timeout_seconds: float
    read_timeout_seconds: float
    chunk_size_bytes: int
    max_attempts: int
    resources: dict[int, str]


@dataclass(frozen=True)
class DownloadResult:
    """Identity and local location of a DVF archive."""

    year: int
    source_url: str
    final_url: str
    filename: str
    bytes_written: int
    sha256: str
    downloaded_at: str
    path: Path
    downloaded: bool


def load_data_config(config_path: Path = DEFAULT_CONFIG_PATH) -> DvfDataConfig:
    """Load and validate the DVF acquisition section from a YAML file."""
    try:
        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        section = content["dvf"]
        resources = {int(year): str(url) for year, url in section["resources"].items()}
        years = tuple(int(year) for year in section["years"])
        raw_directory = Path(section["raw_directory"])
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"Invalid DVF configuration in {config_path}.") from error

    if raw_directory.is_absolute() or any(part == ".." for part in raw_directory.parts):
        raise ValueError("DVF raw_directory must be project-relative.")
    if not years or len(set(years)) != len(years) or set(years) != set(resources):
        raise ValueError("DVF years must exactly match configured resources.")
    if not all(url.startswith(("https://", "http://")) for url in resources.values()):
        raise ValueError("DVF resource URLs must use HTTP(S).")

    try:
        config = DvfDataConfig(
            source_name=str(section["source_name"]),
            source_page=str(section["source_page"]),
            years=years,
            raw_directory=raw_directory,
            manifest_filename=str(section["manifest_filename"]),
            connect_timeout_seconds=float(section["connect_timeout_seconds"]),
            read_timeout_seconds=float(section["read_timeout_seconds"]),
            chunk_size_bytes=int(section["chunk_size_bytes"]),
            max_attempts=int(section["max_attempts"]),
            resources=resources,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid DVF configuration in {config_path}.") from error

    if (
        not config.manifest_filename
        or "/" in config.manifest_filename
        or "\\" in config.manifest_filename
        or config.connect_timeout_seconds <= 0
        or config.read_timeout_seconds <= 0
        or config.chunk_size_bytes <= 0
        or config.max_attempts <= 0
    ):
        raise ValueError("DVF download settings must be positive and relative.")
    return config


def get_dvf_source(config: DvfDataConfig, year: int) -> str:
    """Return the configured resource URL for one DVF year."""
    try:
        return config.resources[year]
    except KeyError as error:
        raise ValueError(f"DVF year {year} is not configured.") from error


def sha256_file(path: Path, chunk_size_bytes: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_raw_directory(config: DvfDataConfig, project_root: Path) -> Path:
    """Resolve the configured project-relative DVF directory."""
    return project_root / config.raw_directory


def _filename_for_year(year: int) -> str:
    """Use a deterministic local archive filename for a configured year."""
    return f"dvf_{year}.zip"


def _load_manifest(manifest_path: Path, config: DvfDataConfig) -> dict[str, Any]:
    """Read a local manifest, or create its in-memory empty representation."""
    if not manifest_path.exists():
        return {
            "source_name": config.source_name,
            "source_page": config.source_page,
            "downloads": {},
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("downloads"), dict):
            raise TypeError("downloads must be an object")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DownloadError(f"Invalid DVF manifest: {manifest_path}") from error
    return manifest


def _write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Atomically replace the manifest after a successful archive replacement."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            temporary_path = Path(file_handle.name)
            json.dump(manifest, file_handle, indent=2, sort_keys=True)
            file_handle.write("\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _entry_to_result(
    entry: dict[str, Any], path: Path, *, downloaded: bool = False
) -> DownloadResult:
    """Convert a validated manifest entry to an idempotent result."""
    required = {"year", "source_url", "final_url", "filename", "bytes", "sha256", "downloaded_at"}
    if not required.issubset(entry):
        raise DownloadError("DVF manifest entry is incomplete.")
    return DownloadResult(
        year=int(entry["year"]),
        source_url=str(entry["source_url"]),
        final_url=str(entry["final_url"]),
        filename=str(entry["filename"]),
        bytes_written=int(entry["bytes"]),
        sha256=str(entry["sha256"]),
        downloaded_at=str(entry["downloaded_at"]),
        path=path,
        downloaded=downloaded,
    )


def _existing_result(
    year: int,
    destination: Path,
    manifest: dict[str, Any],
    source_url: str,
) -> DownloadResult | None:
    """Return a matching archive without a network request, or fail explicitly."""
    entry = manifest["downloads"].get(str(year))
    if not destination.exists() and entry is None:
        return None
    if not destination.exists() or not isinstance(entry, dict):
        raise ExistingDownloadMismatchError(
            f"Existing DVF state for {year} does not match its manifest. Use --force."
        )
    result = _entry_to_result(entry, destination)
    if result.filename != destination.name or result.sha256 != sha256_file(destination):
        raise ExistingDownloadMismatchError(
            f"Existing DVF file for {year} does not match its manifest. Use --force."
        )
    if result.source_url != source_url:
        raise ExistingDownloadMismatchError(
            f"Existing DVF file for {year} has a source_url provenance mismatch. Use --force."
        )
    return result


def _validated_content_length(response: requests.Response) -> int | None:
    """Read a nonnegative Content-Length header, when a server supplied one."""
    content_length = response.headers.get("Content-Length")
    if content_length is None:
        return None
    try:
        expected = int(content_length)
    except ValueError as error:
        raise DownloadValidationError("Invalid Content-Length response header.") from error
    if expected < 0:
        raise DownloadValidationError("Invalid Content-Length response header.")
    return expected


def _download_to_temporary_file(
    source_url: str,
    destination: Path,
    config: DvfDataConfig,
    session: requests.Session,
) -> tuple[Path, int, str, str]:
    """Download valid archive bytes to a sibling temporary file with retries."""
    last_error: requests.RequestException | None = None
    for attempt in range(config.max_attempts):
        temporary_path: Path | None = None
        response: requests.Response | None = None
        keep_temporary_file = False
        try:
            response = session.get(
                source_url,
                stream=True,
                timeout=(config.connect_timeout_seconds, config.read_timeout_seconds),
            )
            response.raise_for_status()
            expected_bytes = _validated_content_length(response)
            digest = hashlib.sha256()
            bytes_written = 0
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".part",
                delete=False,
            ) as file_handle:
                temporary_path = Path(file_handle.name)
                for chunk in response.iter_content(chunk_size=config.chunk_size_bytes):
                    if chunk:
                        file_handle.write(chunk)
                        digest.update(chunk)
                        bytes_written += len(chunk)
                file_handle.flush()
                os.fsync(file_handle.fileno())
            if bytes_written == 0:
                raise DownloadValidationError("Downloaded DVF archive is empty.")
            if expected_bytes is not None and bytes_written != expected_bytes:
                raise DownloadValidationError("Downloaded bytes do not match Content-Length.")
            if not zipfile.is_zipfile(temporary_path):
                raise DownloadValidationError("Downloaded DVF archive is not a valid ZIP file.")
            keep_temporary_file = True
            return temporary_path, bytes_written, digest.hexdigest(), str(response.url)
        except requests.RequestException as error:
            last_error = error
            if attempt + 1 == config.max_attempts:
                raise DownloadError(f"Unable to download DVF from {source_url}.") from error
        except (DownloadValidationError, OSError):
            raise
        finally:
            if response is not None:
                response.close()
            if (
                not keep_temporary_file
                and temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()
    raise DownloadError(f"Unable to download DVF from {source_url}.") from last_error


def _utc_timestamp() -> str:
    """Produce an explicit UTC ISO-8601 timestamp for a manifest entry."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def download_dvf_year(
    year: int,
    config: DvfDataConfig,
    project_root: Path,
    *,
    force: bool = False,
    session: requests.Session | None = None,
) -> DownloadResult:
    """Download one configured DVF ZIP atomically and record its byte identity."""
    source_url = get_dvf_source(config, year)
    raw_directory = _project_raw_directory(config, project_root)
    raw_directory.mkdir(parents=True, exist_ok=True)
    destination = raw_directory / _filename_for_year(year)
    manifest_path = raw_directory / config.manifest_filename
    manifest = _load_manifest(manifest_path, config)
    if not force:
        existing = _existing_result(year, destination, manifest, source_url)
        if existing is not None:
            return existing

    owns_session = session is None
    request_session = requests.Session() if owns_session else session
    temporary_path: Path | None = None
    try:
        temporary_path, bytes_written, digest, final_url = _download_to_temporary_file(
            source_url, destination, config, request_session
        )
        os.replace(temporary_path, destination)
        temporary_path = None
        entry = {
            "year": year,
            "source_url": source_url,
            "final_url": final_url,
            "filename": destination.name,
            "bytes": bytes_written,
            "sha256": digest,
            "downloaded_at": _utc_timestamp(),
        }
        updated_manifest = dict(manifest)
        updated_manifest["source_name"] = config.source_name
        updated_manifest["source_page"] = config.source_page
        updated_manifest["downloads"] = dict(manifest["downloads"])
        updated_manifest["downloads"][str(year)] = entry
        _write_manifest(manifest_path, updated_manifest)
        return _entry_to_result(entry, destination, downloaded=True)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        if owns_session:
            request_session.close()


def download_dvf_years(
    years: list[int] | tuple[int, ...],
    config: DvfDataConfig,
    project_root: Path,
    *,
    force: bool = False,
    session: requests.Session | None = None,
) -> list[DownloadResult]:
    """Acquire selected years in order, preserving each completed manifest entry."""
    return [
        download_dvf_year(year, config, project_root, force=force, session=session)
        for year in years
    ]


def parse_args() -> argparse.Namespace:
    """Parse explicit command-line download choices without downloading yet."""
    parser = argparse.ArgumentParser(description="Download configured official DVF archives.")
    parser.add_argument("--year", type=int, nargs="+", action="append", dest="year_groups")
    parser.add_argument("--force", action="store_true", help="Redownload selected years.")
    return parser.parse_args()


def main() -> None:
    """Run explicitly from the command line to acquire configured DVF years."""
    arguments = parse_args()
    config = load_data_config()
    years = tuple(year for group in arguments.year_groups or [] for year in group) or config.years
    results = download_dvf_years(years, config, DEFAULT_CONFIG_PATH.parents[1], force=arguments.force)
    for result in results:
        status = "downloaded" if result.downloaded else "already verified"
        print(f"{result.year}: {status} ({result.filename}, {result.sha256})")


if __name__ == "__main__":
    main()
