"""Acquire configured INSEE COG ZIPs and record their local byte identity.

Importing this module performs no I/O. Acquisition does not extract CSVs, select
TYPECOM categories or define any join key with DVF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite
from pathlib import Path
from urllib.parse import urlsplit

import requests
import yaml

from real_estate.data.download import (
    DEFAULT_CONFIG_PATH,
    DownloadError,
    DownloadValidationError,
    ExistingDownloadMismatchError,
    sha256_file,
)


def _http_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


@dataclass(frozen=True)
class CogDataConfig:
    """Explicit COG sources and bounded transfer settings, with relative paths."""

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

    def __post_init__(self) -> None:
        if (
            not self.years
            or any(type(year) is not int or not 1000 <= year <= 9999 for year in self.years)
            or len(set(self.years)) != len(self.years)
            or set(self.years) != set(self.resources)
        ):
            raise ValueError("COG years must exactly match configured resources.")
        if not self.source_name or not _http_url(self.source_page) or not all(
            _http_url(url) for url in self.resources.values()
        ):
            raise ValueError("COG requires a source name and HTTP(S) URLs.")
        if self.raw_directory.is_absolute() or ".." in self.raw_directory.parts:
            raise ValueError("COG raw_directory must be project-relative.")
        if (
            not self.manifest_filename or self.manifest_filename in {".", ".."}
            or "/" in self.manifest_filename or "\\" in self.manifest_filename
            or self.manifest_filename in {f"cog_{year}.zip" for year in self.years}
        ):
            raise ValueError("COG manifest_filename must be a separate relative filename.")
        for value in (self.connect_timeout_seconds, self.read_timeout_seconds):
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise ValueError("COG timeouts must be positive finite numbers.")
        for value in (self.chunk_size_bytes, self.max_attempts):
            if type(value) is not int or value <= 0:
                raise ValueError("COG chunk size and max_attempts must be positive integers.")


@dataclass(frozen=True)
class CogDownloadResult:
    """Acquisition provenance and the local location of one complete COG ZIP."""

    year: int
    source_url: str
    final_url: str
    filename: str
    bytes_written: int
    sha256: str
    downloaded_at: str
    path: Path
    downloaded: bool


def load_cog_config(config_path: Path = DEFAULT_CONFIG_PATH) -> CogDataConfig:
    """Read only the COG section; leave the DVF configuration untouched."""
    try:
        section = yaml.safe_load(config_path.read_text(encoding="utf-8"))["cog"]
        return CogDataConfig(
            source_name=section["source_name"],
            source_page=section["source_page"],
            years=tuple(section["years"]),
            raw_directory=Path(section["raw_directory"]),
            manifest_filename=section["manifest_filename"],
            connect_timeout_seconds=section["connect_timeout_seconds"],
            read_timeout_seconds=section["read_timeout_seconds"],
            chunk_size_bytes=section["chunk_size_bytes"],
            max_attempts=section["max_attempts"],
            resources=dict(section["resources"]),
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"Invalid COG configuration: {error}") from error


def get_cog_source(config: CogDataConfig, year: int) -> str:
    """Return the configured resource URL, without claiming immutability."""
    if type(year) is not int or year not in config.years:
        raise ValueError(f"COG year {year} is not configured.")
    return config.resources[year]


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate manifest key.")
        result[key] = value
    return result


def _load_manifest(path: Path, config: CogDataConfig) -> dict:
    """Start an in-memory manifest only if absent; never reset invalid provenance."""
    if not path.exists():
        return {"source_name": config.source_name, "source_page": config.source_page,
                "downloads": {}}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("downloads"), dict):
            raise TypeError("downloads must be an object")
        return manifest
    except (OSError, TypeError, ValueError) as error:
        raise DownloadError("Invalid COG manifest; existing provenance was preserved.") from error


def _result(entry: dict, path: Path, *, downloaded: bool) -> CogDownloadResult:
    return CogDownloadResult(
        year=entry["year"], source_url=entry["source_url"], final_url=entry["final_url"],
        filename=entry["filename"], bytes_written=entry["bytes"], sha256=entry["sha256"],
        downloaded_at=entry["downloaded_at"], path=path, downloaded=downloaded,
    )


def _existing_result(
    year: int, destination: Path, manifest: dict, source_url: str,
) -> CogDownloadResult | None:
    """Verify provenance, filename, year, size and hash before skipping the network."""
    downloads = manifest["downloads"]
    if not destination.exists() and str(year) not in downloads:
        return None
    entry = downloads.get(str(year))
    try:
        if not destination.is_file() or not isinstance(entry, dict):
            raise ValueError("archive and manifest entry must both exist")
        if entry["source_url"] != source_url:
            raise ValueError("source_url provenance mismatch")
        if type(entry["year"]) is not int or entry["year"] != year:
            raise ValueError("year mismatch")
        if entry["filename"] != destination.name:
            raise ValueError("filename mismatch")
        if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
            raise ValueError("invalid bytes")
        if not _http_url(entry["final_url"]):
            raise ValueError("invalid final_url")
        timestamp = datetime.fromisoformat(entry["downloaded_at"])
        if timestamp.utcoffset() != timedelta(0):
            raise ValueError("downloaded_at must be UTC")
        digest = entry["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise ValueError("invalid sha256")
        if entry["bytes"] != destination.stat().st_size or digest != sha256_file(destination):
            raise ValueError("file size or sha256 does not match manifest")
    except (KeyError, TypeError, ValueError, AttributeError, OSError) as error:
        raise ExistingDownloadMismatchError(
            f"Existing COG state for {year} does not match its manifest: {error}. Use --force."
        ) from error
    return _result(entry, destination, downloaded=False)


def _content_length(response: requests.Response) -> int | None:
    value = response.headers.get("Content-Length")
    if value is None:
        return None
    try:
        length = int(value)
        if length < 0:
            raise ValueError
        return length
    except (TypeError, ValueError) as error:
        raise DownloadValidationError("Invalid Content-Length response header.") from error


def _validate_zip(path: Path, written: int, expected: int | None) -> None:
    """Validate size and ZIP directory structure, without extracting any member."""
    if written == 0:
        raise DownloadValidationError("Downloaded COG archive is empty.")
    if expected is not None and written != expected:
        raise DownloadValidationError("Downloaded bytes do not match Content-Length.")
    try:
        with zipfile.ZipFile(path):
            pass
    except zipfile.BadZipFile as error:
        raise DownloadValidationError("Downloaded COG archive is not a valid ZIP file.") from error


def _download_to_part(
    url: str, destination: Path, config: CogDataConfig, session: requests.Session,
) -> tuple[Path, int, str, str]:
    """Retry network/HTTP failures with a fresh sibling .part for each attempt."""
    for attempt in range(config.max_attempts):
        temporary_path = None
        response = None
        complete = False
        try:
            response = session.get(
                url, stream=True,
                timeout=(config.connect_timeout_seconds, config.read_timeout_seconds),
            )
            response.raise_for_status()
            expected = _content_length(response)
            digest = hashlib.sha256()
            written = 0
            with tempfile.NamedTemporaryFile(
                "wb", dir=destination.parent, prefix=f".{destination.name}.",
                suffix=".part", delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                for chunk in response.iter_content(chunk_size=config.chunk_size_bytes):
                    if chunk:
                        stream.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            _validate_zip(temporary_path, written, expected)
            final_url = str(response.url)
            closing_response, response = response, None
            closing_response.close()
            complete = True
            return temporary_path, written, digest.hexdigest(), final_url
        except requests.RequestException as error:
            if attempt + 1 == config.max_attempts:
                raise DownloadError(f"Unable to download COG after {config.max_attempts} attempts.") from error
        finally:
            try:
                if response is not None:
                    response.close()
            finally:
                if not complete and temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
    raise DownloadError("COG download attempts exhausted.")


def _stage_manifest(path: Path, manifest: dict) -> Path:
    """Flush the complete JSON to a sibling .part before publishing any archive."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".part", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return temporary
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _archive_backup(destination: Path) -> Path:
    """Keep the previous inode available for rollback without copying archive bytes."""
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.previous.",
        suffix=".part", delete=False,
    ) as stream:
        backup = Path(stream.name)
    backup.unlink()
    os.link(destination, backup)
    return backup


def _publish(archive: Path, destination: Path, manifest_path: Path, manifest: dict) -> None:
    """Publish complete files; roll back the archive if manifest replacement fails.

    The two replacements are not a crash-atomic transaction. A process crash
    between them leaves a mismatch that idempotence detects on the next call.
    """
    staged_manifest = _stage_manifest(manifest_path, manifest)
    backup = None
    archive_replaced = False
    try:
        if destination.exists():
            backup = _archive_backup(destination)
        os.replace(archive, destination)
        archive_replaced = True
        os.replace(staged_manifest, manifest_path)
    except BaseException:
        if archive_replaced:
            try:
                if backup is None:
                    destination.unlink()
                else:
                    os.replace(backup, destination)
            except OSError as error:
                # Preserve the last valid inode if even rollback cannot be written.
                recovery = backup
                backup = None
                raise DownloadError(f"COG publication rollback failed; recovery archive: {recovery}") from error
        raise
    finally:
        staged_manifest.unlink(missing_ok=True)
        if backup is not None:
            backup.unlink(missing_ok=True)


def download_cog_year(
    year: int, config: CogDataConfig, project_root: Path, *, force: bool = False,
    session: requests.Session | None = None,
) -> CogDownloadResult:
    """Acquire one configured ZIP explicitly; never extract or transform its data."""
    source_url = get_cog_source(config, year)
    directory = project_root / config.raw_directory
    destination = directory / f"cog_{year}.zip"
    manifest_path = directory / config.manifest_filename
    manifest = _load_manifest(manifest_path, config)
    if not force:
        existing = _existing_result(year, destination, manifest, source_url)
        if existing is not None:
            return existing
    directory.mkdir(parents=True, exist_ok=True)
    owns_session = session is None
    request_session = requests.Session() if owns_session else session
    temporary = None
    try:
        temporary, written, digest, final_url = _download_to_part(
            source_url, destination, config, request_session,
        )
        entry = {
            "year": year, "source_url": source_url, "final_url": final_url,
            "filename": destination.name, "bytes": written, "sha256": digest,
            "downloaded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        updated = manifest | {
            "source_name": config.source_name, "source_page": config.source_page,
            "downloads": manifest["downloads"] | {str(year): entry},
        }
        _publish(temporary, destination, manifest_path, updated)
        return _result(entry, destination, downloaded=True)
    finally:
        try:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        finally:
            if owns_session:
                request_session.close()


def download_cog_years(
    years: list[int] | tuple[int, ...], config: CogDataConfig, project_root: Path, *,
    force: bool = False, session: requests.Session | None = None,
) -> list[CogDownloadResult]:
    """Validate all selected years before acquiring them in the requested order."""
    for year in years:
        get_cog_source(config, year)
    return [download_cog_year(year, config, project_root, force=force, session=session)
            for year in years]


def main(argv: list[str] | None = None) -> int:
    """Acquire explicitly selected years, or all configured COG years when omitted."""
    parser = argparse.ArgumentParser(description="Acquire configured official INSEE COG ZIPs.")
    parser.add_argument("--year", type=int, nargs="+")
    parser.add_argument("--force", action="store_true", help="Explicitly redownload selected years.")
    arguments = parser.parse_args(argv)
    try:
        config = load_cog_config()
        results = download_cog_years(
            tuple(arguments.year) if arguments.year is not None else config.years,
            config, DEFAULT_CONFIG_PATH.parents[1], force=arguments.force,
        )
    except (DownloadError, OSError, ValueError) as error:
        parser.exit(1, f"COG acquisition failed: {error}\n")
    for result in results:
        status = "downloaded" if result.downloaded else "already verified"
        print(f"{result.year}: {status} ({result.filename}, {result.sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
