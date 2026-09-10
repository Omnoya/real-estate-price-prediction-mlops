"""Test DVF acquisition without making HTTP requests outside the test process."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import requests

from real_estate.data import download as download_module
from real_estate.data.download import (
    DEFAULT_CONFIG_PATH,
    DownloadError,
    DownloadValidationError,
    DvfDataConfig,
    ExistingDownloadMismatchError,
    download_dvf_year,
    get_dvf_source,
    load_data_config,
    sha256_file,
)


class FakeResponse:
    """Small HTTP response double that supports requests' streaming interface."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        url: str = "https://files.example.test/dvf.zip",
        content_length: int | None = None,
        status_error: requests.RequestException | None = None,
        stream_error: requests.RequestException | None = None,
    ) -> None:
        self._chunks = chunks
        self._status_error = status_error
        self._stream_error = stream_error
        self.url = url
        self.headers: dict[str, str] = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self.chunk_sizes: list[int] = []
        self.closed = False

    def raise_for_status(self) -> None:
        """Raise the configured HTTP error, if any."""
        if self._status_error is not None:
            raise self._status_error

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        """Yield configured chunks and optionally fail during streaming."""
        self.chunk_sizes.append(chunk_size)
        yield from self._chunks
        if self._stream_error is not None:
            raise self._stream_error

    def close(self) -> None:
        """Track response cleanup."""
        self.closed = True


class FakeSession:
    """Return predefined responses and retain every request invocation."""

    def __init__(self, outcomes: list[FakeResponse | requests.RequestException]) -> None:
        self._outcomes = outcomes
        self.calls: list[tuple[str, bool, tuple[float, float]]] = []
        self.closed = False

    def get(
        self, url: str, *, stream: bool, timeout: tuple[float, float]
    ) -> FakeResponse:
        """Return one fake outcome per request."""
        self.calls.append((url, stream, timeout))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, requests.RequestException):
            raise outcome
        return outcome

    def close(self) -> None:
        """Track whether the caller closes this injected session."""
        self.closed = True


@pytest.fixture
def config() -> DvfDataConfig:
    """Provide a small explicit configuration rooted under tmp_path in each test."""
    return DvfDataConfig(
        source_name="DVF",
        source_page="https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres",
        years=(2021,),
        raw_directory=Path("data/raw/dvf"),
        manifest_filename="manifest.json",
        connect_timeout_seconds=1,
        read_timeout_seconds=2,
        chunk_size_bytes=3,
        max_attempts=1,
        resources={2021: "https://www.data.gouv.fr/api/1/datasets/r/test-2021"},
    )


@pytest.fixture
def zip_bytes() -> bytes:
    """Build a valid but tiny ZIP payload in memory."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("valeursfoncieres-2021.txt", "synthetic DVF")
    return buffer.getvalue()


def test_configured_source_is_explicit_and_unknown_year_fails() -> None:
    """The committed config matches all five currently selected resources."""
    data_config = load_data_config(DEFAULT_CONFIG_PATH)
    expected_resources = {
        2021: "https://www.data.gouv.fr/api/1/datasets/r/947677ab-ad21-48f4-a9ac-ad217c99cf39",
        2022: "https://www.data.gouv.fr/api/1/datasets/r/be6e092d-292a-4568-90bf-4254a261ff3b",
        2023: "https://www.data.gouv.fr/api/1/datasets/r/025b9d29-8efb-40bb-8ce6-5bddf97a4e51",
        2024: "https://www.data.gouv.fr/api/1/datasets/r/99a26050-b94f-4ffc-9eb0-73ed28a895d1",
        2025: "https://www.data.gouv.fr/api/1/datasets/r/902db087-b0eb-4cbb-a968-0b499bde5bc4",
    }

    assert data_config.years == (2021, 2022, 2023, 2024, 2025)
    assert data_config.resources == expected_resources
    for year, source_url in expected_resources.items():
        assert get_dvf_source(data_config, year) == source_url
    for year in (2019, 2020):
        with pytest.raises(ValueError, match="not configured"):
            get_dvf_source(data_config, year)


def test_successful_download_creates_hash_manifest_and_records_redirect(
    tmp_path: Path, config: DvfDataConfig, zip_bytes: bytes
) -> None:
    """A completed archive atomically becomes the deterministic local file."""
    response = FakeResponse(
        [zip_bytes[:10], zip_bytes[10:]],
        url="https://files.data.gouv.fr/dvf/2021.zip",
        content_length=len(zip_bytes),
    )
    session = FakeSession([response])

    result = download_dvf_year(2021, config, tmp_path, session=session)

    destination = tmp_path / "data/raw/dvf/dvf_2021.zip"
    manifest_path = tmp_path / "data/raw/dvf/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["downloads"]["2021"]
    expected_sha256 = hashlib.sha256(zip_bytes).hexdigest()
    assert result.downloaded is True
    assert result.path == destination
    assert destination.read_bytes() == zip_bytes
    assert result.sha256 == expected_sha256 == sha256_file(destination)
    assert entry["source_url"] == config.resources[2021]
    assert entry["final_url"] == response.url
    assert entry["filename"] == "dvf_2021.zip"
    assert entry["bytes"] == len(zip_bytes)
    assert entry["sha256"] == expected_sha256
    assert entry["downloaded_at"].endswith("Z")
    assert response.chunk_sizes == [config.chunk_size_bytes]
    assert response.closed is True
    assert session.calls == [
        (config.resources[2021], True, (config.connect_timeout_seconds, config.read_timeout_seconds))
    ]


def test_matching_existing_file_and_manifest_skip_the_network(
    tmp_path: Path, config: DvfDataConfig, zip_bytes: bytes
) -> None:
    """The manifest SHA-256, not existence alone, enables idempotence."""
    download_dvf_year(2021, config, tmp_path, session=FakeSession([FakeResponse([zip_bytes])]))
    session = FakeSession([])

    result = download_dvf_year(2021, config, tmp_path, session=session)

    assert result.downloaded is False
    assert session.calls == []


def test_changed_configured_source_url_fails_without_a_network_request(
    tmp_path: Path, config: DvfDataConfig, zip_bytes: bytes
) -> None:
    """An archive cannot be reused after its configured provenance changes."""
    download_dvf_year(2021, config, tmp_path, session=FakeSession([FakeResponse([zip_bytes])]))
    changed_config = replace(
        config,
        resources={2021: "https://www.data.gouv.fr/api/1/datasets/r/different-resource"},
    )
    session = FakeSession([])

    with pytest.raises(ExistingDownloadMismatchError, match="source_url provenance"):
        download_dvf_year(2021, changed_config, tmp_path, session=session)

    assert session.calls == []
    assert session.closed is False


def test_injected_session_is_not_closed(
    tmp_path: Path, config: DvfDataConfig, zip_bytes: bytes
) -> None:
    """The caller, not the downloader, owns an injected session."""
    session = FakeSession([FakeResponse([zip_bytes])])

    download_dvf_year(2021, config, tmp_path, session=session)

    assert session.closed is False


def test_internal_session_is_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: DvfDataConfig, zip_bytes: bytes
) -> None:
    """A session created by the downloader is closed after a download."""
    internal_session = FakeSession([FakeResponse([zip_bytes])])
    monkeypatch.setattr(download_module.requests, "Session", lambda: internal_session)

    download_dvf_year(2021, config, tmp_path)

    assert internal_session.closed is True


def test_existing_file_with_manifest_hash_mismatch_fails_without_network(
    tmp_path: Path, config: DvfDataConfig, zip_bytes: bytes
) -> None:
    """A tampered file is never silently overwritten without --force."""
    download_dvf_year(2021, config, tmp_path, session=FakeSession([FakeResponse([zip_bytes])]))
    destination = tmp_path / "data/raw/dvf/dvf_2021.zip"
    destination.write_bytes(b"tampered")
    session = FakeSession([])

    with pytest.raises(ExistingDownloadMismatchError, match="does not match"):
        download_dvf_year(2021, config, tmp_path, session=session)
    assert destination.read_bytes() == b"tampered"
    assert session.calls == []


@pytest.mark.parametrize(
    "outcome",
    [
        FakeResponse([], status_error=requests.HTTPError("404")),
        requests.ConnectionError("network unavailable"),
    ],
    ids=["http-error", "network-error"],
)
def test_request_errors_leave_no_final_or_partial_file(
    tmp_path: Path, config: DvfDataConfig, outcome: FakeResponse | requests.RequestException
) -> None:
    """Network and HTTP errors leave no archive, temporary file or manifest."""
    with pytest.raises(DownloadError):
        download_dvf_year(2021, config, tmp_path, session=FakeSession([outcome]))

    raw_directory = tmp_path / "data/raw/dvf"
    assert not (raw_directory / "dvf_2021.zip").exists()
    assert not (raw_directory / "manifest.json").exists()
    assert not list(raw_directory.glob("*.part"))


def test_streaming_error_cleans_the_temporary_file(
    tmp_path: Path, config: DvfDataConfig, zip_bytes: bytes
) -> None:
    """A failed iterator cannot leave partially streamed data behind."""
    response = FakeResponse(
        [zip_bytes[:10]], stream_error=requests.ConnectionError("connection dropped")
    )

    with pytest.raises(DownloadError):
        download_dvf_year(2021, config, tmp_path, session=FakeSession([response]))

    raw_directory = tmp_path / "data/raw/dvf"
    assert not (raw_directory / "dvf_2021.zip").exists()
    assert not (raw_directory / "manifest.json").exists()
    assert not list(raw_directory.glob("*.part"))
    assert response.closed is True


def test_content_length_mismatch_fails_before_final_replacement(
    tmp_path: Path, config: DvfDataConfig, zip_bytes: bytes
) -> None:
    """A declared length must match the number of bytes written."""
    response = FakeResponse([zip_bytes], content_length=len(zip_bytes) + 1)

    with pytest.raises(DownloadValidationError, match="Content-Length"):
        download_dvf_year(2021, config, tmp_path, session=FakeSession([response]))

    raw_directory = tmp_path / "data/raw/dvf"
    assert not (raw_directory / "dvf_2021.zip").exists()
    assert not list(raw_directory.glob("*.part"))


def test_invalid_zip_signature_fails_before_final_replacement(
    tmp_path: Path, config: DvfDataConfig
) -> None:
    """The configured ZIP destination rejects an invalid ZIP payload."""
    response = FakeResponse([b"not a zip"])

    with pytest.raises(DownloadValidationError, match="valid ZIP"):
        download_dvf_year(2021, config, tmp_path, session=FakeSession([response]))

    raw_directory = tmp_path / "data/raw/dvf"
    assert not (raw_directory / "dvf_2021.zip").exists()
    assert not (raw_directory / "manifest.json").exists()
    assert not list(raw_directory.glob("*.part"))


def test_force_explicitly_redownloads_a_verified_year(
    tmp_path: Path, config: DvfDataConfig, zip_bytes: bytes
) -> None:
    """Force bypasses idempotence and replaces the archive only after validation."""
    changed_buffer = io.BytesIO()
    with zipfile.ZipFile(changed_buffer, "w") as archive:
        archive.writestr("valeursfoncieres-2021.txt", "changed synthetic DVF")
    changed_zip = changed_buffer.getvalue()
    download_dvf_year(2021, config, tmp_path, session=FakeSession([FakeResponse([zip_bytes])]))
    session = FakeSession([FakeResponse([changed_zip])])

    result = download_dvf_year(2021, config, tmp_path, force=True, session=session)

    destination = tmp_path / "data/raw/dvf/dvf_2021.zip"
    assert result.downloaded is True
    assert destination.read_bytes() == changed_zip
    assert result.sha256 == hashlib.sha256(changed_zip).hexdigest()
    assert len(session.calls) == 1
