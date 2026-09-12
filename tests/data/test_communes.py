"""Exercise COG acquisition with tiny in-memory ZIPs and no network access."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import requests
import yaml

from real_estate.data import communes
from real_estate.data.communes import (
    DEFAULT_CONFIG_PATH,
    CogDataConfig,
    DownloadError,
    DownloadValidationError,
    ExistingDownloadMismatchError,
    download_cog_year,
    download_cog_years,
    get_cog_source,
    load_cog_config,
)

EXPECTED_RESOURCES = {
    2021: "https://www.insee.fr/fr/statistiques/fichier/5057840/cog_ensemble_2021_csv.zip",
    2022: "https://www.insee.fr/fr/statistiques/fichier/6051727/cog_ensemble_2022_csv.zip",
    2023: "https://www.insee.fr/fr/statistiques/fichier/6800675/cog_ensemble_2023_csv.zip",
    2024: "https://www.insee.fr/fr/statistiques/fichier/7766585/cog_ensemble_2024_csv.zip",
    2025: "https://www.insee.fr/fr/statistiques/fichier/8377162/cog_ensemble_2025_csv.zip",
}


class FakeResponse:
    """Serve chunks, recording closure and optionally interrupting the stream."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        content_length: int | str | None = None,
        status_error: requests.RequestException | None = None,
        stream_error: requests.RequestException | None = None,
        during_stream: Callable[[], None] | None = None,
    ) -> None:
        self.chunks = chunks
        self.headers = (
            {} if content_length is None else {"Content-Length": str(content_length)}
        )
        self.url = "https://files.example.test/current-cog.zip"
        self.status_error = status_error
        self.stream_error = stream_error
        self.during_stream = during_stream
        self.chunk_sizes: list[int] = []
        self.closed = False

    @property
    def content(self) -> bytes:
        """Reject eager response-body reads."""
        raise AssertionError("COG acquisition must use iter_content")

    def raise_for_status(self) -> None:
        if self.status_error is not None:
            raise self.status_error

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        self.chunk_sizes.append(chunk_size)
        for chunk in self.chunks:
            yield chunk
            if self.during_stream is not None:
                self.during_stream()
        if self.stream_error is not None:
            raise self.stream_error

    def close(self) -> None:
        self.closed = True


class FakeSession:
    """Return only explicitly supplied outcomes; never contact a server."""

    def __init__(self, outcomes: list[FakeResponse | requests.RequestException]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, bool, tuple[float, float]]] = []
        self.closed = False

    def get(
        self, url: str, *, stream: bool, timeout: tuple[float, float]
    ) -> FakeResponse:
        self.calls.append((url, stream, timeout))
        assert self.outcomes, "Unexpected request"
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, requests.RequestException):
            raise outcome
        return outcome

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail immediately if an acquisition test reaches requests' real transport."""
    def forbidden_request(*args: object, **kwargs: object) -> None:
        raise AssertionError("Real HTTP requests are forbidden in COG tests")

    monkeypatch.setattr(requests.Session, "request", forbidden_request)


@pytest.fixture
def config() -> CogDataConfig:
    return CogDataConfig(
        source_name="INSEE — Code officiel géographique (COG)",
        source_page="https://www.insee.fr/fr/information/2560452",
        years=(2025,),
        raw_directory=Path("data/raw/cog"),
        manifest_filename="manifest.json",
        connect_timeout_seconds=1,
        read_timeout_seconds=2,
        chunk_size_bytes=7,
        max_attempts=1,
        resources={2025: EXPECTED_RESOURCES[2025]},
    )


def make_zip(payload: str = "TYPECOM,COM\nCOM,00000\nCOMD,00000\n") -> bytes:
    """Include intentionally repeated COM values; acquisition must retain bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("synthetic_communes.csv", payload)
    return buffer.getvalue()


@pytest.fixture
def zip_bytes() -> bytes:
    return make_zip()


def raw_paths(root: Path) -> tuple[Path, Path]:
    directory = root / "data/raw/cog"
    return directory / "cog_2025.zip", directory / "manifest.json"


def seed_download(root: Path, config: CogDataConfig, payload: bytes) -> None:
    download_cog_year(2025, config, root, session=FakeSession([FakeResponse([payload])]))


def assert_no_output(root: Path) -> None:
    archive, manifest = raw_paths(root)
    assert not archive.exists()
    assert not manifest.exists()
    assert not list(archive.parent.glob("*.part"))


def test_config_exactly_matches_all_five_official_resources() -> None:
    actual = load_cog_config(DEFAULT_CONFIG_PATH)
    assert actual.years == (2021, 2022, 2023, 2024, 2025)
    assert actual.resources == EXPECTED_RESOURCES
    assert actual.raw_directory == Path("data/raw/cog")
    assert actual.manifest_filename == "manifest.json"
    for year, url in EXPECTED_RESOURCES.items():
        assert get_cog_source(actual, year) == url


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resources", {}),
        ("raw_directory", "../outside-project"),
        ("manifest_filename", "cog_2025.zip"),
        ("max_attempts", 0),
        ("read_timeout_seconds", float("inf")),
    ],
)
def test_malformed_configuration_fails_explicitly(
    tmp_path: Path, config: CogDataConfig, field: str, value: object
) -> None:
    section = dict(vars(config))
    section["raw_directory"] = str(config.raw_directory)
    section["years"] = list(config.years)
    section[field] = value
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump({"cog": section}), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid COG configuration"):
        load_cog_config(path)


@pytest.mark.parametrize("year", [2020, 2026])
def test_unknown_year_fails_before_request_or_writes(
    tmp_path: Path, config: CogDataConfig, year: int
) -> None:
    session = FakeSession([])
    with pytest.raises(ValueError, match="not configured"):
        get_cog_source(config, year)
    with pytest.raises(ValueError, match="not configured"):
        download_cog_year(year, config, tmp_path, session=session)
    assert session.calls == []
    assert list(tmp_path.iterdir()) == []


def test_streaming_download_records_exact_bytes_provenance_and_utc_manifest(
    tmp_path: Path, config: CogDataConfig, zip_bytes: bytes
) -> None:
    response = FakeResponse(
        [zip_bytes[:9], b"", zip_bytes[9:]], content_length=len(zip_bytes)
    )
    session = FakeSession([response])

    result = download_cog_year(2025, config, tmp_path, session=session)

    archive, manifest_path = raw_paths(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["downloads"]["2025"]
    assert result.downloaded is True
    assert result.path == archive
    assert result.year == 2025
    assert result.filename == "cog_2025.zip"
    assert result.bytes_written == len(zip_bytes)
    assert result.sha256 == hashlib.sha256(zip_bytes).hexdigest()
    assert result.source_url == config.resources[2025]
    assert result.final_url == response.url
    assert archive.read_bytes() == zip_bytes
    assert manifest["source_name"] == config.source_name
    assert manifest["source_page"] == config.source_page
    assert entry == {
        "year": 2025,
        "source_url": config.resources[2025],
        "final_url": response.url,
        "filename": archive.name,
        "bytes": len(zip_bytes),
        "sha256": result.sha256,
        "downloaded_at": result.downloaded_at,
    }
    assert datetime.fromisoformat(entry["downloaded_at"]).utcoffset() == timedelta(0)
    assert response.chunk_sizes == [7]
    assert response.closed is True
    assert session.closed is False
    assert session.calls == [(config.resources[2025], True, (1, 2))]
    assert {path.name for path in archive.parent.iterdir()} == {
        "cog_2025.zip", "manifest.json"
    }


@pytest.mark.parametrize("has_existing", [False, True])
def test_archive_becomes_final_only_after_complete_validated_stream(
    tmp_path: Path, config: CogDataConfig, zip_bytes: bytes, has_existing: bool
) -> None:
    if has_existing:
        seed_download(tmp_path, config, zip_bytes)
    archive, manifest = raw_paths(tmp_path)
    old_manifest = manifest.read_bytes() if has_existing else None
    replacement = make_zip("replacement synthetic bytes")
    inspections: list[bool] = []

    def inspect_incomplete_write() -> None:
        inspections.append(True)
        assert list(archive.parent.glob("*.part"))
        if has_existing:
            assert archive.read_bytes() == zip_bytes
            assert manifest.read_bytes() == old_manifest
        else:
            assert not archive.exists()
            assert not manifest.exists()

    response = FakeResponse(
        [replacement[:10], replacement[10:]], during_stream=inspect_incomplete_write
    )
    download_cog_year(
        2025, config, tmp_path, force=has_existing, session=FakeSession([response])
    )
    assert inspections
    assert archive.read_bytes() == replacement
    assert not list(archive.parent.glob("*.part"))


def test_valid_existing_download_preserves_historical_redirect_without_network(
    tmp_path: Path, config: CogDataConfig, zip_bytes: bytes
) -> None:
    seed_download(tmp_path, config, zip_bytes)
    archive, manifest = raw_paths(tmp_path)
    old_manifest = manifest.read_bytes()
    session = FakeSession([])

    result = download_cog_year(2025, config, tmp_path, session=session)

    assert result.downloaded is False
    assert result.final_url == "https://files.example.test/current-cog.zip"
    assert result.final_url != config.resources[2025]
    assert archive.read_bytes() == zip_bytes
    assert manifest.read_bytes() == old_manifest
    assert session.calls == []
    assert session.closed is False


def test_configured_source_change_is_a_provenance_error_without_network(
    tmp_path: Path, config: CogDataConfig, zip_bytes: bytes
) -> None:
    seed_download(tmp_path, config, zip_bytes)
    changed = replace(config, resources={2025: "https://example.test/revised.zip"})
    session = FakeSession([])
    with pytest.raises(ExistingDownloadMismatchError, match="source_url|provenance"):
        download_cog_year(2025, changed, tmp_path, session=session)
    assert session.calls == []


def test_changed_archive_hash_fails_without_overwrite_or_network(
    tmp_path: Path, config: CogDataConfig, zip_bytes: bytes
) -> None:
    seed_download(tmp_path, config, zip_bytes)
    archive, manifest = raw_paths(tmp_path)
    modified = zip_bytes[:-1] + bytes([zip_bytes[-1] ^ 1])
    archive.write_bytes(modified)
    old_manifest = manifest.read_bytes()
    session = FakeSession([])
    with pytest.raises(ExistingDownloadMismatchError):
        download_cog_year(2025, config, tmp_path, session=session)
    assert session.calls == []
    assert archive.read_bytes() == modified
    assert manifest.read_bytes() == old_manifest


@pytest.mark.parametrize(
    ("field", "value"),
    [("filename", "wrong.zip"), ("bytes", 1), ("year", 2024)],
)
def test_inconsistent_manifest_entry_fails_without_network(
    tmp_path: Path, config: CogDataConfig, zip_bytes: bytes, field: str, value: object
) -> None:
    seed_download(tmp_path, config, zip_bytes)
    _, manifest_path = raw_paths(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["downloads"]["2025"][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    session = FakeSession([])
    with pytest.raises(ExistingDownloadMismatchError):
        download_cog_year(2025, config, tmp_path, session=session)
    assert session.calls == []


@pytest.mark.parametrize("state", ["absent", "invalid-json", "missing-entry", "invalid-entry"])
def test_existing_archive_requires_a_complete_coherent_manifest(
    tmp_path: Path, config: CogDataConfig, zip_bytes: bytes, state: str
) -> None:
    seed_download(tmp_path, config, zip_bytes)
    archive, manifest_path = raw_paths(tmp_path)
    if state == "absent":
        manifest_path.unlink()
    elif state == "invalid-json":
        manifest_path.write_text("{malformed", encoding="utf-8")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if state == "missing-entry":
            manifest["downloads"].pop("2025")
        else:
            manifest["downloads"]["2025"] = {"year": 2025}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    session = FakeSession([])
    with pytest.raises(DownloadError):
        download_cog_year(2025, config, tmp_path, session=session)
    assert session.calls == []
    assert archive.read_bytes() == zip_bytes


def test_force_cannot_discard_unreadable_manifest_provenance(
    tmp_path: Path, config: CogDataConfig, zip_bytes: bytes
) -> None:
    seed_download(tmp_path, config, zip_bytes)
    archive, manifest = raw_paths(tmp_path)
    invalid_manifest = b"{unreadable provenance"
    manifest.write_bytes(invalid_manifest)
    session = FakeSession([])
    with pytest.raises(DownloadError, match="manifest"):
        download_cog_year(2025, config, tmp_path, force=True, session=session)
    assert session.calls == []
    assert archive.read_bytes() == zip_bytes
    assert manifest.read_bytes() == invalid_manifest
    assert not list(archive.parent.glob("*.part"))


def test_manifest_entry_without_archive_is_an_explicit_partial_state(
    tmp_path: Path, config: CogDataConfig, zip_bytes: bytes
) -> None:
    seed_download(tmp_path, config, zip_bytes)
    archive, manifest = raw_paths(tmp_path)
    archive.unlink()
    original_manifest = manifest.read_bytes()
    session = FakeSession([])
    with pytest.raises(ExistingDownloadMismatchError):
        download_cog_year(2025, config, tmp_path, session=session)
    assert session.calls == []
    assert manifest.read_bytes() == original_manifest


@pytest.mark.parametrize("content_length", [0, 1, "invalid", -1])
def test_inconsistent_or_invalid_content_length_cleans_partial_files(
    tmp_path: Path, config: CogDataConfig, zip_bytes: bytes, content_length: int | str
) -> None:
    response = FakeResponse([zip_bytes], content_length=content_length)
    with pytest.raises(DownloadValidationError, match="Content-Length"):
        download_cog_year(2025, config, tmp_path, session=FakeSession([response]))
    assert_no_output(tmp_path)
    assert response.closed is True


@pytest.mark.parametrize("payload", [b"", b"not a ZIP", b"PK\x03\x04truncated"])
def test_empty_or_invalid_zip_is_rejected_and_cleaned(
    tmp_path: Path, config: CogDataConfig, payload: bytes
) -> None:
    response = FakeResponse([payload])
    with pytest.raises(DownloadValidationError):
        download_cog_year(2025, config, tmp_path, session=FakeSession([response]))
    assert_no_output(tmp_path)
    assert response.closed is True


@pytest.mark.parametrize("failure", ["http", "connection", "stream"])
def test_request_failures_leave_no_archive_manifest_or_part(
    tmp_path: Path, config: CogDataConfig, failure: str
) -> None:
    response = FakeResponse([b"PK"], stream_error=requests.ConnectionError("dropped"))
    outcome: FakeResponse | requests.RequestException = response
    if failure == "http":
        response = FakeResponse([], status_error=requests.HTTPError("404"))
        outcome = response
    elif failure == "connection":
        outcome = requests.ConnectionError("unavailable")
    session = FakeSession([outcome])
    with pytest.raises(DownloadError):
        download_cog_year(2025, config, tmp_path, session=session)
    assert_no_output(tmp_path)
    assert session.closed is False
    if isinstance(outcome, FakeResponse):
        assert outcome.closed is True


def test_network_retry_is_bounded_and_can_recover(
    tmp_path: Path, config: CogDataConfig, zip_bytes: bytes
) -> None:
    session = FakeSession([
        requests.ConnectionError("first attempt"), FakeResponse([zip_bytes])
    ])
    result = download_cog_year(
        2025, replace(config, max_attempts=2), tmp_path, session=session
    )
    assert result.path.read_bytes() == zip_bytes
    assert len(session.calls) == 2
    assert not list(result.path.parent.glob("*.part"))


def test_retry_exhaustion_does_not_make_an_additional_request(
    tmp_path: Path, config: CogDataConfig
) -> None:
    session = FakeSession([requests.Timeout("timeout") for _ in range(2)])
    with pytest.raises(DownloadError):
        download_cog_year(2025, replace(config, max_attempts=2), tmp_path, session=session)
    assert len(session.calls) == 2
    assert_no_output(tmp_path)


def test_stream_retry_starts_a_fresh_archive_and_digest(
    tmp_path: Path, config: CogDataConfig, zip_bytes: bytes
) -> None:
    first = FakeResponse([zip_bytes[:15]], stream_error=requests.ReadTimeout("dropped"))
    second = FakeResponse([zip_bytes], content_length=len(zip_bytes))
    session = FakeSession([first, second])
    result = download_cog_year(
        2025, replace(config, max_attempts=2), tmp_path, session=session
    )
    assert result.path.read_bytes() == zip_bytes
    assert result.sha256 == hashlib.sha256(zip_bytes).hexdigest()
    assert first.closed and second.closed
    assert len(session.calls) == 2
    assert not list(result.path.parent.glob("*.part"))


def test_force_explicitly_replaces_existing_archive_and_provenance(
    tmp_path: Path, config: CogDataConfig, zip_bytes: bytes
) -> None:
    seed_download(tmp_path, config, zip_bytes)
    replacement = make_zip("changed synthetic content")
    changed = replace(config, resources={2025: "https://example.test/new-cog.zip"})
    session = FakeSession([FakeResponse([replacement])])
    result = download_cog_year(2025, changed, tmp_path, force=True, session=session)
    _, manifest_path = raw_paths(tmp_path)
    entry = json.loads(manifest_path.read_text(encoding="utf-8"))["downloads"]["2025"]
    assert result.downloaded is True
    assert result.path.read_bytes() == replacement
    assert entry["sha256"] == hashlib.sha256(replacement).hexdigest()
    assert entry["source_url"] == changed.resources[2025]
    assert len(session.calls) == 1


def test_failed_force_preserves_valid_archive_and_manifest(
    tmp_path: Path, config: CogDataConfig, zip_bytes: bytes
) -> None:
    seed_download(tmp_path, config, zip_bytes)
    archive, manifest = raw_paths(tmp_path)
    original_manifest = manifest.read_bytes()
    with pytest.raises(DownloadValidationError):
        download_cog_year(
            2025, config, tmp_path, force=True,
            session=FakeSession([FakeResponse([b"invalid ZIP"])]),
        )
    assert archive.read_bytes() == zip_bytes
    assert manifest.read_bytes() == original_manifest
    assert not list(archive.parent.glob("*.part"))


@pytest.mark.parametrize("has_existing", [False, True])
def test_manifest_publication_failure_rolls_back_archive_and_cleans_part(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: CogDataConfig,
    zip_bytes: bytes,
    has_existing: bool,
) -> None:
    if has_existing:
        seed_download(tmp_path, config, zip_bytes)
    archive, manifest = raw_paths(tmp_path)
    original_manifest = manifest.read_bytes() if has_existing else None
    original_replace = communes.os.replace

    def failing_manifest_replace(source: Path, destination: Path) -> None:
        if Path(destination) == manifest:
            raise OSError("synthetic manifest publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(communes.os, "replace", failing_manifest_replace)
    with pytest.raises((DownloadError, OSError)):
        download_cog_year(
            2025, config, tmp_path, force=has_existing,
            session=FakeSession([FakeResponse([make_zip("replacement")])]),
        )
    if has_existing:
        assert archive.read_bytes() == zip_bytes
        assert manifest.read_bytes() == original_manifest
    else:
        assert_no_output(tmp_path)
    assert not list(archive.parent.glob("*.part"))


@pytest.mark.parametrize("fails", [False, True])
def test_internally_created_session_is_closed_even_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: CogDataConfig,
    zip_bytes: bytes,
    fails: bool,
) -> None:
    session = FakeSession([
        requests.ConnectionError("unavailable") if fails else FakeResponse([zip_bytes])
    ])
    monkeypatch.setattr(communes.requests, "Session", lambda: session)
    if fails:
        with pytest.raises(DownloadError):
            download_cog_year(2025, config, tmp_path)
    else:
        download_cog_year(2025, config, tmp_path)
    assert session.closed is True


@pytest.mark.parametrize("valid_zip", [False, True])
def test_response_close_failure_still_cleans_part_and_closes_internal_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: CogDataConfig,
    zip_bytes: bytes,
    valid_zip: bool,
) -> None:
    response = FakeResponse([zip_bytes if valid_zip else b"invalid ZIP"])
    session = FakeSession([response])

    def failing_response_close() -> None:
        raise OSError("synthetic response close failure")

    monkeypatch.setattr(response, "close", failing_response_close)
    monkeypatch.setattr(communes.requests, "Session", lambda: session)
    with pytest.raises((DownloadError, OSError)):
        download_cog_year(2025, config, tmp_path)
    assert_no_output(tmp_path)
    assert session.closed is True


def test_failed_part_cleanup_still_closes_internal_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: CogDataConfig
) -> None:
    session = FakeSession([FakeResponse([b"invalid ZIP"])])
    original_unlink = Path.unlink
    failures: list[Path] = []

    def fail_first_part_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path.suffix == ".part" and not failures:
            failures.append(path)
            raise OSError("synthetic cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(communes.requests, "Session", lambda: session)
    monkeypatch.setattr(Path, "unlink", fail_first_part_unlink)
    with pytest.raises((DownloadError, OSError)):
        download_cog_year(2025, config, tmp_path)
    assert failures
    assert session.closed is True
    archive, manifest = raw_paths(tmp_path)
    assert not archive.exists()
    assert not manifest.exists()
    for path in archive.parent.glob("*.part"):
        original_unlink(path)


def test_multiple_years_use_distinct_archives_and_preserve_manifest_entries(
    tmp_path: Path, config: CogDataConfig, zip_bytes: bytes
) -> None:
    expanded = replace(
        config, years=(2024, 2025),
        resources={year: EXPECTED_RESOURCES[year] for year in (2024, 2025)},
    )
    session = FakeSession([FakeResponse([zip_bytes]), FakeResponse([zip_bytes])])
    results = download_cog_years([2024, 2025], expanded, tmp_path, session=session)
    assert [result.year for result in results] == [2024, 2025]
    assert [result.filename for result in results] == ["cog_2024.zip", "cog_2025.zip"]
    _, manifest_path = raw_paths(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["downloads"]) == {"2024", "2025"}
    assert session.closed is False


@pytest.mark.parametrize(
    ("arguments", "expected_years", "expected_force"),
    [
        ([], [2024, 2025], False),
        (["--year", "2025"], [2025], False),
        (["--year", "2024", "2025", "--force"], [2024, 2025], True),
    ],
)
def test_cli_passes_only_explicit_configuration_and_options(
    monkeypatch: pytest.MonkeyPatch,
    config: CogDataConfig,
    arguments: list[str],
    expected_years: list[int],
    expected_force: bool,
) -> None:
    expanded = replace(
        config, years=(2024, 2025),
        resources={year: EXPECTED_RESOURCES[year] for year in (2024, 2025)},
    )
    calls: list[tuple[list[int], CogDataConfig, bool]] = []

    def fake_download(
        years: list[int], data_config: CogDataConfig, project_root: Path, *, force: bool
    ) -> list:
        calls.append((list(years), data_config, force))
        return []

    monkeypatch.setattr(communes, "load_cog_config", lambda *args: expanded)
    monkeypatch.setattr(communes, "download_cog_years", fake_download)
    assert communes.main(arguments) == 0
    assert calls == [(expected_years, expanded, expected_force)]
