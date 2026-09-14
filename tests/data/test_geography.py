"""Build annual COG geography from small synthetic ZIPs without network access."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests
import yaml

from real_estate.data import geography
from real_estate.data.communes import CogDataConfig

EXPECTED_MEMBERS = {
    2021: "commune2021.csv",
    2022: "commune_2022.csv",
    2023: "v_commune_2023.csv",
    2024: "v_commune_2024.csv",
    2025: "v_commune_2025.csv",
}
SOURCE_COLUMNS = (
    "TYPECOM", "COM", "REG", "DEP", "CTCD", "ARR", "TNCC", "NCC", "NCCENR",
    "LIBELLE", "CAN", "COMPARENT",
)
OUTPUT_COLUMNS = [
    "source_code_commune", "resolved_geo_type", "source_geo_label",
    "parent_commune_code", "canonical_commune_code", "canonical_commune_label",
    "region_code", "department_code", "cog_year",
]


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexpected HTTP request must fail even if a test fixture is incomplete."""
    def fail_request(*args: object, **kwargs: object) -> None:
        raise AssertionError("Geography tests must never access the network")

    monkeypatch.setattr(requests.Session, "request", fail_request)


@pytest.fixture
def config() -> geography.GeographyConfig:
    return geography.GeographyConfig(
        cog=CogDataConfig(
            source_name="Synthetic COG",
            source_page="https://example.test/cog",
            years=tuple(EXPECTED_MEMBERS),
            raw_directory=Path("data/raw/cog"),
            manifest_filename="manifest.json",
            connect_timeout_seconds=1,
            read_timeout_seconds=2,
            chunk_size_bytes=1024,
            max_attempts=1,
            resources={
                year: f"https://example.test/cog_{year}.zip"
                for year in EXPECTED_MEMBERS
            },
        ),
        current_commune_files=dict(EXPECTED_MEMBERS),
        output_directory=Path("data/interim/cog/geography"),
    )


def cog_row(**overrides: str) -> dict[str, str]:
    """A fictional commune with strings that retain significant leading zeros."""
    row = dict.fromkeys(SOURCE_COLUMNS, "")
    row.update({
        "TYPECOM": "COM", "COM": "01001", "REG": "01", "DEP": "01",
        "CTCD": "01D", "ARR": "011", "TNCC": "0", "NCC": "COMMUNE FICTIVE",
        "NCCENR": "Commune fictive", "LIBELLE": "Commune fictive", "CAN": "0199",
    })
    row.update(overrides)
    return row


def csv_text(rows: list[dict[str, str]], header: tuple[str, ...]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows([[row.get(column, "") for column in header] for row in rows])
    return buffer.getvalue()


def write_raw(
    root: Path,
    config: geography.GeographyConfig,
    rows: list[dict[str, str]],
    *,
    year: int = 2025,
    header: tuple[str, ...] = SOURCE_COLUMNS,
    member: str | None = None,
    text: str | None = None,
) -> tuple[Path, Path]:
    """Create only synthetic test archives and a complete acquisition manifest."""
    archive_path = root / config.cog.raw_directory / f"cog_{year}.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            config.current_commune_files[year] if member is None else member,
            csv_text(rows, header) if text is None else text,
        )
        # These deliberately invalid members must never be used for resolution.
        archive.writestr("v_commune_depuis_1943.csv", "invalid historical data")
        archive.writestr("mvtcommune2021.csv", "invalid movement data")
    manifest_path = archive_path.parent / config.cog.manifest_filename
    refresh_manifest(archive_path, manifest_path, config, year)
    return archive_path, manifest_path


def refresh_manifest(
    archive_path: Path,
    manifest_path: Path,
    config: geography.GeographyConfig,
    year: int = 2025,
) -> None:
    payload = archive_path.read_bytes()
    manifest_path.write_text(json.dumps({
        "source_name": config.cog.source_name,
        "source_page": config.cog.source_page,
        "downloads": {str(year): {
            "year": year,
            "source_url": config.cog.resources[year],
            "final_url": "https://files.example.test/publication.zip",
            "filename": archive_path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "downloaded_at": "2025-01-02T00:00:00+00:00",
        }},
    }), encoding="utf-8")


def output_path(root: Path, config: geography.GeographyConfig) -> Path:
    return root / config.output_directory / "cog_geography_2025.parquet"


def assert_no_parts(root: Path) -> None:
    assert list(root.rglob("*.part")) == []


def test_project_configuration_has_explicit_mapping_for_all_five_years() -> None:
    actual = geography.load_geography_config(geography.DEFAULT_CONFIG_PATH)
    assert actual.cog.years == (2021, 2022, 2023, 2024, 2025)
    assert actual.current_commune_files == EXPECTED_MEMBERS
    assert actual.output_directory == Path("data/interim/cog/geography")
    assert geography.COG_SOURCE_COLUMNS == SOURCE_COLUMNS


@pytest.mark.parametrize("year", EXPECTED_MEMBERS)
def test_public_build_reads_exact_configured_member_for_each_year(
    tmp_path: Path, config: geography.GeographyConfig, year: int,
) -> None:
    archive, manifest = write_raw(tmp_path, config, [cog_row()], year=year)
    before = archive.read_bytes(), manifest.read_bytes()

    result = geography.build_geography_year(year, config, tmp_path)

    assert result.year == year
    assert result.path.name == f"cog_geography_{year}.parquet"
    assert pq.read_table(result.path)["cog_year"].to_pylist() == [year]
    assert (archive.read_bytes(), manifest.read_bytes()) == before
    assert_no_parts(tmp_path)
    assert list(tmp_path.rglob("*.csv")) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current_commune_files", {}),
        ("current_commune_files", {2025: "v_commune_2025.csv"}),
        ("output_directory", "../outside-project"),
        ("output_directory", "/outside-project"),
    ],
)
def test_invalid_geography_configuration_fails(
    tmp_path: Path, config: geography.GeographyConfig, field: str, value: object,
) -> None:
    cog = dict(vars(config.cog))
    cog["raw_directory"] = str(config.cog.raw_directory)
    cog["years"] = list(config.cog.years)
    section = {
        "current_commune_files": config.current_commune_files,
        "output_directory": str(config.output_directory),
    }
    section[field] = value
    cog["geography"] = section
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump({"cog": cog}), encoding="utf-8")

    with pytest.raises((ValueError, geography.GeographyError)):
        geography.load_geography_config(path)


@pytest.mark.parametrize("year", [2020, 2026])
def test_unconfigured_year_fails_without_writes(
    tmp_path: Path, config: geography.GeographyConfig, year: int,
) -> None:
    with pytest.raises((ValueError, geography.GeographyError)):
        geography.build_geography_year(year, config, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_archive_integrity_checks_success_without_writes(
    tmp_path: Path, config: geography.GeographyConfig,
) -> None:
    archive, manifest = write_raw(tmp_path, config, [cog_row()])
    before = archive.read_bytes(), manifest.read_bytes()
    assert geography.verify_cog_archive(2025, config, tmp_path) == archive
    assert (archive.read_bytes(), manifest.read_bytes()) == before
    assert not output_path(tmp_path, config).exists()


@pytest.mark.parametrize("missing", ["archive", "manifest", "year"])
def test_missing_provenance_fails_before_publication(
    tmp_path: Path, config: geography.GeographyConfig, missing: str,
) -> None:
    archive, manifest = write_raw(tmp_path, config, [cog_row()])
    if missing == "archive":
        archive.unlink()
    elif missing == "manifest":
        manifest.unlink()
    else:
        manifest.write_text('{"downloads": {}}', encoding="utf-8")

    with pytest.raises(geography.GeographyProvenanceError):
        geography.build_geography_year(2025, config, tmp_path)
    assert not output_path(tmp_path, config).exists()
    assert_no_parts(tmp_path)


@pytest.mark.parametrize("text", ["not JSON", "[]", "null", '{"downloads": []}'])
def test_invalid_manifest_is_explicit_error(
    tmp_path: Path, config: geography.GeographyConfig, text: str,
) -> None:
    _, manifest = write_raw(tmp_path, config, [cog_row()])
    manifest.write_text(text, encoding="utf-8")
    with pytest.raises(geography.GeographyProvenanceError):
        geography.build_geography_year(2025, config, tmp_path)
    assert_no_parts(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("year", 2024),
        ("filename", "different.zip"),
        ("source_url", "https://example.test/other-resource.zip"),
        ("bytes", 1),
        ("bytes", True),
        ("sha256", "0" * 64),
        ("sha256", "not-a-sha256"),
    ],
)
def test_manifest_disagreement_fails_without_publishing(
    tmp_path: Path, config: geography.GeographyConfig, field: str, value: object,
) -> None:
    _, manifest = write_raw(tmp_path, config, [cog_row()])
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["downloads"]["2025"][field] = value
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(geography.GeographyProvenanceError):
        geography.build_geography_year(2025, config, tmp_path)
    assert not output_path(tmp_path, config).exists()
    assert_no_parts(tmp_path)


def test_duplicate_manifest_year_is_rejected(
    tmp_path: Path, config: geography.GeographyConfig,
) -> None:
    _, manifest = write_raw(tmp_path, config, [cog_row()])
    entry = json.loads(manifest.read_text(encoding="utf-8"))["downloads"]["2025"]
    entry_json = json.dumps(entry)
    manifest.write_text(
        '{"downloads": {"2025": ' + entry_json + ', "2025": ' + entry_json + '}}',
        encoding="utf-8",
    )
    with pytest.raises(geography.GeographyProvenanceError):
        geography.build_geography_year(2025, config, tmp_path)
    assert_no_parts(tmp_path)


def test_archive_tampering_fails_even_with_unchanged_size(
    tmp_path: Path, config: geography.GeographyConfig,
) -> None:
    archive, _ = write_raw(tmp_path, config, [cog_row()])
    payload = bytearray(archive.read_bytes())
    payload[len(payload) // 2] ^= 1
    archive.write_bytes(payload)
    with pytest.raises(geography.GeographyProvenanceError):
        geography.build_geography_year(2025, config, tmp_path)
    assert not output_path(tmp_path, config).exists()


def test_invalid_zip_with_matching_hash_is_rejected(
    tmp_path: Path, config: geography.GeographyConfig,
) -> None:
    archive, manifest = write_raw(tmp_path, config, [cog_row()])
    archive.write_bytes(b"not a ZIP")
    refresh_manifest(archive, manifest, config)
    with pytest.raises(geography.GeographyError):
        geography.build_geography_year(2025, config, tmp_path)
    assert_no_parts(tmp_path)


def test_missing_current_member_does_not_fall_back_to_another_csv(
    tmp_path: Path, config: geography.GeographyConfig,
) -> None:
    write_raw(tmp_path, config, [cog_row()], member="other_communes.csv")
    with pytest.raises(geography.GeographyError):
        geography.build_geography_year(2025, config, tmp_path)
    assert_no_parts(tmp_path)


def test_duplicate_current_member_is_rejected(
    tmp_path: Path, config: geography.GeographyConfig,
) -> None:
    archive, manifest = write_raw(tmp_path, config, [cog_row()])
    with (
        pytest.warns(UserWarning, match="Duplicate name"),
        zipfile.ZipFile(archive, "a") as zipped,
    ):
        zipped.writestr(EXPECTED_MEMBERS[2025], csv_text([cog_row()], SOURCE_COLUMNS))
    refresh_manifest(archive, manifest, config)
    with pytest.raises(geography.GeographyError):
        geography.build_geography_year(2025, config, tmp_path)
    assert_no_parts(tmp_path)


@pytest.mark.parametrize(
    "header",
    [SOURCE_COLUMNS[:-1], SOURCE_COLUMNS + ("COM",)],
    ids=["missing-required-column", "duplicate-column"],
)
def test_invalid_csv_schema_is_rejected(
    tmp_path: Path, config: geography.GeographyConfig, header: tuple[str, ...],
) -> None:
    write_raw(tmp_path, config, [cog_row()], header=header)
    with pytest.raises(geography.GeographySchemaError):
        geography.build_geography_year(2025, config, tmp_path)
    assert_no_parts(tmp_path)


@pytest.mark.parametrize(
    "body",
    ["COM,01001\n", ",".join(["x"] * 13) + "\n", '"unterminated'],
    ids=["short-row", "long-row", "broken-quoting"],
)
def test_malformed_csv_structure_is_rejected(
    tmp_path: Path, config: geography.GeographyConfig, body: str,
) -> None:
    write_raw(tmp_path, config, [], text=",".join(SOURCE_COLUMNS) + "\n" + body)
    with pytest.raises(geography.GeographyError):
        geography.build_geography_year(2025, config, tmp_path)
    assert_no_parts(tmp_path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"COM": ""}, {"COM": "   "}, {"COM": "0101"}, {"COM": "001001"},
        {"TYPECOM": "UNKNOWN"}, {"TYPECOM": ""},
        {"LIBELLE": ""}, {"REG": ""}, {"DEP": ""},
    ],
)
def test_invalid_current_commune_is_rejected(
    tmp_path: Path, config: geography.GeographyConfig, overrides: dict[str, str],
) -> None:
    write_raw(tmp_path, config, [cog_row(**overrides)])
    with pytest.raises(geography.GeographyIntegrityError):
        geography.build_geography_year(2025, config, tmp_path)
    assert not output_path(tmp_path, config).exists()
    assert_no_parts(tmp_path)


@pytest.mark.parametrize("geo_type", ["COM", "ARM", "COMD", "COMA"])
def test_duplicate_type_and_code_cannot_be_deduplicated(
    tmp_path: Path, config: geography.GeographyConfig, geo_type: str,
) -> None:
    duplicate = cog_row(TYPECOM=geo_type, COM="01002", COMPARENT="01001")
    if geo_type == "COM":
        duplicate["COMPARENT"] = ""
    rows = [cog_row(), duplicate, {**duplicate, "LIBELLE": "Different label"}]
    write_raw(tmp_path, config, rows)
    with pytest.raises(geography.GeographyIntegrityError):
        geography.build_geography_year(2025, config, tmp_path)
    assert_no_parts(tmp_path)


def test_single_commune_public_output_and_schema(
    tmp_path: Path, config: geography.GeographyConfig,
) -> None:
    write_raw(tmp_path, config, [cog_row()])
    result = geography.build_geography_year(2025, config, tmp_path)
    table = pq.read_table(result.path)
    assert table.schema == geography.GEOGRAPHY_SCHEMA
    assert table.column_names == OUTPUT_COLUMNS
    assert all(pa.types.is_string(field.type) for field in list(table.schema)[:-1])
    assert table.schema.field("cog_year").type == pa.int32()
    assert table.to_pylist() == [{
        "source_code_commune": "01001",
        "resolved_geo_type": "COM",
        "source_geo_label": "Commune fictive",
        "parent_commune_code": None,
        "canonical_commune_code": "01001",
        "canonical_commune_label": "Commune fictive",
        "region_code": "01",
        "department_code": "01",
        "cog_year": 2025,
    }]
    assert result.path == output_path(tmp_path, config)
    assert result.source_codes == 1
    assert result.resolved_counts == {"COM": 1, "ARM": 0, "COMD": 0, "COMA": 0}
    assert result.to_dict() == {
        "year": 2025, "output": str(result.path), "source_codes": 1,
        "resolved_COM": 1, "resolved_ARM": 0, "resolved_COMD": 0, "resolved_COMA": 0,
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_com_plus_comd_selects_com_regardless_of_input_order(
    tmp_path: Path, config: geography.GeographyConfig, reverse: bool,
) -> None:
    rows = [
        cog_row(),
        cog_row(TYPECOM="COMD", COMPARENT="01001", LIBELLE="Commune déléguée"),
    ]
    write_raw(tmp_path, config, list(reversed(rows)) if reverse else rows)
    result = geography.build_geography_year(2025, config, tmp_path)
    row = pq.read_table(result.path).to_pylist()[0]
    assert result.source_codes == 1
    assert row["resolved_geo_type"] == "COM"
    assert row["source_geo_label"] == "Commune fictive"
    assert row["parent_commune_code"] is None
    assert row["canonical_commune_code"] == "01001"
    assert result.resolved_counts["COMD"] == 0


@pytest.mark.parametrize("geo_type", ["ARM", "COMD", "COMA"])
def test_subordinate_uses_canonical_parent_label_region_and_department(
    tmp_path: Path, config: geography.GeographyConfig, geo_type: str,
) -> None:
    rows = [
        cog_row(),
        cog_row(
            COM="01002", TYPECOM=geo_type, COMPARENT="01001",
            LIBELLE="Entité source", REG="99", DEP="98",
        ),
    ]
    write_raw(tmp_path, config, rows)
    result = geography.build_geography_year(2025, config, tmp_path)
    row = pq.read_table(result.path).to_pylist()[1]
    assert row == {
        "source_code_commune": "01002", "resolved_geo_type": geo_type,
        "source_geo_label": "Entité source", "parent_commune_code": "01001",
        "canonical_commune_code": "01001", "canonical_commune_label": "Commune fictive",
        "region_code": "01", "department_code": "01", "cog_year": 2025,
    }


@pytest.mark.parametrize("parent", ["", " ", "0101", "001001", "99999"])
def test_missing_invalid_or_absent_parent_fails(
    tmp_path: Path, config: geography.GeographyConfig, parent: str,
) -> None:
    write_raw(tmp_path, config, [
        cog_row(), cog_row(COM="01002", TYPECOM="ARM", COMPARENT=parent),
    ])
    with pytest.raises(geography.GeographyIntegrityError):
        geography.build_geography_year(2025, config, tmp_path)
    assert_no_parts(tmp_path)


def test_parent_exists_but_is_not_a_com_fails(
    tmp_path: Path, config: geography.GeographyConfig,
) -> None:
    write_raw(tmp_path, config, [
        cog_row(),
        cog_row(COM="01002", TYPECOM="COMD", COMPARENT="01001"),
        cog_row(COM="01003", TYPECOM="ARM", COMPARENT="01002"),
    ])
    with pytest.raises(geography.GeographyIntegrityError):
        geography.build_geography_year(2025, config, tmp_path)
    assert_no_parts(tmp_path)


def test_ambiguous_com_parent_fails_instead_of_choosing_first(
    tmp_path: Path, config: geography.GeographyConfig,
) -> None:
    write_raw(tmp_path, config, [
        cog_row(), cog_row(LIBELLE="Second parent"),
        cog_row(COM="01002", TYPECOM="COMD", COMPARENT="01001"),
    ])
    with pytest.raises(geography.GeographyIntegrityError):
        geography.build_geography_year(2025, config, tmp_path)
    assert_no_parts(tmp_path)


def test_invalid_comd_is_checked_even_when_direct_com_has_priority(
    tmp_path: Path, config: geography.GeographyConfig,
) -> None:
    write_raw(tmp_path, config, [cog_row(), cog_row(TYPECOM="COMD", COMPARENT="")])
    with pytest.raises(geography.GeographyIntegrityError):
        geography.build_geography_year(2025, config, tmp_path)


def test_priority_arm_then_comd_then_coma_for_same_source_code(
    tmp_path: Path, config: geography.GeographyConfig,
) -> None:
    rows = [cog_row()]
    rows.extend(
        cog_row(COM="01002", TYPECOM=kind, COMPARENT="01001", LIBELLE=kind)
        for kind in ["COMA", "COMD", "ARM"]
    )
    rows.extend(
        cog_row(COM="01003", TYPECOM=kind, COMPARENT="01001", LIBELLE=kind)
        for kind in ["COMA", "COMD"]
    )
    write_raw(tmp_path, config, rows)
    result = geography.build_geography_year(2025, config, tmp_path)
    assert pq.read_table(result.path)["resolved_geo_type"].to_pylist() == [
        "COM", "ARM", "COMD",
    ]


def test_reference_has_one_row_per_distinct_source_code_in_deterministic_order(
    tmp_path: Path, config: geography.GeographyConfig,
) -> None:
    rows = [
        cog_row(COM="2A001", REG="94", DEP="2A"),
        cog_row(COM="01002", TYPECOM="COMD", COMPARENT="01001"),
        cog_row(),
        cog_row(TYPECOM="COMD", COMPARENT="01001"),
        cog_row(COM="97101", REG="01", DEP="971"),
    ]
    write_raw(tmp_path, config, rows)
    result = geography.build_geography_year(2025, config, tmp_path)
    table = pq.read_table(result.path)
    assert result.source_codes == 4
    assert table["source_code_commune"].to_pylist() == ["01001", "01002", "2A001", "97101"]
    assert table["department_code"].to_pylist() == ["01", "01", "2A", "971"]
    geography.validate_geography_parquet(result.path, 2025, 4)


def test_empty_current_communes_cannot_produce_empty_reference(
    tmp_path: Path, config: geography.GeographyConfig,
) -> None:
    write_raw(tmp_path, config, [])
    with pytest.raises(geography.GeographyError):
        geography.build_geography_year(2025, config, tmp_path)
    assert not output_path(tmp_path, config).exists()
    assert_no_parts(tmp_path)


def test_atomic_publication_replaces_a_validated_part_file(
    tmp_path: Path, config: geography.GeographyConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_raw(tmp_path, config, [cog_row()])
    original_replace = geography.os.replace
    replacements = []

    def check_replace(source: str | Path, target: str | Path) -> None:
        source_path, target_path = Path(source), Path(target)
        assert source_path.suffix == ".part"
        assert source_path.parent == target_path.parent
        assert target_path == output_path(tmp_path, config)
        assert not target_path.exists()
        geography.validate_geography_parquet(source_path, 2025, 1)
        replacements.append((source_path, target_path))
        original_replace(source, target)

    monkeypatch.setattr(geography.os, "replace", check_replace)
    geography.build_geography_year(2025, config, tmp_path)
    assert len(replacements) == 1
    assert_no_parts(tmp_path)


def test_existing_destination_refused_and_force_rebuilds_explicitly(
    tmp_path: Path, config: geography.GeographyConfig,
) -> None:
    write_raw(tmp_path, config, [cog_row()])
    result = geography.build_geography_year(2025, config, tmp_path)
    previous = result.path.read_bytes()
    write_raw(tmp_path, config, [cog_row(LIBELLE="Nouvelle commune fictive")])

    with pytest.raises((FileExistsError, geography.GeographyError)):
        geography.build_geography_year(2025, config, tmp_path)
    assert result.path.read_bytes() == previous

    replaced = geography.build_geography_year(2025, config, tmp_path, force=True)
    assert replaced.path == result.path
    assert pq.read_table(replaced.path)["source_geo_label"].to_pylist() == [
        "Nouvelle commune fictive",
    ]
    assert_no_parts(tmp_path)


@pytest.mark.parametrize("existing", [False, True])
def test_validation_failure_cleans_part_and_preserves_existing_destination(
    tmp_path: Path, config: geography.GeographyConfig,
    monkeypatch: pytest.MonkeyPatch, existing: bool,
) -> None:
    write_raw(tmp_path, config, [cog_row()])
    destination = output_path(tmp_path, config)
    previous = None
    if existing:
        geography.build_geography_year(2025, config, tmp_path)
        previous = destination.read_bytes()

    def reject_parquet(path: Path, year: int, expected_rows: int) -> None:
        assert Path(path).suffix == ".part"
        assert Path(path).is_file()
        raise geography.GeographyIntegrityError("Synthetic output validation failure")

    monkeypatch.setattr(geography, "validate_geography_parquet", reject_parquet)
    with pytest.raises(geography.GeographyIntegrityError, match="Synthetic"):
        geography.build_geography_year(2025, config, tmp_path, force=existing)
    assert_no_parts(tmp_path)
    if existing:
        assert destination.read_bytes() == previous
    else:
        assert not destination.exists()


def test_replace_failure_cleans_part_and_preserves_previous_output(
    tmp_path: Path, config: geography.GeographyConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_raw(tmp_path, config, [cog_row()])
    result = geography.build_geography_year(2025, config, tmp_path)
    previous = result.path.read_bytes()

    def fail_replace(*args: object, **kwargs: object) -> None:
        raise OSError("Synthetic replace failure")

    monkeypatch.setattr(geography.os, "replace", fail_replace)
    with pytest.raises((OSError, geography.GeographyError), match="Synthetic"):
        geography.build_geography_year(2025, config, tmp_path, force=True)
    assert result.path.read_bytes() == previous
    assert_no_parts(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_code_commune", "0101"),
        ("resolved_geo_type", "UNKNOWN"),
        ("canonical_commune_code", "0101"),
        ("canonical_commune_code", "01002"),
        ("canonical_commune_label", ""),
        ("region_code", ""),
        ("department_code", ""),
        ("cog_year", 2024),
        ("parent_commune_code", "01001"),
    ],
)
def test_output_validation_rejects_broken_com_invariants(
    tmp_path: Path, config: geography.GeographyConfig, field: str, value: object,
) -> None:
    write_raw(tmp_path, config, [cog_row()])
    result = geography.build_geography_year(2025, config, tmp_path)
    rows = pq.read_table(result.path).to_pylist()
    rows[0][field] = value
    corrupt = tmp_path / "invalid.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=geography.GEOGRAPHY_SCHEMA), corrupt)
    with pytest.raises(geography.GeographyIntegrityError):
        geography.validate_geography_parquet(corrupt, 2025, 1)


@pytest.mark.parametrize("parent", [None, "01003"])
def test_output_validation_rejects_missing_or_different_subordinate_parent(
    tmp_path: Path, config: geography.GeographyConfig, parent: str | None,
) -> None:
    write_raw(tmp_path, config, [
        cog_row(), cog_row(COM="01002", TYPECOM="ARM", COMPARENT="01001"),
    ])
    result = geography.build_geography_year(2025, config, tmp_path)
    rows = pq.read_table(result.path).to_pylist()
    rows[1]["parent_commune_code"] = parent
    corrupt = tmp_path / "invalid.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=geography.GEOGRAPHY_SCHEMA), corrupt)
    with pytest.raises(geography.GeographyIntegrityError):
        geography.validate_geography_parquet(corrupt, 2025, 2)


def test_output_validation_rejects_duplicate_source_codes(
    tmp_path: Path, config: geography.GeographyConfig,
) -> None:
    write_raw(tmp_path, config, [cog_row()])
    result = geography.build_geography_year(2025, config, tmp_path)
    rows = pq.read_table(result.path).to_pylist() * 2
    corrupt = tmp_path / "duplicate.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=geography.GEOGRAPHY_SCHEMA), corrupt)
    with pytest.raises(geography.GeographyIntegrityError):
        geography.validate_geography_parquet(corrupt, 2025, 2)


def test_output_validation_checks_expected_row_count(
    tmp_path: Path, config: geography.GeographyConfig,
) -> None:
    write_raw(tmp_path, config, [cog_row()])
    result = geography.build_geography_year(2025, config, tmp_path)
    with pytest.raises(geography.GeographyIntegrityError):
        geography.validate_geography_parquet(result.path, 2025, 2)


def test_cli_builds_synthetic_reference_and_force_is_explicit(
    tmp_path: Path, config: geography.GeographyConfig,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    write_raw(tmp_path, config, [cog_row()])
    monkeypatch.setattr(geography, "DEFAULT_CONFIG_PATH", tmp_path / "configs/data.yaml")
    monkeypatch.setattr(geography, "load_geography_config", lambda *args: config)

    assert geography.main(["--year", "2025"]) in (None, 0)
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "year": 2025, "output": str(output_path(tmp_path, config)), "source_codes": 1,
        "resolved_COM": 1, "resolved_ARM": 0, "resolved_COMD": 0, "resolved_COMA": 0,
    }
    write_raw(tmp_path, config, [cog_row(LIBELLE="Version reconstruite")])
    assert geography.main(["--year", "2025", "--force"]) in (None, 0)
    assert json.loads(capsys.readouterr().out)["source_codes"] == 1
    assert pq.read_table(output_path(tmp_path, config))["source_geo_label"].to_pylist() == [
        "Version reconstruite",
    ]
    assert_no_parts(tmp_path)


def test_config_cannot_omit_an_annual_member(config: geography.GeographyConfig) -> None:
    members = dict(config.current_commune_files)
    members.pop(2021)
    with pytest.raises((ValueError, geography.GeographyError)):
        replace(config, current_commune_files=members)
