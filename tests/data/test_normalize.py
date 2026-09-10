"""Exercise row-preserving normalization using only tiny local synthetic ZIPs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests

from real_estate.data import normalize as normalize_module
from real_estate.data.normalize import (
    DEFAULT_CONFIG_PATH,
    NORMALIZED_SCHEMA,
    SOURCE_COLUMNS,
    NormalizationConfig,
    NormalizationError,
    NormalizationResult,
    NormalizationValueError,
    RawProvenanceError,
    SourceSchemaError,
    load_normalization_config,
    normalize_dvf_year,
    normalize_parcel,
)

EXPECTED_SOURCE_COLUMNS = (
    "Identifiant de document", "Reference document", "1 Articles CGI",
    "2 Articles CGI", "3 Articles CGI", "4 Articles CGI", "5 Articles CGI",
    "No disposition", "Date mutation", "Nature mutation", "Valeur fonciere",
    "No voie", "B/T/Q", "Type de voie", "Code voie", "Voie", "Code postal",
    "Commune", "Code departement", "Code commune", "Prefixe de section",
    "Section", "No plan", "No Volume", "1er lot", "Surface Carrez du 1er lot",
    "2eme lot", "Surface Carrez du 2eme lot", "3eme lot", "Surface Carrez du 3eme lot",
    "4eme lot", "Surface Carrez du 4eme lot", "5eme lot", "Surface Carrez du 5eme lot",
    "Nombre de lots", "Code type local", "Type local", "Identifiant local",
    "Surface reelle bati", "Nombre pieces principales", "Nature culture",
    "Nature culture speciale", "Surface terrain",
)

EXPECTED_OUTPUT_COLUMNS = [
    "source_year", "source_row_number", "id_mutation", "date_mutation",
    "numero_disposition", "nature_mutation", "valeur_fonciere", "code_postal",
    "nom_commune", "code_departement", "code_commune", "prefixe_section", "section",
    "numero_plan", "id_parcelle", "nombre_lots", "code_type_local", "type_local",
    "surface_reelle_bati", "nombre_pieces_principales", "nature_culture",
    "nature_culture_speciale", "surface_terrain",
]


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if the normalization path ever attempts an HTTP request."""
    def fail_request(*args: object, **kwargs: object) -> None:
        raise AssertionError("Normalization must not access the network")

    monkeypatch.setattr(requests.Session, "request", fail_request)


@pytest.fixture
def config() -> NormalizationConfig:
    return NormalizationConfig(
        years=(2021, 2022, 2023, 2024, 2025),
        raw_directory=Path("data/raw/dvf"),
        manifest_filename="manifest.json",
        output_directory=Path("data/interim/dvf/normalized"),
        batch_size=2,
    )


def source_row(**overrides: str) -> dict[str, str]:
    """Create a purely synthetic source record with every required column."""
    row = dict.fromkeys(EXPECTED_SOURCE_COLUMNS, "")
    row.update({
        "No disposition": "000001",
        "Date mutation": "02/01/2025",
        "Nature mutation": "Vente",
        "Valeur fonciere": "100,00",
        "Code postal": "01000",
        "Commune": "COMMUNE FICTIVE",
        "Code departement": "01",
        "Code commune": "1",
        "Section": "A",
        "No plan": "1",
        "Nombre de lots": "0",
        "Code type local": "1",
        "Type local": "Maison",
        "Surface reelle bati": "50",
        "Nombre pieces principales": "2",
        "Surface terrain": "0",
    })
    row.update(overrides)
    return row


def write_raw(
    root: Path,
    config: NormalizationConfig,
    rows: list[dict[str, str]],
    *,
    header: tuple[str, ...] = EXPECTED_SOURCE_COLUMNS,
    text_override: str | None = None,
) -> tuple[Path, Path]:
    """Write a tiny test archive and its matching acquisition manifest."""
    text_buffer = io.StringIO(newline="")
    writer = csv.writer(text_buffer, delimiter="|", lineterminator="\n")
    writer.writerow(header)
    writer.writerows([[row.get(column, "") for column in header] for row in rows])
    archive_path = root / config.raw_directory / "dvf_2025.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        text = text_buffer.getvalue() if text_override is None else text_override
        archive.writestr("ValeursFoncieres-2025.txt", text.encode("utf-8"))
    manifest_path = archive_path.parent / config.manifest_filename
    manifest_path.write_text(json.dumps({
        "downloads": {"2025": {
            "year": 2025,
            "filename": archive_path.name,
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        }},
    }), encoding="utf-8")
    return archive_path, manifest_path


def assert_no_parts(root: Path) -> None:
    assert list(root.rglob("*.part")) == []


def test_committed_config_supports_all_five_years() -> None:
    config = load_normalization_config(DEFAULT_CONFIG_PATH)
    assert config.years == (2021, 2022, 2023, 2024, 2025)
    assert config.raw_directory == Path("data/raw/dvf")
    assert config.output_directory == Path("data/interim/dvf/normalized")
    assert config.manifest_filename == "manifest.json"
    assert config.batch_size > 0
    assert SOURCE_COLUMNS == EXPECTED_SOURCE_COLUMNS


@pytest.mark.parametrize("batch_size", [1, 2, 3, 20])
def test_grouping_is_contiguous_and_survives_batch_boundaries(
    tmp_path: Path, config: NormalizationConfig, batch_size: int,
) -> None:
    config = replace(config, batch_size=batch_size)
    rows = [
        source_row(),
        source_row(**{"Valeur fonciere": " 100 ", "No disposition": "000002"}),
        source_row(**{"Valeur fonciere": "100,0", "No disposition": "000002"}),
        source_row(**{"Valeur fonciere": "101", "No disposition": "000002"}),
        source_row(**{
            "Date mutation": "03/01/2025", "Valeur fonciere": "101",
            "No disposition": "000002",
        }),
        source_row(),
    ]
    raw_path, manifest_path = write_raw(tmp_path, config, rows)
    raw_before = raw_path.read_bytes()
    manifest_before = manifest_path.read_bytes()

    result = normalize_dvf_year(2025, config, tmp_path)

    table = pq.read_table(result.path)
    with pq.ParquetFile(result.path) as parquet:
        assert all(
            parquet.metadata.row_group(index).num_rows <= batch_size
            for index in range(parquet.metadata.num_row_groups)
        )
    assert result.path == tmp_path / config.output_directory / "dvf_2025.parquet"
    assert result.year == 2025
    assert result.row_count == 6
    assert result.mutation_count == 4
    assert result.raw_sha256 == hashlib.sha256(raw_before).hexdigest()
    assert table["id_mutation"].to_pylist() == [
        "2025-1", "2025-1", "2025-1", "2025-2", "2025-3", "2025-4",
    ]
    assert table["source_row_number"].to_pylist() == [1, 2, 3, 4, 5, 6]
    assert table["source_year"].to_pylist() == [2025] * 6
    assert raw_path.read_bytes() == raw_before
    assert manifest_path.read_bytes() == manifest_before
    assert_no_parts(tmp_path)


def test_exact_amount_comparison_precedes_float_conversion(
    tmp_path: Path, config: NormalizationConfig,
) -> None:
    amounts = ["9007199254740992", "9007199254740993", "9007199254740993,00"]
    write_raw(tmp_path, config, [source_row(**{"Valeur fonciere": x}) for x in amounts])

    table = pq.read_table(normalize_dvf_year(2025, config, tmp_path).path)

    assert table["id_mutation"].to_pylist() == ["2025-1", "2025-2", "2025-2"]


def test_blank_amount_and_zero_have_different_groups(
    tmp_path: Path, config: NormalizationConfig,
) -> None:
    amounts = ["", "  ", "0", "0,00", ""]
    write_raw(tmp_path, config, [source_row(**{"Valeur fonciere": x}) for x in amounts])

    table = pq.read_table(normalize_dvf_year(2025, config, tmp_path).path)

    assert table["valeur_fonciere"].to_pylist() == [None, None, 0.0, 0.0, None]
    assert table["id_mutation"].to_pylist() == [
        "2025-1", "2025-1", "2025-2", "2025-2", "2025-3",
    ]


@pytest.mark.parametrize("field,value", [
    ("Valeur fonciere", "invalid"),
    ("Valeur fonciere", "NaN"),
    ("Valeur fonciere", "Infinity"),
    ("Valeur fonciere", "1e400"),
    ("Surface reelle bati", "invalid"),
    ("Surface terrain", "-Infinity"),
    ("Nombre de lots", "1,5"),
    ("Nombre pieces principales", "invalid"),
    ("Code type local", "invalid"),
    ("Date mutation", "31/02/2025"),
    ("Date mutation", "2025-01-02"),
    ("Date mutation", ""),
])
def test_invalid_values_fail_explicitly(
    tmp_path: Path, config: NormalizationConfig, field: str, value: str,
) -> None:
    write_raw(tmp_path, config, [source_row(**{field: value})])

    with pytest.raises(NormalizationValueError):
        normalize_dvf_year(2025, config, tmp_path)

    assert not (tmp_path / config.output_directory / "dvf_2025.parquet").exists()
    assert_no_parts(tmp_path)


@pytest.mark.parametrize("department,commune,expected", [
    ("1", "2", "01002"),
    ("01", "002", "01002"),
    ("2A", "1", "2A001"),
    ("2B", "1", "2B001"),
    ("971", "1", "97101"),
    ("972", "1", "97201"),
    ("973", "1", "97301"),
    ("974", "1", "97401"),
])
def test_parcel_department_rules_and_padding(
    department: str, commune: str, expected: str,
) -> None:
    parcel = normalize_parcel(department, commune, "", "A", "12")
    assert parcel["code_commune"] == expected
    assert parcel["prefixe_section"] == "000"
    assert parcel["section"] == "0A"
    assert parcel["numero_plan"] == "0012"
    assert parcel["id_parcelle"] == expected + "0000A0012"
    assert len(parcel["id_parcelle"]) == 14
    assert isinstance(parcel["code_departement"], str)


def test_parcel_present_prefix_is_padded_without_truncation() -> None:
    parcel = normalize_parcel(" 01 ", " 2 ", " 4 ", " AB ", " 12 ")
    assert parcel["prefixe_section"] == "004"
    assert parcel["id_parcelle"] == "01002004AB0012"


@pytest.mark.parametrize("components", [
    ("", "1", "", "A", "1"),
    ("01", "", "", "A", "1"),
    ("01", "1", "", "", "1"),
    ("01", "1", "", "A", ""),
    ("XYZ", "1", "", "A", "1"),
    ("01", "1000", "", "A", "1"),
    ("971", "001", "", "A", "1"),
    ("01", "1", "1234", "A", "1"),
    ("01", "1", "", "ABC", "1"),
    ("01", "1", "", "A", "10000"),
    ("01", "1", "", "A", "1.5"),
])
def test_invalid_parcel_components_are_rejected(components: tuple[str, ...]) -> None:
    with pytest.raises(NormalizationValueError):
        normalize_parcel(*components)


def test_output_schema_dates_nulls_zeroes_and_address_exclusion(
    tmp_path: Path, config: NormalizationConfig,
) -> None:
    nullable_fields = [
        "Valeur fonciere", "Nombre de lots", "Code type local",
        "Surface reelle bati", "Nombre pieces principales", "Surface terrain",
    ]
    rows = [
        source_row(**dict.fromkeys(nullable_fields, "")),
        source_row(**dict.fromkeys(nullable_fields, "0")),
    ]
    for row in rows:
        row.update({"No voie": "123", "Voie": "VOIE FICTIVE", "Type de voie": "RUE"})
    write_raw(tmp_path, config, rows)

    table = pq.read_table(normalize_dvf_year(2025, config, tmp_path).path)

    assert table.column_names == EXPECTED_OUTPUT_COLUMNS
    assert table.schema.equals(NORMALIZED_SCHEMA, check_metadata=False)
    assert table.schema.field("source_year").type == pa.int32()
    assert table.schema.field("source_row_number").type == pa.int64()
    assert table["date_mutation"].to_pylist() == ["2025-01-02"] * 2
    assert table["numero_disposition"].to_pylist() == ["000001"] * 2
    assert table["code_postal"].to_pylist() == ["01000"] * 2
    assert table["code_commune"].to_pylist() == ["01001"] * 2
    for field in ("valeur_fonciere", "surface_reelle_bati", "surface_terrain"):
        assert table.schema.field(field).type == pa.float64()
        assert table[field].to_pylist() == [None, 0.0]
        assert math.isfinite(table[field][1].as_py())
    for field in ("nombre_lots", "code_type_local", "nombre_pieces_principales"):
        assert table.schema.field(field).type == pa.int64()
        assert table[field].to_pylist() == [None, 0]
    assert "VOIE FICTIVE" not in str(table.to_pylist())


def test_no_business_qualification_or_deduplication_occurs(
    tmp_path: Path, config: NormalizationConfig,
) -> None:
    row = source_row(**{
        "Nature mutation": "Echange", "Valeur fonciere": "-10",
        "Code type local": "4", "Type local": "Local industriel, commercial ou assimilé",
        "Surface reelle bati": "-5",
    })
    write_raw(tmp_path, config, [row, row.copy()])

    result = normalize_dvf_year(2025, config, tmp_path)
    table = pq.read_table(result.path)

    assert result.row_count == 2
    assert result.mutation_count == 1
    assert table["valeur_fonciere"].to_pylist() == [-10.0, -10.0]
    assert table["surface_reelle_bati"].to_pylist() == [-5.0, -5.0]
    assert table["code_type_local"].to_pylist() == [4, 4]


@pytest.mark.parametrize("failure", [
    "zip_missing", "manifest_missing", "year_missing", "filename_mismatch",
    "sha256_mismatch", "malformed_manifest",
])
def test_raw_provenance_failure_precedes_transformation(
    tmp_path: Path, config: NormalizationConfig, failure: str,
) -> None:
    raw_path, manifest_path = write_raw(tmp_path, config, [source_row()])
    if failure == "zip_missing":
        raw_path.unlink()
    elif failure == "manifest_missing":
        manifest_path.unlink()
    elif failure == "malformed_manifest":
        manifest_path.write_text("invalid JSON", encoding="utf-8")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if failure == "year_missing":
            manifest["downloads"].clear()
        elif failure == "filename_mismatch":
            manifest["downloads"]["2025"]["filename"] = "other.zip"
        else:
            manifest["downloads"]["2025"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RawProvenanceError):
        normalize_dvf_year(2025, config, tmp_path)

    assert not (tmp_path / config.output_directory / "dvf_2025.parquet").exists()
    assert_no_parts(tmp_path)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "extra", "reordered"])
def test_source_header_drift_fails(tmp_path: Path, config: NormalizationConfig, failure: str) -> None:
    columns = list(EXPECTED_SOURCE_COLUMNS)
    if failure == "missing":
        columns.remove("No disposition")
    elif failure == "duplicate":
        columns[-1] = "No disposition"
    elif failure == "extra":
        columns.append("Unexpected column")
    else:
        columns[0], columns[1] = columns[1], columns[0]
    write_raw(tmp_path, config, [source_row()], header=tuple(columns))

    with pytest.raises(SourceSchemaError):
        normalize_dvf_year(2025, config, tmp_path)

    assert_no_parts(tmp_path)


@pytest.mark.parametrize("bad_row", ["too|short", "|" * 43, ""])
def test_invalid_record_width_fails(
    tmp_path: Path, config: NormalizationConfig, bad_row: str,
) -> None:
    text = "|".join(EXPECTED_SOURCE_COLUMNS) + "\n" + bad_row + "\n"
    write_raw(tmp_path, config, [], text_override=text)

    with pytest.raises(SourceSchemaError):
        normalize_dvf_year(2025, config, tmp_path)

    assert_no_parts(tmp_path)


def test_existing_output_requires_force(tmp_path: Path, config: NormalizationConfig) -> None:
    write_raw(tmp_path, config, [source_row()])
    result = normalize_dvf_year(2025, config, tmp_path)
    original = result.path.read_bytes()

    with pytest.raises(FileExistsError):
        normalize_dvf_year(2025, config, tmp_path)

    assert result.path.read_bytes() == original
    assert_no_parts(tmp_path)


def test_force_explicitly_replaces_output(tmp_path: Path, config: NormalizationConfig) -> None:
    write_raw(tmp_path, config, [source_row()])
    first = normalize_dvf_year(2025, config, tmp_path)
    write_raw(tmp_path, config, [source_row(), source_row(**{"Valeur fonciere": "200"})])

    replacement = normalize_dvf_year(2025, config, tmp_path, force=True)

    assert replacement.path == first.path
    assert pq.read_table(replacement.path).num_rows == 2
    assert replacement.mutation_count == 2
    assert_no_parts(tmp_path)


@pytest.mark.parametrize("existing_output", [False, True])
def test_later_batch_failure_cleans_part_and_preserves_existing_output(
    tmp_path: Path, config: NormalizationConfig, existing_output: bool,
) -> None:
    config = replace(config, batch_size=1)
    output = tmp_path / config.output_directory / "dvf_2025.parquet"
    previous_bytes = None
    if existing_output:
        write_raw(tmp_path, config, [source_row()])
        normalize_dvf_year(2025, config, tmp_path)
        previous_bytes = output.read_bytes()
    write_raw(tmp_path, config, [source_row(), source_row(**{"Valeur fonciere": "invalid"})])

    with pytest.raises(NormalizationValueError):
        normalize_dvf_year(2025, config, tmp_path, force=existing_output)

    if existing_output:
        assert output.read_bytes() == previous_bytes
    else:
        assert not output.exists()
    assert_no_parts(tmp_path)


def test_corrupt_archive_is_rejected(tmp_path: Path, config: NormalizationConfig) -> None:
    raw_path, manifest_path = write_raw(tmp_path, config, [source_row()])
    raw_path.write_bytes(b"not a zip")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["downloads"]["2025"]["sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(NormalizationError):
        normalize_dvf_year(2025, config, tmp_path)

    assert_no_parts(tmp_path)


def test_unconfigured_year_fails_without_raw_access(
    tmp_path: Path, config: NormalizationConfig,
) -> None:
    with pytest.raises(ValueError):
        normalize_dvf_year(2020, config, tmp_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("force", [False, True])
def test_cli_selects_year_and_forwards_force_without_real_transformation(
    tmp_path: Path, config: NormalizationConfig, monkeypatch: pytest.MonkeyPatch, force: bool,
) -> None:
    calls: list[tuple[int, NormalizationConfig, bool]] = []

    def fake_normalize(
        year: int, supplied_config: NormalizationConfig, project_root: Path, *, force: bool = False,
    ) -> NormalizationResult:
        calls.append((year, supplied_config, force))
        return NormalizationResult(year, tmp_path / "synthetic.parquet", 2, 1, "0" * 64)

    monkeypatch.setattr(normalize_module, "load_normalization_config", lambda: config)
    monkeypatch.setattr(normalize_module, "normalize_dvf_year", fake_normalize)
    arguments = ["--year", "2025"] + (["--force"] if force else [])

    assert normalize_module.main(arguments) == 0
    assert calls == [(2025, config, force)]
    assert list(tmp_path.iterdir()) == []
