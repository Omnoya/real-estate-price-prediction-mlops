"""Exercise the public geography left join with tiny local Parquet inputs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests
import yaml

from real_estate.data import enrich_geography as enrichment
from real_estate.data.geography import GEOGRAPHY_SCHEMA
from real_estate.data.qualify import QUALIFIED_SCHEMA

GEOGRAPHIC_COLUMNS = (
    "source_code_commune", "resolved_geo_type", "source_geo_label",
    "parent_commune_code", "canonical_commune_code", "canonical_commune_label",
    "region_code", "department_code", "cog_year",
)


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail unexpected requests; all sources below are synthetic local files."""
    def fail_request(*args: object, **kwargs: object) -> None:
        raise AssertionError("Enrichment tests must never access the network")

    monkeypatch.setattr(requests.Session, "request", fail_request)


@pytest.fixture
def config() -> enrichment.EnrichmentConfig:
    return enrichment.EnrichmentConfig(
        years=(2021, 2022, 2023, 2024, 2025),
        qualified_input="data/processed/dvf/qualified/dvf_{year}.parquet",
        geography_input="data/interim/cog/geography/cog_geography_{year}.parquet",
        output="data/processed/dvf/geography/dvf_{year}.parquet",
        batch_size=2,
    )


def dvf_row(**overrides: object) -> dict[str, object]:
    """Include every authoritative qualified column, including nullable metadata."""
    row = {
        "source_year": 2025, "id_mutation": "2025-1", "date_mutation": "2025-01-02",
        "numero_disposition": "000001", "id_parcelle": "010010000A0001",
        "code_postal": "01000", "nom_commune": "Different source label",
        "code_departement": "01", "code_commune": "01001", "code_type_local": 1,
        "type_local": "Maison", "valeur_fonciere": 100000.25,
        "surface_reelle_bati": 80.0, "nombre_pieces_principales": 4,
        "nombre_lots": 0, "surface_terrain": None, "has_dependance": False,
        "prix_m2": 1250.003125, "source_row_count": 1,
        "longitude": None, "latitude": None,
    }
    row.update(overrides)
    return row


def geo_row(**overrides: object) -> dict[str, object]:
    row = {
        "source_code_commune": "01001", "resolved_geo_type": "COM",
        "source_geo_label": "Canonical commune", "parent_commune_code": None,
        "canonical_commune_code": "01001", "canonical_commune_label": "Canonical commune",
        "region_code": "01", "department_code": "01", "cog_year": 2025,
    }
    row.update(overrides)
    return row


def child_row(kind: str = "ARM", code: str = "01002") -> dict[str, object]:
    return geo_row(
        source_code_commune=code, resolved_geo_type=kind,
        source_geo_label=f"Source {kind}", parent_commune_code="01001",
    )


def configured_path(root: Path, template: str, year: int = 2025) -> Path:
    return root / template.format(year=year)


def write_inputs(
    root: Path,
    config: enrichment.EnrichmentConfig,
    dvf: list[dict[str, object]] | None = None,
    geography: list[dict[str, object]] | None = None,
    *,
    year: int = 2025,
    dvf_schema: pa.Schema = QUALIFIED_SCHEMA,
    geography_schema: pa.Schema = GEOGRAPHY_SCHEMA,
) -> tuple[Path, Path]:
    """Only this temporary project receives files, never the real project data."""
    dvf_path = configured_path(root, config.qualified_input, year)
    geo_path = configured_path(root, config.geography_input, year)
    dvf_path.parent.mkdir(parents=True, exist_ok=True)
    geo_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([dvf_row(source_year=year)] if dvf is None else dvf, schema=dvf_schema),
        dvf_path, row_group_size=2,
    )
    pq.write_table(
        pa.Table.from_pylist([geo_row(cog_year=year)] if geography is None else geography,
                             schema=geography_schema),
        geo_path,
    )
    return dvf_path, geo_path


def no_parts(root: Path) -> None:
    assert list(root.rglob("*.part")) == []


def write_config(root: Path, config: enrichment.EnrichmentConfig) -> Path:
    path = root / "configs" / "data.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "dvf": {
            "years": list(config.years),
            "geography_enrichment": {
                "qualified_input": config.qualified_input,
                "geography_input": config.geography_input,
                "output": config.output,
                "batch_size": config.batch_size,
            },
        },
        "cog": {"years": list(config.years)},
    }), encoding="utf-8")
    return path


def test_project_configuration_and_exact_geographic_columns() -> None:
    actual = enrichment.load_enrichment_config(enrichment.DEFAULT_CONFIG_PATH)
    assert actual.years == (2021, 2022, 2023, 2024, 2025)
    assert actual.qualified_input == "data/processed/dvf/qualified/dvf_{year}.parquet"
    assert actual.geography_input == "data/interim/cog/geography/cog_geography_{year}.parquet"
    assert actual.output == "data/processed/dvf/geography/dvf_{year}.parquet"
    assert actual.batch_size > 0
    assert tuple(enrichment.GEOGRAPHIC_COLUMNS) == GEOGRAPHIC_COLUMNS
    assert enrichment.ADDED_GEOGRAPHY_SCHEMA.names == list(GEOGRAPHIC_COLUMNS)
    for field in enrichment.ADDED_GEOGRAPHY_SCHEMA:
        assert field.type == (pa.int32() if field.name == "cog_year" else pa.string())
        assert field.nullable == (field.name not in {"source_code_commune", "cog_year"})


def test_load_synthetic_configuration(tmp_path: Path, config: enrichment.EnrichmentConfig) -> None:
    assert enrichment.load_enrichment_config(write_config(tmp_path, config)) == config


@pytest.mark.parametrize("year", [2020, 2026])
def test_unconfigured_year_rejected_before_writing(
    tmp_path: Path, config: enrichment.EnrichmentConfig, year: int,
) -> None:
    with pytest.raises((ValueError, enrichment.EnrichmentError)):
        enrichment.enrich_dvf_geography_year(year, config, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("year", [2021, 2022, 2023, 2024, 2025])
def test_public_orchestration_for_each_configured_year(
    tmp_path: Path, config: enrichment.EnrichmentConfig, year: int,
) -> None:
    source, reference = write_inputs(tmp_path, config, year=year)
    before = source.read_bytes(), reference.read_bytes()
    result = enrichment.enrich_dvf_geography_year(year, config, tmp_path)
    rows = pq.read_table(result.path).to_pylist()
    assert len(rows) == 1
    assert rows[0]["cog_year"] == rows[0]["source_year"] == year
    assert rows[0]["source_code_commune"] == rows[0]["code_commune"] == "01001"
    assert rows[0]["source_geo_label"] == "Canonical commune"
    assert rows[0]["nom_commune"] == "Different source label"
    assert result.year == year
    assert result.report.rows == result.report.resolved == 1
    assert result.report.unresolved == 0
    assert (source.read_bytes(), reference.read_bytes()) == before
    no_parts(tmp_path)


@pytest.mark.parametrize("kind", ["COM", "ARM", "COMD", "COMA"])
def test_join_copies_reference_without_resolving_it_again(
    tmp_path: Path, config: enrichment.EnrichmentConfig, kind: str,
) -> None:
    expected = geo_row() if kind == "COM" else child_row(kind)
    reference = [geo_row()] if kind == "COM" else [geo_row(), expected]
    write_inputs(tmp_path, config, [dvf_row(code_commune=expected["source_code_commune"])], reference)
    result = enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    actual = pq.read_table(result.path).to_pylist()[0]
    assert {name: actual[name] for name in GEOGRAPHIC_COLUMNS} == expected
    assert actual["code_commune"] == expected["source_code_commune"]
    assert result.report.resolved_counts == {name: int(name == kind) for name in ("COM", "ARM", "COMD", "COMA")}


def test_unresolved_preserves_source_code_and_requested_year_with_exact_nulls(
    tmp_path: Path, config: enrichment.EnrichmentConfig,
) -> None:
    # Matching label and department must never rescue an absent commune code.
    write_inputs(tmp_path, config, [dvf_row(code_commune="01999", nom_commune="Canonical commune")])
    result = enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    row = pq.read_table(result.path).to_pylist()[0]
    assert row["source_code_commune"] == row["code_commune"] == "01999"
    assert row["cog_year"] == row["source_year"] == 2025
    assert all(row[name] is None for name in GEOGRAPHIC_COLUMNS[1:-1])
    assert result.report.rows == result.report.unresolved == 1
    assert result.report.resolved == 0


def test_multiple_batches_preserve_all_source_columns_order_values_and_counts(
    tmp_path: Path, config: enrichment.EnrichmentConfig,
) -> None:
    source_rows = [
        dvf_row(id_mutation="2025-8", has_dependance=True, surface_terrain=0.0),
        dvf_row(id_mutation="2025-3", code_commune="01002", nombre_lots=None),
        dvf_row(id_mutation="2025-9", code_commune="01999", code_postal=None),
        dvf_row(id_mutation="2025-2", code_commune="01003", longitude=-1.25, latitude=48.75),
        dvf_row(id_mutation="2025-7", code_commune="01004", nombre_pieces_principales=None),
        dvf_row(id_mutation="2025-1", nom_commune=None),
    ]
    reference = [geo_row(), child_row("ARM", "01002"), child_row("COMD", "01003"), child_row("COMA", "01004")]
    source, _ = write_inputs(tmp_path, config, source_rows, reference)
    source_table = pq.read_table(source)
    result = enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    output = pq.read_table(result.path)
    assert output.num_rows == source_table.num_rows == 6
    assert output.num_columns == 30
    assert output.column_names == source_table.column_names + list(GEOGRAPHIC_COLUMNS)
    assert output.select(source_table.column_names).equals(source_table)
    assert output["id_mutation"].to_pylist() == [row["id_mutation"] for row in source_rows]
    assert len(set(output["id_mutation"].to_pylist())) == 6
    assert output["source_code_commune"].equals(output["code_commune"])
    assert output["cog_year"].equals(output["source_year"])
    assert result.to_dict() == {
        "year": 2025, "output": str(result.path), "rows": 6, "resolved": 5,
        "unresolved": 1, "resolved_COM": 2, "resolved_ARM": 1,
        "resolved_COMD": 1, "resolved_COMA": 1,
    }
    no_parts(tmp_path)


def test_original_qualified_schema_order_and_metadata_are_preserved(
    tmp_path: Path, config: enrichment.EnrichmentConfig,
) -> None:
    fields = [field.with_nullable(True) for field in reversed(QUALIFIED_SCHEMA)]
    fields[0] = fields[0].with_metadata({b"unit": b"degrees"})
    schema = pa.schema(fields, metadata={b"synthetic-source": b"preserve"})
    source, _ = write_inputs(tmp_path, config, dvf_schema=schema)
    result = enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    source_table = pq.read_table(source)
    actual = pq.read_table(result.path).select(source_table.column_names)
    assert actual.equals(source_table, check_metadata=True)


@pytest.mark.parametrize("missing", ["dvf", "geography"])
def test_missing_input_is_explicit_error(
    tmp_path: Path, config: enrichment.EnrichmentConfig, missing: str,
) -> None:
    source, reference = write_inputs(tmp_path, config)
    (source if missing == "dvf" else reference).unlink()
    with pytest.raises(enrichment.EnrichmentError):
        enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    assert not configured_path(tmp_path, config.output).exists()
    no_parts(tmp_path)


@pytest.mark.parametrize("which", ["dvf", "geography"])
def test_unreadable_parquet_is_explicit_error(
    tmp_path: Path, config: enrichment.EnrichmentConfig, which: str,
) -> None:
    source, reference = write_inputs(tmp_path, config)
    (source if which == "dvf" else reference).write_bytes(b"not parquet")
    with pytest.raises(enrichment.EnrichmentError):
        enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    assert not configured_path(tmp_path, config.output).exists()
    no_parts(tmp_path)


@pytest.mark.parametrize("column", ["source_year", "id_mutation", "code_commune", "code_departement", "prix_m2"])
def test_missing_qualified_columns_fail(
    tmp_path: Path, config: enrichment.EnrichmentConfig, column: str,
) -> None:
    schema = pa.schema([field for field in QUALIFIED_SCHEMA if field.name != column])
    write_inputs(tmp_path, config, dvf_schema=schema)
    with pytest.raises(enrichment.EnrichmentError):
        enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    no_parts(tmp_path)


@pytest.mark.parametrize("alteration", ["missing", "extra", "wrong_type", "wrong_order", "duplicate"])
def test_invalid_geography_schema_is_rejected(
    tmp_path: Path, config: enrichment.EnrichmentConfig, alteration: str,
) -> None:
    source, reference = write_inputs(tmp_path, config)
    table = pq.read_table(reference)
    if alteration == "missing":
        table = table.drop(["region_code"])
    elif alteration == "extra":
        table = table.append_column("unexpected", pa.array(["x"]))
    elif alteration == "wrong_type":
        index = table.schema.get_field_index("cog_year")
        table = table.set_column(index, "cog_year", pa.array(["2025"]))
    elif alteration == "wrong_order":
        table = table.select(list(reversed(table.column_names)))
    else:
        table = table.append_column("region_code", pa.array(["01"]))
    pq.write_table(table, reference)
    with pytest.raises(enrichment.EnrichmentError):
        enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    assert source.exists()
    no_parts(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_year", 2024), ("source_year", None),
        ("code_commune", None), ("code_commune", ""),
        ("code_commune", "1001"), ("code_commune", "010001"),
        ("id_mutation", None),
    ],
)
def test_invalid_dvf_values_are_rejected(
    tmp_path: Path, config: enrichment.EnrichmentConfig, field: str, value: object,
) -> None:
    # Nullable test schemas let malformed values reach the application validator.
    schema = pa.schema([item.with_nullable(True) for item in QUALIFIED_SCHEMA])
    write_inputs(tmp_path, config, [dvf_row(**{field: value})], dvf_schema=schema)
    with pytest.raises(enrichment.EnrichmentError):
        enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    assert not configured_path(tmp_path, config.output).exists()
    no_parts(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_code_commune", ""), ("source_code_commune", "0100"),
        ("cog_year", 2024), ("resolved_geo_type", "UNKNOWN"),
        ("canonical_commune_code", "01002"),
        ("canonical_commune_label", ""), ("region_code", ""),
        ("department_code", ""), ("parent_commune_code", "01002"),
    ],
)
def test_invalid_reference_invariants_are_rejected(
    tmp_path: Path, config: enrichment.EnrichmentConfig, field: str, value: object,
) -> None:
    write_inputs(tmp_path, config, geography=[geo_row(**{field: value})])
    with pytest.raises(enrichment.EnrichmentError):
        enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    no_parts(tmp_path)


def test_duplicate_reference_key_is_rejected_not_multiplied(
    tmp_path: Path, config: enrichment.EnrichmentConfig,
) -> None:
    write_inputs(tmp_path, config, geography=[geo_row(), geo_row()])
    with pytest.raises(enrichment.EnrichmentError):
        enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    assert not configured_path(tmp_path, config.output).exists()
    no_parts(tmp_path)


@pytest.mark.parametrize("alteration", ["absent_parent", "wrong_parent_type", "label", "region", "department"])
def test_reference_canonical_attributes_must_match_unique_com(
    tmp_path: Path, config: enrichment.EnrichmentConfig, alteration: str,
) -> None:
    parent, child = geo_row(), child_row()
    reference = [parent, child]
    if alteration == "absent_parent":
        reference = [child]
    elif alteration == "wrong_parent_type":
        parent.update(resolved_geo_type="COMD", parent_commune_code="01001")
    else:
        field = {"label": "canonical_commune_label", "region": "region_code", "department": "department_code"}[alteration]
        child[field] = "different"
    write_inputs(tmp_path, config, geography=reference)
    with pytest.raises(enrichment.EnrichmentError):
        enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    no_parts(tmp_path)


@pytest.mark.parametrize("department", ["02", None, ""])
def test_resolved_department_divergence_fails(
    tmp_path: Path, config: enrichment.EnrichmentConfig, department: str | None,
) -> None:
    write_inputs(tmp_path, config, [dvf_row(code_departement=department)])
    with pytest.raises(enrichment.EnrichmentIntegrityError):
        enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    assert not configured_path(tmp_path, config.output).exists()
    no_parts(tmp_path)


@pytest.mark.parametrize("empty", ["dvf", "geography"])
def test_empty_input_cannot_publish_empty_output(
    tmp_path: Path, config: enrichment.EnrichmentConfig, empty: str,
) -> None:
    write_inputs(tmp_path, config, dvf=[] if empty == "dvf" else None,
                 geography=[] if empty == "geography" else None)
    with pytest.raises(enrichment.EnrichmentError):
        enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    assert not configured_path(tmp_path, config.output).exists()
    no_parts(tmp_path)


def test_existing_output_refused_and_force_explicitly_rebuilds(
    tmp_path: Path, config: enrichment.EnrichmentConfig,
) -> None:
    write_inputs(tmp_path, config)
    first = enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    before = first.path.read_bytes()
    with pytest.raises(FileExistsError):
        enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    assert first.path.read_bytes() == before
    write_inputs(tmp_path, config, [dvf_row(id_mutation="2025-99")])
    rebuilt = enrichment.enrich_dvf_geography_year(2025, config, tmp_path, force=True)
    assert pq.read_table(rebuilt.path)["id_mutation"].to_pylist() == ["2025-99"]
    no_parts(tmp_path)


def test_part_is_validated_then_replaced_atomically(
    tmp_path: Path, config: enrichment.EnrichmentConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, reference = write_inputs(tmp_path, config)
    before = source.read_bytes(), reference.read_bytes()
    destination = configured_path(tmp_path, config.output)
    events = []
    real_validate = enrichment.validate_enriched_parquet
    real_replace = enrichment.os.replace

    def validate(*args: object, **kwargs: object) -> None:
        assert args[1] == Path(str(destination) + ".part")
        assert args[1].is_file()
        assert not destination.exists()
        real_validate(*args, **kwargs)
        events.append("validated")

    def publish(staged: Path, final: Path) -> None:
        assert events == ["validated"]
        assert staged == Path(str(destination) + ".part")
        assert final == destination
        assert pq.read_table(staged).num_columns == 30
        events.append("replaced")
        real_replace(staged, final)

    monkeypatch.setattr(enrichment, "validate_enriched_parquet", validate)
    monkeypatch.setattr(enrichment.os, "replace", publish)
    enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    assert events == ["validated", "replaced"]
    assert (source.read_bytes(), reference.read_bytes()) == before
    no_parts(tmp_path)


def test_error_in_later_batch_removes_part_preserving_existing_final(
    tmp_path: Path, config: enrichment.EnrichmentConfig,
) -> None:
    write_inputs(tmp_path, config)
    existing = enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    previous = existing.path.read_bytes()
    rows = [dvf_row(id_mutation=f"2025-{n}") for n in range(1, 6)]
    rows[-1]["code_departement"] = "02"
    write_inputs(tmp_path, config, rows)
    with pytest.raises(enrichment.EnrichmentIntegrityError):
        enrichment.enrich_dvf_geography_year(2025, config, tmp_path, force=True)
    assert existing.path.read_bytes() == previous
    no_parts(tmp_path)


@pytest.mark.parametrize("failure", ["validation", "replace"])
def test_publication_failure_cleans_owned_part_and_preserves_final(
    tmp_path: Path, config: enrichment.EnrichmentConfig,
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    write_inputs(tmp_path, config)
    existing = enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    previous = existing.path.read_bytes()

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic publication failure")

    if failure == "validation":
        monkeypatch.setattr(enrichment, "validate_enriched_parquet", fail)
    else:
        monkeypatch.setattr(enrichment.os, "replace", fail)
    with pytest.raises((OSError, enrichment.EnrichmentError)):
        enrichment.enrich_dvf_geography_year(2025, config, tmp_path, force=True)
    assert existing.path.read_bytes() == previous
    no_parts(tmp_path)


def test_existing_part_is_not_overwritten_or_deleted_even_with_force(
    tmp_path: Path, config: enrichment.EnrichmentConfig,
) -> None:
    write_inputs(tmp_path, config)
    destination = configured_path(tmp_path, config.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(destination) + ".part")
    part.write_bytes(b"another execution owns this part")
    with pytest.raises((FileExistsError, enrichment.EnrichmentError)):
        enrichment.enrich_dvf_geography_year(2025, config, tmp_path, force=True)
    assert part.read_bytes() == b"another execution owns this part"
    assert not destination.exists()


@pytest.mark.parametrize("source_template", ["qualified_input", "geography_input"])
def test_output_cannot_replace_input_even_with_force(
    tmp_path: Path, config: enrichment.EnrichmentConfig, source_template: str,
) -> None:
    source, reference = write_inputs(tmp_path, config)
    before = source.read_bytes(), reference.read_bytes()
    with pytest.raises((ValueError, enrichment.EnrichmentError)):
        unsafe = replace(config, output=getattr(config, source_template))
        enrichment.enrich_dvf_geography_year(2025, unsafe, tmp_path, force=True)
    assert (source.read_bytes(), reference.read_bytes()) == before
    no_parts(tmp_path)


def test_reordered_staged_rows_are_rejected_before_publication(
    tmp_path: Path, config: enrichment.EnrichmentConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [dvf_row(id_mutation=f"2025-{n}") for n in range(1, 4)]
    write_inputs(tmp_path, config, rows)
    validate = enrichment.validate_enriched_parquet

    def corrupt_then_validate(*args: object, **kwargs: object) -> None:
        part = args[1]
        table = pq.read_table(part)
        pq.write_table(table.take(pa.array([2, 1, 0])), part)
        validate(*args, **kwargs)

    monkeypatch.setattr(enrichment, "validate_enriched_parquet", corrupt_then_validate)
    with pytest.raises(enrichment.EnrichmentIntegrityError):
        enrichment.enrich_dvf_geography_year(2025, config, tmp_path)
    assert not configured_path(tmp_path, config.output).exists()
    no_parts(tmp_path)


def test_public_batch_rejects_broken_enriched_invariants(
    tmp_path: Path, config: enrichment.EnrichmentConfig,
) -> None:
    _, reference = write_inputs(tmp_path, config)
    index = enrichment.load_geography_index(reference, 2025)
    batch = pa.RecordBatch.from_pylist([dvf_row()], schema=QUALIFIED_SCHEMA)
    enriched = enrichment.enrich_batch(batch, index, 2025)
    enrichment.validate_enriched_batch(batch, enriched, 2025)
    field_index = enriched.schema.get_field_index("source_code_commune")
    corrupt = enriched.set_column(field_index, enriched.schema[field_index], pa.array(["01999"]))
    with pytest.raises(enrichment.EnrichmentIntegrityError):
        enrichment.validate_enriched_batch(batch, corrupt, 2025)


def test_public_batch_rejects_an_index_from_another_cog_year(
    tmp_path: Path, config: enrichment.EnrichmentConfig,
) -> None:
    _, reference = write_inputs(tmp_path, config, year=2024)
    index = enrichment.load_geography_index(reference, 2024)
    batch = pa.RecordBatch.from_pylist([dvf_row()], schema=QUALIFIED_SCHEMA)

    with pytest.raises(enrichment.EnrichmentIntegrityError, match="index year"):
        enrichment.enrich_batch(batch, index, 2025)


def test_cli_runs_on_synthetic_project_and_emits_exact_json(
    tmp_path: Path, config: enrichment.EnrichmentConfig,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    write_inputs(tmp_path, config, [dvf_row(), dvf_row(id_mutation="2025-2", code_commune="01999")])
    monkeypatch.setattr(enrichment, "DEFAULT_CONFIG_PATH", write_config(tmp_path, config))
    assert enrichment.main(["--year", "2025"]) == 0
    output = configured_path(tmp_path, config.output)
    assert json.loads(capsys.readouterr().out) == {
        "year": 2025, "output": str(output), "rows": 2, "resolved": 1, "unresolved": 1,
        "resolved_COM": 1, "resolved_ARM": 0, "resolved_COMD": 0, "resolved_COMA": 0,
    }
    assert enrichment.main(["--year", "2025", "--force"]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] == 2
    no_parts(tmp_path)
