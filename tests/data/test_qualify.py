"""Run qualification only on tiny, temporary normalized Parquet fixtures."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests

from real_estate.data import clean as clean_module
from real_estate.data import qualify as qualify_module
from real_estate.data.clean import build_observation, qualify_mutation
from real_estate.data.normalize import NORMALIZED_SCHEMA
from real_estate.data.qualify import (
    DEFAULT_CONFIG_PATH,
    QUALIFIED_SCHEMA,
    InputIntegrityError,
    QualificationConfig,
    load_qualification_config,
    qualify_dvf_year,
)
from real_estate.data.validate import ExclusionReason

EXPECTED_OUTPUT_COLUMNS = [
    "source_year", "id_mutation", "date_mutation", "numero_disposition",
    "id_parcelle", "code_postal", "nom_commune", "code_departement", "code_commune",
    "code_type_local", "type_local", "valeur_fonciere", "surface_reelle_bati",
    "nombre_pieces_principales", "nombre_lots", "surface_terrain", "has_dependance",
    "prix_m2", "source_row_count", "longitude", "latitude",
]

# Nullable fields also allow deliberately malformed input for integrity tests.
INPUT_SCHEMA = pa.schema([field.with_nullable(True) for field in NORMALIZED_SCHEMA])
ANNEX = {
    "code_type_local": 3,
    "type_local": "Dépendance",
    "surface_reelle_bati": None,
    "nombre_pieces_principales": 0,
}


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """A programming error must never send these tests to a real HTTP server."""
    def fail_request(*args: object, **kwargs: object) -> None:
        raise AssertionError("Qualification must not access the network")

    monkeypatch.setattr(requests.Session, "request", fail_request)


@pytest.fixture
def config() -> QualificationConfig:
    return QualificationConfig(
        years=(2021, 2022, 2023, 2024, 2025),
        input_directory=Path("data/interim/dvf/normalized"),
        output_directory=Path("data/processed/dvf/qualified"),
        batch_size=2,
    )


def row(**overrides: object) -> dict[str, object]:
    """One invented normalized dwelling; never read any real local data."""
    return {
        "source_year": 2025,
        "source_row_number": 1,
        "id_mutation": "2025-1",
        "date_mutation": "2025-01-02",
        "numero_disposition": "000001",
        "nature_mutation": "Vente",
        "valeur_fonciere": 250000.0,
        "code_postal": "01000",
        "nom_commune": "COMMUNE FICTIVE",
        "code_departement": "01",
        "code_commune": "01001",
        "prefixe_section": "000",
        "section": "0A",
        "numero_plan": "0001",
        "id_parcelle": "010010000A0001",
        "nombre_lots": 2,
        "code_type_local": 1,
        "type_local": "Maison",
        "surface_reelle_bati": 100.0,
        "nombre_pieces_principales": 4,
        "nature_culture": "",
        "nature_culture_speciale": "",
        "surface_terrain": 500.0,
    } | overrides


def write_input(
    root: Path, config: QualificationConfig, rows: list[dict[str, object]],
    *, schema: pa.Schema = INPUT_SCHEMA,
) -> Path:
    """Write small input with one row per row group to exercise streaming."""
    path = root / config.input_directory / "dvf_2025.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, row_group_size=1)
    return path


def numbered_rows(overrides: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row(**(change | {"source_row_number": number}))
            for number, change in enumerate(overrides, start=1)]


def assert_no_parts(root: Path) -> None:
    assert list(root.rglob("*.part")) == []


def test_config_uses_normalized_input_and_five_configured_years() -> None:
    settings = load_qualification_config(DEFAULT_CONFIG_PATH)
    assert settings.years == (2021, 2022, 2023, 2024, 2025)
    assert settings.input_directory == Path("data/interim/dvf/normalized")
    assert settings.output_directory == Path("data/processed/dvf/qualified")
    assert settings.batch_size > 0


@pytest.mark.parametrize(("code", "label"), [(1, "Maison"), (2, "Appartement")])
def test_simple_residence_preserves_authoritative_observation(
    tmp_path: Path, config: QualificationConfig, code: int, label: str,
) -> None:
    source = row(code_type_local=code, type_local=label)
    source_path = write_input(tmp_path, config, [source])
    before = source_path.read_bytes()
    result = qualify_dvf_year(2025, config, tmp_path)
    table = pq.read_table(result.path)
    observations = table.to_pylist()

    assert result.year == 2025
    assert result.path == tmp_path / config.output_directory / "dvf_2025.parquet"
    assert table.column_names == EXPECTED_OUTPUT_COLUMNS
    assert table.schema.equals(QUALIFIED_SCHEMA)
    assert len(observations) == 1
    expected = build_observation(pd.DataFrame([
        source | {"longitude": None, "latitude": None},
    ], dtype=object))
    assert expected is not None
    assert observations[0] == expected | {
        "source_year": 2025, "nom_commune": "COMMUNE FICTIVE",
        "nombre_lots": 2, "surface_terrain": 500.0,
    }
    assert observations[0]["prix_m2"] == 2500.0
    assert observations[0]["has_dependance"] is False
    assert result.report.mutations_seen == 1
    assert result.report.mutations_admissible == 1
    assert result.report.mutations_rejected == 0
    assert result.report.retention_rate == 1.0
    assert sum(result.report.rejection_counts.values()) == 0
    assert source_path.read_bytes() == before
    assert_no_parts(tmp_path)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 20])
def test_complete_mutations_cross_batches_and_final_mutation_is_qualified(
    tmp_path: Path, config: QualificationConfig, batch_size: int,
) -> None:
    config = replace(config, batch_size=batch_size)
    rows = numbered_rows([
        ANNEX | {"nombre_lots": 9, "surface_terrain": 9999.0,
                 "nom_commune": "ANNEXE FICTIVE"},
        {"nombre_lots": 2, "surface_terrain": 600.0},
        {"id_mutation": "2025-2"},
        {"id_mutation": "2025-2"},
        {"id_mutation": "2025-3", "code_type_local": 2, "type_local": "Appartement"},
    ])
    write_input(tmp_path, config, rows)
    result = qualify_dvf_year(2025, config, tmp_path)
    observations = pq.read_table(result.path).to_pylist()

    assert [item["id_mutation"] for item in observations] == ["2025-1", "2025-3"]
    assert observations[0]["has_dependance"] is True
    assert observations[0]["source_row_count"] == 2
    assert observations[0]["nombre_lots"] == 2
    assert observations[0]["surface_terrain"] == 600.0
    assert observations[0]["nom_commune"] == "COMMUNE FICTIVE"
    assert observations[0]["surface_reelle_bati"] == 100.0
    assert observations[1]["source_row_count"] == 1
    assert result.report.mutations_seen == 3
    assert result.report.mutations_admissible == 2
    assert result.report.mutations_rejected == 1
    assert result.report.retention_rate == pytest.approx(2 / 3)
    assert result.report.rejection_counts[ExclusionReason.RESIDENTIAL_ROW_COUNT] == 1


def test_canonical_analysis_receives_complete_mutations_once_without_dataframes(
    tmp_path: Path, config: QualificationConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_input(tmp_path, config, numbered_rows([
        {}, ANNEX, {"id_mutation": "2025-2"}, {"id_mutation": "2025-2"},
    ]))
    analyzed = []
    canonical = clean_module.analyze_mutation

    def spy_analyze(rows: list[clean_module.MutationRow]) -> clean_module.MutationAnalysis:
        analyzed.append(([source.id_mutation for source in rows], len(rows)))
        assert all(source.longitude is None and source.latitude is None for source in rows)
        return canonical(rows)

    def forbid_dataframe(*args: object, **kwargs: object) -> None:
        raise AssertionError("The streaming hot path must not build DataFrames")

    monkeypatch.setattr(clean_module, "analyze_mutation", spy_analyze)
    monkeypatch.setattr(pd, "DataFrame", forbid_dataframe)
    result = qualify_dvf_year(2025, replace(config, batch_size=1), tmp_path)
    assert result.report.mutations_seen == 2
    assert analyzed == [(["2025-1", "2025-1"], 2), (["2025-2", "2025-2"], 2)]
    assert result.report.mutations_admissible == 1
    assert result.report.mutations_rejected == 1


def test_sequential_ids_allow_surrounding_whitespace_and_row_number_gaps(
    tmp_path: Path, config: QualificationConfig,
) -> None:
    write_input(tmp_path, config, [
        row(id_mutation=" 2025-1 ", source_row_number=10),
        row(**(ANNEX | {"id_mutation": "2025-1", "source_row_number": 20})),
        row(id_mutation="2025-2", source_row_number=50),
    ])
    result = qualify_dvf_year(2025, config, tmp_path)
    observations = pq.read_table(result.path).to_pylist()
    assert [item["id_mutation"] for item in observations] == ["2025-1", "2025-2"]
    assert result.report.mutations_seen == 2


@pytest.mark.parametrize("batch_size", [1, 2, 20])
@pytest.mark.parametrize(
    "identifiers",
    [
        pytest.param(["2025-1"], id="first-id-is-one"),
        pytest.param(["2025-1", "2025-2"], id="consecutive-identifiers"),
        pytest.param(["2025-1", "2025-1"], id="same-id-on-several-rows"),
        pytest.param(["2025-1", "2025-1", "2025-2", "2025-2"],
                     id="contiguous-groups-across-batches"),
    ],
)
def test_sequential_identifier_integrity_accepts_normalized_groups(
    tmp_path: Path, config: QualificationConfig, batch_size: int,
    identifiers: list[str],
) -> None:
    rows = numbered_rows([
        {"id_mutation": identifier}
        | (ANNEX if index > 0 and identifier == identifiers[index - 1] else {})
        for index, identifier in enumerate(identifiers)
    ])
    write_input(tmp_path, config, rows)
    result = qualify_dvf_year(2025, replace(config, batch_size=batch_size), tmp_path)
    observations = pq.read_table(result.path).to_pylist()
    expected_ids = list(dict.fromkeys(identifiers))
    assert [item["id_mutation"] for item in observations] == expected_ids
    assert result.report.mutations_seen == len(expected_ids)
    assert result.report.mutations_admissible == len(expected_ids)
    assert result.report.mutations_rejected == 0
    assert sum(item["source_row_count"] for item in observations) == len(rows)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        pytest.param([{}, ANNEX | {"numero_disposition": "000002"}],
                     ExclusionReason.INVALID_DISPOSITION, id="multiple-dispositions"),
        pytest.param([{"numero_disposition": " "}],
                     ExclusionReason.INVALID_DISPOSITION, id="empty-disposition"),
        pytest.param([{}, ANNEX | {"numero_disposition": None}],
                     ExclusionReason.INVALID_DISPOSITION, id="missing-disposition-on-annex"),
        pytest.param([{}, ANNEX | {"id_parcelle": "010010000A0002"}],
                     ExclusionReason.INVALID_PARCEL, id="multiple-parcels"),
        pytest.param([{"id_parcelle": ""}],
                     ExclusionReason.INVALID_PARCEL, id="empty-parcel"),
        pytest.param([{}, ANNEX | {"id_parcelle": None}],
                     ExclusionReason.INVALID_PARCEL, id="missing-parcel-on-annex"),
        pytest.param([ANNEX], ExclusionReason.RESIDENTIAL_ROW_COUNT, id="no-residence"),
        pytest.param([{}, {}], ExclusionReason.RESIDENTIAL_ROW_COUNT,
                     id="identical-residential-rows-not-deduplicated"),
        pytest.param([{}, {"code_type_local": 4, "type_local": "Maison"}],
                     ExclusionReason.COMMERCIAL_OR_INDUSTRIAL_LOCAL, id="commercial-code"),
        pytest.param([{"valeur_fonciere": None}],
                     ExclusionReason.INVALID_VALUE, id="missing-value"),
        pytest.param([{}, ANNEX | {"valeur_fonciere": 250001.0}],
                     ExclusionReason.INCONSISTENT_VALUE, id="inconsistent-value"),
        pytest.param([{"valeur_fonciere": float("nan")}],
                     ExclusionReason.INVALID_VALUE, id="nan-value"),
        pytest.param([{"valeur_fonciere": float("inf")}],
                     ExclusionReason.INVALID_VALUE, id="infinite-value"),
        pytest.param([{"surface_reelle_bati": None}],
                     ExclusionReason.INVALID_RESIDENTIAL_SURFACE, id="missing-surface"),
        pytest.param([{"surface_reelle_bati": 0.0}],
                     ExclusionReason.INVALID_RESIDENTIAL_SURFACE, id="zero-surface"),
        pytest.param([{"surface_reelle_bati": -1.0}],
                     ExclusionReason.INVALID_RESIDENTIAL_SURFACE, id="negative-surface"),
        pytest.param([{"surface_reelle_bati": float("inf")}],
                     ExclusionReason.INVALID_RESIDENTIAL_SURFACE, id="infinite-surface"),
        pytest.param([{"valeur_fonciere": 1e308, "surface_reelle_bati": 1e-300}],
                     ExclusionReason.UNREPRESENTABLE_FLOAT, id="unrepresentable-ratio"),
        pytest.param([{"nature_mutation": "Echange"}],
                     ExclusionReason.NOT_A_SALE, id="not-a-sale"),
        pytest.param([{}, ANNEX | {"nature_mutation": "Echange"}],
                     ExclusionReason.NOT_A_SALE, id="mixed-natures"),
        pytest.param([{"code_type_local": None}],
                     ExclusionReason.INVALID_LOCAL_CODE, id="null-local-code"),
        pytest.param([{}, ANNEX | {"code_type_local": 5}],
                     ExclusionReason.INVALID_LOCAL_CODE, id="unknown-local-code-on-annex"),
        pytest.param([{"code_type_local": 0}],
                     ExclusionReason.INVALID_LOCAL_CODE, id="zero-local-code"),
        pytest.param([{"nature_mutation": "Echange", "numero_disposition": "",
                       "id_parcelle": "", "code_type_local": None}],
                     ExclusionReason.NOT_A_SALE, id="first-rejection-priority-preserved"),
        pytest.param([{"code_type_local": 4}],
                     ExclusionReason.RESIDENTIAL_ROW_COUNT,
                     id="residential-count-before-commercial-priority"),
    ],
)
def test_rejections_use_exact_clean_contract_and_never_produce_observations(
    tmp_path: Path, config: QualificationConfig,
    changes: list[dict[str, object]], reason: ExclusionReason,
) -> None:
    rows = numbered_rows(changes)
    oracle_frame = pd.DataFrame(
        [source | {"longitude": None, "latitude": None} for source in rows], dtype=object,
    )
    assert qualify_mutation(oracle_frame).exclusion_reason is reason
    write_input(tmp_path, config, rows)

    result = qualify_dvf_year(2025, config, tmp_path)

    assert pq.read_table(result.path).num_rows == 0
    assert pq.read_schema(result.path).equals(QUALIFIED_SCHEMA)
    assert result.report.mutations_seen == 1
    assert result.report.mutations_admissible == 0
    assert result.report.mutations_rejected == 1
    assert result.report.retention_rate == 0.0
    assert result.report.rejection_counts[reason] == 1
    assert sum(result.report.rejection_counts.values()) == 1
    assert {path.name for path in result.path.parent.iterdir()} == {"dvf_2025.parquet"}


@pytest.mark.parametrize("amount", [0.0, -1.0, 1.0, 6_000_000.0])
def test_no_economic_threshold_or_label_consistency_rule_is_added(
    tmp_path: Path, config: QualificationConfig, amount: float,
) -> None:
    write_input(tmp_path, config, [row(
        valeur_fonciere=amount, surface_reelle_bati=1.0,
        type_local="Dépendance", nombre_pieces_principales=None,
        nombre_lots=None, surface_terrain=None,
    )])
    result = qualify_dvf_year(2025, config, tmp_path)
    observation = pq.read_table(result.path).to_pylist()[0]
    assert observation["prix_m2"] == amount
    assert observation["type_local"] == "Dépendance"
    assert observation["code_type_local"] == 1
    assert observation["has_dependance"] is False
    assert observation["nombre_pieces_principales"] is None
    assert observation["nombre_lots"] is None
    assert observation["surface_terrain"] is None


def test_multiple_rejections_are_aggregated_once_per_mutation(
    tmp_path: Path, config: QualificationConfig,
) -> None:
    write_input(tmp_path, config, numbered_rows([
        {"nature_mutation": "Echange"},
        {"id_mutation": "2025-2", "surface_reelle_bati": 0.0},
        {"id_mutation": "2025-3", "surface_reelle_bati": None},
        {"id_mutation": "2025-4"},
    ]))
    report = qualify_dvf_year(2025, config, tmp_path).report
    assert report.mutations_seen == 4
    assert report.mutations_admissible == 1
    assert report.mutations_rejected == 3
    assert report.retention_rate == 0.25
    assert report.rejection_counts[ExclusionReason.NOT_A_SALE] == 1
    assert report.rejection_counts[ExclusionReason.INVALID_RESIDENTIAL_SURFACE] == 2
    assert sum(report.rejection_counts.values()) == 3
    payload = report.to_dict()
    assert payload["retention_rate"] == 0.25
    assert payload["rejection_counts"]["not_a_sale"] == 1


@pytest.mark.parametrize("batch_size", [1, 3, 50])
def test_mixed_output_and_all_report_counts_match_clean_exactly(
    tmp_path: Path, config: QualificationConfig, batch_size: int,
) -> None:
    """Compare orchestration to the unchanged contract, including output order."""
    groups = [
        [ANNEX | {"nom_commune": "ANNEXE FICTIVE"}, {}],
        [{"nature_mutation": "Echange", "id_parcelle": "", "code_type_local": None}],
        [{"code_type_local": 2, "type_local": "Appartement", "valeur_fonciere": 0.0,
          "nombre_pieces_principales": None, "nombre_lots": None,
          "surface_terrain": None}],
        [{}, {}],
        [{}, ANNEX | {"code_type_local": None}],
        [{"surface_reelle_bati": 0.0}],
        [{"valeur_fonciere": -1.0, "type_local": "Dépendance"}],
    ]
    rows = []
    expected_observations = []
    expected_rejections = dict.fromkeys(ExclusionReason, 0)
    for ordinal, group in enumerate(groups, start=1):
        mutation = [
            row(**(change | {
                "id_mutation": f"2025-{ordinal}",
                "source_row_number": len(rows) + offset,
            }))
            for offset, change in enumerate(group, start=1)
        ]
        rows.extend(mutation)
        frame = pd.DataFrame(
            [source | {"longitude": None, "latitude": None} for source in mutation],
            dtype=object,
        )
        decision = qualify_mutation(frame)
        if decision.admissible:
            observation = build_observation(frame)
            assert observation is not None
            residential = next(source for source in mutation
                               if source["code_type_local"] in (1, 2))
            expected_observations.append(observation | {
                name: residential[name]
                for name in ("source_year", "nom_commune", "nombre_lots", "surface_terrain")
            })
        else:
            expected_rejections[decision.exclusion_reason] += 1

    write_input(tmp_path, config, rows)
    result = qualify_dvf_year(2025, replace(config, batch_size=batch_size), tmp_path)

    table = pq.read_table(result.path)
    assert table.schema.equals(QUALIFIED_SCHEMA)
    assert table.to_pylist() == expected_observations
    assert result.report.mutations_seen == len(groups)
    assert result.report.mutations_admissible == len(expected_observations)
    assert result.report.mutations_rejected == sum(expected_rejections.values())
    assert result.report.retention_rate == len(expected_observations) / len(groups)
    assert result.report.rejection_counts == expected_rejections


def test_empty_input_creates_valid_empty_output_and_zero_counters(
    tmp_path: Path, config: QualificationConfig,
) -> None:
    write_input(tmp_path, config, [])
    result = qualify_dvf_year(2025, config, tmp_path)
    assert pq.read_table(result.path).num_rows == 0
    assert pq.read_schema(result.path).equals(QUALIFIED_SCHEMA)
    assert result.report.mutations_seen == 0
    assert result.report.mutations_admissible == 0
    assert result.report.mutations_rejected == 0
    assert result.report.retention_rate == 0.0
    assert sum(result.report.rejection_counts.values()) == 0


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param([{}, {"source_row_number": 1}], id="repeated-row-number"),
        pytest.param([{"source_row_number": 2}, {"source_row_number": 1}],
                     id="decreasing-row-number"),
        pytest.param([{"source_row_number": 0}], id="zero-row-number"),
        pytest.param([{"source_row_number": -1}], id="negative-row-number"),
        pytest.param([{"source_row_number": None}], id="missing-row-number"),
        pytest.param([{"source_year": 2024}], id="wrong-source-year"),
        pytest.param([{"source_year": None}], id="missing-source-year"),
        pytest.param([{"id_mutation": ""}], id="empty-mutation-id"),
        pytest.param([{"id_mutation": "  "}], id="blank-mutation-id"),
        pytest.param([{"id_mutation": None}], id="missing-mutation-id"),
        pytest.param([{"id_mutation": "2025-2"}], id="first-id-is-not-one"),
        pytest.param([{}, {"id_mutation": "2025-3"}], id="skipped-ordinal"),
        pytest.param([{"id_mutation": "2025-2"}, {"id_mutation": "2025-1"}],
                     id="decreasing-identifiers-with-invalid-first-id"),
        pytest.param([{}, {"id_mutation": "2025-2"}, {"id_mutation": "2025-0"}],
                     id="change-to-lower-ordinal"),
        pytest.param([{"id_mutation": "2024-1"}], id="wrong-year-in-first-id"),
        pytest.param([{}, {"id_mutation": "2024-2"}], id="wrong-year-in-later-id"),
        pytest.param([{"id_mutation": "mutation A"}], id="arbitrary-id"),
        pytest.param([{"id_mutation": "2025"}], id="missing-ordinal"),
        pytest.param([{"id_mutation": "2025-0"}], id="zero-ordinal"),
        pytest.param([{"id_mutation": "2025-01"}], id="leading-zero-ordinal"),
        pytest.param([{}, {"id_mutation": "2025-02"}], id="later-leading-zero-ordinal"),
        pytest.param([{"id_mutation": "2025--1"}], id="negative-ordinal"),
        pytest.param([{"id_mutation": "2025-+1"}], id="signed-ordinal"),
        pytest.param([{"id_mutation": "2025-1.0"}], id="decimal-ordinal"),
        pytest.param([{"id_mutation": "2025-one"}], id="nonnumeric-ordinal"),
        pytest.param([{"id_mutation": "2025- 1"}], id="inner-whitespace"),
        pytest.param([{"id_mutation": "2025-١"}], id="non-ascii-ordinal"),
        pytest.param([{}, {"id_mutation": "2025-2"}, {"id_mutation": "2025-1"}],
                     id="non-contiguous-id"),
        pytest.param([{}, {"id_mutation": "2025-2", "nature_mutation": "Echange"},
                      {"id_mutation": "2025-3"},
                      {"id_mutation": "2025-2", "nature_mutation": "Echange"}],
                     id="non-contiguous-rejected-id-also-fails"),
    ],
)
def test_input_integrity_errors_abort_and_clean_temporaries(
    tmp_path: Path, config: QualificationConfig, changes: list[dict[str, object]],
) -> None:
    config = replace(config, batch_size=1)
    rows = [row(**({"source_row_number": index} | change))
            for index, change in enumerate(changes, start=1)]
    write_input(tmp_path, config, rows)
    with pytest.raises(InputIntegrityError):
        qualify_dvf_year(2025, config, tmp_path)
    assert not (tmp_path / config.output_directory / "dvf_2025.parquet").exists()
    assert_no_parts(tmp_path)
    output_dir = tmp_path / config.output_directory
    assert not output_dir.exists() or list(output_dir.iterdir()) == []


def test_missing_input_and_unsupported_year_fail(
    tmp_path: Path, config: QualificationConfig,
) -> None:
    with pytest.raises((InputIntegrityError, FileNotFoundError)):
        qualify_dvf_year(2025, config, tmp_path)
    with pytest.raises(ValueError):
        qualify_dvf_year(2020, config, tmp_path)
    assert_no_parts(tmp_path)


def test_unreadable_parquet_raises_explicit_input_error(
    tmp_path: Path, config: QualificationConfig,
) -> None:
    source = tmp_path / config.input_directory / "dvf_2025.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"not a parquet file")
    with pytest.raises(InputIntegrityError):
        qualify_dvf_year(2025, config, tmp_path)
    assert_no_parts(tmp_path)


@pytest.mark.parametrize(
    "missing", ["id_mutation", "code_type_local", "source_year", "nom_commune"],
)
def test_missing_required_column_fails(
    tmp_path: Path, config: QualificationConfig, missing: str,
) -> None:
    schema = pa.schema([field for field in INPUT_SCHEMA if field.name != missing])
    write_input(tmp_path, config, [row()], schema=schema)
    with pytest.raises(InputIntegrityError):
        qualify_dvf_year(2025, config, tmp_path)
    assert_no_parts(tmp_path)


def test_duplicate_columns_fail(
    tmp_path: Path, config: QualificationConfig,
) -> None:
    table = pa.Table.from_pylist([row()], schema=INPUT_SCHEMA)
    table = table.append_column("id_mutation", table.column("id_mutation"))
    path = tmp_path / config.input_directory / "dvf_2025.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    with pytest.raises(InputIntegrityError):
        qualify_dvf_year(2025, config, tmp_path)
    assert_no_parts(tmp_path)


def test_existing_output_refused_and_force_replaces_it(
    tmp_path: Path, config: QualificationConfig,
) -> None:
    write_input(tmp_path, config, [row()])
    first = qualify_dvf_year(2025, config, tmp_path)
    before = first.path.read_bytes()
    write_input(tmp_path, config, [row(valeur_fonciere=300000.0)])
    with pytest.raises(FileExistsError):
        qualify_dvf_year(2025, config, tmp_path)
    assert first.path.read_bytes() == before

    second = qualify_dvf_year(2025, config, tmp_path, force=True)
    assert second.path == first.path
    assert pq.read_table(second.path).to_pylist()[0]["prix_m2"] == 3000.0
    assert_no_parts(tmp_path)


def test_failed_forced_run_preserves_previous_output_and_cleans_part(
    tmp_path: Path, config: QualificationConfig,
) -> None:
    config = replace(config, batch_size=1)
    write_input(tmp_path, config, [row()])
    destination = qualify_dvf_year(2025, config, tmp_path).path
    before = destination.read_bytes()
    write_input(tmp_path, config, numbered_rows([
        {"valeur_fonciere": 300000.0},
        {"id_mutation": "2025-2"},
        {"id_mutation": "2025-3", "source_year": 2024},
    ]))
    with pytest.raises(InputIntegrityError):
        qualify_dvf_year(2025, config, tmp_path, force=True)
    assert destination.read_bytes() == before
    assert_no_parts(tmp_path)
    assert list(destination.parent.iterdir()) == [destination]


def test_failed_atomic_replace_preserves_previous_output_and_cleans_part(
    tmp_path: Path, config: QualificationConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_input(tmp_path, config, [row()])
    destination = qualify_dvf_year(2025, config, tmp_path).path
    before = destination.read_bytes()
    write_input(tmp_path, config, [row(valeur_fonciere=300000.0)])

    def fail_replace(source: Path, target: Path) -> None:
        assert source.suffix == ".part"
        assert target == destination
        assert pq.read_table(source).num_rows == 1
        raise OSError("Synthetic atomic replacement failure")

    monkeypatch.setattr(qualify_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="Synthetic atomic replacement failure"):
        qualify_dvf_year(2025, config, tmp_path, force=True)
    assert destination.read_bytes() == before
    assert_no_parts(tmp_path)
    assert list(destination.parent.iterdir()) == [destination]


def test_pipeline_does_not_use_full_file_parquet_readers(
    tmp_path: Path, config: QualificationConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_input(tmp_path, config, numbered_rows([
        {"id_mutation": f"2025-{number}"} for number in range(1, 7)
    ]))

    def forbid_full_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("The qualification pipeline must stream its Parquet input")

    monkeypatch.setattr(pq, "read_table", forbid_full_read)
    monkeypatch.setattr(pd, "read_parquet", forbid_full_read)
    result = qualify_dvf_year(2025, config, tmp_path)
    assert result.report.mutations_seen == 6
    with pq.ParquetFile(result.path) as output:
        assert output.metadata.num_rows == 6


def test_cli_delegates_explicit_year_and_force_without_real_processing(
    tmp_path: Path, config: QualificationConfig,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    write_input(tmp_path, config, [row()])
    result = qualify_dvf_year(2025, config, tmp_path)
    calls = []

    def fake_qualify(
        year: int, settings: QualificationConfig, root: Path, *, force: bool = False,
    ) -> object:
        calls.append((year, settings, root, force))
        return result

    monkeypatch.setattr(qualify_module, "load_qualification_config", lambda: config)
    monkeypatch.setattr(qualify_module, "qualify_dvf_year", fake_qualify)
    assert qualify_module.main(["--year", "2025", "--force"]) == 0
    assert len(calls) == 1
    assert calls[0][0] == 2025
    assert calls[0][1] is config
    assert calls[0][3] is True
    output = capsys.readouterr().out
    assert "mutations_seen" in output
    assert "rejection_counts" in output
