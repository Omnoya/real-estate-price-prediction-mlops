"""Left-join qualified DVF to its annual COG reference without dropping rows.

Only geography is loaded in full. DVF and output validation use bounded Arrow
batches. No acquisition, hierarchy resolution or processing runs at import time.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from real_estate.data.compare import _arrays_equal
from real_estate.data.download import DEFAULT_CONFIG_PATH
from real_estate.data.geography import (
    GEO_PRIORITY,
    GEOGRAPHY_SCHEMA,
    GeographyError,
    validate_geography_parquet,
)
from real_estate.data.qualify import QUALIFIED_SCHEMA

GEOGRAPHIC_COLUMNS = tuple(GEOGRAPHY_SCHEMA.names)
ADDED_GEOGRAPHY_SCHEMA = pa.schema([
    column.with_nullable(column.name not in {"source_code_commune", "cog_year"})
    for column in GEOGRAPHY_SCHEMA
])


class EnrichmentError(RuntimeError):
    """Enrichment cannot safely publish a complete output."""


class EnrichmentIntegrityError(EnrichmentError):
    """An input, join or staged output violates the enrichment contract."""


def _validate_template(template: str) -> None:
    if not isinstance(template, str):
        raise TypeError("Enrichment paths must be string templates.")
    fields = [(name, spec, conversion) for _, name, spec, conversion
              in Formatter().parse(template) if name is not None]
    if fields != [("year", "", None)]:
        raise ValueError("Each enrichment path must contain exactly one plain {year}.")
    path = Path(template.format(year=2025))
    if path.is_absolute() or ".." in path.parts or path.suffix != ".parquet":
        raise ValueError("Enrichment paths must be project-relative Parquet paths.")


@dataclass(frozen=True)
class EnrichmentConfig:
    """Reuse annual configuration with explicit paths and a bounded DVF batch size."""

    years: tuple[int, ...]
    qualified_input: str
    geography_input: str
    output: str
    batch_size: int

    def __post_init__(self) -> None:
        if (
            not self.years
            or any(type(year) is not int or not 1000 <= year <= 9999 for year in self.years)
            or len(set(self.years)) != len(self.years)
        ):
            raise ValueError("Enrichment years must be distinct four-digit integers.")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("Enrichment batch_size must be a positive integer.")
        for template in (self.qualified_input, self.geography_input, self.output):
            _validate_template(template)


@dataclass(frozen=True)
class EnrichmentPaths:
    """Selected inputs, final output and the exclusive sibling .part."""

    qualified: Path
    geography: Path
    output: Path
    part: Path


@dataclass(frozen=True)
class GeographyIndex:
    """Validated annual reference and constant-time source-code lookups."""

    year: int
    table: pa.Table
    positions: dict[str, int]


@dataclass
class EnrichmentReport:
    """Count retained rows and their geographic status without economic filtering."""

    rows: int = 0
    resolved: int = 0
    unresolved: int = 0
    resolved_counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(GEO_PRIORITY, 0))

    def record(self, batch: pa.RecordBatch) -> None:
        """Count at most five statuses per batch using Arrow aggregation."""
        self.rows += batch.num_rows
        for status in pc.value_counts(batch.column("resolved_geo_type")).to_pylist():
            kind, count = status["values"], status["counts"]
            if kind is None:
                self.unresolved += count
            else:
                self.resolved_counts[kind] += count
                self.resolved += count

    def to_dict(self) -> dict[str, int]:
        return {
            "rows": self.rows, "resolved": self.resolved, "unresolved": self.unresolved,
            **{f"resolved_{kind}": self.resolved_counts[kind] for kind in GEO_PRIORITY},
        }


@dataclass(frozen=True)
class EnrichmentResult:
    """Published output and deterministic execution counters."""

    year: int
    path: Path
    report: EnrichmentReport

    def to_dict(self) -> dict[str, object]:
        return {"year": self.year, "output": str(self.path), **self.report.to_dict()}


def load_enrichment_config(config_path: Path = DEFAULT_CONFIG_PATH) -> EnrichmentConfig:
    """Read join settings and require a configured COG for every configured DVF year."""
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        years = tuple(data["dvf"]["years"])
        cog_years = tuple(data["cog"]["years"])
        if any(type(year) is not int for year in cog_years) or not set(years) <= set(cog_years):
            raise ValueError("Each DVF year needs the same configured COG year.")
        section = data["dvf"]["geography_enrichment"]
        return EnrichmentConfig(
            years=years, qualified_input=section["qualified_input"],
            geography_input=section["geography_input"], output=section["output"],
            batch_size=section["batch_size"],
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"Invalid geography enrichment configuration: {error}") from error


def validate_paths(year: int, config: EnrichmentConfig, project_root: Path) -> EnrichmentPaths:
    """Reject unconfigured years, escaping paths and aliases that could overwrite sources."""
    if type(year) is not int or year not in config.years:
        raise ValueError(f"Enrichment year {year} is not configured.")
    root = project_root.resolve()
    qualified, geography, output = (
        root / template.format(year=year)
        for template in (config.qualified_input, config.geography_input, config.output)
    )
    paths = EnrichmentPaths(qualified, geography, output, output.with_suffix(".parquet.part"))
    locations = [path.resolve() for path in (paths.qualified, paths.geography, paths.output, paths.part)]
    if any(not path.is_relative_to(root) for path in locations):
        raise ValueError("Enrichment paths must stay within the project root.")
    if len(set(locations)) != len(locations):
        raise ValueError("Enrichment inputs, output and .part must use distinct paths.")
    return paths


def _open_parquet(path: Path, label: str) -> pq.ParquetFile:
    if not path.is_file():
        raise EnrichmentIntegrityError(f"{label} Parquet is missing.")
    try:
        return pq.ParquetFile(path)
    except (OSError, pa.ArrowException) as error:
        raise EnrichmentIntegrityError(f"Cannot read {label} Parquet.") from error


def load_geography_index(path: Path, year: int) -> GeographyIndex:
    """Reuse the geography validator and check canonical links; never resolve a hierarchy."""
    try:
        with _open_parquet(path, "Geography") as parquet:
            expected_rows = parquet.metadata.num_rows
        validate_geography_parquet(path, year, expected_rows)
        with _open_parquet(path, "Geography") as parquet:
            table = parquet.read().combine_chunks()
        rows = table.to_pylist()
        positions = {row["source_code_commune"]: i for i, row in enumerate(rows)}
        for row in rows:
            position = positions.get(row["canonical_commune_code"])
            if position is None or rows[position]["resolved_geo_type"] != "COM":
                raise EnrichmentIntegrityError("Geography canonical code must identify a unique COM.")
            canonical = rows[position]
            if (
                row["canonical_commune_label"] != canonical["source_geo_label"]
                or row["region_code"] != canonical["region_code"]
                or row["department_code"] != canonical["department_code"]
            ):
                raise EnrichmentIntegrityError("Geography attributes disagree with the canonical COM.")
        return GeographyIndex(year, table, positions)
    except (GeographyError, OSError, pa.ArrowException) as error:
        raise EnrichmentIntegrityError(f"Invalid geography reference: {error}") from error


def validate_dvf_schema(schema: pa.Schema) -> None:
    """Require the 21 qualified columns and their types, preserving source order/nullability."""
    if len(schema) != len(QUALIFIED_SCHEMA) or set(schema.names) != set(QUALIFIED_SCHEMA.names):
        raise EnrichmentIntegrityError("Qualified DVF must contain exactly the 21 expected columns.")
    for column in schema:
        if column.type != QUALIFIED_SCHEMA.field(column.name).type:
            raise EnrichmentIntegrityError(f"Unexpected qualified DVF type for {column.name}.")


def enriched_schema(source_schema: pa.Schema) -> pa.Schema:
    """Append nullable geographic attributes without changing any original field."""
    validate_dvf_schema(source_schema)
    return pa.schema([*source_schema, *ADDED_GEOGRAPHY_SCHEMA], metadata=source_schema.metadata)


def _require_all(condition: pa.Array, message: str) -> None:
    if pc.all(pc.fill_null(condition, False)).as_py() is not True:
        raise EnrichmentIntegrityError(message)


def _nonempty(values: pa.Array) -> pa.Array:
    return pc.greater(pc.utf8_length(pc.utf8_trim_whitespace(values)), 0)


def validate_dvf_batch(batch: pa.RecordBatch, year: int) -> None:
    """Check only join identity and year, leaving qualified business decisions unchanged."""
    validate_dvf_schema(batch.schema)
    if not batch.num_rows:
        raise EnrichmentIntegrityError("Empty qualified DVF batch.")
    _require_all(pc.equal(batch.column("source_year"), year), "DVF source_year differs from requested year.")
    codes = batch.column("code_commune")
    _require_all(
        pc.and_(pc.equal(pc.utf8_length(codes), 5), pc.invert(pc.match_substring_regex(codes, r"\s"))),
        "DVF code_commune must be nonnull and have 5 characters without whitespace.",
    )
    _require_all(_nonempty(batch.column("id_mutation")), "DVF id_mutation must be nonnull and nonempty.")


def enrich_batch(batch: pa.RecordBatch, index: GeographyIndex, year: int) -> pa.RecordBatch:
    """One lookup per DVF row; Arrow take preserves cardinality and source order."""
    if index.year != year:
        raise EnrichmentIntegrityError("Geography index year differs from requested year.")
    validate_dvf_batch(batch, year)
    positions = pa.array(
        [index.positions.get(code) for code in batch.column("code_commune").to_pylist()],
        type=pa.int32(),
    )
    matches = index.table.take(positions)
    added = [batch.column("code_commune")]
    added.extend(matches.column(name).combine_chunks() for name in GEOGRAPHIC_COLUMNS[1:-1])
    added.append(batch.column("source_year"))
    return pa.RecordBatch.from_arrays([*batch.columns, *added], schema=enriched_schema(batch.schema))


def validate_enriched_batch(source: pa.RecordBatch, enriched: pa.RecordBatch, year: int) -> None:
    """Check unchanged ordered source values and every resolved/unresolved invariant."""
    validate_dvf_batch(source, year)
    if enriched.num_rows != source.num_rows or not enriched.schema.equals(
        enriched_schema(source.schema), check_metadata=True,
    ):
        raise EnrichmentIntegrityError("Enrichment row count or 30-column schema changed.")
    # Reuse the existing exact logical comparison, including nulls and NaNs.
    if not all(_arrays_equal(column, enriched.column(i)) for i, column in enumerate(source.columns)):
        raise EnrichmentIntegrityError("Qualified DVF columns or row order changed during enrichment.")
    _require_all(
        pc.equal(enriched.column("source_code_commune"), source.column("code_commune")),
        "Enriched source_code_commune differs from DVF code_commune.",
    )
    _require_all(
        pc.equal(enriched.column("cog_year"), source.column("source_year")),
        "Enriched cog_year differs from DVF source_year.",
    )
    kinds = enriched.column("resolved_geo_type")
    resolved = pc.is_valid(kinds)
    _require_all(
        pc.if_else(resolved, pc.is_in(kinds, value_set=pa.array(GEO_PRIORITY)), True),
        "Unknown enriched resolved_geo_type.",
    )
    for name in GEOGRAPHIC_COLUMNS[1:-1]:
        _require_all(
            pc.if_else(resolved, True, pc.is_null(enriched.column(name))),
            f"Unresolved {name} must be null.",
        )
    for name in ("source_geo_label", "canonical_commune_label", "region_code", "department_code"):
        _require_all(pc.if_else(resolved, _nonempty(enriched.column(name)), True), f"Resolved {name} is empty.")
    canonical = enriched.column("canonical_commune_code")
    _require_all(
        pc.if_else(resolved, pc.equal(pc.utf8_length(canonical), 5), True),
        "Resolved canonical_commune_code must be nonnull and have length 5.",
    )
    _require_all(
        pc.if_else(resolved, pc.equal(enriched.column("department_code"), source.column("code_departement")), True),
        "Resolved department_code differs from DVF code_departement.",
    )
    is_com = pc.fill_null(pc.equal(kinds, "COM"), False)
    parent = enriched.column("parent_commune_code")
    valid_parent = pc.if_else(
        is_com,
        pc.and_(pc.is_null(parent), pc.equal(canonical, enriched.column("source_code_commune"))),
        pc.equal(parent, canonical),
    )
    _require_all(pc.if_else(resolved, valid_parent, True), "Invalid enriched parent/canonical relationship.")


def _aligned_batches(
    source: pq.ParquetFile, output: pq.ParquetFile, batch_size: int,
) -> Iterator[tuple[pa.RecordBatch, pa.RecordBatch]]:
    """Align ordered slices even when the two files have different row groups."""
    left_batches = (b for b in source.iter_batches(batch_size=batch_size) if b.num_rows)
    right_batches = (b for b in output.iter_batches(batch_size=batch_size) if b.num_rows)
    left, right = next(left_batches, None), next(right_batches, None)
    left_offset = right_offset = 0
    while left is not None and right is not None:
        size = min(left.num_rows - left_offset, right.num_rows - right_offset)
        yield left.slice(left_offset, size), right.slice(right_offset, size)
        left_offset += size
        right_offset += size
        if left_offset == left.num_rows:
            left, left_offset = next(left_batches, None), 0
        if right_offset == right.num_rows:
            right, right_offset = next(right_batches, None), 0
    if left is not None or right is not None:
        raise EnrichmentIntegrityError("Source and enriched row streams differ in length.")


def validate_enriched_parquet(
    source_path: Path, part_path: Path, year: int, report: EnrichmentReport, *, batch_size: int,
) -> None:
    """Read back both files in bounded batches before publication, checking ordered values."""
    observed = EnrichmentReport()
    with _open_parquet(source_path, "Qualified DVF") as source, _open_parquet(part_path, "Enriched DVF") as output:
        if source.metadata.num_rows <= 0 or output.metadata.num_rows != source.metadata.num_rows:
            raise EnrichmentIntegrityError("Enriched row count must equal nonempty qualified input.")
        if not output.schema_arrow.equals(enriched_schema(source.schema_arrow), check_metadata=True):
            raise EnrichmentIntegrityError("Enriched Parquet schema mismatch.")
        for original, enriched in _aligned_batches(source, output, batch_size):
            validate_enriched_batch(original, enriched, year)
            observed.record(enriched)
        if observed.rows != source.metadata.num_rows:
            raise EnrichmentIntegrityError("Decoded row count differs from Parquet metadata.")
    if observed.to_dict() != report.to_dict():
        raise EnrichmentIntegrityError("Enrichment counters differ from the published rows.")


def enrich_dvf_geography_year(
    year: int, config: EnrichmentConfig, project_root: Path, *, force: bool = False,
) -> EnrichmentResult:
    """Enrich one configured year, retaining every input row and atomically publishing."""
    paths = validate_paths(year, config, project_root)
    if paths.output.exists() and not force:
        raise FileExistsError("Enriched DVF output exists; use --force to replace it.")
    report = EnrichmentReport()
    owns_part = False
    try:
        index = load_geography_index(paths.geography, year)
        with _open_parquet(paths.qualified, "Qualified DVF") as source:
            output_schema = enriched_schema(source.schema_arrow)
            if source.metadata.num_rows <= 0:
                raise EnrichmentIntegrityError("Qualified DVF input must not be empty.")
            paths.output.parent.mkdir(parents=True, exist_ok=True)
            with paths.part.open("xb") as temporary:
                owns_part = True
                with pq.ParquetWriter(temporary, output_schema) as writer:
                    for batch in source.iter_batches(batch_size=config.batch_size):
                        enriched = enrich_batch(batch, index, year)
                        validate_enriched_batch(batch, enriched, year)
                        writer.write_batch(enriched, row_group_size=config.batch_size)
                        report.record(enriched)
        validate_enriched_parquet(
            paths.qualified, paths.part, year, report, batch_size=config.batch_size,
        )
        with paths.part.open("rb") as stream:
            os.fsync(stream.fileno())
        if paths.output.exists() and not force:
            raise FileExistsError("Enriched DVF output appeared during processing.")
        os.replace(paths.part, paths.output)
    except pa.ArrowException as error:
        raise EnrichmentIntegrityError("Geographic enrichment Parquet processing failed.") from error
    finally:
        if owns_part:
            paths.part.unlink(missing_ok=True)
    return EnrichmentResult(year, paths.output, report)


def main(argv: list[str] | None = None) -> int:
    """Enrich only an explicitly selected year and print deterministic aggregate JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="Replace an existing enriched Parquet.")
    arguments = parser.parse_args(argv)
    try:
        config = load_enrichment_config(DEFAULT_CONFIG_PATH)
        result = enrich_dvf_geography_year(
            arguments.year, config, DEFAULT_CONFIG_PATH.parents[1], force=arguments.force,
        )
    except (EnrichmentError, ValueError, OSError) as error:
        parser.exit(1, f"DVF geography enrichment failed: {error}\n")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
