"""Build an annual COG geography reference from a verified local current CSV.

Acquisition belongs to communes.py. This module never reads DVF, historical
states or movement journals, and performs no acquisition or processing at import.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from real_estate.data.communes import CogDataConfig, get_cog_source, load_cog_config
from real_estate.data.download import DEFAULT_CONFIG_PATH, sha256_file

COG_SOURCE_COLUMNS = (
    "TYPECOM", "COM", "REG", "DEP", "CTCD", "ARR", "TNCC", "NCC",
    "NCCENR", "LIBELLE", "CAN", "COMPARENT",
)
GEO_PRIORITY = ("COM", "ARM", "COMD", "COMA")
GEOGRAPHY_SCHEMA = pa.schema([
    pa.field("source_code_commune", pa.string(), nullable=False),
    pa.field("resolved_geo_type", pa.string(), nullable=False),
    pa.field("source_geo_label", pa.string(), nullable=False),
    pa.field("parent_commune_code", pa.string()),
    pa.field("canonical_commune_code", pa.string(), nullable=False),
    pa.field("canonical_commune_label", pa.string(), nullable=False),
    pa.field("region_code", pa.string(), nullable=False),
    pa.field("department_code", pa.string(), nullable=False),
    pa.field("cog_year", pa.int32(), nullable=False),
])

CogRow = dict[str, str]
CogIndex = dict[str, dict[str, CogRow]]


class GeographyError(RuntimeError):
    """The annual reference cannot be built or published safely."""


class GeographyProvenanceError(GeographyError):
    """The raw archive does not match its existing acquisition manifest."""


class GeographySchemaError(GeographyError):
    """The configured member, CSV header or record structure is invalid."""


class GeographyIntegrityError(GeographyError):
    """COG codes, hierarchy or output invariants violate the contract."""


@dataclass(frozen=True)
class GeographyConfig:
    """Reuse validated acquisition settings and select current members explicitly."""

    cog: CogDataConfig
    current_commune_files: dict[int, str]
    output_directory: Path

    def __post_init__(self) -> None:
        if (
            set(self.current_commune_files) != set(self.cog.years)
            or any(type(year) is not int for year in self.current_commune_files)
        ):
            raise ValueError("COG current_commune_files must match configured years.")
        for member in self.current_commune_files.values():
            if (
                not isinstance(member, str) or not member.endswith(".csv")
                or "/" in member or "\\" in member
            ):
                raise ValueError("Each current COG member must be an explicit CSV filename.")
        if (
            self.output_directory.is_absolute()
            or ".." in self.output_directory.parts
            or self.output_directory == Path(".")
        ):
            raise ValueError("Geography output_directory must be project-relative.")


@dataclass(frozen=True)
class GeographyResult:
    """Published location and deterministic counts after applying COM priority."""

    year: int
    path: Path
    source_codes: int
    resolved_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return the CLI summary without exposing individual reference rows."""
        return {
            "year": self.year,
            "output": str(self.path),
            "source_codes": self.source_codes,
            **{f"resolved_{kind}": self.resolved_counts[kind] for kind in GEO_PRIORITY},
        }


def load_geography_config(config_path: Path = DEFAULT_CONFIG_PATH) -> GeographyConfig:
    """Read acquisition settings and the explicit current-CSV/output mapping."""
    cog = load_cog_config(config_path)
    try:
        section = yaml.safe_load(config_path.read_text(encoding="utf-8"))["cog"]["geography"]
        return GeographyConfig(
            cog=cog,
            current_commune_files=dict(section["current_commune_files"]),
            output_directory=Path(section["output_directory"]),
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError("Invalid COG geography configuration.") from error


def _unique_manifest_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate manifest key.")
        result[key] = value
    return result


def verify_cog_archive(year: int, config: GeographyConfig, project_root: Path) -> Path:
    """Verify year, filename, source_url, bytes and SHA-256 before opening the ZIP.

    final_url remains historical acquisition metadata. It is not compared to the
    configured source URL. Neither the manifest nor the archive is updated here.
    """
    source_url = get_cog_source(config.cog, year)
    archive = project_root / config.cog.raw_directory / f"cog_{year}.zip"
    manifest_path = archive.parent / config.cog.manifest_filename
    if not archive.is_file():
        raise GeographyProvenanceError("COG archive is missing.")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_manifest_keys,
        )
        entry = manifest["downloads"][str(year)]
        if type(entry["year"]) is not int or entry["year"] != year:
            raise ValueError("manifest year mismatch")
        if entry["filename"] != archive.name:
            raise ValueError("manifest filename mismatch")
        if entry["source_url"] != source_url:
            raise ValueError("manifest source_url provenance mismatch")
        size = entry["bytes"]
        if type(size) is not int or size <= 0 or size != archive.stat().st_size:
            raise ValueError("manifest bytes mismatch")
        digest = entry["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid manifest sha256")
        if sha256_file(archive) != digest:
            raise ValueError("archive sha256 mismatch")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise GeographyProvenanceError(f"COG archive/manifest integrity failed: {error}") from error
    return archive


def validate_source_schema(columns: Sequence[str]) -> None:
    """Require exactly the 12 audited names, each once; header order may vary."""
    if len(columns) != len(set(columns)):
        raise GeographySchemaError("Duplicate COG source column.")
    missing = set(COG_SOURCE_COLUMNS) - set(columns)
    extra = set(columns) - set(COG_SOURCE_COLUMNS)
    if missing or extra:
        raise GeographySchemaError(
            f"Invalid COG source columns: missing={sorted(missing)}, extra={sorted(extra)}."
        )


def read_current_communes(archive_path: Path, member: str) -> Iterator[CogRow]:
    """Stream only the explicitly named UTF-8 comma-separated member in the ZIP."""
    try:
        with zipfile.ZipFile(archive_path) as archive:
            matches = [info for info in archive.infolist() if info.filename == member]
            if len(matches) != 1 or matches[0].is_dir():
                raise GeographySchemaError("Configured current COG CSV member is missing or duplicated.")
            with (
                archive.open(matches[0]) as binary,
                io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as stream,
            ):
                reader = csv.reader(stream, delimiter=",", strict=True)
                header = next(reader, None)
                if header is None:
                    raise GeographySchemaError("Current COG CSV header is missing.")
                validate_source_schema(header)
                for row_number, values in enumerate(reader, start=1):
                    if len(values) != len(header):
                        raise GeographySchemaError(f"Invalid COG record width at row {row_number}.")
                    yield dict(zip(header, values))
    except (OSError, zipfile.BadZipFile, UnicodeError, csv.Error, RuntimeError) as error:
        if isinstance(error, GeographyError):
            raise
        raise GeographySchemaError("Cannot read the configured current COG CSV.") from error


def _require_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise GeographyIntegrityError(f"{field} must be a nonempty string.")


def _require_code(value: object, field: str) -> None:
    _require_text(value, field)
    if len(value) != 5 or any(char.isspace() for char in value):
        raise GeographyIntegrityError(f"{field} must contain exactly 5 characters without whitespace.")


def _validate_source_row(row: CogRow) -> None:
    if set(row) != set(COG_SOURCE_COLUMNS) or any(not isinstance(v, str) for v in row.values()):
        raise GeographySchemaError("A COG record must contain the 12 string fields.")
    _require_code(row["COM"], "COM")
    if row["TYPECOM"] not in GEO_PRIORITY:
        raise GeographyIntegrityError("Unknown COG TYPECOM.")
    _require_text(row["LIBELLE"], "LIBELLE")
    if row["TYPECOM"] == "COM":
        _require_text(row["REG"], "REG of canonical COM")
        _require_text(row["DEP"], "DEP of canonical COM")
    else:
        _require_code(row["COMPARENT"], "COMPARENT")


def _canonical_parent(row: CogRow, index: CogIndex) -> CogRow:
    """Find a direct COM parent; never follow chains or use a historical mapping."""
    parents = index.get(row["COMPARENT"])
    if parents is None:
        raise GeographyIntegrityError("COMPARENT parent is absent from the current COG.")
    if "COM" not in parents:
        raise GeographyIntegrityError("COMPARENT must point to exactly one TYPECOM=COM parent.")
    return parents["COM"]


def build_cog_index(rows: Iterable[CogRow]) -> CogIndex:
    """Validate every row and parent before selection, including a masked COMD.

    Keep one annual COG in memory. Duplicate (TYPECOM, COM) pairs are errors;
    distinct TYPECOM rows sharing a code remain available to the resolver.
    """
    index: CogIndex = {}
    for row in rows:
        _validate_source_row(row)
        by_type = index.setdefault(row["COM"], {})
        if row["TYPECOM"] in by_type:
            raise GeographyIntegrityError("Duplicate (TYPECOM, COM): ambiguous COG structure.")
        by_type[row["TYPECOM"]] = dict(row)
    if not index:
        raise GeographyIntegrityError("Current COG must contain at least one row.")
    for by_type in index.values():
        for row in by_type.values():
            if row["TYPECOM"] != "COM":
                _canonical_parent(row, index)
    return index


def resolve_geography(code: str, index: CogIndex, year: int) -> dict[str, object]:
    """Resolve one code in a validated index, using COM then ARM, COMD and COMA."""
    candidates = index.get(code)
    if not candidates:
        raise GeographyIntegrityError("Source code is absent from the current COG index.")
    row = next(candidates[kind] for kind in GEO_PRIORITY if kind in candidates)
    is_commune = row["TYPECOM"] == "COM"
    canonical = row if is_commune else _canonical_parent(row, index)
    return {
        "source_code_commune": code,
        "resolved_geo_type": row["TYPECOM"],
        "source_geo_label": row["LIBELLE"],
        "parent_commune_code": None if is_commune else row["COMPARENT"],
        "canonical_commune_code": canonical["COM"],
        "canonical_commune_label": canonical["LIBELLE"],
        "region_code": canonical["REG"],
        "department_code": canonical["DEP"],
        "cog_year": year,
    }


def _validate_geography_row(row: dict[str, object], year: int) -> None:
    _require_code(row["source_code_commune"], "source_code_commune")
    _require_code(row["canonical_commune_code"], "canonical_commune_code")
    kind = row["resolved_geo_type"]
    if kind not in GEO_PRIORITY:
        raise GeographyIntegrityError("Invalid resolved_geo_type in geography output.")
    for field in ("source_geo_label", "canonical_commune_label", "region_code", "department_code"):
        _require_text(row[field], field)
    if type(row["cog_year"]) is not int or row["cog_year"] != year:
        raise GeographyIntegrityError("Geography cog_year must equal the requested year.")
    parent = row["parent_commune_code"]
    if kind == "COM":
        if parent is not None or row["canonical_commune_code"] != row["source_code_commune"]:
            raise GeographyIntegrityError("COM must have null parent and its own canonical code.")
    else:
        _require_code(parent, "parent_commune_code")
        if parent != row["canonical_commune_code"]:
            raise GeographyIntegrityError("Non-COM parent must equal canonical_commune_code.")


def validate_geography_parquet(path: Path, year: int, expected_rows: int) -> None:
    """Read back the staged Parquet and check every contract invariant."""
    seen: set[str] = set()
    with pq.ParquetFile(path) as parquet:
        if not parquet.schema_arrow.equals(GEOGRAPHY_SCHEMA):
            raise GeographyIntegrityError("Geography Parquet schema mismatch.")
        if parquet.metadata.num_rows <= 0 or parquet.metadata.num_rows != expected_rows:
            raise GeographyIntegrityError("Geography Parquet row count must be positive and match input codes.")
        for batch in parquet.iter_batches(batch_size=10000):
            for row in batch.to_pylist():
                _validate_geography_row(row, year)
                code = row["source_code_commune"]
                if code in seen:
                    raise GeographyIntegrityError("Duplicate source_code_commune in geography output.")
                seen.add(code)
    if len(seen) != expected_rows:
        raise GeographyIntegrityError("Geography Parquet decoded row count mismatch.")


def write_geography_parquet(
    table: pa.Table, destination: Path, year: int, *, force: bool = False,
) -> None:
    """Stage, validate, fsync and atomically replace a reference; preserve failures."""
    if destination.exists() and not force:
        raise FileExistsError("COG geography output exists; use --force to replace it.")
    temporary_path: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".part", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        pq.write_table(table, temporary_path, row_group_size=10000)
        validate_geography_parquet(temporary_path, year, table.num_rows)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        if destination.exists() and not force:
            raise FileExistsError("COG geography output appeared during processing.")
        os.replace(temporary_path, destination)
    except pa.ArrowException as error:
        raise GeographyError("Cannot write or validate geography Parquet.") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_geography_year(
    year: int, config: GeographyConfig, project_root: Path, *, force: bool = False,
) -> GeographyResult:
    """Build only one configured annual reference, with deterministic code order."""
    get_cog_source(config.cog, year)
    destination = project_root / config.output_directory / f"cog_geography_{year}.parquet"
    if destination.exists() and not force:
        raise FileExistsError("COG geography output exists; use --force to replace it.")
    archive = verify_cog_archive(year, config, project_root)
    manifest = archive.parent / config.cog.manifest_filename
    if destination.resolve() in {archive.resolve(), manifest.resolve()}:
        raise GeographyError("Geography output must not replace the raw archive or manifest.")
    index = build_cog_index(read_current_communes(archive, config.current_commune_files[year]))
    rows = [resolve_geography(code, index, year) for code in sorted(index)]
    counts = Counter(row["resolved_geo_type"] for row in rows)
    table = pa.Table.from_pylist(rows, schema=GEOGRAPHY_SCHEMA)
    write_geography_parquet(table, destination, year, force=force)
    return GeographyResult(year, destination, len(rows), {kind: counts[kind] for kind in GEO_PRIORITY})


def main(argv: list[str] | None = None) -> int:
    """Explicitly build one local annual reference and print aggregate JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="Replace an existing COG reference.")
    arguments = parser.parse_args(argv)
    try:
        config = load_geography_config(DEFAULT_CONFIG_PATH)
        result = build_geography_year(
            arguments.year, config, DEFAULT_CONFIG_PATH.parents[1], force=arguments.force,
        )
    except (GeographyError, ValueError, OSError) as error:
        parser.exit(1, f"COG geography failed: {error}\n")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
