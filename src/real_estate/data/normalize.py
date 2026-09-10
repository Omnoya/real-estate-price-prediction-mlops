"""Stream DGFiP source rows into a row-preserving, normalized Parquet file.

id_mutation is synthetic, snapshot-local and non-stable across DVF releases.
It is intended only to group contiguous source rows.
No acquisition, business qualification or transformation runs at import time.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import tempfile
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from real_estate.data.download import DEFAULT_CONFIG_PATH, load_data_config, sha256_file

SOURCE_COLUMNS = (
    "Identifiant de document", "Reference document",
    "1 Articles CGI", "2 Articles CGI", "3 Articles CGI", "4 Articles CGI", "5 Articles CGI",
    "No disposition", "Date mutation", "Nature mutation", "Valeur fonciere",
    "No voie", "B/T/Q", "Type de voie", "Code voie", "Voie",
    "Code postal", "Commune", "Code departement", "Code commune",
    "Prefixe de section", "Section", "No plan", "No Volume",
    "1er lot", "Surface Carrez du 1er lot", "2eme lot", "Surface Carrez du 2eme lot",
    "3eme lot", "Surface Carrez du 3eme lot", "4eme lot", "Surface Carrez du 4eme lot",
    "5eme lot", "Surface Carrez du 5eme lot", "Nombre de lots",
    "Code type local", "Type local", "Identifiant local", "Surface reelle bati",
    "Nombre pieces principales", "Nature culture", "Nature culture speciale", "Surface terrain",
)

NORMALIZED_SCHEMA = pa.schema([
    pa.field("source_year", pa.int32(), nullable=False),
    pa.field("source_row_number", pa.int64(), nullable=False),
    pa.field("id_mutation", pa.string(), nullable=False),
    pa.field("date_mutation", pa.string(), nullable=False),
    pa.field("numero_disposition", pa.string(), nullable=False),
    pa.field("nature_mutation", pa.string(), nullable=False),
    pa.field("valeur_fonciere", pa.float64()),
    pa.field("code_postal", pa.string(), nullable=False),
    pa.field("nom_commune", pa.string(), nullable=False),
    pa.field("code_departement", pa.string(), nullable=False),
    pa.field("code_commune", pa.string(), nullable=False),
    pa.field("prefixe_section", pa.string(), nullable=False),
    pa.field("section", pa.string(), nullable=False),
    pa.field("numero_plan", pa.string(), nullable=False),
    pa.field("id_parcelle", pa.string(), nullable=False),
    pa.field("nombre_lots", pa.int64()),
    pa.field("code_type_local", pa.int64()),
    pa.field("type_local", pa.string(), nullable=False),
    pa.field("surface_reelle_bati", pa.float64()),
    pa.field("nombre_pieces_principales", pa.int64()),
    pa.field("nature_culture", pa.string(), nullable=False),
    pa.field("nature_culture_speciale", pa.string(), nullable=False),
    pa.field("surface_terrain", pa.float64()),
])

_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:[.,][0-9]*)?|[.,][0-9]+)(?:[eE][+-]?[0-9]+)?")


class NormalizationError(RuntimeError):
    """A source cannot be normalized safely; no partial output is published."""


class RawProvenanceError(NormalizationError):
    """The local archive cannot be matched to its acquisition manifest."""


class SourceSchemaError(NormalizationError):
    """The ZIP, header or record structure differs from the source contract."""


class NormalizationValueError(NormalizationError):
    """A value cannot be represented without an invalid date, number or parcel."""


@dataclass(frozen=True)
class NormalizationConfig:
    """Project-relative inputs and bounded batch size, without network settings."""

    years: tuple[int, ...]
    raw_directory: Path
    manifest_filename: str
    output_directory: Path
    batch_size: int

    def __post_init__(self) -> None:
        if (
            not self.years or len(set(self.years)) != len(self.years)
            or any(type(year) is not int or not 1000 <= year <= 9999 for year in self.years)
        ):
            raise ValueError("Normalization years must be distinct four-digit integers.")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("Normalization batch_size must be a positive integer.")
        for path in (self.raw_directory, self.output_directory):
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Normalization directories must be project-relative.")
        if (
            not self.manifest_filename or self.manifest_filename in {".", ".."}
            or "/" in self.manifest_filename or "\\" in self.manifest_filename
        ):
            raise ValueError("Normalization manifest_filename must be a filename.")


@dataclass(frozen=True)
class NormalizationResult:
    """Published output identity and counters from a complete source traversal."""

    year: int
    path: Path
    row_count: int
    mutation_count: int
    raw_sha256: str


@dataclass
class _MutationState:
    """Retain only the preceding exact key, including across batch boundaries."""

    year: int
    counter: int = 0
    previous_key: tuple[str, Decimal | None] | None = None

    def identify(self, date: str, amount: Decimal | None) -> str:
        key = (date, amount)
        if key != self.previous_key:
            self.counter += 1
            self.previous_key = key
        return f"{self.year}-{self.counter}"


def load_normalization_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> NormalizationConfig:
    """Reuse acquisition paths/years and read only normalization-specific settings."""
    acquisition = load_data_config(config_path)
    try:
        section = yaml.safe_load(config_path.read_text(encoding="utf-8"))["dvf"]["normalization"]
        return NormalizationConfig(
            years=acquisition.years,
            raw_directory=acquisition.raw_directory,
            manifest_filename=acquisition.manifest_filename,
            output_directory=Path(section["output_directory"]),
            batch_size=section["batch_size"],
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError("Invalid DVF normalization configuration.") from error


def verify_raw_archive(
    year: int, config: NormalizationConfig, project_root: Path,
) -> tuple[Path, str]:
    """Check year, filename and actual SHA-256 before opening the source as a ZIP."""
    if year not in config.years:
        raise ValueError(f"DVF year {year} is not configured.")
    archive_path = project_root / config.raw_directory / f"dvf_{year}.zip"
    manifest_path = project_root / config.raw_directory / config.manifest_filename
    if not archive_path.is_file():
        raise RawProvenanceError("DVF ZIP is missing.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest["downloads"][str(year)]
        expected_hash = entry["sha256"]
        if entry["year"] != year or entry["filename"] != archive_path.name:
            raise RawProvenanceError("DVF manifest year or filename mismatch.")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise RawProvenanceError("DVF manifest SHA-256 is invalid.")
        actual_hash = sha256_file(archive_path)
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as error:
        raise RawProvenanceError("DVF manifest is missing, invalid or lacks the requested year.") from error
    if actual_hash != expected_hash:
        raise RawProvenanceError("DVF ZIP SHA-256 does not match its manifest.")
    return archive_path, actual_hash


def validate_source_schema(columns: Sequence[str]) -> None:
    """Require the exact 43 source names in order, without duplicate columns."""
    if len(set(columns)) != len(columns):
        raise SourceSchemaError("Duplicate DVF source columns.")
    if tuple(columns) != SOURCE_COLUMNS:
        raise SourceSchemaError("DVF source schema drift: expected the exact 43 columns in order.")


def _source_rows(archive_path: Path) -> Iterator[dict[str, str]]:
    """Read one UTF-8 pipe-delimited text member, validating every record width."""
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) != 1 or not members[0].filename.lower().endswith(".txt"):
                raise SourceSchemaError("DVF ZIP must contain exactly one text file.")
            with (
                archive.open(members[0]) as binary,
                io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as stream,
            ):
                reader = csv.reader(stream, delimiter="|", strict=True)
                validate_source_schema(next(reader, []))
                for row_number, row in enumerate(reader, start=1):
                    if len(row) != len(SOURCE_COLUMNS):
                        raise SourceSchemaError(f"Invalid field count at source row {row_number}.")
                    yield dict(zip(SOURCE_COLUMNS, (value.strip() for value in row)))
    except (OSError, zipfile.BadZipFile, UnicodeError, csv.Error) as error:
        raise SourceSchemaError("Invalid DVF ZIP or source text structure.") from error


def _pad_component(value: str, width: int, field: str, *, letters: bool = False) -> str:
    pattern = r"[A-Z0-9]+" if letters else r"[0-9]+"
    if not value or len(value) > width or re.fullmatch(pattern, value) is None:
        raise NormalizationValueError(f"Invalid parcel component: {field}.")
    return value.zfill(width)


def normalize_parcel(
    department: str, commune: str, prefix: str, section: str, plan: str,
) -> dict[str, str]:
    """Build a 14-character parcel identifier without truncating source components."""
    department, commune, prefix, section, plan = (
        value.strip() for value in (department, commune, prefix, section, plan)
    )
    if re.fullmatch(r"(?:[0-9]{1,2}|2A|2B|97[0-9])", department) is None:
        raise NormalizationValueError("Invalid parcel component: Code departement.")
    if department.startswith("97"):
        if len(department) != 3:
            raise NormalizationValueError("Invalid overseas department width.")
        full_commune = department + _pad_component(commune, 2, "Code commune")
    else:
        department = department.zfill(2)
        full_commune = department + _pad_component(commune, 3, "Code commune")
    prefix = _pad_component(prefix or "000", 3, "Prefixe de section")
    section = _pad_component(section, 2, "Section", letters=True)
    plan = _pad_component(plan, 4, "No plan")
    parcel = full_commune + prefix + section + plan
    if len(parcel) != 14:
        raise NormalizationValueError("Invalid constructed parcel length.")
    return {
        "code_departement": department, "code_commune": full_commune,
        "prefixe_section": prefix, "section": section, "numero_plan": plan,
        "id_parcelle": parcel,
    }


def _number(text: str, field: str, row_number: int) -> Decimal | None:
    """Parse exact finite decimals; only an empty string denotes a missing number."""
    if not text:
        return None
    if _NUMBER.fullmatch(text) is not None:
        try:
            number = Decimal(text.replace(",", "."))
            if number.is_finite():
                return number
        except InvalidOperation:
            pass
    raise NormalizationValueError(f"Invalid numeric field {field} at source row {row_number}.")


def _float(number: Decimal | None, field: str, row_number: int) -> float | None:
    if number is None:
        return None
    value = float(number)
    if not math.isfinite(value) or (value == 0 and number != 0):
        raise NormalizationValueError(
            f"Unrepresentable float field {field} at source row {row_number}."
        )
    return value


def _integer(text: str, field: str, row_number: int) -> int | None:
    number = _number(text, field, row_number)
    if number is None:
        return None
    if number != number.to_integral_value() or not -(2**63) <= number < 2**63:
        raise NormalizationValueError(f"Invalid int64 field {field} at source row {row_number}.")
    return int(number)


def _iso_date(text: str, row_number: int) -> str:
    if re.fullmatch(r"[0-9]{2}/[0-9]{2}/[0-9]{4}", text):
        try:
            day, month, year = map(int, text.split("/"))
            return date(year, month, day).isoformat()
        except ValueError:
            pass
    raise NormalizationValueError(f"Invalid date at source row {row_number}.")


def _normalize_row(
    source: Mapping[str, str], row_number: int, state: _MutationState,
) -> dict[str, object]:
    """Rename and convert one row without applying any business selection."""
    date = _iso_date(source["Date mutation"], row_number)
    amount = _number(source["Valeur fonciere"], "Valeur fonciere", row_number)
    mutation_id = state.identify(date, amount)
    result: dict[str, object] = {
        "source_year": state.year, "source_row_number": row_number,
        "id_mutation": mutation_id, "date_mutation": date,
        "numero_disposition": source["No disposition"],
        "nature_mutation": source["Nature mutation"],
        "valeur_fonciere": _float(amount, "Valeur fonciere", row_number),
        "code_postal": source["Code postal"], "nom_commune": source["Commune"],
        "type_local": source["Type local"], "nature_culture": source["Nature culture"],
        "nature_culture_speciale": source["Nature culture speciale"],
    }
    result.update(normalize_parcel(
        source["Code departement"], source["Code commune"], source["Prefixe de section"],
        source["Section"], source["No plan"],
    ))
    for output, field in (
        ("nombre_lots", "Nombre de lots"), ("code_type_local", "Code type local"),
        ("nombre_pieces_principales", "Nombre pieces principales"),
    ):
        result[output] = _integer(source[field], field, row_number)
    for output, field in (
        ("surface_reelle_bati", "Surface reelle bati"), ("surface_terrain", "Surface terrain"),
    ):
        result[output] = _float(_number(source[field], field, row_number), field, row_number)
    return result


def _normalized_batches(
    archive_path: Path, state: _MutationState, batch_size: int,
) -> Iterator[pa.Table]:
    """Bound memory to one batch while sharing mutation state for the entire file."""
    batch: list[dict[str, object]] = []
    for row_number, source in enumerate(_source_rows(archive_path), start=1):
        batch.append(_normalize_row(source, row_number, state))
        if len(batch) == batch_size:
            yield pa.Table.from_pylist(batch, schema=NORMALIZED_SCHEMA)
            batch.clear()
    if batch:
        yield pa.Table.from_pylist(batch, schema=NORMALIZED_SCHEMA)


def _validate_parquet(path: Path, expected_rows: int) -> None:
    """Check the completed footer, explicit schema and total row count."""
    with pq.ParquetFile(path) as parquet:
        if parquet.metadata.num_rows != expected_rows:
            raise NormalizationError("Normalized Parquet row count mismatch.")
        if not parquet.schema_arrow.equals(NORMALIZED_SCHEMA):
            raise NormalizationError("Normalized Parquet schema mismatch.")


def normalize_dvf_year(
    year: int, config: NormalizationConfig, project_root: Path, *, force: bool = False,
) -> NormalizationResult:
    """Verify local provenance, stream every row and atomically publish one year."""
    archive_path, raw_hash = verify_raw_archive(year, config, project_root)
    destination = project_root / config.output_directory / f"dvf_{year}.parquet"
    if destination.exists() and not force:
        raise FileExistsError("Normalized DVF output exists; use --force to replace it.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    state = _MutationState(year)
    row_count = 0
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".part", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        with pq.ParquetWriter(temporary_path, NORMALIZED_SCHEMA) as writer:
            for batch in _normalized_batches(archive_path, state, config.batch_size):
                writer.write_table(batch, row_group_size=config.batch_size)
                row_count += batch.num_rows
        _validate_parquet(temporary_path, row_count)
        if destination.exists() and not force:
            raise FileExistsError("Normalized DVF output appeared during processing.")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return NormalizationResult(year, destination, row_count, state.counter, raw_hash)


def main(argv: list[str] | None = None) -> int:
    """Normalize one explicitly selected configured year; never acquire raw data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="Replace an existing normalized file.")
    arguments = parser.parse_args(argv)
    try:
        config = load_normalization_config()
        result = normalize_dvf_year(
            arguments.year, config, DEFAULT_CONFIG_PATH.parents[1], force=arguments.force,
        )
    except (NormalizationError, ValueError, OSError) as error:
        parser.exit(1, f"DVF normalization failed: {error}\n")
    print(f"{result.year}: {result.row_count} rows, {result.mutation_count} contiguous groups.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
