"""Execute the existing DVF V1 contract on complete, streamed mutations.

Business decisions and observation construction belong exclusively to clean.py.
This module validates input ordering, adapts missing coordinates, writes admitted
observations and reports aggregate rejections. Importing it does not process data.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import repeat
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from real_estate.data import clean
from real_estate.data.normalize import DEFAULT_CONFIG_PATH, load_normalization_config
from real_estate.data.validate import (
    REQUIRED_COLUMNS,
    ExclusionReason,
    parse_finite_number,
)

REQUIRED_INPUT_COLUMNS = tuple(
    column for column in REQUIRED_COLUMNS if column not in {"longitude", "latitude"}
) + ("source_year", "source_row_number", "nom_commune", "nombre_lots", "surface_terrain")

# Fixed tuple positions avoid allocating a dictionary for every source row.
_MUTATION_COLUMNS = (*REQUIRED_INPUT_COLUMNS, "longitude", "latitude")
_YEAR_INDEX = _MUTATION_COLUMNS.index("source_year")
_ROW_NUMBER_INDEX = _MUTATION_COLUMNS.index("source_row_number")
_ID_INDEX = _MUTATION_COLUMNS.index("id_mutation")
_LOCAL_CODE_INDEX = _MUTATION_COLUMNS.index("code_type_local")
_RESIDENTIAL_METADATA = tuple(
    (column, _MUTATION_COLUMNS.index(column))
    for column in ("source_year", "nom_commune", "nombre_lots", "surface_terrain")
)

QUALIFIED_SCHEMA = pa.schema([
    pa.field("source_year", pa.int32(), nullable=False),
    pa.field("id_mutation", pa.string(), nullable=False),
    pa.field("date_mutation", pa.string()),
    pa.field("numero_disposition", pa.string(), nullable=False),
    pa.field("id_parcelle", pa.string(), nullable=False),
    pa.field("code_postal", pa.string()),
    pa.field("nom_commune", pa.string()),
    pa.field("code_departement", pa.string()),
    pa.field("code_commune", pa.string()),
    pa.field("code_type_local", pa.int64(), nullable=False),
    pa.field("type_local", pa.string()),
    pa.field("valeur_fonciere", pa.float64(), nullable=False),
    pa.field("surface_reelle_bati", pa.float64(), nullable=False),
    pa.field("nombre_pieces_principales", pa.int64()),
    pa.field("nombre_lots", pa.int64()),
    pa.field("surface_terrain", pa.float64()),
    pa.field("has_dependance", pa.bool_(), nullable=False),
    pa.field("prix_m2", pa.float64(), nullable=False),
    pa.field("source_row_count", pa.int64(), nullable=False),
    pa.field("longitude", pa.float64()),
    pa.field("latitude", pa.float64()),
])


class QualificationError(RuntimeError):
    """The execution cannot publish a complete qualified Parquet."""


class InputIntegrityError(QualificationError):
    """The normalized input violates structural or ordering requirements."""


@dataclass(frozen=True)
class QualificationConfig:
    """Reuse configured years/input and bound input/output batch sizes."""

    years: tuple[int, ...]
    input_directory: Path
    output_directory: Path
    batch_size: int

    def __post_init__(self) -> None:
        if (
            not self.years or len(set(self.years)) != len(self.years)
            or any(type(year) is not int or not 1000 <= year <= 9999 for year in self.years)
        ):
            raise ValueError("Qualification years must be distinct four-digit integers.")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("Qualification batch_size must be a positive integer.")
        for directory in (self.input_directory, self.output_directory):
            if directory.is_absolute() or ".." in directory.parts:
                raise ValueError("Qualification directories must be project-relative.")


@dataclass
class QualificationReport:
    """Aggregate the first rejection reason returned by the existing contract."""

    mutations_seen: int = 0
    mutations_admissible: int = 0
    mutations_rejected: int = 0
    rejection_counts: dict[ExclusionReason, int] = field(
        default_factory=lambda: dict.fromkeys(ExclusionReason, 0),
    )

    @property
    def retention_rate(self) -> float:
        """Return admitted / seen, or zero when the input has no mutations."""
        return self.mutations_admissible / self.mutations_seen if self.mutations_seen else 0.0

    def record(self, decision: clean.MutationQualification) -> None:
        """Record exactly one authoritative decision for a complete mutation."""
        self.mutations_seen += 1
        if decision.admissible:
            self.mutations_admissible += 1
        else:
            self.mutations_rejected += 1
            self.rejection_counts[decision.exclusion_reason] += 1

    def to_dict(self) -> dict[str, object]:
        """Expose stable enum values for the CLI's aggregate JSON report."""
        return {
            "mutations_seen": self.mutations_seen,
            "mutations_admissible": self.mutations_admissible,
            "mutations_rejected": self.mutations_rejected,
            "retention_rate": self.retention_rate,
            "rejection_counts": {
                reason.value: count for reason, count in self.rejection_counts.items()
            },
        }


@dataclass(frozen=True)
class QualificationResult:
    """Published file and execution counters; no rejected-row dataset is created."""

    year: int
    path: Path
    report: QualificationReport


def load_qualification_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> QualificationConfig:
    """Read qualification settings while reusing normalization years and output."""
    normalized = load_normalization_config(config_path)
    try:
        section = yaml.safe_load(config_path.read_text(encoding="utf-8"))["dvf"]["qualification"]
        return QualificationConfig(
            years=normalized.years,
            input_directory=normalized.output_directory,
            output_directory=Path(section["output_directory"]),
            batch_size=section["batch_size"],
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError("Invalid DVF qualification configuration.") from error


def validate_normalized_schema(schema: pa.Schema) -> None:
    """Require business inputs and provenance columns without adding business rules."""
    if len(schema.names) != len(set(schema.names)):
        raise InputIntegrityError("Duplicate normalized input columns.")
    missing = set(REQUIRED_INPUT_COLUMNS) - set(schema.names)
    if missing:
        raise InputIntegrityError(f"Missing normalized input columns: {', '.join(sorted(missing))}.")


def _open_normalized(path: Path) -> pq.ParquetFile:
    if not path.is_file():
        raise InputIntegrityError("Normalized DVF input file is missing.")
    try:
        return pq.ParquetFile(path)
    except (OSError, pa.ArrowException) as error:
        raise InputIntegrityError("Cannot read the normalized DVF Parquet.") from error


def _validate_row_identity(
    row: tuple[object, ...], year: int, previous_number: int,
) -> tuple[int, str]:
    """Validate each row before grouping, including rows later rejected by V1."""
    if type(row[_YEAR_INDEX]) is not int or row[_YEAR_INDEX] != year:
        raise InputIntegrityError("Inconsistent source_year in normalized input.")
    number = row[_ROW_NUMBER_INDEX]
    if type(number) is not int or number <= previous_number:
        raise InputIntegrityError("source_row_number must be positive and strictly increasing.")
    mutation_id = row[_ID_INDEX]
    if not isinstance(mutation_id, str) or not mutation_id.strip():
        raise InputIntegrityError("id_mutation must be a nonempty string.")
    return number, mutation_id.strip()


def _complete_mutations(
    parquet: pq.ParquetFile, year: int, batch_size: int,
) -> Iterator[list[tuple[object, ...]]]:
    """Keep complete groups across batches; validate normalize.py's YYYY-N sequence."""
    current_id: str | None = None
    current_rows: list[tuple[object, ...]] = []
    mutation_number = 0
    previous_number = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=list(REQUIRED_INPUT_COLUMNS), use_threads=False,
    ):
        columns = batch.to_pydict()
        # Explicit None coordinates preserve the adapter's previous null semantics.
        for row in zip(
            *(columns[column] for column in REQUIRED_INPUT_COLUMNS),
            repeat(None), repeat(None),
        ):
            previous_number, mutation_id = _validate_row_identity(row, year, previous_number)
            if row[_ID_INDEX] != mutation_id:
                row = (*row[:_ID_INDEX], mutation_id, *row[_ID_INDEX + 1:])
            if mutation_id != current_id:
                expected_id = f"{year}-{mutation_number + 1}"
                if mutation_id != expected_id:
                    raise InputIntegrityError(
                        f"Invalid id_mutation sequence: expected {expected_id}. "
                        "IDs must start at YYYY-1 and advance by one per contiguous group."
                    )
                mutation_number += 1
                if current_rows:
                    yield current_rows
                current_rows = []
                current_id = mutation_id
            current_rows.append(row)
    if current_rows:
        yield current_rows


def _qualify_complete_mutation(
    rows: list[tuple[object, ...]], report: QualificationReport,
) -> dict[str, object] | None:
    """Adapt missing coordinates and delegate every business decision to clean.py."""
    mutation = pd.DataFrame(rows, columns=_MUTATION_COLUMNS, dtype=object)
    decision = clean.qualify_mutation(mutation)
    report.record(decision)
    if not decision.admissible:
        return None
    observation = clean.build_observation(mutation)
    if observation is None:
        raise QualificationError("The V1 observation builder contradicted its admission.")
    residential = next(
        row for row in rows
        if parse_finite_number(row[_LOCAL_CODE_INDEX]) in clean.RESIDENTIAL_CODES
    )
    for column, index in _RESIDENTIAL_METADATA:
        observation[column] = residential[index]
    return observation


def _write_observations(
    parquet: pq.ParquetFile, writer: pq.ParquetWriter,
    year: int, batch_size: int, report: QualificationReport,
) -> None:
    """Buffer only a bounded batch of admitted observations for each Parquet write."""
    observations: list[dict[str, object]] = []
    for mutation in _complete_mutations(parquet, year, batch_size):
        observation = _qualify_complete_mutation(mutation, report)
        del mutation
        if observation is not None:
            observations.append(observation)
        if len(observations) == batch_size:
            writer.write_table(
                pa.Table.from_pylist(observations, schema=QUALIFIED_SCHEMA),
                row_group_size=batch_size,
            )
            observations.clear()
    if observations:
        writer.write_table(
            pa.Table.from_pylist(observations, schema=QUALIFIED_SCHEMA),
            row_group_size=batch_size,
        )


def _validate_output(path: Path, report: QualificationReport) -> None:
    """Validate footer, schema, row count and accounting before publication."""
    with pq.ParquetFile(path) as parquet:
        if not parquet.schema_arrow.equals(QUALIFIED_SCHEMA):
            raise QualificationError("Qualified output schema mismatch.")
        if parquet.metadata.num_rows != report.mutations_admissible:
            raise QualificationError("Qualified output row count mismatch.")
    if (
        report.mutations_seen != report.mutations_admissible + report.mutations_rejected
        or report.mutations_rejected != sum(report.rejection_counts.values())
    ):
        raise QualificationError("Qualification report counters are inconsistent.")


def qualify_dvf_year(
    year: int, config: QualificationConfig, project_root: Path, *, force: bool = False,
) -> QualificationResult:
    """Stream one configured local year and publish only after complete validation."""
    if year not in config.years:
        raise ValueError(f"DVF year {year} is not configured.")
    source = project_root / config.input_directory / f"dvf_{year}.parquet"
    destination = project_root / config.output_directory / f"dvf_{year}.parquet"
    if source.resolve() == destination.resolve():
        raise ValueError("Qualification input and output paths must differ.")
    if destination.exists() and not force:
        raise FileExistsError("Qualified DVF output exists; use --force to replace it.")
    report = QualificationReport()
    temporary_path: Path | None = None
    try:
        with _open_normalized(source) as parquet:
            validate_normalized_schema(parquet.schema_arrow)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=f".{destination.name}.", suffix=".part", delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            with pq.ParquetWriter(temporary_path, QUALIFIED_SCHEMA) as writer:
                _write_observations(parquet, writer, year, config.batch_size, report)
        _validate_output(temporary_path, report)
        if destination.exists() and not force:
            raise FileExistsError("Qualified DVF output appeared during processing.")
        os.replace(temporary_path, destination)
    except pa.ArrowException as error:
        raise QualificationError("Parquet qualification execution failed.") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return QualificationResult(year, destination, report)


def main(argv: list[str] | None = None) -> int:
    """Qualify a selected local year and print aggregate counters as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="Replace an existing qualified file.")
    arguments = parser.parse_args(argv)
    try:
        config = load_qualification_config()
        result = qualify_dvf_year(
            arguments.year, config, DEFAULT_CONFIG_PATH.parents[1], force=arguments.force,
        )
    except (QualificationError, ValueError, OSError) as error:
        parser.exit(1, f"DVF qualification failed: {error}\n")
    print(json.dumps(
        {"year": result.year, "output": str(result.path), **result.report.to_dict()},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
