"""Qualify complete DVF mutations and build one observation per admission.

Callers must provide every source row of a mutation, including dependencies,
commercial premises and repeated residential rows. Completeness cannot be
inferred from an already filtered frame or an isolated input chunk.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import NamedTuple

import pandas as pd

from real_estate.data.validate import (
    REQUIRED_COLUMNS,
    ExclusionReason,
    parse_finite_number,
    validate_input_schema,
)

RESIDENTIAL_CODES = (1, 2)
LOCAL_CODES = (1, 2, 3, 4)
OBSERVATION_SOURCE_COLUMNS = (
    "id_mutation",
    "numero_disposition",
    "date_mutation",
    "valeur_fonciere",
    "id_parcelle",
    "code_commune",
    "code_departement",
    "code_postal",
    "code_type_local",
    "type_local",
    "nombre_pieces_principales",
    "surface_reelle_bati",
    "longitude",
    "latitude",
)


@dataclass(frozen=True)
class MutationQualification:
    """An immutable decision whose admission follows from its exclusion reason."""

    exclusion_reason: ExclusionReason | None = None

    @property
    def admissible(self) -> bool:
        """Whether every V1 condition passed."""
        return self.exclusion_reason is None


class MutationRow(NamedTuple):
    """Business inputs in REQUIRED_COLUMNS order, without scalar coercion."""

    id_mutation: object
    numero_disposition: object
    date_mutation: object
    nature_mutation: object
    valeur_fonciere: object
    id_parcelle: object
    code_commune: object
    code_departement: object
    code_postal: object
    code_type_local: object
    type_local: object
    nombre_pieces_principales: object
    surface_reelle_bati: object
    longitude: object
    latitude: object


@dataclass(frozen=True)
class MutationAnalysis:
    """One decision and its observation, with the source residential row position."""

    decision: MutationQualification
    observation: dict[str, object] | None = None
    residential_index: int | None = None


def _single_identifier(rows: Sequence[MutationRow], column: str) -> str | None:
    """Check completeness and equality, stripping strings without losing zeroes."""
    identifier = None
    for row in rows:
        value = getattr(row, column)
        if not isinstance(value, str):
            if pd.api.types.is_scalar(value) and pd.isna(value):
                return None
            value = str(value)
        value = value.strip()
        if not value or (identifier is not None and value != identifier):
            return None
        identifier = value
    return identifier


def _value_exclusion(rows: Sequence[MutationRow]) -> ExclusionReason | None:
    """Parse each amount once; any invalid value takes priority over differences."""
    first = None
    inconsistent = False
    for row in rows:
        amount = parse_finite_number(row.valeur_fonciere)
        if amount is None:
            return ExclusionReason.INVALID_VALUE
        if first is None:
            first = amount
        elif amount != first:
            inconsistent = True
    if inconsistent:
        return ExclusionReason.INCONSISTENT_VALUE
    return None


def _float_observation_values(residential: MutationRow) -> dict[str, float] | None:
    """Check float representability after the Decimal business checks.

    A positive surface must remain positive after conversion. The ratio is
    checked using the converted operands, so it matches the final dataset.
    """
    try:
        amount = float(residential.valeur_fonciere)
        surface = float(residential.surface_reelle_bati)
    except (ValueError, TypeError, OverflowError):
        return None
    if not isfinite(amount) or not isfinite(surface) or surface <= 0:
        return None
    price = amount / surface
    if not isfinite(price):
        return None
    return {
        "valeur_fonciere": amount,
        "surface_reelle_bati": surface,
        "prix_m2": price,
    }


def _rejected(reason: ExclusionReason) -> MutationAnalysis:
    return MutationAnalysis(MutationQualification(reason))


def analyze_mutation(rows: Sequence[MutationRow]) -> MutationAnalysis:
    """Apply the single canonical V1 evaluator to all rows of one mutation.

    Validate group identity even for early business rejections. Preserve the first
    exclusion reason and construct an observation only after every check passes.
    Callers supply complete groups; no row is deduplicated. The residential index
    lets orchestration copy auxiliary metadata without repeating the analysis.
    """
    if any(not isinstance(row, MutationRow) for row in rows):
        raise TypeError("Expected a sequence of MutationRow values.")
    mutation_id = _single_identifier(rows, "id_mutation")
    if mutation_id is None:
        raise ValueError("Expected one nonempty mutation with one complete id_mutation.")
    if any(not isinstance(row.nature_mutation, str) or row.nature_mutation != "Vente"
           for row in rows):
        return _rejected(ExclusionReason.NOT_A_SALE)
    disposition = _single_identifier(rows, "numero_disposition")
    if disposition is None:
        return _rejected(ExclusionReason.INVALID_DISPOSITION)
    parcel = _single_identifier(rows, "id_parcelle")
    if parcel is None:
        return _rejected(ExclusionReason.INVALID_PARCEL)

    codes = [parse_finite_number(row.code_type_local) for row in rows]
    if any(code not in LOCAL_CODES for code in codes):
        return _rejected(ExclusionReason.INVALID_LOCAL_CODE)
    residential_positions = [i for i, code in enumerate(codes) if code in RESIDENTIAL_CODES]
    if len(residential_positions) != 1:
        return _rejected(ExclusionReason.RESIDENTIAL_ROW_COUNT)
    if 4 in codes:
        return _rejected(ExclusionReason.COMMERCIAL_OR_INDUSTRIAL_LOCAL)
    value_exclusion = _value_exclusion(rows)
    if value_exclusion is not None:
        return _rejected(value_exclusion)

    residential_index = residential_positions[0]
    residential = rows[residential_index]
    surface = parse_finite_number(residential.surface_reelle_bati)
    if surface is None or surface <= 0:
        return _rejected(ExclusionReason.INVALID_RESIDENTIAL_SURFACE)
    numeric_values = _float_observation_values(residential)
    if numeric_values is None:
        return _rejected(ExclusionReason.UNREPRESENTABLE_FLOAT)

    observation = {column: getattr(residential, column) for column in OBSERVATION_SOURCE_COLUMNS}
    observation.update(numeric_values)
    observation.update(
        id_mutation=mutation_id,
        numero_disposition=disposition,
        id_parcelle=parcel,
        code_type_local=int(codes[residential_index]),
        has_dependance=3 in codes,
        source_row_count=len(rows),
    )
    return MutationAnalysis(MutationQualification(), observation, residential_index)


def _dataframe_rows(mutation: pd.DataFrame) -> list[MutationRow]:
    """Adapt the public DataFrame API, retaining its dtype-aware identifier strings.

    Pandas string conversion matters for datetime and object identifier columns.
    Missing identifiers remain missing; uniqueness and rejection belong solely
    to analyze_mutation. No cached decision can outlive a change to the input.
    """
    validate_input_schema(mutation)
    columns = []
    for column in REQUIRED_COLUMNS:
        values = mutation[column]
        if column in {"id_mutation", "numero_disposition", "id_parcelle"}:
            values = values.astype(str).mask(values.isna(), None)
        columns.append(values.tolist())
    return [MutationRow(*row) for row in zip(*columns, strict=True)]


def qualify_mutation(mutation: pd.DataFrame) -> MutationQualification:
    """Validate a complete DataFrame and return the canonical first V1 decision."""
    return analyze_mutation(_dataframe_rows(mutation)).decision


def build_observation(mutation: pd.DataFrame) -> dict[str, object] | None:
    """Build an observation only after admission; return None for exclusions.

    Auxiliary fields are copied from the residential row without additional
    date, geographic or economic filters. Amount, surface and prix_m2 are finite
    floats; the local code is normalized to an integer and its source label is
    preserved. The amount includes dependencies. Input is never modified.
    """
    return analyze_mutation(_dataframe_rows(mutation)).observation
