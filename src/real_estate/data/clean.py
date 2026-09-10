"""Qualify complete DVF mutations and build one observation per admission.

Callers must provide every source row of a mutation, including dependencies,
commercial premises and repeated residential rows. Completeness cannot be
inferred from an already filtered frame or an isolated input chunk.
"""

from dataclasses import dataclass
from math import isfinite

import pandas as pd

from real_estate.data.validate import (
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


def _single_identifier(values: pd.Series) -> str | None:
    """Return one complete identifier, preserving leading zeros in strings."""
    if values.isna().any():
        return None
    identifiers = values.astype(str).str.strip()
    if identifiers.eq("").any() or identifiers.nunique() != 1:
        return None
    return str(identifiers.iloc[0])


def _validate_mutation_input(mutation: pd.DataFrame) -> None:
    """Reject malformed calls rather than assign a business exclusion to them."""
    validate_input_schema(mutation)
    if mutation.empty or _single_identifier(mutation["id_mutation"]) is None:
        raise ValueError("Expected one nonempty mutation with one complete id_mutation.")


def _local_codes(mutation: pd.DataFrame) -> pd.Series:
    """Parse codes without inferring missing or invalid categories from labels."""
    return mutation["code_type_local"].map(parse_finite_number)


def _value_exclusion(mutation: pd.DataFrame) -> ExclusionReason | None:
    """Check every amount before comparing them, without rounding or tolerance."""
    amounts = [parse_finite_number(value) for value in mutation["valeur_fonciere"]]
    if any(amount is None for amount in amounts):
        return ExclusionReason.INVALID_VALUE
    if len(set(amounts)) != 1:
        return ExclusionReason.INCONSISTENT_VALUE
    return None


def _float_observation_values(residential: pd.Series) -> dict[str, float] | None:
    """Check float representability after the Decimal business checks.

    A positive surface must remain positive after conversion. The ratio is
    checked using the converted operands, so it matches the final dataset.
    """
    try:
        amount = float(residential["valeur_fonciere"])
        surface = float(residential["surface_reelle_bati"])
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


def qualify_mutation(mutation: pd.DataFrame) -> MutationQualification:
    """Apply V1 to one complete mutation, returning its first exclusion reason.

    Residential candidates are rows with code 1 or 2, never deduplicated homes.
    Schema errors and missing/multiple mutation IDs raise an exception. Other
    rejections follow the locked contract order, independently of row order.
    Codes must belong to the V1 taxonomy; labels do not override valid codes.
    """
    _validate_mutation_input(mutation)
    if not mutation["nature_mutation"].eq("Vente").fillna(False).all():
        return MutationQualification(ExclusionReason.NOT_A_SALE)
    if _single_identifier(mutation["numero_disposition"]) is None:
        return MutationQualification(ExclusionReason.INVALID_DISPOSITION)
    if _single_identifier(mutation["id_parcelle"]) is None:
        return MutationQualification(ExclusionReason.INVALID_PARCEL)

    codes = _local_codes(mutation)
    if not codes.isin(LOCAL_CODES).all():
        return MutationQualification(ExclusionReason.INVALID_LOCAL_CODE)
    residential = mutation[codes.isin(RESIDENTIAL_CODES)]
    if len(residential) != 1:
        return MutationQualification(ExclusionReason.RESIDENTIAL_ROW_COUNT)
    if codes.eq(4).any():
        return MutationQualification(ExclusionReason.COMMERCIAL_OR_INDUSTRIAL_LOCAL)
    value_exclusion = _value_exclusion(mutation)
    if value_exclusion is not None:
        return MutationQualification(value_exclusion)

    surface = parse_finite_number(residential["surface_reelle_bati"].iloc[0])
    if surface is None or surface <= 0:
        return MutationQualification(ExclusionReason.INVALID_RESIDENTIAL_SURFACE)
    if _float_observation_values(residential.iloc[0]) is None:
        return MutationQualification(ExclusionReason.UNREPRESENTABLE_FLOAT)
    return MutationQualification()


def build_observation(mutation: pd.DataFrame) -> dict[str, object] | None:
    """Build an observation only after admission; return None for exclusions.

    Auxiliary fields are copied from the residential row without additional
    date, geographic or economic filters. Amount, surface and prix_m2 are finite
    floats; the local code is normalized to an integer and its source label is
    preserved. The amount includes dependencies. Input is never modified.
    """
    if not qualify_mutation(mutation).admissible:
        return None
    codes = _local_codes(mutation)
    residential = mutation[codes.isin(RESIDENTIAL_CODES)].iloc[0]
    observation = {column: residential[column] for column in OBSERVATION_SOURCE_COLUMNS}
    for column in ("id_mutation", "numero_disposition", "id_parcelle"):
        observation[column] = _single_identifier(mutation[column])

    numeric_values = _float_observation_values(residential)
    if numeric_values is None:
        return None
    observation.update(numeric_values)
    observation.update(
        code_type_local=int(codes[codes.isin(RESIDENTIAL_CODES)].iloc[0]),
        has_dependance=bool(codes.eq(3).any()),
        source_row_count=len(mutation),
    )
    return observation
