"""Validate the input structure and define the DVF V1 exclusion vocabulary."""

from decimal import Decimal, InvalidOperation
from enum import Enum
from numbers import Number

import pandas as pd

REQUIRED_COLUMNS = (
    "id_mutation",
    "numero_disposition",
    "date_mutation",
    "nature_mutation",
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


class ExclusionReason(str, Enum):
    """Stable reasons, checked in contract order; only the first is returned."""

    NOT_A_SALE = "not_a_sale"
    INVALID_DISPOSITION = "invalid_disposition"
    INVALID_PARCEL = "invalid_parcel"
    INVALID_LOCAL_CODE = "invalid_local_code"
    RESIDENTIAL_ROW_COUNT = "residential_row_count"
    COMMERCIAL_OR_INDUSTRIAL_LOCAL = "commercial_or_industrial_local"
    INVALID_VALUE = "invalid_value"
    INCONSISTENT_VALUE = "inconsistent_value"
    INVALID_RESIDENTIAL_SURFACE = "invalid_residential_surface"
    UNREPRESENTABLE_FLOAT = "unrepresentable_float"


def validate_input_schema(frame: pd.DataFrame) -> None:
    """Require unambiguous columns without coercing or filtering input rows.

    Scalar business values are checked during qualification. Optional metadata
    may be null, and additional columns are allowed.
    An empty frame can have a valid schema, but cannot represent a mutation.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("DVF input must be a pandas DataFrame.")
    if not frame.columns.is_unique:
        raise ValueError("DVF input contains duplicate column names.")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required DVF columns: {', '.join(missing)}")


def parse_finite_number(value: object) -> Decimal | None:
    """Parse a numeric scalar, retaining decimal precision for equality checks.

    Booleans, missing values and nonfinite numbers are invalid. No magnitude
    restriction or float conversion is applied to otherwise finite decimals.
    """
    if isinstance(value, bool) or not isinstance(value, (str, Number, Decimal)):
        return None
    try:
        number = Decimal(str(value).strip())
        if number.is_finite():
            return number
    except (InvalidOperation, ValueError, OverflowError):
        pass
    return None
