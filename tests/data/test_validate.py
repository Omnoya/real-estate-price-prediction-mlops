"""Check input structure separately from per-mutation business decisions."""

import pandas as pd
import pytest

from real_estate.data.validate import REQUIRED_COLUMNS, validate_input_schema


def test_all_required_columns_are_accepted_without_mutating_input() -> None:
    frame = pd.DataFrame({column: [pd.NA] for column in REQUIRED_COLUMNS})
    original = frame.copy(deep=True)

    assert validate_input_schema(frame) is None
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("missing_column", REQUIRED_COLUMNS)
def test_missing_required_column_is_reported(missing_column: str) -> None:
    frame = pd.DataFrame(columns=REQUIRED_COLUMNS).drop(columns=missing_column)

    with pytest.raises(ValueError, match=missing_column):
        validate_input_schema(frame)


def test_additional_columns_are_allowed() -> None:
    frame = pd.DataFrame(columns=[*REQUIRED_COLUMNS, "extra"])

    assert validate_input_schema(frame) is None


def test_local_code_is_required_alongside_local_label() -> None:
    assert "code_type_local" in REQUIRED_COLUMNS
    assert "type_local" in REQUIRED_COLUMNS
    frame = pd.DataFrame(columns=REQUIRED_COLUMNS).drop(columns="code_type_local")

    with pytest.raises(ValueError, match="code_type_local"):
        validate_input_schema(frame)


def test_schema_validation_does_not_qualify_empty_mutations() -> None:
    assert validate_input_schema(pd.DataFrame(columns=REQUIRED_COLUMNS)) is None


@pytest.mark.parametrize("duplicate", ["id_mutation", "extra"])
def test_duplicate_columns_are_rejected(duplicate: str) -> None:
    columns = [*REQUIRED_COLUMNS, "extra", duplicate]
    frame = pd.DataFrame(columns=columns)

    with pytest.raises(ValueError):
        validate_input_schema(frame)


@pytest.mark.parametrize("invalid", [None, [], {}, "not a dataframe"])
def test_non_dataframe_input_is_rejected(invalid: object) -> None:
    with pytest.raises(TypeError):
        validate_input_schema(invalid)
