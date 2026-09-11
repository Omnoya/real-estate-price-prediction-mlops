"""Pin V1 outcomes while comparing the public DataFrame and canonical row APIs."""

from __future__ import annotations

from decimal import Decimal
from math import copysign, isfinite, isnan

import pandas as pd
import pytest

from real_estate.data import clean
from real_estate.data.validate import REQUIRED_COLUMNS, ExclusionReason

BASE = {
    "id_mutation": "2025-1",
    "numero_disposition": "000001",
    "date_mutation": "2025-01-02",
    "nature_mutation": "Vente",
    "valeur_fonciere": "250000.00",
    "id_parcelle": "010010000A0001",
    "code_commune": "01001",
    "code_departement": "01",
    "code_postal": "01000",
    "code_type_local": 1,
    "type_local": "Maison",
    "nombre_pieces_principales": 4,
    "surface_reelle_bati": "100",
    "longitude": None,
    "latitude": None,
}
ANNEX = {
    "code_type_local": 3,
    "type_local": "Dépendance",
    "surface_reelle_bati": None,
    "nombre_pieces_principales": 0,
}


def inputs(changes: list[dict[str, object]]) -> tuple[pd.DataFrame, list[clean.MutationRow]]:
    """Keep synthetic Python scalar types identical for both entry points."""
    source = [BASE | change for change in changes]
    frame = pd.DataFrame(source, dtype=object)
    rows = [clean.MutationRow(*(row[column] for column in REQUIRED_COLUMNS)) for row in source]
    return frame, rows


def assert_same_observation(
    actual: dict[str, object], expected: dict[str, object],
) -> None:
    """Compare each field, distinguishing null sentinels and exact finite floats."""
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        if value is None or value is pd.NA or value is pd.NaT:
            assert actual[name] is value
        elif isinstance(value, float) and isnan(value):
            assert isinstance(actual[name], float) and isnan(actual[name])
        else:
            assert actual[name] == value


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        pytest.param([{}], None, id="house"),
        pytest.param([{"code_type_local": 2, "type_local": "Appartement"}], None,
                     id="apartment"),
        pytest.param([ANNEX, {}, ANNEX], None, id="dependencies-before-and-after-home"),
        pytest.param([{"code_type_local": "1.0", "type_local": "Dépendance"}], None,
                     id="code-overrides-label"),
        pytest.param([{"valeur_fonciere": 0}], None, id="zero-amount-allowed"),
        pytest.param([{"valeur_fonciere": -100}], None, id="negative-amount-allowed"),
        pytest.param([{"valeur_fonciere": "1e-400"}], None,
                     id="amount-underflows-to-allowed-zero"),
        pytest.param([{"valeur_fonciere": Decimal("100.1"),
                       "surface_reelle_bati": Decimal(3),
                       "code_type_local": Decimal("1.00")}], None,
                     id="representable-decimals"),
        pytest.param([{"valeur_fonciere": "250000"},
                      ANNEX | {"valeur_fonciere": Decimal("250000.000")}], None,
                     id="exactly-equivalent-amounts"),
        pytest.param([{"nature_mutation": "Echange"}], ExclusionReason.NOT_A_SALE,
                     id="not-a-sale"),
        pytest.param([{}, ANNEX | {"nature_mutation": "Echange"}],
                     ExclusionReason.NOT_A_SALE, id="mixed-natures"),
        pytest.param([{"nature_mutation": "Vente "}], ExclusionReason.NOT_A_SALE,
                     id="nature-is-not-stripped"),
        pytest.param([{"nature_mutation": pd.NA}], ExclusionReason.NOT_A_SALE,
                     id="missing-nature"),
        pytest.param([{}, ANNEX | {"numero_disposition": "000002"}],
                     ExclusionReason.INVALID_DISPOSITION, id="multiple-dispositions"),
        pytest.param([{}, ANNEX | {"numero_disposition": pd.NA}],
                     ExclusionReason.INVALID_DISPOSITION, id="null-disposition"),
        pytest.param([{}, ANNEX | {"id_parcelle": "010010000A0002"}],
                     ExclusionReason.INVALID_PARCEL, id="multiple-parcels"),
        pytest.param([{"id_parcelle": " "}], ExclusionReason.INVALID_PARCEL,
                     id="blank-parcel"),
        pytest.param([{"code_type_local": None}], ExclusionReason.INVALID_LOCAL_CODE,
                     id="null-local-code"),
        pytest.param([{}, ANNEX | {"code_type_local": 5}],
                     ExclusionReason.INVALID_LOCAL_CODE, id="invalid-annex-code"),
        pytest.param([{"code_type_local": True}], ExclusionReason.INVALID_LOCAL_CODE,
                     id="boolean-code"),
        pytest.param([{"code_type_local": "NaN"}], ExclusionReason.INVALID_LOCAL_CODE,
                     id="nonfinite-code"),
        pytest.param([ANNEX], ExclusionReason.RESIDENTIAL_ROW_COUNT, id="zero-residential"),
        pytest.param([{}, {}], ExclusionReason.RESIDENTIAL_ROW_COUNT,
                     id="identical-residential-rows-stay-distinct"),
        pytest.param([{}, {"code_type_local": 4}],
                     ExclusionReason.COMMERCIAL_OR_INDUSTRIAL_LOCAL, id="commercial"),
        pytest.param([{"valeur_fonciere": ""}], ExclusionReason.INVALID_VALUE,
                     id="empty-value"),
        pytest.param([{}, ANNEX | {"valeur_fonciere": pd.NA}],
                     ExclusionReason.INVALID_VALUE, id="null-value-on-annex"),
        pytest.param([{"valeur_fonciere": "100,00"}], ExclusionReason.INVALID_VALUE,
                     id="comma-is-not-accepted-by-clean"),
        pytest.param([{}, ANNEX | {"valeur_fonciere": "250001"}],
                     ExclusionReason.INCONSISTENT_VALUE, id="inconsistent-values"),
        pytest.param([{"valeur_fonciere": "9007199254740992"},
                      ANNEX | {"valeur_fonciere": "9007199254740993"}],
                     ExclusionReason.INCONSISTENT_VALUE, id="no-float-rounded-equality"),
        pytest.param([{"surface_reelle_bati": None}],
                     ExclusionReason.INVALID_RESIDENTIAL_SURFACE, id="null-surface"),
        pytest.param([{"surface_reelle_bati": ""}],
                     ExclusionReason.INVALID_RESIDENTIAL_SURFACE, id="empty-surface"),
        pytest.param([{"surface_reelle_bati": 0}],
                     ExclusionReason.INVALID_RESIDENTIAL_SURFACE, id="zero-surface"),
        pytest.param([{"surface_reelle_bati": -1}],
                     ExclusionReason.INVALID_RESIDENTIAL_SURFACE, id="negative-surface"),
        pytest.param([{"valeur_fonciere": "1e400"}],
                     ExclusionReason.UNREPRESENTABLE_FLOAT, id="amount-float-overflow"),
        pytest.param([{"surface_reelle_bati": "1e-400"}],
                     ExclusionReason.UNREPRESENTABLE_FLOAT, id="surface-float-underflow"),
        pytest.param([{"valeur_fonciere": "1e308", "surface_reelle_bati": "1e-308"}],
                     ExclusionReason.UNREPRESENTABLE_FLOAT, id="ratio-float-overflow"),
    ],
)
def test_classic_and_canonical_apis_have_exact_business_parity(
    changes: list[dict[str, object]], reason: ExclusionReason | None,
) -> None:
    frame, rows = inputs(changes)
    original = frame.copy(deep=True)
    original_rows = rows.copy()

    analysis = clean.analyze_mutation(rows)
    decision = clean.qualify_mutation(frame)
    observation = clean.build_observation(frame)

    assert analysis.decision == decision
    assert decision.exclusion_reason is reason
    assert decision.admissible is (reason is None)
    if reason is not None:
        assert analysis.observation is None
        assert observation is None
    else:
        assert observation is not None
        assert analysis.observation is not None
        assert_same_observation(analysis.observation, observation)
        assert observation["source_row_count"] == len(rows)
        for column in ("valeur_fonciere", "surface_reelle_bati", "prix_m2"):
            assert type(observation[column]) is float
            assert isfinite(observation[column])
    pd.testing.assert_frame_equal(frame, original)
    assert rows == original_rows


def test_admitted_context_selects_the_residential_row_and_exact_observation() -> None:
    frame, rows = inputs([
        ANNEX | {"code_postal": "99999", "longitude": 0.0},
        {"code_type_local": 2, "type_local": "Appartement"},
        ANNEX,
    ])
    analysis = clean.analyze_mutation(rows)
    expected = {
        "id_mutation": "2025-1", "numero_disposition": "000001",
        "date_mutation": "2025-01-02", "valeur_fonciere": 250000.0,
        "id_parcelle": "010010000A0001", "code_commune": "01001",
        "code_departement": "01", "code_postal": "01000", "code_type_local": 2,
        "type_local": "Appartement", "nombre_pieces_principales": 4,
        "surface_reelle_bati": 100.0, "longitude": None, "latitude": None,
        "has_dependance": True, "source_row_count": 3, "prix_m2": 2500.0,
    }

    assert analysis.decision.admissible is True
    assert analysis.residential_index == 1
    assert_same_observation(analysis.observation, expected)
    assert_same_observation(clean.build_observation(frame), expected)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ([{"nature_mutation": "Echange", "numero_disposition": None}],
         ExclusionReason.NOT_A_SALE),
        ([{"numero_disposition": None, "id_parcelle": None}],
         ExclusionReason.INVALID_DISPOSITION),
        ([{"id_parcelle": None, "code_type_local": None}], ExclusionReason.INVALID_PARCEL),
        ([{}, {"code_type_local": None}], ExclusionReason.INVALID_LOCAL_CODE),
        ([{"code_type_local": 4, "valeur_fonciere": None}],
         ExclusionReason.RESIDENTIAL_ROW_COUNT),
        ([{"valeur_fonciere": None}, {"code_type_local": 4}],
         ExclusionReason.COMMERCIAL_OR_INDUSTRIAL_LOCAL),
        ([{"surface_reelle_bati": 0}, ANNEX | {"valeur_fonciere": 1}],
         ExclusionReason.INCONSISTENT_VALUE),
        ([{}, ANNEX | {"valeur_fonciere": 1}, ANNEX | {"valeur_fonciere": None}],
         ExclusionReason.INVALID_VALUE),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_first_rejection_priority_is_identical_in_both_row_orders(
    changes: list[dict[str, object]], reason: ExclusionReason, reverse: bool,
) -> None:
    frame, rows = inputs(list(reversed(changes)) if reverse else changes)
    analysis = clean.analyze_mutation(rows)
    assert analysis.decision.exclusion_reason is reason
    assert clean.qualify_mutation(frame).exclusion_reason is reason
    assert analysis.observation is None
    assert clean.build_observation(frame) is None


@pytest.mark.parametrize(
    ("amount", "surface"),
    [(Decimal("100.1"), Decimal(3)), ("-0.0", "100"), ("1e-400", "100")],
)
def test_final_float_calculation_keeps_source_conversion_and_signed_zero(
    amount: object, surface: object,
) -> None:
    frame, rows = inputs([{"valeur_fonciere": amount, "surface_reelle_bati": surface}])
    analysis = clean.analyze_mutation(rows)
    expected_price = float(amount) / float(surface)
    for observation in (analysis.observation, clean.build_observation(frame)):
        assert observation is not None
        assert observation["valeur_fonciere"] == float(amount)
        assert observation["surface_reelle_bati"] == float(surface)
        assert observation["prix_m2"] == expected_price
        assert copysign(1, observation["prix_m2"]) == copysign(1, expected_price)


def test_public_builder_reanalyzes_a_modified_frame_instead_of_using_stale_admission() -> None:
    frame, _ = inputs([{}])
    assert clean.qualify_mutation(frame).admissible is True
    assert clean.build_observation(frame)["prix_m2"] == 2500.0

    frame.loc[0, "surface_reelle_bati"] = "200"
    assert clean.build_observation(frame)["prix_m2"] == 1250.0
    frame.loc[0, "code_type_local"] = None
    assert clean.build_observation(frame) is None
    assert clean.qualify_mutation(frame).exclusion_reason is ExclusionReason.INVALID_LOCAL_CODE


@pytest.mark.parametrize("identifier", [[1, 2], {"synthetic": 1}])
def test_dataframe_identifier_adapter_preserves_object_stringification(identifier: object) -> None:
    frame, _ = inputs([
        {"id_mutation": identifier, "numero_disposition": identifier,
         "id_parcelle": identifier},
    ])
    original = frame.copy(deep=True)
    observation = clean.build_observation(frame)
    assert clean.qualify_mutation(frame).admissible is True
    assert observation is not None
    for column in ("id_mutation", "numero_disposition", "id_parcelle"):
        assert observation[column] == str(identifier)
    pd.testing.assert_frame_equal(frame, original)


def test_dataframe_identifier_adapter_preserves_datetime_dtype_formatting() -> None:
    frame, _ = inputs([{}])
    for column in ("id_mutation", "numero_disposition", "id_parcelle"):
        frame[column] = pd.to_datetime(["2025-01-02"])
    original = frame.copy(deep=True)

    observation = clean.build_observation(frame)

    assert observation is not None
    for column in ("id_mutation", "numero_disposition", "id_parcelle"):
        assert observation[column] == "2025-01-02"
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("missing", [None, pd.NA, float("nan")])
def test_nullable_auxiliary_values_are_preserved_without_new_filters(missing: object) -> None:
    columns = ("date_mutation", "code_commune", "code_departement", "code_postal",
               "type_local", "nombre_pieces_principales", "longitude", "latitude")
    frame, rows = inputs([{column: missing for column in columns}])
    analysis = clean.analyze_mutation(rows)
    observation = clean.build_observation(frame)
    assert clean.qualify_mutation(frame).admissible is True
    assert analysis.observation is not None
    assert observation is not None
    assert_same_observation(analysis.observation, observation)
    for column in columns:
        assert pd.isna(observation[column])


@pytest.mark.parametrize("identifier", [None, pd.NA, "", "  ", "different-id"])
def test_invalid_mutation_calls_raise_before_assigning_a_business_rejection(
    identifier: object,
) -> None:
    frame, rows = inputs([
        {"nature_mutation": "Echange"},
        ANNEX | {"id_mutation": identifier},
    ])
    for call in (
        lambda: clean.analyze_mutation(rows),
        lambda: clean.qualify_mutation(frame),
        lambda: clean.build_observation(frame),
    ):
        with pytest.raises(ValueError):
            call()
