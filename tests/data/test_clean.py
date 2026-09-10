"""Exercise the V1 contract with complete, synthetic DVF mutations."""

from decimal import Decimal
from math import isfinite

import pandas as pd
import pytest

from real_estate.data.clean import build_observation, qualify_mutation
from real_estate.data.validate import ExclusionReason


def mutation(*overrides: dict[str, object]) -> pd.DataFrame:
    """Create source rows sharing a valid mutation unless overridden."""
    base = {
        "id_mutation": "2023-1",
        "numero_disposition": "000001",
        "date_mutation": "2023-02-01",
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
        "longitude": 5.2,
        "latitude": 46.2,
    }
    local_codes = {
        "Maison": 1,
        "Appartement": 2,
        "Dépendance": 3,
        "Local industriel. commercial ou assimilé": 4,
        "Local industriel, commercial ou assimilé": 4,
    }
    rows = []
    for overrides_for_row in overrides or ({},):
        row = base | overrides_for_row
        if "type_local" in overrides_for_row and "code_type_local" not in overrides_for_row:
            row["code_type_local"] = local_codes.get(row["type_local"])
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("local_code", "local_type"),
    [(1, "Maison"), ("1", "Maison"), (2, "Appartement"), ("2", "Appartement")],
)
def test_simple_residential_sale_is_admissible(
    local_code: int | str, local_type: str,
) -> None:
    frame = mutation({"code_type_local": local_code, "type_local": local_type})

    decision = qualify_mutation(frame)
    observation = build_observation(frame)

    assert decision.admissible is True
    assert decision.exclusion_reason is None
    assert observation is not None
    assert observation == {
        "id_mutation": "2023-1",
        "numero_disposition": "000001",
        "date_mutation": "2023-02-01",
        "valeur_fonciere": 250000.0,
        "id_parcelle": "010010000A0001",
        "code_commune": "01001",
        "code_departement": "01",
        "code_postal": "01000",
        "code_type_local": int(local_code),
        "type_local": local_type,
        "nombre_pieces_principales": 4,
        "surface_reelle_bati": 100.0,
        "longitude": 5.2,
        "latitude": 46.2,
        "has_dependance": False,
        "source_row_count": 1,
        "prix_m2": 2500.0,
    }
    assert type(observation["code_type_local"]) is int


@pytest.mark.parametrize("dependance_code", [3, "3"])
def test_dependance_is_allowed_and_residential_attributes_are_selected(
    dependance_code: object,
) -> None:
    annex = {
        "code_type_local": dependance_code,
        "type_local": "Dépendance",
        "surface_reelle_bati": None,
        "nombre_pieces_principales": 0,
        "longitude": None,
        "latitude": None,
    }
    frame = mutation(annex, {})
    frame.index = [0, 0]

    decision = qualify_mutation(frame)
    observation = build_observation(frame)

    assert decision.admissible is True
    assert observation is not None
    assert observation["has_dependance"] is True
    assert observation["source_row_count"] == 2
    assert observation["surface_reelle_bati"] == 100
    assert observation["nombre_pieces_principales"] == 4
    assert observation["longitude"] == 5.2
    assert observation["latitude"] == 46.2
    assert observation["prix_m2"] == 2500


@pytest.mark.parametrize(
    ("rows", "has_dependance", "residential_label"),
    [
        ([{"code_type_local": 1, "type_local": "Dépendance"}], False, "Dépendance"),
        ([{"code_type_local": 2, "type_local": "Maison"}], False, "Maison"),
        ([{}, {"code_type_local": 3, "type_local": "Appartement"}], True, "Maison"),
    ],
)
def test_valid_local_codes_take_priority_over_labels(
    rows: list[dict[str, object]], has_dependance: bool, residential_label: str,
) -> None:
    frame = mutation(*rows)

    assert qualify_mutation(frame).admissible is True
    observation = build_observation(frame)
    assert observation is not None
    assert observation["has_dependance"] is has_dependance
    assert observation["type_local"] == residential_label


@pytest.mark.parametrize(
    "invalid",
    [None, pd.NA, "", " ", "unknown", 0, 5, -1, 1.5,
     float("nan"), float("inf"), float("-inf"), True, False],
)
@pytest.mark.parametrize("on_annex", [False, True])
def test_invalid_local_codes_are_not_inferred_from_labels(
    invalid: object, on_annex: bool,
) -> None:
    rows = [{}, {"type_local": "Dépendance"}] if on_annex else [{}]
    rows[-1]["code_type_local"] = invalid
    frame = mutation(*rows)

    assert qualify_mutation(frame).exclusion_reason is ExclusionReason.INVALID_LOCAL_CODE
    assert build_observation(frame) is None


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        pytest.param(
            [{}, {}], ExclusionReason.RESIDENTIAL_ROW_COUNT,
            id="identical-residential-rows-are-not-deduplicated",
        ),
        pytest.param(
            [{}, {"type_local": "Appartement", "surface_reelle_bati": 50}],
            ExclusionReason.RESIDENTIAL_ROW_COUNT, id="two-different-residences",
        ),
        pytest.param(
            [{"type_local": "Dépendance"}],
            ExclusionReason.RESIDENTIAL_ROW_COUNT, id="no-residence",
        ),
        pytest.param(
            [{}, {"type_local": "Dépendance", "id_parcelle": "second-parcel"}],
            ExclusionReason.INVALID_PARCEL, id="multiple-parcels",
        ),
        pytest.param(
            [{}, {"type_local": "Dépendance", "numero_disposition": "000002"}],
            ExclusionReason.INVALID_DISPOSITION, id="multiple-dispositions",
        ),
        pytest.param(
            [{}, {"type_local": "Local industriel. commercial ou assimilé"}],
            ExclusionReason.COMMERCIAL_OR_INDUSTRIAL_LOCAL,
            id="commercial-label-period",
        ),
        pytest.param(
            [{}, {"type_local": "Local industriel, commercial ou assimilé"}],
            ExclusionReason.COMMERCIAL_OR_INDUSTRIAL_LOCAL,
            id="commercial-label-comma",
        ),
        pytest.param(
            [{}, {"type_local": None, "code_type_local": 4}],
            ExclusionReason.COMMERCIAL_OR_INDUSTRIAL_LOCAL,
            id="commercial-numeric-code",
        ),
        pytest.param(
            [{}, {"type_local": None, "code_type_local": "4"}],
            ExclusionReason.COMMERCIAL_OR_INDUSTRIAL_LOCAL,
            id="commercial-string-code",
        ),
        pytest.param(
            [{}, {"type_local": "Maison", "code_type_local": 4}],
            ExclusionReason.COMMERCIAL_OR_INDUSTRIAL_LOCAL,
            id="commercial-code-takes-priority-over-residential-label",
        ),
        pytest.param(
            [{}, {"type_local": "Dépendance", "valeur_fonciere": 250001}],
            ExclusionReason.INCONSISTENT_VALUE, id="different-values",
        ),
        pytest.param(
            [{"nature_mutation": "Echange"}],
            ExclusionReason.NOT_A_SALE, id="not-a-sale",
        ),
        pytest.param(
            [{}, {"type_local": "Dépendance", "nature_mutation": "Echange"}],
            ExclusionReason.NOT_A_SALE, id="mixed-natures",
        ),
        pytest.param(
            [{"nature_mutation": None}],
            ExclusionReason.NOT_A_SALE, id="missing-nature",
        ),
    ],
)
def test_excluded_mutations_never_produce_observations(
    rows: list[dict[str, object]], reason: ExclusionReason,
) -> None:
    frame = mutation(*rows)
    original = frame.copy(deep=True)

    decision = qualify_mutation(frame)

    assert decision.admissible is False
    assert decision.exclusion_reason is reason
    assert build_observation(frame) is None
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    ("column", "reason"),
    [
        ("numero_disposition", ExclusionReason.INVALID_DISPOSITION),
        ("id_parcelle", ExclusionReason.INVALID_PARCEL),
    ],
)
@pytest.mark.parametrize("missing", [None, pd.NA, float("nan"), "", "   "])
def test_identifiers_must_be_present_on_every_row(
    column: str, reason: ExclusionReason, missing: object,
) -> None:
    frame = mutation({}, {"type_local": "Dépendance", column: missing})

    assert qualify_mutation(frame).exclusion_reason is reason
    assert build_observation(frame) is None


@pytest.mark.parametrize(
    "invalid",
    [None, pd.NA, "", " ", "unknown", float("nan"), float("inf"),
     float("-inf"), "NaN", "Infinity", True, False],
)
def test_invalid_amount_on_any_row_excludes_mutation(invalid: object) -> None:
    frame = mutation({}, {"type_local": "Dépendance", "valeur_fonciere": invalid})

    assert qualify_mutation(frame).exclusion_reason is ExclusionReason.INVALID_VALUE
    assert build_observation(frame) is None


@pytest.mark.parametrize(
    "invalid",
    [0, -1, "0", "-2", None, pd.NA, "", "unknown", float("nan"),
     float("inf"), float("-inf"), "NaN", "Infinity", True, False],
)
def test_residential_surface_must_be_finite_and_positive(invalid: object) -> None:
    frame = mutation({"surface_reelle_bati": invalid})

    decision = qualify_mutation(frame)

    assert decision.exclusion_reason is ExclusionReason.INVALID_RESIDENTIAL_SURFACE
    assert build_observation(frame) is None


def test_equivalent_decimal_amounts_are_consistent() -> None:
    frame = mutation(
        {"valeur_fonciere": "250000.00"},
        {"type_local": "Dépendance", "valeur_fonciere": Decimal("250000.000")},
        {"type_local": "Dépendance", "valeur_fonciere": 250000},
    )

    assert qualify_mutation(frame).admissible is True
    assert build_observation(frame)["prix_m2"] == 2500


@pytest.mark.parametrize(
    ("first", "second"),
    [("9007199254740992", "9007199254740993"), ("100.00", "100.000000000001")],
)
def test_different_amounts_are_not_merged_by_float_rounding_or_tolerance(
    first: str, second: str,
) -> None:
    frame = mutation(
        {"valeur_fonciere": first},
        {"type_local": "Dépendance", "valeur_fonciere": second},
    )

    assert qualify_mutation(frame).exclusion_reason is ExclusionReason.INCONSISTENT_VALUE
    assert build_observation(frame) is None


@pytest.mark.parametrize(
    ("amount", "surface"),
    [(1, 1000), (722590020, 119), (375000, 1), (0, 100), (-1, 100)],
)
def test_no_economic_thresholds_are_applied(amount: int, surface: int) -> None:
    frame = mutation({"valeur_fonciere": amount, "surface_reelle_bati": surface})

    assert qualify_mutation(frame).admissible is True
    assert build_observation(frame)["prix_m2"] == pytest.approx(amount / surface)


@pytest.mark.parametrize(
    ("amount", "surface"),
    [
        pytest.param("1e400", "1", id="amount-overflows-float"),
        pytest.param("-1e400", "1", id="negative-amount-overflows-float"),
        pytest.param("1", "1e400", id="surface-overflows-float"),
        pytest.param("1", "1e-400", id="surface-underflows-to-zero"),
        pytest.param("1e308", "1e-308", id="price-overflows-float"),
    ],
)
def test_unrepresentable_float_values_exclude_mutations(
    amount: str, surface: str,
) -> None:
    frame = mutation({"valeur_fonciere": amount, "surface_reelle_bati": surface})

    decision = qualify_mutation(frame)

    assert decision.admissible is False
    assert decision.exclusion_reason is ExclusionReason.UNREPRESENTABLE_FLOAT
    assert build_observation(frame) is None


@pytest.mark.parametrize(
    ("amount", "surface"),
    [("250000.00", "100"), (Decimal("100.1"), Decimal(3)), (722590020, 119)],
)
def test_final_amount_surface_and_price_are_finite_floats(
    amount: object, surface: object,
) -> None:
    frame = mutation({"valeur_fonciere": amount, "surface_reelle_bati": surface})

    observation = build_observation(frame)

    assert observation is not None
    for column in ("valeur_fonciere", "surface_reelle_bati", "prix_m2"):
        assert type(observation[column]) is float
        assert isfinite(observation[column])
    assert observation["valeur_fonciere"] == float(amount)
    assert observation["surface_reelle_bati"] == float(surface)
    assert observation["prix_m2"] == pytest.approx(float(amount) / float(surface))


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"nature_mutation": "Echange", "numero_disposition": None},
         ExclusionReason.NOT_A_SALE),
        ({"numero_disposition": None, "id_parcelle": None},
         ExclusionReason.INVALID_DISPOSITION),
        ({"id_parcelle": None, "type_local": "Dépendance"},
         ExclusionReason.INVALID_PARCEL),
        ({"id_parcelle": None, "code_type_local": None},
         ExclusionReason.INVALID_PARCEL),
        ({"code_type_local": None, "valeur_fonciere": None},
         ExclusionReason.INVALID_LOCAL_CODE),
        ({"type_local": "Local industriel. commercial ou assimilé"},
         ExclusionReason.RESIDENTIAL_ROW_COUNT),
        ({"code_type_local": 4, "valeur_fonciere": None},
         ExclusionReason.RESIDENTIAL_ROW_COUNT),
        ({"valeur_fonciere": None, "surface_reelle_bati": 0},
         ExclusionReason.INVALID_VALUE),
    ],
)
def test_first_failed_rule_determines_reason(
    overrides: dict[str, object], reason: ExclusionReason,
) -> None:
    frame = mutation(overrides)

    assert qualify_mutation(frame).exclusion_reason is reason
    assert qualify_mutation(frame).exclusion_reason is reason
    assert build_observation(frame) is None


def test_commercial_local_precedes_invalid_amount() -> None:
    frame = mutation(
        {"valeur_fonciere": None},
        {"type_local": "Local industriel, commercial ou assimilé"},
    )

    decision = qualify_mutation(frame)

    assert decision.exclusion_reason is ExclusionReason.COMMERCIAL_OR_INDUSTRIAL_LOCAL
    assert build_observation(frame) is None


def test_inconsistent_value_precedes_invalid_surface() -> None:
    frame = mutation(
        {"surface_reelle_bati": 0},
        {"type_local": "Dépendance", "valeur_fonciere": 1},
    )

    assert qualify_mutation(frame).exclusion_reason is ExclusionReason.INCONSISTENT_VALUE
    assert build_observation(frame) is None


def test_exclusion_priority_does_not_depend_on_source_row_order() -> None:
    frame = mutation(
        {"valeur_fonciere": None},
        {"type_local": "Dépendance", "valeur_fonciere": "250001"},
        {"type_local": "Dépendance", "valeur_fonciere": "250002"},
    )

    for candidate in (frame, frame.iloc[::-1]):
        decision = qualify_mutation(candidate)
        assert decision.exclusion_reason is ExclusionReason.INVALID_VALUE
        assert build_observation(candidate) is None


@pytest.mark.parametrize("missing", [None, pd.NA, "", "   "])
def test_missing_mutation_identifier_is_an_input_error(missing: object) -> None:
    frame = mutation({"id_mutation": missing})

    with pytest.raises(ValueError):
        qualify_mutation(frame)
    with pytest.raises(ValueError):
        build_observation(frame)


def test_multiple_mutations_are_an_input_error() -> None:
    frame = mutation({}, {"id_mutation": "2023-2"})

    with pytest.raises(ValueError):
        qualify_mutation(frame)
    with pytest.raises(ValueError):
        build_observation(frame)


def test_empty_mutation_is_an_input_error() -> None:
    frame = mutation().iloc[:0]

    with pytest.raises(ValueError):
        qualify_mutation(frame)
    with pytest.raises(ValueError):
        build_observation(frame)


def test_identifiers_keep_leading_zeroes_and_input_is_unchanged() -> None:
    frame = mutation(
        {"id_mutation": " 2023-1 ", "numero_disposition": " 000001 ",
         "id_parcelle": " 010010000A0001 "},
        {"type_local": "Dépendance", "id_mutation": " 2023-1 "},
    )
    original = frame.copy(deep=True)

    decision = qualify_mutation(frame)
    observation = build_observation(frame)

    assert decision.admissible is True
    assert observation["id_mutation"] == "2023-1"
    assert observation["numero_disposition"] == "000001"
    assert observation["id_parcelle"] == "010010000A0001"
    assert observation["code_departement"] == "01"
    assert observation["code_postal"] == "01000"
    pd.testing.assert_frame_equal(frame, original)


def test_missing_auxiliary_fields_do_not_add_uncontracted_filters() -> None:
    nullable = {
        "date_mutation": pd.NA,
        "code_commune": pd.NA,
        "code_departement": pd.NA,
        "code_postal": pd.NA,
        "nombre_pieces_principales": pd.NA,
        "longitude": pd.NA,
        "latitude": pd.NA,
    }
    frame = mutation(nullable)

    assert qualify_mutation(frame).admissible is True
    observation = build_observation(frame)
    assert observation is not None
    for column in nullable:
        assert pd.isna(observation[column])
