"""Logical comparisons use only tiny, temporary synthetic Parquet files."""

from __future__ import annotations

import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from real_estate.data import compare as compare_module
from real_estate.data.compare import compare_qualified_parquets


def write_pair(
    root: Path, left: pa.Table, right: pa.Table,
) -> tuple[Path, Path]:
    """Deliberately vary storage layout independently of logical values."""
    left_path, right_path = root / "left.parquet", root / "right.parquet"
    pq.write_table(left, left_path, row_group_size=3, compression=None)
    pq.write_table(right, right_path, row_group_size=1, compression="gzip")
    return left_path, right_path


@pytest.fixture
def mixed_table() -> pa.Table:
    return pa.table({
        "text": pa.array(["a", None, "", "d", "e"], type=pa.string()),
        "integer": pa.array([1, None, 0, -1, 5], type=pa.int64()),
        "boolean": pa.array([True, None, False, True, False], type=pa.bool_()),
        "float": pa.array([1.5, None, 0.0, -2.0, 5.0], type=pa.float64()),
    })


def test_equal_values_ignore_storage_layout_and_metadata(
    tmp_path: Path, mixed_table: pa.Table,
) -> None:
    paths = write_pair(
        tmp_path, mixed_table, mixed_table.replace_schema_metadata({b"run": b"new"}),
    )
    assert paths[0].read_bytes() != paths[1].read_bytes()
    assert compare_qualified_parquets(*paths, batch_size=2)


@pytest.mark.parametrize("column,replacement", [
    ("text", "changed"), ("integer", 42), ("boolean", True), ("float", 9.0),
])
def test_changed_value_is_detected(
    tmp_path: Path, mixed_table: pa.Table, column: str, replacement: object,
) -> None:
    rows = mixed_table.to_pylist()
    rows[-1][column] = replacement
    changed = pa.Table.from_pylist(rows, schema=mixed_table.schema)
    assert not compare_qualified_parquets(
        *write_pair(tmp_path, mixed_table, changed), batch_size=2,
    )


@pytest.mark.parametrize("column,zero", [
    ("text", ""), ("integer", 0), ("boolean", False), ("float", 0.0),
])
def test_null_does_not_equal_empty_or_zero(
    tmp_path: Path, mixed_table: pa.Table, column: str, zero: object,
) -> None:
    rows = mixed_table.to_pylist()
    rows[1][column] = zero
    changed = pa.Table.from_pylist(rows, schema=mixed_table.schema)
    assert not compare_qualified_parquets(*write_pair(tmp_path, mixed_table, changed))


@pytest.mark.parametrize("change", ["name", "order", "type", "nullable"])
def test_schema_differences_are_detected(tmp_path: Path, change: str) -> None:
    original = pa.table({"a": [1, 2], "b": [3, 4]})
    if change == "name":
        changed = original.rename_columns(["different", "b"])
    elif change == "order":
        changed = original.select(["b", "a"])
    elif change == "type":
        changed = original.cast(pa.schema([("a", pa.int32()), ("b", pa.int64())]))
    else:
        changed = original.cast(pa.schema([
            pa.field("a", pa.int64(), nullable=False), pa.field("b", pa.int64()),
        ]))
    assert not compare_qualified_parquets(*write_pair(tmp_path, original, changed))


def test_row_count_difference_is_detected(
    tmp_path: Path, mixed_table: pa.Table,
) -> None:
    assert not compare_qualified_parquets(
        *write_pair(tmp_path, mixed_table, mixed_table.slice(0, 4)),
    )


def test_row_order_difference_is_detected(
    tmp_path: Path, mixed_table: pa.Table,
) -> None:
    reordered = mixed_table.take(pa.array([4, 3, 2, 1, 0]))
    assert not compare_qualified_parquets(
        *write_pair(tmp_path, mixed_table, reordered), batch_size=2,
    )


@pytest.mark.parametrize("right,expected", [
    ([float("nan"), None, 0.0, 1.0], True),
    ([float("nan"), None, -0.0, 1.0], True),
    ([None, float("nan"), 0.0, 1.0], False),
    ([float("nan"), None, 0.0, math.nextafter(1.0, 2.0)], False),
])
def test_float_equality_is_exact_with_explicit_nan_semantics(
    tmp_path: Path, right: list[float | None], expected: bool,
) -> None:
    left_table = pa.table({"float": [float("nan"), None, 0.0, 1.0]})
    right_table = pa.table({"float": right})
    assert compare_qualified_parquets(
        *write_pair(tmp_path, left_table, right_table), batch_size=2,
    ) is expected


def test_empty_parquets_compare_with_schema(tmp_path: Path) -> None:
    left = pa.table({"value": pa.array([], type=pa.int64())})
    assert compare_qualified_parquets(*write_pair(tmp_path, left, left))
    changed = pa.table({"value": pa.array([], type=pa.string())})
    assert not compare_qualified_parquets(*write_pair(tmp_path, left, changed))


def test_batch_sizes_can_differ_between_readers(
    tmp_path: Path, mixed_table: pa.Table, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = write_pair(tmp_path, mixed_table, mixed_table)
    original_iter_batches = pq.ParquetFile.iter_batches
    calls = 0

    def uneven_batches(self: pq.ParquetFile, *, batch_size: int):
        nonlocal calls
        calls += 1
        actual_size = 1 if calls == 1 else batch_size
        return original_iter_batches(self, batch_size=actual_size)

    monkeypatch.setattr(compare_module.pq.ParquetFile, "iter_batches", uneven_batches)
    assert compare_qualified_parquets(*paths, batch_size=3)
    assert calls == 2


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size_fails_before_io(tmp_path: Path, batch_size: object) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        compare_qualified_parquets(
            tmp_path / "absent.parquet", tmp_path / "also_absent.parquet",
            batch_size=batch_size,
        )


def test_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compare_qualified_parquets(tmp_path / "absent.parquet", tmp_path / "other.parquet")
