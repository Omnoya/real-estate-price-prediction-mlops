"""Compare qualified Parquet values in source order without loading entire files."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _arrays_equal(left: pa.Array, right: pa.Array) -> bool:
    """Use exact value equality, treating corresponding NaNs as equal, not null."""
    if left.equals(right):
        return True
    if not pa.types.is_floating(left.type):
        return False
    equal_values = pc.fill_null(pc.equal(left, right), False)
    both_null = pc.and_(pc.is_null(left), pc.is_null(right))
    both_nan = pc.fill_null(pc.and_(pc.is_nan(left), pc.is_nan(right)), False)
    return pc.all(pc.or_(pc.or_(equal_values, both_null), both_nan)).as_py()


def compare_qualified_parquets(
    left: Path, right: Path, *, batch_size: int = 10_000,
) -> bool:
    """Compare schema, row count and ordered values using bounded Arrow batches.

    Column names, order, types and nullability must match. File/schema metadata,
    compression and row-group boundaries do not define logical equality.
    Floats use exact equality without tolerance; corresponding NaNs compare equal
    and remain distinct from null. Signed zeros compare equal. I/O errors propagate.
    Neither input is changed and no output is written.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    with closing(pq.ParquetFile(left)) as left_file, closing(
        pq.ParquetFile(right),
    ) as right_file:
        if not left_file.schema_arrow.equals(right_file.schema_arrow):
            return False
        if left_file.metadata.num_rows != right_file.metadata.num_rows:
            return False
        left_batches = (
            batch for batch in left_file.iter_batches(batch_size=batch_size)
            if batch.num_rows
        )
        right_batches = (
            batch for batch in right_file.iter_batches(batch_size=batch_size)
            if batch.num_rows
        )
        left_batch = next(left_batches, None)
        right_batch = next(right_batches, None)
        left_offset = right_offset = 0
        while left_batch is not None and right_batch is not None:
            count = min(
                left_batch.num_rows - left_offset,
                right_batch.num_rows - right_offset,
            )
            left_slice = left_batch.slice(left_offset, count)
            right_slice = right_batch.slice(right_offset, count)
            if not all(
                _arrays_equal(left_column, right_column)
                for left_column, right_column in zip(
                    left_slice.columns, right_slice.columns, strict=True,
                )
            ):
                return False
            left_offset += count
            right_offset += count
            if left_offset == left_batch.num_rows:
                left_batch = next(left_batches, None)
                left_offset = 0
            if right_offset == right_batch.num_rows:
                right_batch = next(right_batches, None)
                right_offset = 0
        return left_batch is None and right_batch is None
