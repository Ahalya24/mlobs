"""
Adapter for pyarrow.Table.

PyArrow uses a chunked array model.  We call .combine_chunks() to get a
contiguous ChunkedArray before converting to Python list, then to numpy.
Nulls in pyarrow columns are represented as None in .to_pylist(); we
convert them to np.nan for numeric columns so _strip_nulls works uniformly.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from mlobs.core.types import (
    CategoricalStats,
    ColumnArray,
    ColumnStats,
    NumericStats,
)


def _is_numeric_pa_type(t: Any) -> bool:
    """Return True if *t* is a numeric PyArrow type."""
    import pyarrow as pa
    return (
        pa.types.is_integer(t)
        or pa.types.is_floating(t)
        or pa.types.is_boolean(t)
        or pa.types.is_decimal(t)
    )


class ArrowAdapter:
    """Stateless adapter for pyarrow.Table objects."""

    def shape(self, df: Any) -> Tuple[int, int]:
        return (int(df.num_rows), int(df.num_columns))

    def column_names(self, df: Any) -> List[str]:
        return list(df.schema.names)

    def column_dtype(self, df: Any, col: str) -> str:
        return str(df.schema.field(col).type)

    def is_numeric(self, df: Any, col: str) -> bool:
        return _is_numeric_pa_type(df.schema.field(col).type)

    def to_numpy(self, df: Any, col: str) -> ColumnArray:
        chunked = df.column(col).combine_chunks()
        raw = chunked.to_pylist()
        if self.is_numeric(df, col):
            return np.array(
                [x if x is not None else np.nan for x in raw],
                dtype=np.float64,
            )
        return np.array(raw, dtype=object)

    def compute_column_stats(self, df: Any, col: str) -> ColumnStats:
        chunked = df.column(col)
        total = int(df.num_rows)
        null_count = int(chunked.null_count)

        if self.is_numeric(df, col):
            arr = self.to_numpy(df, col)
            arr_clean = arr[~np.isnan(arr)]
            n = len(arr_clean)
            numeric_stats = NumericStats(
                count=total,
                null_count=null_count,
                mean=float(np.mean(arr_clean)) if n else float("nan"),
                std=float(np.std(arr_clean, ddof=1)) if n > 1 else float("nan"),
                min=float(np.min(arr_clean)) if n else float("nan"),
                q25=float(np.percentile(arr_clean, 25)) if n else float("nan"),
                q50=float(np.percentile(arr_clean, 50)) if n else float("nan"),
                q75=float(np.percentile(arr_clean, 75)) if n else float("nan"),
                max=float(np.max(arr_clean)) if n else float("nan"),
            )
            return ColumnStats(
                name=col,
                dtype=self.column_dtype(df, col),
                kind="numeric",
                stats=numeric_stats,
            )

        # Categorical / string
        arr_obj = self.to_numpy(df, col)
        values = [v for v in arr_obj if v is not None]
        if values:
            unique_vals, counts = np.unique(values, return_counts=True)
            vc = {str(k): int(v) for k, v in zip(unique_vals, counts)}
            n_unique = len(unique_vals)
        else:
            vc = {}
            n_unique = 0

        cat_stats = CategoricalStats(
            count=total,
            null_count=null_count,
            n_unique=n_unique,
            value_counts=vc,
        )
        return ColumnStats(
            name=col,
            dtype=self.column_dtype(df, col),
            kind="categorical",
            stats=cat_stats,
        )
