"""
Adapter for polars.DataFrame / polars.LazyFrame.

polars uses its own null type (not NaN).  We cast numeric columns to Float64
and fill_null(nan) before converting to numpy so _strip_nulls sees standard
np.nan.  For string/categorical columns, nulls become None in the object array.

LazyFrame is supported transparently — _materialise() calls .collect().
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mlobs.core.types import (
    CategoricalStats,
    ColumnArray,
    ColumnStats,
    NumericStats,
)

# Base dtype name strings that are considered numeric (no parameterisation)
_NUMERIC_POLARS_BASE = frozenset({
    "Int8", "Int16", "Int32", "Int64",
    "UInt8", "UInt16", "UInt32", "UInt64",
    "Float32", "Float64",
    "Boolean",
    "Decimal",
})


def _materialise(df: Any) -> Any:
    """Collect a LazyFrame; return a DataFrame unchanged."""
    if type(df).__name__ == "LazyFrame":
        return df.collect()
    return df


def _dtype_base(dtype: Any) -> str:
    """Return the base name of a polars dtype (strips parameterisation)."""
    return str(dtype).split("(")[0]


class PolarsAdapter:
    """Stateless adapter for polars.DataFrame and polars.LazyFrame."""

    def shape(self, df: Any) -> tuple[int, int]:
        df = _materialise(df)
        return (df.height, df.width)

    def column_names(self, df: Any) -> list[str]:
        df = _materialise(df)
        return list(df.columns)

    def column_dtype(self, df: Any, col: str) -> str:
        df = _materialise(df)
        return str(df[col].dtype)

    def is_numeric(self, df: Any, col: str) -> bool:
        df = _materialise(df)
        return _dtype_base(df[col].dtype) in _NUMERIC_POLARS_BASE

    def to_numpy(self, df: Any, col: str) -> ColumnArray:
        df = _materialise(df)
        series = df[col]
        if self.is_numeric(df, col):
            import polars as pl
            return np.asarray(  # type: ignore[no-any-return]
                series.cast(pl.Float64).fill_null(float("nan")).to_numpy(),
                dtype=np.float64,
            )
        # String / categorical — nulls become None in object array
        return np.asarray(series.to_list(), dtype=object)  # type: ignore[no-any-return]

    def compute_column_stats(self, df: Any, col: str) -> ColumnStats:
        df = _materialise(df)
        series = df[col]
        total = len(series)
        null_count = int(series.null_count())

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
                dtype=str(series.dtype),
                kind="numeric",
                stats=numeric_stats,
            )

        # Categorical
        values = [v for v in series.to_list() if v is not None]
        unique_vals, counts = np.unique(values, return_counts=True)
        vc = {str(k): int(v) for k, v in zip(unique_vals, counts)}
        cat_stats = CategoricalStats(
            count=total,
            null_count=null_count,
            n_unique=len(unique_vals),
            value_counts=vc,
        )
        return ColumnStats(
            name=col,
            dtype=str(series.dtype),
            kind="categorical",
            stats=cat_stats,
        )
