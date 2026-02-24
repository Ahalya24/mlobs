"""
Adapter for pandas.DataFrame.

Null handling: pandas uses NaN for float columns and pd.NA / None for
nullable integer / string types.  We call .to_numpy(dtype=float, na_value=nan)
for numeric columns so all null flavours become np.nan, which _strip_nulls
in detectors handles.  For object columns we pass a raw object array;
None values are handled by _strip_nulls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from mlobs.core.types import (
    CategoricalStats,
    ColumnArray,
    ColumnStats,
    NumericStats,
)

# pandas dtype kinds that are considered numeric
_NUMERIC_KINDS = frozenset("iufcb")  # int, uint, float, complex, bool


class PandasAdapter:
    """Stateless adapter for pandas.DataFrame objects."""

    def shape(self, df: Any) -> Tuple[int, int]:
        return (int(df.shape[0]), int(df.shape[1]))

    def column_names(self, df: Any) -> List[str]:
        return [str(c) for c in df.columns]

    def column_dtype(self, df: Any, col: str) -> str:
        return str(df[col].dtype)

    def is_numeric(self, df: Any, col: str) -> bool:
        return df[col].dtype.kind in _NUMERIC_KINDS

    def to_numpy(self, df: Any, col: str) -> ColumnArray:
        series = df[col]
        if series.dtype.kind in _NUMERIC_KINDS:
            return series.to_numpy(dtype=np.float64, na_value=np.nan)
        return series.to_numpy(dtype=object)

    def compute_column_stats(self, df: Any, col: str) -> ColumnStats:
        series = df[col]
        total = int(len(series))
        null_count = int(series.isna().sum())

        if self.is_numeric(df, col):
            s = series.dropna().astype(float)
            n = len(s)
            numeric_stats = NumericStats(
                count=total,
                null_count=null_count,
                mean=float(s.mean()) if n else float("nan"),
                std=float(s.std()) if n else float("nan"),
                min=float(s.min()) if n else float("nan"),
                q25=float(s.quantile(0.25)) if n else float("nan"),
                q50=float(s.quantile(0.50)) if n else float("nan"),
                q75=float(s.quantile(0.75)) if n else float("nan"),
                max=float(s.max()) if n else float("nan"),
            )
            return ColumnStats(
                name=col,
                dtype=str(series.dtype),
                kind="numeric",
                stats=numeric_stats,
            )

        # Categorical / object
        vc = series.dropna().value_counts()
        cat_stats = CategoricalStats(
            count=total,
            null_count=null_count,
            n_unique=int(series.nunique()),
            value_counts={str(k): int(v) for k, v in vc.items()},
        )
        return ColumnStats(
            name=col,
            dtype=str(series.dtype),
            kind="categorical",
            stats=cat_stats,
        )
