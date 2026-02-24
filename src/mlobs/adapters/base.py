"""
DataFrameAdapter Protocol — the bridge between a backend dataframe and
the mlobs internal representation.

Design: structural subtyping (Protocol)
----------------------------------------
We use typing.Protocol so that adapter classes do not have to inherit from
a common base class.  Any class that implements the six required methods is
a valid DataFrameAdapter.  This lets third-party users write adapters for
custom frame types (e.g. Dask, cuDF) without modifying the mlobs source.

The Protocol is runtime_checkable so isinstance() works in get_adapter().
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from mlobs.core.types import ColumnArray, ColumnStats


@runtime_checkable
class DataFrameAdapter(Protocol):
    """
    Protocol that every backend adapter must satisfy.

    All methods accept the raw dataframe as their first positional argument
    so adapter instances are stateless and reusable across multiple frames.
    """

    def shape(self, df: Any) -> tuple[int, int]:
        """Return (n_rows, n_cols)."""
        ...

    def column_names(self, df: Any) -> list[str]:
        """Return list of column name strings."""
        ...

    def column_dtype(self, df: Any, col: str) -> str:
        """Return the dtype of *col* as a string."""
        ...

    def is_numeric(self, df: Any, col: str) -> bool:
        """Return True if *col* holds numeric data."""
        ...

    def to_numpy(self, df: Any, col: str) -> ColumnArray:
        """
        Extract *col* as a 1-D numpy array.

        Null values should become np.nan for float columns, or None for
        object arrays, so detectors can strip them consistently.
        """
        ...

    def compute_column_stats(self, df: Any, col: str) -> ColumnStats:
        """Compute and return per-column statistics."""
        ...
