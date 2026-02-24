"""
Shared dataclasses used across the mlobs package.

All types here are backend-agnostic.  Adapters produce ColumnStats; detectors
consume ColumnArray (plain numpy arrays).  No backend libraries are imported
in this module.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# A single extracted column as a 1-D numpy array.
ColumnArray = np.ndarray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Per-column statistics
# ---------------------------------------------------------------------------

@dataclass
class NumericStats:
    """Descriptive statistics for a numeric column."""
    count: int
    null_count: int
    mean: float
    std: float
    min: float
    q25: float
    q50: float
    q75: float
    max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "null_count": self.null_count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "q25": self.q25,
            "q50": self.q50,
            "q75": self.q75,
            "max": self.max,
        }


@dataclass
class CategoricalStats:
    """Frequency statistics for a categorical column."""
    count: int
    null_count: int
    n_unique: int
    # Maps category label (as str) to integer frequency count.
    value_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "null_count": self.null_count,
            "n_unique": self.n_unique,
            "value_counts": self.value_counts,
        }


# Keep Union[] notation for Python 3.9 runtime compatibility.
ColumnStatsPayload = Union[NumericStats, CategoricalStats]


@dataclass
class ColumnStats:
    """
    Wraps NumericStats or CategoricalStats with column name and dtype so
    callers don't need isinstance checks.
    """
    name: str
    dtype: str           # e.g. "float64", "int32", "object", "Utf8"
    kind: str            # "numeric" or "categorical"
    stats: ColumnStatsPayload

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "kind": self.kind,
            "stats": self.stats.to_dict(),
        }


# ---------------------------------------------------------------------------
# Pipeline step record
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """
    Snapshot of feature statistics at a named pipeline step.

    Attributes
    ----------
    step_name : human-readable label, e.g. "after_imputation"
    shape     : (n_rows, n_cols) of the observed dataframe
    columns   : per-column statistics keyed by column name
    timestamp : UTC ISO-8601 string, set automatically
    metadata  : arbitrary caller-supplied key-value pairs
    """
    step_name: str
    shape: tuple[int, int]
    columns: dict[str, ColumnStats]
    timestamp: str = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "timestamp": self.timestamp,
            "shape": list(self.shape),
            "metadata": self.metadata,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
        }
