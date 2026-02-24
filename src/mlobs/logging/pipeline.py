"""
PipelineLogger — records feature statistics at named steps in an ML pipeline.

Design
------
- Each call to log_step() produces a StepRecord (defined in core/types) which
  is appended to an internal list.  No I/O happens during logging.
- Adapters are resolved lazily via get_adapter() so the same logger works
  with pandas, polars, and pyarrow frames without any pre-configuration.
- A threading.Lock guards the records list for safe concurrent access.
- PipelineLogger is also a context manager (enter returns self, exit is a no-op
  that subclasses can override to add flush behaviour).
"""

from __future__ import annotations

import pathlib
import threading
from collections.abc import Sequence
from typing import Any

from mlobs.core.types import ColumnStats, StepRecord
from mlobs.logging.formatters import JSONFormatter


class PipelineLogger:
    """
    Records feature statistics at named pipeline steps.

    Parameters
    ----------
    name : str
        Human-readable name for this logger (included in JSON output).
    formatter : JSONFormatter or None
        Formatter used to serialise records.  Defaults to JSONFormatter().

    Examples
    --------
    >>> import pandas as pd
    >>> from mlobs import PipelineLogger
    >>> logger = PipelineLogger(name="my_pipeline")
    >>> df = pd.DataFrame({"age": [25, 30, 35], "cat": ["A", "B", "A"]})
    >>> logger.log_step(df, step_name="raw_input")
    >>> print(logger.to_json())
    """

    def __init__(
        self,
        name: str = "pipeline",
        formatter: JSONFormatter | None = None,
    ) -> None:
        self.name = name
        self._formatter = formatter or JSONFormatter()
        self._records: list[StepRecord] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def log_step(
        self,
        df: Any,
        step_name: str,
        columns: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StepRecord:
        """
        Compute and record statistics for *df* at pipeline step *step_name*.

        Parameters
        ----------
        df : any supported dataframe type (pandas, polars, pyarrow)
        step_name : str
            Label for this checkpoint.
        columns : sequence of str, optional
            Subset of columns to profile.  Profiles all columns if None.
        metadata : dict, optional
            Arbitrary key-value pairs attached to the StepRecord.

        Returns
        -------
        StepRecord
            The recorded snapshot (also stored internally).

        Raises
        ------
        KeyError
            If a name in *columns* is not present in the dataframe.
        TypeError
            If *df* is not a recognised dataframe type.
        """
        # Lazy import to avoid forcing all backends at import time
        from mlobs.adapters import get_adapter

        adapter = get_adapter(df)
        shape = adapter.shape(df)
        all_cols = adapter.column_names(df)
        selected = list(columns) if columns is not None else all_cols

        for col in selected:
            if col not in all_cols:
                raise KeyError(
                    f"Column {col!r} not found in dataframe "
                    f"(available: {all_cols})"
                )

        col_stats: dict[str, ColumnStats] = {
            col: adapter.compute_column_stats(df, col) for col in selected
        }

        record = StepRecord(
            step_name=step_name,
            shape=shape,
            columns=col_stats,
            metadata=dict(metadata) if metadata else {},
        )

        with self._lock:
            self._records.append(record)

        return record

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def records(self) -> list[StepRecord]:
        """Return a snapshot copy of all recorded StepRecord objects."""
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        """Remove all recorded steps."""
        with self._lock:
            self._records.clear()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full log to a Python dict."""
        return {
            "pipeline_name": self.name,
            "steps": [r.to_dict() for r in self.records],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the full log to a JSON string."""
        return self._formatter.format(self.to_dict(), indent=indent)

    def dump(self, path: str, indent: int = 2) -> None:
        """Write the full log as JSON to *path*."""
        pathlib.Path(path).write_text(self.to_json(indent=indent), encoding="utf-8")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> PipelineLogger:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass  # Subclasses can override to add auto-dump behaviour
