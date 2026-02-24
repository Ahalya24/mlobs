"""
mlobs — ML observability for structured dataframes.

Public API
----------
Drift detection:
    NumericDriftDetector      — KS two-sample test
    CategoricalDriftDetector  — Pearson chi-squared test
    PSIDriftDetector          — Population Stability Index
    JSDriftDetector           — Jensen-Shannon Divergence
    DriftReport               — aggregated result container
    ColumnDriftResult         — per-column result

Pipeline logging:
    PipelineLogger            — records feature stats at named pipeline steps
    StepRecord                — single step snapshot (from core.types)
    ColumnStats               — per-column statistics (from core.types)

Adapters:
    get_adapter               — factory: returns the right adapter for a frame
    DataFrameAdapter          — Protocol that all adapters satisfy
"""

from mlobs._version import __version__
from mlobs.adapters import get_adapter
from mlobs.adapters.base import DataFrameAdapter
from mlobs.core.types import ColumnStats, StepRecord
from mlobs.drift.detectors import (
    CategoricalDriftDetector,
    JSDriftDetector,
    NumericDriftDetector,
    PSIDriftDetector,
)
from mlobs.drift.report import ColumnDriftResult, DriftReport
from mlobs.logging.pipeline import PipelineLogger

__all__ = [
    "__version__",
    # core types
    "ColumnStats",
    "StepRecord",
    # drift
    "NumericDriftDetector",
    "CategoricalDriftDetector",
    "PSIDriftDetector",
    "JSDriftDetector",
    "DriftReport",
    "ColumnDriftResult",
    # logging
    "PipelineLogger",
    # adapters
    "DataFrameAdapter",
    "get_adapter",
]
