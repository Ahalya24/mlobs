"""
DriftReport and ColumnDriftResult — result containers for drift detection.

Both dataclasses support to_dict() / from_dict() for JSON round-tripping
without requiring pydantic.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mlobs.core.types import _utc_now


def _generate_run_id() -> str:
    return str(uuid.uuid4())


@dataclass
class ColumnDriftResult:
    """
    Drift test result for a single column.

    Attributes
    ----------
    column_name    : column being tested
    detector       : algorithm used — "ks", "chi2", "psi", "jsd"
    statistic      : primary test statistic value
    p_value        : p-value where applicable; None for PSI and JSD
    threshold      : the decision threshold applied
    drifted        : True if drift was detected
    reference_size : non-null observations in reference
    current_size   : non-null observations in current
    extra          : optional additional output (e.g. per-bin PSI values)
    """
    column_name: str
    detector: str
    statistic: float
    p_value: Optional[float]
    threshold: float
    drifted: bool
    reference_size: int
    current_size: int
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_name": self.column_name,
            "detector": self.detector,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "drifted": self.drifted,
            "reference_size": self.reference_size,
            "current_size": self.current_size,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ColumnDriftResult":
        return cls(
            column_name=d["column_name"],
            detector=d["detector"],
            statistic=d["statistic"],
            p_value=d.get("p_value"),
            threshold=d["threshold"],
            drifted=d["drifted"],
            reference_size=d["reference_size"],
            current_size=d["current_size"],
            extra=d.get("extra", {}),
        )


@dataclass
class DriftReport:
    """
    Aggregated result of a drift detection run.

    Attributes
    ----------
    reference_name : label for the reference dataset
    current_name   : label for the current dataset
    results        : list of per-column results
    run_id         : unique identifier (uuid4) for this run
    timestamp      : UTC ISO-8601 string
    metadata       : arbitrary caller-supplied key-value pairs
    """
    reference_name: str
    current_name: str
    results: List[ColumnDriftResult]
    run_id: str = field(default_factory=_generate_run_id)
    timestamp: str = field(default_factory=_utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def summary(self) -> Dict[str, int]:
        total = len(self.results)
        drifted = sum(1 for r in self.results if r.drifted)
        return {
            "total_columns": total,
            "drifted": drifted,
            "not_drifted": total - drifted,
        }

    @property
    def drifted_columns(self) -> List[str]:
        return [r.column_name for r in self.results if r.drifted]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "reference_name": self.reference_name,
            "current_name": self.current_name,
            "summary": self.summary,
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DriftReport":
        return cls(
            run_id=d["run_id"],
            timestamp=d["timestamp"],
            reference_name=d["reference_name"],
            current_name=d["current_name"],
            metadata=d.get("metadata", {}),
            results=[ColumnDriftResult.from_dict(r) for r in d["results"]],
        )
