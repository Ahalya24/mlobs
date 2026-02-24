"""
JSON formatter for mlobs output.

Handles numpy scalar types, numpy arrays, float NaN/Inf, and datetime
objects so callers don't need to pre-process their data before serialising.

Design: subclass json.JSONEncoder rather than a third-party library to keep
the dependency count at zero.  NaN → null (not the JS NaN literal, which is
invalid JSON).
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def _safe_float(x: float) -> Any:
    """Convert NaN / Inf to JSON-safe representations."""
    if math.isnan(x):
        return None
    if math.isinf(x):
        return str(x)   # "inf" or "-inf"
    return x


def _sanitise(obj: Any) -> Any:
    """Recursively replace non-JSON-safe floats in nested structures."""
    if isinstance(obj, float):
        return _safe_float(obj)
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    return obj


class _MLObsEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles mlobs / numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return _safe_float(float(obj))
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return super().default(obj)

    def iterencode(self, obj: Any, _one_shot: bool = False) -> Any:
        return super().iterencode(_sanitise(obj), _one_shot)


class JSONFormatter:
    """
    Formats mlobs data structures as JSON strings.

    Parameters
    ----------
    sort_keys : bool
        Whether to sort dict keys in the output.  Default False.
    ensure_ascii : bool
        Whether to escape non-ASCII characters.  Default False.
    """

    def __init__(
        self,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
    ) -> None:
        self.sort_keys = sort_keys
        self.ensure_ascii = ensure_ascii

    def format(self, data: Any, indent: int = 2) -> str:
        """Serialise *data* to a JSON string."""
        return json.dumps(
            data,
            cls=_MLObsEncoder,
            indent=indent,
            sort_keys=self.sort_keys,
            ensure_ascii=self.ensure_ascii,
        )
