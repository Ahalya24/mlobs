"""
Tests for drift detectors and DriftReport.

Detector unit tests work directly with numpy arrays — no backend required.
End-to-end tests use pandas (conditionally skipped if not installed).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from mlobs.drift.detectors import (
    CategoricalDriftDetector,
    JSDriftDetector,
    NumericDriftDetector,
    PSIDriftDetector,
)
from mlobs.drift.report import ColumnDriftResult, DriftReport


# ---------------------------------------------------------------------------
# Fixtures (pure numpy — no backend needed)
# ---------------------------------------------------------------------------

@pytest.fixture()
def numeric_ref() -> np.ndarray:
    return np.random.default_rng(42).normal(loc=0.0, scale=1.0, size=500)


@pytest.fixture()
def numeric_stable(numeric_ref: np.ndarray) -> np.ndarray:
    return np.random.default_rng(7).normal(loc=0.0, scale=1.0, size=500)


@pytest.fixture()
def numeric_drifted() -> np.ndarray:
    return np.random.default_rng(99).normal(loc=20.0, scale=1.0, size=500)


@pytest.fixture()
def cat_ref() -> np.ndarray:
    return np.array(["A"] * 200 + ["B"] * 150 + ["C"] * 150)


@pytest.fixture()
def cat_stable() -> np.ndarray:
    return np.array(["A"] * 195 + ["B"] * 155 + ["C"] * 150)


@pytest.fixture()
def cat_drifted() -> np.ndarray:
    return np.array(["D"] * 300 + ["E"] * 200)


# ---------------------------------------------------------------------------
# NumericDriftDetector
# ---------------------------------------------------------------------------

class TestNumericDriftDetector:

    def test_stable_not_flagged(
        self, numeric_ref: np.ndarray, numeric_stable: np.ndarray
    ) -> None:
        # Use a conservative threshold (0.001) so the KS test is very unlikely
        # to produce a false positive for two same-distribution samples.
        det = NumericDriftDetector(p_value_threshold=0.001)
        result = det.detect(numeric_ref, numeric_stable, column_name="val")
        assert result.detector == "ks"
        assert result.drifted is False
        assert result.p_value is not None
        assert result.p_value >= 0.001

    def test_drifted_flagged(
        self, numeric_ref: np.ndarray, numeric_drifted: np.ndarray
    ) -> None:
        det = NumericDriftDetector(p_value_threshold=0.05)
        result = det.detect(numeric_ref, numeric_drifted, column_name="val")
        assert result.drifted is True
        assert result.p_value is not None
        assert result.p_value < 0.05

    def test_sizes_correct(
        self, numeric_ref: np.ndarray, numeric_stable: np.ndarray
    ) -> None:
        det = NumericDriftDetector()
        result = det.detect(numeric_ref, numeric_stable, column_name="val")
        assert result.reference_size == len(numeric_ref)
        assert result.current_size == len(numeric_stable)

    def test_nans_stripped(
        self, numeric_ref: np.ndarray, numeric_stable: np.ndarray
    ) -> None:
        ref_with_nans = np.concatenate([numeric_ref, [np.nan, np.nan, np.nan]])
        det = NumericDriftDetector()
        result = det.detect(ref_with_nans, numeric_stable, column_name="val")
        assert result.reference_size == len(numeric_ref)  # NaNs stripped

    def test_empty_current_returns_warning(
        self, numeric_ref: np.ndarray
    ) -> None:
        det = NumericDriftDetector()
        result = det.detect(numeric_ref, np.array([]), column_name="val")
        assert result.drifted is True
        assert "warning" in result.extra
        assert math.isnan(result.statistic)

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="p_value_threshold"):
            NumericDriftDetector(p_value_threshold=1.5)

    def test_column_name_in_result(
        self, numeric_ref: np.ndarray, numeric_stable: np.ndarray
    ) -> None:
        det = NumericDriftDetector()
        result = det.detect(numeric_ref, numeric_stable, column_name="my_feature")
        assert result.column_name == "my_feature"

    def test_threshold_stored_in_result(
        self, numeric_ref: np.ndarray, numeric_stable: np.ndarray
    ) -> None:
        det = NumericDriftDetector(p_value_threshold=0.01)
        result = det.detect(numeric_ref, numeric_stable, column_name="v")
        assert result.threshold == 0.01


# ---------------------------------------------------------------------------
# CategoricalDriftDetector
# ---------------------------------------------------------------------------

class TestCategoricalDriftDetector:

    def test_stable_not_flagged(
        self, cat_ref: np.ndarray, cat_stable: np.ndarray
    ) -> None:
        det = CategoricalDriftDetector(p_value_threshold=0.05)
        result = det.detect(cat_ref, cat_stable, column_name="cat")
        assert result.detector == "chi2"
        assert result.drifted is False

    def test_drifted_flagged(
        self, cat_ref: np.ndarray, cat_drifted: np.ndarray
    ) -> None:
        det = CategoricalDriftDetector(p_value_threshold=0.05)
        result = det.detect(cat_ref, cat_drifted, column_name="cat")
        assert result.drifted is True
        assert result.p_value is not None
        assert result.p_value < 0.05

    def test_degrees_of_freedom_in_extra(
        self, cat_ref: np.ndarray, cat_stable: np.ndarray
    ) -> None:
        det = CategoricalDriftDetector()
        result = det.detect(cat_ref, cat_stable, column_name="cat")
        assert "degrees_of_freedom" in result.extra
        assert isinstance(result.extra["degrees_of_freedom"], int)

    def test_sizes_correct(
        self, cat_ref: np.ndarray, cat_stable: np.ndarray
    ) -> None:
        det = CategoricalDriftDetector()
        result = det.detect(cat_ref, cat_stable, column_name="cat")
        assert result.reference_size == len(cat_ref)
        assert result.current_size == len(cat_stable)

    def test_empty_current_returns_warning(
        self, cat_ref: np.ndarray
    ) -> None:
        det = CategoricalDriftDetector()
        result = det.detect(cat_ref, np.array([]), column_name="cat")
        assert result.drifted is True
        assert "warning" in result.extra


# ---------------------------------------------------------------------------
# PSIDriftDetector
# ---------------------------------------------------------------------------

class TestPSIDriftDetector:

    def test_stable_not_flagged(
        self, numeric_ref: np.ndarray, numeric_stable: np.ndarray
    ) -> None:
        det = PSIDriftDetector(threshold=0.20)
        result = det.detect(numeric_ref, numeric_stable, column_name="val")
        assert result.detector == "psi"
        assert result.drifted is False

    def test_drifted_flagged(
        self, numeric_ref: np.ndarray, numeric_drifted: np.ndarray
    ) -> None:
        det = PSIDriftDetector(threshold=0.20)
        result = det.detect(numeric_ref, numeric_drifted, column_name="val")
        assert result.drifted is True

    def test_p_value_is_none(
        self, numeric_ref: np.ndarray, numeric_stable: np.ndarray
    ) -> None:
        det = PSIDriftDetector()
        result = det.detect(numeric_ref, numeric_stable, column_name="val")
        assert result.p_value is None

    def test_bin_psi_in_extra(
        self, numeric_ref: np.ndarray, numeric_stable: np.ndarray
    ) -> None:
        det = PSIDriftDetector()
        result = det.detect(numeric_ref, numeric_stable, column_name="val")
        assert "bin_psi" in result.extra
        assert isinstance(result.extra["bin_psi"], list)

    def test_categorical_mode_drifted(
        self, cat_ref: np.ndarray, cat_drifted: np.ndarray
    ) -> None:
        det = PSIDriftDetector(threshold=0.20)
        result = det.detect(
            cat_ref, cat_drifted, column_name="cat", is_categorical=True
        )
        assert result.drifted is True

    def test_categorical_mode_stable(
        self, cat_ref: np.ndarray, cat_stable: np.ndarray
    ) -> None:
        det = PSIDriftDetector(threshold=0.20)
        result = det.detect(
            cat_ref, cat_stable, column_name="cat", is_categorical=True
        )
        assert result.drifted is False

    def test_empty_returns_warning(self, numeric_ref: np.ndarray) -> None:
        det = PSIDriftDetector()
        result = det.detect(numeric_ref, np.array([]), column_name="val")
        assert result.drifted is True
        assert "warning" in result.extra


# ---------------------------------------------------------------------------
# JSDriftDetector
# ---------------------------------------------------------------------------

class TestJSDriftDetector:

    def test_identical_arrays_near_zero(
        self, numeric_ref: np.ndarray
    ) -> None:
        det = JSDriftDetector(threshold=0.1)
        result = det.detect(numeric_ref, numeric_ref.copy(), column_name="val")
        assert result.statistic < 0.05

    def test_drifted_flagged(
        self, numeric_ref: np.ndarray, numeric_drifted: np.ndarray
    ) -> None:
        det = JSDriftDetector(threshold=0.1)
        result = det.detect(numeric_ref, numeric_drifted, column_name="val")
        assert result.drifted is True

    def test_p_value_is_none(
        self, numeric_ref: np.ndarray, numeric_stable: np.ndarray
    ) -> None:
        det = JSDriftDetector()
        result = det.detect(numeric_ref, numeric_stable, column_name="val")
        assert result.p_value is None

    def test_categorical_mode(
        self, cat_ref: np.ndarray, cat_drifted: np.ndarray
    ) -> None:
        det = JSDriftDetector(threshold=0.1)
        result = det.detect(
            cat_ref, cat_drifted, column_name="cat", is_categorical=True
        )
        assert result.drifted is True

    def test_statistic_bounded(
        self, numeric_ref: np.ndarray, numeric_drifted: np.ndarray
    ) -> None:
        det = JSDriftDetector()
        result = det.detect(numeric_ref, numeric_drifted, column_name="val")
        assert 0.0 <= result.statistic <= 1.0


# ---------------------------------------------------------------------------
# DriftReport
# ---------------------------------------------------------------------------

class TestDriftReport:

    def _make_results(self) -> list:
        return [
            ColumnDriftResult("a", "ks", 0.8, 0.001, 0.05, True, 100, 100),
            ColumnDriftResult("b", "ks", 0.02, 0.45, 0.05, False, 100, 100),
            ColumnDriftResult("c", "chi2", 25.0, 0.0001, 0.05, True, 100, 100),
        ]

    def test_summary_counts(self) -> None:
        report = DriftReport("train", "prod", self._make_results())
        s = report.summary
        assert s["total_columns"] == 3
        assert s["drifted"] == 2
        assert s["not_drifted"] == 1

    def test_drifted_columns(self) -> None:
        report = DriftReport("train", "prod", self._make_results())
        assert report.drifted_columns == ["a", "c"]

    def test_to_dict_keys(self) -> None:
        report = DriftReport("train", "prod", self._make_results())
        d = report.to_dict()
        for key in ("run_id", "timestamp", "reference_name", "current_name",
                    "summary", "metadata", "results"):
            assert key in d

    def test_round_trip(self) -> None:
        results = [
            ColumnDriftResult("x", "psi", 0.25, None, 0.20, True, 500, 500),
        ]
        report = DriftReport("ref", "cur", results)
        d = report.to_dict()
        restored = DriftReport.from_dict(d)
        assert restored.run_id == report.run_id
        assert restored.reference_name == "ref"
        assert restored.results[0].column_name == "x"
        assert restored.results[0].p_value is None

    def test_json_valid(self) -> None:
        report = DriftReport("r", "c", self._make_results())
        d = report.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert "results" in parsed

    def test_empty_report(self) -> None:
        report = DriftReport("r", "c", [])
        assert report.summary["total_columns"] == 0
        assert report.drifted_columns == []

    def test_run_id_unique(self) -> None:
        r1 = DriftReport("r", "c", [])
        r2 = DriftReport("r", "c", [])
        assert r1.run_id != r2.run_id
