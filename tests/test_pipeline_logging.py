"""
Tests for PipelineLogger and JSONFormatter.

Most tests use pandas (conditionally skipped if not installed).
JSONFormatter tests use plain Python dicts and numpy arrays — no backend needed.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np
import pytest

from mlobs.logging.formatters import JSONFormatter
from mlobs.logging.pipeline import PipelineLogger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age": rng.integers(18, 65, size=100).astype(float),
        "score": rng.normal(0, 1, size=100),
        "category": np.random.default_rng(1).choice(["A", "B", "C"], size=100),
    })


# ---------------------------------------------------------------------------
# PipelineLogger
# ---------------------------------------------------------------------------

class TestPipelineLogger:

    def test_log_step_records_step(self, sample_df) -> None:
        logger = PipelineLogger(name="test")
        record = logger.log_step(sample_df, step_name="step_1")
        assert len(logger.records) == 1
        assert logger.records[0].step_name == "step_1"
        assert record.step_name == "step_1"

    def test_shape_captured_correctly(self, sample_df) -> None:
        logger = PipelineLogger()
        record = logger.log_step(sample_df, step_name="raw")
        assert record.shape == (100, 3)

    def test_column_subset_selection(self, sample_df) -> None:
        logger = PipelineLogger()
        record = logger.log_step(sample_df, step_name="s", columns=["age"])
        assert "age" in record.columns
        assert "score" not in record.columns
        assert "category" not in record.columns

    def test_unknown_column_raises_key_error(self, sample_df) -> None:
        logger = PipelineLogger()
        with pytest.raises(KeyError, match="nonexistent"):
            logger.log_step(sample_df, step_name="s", columns=["nonexistent"])

    def test_metadata_stored(self, sample_df) -> None:
        logger = PipelineLogger()
        logger.log_step(sample_df, step_name="s", metadata={"model_version": "v1"})
        assert logger.records[0].metadata["model_version"] == "v1"

    def test_multiple_steps_accumulate(self, sample_df) -> None:
        logger = PipelineLogger()
        logger.log_step(sample_df, step_name="step_1")
        logger.log_step(sample_df, step_name="step_2")
        assert len(logger.records) == 2
        assert logger.records[0].step_name == "step_1"
        assert logger.records[1].step_name == "step_2"

    def test_to_dict_structure(self, sample_df) -> None:
        logger = PipelineLogger(name="mylog")
        logger.log_step(sample_df, step_name="raw")
        d = logger.to_dict()
        assert d["pipeline_name"] == "mylog"
        assert len(d["steps"]) == 1
        step = d["steps"][0]
        for key in ("step_name", "timestamp", "shape", "columns", "metadata"):
            assert key in step

    def test_to_json_is_valid(self, sample_df) -> None:
        logger = PipelineLogger()
        logger.log_step(sample_df, step_name="raw")
        json_str = logger.to_json()
        parsed = json.loads(json_str)
        assert "steps" in parsed
        assert len(parsed["steps"]) == 1

    def test_numeric_column_stats_keys(self, sample_df) -> None:
        logger = PipelineLogger()
        logger.log_step(sample_df, step_name="raw")
        d = logger.to_dict()
        age_col = d["steps"][0]["columns"]["age"]
        assert age_col["kind"] == "numeric"
        for key in ("mean", "std", "min", "max", "q25", "q50", "q75",
                    "count", "null_count"):
            assert key in age_col["stats"], f"Missing key: {key}"

    def test_categorical_column_stats_keys(self, sample_df) -> None:
        logger = PipelineLogger()
        logger.log_step(sample_df, step_name="raw")
        d = logger.to_dict()
        cat_col = d["steps"][0]["columns"]["category"]
        assert cat_col["kind"] == "categorical"
        for key in ("count", "null_count", "n_unique", "value_counts"):
            assert key in cat_col["stats"]

    def test_context_manager_usage(self, sample_df) -> None:
        with PipelineLogger(name="ctx") as logger:
            logger.log_step(sample_df, step_name="inside")
        assert len(logger.records) == 1

    def test_clear_empties_records(self, sample_df) -> None:
        logger = PipelineLogger()
        logger.log_step(sample_df, step_name="s")
        assert len(logger.records) == 1
        logger.clear()
        assert len(logger.records) == 0

    def test_dump_writes_json_file(self, sample_df) -> None:
        logger = PipelineLogger()
        logger.log_step(sample_df, step_name="raw")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            logger.dump(path)
            content = pathlib.Path(path).read_text(encoding="utf-8")
            parsed = json.loads(content)
            assert "steps" in parsed
        finally:
            pathlib.Path(path).unlink(missing_ok=True)

    def test_records_is_snapshot_copy(self, sample_df) -> None:
        logger = PipelineLogger()
        logger.log_step(sample_df, step_name="s")
        snap1 = logger.records
        logger.log_step(sample_df, step_name="s2")
        # snap1 should not grow
        assert len(snap1) == 1

    def test_unsupported_backend_raises_type_error(self) -> None:
        logger = PipelineLogger()
        with pytest.raises(TypeError, match="No mlobs adapter"):
            logger.log_step({"a": [1, 2, 3]}, step_name="s")


# ---------------------------------------------------------------------------
# JSONFormatter
# ---------------------------------------------------------------------------

class TestJSONFormatter:

    def test_nan_becomes_null(self) -> None:
        fmt = JSONFormatter()
        result = fmt.format({"val": float("nan")})
        parsed = json.loads(result)
        assert parsed["val"] is None

    def test_inf_becomes_string(self) -> None:
        fmt = JSONFormatter()
        result = fmt.format({"val": float("inf")})
        parsed = json.loads(result)
        assert parsed["val"] == "inf"

    def test_numpy_float_serialised(self) -> None:
        fmt = JSONFormatter()
        result = fmt.format({"val": np.float64(3.14)})
        parsed = json.loads(result)
        assert abs(parsed["val"] - 3.14) < 1e-6

    def test_numpy_int_serialised(self) -> None:
        fmt = JSONFormatter()
        result = fmt.format({"val": np.int32(7)})
        parsed = json.loads(result)
        assert parsed["val"] == 7

    def test_numpy_array_serialised(self) -> None:
        fmt = JSONFormatter()
        result = fmt.format({"arr": np.array([1, 2, 3])})
        parsed = json.loads(result)
        assert parsed["arr"] == [1, 2, 3]

    def test_nested_nan_in_list(self) -> None:
        fmt = JSONFormatter()
        result = fmt.format({"vals": [1.0, float("nan"), 3.0]})
        parsed = json.loads(result)
        assert parsed["vals"][1] is None

    def test_sort_keys(self) -> None:
        fmt = JSONFormatter(sort_keys=True)
        result = fmt.format({"b": 1, "a": 2})
        # Keys should appear in sorted order
        assert result.index('"a"') < result.index('"b"')

    def test_indent(self) -> None:
        fmt = JSONFormatter()
        result = fmt.format({"key": "val"}, indent=4)
        # Indented JSON contains newlines
        assert "\n" in result
