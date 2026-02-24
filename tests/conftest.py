"""
Shared pytest fixtures for mlobs tests.

Fixtures provide reference and current dataframes for each backend
with known drift patterns:
  - numeric_drifted: current mean shifted by +1000 (obvious drift)
  - numeric_stable: current drawn from same distribution (no drift)
  - categorical_drifted: current has completely different categories
  - categorical_stable: current has similar category proportions

Backends are conditionally skipped if the library is not installed so the
test suite degrades gracefully in minimal environments.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Raw data arrays
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

NUMERIC_REF = RNG.normal(loc=0.0, scale=1.0, size=300)
NUMERIC_STABLE = np.random.default_rng(7).normal(loc=0.0, scale=1.0, size=300)
NUMERIC_DRIFTED = np.random.default_rng(99).normal(loc=50.0, scale=1.0, size=300)

CATEGORICAL_REF = np.array(["A"] * 120 + ["B"] * 100 + ["C"] * 80)
CATEGORICAL_STABLE = np.array(["A"] * 115 + ["B"] * 105 + ["C"] * 80)
CATEGORICAL_DRIFTED = np.array(["D"] * 150 + ["E"] * 150)


# ---------------------------------------------------------------------------
# pandas fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pandas_ref_df():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {"age": NUMERIC_REF, "score": NUMERIC_REF * 2.0, "category": CATEGORICAL_REF}
    )


@pytest.fixture()
def pandas_stable_df():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {"age": NUMERIC_STABLE, "score": NUMERIC_STABLE * 2.0,
         "category": CATEGORICAL_STABLE}
    )


@pytest.fixture()
def pandas_drifted_df():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {"age": NUMERIC_DRIFTED, "score": NUMERIC_DRIFTED * 2.0,
         "category": CATEGORICAL_DRIFTED}
    )


# ---------------------------------------------------------------------------
# polars fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def polars_ref_df():
    pl = pytest.importorskip("polars")
    return pl.DataFrame(
        {"age": NUMERIC_REF.tolist(), "score": (NUMERIC_REF * 2.0).tolist(),
         "category": CATEGORICAL_REF.tolist()}
    )


@pytest.fixture()
def polars_stable_df():
    pl = pytest.importorskip("polars")
    return pl.DataFrame(
        {"age": NUMERIC_STABLE.tolist(), "score": (NUMERIC_STABLE * 2.0).tolist(),
         "category": CATEGORICAL_STABLE.tolist()}
    )


@pytest.fixture()
def polars_drifted_df():
    pl = pytest.importorskip("polars")
    return pl.DataFrame(
        {"age": NUMERIC_DRIFTED.tolist(), "score": (NUMERIC_DRIFTED * 2.0).tolist(),
         "category": CATEGORICAL_DRIFTED.tolist()}
    )


# ---------------------------------------------------------------------------
# pyarrow fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def arrow_ref_table():
    pa = pytest.importorskip("pyarrow")
    return pa.table(
        {"age": NUMERIC_REF.tolist(), "score": (NUMERIC_REF * 2.0).tolist(),
         "category": CATEGORICAL_REF.tolist()}
    )


@pytest.fixture()
def arrow_stable_table():
    pa = pytest.importorskip("pyarrow")
    return pa.table(
        {"age": NUMERIC_STABLE.tolist(), "score": (NUMERIC_STABLE * 2.0).tolist(),
         "category": CATEGORICAL_STABLE.tolist()}
    )


@pytest.fixture()
def arrow_drifted_table():
    pa = pytest.importorskip("pyarrow")
    return pa.table(
        {"age": NUMERIC_DRIFTED.tolist(), "score": (NUMERIC_DRIFTED * 2.0).tolist(),
         "category": CATEGORICAL_DRIFTED.tolist()}
    )
