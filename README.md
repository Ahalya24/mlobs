# mlobs

**ML observability for structured dataframes.**

`mlobs` provides drift detection and pipeline step logging for pandas, polars,
and PyArrow dataframes — with no dependency on any external ML platform.

[![CI](https://github.com/your-org/mlobs/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/mlobs/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/mlobs.svg)](https://pypi.org/project/mlobs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/pypi/pyversions/mlobs.svg)](https://pypi.org/project/mlobs/)

---

## Installation

```bash
# Core (numpy + scipy — no backend adapters)
pip install mlobs

# With a specific backend
pip install "mlobs[pandas]"
pip install "mlobs[polars]"
pip install "mlobs[arrow]"

# All backends
pip install "mlobs[all]"

# All backends + dev tools (pytest, mypy, ruff)
pip install "mlobs[dev]"
```

---

## Quick Start

### Drift Detection

```python
import pandas as pd
import numpy as np
from mlobs import NumericDriftDetector, CategoricalDriftDetector, DriftReport
from mlobs import get_adapter

# Load reference (training) and current (production) data
ref = pd.DataFrame({
    "age":      np.random.default_rng(0).normal(30, 5, 1000),
    "income":   np.random.default_rng(1).normal(50000, 10000, 1000),
    "segment":  np.random.default_rng(2).choice(["A", "B", "C"], 1000),
})
cur = pd.DataFrame({
    "age":      np.random.default_rng(0).normal(35, 5, 1000),   # shifted
    "income":   np.random.default_rng(1).normal(50000, 10000, 1000),
    "segment":  np.random.default_rng(3).choice(["B", "C", "D"], 1000),  # shifted
})

adapter = get_adapter(ref)

# Test each column with the appropriate detector
ks  = NumericDriftDetector(p_value_threshold=0.05)
chi = CategoricalDriftDetector(p_value_threshold=0.05)

results = [
    ks.detect(adapter.to_numpy(ref, "age"),
              adapter.to_numpy(cur, "age"),
              column_name="age"),
    ks.detect(adapter.to_numpy(ref, "income"),
              adapter.to_numpy(cur, "income"),
              column_name="income"),
    chi.detect(adapter.to_numpy(ref, "segment"),
               adapter.to_numpy(cur, "segment"),
               column_name="segment"),
]

report = DriftReport(
    reference_name="2024-Q1",
    current_name="2024-Q2",
    results=results,
)

print(report.summary)
# {'total_columns': 3, 'drifted': 2, 'not_drifted': 1}

print(report.drifted_columns)
# ['age', 'segment']

print(report.to_json())
```

### Pipeline Logging

```python
import pandas as pd
from mlobs import PipelineLogger

logger = PipelineLogger(name="preprocessing")

raw = pd.read_csv("data.csv")
logger.log_step(raw, step_name="raw_input", metadata={"source": "data.csv"})

cleaned = raw.dropna()
logger.log_step(cleaned, step_name="after_drop_na")

scaled = cleaned.copy()
scaled["age"] = (scaled["age"] - scaled["age"].mean()) / scaled["age"].std()
logger.log_step(scaled, step_name="after_scaling", columns=["age"])

# Persist the log as JSON
logger.dump("pipeline_log.json")

# Or get as a Python dict / JSON string
d = logger.to_dict()
json_str = logger.to_json()
```

### Context Manager

```python
from mlobs import PipelineLogger

with PipelineLogger(name="my_pipeline") as logger:
    logger.log_step(df1, "step_1")
    logger.log_step(df2, "step_2")

print(logger.records)
```

---

## Supported Drift Tests

| Detector | Column type | Test | p-value |
|---|---|---|---|
| `NumericDriftDetector` | numeric | Kolmogorov-Smirnov two-sample | yes |
| `CategoricalDriftDetector` | categorical | Pearson chi-squared | yes |
| `PSIDriftDetector` | numeric / categorical | Population Stability Index | no |
| `JSDriftDetector` | numeric / categorical | Jensen-Shannon Divergence | no |

**PSI thresholds** (conventional):
- PSI < 0.10 → no significant change
- 0.10 ≤ PSI < 0.20 → moderate change
- PSI ≥ 0.20 → significant drift

---

## Multi-Backend Support

The same API works across all supported backends:

```python
import polars as pl
import pyarrow as pa
from mlobs import get_adapter, NumericDriftDetector

# polars
df_pl = pl.read_parquet("data.parquet")
adapter = get_adapter(df_pl)
arr = adapter.to_numpy(df_pl, "age")

# pyarrow
table = pa.ipc.open_file("data.arrow").read_all()
adapter = get_adapter(table)
arr = adapter.to_numpy(table, "age")
```

---

## API Reference

### Drift Detection

```python
from mlobs import (
    NumericDriftDetector,
    CategoricalDriftDetector,
    PSIDriftDetector,
    JSDriftDetector,
    DriftReport,
    ColumnDriftResult,
)
```

**`NumericDriftDetector(p_value_threshold=0.05, alternative="two-sided")`**
- `.detect(reference, current, column_name="unknown") -> ColumnDriftResult`

**`CategoricalDriftDetector(p_value_threshold=0.05, min_expected_count=5.0)`**
- `.detect(reference, current, column_name="unknown") -> ColumnDriftResult`

**`PSIDriftDetector(threshold=0.20, n_bins=10, epsilon=1e-4)`**
- `.detect(reference, current, column_name="unknown", is_categorical=False) -> ColumnDriftResult`

**`JSDriftDetector(threshold=0.1, n_bins=10, epsilon=1e-4)`**
- `.detect(reference, current, column_name="unknown", is_categorical=False) -> ColumnDriftResult`

**`DriftReport`**
- `.summary -> dict` — `{total_columns, drifted, not_drifted}`
- `.drifted_columns -> list[str]`
- `.to_dict() -> dict`
- `.to_json(indent=2) -> str`
- `.from_dict(d) -> DriftReport` (classmethod)

### Pipeline Logging

```python
from mlobs import PipelineLogger, StepRecord, ColumnStats
```

**`PipelineLogger(name="pipeline")`**
- `.log_step(df, step_name, columns=None, metadata=None) -> StepRecord`
- `.records -> list[StepRecord]`
- `.clear()`
- `.to_dict() -> dict`
- `.to_json(indent=2) -> str`
- `.dump(path)`

### Adapters

```python
from mlobs import get_adapter, DataFrameAdapter
```

**`get_adapter(df) -> DataFrameAdapter`** — auto-detects backend

**`DataFrameAdapter`** Protocol methods (all take `df` as first arg):
- `shape(df) -> (n_rows, n_cols)`
- `column_names(df) -> list[str]`
- `is_numeric(df, col) -> bool`
- `to_numpy(df, col) -> np.ndarray`
- `compute_column_stats(df, col) -> ColumnStats`

---

## Contributing

```bash
git clone https://github.com/your-org/mlobs.git
cd mlobs
pip install -e ".[dev]"
pytest
```

Run linting and type checking:

```bash
ruff check src/ tests/
mypy src/mlobs
```

---

## License

MIT — see [LICENSE](LICENSE).
