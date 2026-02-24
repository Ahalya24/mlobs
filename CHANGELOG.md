# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-24

### Added
- `NumericDriftDetector` — Kolmogorov-Smirnov two-sample test for numeric columns
- `CategoricalDriftDetector` — Pearson chi-squared test for categorical columns
- `PSIDriftDetector` — Population Stability Index with quantile-based binning
- `JSDriftDetector` — Jensen-Shannon Divergence detector
- `DriftReport` and `ColumnDriftResult` dataclasses with `to_dict` / `from_dict`
  serialisation
- `PipelineLogger` — records per-column feature statistics at named pipeline steps
- `JSONFormatter` — handles numpy scalars, arrays, NaN → null
- Backend adapters for **pandas**, **polars** (DataFrame + LazyFrame), and
  **pyarrow** (Table)
- `get_adapter()` factory with lazy imports for each backend
- `DataFrameAdapter` Protocol (PEP 544, runtime-checkable)
- PEP 561 `py.typed` marker for inline type annotations
- GitHub Actions CI matrix (Python 3.9–3.12) and PyPI publish workflow

[Unreleased]: https://github.com/your-org/mlobs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/mlobs/releases/tag/v0.1.0
