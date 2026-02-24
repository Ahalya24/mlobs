"""
Adapter factory — inspects the type of a dataframe and returns the
correct adapter instance.

Imports are lazy so users who have only one backend installed do not
get ImportError from the others.
"""

from __future__ import annotations

from typing import Any

from mlobs.adapters.base import DataFrameAdapter


def get_adapter(df: Any) -> DataFrameAdapter:
    """
    Return the appropriate DataFrameAdapter for *df*.

    Supported types: pandas.DataFrame, polars.DataFrame / LazyFrame,
    pyarrow.Table.

    Raises
    ------
    TypeError
        If no adapter is registered for the type of *df*.
    ImportError
        If the required backend library is not installed.
    """
    module = type(df).__module__.split(".")[0]

    if module == "pandas":
        from mlobs.adapters.pandas_adapter import PandasAdapter
        return PandasAdapter()

    if module == "polars":
        from mlobs.adapters.polars_adapter import PolarsAdapter
        return PolarsAdapter()

    if module == "pyarrow":
        from mlobs.adapters.arrow_adapter import ArrowAdapter
        return ArrowAdapter()

    raise TypeError(
        f"No mlobs adapter for dataframe type {type(df).__name__!r}. "
        "Supported backends: pandas, polars, pyarrow. "
        "Install the required extra, e.g. pip install 'mlobs[pandas]'."
    )
