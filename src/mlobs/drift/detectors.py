"""
Drift detectors.

Each detector follows the same interface:
    detector.detect(reference: ColumnArray, current: ColumnArray,
                    column_name: str = "unknown") -> ColumnDriftResult

All detectors accept raw 1-D numpy arrays (null values stripped internally).
Detector classes are stateless — parameters are set at construction, no data
is stored on the instance.

Detectors
---------
NumericDriftDetector     — KS two-sample test (scipy.stats.ks_2samp)
CategoricalDriftDetector — Pearson chi-squared test (scipy.stats.chi2_contingency)
PSIDriftDetector         — Population Stability Index (pure numpy, quantile bins)
JSDriftDetector          — Jensen-Shannon Divergence (scipy.spatial.distance)
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

from mlobs.core.types import ColumnArray
from mlobs.drift.report import ColumnDriftResult

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _strip_nulls(arr: ColumnArray) -> ColumnArray:
    """Return a copy of *arr* with NaN / None removed."""
    if arr.dtype.kind in ("f", "c"):  # float / complex
        return arr[~np.isnan(arr)]
    # Object dtype (strings, etc.) — treat None and np.nan as null
    mask = np.array(
        [x is not None and not (isinstance(x, float) and np.isnan(x)) for x in arr],
        dtype=bool,
    )
    return arr[mask]


def _align_categories(
    ref: ColumnArray, cur: ColumnArray
) -> tuple:  # -> Tuple[np.ndarray, np.ndarray]
    """
    Build frequency arrays aligned over the union of categories in ref and cur.
    Returns (ref_counts, cur_counts) as int64 arrays in the same category order.
    """
    categories = sorted(set(ref.tolist()) | set(cur.tolist()), key=str)
    cat_index = {c: i for i, c in enumerate(categories)}
    n = len(categories)

    ref_counts = np.zeros(n, dtype=np.int64)
    cur_counts = np.zeros(n, dtype=np.int64)
    for v in ref:
        ref_counts[cat_index[v]] += 1
    for v in cur:
        cur_counts[cat_index[v]] += 1
    return ref_counts, cur_counts


def _empty_result(
    column_name: str, detector: str, threshold: float,
    ref_size: int, cur_size: int,
) -> ColumnDriftResult:
    """Return a ColumnDriftResult indicating an empty-array error."""
    return ColumnDriftResult(
        column_name=column_name,
        detector=detector,
        statistic=float("nan"),
        p_value=None,
        threshold=threshold,
        drifted=True,
        reference_size=ref_size,
        current_size=cur_size,
        extra={"warning": "empty array after null removal"},
    )


# ---------------------------------------------------------------------------
# Numeric: KS test
# ---------------------------------------------------------------------------

class NumericDriftDetector:
    """
    Kolmogorov-Smirnov two-sample test for continuous (numeric) features.

    Parameters
    ----------
    p_value_threshold : float
        p-value below which drift is declared.  Default 0.05.
    alternative : str
        Passed to scipy.stats.ks_2samp.  One of 'two-sided' (default),
        'less', 'greater'.
    """

    def __init__(
        self,
        p_value_threshold: float = 0.05,
        alternative: str = "two-sided",
    ) -> None:
        if not (0.0 < p_value_threshold < 1.0):
            raise ValueError("p_value_threshold must be in (0, 1)")
        self.p_value_threshold = p_value_threshold
        self.alternative = alternative

    def detect(
        self,
        reference: ColumnArray,
        current: ColumnArray,
        column_name: str = "unknown",
    ) -> ColumnDriftResult:
        ref_clean = _strip_nulls(reference.astype(np.float64))
        cur_clean = _strip_nulls(current.astype(np.float64))

        if len(ref_clean) == 0 or len(cur_clean) == 0:
            return _empty_result("ks", "ks", self.p_value_threshold,
                                 len(ref_clean), len(cur_clean))

        result = stats.ks_2samp(ref_clean, cur_clean, alternative=self.alternative)
        return ColumnDriftResult(
            column_name=column_name,
            detector="ks",
            statistic=float(result.statistic),
            p_value=float(result.pvalue),
            threshold=self.p_value_threshold,
            drifted=bool(result.pvalue < self.p_value_threshold),
            reference_size=len(ref_clean),
            current_size=len(cur_clean),
        )


# ---------------------------------------------------------------------------
# Categorical: chi-squared test
# ---------------------------------------------------------------------------

class CategoricalDriftDetector:
    """
    Pearson chi-squared test for categorical features.

    Low-expected-count cells are merged into an '_other_' bucket before
    testing (standard statistical practice: expected count < 5).

    Parameters
    ----------
    p_value_threshold : float
        p-value below which drift is declared.  Default 0.05.
    min_expected_count : float
        Cells with expected count below this are merged.  Default 5.
    """

    def __init__(
        self,
        p_value_threshold: float = 0.05,
        min_expected_count: float = 5.0,
    ) -> None:
        self.p_value_threshold = p_value_threshold
        self.min_expected_count = min_expected_count

    def detect(
        self,
        reference: ColumnArray,
        current: ColumnArray,
        column_name: str = "unknown",
    ) -> ColumnDriftResult:
        ref_clean = _strip_nulls(reference)
        cur_clean = _strip_nulls(current)

        if len(ref_clean) == 0 or len(cur_clean) == 0:
            return _empty_result(column_name, "chi2", self.p_value_threshold,
                                 len(ref_clean), len(cur_clean))

        ref_counts, cur_counts = _align_categories(ref_clean, cur_clean)
        n_ref = int(ref_counts.sum())
        n_cur = int(cur_counts.sum())

        # Expected counts for current = cur_total × (ref proportion)
        ref_proportions = ref_counts / n_ref
        expected_cur = ref_proportions * n_cur

        keep_mask = expected_cur >= self.min_expected_count
        if keep_mask.sum() < 2:
            keep_mask = np.ones(len(ref_counts), dtype=bool)

        ref_agg = ref_counts[keep_mask].copy()
        cur_agg = cur_counts[keep_mask].copy()

        if not keep_mask.all():
            ref_agg = np.append(ref_agg, ref_counts[~keep_mask].sum())
            cur_agg = np.append(cur_agg, cur_counts[~keep_mask].sum())

        # chi2_contingency expects a 2×k observed frequency table
        contingency = np.array([ref_agg, cur_agg])
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

        return ColumnDriftResult(
            column_name=column_name,
            detector="chi2",
            statistic=float(chi2),
            p_value=float(p_value),
            threshold=self.p_value_threshold,
            drifted=bool(p_value < self.p_value_threshold),
            reference_size=n_ref,
            current_size=n_cur,
            extra={"degrees_of_freedom": int(dof)},
        )


# ---------------------------------------------------------------------------
# PSI — Population Stability Index
# ---------------------------------------------------------------------------

class PSIDriftDetector:
    """
    Population Stability Index for numeric or categorical features.

    PSI = Σ (current_pct − reference_pct) × ln(current_pct / reference_pct)

    Conventional thresholds:
        PSI < 0.10  → no significant change
        0.10–0.20   → moderate change (monitor)
        PSI ≥ 0.20  → significant shift (drift detected)

    For numeric features binning uses quantiles of the reference distribution
    (equal-probability bins) to avoid artificially high PSI from sparse bins.

    Parameters
    ----------
    threshold : float
        PSI value at or above which drift is declared.  Default 0.20.
    n_bins : int
        Number of bins for numeric features.  Default 10.
    epsilon : float
        Small constant added to zero-count cells to avoid log(0).  Default 1e-4.
    """

    def __init__(
        self,
        threshold: float = 0.20,
        n_bins: int = 10,
        epsilon: float = 1e-4,
    ) -> None:
        self.threshold = threshold
        self.n_bins = n_bins
        self.epsilon = epsilon

    def detect(
        self,
        reference: ColumnArray,
        current: ColumnArray,
        column_name: str = "unknown",
        is_categorical: bool = False,
    ) -> ColumnDriftResult:
        ref_clean = _strip_nulls(reference)
        cur_clean = _strip_nulls(current)

        if len(ref_clean) == 0 or len(cur_clean) == 0:
            return _empty_result(column_name, "psi", self.threshold,
                                 len(ref_clean), len(cur_clean))

        if is_categorical:
            ref_counts, cur_counts = _align_categories(ref_clean, cur_clean)
        else:
            ref_float = ref_clean.astype(np.float64)
            cur_float = cur_clean.astype(np.float64)
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            bin_edges = np.unique(np.percentile(ref_float, quantiles))
            if len(bin_edges) < 2:
                bin_edges = np.array([ref_float.min(), ref_float.max() + 1e-9])
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            ref_counts, _ = np.histogram(ref_float, bins=bin_edges)
            cur_counts, _ = np.histogram(cur_float, bins=bin_edges)

        n_ref = int(ref_counts.sum())
        n_cur = int(cur_counts.sum())

        ref_pct = (ref_counts.astype(float) / n_ref) + self.epsilon
        cur_pct = (cur_counts.astype(float) / n_cur) + self.epsilon
        # Renormalise so proportions sum to 1 after epsilon shift
        ref_pct /= ref_pct.sum()
        cur_pct /= cur_pct.sum()

        bin_psi = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
        psi_value = float(bin_psi.sum())

        return ColumnDriftResult(
            column_name=column_name,
            detector="psi",
            statistic=psi_value,
            p_value=None,
            threshold=self.threshold,
            drifted=bool(psi_value >= self.threshold),
            reference_size=n_ref,
            current_size=n_cur,
            extra={"bin_psi": bin_psi.tolist()},
        )


# ---------------------------------------------------------------------------
# Jensen-Shannon Divergence
# ---------------------------------------------------------------------------

class JSDriftDetector:
    """
    Jensen-Shannon Divergence detector.

    JSD is symmetric and bounded in [0, 1] (base-2 log).  Suitable for
    both numeric (binned) and categorical inputs.

    Parameters
    ----------
    threshold : float
        JSD at or above which drift is declared.  Default 0.1.
    n_bins : int
        Bins for numeric features (quantile-based from reference).  Default 10.
    epsilon : float
        Smoothing constant added before computing divergence.  Default 1e-4.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        n_bins: int = 10,
        epsilon: float = 1e-4,
    ) -> None:
        self.threshold = threshold
        self.n_bins = n_bins
        self.epsilon = epsilon

    def detect(
        self,
        reference: ColumnArray,
        current: ColumnArray,
        column_name: str = "unknown",
        is_categorical: bool = False,
    ) -> ColumnDriftResult:
        ref_clean = _strip_nulls(reference)
        cur_clean = _strip_nulls(current)

        if len(ref_clean) == 0 or len(cur_clean) == 0:
            return _empty_result(column_name, "jsd", self.threshold,
                                 len(ref_clean), len(cur_clean))

        if is_categorical:
            ref_counts, cur_counts = _align_categories(ref_clean, cur_clean)
        else:
            ref_float = ref_clean.astype(np.float64)
            cur_float = cur_clean.astype(np.float64)
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            bin_edges = np.unique(np.percentile(ref_float, quantiles))
            if len(bin_edges) < 2:
                bin_edges = np.array([ref_float.min(), ref_float.max() + 1e-9])
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            ref_counts, _ = np.histogram(ref_float, bins=bin_edges)
            cur_counts, _ = np.histogram(cur_float, bins=bin_edges)

        ref_pct = ref_counts.astype(float) + self.epsilon
        cur_pct = cur_counts.astype(float) + self.epsilon
        ref_pct /= ref_pct.sum()
        cur_pct /= cur_pct.sum()

        jsd_value = float(jensenshannon(ref_pct, cur_pct, base=2))

        return ColumnDriftResult(
            column_name=column_name,
            detector="jsd",
            statistic=jsd_value,
            p_value=None,
            threshold=self.threshold,
            drifted=bool(jsd_value >= self.threshold),
            reference_size=int(ref_counts.sum()),
            current_size=int(cur_counts.sum()),
        )
