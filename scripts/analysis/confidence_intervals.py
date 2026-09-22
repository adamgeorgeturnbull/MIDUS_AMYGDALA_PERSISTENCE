"""Two-sided 95% effect-estimate CIs, obtained during model fitting.

No file I/O, data loading, fitting, or conversion of directional p values.
Coefficients and mean differences retain their original (unstandardized) units.
Invalid/nonconverged fits produce missing bounds with an explicit status;
in particular, negative variances are never repaired with abs().
"""

import numpy as np
from scipy import stats


def _fields(low, high, method, status="ok", prefix=""):
    stem = prefix + "_" if prefix else ""
    return {stem + "ci_low": float(low), stem + "ci_high": float(high),
            stem + "ci_level": 95.0, stem + "ci_sidedness": "two-sided",
            stem + "ci_method": method, stem + "ci_status": status}


def coefficient_ci(result, term, prefix=""):
    """Use the fitted result's t (OLS) or normal-Wald (MixedLM) interval.

    Check the original covariance and standard error, not any fallback SE
    previously used in a legacy exporter. Preserve existing estimates/p values.
    """
    method = "student_t" if result.use_t else "wald_normal"
    if not bool(getattr(result, "converged", True)):
        return _fields(np.nan, np.nan, method, "nonconverged", prefix)
    try:
        beta = float(result.params[term])
        se = float(result.bse[term])
        variance = float(result.cov_params().loc[term, term])
        if not np.all(np.isfinite([beta, se, variance])) or se <= 0 or variance <= 0:
            return _fields(np.nan, np.nan, method, "invalid_variance", prefix)
        low, high = result.conf_int(alpha=0.05).loc[term]
        if not np.all(np.isfinite([low, high])) or not low <= beta <= high:
            return _fields(np.nan, np.nan, method, "invalid_interval", prefix)
    except (KeyError, ValueError, TypeError, AttributeError, IndexError):
        return _fields(np.nan, np.nan, method, "unavailable", prefix)
    return _fields(low, high, method, prefix=prefix)


def pearson_ci(result):
    """Fisher-z interval from the unrounded scipy PearsonRResult (iid pairs)."""
    if not np.isfinite(result.statistic):
        return _fields(np.nan, np.nan, "fisher_z", "undefined_correlation")
    ci = result.confidence_interval(confidence_level=0.95)
    return _fields(ci.low, ci.high, "fisher_z")


def mean_ci(values):
    """Student-t CI for a mean, or for the mean of paired differences.

    Call only on the exact complete-case values used by the corresponding
    t-test. These raw mean/difference intervals are not standardized d CIs.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)):
        return _fields(np.nan, np.nan, "student_t", "invalid_sample")
    se = stats.sem(x)
    if not np.isfinite(se) or se <= 0:
        return _fields(np.nan, np.nan, "student_t", "invalid_variance")
    low, high = stats.t.interval(0.95, df=len(x) - 1, loc=x.mean(), scale=se)
    return _fields(low, high, "student_t")
