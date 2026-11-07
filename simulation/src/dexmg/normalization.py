"""Native OpenPI statistics with audited protection against collapsed quantiles."""

import numpy as np


NORMALIZATION_POLICY = "q01_q99_collapsed_range_minmax_v1"
NORMALIZATION_EPSILON = 1e-6


def _validated_arrays(states, actions):
    arrays = {}
    for name, values in (("state", states), ("actions", actions)):
        if np.iscomplexobj(values):
            raise ValueError(f"{name}: expected real values")
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name}: expected a numeric 2D array") from error
        if array.ndim != 2 or 0 in array.shape:
            raise ValueError(f"{name}: expected a nonempty 2D array, got shape {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name}: all values must be finite")
        arrays[name] = array
    if len(arrays["state"]) != len(arrays["actions"]):
        raise ValueError("state and actions must have the same frame count")
    return arrays


def _quantile_ranges(array):
    with np.errstate(over="ignore", invalid="ignore"):
        q01, q99 = np.percentile(array, [1, 99], axis=0)
        minimum, maximum = array.min(axis=0), array.max(axis=0)
        fallback = ((q99 - q01) <= NORMALIZATION_EPSILON) & (
            (maximum - minimum) > NORMALIZATION_EPSILON
        )
    return q01, q99, minimum, maximum, fallback


def compute_norm_stats(states, actions):
    """Return OpenPI's JSON structure from aligned frames of real, finite data.

    The caller selects training episodes and manages provenance, sampling and
    file output. Means and population standard deviations (ddof=0) are exact.
    The 1%/99% bounds use a channel's minimum/maximum only when their range is
    at most 1e-6 while the actual channel range exceeds 1e-6. This protects
    sparse varying channels from normalization by an epsilon-only denominator.
    Constant channels retain zero standard deviation and equal bounds. No
    input values are clipped, padded, transformed or modified.
    """
    arrays = _validated_arrays(states, actions)

    stats = {}
    for name, array in arrays.items():
        q01, q99, minimum, maximum, fallback = _quantile_ranges(array)
        with np.errstate(over="ignore", invalid="ignore"):
            values = {
                "mean": array.mean(axis=0),
                "std": array.std(axis=0, ddof=0),
                "q01": np.where(fallback, minimum, q01),
                "q99": np.where(fallback, maximum, q99),
            }
        if not all(np.isfinite(value).all() for value in values.values()):
            raise ValueError(f"{name}: computed statistics are nonfinite")
        stats[name] = {key: value.tolist() for key, value in values.items()}
    return {"norm_stats": stats}


def normalization_diagnostics(states, actions, stats):
    """Audit fallback channels and actual OpenPI amplification of given assets.

    ``stats`` is the complete JSON structure returned by compute_norm_stats.
    Fallback channels are identified from the input data's original quantiles.
    The normalized maxima use the supplied asset bounds without repairing or
    clipping them, so this helper also exposes stale collapsed-range assets.
    """
    arrays = _validated_arrays(states, actions)
    report = {
        "policy": NORMALIZATION_POLICY,
        "fallback_channels": {},
        "max_abs_normalized": {},
        "source_percentiles": {},
    }
    for name, array in arrays.items():
        q01, q99, _, _, fallback = _quantile_ranges(array)
        loaded = {}
        for key in ("mean", "std", "q01", "q99"):
            try:
                value = np.asarray(stats["norm_stats"][name][key], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{name}: missing or invalid {key} statistics") from error
            if value.shape != (array.shape[1],) or not np.isfinite(value).all():
                raise ValueError(f"{name}: {key} statistics must be finite and match the native dimension")
            loaded[key] = value
        if np.any(loaded["q99"] < loaded["q01"]) or np.any(loaded["std"] < 0):
            raise ValueError(f"{name}: statistics contain a negative range or standard deviation")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            normalized = (array - loaded["q01"]) / (
                loaded["q99"] - loaded["q01"] + NORMALIZATION_EPSILON
            ) * 2.0 - 1.0
            maximum = float(np.max(np.abs(normalized)))
        if not np.isfinite(maximum) or not np.isfinite([q01, q99]).all():
            raise ValueError(f"{name}: normalization diagnostics are nonfinite")
        report["fallback_channels"][name] = np.flatnonzero(fallback).tolist()
        report["max_abs_normalized"][name] = maximum
        report["source_percentiles"][name] = {"q01": q01.tolist(), "q99": q99.tolist()}
    return report
