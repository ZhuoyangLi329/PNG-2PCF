#!/usr/bin/env python3
"""Cross-validated smoothing of noisy anchor-sampled radial pair moments."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import UnivariateSpline


DEFAULT_SMOOTHING_FACTORS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def shell_volume_centers(edges: np.ndarray) -> np.ndarray:
    radial_edges = np.asarray(edges, dtype="f8")
    if (
        radial_edges.ndim != 1
        or radial_edges.size < 5
        or radial_edges[0] != 0.0
        or np.any(np.diff(radial_edges) <= 0.0)
    ):
        raise ValueError("radial edges must be ordered and begin at zero")
    lower, upper = radial_edges[:-1], radial_edges[1:]
    return 0.75 * (upper**4 - lower**4) / (upper**3 - lower**3)


def _fit(
    radius: np.ndarray,
    values: np.ndarray,
    uncertainty: np.ndarray,
    factor: float,
) -> UnivariateSpline:
    if not np.isfinite(factor) or factor < 0.0:
        raise ValueError("smoothing factor must be finite and non-negative")
    sigma = np.asarray(uncertainty, dtype="f8")
    positive = sigma[np.isfinite(sigma) & (sigma > 0.0)]
    if positive.size != sigma.size:
        raise ValueError("smoothing uncertainties must be finite and positive")
    return UnivariateSpline(
        np.asarray(radius, dtype="f8"),
        np.asarray(values, dtype="f8"),
        w=1.0 / sigma,
        s=float(factor) * len(radius),
        k=3,
        ext=2,
    )


def apply_smoothing_factor(
    radial_edges: np.ndarray,
    values: np.ndarray,
    uncertainty: np.ndarray,
    factor: float,
) -> np.ndarray:
    """Apply a previously selected factor to a specified pooled estimator."""

    radius = shell_volume_centers(radial_edges)
    data = np.asarray(values, dtype="f8")
    sigma = np.asarray(uncertainty, dtype="f8")
    if data.shape != radius.shape or sigma.shape != radius.shape or not np.all(np.isfinite(data)):
        raise ValueError("pooled smoothing inputs do not match the radial grid")
    smoothed = np.asarray(_fit(radius, data, sigma, float(factor))(radius), dtype="f8")
    if not np.all(np.isfinite(smoothed)):
        raise RuntimeError("pooled smoothing produced non-finite values")
    return smoothed


def cross_validated_smoothing(
    radial_edges: np.ndarray,
    block_values: np.ndarray,
    *,
    factors: tuple[float, ...] = DEFAULT_SMOOTHING_FACTORS,
    require_positive: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a smoothing scale from disjoint blocks without using RSD xi.

    Each fold predicts one block from the mean of the others.  Scores use the
    held-block plus training-mean variance.  The final fit uses the standard
    error of the all-block mean and the selected dimensionless residual budget.
    """

    radius = shell_volume_centers(radial_edges)
    blocks = np.asarray(block_values, dtype="f8")
    candidates = tuple(float(value) for value in factors)
    if blocks.ndim != 2 or blocks.shape[1] != radius.size or blocks.shape[0] < 4:
        raise ValueError("block values must have shape (at least 4, number of radial bins)")
    if not np.all(np.isfinite(blocks)) or not candidates or any(value < 0.0 for value in candidates):
        raise ValueError("invalid block values or smoothing candidates")

    nblock, nradial = blocks.shape
    candidate_rows: list[dict[str, Any]] = []
    for factor in candidates:
        fold_scores = []
        valid = True
        for held in range(nblock):
            training = np.delete(blocks, held, axis=0)
            train_mean = np.mean(training, axis=0)
            train_sd = np.std(training, axis=0, ddof=1)
            train_se = train_sd / np.sqrt(training.shape[0])
            try:
                prediction = np.asarray(_fit(radius, train_mean, train_se, factor)(radius), dtype="f8")
            except (ValueError, RuntimeError):
                valid = False
                break
            if require_positive and np.any(prediction <= 0.0):
                valid = False
                break
            comparison_variance = train_sd**2 * (1.0 + 1.0 / training.shape[0])
            if np.any(comparison_variance <= 0.0):
                raise ValueError("zero block variance prevents cross-validation")
            fold_scores.append(float(np.mean((blocks[held] - prediction) ** 2 / comparison_variance)))
        candidate_rows.append(
            {
                "factor": factor,
                "valid": valid,
                "fold_scores": fold_scores,
                "mean_score": float(np.mean(fold_scores)) if valid else None,
            }
        )

    valid_rows = [row for row in candidate_rows if row["valid"]]
    if not valid_rows:
        raise RuntimeError("no smoothing candidate passed the physical constraints")
    selected = min(valid_rows, key=lambda row: (float(row["mean_score"]), float(row["factor"])))
    pooled = np.mean(blocks, axis=0)
    pooled_se = np.std(blocks, axis=0, ddof=1) / np.sqrt(nblock)
    smoothed = np.asarray(_fit(radius, pooled, pooled_se, float(selected["factor"]))(radius), dtype="f8")
    if not np.all(np.isfinite(smoothed)) or (require_positive and np.any(smoothed <= 0.0)):
        raise RuntimeError("selected final smoothing fit violates physical constraints")
    audit = {
        "selection": "leave-one-anchor-block-out; RSD target was not used",
        "nblocks": int(nblock),
        "nradial": int(nradial),
        "selected_factor": float(selected["factor"]),
        "selected_mean_score": float(selected["mean_score"]),
        "candidates": candidate_rows,
        "weighted_residual_chi2": float(np.sum(((pooled - smoothed) / pooled_se) ** 2)),
        "max_abs_adjustment": float(np.max(np.abs(smoothed - pooled))),
        "rms_adjustment": float(np.sqrt(np.mean((smoothed - pooled) ** 2))),
    }
    return smoothed, audit
