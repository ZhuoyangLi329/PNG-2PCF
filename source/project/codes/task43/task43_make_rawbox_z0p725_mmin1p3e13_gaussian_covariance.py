#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small covariance helpers retained for active Task43 lightcone scripts.

The full historical raw-box builder was archived on 2026-07-07, while two
active jaxpower scripts still import its SPD and correlation helpers.  Keeping
these two dependency-free functions here makes that provenance explicit and
avoids restoring the unrelated historical raw-box pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def nearest_spd(matrix: np.ndarray, *, floor_fraction: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Symmetrize a matrix and floor only eigenvalues below a relative threshold."""
    matrix = np.asarray(matrix, dtype="f8")
    sym = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    max_eigenvalue = float(np.max(eigenvalues))
    floor = float(max(float(floor_fraction) * max_eigenvalue, 1.0e-30))
    floored = np.maximum(eigenvalues, floor)
    covariance = (eigenvectors * floored[None, :]) @ eigenvectors.T
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, {
        "raw_min_eigenvalue": float(np.min(eigenvalues)),
        "raw_max_eigenvalue": max_eigenvalue,
        "floor_eigenvalue": floor,
        "n_floored": int(np.count_nonzero(eigenvalues < floor)),
        "condition_number_after_floor": float(np.linalg.cond(covariance)),
    }


def correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix safely."""
    covariance = np.asarray(covariance, dtype="f8")
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    denominator = np.outer(sigma, sigma)
    correlation = np.zeros_like(covariance)
    np.divide(covariance, denominator, out=correlation, where=denominator > 0.0)
    return correlation
