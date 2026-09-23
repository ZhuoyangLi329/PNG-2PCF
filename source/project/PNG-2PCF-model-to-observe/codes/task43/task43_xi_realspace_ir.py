#!/usr/bin/env python3
"""Independent finite-box real-space xi model with leading BAO resummation.

The frozen P(k) branch is an input to this module and is never modified.  The
only new operation is the standard split P_lin = P_nw + P_w followed by a
Gaussian damping of P_w before applying the existing finite-box shell
operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson
from scipy.special import spherical_jn


DELTA_C = 1.686


def _finite_1d(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype="f8")
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return result


def ir_resummed_power(
    k: np.ndarray,
    linear_power: np.ndarray,
    smooth_power: np.ndarray,
    sigma_bao: float,
) -> np.ndarray:
    """Return P_nw + exp[-k^2 Sigma_BAO^2 / 2] (P_lin - P_nw)."""

    kval = _finite_1d(k, "k")
    plin = _finite_1d(linear_power, "linear power")
    pnow = _finite_1d(smooth_power, "smooth power")
    if not (kval.shape == plin.shape == pnow.shape):
        raise ValueError("k and power arrays must have identical shapes")
    sigma = float(sigma_bao)
    if np.any(kval <= 0.0) or np.any(plin <= 0.0) or np.any(pnow <= 0.0):
        raise ValueError("k and power arrays must be strictly positive")
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma_bao must be finite and non-negative")
    damping = np.exp(-0.5 * (kval * sigma) ** 2)
    return pnow + damping * (plin - pnow)


def bao_displacement_sigma(
    q: np.ndarray,
    smooth_power: np.ndarray,
    *,
    r_bao: float,
    k_ir: float,
) -> float:
    """Compute the leading relative-displacement BAO damping scale.

    The convention matches ``exp(-k^2 Sigma_BAO^2 / 2)`` above:

        Sigma_BAO^2 = 1/(3 pi^2) int_0^kIR dq P_nw(q)
                      [1 - j0(q r_BAO) + 2 j2(q r_BAO)].
    """

    qval = _finite_1d(q, "q")
    pnow = _finite_1d(smooth_power, "smooth power")
    if qval.shape != pnow.shape or qval.size < 8:
        raise ValueError("q and smooth power must have matching shapes with at least eight samples")
    if np.any(qval <= 0.0) or np.any(np.diff(qval) <= 0.0) or np.any(pnow <= 0.0):
        raise ValueError("q must increase strictly and smooth power must be positive")
    radius = float(r_bao)
    cutoff = float(k_ir)
    if not np.isfinite(radius) or radius <= 0.0 or not np.isfinite(cutoff) or cutoff <= qval[0]:
        raise ValueError("r_bao and k_ir must be positive and k_ir must exceed q[0]")
    if cutoff > qval[-1]:
        raise ValueError("q does not cover k_ir; extrapolation is forbidden")

    if cutoff == qval[-1]:
        q_use = qval
        p_use = pnow
    else:
        below = qval < cutoff
        q_use = np.concatenate([qval[below], np.asarray([cutoff])])
        p_cut = np.interp(np.log(cutoff), np.log(qval), pnow)
        p_use = np.concatenate([pnow[below], np.asarray([p_cut])])
    qr = q_use * radius
    kernel = 1.0 - spherical_jn(0, qr) + 2.0 * spherical_jn(2, qr)
    sigma2 = float(simpson(p_use * kernel, x=q_use) / (3.0 * np.pi**2))
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError("computed BAO displacement variance is not positive and finite")
    return float(np.sqrt(sigma2))


@dataclass(frozen=True)
class FiniteBoxIRXiModel:
    """Apply the existing finite-box shell operator to an IR-resummed spectrum."""

    k: np.ndarray
    degeneracy: np.ndarray
    linear_power: np.ndarray
    smooth_power: np.ndarray
    alpha: np.ndarray
    kernel0: np.ndarray
    volume: float

    def __post_init__(self) -> None:
        k = _finite_1d(self.k, "k")
        degeneracy = _finite_1d(self.degeneracy, "degeneracy")
        linear = _finite_1d(self.linear_power, "linear power")
        smooth = _finite_1d(self.smooth_power, "smooth power")
        alpha = _finite_1d(self.alpha, "alpha")
        kernel = np.asarray(self.kernel0, dtype="f8")
        if not (k.shape == degeneracy.shape == linear.shape == smooth.shape == alpha.shape):
            raise ValueError("all mode arrays must have identical shapes")
        if kernel.ndim != 2 or kernel.shape[0] != k.size or not np.all(np.isfinite(kernel)):
            raise ValueError("kernel0 must be finite with one row per mode")
        if np.any(degeneracy <= 0.0) or not np.isfinite(self.volume) or float(self.volume) <= 0.0:
            raise ValueError("mode degeneracies and volume must be positive")
        object.__setattr__(self, "k", k.copy())
        object.__setattr__(self, "degeneracy", degeneracy.copy())
        object.__setattr__(self, "linear_power", linear.copy())
        object.__setattr__(self, "smooth_power", smooth.copy())
        object.__setattr__(self, "alpha", alpha.copy())
        object.__setattr__(self, "kernel0", kernel.copy())
        object.__setattr__(self, "volume", float(self.volume))

    def evaluate(self, *, fnl: float, b1: float, sigma_bao: float, p_fixed: float = 1.0) -> np.ndarray:
        bphi = 2.0 * DELTA_C * (float(b1) - float(p_fixed))
        amplitude = float(b1) + float(fnl) * bphi * self.alpha
        power = ir_resummed_power(self.k, self.linear_power, self.smooth_power, float(sigma_bao))
        return ((self.degeneracy * power * amplitude**2) @ self.kernel0) / self.volume

