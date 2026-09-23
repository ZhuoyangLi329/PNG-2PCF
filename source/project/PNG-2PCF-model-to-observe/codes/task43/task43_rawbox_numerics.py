#!/usr/bin/env python3
"""Numerically strict building blocks for Task 4.3 rawbox validation.

The lattice convention in this module contains both members of every nonzero
Fourier +/- pair.  Covariances are validated in dimensionless correlation
units and are never repaired by silently flooring eigenvalues.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import eval_legendre, factorial2, hyp2f1, sici


RAWBOX_FIT_EDGES = np.asarray(
    [
        [0.003, 0.005],
        [0.005, 0.007],
        [0.007, 0.009],
        [0.009, 0.011],
        [0.013, 0.015],
        [0.017, 0.019],
        [0.021, 0.023],
        [0.029, 0.031],
        [0.037, 0.039],
        [0.045, 0.047],
        [0.053, 0.055],
        [0.061, 0.063],
        [0.069, 0.071],
        [0.077, 0.079],
        [0.085, 0.087],
        [0.093, 0.095],
    ],
    dtype="f8",
)


def _finite(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype="f8")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite entries")
    return result


@dataclass
class GaussianMetric:
    """A fixed-covariance metric validated in dimensionless units."""

    covariance: np.ndarray
    min_correlation_eigenvalue: float = 1.0e-12

    def __post_init__(self) -> None:
        covariance = _finite(self.covariance, "covariance").copy()
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or covariance.size == 0:
            raise ValueError("covariance must be a nonempty square matrix")
        if np.any(np.diag(covariance) <= 0.0):
            raise ValueError("covariance has a non-positive marginal variance")

        self.scale = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(self.scale, self.scale)
        if not np.allclose(correlation, correlation.T, rtol=0.0, atol=1.0e-12):
            raise ValueError("covariance is not symmetric in normalized units")
        correlation = 0.5 * (correlation + correlation.T)
        self.eigenvalues = np.linalg.eigvalsh(correlation)
        if self.eigenvalues[0] <= float(self.min_correlation_eigenvalue):
            raise ValueError(
                "indefinite or unresolved covariance; automatic repair is forbidden: "
                f"lambda_min(R)={self.eigenvalues[0]:.8g}"
            )
        self.cholesky = np.linalg.cholesky(correlation)
        self.correlation = correlation
        self.covariance = 0.5 * (covariance + covariance.T)

    def residual(self, residual: np.ndarray) -> np.ndarray:
        value = _finite(residual, "residual")
        if value.shape != self.scale.shape:
            raise ValueError("residual shape differs from covariance dimension")
        return solve_triangular(self.cholesky, value / self.scale, lower=True)

    def chi2(self, residual: np.ndarray) -> float:
        whitened = self.residual(residual)
        return float(whitened @ whitened)

    def solve(self, value: np.ndarray) -> np.ndarray:
        right = _finite(value, "right hand side")
        if right.ndim not in (1, 2) or right.shape[0] != self.scale.size:
            raise ValueError("right hand side has the wrong shape")
        scale = self.scale if right.ndim == 1 else self.scale[:, None]
        whitened = solve_triangular(self.cholesky, right / scale, lower=True)
        solved = solve_triangular(self.cholesky.T, whitened, lower=False)
        return solved / scale


def assemble_covariance(c_pp: np.ndarray, c_xx: np.ndarray, c_xp: np.ndarray | None = None) -> np.ndarray:
    """Assemble a P/xi covariance while preserving both marginal blocks."""

    metric_p = GaussianMetric(c_pp)
    metric_x = GaussianMetric(c_xx)
    if c_xp is None:
        cross = np.zeros((metric_x.scale.size, metric_p.scale.size), dtype="f8")
    else:
        cross = _finite(c_xp, "cross covariance")
    if cross.shape != (metric_x.scale.size, metric_p.scale.size):
        raise ValueError("c_xp must have xi rows and P columns")
    covariance = np.block([[metric_p.covariance, cross.T], [cross, metric_x.covariance]])
    GaussianMetric(covariance)
    return covariance


def canonical_correlations(c_pp: np.ndarray, c_xx: np.ndarray, c_xp: np.ndarray) -> np.ndarray:
    """Return canonical P/xi correlations for a proposed cross block."""

    metric_p = GaussianMetric(c_pp)
    metric_x = GaussianMetric(c_xx)
    cross = _finite(c_xp, "cross covariance")
    if cross.shape != (metric_x.scale.size, metric_p.scale.size):
        raise ValueError("cross covariance has the wrong shape")
    normalized = cross / np.outer(metric_x.scale, metric_p.scale)
    normalized = solve_triangular(metric_x.cholesky, normalized, lower=True)
    normalized = solve_triangular(metric_p.cholesky, normalized.T, lower=True).T
    return np.linalg.svd(normalized, compute_uv=False)


def conditional_xi_residual(
    residual_p: np.ndarray,
    residual_x: np.ndarray,
    c_pp: np.ndarray,
    c_xx: np.ndarray,
    c_xp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return xi|P residual, Schur covariance, and conditional chi-square."""

    assemble_covariance(c_pp, c_xx, c_xp)
    metric_p = GaussianMetric(c_pp)
    cross = _finite(c_xp, "cross covariance")
    residual = _finite(residual_x, "xi residual") - cross @ metric_p.solve(residual_p)
    schur = _finite(c_xx, "xi covariance") - cross @ metric_p.solve(cross.T)
    metric = GaussianMetric(schur)
    return residual, metric.covariance, metric.chi2(residual)


def frozen_covariance_refit(
    covariance_builder: Callable[[float], np.ndarray],
    fit_fixed_covariance: Callable[[np.ndarray], Any],
    *,
    initial_b1: float,
    b1_index: int,
) -> dict[str, Any]:
    """Choose a covariance with a prefit, freeze it, then refit the MAP."""

    covariance_initial = _finite(covariance_builder(float(initial_b1)), "initial covariance")
    GaussianMetric(covariance_initial)
    prefit = fit_fixed_covariance(covariance_initial)
    prefit_theta = _finite(prefit.x, "prefit parameters")
    if not 0 <= int(b1_index) < prefit_theta.size:
        raise ValueError("b1_index is outside the fitted parameter vector")

    covariance_b1 = float(prefit_theta[int(b1_index)])
    covariance_final = _finite(covariance_builder(covariance_b1), "final frozen covariance")
    metric_final = GaussianMetric(covariance_final)
    final_fit = fit_fixed_covariance(metric_final.covariance)
    final_theta = _finite(final_fit.x, "final parameters")
    return {
        "initial_b1": float(initial_b1),
        "covariance_b1": covariance_b1,
        "prefit": prefit,
        "prefit_theta": prefit_theta,
        "final_fit": final_fit,
        "final_theta": final_theta,
        "covariance": metric_final.covariance,
        "metric": metric_final,
    }


def lorentzian_moments(ksigma: np.ndarray, *, max_power: int = 6, exponent: int = 2) -> np.ndarray:
    """Return half-range even moments of the squared-Lorentzian kernel."""

    value = _finite(ksigma, "k sigma")
    if max_power < 0 or exponent < 0:
        raise ValueError("moment powers must be non-negative")
    return np.stack(
        [
            hyp2f1(exponent, power + 0.5, power + 1.5, -0.5 * value * value) / (2 * power + 1)
            for power in range(max_power + 1)
        ]
    )


def continuous_rsd_poles(
    k: np.ndarray,
    linear_power: np.ndarray,
    amplitude: np.ndarray,
    growth: float,
    sigma_s: float,
) -> dict[int, np.ndarray]:
    """Return continuous-angle ell=0,2 Kaiser x Lorentzian poles."""

    kval, power, bias = np.broadcast_arrays(
        _finite(k, "k"),
        _finite(linear_power, "linear power"),
        _finite(amplitude, "amplitude"),
    )
    if np.any(kval < 0.0) or sigma_s < 0.0 or not np.isfinite(growth):
        raise ValueError("invalid k, growth, or sigma_s")
    moments = lorentzian_moments(kval * float(sigma_s), max_power=3, exponent=2)
    p0_factor = bias**2 * moments[0] + 2.0 * bias * growth * moments[1] + growth**2 * moments[2]
    mu2_factor = bias**2 * moments[1] + 2.0 * bias * growth * moments[2] + growth**2 * moments[3]
    return {
        0: power * p0_factor,
        2: 2.5 * power * (3.0 * mu2_factor - p0_factor),
    }


def angular_total_integrals(
    k: np.ndarray,
    linear_power: np.ndarray,
    amplitude: np.ndarray,
    growth: float,
    sigma_s: float,
    shot_noise: float | np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Return integral_-1^1 [signal+shot]^2 L_a L_b dmu."""

    kval, power, bias, shot = np.broadcast_arrays(
        _finite(k, "k"),
        _finite(linear_power, "linear power"),
        _finite(amplitude, "amplitude"),
        _finite(shot_noise, "shot noise"),
    )
    if np.any(kval < 0.0) or sigma_s < 0.0 or not np.isfinite(growth):
        raise ValueError("invalid covariance angular-integral inputs")
    moments_signal = lorentzian_moments(kval * float(sigma_s), max_power=6, exponent=2)
    moments_squared = lorentzian_moments(kval * float(sigma_s), max_power=6, exponent=4)
    legendre_polynomials = {0: np.asarray([1.0]), 2: np.asarray([-0.5, 1.5])}
    result: dict[tuple[int, int], np.ndarray] = {}
    for ell_a in (0, 2):
        for ell_b in (0, 2):
            product = np.polynomial.polynomial.polymul(
                legendre_polynomials[ell_a], legendre_polynomials[ell_b]
            )
            integral = np.zeros_like(kval)
            for power_index, coefficient in enumerate(product):
                signal_squared = power**2 * sum(
                    math.comb(4, order)
                    * bias ** (4 - order)
                    * growth**order
                    * moments_squared[power_index + order]
                    for order in range(5)
                )
                signal_shot = 2.0 * shot * power * sum(
                    math.comb(2, order)
                    * bias ** (2 - order)
                    * growth**order
                    * moments_signal[power_index + order]
                    for order in range(3)
                )
                shot_squared = shot**2 / (2 * power_index + 1)
                integral += 2.0 * coefficient * (signal_squared + signal_shot + shot_squared)
            result[ell_a, ell_b] = integral
    return result


def shell_kernel(k: np.ndarray, edges: np.ndarray, ell: int) -> np.ndarray:
    """Return i**ell times the volume-averaged j_ell shell kernel."""

    kval = _finite(k, "k").reshape(-1)
    sedges = _finite(edges, "separation edges")
    if ell not in (0, 2) or np.any(kval < 0.0):
        raise ValueError("shell_kernel requires ell=0 or 2 and non-negative k")
    if sedges.ndim != 1 or sedges.size < 2 or sedges[0] < 0.0 or np.any(np.diff(sedges) <= 0.0):
        raise ValueError("invalid separation edges")

    lo = sedges[:-1][None, :]
    hi = sedges[1:][None, :]
    kk = kval[:, None]
    lower = kk * lo
    upper = kk * hi
    if ell == 0:
        primitive = lambda x: np.sin(x) - x * np.cos(x)
    else:
        primitive = lambda x: 3.0 * sici(x)[0] + x * np.cos(x) - 4.0 * np.sin(x)
    denominator = kk**3 * (hi**3 - lo**3) / 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (primitive(upper) - primitive(lower)) / denominator

    small = upper < 0.5
    if np.any(small):
        series = np.zeros_like(result)
        for order in range(7):
            power = ell + 2 * order
            coefficient = (-1.0) ** order / (
                2**order * math.factorial(order) * factorial2(2 * ell + 2 * order + 1)
            )
            radius_moment = 3.0 * (hi ** (power + 3) - lo ** (power + 3)) / (
                (power + 3) * (hi**3 - lo**3)
            )
            series += coefficient * kk**power * radius_moment
        result[small] = series[small]
    return (-1) ** (ell // 2) * result


@dataclass(frozen=True)
class ExactLatticeModes:
    """Full +/- periodic modes selected into observational P bins first."""

    vectors: np.ndarray
    k: np.ndarray
    mu: np.ndarray
    bin_id: np.ndarray
    counts: np.ndarray
    k_mean: np.ndarray
    boxsize: float
    bins: np.ndarray


def exact_lattice_modes(boxsize: float, bins: np.ndarray, *, max_cube_cells: int = 3_000_000) -> ExactLatticeModes:
    """Enumerate low-k modes and assign exact members before any rebinning."""

    fit_bins = _finite(bins, "k bins")
    if boxsize <= 0.0 or fit_bins.ndim != 2 or fit_bins.shape[1] != 2:
        raise ValueError("boxsize must be positive and bins must have shape (N, 2)")
    if np.any(fit_bins[:, 0] < 0.0) or np.any(fit_bins[:, 1] <= fit_bins[:, 0]):
        raise ValueError("invalid k-bin edges")
    kfund = 2.0 * np.pi / float(boxsize)
    nmax = int(np.ceil(float(np.max(fit_bins[:, 1])) / kfund))
    if (2 * nmax + 1) ** 3 > int(max_cube_cells):
        raise ValueError("large lattice enumeration rejected")

    integers = np.arange(-nmax, nmax + 1, dtype="i4")
    vectors = np.stack(np.meshgrid(integers, integers, integers, indexing="ij"), axis=-1).reshape(-1, 3)
    norm2 = np.sum(vectors.astype("i8") ** 2, axis=1)
    kval = kfund * np.sqrt(norm2)
    bin_id = np.full(kval.size, -1, dtype="i4")
    for index, (lower, upper) in enumerate(fit_bins):
        selected = (norm2 > 0) & (kval >= lower) & (kval < upper)
        if np.any(bin_id[selected] >= 0):
            raise ValueError("overlapping k bins are not supported")
        bin_id[selected] = index
    keep = bin_id >= 0
    vectors = vectors[keep]
    norm2 = norm2[keep]
    kval = kval[keep]
    bin_id = bin_id[keep]
    counts = np.bincount(bin_id, minlength=fit_bins.shape[0]).astype("f8")
    if np.any(counts == 0.0):
        raise ValueError("one or more observational P bins contain no lattice modes")
    mu = vectors[:, 2] / np.sqrt(norm2)
    k_mean = np.bincount(bin_id, weights=kval, minlength=fit_bins.shape[0]) / counts
    return ExactLatticeModes(vectors, kval, mu, bin_id, counts, k_mean, float(boxsize), fit_bins.copy())


def mode_operators(
    modes: ExactLatticeModes,
    separation_edges: np.ndarray,
    *,
    ells: tuple[int, ...] = (0, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """Build P and low-k xi operators from exactly the same modes."""

    if any(ell not in (0, 2) for ell in ells):
        raise ValueError("only ell=0 and ell=2 operators are supported")
    selection = modes.bin_id[None, :] == np.arange(modes.counts.size)[:, None]
    p_rows: list[np.ndarray] = []
    xi_rows: list[np.ndarray] = []
    volume = modes.boxsize**3
    for ell in ells:
        angular = (2 * ell + 1) * eval_legendre(ell, modes.mu)
        p_rows.append(selection * angular[None, :] / modes.counts[:, None])
        xi_rows.append(shell_kernel(modes.k, separation_edges, ell).T * angular[None, :] / volume)
    return np.vstack(p_rows), np.vstack(xi_rows)


def gaussian_mode_covariance(operator: np.ndarray, total_power: np.ndarray) -> np.ndarray:
    """Evaluate 2 W diag(T**2) W.T for a full +/- even-mode operator."""

    weights = _finite(operator, "operator")
    total = _finite(total_power, "total power")
    if weights.ndim != 2 or total.shape != (weights.shape[1],) or np.any(total <= 0.0):
        raise ValueError("invalid operator or non-positive total power")
    factored = np.sqrt(2.0) * weights * total[None, :]
    return factored @ factored.T
