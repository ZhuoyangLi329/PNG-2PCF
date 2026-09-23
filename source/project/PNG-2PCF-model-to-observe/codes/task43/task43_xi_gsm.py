#!/usr/bin/env python3
"""Pair-conserving xi-only Gaussian streaming model for rawbox diagnostics.

This module never imports or modifies the production P(k) model.  Distances
and pair displacements use Mpc/h; all variances are central variances in
(Mpc/h)^2.  The empirical provider is an oracle input, not an independent
theory prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.special import eval_legendre


MomentFunction = Callable[[float, float], tuple[float, float, float]]


def _finite_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype="f8")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


@dataclass(frozen=True)
class EmpiricalRadialMoments:
    """Interpolated same-catalog density and pair-velocity moment table."""

    radial_edges: np.ndarray
    radial_knots: np.ndarray
    xi_values: np.ndarray
    v12_values: np.ndarray
    sigma_r2_values: np.ndarray
    sigma_t2_values: np.ndarray
    xi_interpolator: PchipInterpolator
    v12_interpolator: PchipInterpolator
    sigma_r2_interpolator: PchipInterpolator
    sigma_t2_interpolator: PchipInterpolator

    @classmethod
    def from_arrays(
        cls,
        radial_edges: np.ndarray,
        xi_values: np.ndarray,
        v12_values: np.ndarray,
        sigma_r2_values: np.ndarray,
        sigma_t2_values: np.ndarray,
    ) -> "EmpiricalRadialMoments":
        edges = _finite_array(radial_edges, "radial_edges")
        xi = _finite_array(xi_values, "xi_values")
        v12 = _finite_array(v12_values, "v12_values")
        sigma_r2 = _finite_array(sigma_r2_values, "sigma_r2_values")
        sigma_t2 = _finite_array(sigma_t2_values, "sigma_t2_values")
        if edges.ndim != 1 or edges.size < 3 or edges[0] != 0.0 or np.any(np.diff(edges) <= 0.0):
            raise ValueError("radial edges must be ordered and start at zero")
        expected = (edges.size - 1,)
        if any(value.shape != expected for value in (xi, v12, sigma_r2, sigma_t2)):
            raise ValueError("radial table values do not match radial edges")
        if np.any(1.0 + xi <= 0.0):
            raise ValueError("real-space pair density must be strictly positive")
        if np.any(sigma_r2 <= 0.0) or np.any(sigma_t2 <= 0.0):
            raise ValueError("empirical pair variances must be strictly positive")

        # Shell moments are attached to the uniform-volume mean radius.  The
        # first and last values extend only inside their measured edge bins.
        lo, hi = edges[:-1], edges[1:]
        centers = 0.75 * (hi**4 - lo**4) / (hi**3 - lo**3)
        knots = np.concatenate(([edges[0]], centers, [edges[-1]]))

        def interpolator(values: np.ndarray) -> PchipInterpolator:
            padded = np.concatenate(([values[0]], values, [values[-1]]))
            return PchipInterpolator(knots, padded, extrapolate=False)

        result = cls(
            radial_edges=edges.copy(),
            radial_knots=knots,
            xi_values=xi.copy(),
            v12_values=v12.copy(),
            sigma_r2_values=sigma_r2.copy(),
            sigma_t2_values=sigma_t2.copy(),
            xi_interpolator=interpolator(xi),
            v12_interpolator=interpolator(v12),
            sigma_r2_interpolator=interpolator(sigma_r2),
            sigma_t2_interpolator=interpolator(sigma_t2),
        )
        check_radius = np.linspace(edges[0], edges[-1], 20001)
        check = result.evaluate_radial(check_radius)
        if np.any(1.0 + check[0] <= 0.0):
            raise ValueError("xi interpolation creates non-positive pair density")
        if np.any(check[2] <= 0.0) or np.any(check[3] <= 0.0):
            raise ValueError("moment interpolation creates non-positive variance")
        return result

    @classmethod
    def from_npz(cls, path: Path) -> "EmpiricalRadialMoments":
        with np.load(Path(path), allow_pickle=False) as payload:
            return cls.from_arrays(
                payload["radial_edges"],
                payload["full_xi_real"],
                payload["pair_v12_radial"],
                payload["pair_sigma_r2_central"],
                payload["pair_sigma_t2_one_component"],
            )

    @property
    def support(self) -> tuple[float, float]:
        return float(self.radial_edges[0]), float(self.radial_edges[-1])

    @property
    def integration_scale(self) -> float:
        return float(np.sqrt(max(np.max(self.sigma_r2_values), np.max(self.sigma_t2_values))))

    def evaluate_radial(self, radius: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r = _finite_array(radius, "radius")
        lower, upper = self.support
        if np.any(r < lower) or np.any(r > upper):
            raise ValueError(f"radial query lies outside measured support [{lower},{upper}]")
        values = tuple(
            np.asarray(interpolator(r), dtype="f8")
            for interpolator in (
                self.xi_interpolator,
                self.v12_interpolator,
                self.sigma_r2_interpolator,
                self.sigma_t2_interpolator,
            )
        )
        if not all(np.all(np.isfinite(value)) for value in values):
            raise ValueError("non-finite empirical moment interpolation")
        return values  # type: ignore[return-value]

    def at_los(self, transverse: float, real_parallel: float) -> tuple[float, float, float]:
        radius = math.hypot(float(transverse), float(real_parallel))
        xi, v12, sigma_r2, sigma_t2 = self.evaluate_radial(radius)
        mu_real = float(real_parallel) / radius if radius > 0.0 else 0.0
        mean = mu_real * float(v12)
        variance = mu_real**2 * float(sigma_r2) + (1.0 - mu_real**2) * float(sigma_t2)
        return float(xi), mean, variance

    def at_los_array(
        self, transverse: np.ndarray, real_parallel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        perpendicular, los = np.broadcast_arrays(
            _finite_array(transverse, "transverse"), _finite_array(real_parallel, "real_parallel")
        )
        if np.any(perpendicular < 0.0):
            raise ValueError("transverse separation must be non-negative")
        radius = np.hypot(perpendicular, los)
        xi, v12, sigma_r2, sigma_t2 = self.evaluate_radial(radius)
        mu_real = np.divide(los, radius, out=np.zeros_like(radius), where=radius > 0.0)
        mean = mu_real * v12
        variance = mu_real**2 * sigma_r2 + (1.0 - mu_real**2) * sigma_t2
        return xi, mean, variance


def gsm_point(
    transverse: float,
    parallel: float,
    moments: MomentFunction,
    *,
    integration_scale: float,
    zmax: float = 12.0,
    extra_pair_variance: float = 0.0,
    boxsize: float | None = None,
    periodic_images: int = 0,
    epsabs: float = 2.0e-10,
    zero_velocity: bool = False,
) -> tuple[float, float]:
    """Evaluate the pair-conserving Gaussian streaming integral at one point.

    If ``boxsize`` is supplied, neighboring periodized Gaussian images are
    included in the local integral.  The caller must separately demonstrate
    LOS-tail convergence because the empirical radial table is finite.
    """

    settings = np.asarray(
        [transverse, parallel, integration_scale, zmax, extra_pair_variance, epsabs], dtype="f8"
    )
    if (
        not np.all(np.isfinite(settings))
        or transverse < 0.0
        or integration_scale <= 0.0
        or zmax <= 0.0
        or extra_pair_variance < 0.0
        or epsabs <= 0.0
        or int(periodic_images) < 0
    ):
        raise ValueError("invalid separation, integration setting, or variance")
    if boxsize is None and int(periodic_images) != 0:
        raise ValueError("periodic images require a boxsize")
    if boxsize is not None and (not np.isfinite(boxsize) or boxsize <= 0.0):
        raise ValueError("boxsize must be finite and positive")
    if zero_velocity:
        xi, mean, variance = moments(float(transverse), float(parallel))
        if not np.all(np.isfinite([xi, mean, variance])) or xi <= -1.0:
            raise ValueError("invalid zero-velocity moments")
        if mean != 0.0 or variance != 0.0 or extra_pair_variance != 0.0:
            raise ValueError("zero_velocity contradicts supplied moments")
        return float(xi), 0.0

    image_indices = np.arange(-int(periodic_images), int(periodic_images) + 1, dtype="f8")

    def integrand(t: float) -> float:
        real_parallel = float(parallel) + float(integration_scale) * t
        xi, mean, variance = moments(float(transverse), real_parallel)
        variance = float(variance) + float(extra_pair_variance)
        if not np.all(np.isfinite([xi, mean, variance])) or xi <= -1.0 or variance <= 0.0:
            raise ValueError("invalid real-space pair moments")
        residual = float(parallel) - real_parallel - float(mean)
        if boxsize is None:
            residuals = np.asarray([residual])
        else:
            residuals = residual + image_indices * float(boxsize)
        gaussian = np.exp(-0.5 * residuals**2 / variance).sum()
        density = float(integration_scale) * gaussian / math.sqrt(2.0 * math.pi * variance)
        reference = math.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
        return (1.0 + float(xi)) * density - reference

    breaks = [value for value in (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0) if -zmax < value < zmax]
    value, error = quad(
        integrand,
        -float(zmax),
        float(zmax),
        points=breaks,
        epsabs=float(epsabs),
        epsrel=2.0e-8,
        limit=250,
    )
    return float(value), float(error)


def composite_legendre_nodes(zmax: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed quadrature nodes split at the same landmarks as ``gsm_point``."""

    if not np.isfinite(zmax) or zmax <= 0.0 or int(order) < 2:
        raise ValueError("invalid composite quadrature setting")
    breaks = [-float(zmax)]
    breaks.extend(value for value in (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0) if -zmax < value < zmax)
    breaks.append(float(zmax))
    base_nodes, base_weights = np.polynomial.legendre.leggauss(int(order))
    nodes, weights = [], []
    for lower, upper in zip(breaks[:-1], breaks[1:]):
        nodes.append(0.5 * (upper - lower) * base_nodes + 0.5 * (upper + lower))
        weights.append(0.5 * (upper - lower) * base_weights)
    return np.concatenate(nodes), np.concatenate(weights)


def gsm_points_fixed(
    transverse: np.ndarray,
    parallel: np.ndarray,
    provider: EmpiricalRadialMoments,
    *,
    integration_scale: float,
    zmax: float = 12.0,
    quadrature_order: int = 12,
    extra_pair_variance: float = 0.0,
    boxsize: float | None = None,
    periodic_images: int = 0,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Vectorized fixed-quadrature evaluation of the same GSM integral."""

    perpendicular, los_observed = np.broadcast_arrays(
        _finite_array(transverse, "transverse"), _finite_array(parallel, "parallel")
    )
    if np.any(perpendicular < 0.0) or not np.isfinite(integration_scale) or integration_scale <= 0.0:
        raise ValueError("invalid separations or integration scale")
    if not np.isfinite(extra_pair_variance) or extra_pair_variance < 0.0:
        raise ValueError("extra pair variance must be finite and non-negative")
    if int(periodic_images) < 0 or (periodic_images and boxsize is None):
        raise ValueError("periodic images require a boxsize")
    if boxsize is not None and (not np.isfinite(boxsize) or boxsize <= 0.0):
        raise ValueError("boxsize must be finite and positive")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")

    nodes, weights = composite_legendre_nodes(float(zmax), int(quadrature_order))
    reference_integral = float(weights @ (np.exp(-0.5 * nodes**2) / math.sqrt(2.0 * math.pi)))
    flat_perpendicular = perpendicular.ravel()
    flat_parallel = los_observed.ravel()
    output = np.empty_like(flat_parallel)
    image_indices = np.arange(-int(periodic_images), int(periodic_images) + 1, dtype="f8")
    for start in range(0, flat_parallel.size, int(chunk_size)):
        stop = min(start + int(chunk_size), flat_parallel.size)
        observed = flat_parallel[start:stop, None]
        real = observed + float(integration_scale) * nodes[None, :]
        transverse_grid = np.broadcast_to(flat_perpendicular[start:stop, None], real.shape)
        xi, mean, variance = provider.at_los_array(transverse_grid, real)
        variance = variance + float(extra_pair_variance)
        if np.any(1.0 + xi <= 0.0) or np.any(variance <= 0.0):
            raise ValueError("invalid interpolated pair density or variance")
        residual = observed - real - mean
        if boxsize is None:
            gaussian_sum = np.exp(-0.5 * residual**2 / variance)
        else:
            shifted = residual[..., None] + image_indices[None, None, :] * float(boxsize)
            gaussian_sum = np.exp(-0.5 * shifted**2 / variance[..., None]).sum(axis=-1)
        density = float(integration_scale) * gaussian_sum / np.sqrt(2.0 * math.pi * variance)
        output[start:stop] = ((1.0 + xi) * density) @ weights - reference_integral
    return output.reshape(perpendicular.shape)


def shell_quadrature(edges: np.ndarray, nradial: int) -> tuple[np.ndarray, np.ndarray]:
    e = _finite_array(edges, "edges")
    if e.ndim != 1 or e.size < 2 or e[0] < 0.0 or np.any(np.diff(e) <= 0.0):
        raise ValueError("invalid radial shell edges")
    nodes, weights = np.polynomial.legendre.leggauss(int(nradial))
    lower, upper = e[:-1, None], e[1:, None]
    radius = 0.5 * (upper - lower) * nodes[None, :] + 0.5 * (upper + lower)
    raw_weight = 0.5 * (upper - lower) * weights[None, :] * radius**2
    normalization = (upper**3 - lower**3) / 3.0
    return radius, raw_weight / normalization


def fcfc_project_smu(
    evaluate_xi: Callable[[float, float], float],
    radial_edges: np.ndarray,
    mu_edges: np.ndarray,
    *,
    nradial: int = 4,
    nmu_per_bin: int = 1,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Map first, then reproduce FCFC shell/mu-bin and midpoint multipoles."""

    medges = _finite_array(mu_edges, "mu_edges")
    if medges.ndim != 1 or medges[0] != 0.0 or medges[-1] != 1.0 or np.any(np.diff(medges) <= 0.0):
        raise ValueError("mu edges must span [0,1]")
    radii, radial_weights = shell_quadrature(radial_edges, int(nradial))
    mu_nodes, mu_weights = np.polynomial.legendre.leggauss(int(nmu_per_bin))
    mu_lower, mu_upper = medges[:-1, None], medges[1:, None]
    mus = 0.5 * (mu_upper - mu_lower) * mu_nodes[None, :] + 0.5 * (mu_upper + mu_lower)
    normalized_mu_weights = np.broadcast_to(0.5 * mu_weights[None, :], mus.shape)

    smu = np.empty((radii.shape[0], mus.shape[0]), dtype="f8")
    for ishell in range(radii.shape[0]):
        for imu in range(mus.shape[0]):
            total = 0.0
            for iradial, radius in enumerate(radii[ishell]):
                for iangular, mu in enumerate(mus[imu]):
                    transverse = float(radius) * math.sqrt(max(0.0, 1.0 - float(mu) ** 2))
                    parallel = float(radius) * float(mu)
                    total += (
                        float(radial_weights[ishell, iradial])
                        * float(normalized_mu_weights[imu, iangular])
                        * float(evaluate_xi(transverse, parallel))
                    )
            smu[ishell, imu] = total

    mu_midpoint = 0.5 * (medges[:-1] + medges[1:])
    multipoles = {
        ell: (2 * ell + 1) * np.mean(smu * eval_legendre(ell, mu_midpoint)[None, :], axis=1)
        for ell in (0, 2, 4)
    }
    return smu, multipoles


def fcfc_project_smu_vectorized(
    evaluate_xi: Callable[[np.ndarray, np.ndarray], np.ndarray],
    radial_edges: np.ndarray,
    mu_edges: np.ndarray,
    *,
    nradial: int = 4,
    nmu_per_bin: int = 1,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Vectorized equivalent of ``fcfc_project_smu``."""

    medges = _finite_array(mu_edges, "mu_edges")
    if medges.ndim != 1 or medges[0] != 0.0 or medges[-1] != 1.0 or np.any(np.diff(medges) <= 0.0):
        raise ValueError("mu edges must span [0,1]")
    radii, radial_weights = shell_quadrature(radial_edges, int(nradial))
    mu_nodes, mu_weights = np.polynomial.legendre.leggauss(int(nmu_per_bin))
    mu_lower, mu_upper = medges[:-1, None], medges[1:, None]
    mus = 0.5 * (mu_upper - mu_lower) * mu_nodes[None, :] + 0.5 * (mu_upper + mu_lower)
    normalized_mu_weights = np.broadcast_to(0.5 * mu_weights[None, :], mus.shape)

    radius_grid = radii[:, None, :, None]
    mu_grid = mus[None, :, None, :]
    transverse = radius_grid * np.sqrt(np.maximum(0.0, 1.0 - mu_grid**2))
    parallel = radius_grid * mu_grid
    values = np.asarray(evaluate_xi(transverse, parallel), dtype="f8")
    expected_shape = (radii.shape[0], mus.shape[0], radii.shape[1], mus.shape[1])
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        raise ValueError("vectorized xi evaluator returned an invalid array")
    weights = radial_weights[:, None, :, None] * normalized_mu_weights[None, :, None, :]
    smu = np.sum(values * weights, axis=(2, 3))
    mu_midpoint = 0.5 * (medges[:-1] + medges[1:])
    multipoles = {
        ell: (2 * ell + 1) * np.mean(smu * eval_legendre(ell, mu_midpoint)[None, :], axis=1)
        for ell in (0, 2, 4)
    }
    return smu, multipoles
