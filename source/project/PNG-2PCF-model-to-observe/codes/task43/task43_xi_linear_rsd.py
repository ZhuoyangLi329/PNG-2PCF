#!/usr/bin/env python3
"""Independent xi-only linear RSD model for Task 4.3 rawbox validation.

This module reads the frozen historical theory cache but does not modify it or
the production P(k) model.  Continuous angular moments and shell kernels are
evaluated analytically; finite-lattice angular effects are exposed separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import eval_legendre, spherical_jn

from task43_rawbox_numerics import (
    GaussianMetric,
    angular_total_integrals,
    continuous_rsd_poles,
    exact_lattice_modes,
    mode_operators,
    shell_kernel,
)
from task43_rsd_model import DELTA_C


class XiLinearRSDModel:
    """Analytic continuous-angle xi predictor with explicit operator tests."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = Path(cache_path)
        with np.load(self.cache_path, allow_pickle=False) as payload:
            for key in payload.files:
                setattr(self, key, np.asarray(payload[key]))
        self.k_eff = np.asarray(self.k_eff, dtype="f8")
        self.g_nz = np.asarray(self.g_nz, dtype="f8")
        self.pk_dd = np.asarray(self.pk_dd, dtype="f8")
        self.alpha = np.asarray(self.alpha, dtype="f8")
        self.s_edges = np.asarray(self.s_edges, dtype="f8")
        self.ell_values = tuple(int(value) for value in np.asarray(self.ells).ravel())
        if self.ell_values != (0, 2):
            raise ValueError(f"expected frozen ell=(0,2) cache, got {self.ell_values}")
        if np.any(self.k_eff <= 0.0) or np.any(self.g_nz <= 0.0) or np.any(self.pk_dd < 0.0):
            raise ValueError("invalid frozen theory cache")
        self.volume = float(np.asarray(self.volume).item())
        self.boxsize = float(np.asarray(self.boxsize).item())
        self.f_growth = float(np.asarray(self.f_growth).item())
        self.cached_kernels = np.asarray(self.kernels, dtype="f8")
        self.analytic_kernels = np.stack(
            [shell_kernel(self.k_eff, self.s_edges, ell) for ell in self.ell_values]
        )

    def amplitude(self, *, fnl: float, b1: float, p_fixed: float = 1.0) -> np.ndarray:
        bphi = 2.0 * DELTA_C * (float(b1) - float(p_fixed))
        return float(b1) + float(fnl) * bphi * self.alpha

    def evaluate(
        self,
        *,
        fnl: float,
        b1: float,
        sigma_s: float,
        p_fixed: float = 1.0,
        use_cached_kernels: bool = False,
    ) -> dict[int, np.ndarray]:
        amplitude = self.amplitude(fnl=fnl, b1=b1, p_fixed=p_fixed)
        poles = continuous_rsd_poles(
            self.k_eff,
            self.pk_dd,
            amplitude,
            self.f_growth,
            float(sigma_s),
        )
        kernels = self.cached_kernels if use_cached_kernels else self.analytic_kernels
        return {
            ell: ((self.g_nz * poles[ell]) @ kernels[index]) / self.volume
            for index, ell in enumerate(self.ell_values)
        }

    def evaluate_gauss_legendre(
        self,
        *,
        fnl: float,
        b1: float,
        sigma_s: float,
        nmu: int,
        p_fixed: float = 1.0,
        use_cached_kernels: bool = True,
    ) -> dict[int, np.ndarray]:
        if int(nmu) < 2:
            raise ValueError("nmu must be at least 2")
        mu, weight = np.polynomial.legendre.leggauss(int(nmu))
        mu2 = mu**2
        amplitude = self.amplitude(fnl=fnl, b1=b1, p_fixed=p_fixed)
        damping = 1.0 / (
            1.0 + 0.5 * (self.k_eff[:, None] * mu[None, :] * float(sigma_s)) ** 2
        ) ** 2
        pkmu = (
            self.pk_dd[:, None]
            * (amplitude[:, None] + self.f_growth * mu2[None, :]) ** 2
            * damping
        )
        kernels = self.cached_kernels if use_cached_kernels else self.analytic_kernels
        result: dict[int, np.ndarray] = {}
        for index, ell in enumerate(self.ell_values):
            pole = 0.5 * (2 * ell + 1) * np.sum(
                weight[None, :] * pkmu * eval_legendre(ell, mu)[None, :],
                axis=1,
            )
            result[ell] = ((self.g_nz * pole) @ kernels[index]) / self.volume
        return result

    def covariance(
        self,
        *,
        b1: float,
        sigma_s: float,
        nbar: float,
        fnl: float = 0.0,
        p_fixed: float = 1.0,
    ) -> np.ndarray:
        """Return the unfloored analytic Gaussian xi02 covariance."""

        if not np.isfinite(nbar) or nbar <= 0.0:
            raise ValueError("nbar must be positive")
        amplitude = self.amplitude(fnl=fnl, b1=b1, p_fixed=p_fixed)
        angular = angular_total_integrals(
            self.k_eff,
            self.pk_dd,
            amplitude,
            self.f_growth,
            float(sigma_s),
            1.0 / float(nbar),
        )
        nbin = self.s_edges.size - 1
        covariance = np.empty((len(self.ell_values) * nbin,) * 2, dtype="f8")
        for index_a, ell_a in enumerate(self.ell_values):
            for index_b, ell_b in enumerate(self.ell_values):
                radial_weight = (
                    self.g_nz
                    * (2 * ell_a + 1)
                    * (2 * ell_b + 1)
                    * angular[ell_a, ell_b]
                    / self.volume**2
                )
                block = self.analytic_kernels[index_a].T @ (
                    radial_weight[:, None] * self.analytic_kernels[index_b]
                )
                covariance[
                    index_a * nbin : (index_a + 1) * nbin,
                    index_b * nbin : (index_b + 1) * nbin,
                ] = block
        return GaussianMetric(0.5 * (covariance + covariance.T)).covariance

    def realspace_pair_moments(
        self,
        *,
        fnl: float,
        b1: float,
        radial_edges: np.ndarray,
        p_fixed: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """Leading isotropic density/velocity moments on the frozen k support.

        ``v12`` is the signed leading numerator, without a partial ``1+xi``
        resummation.  Velocity moments contain no tracer shot noise or old FoG.
        """

        edges = np.asarray(radial_edges, dtype="f8")
        if (
            edges.ndim != 1
            or edges.size < 2
            or edges[0] < 0.0
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError("invalid radial edges")
        lower, upper = edges[:-1], edges[1:]
        radius = 0.75 * (upper**4 - lower**4) / (upper**3 - lower**3)
        kr = self.k_eff[:, None] * radius[None, :]
        j0 = spherical_jn(0, kr)
        j1 = spherical_jn(1, kr)
        amplitude = self.amplitude(fnl=fnl, b1=b1, p_fixed=p_fixed)
        weighted_power = self.g_nz * self.pk_dd

        xi_kernel = shell_kernel(self.k_eff, edges, 0)
        xi = ((weighted_power * amplitude**2) @ xi_kernel) / self.volume
        v12 = (
            -2.0
            * self.f_growth
            * np.sum(
                (weighted_power * amplitude / self.k_eff)[:, None] * j1,
                axis=0,
            )
            / self.volume
        )
        velocity_weight = weighted_power / self.k_eff**2
        sigma_u2 = self.f_growth**2 * np.sum(velocity_weight) / (3.0 * self.volume)
        psi_parallel = (
            self.f_growth**2
            * np.sum(velocity_weight[:, None] * (j0 - 2.0 * j1 / kr), axis=0)
            / self.volume
        )
        psi_transverse = (
            self.f_growth**2
            * np.sum(velocity_weight[:, None] * (j1 / kr), axis=0)
            / self.volume
        )
        sigma_r2 = 2.0 * (sigma_u2 - psi_parallel)
        sigma_t2 = 2.0 * (sigma_u2 - psi_transverse)
        if (
            np.any(1.0 + xi <= 0.0)
            or np.any(sigma_r2 <= 0.0)
            or np.any(sigma_t2 <= 0.0)
            or not all(np.all(np.isfinite(value)) for value in (xi, v12, sigma_r2, sigma_t2))
        ):
            raise RuntimeError("linear pair moments are non-physical on the requested support")
        return {
            "radial_edges": edges.copy(),
            "radial_volume_centers": radius,
            "xi_real": xi,
            "v12_radial": v12,
            "sigma_r2_central": sigma_r2,
            "sigma_t2_one_component": sigma_t2,
            "sigma_u2_one_component": np.asarray(sigma_u2),
        }

    def lowk_angular_delta(
        self,
        *,
        fnl: float,
        b1: float,
        sigma_s: float,
        k_switch: float,
        p_fixed: float = 1.0,
    ) -> dict[int, np.ndarray]:
        """Return exact-mode minus continuous-angle xi below k_switch."""

        if not np.isfinite(k_switch) or not 0.0 < k_switch <= float(np.max(self.k_eff)):
            raise ValueError("invalid k_switch")
        modes = exact_lattice_modes(self.boxsize, np.asarray([[0.0, float(k_switch)]]))
        _, xi_operator = mode_operators(modes, self.s_edges, ells=self.ell_values)
        power = np.interp(np.log(modes.k), np.log(self.k_eff), self.pk_dd)
        alpha = np.interp(np.log(modes.k), np.log(self.k_eff), self.alpha)
        bphi = 2.0 * DELTA_C * (float(b1) - float(p_fixed))
        amplitude = float(b1) + float(fnl) * bphi * alpha
        damping = 1.0 / (1.0 + 0.5 * (modes.k * modes.mu * float(sigma_s)) ** 2) ** 2
        signal = power * (amplitude + self.f_growth * modes.mu**2) ** 2 * damping
        discrete = xi_operator @ signal

        continuous_poles = continuous_rsd_poles(
            modes.k,
            power,
            amplitude,
            self.f_growth,
            float(sigma_s),
        )
        nbin = self.s_edges.size - 1
        continuous_rows: list[np.ndarray] = []
        for ell in self.ell_values:
            kernel = shell_kernel(modes.k, self.s_edges, ell)
            continuous_rows.append(np.sum(continuous_poles[ell][:, None] * kernel, axis=0) / self.volume)
        continuous = np.concatenate(continuous_rows)
        delta = discrete - continuous
        return {
            ell: delta[index * nbin : (index + 1) * nbin]
            for index, ell in enumerate(self.ell_values)
        }

    def validation_summary(self) -> dict[str, Any]:
        difference = self.cached_kernels - self.analytic_kernels
        return {
            "cache_path": str(self.cache_path),
            "kernel_max_abs_cached_minus_analytic": float(np.max(np.abs(difference))),
            "kernel_relative_l2_cached_minus_analytic": float(
                np.linalg.norm(difference) / np.linalg.norm(self.analytic_kernels)
            ),
            "frozen_cache_unchanged": True,
        }
