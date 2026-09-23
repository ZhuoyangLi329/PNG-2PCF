#!/usr/bin/env python3
"""Strict common-mode rawbox P/xi fits with the canonical BAO mask.

The real-space suite fits P, xi and their joint likelihood.  The redshift-
space suite fits P0+P2, xi0+xi2 and their joint likelihood.  The standard
xi selection is 50 <= s < 350 Mpc/h with 80 <= s < 120 Mpc/h removed;
40 Mpc/h remains available as an explicit comparison option.

The P/xi covariance is constructed as a genuine common-mode covariance.  The
observed P-bin lattice modes generate P-P, P-xi and their contribution to
xi-xi together.  Positive-semidefinite xi covariance from every mode outside
the observed P bins is then added to xi-xi.  No empirical cross rescaling or
eigenvalue repair is allowed.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any, Callable

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import (
    ExactPeriodicPk0Model,
    SN0_SCALE,
    load_pk_x25,
    matching_indices,
    set_affinity,
    split_rhat,
)
from task43_fit_rsd_rawbox_x25 import FastRSDModel, load_x25
from task43_rawbox_numerics import (
    GaussianMetric,
    RAWBOX_FIT_EDGES,
    canonical_correlations,
    exact_lattice_modes,
    gaussian_mode_covariance,
    mode_operators,
)
from task43_rsd_common import (
    OUTPUT_ROOT,
    PHASES,
    P_FIXED,
    S_EDGES,
    atomic_savez,
    atomic_write_json,
    rawbox_xi_primary_mask,
    sha256_file,
)
from task43_rsd_model import DELTA_C, FullDiscreteRSDModel, build_cache


DEFAULT_ROOT = OUTPUT_ROOT / "rawbox" / "kmax0p08_smin50_v1"
P02_DIR = OUTPUT_ROOT / "rawbox" / "pk"
NPHASE = len(PHASES)
REAL_NAMES = ("fNL", "b1")
RSD_NAMES = ("fNL", "b1", "sigma_s")
RSD_P_NAMES = ("fNL", "b1", "sigma_s", "sn0")
REAL_BOUNDS = (
    np.asarray([-500.0, 0.2], dtype="f8"),
    np.asarray([500.0, 10.0], dtype="f8"),
)
RSD_BOUNDS = (
    np.asarray([-500.0, 0.5, 0.0], dtype="f8"),
    np.asarray([500.0, 5.0, 30.0], dtype="f8"),
)
RSD_P_BOUNDS = (
    np.asarray([-500.0, 0.5, 0.0, -1.0], dtype="f8"),
    np.asarray([500.0, 5.0, 30.0, 1.0], dtype="f8"),
)
REAL_STARTS = (
    np.asarray([0.0, 2.6]),
    np.asarray([-100.0, 2.3]),
    np.asarray([100.0, 2.9]),
)
RSD_STARTS = (
    np.asarray([0.0, 2.55, 8.0]),
    np.asarray([-80.0, 2.4, 2.0]),
    np.asarray([80.0, 2.8, 12.0]),
    np.asarray([0.0, 2.5, 20.0]),
)
RSD_P_STARTS = tuple(np.concatenate((start, [sn0])) for start, sn0 in zip(RSD_STARTS, (0.0, 0.2, -0.2, 0.5)))
INITIAL_SCALE = {
    REAL_NAMES: np.asarray([4.0, 0.015]),
    RSD_NAMES: np.asarray([4.0, 0.015, 0.12]),
    RSD_P_NAMES: np.asarray([4.0, 0.015, 0.12, 0.025]),
}
_POOL_LOG_PROBABILITY: Callable[[np.ndarray], float] | None = None


def fit_edges_for_kmax(kmax: float) -> np.ndarray:
    """Return the frozen rawbox P(k) bins below the requested fit cutoff.

    The native Task 4.3 selection has sparse DESI-PNG bins.  A lower cutoff
    therefore removes complete bins rather than clipping a bin edge.  The
    0.08 contract retains the [0.003, 0.005] rawbox fundamental-mode bin and
    the canonical tail through [0.077, 0.079].
    """

    value = float(kmax)
    if not np.isclose(value, 0.08, rtol=0.0, atol=1.0e-12) and not np.isclose(
        value, 0.10, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("--pk-kmax must be 0.08 or 0.10")
    selected = RAWBOX_FIT_EDGES[RAWBOX_FIT_EDGES[:, 1] <= value - 1.0e-12]
    if selected.shape[0] < 1:
        raise RuntimeError(f"no rawbox P(k) bins survive kmax={value}")
    return np.asarray(selected, dtype="f8")


def xi_mask_for_smin(centers: np.ndarray, smin: float) -> np.ndarray:
    """Apply the standard 40/50--350 Mpc/h range and 80--120 BAO mask."""

    values = np.asarray(centers, dtype="f8")
    lower = float(smin)
    if lower not in (40.0, 50.0):
        raise ValueError("--smin must be 40 or 50 Mpc/h")
    return (values >= lower) & (values < 350.0) & ~((values >= 80.0) & (values < 120.0))


def _pool_log_probability(theta: np.ndarray) -> float:
    if _POOL_LOG_PROBABILITY is None:
        raise RuntimeError("MCMC worker likelihood was not initialized")
    return _POOL_LOG_PROBABILITY(theta)


class FastP02Model:
    """Exact parent-mode P0/P2 bin averages with a sigma_s spline."""

    def __init__(self, exact: FullDiscreteRSDModel, edges: np.ndarray, *, sigma_step: float = 0.05) -> None:
        self.inner = ExactPeriodicPk0Model(exact, edges, sigma_step=sigma_step)
        self.legendre2 = 0.5 * (3.0 * self.inner.mu2 - 1.0)
        self.p2_basis = np.empty_like(self.inner.basis)
        for index, sigma_s in enumerate(self.inner.sigma_grid):
            damping = 1.0 / (1.0 + 0.5 * self.inner.k**2 * self.inner.mu2 * sigma_s**2) ** 2
            columns = (
                self.inner.pk_dd * damping,
                self.inner.pk_dd * self.inner.alpha * damping,
                self.inner.pk_dd * self.inner.alpha**2 * damping,
                self.inner.pk_dd * self.inner.mu2 * damping,
                self.inner.pk_dd * self.inner.alpha * self.inner.mu2 * damping,
                self.inner.pk_dd * self.inner.mu2**2 * damping,
            )
            for column, values in enumerate(columns):
                self.p2_basis[index, :, column] = np.bincount(
                    self.inner.bin_id,
                    weights=5.0 * self.legendre2 * values,
                    minlength=self.inner.edges.shape[0],
                ) / self.inner.counts
        self.p2_spline = CubicSpline(self.inner.sigma_grid, self.p2_basis, axis=0)

    @staticmethod
    def _coefficients(theta: np.ndarray, growth: float) -> tuple[np.ndarray, float]:
        fnl, b1, sigma_s, _ = map(float, np.asarray(theta, dtype="f8")[:4])
        q = fnl * 2.0 * DELTA_C * (b1 - P_FIXED)
        coefficients = np.asarray([b1 * b1, 2.0 * b1 * q, q * q, 2.0 * b1 * growth, 2.0 * q * growth, growth * growth])
        return coefficients, sigma_s

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        coefficients, sigma_s = self._coefficients(theta, self.inner.f_growth)
        p0 = self.inner.evaluate(theta)
        p2 = np.asarray(self.p2_spline(sigma_s), dtype="f8") @ coefficients
        return np.concatenate((p0, p2))

    def direct(self, theta: np.ndarray) -> np.ndarray:
        fnl, b1, sigma_s, sn0 = map(float, np.asarray(theta, dtype="f8")[:4])
        q = fnl * 2.0 * DELTA_C * (b1 - P_FIXED)
        amplitude = b1 + q * self.inner.alpha
        damping = 1.0 / (1.0 + 0.5 * self.inner.k**2 * self.inner.mu2 * sigma_s**2) ** 2
        signal = self.inner.pk_dd * (amplitude + self.inner.f_growth * self.inner.mu2) ** 2 * damping
        p0 = np.bincount(self.inner.bin_id, weights=signal, minlength=self.inner.edges.shape[0]) / self.inner.counts
        p0 = p0 + sn0 * SN0_SCALE
        p2 = np.bincount(
            self.inner.bin_id,
            weights=5.0 * self.legendre2 * signal,
            minlength=self.inner.edges.shape[0],
        ) / self.inner.counts
        return np.concatenate((p0, p2))

    def validate(self) -> dict[str, Any]:
        trials = (
            np.asarray([0.0, 2.55, 8.0, 0.0]),
            np.asarray([-75.0, 2.2, 3.37, 0.2]),
            np.asarray([80.0, 2.8, 12.43, -0.3]),
            np.asarray([15.0, 2.5, 0.07, 0.1]),
            np.asarray([-20.0, 2.6, 29.93, -0.1]),
        )
        rows = []
        for theta in trials:
            fast, direct = self.evaluate(theta), self.direct(theta)
            rows.append(
                {
                    "theta": theta.tolist(),
                    "max_abs": float(np.max(np.abs(fast - direct))),
                    "relative_l2": float(np.linalg.norm(fast - direct) / np.linalg.norm(direct)),
                }
            )
        maximum = max(row["relative_l2"] for row in rows)
        return {"status": "pass" if maximum < 1.0e-8 else "fail", "max_relative_l2": maximum, "trials": rows}


def _inside_observed_p_bins(k: np.ndarray) -> np.ndarray:
    kval = np.asarray(k, dtype="f8")
    return np.any(
        (kval[:, None] >= RAWBOX_FIT_EDGES[None, :, 0])
        & (kval[:, None] < RAWBOX_FIT_EDGES[None, :, 1]),
        axis=1,
    )


def common_mode_covariance(
    exact: FullDiscreteRSDModel,
    modes,
    mask: np.ndarray,
    *,
    nbar: float,
    b1: float,
    sigma_s: float | None,
) -> dict[str, Any]:
    """Return strict P, xi and joint blocks with shared observed P modes."""

    ells = (0,) if sigma_s is None else (0, 2)
    p_operator, xi_operator_full = mode_operators(modes, S_EDGES, ells=ells)
    xi_row_ids = np.concatenate(
        [index * mask.size + np.flatnonzero(mask) for index in range(len(ells))]
    )
    xi_operator = xi_operator_full[xi_row_ids]
    power_modes = np.interp(np.log(modes.k), np.log(exact.k_eff), exact.pk_dd)
    shot = 1.0 / float(nbar)
    if sigma_s is None:
        total_modes = power_modes * float(b1) ** 2 + shot
    else:
        mu2 = modes.mu**2
        damping = 1.0 / (1.0 + 0.5 * (modes.k * modes.mu * float(sigma_s)) ** 2) ** 2
        total_modes = power_modes * (float(b1) + float(exact.f_growth) * mu2) ** 2 * damping + shot
    low = gaussian_mode_covariance(np.vstack((p_operator, xi_operator)), total_modes)
    np_rows = p_operator.shape[0]
    c_pp = low[:np_rows, :np_rows]
    c_xp = low[np_rows:, :np_rows]
    c_xx_low = low[np_rows:, np_rows:]

    outside = ~_inside_observed_p_bins(exact.k_eff)
    nselected = int(np.count_nonzero(mask))
    tail = np.zeros_like(c_xx_low)
    if sigma_s is None:
        kernel = np.asarray(exact.kernels[0], dtype="f8")[:, mask]
        weights = (
            2.0
            * np.asarray(exact.g_nz, dtype="f8")[outside]
            * (np.asarray(exact.pk_dd, dtype="f8")[outside] * float(b1) ** 2 + shot) ** 2
            / float(exact.volume) ** 2
        )
        tail = kernel[outside].T @ (weights[:, None] * kernel[outside])
    else:
        x = (exact.k_eff[:, None] * exact.mu[None, :] * float(sigma_s)) ** 2
        damping = 1.0 / (1.0 + 0.5 * x) ** 2
        signal = exact.pk_dd[:, None] * (
            float(b1) + float(exact.f_growth) * exact.mu2[None, :]
        ) ** 2 * damping
        total2 = (signal + shot) ** 2
        kernels = {ell: np.asarray(exact.kernels[index], dtype="f8")[:, mask] for index, ell in enumerate(ells)}
        for ia, ell_a in enumerate(ells):
            for ib, ell_b in enumerate(ells):
                angular = np.sum(
                    exact.wmu[None, :]
                    * total2
                    * exact.legendre[ell_a][None, :]
                    * exact.legendre[ell_b][None, :],
                    axis=1,
                )
                weights = (
                    exact.g_nz[outside]
                    * (2 * ell_a + 1)
                    * (2 * ell_b + 1)
                    * angular[outside]
                    / float(exact.volume) ** 2
                )
                tail[
                    ia * nselected : (ia + 1) * nselected,
                    ib * nselected : (ib + 1) * nselected,
                ] = kernels[ell_a][outside].T @ (weights[:, None] * kernels[ell_b][outside])
    c_xx = 0.5 * (c_xx_low + tail + (c_xx_low + tail).T)
    joint = np.block([[c_pp, c_xp.T], [c_xp, c_xx]])
    metric_p = GaussianMetric(c_pp)
    metric_x = GaussianMetric(c_xx)
    metric_joint = GaussianMetric(joint)
    canonical = canonical_correlations(c_pp, c_xx, c_xp)
    return {
        "pp": metric_p.covariance,
        "xx": metric_x.covariance,
        "xp": c_xp,
        "joint": metric_joint.covariance,
        "xx_low": c_xx_low,
        "xx_tail": tail,
        "canonical_correlations": canonical,
        "correlation_eigenvalue_min": {
            "pp": float(metric_p.eigenvalues[0]),
            "xx": float(metric_x.eigenvalues[0]),
            "joint": float(metric_joint.eigenvalues[0]),
        },
    }


def load_p02_x25(reference_p0: dict[str, Any]) -> dict[str, Any]:
    stacks0, stacks2, hashes, bridges = [], [], [], []
    selected = np.asarray(reference_p0["fine_indices"], dtype="i8")
    measured_selection = None
    for index, phase in enumerate(PHASES):
        path = P02_DIR / f"task43_rsd_rawbox_p02_AbacusSummit_base_c000_{phase}_mmin1p4e13_mesh400.npz"
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        digest = sha256_file(path)
        if metadata.get("status") != "pass" or metadata.get("output_sha256") != digest:
            raise RuntimeError(f"unvalidated P02 measurement: {path}")
        recorded_bridge = metadata.get("p0_bridge_vs_audited", {})
        if (
            float(recorded_bridge.get("rel_l2", np.inf)) >= 1.0e-14
            or float(recorded_bridge.get("max_abs_diff", np.inf)) >= 1.0e-9
        ):
            raise RuntimeError(f"recorded P0 bridge failed for {phase}: {recorded_bridge}")
        with np.load(path, allow_pickle=False) as payload:
            this_selection = matching_indices(np.asarray(payload["k_edges"], dtype="f8"), RAWBOX_FIT_EDGES)
            if measured_selection is None:
                measured_selection = this_selection
            elif not np.array_equal(this_selection, measured_selection):
                raise RuntimeError(f"P02 selected bins changed for {phase}")
            p0 = np.asarray(payload["pk0"], dtype="f8")[this_selection]
            p2 = np.asarray(payload["pk2"], dtype="f8")[this_selection]
        reference = np.asarray(reference_p0["pk0_rsd"], dtype="f8")[index]
        difference = p0 - reference
        relative_l2 = float(np.linalg.norm(difference) / np.linalg.norm(reference))
        maximum_abs = float(np.max(np.abs(difference)))
        if relative_l2 >= 1.0e-14 or maximum_abs >= 1.0e-9:
            raise RuntimeError(f"P02/P0 audited bridge changed for {phase}")
        stacks0.append(p0)
        stacks2.append(p2)
        hashes.append(digest)
        bridges.append(
            {
                "phase": phase,
                "array_equal": bool(np.array_equal(p0, reference)),
                "relative_l2": relative_l2,
                "max_abs_diff": maximum_abs,
            }
        )
    if not np.array_equal(np.asarray(selected, dtype="i8"), np.asarray(measured_selection, dtype="i8")):
        raise RuntimeError("analytic and measured P02 selections differ")
    return {"pk0": np.stack(stacks0), "pk2": np.stack(stacks2), "hashes": hashes, "p0_bridges": bridges}


def fit_map(
    data: np.ndarray,
    evaluate: Callable[[np.ndarray], np.ndarray],
    covariance: np.ndarray,
    *,
    names: tuple[str, ...],
    bounds: tuple[np.ndarray, np.ndarray],
    starts: tuple[np.ndarray, ...],
) -> tuple[dict[str, Any], GaussianMetric]:
    metric = GaussianMetric(covariance)

    def residual(theta: np.ndarray) -> np.ndarray:
        return metric.residual(np.asarray(data, dtype="f8") - np.asarray(evaluate(theta), dtype="f8"))

    solutions = [
        least_squares(
            residual,
            start,
            bounds=bounds,
            max_nfev=3000,
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
        )
        for start in starts
    ]
    best = min(solutions, key=lambda result: float(result.fun @ result.fun))
    theta = np.asarray(best.x, dtype="f8")
    chi2_single = float(best.fun @ best.fun)
    dof = int(np.asarray(data).size - theta.size)
    return (
        {
            "theta": {name: float(value) for name, value in zip(names, theta, strict=True)},
            "chi2_observed_x25_mean_with_single_realization_covariance": chi2_single,
            "dof": dof,
            "pte_observed_x25_mean_with_single_realization_covariance": float(
                chi2_distribution.sf(chi2_single, dof)
            ),
            "success": bool(best.success),
            "message": str(best.message),
        },
        metric,
    )


def run_chain(
    data: np.ndarray,
    evaluate: Callable[[np.ndarray], np.ndarray],
    metric: GaussianMetric,
    map_summary: dict[str, Any],
    *,
    names: tuple[str, ...],
    bounds: tuple[np.ndarray, np.ndarray],
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    nworkers: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    lower, upper = bounds

    def log_probability(theta: np.ndarray) -> float:
        value = np.asarray(theta, dtype="f8")
        if np.any(value < lower) or np.any(value > upper):
            return -np.inf
        residual = metric.residual(np.asarray(data, dtype="f8") - np.asarray(evaluate(value), dtype="f8"))
        return -0.5 * float(residual @ residual)

    center = np.asarray([map_summary["theta"][name] for name in names], dtype="f8")
    rng = np.random.default_rng(int(seed))
    initial = center[None, :] + rng.normal(size=(int(nwalkers), len(names))) * INITIAL_SCALE[names][None, :]
    initial = np.clip(initial, lower + 1.0e-7, upper - 1.0e-7)
    np.random.seed(int(seed))
    global _POOL_LOG_PROBABILITY
    _POOL_LOG_PROBABILITY = log_probability
    try:
        if int(nworkers) == 1:
            sampler = emcee.EnsembleSampler(int(nwalkers), len(names), log_probability)
            sampler.run_mcmc(initial, int(nsteps), progress=False)
        else:
            with mp.get_context("fork").Pool(processes=int(nworkers)) as pool:
                sampler = emcee.EnsembleSampler(int(nwalkers), len(names), _pool_log_probability, pool=pool)
                sampler.run_mcmc(initial, int(nsteps), progress=False)
    finally:
        _POOL_LOG_PROBABILITY = None
    chain = np.asarray(sampler.get_chain(discard=int(burnin)), dtype="f8")
    logp = np.asarray(sampler.get_log_prob(discard=int(burnin)), dtype="f8")
    flat, flat_logp = chain.reshape(-1, len(names)), logp.reshape(-1)
    quantiles = np.percentile(flat, [16.0, 50.0, 84.0], axis=0)
    sigma68 = 0.5 * (quantiles[2] - quantiles[0])
    tau = np.asarray(emcee.autocorr.integrated_time(chain, quiet=True, tol=0), dtype="f8")
    rhat = split_rhat(chain)
    half = chain.shape[0] // 2
    first, second = chain[:half].reshape(-1, len(names)), chain[-half:].reshape(-1, len(names))
    half_shift = np.abs(np.median(first, axis=0) - np.median(second, axis=0)) / sigma68
    imax = int(np.argmax(flat_logp))
    summary = {
        "posterior": {
            name: {
                "q16": float(quantiles[0, index]),
                "q50": float(quantiles[1, index]),
                "q84": float(quantiles[2, index]),
                "sigma68": float(sigma68[index]),
                "mean": float(np.mean(flat[:, index])),
                "std": float(np.std(flat[:, index], ddof=1)),
            }
            for index, name in enumerate(names)
        },
        "map_chain": {name: float(flat[imax, index]) for index, name in enumerate(names)},
        "map_chain_log_probability": float(flat_logp[imax]),
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
        "nwalkers": int(nwalkers),
        "nsteps": int(nsteps),
        "burnin": int(burnin),
        "parallel_workers": int(nworkers),
        "tau": {name: float(tau[index]) for index, name in enumerate(names)},
        "split_rhat": {name: float(rhat[index]) for index, name in enumerate(names)},
        "postburn_length_over_tau": {name: float(chain.shape[0] / tau[index]) for index, name in enumerate(names)},
        "half_chain_shift_sigma": {name: float(half_shift[index]) for index, name in enumerate(names)},
    }
    summary["gates"] = {
        "split_rhat_max_below_1p01": bool(np.max(rhat) < 1.01),
        "postburn_length_min_above_50tau": bool(np.min(chain.shape[0] / tau) > 50.0),
        "half_chain_shift_max_below_0p1sigma": bool(np.max(half_shift) < 0.1),
    }
    return summary, chain, logp


def empirical_covariance_diagnostics(p_stack: np.ndarray, x_stack: np.ndarray, covariance: dict[str, Any]) -> dict[str, Any]:
    empirical = np.cov(np.hstack((p_stack, x_stack)), rowvar=False, ddof=1)
    np_rows = p_stack.shape[1]
    pp_emp = empirical[:np_rows, :np_rows]
    xp_emp = empirical[np_rows:, :np_rows]
    xx_emp = empirical[np_rows:, np_rows:]
    analytic_corr = covariance["xp"] / np.outer(np.sqrt(np.diag(covariance["xx"])), np.sqrt(np.diag(covariance["pp"])))
    empirical_corr = xp_emp / np.outer(np.sqrt(np.diag(xx_emp)), np.sqrt(np.diag(pp_emp)))
    denom = float(np.sum(covariance["xp"] ** 2))
    return {
        "pp_sample_sigma_over_analytic": (np.sqrt(np.diag(pp_emp) / np.diag(covariance["pp"]))).tolist(),
        "xx_sample_sigma_over_analytic": (np.sqrt(np.diag(xx_emp) / np.diag(covariance["xx"]))).tolist(),
        "cross_matrix_correlation": float(np.corrcoef(xp_emp.ravel(), covariance["xp"].ravel())[0, 1]),
        "cross_least_squares_scale_empirical_over_analytic": float(np.sum(xp_emp * covariance["xp"]) / denom),
        "cross_correlation_rms_difference": float(np.sqrt(np.mean((empirical_corr - analytic_corr) ** 2))),
        "cross_correlation_absmax_analytic": float(np.max(np.abs(analytic_corr))),
        "cross_correlation_absmax_empirical": float(np.max(np.abs(empirical_corr))),
    }


def _summary_for_covariance(covariance: dict[str, Any]) -> dict[str, Any]:
    return {
        "shapes": {key: list(np.asarray(covariance[key]).shape) for key in ("pp", "xx", "xp", "joint")},
        "correlation_eigenvalue_min": covariance["correlation_eigenvalue_min"],
        "canonical_correlations": np.asarray(covariance["canonical_correlations"]).tolist(),
        "canonical_correlation_max": float(np.max(covariance["canonical_correlations"])),
        "xi_low_mode_variance_fraction": (
            np.diag(covariance["xx_low"]) / np.diag(covariance["xx"])
        ).tolist(),
        "automatic_eigenvalue_floor": False,
        "cross_rescaling": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--pk-kmax", type=float, choices=(0.08, 0.10), default=0.08)
    parser.add_argument("--smin", type=float, choices=(40.0, 50.0), default=50.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if not 1 <= int(args.threads) <= 8:
        raise ValueError("--threads must be in [1, 8]")
    if int(args.burnin) >= int(args.nsteps):
        raise ValueError("--burnin must be smaller than --nsteps")
    nsteps = 500 if args.smoke else int(args.nsteps)
    burnin = 100 if args.smoke else int(args.burnin)
    out_root = Path(args.out_root) / ("smoke" if args.smoke else "")
    audit_json = out_root / "task43_rawbox_standard_joint_baomask80_120_v1.json"
    covariance_npz = out_root / "task43_rawbox_standard_joint_baomask80_120_covariance_v1.npz"
    if audit_json.exists():
        raise FileExistsError(f"immutable audit exists: {audit_json}")

    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))
    code_hash = sha256_file(Path(__file__))
    # The standard contract is kmax=0.08 and smin=50; alternate cuts remain
    # available explicitly for controlled comparisons.
    global RAWBOX_FIT_EDGES
    RAWBOX_FIT_EDGES = fit_edges_for_kmax(float(args.pk_kmax))
    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    mask = xi_mask_for_smin(centers, float(args.smin))
    excluded = centers[(centers >= float(args.smin)) & (centers < 350.0) & ~mask]
    expected_xi_count = 26 if np.isclose(float(args.smin), 50.0) else 27
    if int(np.count_nonzero(mask)) != expected_xi_count or not np.array_equal(
        excluded, [85.0, 95.0, 105.0, 115.0]
    ):
        raise RuntimeError("canonical BAO mask changed")

    cache_path = build_cache(zeff=0.725, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000")
    exact = FullDiscreteRSDModel(cache_path, nmu=64)
    modes = exact_lattice_modes(float(exact.boxsize), RAWBOX_FIT_EDGES)
    pk = load_pk_x25(RAWBOX_FIT_EDGES)
    if not np.array_equal(modes.counts, np.asarray(pk["nmodes"], dtype="f8")):
        raise RuntimeError("exact lattice mode counts differ from the measurements")
    if not np.allclose(modes.k_mean, np.asarray(pk["k"], dtype="f8"), rtol=0.0, atol=1.0e-14):
        raise RuntimeError("exact lattice mean k differs from the measurements")
    p02 = load_p02_x25(pk)
    xi, xi_metadata, xi_hashes = load_x25()
    nbar = float(np.mean(np.asarray(pk["nbar"], dtype="f8")))
    xi_nbar = float(np.mean([row["nbar_h3_mpc3_from_npz"] for row in xi_metadata]))
    if not np.isclose(nbar, xi_nbar, rtol=0.0, atol=1.0e-15):
        raise RuntimeError("P and xi number densities differ")

    pk_mode_power = np.interp(np.log(modes.k), np.log(exact.k_eff), exact.pk_dd)
    pk_mode_alpha = np.interp(np.log(modes.k), np.log(exact.k_eff), exact.alpha)
    real_p_basis = np.stack(
        [
            np.bincount(modes.bin_id, weights=pk_mode_power * pk_mode_alpha**power, minlength=RAWBOX_FIT_EDGES.shape[0])
            / modes.counts
            for power in range(3)
        ]
    )
    real_x_basis = np.stack(
        [
            (exact.g_nz * exact.pk_dd * exact.alpha**power) @ exact.kernels[0] / float(exact.volume)
            for power in range(3)
        ]
    )

    def evaluate_real_basis(theta: np.ndarray, basis: np.ndarray) -> np.ndarray:
        fnl, b1 = map(float, np.asarray(theta, dtype="f8")[:2])
        q = fnl * 2.0 * DELTA_C * (b1 - P_FIXED)
        return b1**2 * basis[0] + 2.0 * b1 * q * basis[1] + q**2 * basis[2]

    evaluate_real_p = lambda theta: evaluate_real_basis(theta, real_p_basis)
    evaluate_real_x = lambda theta: evaluate_real_basis(theta, real_x_basis) [mask]
    evaluate_real_joint = lambda theta: np.concatenate((evaluate_real_p(theta), evaluate_real_x(theta)))
    data_real_p = np.mean(np.asarray(pk["pk0_real"], dtype="f8"), axis=0)
    data_real_x = np.mean(np.asarray(xi["xi0_real"], dtype="f8"), axis=0)[mask]
    data_real_joint = np.concatenate((data_real_p, data_real_x))

    covariance_real_initial = common_mode_covariance(exact, modes, mask, nbar=nbar, b1=2.5, sigma_s=None)
    real_prefit, _ = fit_map(
        data_real_joint,
        evaluate_real_joint,
        covariance_real_initial["joint"],
        names=REAL_NAMES,
        bounds=REAL_BOUNDS,
        starts=REAL_STARTS,
    )
    covariance_real = common_mode_covariance(
        exact,
        modes,
        mask,
        nbar=nbar,
        b1=float(real_prefit["theta"]["b1"]),
        sigma_s=None,
    )

    p02_model = FastP02Model(exact, RAWBOX_FIT_EDGES)
    p02_validation = p02_model.validate()
    if p02_validation["status"] != "pass":
        raise RuntimeError("P02 spline validation failed")
    xi_rsd_model = FastRSDModel(exact, sigma_step=0.05)
    xi_rsd_validation = xi_rsd_model.validate()
    if xi_rsd_validation["status"] != "pass":
        raise RuntimeError("xi02 spline validation failed")
    data_rsd_p = np.concatenate((np.mean(p02["pk0"], axis=0), np.mean(p02["pk2"], axis=0)))
    data_rsd_x = np.concatenate(
        (
            np.mean(np.asarray(xi["xi0_rsd"], dtype="f8"), axis=0)[mask],
            np.mean(np.asarray(xi["xi2_rsd"], dtype="f8"), axis=0)[mask],
        )
    )

    def evaluate_rsd_x(theta: np.ndarray) -> np.ndarray:
        prediction = xi_rsd_model.evaluate(np.asarray(theta, dtype="f8")[:3])
        return np.concatenate((np.asarray(prediction[0])[mask], np.asarray(prediction[2])[mask]))

    evaluate_rsd_p = lambda theta: p02_model.evaluate(theta)
    evaluate_rsd_joint = lambda theta: np.concatenate((evaluate_rsd_p(theta), evaluate_rsd_x(theta)))
    data_rsd_joint = np.concatenate((data_rsd_p, data_rsd_x))
    covariance_rsd = common_mode_covariance(exact, modes, mask, nbar=nbar, b1=2.55, sigma_s=8.0)

    specifications = (
        ("real_p", data_real_p, evaluate_real_p, covariance_real["pp"], REAL_NAMES, REAL_BOUNDS, REAL_STARTS),
        ("real_xi", data_real_x, evaluate_real_x, covariance_real["xx"], REAL_NAMES, REAL_BOUNDS, REAL_STARTS),
        ("real_joint", data_real_joint, evaluate_real_joint, covariance_real["joint"], REAL_NAMES, REAL_BOUNDS, REAL_STARTS),
        ("rsd_p02", data_rsd_p, evaluate_rsd_p, covariance_rsd["pp"], RSD_P_NAMES, RSD_P_BOUNDS, RSD_P_STARTS),
        ("rsd_xi02", data_rsd_x, evaluate_rsd_x, covariance_rsd["xx"], RSD_NAMES, RSD_BOUNDS, RSD_STARTS),
        ("rsd_joint", data_rsd_joint, evaluate_rsd_joint, covariance_rsd["joint"], RSD_P_NAMES, RSD_P_BOUNDS, RSD_P_STARTS),
    )

    results: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for index, (name, data, evaluate, covariance, names, bounds, starts) in enumerate(specifications):
        map_summary, metric = fit_map(data, evaluate, covariance, names=names, bounds=bounds, starts=starts)
        predictions[name] = np.asarray(evaluate(np.asarray([map_summary["theta"][key] for key in names])), dtype="f8")
        fit_dir = out_root / "fits" / name
        fit_npz = fit_dir / "samples.npz"
        fit_json = fit_dir / "summary.json"
        if fit_npz.exists() != fit_json.exists():
            raise RuntimeError(f"partial resumable output for {name}")
        if fit_npz.exists():
            existing = json.loads(fit_json.read_text(encoding="utf-8"))
            if (
                existing.get("status") != "pass"
                or existing.get("code_sha256") != code_hash
                or existing.get("output_npz_sha256") != sha256_file(fit_npz)
                or existing.get("nsteps") != nsteps
                or existing.get("burnin") != burnin
            ):
                raise RuntimeError(f"resumable output contract failed for {name}")
            results[name] = existing["result"]
            print(json.dumps({"variant": name, "status": "resumed"}), flush=True)
            continue
        chain_summary, chain, logp = run_chain(
            data,
            evaluate,
            metric,
            map_summary,
            names=names,
            bounds=bounds,
            nwalkers=int(args.nwalkers),
            nsteps=nsteps,
            burnin=burnin,
            seed=int(args.seed) + 100 * index,
            nworkers=int(args.threads),
        )
        result = {"parameter_names": list(names), "map": map_summary, "mcmc": chain_summary}
        atomic_savez(
            fit_npz,
            parameter_names=np.asarray(names),
            chain_by_step=chain,
            log_probability_by_step=logp,
            theta_maximum_likelihood=np.asarray([map_summary["theta"][key] for key in names]),
            prediction_maximum_likelihood=predictions[name],
            data=np.asarray(data),
            covariance_single=np.asarray(covariance),
        )
        atomic_write_json(
            fit_json,
            {
                "task": "task43_run_rawbox_joint_baomask_v1",
                "variant": name,
                "status": "pass" if args.smoke or all(chain_summary["gates"].values()) else "validation_failed",
                "result": result,
                "nsteps": nsteps,
                "burnin": burnin,
                "code_sha256": code_hash,
                "output_npz": str(fit_npz),
                "output_npz_sha256": sha256_file(fit_npz),
            },
        )
        results[name] = result
        print(
            json.dumps(
                {
                    "variant": name,
                    "fNL": chain_summary["posterior"]["fNL"],
                    "map": map_summary,
                    "gates": chain_summary["gates"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    p_stack_real = np.asarray(pk["pk0_real"], dtype="f8")
    x_stack_real = np.asarray(xi["xi0_real"], dtype="f8")[:, mask]
    p_stack_rsd = np.concatenate((p02["pk0"], p02["pk2"]), axis=1)
    x_stack_rsd = np.concatenate(
        (np.asarray(xi["xi0_rsd"], dtype="f8")[:, mask], np.asarray(xi["xi2_rsd"], dtype="f8")[:, mask]),
        axis=1,
    )
    numerical_gates = {
        "canonical_mask": bool(
            np.count_nonzero(mask) == expected_xi_count
            and np.array_equal(excluded, [85.0, 95.0, 105.0, 115.0])
        ),
        "p02_model": bool(p02_validation["status"] == "pass"),
        "xi02_model": bool(xi_rsd_validation["status"] == "pass"),
        "real_joint_strict_spd": bool(covariance_real["correlation_eigenvalue_min"]["joint"] > 1.0e-12),
        "rsd_joint_strict_spd": bool(covariance_rsd["correlation_eigenvalue_min"]["joint"] > 1.0e-12),
        "real_canonical_correlation_below_one": bool(np.max(covariance_real["canonical_correlations"]) < 1.0),
        "rsd_canonical_correlation_below_one": bool(np.max(covariance_rsd["canonical_correlations"]) < 1.0),
    }
    if not args.smoke:
        for name in results:
            for gate, passed in results[name]["mcmc"]["gates"].items():
                numerical_gates[f"{name}_{gate}"] = bool(passed)
    status = "pass" if all(numerical_gates.values()) else "validation_failed"
    scientific_closure = {
        name: {
            "single_realization_covariance_pte_above_0p05": bool(
                result["map"]["pte_observed_x25_mean_with_single_realization_covariance"] > 0.05
            ),
            "pte_observed_x25_mean_with_single_realization_covariance": float(
                result["map"]["pte_observed_x25_mean_with_single_realization_covariance"]
            ),
        }
        for name, result in results.items()
    }
    summary = {
        "task": "task43_run_rawbox_joint_baomask_v1",
        "status": status,
        "classification": "standard rawbox real P+xi and RSD P02+xi02 common-mode joint test",
        "fit_contract": {
            "pk_kmax_h_mpc": float(args.pk_kmax),
            "pk_bin_count_per_pole": int(RAWBOX_FIT_EDGES.shape[0]),
            "pk_fit_edges_h_mpc": RAWBOX_FIT_EDGES.tolist(),
            "xi_smin_mpc_h": float(args.smin),
            "xi_smax_mpc_h": 350.0,
            "xi_bin_count_per_pole": int(np.count_nonzero(mask)),
        },
        "mask_policy": {
            "full_range_mpc_h": [float(args.smin), 350.0],
            "excluded_half_open_range_mpc_h": [80.0, 120.0],
            "selected_centers_mpc_h": centers[mask].tolist(),
            "excluded_centers_mpc_h": excluded.tolist(),
            "same_mask_for_xi0_and_xi2": True,
        },
        "parameter_contract": {
            "real": list(REAL_NAMES),
            "rsd_p02": list(RSD_P_NAMES),
            "rsd_xi02": list(RSD_NAMES),
            "rsd_joint": list(RSD_P_NAMES),
            "sn0_definition": "dimensionless nuisance multiplied by 1e4 and applied to P0 only",
        },
        "covariance_contract": {
            "observed_curve": "arithmetic mean of 25 realizations",
            "likelihood": "single-realization covariance; never divided by 25",
            "posterior": "single-realization covariance; never divided by 25",
            "goodness_of_fit": "x25 mean curve evaluated with single-realization covariance; never divided by 25",
            "plot_errorbars": "sqrt(diag(C_single)); never divided by sqrt(25)",
            "construction": "common exact observed-P lattice modes plus independent xi-only complement",
            "automatic_eigenvalue_floor": False,
            "empirical_cross_rescaling": False,
            "real": _summary_for_covariance(covariance_real),
            "rsd": _summary_for_covariance(covariance_rsd),
        },
        "empirical_x25_diagnostics": {
            "real": empirical_covariance_diagnostics(p_stack_real, x_stack_real, covariance_real),
            "rsd": empirical_covariance_diagnostics(p_stack_rsd, x_stack_rsd, covariance_rsd),
        },
        "model_validation": {"p02": p02_validation, "xi02": xi_rsd_validation},
        "real_covariance_prefit": real_prefit,
        "results": results,
        "scientific_closure": scientific_closure,
        "numerical_gates": numerical_gates,
        "inputs": {
            "p0_measurement_sha256": pk["hashes"],
            "p02_measurement_sha256": p02["hashes"],
            "p02_p0_bridge": {
                "thresholds": {"relative_l2": 1.0e-14, "max_abs_diff": 1.0e-9},
                "by_phase": p02["p0_bridges"],
            },
            "xi_measurement_sha256": xi_hashes,
            "theory_cache": str(cache_path),
            "theory_cache_sha256": sha256_file(cache_path),
            "nphase": NPHASE,
            "nbar_mean_h3_mpc3": nbar,
        },
        "mcmc": {
            "nwalkers": int(args.nwalkers),
            "nsteps": nsteps,
            "burnin": burnin,
            "seed_base": int(args.seed),
            "parallel_workers": int(args.threads),
        },
        "outputs": {
            "root": str(out_root),
            "covariance_npz": str(covariance_npz),
        },
        "code_sha256": code_hash,
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    atomic_savez(
        covariance_npz,
        k=np.asarray(pk["k"]),
        k_edges=RAWBOX_FIT_EDGES,
        s_centers=centers,
        xi_mask=mask,
        real_pp=covariance_real["pp"],
        real_xx=covariance_real["xx"],
        real_xp=covariance_real["xp"],
        real_joint=covariance_real["joint"],
        rsd_pp=covariance_rsd["pp"],
        rsd_xx=covariance_rsd["xx"],
        rsd_xp=covariance_rsd["xp"],
        rsd_joint=covariance_rsd["joint"],
    )
    summary["outputs"]["covariance_npz_sha256"] = sha256_file(covariance_npz)
    atomic_write_json(audit_json, summary)
    print(json.dumps({"status": status, "output": str(audit_json), "elapsed_sec": summary["elapsed_sec"]}, sort_keys=True))


if __name__ == "__main__":
    main()
