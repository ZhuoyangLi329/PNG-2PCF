#!/usr/bin/env python3
"""Legacy unmasked x25 periodic-box closure for the Task 4.3.2 RSD model.

The quoted posterior always uses a single-realization covariance.  The
goodness-of-fit of the 25-phase mean instead uses C_mean = C_single / 25.
Those two roles are intentionally kept separate throughout this file.

This file is retained to reproduce the frozen smin-scan baseline.  New
standard rawbox xi0 inference must use
``task43_run_rawbox_baomask_mcmc_v1.py``, whose shared canonical selection is
50 <= s < 350 Mpc/h with 80 <= s < 120 Mpc/h excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

from task43_rsd_common import (
    OUTPUT_ROOT,
    PHASES,
    PLOT_ROOT,
    P_FIXED,
    S_EDGES,
    SMIN_SCAN,
    atomic_savez,
    atomic_write_json,
    sha256_file,
)
from task43_rsd_model import DELTA_C, FullDiscreteRSDModel, build_cache


PARAMETERS = ("fNL", "b1", "sigma_s")
BOUNDS_LO = np.asarray([-500.0, 0.5, 0.0], dtype="f8")
BOUNDS_HI = np.asarray([500.0, 5.0, 30.0], dtype="f8")


def set_affinity(nthreads: int) -> list[int]:
    if not 1 <= int(nthreads) <= 8:
        raise ValueError("--threads must be between 1 and 8")
    available = sorted(os.sched_getaffinity(0))
    selected = available[: int(nthreads)]
    if len(selected) != int(nthreads):
        raise RuntimeError(f"requested {nthreads} CPUs but only {len(available)} are available")
    os.sched_setaffinity(0, selected)
    return selected


def measurement_path(phase: str) -> Path:
    return OUTPUT_ROOT / "rawbox" / "summary" / (
        f"task43_rsd_rawbox_AbacusSummit_base_c000_{phase}_mmin1p4e13_clustering.npz"
    )


def load_x25() -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[str]]:
    stacks = {key: [] for key in ("xi0_real", "xi2_real", "xi0_rsd", "xi2_rsd")}
    metadata_rows: list[dict[str, Any]] = []
    hashes: list[str] = []
    for phase in PHASES:
        path = measurement_path(phase)
        meta_path = path.with_suffix(".json")
        if not path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"missing validated rawbox measurement for {phase}: {path}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        digest = sha256_file(path)
        if metadata.get("status") != "pass" or metadata.get("output_sha256") != digest:
            raise RuntimeError(f"unvalidated rawbox measurement for {phase}: {path}")
        if float(metadata["legacy_real_xi0_bridge"]["max_abs_xi0"]) > 2.0e-14:
            raise RuntimeError(f"legacy real-space estimator bridge failed for {phase}")
        with np.load(path, allow_pickle=False) as payload:
            if not np.array_equal(np.asarray(payload["s_edges"], dtype="f8"), S_EDGES):
                raise RuntimeError(f"separation edges changed in {path}")
            if str(np.asarray(payload["phase"]).item()) != phase:
                raise RuntimeError(f"phase label mismatch in {path}")
            for key in stacks:
                value = np.asarray(payload[key], dtype="f8")
                if value.shape != (S_EDGES.size - 1,) or not np.all(np.isfinite(value)):
                    raise RuntimeError(f"invalid {key} in {path}")
                stacks[key].append(value)
            metadata["nbar_h3_mpc3_from_npz"] = float(np.asarray(payload["nbar"]).item())
        metadata_rows.append(metadata)
        hashes.append(digest)
    return {key: np.stack(values) for key, values in stacks.items()}, metadata_rows, hashes


class FastRSDModel:
    """Cubic-in-sigma surrogate with exact polynomial dependence on b1/fNL."""

    def __init__(self, exact: FullDiscreteRSDModel, *, sigma_step: float = 0.1) -> None:
        self.exact = exact
        self.sigma_grid = np.arange(0.0, 30.0 + 0.5 * float(sigma_step), float(sigma_step))
        self.ells = exact.ell_values
        self.basis = np.empty(
            (self.sigma_grid.size, len(self.ells), 6, S_EDGES.size - 1), dtype="f8"
        )
        mu2 = exact.mu2
        powers = (np.ones_like(mu2), mu2, mu2 * mu2)
        projection = [
            (exact.g_nz[:, None] * exact.kernels[index]) / float(exact.volume)
            for index in range(len(self.ells))
        ]
        for isig, sigma_s in enumerate(self.sigma_grid):
            x = (exact.k_eff[:, None] * exact.mu[None, :] * float(sigma_s)) ** 2
            damping = 1.0 / (1.0 + 0.5 * x) ** 2
            for iell, ell in enumerate(self.ells):
                prefactor = 0.5 * (2 * int(ell) + 1)
                moments = [
                    prefactor
                    * np.sum(
                        exact.wmu[None, :]
                        * damping
                        * power[None, :]
                        * exact.legendre[int(ell)][None, :],
                        axis=1,
                    )
                    for power in powers
                ]
                vectors = np.stack(
                    [
                        exact.pk_dd * moments[0],
                        exact.pk_dd * exact.alpha * moments[0],
                        exact.pk_dd * exact.alpha**2 * moments[0],
                        exact.pk_dd * moments[1],
                        exact.pk_dd * exact.alpha * moments[1],
                        exact.pk_dd * moments[2],
                    ]
                )
                self.basis[isig, iell] = vectors @ projection[iell]
        self.spline = CubicSpline(self.sigma_grid, self.basis, axis=0)

    def evaluate(self, theta: np.ndarray) -> dict[int, np.ndarray]:
        fnl, b1, sigma_s = map(float, np.asarray(theta, dtype="f8")[:3])
        basis = np.asarray(self.spline(sigma_s), dtype="f8")
        q = fnl * 2.0 * DELTA_C * (b1 - P_FIXED)
        f = float(self.exact.f_growth)
        coefficients = np.asarray([b1 * b1, 2.0 * b1 * q, q * q, 2.0 * b1 * f, 2.0 * q * f, f * f])
        return {
            int(ell): coefficients @ basis[iell]
            for iell, ell in enumerate(self.ells)
        }

    def validate(self) -> dict[str, Any]:
        trials = (
            np.asarray([0.0, 2.55, 8.0]),
            np.asarray([-75.0, 2.2, 3.37]),
            np.asarray([80.0, 2.8, 12.43]),
            np.asarray([15.0, 2.5, 0.07]),
            np.asarray([-20.0, 2.6, 29.93]),
        )
        rows = []
        for theta in trials:
            fast = self.evaluate(theta)
            exact = self.exact.evaluate(fnl=theta[0], b1=theta[1], sigma_s=theta[2], p_fixed=P_FIXED)
            for ell in self.ells:
                delta = np.asarray(fast[int(ell)]) - np.asarray(exact[int(ell)])
                rows.append(
                    {
                        "theta": theta.tolist(),
                        "ell": int(ell),
                        "max_abs": float(np.max(np.abs(delta))),
                        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(exact[int(ell)]), 1.0e-30)),
                    }
                )
        maximum = max(row["relative_l2"] for row in rows)
        return {"status": "pass" if maximum < 1.0e-7 else "fail", "max_relative_l2": maximum, "trials": rows}


def periodic_gaussian_covariance(
    model: FullDiscreteRSDModel,
    *,
    b1: float,
    sigma_s: float,
    nbar: float,
) -> np.ndarray:
    """Gaussian covariance of shell-averaged periodic-box xi multipoles."""
    x = (model.k_eff[:, None] * model.mu[None, :] * float(sigma_s)) ** 2
    damping = 1.0 / (1.0 + 0.5 * x) ** 2
    signal = model.pk_dd[:, None] * (
        float(b1) + float(model.f_growth) * model.mu2[None, :]
    ) ** 2 * damping
    total2 = (signal + 1.0 / float(nbar)) ** 2
    nbin = S_EDGES.size - 1
    covariance = np.empty((len(model.ell_values) * nbin,) * 2, dtype="f8")
    for ia, ella in enumerate(model.ell_values):
        for ib, ellb in enumerate(model.ell_values):
            angular = np.sum(
                model.wmu[None, :]
                * total2
                * model.legendre[int(ella)][None, :]
                * model.legendre[int(ellb)][None, :],
                axis=1,
            )
            weight = (
                model.g_nz
                * (2 * int(ella) + 1)
                * (2 * int(ellb) + 1)
                * angular
                / float(model.volume) ** 2
            )
            block = model.kernels[ia].T @ (weight[:, None] * model.kernels[ib])
            covariance[ia * nbin : (ia + 1) * nbin, ib * nbin : (ib + 1) * nbin] = block
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(1.0e-12 * float(eigenvalues[-1]), 1.0e-30)
    covariance = (eigenvectors * np.maximum(eigenvalues, floor)[None, :]) @ eigenvectors.T
    return 0.5 * (covariance + covariance.T)


def vector_indices(ells: tuple[int, ...], mask: np.ndarray, all_ells: tuple[int, ...]) -> np.ndarray:
    nbin = mask.size
    indices: list[int] = []
    for ell in ells:
        offset = all_ells.index(int(ell)) * nbin
        indices.extend((offset + np.flatnonzero(mask)).tolist())
    return np.asarray(indices, dtype="i8")


def data_vector(values: dict[int, np.ndarray], ells: tuple[int, ...], mask: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(values[int(ell)])[mask] for ell in ells])


def fit_map(
    model: FastRSDModel,
    data: dict[int, np.ndarray],
    covariance_single: np.ndarray,
    *,
    smin: float,
    ells: tuple[int, ...],
    nphase_mean: int,
) -> dict[str, Any]:
    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    mask = centers >= float(smin)
    ids = vector_indices(ells, mask, model.ells)
    covariance = covariance_single[np.ix_(ids, ids)]
    chol = np.linalg.cholesky(covariance)
    vector = data_vector(data, ells, mask)

    def residual(theta: np.ndarray) -> np.ndarray:
        prediction = data_vector(model.evaluate(theta), ells, mask)
        return np.linalg.solve(chol, vector - prediction)

    starts = (
        np.asarray([0.0, 2.5, 8.0]),
        np.asarray([-100.0, 2.3, 4.0]),
        np.asarray([100.0, 2.7, 12.0]),
    )
    solutions = [
        least_squares(
            residual,
            start,
            bounds=(BOUNDS_LO, BOUNDS_HI),
            max_nfev=1000,
            xtol=1.0e-11,
            ftol=1.0e-11,
            gtol=1.0e-11,
        )
        for start in starts
    ]
    result = min(solutions, key=lambda item: float(item.fun @ item.fun))
    chi2_single = float(result.fun @ result.fun)
    chi2_mean = float(nphase_mean * chi2_single)
    dof = int(vector.size - len(PARAMETERS))
    fisher_cov = np.linalg.pinv(result.jac.T @ result.jac, rcond=1.0e-12)
    sigma = np.sqrt(np.clip(np.diag(fisher_cov), 0.0, np.inf))
    prediction = model.evaluate(result.x)
    return {
        "smin_mpc_h": float(smin),
        "ells": [int(ell) for ell in ells],
        "ndata": int(vector.size),
        "dof": dof,
        "success": bool(result.success),
        "message": str(result.message),
        "theta": {name: float(value) for name, value in zip(PARAMETERS, result.x, strict=True)},
        "sigma_laplace_single": {name: float(value) for name, value in zip(PARAMETERS, sigma, strict=True)},
        "covariance_laplace_single": fisher_cov.tolist(),
        "chi2_single_covariance": chi2_single,
        "pte_single_covariance": float(chi2_distribution.sf(chi2_single, dof)),
        "chi2_mean_covariance": chi2_mean,
        "pte_mean_covariance": float(chi2_distribution.sf(chi2_mean, dof)),
        "residual_rms_sigma_single": float(np.sqrt(np.mean(result.fun**2))),
        "at_parameter_boundary": bool(
            np.any(np.isclose(result.x, BOUNDS_LO, atol=[1.0, 0.01, 0.01]))
            or np.any(np.isclose(result.x, BOUNDS_HI, atol=[1.0, 0.01, 0.01]))
        ),
        "prediction": {str(int(ell)): np.asarray(prediction[int(ell)]).tolist() for ell in model.ells},
    }


def split_rhat(chain: np.ndarray) -> np.ndarray:
    values = np.asarray(chain, dtype="f8")
    nstep, nwalker, ndim = values.shape
    half = nstep // 2
    if half < 2:
        return np.full(ndim, np.inf)
    split = np.concatenate([values[:half].transpose(1, 0, 2), values[-half:].transpose(1, 0, 2)], axis=0)
    means = np.mean(split, axis=1)
    within = np.mean(np.var(split, axis=1, ddof=1), axis=0)
    between = half * np.var(means, axis=0, ddof=1)
    var_hat = (half - 1.0) / half * within + between / half
    return np.sqrt(var_hat / within)


def run_mcmc(
    model: FastRSDModel,
    mean: dict[int, np.ndarray],
    covariance_single: np.ndarray,
    nominal: dict[str, Any],
    *,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    mask = centers >= 50.0
    ids = vector_indices((0,), mask, model.ells)
    covariance = covariance_single[np.ix_(ids, ids)]
    precision = np.linalg.inv(covariance)
    vector = data_vector(mean, (0,), mask)

    def log_probability(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype="f8")
        if np.any(theta < BOUNDS_LO) or np.any(theta > BOUNDS_HI):
            return -np.inf
        diff = vector - data_vector(model.evaluate(theta), (0,), mask)
        return -0.5 * float(diff @ precision @ diff)

    center = np.asarray([nominal["theta"][name] for name in PARAMETERS], dtype="f8")
    rng = np.random.default_rng(int(seed))
    scale = np.asarray([4.0, 0.015, 0.08], dtype="f8")
    initial = center[None, :] + rng.normal(size=(int(nwalkers), 3)) * scale[None, :]
    initial = np.clip(initial, BOUNDS_LO + 1.0e-6, BOUNDS_HI - 1.0e-6)
    sampler = emcee.EnsembleSampler(int(nwalkers), 3, log_probability)
    sampler.run_mcmc(initial, int(nsteps), progress=False)
    post = np.asarray(sampler.get_chain(discard=int(burnin)), dtype="f8")
    logp = np.asarray(sampler.get_log_prob(discard=int(burnin)), dtype="f8")
    flat = post.reshape(-1, post.shape[-1])
    flat_logp = logp.reshape(-1)
    quantiles = np.percentile(flat, [16.0, 50.0, 84.0], axis=0)
    sigma68 = 0.5 * (quantiles[2] - quantiles[0])
    try:
        tau = np.asarray(emcee.autocorr.integrated_time(post, quiet=True, tol=0), dtype="f8")
    except Exception:
        tau = np.full(3, np.inf)
    rhat = split_rhat(post)
    half = post.shape[0] // 2
    first = post[:half].reshape(-1, 3)
    second = post[-half:].reshape(-1, 3)
    half_shift = np.abs(np.median(first, axis=0) - np.median(second, axis=0)) / sigma68
    imax = int(np.argmax(flat_logp))
    summary = {
        "parameter_names": list(PARAMETERS),
        "posterior": {
            name: {
                "q16": float(quantiles[0, index]),
                "q50": float(quantiles[1, index]),
                "q84": float(quantiles[2, index]),
                "sigma68": float(sigma68[index]),
                "mean": float(np.mean(flat[:, index])),
                "std": float(np.std(flat[:, index], ddof=1)),
            }
            for index, name in enumerate(PARAMETERS)
        },
        "map_chain": {name: float(flat[imax, index]) for index, name in enumerate(PARAMETERS)},
        "map_chain_log_probability": float(flat_logp[imax]),
        "nwalkers": int(nwalkers),
        "nsteps": int(nsteps),
        "burnin": int(burnin),
        "postburn_steps_per_walker": int(post.shape[0]),
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
        "tau": {name: float(tau[index]) for index, name in enumerate(PARAMETERS)},
        "postburn_length_over_tau": {
            name: float(post.shape[0] / tau[index]) for index, name in enumerate(PARAMETERS)
        },
        "split_rhat": {name: float(rhat[index]) for index, name in enumerate(PARAMETERS)},
        "half_chain_shift_sigma": {
            name: float(half_shift[index]) for index, name in enumerate(PARAMETERS)
        },
    }
    summary["gates"] = {
        "split_rhat_max_below_1p01": bool(np.max(rhat) < 1.01),
        "postburn_length_min_above_50tau": bool(np.min(post.shape[0] / tau) > 50.0),
        "half_chain_shift_max_below_0p1sigma": bool(np.max(half_shift) < 0.1),
    }
    return summary, post, logp


def covariance_diagnostics(covariance: np.ndarray) -> dict[str, Any]:
    covariance = 0.5 * (np.asarray(covariance) + np.asarray(covariance).T)
    sigma = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(sigma, sigma)
    eig = np.linalg.eigvalsh(covariance)
    corr_eig = np.linalg.eigvalsh(correlation)
    return {
        "shape": list(covariance.shape),
        "sigma_min": float(np.min(sigma)),
        "sigma_max": float(np.max(sigma)),
        "eigenvalue_min": float(eig[0]),
        "eigenvalue_max": float(eig[-1]),
        "condition_number": float(eig[-1] / eig[0]),
        "correlation_eigenvalue_min": float(corr_eig[0]),
        "correlation_condition_number": float(corr_eig[-1] / corr_eig[0]),
        "correlation_offdiagonal_absmax": float(np.max(np.abs(correlation - np.eye(correlation.shape[0])))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--b1-cov", type=float, default=2.55)
    parser.add_argument("--sigma-s-cov", type=float, default=8.0)
    parser.add_argument("--sigma-grid-step", type=float, default=0.05)
    parser.add_argument("--nmu", type=int, default=64)
    parser.add_argument("--nwalkers", type=int, default=48)
    parser.add_argument("--nsteps", type=int, default=8000)
    parser.add_argument("--burnin", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=430325)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=OUTPUT_ROOT / "rawbox" / "closure" / "task43_rsd_rawbox_x25_fulldiscrete_lorentzian",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=PLOT_ROOT / "task43_rsd_rawbox_x25_fulldiscrete_lorentzian.pdf",
    )
    args = parser.parse_args()
    if int(args.burnin) >= int(args.nsteps):
        raise ValueError("--burnin must be smaller than --nsteps")
    output_npz = args.output_prefix.with_suffix(".npz")
    output_json = args.output_prefix.with_suffix(".json")
    if any(path.exists() for path in (output_npz, output_json, args.plot)):
        raise FileExistsError("immutable x25 closure outputs already exist; use a new output tag")
    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))
    stacks, input_metadata, input_hashes = load_x25()
    mean = {0: np.mean(stacks["xi0_rsd"], axis=0), 2: np.mean(stacks["xi2_rsd"], axis=0)}
    all_vector = np.concatenate([stacks["xi0_rsd"], stacks["xi2_rsd"]], axis=1)
    scatter_covariance = np.cov(all_vector, rowvar=False, ddof=1)
    nbar_values = np.asarray([row["nbar_h3_mpc3_from_npz"] for row in input_metadata])

    cache = build_cache(zeff=0.725, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000")
    exact = FullDiscreteRSDModel(cache, nmu=int(args.nmu))
    fast = FastRSDModel(exact, sigma_step=float(args.sigma_grid_step))
    surrogate_validation = fast.validate()
    if surrogate_validation["status"] != "pass":
        raise RuntimeError(f"fast RSD surrogate failed validation: {surrogate_validation}")
    covariance_single = periodic_gaussian_covariance(
        exact,
        b1=float(args.b1_cov),
        sigma_s=float(args.sigma_s_cov),
        nbar=float(np.mean(nbar_values)),
    )

    fits: list[dict[str, Any]] = []
    for smin in SMIN_SCAN:
        for ells in ((0,), (0, 2)):
            fits.append(
                fit_map(
                    fast,
                    mean,
                    covariance_single,
                    smin=float(smin),
                    ells=ells,
                    nphase_mean=len(PHASES),
                )
            )
    nominal = next(item for item in fits if item["smin_mpc_h"] == 50.0 and item["ells"] == [0])
    mcmc, chain, logp = run_mcmc(
        fast,
        mean,
        covariance_single,
        nominal,
        nwalkers=int(args.nwalkers),
        nsteps=int(args.nsteps),
        burnin=int(args.burnin),
        seed=int(args.seed),
    )

    phase_fits = []
    for iphase, phase in enumerate(PHASES):
        result = fit_map(
            fast,
            {0: stacks["xi0_rsd"][iphase], 2: stacks["xi2_rsd"][iphase]},
            covariance_single,
            smin=50.0,
            ells=(0,),
            nphase_mean=1,
        )
        phase_fits.append({"phase": phase, **result})
    ensemble_chi2 = float(sum(item["chi2_single_covariance"] for item in phase_fits))
    ensemble_dof = int(sum(item["dof"] for item in phase_fits))
    ensemble_pte = float(chi2_distribution.sf(ensemble_chi2, ensemble_dof))

    posterior = mcmc["posterior"]
    fnl_null_ratio = abs(posterior["fNL"]["q50"]) / posterior["fNL"]["sigma68"]
    reference_sigma = np.asarray([posterior[name]["sigma68"] for name in PARAMETERS])
    reference_theta = np.asarray([nominal["theta"][name] for name in PARAMETERS])
    scale_rows = [item for item in fits if item["ells"] == [0] and item["smin_mpc_h"] in (80.0, 100.0, 120.0)]
    scale_shifts = {
        str(int(item["smin_mpc_h"])): {
            name: float(abs(item["theta"][name] - reference_theta[index]) / reference_sigma[index])
            for index, name in enumerate(PARAMETERS)
        }
        for item in scale_rows
    }
    max_scale_shift = max(value for row in scale_shifts.values() for value in row.values())
    primary_gates = {
        "null_abs_median_over_sigma68_single_below_0p3": bool(fnl_null_ratio < 0.3),
        "mean_pte_Cmean_above_0p05": bool(nominal["pte_mean_covariance"] > 0.05),
        "phase_ensemble_profile_pte_above_0p05": bool(ensemble_pte > 0.05),
        "nested_scale_shift_max_below_0p3sigma_ref": bool(max_scale_shift < 0.3),
        **mcmc["gates"],
    }
    status = "pass" if all(primary_gates.values()) else "validation_failed"

    scatter_sigma = np.sqrt(np.diag(scatter_covariance))
    covariance_sigma = np.sqrt(np.diag(covariance_single))
    centered = all_vector - np.mean(all_vector, axis=0)
    precision_full = np.linalg.inv(covariance_single)
    scatter_chi2 = np.einsum("ij,jk,ik->i", centered, precision_full, centered)
    covariance_comparison = {
        "sample_std_over_gaussian_sigma": (scatter_sigma / covariance_sigma).tolist(),
        "sample_std_over_gaussian_sigma_median": float(np.median(scatter_sigma / covariance_sigma)),
        "sample_std_over_gaussian_sigma_xi0_median": float(np.median(scatter_sigma[: S_EDGES.size - 1] / covariance_sigma[: S_EDGES.size - 1])),
        "sample_std_over_gaussian_sigma_xi2_median": float(np.median(scatter_sigma[S_EDGES.size - 1 :] / covariance_sigma[S_EDGES.size - 1 :])),
        "phase_chi2_about_empirical_mean_full_xi02": scatter_chi2.tolist(),
        "phase_chi2_mean_about_empirical_mean_full_xi02": float(np.mean(scatter_chi2)),
        "expected_phase_chi2_mean_about_empirical_mean": float((len(PHASES) - 1) / len(PHASES) * all_vector.shape[1]),
    }

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    temporary_plot = args.plot.with_name(f".{args.plot.name}.{os.getpid()}.tmp.pdf")
    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    nominal_prediction = {int(key): np.asarray(value) for key, value in nominal["prediction"].items()}
    with PdfPages(temporary_plot) as pdf:
        figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex="col")
        for column, ell in enumerate((0, 2)):
            sigma = covariance_sigma[column * centers.size : (column + 1) * centers.size]
            axes[0, column].errorbar(
                centers,
                centers**2 * mean[ell],
                yerr=centers**2 * sigma / np.sqrt(len(PHASES)),
                fmt="o",
                ms=3,
                label="x25 mean +/- Gaussian SEM",
            )
            axes[0, column].plot(centers, centers**2 * nominal_prediction[ell], lw=1.5, label="xi0 nominal model")
            axes[0, column].axvline(50.0, color="0.3", ls="--", lw=0.8)
            axes[0, column].set_ylabel(rf"$s^2\xi_{ell}(s)$")
            axes[0, column].legend(frameon=False, fontsize=8)
            residual = (mean[ell] - nominal_prediction[ell]) / (sigma / np.sqrt(len(PHASES)))
            axes[1, column].axhline(0.0, color="0.4", lw=0.8)
            axes[1, column].plot(centers, residual, "o-", ms=3, lw=0.8)
            axes[1, column].set(xlabel=r"$s\,[h^{-1}{\rm Mpc}]$", ylabel=r"residual / $\sigma_{\rm mean}$")
        figure.suptitle(f"Task 4.3.2 rawbox x25 closure: {status}")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(3, 1, figsize=(8.0, 8.2), sharex=True)
        colors = {(0,): "#1f4e79", (0, 2): "#b22222"}
        for ells in ((0,), (0, 2)):
            subset = [item for item in fits if tuple(item["ells"]) == ells]
            x = np.asarray([item["smin_mpc_h"] for item in subset])
            for iparam, name in enumerate(PARAMETERS):
                y = np.asarray([item["theta"][name] for item in subset])
                error = np.asarray([item["sigma_laplace_single"][name] for item in subset])
                axes[iparam].errorbar(x, y, yerr=error, marker="o", color=colors[ells], label="xi0" if ells == (0,) else "xi0+xi2")
                axes[iparam].set_ylabel(name)
        axes[0].axhline(0.0, color="0.4", lw=0.8)
        axes[-1].set_xlabel(r"$s_{\min}\,[h^{-1}{\rm Mpc}]$")
        axes[0].legend(frameon=False)
        figure.suptitle("Frozen scale scan (single-realization covariance errors)")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        phase_fnl = np.asarray([item["theta"]["fNL"] for item in phase_fits])
        phase_pte = np.asarray([item["pte_single_covariance"] for item in phase_fits])
        figure, axes = plt.subplots(2, 1, figsize=(8.5, 6.5))
        axes[0].axhline(0.0, color="0.4", lw=0.8)
        axes[0].plot(np.arange(len(PHASES)), phase_fnl / posterior["fNL"]["sigma68"], "o-")
        axes[0].set(ylabel=r"profile $f_{\rm NL}/\sigma_{\rm single}$", xticks=np.arange(len(PHASES)), xticklabels=PHASES)
        axes[0].tick_params(axis="x", rotation=90, labelsize=7)
        axes[1].hist(phase_pte, bins=np.linspace(0.0, 1.0, 11), histtype="stepfilled", alpha=0.65)
        axes[1].axvline(0.05, color="#b22222", ls="--")
        axes[1].set(xlabel="per-phase profile PTE", ylabel="count")
        figure.suptitle(f"Phase audit: aggregate PTE={ensemble_pte:.3g}")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)
    temporary_plot.replace(args.plot)

    atomic_savez(
        output_npz,
        s=centers,
        s_edges=S_EDGES,
        phases=np.asarray(PHASES),
        xi0_rsd=stacks["xi0_rsd"],
        xi2_rsd=stacks["xi2_rsd"],
        xi0_real=stacks["xi0_real"],
        xi2_real=stacks["xi2_real"],
        xi0_rsd_mean=mean[0],
        xi2_rsd_mean=mean[2],
        covariance_single_realization=covariance_single,
        covariance_of_mean=covariance_single / len(PHASES),
        scatter_covariance_x25=scatter_covariance,
        chain_by_step=chain,
        log_probability_by_step=logp,
        phase_profile_theta=np.asarray([[item["theta"][name] for name in PARAMETERS] for item in phase_fits]),
        phase_profile_chi2=np.asarray([item["chi2_single_covariance"] for item in phase_fits]),
        nominal_prediction_xi0=nominal_prediction[0],
        nominal_prediction_xi2=nominal_prediction[2],
    )
    payload = {
        "task": "task43_fit_rsd_rawbox_x25",
        "status": status,
        "classification": "production periodic-box RSD closure; lightcone window closure remains separate",
        "phases": list(PHASES),
        "nphase": len(PHASES),
        "input_paths": [str(measurement_path(phase)) for phase in PHASES],
        "input_sha256": input_hashes,
        "legacy_real_xi0_bridge_max_abs": float(
            max(row["legacy_real_xi0_bridge"]["max_abs_xi0"] for row in input_metadata)
        ),
        "legacy_real_xi0_bridge_tolerance": 2.0e-14,
        "cpu_affinity": cpus,
        "model": {
            "name": "FullDiscrete shell-averaged Kaiser x squared-Lorentzian FoG",
            "p_fixed": P_FIXED,
            "theory_cache": str(cache),
            "nmu": int(args.nmu),
            "sigma_surrogate_grid_step_mpc_h": float(args.sigma_grid_step),
            "surrogate_validation": surrogate_validation,
        },
        "covariance_policy": {
            "definition": "periodic-box Gaussian xi0/xi2 covariance including Poisson shot noise",
            "fiducial": {"fNL": 0.0, "b1": float(args.b1_cov), "sigma_s_mpc_h": float(args.sigma_s_cov)},
            "nbar_mean_h3_mpc3": float(np.mean(nbar_values)),
            "quoted_posterior": "C_single, never divided by 25",
            "mean_goodness_of_fit": "C_mean=C_single/25",
            "phase_profiles": "C_single",
            "diagnostics": covariance_diagnostics(covariance_single),
            "x25_scatter_comparison": covariance_comparison,
        },
        "fits": fits,
        "nominal": nominal,
        "mcmc_nominal": mcmc,
        "phase_profiles": phase_fits,
        "phase_ensemble": {"chi2": ensemble_chi2, "dof": ensemble_dof, "pte": ensemble_pte},
        "scale_stability": {
            "reference": "xi0 smin=50; sigma_ref is nominal C_single posterior sigma68",
            "nested_shifts_sigma_ref": scale_shifts,
            "max_shift_sigma_ref": float(max_scale_shift),
        },
        "primary_null_abs_median_over_sigma68_single": float(fnl_null_ratio),
        "primary_gates": primary_gates,
        "output_npz": str(output_npz),
        "output_plot_pdf": str(args.plot),
        "elapsed_sec": time.perf_counter() - started,
    }
    atomic_write_json(output_json, payload)
    print(json.dumps({"status": status, "gates": primary_gates, "output": str(output_json)}, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
