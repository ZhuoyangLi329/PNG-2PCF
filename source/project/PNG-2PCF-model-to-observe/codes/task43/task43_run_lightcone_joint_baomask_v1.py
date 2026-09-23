#!/usr/bin/env python3
"""Standard BAO-masked Task 4.3 lightcone P/xi consistency fits.

The real-space branch retains the frozen 0.6 < z < 0.8 Task 4.3 data and
radial single-term RIC model.  The RSD branch retains the active box-safe
0.4 < z_obs < 0.8 data, geometry-window P model, and formal-GIC xi model.
The standard xi fit selection is 50 <= s < 350 Mpc/h with
80 <= s < 120 Mpc/h removed for every fitted xi multipole; 40 Mpc/h remains
available as an explicit comparison option.

Only the observed curves are averages over 25 phases.  Likelihoods,
posteriors, goodness-of-fit values, residual normalizations, and plot error
bars use the single-realization covariance without division by 25.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

from task43_fit_rsd_lightcone_x25 import FastWindowRSDModel, load_window
from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import set_affinity, split_rhat
from task43_joint_pkxi_fit import (
    JOINT_ROOT as REAL_SOURCE_ROOT,
    MEAN_XI_PATH as REAL_XI_PATH,
    PK_PAYLOAD_PATH as REAL_P_PAYLOAD,
    RIC_OPERATOR_PATH as REAL_RIC_OPERATOR,
    JointModel as RealJointModel,
)
from task43_rsd_boxsafe_p02_increment import (
    COV_NPZ as RSD_P_COVARIANCE,
    MEASURE_DIR as RSD_P_MEASURE_DIR,
    PAYLOAD_NPZ as RSD_P_PAYLOAD,
    WindowConvolvedP02Model,
)
from task43_rsd_common import OUTPUT_ROOT, PHASES, atomic_savez, atomic_write_json, sha256_file
from task43_rsd_joint_p02xi02_fit import (
    ELL02_DECONV_NPZ as RSD_X_DECONV,
    ELL2_SUMMARY as RSD_X_SUMMARY,
    MEAN_WINDOW_NPZ as RSD_MEAN_WINDOW,
    RAW_NPZ as RSD_RAW_COVARIANCE,
)
from task43_rsd_model import FullDiscreteRSDModel, build_cache


DEFAULT_ROOT = OUTPUT_ROOT / "lightcone" / "kmax0p08_smin50_v1"
REAL_COVARIANCE = REAL_SOURCE_ROOT / "covariance" / "task43_joint_cov_64_s50_350.npz"
REAL_COVARIANCE_AUDIT = REAL_COVARIANCE.with_suffix(".json")
REAL_FIT_AUDIT = REAL_SOURCE_ROOT / "audits" / "task43_joint_pkxi_fit_summary.json"
REAL_S30_ROOT = OUTPUT_ROOT.parent / "joint_pkxi_s30_350_v1"
REAL_S30_COVARIANCE = REAL_S30_ROOT / "covariance" / "task43_joint_cov_64_s30_350_v1.npz"
REAL_S30_COVARIANCE_AUDIT = REAL_S30_COVARIANCE.with_suffix(".json")
REAL_S30_XI_PATH = OUTPUT_ROOT.parent / "rmin_scan" / "summary" / "task43_mean_xi_mmin1p4e13_x25_s30_350_ds10_fkpP010000.npz"
REAL_S30_RIC_OPERATOR = (
    OUTPUT_ROOT.parent
    / "rmin_scan/operators/task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_"
    "seed20260712_L2000_s30_350_ds10.npz"
)
REAL_S30_FIT_AUDIT = OUTPUT_ROOT.parent / "rmin_scan/audits/task43_rmin_scan_hygiene.json"

PARAMETER_ORDER = ("fNL", "b1", "sigma_s", "sn0")
BOUNDS = {
    "fNL": (-500.0, 500.0),
    "b1": (0.5, 5.0),
    "sigma_s": (0.0, 30.0),
    "sn0": (-1.0, 1.0),
}
INITIAL_SCALE = {"fNL": 4.0, "b1": 0.015, "sigma_s": 0.12, "sn0": 0.025}
REAL_STARTS = (
    np.asarray([0.0, 2.5, 0.0]),
    np.asarray([-80.0, 2.3, 0.2]),
    np.asarray([80.0, 2.8, -0.2]),
)
RSD_STARTS = (
    np.asarray([0.0, 2.5, 8.0, 0.0]),
    np.asarray([-80.0, 2.3, 3.0, 0.2]),
    np.asarray([80.0, 2.8, 12.0, -0.2]),
    np.asarray([0.0, 2.5, 20.0, 0.5]),
)
_POOL_LOG_PROBABILITY: Callable[[np.ndarray], float] | None = None


def _pool_log_probability(theta: np.ndarray) -> float:
    if _POOL_LOG_PROBABILITY is None:
        raise RuntimeError("MCMC worker likelihood was not initialized")
    return _POOL_LOG_PROBABILITY(theta)


def lightcone_xi_primary_mask(s: np.ndarray, *, smin: float = 50.0) -> np.ndarray:
    centers = np.asarray(s, dtype="f8")
    return (centers >= float(smin)) & (centers < 350.0) & ~((centers >= 80.0) & (centers < 120.0))


def select_pk_bins(payload: dict[str, np.ndarray], kmax: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select complete frozen P(k) bins below the requested kmax."""

    value = float(kmax)
    if not np.isclose(value, 0.08, rtol=0.0, atol=1.0e-12) and not np.isclose(
        value, 0.10, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("--pk-kmax must be 0.08 or 0.10")
    edges = np.asarray(payload["k_edges"], dtype="f8")
    keep = np.flatnonzero(edges[:, 1] <= value - 1.0e-12)
    if keep.size == 0:
        raise RuntimeError(f"no lightcone P(k) bins survive kmax={value}")
    return (
        np.asarray(payload["fit_bin_indices"], dtype="i8")[keep],
        keep.astype("i8"),
        np.asarray(payload["k_obs"], dtype="f8")[keep],
        edges[keep],
    )


class ScaledGaussianMetric:
    """Strict covariance metric evaluated in correlation-normalized units."""

    def __init__(self, covariance: np.ndarray) -> None:
        self.covariance = 0.5 * (np.asarray(covariance, dtype="f8") + np.asarray(covariance, dtype="f8").T)
        self.scale = np.sqrt(np.diag(self.covariance))
        if np.any(~np.isfinite(self.scale)) or np.any(self.scale <= 0.0):
            raise ValueError("covariance diagonal is not finite and positive")
        self.correlation = self.covariance / np.outer(self.scale, self.scale)
        self.correlation = 0.5 * (self.correlation + self.correlation.T)
        self.eigenvalues = np.linalg.eigvalsh(self.correlation)
        if float(self.eigenvalues[0]) <= 1.0e-12:
            raise ValueError(f"strict correlation SPD gate failed: {self.eigenvalues[0]:.3e}")
        self.cholesky = np.linalg.cholesky(self.correlation)

    def residual(self, difference: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.cholesky, np.asarray(difference, dtype="f8") / self.scale)

    def chi2(self, difference: np.ndarray) -> float:
        residual = self.residual(difference)
        return float(residual @ residual)


@dataclass(frozen=True)
class FitSpec:
    name: str
    group: str
    data: np.ndarray
    phase_data: np.ndarray
    evaluate: Callable[[np.ndarray], np.ndarray]
    covariance: np.ndarray
    parameter_names: tuple[str, ...]
    starts: tuple[np.ndarray, ...]


def bounds_for(names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([BOUNDS[name][0] for name in names], dtype="f8"),
        np.asarray([BOUNDS[name][1] for name in names], dtype="f8"),
    )


def fit_maximum_likelihood(spec: FitSpec, metric: ScaledGaussianMetric) -> tuple[dict[str, Any], np.ndarray]:
    lower, upper = bounds_for(spec.parameter_names)

    def residual(theta: np.ndarray) -> np.ndarray:
        return metric.residual(spec.data - spec.evaluate(np.asarray(theta, dtype="f8")))

    starts = tuple(start[: len(spec.parameter_names)] for start in spec.starts)
    solutions = [
        least_squares(
            residual,
            start,
            bounds=(lower, upper),
            max_nfev=3000,
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
        )
        for start in starts
    ]
    best = min(solutions, key=lambda result: float(result.fun @ result.fun))
    theta = np.asarray(best.x, dtype="f8")
    chi2 = float(best.fun @ best.fun)
    dof = int(spec.data.size - theta.size)
    return (
        {
            "theta": {name: float(value) for name, value in zip(spec.parameter_names, theta, strict=True)},
            "chi2_observed_x25_mean_with_single_realization_covariance": chi2,
            "dof": dof,
            "pte_observed_x25_mean_with_single_realization_covariance": float(chi2_distribution.sf(chi2, dof)),
            "success": bool(best.success),
            "message": str(best.message),
        },
        theta,
    )


def run_chain(
    spec: FitSpec,
    metric: ScaledGaussianMetric,
    theta_ml: np.ndarray,
    *,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    nworkers: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    lower, upper = bounds_for(spec.parameter_names)

    def log_probability(theta: np.ndarray) -> float:
        values = np.asarray(theta, dtype="f8")
        if np.any(values < lower) or np.any(values > upper):
            return -np.inf
        return -0.5 * metric.chi2(spec.data - spec.evaluate(values))

    rng = np.random.default_rng(int(seed))
    scales = np.asarray([INITIAL_SCALE[name] for name in spec.parameter_names], dtype="f8")
    initial = theta_ml[None, :] + rng.normal(size=(int(nwalkers), len(spec.parameter_names))) * scales[None, :]
    initial = np.clip(initial, lower + 1.0e-7, upper - 1.0e-7)
    np.random.seed(int(seed))
    global _POOL_LOG_PROBABILITY
    _POOL_LOG_PROBABILITY = log_probability
    try:
        if int(nworkers) == 1:
            sampler = emcee.EnsembleSampler(int(nwalkers), len(spec.parameter_names), log_probability)
            sampler.run_mcmc(initial, int(nsteps), progress=False)
        else:
            with mp.get_context("fork").Pool(processes=int(nworkers)) as pool:
                sampler = emcee.EnsembleSampler(
                    int(nwalkers), len(spec.parameter_names), _pool_log_probability, pool=pool
                )
                sampler.run_mcmc(initial, int(nsteps), progress=False)
    finally:
        _POOL_LOG_PROBABILITY = None
    chain = np.asarray(sampler.get_chain(discard=int(burnin)), dtype="f8")
    logp = np.asarray(sampler.get_log_prob(discard=int(burnin)), dtype="f8")
    flat = chain.reshape(-1, len(spec.parameter_names))
    flat_logp = logp.reshape(-1)
    quantiles = np.percentile(flat, [16.0, 50.0, 84.0], axis=0)
    sigma68 = 0.5 * (quantiles[2] - quantiles[0])
    tau = np.asarray(emcee.autocorr.integrated_time(chain, quiet=True, tol=0), dtype="f8")
    rhat = split_rhat(chain)
    half = chain.shape[0] // 2
    first, second = chain[:half].reshape(-1, chain.shape[-1]), chain[-half:].reshape(-1, chain.shape[-1])
    half_shift = np.abs(np.median(first, axis=0) - np.median(second, axis=0)) / sigma68
    imax = int(np.argmax(flat_logp))
    return (
        {
            "posterior": {
                name: {
                    "q16": float(quantiles[0, index]),
                    "q50": float(quantiles[1, index]),
                    "q84": float(quantiles[2, index]),
                    "sigma68": float(sigma68[index]),
                    "mean": float(np.mean(flat[:, index])),
                    "std": float(np.std(flat[:, index], ddof=1)),
                }
                for index, name in enumerate(spec.parameter_names)
            },
            "map_chain": {name: float(flat[imax, index]) for index, name in enumerate(spec.parameter_names)},
            "map_chain_log_probability": float(flat_logp[imax]),
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
            "nwalkers": int(nwalkers),
            "nsteps": int(nsteps),
            "burnin": int(burnin),
            "parallel_workers": int(nworkers),
            "tau": {name: float(tau[index]) for index, name in enumerate(spec.parameter_names)},
            "split_rhat": {name: float(rhat[index]) for index, name in enumerate(spec.parameter_names)},
            "postburn_length_over_tau": {
                name: float(chain.shape[0] / tau[index]) for index, name in enumerate(spec.parameter_names)
            },
            "half_chain_shift_sigma": {
                name: float(half_shift[index]) for index, name in enumerate(spec.parameter_names)
            },
            "gates": {
                "split_rhat_max_below_1p01": bool(np.max(rhat) < 1.01),
                "postburn_length_min_above_50tau": bool(np.min(chain.shape[0] / tau) > 50.0),
                "half_chain_shift_max_below_0p1sigma": bool(np.max(half_shift) < 0.1),
            },
        },
        chain,
        logp,
    )


def phase_diagnostics(spec: FitSpec, metric: ScaledGaussianMetric, theta_ml: np.ndarray) -> dict[str, Any]:
    prediction = spec.evaluate(theta_ml)
    chi2_values = np.asarray([metric.chi2(row - prediction) for row in spec.phase_data], dtype="f8")
    dof = int(spec.data.size - len(spec.parameter_names))
    pte = chi2_distribution.sf(chi2_values, dof)
    return {
        "definition": "each phase at the x25-mean maximum-likelihood model, using C_single",
        "dof": dof,
        "chi2_by_phase": chi2_values.tolist(),
        "chi2_mean": float(np.mean(chi2_values)),
        "chi2_median": float(np.median(chi2_values)),
        "pte_by_phase": pte.tolist(),
        "fraction_pte_above_0p05": float(np.mean(pte > 0.05)),
    }


def load_real_specs(
    *,
    smin: float = 50.0,
    pk_kmax: float = 0.10,
    covariance_path: Path = REAL_COVARIANCE,
    covariance_audit_path: Path = REAL_COVARIANCE_AUDIT,
    fit_audit_path: Path = REAL_FIT_AUDIT,
    xi_path: Path = REAL_XI_PATH,
    ric_operator_path: Path = REAL_RIC_OPERATOR,
) -> tuple[list[FitSpec], dict[str, Any], dict[str, np.ndarray]]:
    for path in (covariance_path, covariance_audit_path, fit_audit_path, REAL_P_PAYLOAD, xi_path, ric_operator_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    covariance_audit = json.loads(covariance_audit_path.read_text(encoding="utf-8"))
    fit_audit = json.loads(fit_audit_path.read_text(encoding="utf-8"))
    if covariance_audit.get("status") not in ("done", "pass") or fit_audit.get("status") not in ("done", "pass"):
        raise RuntimeError("real-space source audit did not complete")
    namespace = argparse.Namespace(
        pk_payload=REAL_P_PAYLOAD,
        xi_path=xi_path,
        ric_operator=ric_operator_path,
        cosmology="abacus_c000",
        p_fixed=1.0,
    )
    model = RealJointModel(args=namespace)
    mask = lightcone_xi_primary_mask(model.xi_s, smin=smin)
    expected = (model.xi_s >= float(smin)) & (model.xi_s < 350.0) & ~(
        (model.xi_s >= 80.0) & (model.xi_s < 120.0)
    )
    excluded = model.xi_s[(model.xi_s >= float(smin)) & (model.xi_s < 350.0) & ~mask]
    if not np.array_equal(mask, expected) or not np.array_equal(excluded, [85.0, 95.0, 105.0, 115.0]):
        raise RuntimeError("real-space BAO mask changed")
    with np.load(REAL_P_PAYLOAD, allow_pickle=False) as payload:
        fit_indices, payload_indices, k, fit_edges = select_pk_bins(
            {key: np.asarray(payload[key]) for key in ("fit_bin_indices", "k_obs", "k_edges")}, pk_kmax
        )
    with np.load(covariance_path, allow_pickle=False) as payload:
        source_joint = np.asarray(payload["joint_covariance"], dtype="f8")
    ids = np.concatenate((payload_indices, 15 + np.flatnonzero(mask)))
    c_joint = source_joint[np.ix_(ids, ids)]
    n_p = int(payload_indices.size)
    c_pp = c_joint[:n_p, :n_p]
    c_px = c_joint[:n_p, n_p:]
    c_xx = c_joint[n_p:, n_p:]
    with np.load(REAL_P_PAYLOAD, allow_pickle=False) as payload:
        p_stack = np.asarray(payload["pk_stack"], dtype="f8")[:, payload_indices]
    with np.load(xi_path, allow_pickle=False) as payload:
        x_stack = np.asarray(payload["xi0_all"], dtype="f8")[:, mask]
    data_p, data_x = np.mean(p_stack, axis=0), np.mean(x_stack, axis=0)

    def evaluate_p(theta: np.ndarray) -> np.ndarray:
        return model.model_pk_vector(theta)[payload_indices]

    def evaluate_x(theta: np.ndarray) -> np.ndarray:
        return model.model_xi_vector(theta)[mask]

    def evaluate_joint(theta: np.ndarray) -> np.ndarray:
        return np.concatenate((evaluate_p(theta), evaluate_x(theta)))

    specs = [
        FitSpec("real_p0", "real", data_p, p_stack, evaluate_p, c_pp, ("fNL", "b1", "sn0"), REAL_STARTS),
        FitSpec("real_xi0", "real", data_x, x_stack, evaluate_x, c_xx, ("fNL", "b1"), REAL_STARTS),
        FitSpec(
            "real_joint_p0xi0",
            "real",
            np.concatenate((data_p, data_x)),
            np.hstack((p_stack, x_stack)),
            evaluate_joint,
            c_joint,
            ("fNL", "b1", "sn0"),
            REAL_STARTS,
        ),
    ]
    metadata = {
        "redshift_range": "0.6 < z < 0.8",
        "model": "real-space window-convolved P0 and radial single-term RIC xi0",
        "pk_kmax_h_mpc": float(pk_kmax),
        "pk_bins": n_p,
        "pk_fit_edges_h_mpc": fit_edges.tolist(),
        "source_covariance": str(covariance_path),
        "source_covariance_sha256": sha256_file(covariance_path),
        "source_covariance_audit_sha256": sha256_file(covariance_audit_path),
        "source_fit_audit_sha256": sha256_file(fit_audit_path),
        "p_payload_sha256": sha256_file(REAL_P_PAYLOAD),
        "xi_summary_sha256": sha256_file(xi_path),
        "ric_operator_sha256": sha256_file(ric_operator_path),
    }
    arrays = {"real_k": k, "real_s": model.xi_s, "real_xi_mask": mask, "real_pp": c_pp, "real_xx": c_xx, "real_px": c_px, "real_joint": c_joint}
    return specs, metadata, arrays


def load_rsd_specs(
    *, smin: float = 50.0, pk_kmax: float = 0.10
) -> tuple[list[FitSpec], dict[str, Any], dict[str, np.ndarray]]:
    for path in (RSD_P_PAYLOAD, RSD_P_COVARIANCE, RSD_RAW_COVARIANCE, RSD_X_DECONV, RSD_X_SUMMARY, RSD_MEAN_WINDOW):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    from task43_fit_rsd_lightcone_pk0_vs_xi0_smin50 import load_pk

    payload = load_pk(RSD_P_PAYLOAD)
    fit_indices, payload_indices, k_obs, fit_edges = select_pk_bins(payload, pk_kmax)
    keep_p2 = np.flatnonzero(k_obs >= 0.015 - 1.0e-12)
    zeff = float(np.asarray(payload["zeff"]).item())
    p_stacks, window_rows, theory_k, theory_ell = [], None, None, None
    measurement_hashes = []
    for phase in PHASES:
        path = RSD_P_MEASURE_DIR / f"task43_rsd_lightcone_p02_{phase}_mesh256_kmax0p300_dk0p002.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        measurement_hashes.append(sha256_file(path))
        with np.load(path, allow_pickle=False) as data:
            p_stacks.append(
                np.concatenate(
                    (
                        np.asarray(data["pk0"], dtype="f8")[fit_indices],
                        np.asarray(data["pk2"], dtype="f8")[fit_indices][keep_p2],
                    )
                )
            )
            if phase == "ph000":
                window_rows = np.asarray(data["window_matrix"], dtype="f8")
                theory_k = np.asarray(data["theory_k"], dtype="f8")
                theory_ell = np.asarray(data["theory_ell"], dtype="i8")
    p_stack = np.stack(p_stacks)
    p_row_ids = np.concatenate((fit_indices, 150 + fit_indices[keep_p2]))
    window = window_rows[p_row_ids]
    p_model = WindowConvolvedP02Model(window, theory_k, theory_ell, zeff=zeff, sigma_step=0.05)
    p_validation = p_model.validate()
    if p_validation["status"] != "pass":
        raise RuntimeError("RSD P02 model validation failed")

    with np.load(RSD_X_SUMMARY, allow_pickle=False) as payload_x:
        s = np.asarray(payload_x["s"], dtype="f8")
        x_phase_full = np.asarray(payload_x["xi_multipoles_by_phase"], dtype="f8")
        zeff_x = float(np.mean(np.asarray(payload_x["zeff_by_phase"], dtype="f8")))
    mask = lightcone_xi_primary_mask(s, smin=smin)
    expected = (s >= float(smin)) & (s < 350.0) & ~((s >= 80.0) & (s < 120.0))
    excluded = s[(s >= float(smin)) & (s < 350.0) & ~mask]
    if not np.array_equal(mask, expected) or not np.array_equal(excluded, [85.0, 95.0, 105.0, 115.0]):
        raise RuntimeError("RSD BAO mask changed")
    x_stack = np.concatenate((x_phase_full[:, 0, mask], x_phase_full[:, 1, mask]), axis=1)
    theory_cache = build_cache(zeff=zeff_x, boxsize=2000.0, kmax=5.0, ells=(0, 2), cosmology="abacus_c000")
    x_model = FastWindowRSDModel(
        FullDiscreteRSDModel(theory_cache, nmu=64),
        {"mean": load_window(RSD_MEAN_WINDOW)},
        sigma_step=0.05,
    )
    x_validation = x_model.validate()
    if x_validation["status"] != "pass":
        raise RuntimeError("RSD xi02 model validation failed")

    with np.load(RSD_RAW_COVARIANCE, allow_pickle=False) as raw:
        cov_pk = np.asarray(raw["covariance_pk"], dtype="f8")
        projection = np.asarray(raw["projection_matrix"], dtype="f8")
        raw_meta = json.loads(str(np.asarray(raw["meta_json"]).item()))
    with np.load(RSD_X_DECONV, allow_pickle=False) as deconv:
        c_xx_full = np.asarray(deconv["covariance_single_realization"], dtype="f8")
        rinv = np.asarray(deconv["rr_window_inverse"], dtype="f8")
    k_grid = raw_meta["k_grid"]
    nper = int(k_grid["nk"])
    edges = float(k_grid["kmin"]) + float(k_grid["dk"]) * np.arange(nper + 1, dtype="f8")
    fine_centers = 0.5 * (edges[:-1] + edges[1:])
    selected = []
    for lo, hi in np.asarray(fit_edges, dtype="f8"):
        inside = np.flatnonzero((fine_centers >= lo - 1.0e-12) & (fine_centers < hi - 1.0e-12))
        if inside.size != 1:
            raise RuntimeError(f"RSD P bin [{lo},{hi}] has {inside.size} fine covariance bins")
        selected.append(int(inside[0]))
    selected = np.asarray(selected, dtype="i8")
    p_cov_ids = np.concatenate((selected, nper + selected[keep_p2]))
    c_pp = cov_pk[np.ix_(p_cov_ids, p_cov_ids)]
    c_xp_full = rinv @ (projection[: 2 * s.size] @ cov_pk[:, p_cov_ids])
    x_row_ids = np.concatenate((np.flatnonzero(mask), s.size + np.flatnonzero(mask)))
    c_xp = c_xp_full[x_row_ids]
    c_xx = c_xx_full[np.ix_(x_row_ids, x_row_ids)]
    c_joint = np.block([[c_pp, c_xp.T], [c_xp, c_xx]])

    with np.load(RSD_P_COVARIANCE, allow_pickle=False) as covariance_p:
        cov450 = np.asarray(covariance_p["covariance_full"], dtype="f8")
    ids128 = np.concatenate((fit_indices, 150 + fit_indices[keep_p2]))
    reference_pp = cov450[np.ix_(ids128, ids128)]
    bridge_relfro = float(np.linalg.norm(c_pp - reference_pp) / np.linalg.norm(reference_pp))
    bridge_sigma = np.sqrt(np.diag(c_pp) / np.diag(reference_pp))
    if not (0.95 <= float(np.median(bridge_sigma)) <= 1.05 and bridge_relfro < 0.10):
        raise RuntimeError("RSD P02 covariance bridge failed")

    def evaluate_p02(theta: np.ndarray) -> np.ndarray:
        return p_model.evaluate(theta)

    def evaluate_xi02(theta: np.ndarray) -> np.ndarray:
        prediction = x_model.evaluate(np.asarray(theta, dtype="f8")[:3], model="formal_gic", window_key="mean")
        return np.concatenate((np.asarray(prediction[0])[mask], np.asarray(prediction[2])[mask]))

    def evaluate_full(theta: np.ndarray) -> np.ndarray:
        return np.concatenate((evaluate_p02(theta), evaluate_xi02(theta)))

    data_p, data_x = np.mean(p_stack, axis=0), np.mean(x_stack, axis=0)
    n0 = int(fit_indices.size)
    nx0 = int(np.count_nonzero(mask))
    c_pp0, c_xx0, c_xp0 = c_pp[:n0, :n0], c_xx[:nx0, :nx0], c_xp[:nx0, :n0]
    c_joint0 = np.block([[c_pp0, c_xp0.T], [c_xp0, c_xx0]])
    data_p0, data_x0 = data_p[:n0], data_x[:nx0]
    p_stack0, x_stack0 = p_stack[:, :n0], x_stack[:, :nx0]

    def evaluate_p0(theta: np.ndarray) -> np.ndarray:
        return evaluate_p02(theta)[:n0]

    def evaluate_xi0(theta: np.ndarray) -> np.ndarray:
        return evaluate_xi02(theta)[:nx0]

    def evaluate_mono_joint(theta: np.ndarray) -> np.ndarray:
        return np.concatenate((evaluate_p0(theta), evaluate_xi0(theta)))

    specs = [
        FitSpec("rsd_p0", "rsd_monopole", data_p0, p_stack0, evaluate_p0, c_pp0, PARAMETER_ORDER, RSD_STARTS),
        FitSpec("rsd_xi0", "rsd_monopole", data_x0, x_stack0, evaluate_xi0, c_xx0, PARAMETER_ORDER[:3], RSD_STARTS),
        FitSpec(
            "rsd_joint_p0xi0",
            "rsd_monopole",
            np.concatenate((data_p0, data_x0)),
            np.hstack((p_stack0, x_stack0)),
            evaluate_mono_joint,
            c_joint0,
            PARAMETER_ORDER,
            RSD_STARTS,
        ),
        FitSpec("rsd_p02", "rsd_multipole", data_p, p_stack, evaluate_p02, c_pp, PARAMETER_ORDER, RSD_STARTS),
        FitSpec("rsd_xi02", "rsd_multipole", data_x, x_stack, evaluate_xi02, c_xx, PARAMETER_ORDER[:3], RSD_STARTS),
        FitSpec(
            "rsd_joint_p02xi02",
            "rsd_multipole",
            np.concatenate((data_p, data_x)),
            np.hstack((p_stack, x_stack)),
            evaluate_full,
            c_joint,
            PARAMETER_ORDER,
            RSD_STARTS,
        ),
    ]
    metadata = {
        "redshift_range": "0.4 < z_obs < 0.8 (box-safe)",
        "model": "geometry-window P02 and formal-GIC RR-deconvolved xi02",
        "pk_kmax_h_mpc": float(pk_kmax),
        "pk_fit_edges_h_mpc": fit_edges.tolist(),
        "p0_bins": n0,
        "p2_bins": int(keep_p2.size),
        "p2_kmin_h_mpc": 0.015,
        "p_covariance_bridge": {
            "relative_frobenius": bridge_relfro,
            "sigma_ratio_median": float(np.median(bridge_sigma)),
        },
        "raw_covariance_sha256": sha256_file(RSD_RAW_COVARIANCE),
        "xi_deconvolved_covariance_sha256": sha256_file(RSD_X_DECONV),
        "xi_summary_sha256": sha256_file(RSD_X_SUMMARY),
        "p_payload_sha256": sha256_file(RSD_P_PAYLOAD),
        "p_measurement_sha256": measurement_hashes,
        "mean_formal_gic_window_sha256": sha256_file(RSD_MEAN_WINDOW),
        "theory_cache": str(theory_cache),
        "theory_cache_sha256": sha256_file(theory_cache),
        "model_validation": {"p02": p_validation, "xi02": x_validation},
    }
    arrays = {
        "rsd_k": k_obs,
        "rsd_p2_keep_indices": keep_p2,
        "rsd_s": s,
        "rsd_xi_mask": mask,
        "rsd_pp": c_pp,
        "rsd_xx": c_xx,
        "rsd_xp": c_xp,
        "rsd_joint": c_joint,
        "rsd_pp0": c_pp0,
        "rsd_xx0": c_xx0,
        "rsd_xp0": c_xp0,
        "rsd_joint0": c_joint0,
    }
    return specs, metadata, arrays


def independent_tension(left: dict[str, Any], right: dict[str, Any], parameter: str) -> dict[str, float | str]:
    lhs, rhs = left["mcmc"]["posterior"][parameter], right["mcmc"]["posterior"][parameter]
    difference = float(rhs["q50"] - lhs["q50"])
    denominator = float(np.hypot(lhs["sigma68"], rhs["sigma68"]))
    return {
        "definition": "abs(q50_right-q50_left)/hypot(sigma68_left,sigma68_right); ignores cross-correlation",
        "signed_q50_difference_right_minus_left": difference,
        "quadrature_sigma68": denominator,
        "tension_sigma": abs(difference) / denominator,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--pk-kmax", type=float, choices=(0.08, 0.10), default=0.08)
    parser.add_argument("--smin", type=float, choices=(40.0, 50.0), default=50.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if not 1 <= int(args.threads) <= 8:
        raise ValueError("--threads must be in [1,8]")
    if int(args.burnin) >= int(args.nsteps):
        raise ValueError("--burnin must be smaller than --nsteps")
    nsteps = 400 if args.smoke else int(args.nsteps)
    burnin = 100 if args.smoke else int(args.burnin)
    out_root = Path(args.out_root) / ("smoke" if args.smoke else "")
    audit_json = out_root / "task43_lightcone_standard_joint_baomask80_120_v1.json"
    covariance_npz = out_root / "task43_lightcone_standard_joint_baomask80_120_covariance_v1.npz"
    if audit_json.exists():
        raise FileExistsError(f"immutable audit exists: {audit_json}")
    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))
    code_hash = sha256_file(Path(__file__))

    if np.isclose(float(args.smin), 40.0, rtol=0.0, atol=1.0e-12):
        real_covariance = REAL_S30_COVARIANCE
        real_covariance_audit = REAL_S30_COVARIANCE_AUDIT
        real_fit_audit = REAL_S30_FIT_AUDIT
        real_xi_path = REAL_S30_XI_PATH
        real_ric_operator = REAL_S30_RIC_OPERATOR
    else:
        real_covariance = REAL_COVARIANCE
        real_covariance_audit = REAL_COVARIANCE_AUDIT
        real_fit_audit = REAL_FIT_AUDIT
        real_xi_path = REAL_XI_PATH
        real_ric_operator = REAL_RIC_OPERATOR
    real_specs, real_metadata, real_arrays = load_real_specs(
        smin=float(args.smin),
        pk_kmax=float(args.pk_kmax),
        covariance_path=real_covariance,
        covariance_audit_path=real_covariance_audit,
        fit_audit_path=real_fit_audit,
        xi_path=real_xi_path,
        ric_operator_path=real_ric_operator,
    )
    rsd_specs, rsd_metadata, rsd_arrays = load_rsd_specs(smin=float(args.smin), pk_kmax=float(args.pk_kmax))
    specs = real_specs + rsd_specs
    results: dict[str, Any] = {}
    numerical_gates: dict[str, bool] = {}
    for index, spec in enumerate(specs):
        metric = ScaledGaussianMetric(spec.covariance)
        numerical_gates[f"{spec.name}_strict_spd"] = bool(metric.eigenvalues[0] > 1.0e-12)
        map_summary, theta_ml = fit_maximum_likelihood(spec, metric)
        prediction = np.asarray(spec.evaluate(theta_ml), dtype="f8")
        fit_root = out_root / "fits" / spec.name
        fit_npz, fit_json = fit_root / "samples.npz", fit_root / "summary.json"
        if fit_npz.exists() != fit_json.exists():
            raise RuntimeError(f"partial resumable output for {spec.name}")
        if fit_npz.exists():
            existing = json.loads(fit_json.read_text(encoding="utf-8"))
            if (
                existing.get("status") != "pass"
                or existing.get("code_sha256") != code_hash
                or existing.get("output_npz_sha256") != sha256_file(fit_npz)
                or existing.get("nsteps") != nsteps
                or existing.get("burnin") != burnin
            ):
                raise RuntimeError(f"resumable output contract failed for {spec.name}")
            results[spec.name] = existing["result"]
            for gate, passed in results[spec.name]["mcmc"]["gates"].items():
                if not args.smoke:
                    numerical_gates[f"{spec.name}_{gate}"] = bool(passed)
            print(json.dumps({"variant": spec.name, "status": "resumed"}), flush=True)
            continue
        chain_summary, chain, logp = run_chain(
            spec,
            metric,
            theta_ml,
            nwalkers=int(args.nwalkers),
            nsteps=nsteps,
            burnin=burnin,
            seed=int(args.seed) + 100 * index,
            nworkers=int(args.threads),
        )
        result = {
            "group": spec.group,
            "parameter_names": list(spec.parameter_names),
            "map": map_summary,
            "mcmc": chain_summary,
            "phase_diagnostics": phase_diagnostics(spec, metric, theta_ml),
            "correlation_eigenvalue_min": float(metric.eigenvalues[0]),
        }
        atomic_savez(
            fit_npz,
            parameter_names=np.asarray(spec.parameter_names),
            chain_by_step=chain,
            log_probability_by_step=logp,
            theta_maximum_likelihood=theta_ml,
            prediction_maximum_likelihood=prediction,
            data=np.asarray(spec.data),
            phase_data=np.asarray(spec.phase_data),
            covariance_single=np.asarray(spec.covariance),
        )
        fit_status = "pass" if args.smoke or all(chain_summary["gates"].values()) else "validation_failed"
        atomic_write_json(
            fit_json,
            {
                "task": "task43_run_lightcone_joint_baomask_v1",
                "variant": spec.name,
                "status": fit_status,
                "result": result,
                "nsteps": nsteps,
                "burnin": burnin,
                "code_sha256": code_hash,
                "output_npz": str(fit_npz),
                "output_npz_sha256": sha256_file(fit_npz),
            },
        )
        results[spec.name] = result
        if not args.smoke:
            for gate, passed in chain_summary["gates"].items():
                numerical_gates[f"{spec.name}_{gate}"] = bool(passed)
        print(
            json.dumps(
                {
                    "variant": spec.name,
                    "fNL": chain_summary["posterior"]["fNL"],
                    "maximum_likelihood": map_summary["theta"],
                    "gates": chain_summary["gates"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    expected_xi_count = 26 if np.isclose(float(args.smin), 50.0) else 27
    numerical_gates["canonical_real_mask"] = bool(np.count_nonzero(real_arrays["real_xi_mask"]) == expected_xi_count)
    numerical_gates["canonical_rsd_mask"] = bool(np.count_nonzero(rsd_arrays["rsd_xi_mask"]) == expected_xi_count)
    status = "pass" if all(numerical_gates.values()) else "validation_failed"
    metrics = {
        "real": {
            "joint_fNL_improvement_over_best_marginal_fraction": 1.0
            - results["real_joint_p0xi0"]["mcmc"]["posterior"]["fNL"]["sigma68"]
            / min(
                results["real_p0"]["mcmc"]["posterior"]["fNL"]["sigma68"],
                results["real_xi0"]["mcmc"]["posterior"]["fNL"]["sigma68"],
            ),
            "marginal_tensions": {
                parameter: independent_tension(results["real_p0"], results["real_xi0"], parameter)
                for parameter in ("fNL", "b1")
            },
        },
        "rsd_monopole": {
            "joint_fNL_improvement_over_best_marginal_fraction": 1.0
            - results["rsd_joint_p0xi0"]["mcmc"]["posterior"]["fNL"]["sigma68"]
            / min(
                results["rsd_p0"]["mcmc"]["posterior"]["fNL"]["sigma68"],
                results["rsd_xi0"]["mcmc"]["posterior"]["fNL"]["sigma68"],
            ),
            "marginal_tensions": {
                parameter: independent_tension(results["rsd_p0"], results["rsd_xi0"], parameter)
                for parameter in ("fNL", "b1", "sigma_s")
            },
        },
        "rsd_multipole": {
            "joint_fNL_improvement_over_best_marginal_fraction": 1.0
            - results["rsd_joint_p02xi02"]["mcmc"]["posterior"]["fNL"]["sigma68"]
            / min(
                results["rsd_p02"]["mcmc"]["posterior"]["fNL"]["sigma68"],
                results["rsd_xi02"]["mcmc"]["posterior"]["fNL"]["sigma68"],
            ),
            "marginal_tensions": {
                parameter: independent_tension(results["rsd_p02"], results["rsd_xi02"], parameter)
                for parameter in ("fNL", "b1", "sigma_s")
            },
        },
    }
    summary = {
        "task": "task43_run_lightcone_joint_baomask_v1",
        "status": status,
        "classification": "standard Task 4.3 lightcone P/xi consistency test with canonical BAO mask",
        "fit_contract": {
            "pk_kmax_h_mpc": float(args.pk_kmax),
            "xi_smin_mpc_h": float(args.smin),
            "xi_smax_mpc_h": 350.0,
            "pk_bins_real": int(real_arrays["real_k"].size),
            "pk_bins_rsd_p0": int(rsd_metadata["p0_bins"]),
            "pk_bins_rsd_p2": int(rsd_metadata["p2_bins"]),
        },
        "comparison_scope": (
            "within-space P/xi consistency only; real and RSD redshift ranges differ and their difference is not a pure RSD comparison"
        ),
        "mask_policy": {
            "full_range_mpc_h": [float(args.smin), 350.0],
            "excluded_half_open_range_mpc_h": [80.0, 120.0],
            "selected_centers_mpc_h": real_arrays["real_s"][real_arrays["real_xi_mask"]].tolist(),
            "excluded_centers_mpc_h": [85.0, 95.0, 105.0, 115.0],
            "same_mask_for_rsd_xi0_and_xi2": True,
        },
        "covariance_contract": {
            "observed_curve": "arithmetic mean of 25 realizations",
            "likelihood": "single-realization covariance; never divided by 25",
            "posterior": "single-realization covariance; never divided by 25",
            "goodness_of_fit": "x25 mean and each phase evaluated with C_single; never divided by 25",
            "plot_errorbars": "sqrt(diag(C_single)); never divided by sqrt(25)",
            "selection": "exact principal submatrices/cross blocks; no empirical rescaling and no new eigenvalue floor",
        },
        "real": real_metadata,
        "rsd": rsd_metadata,
        "results": results,
        "metrics": metrics,
        "numerical_gates": numerical_gates,
        "mcmc": {
            "nwalkers": int(args.nwalkers),
            "nsteps": nsteps,
            "burnin": burnin,
            "seed_base": int(args.seed),
            "parallel_workers": int(args.threads),
        },
        "outputs": {"root": str(out_root), "covariance_npz": str(covariance_npz)},
        "code_sha256": code_hash,
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    atomic_savez(covariance_npz, **real_arrays, **rsd_arrays)
    summary["outputs"]["covariance_npz_sha256"] = sha256_file(covariance_npz)
    atomic_write_json(audit_json, summary)
    print(json.dumps({"status": status, "output": str(audit_json), "elapsed_sec": summary["elapsed_sec"]}, sort_keys=True))


if __name__ == "__main__":
    main()
