#!/usr/bin/env python3
"""Infer fNL from matched halo P0/xi0 with a Gaussian prior on p."""

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
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

from task44_fit_pngbase_hodhost_pk_strictk0p006 import load_measurement as load_halo_pk
from task44_fit_pngbase_hodhost_xi_smax150 import (
    load_halo_xi,
    load_lrg_xi,
    lrg_xi_contract,
    slice_theory_to_fit_range,
)
from task44_fit_pngbase_hodmap_strictk0p006 import load_measurement as load_lrg_pk
from task44_fit_pngbase_hodmap_rawbox import (
    ExactPeriodicPkModel,
    FullDiscreteXiModel,
    build_theory_cache,
    covariance_iteration_summary,
    load_theory,
    regularize_covariance,
    summarize_chain,
)
from task44_pngbase_hodhost_mmin1e13_common import (
    BOX_SIZE,
    BOX_VOLUME,
    CATALOGS,
    K_FUND,
    LRG_CONSERVATIVE_PK_FIT_EDGES,
    LRG_CONSERVATIVE_PK_KMAX,
    LRG_CONSERVATIVE_XI_FIT_EDGES,
    LRG_CONSERVATIVE_XI_SMAX,
    LRG_CONSERVATIVE_XI_SMIN,
    LRG_EXTENDED_XI_SMAX,
    P_GAUSSIAN_PRIOR_MEAN,
    P_GAUSSIAN_PRIOR_SIGMA,
    PK_KMAX_CONTRACT,
    PK_STRICT_FIT_EDGES,
    PK_STRICT_KMIN,
    REDSHIFT,
    XI_FIT_EDGES,
    XI_FIT_SMAX,
    XI_FIT_SMIN,
    atomic_savez,
    atomic_write_json,
    ensure_output_dirs,
    get_spec,
    host_catalog_path,
    host_pk_gaussianp_fit_prefix,
    host_xi_gaussianp_fit_prefix,
    lrg_pk_gaussianp_fit_prefix,
    lrg_pk_conservative_gaussianp_fit_prefix,
    lrg_xi_gaussianp_fit_prefix,
    lrg_xi_conservative_gaussianp_fit_prefix,
    lrg_xi_extended_gaussianp_fit_prefix,
    set_cpu_affinity,
    sha256_file,
    to_jsonable,
)
from task44_pngbase_hodmap_rawbox_common import DELTA_C, SN0_SCALE


PK_PRIORS = {"b1": (0.5, 5.0), "fnl": (-500.0, 500.0), "p": (-5.0, 5.0), "sn0": (-1.0, 1.0)}
XI_PRIORS = {"b1": (0.5, 5.0), "fnl": (-500.0, 500.0), "p": (-5.0, 5.0)}


class ExactPeriodicPkGaussianPModel(ExactPeriodicPkModel):
    names = ("b1", "fnl", "p", "sn0")
    priors = PK_PRIORS

    def __init__(self, theory: dict[str, Any], edges: np.ndarray) -> None:
        super().__init__(theory, edges, fnl=0.0)

    def coefficients(self, theta: np.ndarray) -> np.ndarray:
        b1, fnl, p, _sn0 = map(float, np.asarray(theta, dtype="f8")[:4])
        q = fnl * 2.0 * DELTA_C * (b1 - p)
        return np.asarray([b1**2, 2.0 * b1 * q, q**2], dtype="f8")

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        values = np.asarray(theta, dtype="f8")
        return self.basis @ self.coefficients(values) + float(values[3]) * SN0_SCALE

    def signal_modes(self, theta: np.ndarray) -> np.ndarray:
        b1, fnl, p, _sn0 = map(float, np.asarray(theta, dtype="f8")[:4])
        q = fnl * 2.0 * DELTA_C * (b1 - p)
        return self.pk_dd_mode * (b1 + q * self.alpha_mode) ** 2

    def covariance(self, theta: np.ndarray, *, nbar: float) -> np.ndarray:
        values = np.asarray(theta, dtype="f8")
        total2 = (self.signal_modes(values) + float(values[3]) * SN0_SCALE + 1.0 / float(nbar)) ** 2
        summed = np.bincount(self.bin_id, weights=total2, minlength=self.edges.shape[0])
        return np.diag(2.0 * summed / self.counts**2)


class FullDiscreteXiGaussianPModel(FullDiscreteXiModel):
    names = ("b1", "fnl", "p")
    priors = XI_PRIORS

    def __init__(self, theory: dict[str, Any]) -> None:
        super().__init__(theory, fnl=0.0)

    def coefficients(self, theta: np.ndarray) -> np.ndarray:
        b1, fnl, p = map(float, np.asarray(theta, dtype="f8")[:3])
        q = fnl * 2.0 * DELTA_C * (b1 - p)
        return np.asarray([b1**2, 2.0 * b1 * q, q**2], dtype="f8")

    def signal_modes(self, theta: np.ndarray) -> np.ndarray:
        b1, fnl, p = map(float, np.asarray(theta, dtype="f8")[:3])
        q = fnl * 2.0 * DELTA_C * (b1 - p)
        return self.pk_dd * (b1 + q * self.alpha) ** 2


def bounds_for(model: Any) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray([model.priors[name][0] for name in model.names], dtype="f8")
    upper = np.asarray([model.priors[name][1] for name in model.names], dtype="f8")
    return lower, upper


def prior_chi2(theta: np.ndarray, model: Any) -> float:
    values = np.asarray(theta, dtype="f8")
    p = float(values[tuple(model.names).index("p")])
    return float(((p - P_GAUSSIAN_PRIOR_MEAN) / P_GAUSSIAN_PRIOR_SIGMA) ** 2)


def fit_map(model: Any, data: np.ndarray, covariance: np.ndarray, *, baseline_fnl: float) -> dict[str, Any]:
    chol = np.linalg.cholesky(covariance)
    lower, upper = bounds_for(model)

    def data_residual(theta: np.ndarray) -> np.ndarray:
        return np.linalg.solve(chol, np.asarray(data, dtype="f8") - model.evaluate(theta))

    def posterior_residual(theta: np.ndarray) -> np.ndarray:
        values = np.asarray(theta, dtype="f8")
        p = float(values[tuple(model.names).index("p")])
        return np.concatenate(
            [data_residual(values), np.asarray([(p - P_GAUSSIAN_PRIOR_MEAN) / P_GAUSSIAN_PRIOR_SIGMA])]
        )

    fnl_starts = tuple(dict.fromkeys((-150.0, 0.0, 30.0, float(baseline_fnl), 100.0, 200.0)))
    p_starts = (
        P_GAUSSIAN_PRIOR_MEAN - P_GAUSSIAN_PRIOR_SIGMA,
        P_GAUSSIAN_PRIOR_MEAN,
        P_GAUSSIAN_PRIOR_MEAN + P_GAUSSIAN_PRIOR_SIGMA,
    )
    if tuple(model.names) == ("b1", "fnl", "p", "sn0"):
        starts = [
            np.asarray([b1, fnl, p, 0.0], dtype="f8")
            for b1 in (1.7, 2.0, 2.3)
            for fnl in fnl_starts
            for p in p_starts
        ]
    elif tuple(model.names) == ("b1", "fnl", "p"):
        starts = [
            np.asarray([b1, fnl, p], dtype="f8")
            for b1 in (1.7, 2.0, 2.3)
            for fnl in fnl_starts
            for p in p_starts
        ]
    else:
        raise ValueError(f"unsupported parameters: {model.names}")
    solutions = [
        least_squares(
            posterior_residual,
            np.clip(start, lower + 1.0e-8, upper - 1.0e-8),
            bounds=(lower, upper),
            max_nfev=7000,
            xtol=1.0e-13,
            ftol=1.0e-13,
            gtol=1.0e-13,
        )
        for start in starts
    ]
    best = min(solutions, key=lambda result: float(result.fun @ result.fun))
    residual = data_residual(best.x)
    chi2_data = float(residual @ residual)
    chi2_p_prior = prior_chi2(best.x, model)
    dof = int(np.asarray(data).size - len(model.names))
    boundary_tolerance = 0.002 * (upper - lower)
    return {
        "theta": {name: float(value) for name, value in zip(model.names, best.x, strict=True)},
        "chi2": chi2_data,
        "chi2_data": chi2_data,
        "chi2_p_gaussian_prior": chi2_p_prior,
        "minus2_logposterior_without_constants": chi2_data + chi2_p_prior,
        "dof": dof,
        "pte_data_only": float(chi2_distribution.sf(chi2_data, dof)),
        "success": bool(best.success),
        "message": str(best.message),
        "at_parameter_boundary": bool(
            np.any(best.x - lower < boundary_tolerance) or np.any(upper - best.x < boundary_tolerance)
        ),
        "prediction": np.asarray(model.evaluate(best.x), dtype="f8").tolist(),
    }


def run_mcmc(
    model: Any,
    data: np.ndarray,
    covariance: np.ndarray,
    nominal: dict[str, Any],
    *,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    precision = np.linalg.inv(covariance)
    lower, upper = bounds_for(model)

    def log_probability(theta: np.ndarray) -> float:
        values = np.asarray(theta, dtype="f8")
        if np.any(values < lower) or np.any(values > upper):
            return -np.inf
        difference = np.asarray(data, dtype="f8") - model.evaluate(values)
        return -0.5 * float(difference @ precision @ difference + prior_chi2(values, model))

    center = np.asarray([nominal["theta"][name] for name in model.names], dtype="f8")
    scale_map = {"b1": 0.015, "fnl": 12.0, "p": 0.08, "sn0": 0.025}
    scale = np.asarray([scale_map[name] for name in model.names], dtype="f8")
    rng = np.random.default_rng(int(seed))
    initial = center[None, :] + rng.normal(size=(int(nwalkers), len(model.names))) * scale[None, :]
    initial = np.clip(initial, lower + 1.0e-8, upper - 1.0e-8)
    np.random.seed(int(seed))
    sampler = emcee.EnsembleSampler(int(nwalkers), len(model.names), log_probability)
    sampler.run_mcmc(initial, int(nsteps), progress=False)
    chain = np.asarray(sampler.get_chain(discard=int(burnin)), dtype="f8")
    logp = np.asarray(sampler.get_log_prob(discard=int(burnin)), dtype="f8")
    summary = summarize_chain(chain, logp, tuple(model.names), model.priors)
    acceptance = float(np.mean(sampler.acceptance_fraction))
    summary.update(
        {
            "nwalkers": int(nwalkers),
            "nsteps": int(nsteps),
            "burnin": int(burnin),
            "postburn_steps_per_walker": int(chain.shape[0]),
            "acceptance_fraction_mean": acceptance,
            "p_gaussian_prior": {
                "mean": P_GAUSSIAN_PRIOR_MEAN,
                "sigma": P_GAUSSIAN_PRIOR_SIGMA,
            },
        }
    )
    summary["gates"]["acceptance_between_0p10_0p80"] = bool(0.10 < acceptance < 0.80)
    return summary, chain, logp


def _validated_existing(prefix: Path) -> bool:
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")
    if not npz_path.is_file() or not json_path.is_file():
        return False
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    return summary.get("status") == "pass" and summary.get("output_npz_sha256") == sha256_file(npz_path)


def fit_contract(
    tracer: str, *, kmax_contract: float, smin: float, smax: float
) -> tuple[np.ndarray, np.ndarray]:
    if tracer == "halo":
        if not np.isclose(kmax_contract, PK_KMAX_CONTRACT, rtol=0.0, atol=1.0e-13):
            raise ValueError("the halo P0 contract remains frozen at kmax=0.100")
        if not (
            np.isclose(smin, XI_FIT_SMIN, rtol=0.0, atol=1.0e-13)
            and np.isclose(smax, XI_FIT_SMAX, rtol=0.0, atol=1.0e-13)
        ):
            raise ValueError("the halo xi contract remains frozen at s=30--150")
        return np.asarray(PK_STRICT_FIT_EDGES, dtype="f8"), np.asarray(XI_FIT_EDGES, dtype="f8")
    if tracer != "lrg":
        raise ValueError(tracer)
    if np.isclose(kmax_contract, PK_KMAX_CONTRACT, rtol=0.0, atol=1.0e-13):
        pk_edges = np.asarray(PK_STRICT_FIT_EDGES, dtype="f8")
    elif np.isclose(kmax_contract, LRG_CONSERVATIVE_PK_KMAX, rtol=0.0, atol=1.0e-13):
        pk_edges = np.asarray(LRG_CONSERVATIVE_PK_FIT_EDGES, dtype="f8")
    else:
        raise ValueError(f"unsupported LRG P0 kmax: {kmax_contract}")
    xi_edges, _xi_centers = lrg_xi_contract(smin, smax)
    return pk_edges, xi_edges


def resume_one(
    tag: str,
    probe: str,
    *,
    tracer: str,
    theory: dict[str, Any],
    kmax_contract: float,
    smin: float,
    smax: float,
    extra_steps: int,
    seed: int,
) -> None:
    """Continue an existing final-covariance ensemble without repeating its burn-in or MAP work."""

    pk_edges, xi_edges = fit_contract(
        tracer, kmax_contract=kmax_contract, smin=smin, smax=smax
    )
    if tracer == "halo":
        prefix = host_pk_gaussianp_fit_prefix(tag) if probe == "pk" else host_xi_gaussianp_fit_prefix(tag)
        measured = load_halo_pk(tag) if probe == "pk" else load_halo_xi(tag)
    elif tracer == "lrg":
        if probe == "pk":
            prefix = (
                lrg_pk_gaussianp_fit_prefix(tag)
                if np.isclose(kmax_contract, PK_KMAX_CONTRACT, rtol=0.0, atol=1.0e-13)
                else lrg_pk_conservative_gaussianp_fit_prefix(tag)
            )
            measured = load_lrg_pk(tag, fit_edges=pk_edges)
        else:
            if np.isclose(smax, LRG_EXTENDED_XI_SMAX, rtol=0.0, atol=1.0e-13):
                prefix = lrg_xi_extended_gaussianp_fit_prefix(tag)
            elif np.isclose(smin, XI_FIT_SMIN, rtol=0.0, atol=1.0e-13):
                prefix = lrg_xi_gaussianp_fit_prefix(tag)
            else:
                prefix = lrg_xi_conservative_gaussianp_fit_prefix(tag)
            measured = load_lrg_xi(tag, smin=smin, smax=smax)
    else:
        raise ValueError(tracer)
    model: Any = (
        ExactPeriodicPkGaussianPModel(theory, pk_edges)
        if probe == "pk"
        else FullDiscreteXiGaussianPModel(slice_theory_to_fit_range(theory, fit_edges=xi_edges))
    )
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"cannot resume missing fit: {npz_path} / {json_path}")
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    if summary.get("output_npz_sha256") != sha256_file(npz_path):
        raise RuntimeError(f"cannot resume fit with invalid hash: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files if name != "summary_json"}
    old_chain = np.asarray(arrays["chain_final_by_step"], dtype="f8")
    old_logp = np.asarray(arrays["logp_final_by_step"], dtype="f8")
    if old_chain.ndim != 3 or old_chain.shape[:2] != old_logp.shape:
        raise RuntimeError(f"invalid resumable chain shape: {old_chain.shape} / {old_logp.shape}")

    covariance = np.asarray(arrays["covariance_final"], dtype="f8")
    precision = np.linalg.inv(covariance)
    lower, upper = bounds_for(model)

    def log_probability(theta: np.ndarray) -> float:
        values = np.asarray(theta, dtype="f8")
        if np.any(values < lower) or np.any(values > upper):
            return -np.inf
        difference = np.asarray(measured["data"], dtype="f8") - model.evaluate(values)
        return -0.5 * float(difference @ precision @ difference + prior_chi2(values, model))

    np.random.seed(int(seed))
    sampler = emcee.EnsembleSampler(old_chain.shape[1], old_chain.shape[2], log_probability)
    sampler.run_mcmc(old_chain[-1], int(extra_steps), progress=False)
    added_chain = np.asarray(sampler.get_chain(), dtype="f8")
    added_logp = np.asarray(sampler.get_log_prob(), dtype="f8")
    combined_chain = np.concatenate([old_chain, added_chain], axis=0)
    combined_logp = np.concatenate([old_logp, added_logp], axis=0)
    final_mcmc = summarize_chain(combined_chain, combined_logp, tuple(model.names), model.priors)
    old_nsteps = int(summary["mcmc"]["nsteps"])
    old_acceptance = float(summary["mcmc"]["acceptance_fraction_mean"])
    new_acceptance = float(np.mean(sampler.acceptance_fraction))
    acceptance = (old_acceptance * old_nsteps + new_acceptance * int(extra_steps)) / (
        old_nsteps + int(extra_steps)
    )
    final_mcmc.update(
        {
            "nwalkers": int(old_chain.shape[1]),
            "nsteps": old_nsteps + int(extra_steps),
            "burnin": int(summary["mcmc"]["burnin"]),
            "postburn_steps_per_walker": int(combined_chain.shape[0]),
            "acceptance_fraction_mean": acceptance,
            "p_gaussian_prior": {
                "mean": P_GAUSSIAN_PRIOR_MEAN,
                "sigma": P_GAUSSIAN_PRIOR_SIGMA,
            },
        }
    )
    final_mcmc["gates"]["acceptance_between_0p10_0p80"] = bool(0.10 < acceptance < 0.80)
    summary["mcmc"] = final_mcmc
    summary["covariance"]["iteration_posterior_comparison"] = covariance_iteration_summary(
        summary["mcmc_initial_covariance"], final_mcmc
    )
    final_delta = np.asarray(
        [summary["covariance"]["final_map_minus_fiducial"][name] for name in model.names], dtype="f8"
    )
    posterior_sigmas = np.asarray(
        [final_mcmc["posterior"][name]["sigma68"] for name in model.names], dtype="f8"
    )
    summary["gates"]["covariance_final_shift_below_0p01sigma_all_parameters"] = bool(
        np.all(np.abs(final_delta) < 0.01 * posterior_sigmas)
    )
    summary["gates"]["mcmc_all_convergence_gates"] = bool(all(final_mcmc["gates"].values()))
    summary["status"] = "pass" if all(summary["gates"].values()) else "review"
    summary.setdefault("mcmc_resume_history", []).append(
        {
            "extra_steps": int(extra_steps),
            "seed": int(seed),
            "old_postburn_steps_per_walker": int(old_chain.shape[0]),
            "new_postburn_steps_per_walker": int(combined_chain.shape[0]),
            "extension_acceptance_fraction_mean": new_acceptance,
        }
    )
    arrays["chain_final_by_step"] = combined_chain
    arrays["logp_final_by_step"] = combined_logp
    summary.pop("output_npz", None)
    summary.pop("output_npz_sha256", None)
    arrays["summary_json"] = np.asarray(json.dumps(to_jsonable(summary), sort_keys=True))
    atomic_savez(npz_path, **arrays)
    summary["output_npz"] = str(npz_path)
    summary["output_npz_sha256"] = sha256_file(npz_path)
    atomic_write_json(json_path, summary)
    fnl = final_mcmc["posterior"]["fnl"]
    print(
        f"[resumed] {tag} {tracer} {probe} +{int(extra_steps)} steps "
        f"fNL={fnl['q50']:.2f} -{fnl['q50']-fnl['q16']:.2f}/+{fnl['q84']-fnl['q50']:.2f} "
        f"maxRhat={max(final_mcmc['split_rhat'].values()):.5f} status={summary['status']}",
        flush=True,
    )


def run_one(
    tag: str,
    probe: str,
    *,
    tracer: str,
    theory: dict[str, Any],
    theory_cache_path: Path,
    kmax_contract: float,
    smin: float,
    smax: float,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    overwrite: bool,
) -> None:
    spec = get_spec(tag)
    pk_edges, xi_edges = fit_contract(
        tracer, kmax_contract=kmax_contract, smin=smin, smax=smax
    )
    if tracer == "halo":
        tracer_token = "hodhost_mmin1e13"
        tracer_description = (
            "unit-weight unique occupied HOD hosts with official cleaned CompaSO M>=1e13 Msun/h"
        )
        source_catalog = host_catalog_path(tag)
    elif tracer == "lrg":
        tracer_token = "hodmap_lrg"
        tracer_description = "HOD-MAP LRG galaxies"
        source_catalog = Path(spec.path)
    else:
        raise ValueError(tracer)
    if probe == "pk":
        if tracer == "halo":
            prefix = host_pk_gaussianp_fit_prefix(tag)
            measured = load_halo_pk(tag)
        else:
            prefix = (
                lrg_pk_gaussianp_fit_prefix(tag)
                if np.isclose(kmax_contract, PK_KMAX_CONTRACT, rtol=0.0, atol=1.0e-13)
                else lrg_pk_conservative_gaussianp_fit_prefix(tag)
            )
            measured = load_lrg_pk(tag, fit_edges=pk_edges)
        model: Any = ExactPeriodicPkGaussianPModel(theory, pk_edges)
        theta_initial = np.asarray([2.0, spec.fnl, P_GAUSSIAN_PRIOR_MEAN, 0.0], dtype="f8")
        parameter_tolerance = np.asarray([1.0e-5, 1.0e-3, 1.0e-4, 1.0e-5], dtype="f8")
        coordinate_edges = np.asarray(measured["k_edges"], dtype="f8")
        radial_edges = np.asarray([], dtype="f8")
        mode_counts_match = bool(
            np.array_equal(np.asarray(model.counts, dtype="i8"), np.asarray(measured["nmodes"], dtype="i8"))
        )
        mode_means_match = bool(np.allclose(model.k_mean, measured["k"], rtol=0.0, atol=1.0e-14))
        if not mode_counts_match or not mode_means_match:
            raise RuntimeError(f"{tracer} P0 exact-mode mismatch for {tag}")
    elif probe == "xi":
        if tracer == "halo":
            prefix = host_xi_gaussianp_fit_prefix(tag)
            measured = load_halo_xi(tag)
        else:
            if np.isclose(smax, LRG_EXTENDED_XI_SMAX, rtol=0.0, atol=1.0e-13):
                prefix = lrg_xi_extended_gaussianp_fit_prefix(tag)
            elif np.isclose(smin, XI_FIT_SMIN, rtol=0.0, atol=1.0e-13):
                prefix = lrg_xi_gaussianp_fit_prefix(tag)
            else:
                prefix = lrg_xi_conservative_gaussianp_fit_prefix(tag)
            measured = load_lrg_xi(tag, smin=smin, smax=smax)
        model = FullDiscreteXiGaussianPModel(slice_theory_to_fit_range(theory, fit_edges=xi_edges))
        theta_initial = np.asarray([2.0, spec.fnl, P_GAUSSIAN_PRIOR_MEAN], dtype="f8")
        parameter_tolerance = np.asarray([1.0e-5, 1.0e-3, 1.0e-4], dtype="f8")
        coordinate_edges = np.column_stack([xi_edges[:-1], xi_edges[1:]])
        radial_edges = np.asarray(xi_edges, dtype="f8")
        mode_counts_match = True
        mode_means_match = True
    else:
        raise ValueError(probe)

    output_npz = prefix.with_suffix(".npz")
    output_json = prefix.with_suffix(".json")
    if _validated_existing(prefix) and not overwrite:
        print(f"[skip] validated {output_json}", flush=True)
        return
    if not overwrite and (output_npz.exists() or output_json.exists()):
        raise FileExistsError(f"partial/unvalidated Gaussian-p fit exists: {output_npz} / {output_json}")
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    covariance_initial_raw = model.covariance(theta_initial, nbar=measured["nbar"])
    covariance_initial, covariance_initial_metadata = regularize_covariance(covariance_initial_raw)
    map_initial = fit_map(model, measured["data"], covariance_initial, baseline_fnl=spec.fnl)

    theta_covariance = theta_initial.copy()
    fixed_point_history: list[dict[str, Any]] = []
    fixed_point_converged = False
    for iteration in range(12):
        raw = model.covariance(theta_covariance, nbar=measured["nbar"])
        matrix, diagnostics = regularize_covariance(raw)
        current_map = fit_map(model, measured["data"], matrix, baseline_fnl=spec.fnl)
        theta_fit = np.asarray([current_map["theta"][name] for name in model.names], dtype="f8")
        delta = theta_fit - theta_covariance
        fixed_point_history.append(
            {
                "iteration": iteration,
                "covariance_fiducial": {
                    name: float(value) for name, value in zip(model.names, theta_covariance, strict=True)
                },
                "map": current_map,
                "delta_map_minus_covariance_fiducial": {
                    name: float(value) for name, value in zip(model.names, delta, strict=True)
                },
                "covariance_diagnostics": diagnostics,
            }
        )
        theta_covariance = theta_fit
        if iteration >= 1 and np.all(np.abs(delta) < parameter_tolerance):
            fixed_point_converged = True
            break

    covariance_final_raw = model.covariance(theta_covariance, nbar=measured["nbar"])
    covariance_final, covariance_final_metadata = regularize_covariance(covariance_final_raw)
    map_final = fit_map(model, measured["data"], covariance_final, baseline_fnl=spec.fnl)
    theta_map_final = np.asarray([map_final["theta"][name] for name in model.names], dtype="f8")
    fixed_point_final_delta = theta_map_final - theta_covariance

    initial_mcmc, initial_chain, initial_logp = run_mcmc(
        model,
        measured["data"],
        covariance_initial,
        map_initial,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burnin=burnin,
        seed=seed + 101,
    )
    final_mcmc, final_chain, final_logp = run_mcmc(
        model,
        measured["data"],
        covariance_final,
        map_final,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burnin=burnin,
        seed=seed,
    )
    covariance_posterior_comparison = covariance_iteration_summary(initial_mcmc, final_mcmc)
    final_prediction = np.asarray(model.evaluate(theta_map_final), dtype="f8")
    sigma = np.sqrt(np.diag(covariance_final))
    residual_sigma = (np.asarray(measured["data"], dtype="f8") - final_prediction) / sigma
    posterior_sigmas = np.asarray(
        [float(final_mcmc["posterior"][name]["sigma68"]) for name in model.names], dtype="f8"
    )
    if tracer == "halo":
        source_catalog_gate = bool(
            sha256_file(source_catalog) == measured["metadata"]["source_host_catalog_sha256"]
        )
    elif "input_hdf5_sha256" in measured["metadata"]:
        source_catalog_gate = bool(
            sha256_file(source_catalog) == measured["metadata"]["input_hdf5_sha256"]
        )
    elif "source_hdf5_sha256" in measured["metadata"]:
        source_catalog_gate = bool(
            sha256_file(source_catalog) == measured["metadata"]["source_hdf5_sha256"]
        )
    else:
        source_catalog_gate = bool(
            measured["metadata"].get("input_hdf5") == spec.path
            and sha256_file(Path(measured["metadata"]["input_ascii"]))
            == measured["metadata"]["input_ascii_sha256"]
        )

    gates = {
        "p_gaussian_prior_mean_exact": P_GAUSSIAN_PRIOR_MEAN == 0.7072363788448802,
        "p_gaussian_prior_sigma_exact": P_GAUSSIAN_PRIOR_SIGMA == 0.2694701311625559,
        "fnl_is_free_not_fixed": "fnl" in model.names,
        "p_is_sampled_not_fixed": "p" in model.names,
        "input_tracer_matches_requested": bool(
            tracer_token in str(measured["path"])
            if tracer == "halo"
            else (
                measured["metadata"].get("input_hdf5") == spec.path
                if probe == "pk"
                else measured["metadata"].get("source_hdf5", measured["metadata"].get("input_hdf5"))
                == spec.path
            )
        ),
        "source_catalog_or_measurement_provenance_valid": source_catalog_gate,
        "current_range_exact": bool(
            np.array_equal(coordinate_edges, pk_edges)
            if probe == "pk"
            else np.array_equal(radial_edges, xi_edges)
        ),
        "pk_exact_mode_contract": mode_counts_match and mode_means_match,
        "single_box_covariance_not_divided": True,
        "map_optimizer_success": bool(map_final["success"]),
        "map_not_at_uniform_prior_boundary": bool(not map_final["at_parameter_boundary"]),
        "covariance_fixed_point_converged": bool(fixed_point_converged),
        "covariance_final_shift_below_0p01sigma_all_parameters": bool(
            np.all(np.abs(fixed_point_final_delta) < 0.01 * posterior_sigmas)
        ),
        "mcmc_all_convergence_gates": bool(all(final_mcmc["gates"].values())),
    }
    status = "pass" if all(gates.values()) else "review"
    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "task": f"task44_fit_pngbase_{tracer_token}_gaussianp_freefnl",
        "status": status,
        "tag": tag,
        "probe": probe,
        "realization": spec.realization,
        "tracer": tracer_description,
        "geometry": "full periodic real-space cube",
        "boxsize_mpc_h": BOX_SIZE,
        "volume_mpc_h3": BOX_VOLUME,
        "kfund_h_mpc": K_FUND,
        "redshift": REDSHIFT,
        "input_baseline_fnl": spec.fnl,
        "free_parameters": list(model.names),
        "uniform_bounds": {name: list(model.priors[name]) for name in model.names},
        "p_gaussian_prior": {
            "mean": P_GAUSSIAN_PRIOR_MEAN,
            "sigma": P_GAUSSIAN_PRIOR_SIGMA,
            "same_for_both_mocks": True,
            "included_in_map_and_mcmc": True,
        },
        "model": "[b1 + fNL*2*delta_c*(b1-p)*alpha(k)]^2 Pm"
        + (" + sn0*1e4" if probe == "pk" else ""),
        "input": {
            "path": str(measured["path"]),
            "sha256": measured["sha256"],
            "source_catalog": str(source_catalog),
            "source_catalog_sha256": sha256_file(source_catalog),
            **(
                {
                    "source_host_catalog": str(source_catalog),
                    "source_host_catalog_sha256": sha256_file(source_catalog),
                }
                if tracer == "halo"
                else {
                    "source_hdf5": str(source_catalog),
                    "source_hdf5_sha256": sha256_file(source_catalog),
                }
            ),
            "ndata": measured["ndata"],
            "nbar_h3_mpc3": measured["nbar"],
        },
        "fit_range": (
            {
                "kmin_h_mpc": PK_STRICT_KMIN,
                "kmax_bin_center_h_mpc": kmax_contract,
                "edges_h_mpc": pk_edges.tolist(),
                "nbins": int(pk_edges.shape[0]),
            }
            if probe == "pk"
            else {
                "smin_mpc_h": smin,
                "smax_mpc_h": float(xi_edges[-1]),
                "edges_mpc_h": xi_edges.tolist(),
                "nbins": int(xi_edges.size - 1),
            }
        ),
        "theory": {
            "path": str(theory_cache_path),
            "sha256": sha256_file(theory_cache_path),
            "cosmology": "abacus_c000",
            "periodic_discrete_modes": True,
            "survey_window": False,
            "gic_ric_aic": False,
        },
        "covariance": {
            "type": "single-periodic-box analytic Gaussian",
            "not_divided_by_number_of_boxes": True,
            "initial_fiducial": {
                name: float(value) for name, value in zip(model.names, theta_initial, strict=True)
            },
            "initial_diagnostics": covariance_initial_metadata,
            "initial_map": map_initial,
            "fixed_point_converged": fixed_point_converged,
            "fixed_point_parameter_tolerance": {
                name: float(value) for name, value in zip(model.names, parameter_tolerance, strict=True)
            },
            "fixed_point_history": fixed_point_history,
            "final_fiducial": {
                name: float(value) for name, value in zip(model.names, theta_covariance, strict=True)
            },
            "final_map_minus_fiducial": {
                name: float(value) for name, value in zip(model.names, fixed_point_final_delta, strict=True)
            },
            "final_diagnostics": covariance_final_metadata,
            "iteration_posterior_comparison": covariance_posterior_comparison,
            "hartlap_percival": False,
        },
        "map": map_final,
        "mcmc": final_mcmc,
        "mcmc_initial_covariance": initial_mcmc,
        "residual_diagnostic_not_plotted": {
            "max_abs_sigma": float(np.max(np.abs(residual_sigma))),
            "rms_sigma": float(np.sqrt(np.mean(residual_sigma**2))),
        },
        "fit_quality_policy": "data chi2/dof and PTE are diagnostics, not pass/fail gates",
        "gates": gates,
        "elapsed_sec": elapsed,
    }
    atomic_savez(
        output_npz,
        coordinate=np.asarray(measured["k"] if probe == "pk" else measured["s"], dtype="f8"),
        coordinate_edges=np.asarray(coordinate_edges, dtype="f8"),
        radial_edges=np.asarray(radial_edges, dtype="f8"),
        nmodes=np.asarray(measured["nmodes"] if probe == "pk" else [], dtype="f8"),
        data=np.asarray(measured["data"], dtype="f8"),
        prediction_map=final_prediction,
        residual_sigma=np.asarray(residual_sigma, dtype="f8"),
        covariance_initial=np.asarray(covariance_initial, dtype="f8"),
        covariance_final=np.asarray(covariance_final, dtype="f8"),
        chain_initial_by_step=np.asarray(initial_chain, dtype="f8"),
        logp_initial_by_step=np.asarray(initial_logp, dtype="f8"),
        chain_final_by_step=np.asarray(final_chain, dtype="f8"),
        logp_final_by_step=np.asarray(final_logp, dtype="f8"),
        parameter_names=np.asarray(model.names),
        tracer=np.asarray(tracer_token),
        input_baseline_fnl=np.asarray(spec.fnl, dtype="f8"),
        p_prior_mean=np.asarray(P_GAUSSIAN_PRIOR_MEAN, dtype="f8"),
        p_prior_sigma=np.asarray(P_GAUSSIAN_PRIOR_SIGMA, dtype="f8"),
        nbar=np.asarray(measured["nbar"], dtype="f8"),
        ndata=np.asarray(measured["ndata"], dtype="i8"),
        boxsize=np.asarray(BOX_SIZE, dtype="f8"),
        volume=np.asarray(BOX_VOLUME, dtype="f8"),
        kfund=np.asarray(K_FUND, dtype="f8"),
        summary_json=np.asarray(json.dumps(to_jsonable(summary), sort_keys=True)),
    )
    summary["output_npz"] = str(output_npz)
    summary["output_npz_sha256"] = sha256_file(output_npz)
    atomic_write_json(output_json, summary)
    fnl = final_mcmc["posterior"]["fnl"]
    print(
        f"[done] {tag} {tracer} {probe} Gaussian-p free-fNL "
        f"fNL={fnl['q50']:.2f} -{fnl['q50']-fnl['q16']:.2f}/+{fnl['q84']-fnl['q50']:.2f} "
        f"chi2/dof={map_final['chi2_data']:.2f}/{map_final['dof']} status={status}",
        flush=True,
    )
    if status != "pass":
        print(f"[review] failed gates: {[name for name, value in gates.items() if not value]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", action="append", choices=tuple(CATALOGS))
    parser.add_argument("--probe", action="append", choices=("pk", "xi"))
    parser.add_argument("--tracer", choices=("halo", "lrg", "both"), default="halo")
    parser.add_argument("--kmax", type=float, choices=(0.08, 0.10), default=0.10)
    parser.add_argument("--smin", type=float, choices=(30.0, 50.0), default=30.0)
    parser.add_argument("--smax", type=float, choices=(150.0, 350.0), default=150.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=12_000)
    parser.add_argument("--burnin", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--resume-steps", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if int(args.burnin) >= int(args.nsteps):
        raise ValueError("burnin must be less than nsteps")
    set_cpu_affinity(int(args.threads))
    ensure_output_dirs()
    theory_cache_path = build_theory_cache(kmax=5.0, smin=float(args.smin), overwrite=False)
    theory = load_theory(theory_cache_path, smin=float(args.smin))
    tags = list(CATALOGS) if not args.tag else [get_spec(tag).tag for tag in args.tag]
    probes = ["pk", "xi"] if not args.probe else list(dict.fromkeys(args.probe))
    tracers = ("halo", "lrg") if args.tracer == "both" else (args.tracer,)
    for tracer in tracers:
        for tag in tags:
            for probe in probes:
                seed_offset = (
                    (0 if tag == "c300" else 3000)
                    + (0 if probe == "pk" else 10_000)
                    + (0 if tracer == "halo" else 30_000)
                    + (80_000 if np.isclose(args.kmax, 0.08) else 0)
                    + (50_000 if np.isclose(args.smin, 50.0) else 0)
                    + (350_000 if np.isclose(args.smax, 350.0) else 0)
                )
                if int(args.resume_steps) > 0:
                    resume_one(
                        tag,
                        probe,
                        tracer=tracer,
                        theory=theory,
                        kmax_contract=float(args.kmax),
                        smin=float(args.smin),
                        smax=float(args.smax),
                        extra_steps=int(args.resume_steps),
                        seed=int(args.seed) + seed_offset,
                    )
                else:
                    run_one(
                        tag,
                        probe,
                        tracer=tracer,
                        theory=theory,
                        theory_cache_path=theory_cache_path,
                        kmax_contract=float(args.kmax),
                        smin=float(args.smin),
                        smax=float(args.smax),
                        nwalkers=int(args.nwalkers),
                        nsteps=int(args.nsteps),
                        burnin=int(args.burnin),
                        seed=int(args.seed) + seed_offset,
                        overwrite=bool(args.overwrite),
                    )


if __name__ == "__main__":
    main()
