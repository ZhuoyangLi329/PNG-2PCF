#!/usr/bin/env python3
"""Free-growth diagnostic for the Task 4.3 box-safe halo lightcone.

The fixed-growth standard BAO-masked P02, xi02, and joint likelihoods are
reused without changing their data vectors or single-realization covariance.
Only one parameter is added: the linear growth rate f.
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
    os.environ[_name] = "1"

import emcee
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

from task43_fit_rsd_lightcone_x25 import FastWindowRSDModel, load_window
from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import set_affinity
from task43_rsd_boxsafe_p02_increment import (
    MEASURE_DIR as RSD_P_MEASURE_DIR,
    PAYLOAD_NPZ as RSD_P_PAYLOAD,
    WindowConvolvedP02Model,
)
from task43_rsd_common import PHASES, PROJECT_ROOT, atomic_savez, atomic_write_json, sha256_file
from task43_rsd_joint_p02xi02_fit import ELL2_SUMMARY as RSD_X_SUMMARY, MEAN_WINDOW_NPZ as RSD_MEAN_WINDOW
from task43_rsd_model import DELTA_C, FullDiscreteRSDModel, build_cache
from task43_run_lightcone_joint_baomask_v1 import (
    DEFAULT_ROOT as FIXED_ROOT,
    ScaledGaussianMetric,
    lightcone_xi_primary_mask,
)


OUTPUT_ROOT = PROJECT_ROOT / "outputs/task43_outputs/rsd_validation/lightcone_free_growth_baomask80_120_v1"
AUDIT_PATH = OUTPUT_ROOT / "task43_lightcone_free_growth_baomask80_120_v1.json"
ARRAY_PATH = OUTPUT_ROOT / "task43_lightcone_free_growth_baomask80_120_arrays_v1.npz"
FIXED_AUDIT_PATH = FIXED_ROOT / "task43_lightcone_standard_joint_baomask80_120_v1.json"
VARIANTS = ("rsd_p02", "rsd_xi02", "rsd_joint_p02xi02")
FREE_NAMES = {
    "rsd_p02": ("fNL", "b1", "sigma_s", "f_growth", "sn0"),
    "rsd_xi02": ("fNL", "b1", "sigma_s", "f_growth"),
    "rsd_joint_p02xi02": ("fNL", "b1", "sigma_s", "f_growth", "sn0"),
}
BOUNDS = {
    "fNL": (-500.0, 500.0),
    "b1": (0.5, 5.0),
    "sigma_s": (0.0, 30.0),
    "f_growth": (0.2, 1.4),
    "sn0": (-1.0, 1.0),
}
INITIAL_SCALE = {"fNL": 4.0, "b1": 0.015, "sigma_s": 0.12, "f_growth": 0.012, "sn0": 0.025}
_POOL_LOG_PROBABILITY: Callable[[np.ndarray], float] | None = None


def _pool_log_probability(theta: np.ndarray) -> float:
    if _POOL_LOG_PROBABILITY is None:
        raise RuntimeError("MCMC worker likelihood was not initialized")
    return _POOL_LOG_PROBABILITY(theta)


def growth_coefficients(fnl: float, b1: float, f_growth: float) -> np.ndarray:
    """Polynomial coefficients for [b1 + Delta_b_PNG + f mu^2]^2."""
    q = float(fnl) * 2.0 * DELTA_C * (float(b1) - 1.0)
    return np.asarray(
        [b1 * b1, 2.0 * b1 * q, q * q, 2.0 * b1 * f_growth, 2.0 * q * f_growth, f_growth * f_growth],
        dtype="f8",
    )


class FreeGrowthPrediction:
    """The standard P02 and formal-GIC xi02 surrogates with explicit f."""

    def __init__(self) -> None:
        from task43_fit_rsd_lightcone_pk0_vs_xi0_smin50 import load_pk

        payload = load_pk(RSD_P_PAYLOAD)
        self.k0 = np.asarray(payload["k_obs"], dtype="f8")
        fit_indices = np.asarray(payload["fit_bin_indices"], dtype="i8")
        self.p2_keep = np.flatnonzero(self.k0 >= 0.015 - 1.0e-12)
        zeff_p = float(np.asarray(payload["zeff"]).item())
        phase0 = RSD_P_MEASURE_DIR / "task43_rsd_lightcone_p02_ph000_mesh256_kmax0p300_dk0p002.npz"
        with np.load(phase0, allow_pickle=False) as data:
            p_row_ids = np.concatenate((fit_indices, 150 + fit_indices[self.p2_keep]))
            window = np.asarray(data["window_matrix"], dtype="f8")[p_row_ids]
            theory_k = np.asarray(data["theory_k"], dtype="f8")
            theory_ell = np.asarray(data["theory_ell"], dtype="i8")
        self.p_model = WindowConvolvedP02Model(window, theory_k, theory_ell, zeff=zeff_p, sigma_step=0.05)

        with np.load(RSD_X_SUMMARY, allow_pickle=False) as payload_x:
            self.s = np.asarray(payload_x["s"], dtype="f8")
            zeff_x = float(np.mean(np.asarray(payload_x["zeff_by_phase"], dtype="f8")))
        self.x_mask = lightcone_xi_primary_mask(self.s, smin=50.0)
        excluded = self.s[(self.s >= 50.0) & (self.s < 350.0) & ~self.x_mask]
        if not np.array_equal(excluded, [85.0, 95.0, 105.0, 115.0]):
            raise RuntimeError("canonical BAO exclusion changed")
        cache = build_cache(zeff=zeff_x, boxsize=2000.0, kmax=5.0, ells=(0, 2), cosmology="abacus_c000")
        self.x_model = FastWindowRSDModel(
            FullDiscreteRSDModel(cache, nmu=64),
            {"mean": load_window(RSD_MEAN_WINDOW)},
            sigma_step=0.05,
        )
        self.zeff_p = zeff_p
        self.zeff_x = zeff_x
        self.fiducial_growth_p = float(self.p_model.f_growth)
        self.fiducial_growth_x = float(self.x_model.exact.f_growth)
        if not np.isclose(self.fiducial_growth_p, self.fiducial_growth_x, rtol=0.0, atol=1.0e-14):
            raise RuntimeError("P02 and xi02 fiducial growth rates differ")

    @property
    def fiducial_growth(self) -> float:
        return self.fiducial_growth_p

    def p02(self, theta: np.ndarray) -> np.ndarray:
        fnl, b1, sigma_s, f_growth, sn0 = map(float, np.asarray(theta, dtype="f8"))
        coefficients = growth_coefficients(fnl, b1, f_growth)
        return np.asarray(self.p_model.spline(sigma_s), dtype="f8") @ coefficients + sn0 * self.p_model.shot_response

    def xi02(self, theta: np.ndarray) -> np.ndarray:
        fnl, b1, sigma_s, f_growth = map(float, np.asarray(theta, dtype="f8")[:4])
        coefficients = growth_coefficients(fnl, b1, f_growth)
        basis = np.asarray(self.x_model.base.spline(sigma_s), dtype="f8")
        values = {
            int(ell): coefficients @ basis[index]
            for index, ell in enumerate(self.x_model.exact.ell_values)
        }
        gic_index = self.x_model.window_index["mean"]
        gic_basis = np.asarray(self.x_model.gic_spline(sigma_s), dtype="f8")[gic_index]
        values[0] -= float(coefficients @ gic_basis)
        return np.concatenate((values[0][self.x_mask], values[2][self.x_mask]))

    def joint(self, theta: np.ndarray) -> np.ndarray:
        values = np.asarray(theta, dtype="f8")
        return np.concatenate((self.p02(values), self.xi02(values[:4])))


@dataclass(frozen=True)
class FitSpec:
    name: str
    names: tuple[str, ...]
    data: np.ndarray
    phase_data: np.ndarray
    covariance: np.ndarray
    evaluate: Callable[[np.ndarray], np.ndarray]
    fixed_theta: np.ndarray
    fixed_prediction: np.ndarray
    block_slices: dict[str, tuple[int, int]]


def load_fixed_variant(name: str, fixed_audit: dict[str, Any]) -> dict[str, np.ndarray]:
    path = FIXED_ROOT / "fits" / name / "samples.npz"
    summary_path = path.with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "pass" or summary.get("output_npz_sha256") != sha256_file(path):
        raise RuntimeError(f"fixed-f source failed validation: {name}")
    if not all(summary["result"]["mcmc"]["gates"].values()):
        raise RuntimeError(f"fixed-f source chain failed convergence: {name}")
    if fixed_audit["results"][name]["parameter_names"] != np.load(path, allow_pickle=False)["parameter_names"].tolist():
        raise RuntimeError(f"fixed-f parameter contract changed: {name}")
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def expand_fixed_theta(name: str, theta: np.ndarray, f_growth: float) -> np.ndarray:
    values = list(map(float, np.asarray(theta, dtype="f8")))
    if name in ("rsd_p02", "rsd_joint_p02xi02"):
        return np.asarray([values[0], values[1], values[2], f_growth, values[3]], dtype="f8")
    return np.asarray([values[0], values[1], values[2], f_growth], dtype="f8")


def build_specs(models: FreeGrowthPrediction, fixed_audit: dict[str, Any]) -> tuple[dict[str, FitSpec], dict[str, Any]]:
    fixed = {name: load_fixed_variant(name, fixed_audit) for name in VARIANTS}
    np0, np2 = models.k0.size, models.p2_keep.size
    nx0 = int(np.count_nonzero(models.x_mask))
    slices = {
        "rsd_p02": {"P0": (0, np0), "P2": (np0, np0 + np2)},
        "rsd_xi02": {"xi0": (0, nx0), "xi2": (nx0, 2 * nx0)},
        "rsd_joint_p02xi02": {
            "P0": (0, np0),
            "P2": (np0, np0 + np2),
            "xi0": (np0 + np2, np0 + np2 + nx0),
            "xi2": (np0 + np2 + nx0, np0 + np2 + 2 * nx0),
        },
    }
    evaluators = {"rsd_p02": models.p02, "rsd_xi02": models.xi02, "rsd_joint_p02xi02": models.joint}
    specs: dict[str, FitSpec] = {}
    reproduction: dict[str, Any] = {}
    for name in VARIANTS:
        source = fixed[name]
        free_at_fixed = expand_fixed_theta(name, source["theta_maximum_likelihood"], models.fiducial_growth)
        prediction = evaluators[name](free_at_fixed)
        reference = np.asarray(source["prediction_maximum_likelihood"], dtype="f8")
        max_abs = float(np.max(np.abs(prediction - reference)))
        relative_l2 = float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1.0e-300))
        if relative_l2 >= 1.0e-12:
            raise RuntimeError(f"free-f model does not reproduce fixed-f prediction for {name}: {relative_l2}")
        reproduction[name] = {"max_abs": max_abs, "relative_l2": relative_l2}
        specs[name] = FitSpec(
            name=name,
            names=FREE_NAMES[name],
            data=np.asarray(source["data"], dtype="f8"),
            phase_data=np.asarray(source["phase_data"], dtype="f8"),
            covariance=np.asarray(source["covariance_single"], dtype="f8"),
            evaluate=evaluators[name],
            fixed_theta=free_at_fixed,
            fixed_prediction=reference,
            block_slices=slices[name],
        )
    return specs, reproduction


def bounds_for(names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([BOUNDS[name][0] for name in names], dtype="f8"),
        np.asarray([BOUNDS[name][1] for name in names], dtype="f8"),
    )


def fit_map(spec: FitSpec) -> tuple[dict[str, Any], np.ndarray]:
    metric = ScaledGaussianMetric(spec.covariance)
    lower, upper = bounds_for(spec.names)

    def residual(theta: np.ndarray) -> np.ndarray:
        return metric.residual(spec.data - spec.evaluate(np.asarray(theta, dtype="f8")))

    starts = [spec.fixed_theta.copy()]
    for f_growth in (0.45, 0.65, 0.80, 0.95, 1.15):
        for fnl in (-80.0, 0.0, 80.0):
            start = spec.fixed_theta.copy()
            start[0] = fnl
            start[3] = f_growth
            starts.append(start)
    solutions = [
        least_squares(
            residual,
            np.clip(start, lower + 1.0e-9, upper - 1.0e-9),
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=5000,
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
        )
        for start in starts
    ]
    good = [solution for solution in solutions if solution.success]
    if not good:
        raise RuntimeError(f"no free-f MAP start converged for {spec.name}")
    best = min(good, key=lambda solution: float(solution.fun @ solution.fun))
    theta = np.asarray(best.x, dtype="f8")
    chi2 = float(best.fun @ best.fun)
    dof = int(spec.data.size - theta.size)
    return (
        {
            "theta": {name: float(value) for name, value in zip(spec.names, theta, strict=True)},
            "chi2": chi2,
            "dof": dof,
            "pte": float(chi2_distribution.sf(chi2, dof)),
            "success": bool(best.success),
            "message": str(best.message),
            "number_of_starts": len(starts),
        },
        theta,
    )


def chain_diagnostics(chain: np.ndarray, names: tuple[str, ...]) -> dict[str, Any]:
    from task44_refinement_likelihood import chain_diagnostics as validated_diagnostics

    return validated_diagnostics(np.asarray(chain, dtype="f8"), names)


def run_ensemble(
    spec: FitSpec,
    theta_ml: np.ndarray,
    *,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    nworkers: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    metric = ScaledGaussianMetric(spec.covariance)
    lower, upper = bounds_for(spec.names)

    def log_probability(theta: np.ndarray) -> float:
        values = np.asarray(theta, dtype="f8")
        if np.any(~np.isfinite(values)) or np.any(values < lower) or np.any(values > upper):
            return -np.inf
        return -0.5 * metric.chi2(spec.data - spec.evaluate(values))

    rng = np.random.default_rng(seed)
    scales = np.asarray([INITIAL_SCALE[name] for name in spec.names], dtype="f8")
    initial = theta_ml[None, :] + rng.normal(size=(nwalkers, len(spec.names))) * scales[None, :]
    initial = np.clip(initial, lower + 1.0e-8, upper - 1.0e-8)
    np.random.seed(seed)
    global _POOL_LOG_PROBABILITY
    _POOL_LOG_PROBABILITY = log_probability
    started = time.perf_counter()
    try:
        if nworkers == 1:
            sampler = emcee.EnsembleSampler(nwalkers, len(spec.names), log_probability)
            sampler.run_mcmc(initial, nsteps, progress=False)
        else:
            with mp.get_context("fork").Pool(processes=nworkers) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, len(spec.names), _pool_log_probability, pool=pool)
                sampler.run_mcmc(initial, nsteps, progress=False)
    finally:
        _POOL_LOG_PROBABILITY = None
    chain = np.asarray(sampler.get_chain(discard=burnin), dtype="f8")
    logp = np.asarray(sampler.get_log_prob(discard=burnin), dtype="f8")
    summary = chain_diagnostics(chain, spec.names)
    acceptance = float(np.mean(sampler.acceptance_fraction))
    summary.update(
        seed=seed,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burnin=burnin,
        acceptance_fraction=acceptance,
        elapsed_seconds=float(time.perf_counter() - started),
    )
    summary["gates"]["acceptance_between_0p1_0p8"] = 0.1 < acceptance < 0.8
    summary["all_internal_gates"] = bool(all(summary["gates"].values()))
    return summary, chain, logp


def phase_diagnostics(spec: FitSpec, prediction: np.ndarray) -> dict[str, Any]:
    metric = ScaledGaussianMetric(spec.covariance)
    values = np.asarray([metric.chi2(row - prediction) for row in spec.phase_data], dtype="f8")
    dof = int(spec.data.size - len(spec.names))
    pte = chi2_distribution.sf(values, dof)
    return {
        "definition": "each phase at the x25-mean free-f ML model, using C_single",
        "dof": dof,
        "chi2_by_phase": values,
        "chi2_mean": float(np.mean(values)),
        "chi2_median": float(np.median(values)),
        "pte_by_phase": pte,
        "fraction_pte_above_0p05": float(np.mean(pte > 0.05)),
    }


def marginal_block_chi2(spec: FitSpec, prediction: np.ndarray) -> dict[str, float]:
    residual = spec.data - np.asarray(prediction, dtype="f8")
    out: dict[str, float] = {}
    for label, (start, stop) in spec.block_slices.items():
        covariance = spec.covariance[start:stop, start:stop]
        out[label] = ScaledGaussianMetric(covariance).chi2(residual[start:stop])
    return out


def posterior_summary(chains: list[np.ndarray], names: tuple[str, ...], theta_ml: np.ndarray) -> dict[str, Any]:
    flat = np.concatenate([chain.reshape(-1, len(names)) for chain in chains], axis=0)
    quantiles = np.percentile(flat, [16.0, 50.0, 84.0], axis=0)
    return {
        name: {
            "maximum_likelihood": float(theta_ml[index]),
            "q16": float(quantiles[0, index]),
            "q50": float(quantiles[1, index]),
            "q84": float(quantiles[2, index]),
            "sigma68": float(0.5 * (quantiles[2, index] - quantiles[0, index])),
        }
        for index, name in enumerate(names)
    }


def run_variant(
    spec: FitSpec,
    *,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed_base: int,
    nworkers: int,
    map_only: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    map_result, theta_ml = fit_map(spec)
    prediction = spec.evaluate(theta_ml)
    fixed_metric = ScaledGaussianMetric(spec.covariance)
    fixed_chi2 = fixed_metric.chi2(spec.data - spec.fixed_prediction)
    delta_chi2 = fixed_chi2 - map_result["chi2"]
    result: dict[str, Any] = {
        "parameter_names": spec.names,
        "maximum_likelihood": map_result,
        "fixed_growth_reference": {
            "theta": {name: float(value) for name, value in zip(spec.names, spec.fixed_theta, strict=True)},
            "chi2": fixed_chi2,
            "dof": int(spec.data.size - len(spec.fixed_theta) + 1),
        },
        "delta_chi2_fixed_minus_free": float(delta_chi2),
        "likelihood_ratio_pte_chi2_1_approximate": float(chi2_distribution.sf(max(delta_chi2, 0.0), 1)),
        "phase_diagnostics": phase_diagnostics(spec, prediction),
        "marginal_block_chi2": {
            "definition": "r_block^T C_block^-1 r_block; blocks do not sum when cross-covariance is nonzero",
            "fixed_growth": marginal_block_chi2(spec, spec.fixed_prediction),
            "free_growth": marginal_block_chi2(spec, prediction),
        },
    }
    arrays = {
        "data": spec.data,
        "phase_data": spec.phase_data,
        "covariance_single": spec.covariance,
        "fixed_prediction": spec.fixed_prediction,
        "free_prediction": prediction,
        "fixed_theta": spec.fixed_theta,
        "free_theta": theta_ml,
        "parameter_names": np.asarray(spec.names),
    }
    if map_only:
        result["status"] = "map_complete"
        return result, arrays

    from task44_refinement_likelihood import independent_ensemble_check

    summaries: list[dict[str, Any]] = []
    chains: list[np.ndarray] = []
    logps: list[np.ndarray] = []
    chain_paths: list[str] = []
    for index, seed in enumerate((seed_base, seed_base + 1)):
        summary, chain, logp = run_ensemble(
            spec,
            theta_ml,
            nwalkers=nwalkers,
            nsteps=nsteps,
            burnin=burnin,
            seed=seed,
            nworkers=nworkers,
        )
        path = OUTPUT_ROOT / "fits" / spec.name / f"ensemble_seed{seed}_steps{nsteps}.npz"
        atomic_savez(path, chain_by_step=chain, log_probability_by_step=logp, parameter_names=np.asarray(spec.names))
        summary["output_npz"] = str(path)
        summary["output_npz_sha256"] = sha256_file(path)
        summaries.append(summary)
        chains.append(chain)
        logps.append(logp)
        chain_paths.append(str(path))
        print(json.dumps({"variant": spec.name, "seed": seed, "gates": summary["gates"]}, sort_keys=True), flush=True)
    independent = independent_ensemble_check(chains, summaries)
    result.update(
        status="sampling_validated" if independent["pass"] else "sampling_review",
        ensembles=summaries,
        independent_check=independent,
        posterior=posterior_summary(chains, spec.names, theta_ml),
        chain_paths=chain_paths,
    )
    arrays["chain_by_ensemble"] = np.stack(chains)
    arrays["log_probability_by_ensemble"] = np.stack(logps)
    return result, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nwalkers", type=int, default=48)
    parser.add_argument("--nsteps", type=int, default=24000)
    parser.add_argument("--burnin", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--map-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        raise ValueError("--threads must be in [1,8]")
    if args.nwalkers < 2 * max(len(names) for names in FREE_NAMES.values()):
        raise ValueError("too few walkers")
    if not 0 <= args.burnin < args.nsteps:
        raise ValueError("burnin must satisfy 0 <= burnin < nsteps")
    cpus = set_affinity(args.threads)
    started = time.perf_counter()
    fixed_audit = json.loads(FIXED_AUDIT_PATH.read_text(encoding="utf-8"))
    if fixed_audit.get("status") != "pass" or not all(fixed_audit["numerical_gates"].values()):
        raise RuntimeError("fixed-growth source audit failed")
    models = FreeGrowthPrediction()
    specs, reproduction = build_specs(models, fixed_audit)
    selected_variants = tuple(dict.fromkeys(args.variants))
    retained_provenance: dict[str, Any] | None = None
    if set(selected_variants) != set(VARIANTS):
        if not AUDIT_PATH.is_file() or not ARRAY_PATH.is_file():
            raise RuntimeError("partial rerun requires an existing complete set of variant outputs")
        previous_audit_sha256 = sha256_file(AUDIT_PATH)
        previous = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        arrays = load_npz(ARRAY_PATH)
        results = dict(previous["results"])
        for name in set(VARIANTS) - set(selected_variants):
            if results[name].get("status") != "sampling_validated":
                raise RuntimeError(f"cannot retain unvalidated variant {name}")
        retained_provenance = {
            "audit_sha256_before_partial_rerun": previous_audit_sha256,
            "driver_sha256_before_partial_rerun": previous.get("driver_sha256"),
            "retained_variants": sorted(set(VARIANTS) - set(selected_variants)),
        }
    else:
        results = {}
        arrays = {"k0": models.k0, "k2": models.k0[models.p2_keep], "s": models.s[models.x_mask]}
    for name in selected_variants:
        index = VARIANTS.index(name)
        result, variant_arrays = run_variant(
            specs[name],
            nwalkers=args.nwalkers,
            nsteps=args.nsteps,
            burnin=args.burnin,
            seed_base=args.seed + 100 * index,
            nworkers=args.threads,
            map_only=args.map_only,
        )
        results[name] = result
        for key, value in variant_arrays.items():
            arrays[f"{name}_{key}"] = value
        print(
            json.dumps(
                {
                    "variant": name,
                    "status": result["status"],
                    "free_growth_ML": result["maximum_likelihood"]["theta"]["f_growth"],
                    "delta_chi2": result["delta_chi2_fixed_minus_free"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    status = "map_complete" if args.map_only else (
        "pass" if all(result["status"] == "sampling_validated" for result in results.values()) else "validation_failed"
    )
    atomic_savez(ARRAY_PATH, **arrays)
    audit = {
        "task": "task43_lightcone_free_growth_baomask80_120_v1",
        "status": status,
        "scope": "box-safe 0.4 < z_obs < 0.8 halo lightcone RSD P02, xi02, and their full-cross joint",
        "change_from_standard": "f_growth is free; all data, masks, windows, and covariance matrices are unchanged",
        "growth_prior": {"type": "uniform", "bounds": BOUNDS["f_growth"]},
        "fiducial": {"zeff_p": models.zeff_p, "zeff_x": models.zeff_x, "f_growth": models.fiducial_growth},
        "covariance_contract": {
            "physical": "single-realization normalization; divisor=1",
            "observed_curve": "mean of 25 phases",
            "free_and_fixed_identical": True,
        },
        "fixed_prediction_reproduction": reproduction,
        "results": results,
        "execution": {
            "cpus": cpus,
            "threads": args.threads,
            "nwalkers": args.nwalkers,
            "nsteps": args.nsteps,
            "burnin": args.burnin,
            "elapsed_seconds": float(time.perf_counter() - started),
            "host": os.uname().nodename,
            "selected_variants": selected_variants,
            "retained_provenance": retained_provenance,
        },
        "inputs": {
            "fixed_audit": str(FIXED_AUDIT_PATH),
            "fixed_audit_sha256": sha256_file(FIXED_AUDIT_PATH),
            "p_payload": str(RSD_P_PAYLOAD),
            "p_payload_sha256": sha256_file(RSD_P_PAYLOAD),
            "xi_summary": str(RSD_X_SUMMARY),
            "xi_summary_sha256": sha256_file(RSD_X_SUMMARY),
            "mean_window": str(RSD_MEAN_WINDOW),
            "mean_window_sha256": sha256_file(RSD_MEAN_WINDOW),
        },
        "outputs": {"arrays": str(ARRAY_PATH), "arrays_sha256": sha256_file(ARRAY_PATH)},
        "driver_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_json(AUDIT_PATH, audit)
    print(json.dumps({"status": status, "audit": str(AUDIT_PATH), "arrays": str(ARRAY_PATH)}, sort_keys=True))


if __name__ == "__main__":
    main()
