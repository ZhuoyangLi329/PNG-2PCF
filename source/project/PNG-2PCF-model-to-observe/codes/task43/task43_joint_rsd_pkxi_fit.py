#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Joint P0(k)+xi0(s) fNL fit for the boxsafe redshift-space lightcone.

Five variants under one contract:

* ``pk_marginal``  : 15-bin window-convolved P0, free (fNL, b1, sigma_s, sn0);
* ``xi_marginal``  : 30-bin formal-GIC xi0 (s=50..350), free (fNL, b1, sigma_s);
* ``joint``        : 45 bins with the full cross-covariance;
* ``joint_naive``  : cross-covariance set to zero (misspecification diagnostic);
* ``joint_half``   : cross-covariance halved (sensitivity diagnostic).

Everything else mirrors the audited boxsafe pair: the P0 model is the frozen
WindowConvolvedPk0Model surrogate (continuous ell024 W_geom window, squared
Lorentzian FoG, sn0 through the shot response), the xi0 model is the frozen
FastWindowRSDModel (FullDiscrete kmax=5 cache + equal-phase mean formal-GIC
window, no stochastic term), sigma_s is shared, and the quoted posterior uses
the single-lightcone covariance (never divided by 25).

Priors for every variant are the intersection of the audited sides:
fNL in [-500, 500], b1 in [0.5, 5] (the audited xi0 side allowed [0.2, 10];
the posterior sits near 2.34 +- 0.1 so this restriction is immaterial),
sigma_s in [0, 30], sn0 in [-1, 1] (P0 side only).

Precision uses the correlation-matrix-normalized absolute-floor inversion
carried over from the realspace joint experiment: the 45x45 spectrum spans ~17
decades and a plain ``np.linalg.inv`` silently drops the xi-side modes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import numpy as np
from scipy.optimize import least_squares

from task43_fit_rsd_lightcone_pk0_vs_xi0_smin50 import (
    WindowConvolvedPk0Model,
    load_pk,
)
from task43_fit_rsd_lightcone_x25 import FastWindowRSDModel, load_window
from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import set_affinity, split_rhat
from task43_rsd_common import OUTPUT_ROOT, atomic_savez, atomic_write_json, sha256_file
from task43_rsd_model import FullDiscreteRSDModel, build_cache


BOXSAFE_ROOT = OUTPUT_ROOT / "lightcone_boxsafe_zobs0p4_0p8"
PAYLOAD_NPZ = (
    BOXSAFE_ROOT / "pk_summary"
    / "task43_rsd_boxsafe_lightcone_pk0_x25_kmin0p004291_kmax0p10_l0only_15bin.npz"
)
CLOSURE_NPZ = (
    BOXSAFE_ROOT / "closure" / "task43_rsd_boxsafe_lightcone_x25_xi0_smin50_formalgic_lorentzian.npz"
)
CLOSURE_JSON = CLOSURE_NPZ.with_suffix(".json")
MEAN_WINDOW_NPZ = (
    BOXSAFE_ROOT / "formal_gic_windows"
    / "task43_rsd_boxsafe_formal_gic_x25_equal_phase_mean_nsub200000_seedbase430340.npz"
)
JOINT_COV_NPZ = (
    BOXSAFE_ROOT / "joint_pkxi_s50_350" / "covariance"
    / "task43_rsd_joint_cov_boxsafe_mesh64_s50_350.npz"
)
FITS_DIR = BOXSAFE_ROOT / "joint_pkxi_s50_350" / "fits"
AUDIT_DIR = BOXSAFE_ROOT / "joint_pkxi_s50_350" / "audits"

JOINT_PARAMS = ("fNL", "b1", "sigma_s", "sn0")
BOUNDS_LO = np.asarray([-500.0, 0.5, 0.0, -1.0], dtype="f8")
BOUNDS_HI = np.asarray([500.0, 5.0, 30.0, 1.0], dtype="f8")
INIT_SCALE = np.asarray([4.0, 0.015, 0.12, 0.025], dtype="f8")
OPTIMIZER_STARTS = (
    np.asarray([0.0, 2.55, 8.0, 0.0], dtype="f8"),
    np.asarray([-80.0, 2.4, 4.0, 0.2], dtype="f8"),
    np.asarray([80.0, 2.7, 12.0, -0.2], dtype="f8"),
    np.asarray([0.0, 2.5, 15.0, 0.5], dtype="f8"),
)
VARIANTS = ("pk_marginal", "xi_marginal", "joint", "joint_naive", "joint_half")
FISHER_STEPS = np.asarray([1.0, 0.01, 0.05, 0.01], dtype="f8")


def build_precision(covariance: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Correlation-normalized inversion with an absolute eigenvalue floor only."""
    cov = 0.5 * (np.asarray(covariance, dtype="f8") + np.asarray(covariance, dtype="f8").T)
    scale = np.sqrt(np.clip(np.diag(cov), 1.0e-300, None))
    scaled = cov / np.outer(scale, scale)
    evals, evecs = np.linalg.eigh(scaled)
    floor = 1.0e-14
    inv_evals = np.where(evals >= floor, 1.0 / np.maximum(evals, floor), 0.0)
    inv_scaled = (evecs * inv_evals[None, :]) @ evecs.T
    meta = {
        "scaled_eigenvalues": {
            "min": float(evals[0]),
            "max": float(evals[-1]),
            "n_below_floor": int(np.count_nonzero(evals < floor)),
        }
    }
    return inv_scaled / np.outer(scale, scale), meta


def summarize_chain(chain: np.ndarray, logp: np.ndarray, names: tuple[str, ...]) -> dict[str, Any]:
    flat = np.asarray(chain, dtype="f8").reshape(-1, len(names))
    flat_logp = np.asarray(logp, dtype="f8").reshape(-1)
    quantiles = np.percentile(flat, [16.0, 50.0, 84.0], axis=0)
    sigma68 = 0.5 * (quantiles[2] - quantiles[0])
    try:
        tau = np.asarray(emcee.autocorr.integrated_time(chain, quiet=True, tol=0), dtype="f8")
    except Exception:
        tau = np.full(chain.shape[-1], np.inf)
    rhat = split_rhat(chain)
    half = chain.shape[0] // 2
    first = chain[:half].reshape(-1, chain.shape[-1])
    second = chain[-half:].reshape(-1, chain.shape[-1])
    half_shift = np.abs(np.median(first, axis=0) - np.median(second, axis=0)) / sigma68
    imax = int(np.argmax(flat_logp))
    return {
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
        "tau": {name: float(tau[index]) for index, name in enumerate(names)},
        "postburn_length_over_tau": {name: float(chain.shape[0] / tau[index]) for index, name in enumerate(names)},
        "split_rhat": {name: float(rhat[index]) for index, name in enumerate(names)},
        "half_chain_shift_sigma": {name: float(half_shift[index]) for index, name in enumerate(names)},
        "gates": {
            "split_rhat_max_below_1p01": bool(np.max(rhat) < 1.01),
            "postburn_length_min_above_50tau": bool(np.min(chain.shape[0] / tau) > 50.0),
            "half_chain_shift_max_below_0p1sigma": bool(np.max(half_shift) < 0.1),
        },
    }


def fit_map(
    evaluate: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    covariance: np.ndarray,
    nparams: int,
) -> dict[str, Any]:
    chol = np.linalg.cholesky(0.5 * (np.asarray(covariance, dtype="f8") + np.asarray(covariance, dtype="f8").T))

    def residual(theta: np.ndarray) -> np.ndarray:
        return np.linalg.solve(chol, np.asarray(data) - evaluate(theta))

    solutions = [
        least_squares(
            residual,
            start[:nparams],
            bounds=(BOUNDS_LO[:nparams], BOUNDS_HI[:nparams]),
            max_nfev=3000,
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
        )
        for start in OPTIMIZER_STARTS
    ]
    best = min(solutions, key=lambda result: float(result.fun @ result.fun))
    chi2 = float(best.fun @ best.fun)
    return {
        "theta": [float(v) for v in best.x],
        "theta_map": np.asarray(best.x, dtype="f8"),
        "chi2": chi2,
        "dof": int(np.asarray(data).size - nparams),
        "success": bool(best.success),
    }


def run_chain(
    evaluate: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    precision: np.ndarray,
    nominal_theta: np.ndarray,
    nparams: int,
    *,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    names = JOINT_PARAMS[:nparams]

    def log_probability(theta: np.ndarray) -> float:
        values = np.asarray(theta, dtype="f8")
        if np.any(values < BOUNDS_LO[:nparams]) or np.any(values > BOUNDS_HI[:nparams]):
            return -np.inf
        diff = np.asarray(data) - evaluate(values)
        return -0.5 * float(diff @ precision @ diff)

    rng = np.random.default_rng(int(seed))
    initial = np.asarray(nominal_theta, dtype="f8")[None, :] + rng.normal(
        size=(int(nwalkers), nparams)
    ) * INIT_SCALE[:nparams][None, :]
    initial = np.clip(initial, BOUNDS_LO[:nparams] + 1.0e-7, BOUNDS_HI[:nparams] - 1.0e-7)
    sampler = emcee.EnsembleSampler(int(nwalkers), nparams, log_probability)
    sampler.run_mcmc(initial, int(nsteps), progress=False)
    chain = np.asarray(sampler.get_chain(discard=int(burnin)), dtype="f8")
    logp = np.asarray(sampler.get_log_prob(discard=int(burnin)), dtype="f8")
    summary = summarize_chain(chain, logp, names)
    summary.update(
        {
            "nwalkers": int(nwalkers),
            "nsteps": int(nsteps),
            "burnin": int(burnin),
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
        }
    )
    return summary, chain, logp


def fisher_covariance(
    evaluate: Callable[[np.ndarray], np.ndarray],
    precision: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    nparams = int(theta.size)
    columns = []
    for index in range(nparams):
        step = float(FISHER_STEPS[index])
        plus, minus = np.asarray(theta, dtype="f8").copy(), np.asarray(theta, dtype="f8").copy()
        plus[index] += step
        minus[index] -= step
        columns.append((evaluate(plus) - evaluate(minus)) / (2.0 * step))
    jacobian = np.column_stack(columns)
    return np.linalg.inv(jacobian.T @ precision @ jacobian)


class JointModel:
    """P0 window surrogate + formal-GIC xi0 surrogate, sigma_s shared."""

    def __init__(self, pk_model: WindowConvolvedPk0Model, xi_model: FastWindowRSDModel, mask: np.ndarray) -> None:
        self.pk_model = pk_model
        self.xi_model = xi_model
        self.mask = np.asarray(mask, dtype=bool)

    def prediction(self, theta: np.ndarray) -> np.ndarray:
        values = np.asarray(theta, dtype="f8")
        p0 = self.pk_model.evaluate(values)
        xi = self.xi_model.evaluate(values[:3], model="formal_gic", window_key="mean")[0]
        return np.concatenate([p0, xi[self.mask]])

    def prediction_pk(self, theta: np.ndarray) -> np.ndarray:
        return self.pk_model.evaluate(np.asarray(theta, dtype="f8"))

    def prediction_xi(self, theta: np.ndarray) -> np.ndarray:
        values = np.asarray(theta, dtype="f8")
        xi = self.xi_model.evaluate(values[:3], model="formal_gic", window_key="mean")[0]
        return xi[self.mask]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--variants", type=str, default=",".join(VARIANTS))
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--smoke", action="store_true", help="64x400 plumbing check for the joint variant only")
    args = parser.parse_args()
    nsteps = 400 if args.smoke else int(args.nsteps)
    burnin = 100 if args.smoke else int(args.burnin)
    variants = ("joint",) if args.smoke else tuple(v.strip() for v in args.variants.split(",") if v.strip())
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants {unknown}")

    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))

    pk_payload = load_pk(PAYLOAD_NPZ)
    closure_summary = json.loads(CLOSURE_JSON.read_text(encoding="utf-8"))
    if closure_summary.get("output_npz_sha256") != sha256_file(CLOSURE_NPZ):
        raise RuntimeError("closure npz hash gate failed")
    with np.load(CLOSURE_NPZ, allow_pickle=False) as data:
        s_full = np.asarray(data["s"], dtype="f8")
        xi_mean_full = np.asarray(data["xi_multipoles_mean"], dtype="f8")[0]
    mask = s_full >= 50.0
    if int(np.count_nonzero(mask)) != 30:
        raise RuntimeError("xi fit mask is not the 30-bin s=50..350 selection")
    xi_mean = xi_mean_full[mask]
    pk_mean = np.asarray(pk_payload["pk_mean"], dtype="f8")

    pk_model = WindowConvolvedPk0Model(PAYLOAD_NPZ, sigma_step=0.05, nmu=96)
    pk_validation = pk_model.validate()
    if pk_validation["status"] != "pass":
        raise RuntimeError(f"P0 surrogate failed: {pk_validation}")
    zeff = float(np.asarray(pk_payload["zeff"]).item())
    theory_cache = build_cache(zeff=zeff, boxsize=2000.0, kmax=5.0, ells=(0, 2), cosmology="abacus_c000")
    xi_model = FastWindowRSDModel(
        FullDiscreteRSDModel(theory_cache, nmu=64),
        {"mean": load_window(MEAN_WINDOW_NPZ)},
        sigma_step=0.05,
    )
    xi_validation = xi_model.validate()
    if xi_validation["status"] != "pass":
        raise RuntimeError(f"xi0 surrogate failed: {xi_validation}")

    model = JointModel(pk_model, xi_model, mask)
    joint_data = np.concatenate([pk_mean, xi_mean])

    with np.load(JOINT_COV_NPZ, allow_pickle=False) as data:
        joint_npz = {key: np.asarray(data[key]) for key in data.files}
    cov_audit = json.loads(JOINT_COV_NPZ.with_suffix(".json").read_text(encoding="utf-8"))
    if cov_audit.get("status") != "done":
        raise RuntimeError("joint covariance audit is not done")

    cross_scales = {"joint": 1.0, "joint_naive": 0.0, "joint_half": 0.5}
    results: dict[str, dict[str, Any]] = {}
    chains: dict[str, np.ndarray] = {}
    for index, variant in enumerate(variants):
        nparams = 3 if variant == "xi_marginal" else 4
        if variant == "pk_marginal":
            evaluate, data = model.prediction_pk, pk_mean
            cov = np.asarray(joint_npz["c_pp"], dtype="f8")
        elif variant == "xi_marginal":
            evaluate, data = model.prediction_xi, xi_mean
            cov = np.asarray(joint_npz["c_xx"], dtype="f8")
        else:
            evaluate, data = model.prediction, joint_data
            c_pp = np.asarray(joint_npz["c_pp"], dtype="f8")
            c_px = cross_scales[variant] * np.asarray(joint_npz["c_px"], dtype="f8")
            cov = np.block([[c_pp, c_px.T], [c_px, np.asarray(joint_npz["c_xx"], dtype="f8")]])
        precision, precision_meta = build_precision(cov)
        nominal = fit_map(evaluate, data, cov, nparams)
        theta_map = np.asarray(nominal.pop("theta_map"), dtype="f8")
        summary, chain, logp = run_chain(
            evaluate,
            data,
            precision,
            theta_map,
            nparams,
            nwalkers=int(args.nwalkers),
            nsteps=nsteps,
            burnin=burnin,
            seed=int(args.seed) + 100 * index,
        )
        fisher = fisher_covariance(evaluate, precision, theta_map)
        summary["nominal"] = nominal
        summary["precision_meta"] = precision_meta
        summary["fisher_sigma"] = {
            name: float(np.sqrt(max(np.diag(fisher)[i], 0.0))) for i, name in enumerate(JOINT_PARAMS[:nparams])
        }
        summary["mcmc_over_fisher_sigma_fNL"] = float(
            summary["posterior"]["fNL"]["sigma68"] / max(summary["fisher_sigma"]["fNL"], 1.0e-300)
        )
        results[variant] = summary
        chains[variant] = chain
        out_dir = FITS_DIR / ("smoke" if args.smoke else variant)
        out_npz = out_dir / "samples.npz"
        if out_npz.exists():
            raise FileExistsError(f"immutable chain output exists: {out_npz}")
        atomic_savez(
            out_npz,
            chain_by_step=chain,
            log_probability_by_step=logp,
            parameter_names=np.asarray(JOINT_PARAMS[:nparams]),
            prediction_map=evaluate(theta_map),
            data=data,
        )
        print(
            json.dumps(
                {
                    "variant": variant,
                    "fNL": summary["posterior"]["fNL"],
                    "gates": summary["gates"],
                    "mcmc_over_fisher_sigma_fNL": summary["mcmc_over_fisher_sigma_fNL"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not args.smoke and not all(summary["gates"].values()):
            raise SystemExit(f"MCMC gates failed for {variant}: {summary['gates']}")

    if args.smoke:
        print(json.dumps({"status": "smoke_ok", "elapsed_sec": time.perf_counter() - started}, sort_keys=True))
        return

    pk_sigma = results["pk_marginal"]["posterior"]["fNL"]["sigma68"]
    xi_sigma = results["xi_marginal"]["posterior"]["fNL"]["sigma68"]
    joint_sigma = results["joint"]["posterior"]["fNL"]["sigma68"]
    naive_sigma = results["joint_naive"]["posterior"]["fNL"]["sigma68"]
    metrics = {
        "best_marginal_sigma68": float(min(pk_sigma, xi_sigma)),
        "best_marginal_variant": "pk_marginal" if pk_sigma <= xi_sigma else "xi_marginal",
        "joint_sigma68": float(joint_sigma),
        "improvement_vs_best_marginal": float(1.0 - joint_sigma / min(pk_sigma, xi_sigma)),
        "naive_sigma68": float(naive_sigma),
        "naive_improvement_vs_best_marginal": float(1.0 - naive_sigma / min(pk_sigma, xi_sigma)),
        "cross_covariance_cost_ratio_joint_over_naive": float(joint_sigma / naive_sigma),
        "b1_sigma": {
            name: float(results[name]["posterior"]["b1"]["sigma68"])
            for name in ("pk_marginal", "xi_marginal", "joint")
        },
        "sigma_s_sigma": {
            name: float(results[name]["posterior"]["sigma_s"]["sigma68"])
            for name in ("pk_marginal", "xi_marginal", "joint")
        },
        "audited_references": {
            "pk0_comparison_fNL_q16_q50_q84": [-50.58, -15.55, 19.47],
            "xi0_closure_fNL_q16_q50_q84": [-61.68, -21.73, 12.58],
            "note": "audited posteriors quoted from the frozen pair audit for cross-checking only",
        },
    }
    audit = {
        "task": "task43_joint_rsd_pkxi_fit",
        "status": "complete",
        "scope": (
            "boxsafe 0.4<zobs<0.8 RSD lightcone; P0 15 bins (geometry-only W_geom[P0,P2,P4], "
            "free sn0) + formal-GIC xi0 s=50..350 (no stochastic term); shared (fNL, b1, sigma_s)"
        ),
        "caveats": [
            "diagnostic Gaussian survey-window covariance family, not science-ready",
            "the frozen RSD closure status is validation_failed (xi0 mean-shape PTE=1.3e-3); "
            "the joint gain is a covariance-structure diagnostic within that family",
            "joint priors use b1 in [0.5, 5] (P0-side intersection; audited xi0 side allowed [0.2, 10])",
        ],
        "mcmc": {"nwalkers": int(args.nwalkers), "nsteps": int(args.nsteps), "burnin": int(burnin), "seed": int(args.seed)},
        "priors": {name: [float(BOUNDS_LO[i]), float(BOUNDS_HI[i])] for i, name in enumerate(JOINT_PARAMS)},
        "models": {
            "pk0": {"window": "W_geom[P0,P2,P4] frozen payload surrogate", "surrogate_validation": pk_validation["status"]},
            "xi0": {
                "theory_cache": str(theory_cache),
                "theory_cache_sha256": sha256_file(theory_cache),
                "mean_window": str(MEAN_WINDOW_NPZ),
                "mean_window_sha256": sha256_file(MEAN_WINDOW_NPZ),
                "surrogate_validation": xi_validation["status"],
            },
        },
        "inputs": {
            "joint_covariance": {"path": str(JOINT_COV_NPZ), "sha256": sha256_file(JOINT_COV_NPZ)},
            "pk_payload": {"path": str(PAYLOAD_NPZ), "sha256": sha256_file(PAYLOAD_NPZ)},
            "closure_npz": {"path": str(CLOSURE_NPZ), "sha256": sha256_file(CLOSURE_NPZ)},
        },
        "results": results,
        "metrics": metrics,
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    out_json = AUDIT_DIR / "task43_joint_rsd_pkxi_fit_summary.json"
    if out_json.exists():
        raise FileExistsError(f"immutable fit summary exists: {out_json}")
    atomic_write_json(out_json, audit)
    print(json.dumps({"status": "complete", "metrics": metrics, "output": str(out_json)}, sort_keys=True))


if __name__ == "__main__":
    main()
