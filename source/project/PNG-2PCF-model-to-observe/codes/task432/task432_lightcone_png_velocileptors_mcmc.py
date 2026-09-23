#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCMC for the hybrid PNG velocileptors GSM on the standard lightcone mock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_DIR = PROJECT_ROOT / "codes" / "task43"
TASK432_DIR = PROJECT_ROOT / "codes" / "task432"
OUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "task432_model_repair" / "lightcone_png_velocileptors_gsm"

for path in (TASK43_DIR, TASK432_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from task43_run_lightcone_joint_baomask_v1 import load_rsd_specs  # noqa: E402
from task432_linear_gsm_rawbox import precision_from_covariance  # noqa: E402
from task432_lightcone_png_velocileptors import (  # noqa: E402
    LightconePNGVelocileptors,
    RSD_P_PAYLOAD,
    RSD_X_SUMMARY,
    build_cache,
)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sample_chain(name, data, covariance, evaluate, start, lower, upper, *, nwalkers, nsteps, burnin, seed):
    precision = precision_from_covariance(covariance)

    def log_prob(theta):
        theta = np.asarray(theta, dtype="f8")
        if np.any(theta <= lower) or np.any(theta >= upper):
            return -np.inf
        residual = np.asarray(data, dtype="f8") - np.asarray(evaluate(theta), dtype="f8")
        return -0.5 * float(residual @ precision @ residual)

    rng = np.random.default_rng(int(seed))
    spread = np.asarray([5.0, 0.04] if len(start) == 2 else [5.0, 0.04, 0.5, 0.03], dtype="f8")
    initial = np.asarray(start, dtype="f8")[None, :] + rng.normal(size=(int(nwalkers), len(start))) * spread[None, :]
    initial = np.maximum(initial, lower[None, :] + 1.0e-7)
    initial = np.minimum(initial, upper[None, :] - 1.0e-7)
    np.random.seed(int(seed))
    sampler = emcee.EnsembleSampler(int(nwalkers), len(start), log_prob)
    sampler.run_mcmc(initial, int(nsteps), progress=False, skip_initial_state_check=True)
    chain = sampler.get_chain(discard=int(burnin), flat=True)
    logp = sampler.get_log_prob(discard=int(burnin), flat=True)
    tau = None
    try:
        tau = np.asarray(emcee.autocorr.integrated_time(sampler.get_chain(discard=int(burnin)), tol=0), dtype="f8")
    except Exception:
        pass
    return {
        "name": name,
        "chain": chain,
        "logp": logp,
        "summary": {
            "parameter_names": ["fNL", "b1"] if len(start) == 2 else ["fNL", "b1", "sigma_s", "sn0"],
            "nwalkers": int(nwalkers),
            "nsteps": int(nsteps),
            "burnin": int(burnin),
            "seed": int(seed),
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
            "tau": None if tau is None else tau.tolist(),
            "postburn_length_over_tau": None if tau is None else (sampler.get_chain(discard=int(burnin)).shape[0] / tau).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwalkers", type=int, default=16)
    parser.add_argument("--nsteps", type=int, default=900)
    parser.add_argument("--burnin", type=int, default=200)
    parser.add_argument("--nint", type=int, default=500)
    args = parser.parse_args()

    specs, metadata, arrays = load_rsd_specs(smin=50.0, pk_kmax=0.08)
    by_name = {spec.name: spec for spec in specs}
    spec_p = by_name["rsd_p02"]
    spec_x = by_name["rsd_xi02"]
    spec_joint = by_name["rsd_joint_p02xi02"]
    with np.load(RSD_P_PAYLOAD, allow_pickle=False) as payload:
        zeff = float(np.asarray(payload["zeff"]).item())
    cache = build_cache(zeff=zeff, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000")
    with np.load(cache, allow_pickle=False) as payload:
        k, pk, alpha = payload["k_eff"], payload["pk_dd"], payload["alpha"]
        f_growth = float(np.asarray(payload["f_growth"]).item())
    with np.load(RSD_X_SUMMARY, allow_pickle=False) as payload:
        s = np.asarray(payload["s"], dtype="f8")
    # The loader's data vector is already masked in the same order as this mask.
    mask = (s >= 50.0) & (s < 350.0) & ~((s >= 80.0) & (s < 120.0))
    model = LightconePNGVelocileptors(k, pk, alpha, f_growth, s)

    def evaluate_x(theta):
        values = model.evaluate(fnl=float(theta[0]), b1=float(theta[1]), nint=int(args.nint))
        return np.concatenate((values[0][mask], values[2][mask]))

    def evaluate_joint(theta):
        return np.concatenate((spec_p.evaluate(np.asarray(theta, dtype="f8")), evaluate_x(theta)))

    xi_result = sample_chain(
        "xi_only",
        spec_x.data,
        spec_x.covariance,
        evaluate_x,
        np.asarray([-5.6, 2.437]),
        np.asarray([-500.0, 0.5]),
        np.asarray([500.0, 5.0]),
        nwalkers=args.nwalkers,
        nsteps=args.nsteps,
        burnin=args.burnin,
        seed=20260926,
    )
    joint_result = sample_chain(
        "joint",
        spec_joint.data,
        spec_joint.covariance,
        evaluate_joint,
        np.asarray([-7.7, 2.39, 2.17, 0.07]),
        np.asarray([-500.0, 0.5, 0.0, -1.0]),
        np.asarray([500.0, 5.0, 30.0, 1.0]),
        nwalkers=args.nwalkers,
        nsteps=args.nsteps,
        burnin=args.burnin,
        seed=20260927,
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {"metadata": metadata, "xi_only": xi_result["summary"], "joint": joint_result["summary"], "nint": int(args.nint)}
    np.savez_compressed(OUT_ROOT / "lightcone_png_xi_only_samples.npz", samples=xi_result["chain"], logp=xi_result["logp"])
    np.savez_compressed(OUT_ROOT / "lightcone_png_joint_samples.npz", samples=joint_result["chain"], logp=joint_result["logp"])
    atomic_json(OUT_ROOT / "task432_lightcone_png_velocileptors_mcmc_summary.json", summary)
    print(json.dumps({"status": "complete", "xi_only": xi_result["summary"], "joint": joint_result["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
