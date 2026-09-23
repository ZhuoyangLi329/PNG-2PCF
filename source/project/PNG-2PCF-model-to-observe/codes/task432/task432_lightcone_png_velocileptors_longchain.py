#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Long-chain P02/xi02/joint fits for the hybrid PNG lightcone model.

The chain contract matches the standard Task43 lightcone fit:

    64 walkers x 30000 steps, burn-in 5000, C_single, no /25.

P02 uses the frozen Task43 P-side model.  xi02 uses the validated
velocileptors/GSM hybrid PNG response.  The joint fit shares fNL and b1 and
retains P-side sigma_s and sn0; no xi-side sigma_FOG is added.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_DIR = PROJECT_ROOT / "codes" / "task43"
TASK432_DIR = PROJECT_ROOT / "codes" / "task432"
OUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "task432_model_repair" / "lightcone_png_velocileptors_gsm_longchain"

for path in (TASK43_DIR, TASK432_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from task43_run_lightcone_joint_baomask_v1 import ScaledGaussianMetric, load_rsd_specs  # noqa: E402
from task432_lightcone_png_velocileptors import (  # noqa: E402
    LightconePNGVelocileptors,
    RSD_P_PAYLOAD,
    RSD_X_SUMMARY,
    build_cache,
)


CURRENT = {}


def jsonable(value: Any) -> Any:
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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate_current(theta: np.ndarray) -> np.ndarray:
    kind = CURRENT["kind"]
    if kind == "p02":
        return np.asarray(CURRENT["p_spec"].evaluate(np.asarray(theta, dtype="f8")), dtype="f8")
    if kind == "xi02":
        values = CURRENT["x_model"].evaluate(fnl=float(theta[0]), b1=float(theta[1]), nint=CURRENT["nint"])
        mask = CURRENT["mask"]
        return np.concatenate((values[0][mask], values[2][mask]))
    if kind == "joint":
        p = np.asarray(CURRENT["p_spec"].evaluate(np.asarray(theta, dtype="f8")), dtype="f8")
        values = CURRENT["x_model"].evaluate(fnl=float(theta[0]), b1=float(theta[1]), nint=CURRENT["nint"])
        mask = CURRENT["mask"]
        x = np.concatenate((values[0][mask], values[2][mask]))
        return np.concatenate((p, x))
    raise RuntimeError(kind)


def log_probability(theta: np.ndarray) -> float:
    values = np.asarray(theta, dtype="f8")
    lower = CURRENT["lower"]
    upper = CURRENT["upper"]
    if np.any(values <= lower) or np.any(values >= upper):
        return -np.inf
    residual = CURRENT["data"] - evaluate_current(values)
    return -0.5 * CURRENT["metric"].chi2(residual)


def split_rhat(chain: np.ndarray) -> np.ndarray:
    nstep, nwalkers, ndim = chain.shape
    half = nstep // 2
    split = np.concatenate((chain[:half], chain[nstep - half :]), axis=1)
    n, m, _ = split.shape
    chain_mean = np.mean(split, axis=0)
    overall = np.mean(chain_mean, axis=0)
    between = n * np.var(chain_mean, axis=0, ddof=1)
    within = np.mean(np.var(split, axis=0, ddof=1), axis=0)
    return np.sqrt(((n - 1.0) * within + between) / (n * within))


def run_fit(name: str, spec: Any, *, x_model: Any, mask: np.ndarray, nint: int, nwalkers: int, nsteps: int, burnin: int, seed: int, workers: int, start: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    CURRENT.clear()
    CURRENT.update(
        {
            "kind": name,
            "data": np.asarray(spec.data, dtype="f8"),
            "metric": ScaledGaussianMetric(spec.covariance),
            "p_spec": spec if name != "xi02" else None,
            "x_model": x_model,
            "mask": mask,
            "nint": int(nint),
            "lower": np.asarray(lower, dtype="f8"),
            "upper": np.asarray(upper, dtype="f8"),
        }
    )
    rng = np.random.default_rng(int(seed))
    scale = np.asarray([4.0, 0.015] if name == "xi02" else [4.0, 0.015, 0.12, 0.025], dtype="f8")
    initial = np.asarray(start, dtype="f8")[None, :] + rng.normal(size=(int(nwalkers), len(start))) * scale[None, :]
    initial = np.maximum(initial, lower[None, :] + 1.0e-7)
    initial = np.minimum(initial, upper[None, :] - 1.0e-7)
    np.random.seed(int(seed))
    context = mp.get_context("fork")
    with context.Pool(processes=int(workers)) as pool:
        sampler = emcee.EnsembleSampler(int(nwalkers), len(start), log_probability, pool=pool)
        sampler.run_mcmc(initial, int(nsteps), progress=False, skip_initial_state_check=True)
    chain = np.asarray(sampler.get_chain(), dtype="f8")
    logp = np.asarray(sampler.get_log_prob(), dtype="f8")
    post = chain[int(burnin) :]
    flat = post.reshape(-1, post.shape[-1])
    flat_logp = logp[int(burnin) :].reshape(-1)
    try:
        tau = np.asarray(emcee.autocorr.integrated_time(post, quiet=True, tol=0), dtype="f8")
    except Exception:
        tau = np.full(flat.shape[1], np.nan)
    rhat = split_rhat(post)
    q = np.percentile(flat, [16.0, 50.0, 84.0], axis=0)
    names = ["fNL", "b1"] if name == "xi02" else ["fNL", "b1", "sigma_s", "sn0"]
    imax = int(np.argmax(flat_logp))
    half = post.shape[0] // 2
    half_shift = np.abs(np.median(post[:half].reshape(-1, post.shape[-1]), axis=0) - np.median(post[-half:].reshape(-1, post.shape[-1]), axis=0)) / (0.5 * (q[2] - q[0]))
    summary = {
        "name": name,
        "parameter_names": names,
        "posterior": {
            param: {
                "q16": float(q[0, i]), "q50": float(q[1, i]), "q84": float(q[2, i]), "sigma68": float(0.5 * (q[2, i] - q[0, i]))
            }
            for i, param in enumerate(names)
        },
        "map_chain": {param: float(flat[imax, i]) for i, param in enumerate(names)},
        "map_chain_log_probability": float(flat_logp[imax]),
        "mcmc": {"nwalkers": int(nwalkers), "nsteps": int(nsteps), "burnin": int(burnin), "workers": int(workers), "seed": int(seed)},
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
        "tau": {param: float(tau[i]) for i, param in enumerate(names)},
        "postburn_length_over_tau": {param: float(post.shape[0] / tau[i]) for i, param in enumerate(names)},
        "split_rhat": {param: float(rhat[i]) for i, param in enumerate(names)},
        "half_chain_shift_sigma": {param: float(half_shift[i]) for i, param in enumerate(names)},
        "gates": {
            "split_rhat_max_below_1p01": bool(np.nanmax(rhat) < 1.01),
            "postburn_length_min_above_50tau": bool(np.nanmin(post.shape[0] / tau) > 50.0),
            "half_chain_shift_max_below_0p1sigma": bool(np.nanmax(half_shift) < 0.1),
        },
    }
    outdir = OUT_ROOT / "fits" / name
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "samples.npz", chain_by_step=chain, log_probability_by_step=logp, parameter_names=np.asarray(names))
    atomic_json(outdir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--nint", type=int, default=400)
    args = parser.parse_args()
    specs, metadata, arrays = load_rsd_specs(smin=50.0, pk_kmax=0.08)
    by_name = {spec.name: spec for spec in specs}
    with np.load(RSD_P_PAYLOAD, allow_pickle=False) as payload:
        zeff = float(np.asarray(payload["zeff"]).item())
    cache = build_cache(zeff=zeff, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000")
    with np.load(cache, allow_pickle=False) as payload:
        k, pk, alpha = payload["k_eff"], payload["pk_dd"], payload["alpha"]
        f_growth = float(np.asarray(payload["f_growth"]).item())
    with np.load(RSD_X_SUMMARY, allow_pickle=False) as payload:
        s = np.asarray(payload["s"], dtype="f8")
    mask = (s >= 50.0) & (s < 350.0) & ~((s >= 80.0) & (s < 120.0))
    x_model = LightconePNGVelocileptors(k, pk, alpha, f_growth, s)
    lower_x, upper_x = np.asarray([-500.0, 0.5]), np.asarray([500.0, 5.0])
    lower_j, upper_j = np.asarray([-500.0, 0.5, 0.0, -1.0]), np.asarray([500.0, 5.0, 30.0, 1.0])
    results = {
        "xi02": run_fit("xi02", by_name["rsd_xi02"], x_model=x_model, mask=mask, nint=args.nint, nwalkers=args.nwalkers, nsteps=args.nsteps, burnin=args.burnin, seed=20260926, workers=args.workers, start=np.asarray([-5.6, 2.437]), lower=lower_x, upper=upper_x),
        "joint": run_fit("joint", by_name["rsd_joint_p02xi02"], x_model=x_model, mask=mask, nint=args.nint, nwalkers=args.nwalkers, nsteps=args.nsteps, burnin=args.burnin, seed=20260927, workers=args.workers, start=np.asarray([-7.7, 2.39, 2.17, 0.07]), lower=lower_j, upper=upper_j),
    }
    atomic_json(OUT_ROOT / "task432_lightcone_png_velocileptors_longchain_summary.json", {"metadata": metadata, "results": results, "model": "velocileptors GSM + minimal linear PNG response; EFT/bias higher operators fixed zero"})
    print(json.dumps({"status": "complete", "results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
