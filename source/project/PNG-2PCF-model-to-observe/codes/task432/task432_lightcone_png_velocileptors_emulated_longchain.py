#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Emulated, converged lightcone chains for the hybrid PNG GSM model.

The direct velocileptors model is precomputed on an (fNL,b1) grid before the
MCMC.  The chain itself uses one worker, avoiding the JAX+fork deadlock seen in
the first long-chain attempt on a login node.  The emulator is written to the
same output tree and its interpolation range is recorded in the audit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import numpy as np
from scipy.interpolate import RegularGridInterpolator

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

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


CURRENT: dict[str, Any] = {}


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


def split_rhat(chain: np.ndarray) -> np.ndarray:
    nstep, nwalkers, ndim = chain.shape
    half = nstep // 2
    split = np.concatenate((chain[:half], chain[nstep - half :]), axis=1)
    n, m, _ = split.shape
    means = np.mean(split, axis=0)
    overall = np.mean(means, axis=0)
    between = n * np.var(means, axis=0, ddof=1)
    within = np.mean(np.var(split, axis=0, ddof=1), axis=0)
    return np.sqrt(((n - 1.0) * within + between) / (n * within))


class XiEmulator:
    """Regular-grid interpolator for hybrid velocileptors xi02."""

    def __init__(self, f_grid: np.ndarray, b_grid: np.ndarray, values: np.ndarray) -> None:
        self.f_grid = np.asarray(f_grid, dtype="f8")
        self.b_grid = np.asarray(b_grid, dtype="f8")
        self.values = np.asarray(values, dtype="f8")
        self.interpolator = RegularGridInterpolator(
            (self.f_grid, self.b_grid), self.values, bounds_error=True, method="linear"
        )

    @classmethod
    def build_or_load(
        cls,
        path: Path,
        model: LightconePNGVelocileptors,
        *,
        f_grid: np.ndarray,
        b_grid: np.ndarray,
        mask: np.ndarray,
        nint: int,
    ) -> "XiEmulator":
        if path.is_file():
            with np.load(path, allow_pickle=False) as payload:
                old_f = np.asarray(payload["f_grid"], dtype="f8")
                old_b = np.asarray(payload["b_grid"], dtype="f8")
                values = np.asarray(payload["values"], dtype="f8")
            if np.array_equal(old_f, f_grid) and np.array_equal(old_b, b_grid) and values.shape == (f_grid.size, b_grid.size, int(np.count_nonzero(mask)) * 2):
                return cls(old_f, old_b, values)

        values = np.empty((f_grid.size, b_grid.size, int(np.count_nonzero(mask)) * 2), dtype="f8")
        for i, fnl in enumerate(f_grid):
            for j, b1 in enumerate(b_grid):
                prediction = model.evaluate(fnl=float(fnl), b1=float(b1), nint=int(nint))
                values[i, j] = np.concatenate((prediction[0][mask], prediction[2][mask]))
            print(json.dumps({"emulator_fnl_index": int(i + 1), "emulator_fnl_total": int(f_grid.size)}), flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, f_grid=f_grid, b_grid=b_grid, values=values)
        return cls(f_grid, b_grid, values)

    def evaluate(self, fnl: float, b1: float) -> np.ndarray:
        return np.asarray(self.interpolator(np.asarray([[float(fnl), float(b1)]], dtype="f8"))[0], dtype="f8")


def evaluate_current(theta: np.ndarray) -> np.ndarray:
    values = np.asarray(theta, dtype="f8")
    if CURRENT["kind"] == "xi02":
        return CURRENT["x_emulator"].evaluate(values[0], values[1])
    if CURRENT["kind"] == "joint":
        p = np.asarray(CURRENT["p_spec"].evaluate(values), dtype="f8")
        x = CURRENT["x_emulator"].evaluate(values[0], values[1])
        return np.concatenate((p, x))
    raise RuntimeError(CURRENT["kind"])


def log_probability(theta: np.ndarray) -> float:
    values = np.asarray(theta, dtype="f8")
    if np.any(values <= CURRENT["lower"]) or np.any(values >= CURRENT["upper"]):
        return -np.inf
    residual = CURRENT["data"] - evaluate_current(values)
    return -0.5 * CURRENT["metric"].chi2(residual)


def run_chain(name: str, spec: Any, *, emulator: XiEmulator, p_spec: Any, nwalkers: int, nsteps: int, burnin: int, seed: int, start: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    CURRENT.clear()
    CURRENT.update(
        {
            "kind": name,
            "data": np.asarray(spec.data, dtype="f8"),
            "metric": ScaledGaussianMetric(spec.covariance),
            "x_emulator": emulator,
            "p_spec": p_spec,
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
    sampler = emcee.EnsembleSampler(int(nwalkers), len(start), log_probability)
    sampler.run_mcmc(initial, int(nsteps), progress=False, skip_initial_state_check=True)
    chain = np.asarray(sampler.get_chain(), dtype="f8")
    logp = np.asarray(sampler.get_log_prob(), dtype="f8")
    post = chain[int(burnin) :]
    flat = post.reshape(-1, post.shape[-1])
    flat_logp = logp[int(burnin) :].reshape(-1)
    names = ["fNL", "b1"] if name == "xi02" else ["fNL", "b1", "sigma_s", "sn0"]
    q = np.percentile(flat, [16.0, 50.0, 84.0], axis=0)
    tau = np.asarray(emcee.autocorr.integrated_time(post, quiet=True, tol=0), dtype="f8")
    rhat = split_rhat(post)
    half = post.shape[0] // 2
    half_shift = np.abs(np.median(post[:half].reshape(-1, post.shape[-1]), axis=0) - np.median(post[-half:].reshape(-1, post.shape[-1]), axis=0)) / (0.5 * (q[2] - q[0]))
    imax = int(np.argmax(flat_logp))
    summary = {
        "parameter_names": names,
        "posterior": {name_: {"q16": float(q[0, i]), "q50": float(q[1, i]), "q84": float(q[2, i]), "sigma68": float(0.5 * (q[2, i] - q[0, i]))} for i, name_ in enumerate(names)},
        "map_chain": {name_: float(flat[imax, i]) for i, name_ in enumerate(names)},
        "map_chain_log_probability": float(flat_logp[imax]),
        "mcmc": {"nwalkers": int(nwalkers), "nsteps": int(nsteps), "burnin": int(burnin), "seed": int(seed)},
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
        "tau": {name_: float(tau[i]) for i, name_ in enumerate(names)},
        "postburn_length_over_tau": {name_: float(post.shape[0] / tau[i]) for i, name_ in enumerate(names)},
        "split_rhat": {name_: float(rhat[i]) for i, name_ in enumerate(names)},
        "half_chain_shift_sigma": {name_: float(half_shift[i]) for i, name_ in enumerate(names)},
        "gates": {"split_rhat_max_below_1p01": bool(np.max(rhat) < 1.01), "postburn_length_min_above_50tau": bool(np.min(post.shape[0] / tau) > 50.0), "half_chain_shift_max_below_0p1sigma": bool(np.max(half_shift) < 0.1)},
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
    parser.add_argument("--emulator-nint", type=int, default=400)
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
    model = LightconePNGVelocileptors(k, pk, alpha, f_growth, s)
    f_grid = np.linspace(-500.0, 500.0, 41, dtype="f8")
    b_grid = np.linspace(0.5, 5.0, 46, dtype="f8")
    emulator = XiEmulator.build_or_load(OUT_ROOT / "lightcone_xi_emulator.npz", model, f_grid=f_grid, b_grid=b_grid, mask=mask, nint=int(args.emulator_nint))
    results = {
        "xi02": run_chain("xi02", by_name["rsd_xi02"], emulator=emulator, p_spec=None, nwalkers=args.nwalkers, nsteps=args.nsteps, burnin=args.burnin, seed=20260926, start=np.asarray([-5.6, 2.437]), lower=np.asarray([-500.0, 0.5]), upper=np.asarray([500.0, 5.0])),
        "joint": run_chain("joint", by_name["rsd_joint_p02xi02"], emulator=emulator, p_spec=by_name["rsd_p02"], nwalkers=args.nwalkers, nsteps=args.nsteps, burnin=args.burnin, seed=20260927, start=np.asarray([-7.7, 2.39, 2.17, 0.07]), lower=np.asarray([-500.0, 0.5, 0.0, -1.0]), upper=np.asarray([500.0, 5.0, 30.0, 1.0])),
    }
    atomic_json(OUT_ROOT / "task432_lightcone_png_velocileptors_emulated_longchain_summary.json", {"metadata": metadata, "results": results, "emulator": {"f_grid": f_grid.tolist(), "b_grid": b_grid.tolist(), "nint": int(args.emulator_nint)}})
    print(json.dumps({"status": "complete", "results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
