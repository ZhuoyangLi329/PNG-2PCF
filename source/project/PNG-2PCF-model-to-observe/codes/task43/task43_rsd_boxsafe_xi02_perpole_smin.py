#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-pole smin for the xi quadrupole increment (fix-3 analog on the xi side).

xi0 keeps s>=50; xi2 enters only above smin2 in {80, 120}.  A xi0-only control
at s>=50 with the same machinery gates the reproduction against the frozen
common-mask result.  3 params (fNL, b1, sigma_s), formal-GIC mean window,
ell02 RR-deconvolved covariance with per-pole row selection.
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
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

from task43_fit_rsd_lightcone_x25 import FastWindowRSDModel, load_inputs, load_window, set_affinity
from task43_joint_rsd_pkxi_fit import summarize_chain
from task43_rsd_common import OUTPUT_ROOT, PHASES, atomic_savez, atomic_write_json, sha256_file
from task43_rsd_model import FullDiscreteRSDModel, build_cache


BOXSAFE_ROOT = OUTPUT_ROOT / "lightcone_boxsafe_zobs0p4_0p8"
ELL2_DIR = BOXSAFE_ROOT / "ell2_increment"
SUMMARY_NPZ = ELL2_DIR / "task43_rsd_boxsafe_x25_mean_xi02_s30_350_ds10.npz"
ELL02_DECONV_NPZ = (
    BOXSAFE_ROOT / "covariance"
    / "task43_rsd_boxsafe_ph000_jaxpower_rrdeconv_ell02_rsdpoles024_win02468_mesh64_p1_b1cov2p50"
    "_sigmas7p50_nran100k_ndata50k_seed20260817_rrnran300k_rrseed430420_kbox_fkpP010000_s30_350_ds10.npz"
)
MEAN_WINDOW_NPZ = (
    BOXSAFE_ROOT / "formal_gic_windows"
    / "task43_rsd_boxsafe_formal_gic_x25_equal_phase_mean_nsub200000_seedbase430340.npz"
)
OUT_DIR = ELL2_DIR / "perpole_smin"
AUDIT_JSON = OUT_DIR / "audits" / "task43_rsd_boxsafe_xi02_perpole_smin_summary.json"

BOUNDS_LO = np.asarray([-500.0, 0.2, 0.0], dtype="f8")
BOUNDS_HI = np.asarray([500.0, 10.0, 30.0], dtype="f8")
INIT_SCALE = np.asarray([4.0, 0.015, 0.10], dtype="f8")
OPTIMIZER_STARTS = (
    np.asarray([0.0, 2.55, 8.0], dtype="f8"),
    np.asarray([-100.0, 2.3, 4.0], dtype="f8"),
    np.asarray([100.0, 2.8, 12.0], dtype="f8"),
)
VARIANTS = (
    ("ell0_s50_control", 0, None),
    ("ell0s50_ell2s80", 2, 80.0),
    ("ell0s50_ell2s120", 2, 120.0),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    if AUDIT_JSON.exists():
        raise FileExistsError(f"immutable per-pole summary exists: {AUDIT_JSON}")

    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))
    inputs = load_inputs(SUMMARY_NPZ, ELL02_DECONV_NPZ)
    mean_xi = np.mean(inputs["xi"], axis=0)
    mean_rr = np.mean(inputs["rr"], axis=0)
    zeff = float(np.mean(inputs["zeff"]))
    theory_cache = build_cache(zeff=zeff, boxsize=2000.0, kmax=5.0, ells=(0, 2), cosmology="abacus_c000")
    model = FastWindowRSDModel(
        FullDiscreteRSDModel(theory_cache, nmu=64),
        {"mean": load_window(MEAN_WINDOW_NPZ)},
        sigma_step=0.05,
    )
    if model.validate()["status"] != "pass":
        raise RuntimeError("xi surrogate failed")
    s = np.asarray(inputs["s"], dtype="f8")
    ns = s.size
    cov64 = np.asarray(inputs["covariance"], dtype="f8")
    mask0 = s >= 50.0

    def make_eval(ells_fit: tuple[int, ...], smin2: float | None):
        mask2 = s >= (50.0 if smin2 is None else float(smin2))

        def evaluate(theta: np.ndarray) -> np.ndarray:
            values = model.evaluate(np.asarray(theta, dtype="f8")[:3], model="formal_gic", window_key="mean")
            parts = [np.asarray(values[0])[mask0]]
            if len(ells_fit) == 2:
                parts.append(np.asarray(values[2])[mask2])
            return np.concatenate(parts)

        def ids() -> np.ndarray:
            blocks = [np.flatnonzero(mask0)]
            if len(ells_fit) == 2:
                blocks.append(ns + np.flatnonzero(mask2))
            return np.concatenate(blocks)

        return evaluate, ids, mask2

    results: dict[str, Any] = {}
    for index, (key, nells, smin2) in enumerate(VARIANTS):
        ells_fit = (0,) if nells == 0 else (0, 2)
        evaluate, ids_of, mask2 = make_eval(ells_fit, smin2)
        ids = ids_of()
        data_parts = [np.asarray(mean_xi[0])[mask0]]
        if nells == 2:
            data_parts.append(np.asarray(mean_xi[1])[mask2])
        data_v = np.concatenate(data_parts)
        cov_v = cov64[np.ix_(ids, ids)]
        precision = np.linalg.inv(cov_v)
        chol = np.linalg.cholesky(cov_v)

        def residual(theta: np.ndarray) -> np.ndarray:
            return np.linalg.solve(chol, data_v - evaluate(theta))

        solutions = [
            least_squares(residual, start, bounds=(BOUNDS_LO, BOUNDS_HI), max_nfev=3000,
                          xtol=1.0e-12, ftol=1.0e-12, gtol=1.0e-12)
            for start in OPTIMIZER_STARTS
        ]
        best = min(solutions, key=lambda r: float(r.fun @ r.fun))
        theta_map = np.asarray(best.x, dtype="f8")

        def log_probability(theta: np.ndarray) -> float:
            values = np.asarray(theta, dtype="f8")
            if np.any(values < BOUNDS_LO) or np.any(values > BOUNDS_HI):
                return -np.inf
            diff = data_v - evaluate(values)
            return -0.5 * float(diff @ precision @ diff)

        rng = np.random.default_rng(int(args.seed) + 100 * index)
        initial = theta_map[None, :] + rng.normal(size=(int(args.nwalkers), 3)) * INIT_SCALE[None, :]
        initial = np.clip(initial, BOUNDS_LO + 1.0e-7, BOUNDS_HI - 1.0e-7)
        sampler = emcee.EnsembleSampler(int(args.nwalkers), 3, log_probability)
        sampler.run_mcmc(initial, int(args.nsteps), progress=False)
        chain = np.asarray(sampler.get_chain(discard=int(args.burnin)), dtype="f8")
        logp = np.asarray(sampler.get_log_prob(discard=int(args.burnin)), dtype="f8")
        summary = summarize_chain(chain, logp, ("fNL", "b1", "sigma_s"))
        summary.update({"nwalkers": int(args.nwalkers), "nsteps": int(args.nsteps), "burnin": int(args.burnin)})
        if not all(summary["gates"].values()):
            raise SystemExit(f"MCMC gates failed for {key}: {summary['gates']}")

        prediction = evaluate(theta_map)
        pte: dict[str, Any] = {}
        for name, block_ids, seg in (
            ("ell0", np.flatnonzero(mask0), slice(0, int(np.count_nonzero(mask0)))),
            ("ell2", ns + np.flatnonzero(mask2), slice(int(np.count_nonzero(mask0)), None)),
        ):
            if name == "ell2" and nells == 0:
                continue
            block = cov64[np.ix_(block_ids, block_ids)] / len(PHASES)
            r = data_v[seg] - prediction[seg]
            c2 = float(r @ np.linalg.solve(block, r))
            dof = int(r.size - 3)
            pte[name] = {"chi2_Cmean": c2, "nbins": int(r.size), "dof": dof, "pte": float(chi2_distribution.sf(c2, dof))}

        results[key] = {
            "ells": list(ells_fit),
            "smin2": smin2,
            "posterior": summary["posterior"],
            "map_theta": [float(v) for v in theta_map],
            "gates": summary["gates"],
            "per_pole_mean_goodness": pte,
        }
        out_npz = OUT_DIR / "fits" / key / "samples.npz"
        if out_npz.exists():
            raise FileExistsError(f"immutable chain output exists: {out_npz}")
        atomic_savez(out_npz, chain_by_step=chain, log_probability_by_step=logp, prediction_map=prediction, data=data_v, ids=np.asarray(ids))
        print(json.dumps({"key": key, "fNL": summary["posterior"]["fNL"], "pte": pte}, sort_keys=True), flush=True)

    metrics = {
        "baseline_common_mask": {
            "ell0_s50_only": {"sigma_fNL": 36.54517778531864},
            "ell02_common_s80": {"sigma_fNL": 39.03515481323662, "ell2_pte": 3.324987352102397e-11},
        },
        "per_pole": {
            key: {
                "sigma_fNL": results[key]["posterior"]["fNL"]["sigma68"],
                "sigma_sigma_s": results[key]["posterior"]["sigma_s"]["sigma68"],
                "ell2_pte": results[key]["per_pole_mean_goodness"].get("ell2", {}).get("pte"),
                "ell0_pte": results[key]["per_pole_mean_goodness"]["ell0"]["pte"],
            }
            for key in results
        },
    }
    audit = {
        "task": "task43_rsd_boxsafe_xi02_perpole_smin",
        "status": "complete",
        "scope": (
            "per-pole smin: xi0 fixed at s>=50, xi2 entering above smin2 in {80,120}; "
            "formal-GIC mean window, ell02 RR-deconvolved covariance, x25 mean"
        ),
        "caveats": [
            "attribution diagnostic against the analytic Gaussian covariance; not a science claim",
            "priors follow the audited xi side (fNL +-500, b1 [0.2,10], sigma_s [0,30])",
            "fix-2 (EZmock empirical covariance) remains unexamined",
        ],
        "mcmc": {"nwalkers": int(args.nwalkers), "nsteps": int(args.nsteps), "burnin": int(args.burnin), "seed": int(args.seed)},
        "inputs": {
            "summary": {"path": str(SUMMARY_NPZ), "sha256": sha256_file(SUMMARY_NPZ)},
            "covariance": {"path": str(ELL02_DECONV_NPZ), "sha256": sha256_file(ELL02_DECONV_NPZ)},
            "mean_window": {"path": str(MEAN_WINDOW_NPZ), "sha256": sha256_file(MEAN_WINDOW_NPZ)},
        },
        "results": results,
        "metrics": metrics,
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    atomic_write_json(AUDIT_JSON, audit)
    print(json.dumps({"status": "complete", "metrics": metrics, "output": str(AUDIT_JSON)}, sort_keys=True))


if __name__ == "__main__":
    main()
