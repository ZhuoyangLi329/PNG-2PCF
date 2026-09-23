#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fix-3 diagnostic: per-pole k_min cut for P2 (P0 keeps all 15 bins).

For each kmin2 in {0.015, 0.02, 0.03} h/Mpc, refit (fNL, b1, sigma_s, sn0)
with P0 on all 15 DESI bins and P2 only on bins with k_obs >= kmin2, using
the frozen measurement, window and 450x450 covariance blocks.  Reports the
retained-bin per-pole PTE and the per-bin residual/sigma of P2 to locate the
chi2 concentration.  This is an attribution diagnostic: improvement, not a
PTE > 0.05 target, is the outcome.
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

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import set_affinity
from task43_joint_rsd_pkxi_fit import (
    BOUNDS_HI,
    BOUNDS_LO,
    JOINT_PARAMS,
    OPTIMIZER_STARTS,
    build_precision,
    fisher_covariance,
    run_chain,
)
from task43_rsd_common import OUTPUT_ROOT, PHASES, atomic_savez, atomic_write_json, sha256_file
from task43_rsd_boxsafe_p02_increment import (
    COV_NPZ,
    MEASURE_DIR,
    PAYLOAD_NPZ,
    WindowConvolvedP02Model,
)

BOXSAFE_ROOT = OUTPUT_ROOT / "lightcone_boxsafe_zobs0p4_0p8"
OUT_DIR = BOXSAFE_ROOT / "p02_increment" / "kminscan"
AUDIT_JSON = OUT_DIR / "audits" / "task43_rsd_boxsafe_p02_kminscan_summary.json"
KMIN2_VALUES = (0.015, 0.02, 0.03)
BASELINE_FNL_SIGMA = 35.33414239117636  # p0_control, frozen from the increment audit
BASELINE_P02_SIGMA = 29.185197596964386  # full-bin p02, frozen from the increment audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    if AUDIT_JSON.exists():
        raise FileExistsError(f"immutable kminscan summary exists: {AUDIT_JSON}")

    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))
    from task43_fit_rsd_lightcone_pk0_vs_xi0_smin50 import load_pk

    payload = load_pk(PAYLOAD_NPZ)
    fit_indices = np.asarray(payload["fit_bin_indices"], dtype="i8")
    k_obs = np.asarray(payload["k_obs"], dtype="f8")
    zeff = float(np.asarray(payload["zeff"]).item())

    stacks, window_rows, theory = [], None, None
    for phase in PHASES:
        path = MEASURE_DIR / (
            f"task43_rsd_lightcone_p02_{phase}_mesh256_"
            + "kmax0.300_dk0.002".replace(".", "p") + ".npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as d:
            pk0 = np.asarray(d["pk0"], dtype="f8")[fit_indices]
            pk2 = np.asarray(d["pk2"], dtype="f8")[fit_indices]
            if phase == "ph000":
                window_rows = np.asarray(d["window_matrix"], dtype="f8")
                theory = (np.asarray(d["theory_k"], dtype="f8"), np.asarray(d["theory_ell"], dtype="i8"))
        stacks.append(np.concatenate([pk0, pk2]))
    mean_full = np.mean(np.stack(stacks), axis=0)  # (30,)
    window = window_rows[np.concatenate([fit_indices, 150 + fit_indices]), :]
    with np.load(COV_NPZ, allow_pickle=False) as d:
        cov_full = np.asarray(d["covariance_full"], dtype="f8")
    ids_full = np.concatenate([fit_indices, 150 + fit_indices])
    cov30 = cov_full[np.ix_(ids_full, ids_full)]

    model = WindowConvolvedP02Model(window, theory[0], theory[1], zeff=zeff)
    validation = model.validate()
    if validation["status"] != "pass":
        raise RuntimeError(f"P02 surrogate failed: {validation}")

    results: dict[str, Any] = {}
    for index, kmin2 in enumerate(KMIN2_VALUES):
        keep = np.flatnonzero(k_obs >= float(kmin2) - 1.0e-12)
        nkeep = int(keep.size)
        rows = np.concatenate([np.arange(15), 15 + keep])
        data_v = mean_full[rows]
        cov_v = cov30[np.ix_(rows, rows)]
        precision, precision_meta = build_precision(cov_v)
        chol = np.linalg.cholesky(cov_v)

        def evaluate(theta: np.ndarray) -> np.ndarray:
            return model.evaluate(theta)[rows]

        def residual(theta: np.ndarray) -> np.ndarray:
            return np.linalg.solve(chol, data_v - evaluate(theta))

        solutions = [
            least_squares(residual, start, bounds=(BOUNDS_LO, BOUNDS_HI), max_nfev=3000,
                          xtol=1.0e-12, ftol=1.0e-12, gtol=1.0e-12)
            for start in OPTIMIZER_STARTS
        ]
        best = min(solutions, key=lambda r: float(r.fun @ r.fun))
        theta_map = np.asarray(best.x, dtype="f8")
        summary, chain, logp = run_chain(
            evaluate, data_v, precision, theta_map, 4,
            nwalkers=int(args.nwalkers), nsteps=int(args.nsteps),
            burnin=int(args.burnin), seed=int(args.seed) + 100 * index,
        )
        if not all(summary["gates"].values()):
            raise SystemExit(f"MCMC gates failed for kmin2={kmin2}: {summary['gates']}")
        fisher = fisher_covariance(evaluate, precision, theta_map)
        prediction = evaluate(theta_map)
        block_p0 = cov30[:15, :15] / len(PHASES)
        block_p2 = cov30[np.ix_(15 + keep, 15 + keep)] / len(PHASES)
        r0 = data_v[:15] - prediction[:15]
        r2 = data_v[15:] - prediction[15:]
        chi2_p0 = float(r0 @ np.linalg.solve(block_p0, r0))
        chi2_p2 = float(r2 @ np.linalg.solve(block_p2, r2))
        sig_p2 = np.sqrt(np.diag(block_p2))
        sig_p0 = np.sqrt(np.diag(block_p0))
        res_p2 = r2 / sig_p2
        res_p0 = r0 / sig_p0
        chi2_joint = float(best.fun @ best.fun)
        key = f"kmin2_{kmin2:.3f}".replace(".", "p")
        results[key] = {
            "kmin2_h_mpc": float(kmin2),
            "p2_kept_bins": int(nkeep),
            "p2_kept_k": [float(v) for v in k_obs[keep]],
            "posterior": summary["posterior"],
            "gates": summary["gates"],
            "map_theta": {name: float(v) for name, v in zip(JOINT_PARAMS, theta_map)},
            "fisher_sigma": {
                name: float(np.sqrt(max(np.diag(fisher)[i], 0.0))) for i, name in enumerate(JOINT_PARAMS)
            },
            "chi2": {
                "joint_single": chi2_joint,
                "joint_dof": int(data_v.size - 4),
                "joint_pte": float(chi2_distribution.sf(chi2_joint, data_v.size - 4)),
                "ell0_Cmean": chi2_p0,
                "ell0_dof": 11,
                "ell0_pte": float(chi2_distribution.sf(chi2_p0, 11)),
                "ell2_Cmean_kept": chi2_p2,
                "ell2_dof": int(nkeep - 4),
                "ell2_pte": float(chi2_distribution.sf(chi2_p2, nkeep - 4)),
            },
            "per_bin_residual_sigma": {
                "ell2_kept_k": [float(v) for v in k_obs[keep]],
                "ell2_residual_sigma": [float(v) for v in res_p2],
                "ell0_residual_sigma": [float(v) for v in res_p0],
            },
        }
        out_npz = OUT_DIR / "fits" / key / "samples.npz"
        if out_npz.exists():
            raise FileExistsError(f"immutable chain output exists: {out_npz}")
        atomic_savez(out_npz, chain_by_step=chain, log_probability_by_step=logp, prediction_map=prediction, data=data_v, rows=np.asarray(rows))
        print(json.dumps({"kmin2": kmin2, "fNL": summary["posterior"]["fNL"], "ell2_pte": results[key]["chi2"]["ell2_pte"], "ell0_pte": results[key]["chi2"]["ell0_pte"]}, sort_keys=True), flush=True)

    metrics = {
        "baseline_full_bin": {
            "sigma_fNL_p0_control": BASELINE_FNL_SIGMA,
            "sigma_fNL_p02_full": BASELINE_P02_SIGMA,
            "ell2_pte_full": 2.493582306388858e-06,
        },
        "trend": {
            key: {
                "sigma_fNL": results[key]["posterior"]["fNL"]["sigma68"],
                "sigma_sigma_s": results[key]["posterior"]["sigma_s"]["sigma68"],
                "ell2_pte": results[key]["chi2"]["ell2_pte"],
                "ell0_pte": results[key]["chi2"]["ell0_pte"],
                "p2_kept": results[key]["p2_kept_bins"],
            }
            for key in results
        },
    }
    audit = {
        "task": "task43_rsd_boxsafe_p02_kminscan",
        "status": "complete",
        "scope": (
            "fix-3 diagnostic: P0 keeps all 15 DESI bins, P2 restricted to k>=kmin2 "
            "in {0.015, 0.02, 0.03} h/Mpc; frozen measurement/window/450x450 covariance"
        ),
        "caveats": [
            "attribution diagnostic against the analytic Gaussian covariance; not a science claim",
            "per-pole PTE dof subtracts 4 shared parameters (nbins-4)",
            "fix-2 (EZmock empirical covariance) remains unexamined",
        ],
        "mcmc": {"nwalkers": int(args.nwalkers), "nsteps": int(args.nsteps), "burnin": int(args.burnin), "seed": int(args.seed)},
        "surrogate_validation": validation["status"],
        "inputs": {
            "covariance": {"path": str(COV_NPZ), "sha256": sha256_file(COV_NPZ)},
            "payload": {"path": str(PAYLOAD_NPZ), "sha256": sha256_file(PAYLOAD_NPZ)},
            "measure_dir": str(MEASURE_DIR),
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
