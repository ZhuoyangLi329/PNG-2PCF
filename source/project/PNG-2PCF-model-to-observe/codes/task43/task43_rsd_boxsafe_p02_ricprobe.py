#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fix-1 probe (B1a): real-space radial-RIC operator applied to the RSD P rows.

The audited real-space single-term RIC operator (ph000, factorized kernel,
geometry = Task43 realspace 0.6<z<0.8 selection) has a theory grid bitwise
identical to the P02 window, and its pk_ric matrix is a theory-agnostic
geometry object acting on the ell=0 theory columns.  This probe pads it with
zero P2 rows and refits with W_eff = W_geom - W_RIC.

Scope and caveats (pre-registered):
* the operator geometry is the REAL-SPACE selection; the boxsafe 0.4<zobs<0.8
  selection differs, so the RIC amplitude is mis-geometry by O(selection
  difference).  This pass only tests whether an O(RIC) correction can move
  the low-k residuals at all, and validates the machinery for B1b;
* P2 observed rows carry no RIC here: the single auto-term with pair-(s,mu)
  machinery needed for P2 rows is out of scope;
* guard: the P0-control PTE must stay >= 0.01 (baseline 0.496); a collapse
  would indicate wrong-sign or double counting with sn0.
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
from task43_rsd_boxsafe_p02_increment import (
    COV_NPZ,
    MEASURE_DIR,
    PAYLOAD_NPZ,
    WindowConvolvedP02Model,
)
from task43_rsd_common import OUTPUT_ROOT, PHASES, atomic_savez, atomic_write_json, sha256_file


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
RIC_OPERATOR_NPZ = (
    PROJECT_ROOT / "plots/outputs/task43_outputs/ric_singleterm/operators"
    / "task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712_L2000.npz"
)
BOXSAFE_ROOT = OUTPUT_ROOT / "lightcone_boxsafe_zobs0p4_0p8"
OUT_DIR = BOXSAFE_ROOT / "p02_increment" / "ricprobe"
AUDIT_JSON = OUT_DIR / "audits" / "task43_rsd_boxsafe_p02_ricprobe_summary.json"
P0_CONTROL_PTE_GUARD = 0.01


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--ric-operator", type=Path, default=RIC_OPERATOR_NPZ)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--p2-kmin2", type=float, default=None, help="Optional per-pole P2 k_min cut for a combined (fix1+fix3) variant")
    args = parser.parse_args()
    audit_json = Path(args.out_dir) / "audits" / Path(AUDIT_JSON).name
    if audit_json.exists():
        raise FileExistsError(f"immutable ricprobe summary exists: {AUDIT_JSON}")

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
        with np.load(path, allow_pickle=False) as d:
            stacks.append(np.concatenate([np.asarray(d["pk0"], dtype="f8")[fit_indices], np.asarray(d["pk2"], dtype="f8")[fit_indices]]))
            if phase == "ph000":
                window_rows = np.asarray(d["window_matrix"], dtype="f8")
                theory = (np.asarray(d["theory_k"], dtype="f8"), np.asarray(d["theory_ell"], dtype="i8"))
    mean_full = np.mean(np.stack(stacks), axis=0)
    window = window_rows[np.concatenate([fit_indices, 150 + fit_indices]), :]

    with np.load(Path(args.ric_operator), allow_pickle=False) as d:
        ric15 = np.asarray(d["pk_ric_matrix"], dtype="f8")
        ric_k = np.asarray(d["pk_theory_k"], dtype="f8")
        ric_ell = np.asarray(d["pk_theory_ell"], dtype="i8")
        ric_meta = json.loads(str(np.asarray(d["meta_json"]).item())) if "meta_json" in d.files else {}
    if not np.array_equal(ric_k, theory[0]) or not np.array_equal(ric_ell, theory[1]):
        raise RuntimeError("RIC operator theory grid differs from the P02 window grid")
    ric30 = np.zeros_like(window)
    ric30[:15, :] = ric15
    window_eff = window - ric30
    ric_scale = float(np.linalg.norm(ric15) / np.linalg.norm(window[:15, :]))

    model = WindowConvolvedP02Model(window_eff, theory[0], theory[1], zeff=zeff)
    validation = model.validate()
    if validation["status"] != "pass":
        raise RuntimeError(f"P02(RIC) surrogate failed: {validation}")
    with np.load(COV_NPZ, allow_pickle=False) as d:
        cov_full = np.asarray(d["covariance_full"], dtype="f8")
    ids = np.concatenate([fit_indices, 150 + fit_indices])
    cov30 = cov_full[np.ix_(ids, ids)]

    if args.p2_kmin2 is not None:
        keep = np.flatnonzero(k_obs >= float(args.p2_kmin2) - 1.0e-12)
        variant_rows = (("p0_control_ric", np.arange(15)), ("p02_ric_kmin2", np.concatenate([np.arange(15), 15 + keep])))
    else:
        variant_rows = (("p0_control_ric", np.arange(15)), ("p02_ric", np.arange(30)))
    results: dict[str, Any] = {}
    for index, (variant, rows_fit) in enumerate(variant_rows):
        sl = rows_fit
        data_v = mean_full[sl]
        cov_v = cov30[np.ix_(sl, sl)]
        precision, precision_meta = build_precision(cov_v)
        chol = np.linalg.cholesky(cov_v)

        def evaluate(theta: np.ndarray) -> np.ndarray:
            return model.evaluate(theta)[sl]

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
            raise SystemExit(f"MCMC gates failed for {variant}: {summary['gates']}")
        prediction = evaluate(theta_map)
        pte: dict[str, Any] = {}
        for pole, name, off in ((0, "ell0", 0), (2, "ell2", 15)):
            if pole == 2 and variant == "p0_control_ric":
                continue
            n_p2 = 15 if args.p2_kmin2 is None else int(keep.size)
            n_p0 = 15
            off2 = off if off == 0 else n_p0
            size2 = 15 if off == 0 else n_p2
            block = cov30[np.ix_(sl, sl)][np.ix_(
                np.arange(off2, off2 + size2), np.arange(off2, off2 + size2))] / len(PHASES)
            r = data_v[off2:off2 + size2] - prediction[off2:off2 + size2]
            c2 = float(r @ np.linalg.solve(block, r))
            dof2 = size2 - 4
            pte[name] = {"chi2_Cmean": c2, "dof": dof2, "pte": float(chi2_distribution.sf(c2, dof2))}
        results[variant] = {
            "posterior": summary["posterior"],
            "gates": summary["gates"],
            "map_theta": {name: float(v) for name, v in zip(JOINT_PARAMS, theta_map)},
            "per_pole_mean_goodness": pte,
        }
        out_npz = Path(args.out_dir) / "fits" / variant / "samples.npz"
        if out_npz.exists():
            raise FileExistsError(f"immutable chain output exists: {out_npz}")
        atomic_savez(out_npz, chain_by_step=chain, log_probability_by_step=logp, prediction_map=prediction, data=data_v)
        print(json.dumps({"variant": variant, "fNL": summary["posterior"]["fNL"], "pte": pte}, sort_keys=True), flush=True)

    p0_guard = results["p0_control_ric"]["per_pole_mean_goodness"]["ell0"]["pte"]
    guard_pass = bool(p0_guard >= P0_CONTROL_PTE_GUARD)
    audit = {
        "task": "task43_rsd_boxsafe_p02_ricprobe",
        "status": "complete" if guard_pass else "p0_guard_failed",
        "scope": (
            "fix-1 probe: audited real-space single-term RIC operator (theory grid bitwise identical) "
            "padded with zero P2 rows; W_eff = W_geom - W_RIC on the (30, 981) window"
        ),
        "caveats": [
            "operator geometry is the REAL-SPACE 0.6<z<0.8 selection; boxsafe amplitude mis-geometry",
            "P2 observed rows carry no RIC (pair-(s,mu) machinery out of scope)",
            "attribution diagnostic against the analytic Gaussian covariance; not a science claim",
            "fix-2 (EZmock empirical covariance) remains unexamined",
        ],
        "ric_operator": {
            "path": str(RIC_OPERATOR_NPZ),
            "sha256": sha256_file(RIC_OPERATOR_NPZ),
            "norm_ratio_vs_geom_p0_rows": ric_scale,
            "meta_keys": sorted(ric_meta.keys())[:10] if ric_meta else [],
        },
        "guards": {"p0_control_pte": p0_guard, "threshold": P0_CONTROL_PTE_GUARD, "pass": guard_pass},
        "baseline": {"p0_control_pte": 0.4963, "p02_ell2_pte": 2.493582306388858e-06},
        "results": results,
        "mcmc": {"nwalkers": int(args.nwalkers), "nsteps": int(args.nsteps), "burnin": int(args.burnin), "seed": int(args.seed)},
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    audit["ric_operator"]["path"] = str(args.ric_operator)
    atomic_write_json(audit_json, audit)
    print(json.dumps({"status": audit["status"], "guards": audit["guards"], "output": str(audit_json), "results": {k: v["per_pole_mean_goodness"] for k, v in results.items()}}, sort_keys=True))
    if not guard_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
