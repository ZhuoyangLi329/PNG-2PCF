#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Four-way joint fit: P0+P2 (k>=0.015) + xi0 (s>=50) + xi2 (s>=80).

Adopted standard configuration (user decision 2026-09-06).  All covariance
blocks and the P-xi cross terms come from the SAME frozen 4497-mode jaxpower
run: C_pp and C_px from covariance_pk, C_xx from the audited ell02
RR-deconvolved matrix.  Five variants: p02_marginal, xi02_marginal, joint,
joint_naive (cross zeroed), joint_half (cross halved).  Free
(fNL, b1, sigma_s, sn0); sigma_s shared; sn0 on the P ell=0 theory only.
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

import numpy as np
from scipy.optimize import least_squares

from task43_fit_rsd_lightcone_x25 import FastWindowRSDModel, load_window
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
from task43_rsd_model import FullDiscreteRSDModel, build_cache


BOXSAFE_ROOT = OUTPUT_ROOT / "lightcone_boxsafe_zobs0p4_0p8"
RAW_NPZ = (
    BOXSAFE_ROOT / "covariance"
    / "task43_rsd_boxsafe_ph000_jaxpower_raw_xi0_rsdpoles024_win02468_mesh64_p1_b1cov2p50"
    "_sigmas7p50_nran100k_ndata50k_seed20260817_kbox_fkpP010000_s30_350_ds10.npz"
)
ELL02_DECONV_NPZ = (
    BOXSAFE_ROOT / "covariance"
    / "task43_rsd_boxsafe_ph000_jaxpower_rrdeconv_ell02_rsdpoles024_win02468_mesh64_p1_b1cov2p50"
    "_sigmas7p50_nran100k_ndata50k_seed20260817_rrnran300k_rrseed430420_kbox_fkpP010000_s30_350_ds10.npz"
)
ELL2_SUMMARY = BOXSAFE_ROOT / "ell2_increment" / "task43_rsd_boxsafe_x25_mean_xi02_s30_350_ds10.npz"
MEAN_WINDOW_NPZ = (
    BOXSAFE_ROOT / "formal_gic_windows"
    / "task43_rsd_boxsafe_formal_gic_x25_equal_phase_mean_nsub200000_seedbase430340.npz"
)
OUT_DIR = BOXSAFE_ROOT / "joint_p02xi02"
AUDIT_JSON = OUT_DIR / "audits" / "task43_rsd_joint_p02xi02_fit_summary.json"
P2_KMIN = 0.015
XI0_SMIN = 50.0
XI2_SMIN = 80.0
GATE_INTERNAL_RELFRO = 1.0e-10
GATE_DECONV_RELFRO = 1.0e-8
GATE_PP_SIGMA = (0.95, 1.05)
GATE_PP_RELFRO = 0.10
PSD_ABS_FLOOR = 1.0e-30


def rel_fro(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1.0e-300))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out_root = Path(args.out_dir)
    audit_json = out_root / "audits" / Path(AUDIT_JSON).name
    nsteps = 400 if args.smoke else int(args.nsteps)
    burnin = 100 if args.smoke else int(args.burnin)
    if audit_json.exists():
        raise FileExistsError(f"immutable joint summary exists: {audit_json}")

    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))

    # ---- frozen inputs -------------------------------------------------
    from task43_fit_rsd_lightcone_pk0_vs_xi0_smin50 import load_pk

    payload = load_pk(PAYLOAD_NPZ)
    k_obs = np.asarray(payload["k_obs"], dtype="f8")
    fit_indices = np.asarray(payload["fit_bin_indices"], dtype="i8")
    zeff = float(np.asarray(payload["zeff"]).item())
    keep_p2 = np.flatnonzero(k_obs >= P2_KMIN - 1.0e-12)
    with np.load(RAW_NPZ, allow_pickle=False) as d:
        cov_pk = np.asarray(d["covariance_pk"], dtype="f8")
        proj = np.asarray(d["projection_matrix"], dtype="f8")
        cov_xi_multi = np.asarray(d["covariance_xi_multipoles"], dtype="f8")
        raw_meta = json.loads(str(np.asarray(d["meta_json"]).item()))
    with np.load(ELL02_DECONV_NPZ, allow_pickle=False) as d:
        deconv64 = np.asarray(d["covariance_single_realization"], dtype="f8")
        rinv64 = np.asarray(d["rr_window_inverse"], dtype="f8")
        s_full = np.asarray(d["s"], dtype="f8")
    with np.load(ELL2_SUMMARY, allow_pickle=False) as d:
        mean_xi = np.asarray(d["xi_multipoles_mean"], dtype="f8")
    ns = s_full.size

    k_grid = raw_meta["k_grid"]
    nper = int(k_grid["nk"])
    edges = float(k_grid["kmin"]) + float(k_grid["dk"]) * np.arange(nper + 1, dtype="f8")
    centers_fine = 0.5 * (edges[:-1] + edges[1:])
    k_edges = np.asarray(payload["k_edges"], dtype="f8")
    selected: list[int] = []
    for lo, hi in k_edges:
        inside = np.flatnonzero((centers_fine >= float(lo) - 1.0e-12) & (centers_fine < float(hi) - 1.0e-12))
        if inside.size != 1:
            raise RuntimeError(f"payload bin [{lo},{hi}] contains {inside.size} fine bins")
        selected.append(int(inside[0]))
    sel15 = np.asarray(selected, dtype="i8")

    # ---- covariance assembly from one family ---------------------------
    internal = rel_fro(proj @ cov_pk @ proj.T, cov_xi_multi)
    if internal > GATE_INTERNAL_RELFRO:
        raise RuntimeError(f"internal consistency gate failed: {internal:.3e}")
    raw_64 = cov_xi_multi[: 2 * ns, : 2 * ns]
    deconv_check = rel_fro(rinv64 @ raw_64 @ rinv64.T, deconv64)
    if deconv_check > GATE_DECONV_RELFRO:
        raise RuntimeError(f"ell02 deconvolution reproduction failed: {deconv_check:.3e}")

    ids_p = np.concatenate([sel15, nper + sel15[keep_p2]])  # ell0 rows then ell2 rows
    c_pp = cov_pk[np.ix_(ids_p, ids_p)]
    proj64 = proj[: 2 * ns, :]
    c_px_full = rinv64 @ (proj64 @ cov_pk[:, ids_p])
    mask0 = s_full >= XI0_SMIN
    mask2 = s_full >= XI2_SMIN
    row_ids = np.concatenate([np.flatnonzero(mask0), ns + np.flatnonzero(mask2)])
    c_px = c_px_full[row_ids, :]
    c_xx = deconv64[np.ix_(row_ids, row_ids)]

    # P-side bridge against the mesh128 450x450 selection used by the p02 fits
    with np.load(COV_NPZ, allow_pickle=False) as d:
        cov_full450 = np.asarray(d["covariance_full"], dtype="f8")
    ids128 = np.concatenate([fit_indices, 150 + fit_indices[keep_p2]])
    cov128_sel = cov_full450[np.ix_(ids128, ids128)]
    pp_bridge_relfro = rel_fro(c_pp, cov128_sel)
    pp_sigma_ratio = np.sqrt(np.diag(c_pp) / np.diag(cov128_sel))
    pp_gate = bool(GATE_PP_SIGMA[0] <= float(np.median(pp_sigma_ratio)) <= GATE_PP_SIGMA[1] and pp_bridge_relfro < GATE_PP_RELFRO)
    if not args.smoke and not pp_gate:
        raise RuntimeError(
            f"pp bridge gate failed: sigma_med={float(np.median(pp_sigma_ratio)):.4f}, relFro={pp_bridge_relfro:.4f}"
        )

    joint_raw = np.block([[c_pp, c_px.T], [c_px, c_xx]])
    sym = 0.5 * (joint_raw + joint_raw.T)
    evals, evecs = np.linalg.eigh(sym)
    spd_meta = {"raw_min_eigenvalue": float(evals[0]), "repaired": False}
    if float(evals[0]) > 0.0:
        joint_final = sym
    else:
        floored = np.maximum(evals, PSD_ABS_FLOOR)
        joint_final = (evecs * floored[None, :]) @ evecs.T
        joint_final = 0.5 * (joint_final + joint_final.T)
        spd_meta.update({"repaired": True, "floor": PSD_ABS_FLOOR, "n_floored": int(np.count_nonzero(evals < PSD_ABS_FLOOR))})
    sig_p = np.sqrt(np.clip(np.diag(c_pp), 1e-300, None))
    sig_x = np.sqrt(np.clip(np.diag(c_xx), 1e-300, None))
    cross_corr = c_px / np.outer(sig_x, sig_p)
    eig_final = np.linalg.eigvalsh(joint_final)

    # ---- models --------------------------------------------------------
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
    mean_p_full = np.mean(np.stack(stacks), axis=0)  # (30,)
    window = window_rows[np.concatenate([fit_indices, 150 + fit_indices]), :]
    model_p = WindowConvolvedP02Model(window, theory[0], theory[1], zeff=zeff)
    if model_p.validate()["status"] != "pass":
        raise RuntimeError("P02 surrogate failed")
    zeff_xi = float(np.mean(np.asarray(np.load(ELL2_SUMMARY, allow_pickle=False)["zeff_by_phase"])))
    cache = build_cache(zeff=zeff_xi, boxsize=2000.0, kmax=5.0, ells=(0, 2), cosmology="abacus_c000")
    model_x = FastWindowRSDModel(FullDiscreteRSDModel(cache, nmu=64), {"mean": load_window(MEAN_WINDOW_NPZ)}, sigma_step=0.05)
    if model_x.validate()["status"] != "pass":
        raise RuntimeError("xi surrogate failed")

    data_p = np.concatenate([mean_p_full[:15], mean_p_full[15:][keep_p2]])
    data_x = np.concatenate([mean_xi[0][mask0], mean_xi[1][mask2]])
    data_joint = np.concatenate([data_p, data_x])
    npk = int(data_p.size)
    nxi = int(data_x.size)

    def eval_p(theta: np.ndarray) -> np.ndarray:
        vec = model_p.evaluate(np.asarray(theta, dtype="f8"))
        return np.concatenate([vec[:15], vec[15:][keep_p2]])

    def eval_x(theta: np.ndarray) -> np.ndarray:
        values = model_x.evaluate(np.asarray(theta, dtype="f8")[:3], model="formal_gic", window_key="mean")
        return np.concatenate([np.asarray(values[0])[mask0], np.asarray(values[2])[mask2]])

    def eval_joint(theta: np.ndarray) -> np.ndarray:
        return np.concatenate([eval_p(theta), eval_x(theta)])

    cross_scales = {"joint": 1.0, "joint_naive": 0.0, "joint_half": 0.5}
    variants = {"p02_marginal": eval_p, "xi02_marginal": eval_x}
    results: dict[str, Any] = {}
    index = 0
    for name in ("p02_marginal", "xi02_marginal", "joint", "joint_naive", "joint_half"):
        if name in variants:
            evaluate, data_v, cov_v = variants[name], None, None
        else:
            evaluate = eval_joint
        if name == "p02_marginal":
            data_v, cov_v = data_p, c_pp
        elif name == "xi02_marginal":
            data_v, cov_v = data_x, c_xx
        else:
            scale = cross_scales[name]
            cov_v = np.block([[c_pp, scale * c_px.T], [scale * c_px, c_xx]])
            data_v = data_joint
        precision, precision_meta = build_precision(cov_v)
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
        summary, chain, logp = run_chain(
            evaluate, data_v, precision, theta_map, 4,
            nwalkers=int(args.nwalkers), nsteps=nsteps, burnin=burnin,
            seed=int(args.seed) + 100 * index,
        )
        index += 1
        if not args.smoke and not all(summary["gates"].values()):
            raise SystemExit(f"MCMC gates failed for {name}: {summary['gates']}")
        if name == "xi02_marginal":
            fisher = fisher_covariance(lambda t: eval_x(np.concatenate([t, [0.0]])), precision, theta_map[:3])
            fisher_names = JOINT_PARAMS[:3]
        else:
            fisher = fisher_covariance(evaluate, precision, theta_map)
            fisher_names = JOINT_PARAMS
        summary["nominal"] = {"theta": [float(v) for v in theta_map], "chi2": float(best.fun @ best.fun), "dof": int(data_v.size - 4)}
        summary["precision_meta"] = precision_meta
        summary["fisher_sigma"] = {n: float(np.sqrt(max(np.diag(fisher)[i], 0.0))) for i, n in enumerate(fisher_names)}
        summary["mcmc_over_fisher_sigma_fNL"] = float(summary["posterior"]["fNL"]["sigma68"] / max(summary["fisher_sigma"]["fNL"], 1.0e-300))
        results[name] = summary
        out_dir = out_root / ("fits" if not args.smoke else "smoke") / name
        out_npz = out_dir / "samples.npz"
        if out_npz.exists():
            raise FileExistsError(f"immutable chain output exists: {out_npz}")
        atomic_savez(out_npz, chain_by_step=chain, log_probability_by_step=logp, prediction_map=evaluate(theta_map), data=data_v)
        print(json.dumps({"variant": name, "fNL": summary["posterior"]["fNL"], "gates": summary["gates"]}, sort_keys=True), flush=True)

    if args.smoke:
        print(json.dumps({"status": "smoke_ok", "elapsed_sec": time.perf_counter() - started}, sort_keys=True))
        return

    sig_p02 = results["p02_marginal"]["posterior"]["fNL"]["sigma68"]
    sig_xi = results["xi02_marginal"]["posterior"]["fNL"]["sigma68"]
    sig_j = results["joint"]["posterior"]["fNL"]["sigma68"]
    sig_n = results["joint_naive"]["posterior"]["fNL"]["sigma68"]
    metrics = {
        "best_marginal_sigma68": float(min(sig_p02, sig_xi)),
        "best_marginal_variant": "p02_marginal" if sig_p02 <= sig_xi else "xi02_marginal",
        "joint_sigma68": float(sig_j),
        "improvement_vs_best_marginal": float(1.0 - sig_j / min(sig_p02, sig_xi)),
        "naive_sigma68": float(sig_n),
        "naive_improvement": float(1.0 - sig_n / min(sig_p02, sig_xi)),
        "cross_covariance_cost_ratio": float(sig_j / sig_n),
        "sigma_sigma_s": {n: float(results[n]["posterior"]["sigma_s"]["sigma68"]) for n in ("p02_marginal", "xi02_marginal", "joint")},
        "sigma_b1": {n: float(results[n]["posterior"]["b1"]["sigma68"]) for n in ("p02_marginal", "xi02_marginal", "joint")},
    }
    audit = {
        "task": "task43_rsd_joint_p02xi02_fit",
        "status": "complete",
        "scope": (
            f"four-way joint under the adopted standard: P0 15 bins + P2 k>={P2_KMIN} ({int(keep_p2.size)} bins) + "
            f"xi0 s>={XI0_SMIN:.0f} ({int(np.count_nonzero(mask0))} bins) + xi2 s>={XI2_SMIN:.0f} ({int(np.count_nonzero(mask2))} bins); "
            "sigma_s shared, sn0 P-side only; covariance blocks and cross terms from the frozen 4497-mode run"
        ),
        "caveats": [
            "diagnostic Gaussian covariance family; the frozen RSD closure remains validation_failed",
            "xi2 per-pole PTE 3.4e-13 and P2 kmin PTE 5.1e-4 still reject the model; the joint inherits this",
            "priors b1 [0.5,5] (joint-experiment convention); fix-2 (EZmock covariance) unexamined",
        ],
        "gates": {
            "internal_projection_relfro": internal,
            "ell02_deconv_reproduction_relfro": deconv_check,
            "pp_bridge": {"sigma_ratio_median": float(np.median(pp_sigma_ratio)), "relfro": pp_bridge_relfro, "pass": pp_gate},
        },
        "joint_covariance": {
            "shape": [int(joint_final.shape[0])] * 2,
            "block_shapes": {"pp": list(c_pp.shape), "px": list(c_px.shape), "xx": list(c_xx.shape)},
            "spd": spd_meta,
            "eigenvalue_min": float(eig_final[0]),
            "condition": float(eig_final[-1] / max(eig_final[0], 1.0e-300)),
            "cross_corr_abs_max": float(np.max(np.abs(cross_corr))),
            "psd_policy": "raw kept when SPD; absolute floor 1e-30 otherwise",
        },
        "mcmc": {"nwalkers": int(args.nwalkers), "nsteps": int(args.nsteps), "burnin": int(args.burnin), "seed": int(args.seed)},
        "inputs": {
            "raw": {"path": str(RAW_NPZ), "sha256": sha256_file(RAW_NPZ)},
            "ell02_deconv": {"path": str(ELL02_DECONV_NPZ), "sha256": sha256_file(ELL02_DECONV_NPZ)},
            "payload": {"path": str(PAYLOAD_NPZ), "sha256": sha256_file(PAYLOAD_NPZ)},
            "cov450": {"path": str(COV_NPZ), "sha256": sha256_file(COV_NPZ)},
        },
        "results": results,
        "metrics": metrics,
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    atomic_write_json(audit_json, audit)
    print(json.dumps({"status": "complete", "metrics": metrics, "output": str(audit_json)}, sort_keys=True))


if __name__ == "__main__":
    main()
