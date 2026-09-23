#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Periodic-box joint P0(k)+xi0(s) fit: the no-window control experiment.

Data: the frozen rawbox x25 measurements (16-bin P0 primary with kmin=0.003;
xi0 with smin in {50, 120}).  Covariance: the audited periodic-box Gaussian
conventions for both diagonal blocks (ExactPeriodicPk0Model.gaussian_covariance
for P, periodic_gaussian_covariance for xi, fiducial b1=2.55/sigma_s=8.0) plus
a NEW mode-level cross block

    C_px[i, j] = p * sum_{q in bin i} g_q <total^2 L_0>_q K_0(q, j) / (N_i V)

with the same shell grid, kernels and angular machinery as the audited xi
covariance, N_i the measured mode count of bin i, and a prefactor p in
{1, sqrt(2), 2}: the two audited diagonal blocks differ by a factor 2 in their
mode-pair bookkeeping, so the cross normalization is bracketed and checked
against the x25 empirical cross profile.  Five fit variants at the primary
prefactor plus a cross-prefactor systematic.
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

from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import (
    BOUNDS_HI,
    BOUNDS_LO,
    RAWBOX_FIT_EDGES,
    ExactPeriodicPk0Model,
    load_pk_x25,
    set_affinity,
)
from task43_fit_rsd_rawbox_x25 import (
    FastRSDModel,
    S_EDGES,
    load_x25,
    periodic_gaussian_covariance,
)
from task43_joint_rsd_pkxi_fit import (
    JOINT_PARAMS,
    OPTIMIZER_STARTS,
    build_precision,
    fisher_covariance,
    run_chain,
)
from task43_rsd_model import FullDiscreteRSDModel, build_cache


OUTPUT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/rsd_validation")
OUT_DIR = OUTPUT_ROOT / "rawbox" / "joint_pkxi"
AUDIT_JSON = OUT_DIR / "audits" / "task43_rsd_rawbox_joint_pkxi_summary.json"
COV_FIDUCIAL = {"b1": 2.55, "sigma_s": 8.0, "fnl": 0.0}
CROSS_PREFACTORS = (1.0, np.sqrt(2.0), 2.0)
PRIMARY_PREFACTOR = 1.0


def mode_level_cross(exact: FullDiscreteRSDModel, *, k_edges: np.ndarray, nmodes: np.ndarray, b1: float, sigma_s: float, nbar: float) -> np.ndarray:
    """C_px[i, j] per-shell aggregates times the ell=0 shell-averaged kernels."""
    x = (exact.k_eff[:, None] * exact.mu[None, :] * float(sigma_s)) ** 2
    damping = 1.0 / (1.0 + 0.5 * x) ** 2
    signal = exact.pk_dd[:, None] * (float(b1) + float(exact.f_growth) * exact.mu2[None, :]) ** 2 * damping
    total2_mu = (signal + 1.0 / float(nbar)) ** 2
    angular0 = np.sum(exact.wmu[None, :] * total2_mu, axis=1)  # <total^2 L_0>
    nbin_s = S_EDGES.size - 1
    kernel0 = exact.kernels[0]  # (nmodes_theory, nbin_s)
    bin_center = 0.5 * (k_edges[:, 0] + k_edges[:, 1])
    del bin_center
    aggregate = np.zeros((k_edges.shape[0], np.asarray(exact.k_eff).size), dtype="f8")
    kval = np.asarray(exact.k_eff, dtype="f8")
    assigned = np.zeros(kval.size, dtype=bool)
    for ibin, (lo, hi) in enumerate(k_edges):
        inside = (kval >= float(lo)) & (kval < float(hi))
        aggregate[ibin, inside] = np.asarray(exact.g_nz)[inside] * angular0[inside]
        assigned |= inside
    if not np.all(assigned[np.asarray(exact.g_nz) > 0]):
        # shells outside the P fit bins simply carry no P-bin weight
        pass
    shell_weight = aggregate @ kernel0  # (nbin_k, nbin_s)
    return shell_weight / (np.asarray(nmodes, dtype="f8")[:, None] * float(exact.volume))


def empirical_blocks(pk_stack: np.ndarray, xi_stack: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    p = np.asarray(pk_stack, dtype="f8")
    x = np.asarray(xi_stack, dtype="f8")[:, mask]
    return {
        "pp": np.cov(p, rowvar=False, ddof=1),
        "xx": np.cov(x, rowvar=False, ddof=1),
        "px": np.array([[np.cov(p[:, i], x[:, j], ddof=1)[0, 1] for j in range(x.shape[1])] for i in range(p.shape[1])]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    nsteps = 400 if args.smoke else int(args.nsteps)
    burnin = 100 if args.smoke else int(args.burnin)
    if AUDIT_JSON.exists():
        raise FileExistsError(f"immutable joint summary exists: {AUDIT_JSON}")

    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))

    pk = load_pk_x25(RAWBOX_FIT_EDGES)
    xi, metadata_rows, xi_hashes = load_x25()
    k_edges = np.asarray(pk["k_edges"], dtype="f8")
    k_obs = np.asarray(pk["k"], dtype="f8")
    nmodes = np.asarray(pk["nmodes"], dtype="f8")
    data_p = np.mean(np.asarray(pk["pk0_rsd"], dtype="f8"), axis=0)
    nbar = float(np.mean([row["nbar_h3_mpc3_from_npz"] for row in metadata_rows]))

    exact = FullDiscreteRSDModel(
        build_cache(zeff=0.725, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000"), nmu=64
    )
    pk_model = ExactPeriodicPk0Model(exact, k_edges)
    xi_model = FastRSDModel(exact, sigma_step=0.05)

    # ---- covariance blocks ------------------------------------------------
    c_pp = pk_model.gaussian_covariance(b1=COV_FIDUCIAL["b1"], sigma_s=COV_FIDUCIAL["sigma_s"], nbar=nbar)
    c_xx64 = periodic_gaussian_covariance(
        exact, b1=COV_FIDUCIAL["b1"], sigma_s=COV_FIDUCIAL["sigma_s"], nbar=nbar
    )
    cross_unit = mode_level_cross(
        exact, k_edges=k_edges, nmodes=nmodes,
        b1=COV_FIDUCIAL["b1"], sigma_s=COV_FIDUCIAL["sigma_s"], nbar=nbar,
    )

    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    smins = {"s50": 50.0, "s120": 120.0}
    masks = {key: centers >= value for key, value in smins.items()}
    xi_mean = np.mean(np.asarray(xi["xi0_rsd"], dtype="f8"), axis=0)

    # empirical calibration of the cross prefactor (diagnostic 1c)
    emp = {
        key: empirical_blocks(np.asarray(pk["pk0_rsd"], dtype="f8"), np.asarray(xi["xi0_rsd"], dtype="f8"), masks[key])
        for key in masks
    }
    prefactor_report: dict[str, Any] = {}
    for key, mask in masks.items():
        ids0 = np.flatnonzero(mask)
        cxx_sel = c_xx64[np.ix_(ids0, ids0)]
        px_emp = emp[key]["px"]
        denom_e = np.sqrt(np.outer(np.diag(emp[key]["pp"]), np.diag(emp[key]["xx"])))
        corr_emp = px_emp / denom_e
        denom_a = np.sqrt(np.outer(np.diag(c_pp), np.diag(cxx_sel)))
        rows = {}
        for p in CROSS_PREFACTORS:
            corr_a = (p * cross_unit[:, mask]) / denom_a
            rows[f"p={p:.4f}"] = {
                "corr_rms_emp": float(np.sqrt(np.mean((corr_a - corr_emp) ** 2))),
                "corr_profile_analytic_mean": float(np.mean(corr_a)),
                "corr_profile_empirical_mean": float(np.mean(corr_emp)),
                "scale_ratio_emp_over_ana": float(
                    np.sum(px_emp * (p * cross_unit[:, mask])) / max(np.sum((p * cross_unit[:, mask]) ** 2), 1e-300)
                ),
            }
        prefactor_report[key] = rows
    emp_diag = {
        key: {
            "pp_sigma_ratio_median": float(np.median(np.sqrt(np.diag(emp[key]["pp"]) / np.diag(c_pp)))),
            "xx_sigma_ratio_median": float(
                np.median(
                    np.sqrt(
                        np.diag(emp[key]["xx"])
                        / np.diag(c_xx64[np.ix_(np.flatnonzero(masks[key]), np.flatnonzero(masks[key]))])
                    )
                )
            ),
            "px_emp_absmax": float(np.max(np.abs(emp[key]["px"]))),
        }
        for key in masks
    }

    def fit_variant(name: str, evaluate, data_v: np.ndarray, cov_v: np.ndarray, seed_offset: int) -> dict[str, Any]:
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
            seed=int(args.seed) + 100 * seed_offset,
        )
        nparam = 3 if name.startswith("xi0_marginal") else 4
        if name.startswith("xi0_marginal"):
            fisher = fisher_covariance(lambda t: evaluate(np.concatenate([t, [0.0]])), precision, theta_map[:3])
            fisher_names = JOINT_PARAMS[:3]
        else:
            fisher = fisher_covariance(evaluate, precision, theta_map)
            fisher_names = JOINT_PARAMS
        summary["nominal"] = {"theta": [float(v) for v in theta_map], "chi2": float(best.fun @ best.fun), "dof": int(data_v.size - nparam)}
        summary["precision_meta"] = precision_meta
        summary["fisher_sigma"] = {n: float(np.sqrt(max(np.diag(fisher)[i], 0.0))) for i, n in enumerate(fisher_names)}
        summary["mcmc_over_fisher_sigma_fNL"] = float(
            summary["posterior"]["fNL"]["sigma68"] / max(summary["fisher_sigma"]["fNL"], 1.0e-300)
        )
        out_npz = OUT_DIR / ("fits" if not args.smoke else "smoke") / name / "samples.npz"
        if out_npz.exists():
            raise FileExistsError(f"immutable chain output exists: {out_npz}")
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_npz, chain_by_step=chain, log_probability_by_step=logp, prediction_map=evaluate(theta_map), data=data_v)
        return summary

    def eval_p(theta: np.ndarray) -> np.ndarray:
        return pk_model.evaluate(np.asarray(theta, dtype="f8"))

    def make_eval_xi(mask: np.ndarray):
        def evaluate(theta: np.ndarray) -> np.ndarray:
            return np.asarray(xi_model.evaluate(np.asarray(theta, dtype="f8")[:3])[0])[mask]
        return evaluate

    results: dict[str, Any] = {}
    order = [
        ("p0_marginal", lambda: (eval_p, data_p, c_pp)),
        ("xi0_marginal_s50", lambda: (make_eval_xi(masks["s50"]), xi_mean[masks["s50"]], c_xx64[np.ix_(np.flatnonzero(masks["s50"]), np.flatnonzero(masks["s50"]))])),
        ("joint_s50", lambda: (make_joint(masks["s50"], PRIMARY_PREFACTOR), None, None)),
        ("joint_naive_s50", lambda: (make_joint(masks["s50"], 0.0), None, None)),
        ("joint_s50_psqrt2", lambda: (make_joint(masks["s50"], float(np.sqrt(2.0))), None, None)),
        ("joint_s120", lambda: (make_joint(masks["s120"], PRIMARY_PREFACTOR), None, None)),
    ]

    def make_joint(mask: np.ndarray, prefactor: float):
        def evaluate(theta: np.ndarray) -> np.ndarray:
            return np.concatenate([eval_p(theta), make_eval_xi(mask)(theta)])

        def build():
            ids0 = np.flatnonzero(mask)
            cxx = c_xx64[np.ix_(ids0, ids0)]
            cpx = prefactor * cross_unit[:, mask]
            data = np.concatenate([data_p, xi_mean[mask]])
            cov = np.block([[c_pp, cpx], [cpx.T, cxx]])
            cov = 0.5 * (cov + cov.T)
            evals, evecs = np.linalg.eigh(cov)
            floor = max(1.0e-14 * float(evals[-1]), 1.0e-300)
            if float(evals[0]) < floor:
                cov = (evecs * np.maximum(evals, floor)[None, :]) @ evecs.T
                cov = 0.5 * (cov + cov.T)
            return data, cov

        evaluate.build = build
        return evaluate

    index = 0
    for name, supplier in order:
        if name in ("p0_marginal", "xi0_marginal_s50"):
            evaluate, data_v, cov_v = supplier()
        else:
            evaluate, _, _ = supplier()
            data_v, cov_v = evaluate.build()
        summary = fit_variant(name, evaluate, data_v, cov_v, index)
        index += 1
        if not args.smoke and not all(summary["gates"].values()):
            raise SystemExit(f"MCMC gates failed for {name}: {summary['gates']}")
        results[name] = summary
        print(json.dumps({"variant": name, "fNL": summary["posterior"]["fNL"], "gates": summary["gates"]}, sort_keys=True), flush=True)

    if args.smoke:
        print(json.dumps({"status": "smoke_ok", "elapsed_sec": time.perf_counter() - started}, sort_keys=True))
        return

    sig = {n: results[n]["posterior"]["fNL"]["sigma68"] for n in results}
    metrics = {
        "p0_marginal_sigma": sig["p0_marginal"],
        "xi0_marginal_s50_sigma": sig["xi0_marginal_s50"],
        "best_marginal_sigma": float(min(sig["p0_marginal"], sig["xi0_marginal_s50"])),
        "joint_s50_sigma": sig["joint_s50"],
        "joint_s50_improvement": float(1.0 - sig["joint_s50"] / min(sig["p0_marginal"], sig["xi0_marginal_s50"])),
        "joint_naive_s50_sigma": sig["joint_naive_s50"],
        "naive_cost_ratio": float(sig["joint_s50"] / sig["joint_naive_s50"]),
        "joint_s50_psqrt2_sigma": sig["joint_s50_psqrt2"],
        "prefactor_sensitivity_sigma": float(abs(sig["joint_s50_psqrt2"] - sig["joint_s50"]) / sig["joint_s50"]),
        "joint_s120_sigma": sig["joint_s120"],
        "lightcone_reference": {"p0_15bin": 35.33, "xi0_s50": 36.53, "joint": 33.62, "note": "boxsafe lightcone monopole joint for comparison"},
    }
    audit = {
        "task": "task43_rsd_rawbox_joint_pkxi",
        "status": "complete",
        "scope": (
            "periodic-box control: 16-bin P0 (kmin=0.003, free sn0) + xi0 (smin 50/120), shared (fNL,b1,sigma_s); "
            "audited periodic-Gaussian diagonal blocks + new mode-level cross block"
        ),
        "caveats": [
            "diagnostic Gaussian covariance family; the rawbox closure remains validation_failed (xi0 shape)",
            "the two audited diagonal blocks differ by a factor 2 in mode-pair bookkeeping; the cross prefactor is bracketed p in {1,sqrt2,2} and calibrated against the x25 empirical cross profile",
            "fix-2 (EZmock empirical covariance) remains unexamined",
        ],
        "covariance": {
            "fiducial": COV_FIDUCIAL,
            "nbar_mean": nbar,
            "pp": {"diag_only": True, "source": "ExactPeriodicPk0Model.gaussian_covariance"},
            "xx": {"source": "periodic_gaussian_covariance", "shape": [64, 64]},
            "cross_formula": "p * sum_q g_q <total^2 L_0> K_0(q,j) / (N_i V); theory-shell aggregation on the FullDiscrete grid",
            "primary_prefactor": PRIMARY_PREFACTOR,
            "prefactor_report": prefactor_report,
            "empirical_diagnostic": emp_diag,
        },
        "mcmc": {"nwalkers": int(args.nwalkers), "nsteps": int(args.nsteps), "burnin": int(args.burnin), "seed": int(args.seed)},
        "inputs": {
            "pk": {"k_edges": k_edges.tolist(), "nmodes": nmodes.tolist()},
            "nphase": 25,
            "xi_summary_first": str(metadata_rows[0].get("path", "")),
        },
        "results": results,
        "metrics": metrics,
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    from task43_rsd_common import atomic_write_json

    atomic_write_json(AUDIT_JSON, audit)
    print(json.dumps({"status": "complete", "metrics": metrics, "output": str(AUDIT_JSON)}, sort_keys=True))


if __name__ == "__main__":
    main()
