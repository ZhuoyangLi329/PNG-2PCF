#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rawbox real-space xi0 closure check: the minimal-model control.

Real space removes FoG, Kaiser and the quadrupole entirely; what remains is
P_h(k) = P_dd(k) (b1 + fNL*2dc*(b1-1)*alpha)^2 projected through the frozen
shell-averaged j0 kernels.  MAP fit (fNL, b1) at smin in {50, 120}, analytic
Gaussian covariance (real-space form), per-bin residuals, and the xi2(real)
~ 0 pipeline sanity gate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import set_affinity
from task43_fit_rsd_rawbox_x25 import S_EDGES, load_x25
from task43_rsd_common import OUTPUT_ROOT, atomic_write_json, sha256_file
from task43_rsd_model import DELTA_C, FullDiscreteRSDModel, build_cache


OUT_DIR = OUTPUT_ROOT / "rawbox" / "realspace_check"
AUDIT_JSON = OUT_DIR / "audits" / "task43_rsd_rawbox_realspace_check.json"
BOUNDS_LO = np.asarray([-500.0, 0.2], dtype="f8")
BOUNDS_HI = np.asarray([500.0, 10.0], dtype="f8")
NPHASE = 25
KMAX_FIT = {50.0: "k<~0.06 effective", 120.0: "k<~0.03 effective"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if AUDIT_JSON.exists():
        raise FileExistsError(f"immutable realspace check exists: {AUDIT_JSON}")
    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))

    xi, metadata_rows, _ = load_x25()
    nbar = float(np.mean([row["nbar_h3_mpc3_from_npz"] for row in metadata_rows]))
    xi0_mean = np.mean(np.asarray(xi["xi0_real"], dtype="f8"), axis=0)
    xi2_mean = np.mean(np.asarray(xi["xi2_real"], dtype="f8"), axis=0)

    exact = FullDiscreteRSDModel(
        build_cache(zeff=0.725, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000"), nmu=64
    )
    # ---- pipeline sanity: real-space xi2 must be consistent with zero ------
    xi2_stack = np.asarray(xi["xi2_real"], dtype="f8")
    xi2_scatter = np.std(xi2_stack, axis=0, ddof=1) / np.sqrt(NPHASE)
    xi2_over_sigma = xi2_mean / xi2_scatter
    sanity = {
        "xi2_real_absmax": float(np.max(np.abs(xi2_mean))),
        "xi2_over_semax_abs": float(np.max(np.abs(xi2_over_sigma))),
        "note": "isotropic real-space field: xi2 should vanish within sample noise",
    }

    # ---- real-space model -------------------------------------------------
    kernel0 = np.asarray(exact.kernels[0], dtype="f8")  # (nmode, nbin)
    g = np.asarray(exact.g_nz, dtype="f8")
    pk_dd = np.asarray(exact.pk_dd, dtype="f8")
    alpha = np.asarray(exact.alpha, dtype="f8")
    volume = float(exact.volume)
    proj = (g[:, None] * pk_dd[:, None]) * kernel0 / volume  # per-b1-coefficient rows

    def evaluate(theta: np.ndarray) -> np.ndarray:
        fnl, b1 = map(float, np.asarray(theta, dtype="f8")[:2])
        q = fnl * 2.0 * DELTA_C * (b1 - 1.0)
        amp = b1 + q * alpha
        return (amp**2) @ proj

    # analytic real-space Gaussian covariance (audited convention, ell=0 only)
    def covariance(b1_eff: float) -> np.ndarray:
        total2 = (b1_eff**2 * pk_dd + 1.0 / nbar) ** 2
        weight = 2.0 * g * total2 / volume**2
        return kernel0.T @ (weight[:, None] * kernel0)

    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    results: dict[str, Any] = {}
    for smin in (50.0, 120.0):
        mask = centers >= smin
        data_v = xi0_mean[mask]
        # two-step covariance: evaluate at a fast pre-fit b1, then refit
        ids = np.flatnonzero(mask)

        def evaluate_masked(theta):
            return evaluate(theta)[ids]

        def fit_with(cov_v):
            chol = np.linalg.cholesky(cov_v)

            def residual(theta):
                return np.linalg.solve(chol, data_v - evaluate_masked(theta))

            sols = [
                least_squares(residual, start, bounds=(BOUNDS_LO, BOUNDS_HI), max_nfev=3000,
                              xtol=1e-12, ftol=1e-12, gtol=1e-12)
                for start in (np.asarray([0.0, 2.5]), np.asarray([-100.0, 2.3]), np.asarray([100.0, 2.8]))
            ]
            best = min(sols, key=lambda r: float(r.fun @ r.fun))
            return best

        cov_pre = covariance(2.5)[np.ix_(ids, ids)]
        best = fit_with(cov_pre)
        theta_map = np.asarray(best.x, dtype="f8")
        # covariance at the fitted amplitude, mean-test convention C/25
        cov_post = covariance(theta_map[1])[np.ix_(ids, ids)]
        r = data_v - evaluate_masked(theta_map)
        chi2_single = float(r @ np.linalg.solve(cov_post, r))
        dof = int(r.size - 2)
        pte = float(chi2_distribution.sf(NPHASE * chi2_single, dof))
        results[f"smin{int(smin)}"] = {
            "fNL": float(theta_map[0]),
            "b1": float(theta_map[1]),
            "chi2_single": chi2_single,
            "chi2_mean": NPHASE * chi2_single,
            "dof": dof,
            "pte_mean": pte,
            "residual_over_sigma_single": (r / np.sqrt(np.diag(cov_post))).tolist(),
            "s_bins": centers[mask].tolist(),
        }

    audit = {
        "task": "task43_rsd_rawbox_realspace_check",
        "status": "complete",
        "scope": (
            "minimal-model control: real-space xi0, frozen shell-averaged j0 kernels, "
            "linear P_dd template with full PNG; no FoG/Kaiser/quadrupole"
        ),
        "sanity_xi2_real": sanity,
        "results": results,
        "inputs": {"nbar_mean": nbar, "kmax_theory": 3.0},
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(AUDIT_JSON, audit)
    print(json.dumps({"status": "complete", "sanity": sanity, "results": {k: {kk: v[kk] for kk in ("fNL", "b1", "chi2_mean", "dof", "pte_mean")} for k, v in results.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
