#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rawbox real-space P0(k) closure: the P-side twin of the xi0 real-space check.

Same linear P_dd template, same 16 frozen bins (k=0.003..0.095), same x25
mean, params (fNL, b1) only.  P_h(k) = P_dd(k)(b1 + fNL*2dc*(b1-1)*alpha)^2
as an exact parent-mode bin average; Gaussian mode-count covariance with the
real-space signal (no Kaiser, no FoG).  MAP + PTE + MCMC chains for contour.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import emcee
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import (
    RAWBOX_FIT_EDGES,
    load_pk_x25,
    set_affinity,
)
from task43_rsd_common import OUTPUT_ROOT, atomic_savez, atomic_write_json
from task43_rsd_model import DELTA_C, FullDiscreteRSDModel, build_cache


OUT_DIR = OUTPUT_ROOT / "rawbox" / "realspace_check"
AUDIT_JSON = OUT_DIR / "audits" / "task43_rsd_rawbox_realspace_pk_check.json"
BOUNDS_LO = np.asarray([-500.0, 0.2], dtype="f8")
BOUNDS_HI = np.asarray([500.0, 10.0], dtype="f8")
NPHASE = 25


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    if AUDIT_JSON.exists():
        raise FileExistsError(f"immutable pk realspace check exists: {AUDIT_JSON}")
    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))

    pk = load_pk_x25(RAWBOX_FIT_EDGES)
    k_obs = np.asarray(pk["k"], dtype="f8")
    nmodes = np.asarray(pk["nmodes"], dtype="f8")
    data_v = np.mean(np.asarray(pk["pk0_real"], dtype="f8"), axis=0)
    nbar = float(np.mean(np.asarray(pk["nbar"], dtype="f8")))

    # parent-mode machinery for the model (real space: no Kaiser, no FoG)
    exact = FullDiscreteRSDModel(
        build_cache(zeff=0.725, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000"), nmu=64
    )
    kfund = 2.0 * np.pi / 2000.0
    nmax = int(np.ceil(float(np.max(RAWBOX_FIT_EDGES)) / kfund))
    integers = np.arange(-nmax, nmax + 1, dtype="i4")
    nx, ny, nz = np.meshgrid(integers, integers, integers, indexing="ij")
    n2 = (nx.astype("f8") ** 2 + ny.astype("f8") ** 2 + nz.astype("f8") ** 2).ravel()
    kval = kfund * np.sqrt(n2)
    bin_id = np.full(kval.size, -1, dtype="i4")
    for ibin, (lo, hi) in enumerate(RAWBOX_FIT_EDGES):
        bin_id[(kval >= lo) & (kval < hi)] = ibin
    keep = bin_id >= 0
    k_modes = kval[keep]
    b_id = bin_id[keep]
    counts = np.bincount(b_id, minlength=RAWBOX_FIT_EDGES.shape[0]).astype("f8")
    pk_dd = np.interp(np.log(k_modes), np.log(np.asarray(exact.k_eff, dtype="f8")), np.asarray(exact.pk_dd, dtype="f8"))
    alpha = np.interp(np.log(k_modes), np.log(np.asarray(exact.k_eff, dtype="f8")), np.asarray(exact.alpha, dtype="f8"))

    def evaluate(theta: np.ndarray) -> np.ndarray:
        fnl, b1 = map(float, np.asarray(theta, dtype="f8")[:2])
        signal = pk_dd * (b1 + fnl * 2.0 * DELTA_C * (b1 - 1.0) * alpha) ** 2
        return np.bincount(b_id, weights=signal, minlength=counts.size) / counts

    def covariance(b1: float) -> np.ndarray:
        total2 = (b1**2 * pk_dd + 1.0 / nbar) ** 2
        summed = np.bincount(b_id, weights=total2, minlength=counts.size)
        return np.diag(2.0 * summed / counts**2)

    def fit_with(cov_v):
        chol = np.linalg.cholesky(cov_v)

        def residual(theta):
            return np.linalg.solve(chol, data_v - evaluate(theta))

        sols = [
            least_squares(residual, s, bounds=(BOUNDS_LO, BOUNDS_HI), max_nfev=3000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            for s in (np.asarray([0.0, 2.5]), np.asarray([-100.0, 2.3]), np.asarray([100.0, 2.8]))
        ]
        return min(sols, key=lambda r: float(r.fun @ r.fun))

    best = fit_with(covariance(2.5))
    theta_map = np.asarray(best.x, dtype="f8")
    cov_v = covariance(theta_map[1])
    precision = np.linalg.inv(cov_v)
    r = data_v - evaluate(theta_map)
    chi2_single = float(r @ precision @ r)
    dof = int(r.size - 2)
    pte = float(chi2_distribution.sf(NPHASE * chi2_single, dof))

    def log_probability(theta):
        v = np.asarray(theta, dtype="f8")
        if np.any(v < BOUNDS_LO) or np.any(v > BOUNDS_HI):
            return -np.inf
        d = data_v - evaluate(v)
        return -0.5 * float(d @ precision @ d)

    rng = np.random.default_rng(int(args.seed))
    initial = theta_map[None, :] + rng.normal(size=(int(args.nwalkers), 2)) * np.asarray([4.0, 0.015])
    initial = np.clip(initial, BOUNDS_LO + 1e-7, BOUNDS_HI - 1e-7)
    sampler = emcee.EnsembleSampler(int(args.nwalkers), 2, log_probability)
    sampler.run_mcmc(initial, int(args.nsteps), progress=False)
    chain = np.asarray(sampler.get_chain(discard=int(args.burnin)), dtype="f8")
    logp = np.asarray(sampler.get_log_prob(discard=int(args.burnin)), dtype="f8")
    flat = chain.reshape(-1, 2)
    q = np.percentile(flat, [16, 50, 84], axis=0)
    out_npz = OUT_DIR / "fits" / "pk0_s16" / "samples.npz"
    if out_npz.exists():
        raise FileExistsError(out_npz)
    atomic_savez(out_npz, chain_by_step=chain, log_probability_by_step=logp, prediction_map=evaluate(theta_map), data=data_v)

    audit = {
        "task": "task43_rsd_rawbox_realspace_pk_check",
        "status": "complete",
        "scope": "real-space P0(k), frozen 16 bins (kmin=0.003, kmax=0.095), linear template exact parent-mode bin average",
        "map": {"fNL": float(theta_map[0]), "b1": float(theta_map[1])},
        "chi2_single": chi2_single,
        "chi2_mean": NPHASE * chi2_single,
        "dof": dof,
        "pte_mean": pte,
        "posterior": {
            "fNL": {"q16": float(q[0, 0]), "q50": float(q[1, 0]), "q84": float(q[2, 0])},
            "b1": {"q16": float(q[0, 1]), "q50": float(q[1, 1]), "q84": float(q[2, 1])},
        },
        "acceptance": float(np.mean(sampler.acceptance_fraction)),
        "k_obs": k_obs.tolist(),
        "residual_over_sigma_single": (r / np.sqrt(np.diag(cov_v))).tolist(),
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(AUDIT_JSON, audit)
    print(json.dumps({"status": "complete", "map": audit["map"], "chi2_mean": audit["chi2_mean"], "dof": dof, "pte_mean": pte,
                      "posterior": audit["posterior"]}, sort_keys=True))


if __name__ == "__main__":
    main()
