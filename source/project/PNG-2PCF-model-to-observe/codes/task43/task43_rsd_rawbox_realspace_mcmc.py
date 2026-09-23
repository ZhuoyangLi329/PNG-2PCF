#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCMC chains for the rawbox real-space xi0 check (2 params: fNL, b1)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import emcee
from scipy.optimize import least_squares

from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import set_affinity
from task43_fit_rsd_rawbox_x25 import S_EDGES, load_x25
from task43_rsd_common import OUTPUT_ROOT, atomic_savez, atomic_write_json
from task43_rsd_model import DELTA_C, FullDiscreteRSDModel, build_cache


OUT_DIR = OUTPUT_ROOT / "rawbox" / "realspace_check"
BOUNDS_LO = np.asarray([-500.0, 0.2], dtype="f8")
BOUNDS_HI = np.asarray([500.0, 10.0], dtype="f8")
NPHASE = 25


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    cpus = set_affinity(int(args.threads))

    xi, metadata_rows, _ = load_x25()
    nbar = float(np.mean([row["nbar_h3_mpc3_from_npz"] for row in metadata_rows]))
    xi0_mean = np.mean(np.asarray(xi["xi0_real"], dtype="f8"), axis=0)
    exact = FullDiscreteRSDModel(
        build_cache(zeff=0.725, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000"), nmu=64
    )
    kernel0 = np.asarray(exact.kernels[0], dtype="f8")
    g = np.asarray(exact.g_nz, dtype="f8")
    pk_dd = np.asarray(exact.pk_dd, dtype="f8")
    alpha = np.asarray(exact.alpha, dtype="f8")
    proj = (g[:, None] * pk_dd[:, None]) * kernel0 / float(exact.volume)

    def evaluate(theta):
        fnl, b1 = map(float, np.asarray(theta, dtype="f8")[:2])
        amp = b1 + fnl * 2.0 * DELTA_C * (b1 - 1.0) * alpha
        return (amp**2) @ proj

    def covariance(b1):
        total2 = (b1**2 * pk_dd + 1.0 / nbar) ** 2
        return kernel0.T @ ((2.0 * g * total2 / float(exact.volume) ** 2)[:, None] * kernel0)

    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    results = {}
    for index, smin in enumerate((50.0, 120.0)):
        ids = np.flatnonzero(centers >= smin)
        data_v = xi0_mean[ids]

        def eval_m(theta):
            return evaluate(theta)[ids]

        def fit_with(cov_v):
            chol = np.linalg.cholesky(cov_v)

            def residual(theta):
                return np.linalg.solve(chol, data_v - eval_m(theta))

            sols = [
                least_squares(residual, s, bounds=(BOUNDS_LO, BOUNDS_HI), max_nfev=3000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                for s in (np.asarray([0.0, 2.5]), np.asarray([-100.0, 2.3]), np.asarray([100.0, 2.8]))
            ]
            return min(sols, key=lambda r: float(r.fun @ r.fun))

        best = fit_with(covariance(2.5)[np.ix_(ids, ids)])
        theta_map = np.asarray(best.x, dtype="f8")
        cov_v = covariance(theta_map[1])[np.ix_(ids, ids)]
        precision = np.linalg.inv(cov_v)

        def log_probability(theta):
            v = np.asarray(theta, dtype="f8")
            if np.any(v < BOUNDS_LO) or np.any(v > BOUNDS_HI):
                return -np.inf
            d = data_v - eval_m(v)
            return -0.5 * float(d @ precision @ d)

        rng = np.random.default_rng(int(args.seed) + 100 * index)
        initial = theta_map[None, :] + rng.normal(size=(int(args.nwalkers), 2)) * np.asarray([4.0, 0.015])
        initial = np.clip(initial, BOUNDS_LO + 1e-7, BOUNDS_HI - 1e-7)
        sampler = emcee.EnsembleSampler(int(args.nwalkers), 2, log_probability)
        sampler.run_mcmc(initial, int(args.nsteps), progress=False)
        chain = np.asarray(sampler.get_chain(discard=int(args.burnin)), dtype="f8")
        logp = np.asarray(sampler.get_log_prob(discard=int(args.burnin)), dtype="f8")
        flat = chain.reshape(-1, 2)
        q = np.percentile(flat, [16, 50, 84], axis=0)
        key = f"smin{int(smin)}"
        out_npz = OUT_DIR / "fits" / key / "samples.npz"
        if out_npz.exists():
            raise FileExistsError(out_npz)
        atomic_savez(out_npz, chain_by_step=chain, log_probability_by_step=logp, prediction_map=eval_m(theta_map), data=data_v)
        results[key] = {
            "fNL": {"q16": float(q[0, 0]), "q50": float(q[1, 0]), "q84": float(q[2, 0])},
            "b1": {"q16": float(q[0, 1]), "q50": float(q[1, 1]), "q84": float(q[2, 1])},
            "acceptance": float(np.mean(sampler.acceptance_fraction)),
        }
        print(json.dumps({key: results[key]}, sort_keys=True), flush=True)

    atomic_write_json(OUT_DIR / "audits" / "task43_rsd_rawbox_realspace_mcmc.json", {
        "task": "task43_rsd_rawbox_realspace_mcmc",
        "status": "complete",
        "mcmc": {"nwalkers": int(args.nwalkers), "nsteps": int(args.nsteps), "burnin": int(args.burnin), "seed": int(args.seed)},
        "results": results,
        "cpu_affinity": cpus,
    })


if __name__ == "__main__":
    main()
