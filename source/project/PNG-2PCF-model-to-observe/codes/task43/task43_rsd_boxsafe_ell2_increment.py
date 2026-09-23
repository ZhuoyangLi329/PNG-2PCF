#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quadrupole-increment test: does adding xi2 to xi0 move the RSD constraints?

For every smin in {50, 80, 100} run two fits with the audited closure
machinery (fit_one, formal-GIC, mean window):

* ell=(0,)   control  on the ell0 block of the new 64x64 ell02 covariance;
* ell=(0, 2) increment on the full 64x64 ell02 covariance;

and report sigma(fNL), sigma(b1), sigma(sigma_s), posterior correlations and
per-pole mean-shape PTEs.  The comparison at matched smin isolates "adding
xi2" from any covariance-family change; the known small-s xi2 mismodeling
makes smin=80 the primary branch and smin=50 the diagnostic one.
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
from scipy.stats import chi2 as chi2_distribution

from task43_fit_rsd_lightcone_x25 import (
    FastWindowRSDModel,
    convergence_diagnostics,
    fit_one,
    load_inputs,
    load_window,
    set_affinity,
)
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
AUDIT_JSON = ELL2_DIR / "audits" / "task43_rsd_boxsafe_ell2_increment_summary.json"

SMIN_VALUES = (50.0, 80.0, 100.0)
PRIMARY_SMIN = 80.0


def chain_statistics(chain: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(chain, dtype="f8")
    names = ("fNL", "b1", "sigma_s")
    quantiles = np.percentile(flat, [16.0, 50.0, 84.0], axis=0)
    sigma68 = 0.5 * (quantiles[2] - quantiles[0])
    corr = np.corrcoef(flat.T)
    return {
        "posterior": {
            name: {
                "q16": float(quantiles[0, i]),
                "q50": float(quantiles[1, i]),
                "q84": float(quantiles[2, i]),
                "sigma68": float(sigma68[i]),
            }
            for i, name in enumerate(names)
        },
        "posterior_correlations": {
            "fNL_b1": float(corr[0, 1]),
            "fNL_sigma_s": float(corr[0, 2]),
            "b1_sigma_s": float(corr[1, 2]),
        },
    }


def per_pole_mean_pte(
    inputs: dict[str, Any],
    data_by_pole: dict[int, np.ndarray],
    model_by_pole: dict[int, np.ndarray],
    *,
    smin: float,
    nparams: int = 3,
) -> dict[str, Any]:
    s = np.asarray(inputs["s"], dtype="f8")
    mask = s >= float(smin)
    nbin = int(np.count_nonzero(mask))
    cov = np.asarray(inputs["covariance"], dtype="f8")
    out: dict[str, Any] = {}
    for ell, vector in data_by_pole.items():
        iell = inputs["covariance_ells"].index(int(ell))
        ids = iell * s.size + np.flatnonzero(mask)
        block = cov[np.ix_(ids, ids)] / len(PHASES)
        residual = vector - model_by_pole[int(ell)]
        chi2 = float(residual @ np.linalg.solve(block, residual))
        dof = max(nbin - nparams, 1)
        out[f"ell{ell}"] = {"chi2_Cmean": chi2, "dof": dof, "pte": float(chi2_distribution.sf(chi2, dof))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if AUDIT_JSON.exists():
        raise FileExistsError(f"immutable increment summary exists: {AUDIT_JSON}")

    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))
    inputs = load_inputs(SUMMARY_NPZ, ELL02_DECONV_NPZ)
    mean_xi = np.mean(inputs["xi"], axis=0)
    mean_rr = np.mean(inputs["rr"], axis=0)
    zeff_mean = float(np.mean(inputs["zeff"]))
    theory_cache = build_cache(zeff=zeff_mean, boxsize=2000.0, kmax=5.0, ells=(0, 2), cosmology="abacus_c000")
    model = FastWindowRSDModel(
        FullDiscreteRSDModel(theory_cache, nmu=64),
        {"mean": load_window(MEAN_WINDOW_NPZ)},
        sigma_step=0.05,
    )
    surrogate = model.validate()
    if surrogate["status"] != "pass":
        raise RuntimeError(f"ell02 likelihood surrogate failed: {surrogate}")

    results: dict[str, Any] = {}
    fits_dir = ELL2_DIR / "fits"
    index = 0
    for smin in SMIN_VALUES:
        for ells in ((0,), (0, 2)):
            index += 1
            fit, chain, logp, model_map = fit_one(
                inputs=inputs,
                xi_by_ell=mean_xi,
                rr=mean_rr,
                fast_model=model,
                window_key="mean",
                model="formal_gic",
                ells=ells,
                smin=float(smin),
                nwalkers=int(args.nwalkers),
                nsteps=int(args.nsteps),
                burnin=int(args.burnin),
                seed=int(args.seed) + 100 * index,
                optimizer_only=False,
            )
            convergence = convergence_diagnostics(
                chain,
                names=list(fit["parameter_names"]),
                nwalkers=int(args.nwalkers),
                nsteps=int(args.nsteps),
                burnin=int(args.burnin),
            )
            gates = {key: bool(value) for key, value in convergence["gates"].items()}
            if not all(gates.values()):
                raise SystemExit(f"MCMC gates failed for smin={smin}, ells={ells}: {gates}")
            stats = chain_statistics(chain)
            s = np.asarray(inputs["s"], dtype="f8")
            mask = s >= float(smin)
            data_by_pole = {int(ell): np.asarray(mean_xi[i], dtype="f8")[mask] for i, ell in enumerate(ells)}
            model_by_pole = {
                int(ell): np.asarray(model_map, dtype="f8")[i * int(np.count_nonzero(mask)):(i + 1) * int(np.count_nonzero(mask))]
                for i, ell in enumerate(ells)
            }
            key = f"smin{int(smin):03d}_ell{''.join(str(e) for e in ells)}"
            results[key] = {
                "smin_mpc_h": float(smin),
                "ells": [int(e) for e in ells],
                **stats,
                "gates": gates,
                "map_theta": {k: float(v) for k, v in zip(("fNL", "b1", "sigma_s"), np.asarray(fit["optimizer"]["x"], dtype="f8"))},
                "per_pole_mean_goodness": per_pole_mean_pte(inputs, data_by_pole, model_by_pole, smin=float(smin)),
                "audited_context": fit.get("task43_fit_context", {}),
            }
            out_dir = fits_dir / key
            out_npz = out_dir / "samples.npz"
            if out_npz.exists():
                raise FileExistsError(f"immutable chain output exists: {out_npz}")
            atomic_savez(out_npz, chain_flat=chain, log_probability=logp, model_map=model_map, smin=np.asarray(float(smin)))
            print(json.dumps({"key": key, "fNL": stats["posterior"]["fNL"], "gates": gates}, sort_keys=True), flush=True)

    metrics: dict[str, Any] = {}
    for smin in SMIN_VALUES:
        control = results[f"smin{int(smin):03d}_ell0"]
        increment = results[f"smin{int(smin):03d}_ell02"]
        metrics[f"smin{int(smin)}"] = {
            "sigma_fNL": {"ell0": control["posterior"]["fNL"]["sigma68"], "ell02": increment["posterior"]["fNL"]["sigma68"]},
            "sigma_b1": {"ell0": control["posterior"]["b1"]["sigma68"], "ell02": increment["posterior"]["b1"]["sigma68"]},
            "sigma_sigma_s": {"ell0": control["posterior"]["sigma_s"]["sigma68"], "ell02": increment["posterior"]["sigma_s"]["sigma68"]},
            "improvement_fNL": float(1.0 - increment["posterior"]["fNL"]["sigma68"] / control["posterior"]["fNL"]["sigma68"]),
            "improvement_sigma_s": float(1.0 - increment["posterior"]["sigma_s"]["sigma68"] / control["posterior"]["sigma_s"]["sigma68"]),
            "correlations": {"ell0": control["posterior_correlations"], "ell02": increment["posterior_correlations"]},
            "per_pole_pte": increment["per_pole_mean_goodness"],
        }

    audit = {
        "task": "task43_rsd_boxsafe_ell2_increment",
        "status": "complete",
        "scope": (
            "boxsafe 0.4<zobs<0.8 RSD lightcone; formal-GIC xi multipoles from the frozen x25 mean, "
            "ell02 RR-deconvolved jaxpower covariance; matched-smin xi0 control vs xi0+xi2 increment"
        ),
        "caveats": [
            "diagnostic Gaussian covariance; the frozen RSD closure remains validation_failed",
            "xi2 small-s mismodeling is expected below smin~80 (frozen evidence from the lightcone smin scan)",
            "primary branch is smin=80; smin=50 is diagnostic only",
        ],
        "mcmc": {"nwalkers": int(args.nwalkers), "nsteps": int(args.nsteps), "burnin": int(args.burnin), "seed": int(args.seed)},
        "primary_smin_mpc_h": PRIMARY_SMIN,
        "inputs": {
            "summary_npz": {"path": str(SUMMARY_NPZ), "sha256": sha256_file(SUMMARY_NPZ)},
            "ell02_covariance": {"path": str(ELL02_DECONV_NPZ), "sha256": sha256_file(ELL02_DECONV_NPZ)},
            "mean_window": {"path": str(MEAN_WINDOW_NPZ), "sha256": sha256_file(MEAN_WINDOW_NPZ)},
            "theory_cache": {"path": str(theory_cache), "sha256": sha256_file(theory_cache)},
        },
        "surrogate_validation_status": surrogate["status"],
        "results": results,
        "metrics": metrics,
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    atomic_write_json(AUDIT_JSON, audit)
    print(json.dumps({"status": "complete", "metrics": metrics, "output": str(AUDIT_JSON)}, sort_keys=True))


if __name__ == "__main__":
    main()
