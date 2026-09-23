#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce original Task43 RSD lightcone chains under the exact kmax=0.08 contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_DIR = PROJECT_ROOT / "codes" / "task43"
if str(TASK43_DIR) not in sys.path:
    sys.path.insert(0, str(TASK43_DIR))

from task43_run_lightcone_joint_baomask_v1 import (  # noqa: E402
    atomic_savez,
    atomic_write_json,
    fit_maximum_likelihood,
    load_rsd_specs,
    run_chain,
    ScaledGaussianMetric,
)


OUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "lightcone_original_kmax0p08_rsd"


def main() -> None:
    specs, metadata, arrays = load_rsd_specs(smin=50.0, pk_kmax=0.08)
    wanted = {"rsd_p02", "rsd_xi02", "rsd_joint_p02xi02"}
    results = {}
    for index, spec in enumerate(spec for spec in specs if spec.name in wanted):
        metric = ScaledGaussianMetric(spec.covariance)
        map_summary, theta_ml = fit_maximum_likelihood(spec, metric)
        chain_summary, chain, logp = run_chain(
            spec,
            metric,
            theta_ml,
            nwalkers=64,
            nsteps=30000,
            burnin=5000,
            seed=20260940 + index,
            nworkers=8,
        )
        result = {
            "group": spec.group,
            "parameter_names": list(spec.parameter_names),
            "map": map_summary,
            "mcmc": chain_summary,
            "correlation_eigenvalue_min": float(metric.eigenvalues[0]),
        }
        output_dir = OUT_ROOT / "fits" / spec.name
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_savez(
            output_dir / "samples.npz",
            parameter_names=np.asarray(spec.parameter_names),
            chain_by_step=chain,
            log_probability_by_step=logp,
            theta_maximum_likelihood=theta_ml,
            data=np.asarray(spec.data),
            covariance_single=np.asarray(spec.covariance),
        )
        atomic_write_json(output_dir / "summary.json", {"task": "task432_lightcone_original_kmax0p08_longchain", "status": "complete", "result": result})
        results[spec.name] = result
        print(json.dumps({"variant": spec.name, "posterior": chain_summary["posterior"], "gates": chain_summary["gates"]}, sort_keys=True), flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(OUT_ROOT / "task432_lightcone_original_kmax0p08_summary.json", {"metadata": metadata, "results": results, "contract": "P0=13, P2=9, kmax=0.08, C_single, BAO-masked xi02"})
    print(json.dumps({"status": "complete", "output": str(OUT_ROOT)}, sort_keys=True))


if __name__ == "__main__":
    main()
