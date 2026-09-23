#!/usr/bin/env python3
"""Run Task 4.3 lightcone xi/joint fits with smin=40 and the BAO mask.

The three P-only chains are invariant under an xi separation cut and are
therefore reused, with hash validation, from the standard smin=50 analysis.
The six xi-containing fits are rerun with the single additional s=45 Mpc/h
bin in each fitted xi multipole.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

import task43_run_lightcone_joint_baomask_v1 as standard
from task43_rsd_common import OUTPUT_ROOT, PROJECT_ROOT, atomic_savez, atomic_write_json, sha256_file


DEFAULT_ROOT = OUTPUT_ROOT / "lightcone_standard_joint_smin40_baomask80_120_v1"
STANDARD_ROOT = standard.DEFAULT_ROOT
STANDARD_AUDIT = STANDARD_ROOT / "task43_lightcone_standard_joint_baomask80_120_v1.json"
REAL_JOINT_COVARIANCE = (
    PROJECT_ROOT
    / "outputs/task43_outputs/joint_pkxi_s30_350_v1/covariance"
    / "task43_joint_cov_64_s30_350_v1.npz"
)
REAL_JOINT_AUDIT = REAL_JOINT_COVARIANCE.with_suffix(".json")
REAL_XI_PATH = (
    PROJECT_ROOT
    / "outputs/task43_outputs/rmin_scan/summary"
    / "task43_mean_xi_mmin1p4e13_x25_s30_350_ds10_fkpP010000.npz"
)
REAL_RIC_OPERATOR = (
    PROJECT_ROOT
    / "outputs/task43_outputs/rmin_scan/operators"
    / "task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_"
    "seed20260712_L2000_s30_350_ds10.npz"
)
REAL_RMIN_AUDIT = PROJECT_ROOT / "outputs/task43_outputs/rmin_scan/audits/task43_rmin_scan_hygiene.json"

P_ONLY_NAMES = ("real_p0", "rsd_p0", "rsd_p02")
AFFECTED_NAMES = (
    "real_xi0",
    "real_joint_p0xi0",
    "rsd_xi0",
    "rsd_joint_p0xi0",
    "rsd_xi02",
    "rsd_joint_p02xi02",
)


def load_standard_p_results() -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    if not STANDARD_AUDIT.is_file():
        raise FileNotFoundError(STANDARD_AUDIT)
    audit = json.loads(STANDARD_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or not all(audit["numerical_gates"].values()):
        raise RuntimeError("standard smin=50 audit did not pass")
    results: dict[str, Any] = {}
    provenance: dict[str, dict[str, str]] = {}
    for name in P_ONLY_NAMES:
        summary_path = STANDARD_ROOT / "fits" / name / "summary.json"
        samples_path = STANDARD_ROOT / "fits" / name / "samples.npz"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("status") != "pass"
            or not all(summary["result"]["mcmc"]["gates"].values())
            or summary.get("output_npz_sha256") != sha256_file(samples_path)
            or summary["result"] != audit["results"][name]
        ):
            raise RuntimeError(f"standard P-only provenance failed for {name}")
        results[name] = summary["result"]
        provenance[name] = {
            "selection": "unchanged P-only standard chain; independent of xi smin",
            "summary_json": str(summary_path),
            "summary_json_sha256": sha256_file(summary_path),
            "samples_npz": str(samples_path),
            "samples_npz_sha256": sha256_file(samples_path),
        }
    return results, provenance


def parameter_change(new: dict[str, Any], old: dict[str, Any], parameter: str) -> dict[str, float | str]:
    new_posterior = new["mcmc"]["posterior"][parameter]
    old_posterior = old["mcmc"]["posterior"][parameter]
    new_ml = float(new["map"]["theta"][parameter])
    old_ml = float(old["map"]["theta"][parameter])
    old_sigma = float(old_posterior["sigma68"])
    return {
        "definition": "smin40 minus smin50; ML shift normalized by the old posterior sigma68",
        "maximum_likelihood_smin40_minus_smin50": new_ml - old_ml,
        "maximum_likelihood_shift_over_smin50_sigma68": (new_ml - old_ml) / old_sigma,
        "sigma68_smin40_over_smin50": float(new_posterior["sigma68"]) / old_sigma,
        "q50_smin40_minus_smin50": float(new_posterior["q50"]) - float(old_posterior["q50"]),
    }


def independent_tension(left: dict[str, Any], right: dict[str, Any], parameter: str) -> dict[str, float | str]:
    left_posterior = left["mcmc"]["posterior"][parameter]
    right_posterior = right["mcmc"]["posterior"][parameter]
    difference = float(right["map"]["theta"][parameter] - left["map"]["theta"][parameter])
    denominator = float(np.hypot(left_posterior["sigma68"], right_posterior["sigma68"]))
    return {
        "definition": "abs(ML_right-ML_left)/hypot(sigma68_left,sigma68_right); ignores cross-correlation",
        "signed_maximum_likelihood_difference_right_minus_left": difference,
        "quadrature_sigma68": denominator,
        "tension_sigma": abs(difference) / denominator,
    }


def comparison_metrics(results: dict[str, Any], old_results: dict[str, Any]) -> dict[str, Any]:
    changes = {
        name: {
            parameter: parameter_change(results[name], old_results[name], parameter)
            for parameter in results[name]["parameter_names"]
        }
        for name in AFFECTED_NAMES
    }
    within_space = {}
    for group, p_name, xi_name, joint_name, shared in (
        ("real", "real_p0", "real_xi0", "real_joint_p0xi0", ("fNL", "b1")),
        ("rsd_monopole", "rsd_p0", "rsd_xi0", "rsd_joint_p0xi0", ("fNL", "b1", "sigma_s")),
        ("rsd_multipole", "rsd_p02", "rsd_xi02", "rsd_joint_p02xi02", ("fNL", "b1", "sigma_s")),
    ):
        within_space[group] = {
            "p_vs_xi_ml_tensions": {
                parameter: independent_tension(results[p_name], results[xi_name], parameter)
                for parameter in shared
            },
            "joint_fNL_sigma68_improvement_over_best_marginal_fraction": 1.0
            - results[joint_name]["mcmc"]["posterior"]["fNL"]["sigma68"]
            / min(
                results[p_name]["mcmc"]["posterior"]["fNL"]["sigma68"],
                results[xi_name]["mcmc"]["posterior"]["fNL"]["sigma68"],
            ),
        }
    return {"smin40_vs_smin50": changes, "smin40_within_space": within_space}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if not 1 <= int(args.threads) <= 8:
        raise ValueError("--threads must be in [1,8]")
    if int(args.burnin) >= int(args.nsteps):
        raise ValueError("--burnin must be smaller than --nsteps")

    nsteps = 400 if args.smoke else int(args.nsteps)
    burnin = 100 if args.smoke else int(args.burnin)
    out_root = Path(args.out_root) / ("smoke" if args.smoke else "")
    audit_json = out_root / "task43_lightcone_joint_smin40_baomask80_120_v1.json"
    covariance_npz = out_root / "task43_lightcone_joint_smin40_baomask80_120_covariance_v1.npz"
    if audit_json.exists():
        raise FileExistsError(f"immutable audit exists: {audit_json}")
    started = time.perf_counter()
    cpus = standard.set_affinity(int(args.threads))
    code_hash = sha256_file(Path(__file__))
    dependency_hash = sha256_file(Path(standard.__file__))

    real_specs, real_metadata, real_arrays = standard.load_real_specs(
        smin=40.0,
        covariance_path=REAL_JOINT_COVARIANCE,
        covariance_audit_path=REAL_JOINT_AUDIT,
        fit_audit_path=REAL_RMIN_AUDIT,
        xi_path=REAL_XI_PATH,
        ric_operator_path=REAL_RIC_OPERATOR,
    )
    rsd_specs, rsd_metadata, rsd_arrays = standard.load_rsd_specs(smin=40.0)
    all_specs = {spec.name: spec for spec in real_specs + rsd_specs}
    specs = [all_specs[name] for name in AFFECTED_NAMES]
    results, p_provenance = load_standard_p_results()
    old_audit = json.loads(STANDARD_AUDIT.read_text(encoding="utf-8"))
    fit_sources: dict[str, Any] = dict(p_provenance)
    numerical_gates: dict[str, bool] = {}

    for index, spec in enumerate(specs):
        metric = standard.ScaledGaussianMetric(spec.covariance)
        numerical_gates[f"{spec.name}_strict_spd"] = bool(metric.eigenvalues[0] > 1.0e-12)
        map_summary, theta_ml = standard.fit_maximum_likelihood(spec, metric)
        prediction = np.asarray(spec.evaluate(theta_ml), dtype="f8")
        fit_root = out_root / "fits" / spec.name
        fit_npz, fit_json = fit_root / "samples.npz", fit_root / "summary.json"
        if fit_npz.exists() != fit_json.exists():
            raise RuntimeError(f"partial resumable output for {spec.name}")
        if fit_npz.exists():
            existing = json.loads(fit_json.read_text(encoding="utf-8"))
            if (
                existing.get("status") != "pass"
                or existing.get("code_sha256") != code_hash
                or existing.get("dependency_sha256") != dependency_hash
                or existing.get("output_npz_sha256") != sha256_file(fit_npz)
                or existing.get("nsteps") != nsteps
                or existing.get("burnin") != burnin
            ):
                raise RuntimeError(f"resumable output contract failed for {spec.name}")
            results[spec.name] = existing["result"]
            fit_sources[spec.name] = {
                "selection": "smin40 BAO-mask chain",
                "summary_json": str(fit_json),
                "summary_json_sha256": sha256_file(fit_json),
                "samples_npz": str(fit_npz),
                "samples_npz_sha256": sha256_file(fit_npz),
            }
            if not args.smoke:
                for gate, passed in results[spec.name]["mcmc"]["gates"].items():
                    numerical_gates[f"{spec.name}_{gate}"] = bool(passed)
            print(json.dumps({"variant": spec.name, "status": "resumed"}), flush=True)
            continue

        chain_summary, chain, logp = standard.run_chain(
            spec,
            metric,
            theta_ml,
            nwalkers=int(args.nwalkers),
            nsteps=nsteps,
            burnin=burnin,
            seed=int(args.seed) + 100 * index,
            nworkers=int(args.threads),
        )
        result = {
            "group": spec.group,
            "parameter_names": list(spec.parameter_names),
            "map": map_summary,
            "mcmc": chain_summary,
            "phase_diagnostics": standard.phase_diagnostics(spec, metric, theta_ml),
            "correlation_eigenvalue_min": float(metric.eigenvalues[0]),
        }
        atomic_savez(
            fit_npz,
            parameter_names=np.asarray(spec.parameter_names),
            chain_by_step=chain,
            log_probability_by_step=logp,
            theta_maximum_likelihood=theta_ml,
            prediction_maximum_likelihood=prediction,
            data=np.asarray(spec.data),
            phase_data=np.asarray(spec.phase_data),
            covariance_single=np.asarray(spec.covariance),
        )
        fit_status = "pass" if args.smoke or all(chain_summary["gates"].values()) else "validation_failed"
        atomic_write_json(
            fit_json,
            {
                "task": "task43_run_lightcone_joint_baomask_smin40_v1",
                "variant": spec.name,
                "status": fit_status,
                "result": result,
                "nsteps": nsteps,
                "burnin": burnin,
                "code_sha256": code_hash,
                "dependency_sha256": dependency_hash,
                "output_npz": str(fit_npz),
                "output_npz_sha256": sha256_file(fit_npz),
            },
        )
        results[spec.name] = result
        fit_sources[spec.name] = {
            "selection": "smin40 BAO-mask chain",
            "summary_json": str(fit_json),
            "summary_json_sha256": sha256_file(fit_json),
            "samples_npz": str(fit_npz),
            "samples_npz_sha256": sha256_file(fit_npz),
        }
        if not args.smoke:
            for gate, passed in chain_summary["gates"].items():
                numerical_gates[f"{spec.name}_{gate}"] = bool(passed)
        print(
            json.dumps(
                {
                    "variant": spec.name,
                    "fNL": chain_summary["posterior"]["fNL"],
                    "maximum_likelihood": map_summary["theta"],
                    "gates": chain_summary["gates"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    numerical_gates["canonical_real_smin40_mask"] = bool(np.count_nonzero(real_arrays["real_xi_mask"]) == 27)
    numerical_gates["canonical_rsd_smin40_mask"] = bool(np.count_nonzero(rsd_arrays["rsd_xi_mask"]) == 27)
    status = "pass" if all(numerical_gates.values()) else "validation_failed"
    summary = {
        "task": "task43_run_lightcone_joint_baomask_smin40_v1",
        "status": status,
        "classification": "Task 4.3 lightcone smin=40 sensitivity test with the canonical BAO mask",
        "comparison_scope": (
            "compare smin40 with the standard smin50 result within each space; real and RSD redshift ranges differ"
        ),
        "mask_policy": {
            "full_range_mpc_h": [40.0, 350.0],
            "excluded_half_open_range_mpc_h": [80.0, 120.0],
            "selected_centers_mpc_h": real_arrays["real_s"][real_arrays["real_xi_mask"]].tolist(),
            "new_centers_relative_to_smin50_mpc_h": [45.0],
            "excluded_centers_mpc_h": [85.0, 95.0, 105.0, 115.0],
            "same_mask_for_rsd_xi0_and_xi2": True,
        },
        "covariance_contract": {
            "observed_curve": "arithmetic mean of 25 realizations",
            "likelihood": "single-realization covariance; never divided by 25",
            "posterior": "single-realization covariance; never divided by 25",
            "goodness_of_fit": "x25 mean and each phase evaluated with C_single; never divided by 25",
            "plot_errorbars": "sqrt(diag(C_single)); never divided by sqrt(25)",
            "selection": "exact principal submatrices/cross blocks; no empirical rescaling and no new eigenvalue floor",
        },
        "real": real_metadata,
        "rsd": rsd_metadata,
        "results": results,
        "fit_sources": fit_sources,
        "metrics": comparison_metrics(results, old_audit["results"]),
        "numerical_gates": numerical_gates,
        "mcmc": {
            "new_chains": list(AFFECTED_NAMES),
            "reused_p_only_chains": list(P_ONLY_NAMES),
            "nwalkers": int(args.nwalkers),
            "nsteps": nsteps,
            "burnin": burnin,
            "seed_base": int(args.seed),
            "parallel_workers": int(args.threads),
        },
        "baseline": {"audit": str(STANDARD_AUDIT), "audit_sha256": sha256_file(STANDARD_AUDIT)},
        "outputs": {"root": str(out_root), "covariance_npz": str(covariance_npz)},
        "code_sha256": code_hash,
        "dependency_sha256": dependency_hash,
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    atomic_savez(covariance_npz, **real_arrays, **rsd_arrays)
    summary["outputs"]["covariance_npz_sha256"] = sha256_file(covariance_npz)
    atomic_write_json(audit_json, summary)
    print(json.dumps({"status": status, "output": str(audit_json), "elapsed_sec": summary["elapsed_sec"]}, sort_keys=True))


if __name__ == "__main__":
    main()
