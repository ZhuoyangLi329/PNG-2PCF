#!/usr/bin/env python3
"""Strict BAO-masked rawbox RSD P0, xi0, and joint fits.

This is the monopole-only companion to task43_run_rawbox_joint_baomask_v1.
It selects the P0/xi0 principal submatrix of that run's immutable strict
common-mode covariance, then fits P0, xi0, and their joint vector under the
same single-realization covariance contract.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import ExactPeriodicPk0Model, load_pk_x25, set_affinity
from task43_fit_rsd_rawbox_x25 import FastRSDModel, load_x25
from task43_rawbox_numerics import canonical_correlations
from task43_rsd_common import OUTPUT_ROOT, S_EDGES, atomic_savez, atomic_write_json, rawbox_xi_primary_mask, sha256_file
from task43_rsd_model import FullDiscreteRSDModel, build_cache
from task43_run_rawbox_joint_baomask_v1 import (
    RSD_BOUNDS,
    RSD_NAMES,
    RSD_P_BOUNDS,
    RSD_P_NAMES,
    RSD_P_STARTS,
    RSD_STARTS,
    fit_edges_for_kmax,
    empirical_covariance_diagnostics,
    fit_map,
    run_chain,
    xi_mask_for_smin,
)


SOURCE_ROOT = OUTPUT_ROOT / "rawbox" / "kmax0p08_smin50_v1"
SOURCE_AUDIT = SOURCE_ROOT / "task43_rawbox_standard_joint_baomask80_120_v1.json"
SOURCE_COVARIANCE = SOURCE_ROOT / "task43_rawbox_standard_joint_baomask80_120_covariance_v1.npz"
DEFAULT_ROOT = OUTPUT_ROOT / "rawbox" / "monopole_kmax0p08_smin50_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--pk-kmax", type=float, choices=(0.08, 0.10), default=0.08)
    parser.add_argument("--smin", type=float, choices=(40.0, 50.0), default=50.0)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if not 1 <= int(args.threads) <= 8:
        raise ValueError("--threads must be in [1, 8]")
    if int(args.burnin) >= int(args.nsteps):
        raise ValueError("--burnin must be smaller than --nsteps")
    nsteps = 500 if args.smoke else int(args.nsteps)
    burnin = 100 if args.smoke else int(args.burnin)
    fit_edges = fit_edges_for_kmax(float(args.pk_kmax))
    source_root = Path(args.source_root) if args.source_root is not None else SOURCE_ROOT
    source_audit_path = source_root / "task43_rawbox_standard_joint_baomask80_120_v1.json"
    source_covariance_path = source_root / "task43_rawbox_standard_joint_baomask80_120_covariance_v1.npz"
    out_root = Path(args.out_root) / ("smoke" if args.smoke else "")
    audit_json = out_root / "task43_rawbox_monopole_joint_baomask80_120_v1.json"
    if audit_json.exists():
        raise FileExistsError(f"immutable audit exists: {audit_json}")

    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))
    code_hash = sha256_file(Path(__file__))
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if source_audit.get("status") != "pass" or not all(source_audit["numerical_gates"].values()):
        raise RuntimeError("source four-way audit did not pass")
    if source_audit["outputs"]["covariance_npz_sha256"] != sha256_file(source_covariance_path):
        raise RuntimeError("source covariance hash changed")
    source_contract = source_audit.get("fit_contract", {})
    if not np.isclose(float(source_contract.get("pk_kmax_h_mpc", np.nan)), float(args.pk_kmax)):
        raise RuntimeError("source P(k) cutoff does not match monopole request")
    if not np.isclose(float(source_contract.get("xi_smin_mpc_h", np.nan)), float(args.smin)):
        raise RuntimeError("source xi smin does not match monopole request")
    contract = source_audit["covariance_contract"]
    if contract["likelihood"] != "single-realization covariance; never divided by 25":
        raise RuntimeError("source likelihood covariance contract changed")

    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    mask = xi_mask_for_smin(centers, float(args.smin))
    expected_xi_count = 26 if np.isclose(float(args.smin), 50.0) else 27
    if int(np.count_nonzero(mask)) != expected_xi_count:
        raise RuntimeError("canonical xi mask changed")
    with np.load(source_covariance_path, allow_pickle=False) as payload:
        stored_mask = np.asarray(payload["xi_mask"], dtype=bool)
        if not np.array_equal(mask, stored_mask):
            raise RuntimeError("source covariance mask changed")
        pp_full = np.asarray(payload["rsd_pp"], dtype="f8")
        xx_full = np.asarray(payload["rsd_xx"], dtype="f8")
        xp_full = np.asarray(payload["rsd_xp"], dtype="f8")
        joint_full = np.asarray(payload["rsd_joint"], dtype="f8")
    nk, ns = fit_edges.shape[0], int(np.count_nonzero(mask))
    c_pp = pp_full[:nk, :nk]
    c_xx = xx_full[:ns, :ns]
    c_xp = xp_full[:ns, :nk]
    c_joint = np.block([[c_pp, c_xp.T], [c_xp, c_xx]])
    source_ids = np.concatenate((np.arange(nk), 2 * nk + np.arange(ns)))
    if not np.array_equal(c_joint, joint_full[np.ix_(source_ids, source_ids)]):
        raise RuntimeError("monopole covariance is not an exact principal submatrix")
    eig = np.linalg.eigvalsh(
        c_joint / np.outer(np.sqrt(np.diag(c_joint)), np.sqrt(np.diag(c_joint)))
    )
    canonical = canonical_correlations(c_pp, c_xx, c_xp)
    if float(eig[0]) <= 1.0e-12 or float(np.max(canonical)) >= 1.0:
        raise RuntimeError("monopole joint covariance is not strict SPD")

    cache_path = build_cache(zeff=0.725, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000")
    exact = FullDiscreteRSDModel(cache_path, nmu=64)
    p_model = ExactPeriodicPk0Model(exact, fit_edges, sigma_step=0.05)
    x_model = FastRSDModel(exact, sigma_step=0.05)
    p_validation, x_validation = p_model.validate(), x_model.validate()
    if p_validation["status"] != "pass" or x_validation["status"] != "pass":
        raise RuntimeError("monopole model surrogate validation failed")

    pk = load_pk_x25(fit_edges)
    xi, xi_metadata, xi_hashes = load_x25()
    data_p = np.mean(np.asarray(pk["pk0_rsd"], dtype="f8"), axis=0)
    data_x = np.mean(np.asarray(xi["xi0_rsd"], dtype="f8"), axis=0)[mask]
    data_joint = np.concatenate((data_p, data_x))

    def evaluate_p(theta: np.ndarray) -> np.ndarray:
        return p_model.evaluate(np.asarray(theta, dtype="f8"))

    def evaluate_x(theta: np.ndarray) -> np.ndarray:
        return np.asarray(x_model.evaluate(np.asarray(theta, dtype="f8")[:3])[0], dtype="f8")[mask]

    def evaluate_joint(theta: np.ndarray) -> np.ndarray:
        return np.concatenate((evaluate_p(theta), evaluate_x(theta)))

    specifications = (
        ("rsd_p0", data_p, evaluate_p, c_pp, RSD_P_NAMES, RSD_P_BOUNDS, RSD_P_STARTS),
        ("rsd_xi0", data_x, evaluate_x, c_xx, RSD_NAMES, RSD_BOUNDS, RSD_STARTS),
        ("rsd_joint_p0xi0", data_joint, evaluate_joint, c_joint, RSD_P_NAMES, RSD_P_BOUNDS, RSD_P_STARTS),
    )
    results: dict[str, Any] = {}
    for index, (name, data, evaluate, covariance, names, bounds, starts) in enumerate(specifications):
        map_summary, metric = fit_map(data, evaluate, covariance, names=names, bounds=bounds, starts=starts)
        theta_ml = np.asarray([map_summary["theta"][key] for key in names], dtype="f8")
        prediction = np.asarray(evaluate(theta_ml), dtype="f8")
        fit_dir = out_root / "fits" / name
        fit_npz, fit_json = fit_dir / "samples.npz", fit_dir / "summary.json"
        if fit_npz.exists() != fit_json.exists():
            raise RuntimeError(f"partial resumable output for {name}")
        if fit_npz.exists():
            existing = json.loads(fit_json.read_text(encoding="utf-8"))
            if (
                existing.get("status") != "pass"
                or existing.get("code_sha256") != code_hash
                or existing.get("output_npz_sha256") != sha256_file(fit_npz)
                or existing.get("nsteps") != nsteps
                or existing.get("burnin") != burnin
            ):
                raise RuntimeError(f"resumable output contract failed for {name}")
            results[name] = existing["result"]
            print(json.dumps({"variant": name, "status": "resumed"}), flush=True)
            continue
        chain_summary, chain, logp = run_chain(
            data,
            evaluate,
            metric,
            map_summary,
            names=names,
            bounds=bounds,
            nwalkers=int(args.nwalkers),
            nsteps=nsteps,
            burnin=burnin,
            seed=int(args.seed) + 100 * index,
            nworkers=int(args.threads),
        )
        result = {"parameter_names": list(names), "map": map_summary, "mcmc": chain_summary}
        atomic_savez(
            fit_npz,
            parameter_names=np.asarray(names),
            chain_by_step=chain,
            log_probability_by_step=logp,
            theta_maximum_likelihood=theta_ml,
            prediction_maximum_likelihood=prediction,
            data=np.asarray(data),
            covariance_single=np.asarray(covariance),
        )
        atomic_write_json(
            fit_json,
            {
                "task": "task43_run_rawbox_monopole_joint_baomask_v1",
                "variant": name,
                "status": "pass" if args.smoke or all(chain_summary["gates"].values()) else "validation_failed",
                "result": result,
                "nsteps": nsteps,
                "burnin": burnin,
                "code_sha256": code_hash,
                "output_npz": str(fit_npz),
                "output_npz_sha256": sha256_file(fit_npz),
            },
        )
        results[name] = result
        print(json.dumps({"variant": name, "fNL": chain_summary["posterior"]["fNL"], "gates": chain_summary["gates"]}, sort_keys=True), flush=True)

    numerical_gates = {
        "source_audit": True,
        "canonical_mask": True,
        "principal_submatrix_exact": True,
        "joint_correlation_eigenvalue_min_above_1e_12": bool(eig[0] > 1.0e-12),
        "canonical_correlation_below_one": bool(np.max(canonical) < 1.0),
        "p0_model": bool(p_validation["status"] == "pass"),
        "xi0_model": bool(x_validation["status"] == "pass"),
    }
    if not args.smoke:
        for name, result in results.items():
            for gate, passed in result["mcmc"]["gates"].items():
                numerical_gates[f"{name}_{gate}"] = bool(passed)
    status = "pass" if all(numerical_gates.values()) else "validation_failed"
    p_sigma = float(results["rsd_p0"]["mcmc"]["posterior"]["fNL"]["sigma68"])
    x_sigma = float(results["rsd_xi0"]["mcmc"]["posterior"]["fNL"]["sigma68"])
    joint_sigma = float(results["rsd_joint_p0xi0"]["mcmc"]["posterior"]["fNL"]["sigma68"])
    summary = {
        "task": "task43_run_rawbox_monopole_joint_baomask_v1",
        "status": status,
        "classification": "standard rawbox RSD P0, BAO-masked xi0, and strict common-mode joint test",
        "mask_policy": source_audit["mask_policy"],
        "fit_contract": {
            "pk_kmax_h_mpc": float(args.pk_kmax),
            "pk_bin_count_per_pole": int(fit_edges.shape[0]),
            "pk_fit_edges_h_mpc": fit_edges.tolist(),
            "xi_smin_mpc_h": float(args.smin),
            "xi_smax_mpc_h": 350.0,
            "xi_bin_count_per_pole": int(ns),
        },
        "covariance_contract": {
            "observed_curve": "arithmetic mean of 25 realizations",
            "likelihood": "single-realization covariance; never divided by 25",
            "posterior": "single-realization covariance; never divided by 25",
            "goodness_of_fit": "x25 mean curve evaluated with single-realization covariance; never divided by 25",
            "plot_errorbars": "sqrt(diag(C_single)); never divided by sqrt(25)",
            "construction": "exact P0/xi0 principal submatrix of the strict four-way common-mode covariance",
            "automatic_eigenvalue_floor": False,
            "empirical_cross_rescaling": False,
            "correlation_eigenvalue_min": float(eig[0]),
            "canonical_correlation_max": float(np.max(canonical)),
        },
        "results": results,
        "metrics": {
            "best_marginal_sigma68_fNL": min(p_sigma, x_sigma),
            "joint_sigma68_fNL": joint_sigma,
            "joint_improvement_over_best_marginal_fraction": 1.0 - joint_sigma / min(p_sigma, x_sigma),
        },
        "empirical_x25_diagnostics": empirical_covariance_diagnostics(
            np.asarray(pk["pk0_rsd"], dtype="f8"),
            np.asarray(xi["xi0_rsd"], dtype="f8")[:, mask],
            {"pp": c_pp, "xx": c_xx, "xp": c_xp},
        ),
        "model_validation": {"p0": p_validation, "xi0": x_validation},
        "numerical_gates": numerical_gates,
        "inputs": {
            "source_audit": str(source_audit_path),
            "source_audit_sha256": sha256_file(source_audit_path),
            "source_covariance": str(source_covariance_path),
            "source_covariance_sha256": sha256_file(source_covariance_path),
            "p0_measurement_sha256": pk["hashes"],
            "xi_measurement_sha256": xi_hashes,
            "xi_nbar_by_phase": [float(row["nbar_h3_mpc3_from_npz"]) for row in xi_metadata],
            "theory_cache": str(cache_path),
            "theory_cache_sha256": sha256_file(cache_path),
        },
        "mcmc": {
            "nwalkers": int(args.nwalkers),
            "nsteps": nsteps,
            "burnin": burnin,
            "seed_base": int(args.seed),
            "parallel_workers": int(args.threads),
        },
        "outputs": {"root": str(out_root)},
        "code_sha256": code_hash,
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    atomic_write_json(audit_json, summary)
    print(json.dumps({"status": status, "output": str(audit_json), "elapsed_sec": summary["elapsed_sec"]}, sort_keys=True))


if __name__ == "__main__":
    main()
