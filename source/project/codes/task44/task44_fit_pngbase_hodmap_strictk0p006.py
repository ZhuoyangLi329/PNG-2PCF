#!/usr/bin/env python3
"""Fixed-baseline-fNL/free-p fits to the strict-kmin galaxy P0 products."""

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

from task44_fit_pngbase_hodmap_rawbox import (
    ExactPeriodicPkModel,
    build_theory_cache,
    covariance_iteration_summary,
    fit_map,
    load_theory,
    regularize_covariance,
    run_mcmc,
)
from task44_pngbase_hodhost_mmin1e13_common import (
    BOX_SIZE,
    BOX_VOLUME,
    CATALOGS,
    K_FUND,
    LRG_CONSERVATIVE_PK_FIT_EDGES,
    LRG_CONSERVATIVE_PK_KMAX,
    PK_KMAX_CONTRACT,
    PK_STRICT_FIT_EDGES,
    PK_STRICT_KMIN,
    REDSHIFT,
    atomic_savez,
    atomic_write_json,
    ensure_output_dirs,
    get_spec,
    lrg_pk_conservative_fit_prefix,
    set_cpu_affinity,
    sha256_file,
    strict_fit_prefix,
    strict_pk_metadata_path,
    strict_pk_path,
    to_jsonable,
)


def load_measurement(
    tag: str,
    *,
    mesh: int = 400,
    fit_edges: np.ndarray = PK_STRICT_FIT_EDGES,
) -> dict[str, Any]:
    path = strict_pk_path(tag, mesh=mesh)
    metadata_path = strict_pk_metadata_path(tag, mesh=mesh)
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing strict P0 measurement: {path} / {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "pass" or metadata.get("output_sha256") != sha256_file(path):
        raise RuntimeError(f"invalid strict P0 metadata/hash: {path}")
    with np.load(path, allow_pickle=False) as data:
        result = {
            "path": path,
            "sha256": metadata["output_sha256"],
            "metadata": metadata,
            "k": np.asarray(data["k"], dtype="f8"),
            "k_edges": np.asarray(data["k_edges"], dtype="f8"),
            "nmodes": np.asarray(data["nmodes"], dtype="f8"),
            "data": np.asarray(data["pk0"], dtype="f8"),
            "nbar": float(np.asarray(data["nbar"]).item()),
            "ndata": int(np.asarray(data["ndata"]).item()),
            "mesh": int(np.asarray(data["mesh"]).item()),
        }
    if not np.array_equal(result["k_edges"], PK_STRICT_FIT_EDGES):
        raise RuntimeError(f"strict fit edges changed in {path}")
    expected = np.asarray(fit_edges, dtype="f8")
    if not np.array_equal(result["k_edges"][: expected.shape[0]], expected):
        raise RuntimeError(f"requested fit edges are not an exact prefix of the strict measurement: {path}")
    result["source_total_bins"] = int(result["data"].size)
    for name in ("k", "k_edges", "nmodes", "data"):
        result[name] = np.asarray(result[name])[: expected.shape[0]]
    return result


def _validated_existing(prefix: Path) -> bool:
    output_npz = prefix.with_suffix(".npz")
    output_json = prefix.with_suffix(".json")
    if not output_npz.is_file() or not output_json.is_file():
        return False
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    return summary.get("status") == "pass" and summary.get("output_npz_sha256") == sha256_file(output_npz)


def run_one(
    tag: str,
    *,
    theory: dict[str, Any],
    theory_cache_path: Path,
    kmax_contract: float,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    overwrite: bool,
) -> None:
    spec = get_spec(tag)
    if np.isclose(kmax_contract, PK_KMAX_CONTRACT, rtol=0.0, atol=1.0e-13):
        fit_edges = np.asarray(PK_STRICT_FIT_EDGES, dtype="f8")
        prefix = strict_fit_prefix(tag)
    elif np.isclose(kmax_contract, LRG_CONSERVATIVE_PK_KMAX, rtol=0.0, atol=1.0e-13):
        fit_edges = np.asarray(LRG_CONSERVATIVE_PK_FIT_EDGES, dtype="f8")
        prefix = lrg_pk_conservative_fit_prefix(tag)
    else:
        raise ValueError(f"unsupported kmax bin-centre contract: {kmax_contract}")
    output_npz = prefix.with_suffix(".npz")
    output_json = prefix.with_suffix(".json")
    if _validated_existing(prefix) and not overwrite:
        print(f"[skip] validated {output_json}", flush=True)
        return
    if not overwrite and (output_npz.exists() or output_json.exists()):
        raise FileExistsError(f"partial/unvalidated strict-fit output exists: {output_npz} / {output_json}")
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    measured = load_measurement(tag, fit_edges=fit_edges)
    model = ExactPeriodicPkModel(theory, fit_edges, fnl=spec.fnl)
    mode_counts_match = bool(
        np.array_equal(np.asarray(model.counts, dtype="i8"), np.asarray(measured["nmodes"], dtype="i8"))
    )
    mode_means_match = bool(np.allclose(model.k_mean, measured["k"], rtol=0.0, atol=1.0e-14))
    if not mode_counts_match or not mode_means_match:
        raise RuntimeError(
            f"strict exact-mode model/estimator mismatch for {tag}: "
            f"counts={mode_counts_match}, means={mode_means_match}"
        )

    started = time.perf_counter()
    theta_initial = np.asarray([2.0, 1.4, 0.0], dtype="f8")
    covariance_initial_raw = model.covariance(theta_initial, nbar=measured["nbar"])
    covariance_initial, covariance_initial_metadata = regularize_covariance(covariance_initial_raw)
    map_initial = fit_map(model, measured["data"], covariance_initial)

    theta_covariance = theta_initial.copy()
    parameter_tolerance = np.asarray([1.0e-5, 1.0e-4, 1.0e-5], dtype="f8")
    fixed_point_history: list[dict[str, Any]] = []
    fixed_point_converged = False
    for iteration in range(12):
        raw = model.covariance(theta_covariance, nbar=measured["nbar"])
        matrix, diagnostics = regularize_covariance(raw)
        current_map = fit_map(model, measured["data"], matrix)
        theta_fit = np.asarray([current_map["theta"][name] for name in model.names], dtype="f8")
        delta = theta_fit - theta_covariance
        fixed_point_history.append(
            {
                "iteration": iteration,
                "covariance_fiducial": {
                    name: float(value) for name, value in zip(model.names, theta_covariance, strict=True)
                },
                "map": current_map,
                "delta_map_minus_covariance_fiducial": {
                    name: float(value) for name, value in zip(model.names, delta, strict=True)
                },
                "covariance_diagnostics": diagnostics,
            }
        )
        theta_covariance = theta_fit
        if iteration >= 1 and np.all(np.abs(delta) < parameter_tolerance):
            fixed_point_converged = True
            break

    covariance_final_raw = model.covariance(theta_covariance, nbar=measured["nbar"])
    covariance_final, covariance_final_metadata = regularize_covariance(covariance_final_raw)
    map_final = fit_map(model, measured["data"], covariance_final)
    theta_map_final = np.asarray([map_final["theta"][name] for name in model.names], dtype="f8")
    fixed_point_final_delta = theta_map_final - theta_covariance

    initial_mcmc, initial_chain, initial_logp = run_mcmc(
        model,
        measured["data"],
        covariance_initial,
        map_initial,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burnin=burnin,
        seed=seed + 101,
    )
    final_mcmc, final_chain, final_logp = run_mcmc(
        model,
        measured["data"],
        covariance_final,
        map_final,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burnin=burnin,
        seed=seed,
    )
    covariance_posterior_comparison = covariance_iteration_summary(initial_mcmc, final_mcmc)
    final_prediction = np.asarray(model.evaluate(theta_map_final), dtype="f8")
    sigma = np.sqrt(np.diag(covariance_final))
    residual_sigma = (np.asarray(measured["data"], dtype="f8") - final_prediction) / sigma
    p_sigma = float(final_mcmc["posterior"]["p"]["sigma68"])

    gates = {
        "fixed_fnl_equals_catalog_baseline": bool(model.fnl == spec.fnl and spec.fnl in (30.0, 100.0)),
        "free_parameters_are_b1_p_sn0": tuple(model.names) == ("b1", "p", "sn0"),
        "strict_physical_kmin_is_0p006": PK_STRICT_KMIN == 0.006,
        "first_shell_is_0p006_to_0p007": bool(np.array_equal(measured["k_edges"][0], [0.006, 0.007])),
        "first_shell_has_six_signed_modes": int(measured["nmodes"][0]) == 6,
        "uses_exact_requested_strict_shells": bool(np.array_equal(measured["k_edges"], fit_edges)),
        "last_shell_matches_requested_contract": bool(np.array_equal(measured["k_edges"][-1], fit_edges[-1])),
        "kmax_bin_center_matches_requested_contract": bool(
            np.isclose(np.mean(measured["k_edges"][-1]), kmax_contract, rtol=0.0, atol=1.0e-13)
        ),
        "exact_theory_and_estimator_mode_counts_match": mode_counts_match,
        "exact_theory_and_estimator_mode_means_match": mode_means_match,
        "covariance_uses_same_mode_sets": covariance_final.shape == (fit_edges.shape[0], fit_edges.shape[0]),
        "single_box_covariance_not_divided": True,
        "map_optimizer_success": bool(map_final["success"]),
        "map_not_at_prior_boundary": bool(not map_final["at_parameter_boundary"]),
        "covariance_fixed_point_converged": bool(fixed_point_converged),
        "covariance_final_p_shift_below_0p01sigma": bool(
            abs(float(fixed_point_final_delta[1])) < 0.01 * p_sigma
        ),
        "mcmc_all_convergence_gates": bool(all(final_mcmc["gates"].values())),
    }
    status = "pass" if all(gates.values()) else "review"
    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "task": f"task44_fit_pngbase_hodmap_strictk0p006_kcentermax{kmax_contract:.3f}",
        "status": status,
        "tag": tag,
        "realization": spec.realization,
        "geometry": "full periodic real-space cube",
        "boxsize_mpc_h": BOX_SIZE,
        "volume_mpc_h3": BOX_VOLUME,
        "kfund_h_mpc": K_FUND,
        "redshift": REDSHIFT,
        "catalog_baseline_fnl": spec.fnl,
        "fixed_fnl": spec.fnl,
        "free_parameters": list(model.names),
        "priors": {name: list(model.priors[name]) for name in model.names},
        "model": "[b1 + fNL*2*delta_c*(b1-p)*alpha(k)]^2 Pm + sn0*1e4; full fNL^2",
        "input": {
            "path": str(measured["path"]),
            "sha256": measured["sha256"],
            "ndata": measured["ndata"],
            "nbar_h3_mpc3": measured["nbar"],
            "mesh": measured["mesh"],
        },
        "fit_range": {
            "physical_mode_rule": "k>=0.006 h/Mpc",
            "kmin_edge_h_mpc": PK_STRICT_KMIN,
            "first_shell_h_mpc": measured["k_edges"][0].tolist(),
            "kmax_bin_center_h_mpc": kmax_contract,
            "kmax_bin_upper_edge_h_mpc": float(measured["k_edges"][-1, 1]),
            "last_shell_h_mpc": measured["k_edges"][-1].tolist(),
            "nbins": int(measured["data"].size),
            "source_measurement_total_bins": measured["source_total_bins"],
            "k_edges_h_mpc": measured["k_edges"].tolist(),
        },
        "theory": {
            "path": str(theory_cache_path),
            "sha256": sha256_file(theory_cache_path),
            "cosmology": "abacus_c000",
            "periodic_discrete_modes": True,
            "survey_window": False,
            "gic_ric": False,
        },
        "covariance": {
            "type": "single-periodic-box analytic Gaussian",
            "mode_sets": "identical signed lattice shells used by estimator and exact theory average",
            "volume_mpc_h3": BOX_VOLUME,
            "nbar_h3_mpc3": measured["nbar"],
            "not_divided_by_number_of_boxes": True,
            "initial_fiducial": {
                name: float(value) for name, value in zip(model.names, theta_initial, strict=True)
            },
            "initial_diagnostics": covariance_initial_metadata,
            "initial_map": map_initial,
            "fixed_point_converged": fixed_point_converged,
            "fixed_point_parameter_tolerance": {
                name: float(value) for name, value in zip(model.names, parameter_tolerance, strict=True)
            },
            "fixed_point_history": fixed_point_history,
            "final_fiducial": {
                name: float(value) for name, value in zip(model.names, theta_covariance, strict=True)
            },
            "final_map_minus_fiducial": {
                name: float(value) for name, value in zip(model.names, fixed_point_final_delta, strict=True)
            },
            "final_diagnostics": covariance_final_metadata,
            "iteration_posterior_comparison": covariance_posterior_comparison,
            "hartlap_percival": False,
        },
        "map": map_final,
        "mcmc": final_mcmc,
        "mcmc_initial_covariance": initial_mcmc,
        "residual_diagnostic_not_plotted": {
            "max_abs_sigma": float(np.max(np.abs(residual_sigma))),
            "rms_sigma": float(np.sqrt(np.mean(residual_sigma**2))),
        },
        "gates": gates,
        "elapsed_sec": elapsed,
    }
    atomic_savez(
        output_npz,
        coordinate=np.asarray(measured["k"], dtype="f8"),
        coordinate_edges=np.asarray(measured["k_edges"], dtype="f8"),
        nmodes=np.asarray(measured["nmodes"], dtype="f8"),
        data=np.asarray(measured["data"], dtype="f8"),
        prediction_map=final_prediction,
        residual_sigma=np.asarray(residual_sigma, dtype="f8"),
        covariance_initial=np.asarray(covariance_initial, dtype="f8"),
        covariance_final=np.asarray(covariance_final, dtype="f8"),
        chain_initial_by_step=np.asarray(initial_chain, dtype="f8"),
        logp_initial_by_step=np.asarray(initial_logp, dtype="f8"),
        chain_final_by_step=np.asarray(final_chain, dtype="f8"),
        logp_final_by_step=np.asarray(final_logp, dtype="f8"),
        parameter_names=np.asarray(model.names),
        catalog_baseline_fnl=np.asarray(spec.fnl, dtype="f8"),
        fixed_fnl=np.asarray(spec.fnl, dtype="f8"),
        nbar=np.asarray(measured["nbar"], dtype="f8"),
        ndata=np.asarray(measured["ndata"], dtype="i8"),
        boxsize=np.asarray(BOX_SIZE, dtype="f8"),
        volume=np.asarray(BOX_VOLUME, dtype="f8"),
        kfund=np.asarray(K_FUND, dtype="f8"),
        summary_json=np.asarray(json.dumps(to_jsonable(summary), sort_keys=True)),
    )
    summary["output_npz"] = str(output_npz)
    summary["output_npz_sha256"] = sha256_file(output_npz)
    atomic_write_json(output_json, summary)
    p = final_mcmc["posterior"]["p"]
    print(
        f"[done] {tag} fixed fNL={spec.fnl:g} kmax={kmax_contract:.3f} p={p['q50']:.4f} "
        f"-{p['q50']-p['q16']:.4f}/+{p['q84']-p['q50']:.4f} "
        f"chi2/dof={map_final['chi2']:.2f}/{map_final['dof']} status={status}",
        flush=True,
    )
    if status != "pass":
        print(f"[review] failed gates: {[name for name, value in gates.items() if not value]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", action="append", choices=tuple(CATALOGS))
    parser.add_argument("--kmax", type=float, choices=(0.08, 0.10), default=0.10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=10_000)
    parser.add_argument("--burnin", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if int(args.burnin) >= int(args.nsteps):
        raise ValueError("burnin must be less than nsteps")
    set_cpu_affinity(int(args.threads))
    ensure_output_dirs()
    theory_cache_path = build_theory_cache(kmax=5.0, smin=30.0, overwrite=False)
    theory = load_theory(theory_cache_path, smin=30.0)
    tags = list(CATALOGS) if not args.tag else [get_spec(tag).tag for tag in args.tag]
    for tag in tags:
        tag_offset = 0 if tag == "c300" else 3000
        run_one(
            tag,
            theory=theory,
            theory_cache_path=theory_cache_path,
            kmax_contract=float(args.kmax),
            nwalkers=int(args.nwalkers),
            nsteps=int(args.nsteps),
            burnin=int(args.burnin),
            seed=int(args.seed) + tag_offset + (80_000 if np.isclose(args.kmax, 0.08) else 0),
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    main()
