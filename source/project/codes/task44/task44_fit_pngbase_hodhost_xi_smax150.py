#!/usr/bin/env python3
"""Fixed-baseline-fNL/free-p fits to Task44 LRG or occupied-host xi0.

The LRG likelihood supports the audited 30--150, 50--150, and 30--350
pair-count ranges.  The mean and Gaussian covariance use the same
full-discrete periodic-box model, with the radial kernel sliced exactly to
the requested native 10 Mpc/h shells.
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

from task44_fit_pngbase_hodmap_rawbox import (
    FullDiscreteXiModel,
    build_theory_cache,
    covariance_iteration_summary,
    fit_map,
    load_theory,
    regularize_covariance,
    run_mcmc,
)
from task44_pngbase_hodmap_rawbox_common import (
    xi_metadata_path as lrg_xi_metadata_path,
    xi_path as lrg_xi_path,
)
from task44_pngbase_hodhost_mmin1e13_common import (
    BOX_SIZE,
    BOX_VOLUME,
    CATALOGS,
    K_FUND,
    LRG_EXTENDED_XI_FIT_CENTERS,
    LRG_EXTENDED_XI_FIT_EDGES,
    LRG_EXTENDED_XI_SMAX,
    LRG_EXTENDED_XI_SMIN,
    LRG_CONSERVATIVE_XI_FIT_CENTERS,
    LRG_CONSERVATIVE_XI_FIT_EDGES,
    LRG_CONSERVATIVE_XI_SMAX,
    LRG_CONSERVATIVE_XI_SMIN,
    REDSHIFT,
    XI_FIT_CENTERS,
    XI_FIT_EDGES,
    XI_FIT_SMAX,
    XI_FIT_SMIN,
    atomic_savez,
    atomic_write_json,
    ensure_output_dirs,
    get_spec,
    host_xi_fit_prefix,
    lrg_xi_conservative_fit_prefix,
    lrg_xi_extended_fit_prefix,
    lrg_xi_fit_prefix,
    set_cpu_affinity,
    sha256_file,
    to_jsonable,
    xi_engine_metadata_path,
    xi_engine_path,
)


def load_halo_xi(tag: str) -> dict[str, Any]:
    path = xi_engine_path(tag, "fcfc")
    metadata_path = xi_engine_metadata_path(tag, "fcfc")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing halo xi product: {path} / {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "pass" or metadata.get("output_sha256") != sha256_file(path):
        raise RuntimeError(f"invalid halo xi metadata/hash: {path}")
    with np.load(path, allow_pickle=False) as data:
        full_edges = np.asarray(data["s_edges"], dtype="f8")
        full_centers = np.asarray(data["s"], dtype="f8")
        full_xi = np.asarray(data["xi0"], dtype="f8")
        ndata = int(np.asarray(data["ndata"]).item())
    if not np.array_equal(full_edges[: XI_FIT_EDGES.size], XI_FIT_EDGES):
        raise RuntimeError(f"halo xi does not contain the frozen 30--150 shells: {path}")
    nbins = XI_FIT_EDGES.size - 1
    result = {
        "path": path,
        "sha256": metadata["output_sha256"],
        "metadata": metadata,
        "s": full_centers[:nbins],
        "s_edges": full_edges[: nbins + 1],
        "data": full_xi[:nbins],
        "ndata": ndata,
        "nbar": float(ndata / BOX_VOLUME),
        "source_total_bins": int(full_xi.size),
    }
    if not np.array_equal(result["s"], XI_FIT_CENTERS):
        raise RuntimeError(f"halo xi centers changed for {tag}")
    return result


def lrg_xi_contract(smin: float, smax: float) -> tuple[np.ndarray, np.ndarray]:
    if np.isclose(smin, XI_FIT_SMIN, rtol=0.0, atol=1.0e-13) and np.isclose(
        smax, XI_FIT_SMAX, rtol=0.0, atol=1.0e-13
    ):
        fit_edges = XI_FIT_EDGES
        fit_centers = XI_FIT_CENTERS
    elif np.isclose(smin, LRG_CONSERVATIVE_XI_SMIN, rtol=0.0, atol=1.0e-13) and np.isclose(
        smax, LRG_CONSERVATIVE_XI_SMAX, rtol=0.0, atol=1.0e-13
    ):
        fit_edges = LRG_CONSERVATIVE_XI_FIT_EDGES
        fit_centers = LRG_CONSERVATIVE_XI_FIT_CENTERS
    elif np.isclose(smin, LRG_EXTENDED_XI_SMIN, rtol=0.0, atol=1.0e-13) and np.isclose(
        smax, LRG_EXTENDED_XI_SMAX, rtol=0.0, atol=1.0e-13
    ):
        fit_edges = LRG_EXTENDED_XI_FIT_EDGES
        fit_centers = LRG_EXTENDED_XI_FIT_CENTERS
    else:
        raise ValueError(f"unsupported LRG xi range: {smin:g}--{smax:g}")
    return np.asarray(fit_edges, dtype="f8"), np.asarray(fit_centers, dtype="f8")


def load_lrg_xi(tag: str, *, smin: float = 30.0, smax: float = 150.0) -> dict[str, Any]:
    fit_edges, fit_centers = lrg_xi_contract(smin, smax)
    path = lrg_xi_path(tag, smin=smin)
    metadata_path = lrg_xi_metadata_path(tag, smin=smin)
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing LRG xi product: {path} / {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "pass" or metadata.get("output_sha256") != sha256_file(path):
        raise RuntimeError(f"invalid LRG xi metadata/hash: {path}")
    with np.load(path, allow_pickle=False) as data:
        full_edges = np.asarray(data["s_edges"], dtype="f8")
        full_centers = np.asarray(data["s"], dtype="f8")
        full_xi = np.asarray(data["xi0"], dtype="f8")
        ndata = int(np.asarray(data["ndata"]).item())
    if not np.array_equal(full_edges[: fit_edges.size], fit_edges):
        raise RuntimeError(f"LRG xi does not contain the requested {smin:g}--{smax:g} shells: {path}")
    nbins = fit_edges.size - 1
    result = {
        "path": path,
        "sha256": metadata["output_sha256"],
        "metadata": metadata,
        "s": full_centers[:nbins],
        "s_edges": full_edges[: nbins + 1],
        "data": full_xi[:nbins],
        "ndata": ndata,
        "nbar": float(ndata / BOX_VOLUME),
        "source_total_bins": int(full_xi.size),
    }
    if not np.array_equal(result["s"], fit_centers):
        raise RuntimeError(f"LRG xi centers changed for {tag}")
    return result


def slice_theory_to_fit_range(
    theory: dict[str, Any],
    *,
    fit_edges: np.ndarray = XI_FIT_EDGES,
) -> dict[str, Any]:
    source_edges = np.asarray(theory["s_edges"], dtype="f8")
    requested_edges = np.asarray(fit_edges, dtype="f8")
    if not np.array_equal(source_edges[: requested_edges.size], requested_edges):
        raise RuntimeError(
            f"theory radial kernel does not begin with the requested shells through smax={requested_edges[-1]:g}"
        )
    nbins = requested_edges.size - 1
    sliced = dict(theory)
    sliced["shell_j0"] = np.asarray(theory["shell_j0"], dtype="f8")[:, :nbins]
    sliced["s_edges"] = requested_edges
    if np.asarray(sliced["shell_j0"]).shape[1] != nbins:
        raise RuntimeError(f"sliced theory kernel does not have the requested {nbins} radial shells")
    return sliced


def _validated_existing(prefix: Path) -> bool:
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")
    if not npz_path.is_file() or not json_path.is_file():
        return False
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    return summary.get("status") == "pass" and summary.get("output_npz_sha256") == sha256_file(npz_path)


def run_one(
    tag: str,
    *,
    tracer: str,
    theory: dict[str, Any],
    theory_cache_path: Path,
    smin: float,
    smax: float,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    overwrite: bool,
) -> None:
    spec = get_spec(tag)
    if tracer == "halo":
        if not (
            np.isclose(smin, XI_FIT_SMIN, rtol=0.0, atol=1.0e-13)
            and np.isclose(smax, XI_FIT_SMAX, rtol=0.0, atol=1.0e-13)
        ):
            raise ValueError("the halo extension remains frozen at s=30--150; only LRG is being changed")
        fit_edges = np.asarray(XI_FIT_EDGES, dtype="f8")
        fit_centers = np.asarray(XI_FIT_CENTERS, dtype="f8")
        prefix = host_xi_fit_prefix(tag)
        measured = load_halo_xi(tag)
        tracer_token = "hodhost_mmin1e13"
        tracer_description = (
            "unit-weight unique occupied HOD hosts with official cleaned CompaSO M>=1e13 Msun/h"
        )
    elif tracer == "lrg":
        fit_edges, fit_centers = lrg_xi_contract(smin, smax)
        if np.isclose(smax, LRG_EXTENDED_XI_SMAX, rtol=0.0, atol=1.0e-13):
            prefix = lrg_xi_extended_fit_prefix(tag)
        elif np.isclose(smin, XI_FIT_SMIN, rtol=0.0, atol=1.0e-13):
            prefix = lrg_xi_fit_prefix(tag)
        else:
            prefix = lrg_xi_conservative_fit_prefix(tag)
        measured = load_lrg_xi(tag, smin=smin, smax=smax)
        tracer_token = "hodmap_lrg"
        tracer_description = "HOD-MAP LRG galaxies"
    else:
        raise ValueError(tracer)
    output_npz = prefix.with_suffix(".npz")
    output_json = prefix.with_suffix(".json")
    if _validated_existing(prefix) and not overwrite:
        print(f"[skip] validated {output_json}", flush=True)
        return
    if not overwrite and (output_npz.exists() or output_json.exists()):
        raise FileExistsError(f"partial/unvalidated {tracer}-xi fit exists: {output_npz} / {output_json}")
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    fit_theory = slice_theory_to_fit_range(theory, fit_edges=fit_edges)
    model = FullDiscreteXiModel(fit_theory, fnl=spec.fnl)
    if np.asarray(model.basis).shape != (fit_centers.size, 3):
        raise RuntimeError(f"unexpected {tracer}-xi model basis shape: {np.asarray(model.basis).shape}")

    started = time.perf_counter()
    theta_initial = np.asarray([2.0, 1.4], dtype="f8")
    covariance_initial_raw = model.covariance(theta_initial, nbar=measured["nbar"])
    covariance_initial, covariance_initial_metadata = regularize_covariance(covariance_initial_raw)
    map_initial = fit_map(model, measured["data"], covariance_initial)

    theta_covariance = theta_initial.copy()
    parameter_tolerance = np.asarray([1.0e-5, 1.0e-4], dtype="f8")
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
        "free_parameters_are_b1_p": tuple(model.names) == ("b1", "p"),
        "fit_edges_match_requested_range_step10": bool(np.array_equal(measured["s_edges"], fit_edges)),
        "fit_centers_match_requested_contract": bool(np.array_equal(measured["s"], fit_centers)),
        "uses_exact_requested_pair_count_bins": int(measured["data"].size) == fit_centers.size,
        "source_is_validated_xi": measured["metadata"].get("status") == "pass",
        "input_tracer_matches_requested": bool(
            tracer_token in str(measured["path"])
            if tracer == "halo"
            else measured["metadata"].get("source_hdf5", measured["metadata"].get("input_hdf5"))
            == spec.path
        ),
        "theory_and_covariance_use_same_shell_kernels": bool(
            np.asarray(model.kernel).shape[1] == fit_centers.size
            and covariance_final.shape == (fit_centers.size, fit_centers.size)
        ),
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
        "task": f"task44_fit_pngbase_{tracer_token}_xi_smin{int(smin)}_smax{int(smax)}",
        "status": status,
        "tag": tag,
        "realization": spec.realization,
        "tracer": tracer_description,
        "geometry": "full periodic real-space cube",
        "boxsize_mpc_h": BOX_SIZE,
        "volume_mpc_h3": BOX_VOLUME,
        "kfund_h_mpc": K_FUND,
        "redshift": REDSHIFT,
        "catalog_baseline_fnl": spec.fnl,
        "fixed_fnl": spec.fnl,
        "free_parameters": list(model.names),
        "priors": {name: list(model.priors[name]) for name in model.names},
        "model": "full-discrete shell-averaged [b1 + fNL*2*delta_c*(b1-p)*alpha(k)]^2 Pm",
        "mean_stochastic_term": "zero at s>0",
        "input": {
            "path": str(measured["path"]),
            "sha256": measured["sha256"],
            "engine": "FCFC, independently validated against pycorr and GPU CuCount",
            "ndata": measured["ndata"],
            "nbar_h3_mpc3": measured["nbar"],
        },
        "fit_range": {
            "smin_mpc_h": smin,
            "smax_mpc_h": float(fit_edges[-1]),
            "edge_convention": f"[low, high); final shell [{fit_edges[-2]:g},{fit_edges[-1]:g})",
            "s_edges_mpc_h": fit_edges.tolist(),
            "nbins": int(measured["data"].size),
            "source_measurement_total_bins_to_350": measured["source_total_bins"],
            "bin_selection": f"independent FCFC pair-count shells through smax={smax:g}; no radial rebinning",
        },
        "theory": {
            "path": str(theory_cache_path),
            "sha256": sha256_file(theory_cache_path),
            "cosmology": "abacus_c000",
            "periodic_discrete_modes": True,
            "radial_kernel": (
                f"exact [{smin:g},{smin + 10:g}), ..., "
                f"[{fit_edges[-2]:g},{fit_edges[-1]:g}) column slice"
            ),
            "survey_window": False,
            "gic_ric_aic": False,
        },
        "covariance": {
            "type": "single-periodic-box analytic Gaussian xi covariance",
            "same_shell_kernels_as_mean": True,
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
        "fit_quality_policy": "chi2/dof and PTE are diagnostics, not pass/fail gates",
        "gates": gates,
        "elapsed_sec": elapsed,
    }
    atomic_savez(
        output_npz,
        coordinate=np.asarray(measured["s"], dtype="f8"),
        coordinate_edges=np.column_stack([fit_edges[:-1], fit_edges[1:]]),
        radial_edges=np.asarray(fit_edges, dtype="f8"),
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
        tracer=np.asarray(tracer_token),
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
        f"[done] {tag} {tracer} xi fixed fNL={spec.fnl:g} s={smin:g}--{smax:g} "
        f"p={p['q50']:.4f} -{p['q50']-p['q16']:.4f}/+{p['q84']-p['q50']:.4f} "
        f"chi2/dof={map_final['chi2']:.2f}/{map_final['dof']} status={status}",
        flush=True,
    )
    if status != "pass":
        print(f"[review] failed gates: {[name for name, value in gates.items() if not value]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", action="append", choices=tuple(CATALOGS))
    parser.add_argument("--tracer", choices=("halo", "lrg", "both"), default="halo")
    parser.add_argument("--smin", type=float, choices=(30.0, 50.0), default=30.0)
    parser.add_argument("--smax", type=float, choices=(150.0, 350.0), default=150.0)
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
    theory_cache_path = build_theory_cache(kmax=5.0, smin=float(args.smin), overwrite=False)
    theory = load_theory(theory_cache_path, smin=float(args.smin))
    tags = list(CATALOGS) if not args.tag else [get_spec(tag).tag for tag in args.tag]
    tracers = ("halo", "lrg") if args.tracer == "both" else (args.tracer,)
    for tracer in tracers:
        for tag in tags:
            tag_offset = 0 if tag == "c300" else 3000
            tracer_offset = 0 if tracer == "halo" else 30_000
            run_one(
                tag,
                tracer=tracer,
                theory=theory,
                theory_cache_path=theory_cache_path,
                smin=float(args.smin),
                smax=float(args.smax),
                nwalkers=int(args.nwalkers),
                nsteps=int(args.nsteps),
                burnin=int(args.burnin),
                seed=(
                    int(args.seed)
                    + tag_offset
                    + tracer_offset
                    + int(args.smin) * 1000
                    + int(args.smax) * 100
                ),
                overwrite=bool(args.overwrite),
            )


if __name__ == "__main__":
    main()
