#!/usr/bin/env python3
"""Targeted likelihood diagnostics for PNG-base P0--xi0 consistency.

The authoritative P0 fit uses the native periodic-box selection: all 49
contiguous 0.002-wide bins from [0.003, 0.005) through [0.099, 0.101).  This
script uses that fit only to propagate its residual constant stochastic power
into the xi0 Gaussian covariance; the xi0 mean retains the correct zero
contact term at positive separation.

Each variant uses a self-consistent fixed covariance and a 64x10000 emcee chain
by default.  All outputs live in a separate ``consistency_diagnostics`` tree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_CODE_DIR = PROJECT_ROOT / "codes" / "task43"
TASK44_CODE_DIR = PROJECT_ROOT / "codes" / "task44"
for _directory in (TASK43_CODE_DIR, TASK44_CODE_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from task44_fit_validation import make_fit_contract, reusable_fit
from task44_fit_pngbase_hodmap_rawbox import (  # noqa: E402
    ExactPeriodicPkModel,
    FullDiscreteXiFixedPModel,
    FullDiscreteXiModel,
    fit_map,
    load_pk_data,
    load_theory,
    load_xi_data,
    regularize_covariance,
    run_mcmc,
)
from task44_pngbase_hodmap_rawbox_common import (  # noqa: E402
    BOX_SIZE,
    BOX_VOLUME,
    K_FUND,
    OUTPUT_ROOT,
    PK_PRIMARY_FIT_EDGES,
    SN0_SCALE,
    atomic_savez,
    atomic_write_json,
    fit_prefix,
    get_spec,
    set_cpu_affinity,
    sha256_file,
    theory_path,
    to_jsonable,
)


DIAGNOSTIC_ROOT = OUTPUT_ROOT / "consistency_diagnostics"
CONTIGUOUS_EDGES = PK_PRIMARY_FIT_EDGES


class FullDiscreteXiResidualStochasticCovModel(FullDiscreteXiModel):
    """xi mean with no contact term and covariance with residual stochastic P."""

    def __init__(self, theory: dict[str, Any], *, fnl: float, residual_stochastic: float) -> None:
        super().__init__(theory, fnl=fnl)
        self.residual_stochastic = float(residual_stochastic)

    def covariance(self, theta: np.ndarray, *, nbar: float) -> np.ndarray:
        total2 = (
            self.signal_modes(theta)
            + 1.0 / float(nbar)
            + self.residual_stochastic
        ) ** 2
        weighted_kernel = (self.g * total2)[:, None] * self.kernel
        covariance = 2.0 * (self.kernel.T @ weighted_kernel) / self.volume**2
        return 0.5 * (covariance + covariance.T)


class FullDiscreteXiFixedPResidualStochasticCovModel(FullDiscreteXiFixedPModel):
    """Fixed-p xi mean and covariance including residual stochastic power."""

    def __init__(self, theory: dict[str, Any], *, fixed_p: float, residual_stochastic: float) -> None:
        super().__init__(theory, fixed_p=fixed_p)
        self.residual_stochastic = float(residual_stochastic)

    def covariance(self, theta: np.ndarray, *, nbar: float) -> np.ndarray:
        total2 = (
            self.signal_modes(theta)
            + 1.0 / float(nbar)
            + self.residual_stochastic
        ) ** 2
        weighted_kernel = (self.g * total2)[:, None] * self.kernel
        covariance = 2.0 * (self.kernel.T @ weighted_kernel) / self.volume**2
        return 0.5 * (covariance + covariance.T)


def primary_pk_summary(tag: str) -> tuple[Path, dict[str, Any]]:
    path = fit_prefix(tag, "pk", kmin_edge=0.003).with_suffix(".json")
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise RuntimeError(f"primary P0 fit is not validated: {path}")
    return path, payload


def variant_prefix(tag: str, kind: str, *, smin: float | None = None) -> Path:
    if kind == "pk_contiguous49":
        label = "pk0_contiguous49_kcentermax0p100_fixedfnl_freep"
    elif kind == "xi_stochastic_cov":
        if smin not in (30.0, 50.0):
            raise ValueError(f"unsupported smin={smin}")
        label = (
            f"xi0_smin{int(smin)}_smax350_pstochcov_"
            "from_periodiccontiguous49_primarypk_fixedfnl_freep"
        )
    elif kind == "xi_fixedp1_stochastic_cov":
        if smin not in (30.0, 50.0):
            raise ValueError(f"unsupported smin={smin}")
        label = (
            f"xi0_smin{int(smin)}_smax350_pstochcov_"
            "from_periodiccontiguous49_primarypk_fixedp1_freefnl"
        )
    else:
        raise ValueError(kind)
    return DIAGNOSTIC_ROOT / "fits" / tag / label / f"task44_pngbase_{tag}_{label}"


def fixed_point(
    model: Any,
    data: np.ndarray,
    *,
    nbar: float,
    initial: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, dict[str, Any], list[dict[str, Any]], bool]:
    theta_cov = np.asarray(initial, dtype="f8")
    tolerance = np.asarray(
        [1.0e-4 if name == "p" else (1.0e-3 if name == "fnl" else 1.0e-5) for name in model.names],
        dtype="f8",
    )
    history: list[dict[str, Any]] = []
    converged = False
    for iteration in range(12):
        covariance, covariance_meta = regularize_covariance(model.covariance(theta_cov, nbar=nbar))
        map_result = fit_map(model, data, covariance)
        theta_fit = np.asarray([map_result["theta"][name] for name in model.names], dtype="f8")
        delta = theta_fit - theta_cov
        history.append(
            {
                "iteration": iteration,
                "covariance_fiducial": {
                    name: float(value) for name, value in zip(model.names, theta_cov, strict=True)
                },
                "map": map_result,
                "delta": {name: float(value) for name, value in zip(model.names, delta, strict=True)},
                "covariance_diagnostics": covariance_meta,
            }
        )
        theta_cov = theta_fit
        if iteration >= 1 and np.all(np.abs(delta) < tolerance):
            converged = True
            break
    covariance, covariance_meta = regularize_covariance(model.covariance(theta_cov, nbar=nbar))
    map_result = fit_map(model, data, covariance)
    return theta_cov, map_result, covariance, covariance_meta, history, converged


def variant_cache_contract(*, tag, model, measured, theory, theory_cache, nwalkers, nsteps,
                           burnin, seed, source_path=None):
    """在运行条件 covariance 变体前验证真实输入与运行合同，不信任旧输出文件名。"""
    inputs = dict(measurement=measured["path"], theory=theory_cache)
    if source_path is not None:
        inputs["stochastic_source_fit"] = source_path
    return make_fit_contract(
        config=dict(tag=tag, model_class=type(model).__name__, names=model.names, priors=model.priors,
                    nbar=measured["nbar"], nwalkers=nwalkers, nsteps=nsteps, burnin=burnin, seed=seed,
                    fixed_p=getattr(model, "fixed_p", None), fixed_fnl=getattr(model, "fnl", None),
                    residual_stochastic=getattr(model, "residual_stochastic", None)),
        inputs=inputs,
        code=[Path(__file__), Path(__file__).with_name("task44_fit_pngbase_hodmap_rawbox.py"),
              Path(__file__).with_name("task44_pngbase_hodmap_rawbox_common.py"),
              Path(__file__).with_name("task44_fit_validation.py")], theory=theory,
    )


def write_fit(
    *,
    prefix: Path,
    summary: dict[str, Any],
    coordinate: np.ndarray,
    coordinate_edges: np.ndarray,
    data: np.ndarray,
    prediction: np.ndarray,
    covariance: np.ndarray,
    chain: np.ndarray,
    logp: np.ndarray,
    parameter_names: tuple[str, ...],
) -> None:
    output_npz = prefix.with_suffix(".npz")
    output_json = prefix.with_suffix(".json")
    atomic_savez(
        output_npz,
        coordinate=np.asarray(coordinate, dtype="f8"),
        coordinate_edges=np.asarray(coordinate_edges, dtype="f8"),
        data=np.asarray(data, dtype="f8"),
        prediction_map=np.asarray(prediction, dtype="f8"),
        covariance_final=np.asarray(covariance, dtype="f8"),
        residual_sigma=(np.asarray(data, dtype="f8") - np.asarray(prediction, dtype="f8"))
        / np.sqrt(np.diag(covariance)),
        chain_final_by_step=np.asarray(chain, dtype="f8"),
        logp_final_by_step=np.asarray(logp, dtype="f8"),
        parameter_names=np.asarray(parameter_names),
        summary_json=np.asarray(json.dumps(to_jsonable(summary), sort_keys=True)),
    )
    summary["output_npz"] = str(output_npz)
    summary["output_npz_sha256"] = sha256_file(output_npz)
    atomic_write_json(output_json, summary)


def run_pk_contiguous(
    tag: str,
    *,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    overwrite: bool,
) -> None:
    prefix = variant_prefix(tag, "pk_contiguous49")
    started = time.perf_counter()
    theory_cache = theory_path(5.0, smin=30.0)
    theory = load_theory(theory_cache, smin=30.0)
    measured = load_pk_data(tag, CONTIGUOUS_EDGES)
    spec = get_spec(tag)
    model = ExactPeriodicPkModel(theory, CONTIGUOUS_EDGES, fnl=spec.fnl)
    if not np.array_equal(model.counts, measured["nmodes"]):
        raise RuntimeError(f"contiguous P0 mode counts mismatch for {tag}")
    if not np.allclose(model.k_mean, measured["k"], rtol=0.0, atol=1.0e-14):
        raise RuntimeError(f"contiguous P0 mean-k mismatch for {tag}")
    cache_contract = variant_cache_contract(
        tag=tag, model=model, measured=measured, theory=theory, theory_cache=theory_cache,
        nwalkers=nwalkers, nsteps=nsteps, burnin=burnin, seed=seed,
    )
    if reusable_fit(prefix, cache_contract, overwrite=overwrite):
        print(f"[skip verified] {prefix}", flush=True)
        return
    _fiducial, map_result, covariance, covariance_meta, history, fixed_point_converged = fixed_point(
        model,
        measured["data"],
        nbar=measured["nbar"],
        initial=np.asarray([2.0, 1.4, 0.0], dtype="f8"),
    )
    mcmc, chain, logp = run_mcmc(
        model,
        measured["data"],
        covariance,
        map_result,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burnin=burnin,
        seed=seed,
    )
    prediction = np.asarray(model.evaluate(np.asarray([map_result["theta"][name] for name in model.names])), dtype="f8")
    gates = {
        "exact_mode_counts_match": True,
        "exact_mean_k_matches": True,
        "all_49_bins_present": int(measured["data"].size) == 49,
        "first_edge_is_0p003": bool(np.isclose(CONTIGUOUS_EDGES[0, 0], 0.003, rtol=0.0, atol=1.0e-14)),
        "last_bin_center_is_0p100": bool(np.isclose(np.mean(CONTIGUOUS_EDGES[-1]), 0.100, rtol=0.0, atol=1.0e-14)),
        "fixed_point_converged": fixed_point_converged,
        "map_optimizer_success": bool(map_result["success"]),
        "map_not_at_boundary": bool(not map_result["at_parameter_boundary"]),
        "all_mcmc_gates": bool(all(mcmc["gates"].values())),
    }
    summary = {
        "cache_contract": cache_contract,
        "task": "task44_pngbase_pk_xi_consistency_pk_contiguous49",
        "status": "pass" if all(gates.values()) else "review",
        "tag": tag,
        "catalog_baseline_fnl": spec.fnl,
        "fixed_fnl": spec.fnl,
        "free_parameters": list(model.names),
        "geometry": "full periodic real-space cube",
        "boxsize_mpc_h": BOX_SIZE,
        "volume_mpc_h3": BOX_VOLUME,
        "kfund_h_mpc": K_FUND,
        "fit_range": {
            "edge_min_h_mpc": float(CONTIGUOUS_EDGES[0, 0]),
            "edge_max_h_mpc": float(CONTIGUOUS_EDGES[-1, 1]),
            "last_bin_center_h_mpc": float(np.mean(CONTIGUOUS_EDGES[-1])),
            "nbins": int(measured["data"].size),
            "selection": "all contiguous measured 0.002-wide bins",
        },
        "input": {"path": str(measured["path"]), "sha256": measured["sha256"], "nbar": measured["nbar"]},
        "theory": {"path": str(theory_cache), "sha256": sha256_file(theory_cache)},
        "covariance": {
            "type": "single-periodic-box analytic Gaussian with fitted residual stochastic power",
            "fixed_point_history": history,
            "final_diagnostics": covariance_meta,
        },
        "map": map_result,
        "mcmc": mcmc,
        "gates": gates,
        "elapsed_sec": time.perf_counter() - started,
    }
    write_fit(
        prefix=prefix,
        summary=summary,
        coordinate=measured["k"],
        coordinate_edges=measured["k_edges"],
        data=measured["data"],
        prediction=prediction,
        covariance=covariance,
        chain=chain,
        logp=logp,
        parameter_names=tuple(model.names),
    )
    row = mcmc["posterior"]["p"]
    print(
        f"[done] {tag} contiguous49 P0 p={row['q50']:.5f} "
        f"-{row['q50'] - row['q16']:.5f}/+{row['q84'] - row['q50']:.5f} "
        f"chi2/dof={map_result['chi2']:.2f}/{map_result['dof']} status={summary['status']}",
        flush=True,
    )


def run_xi_stochastic_cov(
    tag: str,
    *,
    smin: float,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    overwrite: bool,
) -> None:
    prefix = variant_prefix(tag, "xi_stochastic_cov", smin=smin)
    started = time.perf_counter()
    source_path, source = primary_pk_summary(tag)
    residual_stochastic = float(source["map"]["theta"]["sn0"]) * SN0_SCALE
    theory_cache = theory_path(5.0, smin=smin)
    theory = load_theory(theory_cache, smin=smin)
    measured = load_xi_data(tag, smin=smin)
    spec = get_spec(tag)
    model = FullDiscreteXiResidualStochasticCovModel(
        theory,
        fnl=spec.fnl,
        residual_stochastic=residual_stochastic,
    )
    poisson_model = FullDiscreteXiModel(theory, fnl=spec.fnl)
    cache_contract = variant_cache_contract(
        tag=tag, model=model, measured=measured, theory=theory, theory_cache=theory_cache,
        nwalkers=nwalkers, nsteps=nsteps, burnin=burnin, seed=seed, source_path=source_path,
    )
    if reusable_fit(prefix, cache_contract, overwrite=overwrite):
        print(f"[skip verified] {prefix}", flush=True)
        return
    _fiducial, map_result, covariance, covariance_meta, history, fixed_point_converged = fixed_point(
        model,
        measured["data"],
        nbar=measured["nbar"],
        initial=np.asarray([2.0, 1.4], dtype="f8"),
    )
    mcmc, chain, logp = run_mcmc(
        model,
        measured["data"],
        covariance,
        map_result,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burnin=burnin,
        seed=seed,
    )
    theta_map = np.asarray([map_result["theta"][name] for name in model.names], dtype="f8")
    prediction = np.asarray(model.evaluate(theta_map), dtype="f8")
    poisson_covariance, _ = regularize_covariance(poisson_model.covariance(theta_map, nbar=measured["nbar"]))
    sigma_ratio = np.sqrt(np.diag(covariance) / np.diag(poisson_covariance))
    gates = {
        "source_primary_pk_status_pass": source.get("status") == "pass",
        "residual_stochastic_is_positive": residual_stochastic > 0.0,
        "xi_mean_unchanged_by_stochastic_covariance": True,
        "fixed_point_converged": fixed_point_converged,
        "map_optimizer_success": bool(map_result["success"]),
        "map_not_at_boundary": bool(not map_result["at_parameter_boundary"]),
        "all_mcmc_gates": bool(all(mcmc["gates"].values())),
    }
    summary = {
        "cache_contract": cache_contract,
        "task": "task44_pngbase_pk_xi_consistency_xi_stochastic_covariance",
        "status": "pass" if all(gates.values()) else "review",
        "tag": tag,
        "catalog_baseline_fnl": spec.fnl,
        "fixed_fnl": spec.fnl,
        "free_parameters": list(model.names),
        "geometry": "full periodic real-space cube",
        "boxsize_mpc_h": BOX_SIZE,
        "volume_mpc_h3": BOX_VOLUME,
        "kfund_h_mpc": K_FUND,
        "fit_range": {"smin_mpc_h": smin, "smax_mpc_h": 350.0, "nbins": int(measured["data"].size)},
        "input": {"path": str(measured["path"]), "sha256": measured["sha256"], "nbar": measured["nbar"]},
        "theory": {"path": str(theory_cache), "sha256": sha256_file(theory_cache)},
        "residual_stochastic": {
            "power_mpc_h3": residual_stochastic,
            "sn0_units_1e4": residual_stochastic / SN0_SCALE,
            "source_primary_pk_json": str(source_path),
            "source_primary_pk_json_sha256": sha256_file(source_path),
            "mean_policy": "zero contact term for s>0",
            "covariance_policy": "add residual stochastic to P_total before squaring",
        },
        "covariance": {
            "type": "single-periodic-box analytic Gaussian including Poisson plus P0 residual stochastic",
            "fixed_point_history": history,
            "final_diagnostics": covariance_meta,
            "sigma_ratio_vs_poisson_only": {
                "min": float(np.min(sigma_ratio)),
                "median": float(np.median(sigma_ratio)),
                "max": float(np.max(sigma_ratio)),
            },
        },
        "map": map_result,
        "mcmc": mcmc,
        "gates": gates,
        "elapsed_sec": time.perf_counter() - started,
    }
    write_fit(
        prefix=prefix,
        summary=summary,
        coordinate=measured["s"],
        coordinate_edges=np.column_stack([measured["s_edges"][:-1], measured["s_edges"][1:]]),
        data=measured["data"],
        prediction=prediction,
        covariance=covariance,
        chain=chain,
        logp=logp,
        parameter_names=tuple(model.names),
    )
    row = mcmc["posterior"]["p"]
    print(
        f"[done] {tag} xi smin={smin:g} +Pstoch covariance p={row['q50']:.5f} "
        f"-{row['q50'] - row['q16']:.5f}/+{row['q84'] - row['q50']:.5f} "
        f"chi2/dof={map_result['chi2']:.2f}/{map_result['dof']} status={summary['status']}",
        flush=True,
    )


def run_xi_fixedp1_stochastic_cov(
    tag: str,
    *,
    smin: float,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    overwrite: bool,
) -> None:
    prefix = variant_prefix(tag, "xi_fixedp1_stochastic_cov", smin=smin)
    started = time.perf_counter()
    source_path, source = primary_pk_summary(tag)
    residual_stochastic = float(source["map"]["theta"]["sn0"]) * SN0_SCALE
    theory_cache = theory_path(5.0, smin=smin)
    theory = load_theory(theory_cache, smin=smin)
    measured = load_xi_data(tag, smin=smin)
    spec = get_spec(tag)
    model = FullDiscreteXiFixedPResidualStochasticCovModel(
        theory,
        fixed_p=1.0,
        residual_stochastic=residual_stochastic,
    )
    cache_contract = variant_cache_contract(
        tag=tag, model=model, measured=measured, theory=theory, theory_cache=theory_cache,
        nwalkers=nwalkers, nsteps=nsteps, burnin=burnin, seed=seed, source_path=source_path,
    )
    if reusable_fit(prefix, cache_contract, overwrite=overwrite):
        print(f"[skip verified] {prefix}", flush=True)
        return
    _fiducial, map_result, covariance, covariance_meta, history, fixed_point_converged = fixed_point(
        model,
        measured["data"],
        nbar=measured["nbar"],
        initial=np.asarray([2.0, spec.fnl], dtype="f8"),
    )
    mcmc, chain, logp = run_mcmc(
        model,
        measured["data"],
        covariance,
        map_result,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burnin=burnin,
        seed=seed,
    )
    theta_map = np.asarray([map_result["theta"][name] for name in model.names], dtype="f8")
    prediction = np.asarray(model.evaluate(theta_map), dtype="f8")
    gates = {
        "source_primary_pk_status_pass": source.get("status") == "pass",
        "fixed_p_is_1": True,
        "residual_stochastic_is_positive": residual_stochastic > 0.0,
        "xi_mean_has_zero_stochastic_contact_at_s_positive": True,
        "fixed_point_converged": fixed_point_converged,
        "map_optimizer_success": bool(map_result["success"]),
        "map_not_at_boundary": bool(not map_result["at_parameter_boundary"]),
        "all_mcmc_gates": bool(all(mcmc["gates"].values())),
    }
    summary = {
        "cache_contract": cache_contract,
        "task": "task44_pngbase_pk_xi_consistency_xi_fixedp1_stochastic_covariance",
        "status": "pass" if all(gates.values()) else "review",
        "tag": tag,
        "catalog_baseline_fnl": spec.fnl,
        "fixed_p": 1.0,
        "free_parameters": list(model.names),
        "geometry": "full periodic real-space cube",
        "boxsize_mpc_h": BOX_SIZE,
        "volume_mpc_h3": BOX_VOLUME,
        "kfund_h_mpc": K_FUND,
        "fit_range": {"smin_mpc_h": smin, "smax_mpc_h": 350.0, "nbins": int(measured["data"].size)},
        "input": {"path": str(measured["path"]), "sha256": measured["sha256"], "nbar": measured["nbar"]},
        "theory": {"path": str(theory_cache), "sha256": sha256_file(theory_cache)},
        "residual_stochastic": {
            "power_mpc_h3": residual_stochastic,
            "sn0_units_1e4": residual_stochastic / SN0_SCALE,
            "source_primary_pk_json": str(source_path),
            "source_primary_pk_json_sha256": sha256_file(source_path),
            "mean_policy": "zero contact term for s>0",
            "covariance_policy": "add residual stochastic to P_total before squaring",
        },
        "covariance": {
            "type": "single-periodic-box analytic Gaussian including Poisson plus P0 residual stochastic",
            "fixed_point_history": history,
            "final_diagnostics": covariance_meta,
        },
        "map": map_result,
        "mcmc": mcmc,
        "gates": gates,
        "elapsed_sec": time.perf_counter() - started,
    }
    write_fit(
        prefix=prefix,
        summary=summary,
        coordinate=measured["s"],
        coordinate_edges=np.column_stack([measured["s_edges"][:-1], measured["s_edges"][1:]]),
        data=measured["data"],
        prediction=prediction,
        covariance=covariance,
        chain=chain,
        logp=logp,
        parameter_names=tuple(model.names),
    )
    row = mcmc["posterior"]["fnl"]
    print(
        f"[done] {tag} xi smin={smin:g} fixed-p1 +Pstoch covariance fNL={row['q50']:.4f} "
        f"-{row['q50'] - row['q16']:.4f}/+{row['q84'] - row['q50']:.4f} "
        f"chi2/dof={map_result['chi2']:.2f}/{map_result['dof']} status={summary['status']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", action="append", choices=("c300", "c302"))
    parser.add_argument("--variant", choices=("all", "pk", "xi", "fixedp1"), default="all")
    parser.add_argument("--smin", action="append", type=float)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=10_000)
    parser.add_argument("--burnin", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.burnin >= args.nsteps:
        raise ValueError("burnin must be smaller than nsteps")
    set_cpu_affinity(args.threads)
    tags = args.tag or ["c300", "c302"]
    smins = args.smin or [30.0]
    if any(value not in (30.0, 50.0) for value in smins):
        raise ValueError("audited smin choices are 30 or 50 Mpc/h")
    for tag in tags:
        tag_offset = 0 if tag == "c300" else 3000
        # The diagnostic P0 duplicate is retained only as an explicit
        # compatibility command.  The normal ``all`` workflow uses the
        # canonical fit in FIT_DIR and does not create a second primary.
        if args.variant == "pk":
            run_pk_contiguous(
                tag,
                nwalkers=args.nwalkers,
                nsteps=args.nsteps,
                burnin=args.burnin,
                seed=args.seed + tag_offset + 80_000,
                overwrite=args.overwrite,
            )
        if args.variant in ("all", "xi"):
            for smin in smins:
                run_xi_stochastic_cov(
                    tag,
                    smin=smin,
                    nwalkers=args.nwalkers,
                    nsteps=args.nsteps,
                    burnin=args.burnin,
                    seed=args.seed + tag_offset + 90_000 + int(smin),
                    overwrite=args.overwrite,
                )
        if args.variant in ("all", "fixedp1"):
            for smin in smins:
                run_xi_fixedp1_stochastic_cov(
                    tag,
                    smin=smin,
                    nwalkers=args.nwalkers,
                    nsteps=args.nsteps,
                    burnin=args.burnin,
                    seed=args.seed + tag_offset + 100_000 + int(smin),
                    overwrite=args.overwrite,
                )


if __name__ == "__main__":
    main()
