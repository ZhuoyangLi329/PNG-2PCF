#!/usr/bin/env python3
"""Refit rawbox real-space P0 and xi0 with one final frozen covariance.

This is a versioned MAP-only diagnostic.  It preserves the historical outputs
and quantifies the effect of the old prefit/update-C/no-refit execution order.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

from task43_rawbox_numerics import (
    GaussianMetric,
    RAWBOX_FIT_EDGES,
    exact_lattice_modes,
    frozen_covariance_refit,
)
from task43_rsd_common import OUTPUT_ROOT, PHASES, S_EDGES, atomic_savez, atomic_write_json, sha256_file
from task43_rsd_model import DELTA_C, FullDiscreteRSDModel, build_cache


OUT_DIR = OUTPUT_ROOT / "rawbox" / "xi_rsd_v2" / "stage01_realspace_finalmetric"
OUT_JSON = OUT_DIR / "task43_rsd_rawbox_realspace_finalmetric_v2.json"
OUT_NPZ = OUT_DIR / "task43_rsd_rawbox_realspace_finalmetric_v2.npz"
BOUNDS_LO = np.asarray([-500.0, 0.2], dtype="f8")
BOUNDS_HI = np.asarray([500.0, 10.0], dtype="f8")
OPTIMIZER_STARTS = (
    np.asarray([0.0, 2.5]),
    np.asarray([-100.0, 2.3]),
    np.asarray([100.0, 2.8]),
)
NPHASE = 25


def set_affinity(nthreads: int) -> list[int]:
    if not 1 <= int(nthreads) <= 8:
        raise ValueError("--threads must be in [1, 8]")
    available = sorted(os.sched_getaffinity(0))
    selected = available[: int(nthreads)]
    if len(selected) != int(nthreads):
        raise RuntimeError(f"requested {nthreads} CPUs but only {len(available)} are available")
    os.sched_setaffinity(0, selected)
    return selected


def matching_indices(measured_edges: np.ndarray, fit_edges: np.ndarray) -> np.ndarray:
    indices: list[int] = []
    for edge in np.asarray(fit_edges, dtype="f8"):
        matches = np.flatnonzero(np.all(np.isclose(measured_edges, edge, rtol=0.0, atol=1.0e-12), axis=1))
        if matches.size != 1:
            raise RuntimeError(f"expected exactly one measured bin for {edge.tolist()}")
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype="i4")


def load_pk_x25() -> dict[str, Any]:
    stacks: list[np.ndarray] = []
    hashes: list[str] = []
    nbar: list[float] = []
    k = nmodes = selected = None
    for phase in PHASES:
        path = OUTPUT_ROOT / "rawbox" / "pk" / (
            f"task43_rsd_rawbox_pk0_AbacusSummit_base_c000_{phase}_mmin1p4e13_mesh400.npz"
        )
        metadata_path = path.with_suffix(".json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"missing validated P0 measurement: {path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        digest = sha256_file(path)
        if metadata.get("status") != "pass" or metadata.get("output_sha256") != digest:
            raise RuntimeError(f"unvalidated P0 measurement: {path}")
        with np.load(path, allow_pickle=False) as payload:
            this_edges = np.asarray(payload["k_edges"], dtype="f8")
            this_selected = matching_indices(this_edges, RAWBOX_FIT_EDGES)
            this_k = np.asarray(payload["k"], dtype="f8")[this_selected]
            this_nmodes = np.asarray(payload["nmodes"], dtype="f8")[this_selected]
            if selected is None:
                selected = this_selected
                k = this_k
                nmodes = this_nmodes
            elif not np.array_equal(this_selected, selected):
                raise RuntimeError(f"P0 selected-bin indices changed for {phase}")
            elif not np.allclose(this_k, k, rtol=0.0, atol=1.0e-14):
                raise RuntimeError(f"P0 mean k changed for {phase}")
            elif not np.array_equal(this_nmodes, nmodes):
                raise RuntimeError(f"P0 Nmodes changed for {phase}")
            stacks.append(np.asarray(payload["pk0_real"], dtype="f8")[this_selected])
            nbar.append(float(np.asarray(payload["nbar"]).item()))
        hashes.append(digest)
    return {
        "k": np.asarray(k, dtype="f8"),
        "k_edges": RAWBOX_FIT_EDGES.copy(),
        "nmodes": np.asarray(nmodes, dtype="f8"),
        "pk0_real": np.stack(stacks),
        "nbar": np.asarray(nbar, dtype="f8"),
        "hashes": hashes,
    }


def load_xi_x25() -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[str]]:
    stacks = {key: [] for key in ("xi0_real", "xi2_real", "xi0_rsd", "xi2_rsd")}
    metadata_rows: list[dict[str, Any]] = []
    hashes: list[str] = []
    for phase in PHASES:
        path = OUTPUT_ROOT / "rawbox" / "summary" / (
            f"task43_rsd_rawbox_AbacusSummit_base_c000_{phase}_mmin1p4e13_clustering.npz"
        )
        metadata_path = path.with_suffix(".json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"missing validated xi measurement: {path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        digest = sha256_file(path)
        if metadata.get("status") != "pass" or metadata.get("output_sha256") != digest:
            raise RuntimeError(f"unvalidated xi measurement: {path}")
        with np.load(path, allow_pickle=False) as payload:
            if not np.array_equal(np.asarray(payload["s_edges"], dtype="f8"), S_EDGES):
                raise RuntimeError(f"xi separation bins changed for {phase}")
            for key in stacks:
                stacks[key].append(np.asarray(payload[key], dtype="f8"))
            metadata["nbar_h3_mpc3_from_npz"] = float(np.asarray(payload["nbar"]).item())
        metadata_rows.append(metadata)
        hashes.append(digest)
    return {key: np.stack(value) for key, value in stacks.items()}, metadata_rows, hashes


def fit_branch(
    *,
    name: str,
    data: np.ndarray,
    evaluate: Callable[[np.ndarray], np.ndarray],
    covariance_builder: Callable[[float], np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the historical prefit and the corrected final-metric refit."""

    data_vector = np.asarray(data, dtype="f8")

    def fit_fixed_covariance(covariance: np.ndarray):
        metric = GaussianMetric(covariance)

        def residual(theta: np.ndarray) -> np.ndarray:
            return metric.residual(data_vector - evaluate(theta))

        solutions = [
            least_squares(
                residual,
                start,
                bounds=(BOUNDS_LO, BOUNDS_HI),
                max_nfev=3000,
                xtol=1.0e-12,
                ftol=1.0e-12,
                gtol=1.0e-12,
            )
            for start in OPTIMIZER_STARTS
        ]
        return min(solutions, key=lambda result: float(result.fun @ result.fun))

    refit = frozen_covariance_refit(
        covariance_builder,
        fit_fixed_covariance,
        initial_b1=2.5,
        b1_index=1,
    )
    theta_prefit = np.asarray(refit["prefit_theta"], dtype="f8")
    theta_final = np.asarray(refit["final_theta"], dtype="f8")
    prediction_prefit = np.asarray(evaluate(theta_prefit), dtype="f8")
    prediction_final = np.asarray(evaluate(theta_final), dtype="f8")
    residual_prefit = data_vector - prediction_prefit
    residual_final = data_vector - prediction_final
    metric: GaussianMetric = refit["metric"]
    chi2_prefit_in_final_metric = metric.chi2(residual_prefit)
    chi2_single = metric.chi2(residual_final)
    dof = int(data_vector.size - theta_final.size)
    chi2_mean = float(NPHASE * chi2_single)
    objective_improvement = float(chi2_prefit_in_final_metric - chi2_single)
    tolerance = 1.0e-10 * max(1.0, abs(chi2_prefit_in_final_metric))
    if objective_improvement < -tolerance:
        raise RuntimeError(f"{name}: final-metric refit increased chi-square")

    summary = {
        "name": name,
        "ndata": int(data_vector.size),
        "dof": dof,
        "initial_covariance_b1": 2.5,
        "final_frozen_covariance_b1": float(refit["covariance_b1"]),
        "prefit_map": {"fNL": float(theta_prefit[0]), "b1": float(theta_prefit[1])},
        "final_map": {"fNL": float(theta_final[0]), "b1": float(theta_final[1])},
        "map_shift": {
            "fNL": float(theta_final[0] - theta_prefit[0]),
            "b1": float(theta_final[1] - theta_prefit[1]),
        },
        "chi2_prefit_in_final_metric": float(chi2_prefit_in_final_metric),
        "chi2_single_final": float(chi2_single),
        "chi2_mean_final": chi2_mean,
        "objective_improvement": max(objective_improvement, 0.0),
        "pte_mean_final": float(chi2_distribution.sf(chi2_mean, dof)),
        "optimizer": {
            "prefit_success": bool(refit["prefit"].success),
            "final_success": bool(refit["final_fit"].success),
            "final_message": str(refit["final_fit"].message),
        },
        "metric": {
            "definition": "C fixed at the prefit b1, then MAP/prediction/chi2 all recomputed under that same C",
            "correlation_eigenvalue_min": float(metric.eigenvalues[0]),
            "correlation_eigenvalue_max": float(metric.eigenvalues[-1]),
            "unit_normalized_cholesky": True,
            "automatic_eigenvalue_floor": False,
        },
    }
    arrays = {
        "data": data_vector,
        "covariance": metric.covariance,
        "theta_prefit": theta_prefit,
        "theta_final": theta_final,
        "prediction_prefit": prediction_prefit,
        "prediction_final": prediction_final,
        "residual_prefit": residual_prefit,
        "residual_final": residual_final,
    }
    return summary, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if OUT_JSON.exists() or OUT_NPZ.exists():
        raise FileExistsError(f"immutable versioned output already exists: {OUT_JSON} / {OUT_NPZ}")

    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))
    cache_path = build_cache(
        zeff=0.725,
        boxsize=2000.0,
        kmax=3.0,
        ells=(0, 2),
        cosmology="abacus_c000",
    )
    exact = FullDiscreteRSDModel(cache_path, nmu=64)

    pk = load_pk_x25()
    pk_modes = exact_lattice_modes(float(exact.boxsize), RAWBOX_FIT_EDGES)
    if not np.array_equal(pk_modes.counts, np.asarray(pk["nmodes"], dtype="f8")):
        raise RuntimeError("P model lattice counts differ from measured Nmodes")
    if not np.allclose(pk_modes.k_mean, np.asarray(pk["k"], dtype="f8"), rtol=0.0, atol=1.0e-14):
        raise RuntimeError("P model lattice mean k differs from the measurement")
    pk_model_power = np.interp(np.log(pk_modes.k), np.log(exact.k_eff), exact.pk_dd)
    pk_model_alpha = np.interp(np.log(pk_modes.k), np.log(exact.k_eff), exact.alpha)
    p_data = np.mean(np.asarray(pk["pk0_real"], dtype="f8"), axis=0)
    p_nbar = float(np.mean(np.asarray(pk["nbar"], dtype="f8")))

    def evaluate_p(theta: np.ndarray) -> np.ndarray:
        fnl, b1 = map(float, np.asarray(theta, dtype="f8")[:2])
        amplitude = b1 + fnl * 2.0 * DELTA_C * (b1 - 1.0) * pk_model_alpha
        signal = pk_model_power * amplitude**2
        return np.bincount(
            pk_modes.bin_id,
            weights=signal,
            minlength=RAWBOX_FIT_EDGES.shape[0],
        ) / pk_modes.counts

    def covariance_p(b1: float) -> np.ndarray:
        total2 = (float(b1) ** 2 * pk_model_power + 1.0 / p_nbar) ** 2
        summed = np.bincount(
            pk_modes.bin_id,
            weights=total2,
            minlength=RAWBOX_FIT_EDGES.shape[0],
        )
        return np.diag(2.0 * summed / pk_modes.counts**2)

    summary_p, arrays_p = fit_branch(
        name="realspace_p0",
        data=p_data,
        evaluate=evaluate_p,
        covariance_builder=covariance_p,
    )

    xi, xi_metadata, xi_hashes = load_xi_x25()
    xi_nbar = float(np.mean([row["nbar_h3_mpc3_from_npz"] for row in xi_metadata]))
    xi_data_full = np.mean(np.asarray(xi["xi0_real"], dtype="f8"), axis=0)
    kernel0 = np.asarray(exact.kernels[0], dtype="f8")
    degeneracy = np.asarray(exact.g_nz, dtype="f8")
    pk_dd = np.asarray(exact.pk_dd, dtype="f8")
    alpha = np.asarray(exact.alpha, dtype="f8")
    volume = float(exact.volume)
    projection = degeneracy[:, None] * pk_dd[:, None] * kernel0 / volume

    def evaluate_xi_full(theta: np.ndarray) -> np.ndarray:
        fnl, b1 = map(float, np.asarray(theta, dtype="f8")[:2])
        amplitude = b1 + fnl * 2.0 * DELTA_C * (b1 - 1.0) * alpha
        return amplitude**2 @ projection

    def covariance_xi_full(b1: float) -> np.ndarray:
        total2 = (float(b1) ** 2 * pk_dd + 1.0 / xi_nbar) ** 2
        weight = 2.0 * degeneracy * total2 / volume**2
        return kernel0.T @ (weight[:, None] * kernel0)

    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    xi_summaries: dict[str, Any] = {}
    xi_arrays: dict[str, np.ndarray] = {}
    for smin in (50.0, 120.0):
        mask = centers >= smin
        ids = np.flatnonzero(mask)

        def evaluate_xi(theta: np.ndarray, selected: np.ndarray = ids) -> np.ndarray:
            return evaluate_xi_full(theta)[selected]

        def covariance_xi(b1: float, selected: np.ndarray = ids) -> np.ndarray:
            covariance = covariance_xi_full(b1)
            return covariance[np.ix_(selected, selected)]

        key = f"realspace_xi0_smin{int(smin)}"
        summary, arrays = fit_branch(
            name=key,
            data=xi_data_full[ids],
            evaluate=evaluate_xi,
            covariance_builder=covariance_xi,
        )
        summary["smin_mpc_h"] = smin
        summary["s_bins_mpc_h"] = centers[ids].tolist()
        xi_summaries[key] = summary
        for array_name, value in arrays.items():
            xi_arrays[f"{key}_{array_name}"] = value

    output_arrays: dict[str, np.ndarray] = {
        "pk_k": np.asarray(pk["k"], dtype="f8"),
        "pk_k_edges": np.asarray(pk["k_edges"], dtype="f8"),
        "pk_nmodes": np.asarray(pk["nmodes"], dtype="f8"),
        "xi_s_edges": np.asarray(S_EDGES, dtype="f8"),
    }
    output_arrays.update({f"realspace_p0_{key}": value for key, value in arrays_p.items()})
    output_arrays.update(xi_arrays)
    atomic_savez(OUT_NPZ, **output_arrays)

    audit = {
        "task": "task43_rsd_rawbox_realspace_finalmetric_v2",
        "status": "complete",
        "scientific_status": "diagnostic_only",
        "scope": "MAP-only measurement of the final-frozen-covariance refit effect; historical outputs remain immutable",
        "frozen_p_contract": {
            "model": "existing exact parent-mode real-space P0 model; no P template or measurement changes",
            "measurement_sha256": list(pk["hashes"]),
            "nmodes_exact_gate": True,
            "mean_k_gate": True,
        },
        "inputs": {
            "theory_cache": str(cache_path),
            "theory_cache_sha256": sha256_file(cache_path),
            "xi_measurement_sha256": xi_hashes,
            "nphase": NPHASE,
        },
        "results": {"realspace_p0": summary_p, **xi_summaries},
        "output_npz": str(OUT_NPZ),
        "output_npz_sha256": sha256_file(OUT_NPZ),
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    atomic_write_json(OUT_JSON, audit)
    print(json.dumps(audit["results"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
