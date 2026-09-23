#!/usr/bin/env python3
"""Run the Task 4.3 xi-only linear/operator closure without fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from task43_rawbox_numerics import GaussianMetric
from task43_rsd_common import OUTPUT_ROOT, S_EDGES, atomic_savez, atomic_write_json, sha256_file
from task43_rsd_model import build_cache
from task43_rsd_rawbox_realspace_finalmetric_v2 import load_xi_x25, set_affinity
from task43_xi_linear_rsd import XiLinearRSDModel


OUT_DIR = OUTPUT_ROOT / "rawbox" / "xi_rsd_v2" / "stage02_linear_operator_closure"
OUT_JSON = OUT_DIR / "task43_xi_linear_operator_closure_v2.json"
OUT_NPZ = OUT_DIR / "task43_xi_linear_operator_closure_v2.npz"
NPHASE = 25
NUMERICAL_BUDGET = 0.01
P_REFERENCE = OUTPUT_ROOT / "rawbox" / "comparison" / (
    "task43_rsd_rawbox_x25_pk0_kmin0p003_vs_xi0_smin50_l0only_longchain"
)
PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")


def selected_indices() -> dict[str, np.ndarray]:
    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    nbin = centers.size
    xi0 = np.flatnonzero(centers >= 50.0)
    xi2 = nbin + np.flatnonzero(centers >= 80.0)
    return {"xi0_smin50": xi0, "xi02_smin50_80": np.concatenate([xi0, xi2])}


def vector(values: dict[int, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(values[0], dtype="f8"), np.asarray(values[2], dtype="f8")])


def comparison(
    reference: dict[int, np.ndarray],
    candidate: dict[int, np.ndarray],
    metrics: dict[str, GaussianMetric],
    indices: dict[str, np.ndarray],
) -> dict[str, Any]:
    delta = vector(candidate) - vector(reference)
    result: dict[str, Any] = {
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(vector(reference)), 1.0e-300)),
    }
    for name, selected in indices.items():
        result[f"delta_chi2_Cmean_{name}"] = float(NPHASE * metrics[name].chi2(delta[selected]))
    return result


def frozen_p_manifest() -> dict[str, str]:
    paths = {
        "primary_result_json": P_REFERENCE.with_suffix(".json"),
        "primary_result_npz": P_REFERENCE.with_suffix(".npz"),
        "primary_fit_code": PROJECT_ROOT / "codes" / "task43" / "task43_fit_rsd_rawbox_pk0_vs_xi0_smin50.py",
        "frozen_theory_code": PROJECT_ROOT / "codes" / "task43" / "task43_rsd_model.py",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen P reference {name}: {path}")
    return {str(path): sha256_file(path) for path in paths.values()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if OUT_JSON.exists() or OUT_NPZ.exists():
        raise FileExistsError(f"immutable versioned output already exists: {OUT_JSON} / {OUT_NPZ}")

    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))
    p_manifest_before = frozen_p_manifest()
    cache_path = build_cache(
        zeff=0.725,
        boxsize=2000.0,
        kmax=3.0,
        ells=(0, 2),
        cosmology="abacus_c000",
    )
    model = XiLinearRSDModel(cache_path)
    _, metadata, xi_hashes = load_xi_x25()
    nbar = float(np.mean([row["nbar_h3_mpc3_from_npz"] for row in metadata]))
    covariance = model.covariance(b1=2.55, sigma_s=8.0, nbar=nbar)
    indices = selected_indices()
    metrics = {
        name: GaussianMetric(covariance[np.ix_(selected, selected)])
        for name, selected in indices.items()
    }

    cases = {
        "fiducial_no_fog": {"fnl": 0.0, "b1": 2.55, "sigma_s": 0.0},
        "fiducial_fog": {"fnl": 0.0, "b1": 2.55, "sigma_s": 8.0},
        "positive_png": {"fnl": 100.0, "b1": 2.55, "sigma_s": 8.0},
        "negative_png": {"fnl": -100.0, "b1": 2.55, "sigma_s": 8.0},
        "sigma_prior_edge": {"fnl": 0.0, "b1": 2.55, "sigma_s": 30.0},
    }
    case_results: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "covariance_single": covariance,
        "s_edges": np.asarray(S_EDGES, dtype="f8"),
    }
    gate_values: list[float] = []
    for name, parameters in cases.items():
        analytic = model.evaluate(**parameters, use_cached_kernels=False)
        analytic_cached_kernel = model.evaluate(**parameters, use_cached_kernels=True)
        gl64 = model.evaluate_gauss_legendre(
            **parameters,
            nmu=64,
            use_cached_kernels=False,
        )
        gl256 = model.evaluate_gauss_legendre(
            **parameters,
            nmu=256,
            use_cached_kernels=False,
        )
        case_results[name] = {
            "parameters": parameters,
            "gl64_minus_analytic": comparison(analytic, gl64, metrics, indices),
            "gl256_minus_analytic": comparison(analytic, gl256, metrics, indices),
            "cached_minus_analytic_shell_kernel": comparison(
                analytic,
                analytic_cached_kernel,
                metrics,
                indices,
            ),
        }
        if name != "sigma_prior_edge":
            for probe in indices:
                gate_values.append(case_results[name]["gl64_minus_analytic"][f"delta_chi2_Cmean_{probe}"])
                gate_values.append(
                    case_results[name]["cached_minus_analytic_shell_kernel"][f"delta_chi2_Cmean_{probe}"]
                )
        for ell in (0, 2):
            arrays[f"{name}_analytic_xi{ell}"] = analytic[ell]
            arrays[f"{name}_gl64_xi{ell}"] = gl64[ell]

    lowk_scan: dict[str, Any] = {}
    fiducial = model.evaluate(fnl=0.0, b1=2.55, sigma_s=8.0)
    for k_switch in (0.005, 0.01, 0.02, 0.04, 0.06, 0.095):
        delta = model.lowk_angular_delta(
            fnl=0.0,
            b1=2.55,
            sigma_s=8.0,
            k_switch=k_switch,
        )
        corrected = {ell: fiducial[ell] + delta[ell] for ell in (0, 2)}
        key = f"k_switch_{k_switch:.3f}"
        lowk_scan[key] = comparison(fiducial, corrected, metrics, indices)
        for ell in (0, 2):
            arrays[f"{key}_angular_delta_xi{ell}"] = delta[ell]

    maximum_gate_value = float(max(gate_values))
    gates = {
        "analytic_angle_and_shell_numerics_below_delta_chi2_budget": maximum_gate_value < NUMERICAL_BUDGET,
        "frozen_p_hashes_unchanged": frozen_p_manifest() == p_manifest_before,
        "covariance_unfloored_and_normalized_spd": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"linear operator closure gates failed: {gates}")

    atomic_savez(OUT_NPZ, **arrays)
    audit = {
        "task": "task43_run_xi_linear_operator_closure_v2",
        "status": "pass",
        "scientific_scope": "xi-only no-fit numerical/operator closure; not a calibrated GSM or data-fit result",
        "frozen_p_sha256": p_manifest_before,
        "inputs": {
            "theory_cache": str(cache_path),
            "theory_cache_sha256": sha256_file(cache_path),
            "xi_measurement_sha256": xi_hashes,
            "nbar": nbar,
            "nphase": NPHASE,
        },
        "model_validation": model.validation_summary(),
        "covariance": {
            "definition": "analytic angular moments and analytic shell kernels; no eigenvalue floor",
            "correlation_eigenvalue_min": float(GaussianMetric(covariance).eigenvalues[0]),
            "correlation_eigenvalue_max": float(GaussianMetric(covariance).eigenvalues[-1]),
        },
        "numerical_budget_delta_chi2_Cmean": NUMERICAL_BUDGET,
        "maximum_gated_delta_chi2_Cmean": maximum_gate_value,
        "gates": gates,
        "cases": case_results,
        "finite_mode_angular_scan": {
            "interpretation": "exact finite-lattice minus continuous-angle contribution below k_switch; an operator difference, not quadrature error",
            "rows": lowk_scan,
        },
        "output_npz": str(OUT_NPZ),
        "output_npz_sha256": sha256_file(OUT_NPZ),
        "code_sha256": sha256_file(Path(__file__)),
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    atomic_write_json(OUT_JSON, audit)
    print(json.dumps({"gates": gates, "maximum_gated_delta_chi2_Cmean": maximum_gate_value, "cases": case_results, "finite_mode_angular_scan": lowk_scan}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
