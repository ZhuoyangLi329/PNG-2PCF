#!/usr/bin/env python3
"""Compare refined Gaussian moments with exact same-anchor pair mapping."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from task43_pair_moment_smoothing import apply_smoothing_factor, cross_validated_smoothing
from task43_rawbox_numerics import GaussianMetric
from task43_rsd_common import OUTPUT_ROOT, atomic_savez, atomic_write_json, sha256_file
from task43_rsd_rawbox_realspace_finalmetric_v2 import set_affinity
from task43_run_empirical_gsm_ph000_v2 import (
    LINEAR_AUDIT_NPZ,
    SUMMARY_NPZ,
    evaluate_provider,
    metric_rows,
    vector,
)
from task43_run_xi_linear_operator_closure_v2 import frozen_p_manifest
from task43_xi_gsm import EmpiricalRadialMoments, fcfc_project_smu_vectorized


PAIR_NPZ = OUTPUT_ROOT / "rawbox" / "xi_rsd_v2" / "stage03_pair_moments" / (
    "task43_rawbox_pair_moments_ph000_a8192_v3.npz"
)
PAIR_JSON = PAIR_NPZ.with_suffix(".json")
OLD_ORACLE_NPZ = OUTPUT_ROOT / "rawbox" / "xi_rsd_v2" / "stage04_empirical_gsm_ph000" / (
    "task43_empirical_gsm_ph000_v2.npz"
)
OUT_DIR = OUTPUT_ROOT / "rawbox" / "xi_rsd_v2" / "stage05_empirical_gsm_refined_ph000"
OUT_NPZ = OUT_DIR / "task43_empirical_gsm_refined_ph000_v3.npz"
OUT_JSON = OUT_NPZ.with_suffix(".json")
FINAL_SETTINGS = dict(quadrature_order=96, nradial=12, nmu_per_bin=2, zmax=10.0, periodic_images=1)
LOW_SETTINGS = dict(quadrature_order=64, nradial=10, nmu_per_bin=2, zmax=10.0, periodic_images=1)
NUMERICAL_BUDGET = 0.01


def validated_pair_input() -> dict[str, Any]:
    if not PAIR_NPZ.is_file() or not PAIR_JSON.is_file():
        raise FileNotFoundError(f"missing refined pair input: {PAIR_NPZ}")
    metadata = json.loads(PAIR_JSON.read_text(encoding="utf-8"))
    if metadata.get("status") != "pass" or metadata.get("output_npz_sha256") != sha256_file(PAIR_NPZ):
        raise RuntimeError("refined pair input failed status/hash validation")
    return metadata


def zero_velocity_projection(
    provider: EmpiricalRadialMoments, radial_edges: np.ndarray, mu_edges: np.ndarray
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    return fcfc_project_smu_vectorized(
        lambda transverse, parallel: provider.evaluate_radial(np.hypot(transverse, parallel))[0],
        radial_edges,
        mu_edges,
        nradial=12,
        nmu_per_bin=2,
    )


def rms(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values, dtype="f8")[mask]
    return float(np.sqrt(np.mean(selected**2)))


def comparison_rows(
    model_poles: dict[int, np.ndarray],
    zero_poles: dict[int, np.ndarray],
    target_poles: dict[int, np.ndarray],
    target_zero_poles: dict[int, np.ndarray],
    covariance_single: np.ndarray,
) -> dict[str, Any]:
    centers = np.arange(35.0, 350.0, 10.0)
    mask = centers >= 50.0
    residual = vector(target_poles) - vector(model_poles)
    model_increment = {ell: model_poles[ell] - zero_poles[ell] for ell in (0, 2)}
    target_increment = {ell: target_poles[ell] - target_zero_poles[ell] for ell in (0, 2)}
    increment_residual = {ell: target_increment[ell] - model_increment[ell] for ell in (0, 2)}
    return {
        "chi2_Csingle": metric_rows(residual, covariance_single),
        "rms_residual_smin50": {
            "xi0": rms(residual[:32], mask),
            "xi2": rms(residual[32:], mask),
        },
        "rms_rsd_increment_residual_smin50": {
            f"xi{ell}": rms(increment_residual[ell], mask) for ell in (0, 2)
        },
    }


def main() -> None:
    if OUT_NPZ.exists() or OUT_JSON.exists():
        raise FileExistsError(f"immutable output exists: {OUT_NPZ} / {OUT_JSON}")
    started = time.perf_counter()
    cpus = set_affinity(1)
    pair_metadata = validated_pair_input()
    frozen_before = frozen_p_manifest()
    with np.load(PAIR_NPZ, allow_pickle=False) as payload:
        radial_moment_edges = np.asarray(payload["radial_edges"], dtype="f8")
        radial_edges = np.asarray(payload["mapping_s_edges"], dtype="f8")
        mu_edges = np.asarray(payload["mapping_mu_edges"], dtype="f8")
        full_xi = np.asarray(payload["full_xi_real"], dtype="f8")
        anchor_xi = np.asarray(payload["pair_xi_anchor_sample"], dtype="f8")
        pooled = {
            "v12": np.asarray(payload["pair_v12_radial"], dtype="f8"),
            "sigma_r2": np.asarray(payload["pair_sigma_r2_central"], dtype="f8"),
            "sigma_t2": np.asarray(payload["pair_sigma_t2_one_component"], dtype="f8"),
        }
        errors = {
            "v12": np.asarray(payload["se_v12_radial"], dtype="f8"),
            "sigma_r2": np.asarray(payload["se_sigma_r2_central"], dtype="f8"),
            "sigma_t2": np.asarray(payload["se_sigma_t2_one_component"], dtype="f8"),
        }
        blocks = {
            "v12": np.asarray(payload["block_v12_radial"], dtype="f8"),
            "sigma_r2": np.asarray(payload["block_sigma_r2_central"], dtype="f8"),
            "sigma_t2": np.asarray(payload["block_sigma_t2_one_component"], dtype="f8"),
        }
        anchor_smu = np.asarray(payload["mapping_xi_smu_rsd"], dtype="f8")
        anchor_real_poles = {
            0: np.asarray(payload["mapping_xi0_real"], dtype="f8"),
            2: np.asarray(payload["mapping_xi2_real"], dtype="f8"),
        }
        anchor_rsd_poles = {
            0: np.asarray(payload["mapping_xi0_rsd"], dtype="f8"),
            2: np.asarray(payload["mapping_xi2_rsd"], dtype="f8"),
        }
    with np.load(SUMMARY_NPZ, allow_pickle=False) as summary:
        full_real_poles = {0: np.asarray(summary["xi0_real"], dtype="f8"), 2: np.asarray(summary["xi2_real"], dtype="f8")}
        full_rsd_poles = {0: np.asarray(summary["xi0_rsd"], dtype="f8"), 2: np.asarray(summary["xi2_rsd"], dtype="f8")}
    with np.load(LINEAR_AUDIT_NPZ, allow_pickle=False) as linear:
        covariance_single = np.asarray(linear["covariance_single"], dtype="f8")
    GaussianMetric(covariance_single)

    smoothing_audit: dict[str, Any] = {}
    smoothed: dict[str, np.ndarray] = {}
    for name in ("v12", "sigma_r2", "sigma_t2"):
        _, selection = cross_validated_smoothing(
            radial_moment_edges,
            blocks[name],
            require_positive=name.startswith("sigma"),
        )
        value = apply_smoothing_factor(
            radial_moment_edges,
            pooled[name],
            errors[name],
            selection["selected_factor"],
        )
        if name.startswith("sigma") and np.any(value <= 0.0):
            raise RuntimeError("target-blind smoothing produced a non-positive variance")
        smoothed[name] = value
        smoothing_audit[name] = selection

    providers = {
        "full_density_raw_moments": EmpiricalRadialMoments.from_arrays(
            radial_moment_edges, full_xi, pooled["v12"], pooled["sigma_r2"], pooled["sigma_t2"]
        ),
        "full_density_smoothed_moments": EmpiricalRadialMoments.from_arrays(
            radial_moment_edges, full_xi, smoothed["v12"], smoothed["sigma_r2"], smoothed["sigma_t2"]
        ),
        "anchor_density_raw_moments": EmpiricalRadialMoments.from_arrays(
            radial_moment_edges, anchor_xi, pooled["v12"], pooled["sigma_r2"], pooled["sigma_t2"]
        ),
        "anchor_density_smoothed_moments": EmpiricalRadialMoments.from_arrays(
            radial_moment_edges, anchor_xi, smoothed["v12"], smoothed["sigma_r2"], smoothed["sigma_t2"]
        ),
    }
    predictions = {
        name: evaluate_provider(provider, radial_edges, mu_edges, **FINAL_SETTINGS)
        for name, provider in providers.items()
    }
    zeros = {
        name: zero_velocity_projection(provider, radial_edges, mu_edges) for name, provider in providers.items()
    }

    low_smu, low_poles = evaluate_provider(
        providers["full_density_raw_moments"], radial_edges, mu_edges, **LOW_SETTINGS
    )
    final_smu, final_poles = predictions["full_density_raw_moments"]
    numerical_delta = vector(low_poles) - vector(final_poles)
    numerical_rows = metric_rows(numerical_delta, covariance_single, covariance_multiplier=1.0 / 25.0)
    maximum_numerical_delta_chi2 = float(max(numerical_rows.values()))

    closure = {
        "full_density_raw_moments_vs_full_fcfc": comparison_rows(
            predictions["full_density_raw_moments"][1],
            zeros["full_density_raw_moments"][1],
            full_rsd_poles,
            full_real_poles,
            covariance_single,
        ),
        "full_density_smoothed_moments_vs_full_fcfc": comparison_rows(
            predictions["full_density_smoothed_moments"][1],
            zeros["full_density_smoothed_moments"][1],
            full_rsd_poles,
            full_real_poles,
            covariance_single,
        ),
        "anchor_density_raw_moments_vs_exact_anchor_pdf": comparison_rows(
            predictions["anchor_density_raw_moments"][1],
            zeros["anchor_density_raw_moments"][1],
            anchor_rsd_poles,
            anchor_real_poles,
            covariance_single,
        ),
        "anchor_density_smoothed_moments_vs_exact_anchor_pdf": comparison_rows(
            predictions["anchor_density_smoothed_moments"][1],
            zeros["anchor_density_smoothed_moments"][1],
            anchor_rsd_poles,
            anchor_real_poles,
            covariance_single,
        ),
    }
    with np.load(OLD_ORACLE_NPZ, allow_pickle=False) as old:
        old_vector = np.concatenate([old["empirical_gsm_xi0"], old["empirical_gsm_xi2"]])
    v2_to_v3 = vector(predictions["full_density_raw_moments"][1]) - old_vector

    gates = {
        "refined_input_hash_and_status_valid": True,
        "low_vs_final_numerics_below_delta_chi2_Cmean_budget": maximum_numerical_delta_chi2
        < NUMERICAL_BUDGET,
        "target_blind_smoothing_keeps_variances_positive": True,
        "frozen_p_hashes_unchanged": frozen_p_manifest() == frozen_before,
    }
    if not all(gates.values()):
        raise RuntimeError(f"refined GSM gates failed: {gates}")

    arrays: dict[str, np.ndarray] = {
        "s_edges": radial_edges,
        "mu_edges": mu_edges,
        "covariance_single": covariance_single,
        "full_fcfc_xi0_real": full_real_poles[0],
        "full_fcfc_xi2_real": full_real_poles[2],
        "full_fcfc_xi0_rsd": full_rsd_poles[0],
        "full_fcfc_xi2_rsd": full_rsd_poles[2],
        "anchor_exact_smu_rsd": anchor_smu,
        "anchor_exact_xi0_real": anchor_real_poles[0],
        "anchor_exact_xi2_real": anchor_real_poles[2],
        "anchor_exact_xi0_rsd": anchor_rsd_poles[0],
        "anchor_exact_xi2_rsd": anchor_rsd_poles[2],
        "smoothed_v12": smoothed["v12"],
        "smoothed_sigma_r2": smoothed["sigma_r2"],
        "smoothed_sigma_t2": smoothed["sigma_t2"],
        "low_minus_final_prediction_vector": numerical_delta,
        "v3_minus_v2_prediction_vector": v2_to_v3,
    }
    for name, (smu, poles) in predictions.items():
        arrays[f"{name}_smu"] = smu
        arrays[f"{name}_xi0"] = poles[0]
        arrays[f"{name}_xi2"] = poles[2]
        arrays[f"{name}_zero_xi0"] = zeros[name][1][0]
        arrays[f"{name}_zero_xi2"] = zeros[name][1][2]
    atomic_savez(OUT_NPZ, **arrays)

    audit = {
        "task": "task43_run_empirical_gsm_refined_ph000_v3",
        "status": "pass",
        "scientific_scope": "refined same-catalog Gaussian-moment oracle versus exact same-anchor PDF mapping",
        "phase": "ph000",
        "inputs": {
            str(path): sha256_file(path)
            for path in (PAIR_NPZ, SUMMARY_NPZ, LINEAR_AUDIT_NPZ, OLD_ORACLE_NPZ)
        },
        "frozen_p_sha256": frozen_before,
        "settings": {"final": FINAL_SETTINGS, "low_resolution_check": LOW_SETTINGS},
        "numerical": {
            "budget_delta_chi2_Cmean": NUMERICAL_BUDGET,
            "maximum_delta_chi2_Cmean": maximum_numerical_delta_chi2,
            "rows": numerical_rows,
            "max_abs_smu": float(np.max(np.abs(low_smu - final_smu))),
        },
        "target_blind_smoothing": smoothing_audit,
        "closure": closure,
        "v3_minus_v2": {
            "max_abs_prediction": float(np.max(np.abs(v2_to_v3))),
            "delta_chi2_Cmean": metric_rows(v2_to_v3, covariance_single, covariance_multiplier=1.0 / 25.0),
        },
        "interpretation_contract": {
            "exact_anchor_pdf": "direct mapped same-anchor pairs; validates units/sign/periodicity but retains anchor noise",
            "Gaussian_moment": "uses only pair-weighted mean and central radial/transverse variances",
            "RSD_increment": "(RSD-real) comparison reduces common anchor density fluctuations",
            "no_parameter_fit": True,
            "extra_pair_variance": 0.0,
        },
        "gates": gates,
        "output_npz": str(OUT_NPZ),
        "output_npz_sha256": sha256_file(OUT_NPZ),
        "code_sha256": sha256_file(Path(__file__)),
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
        "source_pair_sampling": pair_metadata["pair_sampling"],
    }
    atomic_write_json(OUT_JSON, audit)
    print(
        json.dumps(
            {
                "gates": gates,
                "numerical": audit["numerical"],
                "closure": closure,
                "v3_minus_v2": audit["v3_minus_v2"],
                "elapsed_sec": audit["elapsed_sec"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
