#!/usr/bin/env python3
"""Run the ph000 measured-moment GSM oracle and numerical closure audit."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.special import eval_legendre

from task43_rawbox_numerics import GaussianMetric
from task43_rsd_common import OUTPUT_ROOT, S_EDGES, atomic_savez, atomic_write_json, sha256_file
from task43_rsd_rawbox_realspace_finalmetric_v2 import set_affinity
from task43_run_xi_linear_operator_closure_v2 import frozen_p_manifest
from task43_xi_gsm import (
    EmpiricalRadialMoments,
    fcfc_project_smu_vectorized,
    gsm_point,
    gsm_points_fixed,
)


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
PAIR_NPZ = OUTPUT_ROOT / "rawbox" / "xi_rsd_v2" / "stage03_pair_moments" / (
    "task43_rawbox_pair_moments_ph000_v2.npz"
)
PAIR_JSON = PAIR_NPZ.with_suffix(".json")
SUMMARY_NPZ = OUTPUT_ROOT / "rawbox" / "summary" / (
    "task43_rsd_rawbox_AbacusSummit_base_c000_ph000_mmin1p4e13_clustering.npz"
)
SUMMARY_JSON = SUMMARY_NPZ.with_suffix(".json")
SMU_TEXT = OUTPUT_ROOT / "rawbox" / "fcfc_text" / "xi_smu_ph000_rsd_s30_350_ds10_mu120.txt"
LINEAR_AUDIT_NPZ = OUTPUT_ROOT / "rawbox" / "xi_rsd_v2" / "stage02_linear_operator_closure" / (
    "task43_xi_linear_operator_closure_v2.npz"
)
OLD_CLOSURE_NPZ = OUTPUT_ROOT / "rawbox" / "closure" / (
    "task43_rsd_rawbox_x25_fulldiscrete_lorentzian.npz"
)
OLD_CLOSURE_JSON = OLD_CLOSURE_NPZ.with_suffix(".json")
OUT_DIR = OUTPUT_ROOT / "rawbox" / "xi_rsd_v2" / "stage04_empirical_gsm_ph000"
OUT_NPZ = OUT_DIR / "task43_empirical_gsm_ph000_v2.npz"
OUT_JSON = OUT_NPZ.with_suffix(".json")
NPHASE = 25
NUMERICAL_BUDGET = 0.01


def validated_json_npz(npz_path: Path, json_path: Path) -> dict[str, Any]:
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"missing validated input: {npz_path} / {json_path}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    digest = sha256_file(npz_path)
    if metadata.get("status") != "pass" or metadata.get("output_npz_sha256", metadata.get("output_sha256")) != digest:
        raise RuntimeError(f"input hash/status validation failed: {npz_path}")
    return metadata


def load_measurement() -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray, np.ndarray]:
    metadata = validated_json_npz(SUMMARY_NPZ, SUMMARY_JSON)
    if int(metadata.get("nmu", -1)) != 120:
        raise RuntimeError("FCFC measurement does not use the required 120 mu bins")
    with np.load(SUMMARY_NPZ, allow_pickle=False) as payload:
        edges = np.asarray(payload["s_edges"], dtype="f8")
        observed = {0: np.asarray(payload["xi0_rsd"], dtype="f8"), 2: np.asarray(payload["xi2_rsd"], dtype="f8")}
    if not np.array_equal(edges, S_EDGES):
        raise RuntimeError("FCFC radial edges changed")
    table = np.loadtxt(SMU_TEXT, comments="#", dtype="f8")
    if table.shape != (120 * 32, 5):
        raise RuntimeError("unexpected FCFC s-mu text shape")
    smu = table[:, 4].reshape(120, 32).T
    mu_edges = np.concatenate([table[::32, 2], table[-32:-31, 3]])
    mu_midpoint = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    reprojection = {
        0: np.mean(smu, axis=1),
        2: 5.0 * np.mean(smu * eval_legendre(2, mu_midpoint)[None, :], axis=1),
    }
    bridge = max(float(np.max(np.abs(reprojection[ell] - observed[ell]))) for ell in (0, 2))
    if bridge > 1.0e-10:
        raise RuntimeError(f"FCFC finite-mu reprojection failed: {bridge}")
    return smu, observed, edges, mu_edges


def selected_indices(ells: tuple[int, ...], smin: float) -> np.ndarray:
    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    base = np.flatnonzero(centers >= float(smin))
    return np.concatenate([base + (0 if ell == 0 else centers.size) for ell in ells])


def vector(poles: dict[int, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(poles[0], dtype="f8"), np.asarray(poles[2], dtype="f8")])


def metric_rows(
    residual: np.ndarray,
    covariance_single: np.ndarray,
    *,
    covariance_multiplier: float = 1.0,
) -> dict[str, float]:
    rows: dict[str, float] = {}
    for ells in ((0,), (0, 2)):
        for smin in (50.0, 80.0, 100.0, 120.0):
            index = selected_indices(ells, smin)
            metric = GaussianMetric(float(covariance_multiplier) * covariance_single[np.ix_(index, index)])
            rows[f"ell{''.join(map(str, ells))}_smin{int(smin)}"] = metric.chi2(residual[index])
    return rows


def evaluate_provider(
    provider: EmpiricalRadialMoments,
    radial_edges: np.ndarray,
    mu_edges: np.ndarray,
    *,
    quadrature_order: int,
    nradial: int,
    nmu_per_bin: int,
    zmax: float,
    periodic_images: int,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    scale = provider.integration_scale

    def evaluator(transverse: np.ndarray, parallel: np.ndarray) -> np.ndarray:
        return gsm_points_fixed(
            transverse,
            parallel,
            provider,
            integration_scale=scale,
            zmax=float(zmax),
            quadrature_order=int(quadrature_order),
            extra_pair_variance=0.0,
            boxsize=2000.0,
            periodic_images=int(periodic_images),
        )

    return fcfc_project_smu_vectorized(
        evaluator,
        radial_edges,
        mu_edges,
        nradial=int(nradial),
        nmu_per_bin=int(nmu_per_bin),
    )


def old_predictions() -> dict[str, dict[int, np.ndarray]]:
    metadata = json.loads(OLD_CLOSURE_JSON.read_text(encoding="utf-8"))
    result: dict[str, dict[int, np.ndarray]] = {}
    for label, ells in (("old_x25_mean_fit_ell0", [0]), ("old_x25_mean_fit_ell02", [0, 2])):
        row = next(item for item in metadata["fits"] if item["ells"] == ells and item["smin_mpc_h"] == 50.0)
        prediction = {int(ell): np.asarray(row["prediction"][str(ell)], dtype="f8") for ell in ells}
        if 2 not in prediction:
            prediction[2] = np.full(32, np.nan)
        result[label] = prediction
    phase = next(item for item in metadata["phase_profiles"] if item["phase"] == "ph000")
    result["old_ph000_profile_fit_ell0"] = {
        0: np.asarray(phase["prediction"]["0"], dtype="f8"),
        2: np.full(32, np.nan),
    }
    return result


def main() -> None:
    if OUT_NPZ.exists() or OUT_JSON.exists():
        raise FileExistsError(f"immutable output exists: {OUT_NPZ} / {OUT_JSON}")
    started = time.perf_counter()
    cpus = set_affinity(1)
    pair_metadata = validated_json_npz(PAIR_NPZ, PAIR_JSON)
    frozen_p_before = frozen_p_manifest()
    observed_smu, observed, radial_edges, mu_edges = load_measurement()
    with np.load(LINEAR_AUDIT_NPZ, allow_pickle=False) as payload:
        covariance_single = np.asarray(payload["covariance_single"], dtype="f8")
    GaussianMetric(covariance_single)
    provider = EmpiricalRadialMoments.from_npz(PAIR_NPZ)

    settings = {
        "final": dict(quadrature_order=96, nradial=12, nmu_per_bin=2, zmax=10.0, periodic_images=1),
        "quadrature64": dict(quadrature_order=64, nradial=12, nmu_per_bin=2, zmax=10.0, periodic_images=1),
        "radial10": dict(quadrature_order=96, nradial=10, nmu_per_bin=2, zmax=10.0, periodic_images=1),
        "mu1": dict(quadrature_order=96, nradial=12, nmu_per_bin=1, zmax=10.0, periodic_images=1),
        "zmax8": dict(quadrature_order=96, nradial=12, nmu_per_bin=2, zmax=8.0, periodic_images=1),
        "no_periodic_images": dict(
            quadrature_order=96, nradial=12, nmu_per_bin=2, zmax=10.0, periodic_images=0
        ),
    }
    predictions = {
        name: evaluate_provider(provider, radial_edges, mu_edges, **setting)
        for name, setting in settings.items()
    }
    final_smu, final_poles = predictions["final"]
    final_vector = vector(final_poles)

    numerical: dict[str, Any] = {}
    maximum_numerical_delta_chi2 = 0.0
    for name in settings:
        if name == "final":
            continue
        delta = vector(predictions[name][1]) - final_vector
        rows = metric_rows(delta, covariance_single, covariance_multiplier=1.0 / NPHASE)
        maximum_numerical_delta_chi2 = max(maximum_numerical_delta_chi2, max(rows.values()))
        numerical[name] = {
            "settings": settings[name],
            "max_abs_smu": float(np.max(np.abs(predictions[name][0] - final_smu))),
            "max_abs_poles": float(np.max(np.abs(delta))),
            "delta_chi2_Cmean": rows,
        }

    adaptive_spots: list[dict[str, float]] = []
    centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    mu_midpoint = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    for ishell, imu in ((0, 0), (0, 119), (2, 60), (9, 90), (21, 30), (31, 119)):
        radius, mu = float(centers[ishell]), float(mu_midpoint[imu])
        transverse, parallel = radius * np.sqrt(1.0 - mu**2), radius * mu
        adaptive, error = gsm_point(
            transverse,
            parallel,
            provider.at_los,
            integration_scale=provider.integration_scale,
            zmax=10.0,
            boxsize=2000.0,
            periodic_images=1,
        )
        fixed = float(
            gsm_points_fixed(
                np.asarray(transverse),
                np.asarray(parallel),
                provider,
                integration_scale=provider.integration_scale,
                zmax=10.0,
                quadrature_order=96,
                boxsize=2000.0,
                periodic_images=1,
            )
        )
        adaptive_spots.append(
            {
                "s": radius,
                "mu": mu,
                "fixed": fixed,
                "adaptive": adaptive,
                "abs_difference": abs(fixed - adaptive),
                "adaptive_quadrature_error": error,
            }
        )

    observed_vector = vector(observed)
    oracle_residual = observed_vector - final_vector
    closure = {
        "empirical_gsm_oracle": {
            "chi2_Csingle": metric_rows(oracle_residual, covariance_single),
            "smu_rms_smin50": float(np.sqrt(np.mean((observed_smu[2:] - final_smu[2:]) ** 2))),
            "smu_max_abs_smin50": float(np.max(np.abs(observed_smu[2:] - final_smu[2:]))),
        }
    }
    old = old_predictions()
    for name, prediction in old.items():
        if np.all(np.isfinite(prediction[2])):
            residual = observed_vector - vector(prediction)
            rows = metric_rows(residual, covariance_single)
        else:
            index = selected_indices((0,), 50.0)
            residual0 = observed[0] - prediction[0]
            rows = {
                "ell0_smin50": GaussianMetric(covariance_single[np.ix_(index, index)]).chi2(
                    residual0[index]
                )
            }
        closure[name] = {"chi2_Csingle": rows}

    block_predictions = []
    with np.load(PAIR_NPZ, allow_pickle=False) as payload:
        los_count = np.asarray(payload["pair_los_count"], dtype="f8")
        los_mean = np.asarray(payload["pair_los_aligned_mean"], dtype="f8")
        los_variance = np.asarray(payload["pair_los_variance_central"], dtype="f8")
        mu_abs_mean = np.asarray(payload["pair_mu_abs_mean"], dtype="f8")
        mu2_mean = np.asarray(payload["pair_mu2_mean"], dtype="f8")
        radial_mean = np.asarray(payload["pair_v12_radial"], dtype="f8")
        radial_variance = np.asarray(payload["pair_sigma_r2_central"], dtype="f8")
        transverse_variance = np.asarray(payload["pair_sigma_t2_one_component"], dtype="f8")
        tensor_mean = mu_abs_mean * radial_mean[:, None]
        tensor_variance = (
            mu2_mean * radial_variance[:, None]
            + (1.0 - mu2_mean) * transverse_variance[:, None]
        )
        isotropy_mask = los_count >= 100.0
        mean_delta = (los_mean - tensor_mean)[isotropy_mask]
        variance_delta = (los_variance - tensor_variance)[isotropy_mask]
        variance_fraction = np.abs(variance_delta) / tensor_variance[isotropy_mask]
        isotropy_diagnostic = {
            "minimum_los_bin_count_used": 100,
            "mean_max_abs_mpc_h": float(np.max(np.abs(mean_delta))),
            "mean_rms_mpc_h": float(np.sqrt(np.mean(mean_delta**2))),
            "variance_max_abs_mpc_h2": float(np.max(np.abs(variance_delta))),
            "variance_rms_mpc_h2": float(np.sqrt(np.mean(variance_delta**2))),
            "variance_fractional_abs_median": float(np.median(variance_fraction)),
            "variance_fractional_abs_p95": float(np.quantile(variance_fraction, 0.95)),
        }
        for block_index in range(int(pair_metadata["pair_sampling"]["nblocks"])):
            block_provider = EmpiricalRadialMoments.from_arrays(
                payload["radial_edges"],
                payload["full_xi_real"],
                payload["block_v12_radial"][block_index],
                payload["block_sigma_r2_central"][block_index],
                payload["block_sigma_t2_one_component"][block_index],
            )
            _, block_poles = evaluate_provider(
                block_provider,
                radial_edges,
                mu_edges,
                quadrature_order=32,
                nradial=8,
                nmu_per_bin=2,
                zmax=10.0,
                periodic_images=1,
            )
            block_predictions.append(vector(block_poles))
    block_prediction_array = np.stack(block_predictions)
    prediction_se = np.std(block_prediction_array, axis=0, ddof=1) / np.sqrt(block_prediction_array.shape[0])

    gates = {
        "all_numerical_refinements_below_delta_chi2_Cmean_budget": maximum_numerical_delta_chi2
        < NUMERICAL_BUDGET,
        "adaptive_spot_max_abs_below_1e-8": max(row["abs_difference"] for row in adaptive_spots) < 1.0e-8,
        "fcfc_input_reprojection_exact": True,
        "empirical_variances_positive_without_clipping": True,
        "frozen_p_hashes_unchanged": frozen_p_manifest() == frozen_p_before,
    }
    if not all(gates.values()):
        raise RuntimeError(f"empirical GSM numerical gates failed: {gates}")

    arrays: dict[str, np.ndarray] = {
        "s_edges": radial_edges,
        "mu_edges": mu_edges,
        "observed_smu_ph000": observed_smu,
        "observed_xi0_ph000": observed[0],
        "observed_xi2_ph000": observed[2],
        "empirical_gsm_smu": final_smu,
        "empirical_gsm_xi0": final_poles[0],
        "empirical_gsm_xi2": final_poles[2],
        "covariance_single": covariance_single,
        "block_prediction_vector_lowres": block_prediction_array,
        "pair_sampling_prediction_se_lowres": prediction_se,
    }
    for name, (smu, poles) in predictions.items():
        arrays[f"{name}_smu"] = smu
        arrays[f"{name}_xi0"] = poles[0]
        arrays[f"{name}_xi2"] = poles[2]
    for name, prediction in old.items():
        arrays[f"{name}_xi0"] = prediction[0]
        arrays[f"{name}_xi2"] = prediction[2]
    atomic_savez(OUT_NPZ, **arrays)

    audit = {
        "task": "task43_run_empirical_gsm_ph000_v2",
        "status": "pass",
        "scientific_scope": (
            "same-catalog empirical-density/velocity-moment GSM oracle; no fitted parameters and not an "
            "independent theory prediction"
        ),
        "phase": "ph000",
        "inputs": {
            str(path): sha256_file(path)
            for path in (PAIR_NPZ, SUMMARY_NPZ, SMU_TEXT, LINEAR_AUDIT_NPZ, OLD_CLOSURE_NPZ)
        },
        "frozen_p_sha256": frozen_p_before,
        "model_contract": {
            "density": "full-catalog real-space xi from Corrfunc, bridged to FCFC",
            "moments": "uniform-anchor pair-weighted signed mean and central radial/transverse variances",
            "mapping": "pair-conserving Gaussian streaming of 1+xi using real-space mu",
            "extra_pair_variance": 0.0,
            "projection": "mapped xi averaged in shell volume and each FCFC mu bin; multipoles use mu-bin midpoint",
            "radial_support_mpc_h": provider.support,
            "integration_scale_mpc_h": provider.integration_scale,
            "final_settings": settings["final"],
        },
        "numerical_budget_delta_chi2_Cmean": NUMERICAL_BUDGET,
        "maximum_numerical_delta_chi2_Cmean": maximum_numerical_delta_chi2,
        "numerical_refinements": numerical,
        "adaptive_spots": adaptive_spots,
        "moment_isotropy_diagnostic": {
            "source": "direct absolute-mu-binned LOS moments versus radial tensor reconstruction",
            "interpretation": "diagnostic only; deviations are not absorbed into an extra dispersion",
            **isotropy_diagnostic,
        },
        "pair_sampling_prediction_uncertainty": {
            "nblocks": int(block_prediction_array.shape[0]),
            "definition": "SE across disjoint anchor-block moment predictions; density xi held at full catalog",
            "median_xi0_se": float(np.median(prediction_se[:32])),
            "max_xi0_se": float(np.max(prediction_se[:32])),
            "median_xi2_se": float(np.median(prediction_se[32:])),
            "max_xi2_se": float(np.max(prediction_se[32:])),
            "caveat": "oracle inputs and RSD target share a catalog; cross-covariance is not calibrated here",
        },
        "closure": closure,
        "gates": gates,
        "output_npz": str(OUT_NPZ),
        "output_npz_sha256": sha256_file(OUT_NPZ),
        "code_sha256": sha256_file(Path(__file__)),
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    atomic_write_json(OUT_JSON, audit)
    print(
        json.dumps(
            {
                "gates": gates,
                "maximum_numerical_delta_chi2_Cmean": maximum_numerical_delta_chi2,
                "closure": closure,
                "pair_sampling_prediction_uncertainty": audit["pair_sampling_prediction_uncertainty"],
                "elapsed_sec": audit["elapsed_sec"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
