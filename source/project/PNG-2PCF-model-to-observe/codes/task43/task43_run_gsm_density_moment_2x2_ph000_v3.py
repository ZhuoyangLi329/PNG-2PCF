#!/usr/bin/env python3
"""Measured/theory density and moment 2x2 GSM diagnosis for ph000."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from task43_pair_moment_smoothing import apply_smoothing_factor, cross_validated_smoothing
from task43_rawbox_numerics import GaussianMetric
from task43_rsd_common import OUTPUT_ROOT, atomic_savez, atomic_write_json, sha256_file
from task43_rsd_model import build_cache
from task43_rsd_rawbox_realspace_finalmetric_v2 import set_affinity
from task43_run_empirical_gsm_ph000_v2 import LINEAR_AUDIT_NPZ, SUMMARY_NPZ, evaluate_provider, metric_rows, vector
from task43_run_xi_linear_operator_closure_v2 import frozen_p_manifest
from task43_xi_gsm import EmpiricalRadialMoments
from task43_xi_linear_rsd import XiLinearRSDModel


PAIR_NPZ = OUTPUT_ROOT / "rawbox" / "xi_rsd_v2" / "stage03_pair_moments" / (
    "task43_rawbox_pair_moments_ph000_a8192_v3.npz"
)
PAIR_JSON = PAIR_NPZ.with_suffix(".json")
REALSPACE_NPZ = OUTPUT_ROOT / "rawbox" / "xi_rsd_v2" / "stage01_realspace_finalmetric" / (
    "task43_rsd_rawbox_realspace_finalmetric_v2.npz"
)
OUT_DIR = OUTPUT_ROOT / "rawbox" / "xi_rsd_v2" / "stage06_gsm_density_moment_2x2_ph000"
OUT_NPZ = OUT_DIR / "task43_gsm_density_moment_2x2_ph000_v3.npz"
OUT_JSON = OUT_NPZ.with_suffix(".json")
SETTINGS = dict(quadrature_order=96, nradial=12, nmu_per_bin=2, zmax=10.0, periodic_images=1)
PARAMETERS = {"fnl": -3.8386363770476493, "b1": 2.6492255894484416}


def main() -> None:
    if OUT_NPZ.exists() or OUT_JSON.exists():
        raise FileExistsError(f"immutable output exists: {OUT_NPZ} / {OUT_JSON}")
    started = time.perf_counter()
    cpus = set_affinity(1)
    pair_metadata = json.loads(PAIR_JSON.read_text(encoding="utf-8"))
    if pair_metadata.get("status") != "pass" or pair_metadata.get("output_npz_sha256") != sha256_file(PAIR_NPZ):
        raise RuntimeError("refined pair input failed status/hash validation")
    frozen_before = frozen_p_manifest()

    cache_path = build_cache(
        zeff=0.725,
        boxsize=2000.0,
        kmax=3.0,
        ells=(0, 2),
        cosmology="abacus_c000",
    )
    linear_model = XiLinearRSDModel(cache_path)
    with np.load(PAIR_NPZ, allow_pickle=False) as payload:
        moment_edges = np.asarray(payload["radial_edges"], dtype="f8")
        output_edges = np.asarray(payload["mapping_s_edges"], dtype="f8")
        mu_edges = np.asarray(payload["mapping_mu_edges"], dtype="f8")
        measured_density = np.asarray(payload["full_xi_real"], dtype="f8")
        measured_raw = {
            "v12": np.asarray(payload["pair_v12_radial"], dtype="f8"),
            "sigma_r2": np.asarray(payload["pair_sigma_r2_central"], dtype="f8"),
            "sigma_t2": np.asarray(payload["pair_sigma_t2_one_component"], dtype="f8"),
        }
        measured_se = {
            "v12": np.asarray(payload["se_v12_radial"], dtype="f8"),
            "sigma_r2": np.asarray(payload["se_sigma_r2_central"], dtype="f8"),
            "sigma_t2": np.asarray(payload["se_sigma_t2_one_component"], dtype="f8"),
        }
        measured_blocks = {
            "v12": np.asarray(payload["block_v12_radial"], dtype="f8"),
            "sigma_r2": np.asarray(payload["block_sigma_r2_central"], dtype="f8"),
            "sigma_t2": np.asarray(payload["block_sigma_t2_one_component"], dtype="f8"),
        }
    measured = {}
    smoothing = {}
    for name in measured_raw:
        _, audit = cross_validated_smoothing(
            moment_edges,
            measured_blocks[name],
            require_positive=name.startswith("sigma"),
        )
        measured[name] = apply_smoothing_factor(
            moment_edges,
            measured_raw[name],
            measured_se[name],
            audit["selected_factor"],
        )
        smoothing[name] = audit

    theory = linear_model.realspace_pair_moments(
        fnl=PARAMETERS["fnl"],
        b1=PARAMETERS["b1"],
        radial_edges=moment_edges,
    )
    theory_density = np.asarray(theory["xi_real"], dtype="f8")
    providers = {
        "measured_density_measured_moments": EmpiricalRadialMoments.from_arrays(
            moment_edges,
            measured_density,
            measured["v12"],
            measured["sigma_r2"],
            measured["sigma_t2"],
        ),
        "theory_density_measured_moments": EmpiricalRadialMoments.from_arrays(
            moment_edges,
            theory_density,
            measured["v12"],
            measured["sigma_r2"],
            measured["sigma_t2"],
        ),
        "measured_density_theory_moments": EmpiricalRadialMoments.from_arrays(
            moment_edges,
            measured_density,
            theory["v12_radial"],
            theory["sigma_r2_central"],
            theory["sigma_t2_one_component"],
        ),
        "theory_density_theory_moments": EmpiricalRadialMoments.from_arrays(
            moment_edges,
            theory_density,
            theory["v12_radial"],
            theory["sigma_r2_central"],
            theory["sigma_t2_one_component"],
        ),
    }
    predictions = {
        name: evaluate_provider(provider, output_edges, mu_edges, **SETTINGS)
        for name, provider in providers.items()
    }
    with np.load(SUMMARY_NPZ, allow_pickle=False) as summary:
        observed = {
            0: np.asarray(summary["xi0_rsd"], dtype="f8"),
            2: np.asarray(summary["xi2_rsd"], dtype="f8"),
        }
    with np.load(LINEAR_AUDIT_NPZ, allow_pickle=False) as linear:
        covariance_single = np.asarray(linear["covariance_single"], dtype="f8")
    GaussianMetric(covariance_single)

    rows = {}
    for name, (_, poles) in predictions.items():
        residual = vector(observed) - vector(poles)
        rows[name] = {
            "chi2_Csingle": metric_rows(residual, covariance_single),
            "rms_xi0_smin50": float(np.sqrt(np.mean(residual[2:32] ** 2))),
            "rms_xi2_smin50": float(np.sqrt(np.mean(residual[34:] ** 2))),
        }
    baseline = vector(predictions["measured_density_measured_moments"][1])
    swaps = {}
    for name in providers:
        if name == "measured_density_measured_moments":
            continue
        delta = vector(predictions[name][1]) - baseline
        swaps[name] = {
            "max_abs_prediction_delta": float(np.max(np.abs(delta))),
            "delta_chi2_Cmean_norm": metric_rows(
                delta, covariance_single, covariance_multiplier=1.0 / 25.0
            ),
        }

    with np.load(REALSPACE_NPZ, allow_pickle=False) as realspace:
        real_data = np.asarray(realspace["realspace_xi0_smin50_data"], dtype="f8")
        real_covariance = np.asarray(realspace["realspace_xi0_smin50_covariance"], dtype="f8")
        amplitude = linear_model.amplitude(
            fnl=PARAMETERS["fnl"], b1=PARAMETERS["b1"]
        )
        theory_real_prediction = (
            (linear_model.g_nz * linear_model.pk_dd * amplitude**2)
            @ linear_model.analytic_kernels[0]
            / linear_model.volume
        )[2:]
    realspace_chi2_mean = 25.0 * GaussianMetric(real_covariance).chi2(real_data - theory_real_prediction)

    gates = {
        "same_frozen_P_parameters_used_for_all_theory_inputs": True,
        "measured_moment_smoothing_target_blind": True,
        "all_variances_positive_without_clipping": all(
            np.all(provider.sigma_r2_values > 0.0) and np.all(provider.sigma_t2_values > 0.0)
            for provider in providers.values()
        ),
        "frozen_p_hashes_unchanged": frozen_p_manifest() == frozen_before,
    }
    if not all(gates.values()):
        raise RuntimeError(f"2x2 GSM gates failed: {gates}")

    arrays = {
        "radial_moment_edges": moment_edges,
        "output_s_edges": output_edges,
        "mu_edges": mu_edges,
        "measured_density": measured_density,
        "theory_density": theory_density,
        "measured_v12": measured["v12"],
        "theory_v12": theory["v12_radial"],
        "measured_sigma_r2": measured["sigma_r2"],
        "theory_sigma_r2": theory["sigma_r2_central"],
        "measured_sigma_t2": measured["sigma_t2"],
        "theory_sigma_t2": theory["sigma_t2_one_component"],
        "observed_xi0_rsd": observed[0],
        "observed_xi2_rsd": observed[2],
        "covariance_single": covariance_single,
    }
    for name, (smu, poles) in predictions.items():
        arrays[f"{name}_smu"] = smu
        arrays[f"{name}_xi0"] = poles[0]
        arrays[f"{name}_xi2"] = poles[2]
    atomic_savez(OUT_NPZ, **arrays)
    audit = {
        "task": "task43_run_gsm_density_moment_2x2_ph000_v3",
        "status": "pass",
        "scientific_scope": "no-fit measured/theory density-moment swap; leading theory moments only",
        "phase": "ph000",
        "frozen_P_parameters": PARAMETERS,
        "frozen_p_sha256": frozen_before,
        "inputs": {
            str(path): sha256_file(path)
            for path in (PAIR_NPZ, SUMMARY_NPZ, LINEAR_AUDIT_NPZ, REALSPACE_NPZ, cache_path)
        },
        "settings": SETTINGS,
        "theory_moment_contract": {
            "density": "B(k)^2 Pm on frozen k support",
            "v12": "leading signed density-velocity numerator with B(k) inside the mode sum",
            "variance": "leading matter pair velocity covariance; no shot, old FoG, or fitted extra variance",
            "sigma_u2_one_component": float(theory["sigma_u2_one_component"]),
        },
        "target_blind_smoothing": smoothing,
        "moment_differences_theory_minus_measured": {
            name: {
                "rms": float(np.sqrt(np.mean((theory[key] - measured[name]) ** 2))),
                "max_abs": float(np.max(np.abs(theory[key] - measured[name]))),
            }
            for name, key in (
                ("v12", "v12_radial"),
                ("sigma_r2", "sigma_r2_central"),
                ("sigma_t2", "sigma_t2_one_component"),
            )
        },
        "frozen_P_realspace_xi_chi2_Cmean_smin50": realspace_chi2_mean,
        "closure": rows,
        "swaps_relative_to_measured_density_measured_moments": swaps,
        "interpretation_contract": {
            "density_swap": "isolates the frozen real-space xi template at fixed measured velocity moments",
            "moment_swap": "isolates leading theory velocity moments at fixed measured real-space xi",
            "no_parameter_fit": True,
            "extra_pair_variance": 0.0,
        },
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
                "frozen_P_realspace_xi_chi2_Cmean_smin50": realspace_chi2_mean,
                "moment_differences": audit["moment_differences_theory_minus_measured"],
                "closure": rows,
                "swaps": swaps,
                "elapsed_sec": audit["elapsed_sec"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
