#!/usr/bin/env python3
"""Final machine audit for the Task 4.3.2 lightcone P0-vs-xi0 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2 as chi2_distribution

from task43_fit_rsd_lightcone_pk0_vs_xi0_smin50 import (
    DEFAULT_PK_PAYLOAD,
    WindowConvolvedPk0Model,
    fit_map,
    load_pk,
)
from task43_rsd_common import OUTPUT_ROOT, PHASES, atomic_savez, atomic_write_json, sha256_file
from task43_rsd_lightcone_pk0_contract import (
    TASK43_REALSPACE_FIT_EDGES,
    TASK43_REALSPACE_OBSERVED_KMIN,
    WINDOW_THEORY_KMIN,
)


DEFAULT_COMPARISON = (
    OUTPUT_ROOT
    / "lightcone/comparison/"
    "task43_rsd_lightcone_x25_pk0_vs_xi0_smin50_task43realspacewindow_l0only_longchain.json"
)
DEFAULT_OUTPUT_PREFIX = (
    OUTPUT_ROOT
    / "audits/"
    "task43_rsd_lightcone_pk0_vs_xi0_smin50_task43realspacewindow_l0only_audit"
)


def validate_comparison(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise RuntimeError(f"comparison is not complete: {path}")
    for key in ("output_npz", "output_fit_pdf", "output_contour_pdf"):
        product = Path(result[key])
        if not product.is_file() or sha256_file(product) != result[f"{key}_sha256"]:
            raise RuntimeError(f"comparison provenance gate failed for {key}: {product}")
    contour_json = Path(result["output_contour_pdf"]).with_suffix(".json")
    contour = json.loads(contour_json.read_text(encoding="utf-8"))
    if contour.get("status") != "pass" or contour.get("comparison_npz_sha256") != result["output_npz_sha256"]:
        raise RuntimeError("contour metadata gate failed")
    result["path"] = str(path)
    result["sha256"] = sha256_file(path)
    result["contour_json"] = str(contour_json)
    result["contour_json_sha256"] = sha256_file(contour_json)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--pk-payload", type=Path, default=DEFAULT_PK_PAYLOAD)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args()
    output_npz, output_json = args.output_prefix.with_suffix(".npz"), args.output_prefix.with_suffix(".json")
    if output_npz.exists() or output_json.exists():
        raise FileExistsError(f"immutable audit output exists: {output_npz} / {output_json}")

    comparison = validate_comparison(args.comparison)
    pk = load_pk(args.pk_payload)
    model = WindowConvolvedPk0Model(args.pk_payload)
    surrogate = model.validate()
    if surrogate["status"] != "pass":
        raise RuntimeError(f"forward model validation failed: {surrogate}")
    covariance = np.asarray(pk["covariance_single_realization"], dtype="f8")
    phase_rows: list[dict[str, Any]] = []
    phase_predictions = []
    for phase, values in zip(np.asarray(pk["phases"]).astype(str), np.asarray(pk["pk_stack"]), strict=True):
        fit = fit_map(model, np.asarray(values, dtype="f8"), covariance)
        phase_rows.append(
            {
                "phase": phase,
                "chi2": float(fit["chi2_single_covariance"]),
                "dof_nominal": int(fit["dof"]),
                "pte_nominal": float(chi2_distribution.sf(fit["chi2_single_covariance"], fit["dof"])),
                "theta": fit["theta"],
                "at_parameter_boundary": bool(fit["at_parameter_boundary"]),
            }
        )
        phase_predictions.append(np.asarray(fit["prediction"], dtype="f8"))

    phase_chi2 = np.asarray([row["chi2"] for row in phase_rows], dtype="f8")
    phase_theta = np.asarray(
        [[row["theta"][name] for name in ("fNL", "b1", "sigma_s", "sn0")] for row in phase_rows],
        dtype="f8",
    )
    aggregate_chi2 = float(np.sum(phase_chi2))
    aggregate_dof = int(len(PHASES) * (TASK43_REALSPACE_FIT_EDGES.shape[0] - 4))
    aggregate_pte = float(chi2_distribution.sf(aggregate_chi2, aggregate_dof))

    pk_posterior = comparison["pk0"]["mcmc"]["posterior"]
    xi_posterior = comparison["xi0"]["posterior"]
    pk_null_ratio = abs(float(pk_posterior["fNL"]["q50"])) / float(pk_posterior["fNL"]["sigma68"])
    xi_null_ratio = abs(float(xi_posterior["fNL"]["q50"])) / float(xi_posterior["fNL"]["sigma68"])
    pk_mean_pte = float(comparison["pk0"]["nominal"]["pte_mean_covariance"])
    xi_mean_pte = float(comparison["xi0"]["mean_goodness"]["pte"])
    pk_mean = np.asarray(pk["pk_mean"], dtype="f8")
    pk_prediction = np.asarray(comparison["pk0"]["nominal"]["prediction"], dtype="f8")
    sigma_mean = np.sqrt(np.diag(covariance) / len(PHASES))
    normalized_mean_residual = (pk_mean - pk_prediction) / sigma_mean

    gates = {
        "provenance_and_hashes": True,
        "task43_realspace_kmin_contract": bool(
            np.isclose(float(pk["kmin_fit_observed"]), TASK43_REALSPACE_OBSERVED_KMIN, rtol=0.0, atol=1.0e-15)
            and np.isclose(float(pk["window_theory_kmin"]), WINDOW_THEORY_KMIN, rtol=0.0, atol=1.0e-15)
        ),
        "window_surrogate_direct_agreement": surrogate["status"] == "pass",
        "pk0_mcmc_convergence": bool(all(comparison["pk0"]["mcmc"]["gates"].values())),
        "pk0_null_ratio_below_0p3": bool(pk_null_ratio < 0.3),
        "pk0_mean_pte_above_0p05": bool(pk_mean_pte > 0.05),
        "pk0_phase_profile_aggregate_pte_above_0p05": bool(aggregate_pte > 0.05),
        "xi0_mcmc_convergence": bool(all(comparison["xi0"]["convergence"].values())),
        "xi0_null_ratio_below_0p3": bool(xi_null_ratio < 0.3),
        "xi0_mean_pte_above_0p05": bool(xi_mean_pte > 0.05),
    }
    pk_science_pass = bool(
        gates["pk0_mcmc_convergence"]
        and gates["pk0_null_ratio_below_0p3"]
        and gates["pk0_mean_pte_above_0p05"]
        and gates["pk0_phase_profile_aggregate_pte_above_0p05"]
    )
    xi_science_pass = bool(
        gates["xi0_mcmc_convergence"]
        and gates["xi0_null_ratio_below_0p3"]
        and gates["xi0_mean_pte_above_0p05"]
    )

    atomic_savez(
        output_npz,
        phases=np.asarray(PHASES),
        phase_profile_chi2=phase_chi2,
        phase_profile_theta=phase_theta,
        phase_profile_predictions=np.asarray(phase_predictions, dtype="f8"),
        pk0_normalized_mean_residual=normalized_mean_residual,
    )
    payload = {
        "task": "task43_audit_rsd_lightcone_pk0_vs_xi0",
        "status": "complete",
        "science_validation": "pass" if pk_science_pass and xi_science_pass else "validation_failed",
        "task44_greenlight": False,
        "scope": "RSD lightcone P0(kmax=0.10) vs formal-GIC xi0(smin=50), observed ell=0 only",
        "comparison_json": comparison["path"],
        "comparison_json_sha256": comparison["sha256"],
        "comparison_npz": comparison["output_npz"],
        "comparison_npz_sha256": comparison["output_npz_sha256"],
        "fit_pdf": comparison["output_fit_pdf"],
        "fit_pdf_sha256": comparison["output_fit_pdf_sha256"],
        "contour_pdf": comparison["output_contour_pdf"],
        "contour_pdf_sha256": comparison["output_contour_pdf_sha256"],
        "contour_json": comparison["contour_json"],
        "contour_json_sha256": comparison["contour_json_sha256"],
        "pk_payload": str(args.pk_payload),
        "pk_payload_sha256": pk["sha256"],
        "k_contract": {
            "measurement_grid": [0.001, 0.3001, 0.002],
            "observed_fit_kmin": float(pk["kmin_fit_observed"]),
            "window_theory_kmin": float(pk["window_theory_kmin"]),
            "fit_edges": np.asarray(pk["k_edges"], dtype="f8").tolist(),
            "kmax_fit": float(pk["kmax_fit"]),
        },
        "pk0": {
            "posterior": pk_posterior,
            "null_ratio_abs_median_over_sigma68": pk_null_ratio,
            "mean_goodness": comparison["pk0"]["nominal"],
            "mean_normalized_residual_absmax": float(np.max(np.abs(normalized_mean_residual))),
            "phase_profile_aggregate": {
                "chi2": aggregate_chi2,
                "dof_nominal": aggregate_dof,
                "pte_nominal": aggregate_pte,
                "mean_chi2": float(np.mean(phase_chi2)),
                "parameter_boundary_count": int(sum(row["at_parameter_boundary"] for row in phase_rows)),
                "note": "profile fit diagnostic; frequent sigma_s/sn0 boundaries reflect weak per-phase nuisance constraints",
            },
            "phase_profiles": phase_rows,
            "science_validation": "pass" if pk_science_pass else "validation_failed",
        },
        "xi0": {
            "posterior": xi_posterior,
            "null_ratio_abs_median_over_sigma68": xi_null_ratio,
            "mean_goodness": comparison["xi0"]["mean_goodness"],
            "science_validation": "pass" if xi_science_pass else "validation_failed",
        },
        "comparison": comparison["comparison"],
        "covariance_scatter_diagnostic": comparison["covariance"]["x25_scatter_diagnostic"],
        "gates": gates,
        "scale_stability": {
            "status": "not_evaluated_in_user_frozen_single-cut_comparison",
            "frozen_scales": {"pk0_kmax": 0.10, "xi0_smin": 50.0, "xi0_smax": 350.0},
        },
        "interpretation": (
            "P0 mean and phase-profile shapes pass, but its fNL null ratio 0.318 narrowly misses the preregistered "
            "0.3 gate. xi0 independently fails both its null-ratio and mean-shape PTE gates. Their fNL centers "
            "agree closely, so the result is validation_failed rather than a P0-vs-xi0 inconsistency."
        ),
        "output_npz": str(output_npz),
        "output_npz_sha256": sha256_file(output_npz),
    }
    atomic_write_json(output_json, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "science_validation": payload["science_validation"],
                "pk0_null_ratio": pk_null_ratio,
                "pk0_mean_pte": pk_mean_pte,
                "pk0_phase_pte": aggregate_pte,
                "xi0_null_ratio": xi_null_ratio,
                "xi0_mean_pte": xi_mean_pte,
                "output": str(output_json),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
