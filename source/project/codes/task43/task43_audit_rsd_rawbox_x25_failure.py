#!/usr/bin/env python3
"""Explain the failed x25 rawbox gate without redefining the frozen primary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution
from scipy.stats import f as f_distribution

from task43_fit_rsd_rawbox_x25 import FastRSDModel, load_x25
from task43_rsd_common import OUTPUT_ROOT, S_EDGES, atomic_write_json
from task43_rsd_model import FullDiscreteRSDModel


PARAMETERS = ("fNL", "b1", "sigma_s")


def fit_rsd(
    model: FastRSDModel,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    smin: float = 50.0,
) -> dict[str, Any]:
    s = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    mask = s >= float(smin)
    chol = np.linalg.cholesky(covariance[np.ix_(mask, mask)])

    def residual(theta: np.ndarray) -> np.ndarray:
        return np.linalg.solve(chol, mean[mask] - model.evaluate(theta)[0][mask])

    solutions = [
        least_squares(residual, start, bounds=([-500.0, 0.5, 0.0], [500.0, 5.0, 30.0]))
        for start in ([0.0, 2.5, 8.0], [-50.0, 2.3, 4.0], [50.0, 2.7, 12.0])
    ]
    result = min(solutions, key=lambda item: float(item.fun @ item.fun))
    chi2_mean = float(25.0 * (result.fun @ result.fun))
    dof = int(np.count_nonzero(mask) - 3)
    return {
        "theta": {name: float(value) for name, value in zip(PARAMETERS, result.x, strict=True)},
        "chi2_mean": chi2_mean,
        "dof": dof,
        "pte_chi2_nominal": float(chi2_distribution.sf(chi2_mean, dof)),
    }


def coarsened_empirical_tests(
    model: FastRSDModel,
    samples: np.ndarray,
    *,
    group_sizes: tuple[int, ...] = (2, 3, 5, 6),
) -> list[dict[str, Any]]:
    s = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    mask = s >= 50.0
    values = np.asarray(samples[:, mask], dtype="f8")
    offset = int(np.flatnonzero(mask)[0])
    rows = []
    for group in group_sizes:
        if values.shape[1] % int(group):
            continue
        transform = []
        for start in range(0, values.shape[1], int(group)):
            ids = np.arange(start, start + int(group))
            low = S_EDGES[offset + ids]
            high = S_EDGES[offset + ids + 1]
            volume = high**3 - low**3
            row = np.zeros(values.shape[1], dtype="f8")
            row[ids] = volume / np.sum(volume)
            transform.append(row)
        transform_array = np.stack(transform)
        coarse = values @ transform_array.T
        mean = np.mean(coarse, axis=0)
        sample_covariance = np.cov(coarse, rowvar=False, ddof=1)
        chol = np.linalg.cholesky(sample_covariance)

        def residual(theta: np.ndarray) -> np.ndarray:
            prediction = model.evaluate(theta)[0][mask] @ transform_array.T
            return np.linalg.solve(chol, mean - prediction)

        solutions = [
            least_squares(residual, start, bounds=([-500.0, 0.5, 0.0], [500.0, 5.0, 30.0]))
            for start in ([0.0, 2.5, 8.0], [-50.0, 2.3, 4.0], [50.0, 2.7, 12.0])
        ]
        result = min(solutions, key=lambda item: float(item.fun @ item.fun))
        nreal, ndim = coarse.shape
        nparam = 3
        dof = ndim - nparam
        t2 = float(nreal * (result.fun @ result.fun))
        hartlap = float((nreal - ndim - 2) / (nreal - 1))
        fixed_model_f = float((nreal - ndim) / (ndim * (nreal - 1)) * t2)
        rows.append(
            {
                "group_size_original_bins": int(group),
                "ndata_coarse": int(ndim),
                "theta": {name: float(value) for name, value in zip(PARAMETERS, result.x, strict=True)},
                "hotelling_t2": t2,
                "fixed_model_hotelling_pte_conservative_reference": float(
                    f_distribution.sf(fixed_model_f, ndim, nreal - ndim)
                ),
                "hartlap_factor": hartlap,
                "hartlap_chi2_heuristic": float(hartlap * t2),
                "hartlap_chi2_heuristic_dof_after_fit": int(dof),
                "hartlap_chi2_heuristic_pte": float(
                    chi2_distribution.sf(hartlap * t2, dof)
                ),
            }
        )
    return rows


def real_space_control(
    exact: FullDiscreteRSDModel,
    samples: np.ndarray,
    *,
    nbar: float,
) -> list[dict[str, Any]]:
    s = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    mean = np.mean(samples, axis=0)
    kernel = exact.kernels[exact.ell_values.index(0)]
    projection = exact.g_nz[:, None] * kernel / float(exact.volume)

    def evaluate(theta: np.ndarray) -> np.ndarray:
        fnl, b1 = map(float, theta)
        q = fnl * 2.0 * 1.686 * (b1 - 1.0)
        return (exact.pk_dd * (b1 + q * exact.alpha) ** 2) @ projection

    b1_cov = 2.5
    total = exact.pk_dd * b1_cov**2 + 1.0 / float(nbar)
    weight = 2.0 * exact.g_nz * total**2 / float(exact.volume) ** 2
    covariance = kernel.T @ (weight[:, None] * kernel)
    rows = []
    for smin in (30.0, 40.0, 50.0, 80.0, 100.0, 120.0):
        mask = s >= smin
        chol = np.linalg.cholesky(covariance[np.ix_(mask, mask)])

        def residual(theta: np.ndarray) -> np.ndarray:
            return np.linalg.solve(chol, mean[mask] - evaluate(theta)[mask])

        solutions = [
            least_squares(residual, start, bounds=([-500.0, 0.5], [500.0, 5.0]))
            for start in ([0.0, 2.5], [-50.0, 2.3], [50.0, 2.7])
        ]
        result = min(solutions, key=lambda item: float(item.fun @ item.fun))
        chi2_mean = float(25.0 * (result.fun @ result.fun))
        dof = int(np.count_nonzero(mask) - 2)
        rows.append(
            {
                "smin_mpc_h": smin,
                "theta": {"fNL": float(result.x[0]), "b1": float(result.x[1])},
                "chi2_mean": chi2_mean,
                "dof": dof,
                "pte": float(chi2_distribution.sf(chi2_mean, dof)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--closure-prefix",
        type=Path,
        default=OUTPUT_ROOT / "rawbox" / "closure" / "task43_rsd_rawbox_x25_fulldiscrete_lorentzian",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "audits" / "task43_rsd_rawbox_x25_failure_audit.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    closure_json = args.closure_prefix.with_suffix(".json")
    closure_npz = args.closure_prefix.with_suffix(".npz")
    closure = json.loads(closure_json.read_text(encoding="utf-8"))
    if closure.get("status") != "validation_failed":
        raise RuntimeError("failure audit requires an immutable validation_failed closure")
    with np.load(closure_npz, allow_pickle=False) as stored:
        covariance = np.asarray(stored["covariance_single_realization"], dtype="f8")
    stacks, metadata, _ = load_x25()
    nbin = S_EDGES.size - 1
    analytic_xi0 = covariance[:nbin, :nbin]
    scatter_xi0 = np.cov(stacks["xi0_rsd"], rowvar=False, ddof=1)
    analytic_sigma = np.sqrt(np.diag(analytic_xi0))
    scatter_sigma = np.sqrt(np.diag(scatter_xi0))
    analytic_correlation = analytic_xi0 / np.outer(analytic_sigma, analytic_sigma)
    hybrid_covariance = analytic_correlation * np.outer(scatter_sigma, scatter_sigma)
    empirical_diagonal = np.diag(scatter_sigma**2)
    cache = Path(closure["model"]["theory_cache"])
    exact = FullDiscreteRSDModel(cache, nmu=int(closure["model"]["nmu"]))
    model = FastRSDModel(exact, sigma_step=float(closure["model"]["sigma_surrogate_grid_step_mpc_h"]))
    mean = np.mean(stacks["xi0_rsd"], axis=0)
    standardized = (stacks["xi0_rsd"] - mean) / analytic_sigma
    empirical_correlation = np.corrcoef(standardized, rowvar=False)
    correlation_delta = empirical_correlation - analytic_correlation
    nbar = float(np.mean([row["nbar_h3_mpc3_from_npz"] for row in metadata]))
    primary = closure["nominal"]
    payload = {
        "task": "task43_audit_rsd_rawbox_x25_failure",
        "status": "pass",
        "scientific_status": "validation_failed_covariance_and_model_shape_unresolved",
        "frozen_primary_unchanged": True,
        "closure_json": str(closure_json),
        "closure_npz": str(closure_npz),
        "primary_recap": {
            "fNL_posterior": closure["mcmc_nominal"]["posterior"]["fNL"],
            "null_ratio": closure["primary_null_abs_median_over_sigma68_single"],
            "mean_chi2": primary["chi2_mean_covariance"],
            "mean_dof": primary["dof"],
            "mean_pte": primary["pte_mean_covariance"],
            "phase_ensemble": closure["phase_ensemble"],
            "scale_stability": closure["scale_stability"],
        },
        "analytic_covariance_ensemble_gate": {
            "phase_chi2_mean_full_xi02": closure["covariance_policy"]["x25_scatter_comparison"]["phase_chi2_mean_about_empirical_mean_full_xi02"],
            "expected_phase_chi2_mean_full_xi02": closure["covariance_policy"]["x25_scatter_comparison"]["expected_phase_chi2_mean_about_empirical_mean"],
            "ratio_observed_to_expected": float(
                closure["covariance_policy"]["x25_scatter_comparison"]["phase_chi2_mean_about_empirical_mean_full_xi02"]
                / closure["covariance_policy"]["x25_scatter_comparison"]["expected_phase_chi2_mean_about_empirical_mean"]
            ),
            "xi0_sample_std_over_gaussian_sigma_median": closure["covariance_policy"]["x25_scatter_comparison"]["sample_std_over_gaussian_sigma_xi0_median"],
            "correlation_delta_frobenius": float(np.linalg.norm(correlation_delta)),
            "correlation_delta_absmax": float(np.max(np.abs(correlation_delta))),
            "interpretation": "diagonal scale is close, but the Gaussian precision fails the x25 multivariate scatter gate; primary GoF cannot be assigned to model shape alone",
        },
        "post_failure_covariance_diagnostics_not_new_primary": {
            "analytic_correlation_empirical_diagonal": fit_rsd(model, mean, hybrid_covariance),
            "empirical_diagonal": fit_rsd(model, mean, empirical_diagonal),
            "coarsened_empirical_covariance": coarsened_empirical_tests(model, stacks["xi0_rsd"]),
            "warning": "post-failure diagnostics; finite x25 covariance and rebinning were not the frozen primary",
        },
        "paired_real_space_control": {
            "fits": real_space_control(exact, stacks["xi0_real"], nbar=nbar),
            "interpretation": "the same linear PNG template also fails the analytic-Cmean gate below 120 Mpc/h without RSD, so the discrepancy is not uniquely generated by the RSD coordinate map",
        },
        "decision": {
            "rawbox_gate_passed": False,
            "task44_model_greenlight": False,
            "lightcone_next_role": "diagnostic discrimination of survey window/GIC and covariance, not a rescue of the failed frozen rawbox primary",
            "required_before_science_claim": [
                "validate a non-Gaussian/empirical covariance or precision against phase scatter",
                "add and prevalidate nonlinear BAO/broadband or scale-dependent bias freedom, or adopt a justified large-scale cut",
                "repeat null and scale-stability gates without post-hoc covariance selection",
            ],
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"status": "pass", "scientific_status": payload["scientific_status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
