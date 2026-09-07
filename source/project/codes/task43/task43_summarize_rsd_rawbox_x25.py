#!/usr/bin/env python3
"""Summarize the Task 4.3.2 ph000--ph024 periodic raw-box RSD closure.

This stage deliberately uses the measured phase-to-phase diagonal scatter for
stable, transparent closure diagnostics.  It stores the full empirical sample
covariance, but does not invert its rank-deficient 64-dimensional xi0+xi2
block.  The survey-window JAXpower covariance is a separate light-cone stage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

from task43_rsd_common import OUTPUT_ROOT, PHASES, PLOT_ROOT, atomic_savez, atomic_write_json
from task43_rsd_model import FullDiscreteRSDModel, build_cache


SMIN_SCAN = (30.0, 40.0, 50.0, 80.0, 100.0, 120.0)


def measurement_path(phase: str) -> Path:
    return OUTPUT_ROOT / "rawbox" / "summary" / (
        f"task43_rsd_rawbox_AbacusSummit_base_c000_{phase}_mmin1p4e13_clustering.npz"
    )


def concatenate_by_ell(values: np.ndarray, mask: np.ndarray, ells: tuple[int, ...]) -> np.ndarray:
    row = {0: 0, 2: 1}
    return np.concatenate([np.asarray(values[row[int(ell)], mask], dtype="f8") for ell in ells])


def fit_vector(
    model: FullDiscreteRSDModel,
    s: np.ndarray,
    vector: np.ndarray,
    sigma_single: np.ndarray,
    *,
    ells: tuple[int, ...],
    smin: float,
    mean_covariance_factor: float,
) -> dict[str, Any]:
    """Fit one vector and quote curvature errors in the single-volume convention."""

    mask = np.asarray(s >= float(smin))
    data = concatenate_by_ell(vector, mask, ells)
    error_single = concatenate_by_ell(sigma_single, mask, ells)
    positive = error_single[error_single > 0.0]
    if positive.size != error_single.size:
        raise RuntimeError(f"zero rawbox phase scatter in fit vector ells={ells} smin={smin}")
    # The optimizer scaling changes for a mean vector, but the quoted curvature
    # is always rebuilt using C_single as required by the frozen contract.
    error_fit = error_single * float(mean_covariance_factor) ** 0.5

    def prediction(theta: np.ndarray) -> np.ndarray:
        poles = model.evaluate(fnl=theta[0], b1=theta[1], sigma_s=theta[2])
        stacked = np.stack([poles[0], poles[2]])
        return concatenate_by_ell(stacked, mask, ells)

    def residual(theta: np.ndarray) -> np.ndarray:
        return (prediction(theta) - data) / error_fit

    starts = ([0.0, 2.6, 7.0], [-80.0, 2.5, 12.0], [80.0, 2.7, 2.0])
    candidates = [
        least_squares(
            residual,
            start,
            bounds=([-500.0, 0.5, 0.0], [500.0, 5.0, 30.0]),
            max_nfev=1000,
            xtol=1.0e-11,
            ftol=1.0e-11,
            gtol=1.0e-11,
        )
        for start in starts
    ]
    result = min(candidates, key=lambda item: float(np.dot(item.fun, item.fun)))
    theta = np.asarray(result.x, dtype="f8")
    model_vector = prediction(theta)
    residual_single = (model_vector - data) / error_single
    chi2_single = float(np.dot(residual_single, residual_single))
    chi2_used = float(np.dot(result.fun, result.fun))
    dof = int(data.size - theta.size)

    # Re-evaluate a finite-difference Jacobian in the C_single metric even when
    # fitting a phase mean with C_mean=C_single/Nphase.
    steps = np.asarray([0.05, 2.0e-4, 2.0e-3], dtype="f8")
    jacobian = np.empty((data.size, theta.size), dtype="f8")
    for iparam, step in enumerate(steps):
        upper = theta.copy()
        lower = theta.copy()
        upper[iparam] += step
        lower[iparam] -= step
        jacobian[:, iparam] = (prediction(upper) - prediction(lower)) / (2.0 * step * error_single)
    fisher_single = jacobian.T @ jacobian
    covariance_parameter_single = np.linalg.pinv(fisher_single, rcond=1.0e-12)
    error_parameter_single = np.sqrt(np.clip(np.diag(covariance_parameter_single), 0.0, np.inf))
    return {
        "ells": [int(ell) for ell in ells],
        "smin_mpc_h": float(smin),
        "parameters": {"fnl": float(theta[0]), "b1": float(theta[1]), "sigma_s_mpc_h": float(theta[2])},
        "parameter_sigma_single": {
            "fnl": float(error_parameter_single[0]),
            "b1": float(error_parameter_single[1]),
            "sigma_s_mpc_h": float(error_parameter_single[2]),
        },
        "parameter_covariance_single": covariance_parameter_single.tolist(),
        "fnl_pull_single": float(theta[0] / error_parameter_single[0]),
        "chi2_single_convention": chi2_single,
        "chi2_fit_covariance": chi2_used,
        "fit_covariance_factor_relative_to_single": float(mean_covariance_factor),
        "dof_nominal": dof,
        "pte_fit_covariance": float(chi2_distribution.sf(chi2_used, dof)),
        "success": bool(result.success),
        "message": str(result.message),
        "prediction": model_vector.tolist(),
        "normalized_residual_single": residual_single.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "rawbox" / "summary_x25" / "task43_rsd_rawbox_closure_x25.npz",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=PLOT_ROOT / "task43_rsd_rawbox_closure_x25.pdf",
    )
    args = parser.parse_args()
    metadata_path = args.output.with_suffix(".json")
    for path in (args.output, metadata_path, args.plot):
        if path.exists():
            raise FileExistsError(path)

    samples_rsd: list[np.ndarray] = []
    samples_real: list[np.ndarray] = []
    paths: list[str] = []
    ndata: list[int] = []
    for phase in PHASES:
        path = measurement_path(phase)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            phase_file = str(np.asarray(payload["phase"]).item())
            if phase_file != phase:
                raise RuntimeError(f"phase-label mismatch: expected {phase}, got {phase_file} in {path}")
            if not paths:
                s = np.asarray(payload["s"], dtype="f8")
                s_edges = np.asarray(payload["s_edges"], dtype="f8")
            elif not np.array_equal(s_edges, np.asarray(payload["s_edges"], dtype="f8")):
                raise RuntimeError(f"rawbox separation edges differ in {path}")
            samples_rsd.append(np.stack([payload["xi0_rsd"], payload["xi2_rsd"]]).astype("f8"))
            samples_real.append(np.stack([payload["xi0_real"], payload["xi2_real"]]).astype("f8"))
            ndata.append(int(np.asarray(payload["ndata"]).item()))
        paths.append(str(path))

    rsd = np.stack(samples_rsd)
    real = np.stack(samples_real)
    nphase = int(rsd.shape[0])
    if nphase != 25:
        raise RuntimeError(f"Task4.3.2 production requires 25 phases, found {nphase}")
    mean_rsd = np.mean(rsd, axis=0)
    mean_real = np.mean(real, axis=0)
    scatter_rsd = np.std(rsd, axis=0, ddof=1)
    vector_rsd = np.concatenate([rsd[:, 0, :], rsd[:, 1, :]], axis=1)
    covariance_single = np.cov(vector_rsd, rowvar=False, ddof=1)
    covariance_mean = covariance_single / float(nphase)
    covariance_rank = int(np.linalg.matrix_rank(covariance_single))

    model_path = build_cache(zeff=0.725, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000")
    model = FullDiscreteRSDModel(model_path)
    mean_fits: list[dict[str, Any]] = []
    for ells in ((0,), (0, 2)):
        for smin in SMIN_SCAN:
            mean_fits.append(
                fit_vector(
                    model,
                    s,
                    mean_rsd,
                    scatter_rsd,
                    ells=ells,
                    smin=smin,
                    mean_covariance_factor=1.0 / float(nphase),
                )
            )

    primary = next(item for item in mean_fits if item["ells"] == [0] and item["smin_mpc_h"] == 50.0)
    primary_mask = s >= 50.0
    primary_prediction_full = model.evaluate(
        fnl=primary["parameters"]["fnl"],
        b1=primary["parameters"]["b1"],
        sigma_s=primary["parameters"]["sigma_s_mpc_h"],
    )
    phase_fits = [
        fit_vector(
            model,
            s,
            rsd[iphase],
            scatter_rsd,
            ells=(0,),
            smin=50.0,
            mean_covariance_factor=1.0,
        )
        for iphase in range(nphase)
    ]
    phase_fnl = np.asarray([item["parameters"]["fnl"] for item in phase_fits], dtype="f8")
    phase_pte = np.asarray([item["pte_fit_covariance"] for item in phase_fits], dtype="f8")
    sigma_fnl_single = float(primary["parameter_sigma_single"]["fnl"])

    scale_reference = primary
    scale_stability: list[dict[str, Any]] = []
    for item in mean_fits:
        if item["ells"] != [0] or item["smin_mpc_h"] < 50.0:
            continue
        row: dict[str, Any] = {"smin_mpc_h": item["smin_mpc_h"]}
        for key, sigma_key in (("fnl", "fnl"), ("b1", "b1"), ("sigma_s_mpc_h", "sigma_s_mpc_h")):
            delta = float(item["parameters"][key] - scale_reference["parameters"][key])
            sigma_combined = float(
                np.hypot(
                    item["parameter_sigma_single"][sigma_key],
                    scale_reference["parameter_sigma_single"][sigma_key],
                )
            )
            row[f"delta_{key}"] = delta
            row[f"delta_{key}_over_combined_sigma_single"] = delta / sigma_combined
        scale_stability.append(row)

    gates = {
        "all_25_phases_present_and_finite": bool(
            len(paths) == 25 and np.all(np.isfinite(rsd)) and np.all(np.isfinite(real))
        ),
        "empirical_covariance_rank_is_expected_nphase_minus_one": covariance_rank == 24,
        "primary_abs_fnl_over_sigma_single_below_0p3": abs(float(primary["fnl_pull_single"])) < 0.3,
        "primary_mean_pte_above_0p05": float(primary["pte_fit_covariance"]) > 0.05,
        "phase_median_pte_above_0p05": float(np.median(phase_pte)) > 0.05,
        "phase_low_pte_fraction_below_0p20": float(np.mean(phase_pte <= 0.05)) < 0.20,
        "smin50_to100_fnl_shift_below_0p3_sigma_single": abs(
            next(row for row in scale_stability if row["smin_mpc_h"] == 100.0)[
                "delta_fnl_over_combined_sigma_single"
            ]
        )
        < 0.3,
    }
    status = "pass" if all(gates.values()) else "validation_failed"

    atomic_savez(
        args.output,
        s=s,
        s_edges=s_edges,
        phases=np.asarray(PHASES),
        ndata=np.asarray(ndata, dtype="i8"),
        xi02_rsd_by_phase=rsd,
        xi02_real_by_phase=real,
        xi02_rsd_mean=mean_rsd,
        xi02_real_mean=mean_real,
        xi02_rsd_scatter_single=scatter_rsd,
        covariance_xi02_sample_single=covariance_single,
        covariance_xi02_sample_mean=covariance_mean,
        primary_prediction_xi02=np.stack([primary_prediction_full[0], primary_prediction_full[2]]),
        primary_mask=primary_mask,
        primary_phase_fnl=phase_fnl,
        primary_phase_pte=phase_pte,
    )
    metadata = {
        "task": "task43_summarize_rsd_rawbox_x25",
        "status": status,
        "classification": (
            "periodic rawbox RSD closure using diagonal empirical phase scatter; "
            "not the lightcone JAXpower/RR-deconvolved posterior covariance"
        ),
        "phases": list(PHASES),
        "measurement_paths": paths,
        "nphase": nphase,
        "covariance_conventions": {
            "C_single": "sample covariance across 25 independent phases",
            "C_mean": "C_single/25",
            "fit_weighting": "diagonal of C_mean for the phase mean; diagonal of C_single for individual phases",
            "quoted_parameter_errors": "local curvature rebuilt with diagonal C_single",
            "full_sample_covariance_rank": covariance_rank,
            "full_sample_covariance_inverted": False,
        },
        "theory_cache": str(model_path),
        "model": "shell-averaged FullDiscrete Kaiser x squared-Lorentzian FoG, p_fixed=1",
        "primary": primary,
        "mean_fits": mean_fits,
        "phase_primary_fits": dict(zip(PHASES, phase_fits)),
        "phase_summary": {
            "fnl_mean": float(np.mean(phase_fnl)),
            "fnl_std": float(np.std(phase_fnl, ddof=1)),
            "fnl_mean_over_primary_sigma_single": float(np.mean(phase_fnl) / sigma_fnl_single),
            "pte_median": float(np.median(phase_pte)),
            "pte_min": float(np.min(phase_pte)),
            "fraction_pte_le_0p05": float(np.mean(phase_pte <= 0.05)),
        },
        "scale_stability_xi0": scale_stability,
        "gates": gates,
        "output_path": str(args.output),
        "plot_path": str(args.plot),
    }
    atomic_write_json(metadata_path, metadata)

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(7.3, 7.2), sharex=True)
    labels = (r"$\xi_0$", r"$\xi_2$")
    for iell, axis in enumerate(axes):
        for iphase in range(nphase):
            axis.plot(s, s**2 * rsd[iphase, iell], color="0.80", lw=0.45, alpha=0.55)
        axis.errorbar(
            s,
            s**2 * mean_rsd[iell],
            yerr=s**2 * scatter_rsd[iell] / np.sqrt(float(nphase)),
            fmt="o",
            ms=2.8,
            color="#1f4e79",
            label=r"x25 mean $\pm\sqrt{\mathrm{diag}(C_{\rm mean})}$",
        )
        axis.plot(
            s,
            s**2 * primary_prediction_full[2 * iell],
            color="#b22222",
            lw=1.5,
            label=r"primary $\xi_0$ fit, $s_{\min}=50$",
        )
        axis.axvline(50.0, color="0.25", ls="--", lw=0.8)
        axis.axhline(0.0, color="0.45", lw=0.6)
        axis.set_ylabel(rf"$s^2 {labels[iell]}(s)$")
        axis.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel(r"$s\,[h^{-1}{\rm Mpc}]$")
    figure.suptitle(
        rf"Task 4.3.2 rawbox x25: $f_{{\rm NL}}={primary['parameters']['fnl']:.2f}"
        rf"\pm{primary['parameter_sigma_single']['fnl']:.2f}$ (single-volume convention)"
    )
    figure.tight_layout()
    temporary = args.plot.with_name(f".{args.plot.name}.{os.getpid()}.tmp.pdf")
    figure.savefig(temporary)
    plt.close(figure)
    temporary.replace(args.plot)

    print(
        json.dumps(
            {
                "status": status,
                "output": str(args.output),
                "plot": str(args.plot),
                "primary_fnl": primary["parameters"]["fnl"],
                "primary_sigma_fnl_single": primary["parameter_sigma_single"]["fnl"],
                "primary_pte_mean": primary["pte_fit_covariance"],
            },
            sort_keys=True,
        )
    )
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
