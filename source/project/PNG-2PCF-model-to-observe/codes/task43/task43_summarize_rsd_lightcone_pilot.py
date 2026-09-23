#!/usr/bin/env python3
"""Summarize the ph000--ph002 radial-LOS light-cone RSD pilot."""

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

from task43_rsd_common import OUTPUT_ROOT, PLOT_ROOT, atomic_savez, atomic_write_json
from task43_rsd_model import FullDiscreteRSDModel, build_cache


PILOT_PHASES = ("ph000", "ph001", "ph002")
SMIN_SCAN = (30.0, 40.0, 50.0, 80.0, 100.0, 120.0)


def measurement_path(phase: str) -> Path:
    return OUTPUT_ROOT / "lightcone" / "xi" / (
        f"task43_rsd_xi0_AbacusSummit_base_c000_{phase}_mmin1p4e13_"
        "zobs0p6_0p8_x25_s30_350_ds10.npz"
    )


def fit_one(
    model: FullDiscreteRSDModel,
    s: np.ndarray,
    mean: np.ndarray,
    scatter: np.ndarray,
    *,
    ells: tuple[int, ...],
    smin: float,
) -> dict[str, Any]:
    mask = s >= float(smin)
    row = {0: 0, 2: 1}
    data = np.concatenate([mean[row[ell], mask] for ell in ells])
    error = np.concatenate([scatter[row[ell], mask] for ell in ells])
    positive = error[error > 0.0]
    if not positive.size:
        raise RuntimeError("pilot scatter is identically zero")
    error_floor = 0.25 * float(np.median(positive))
    error = np.maximum(error, error_floor)

    def prediction(theta: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        poles = model.evaluate(fnl=theta[0], b1=theta[1], sigma_s=theta[2])
        return np.concatenate([poles[ell][mask] for ell in ells]), poles

    def residual(theta: np.ndarray) -> np.ndarray:
        vector, _ = prediction(theta)
        return (vector - data) / error

    candidates = [
        least_squares(
            residual,
            start,
            bounds=([-500.0, 0.5, 0.0], [500.0, 5.0, 30.0]),
            max_nfev=700,
            xtol=1.0e-10,
            ftol=1.0e-10,
            gtol=1.0e-10,
        )
        for start in ([0.0, 2.6, 7.0], [-100.0, 2.5, 12.0], [100.0, 2.7, 2.0])
    ]
    result = min(candidates, key=lambda item: float(np.dot(item.fun, item.fun)))
    _, poles = prediction(result.x)
    chi2 = float(np.dot(result.fun, result.fun))
    return {
        "ells": list(ells),
        "smin_mpc_h": float(smin),
        "parameters": {
            "fnl": float(result.x[0]),
            "b1": float(result.x[1]),
            "sigma_s_mpc_h": float(result.x[2]),
        },
        "chi2_diagonal_pilot": chi2,
        "ndata": int(data.size),
        "dof_nominal": int(data.size - 3),
        "reduced_chi2_diagonal_pilot": chi2 / max(1, data.size - 3),
        "scatter_floor": error_floor,
        "success": bool(result.success),
        "prediction_xi0": poles[0].tolist(),
        "prediction_xi2": poles[2].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "lightcone" / "pilot" / "task43_rsd_lightcone_pilot_x3.npz",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=PLOT_ROOT / "task43_rsd_lightcone_pilot_x3.pdf",
    )
    args = parser.parse_args()
    metadata_path = args.output.with_suffix(".json")
    for path in (args.output, metadata_path, args.plot):
        if path.exists():
            raise FileExistsError(path)

    samples: list[np.ndarray] = []
    zeff: list[float] = []
    paths: list[str] = []
    for phase in PILOT_PHASES:
        path = measurement_path(phase)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            if not paths:
                s = np.asarray(payload["s"], dtype="f8")
                s_edges = np.asarray(payload["s_edges"], dtype="f8")
            elif not np.array_equal(s_edges, np.asarray(payload["s_edges"], dtype="f8")):
                raise RuntimeError("lightcone pilot separation edges differ")
            samples.append(np.asarray(payload["xi_multipoles"], dtype="f8"))
            zeff.append(float(np.asarray(payload["zeff"]).item()))
        paths.append(str(path))
    stack = np.stack(samples)
    mean = np.mean(stack, axis=0)
    scatter = np.std(stack, axis=0, ddof=1)
    zeff_mean = float(np.mean(zeff))
    model_path = build_cache(zeff=zeff_mean, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000")
    model = FullDiscreteRSDModel(model_path)
    fits = [
        fit_one(model, s, mean, scatter, ells=ells, smin=smin)
        for ells in ((0,), (0, 2))
        for smin in SMIN_SCAN
    ]
    primary = next(item for item in fits if item["ells"] == [0] and item["smin_mpc_h"] == 50.0)
    joint = next(item for item in fits if item["ells"] == [0, 2] and item["smin_mpc_h"] == 80.0)
    joint100 = next(item for item in fits if item["ells"] == [0, 2] and item["smin_mpc_h"] == 100.0)
    gates = {
        "all_finite": bool(np.all(np.isfinite(stack))),
        "all_rr_products_present": len(paths) == 3,
        "primary_abs_fnl_below_100": abs(primary["parameters"]["fnl"]) < 100.0,
        "joint_smin80_abs_fnl_below_100": abs(joint["parameters"]["fnl"]) < 100.0,
        "joint_smin80_reduced_diagonal_chi2_below_3": joint["reduced_chi2_diagonal_pilot"] < 3.0,
        "joint_smin80_to100_fnl_shift_below_50": abs(
            joint["parameters"]["fnl"] - joint100["parameters"]["fnl"]
        )
        < 50.0,
    }
    status = "pass" if all(gates.values()) else "validation_failed"
    atomic_savez(
        args.output,
        s=s,
        s_edges=s_edges,
        phases=np.asarray(PILOT_PHASES),
        zeff_by_phase=np.asarray(zeff),
        xi02_by_phase=stack,
        xi02_mean=mean,
        xi02_scatter_single=scatter,
        primary_prediction=np.stack([primary["prediction_xi0"], primary["prediction_xi2"]]),
        joint_prediction=np.stack([joint["prediction_xi0"], joint["prediction_xi2"]]),
    )
    metadata = {
        "task": "task43_summarize_rsd_lightcone_pilot",
        "status": status,
        "classification": "x3 radial-LOS model-shape gate only; not a covariance or quoted posterior",
        "phases": list(PILOT_PHASES),
        "measurement_paths": paths,
        "zeff_by_phase": zeff,
        "zeff_mean": zeff_mean,
        "theory_cache": str(model_path),
        "model": "shell-averaged FullDiscrete Kaiser x squared-Lorentzian FoG, p_fixed=1",
        "pilot_error": "per-bin x3 single-realization scatter with a 0.25 median-scatter floor",
        "fits": fits,
        "gates": gates,
        "output_path": str(args.output),
        "plot_path": str(args.plot),
    }
    atomic_write_json(metadata_path, metadata)

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    prediction = np.asarray(joint["prediction_xi0"]), np.asarray(joint["prediction_xi2"])
    for iell, axis in enumerate(axes):
        for iphase, phase in enumerate(PILOT_PHASES):
            axis.plot(
                s,
                s**2 * stack[iphase, iell],
                color="0.75",
                lw=0.8,
                alpha=0.8,
                label="individual phases" if iphase == 0 else None,
            )
        axis.errorbar(
            s,
            s**2 * mean[iell],
            yerr=s**2 * scatter[iell],
            fmt="o",
            ms=3.0,
            color="#1f4e79",
            label="x3 mean ± single-phase scatter",
        )
        axis.plot(s, s**2 * prediction[iell], color="#b22222", lw=1.6, label="joint fit, smin=80")
        axis.axvline(80.0, color="0.25", ls="--", lw=0.9)
        axis.axhline(0.0, color="0.4", lw=0.6)
        axis.set_ylabel((r"$s^2\xi_0(s)$", r"$s^2\xi_2(s)$")[iell])
        axis.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel(r"$s\,[h^{-1}{\rm Mpc}]$")
    figure.suptitle(
        rf"Task 4.3.2 lightcone x3: $f_{{\rm NL}}={joint['parameters']['fnl']:.1f}$, "
        rf"$b_1={joint['parameters']['b1']:.3f}$, $\sigma_s={joint['parameters']['sigma_s_mpc_h']:.2f}$"
    )
    figure.tight_layout()
    temporary = args.plot.with_name(f".{args.plot.name}.{os.getpid()}.tmp.pdf")
    figure.savefig(temporary)
    plt.close(figure)
    temporary.replace(args.plot)
    print(json.dumps({"status": status, "output": str(args.output), "plot": str(args.plot)}, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
