#!/usr/bin/env python3
"""Summarize the ph000--ph002 rawbox RSD model-shape pilot."""

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


PHASES = ("ph000", "ph001", "ph002")


def measurement_path(phase: str) -> Path:
    return OUTPUT_ROOT / "rawbox" / "summary" / (
        f"task43_rsd_rawbox_AbacusSummit_base_c000_{phase}_mmin1p4e13_clustering.npz"
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
    error_floor = 0.25 * float(np.median(error))
    error = np.maximum(error, error_floor)

    def residual(theta: np.ndarray) -> np.ndarray:
        prediction = model.evaluate(fnl=theta[0], b1=theta[1], sigma_s=theta[2])
        vector = np.concatenate([prediction[ell][mask] for ell in ells])
        return (vector - data) / error

    starts = ([0.0, 2.4, 4.0], [-100.0, 2.4, 7.0], [100.0, 2.4, 1.0])
    fits = [
        least_squares(
            residual,
            start,
            bounds=([-500.0, 0.5, 0.0], [500.0, 5.0, 30.0]),
            max_nfev=500,
            xtol=1.0e-10,
            ftol=1.0e-10,
            gtol=1.0e-10,
        )
        for start in starts
    ]
    result = min(fits, key=lambda item: float(np.dot(item.fun, item.fun)))
    prediction = model.evaluate(fnl=result.x[0], b1=result.x[1], sigma_s=result.x[2])
    chi2 = float(np.dot(result.fun, result.fun))
    return {
        "ells": list(ells),
        "smin_mpc_h": float(smin),
        "parameters": {"fnl": float(result.x[0]), "b1": float(result.x[1]), "sigma_s_mpc_h": float(result.x[2])},
        "chi2_diagonal_pilot": chi2,
        "ndata": int(data.size),
        "dof_nominal": int(data.size - 3),
        "reduced_chi2_diagonal_pilot": chi2 / max(1, data.size - 3),
        "scatter_floor": error_floor,
        "success": bool(result.success),
        "message": str(result.message),
        "prediction_xi0": prediction[0].tolist(),
        "prediction_xi2": prediction[2].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "rawbox" / "pilot" / "task43_rsd_rawbox_pilot_x3.npz")
    parser.add_argument("--plot", type=Path, default=PLOT_ROOT / "task43_rsd_rawbox_pilot_x3.pdf")
    args = parser.parse_args()
    metadata_path = args.output.with_suffix(".json")
    for path in (args.output, metadata_path, args.plot):
        if path.exists():
            raise FileExistsError(path)

    arrays = []
    paths = []
    for phase in PHASES:
        path = measurement_path(phase)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            if not paths:
                s = np.asarray(data["s"], dtype="f8")
                s_edges = np.asarray(data["s_edges"], dtype="f8")
            elif not np.array_equal(s_edges, np.asarray(data["s_edges"], dtype="f8")):
                raise RuntimeError("rawbox pilot separation edges differ")
            arrays.append(np.stack([data["xi0_rsd"], data["xi2_rsd"]]).astype("f8"))
        paths.append(str(path))
    samples = np.stack(arrays)
    mean = np.mean(samples, axis=0)
    scatter = np.std(samples, axis=0, ddof=1)
    model_path = build_cache(zeff=0.725, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000")
    model = FullDiscreteRSDModel(model_path)
    fits = []
    for ells in ((0,), (0, 2)):
        for smin in (30.0, 50.0, 80.0):
            fits.append(fit_one(model, s, mean, scatter, ells=ells, smin=smin))
    primary = next(item for item in fits if item["ells"] == [0] and item["smin_mpc_h"] == 50.0)
    joint = next(item for item in fits if item["ells"] == [0, 2] and item["smin_mpc_h"] == 50.0)
    stability = abs(
        joint["parameters"]["fnl"]
        - next(item for item in fits if item["ells"] == [0, 2] and item["smin_mpc_h"] == 80.0)["parameters"]["fnl"]
    )
    gates = {
        "all_finite": bool(np.all(np.isfinite(samples))),
        "primary_abs_fnl_below_100": abs(primary["parameters"]["fnl"]) < 100.0,
        "joint_abs_fnl_below_100": abs(joint["parameters"]["fnl"]) < 100.0,
        "joint_reduced_diagonal_chi2_below_3": joint["reduced_chi2_diagonal_pilot"] < 3.0,
        "joint_smin50_to80_fnl_shift_below_50": stability < 50.0,
    }
    status = "pass" if all(gates.values()) else "fail"
    atomic_savez(
        args.output,
        s=s,
        s_edges=s_edges,
        phases=np.asarray(PHASES),
        xi02_rsd_by_phase=samples,
        xi02_rsd_mean=mean,
        xi02_rsd_scatter_single=scatter,
        primary_prediction=np.stack([primary["prediction_xi0"], primary["prediction_xi2"]]),
        joint_prediction=np.stack([joint["prediction_xi0"], joint["prediction_xi2"]]),
    )
    metadata = {
        "task": "task43_summarize_rsd_rawbox_pilot",
        "status": status,
        "classification": "x3 model-shape and implementation gate only; not a quoted covariance/posterior",
        "phases": list(PHASES),
        "measurement_paths": paths,
        "theory_cache": str(model_path),
        "model": "shell-averaged FullDiscrete Kaiser x Lorentzian FoG, p_fixed=1",
        "pilot_error": "per-bin x3 single-realization scatter with a 0.25 median-scatter floor",
        "fits": fits,
        "gates": gates,
        "output_path": str(args.output),
        "plot_path": str(args.plot),
    }
    atomic_write_json(metadata_path, metadata)

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    labels = (r"$\xi_0$", r"$\xi_2$")
    prediction = np.asarray(joint["prediction_xi0"]), np.asarray(joint["prediction_xi2"])
    for index, axis in enumerate(axes):
        for iphase, phase in enumerate(PHASES):
            axis.plot(s, s**2 * samples[iphase, index], color="0.75", lw=0.8, alpha=0.8, label="individual phases" if iphase == 0 else None)
        axis.errorbar(s, s**2 * mean[index], yerr=s**2 * scatter[index], fmt="o", ms=3.0, color="#1f4e79", label="x3 mean ± single-phase scatter")
        axis.plot(s, s**2 * prediction[index], color="#b22222", lw=1.6, label="joint fit, smin=50")
        axis.axvline(50.0, color="0.25", ls="--", lw=0.9)
        axis.axhline(0.0, color="0.4", lw=0.6)
        axis.set_ylabel(rf"$s^2 {labels[index]}(s)$")
        axis.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel(r"$s\,[h^{-1}{\rm Mpc}]$")
    figure.suptitle(
        rf"Task 4.3.2 rawbox x3 pilot: $f_{{\rm NL}}={joint['parameters']['fnl']:.1f}$, "
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
