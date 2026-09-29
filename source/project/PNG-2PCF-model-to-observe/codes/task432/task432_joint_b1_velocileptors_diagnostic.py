#!/usr/bin/env python3
"""Diagnostic for the Task432 joint-b1 displacement.

This file intentionally does not modify or replace the formal full-RIC chains.
It compares the frozen full-RIC hybrid model with the already validated
velocileptors xi emulator under the same data/covariance vectors.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
COMPILED = ROOT / "outputs/task43_outputs/rsd_validation/task432_model_repair/full_ric_p02xi02_v1/pilot/mean4xi_mean2P_prodrr_stochastic/engine.npz"
VEL_EMULATOR = ROOT / "outputs/task43_outputs/rsd_validation/task432_model_repair/lightcone_png_velocileptors_gsm_longchain/lightcone_xi_emulator.npz"
COVARIANCES = {
    "ezmock1000": ROOT / "outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_ezmock1000_0918_contract_v1/frozen_inputs.npz",
    "jaxpower": ROOT / "outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_jaxpower_0918_contract_v1/frozen_inputs.npz",
}
OUT = ROOT / "outputs/task43_outputs/rsd_validation/task432_model_repair/joint_b1_velocileptors_diagnostic/summary.json"

from task432_full_ric_compile import Engine  # noqa: E402
from task432_hybrid_jaxpower_0918_contract import Metric  # noqa: E402


def fit_ml(data, covariance, predict, lower, upper, start):
    metric = Metric(covariance)
    def residual(theta):
        return metric.white(np.asarray(data, dtype="f8") - np.asarray(predict(theta), dtype="f8"))
    result = least_squares(
        residual,
        np.asarray(start, dtype="f8"),
        bounds=(np.asarray(lower, dtype="f8"), np.asarray(upper, dtype="f8")),
        x_scale="jac",
        max_nfev=1500,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-11,
    )
    return {"theta": result.x.tolist(), "chi2": float(result.fun @ result.fun), "success": bool(result.success)}


def main():
    from task432_full_ric_compile import Engine

    velocity_payload = np.load(VEL_EMULATOR, allow_pickle=False)
    velocity_interp = RegularGridInterpolator(
        (velocity_payload["f_grid"], velocity_payload["b_grid"]),
        velocity_payload["values"],
        bounds_error=True,
    )
    result = {
        "status": "diagnostic_only",
        "scope": "Frozen 74-vector, full-RIC P model, same EZmock1000/jaxpower C_single; xi replacement uses the existing validated velocileptors/GSM emulator.",
        "formal_result_unchanged": True,
        "caveat": "The imported xi emulator was produced under the previous formal-GIC contract. Existing full-RIC sn0 response is added for sensitivity only; no new full-RIC velocileptors chain is promoted.",
        "cross_scan": {},
        "split_b": {},
        "model_comparison": {},
    }

    for name, path in COVARIANCES.items():
        covariance = np.load(path, allow_pickle=False)["covariance"]
        p_engine = Engine("p02", COMPILED)
        x_engine = Engine("xi02", COMPILED)
        j_engine = Engine("joint", COMPILED)
        data = np.asarray(j_engine.data, dtype="f8")
        p_data, x_data = data[:22], data[22:]

        def p_predict(theta):
            return p_engine.predict(np.asarray(theta, dtype="f8")[None, :])[0]

        def x_hybrid(theta):
            return x_engine.predict(np.asarray(theta, dtype="f8")[None, :])[0]

        def x_velocity(fnl, b1, sn0):
            base = velocity_interp([float(fnl), float(b1)])[0]
            sn_response = x_hybrid([float(fnl), float(b1), float(sn0)]) - x_hybrid([float(fnl), float(b1), 0.0])
            return base + sn_response

        def shared_predict(theta):
            theta = np.asarray(theta, dtype="f8")
            return np.r_[p_predict(theta), x_hybrid([theta[0], theta[1], theta[3]])]

        def velocity_joint_predict(theta):
            theta = np.asarray(theta, dtype="f8")
            return np.r_[p_predict(theta), x_velocity(theta[0], theta[1], theta[3])]

        entry = {"cross_scan": [], "split_b": {}, "model_comparison": {}}
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            cov = covariance.copy()
            cov[:22, 22:] *= lam
            cov[22:, :22] *= lam
            fit = fit_ml(data, cov, shared_predict, j_engine.lower, j_engine.upper, [3.0, 2.38, 2.2, 0.08])
            fit["lambda"] = lam
            entry["cross_scan"].append(fit)

        def split_predict(theta):
            theta = np.asarray(theta, dtype="f8")
            return np.r_[p_predict([theta[0], theta[1], theta[3], theta[4]]), x_hybrid([theta[0], theta[2], theta[4]])]

        split = fit_ml(
            data,
            covariance,
            split_predict,
            [-500.0, 0.5, 0.5, 0.0, -1.0],
            [500.0, 5.0, 5.0, 30.0, 1.0],
            [0.0, 2.41, 2.42, 2.2, 0.08],
        )
        shared = entry["cross_scan"][-1]
        split["delta_chi2_vs_shared"] = float(shared["chi2"] - split["chi2"])
        split["b1_difference_P_minus_xi"] = float(split["theta"][1] - split["theta"][2])
        entry["split_b"] = split

        baseline = fit_ml(data, covariance, shared_predict, j_engine.lower, j_engine.upper, [3.0, 2.38, 2.2, 0.08])
        velocity = fit_ml(data, covariance, velocity_joint_predict, j_engine.lower, j_engine.upper, [3.0, 2.40, 2.2, 0.08])
        p_only = fit_ml(p_data, covariance[:22, :22], p_predict, p_engine.lower, p_engine.upper, [-6.0, 2.41, 2.3, 0.05])
        x_only_velocity = fit_ml(x_data, covariance[22:, 22:], lambda t: x_velocity(t[0], t[1], t[2]), [-500.0, 0.5, -1.0], [500.0, 5.0, 1.0], [0.0, 2.43, 0.0])
        entry["model_comparison"] = {
            "p_only_fullric": p_only,
            "joint_fullric_hybrid": baseline,
            "xi_only_velocileptors": x_only_velocity,
            "joint_velocileptors_xi_sensitivity": velocity,
        }
        result["cross_scan"][name] = entry["cross_scan"]
        result["split_b"][name] = entry["split_b"]
        result["model_comparison"][name] = entry["model_comparison"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
