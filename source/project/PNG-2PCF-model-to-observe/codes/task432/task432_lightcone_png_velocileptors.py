#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit the hybrid PNG velocileptors GSM to the standard 4.3 lightcone mock.

This script reuses the frozen Task43 lightcone loader for all data vectors and
covariance blocks.  It replaces only the xi02 theory with a velocileptors
Gaussian CLPT/GSM baseline plus the minimal PNG response used in the rawbox
diagnostic.  The P02 theory and covariance remain the standard Task43 ones.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.optimize import minimize

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_DIR = PROJECT_ROOT / "codes" / "task43"
TASK432_DIR = PROJECT_ROOT / "codes" / "task432"
OUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "task432_model_repair" / "lightcone_png_velocileptors_gsm"
PLOT_PATH = PROJECT_ROOT / "plots" / "task43" / "rsd_validation" / "task432_model_repair" / "task432_lightcone_png_velocileptors_gsm.pdf"

for path in (TASK43_DIR, TASK432_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from task43_run_lightcone_joint_baomask_v1 import load_rsd_specs, RSD_P_PAYLOAD, RSD_X_SUMMARY  # noqa: E402
from task43_rsd_model import build_cache  # noqa: E402
from task43_rsd_common import rawbox_xi_primary_mask  # noqa: E402
from task432_linear_gsm_rawbox import precision_from_covariance  # noqa: E402
from task44_hybrid_gsm import _load_gsm_class  # noqa: E402


DELTA_C = 1.686
P_FIXED = 1.0


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class LightconePNGVelocileptors:
    """Direct-s-at-center version of the hybrid PNG velocileptors GSM."""

    def __init__(self, k: np.ndarray, pk: np.ndarray, alpha: np.ndarray, f_growth: float, s: np.ndarray) -> None:
        self.k = np.asarray(k, dtype="f8")
        self.pk = np.asarray(pk, dtype="f8")
        self.alpha = np.asarray(alpha, dtype="f8")
        self.f_growth = float(f_growth)
        self.s = np.asarray(s, dtype="f8")
        GaussianStreamingModel = _load_gsm_class()
        self.gsm = GaussianStreamingModel(
            self.k,
            self.pk,
            kmin=0.003,
            kmax=0.5,
            nk=100,
            N=1600,
            threads=1,
            cutoff=10.0,
        )

    def _set_cumulants(self, fnl: float, b1_eulerian: float) -> None:
        b1_l = float(b1_eulerian) - 1.0
        self.gsm.compute_cumulants(b1_l, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if float(fnl) == 0.0:
            return
        q = float(fnl) * 2.0 * DELTA_C * (float(b1_eulerian) - P_FIXED)
        alpha_int = np.interp(np.log(self.gsm.kint), np.log(self.k), self.alpha)
        p_int = np.asarray(self.gsm.plin, dtype="f8")
        delta_p = (2.0 * float(b1_eulerian) * q * alpha_int + q**2 * alpha_int**2) * p_int
        qint, delta_xi = self.gsm.sph_gsm.sph(0, delta_p * self.gsm.window)
        self.gsm.xieft += np.interp(self.gsm.rint, qint, delta_xi)
        delta_v = -2.0 * q * alpha_int * p_int / self.gsm.kint
        qint, delta_v_xi = self.gsm.sph_gsm.sph(1, delta_v * self.gsm.window)
        self.gsm.veft += np.interp(self.gsm.rint, qint, delta_v_xi)

    def evaluate(self, *, fnl: float, b1: float, nint: int = 600) -> dict[int, np.ndarray]:
        self._set_cumulants(float(fnl), float(b1))
        b1_l = float(b1) - 1.0
        xi0 = np.empty_like(self.s)
        xi2 = np.empty_like(self.s)
        for index, radius in enumerate(self.s):
            pole0, pole2, _ = self.gsm.compute_xi_ell(
                float(radius),
                self.f_growth,
                b1_l,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                rwidth=100.0,
                Nint=int(nint),
                ngauss=4,
                update_cumulants=False,
            )
            xi0[index] = float(pole0)
            xi2[index] = float(pole2)
        return {0: xi0, 2: xi2}


def run_map(data: np.ndarray, covariance: np.ndarray, evaluate: Callable[[np.ndarray], np.ndarray], start: np.ndarray, names: tuple[str, ...]) -> dict[str, Any]:
    precision = precision_from_covariance(covariance)
    lower = np.asarray([-500.0, 0.5] if len(names) == 2 else [-500.0, 0.5, 0.0, -1.0], dtype="f8")
    upper = np.asarray([500.0, 5.0] if len(names) == 2 else [500.0, 5.0, 30.0, 1.0], dtype="f8")

    def objective(theta: np.ndarray) -> float:
        delta = np.asarray(data, dtype="f8") - np.asarray(evaluate(theta), dtype="f8")
        return float(delta @ precision @ delta)

    result = minimize(
        objective,
        np.asarray(start, dtype="f8"),
        method="Powell",
        bounds=list(zip(lower, upper)),
        options={"maxiter": 120, "xtol": 1.0e-6, "ftol": 1.0e-8},
    )
    return {"parameter_names": list(names), "map_theta": np.asarray(result.x).tolist(), "map_chi2": float(result.fun), "message": str(result.message), "success": bool(result.success)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwalkers", type=int, default=16)
    parser.add_argument("--nsteps", type=int, default=900)
    parser.add_argument("--burnin", type=int, default=200)
    parser.add_argument("--mcmc", action="store_true")
    args = parser.parse_args()

    specs, metadata, arrays = load_rsd_specs(smin=50.0, pk_kmax=0.08)
    spec_p = next(spec for spec in specs if spec.name == "rsd_p02")
    spec_x = next(spec for spec in specs if spec.name == "rsd_xi02")
    spec_joint = next(spec for spec in specs if spec.name == "rsd_joint_p02xi02")
    with np.load(RSD_P_PAYLOAD, allow_pickle=False) as payload:
        zeff = float(np.asarray(payload["zeff"]).item())
    cache = build_cache(zeff=zeff, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000")
    with np.load(cache, allow_pickle=False) as payload:
        k = np.asarray(payload["k_eff"], dtype="f8")
        pk = np.asarray(payload["pk_dd"], dtype="f8")
        alpha = np.asarray(payload["alpha"], dtype="f8")
        f_growth = float(np.asarray(payload["f_growth"]).item())
    with np.load(RSD_X_SUMMARY, allow_pickle=False) as payload:
        s = np.asarray(payload["s"], dtype="f8")
    mask = rawbox_xi_primary_mask(s)
    model_x = LightconePNGVelocileptors(k, pk, alpha, f_growth, s)

    def evaluate_x(theta: np.ndarray) -> np.ndarray:
        values = model_x.evaluate(fnl=float(theta[0]), b1=float(theta[1]), nint=600)
        return np.concatenate((values[0][mask], values[2][mask]))

    def evaluate_joint(theta: np.ndarray) -> np.ndarray:
        return np.concatenate((spec_p.evaluate(np.asarray(theta, dtype="f8")), evaluate_x(theta)))

    xi_map = run_map(spec_x.data, spec_x.covariance, evaluate_x, np.asarray([0.0, 2.4]), ("fNL", "b1"))
    # Include the standard P-only ML point and the standard joint ML point as
    # starts.  The P/xi covariance can make the four-dimensional surface
    # multimodal enough that a single generic start is not sufficient.
    joint_starts = [
        np.asarray([-1.575, 2.364, 1.04, 0.123]),
        np.asarray([5.715, 2.338, 0.5, 0.168]),
        np.asarray([0.0, 2.4, 1.0, 0.15]),
    ]
    joint_solutions = [
        run_map(spec_joint.data, spec_joint.covariance, evaluate_joint, start, ("fNL", "b1", "sigma_s", "sn0"))
        for start in joint_starts
    ]
    joint_map = min(joint_solutions, key=lambda item: item["map_chi2"])
    result = {"xi_map": xi_map, "joint_map": joint_map, "metadata": metadata, "model": "velocileptors GSM + minimal linear PNG response; EFT/bias higher operators fixed zero"}
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT_ROOT / "task432_lightcone_png_velocileptors_gsm_map.json", result)
    print(json.dumps({"status": "complete", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
