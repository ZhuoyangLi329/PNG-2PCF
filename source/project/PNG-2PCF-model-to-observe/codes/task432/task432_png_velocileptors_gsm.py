#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hybrid PNG response diagnostic on top of velocileptors GSM.

目的
----
验证当前 PNG scale-dependent bias 是否可以在一个已经通过 xi-only closure
的 CLPT-GS/GSM baseline 上一致接入。

实现范围
--------
- velocileptors 提供 Gaussian CLPT/GSM baseline；
- b1_E、fNL 是 xi-only fit parameters；
- PNG response 使用当前 contract:

      b_PNG(k) = b1_E + q * alpha(k)
      q = fNL * 2 * Delta_c * (b1_E - p), p=1

- 对 real-space tracer P 增加线性 PNG response；
- 对 density--velocity cross moment 增加对应线性 response；
- velocity--velocity moment、EFT alpha、b2/bs/b3、s2fog 全部固定为零；
- 这是用于定位 PNG 接入位置的 hybrid diagnostic，不是最终 CLPT-PNG theory。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.optimize import least_squares

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_DIR = PROJECT_ROOT / "codes" / "task43"
TASK432_DIR = PROJECT_ROOT / "codes" / "task432"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "task432_model_repair" / "png_velocileptors_gsm"
PLOT_PATH = PROJECT_ROOT / "plots" / "task43" / "rsd_validation" / "task432_model_repair" / "task432_png_velocileptors_gsm.pdf"

for path in (TASK43_DIR, TASK432_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from task43_rsd_common import rawbox_xi_primary_mask  # noqa: E402
from task432_linear_gsm_rawbox import build_models_and_covariance, precision_from_covariance  # noqa: E402
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


class PNGVelocileptorsGSM:
    """Minimal linear PNG response extension around Gaussian CLPT/GSM."""

    def __init__(self, k: np.ndarray, pk: np.ndarray, alpha: np.ndarray, f_growth: float, s_edges: np.ndarray) -> None:
        self.k = np.asarray(k, dtype="f8")
        self.pk = np.asarray(pk, dtype="f8")
        self.alpha = np.asarray(alpha, dtype="f8")
        self.f_growth = float(f_growth)
        self.s_edges = np.asarray(s_edges, dtype="f8")
        self.radii, self.radial_weights = self._shell_nodes(self.s_edges, 4)
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

    @staticmethod
    def _shell_nodes(edges: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
        nodes, weights = np.polynomial.legendre.leggauss(int(order))
        lo = np.asarray(edges[:-1], dtype="f8")[:, None]
        hi = np.asarray(edges[1:], dtype="f8")[:, None]
        radius = 0.5 * (hi - lo) * nodes[None, :] + 0.5 * (hi + lo)
        raw = 0.5 * (hi - lo) * weights[None, :] * radius**2
        return radius, raw / ((hi**3 - lo**3) / 3.0)

    def _png_correct_cumulants(self, fnl: float, b1_eulerian: float) -> None:
        """Add only linear PNG response to the parent CLPT/GSM cumulants."""

        b1_lagrangian = float(b1_eulerian) - 1.0
        self.gsm.compute_cumulants(
            b1_lagrangian,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        if float(fnl) == 0.0:
            return

        q = float(fnl) * 2.0 * DELTA_C * (float(b1_eulerian) - P_FIXED)
        alpha_kv = np.interp(np.log(self.gsm.kv), np.log(self.k), self.alpha)
        p_kv = np.interp(np.log(self.gsm.kv), np.log(self.k), self.pk)
        p_linear_int = np.asarray(self.gsm.plin, dtype="f8")
        alpha_int = np.interp(np.log(self.gsm.kint), np.log(self.k), self.alpha)

        # Real-space tracer response:
        # Delta P = [2 b_E q alpha + q^2 alpha^2] P_L.
        delta_p_kv = (2.0 * float(b1_eulerian) * q * alpha_kv + q**2 * alpha_kv**2) * p_kv
        delta_p_int = (2.0 * float(b1_eulerian) * q * alpha_int + q**2 * alpha_int**2) * p_linear_int
        qint, delta_xi = self.gsm.sph_gsm.sph(0, delta_p_int * self.gsm.window)
        self.gsm.xieft += np.interp(self.gsm.rint, qint, delta_xi)

        # Leading density--velocity response; velocity--velocity remains the
        # Gaussian CLPT baseline in this minimal diagnostic.
        delta_v_int = -2.0 * q * alpha_int * p_linear_int / self.gsm.kint
        qint, delta_v = self.gsm.sph_gsm.sph(1, delta_v_int * self.gsm.window)
        # compute_xi_rsd applies the common growth factor f to ``veft`` later,
        # so the response table itself must not multiply by f a second time.
        self.gsm.veft += np.interp(self.gsm.rint, qint, delta_v)

    def evaluate(self, *, fnl: float, b1_eulerian: float, nint: int = 800) -> dict[int, np.ndarray]:
        self._png_correct_cumulants(float(fnl), float(b1_eulerian))
        b1_lagrangian = float(b1_eulerian) - 1.0
        xi0_nodes = np.empty_like(self.radii)
        xi2_nodes = np.empty_like(self.radii)
        first = True
        for ibin in range(self.radii.shape[0]):
            for inode, radius in enumerate(self.radii[ibin]):
                xi0, xi2, _ = self.gsm.compute_xi_ell(
                    float(radius),
                    self.f_growth,
                    b1_lagrangian,
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
                first = False
                xi0_nodes[ibin, inode] = float(xi0)
                xi2_nodes[ibin, inode] = float(xi2)
        return {0: np.sum(self.radial_weights * xi0_nodes, axis=1), 2: np.sum(self.radial_weights * xi2_nodes, axis=1)}


def main() -> None:
    products = build_models_and_covariance(xi_angle_mode="continuous", k_switch=0.01)
    cache_path = Path(products["metadata"]["cache"])
    with np.load(cache_path, allow_pickle=False) as payload:
        k = np.asarray(payload["k_eff"], dtype="f8")
        pk = np.asarray(payload["pk_dd"], dtype="f8")
        alpha = np.asarray(payload["alpha"], dtype="f8")
        f_growth = float(np.asarray(payload["f_growth"]).item())
        s_edges = np.asarray(payload["s_edges"], dtype="f8")
    mask = rawbox_xi_primary_mask(0.5 * (s_edges[:-1] + s_edges[1:]))
    data = np.asarray(products["data"]["x"], dtype="f8")
    precision = precision_from_covariance(products["covariances"]["x"])
    model = PNGVelocileptorsGSM(k, pk, alpha, f_growth, s_edges)

    def residual(theta: np.ndarray) -> np.ndarray:
        prediction = model.evaluate(fnl=float(theta[0]), b1_eulerian=float(theta[1]), nint=600)
        vector = np.concatenate([prediction[0][mask], prediction[2][mask]])
        return data - vector

    def objective(theta: np.ndarray) -> float:
        r = residual(theta)
        return float(r @ precision @ r)

    starts = [np.asarray([0.0, 2.55]), np.asarray([-20.0, 2.5]), np.asarray([20.0, 2.6])]
    solutions = [least_squares(residual, start, bounds=(np.asarray([-500.0, 0.5]), np.asarray([500.0, 5.0])), max_nfev=50) for start in starts]
    best = min(solutions, key=lambda result: float(result.fun @ result.fun))
    best_theta = np.asarray(best.x, dtype="f8")
    best_prediction = model.evaluate(fnl=float(best_theta[0]), b1_eulerian=float(best_theta[1]), nint=1000)
    best_vector = np.concatenate([best_prediction[0][mask], best_prediction[2][mask]])
    best_chi2 = float((data - best_vector) @ precision @ (data - best_vector))

    output_dir = OUTPUT_ROOT
    output_dir.joinpath("png_velocileptors_gsm").mkdir(parents=True, exist_ok=True)
    audit = {
        "task": "Task432 hybrid PNG velocileptors GSM xi-only diagnostic",
        "status": "complete",
        "contract": products["metadata"],
        "model": {
            "baseline": "velocileptors Gaussian CLPT/GSM",
            "png_response": "linear tracer P and density-velocity response using current b_PNG(k)",
            "p_fixed": P_FIXED,
            "b2_bs_b3": "fixed zero",
            "eft_parameters": "all zero",
            "s2fog": 0.0,
            "kmax_h_mpc": 0.5,
        },
        "fit": {"parameter_names": ["fNL", "b1"], "map_theta": best_theta.tolist(), "map_chi2": best_chi2},
        "note": "Hybrid PNG response diagnostic; not yet a complete CLPT-PNG operator basis.",
    }
    atomic_json(output_dir / "png_velocileptors_gsm" / "task432_png_velocileptors_gsm_summary.json", audit)
    print(json.dumps({"status": "complete", "map_theta": best_theta.tolist(), "map_chi2": best_chi2}, sort_keys=True))


if __name__ == "__main__":
    main()
