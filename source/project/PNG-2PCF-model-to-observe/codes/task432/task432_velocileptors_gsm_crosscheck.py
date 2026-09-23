#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-check Task432 xi-only RSD against velocileptors GSM.

本脚本使用与 task432 rawbox 相同的 xi0/xi2 data、mask 和固定 covariance，
但调用 velocileptors 的 GaussianStreamingModel，设置所有 EFT-like
counterterms 为零，只保留 Lagrangian bias b1^L=b1^E-1、linear P(k) 和
Gaussian streaming mapping。

这不是最终 PNG model；它只回答一个关键问题：当前自定义 linear-moment
GSM 的 xi shape mismatch 是否来自实现 convention，而不是 GSM 方法本身。
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
from scipy.optimize import minimize_scalar

# velocileptors 3.1 uses np.trapezoid, while the frozen desilike environment
# currently exposes numpy 1.26.  This compatibility alias does not alter any
# theory formula or production file.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_DIR = PROJECT_ROOT / "codes" / "task43"
TASK432_DIR = PROJECT_ROOT / "codes" / "task432"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "task432_model_repair" / "velocileptors_gsm_crosscheck"
PLOT_PATH = PROJECT_ROOT / "plots" / "task43" / "rsd_validation" / "task432_model_repair" / "task432_velocileptors_gsm_crosscheck.pdf"

for path in (TASK43_DIR, TASK432_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from task43_rsd_common import rawbox_xi_primary_mask  # noqa: E402
from task432_linear_gsm_rawbox import build_models_and_covariance, precision_from_covariance  # noqa: E402
from task44_hybrid_gsm import _load_gsm_class  # noqa: E402


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


def shell_nodes(edges: np.ndarray, order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    lo = np.asarray(edges[:-1], dtype="f8")[:, None]
    hi = np.asarray(edges[1:], dtype="f8")[:, None]
    radius = 0.5 * (hi - lo) * nodes[None, :] + 0.5 * (hi + lo)
    raw = 0.5 * (hi - lo) * weights[None, :] * radius**2
    return radius, raw / ((hi**3 - lo**3) / 3.0)


class VelocileptorsXiModel:
    def __init__(self, k: np.ndarray, pk: np.ndarray, f_growth: float, s_edges: np.ndarray) -> None:
        GaussianStreamingModel = _load_gsm_class()
        self.f_growth = float(f_growth)
        self.s_edges = np.asarray(s_edges, dtype="f8")
        self.radii, self.radial_weights = shell_nodes(self.s_edges, order=4)
        self.gsm = GaussianStreamingModel(
            np.asarray(k, dtype="f8"),
            np.asarray(pk, dtype="f8"),
            kmin=0.003,
            kmax=0.5,
            nk=100,
            N=1600,
            threads=1,
            cutoff=10.0,
        )

    def evaluate(self, b1_eulerian: float, *, nint: int = 800) -> dict[int, np.ndarray]:
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
                    0.0,  # b2^L fixed to zero for this RSD cross-check
                    0.0,  # bs^L fixed to zero
                    0.0,  # b3^L fixed to zero
                    0.0,  # alpha real-space EFT set to zero
                    0.0,  # alpha_v set to zero
                    0.0,  # alpha_s0 set to zero
                    0.0,  # alpha_s2 set to zero
                    0.0,  # s2fog set to zero
                    rwidth=100.0,
                    Nint=int(nint),
                    ngauss=4,
                    update_cumulants=first,
                )
                first = False
                xi0_nodes[ibin, inode] = float(xi0)
                xi2_nodes[ibin, inode] = float(xi2)
        return {
            0: np.sum(self.radial_weights * xi0_nodes, axis=1),
            2: np.sum(self.radial_weights * xi2_nodes, axis=1),
        }


def main() -> None:
    products = build_models_and_covariance(xi_angle_mode="continuous", k_switch=0.01)
    cache_path = Path(products["metadata"]["cache"])
    with np.load(cache_path, allow_pickle=False) as payload:
        k = np.asarray(payload["k_eff"], dtype="f8")
        pk = np.asarray(payload["pk_dd"], dtype="f8")
        f_growth = float(np.asarray(payload["f_growth"]).item())
        s_edges = np.asarray(payload["s_edges"], dtype="f8")
    mask = rawbox_xi_primary_mask(0.5 * (s_edges[:-1] + s_edges[1:]))
    data = np.asarray(products["data"]["x"], dtype="f8")
    covariance = products["covariances"]["x"]
    precision = precision_from_covariance(covariance)
    model = VelocileptorsXiModel(k, pk, f_growth, s_edges)

    cache: dict[float, dict[int, np.ndarray]] = {}

    def prediction(b1: float, nint: int = 800) -> np.ndarray:
        key = float(np.round(b1, 8))
        if key not in cache:
            values = model.evaluate(float(b1), nint=int(nint))
            cache[key] = values
        values = cache[key]
        return np.concatenate([values[0][mask], values[2][mask]])

    def chi2(b1: float) -> float:
        residual = data - prediction(float(b1), nint=600)
        return float(residual @ precision @ residual)

    best = minimize_scalar(chi2, bounds=(1.5, 3.5), method="bounded", options={"xatol": 2.0e-3, "maxiter": 40})
    best_b1 = float(best.x)
    best_prediction = model.evaluate(best_b1, nint=1000)
    best_vector = np.concatenate([best_prediction[0][mask], best_prediction[2][mask]])
    best_chi2 = float((data - best_vector) @ precision @ (data - best_vector))
    fixed_prediction = model.evaluate(2.55, nint=1000)
    fixed_vector = np.concatenate([fixed_prediction[0][mask], fixed_prediction[2][mask]])
    fixed_chi2 = float((data - fixed_vector) @ precision @ (data - fixed_vector))

    centers = 0.5 * (s_edges[:-1] + s_edges[1:])[mask]
    n0 = int(np.count_nonzero(mask))
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_pdf = PLOT_PATH.with_name(f".{PLOT_PATH.name}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary_pdf) as pdf:
        figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.5), sharex=True)
        for axis, ell, offset in ((axes[0], 0, 0), (axes[1], 2, n0)):
            measured = data[offset : offset + n0]
            predicted = best_prediction[ell][mask]
            fixed = fixed_prediction[ell][mask]
            axis.plot(centers, measured, "o", ms=2.8, color="#2F2F2F", label="measured mean")
            axis.plot(centers, predicted, color="#4C72B0", lw=1.8, label=f"velocileptors GSM, b1={best_b1:.3f}")
            axis.plot(centers, fixed, color="#C44E52", lw=1.2, ls="--", label="velocileptors GSM, b1=2.55")
            axis.axhline(0.0, color="0.6", lw=0.7)
            axis.set_ylabel(f"xi{ell}(s)")
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False, fontsize=8)
        axes[1].set_xlabel(r"$s\ [h^{-1}\mathrm{Mpc}]$")
        figure.suptitle(f"velocileptors GSM cross-check: xi-only MAP chi2={best_chi2:.3f}", fontsize=13)
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8.5, 4.8))
        residual = best_vector - data
        axis.plot(np.arange(residual.size), residual, color="#4C72B0", lw=1.0)
        axis.axhline(0.0, color="0.5", lw=0.8)
        axis.set_xlabel("masked xi0/xi2 data-vector index")
        axis.set_ylabel("model - measured")
        axis.set_title("velocileptors GSM residual")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    temporary_pdf.replace(PLOT_PATH)

    audit = {
        "task": "Task432 velocileptors GSM xi-only cross-check",
        "status": "complete",
        "contract": products["metadata"],
        "model": {
            "package": "velocileptors 3.1",
            "kmax_h_mpc": 0.5,
            "b1_lagrangian": "b1_eulerian - 1",
            "b2_bs_b3": "fixed zero",
            "eft_parameters": "all zero",
            "s2fog": 0.0,
            "numpy_compatibility": "np.trapezoid alias to np.trapz",
        },
        "fixed_b1": {"b1": 2.55, "chi2": fixed_chi2},
        "map": {"b1": best_b1, "chi2": best_chi2, "optimizer_success": bool(best.success), "optimizer_message": str(best.message)},
        "output_pdf": str(PLOT_PATH),
        "elapsed_sec": None,
    }
    atomic_json(OUTPUT_ROOT / "velocileptors_gsm_crosscheck" / "task432_velocileptors_gsm_crosscheck.json", audit)
    print(json.dumps({"status": "complete", "b1_map": best_b1, "chi2_map": best_chi2, "chi2_b1_2p55": fixed_chi2, "pdf": str(PLOT_PATH)}, sort_keys=True))


if __name__ == "__main__":
    main()
