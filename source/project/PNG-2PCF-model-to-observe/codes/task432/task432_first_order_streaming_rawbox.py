#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task432 first-order streaming xi-only rawbox baseline.

这个脚本是当前 linear-input full GSM 的 consistency control：

    xi_s = xi_real - d[mu*v12]/dy + 0.5*d2[sigma12^2]/dy2

它只使用 linear real-space/velocity moments，不使用 EFT、不使用
sigma_FOG，也不把 P02 放进本次 xi-only likelihood。该表达式是
Reid & White (2011) streaming expansion 的一阶形式，必须先通过
Kaiser-limit 与 xi-only closure 检查，才考虑 full Gaussian mapping。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.optimize import least_squares


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_DIR = PROJECT_ROOT / "codes" / "task43"
TASK432_DIR = PROJECT_ROOT / "codes" / "task432"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "task432_model_repair"
PLOT_ROOT = PROJECT_ROOT / "plots" / "task43" / "rsd_validation" / "task432_model_repair"

for _path in (TASK43_DIR, TASK432_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from task43_rsd_common import rawbox_xi_primary_mask  # noqa: E402
from task432_gsm_kaiser_limit_audit import first_order_streaming  # noqa: E402
from task432_linear_gsm_rawbox import (  # noqa: E402
    LinearRadialMomentProvider,
    build_models_and_covariance,
    precision_from_covariance,
)


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


class FirstOrderStreamingModel:
    """Evaluate shell-averaged xi0/xi2 from first-order streaming."""

    def __init__(
        self,
        cache_path: Path,
        s_edges: np.ndarray,
        *,
        kmax_gsm: float,
        radial_order: int = 4,
        angular_order: int = 16,
        finite_difference_h: float = 0.05,
    ) -> None:
        self.basis = LinearRadialMomentProvider(cache_path, kmax_gsm=float(kmax_gsm), n_radial=2800)
        self.s_edges = np.asarray(s_edges, dtype="f8")
        self.radial_order = int(radial_order)
        self.angular_order = int(angular_order)
        self.finite_difference_h = float(finite_difference_h)
        self.centers = 0.5 * (self.s_edges[:-1] + self.s_edges[1:])
        self.mask = rawbox_xi_primary_mask(self.centers)

    def evaluate(self, *, fnl: float, b1: float) -> dict[str, np.ndarray]:
        provider = self.basis.provider(fnl=float(fnl), b1=float(b1))
        values = first_order_streaming(
            provider,
            self.s_edges,
            radial_order=self.radial_order,
            angular_order=self.angular_order,
            finite_difference_h=self.finite_difference_h,
        )
        return {key: np.asarray(value, dtype="f8") for key, value in values.items()}

    def vector(self, *, fnl: float, b1: float) -> np.ndarray:
        values = self.evaluate(fnl=fnl, b1=b1)
        return np.concatenate([values["total0"][self.mask], values["total2"][self.mask]])


def map_fit(
    data: np.ndarray,
    covariance: np.ndarray,
    model: FirstOrderStreamingModel,
    starts: list[np.ndarray],
) -> dict[str, Any]:
    precision = precision_from_covariance(covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (precision + precision.T))
    floor = max(float(eigenvalues[-1]) * 1.0e-14, 1.0e-300)
    square_root_transpose = (eigenvectors * np.sqrt(np.maximum(eigenvalues, floor))[None, :]).T

    def residual(theta: np.ndarray) -> np.ndarray:
        return square_root_transpose @ (data - model.vector(fnl=float(theta[0]), b1=float(theta[1])))

    bounds = (np.asarray([-500.0, 0.5]), np.asarray([500.0, 5.0]))
    solutions = [
        least_squares(
            residual,
            np.asarray(start, dtype="f8"),
            bounds=bounds,
            max_nfev=300,
            xtol=1.0e-9,
            ftol=1.0e-9,
            gtol=1.0e-9,
        )
        for start in starts
    ]
    best = min(solutions, key=lambda item: float(item.fun @ item.fun))
    return {
        "parameter_names": ["fNL", "b1"],
        "map_theta": np.asarray(best.x, dtype="f8").tolist(),
        "map_chi2": float(best.fun @ best.fun),
        "nfev": int(best.nfev),
        "message": str(best.message),
    }


def make_pdf(path: Path, model: FirstOrderStreamingModel, data: np.ndarray, prediction: dict[str, np.ndarray], map_fit_result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.pdf")
    centers = model.centers[model.mask]
    n0 = int(np.count_nonzero(model.mask))
    with PdfPages(temporary) as pdf:
        figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.5), sharex=True)
        for axis, ell, key in ((axes[0], 0, "total0"), (axes[1], 2, "total2")):
            measured = data[:n0] if ell == 0 else data[n0:]
            axis.errorbar(centers, measured, fmt="o", ms=2.8, color="#2F2F2F", label="measured mean")
            axis.plot(centers, prediction[key][model.mask], color="#4C72B0", lw=1.8, label="first-order streaming")
            axis.plot(centers, prediction["real" + str(ell)][model.mask], color="#777777", lw=1.0, ls="--", label="real-space term")
            axis.axhline(0.0, color="0.6", lw=0.7)
            axis.set_ylabel(f"xi{ell}(s)")
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False, fontsize=8)
        axes[1].set_xlabel(r"$s\ [h^{-1}\mathrm{Mpc}]$")
        figure.suptitle(f"Task432 first-order streaming xi-only; MAP chi2={map_fit_result['map_chi2']:.3f}", fontsize=13)
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8.5, 4.8))
        residual = np.concatenate([prediction["total0"][model.mask], prediction["total2"][model.mask]]) - data
        axis.plot(np.arange(residual.size), residual, color="#4C72B0", lw=1.0)
        axis.axhline(0.0, color="0.5", lw=0.8)
        axis.set_xlabel("masked xi0/xi2 data-vector index")
        axis.set_ylabel("model - measured")
        axis.set_title("first-order streaming residual")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kmax-gsm", type=float, default=0.5)
    parser.add_argument("--radial-order", type=int, default=4)
    parser.add_argument("--angular-order", type=int, default=16)
    parser.add_argument("--finite-difference-h", type=float, default=0.05)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT / "rawbox_first_order_streaming")
    args = parser.parse_args()
    started = time.perf_counter()

    products = build_models_and_covariance(xi_angle_mode="continuous", k_switch=0.01)
    cache_path = Path(products["metadata"]["cache"])
    with np.load(cache_path, allow_pickle=False) as payload:
        s_edges = np.asarray(payload["s_edges"], dtype="f8")
    model = FirstOrderStreamingModel(
        cache_path,
        s_edges,
        kmax_gsm=float(args.kmax_gsm),
        radial_order=int(args.radial_order),
        angular_order=int(args.angular_order),
        finite_difference_h=float(args.finite_difference_h),
    )
    data = np.asarray(products["data"]["x"], dtype="f8")
    fit = map_fit(
        data,
        products["covariances"]["x"],
        model,
        starts=[np.asarray([0.0, 2.55]), np.asarray([-20.0, 2.5]), np.asarray([20.0, 2.6])],
    )
    prediction = model.evaluate(fnl=float(fit["map_theta"][0]), b1=float(fit["map_theta"][1]))
    audit = {
        "task": "Task432 first-order streaming xi-only rawbox baseline",
        "status": "complete",
        "contract": products["metadata"],
        "model": {
            "kmax_gsm_h_mpc": float(args.kmax_gsm),
            "radial_order": int(args.radial_order),
            "angular_order": int(args.angular_order),
            "finite_difference_h": float(args.finite_difference_h),
            "new_parameters": [],
            "formula": "xi_real - d[mu*v12]/dy + 0.5*d2[sigma12^2]/dy2",
        },
        "fit": fit,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    output_root = Path(args.output_root)
    atomic_json(output_root / "audits" / "task432_first_order_streaming_summary.json", audit)
    make_pdf(PLOT_ROOT / "task432_first_order_streaming_xi_only.pdf", model, data, prediction, fit)
    print(json.dumps({"status": "complete", "map": fit, "audit": str(output_root / 'audits' / 'task432_first_order_streaming_summary.json')}, sort_keys=True))


if __name__ == "__main__":
    main()
