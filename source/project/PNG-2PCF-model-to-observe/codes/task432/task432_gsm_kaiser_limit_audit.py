#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kaiser-limit and first-order streaming audit for the Task432 GSM.

本 audit 不做 likelihood，不产生 PNG，也不修改 Task43 production。

对同一个 frozen linear P(k)、b1、f_growth，比较：

1. Fourier-space continuous Kaiser xi0/xi2；
2. first-order streaming expansion；
3. full Gaussian streaming integral；
4. real-space xi、v12 mapping term、sigma12 mapping term的分解。

first-order streaming expansion 使用：

    xi_s = xi_real - d[mu*v12]/dy + 0.5*d2[sigma12^2]/dy2

这个测试的目的，是把“线性 moments + full Gaussian exponent”产生的高阶
mapping correction，与 v12/variance 的 normalization 或 projection bug 分开。
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


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_DIR = PROJECT_ROOT / "codes" / "task43"
TASK432_DIR = PROJECT_ROOT / "codes" / "task432"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "task432_model_repair"
PLOT_ROOT = PROJECT_ROOT / "plots" / "task43" / "rsd_validation" / "task432_model_repair"

for _path in (TASK43_DIR, TASK432_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from task43_rawbox_numerics import continuous_rsd_poles, shell_kernel  # noqa: E402
from task43_rsd_common import rawbox_xi_primary_mask  # noqa: E402
from task432_linear_gsm_rawbox import LinearGSMModel, LinearRadialMomentProvider, build_models_and_covariance  # noqa: E402


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


def shell_nodes(edges: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    lo = np.asarray(edges[:-1], dtype="f8")[:, None]
    hi = np.asarray(edges[1:], dtype="f8")[:, None]
    radius = 0.5 * (hi - lo) * nodes[None, :] + 0.5 * (hi + lo)
    raw = 0.5 * (hi - lo) * weights[None, :] * radius**2
    normalized = raw / ((hi**3 - lo**3) / 3.0)
    return radius, normalized


def multipoles_from_values(values: np.ndarray, mu: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Project one angular shell with the same full-range Legendre convention."""

    l2 = 0.5 * (3.0 * mu**2 - 1.0)
    return float(0.5 * (values @ weights)), float(values @ (2.5 * weights * l2))


def first_order_streaming(
    provider: Any,
    edges: np.ndarray,
    *,
    radial_order: int,
    angular_order: int,
    finite_difference_h: float,
) -> dict[str, np.ndarray]:
    """Compute real, v12, sigma12 and total first-order streaming multipoles."""

    radii, radial_weights = shell_nodes(edges, radial_order)
    mu, mu_weights = np.polynomial.legendre.leggauss(angular_order)
    nbin = edges.size - 1
    total0 = np.zeros(nbin, dtype="f8")
    total2 = np.zeros(nbin, dtype="f8")
    real0 = np.zeros(nbin, dtype="f8")
    real2 = np.zeros(nbin, dtype="f8")
    v0 = np.zeros(nbin, dtype="f8")
    v2 = np.zeros(nbin, dtype="f8")
    sigma0 = np.zeros(nbin, dtype="f8")
    sigma2 = np.zeros(nbin, dtype="f8")

    h = float(finite_difference_h)
    for ibin in range(nbin):
        for inode, radius in enumerate(radii[ibin]):
            transverse = float(radius) * np.sqrt(np.maximum(1.0 - mu**2, 0.0))
            parallel = float(radius) * mu
            xi, mean, variance = provider.at_los_array(transverse, parallel)
            mean_plus = np.empty_like(mean)
            mean_minus = np.empty_like(mean)
            variance_plus = np.empty_like(variance)
            variance_minus = np.empty_like(variance)
            for index, (trans, par) in enumerate(zip(transverse, parallel)):
                _, mean_p, variance_p = provider.at_los_array(
                    np.asarray([trans], dtype="f8"), np.asarray([par + h], dtype="f8")
                )
                _, mean_m, variance_m = provider.at_los_array(
                    np.asarray([trans], dtype="f8"), np.asarray([par - h], dtype="f8")
                )
                mean_plus[index] = mean_p[0]
                mean_minus[index] = mean_m[0]
                variance_plus[index] = variance_p[0]
                variance_minus[index] = variance_m[0]
            d_mean = (mean_plus - mean_minus) / (2.0 * h)
            d2_variance = (variance_plus - 2.0 * variance + variance_minus) / h**2
            v_term = -d_mean
            sigma_term = 0.5 * d2_variance
            total = xi + v_term + sigma_term
            real_bin0, real_bin2 = multipoles_from_values(xi, mu, mu_weights)
            v_bin0, v_bin2 = multipoles_from_values(v_term, mu, mu_weights)
            sigma_bin0, sigma_bin2 = multipoles_from_values(sigma_term, mu, mu_weights)
            total_bin0, total_bin2 = multipoles_from_values(total, mu, mu_weights)
            weight = radial_weights[ibin, inode]
            real0[ibin] += weight * real_bin0
            real2[ibin] += weight * real_bin2
            v0[ibin] += weight * v_bin0
            v2[ibin] += weight * v_bin2
            sigma0[ibin] += weight * sigma_bin0
            sigma2[ibin] += weight * sigma_bin2
            total0[ibin] += weight * total_bin0
            total2[ibin] += weight * total_bin2
    return {
        "real0": real0,
        "real2": real2,
        "v0": v0,
        "v2": v2,
        "sigma0": sigma0,
        "sigma2": sigma2,
        "total0": total0,
        "total2": total2,
    }


def kaiser_multipoles(cache_path: Path, *, fnl: float, b1: float) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as payload:
        k = np.asarray(payload["k_eff"], dtype="f8")
        g_nz = np.asarray(payload["g_nz"], dtype="f8")
        pk = np.asarray(payload["pk_dd"], dtype="f8")
        alpha = np.asarray(payload["alpha"], dtype="f8")
        edges = np.asarray(payload["s_edges"], dtype="f8")
        volume = float(np.asarray(payload["volume"]).item())
        growth = float(np.asarray(payload["f_growth"]).item())
    amplitude = float(b1) + float(fnl) * 2.0 * 1.686 * (float(b1) - 1.0) * alpha
    poles = continuous_rsd_poles(k, pk, amplitude, growth, 0.0)
    return {
        "xi0": np.sum(g_nz[:, None] * poles[0][:, None] * shell_kernel(k, edges, 0), axis=0) / volume,
        "xi2": np.sum(g_nz[:, None] * poles[2][:, None] * shell_kernel(k, edges, 2), axis=0) / volume,
        "edges": edges,
    }


def relative_l2(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    aa = np.asarray(a)[mask]
    bb = np.asarray(b)[mask]
    return float(np.linalg.norm(aa - bb) / max(np.linalg.norm(bb), 1.0e-300))


def make_pdf(path: Path, rows: list[dict[str, Any]], centers: np.ndarray, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary) as pdf:
        for row in rows:
            figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.8), sharex=True)
            plot_s = centers[mask]
            for axis, ell, labels in (
                (axes[0], 0, ("kaiser", "first_order", "full_gsm", "real", "v_term", "sigma_term")),
                (axes[1], 2, ("kaiser", "first_order", "full_gsm", "real", "v_term", "sigma_term")),
            ):
                curves = row["curves"][f"xi{ell}"]
                colors = {"kaiser": "#2F2F2F", "first_order": "#4C72B0", "full_gsm": "#C44E52", "real": "#777777", "v_term": "#55A868", "sigma_term": "#C44E52"}
                styles = {"kaiser": "-", "first_order": "-", "full_gsm": "-", "real": "--", "v_term": ":", "sigma_term": "-."}
                for key in labels:
                    axis.plot(plot_s, curves[key][mask], color=colors[key], ls=styles[key], lw=1.25 if key not in {"kaiser", "full_gsm"} else 1.8, label=key)
                axis.axhline(0.0, color="0.6", lw=0.7)
                axis.set_ylabel(f"xi{ell}(s)")
                axis.grid(True, alpha=0.25)
                axis.legend(frameon=False, ncol=3, fontsize=8)
            axes[1].set_xlabel(r"$s\ [h^{-1}\mathrm{Mpc}]$")
            figure.suptitle(f"GSM Kaiser-limit audit: kmax={row['kmax_gsm_h_mpc']:.3g} h/Mpc; h={row['finite_difference_h']:.3g} Mpc/h", fontsize=13)
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

        figure, axis = plt.subplots(figsize=(9.0, 5.0))
        kmax = np.asarray([row["kmax_gsm_h_mpc"] for row in rows])
        for ell in (0, 2):
            axis.plot(kmax, [row["relative_l2"][f"first_order_vs_kaiser_xi{ell}"] for row in rows], marker="o", label=f"first-order xi{ell}")
            axis.plot(kmax, [row["relative_l2"][f"full_gsm_vs_kaiser_xi{ell}"] for row in rows], marker="s", label=f"full GSM xi{ell}")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$k_{\max}^{\rm GSM}\ [h\,\mathrm{Mpc}^{-1}]$")
        axis.set_ylabel("relative L2 difference to Kaiser")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False, ncol=2)
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kmax-list", default="0.15,0.25,0.50,1.0")
    parser.add_argument("--b1", type=float, default=2.55)
    parser.add_argument("--fnl", type=float, default=0.0)
    parser.add_argument("--finite-difference-h", type=float, default=0.05)
    parser.add_argument("--radial-order", type=int, default=4)
    parser.add_argument("--angular-order", type=int, default=16)
    args = parser.parse_args()
    started = time.perf_counter()

    products = build_models_and_covariance(xi_angle_mode="continuous", k_switch=0.01)
    cache_path = Path(products["metadata"]["cache"])
    with np.load(cache_path, allow_pickle=False) as payload:
        edges = np.asarray(payload["s_edges"], dtype="f8")
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = rawbox_xi_primary_mask(centers)
    rows: list[dict[str, Any]] = []
    for value in [float(item) for item in str(args.kmax_list).split(",") if item.strip()]:
        basis = LinearRadialMomentProvider(cache_path, kmax_gsm=value, n_radial=2800)
        provider = basis.provider(fnl=float(args.fnl), b1=float(args.b1))
        first = first_order_streaming(
            provider,
            edges,
            radial_order=int(args.radial_order),
            angular_order=int(args.angular_order),
            finite_difference_h=float(args.finite_difference_h),
        )
        gsm = LinearGSMModel(basis, edges, shell_order=4, angular_order=12, stream_order=8, zmax=8.0).evaluate(fnl=float(args.fnl), b1=float(args.b1), sigma_fog=0.0)
        kaiser = kaiser_multipoles(cache_path, fnl=float(args.fnl), b1=float(args.b1))
        curves = {
            "xi0": {"kaiser": kaiser["xi0"], "first_order": first["total0"], "full_gsm": gsm[0], "real": first["real0"], "v_term": first["v0"], "sigma_term": first["sigma0"]},
            "xi2": {"kaiser": kaiser["xi2"], "first_order": first["total2"], "full_gsm": gsm[2], "real": first["real2"], "v_term": first["v2"], "sigma_term": first["sigma2"]},
        }
        rows.append({
            "kmax_gsm_h_mpc": value,
            "finite_difference_h": float(args.finite_difference_h),
            "curves": curves,
            "relative_l2": {
                "first_order_vs_kaiser_xi0": relative_l2(first["total0"], kaiser["xi0"], mask),
                "first_order_vs_kaiser_xi2": relative_l2(first["total2"], kaiser["xi2"], mask),
                "full_gsm_vs_kaiser_xi0": relative_l2(gsm[0], kaiser["xi0"], mask),
                "full_gsm_vs_kaiser_xi2": relative_l2(gsm[2], kaiser["xi2"], mask),
                "full_gsm_vs_first_order_xi0": relative_l2(gsm[0], first["total0"], mask),
                "full_gsm_vs_first_order_xi2": relative_l2(gsm[2], first["total2"], mask),
            },
        })
        print(json.dumps({"kmax": value, "relative_l2": rows[-1]["relative_l2"]}, sort_keys=True))

    output_dir = OUTPUT_ROOT / "gsm_kaiser_limit_audit"
    audit_path = output_dir / "task432_gsm_kaiser_limit_audit.json"
    pdf_path = PLOT_ROOT / "task432_gsm_kaiser_limit_audit.pdf"
    audit = {
        "task": "Task432 GSM Kaiser-limit and first-order streaming audit",
        "status": "complete",
        "contract": products["metadata"],
        "fnl": float(args.fnl),
        "b1": float(args.b1),
        "finite_difference_h": float(args.finite_difference_h),
        "radial_order": int(args.radial_order),
        "angular_order": int(args.angular_order),
        "rows": rows,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    atomic_json(audit_path, audit)
    make_pdf(pdf_path, rows, centers, mask)
    print(json.dumps({"status": "complete", "audit": str(audit_path), "pdf": str(pdf_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
