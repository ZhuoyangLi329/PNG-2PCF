#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8：1Gpc 直接用测量 P(k) 建模，且只积到测量 k_max
===========================================================

说明
----
上一轮“直接用测量 P(k)”失败的根本原因，不是 direct-P 思路本身，
而是把只测到 `k ~ 0.3` 的 P(k) 错误地外延到了 `k=20`。

本脚本修正这个问题：

1. 只使用 1Gpc；
2. 直接采用测量均值 P(k)；
3. 全离散求和只做到测量的最后一个 k-bin 上限 `k_max,data`；
4. 同时对比：
   - 原始全离散 best-fit `kmax=20`
   - best-fit 但截到 `k_max,data`
   - measured P(k) + interp
   - measured P(k) + step

先看这条路在“当前已有 k-bin”下是否成立，尤其是 `fnl=0`。
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft


THIS_DIR = Path(__file__).resolve().parent
MISSION5_DIR = THIS_DIR.parent / "mission5_log"
if str(MISSION5_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION5_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import mission8_exp_window_box_finish as m8finish
import task5_ir_window_solution_3gpc_multitype as m5


KMIN_GLOBAL = 1.0e-4
KMAX_FULL = 20.0
FFTLOG_N = m5.FFTLOG_N
FFTLOG_PADDING = m5.FFTLOG_PADDING
LARGE_SCALE_MIN = m5.LARGE_SCALE_MIN

PK100_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat"
PCF100_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat"
PK0_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat"
PCF0_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut_fnl0/pcf_rsd_N*.dat"

BOX_SIZE = 1000.0
K_FUND = 2.0 * np.pi / BOX_SIZE
RID_MIN = 1
RID_MAX = 50

OUT_METRIC_CSV = THIS_DIR / "mission8_measured_pk_1gpc_cutkdata_metrics.csv"
OUT_SUMMARY_MD = THIS_DIR / "mission8_measured_pk_1gpc_cutkdata_summary_20260313.md"
OUT_FIG = THIS_DIR / "mission8_measured_pk_1gpc_cutkdata_compare.png"


def compute_shell_degeneracy_fft(qmax: int, nmax: int) -> np.ndarray:
    a = np.zeros(qmax + 1, dtype=np.float32)
    a[0] = 1.0
    n = np.arange(1, nmax + 1, dtype=np.int64)
    sq = np.square(n)
    sq = sq[sq <= qmax]
    a[sq] = 2.0
    nfft = next_fast_len(3 * qmax + 1)
    print(f"[INFO] degeneracy FFT: qmax={qmax}, nfft={nfft}")
    fa = rfft(a, n=nfft)
    conv = irfft(fa * fa * fa, n=nfft)
    return np.rint(conv[: qmax + 1]).astype(np.int64)


def spherical_bessel_j0(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = x != 0.0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def build_metric_row(method: str, s: np.ndarray, xi_data: np.ndarray, xi_std: np.ndarray, xi_model: np.ndarray, sample: str) -> Dict[str, object]:
    row = dict(m5.compute_alignment_metrics(s, xi_data, xi_std, xi_model, tag=method))
    row["sample"] = sample
    return row


def xi_from_shellsum_with_pfunc(
    s_data: np.ndarray,
    gq: np.ndarray,
    qmax_use: int,
    pfunc,
    q_chunk: int = 400000,
) -> np.ndarray:
    volume = BOX_SIZE**3
    xi = np.zeros_like(s_data, dtype=np.float64)
    q_nonzero = np.nonzero(gq[1 : qmax_use + 1])[0] + 1
    for start in range(0, q_nonzero.size, q_chunk):
        q_block = q_nonzero[start : start + q_chunk]
        kvals = K_FUND * np.sqrt(q_block.astype(np.float64))
        pvals = pfunc(kvals)
        xi += np.sum((gq[q_block].astype(np.float64) * pvals)[:, None] * spherical_bessel_j0(np.outer(kvals, s_data)), axis=0) / volume
    return xi


def main() -> None:
    print("[INFO] Mission8 measured-Pk 1Gpc cut-kdata start")
    data100_raw = m5.load_mock_data("1Gpc_fnl100", PK100_GLOB, PCF100_GLOB, RID_MIN, RID_MAX)
    data0_raw = m5.load_mock_data("1Gpc_fnl0", PK0_GLOB, PCF0_GLOB, RID_MIN, RID_MAX)
    data100 = m8finish.filter_zero_std_pk_bins(data100_raw, tag="1Gpc fnl100")
    data0 = m8finish.filter_zero_std_pk_bins(data0_raw, tag="1Gpc fnl0")

    fit100 = m5.fit_best_pk(data100)
    fit0 = m5.fit_best_pk(data0)
    bestfit100 = fit100["bestfit"]
    bestfit0 = fit0["bestfit"]
    print(f"[INFO] bestfit100 = {bestfit100}")
    print(f"[INFO] bestfit0   = {bestfit0}")

    k_fft = np.geomspace(KMIN_GLOBAL / FFTLOG_PADDING, KMAX_FULL * FFTLOG_PADDING, FFTLOG_N)
    p100_fft = m5.build_theory_p0(k_fft, bestfit100)
    p0_fft = m5.build_theory_p0(k_fft, bestfit0)
    xi100_exp = m8finish.evaluate_exp_window_model(data100.scen, k_fft, p100_fft, K_FUND, 4.0)
    xi0_exp = m8finish.evaluate_exp_window_model(data0.scen, k_fft, p0_fft, K_FUND, 4.0)

    qmax_full = int(np.floor((KMAX_FULL / K_FUND) ** 2))
    nmax_full = int(np.floor(KMAX_FULL / K_FUND))
    gq = compute_shell_degeneracy_fft(qmax_full, nmax_full)

    k_dense = np.geomspace(K_FUND, KMAX_FULL, 300000)
    p100_dense = m5.build_theory_p0(k_dense, bestfit100)
    p0_dense = m5.build_theory_p0(k_dense, bestfit0)

    kmax_data_100 = float(data100.kmax[-1])
    kmax_data_0 = float(data0.kmax[-1])
    kmax_data = min(kmax_data_100, kmax_data_0)
    qmax_data = int(np.floor((kmax_data / K_FUND) ** 2))
    print(f"[INFO] measured kmax_data = {kmax_data:.6f}, qmax_data = {qmax_data}")

    pmean100 = np.mean(data100.p0_mocks, axis=0)
    pmean0 = np.mean(data0.p0_mocks, axis=0)

    def p100_best(kvals: np.ndarray) -> np.ndarray:
        return np.interp(kvals, k_dense, p100_dense)

    def p0_best(kvals: np.ndarray) -> np.ndarray:
        return np.interp(kvals, k_dense, p0_dense)

    def p100_meas_interp(kvals: np.ndarray) -> np.ndarray:
        return np.interp(kvals, data100.kcen, pmean100, left=0.0, right=0.0)

    def p0_meas_interp(kvals: np.ndarray) -> np.ndarray:
        return np.interp(kvals, data0.kcen, pmean0, left=0.0, right=0.0)

    def p100_meas_step(kvals: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(data100.kmax, kvals, side="left")
        out = np.zeros_like(kvals)
        mask = idx < len(pmean100)
        out[mask] = pmean100[idx[mask]]
        return out

    def p0_meas_step(kvals: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(data0.kmax, kvals, side="left")
        out = np.zeros_like(kvals)
        mask = idx < len(pmean0)
        out[mask] = pmean0[idx[mask]]
        return out

    xi100_best_full = xi_from_shellsum_with_pfunc(data100.scen, gq, qmax_full, p100_best)
    xi0_best_full = xi_from_shellsum_with_pfunc(data0.scen, gq, qmax_full, p0_best)
    xi100_best_cut = xi_from_shellsum_with_pfunc(data100.scen, gq, qmax_data, p100_best)
    xi0_best_cut = xi_from_shellsum_with_pfunc(data0.scen, gq, qmax_data, p0_best)
    xi100_meas_interp = xi_from_shellsum_with_pfunc(data100.scen, gq, qmax_data, p100_meas_interp)
    xi0_meas_interp = xi_from_shellsum_with_pfunc(data0.scen, gq, qmax_data, p0_meas_interp)
    xi100_meas_step = xi_from_shellsum_with_pfunc(data100.scen, gq, qmax_data, p100_meas_step)
    xi0_meas_step = xi_from_shellsum_with_pfunc(data0.scen, gq, qmax_data, p0_meas_step)

    rows: List[Dict[str, object]] = [
        build_metric_row("1Gpc_fnl100_ExpWindow", data100.scen, data100.xi_mean, data100.xi_std, xi100_exp, "fnl100"),
        build_metric_row("1Gpc_fnl100_AllDiscreteBestfit_k20", data100.scen, data100.xi_mean, data100.xi_std, xi100_best_full, "fnl100"),
        build_metric_row("1Gpc_fnl100_AllDiscreteBestfit_kdata", data100.scen, data100.xi_mean, data100.xi_std, xi100_best_cut, "fnl100"),
        build_metric_row("1Gpc_fnl100_MeasuredPkInterp_kdata", data100.scen, data100.xi_mean, data100.xi_std, xi100_meas_interp, "fnl100"),
        build_metric_row("1Gpc_fnl100_MeasuredPkStep_kdata", data100.scen, data100.xi_mean, data100.xi_std, xi100_meas_step, "fnl100"),
        build_metric_row("1Gpc_fnl0_ExpWindow", data0.scen, data0.xi_mean, data0.xi_std, xi0_exp, "fnl0"),
        build_metric_row("1Gpc_fnl0_AllDiscreteBestfit_k20", data0.scen, data0.xi_mean, data0.xi_std, xi0_best_full, "fnl0"),
        build_metric_row("1Gpc_fnl0_AllDiscreteBestfit_kdata", data0.scen, data0.xi_mean, data0.xi_std, xi0_best_cut, "fnl0"),
        build_metric_row("1Gpc_fnl0_MeasuredPkInterp_kdata", data0.scen, data0.xi_mean, data0.xi_std, xi0_meas_interp, "fnl0"),
        build_metric_row("1Gpc_fnl0_MeasuredPkStep_kdata", data0.scen, data0.xi_mean, data0.xi_std, xi0_meas_step, "fnl0"),
    ]

    with OUT_METRIC_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] saved metrics: {OUT_METRIC_CSV}")

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.8))
    for row_idx, (sample, xi_data, xi_std, s, curves) in enumerate([
        ("fnl100", data100.xi_mean, data100.xi_std, data100.scen, {
            "Exp": xi100_exp,
            "Best20": xi100_best_full,
            "BestData": xi100_best_cut,
            "MeasInterp": xi100_meas_interp,
            "MeasStep": xi100_meas_step,
        }),
        ("fnl0", data0.xi_mean, data0.xi_std, data0.scen, {
            "Exp": xi0_exp,
            "Best20": xi0_best_full,
            "BestData": xi0_best_cut,
            "MeasInterp": xi0_meas_interp,
            "MeasStep": xi0_meas_step,
        }),
    ]):
        ax = axes[row_idx, 0]
        ax.errorbar(s, s**2 * xi_data, yerr=s**2 * xi_std, fmt="o", ms=2.8, capsize=2, color="black", label="Measured")
        colors = {"Exp": "tab:red", "Best20": "tab:green", "BestData": "tab:olive", "MeasInterp": "tab:orange", "MeasStep": "tab:blue"}
        for name, arr in curves.items():
            ax.plot(s, s**2 * arr, "-", lw=1.5, color=colors[name], label=name)
        ax.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(f"1Gpc {sample}: curves")
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2 \xi_0(r)$")
        ax.legend(fontsize=8)

        axr = axes[row_idx, 1]
        r2_data = s**2 * xi_data
        r2_std = np.maximum(s**2 * xi_std, 1e-12)
        for name, arr in curves.items():
            resid = (r2_data - s**2 * arr) / r2_std
            axr.plot(s, resid, "-", lw=1.2, color=colors[name], label=name)
        axr.axhline(0.0, color="black", lw=1.0)
        axr.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.axhline(-1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        axr.set_title(f"1Gpc {sample}: residuals")
        axr.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        axr.set_ylabel(r"$(Data-Model)/\sigma$")
        axr.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved figure: {OUT_FIG}")

    with OUT_SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("# Mission 8：1Gpc 直接用测量 P(k) 且只积到测量 k_max\n\n")
        f.write(f"测量的最后一个 k-bin 上限为 `k_max,data = {kmax_data:.6f}`。\n\n")
        f.write("| Sample | Method | mean_abs_sigma | chi2_ndof | mean_sigma |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for sample in ["fnl100", "fnl0"]:
            subset = [row for row in rows if row["sample"] == sample]
            subset = sorted(subset, key=lambda row: float(row["mean_abs_sigma"]))
            for row in subset:
                f.write(
                    f"| {sample} | {row['method']} | {float(row['mean_abs_sigma']):.4f} | "
                    f"{float(row['chi2_ndof']):.4f} | {float(row['mean_sigma']):.4f} |\n"
                )
            f.write("\n")
    print(f"[INFO] saved summary: {OUT_SUMMARY_MD}")
    print("[INFO] Mission8 measured-Pk 1Gpc cut-kdata done")


if __name__ == "__main__":
    main()
