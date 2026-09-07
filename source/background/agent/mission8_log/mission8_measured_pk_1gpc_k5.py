#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8：1Gpc 直接用重测 P(k)（kmax≈5）做全离散 2PCF 建模
============================================================

目标
----
1. 读取新测量的 `pk_remeasure_g1024_k5` / `pk_remeasure_g1024_k5_fnl0`；
2. 直接采用测量均值 P0(k)，不经过理论外延；
3. 全离散 shell 求和只做到测量数据自身的最后一个 k-bin；
4. 与已有基准比较：
   - Exp-window best-fit
   - All-discrete best-fit kmax=20
   - All-discrete best-fit 截到 kmax,data
   - Measured P(k) + bin-step
   - Measured P(k) + linear-interp

说明
----
这个脚本默认在 desilike 环境运行，因为它复用了 mission5 的拟合接口。
"""

from __future__ import annotations

import csv
import sys
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

BOX_SIZE = 1000.0
K_FUND = 2.0 * np.pi / BOX_SIZE
RID_MIN = 1
RID_MAX = 50

PK100_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_remeasure_g1024_k5/pk_rsd_N*.dat"
PK0_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_remeasure_g1024_k5_fnl0/pk_rsd_N*.dat"
PCF100_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat"
PCF0_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut_fnl0/pcf_rsd_N*.dat"

OUT_METRIC_CSV = THIS_DIR / "mission8_measured_pk_1gpc_k5_metrics.csv"
OUT_SUMMARY_MD = THIS_DIR / "mission8_measured_pk_1gpc_k5_summary_20260313.md"
OUT_FIG = THIS_DIR / "mission8_measured_pk_1gpc_k5_compare.png"


def compute_shell_degeneracy_fft(qmax: int, nmax: int) -> np.ndarray:
    """用 FFT 卷积计算三维壳层简并度 g_q。"""
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
    """球贝塞尔函数 j0(x)=sin(x)/x。"""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = x != 0.0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def xi_from_shellsum_with_pfunc(
    s_data: np.ndarray,
    gq: np.ndarray,
    qmax_use: int,
    pfunc,
    q_chunk: int = 400000,
) -> np.ndarray:
    """按离散壳层对 P(k_q) 求和，得到 xi_0(r)。"""
    volume = BOX_SIZE ** 3
    xi = np.zeros_like(s_data, dtype=np.float64)
    q_nonzero = np.nonzero(gq[1 : qmax_use + 1])[0] + 1
    for start in range(0, q_nonzero.size, q_chunk):
        q_block = q_nonzero[start : start + q_chunk]
        kvals = K_FUND * np.sqrt(q_block.astype(np.float64))
        pvals = pfunc(kvals)
        j0 = spherical_bessel_j0(np.outer(kvals, s_data))
        xi += np.sum((gq[q_block].astype(np.float64) * pvals)[:, None] * j0, axis=0) / volume
    return xi


def build_metric_row(
    method: str,
    s: np.ndarray,
    xi_data: np.ndarray,
    xi_std: np.ndarray,
    xi_model: np.ndarray,
    sample: str,
) -> Dict[str, object]:
    row = dict(m5.compute_alignment_metrics(s, xi_data, xi_std, xi_model, tag=method))
    row["sample"] = sample
    return row


def require_inputs() -> None:
    """在正式计算前检查输入文件是否已经生成。"""
    import glob

    n100 = len(glob.glob(PK100_GLOB))
    n0 = len(glob.glob(PK0_GLOB))
    if n100 == 0 or n0 == 0:
        raise SystemExit(
            "重测 P(k) 文件尚未生成完成。\n"
            f"fnl100 files = {n100}, fnl0 files = {n0}\n"
            "请等待 sbatch 任务完成后再运行本脚本。"
        )


def summarize_best(rows: List[Dict[str, object]], sample: str) -> Dict[str, object]:
    """取同一样本下 mean_abs_sigma 最小的一行。"""
    subset = [row for row in rows if row["sample"] == sample]
    return min(subset, key=lambda row: float(row["mean_abs_sigma"]))


def main() -> None:
    print("[INFO] Mission8 measured-Pk 1Gpc k5 start")
    require_inputs()

    data100_raw = m5.load_mock_data("1Gpc_fnl100_remeasure", PK100_GLOB, PCF100_GLOB, RID_MIN, RID_MAX)
    data0_raw = m5.load_mock_data("1Gpc_fnl0_remeasure", PK0_GLOB, PCF0_GLOB, RID_MIN, RID_MAX)
    data100 = m8finish.filter_zero_std_pk_bins(data100_raw, tag="1Gpc fnl100 k5")
    data0 = m8finish.filter_zero_std_pk_bins(data0_raw, tag="1Gpc fnl0 k5")

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

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.8))
    plot_specs = [
        ("fnl100", data100.scen, data100.xi_mean, data100.xi_std, {
            "Exp": xi100_exp,
            "Best20": xi100_best_full,
            "BestData": xi100_best_cut,
            "MeasStep": xi100_meas_step,
            "MeasInterp": xi100_meas_interp,
        }),
        ("fnl0", data0.scen, data0.xi_mean, data0.xi_std, {
            "Exp": xi0_exp,
            "Best20": xi0_best_full,
            "BestData": xi0_best_cut,
            "MeasStep": xi0_meas_step,
            "MeasInterp": xi0_meas_interp,
        }),
    ]
    colors = {
        "Exp": "tab:red",
        "Best20": "tab:green",
        "BestData": "tab:olive",
        "MeasStep": "tab:blue",
        "MeasInterp": "tab:orange",
    }

    for row_idx, (sample, s, xi_data, xi_std, curves) in enumerate(plot_specs):
        ax = axes[row_idx, 0]
        ax.errorbar(s, s ** 2 * xi_data, yerr=s ** 2 * xi_std, fmt="o", ms=2.8, capsize=2, color="black", label="Measured")
        for name, arr in curves.items():
            ax.plot(s, s ** 2 * arr, "-", lw=1.5, color=colors[name], label=name)
        ax.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(f"1Gpc {sample}: curves")
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2 \xi_0(r)$")
        ax.legend(fontsize=8)

        axr = axes[row_idx, 1]
        r2_data = s ** 2 * xi_data
        r2_std = np.maximum(s ** 2 * xi_std, 1e-12)
        for name, arr in curves.items():
            resid = (r2_data - s ** 2 * arr) / r2_std
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

    best100 = summarize_best(rows, "fnl100")
    best0 = summarize_best(rows, "fnl0")
    with OUT_SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("# Mission 8：1Gpc 重测 P(k) 到 kmax≈5 的全离散建模\n\n")
        f.write(f"- 低-k 基模：`k_f = {K_FUND:.8f}`\n")
        f.write(f"- 测量数据公共上限：`kmax_data = {kmax_data:.6f}`\n")
        f.write(f"- `fnl100` 最优方法：`{best100['method']}`，`mean_abs_sigma={float(best100['mean_abs_sigma']):.4f}`，`mean_sigma={float(best100['mean_sigma']):.4f}`\n")
        f.write(f"- `fnl0` 最优方法：`{best0['method']}`，`mean_abs_sigma={float(best0['mean_abs_sigma']):.4f}`，`mean_sigma={float(best0['mean_sigma']):.4f}`\n\n")
        f.write("## 全部指标\n\n")
        f.write("| sample | method | mean_abs_sigma | mean_sigma | chi2_ndof |\n")
        f.write("| --- | --- | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(
                f"| {row['sample']} | {row['method']} | "
                f"{float(row['mean_abs_sigma']):.4f} | {float(row['mean_sigma']):.4f} | "
                f"{float(row['chi2_ndof']):.4f} |\n"
            )
    print(f"[INFO] saved summary: {OUT_SUMMARY_MD}")


if __name__ == "__main__":
    main()
