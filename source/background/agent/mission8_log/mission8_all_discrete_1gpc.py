#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8：1Gpc 的“全离散”测试
================================

目标
----
按用户最新要求，测试下面这个版本：

1. 完全不用连续 FFTLog 部分；
2. 从最低非零模开始，到 kmax=20 为止，全部用离散求和；
3. 看 1Gpc fastPM 的 fnl=100 / fnl=0 情况下，这个“全离散”模型会怎么样。

代码大纲
--------
A. 读取 1Gpc fnl100 / fnl0 的测量 pk / pcf，并做 best-fit。
B. 用 FFT 卷积的方式精确计算三维离散 shell 简并度 g_q：
   - 一维平方数数组 A(q)
   - 三维计数 g = A * A * A
C. 构造全离散相关函数：
       xi_0(r) = (1/V) * sum_{q>=1, k_q<=kmax} g_q P(k_q) j0(k_q r)
D. 与 baseline / exp 窗口法比较。

说明
----
1. 这里的“全离散”针对的是 xi_0(r) 单极矩，因此按 shell 求和就足够；
   不需要逐模式平面波再做角平均。
2. 对 1Gpc，qmax=(kmax/k_f)^2 约为 1e7，借助 FFT 卷积仍然可以算。
3. 对 3Gpc，同样做法的 qmax 约为 9e7，量级明显更大；本脚本先只跑 1Gpc。
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
from scipy.fft import next_fast_len, rfft, irfft


# ============================================================
# 复用已有模块
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
MISSION5_DIR = THIS_DIR.parent / "mission5_log"
if str(MISSION5_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION5_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import task5_ir_window_solution_3gpc_multitype as m5
import mission8_exp_window_box_finish as m8finish


# ============================================================
# 常数与输出
# ============================================================

BOX_SIZE = 1000.0
K_FUND = 2.0 * np.pi / BOX_SIZE
KMAX_INT = 20.0
QMAX = int(np.floor((KMAX_INT / K_FUND) ** 2))
NMAX = int(np.floor(KMAX_INT / K_FUND))
FFTLOG_N = m5.FFTLOG_N
FFTLOG_PADDING = m5.FFTLOG_PADDING
KMIN_GLOBAL = 1.0e-4
EDGE_TAPER_FRAC = m5.EDGE_TAPER_FRAC

PK100_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat"
PCF100_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat"
PK0_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat"
PCF0_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut_fnl0/pcf_rsd_N*.dat"
RID_MIN = 1
RID_MAX = 50

OUT_METRIC_CSV = THIS_DIR / "mission8_all_discrete_1gpc_metrics.csv"
OUT_SUMMARY_MD = THIS_DIR / "mission8_all_discrete_1gpc_summary_20260313.md"
OUT_FIG = THIS_DIR / "mission8_all_discrete_1gpc_compare.png"


# ============================================================
# 辅助函数
# ============================================================

def spherical_bessel_j0(x: np.ndarray | float) -> np.ndarray:
    """
    球贝塞尔函数 j0(x)=sin(x)/x。
    """
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = x != 0.0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def compute_shell_degeneracy_fft(qmax: int, nmax: int) -> np.ndarray:
    """
    用 FFT 卷积计算三维离散 shell 的简并度 g_q。

    思路
    ----
    一维上，n^2=q 的表示数为：
    - q=0 时只有 n=0，所以计数 1
    - q>0 时有 ±n，所以计数 2

    记这个一维数组为 A(q)，则三维简并度就是三次卷积：
        g = A * A * A

    参数
    ----------
    qmax : int
        需要保留到的最大 q。
    nmax : int
        允许的最大 |n|，这里取 floor(kmax/k_f)。

    返回
    ----------
    ndarray
        长度 qmax+1 的整数数组 g_q。
    """
    a = np.zeros(qmax + 1, dtype=np.float64)
    a[0] = 1.0
    n = np.arange(1, nmax + 1, dtype=np.int64)
    sq = np.square(n)
    sq = sq[sq <= qmax]
    a[sq] = 2.0

    nfft = next_fast_len(3 * qmax + 1)
    fa = rfft(a, n=nfft)
    conv = irfft(fa * fa * fa, n=nfft)
    g = np.rint(conv[: qmax + 1]).astype(np.int64)
    return g


def xi_from_full_discrete_shell_sum(
    s_data: np.ndarray,
    gq: np.ndarray,
    k_dense: np.ndarray,
    p_dense: np.ndarray,
    box_size: float,
    q_chunk: int = 500000,
) -> np.ndarray:
    """
    用全离散 shell 求和计算 xi_0(r)。

    公式
    ----------
        xi_0(r) = (1/V) * sum_{q>=1}^{qmax} g_q P(k_q) j0(k_q r)

    参数
    ----------
    s_data : ndarray
        测量的 r/s 网格。
    gq : ndarray
        每个 q 的简并度。
    k_dense, p_dense : ndarray
        稠密理论 P(k) 插值表。
    box_size : float
        盒长。
    q_chunk : int
        分块大小，防止一次性分配过大内存。

    返回
    ----------
    ndarray
        与 s_data 同长度的 xi_0(r)。
    """
    volume = float(box_size) ** 3
    xi = np.zeros_like(s_data, dtype=np.float64)

    q_nonzero = np.nonzero(gq[1:])[0] + 1
    for start in range(0, q_nonzero.size, q_chunk):
        q_block = q_nonzero[start : start + q_chunk]
        kvals = K_FUND * np.sqrt(q_block.astype(np.float64))
        pvals = np.interp(kvals, k_dense, p_dense)
        weights = gq[q_block].astype(np.float64) * pvals
        j0 = spherical_bessel_j0(np.outer(kvals, s_data))
        xi += np.sum(weights[:, None] * j0, axis=0) / volume
    return xi


def build_metric_row(method: str, s: np.ndarray, xi_data: np.ndarray, xi_std: np.ndarray, xi_model: np.ndarray) -> Dict[str, object]:
    """
    统一包装一行指标。
    """
    return dict(m5.compute_alignment_metrics(s, xi_data, xi_std, xi_model, tag=method))


def main() -> None:
    """
    执行 1Gpc 的全离散测试。
    """
    print("[INFO] Mission8 1Gpc all-discrete start")
    print(f"[INFO] K_FUND = {K_FUND:.8f}, QMAX = {QMAX}, NMAX = {NMAX}")

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

    k_dense = np.geomspace(K_FUND, KMAX_INT, 300000)
    p100_dense = m5.build_theory_p0(k_dense, bestfit100)
    p0_dense = m5.build_theory_p0(k_dense, bestfit0)

    # 参照方法
    s = data100.scen
    xi_baseline = m5.evaluate_unwindowed_with_kmin(s, np.geomspace(KMIN_GLOBAL / FFTLOG_PADDING, KMAX_INT * FFTLOG_PADDING, FFTLOG_N), m5.build_theory_p0(np.geomspace(KMIN_GLOBAL / FFTLOG_PADDING, KMAX_INT * FFTLOG_PADDING, FFTLOG_N), bestfit100), kmin=K_FUND, tag="1Gpc_baseline")
    xi_exp = m8finish.evaluate_exp_window_model(
        s_data=s,
        k_grid=np.geomspace(KMIN_GLOBAL / FFTLOG_PADDING, KMAX_INT * FFTLOG_PADDING, FFTLOG_N),
        p0_grid=m5.build_theory_p0(np.geomspace(KMIN_GLOBAL / FFTLOG_PADDING, KMAX_INT * FFTLOG_PADDING, FFTLOG_N), bestfit100),
        k_fund=K_FUND,
        x_power=4.0,
    )

    # 全离散简并度
    print("[INFO] computing full discrete shell degeneracy via FFT convolution...")
    gq = compute_shell_degeneracy_fft(QMAX, NMAX)
    print("[INFO] shell degeneracy computed")

    # 全离散 xi
    xi_discrete100 = xi_from_full_discrete_shell_sum(s, gq, k_dense, p100_dense, BOX_SIZE)
    xi_discrete0 = xi_from_full_discrete_shell_sum(data0.scen, gq, k_dense, p0_dense, BOX_SIZE)

    rows: List[Dict[str, object]] = [
        build_metric_row("1Gpc_Baseline", s, data100.xi_mean, data100.xi_std, xi_baseline),
        build_metric_row("1Gpc_ExpWindow", s, data100.xi_mean, data100.xi_std, xi_exp),
        build_metric_row("1Gpc_AllDiscrete", s, data100.xi_mean, data100.xi_std, xi_discrete100),
        build_metric_row("1Gpc_fnl0_AllDiscrete", data0.scen, data0.xi_mean, data0.xi_std, xi_discrete0),
    ]

    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with OUT_METRIC_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] saved metrics: {OUT_METRIC_CSV}")

    # 图
    fig, axes = plt.subplots(2, 1, figsize=(9.3, 9.0))
    ax = axes[0]
    ax.errorbar(s, s**2 * data100.xi_mean, yerr=s**2 * data100.xi_std,
                fmt="o", ms=3.2, capsize=2, color="black", label="Measured mean")
    ax.plot(s, s**2 * xi_baseline, "-", lw=1.8, color="tab:blue", label="Baseline")
    ax.plot(s, s**2 * xi_exp, "-", lw=1.8, color="tab:red", label="Exp window x=4")
    ax.plot(s, s**2 * xi_discrete100, "-", lw=1.8, color="tab:green", label="All discrete to kmax=20")
    ax.axvline(m5.LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
    ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$r^2 \xi_0(r)$")
    ax.set_title("Mission8: 1Gpc all-discrete test")
    ax.legend(fontsize=9)

    axr = axes[1]
    r2_data = s**2 * data100.xi_mean
    r2_std = np.maximum(s**2 * data100.xi_std, 1e-12)
    for arr, color, label, marker in [
        (xi_baseline, "tab:blue", "Baseline", "o"),
        (xi_exp, "tab:red", "Exp window", "s"),
        (xi_discrete100, "tab:green", "All discrete", "^"),
    ]:
        resid = (r2_data - s**2 * arr) / r2_std
        axr.plot(s, resid, marker + "-", ms=3.0, lw=1.3, color=color, label=label)
    axr.axhline(0.0, color="black", lw=1.0)
    axr.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
    axr.axhline(-1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
    axr.axvline(m5.LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
    axr.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
    axr.set_ylabel(r"$(Data-Model)/\sigma$")
    axr.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved figure: {OUT_FIG}")

    # 总结
    def get_row(name: str) -> Dict[str, object]:
        for row in rows:
            if row["method"] == name:
                return row
        raise KeyError(name)

    rb = get_row("1Gpc_Baseline")
    re = get_row("1Gpc_ExpWindow")
    rd = get_row("1Gpc_AllDiscrete")
    r0 = get_row("1Gpc_fnl0_AllDiscrete")

    with OUT_SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("# Mission 8：1Gpc 全离散测试\n\n")
        f.write("这次按你的要求，不再保留任何连续高-k 部分，而是从最低非零模开始，到 `kmax=20` 为止全部用离散 shell 求和。\n\n")
        f.write("## 结果\n\n")
        f.write("| 方法 | mean|Δ/σ| | chi2/ndof | mean(Δ/σ) |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in [rb, re, rd]:
            f.write(f"| {row['method']} | {float(row['mean_abs_sigma']):.4f} | {float(row['chi2_ndof']):.4f} | {float(row['mean_sigma']):.4f} |\n")
        f.write("\n")
        f.write(f"- fnl0 的全离散 sanity：mean|Δ/σ|={float(r0['mean_abs_sigma']):.4f}, chi2/ndof={float(r0['chi2_ndof']):.4f}\n")
        f.write(f"- 本次全离散求和使用 `qmax={QMAX}`，即直到 `kmax=20`。\n\n")
        f.write("## 解释\n\n")
        f.write("1. 这个版本已经把“连续高-k 部分会不会污染结论”这个问题排除了，因为完全没有连续部分。\n")
        f.write("2. 如果全离散结果仍然明显不如 exp 窗口，那说明当前 1Gpc 的关键问题不是“连续拼接”本身，而是更深层的理论/测量映射差异。\n")
        f.write("3. 由于这里算的是 xi0(r) 单极矩，所以全离散模式求和与全离散 shell 求和是等价的；用 shell 求和只是为了让计算量可控。\n")

    print(f"[INFO] saved summary: {OUT_SUMMARY_MD}")
    print("[INFO] Mission8 1Gpc all-discrete done")


if __name__ == "__main__":
    main()
