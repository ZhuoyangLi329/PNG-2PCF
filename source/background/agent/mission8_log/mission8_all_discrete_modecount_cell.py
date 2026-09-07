#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8：用 mode-count preserving k-cell 替代 delta-shell
===========================================================

物理动机
--------
原始全离散方法把每个离散 shell 写成径向 delta：

    P_eff(k) = Σ_q C_q δ(k-k_q)

这是严格的有限盒 Fourier 模表达，但如果我们把连续理论 P(k)
视为离散壳层对连续相空间的采样，那么更物理的近似是：

1. 每个 shell 不再对应“零宽度 delta”；
2. 而对应一个有限的 radial k-cell；
3. 这个 cell 的体积不靠调参，而是直接由该 shell 的离散模式数 g_q 决定。

具体地，要求每个 cell 的连续相空间模数恰好等于离散模数：

    V / (2π)^3 * 4π/3 * (k_hi^3 - k_lo^3) = g_q

即

    k_hi^3 - k_lo^3 = 6π² g_q / V

这样可保证：

1. 每个 shell 的 mode count 被严格保留；
2. 在 r=0 处，cell 模型与原始全离散的权重完全一致；
3. 与“人为高斯展宽”不同，这里没有额外自由参数。
"""

from __future__ import annotations

import csv
import math
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
KMAX_INT = 20.0
FFTLOG_N = m5.FFTLOG_N
FFTLOG_PADDING = m5.FFTLOG_PADDING
LARGE_SCALE_MIN = m5.LARGE_SCALE_MIN

OUT_METRIC_CSV = THIS_DIR / "mission8_all_discrete_modecount_cell_metrics.csv"
OUT_SUMMARY_MD = THIS_DIR / "mission8_all_discrete_modecount_cell_summary_20260313.md"
OUT_FIG = THIS_DIR / "mission8_all_discrete_modecount_cell_compare.png"


@dataclass
class BoxConfig:
    """
    单个盒长配置。
    """

    tag: str
    box_size: float
    pk100_glob: str
    pcf100_glob: str
    pk0_glob: str
    pcf0_glob: str
    rid_min: int
    rid_max: int
    drop_zero_std: bool
    x_power: float
    q_chunk: int


BOXES = [
    BoxConfig(
        tag="1Gpc",
        box_size=1000.0,
        pk100_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat",
        pcf100_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat",
        pk0_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
        pcf0_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut_fnl0/pcf_rsd_N*.dat",
        rid_min=1,
        rid_max=50,
        drop_zero_std=True,
        x_power=4.0,
        q_chunk=400000,
    ),
    BoxConfig(
        tag="3Gpc",
        box_size=3000.0,
        pk100_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat",
        pcf100_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",
        pk0_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
        pcf0_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat",
        rid_min=2,
        rid_max=80,
        drop_zero_std=False,
        x_power=12.0,
        q_chunk=250000,
    ),
]


def compute_shell_degeneracy_fft(qmax: int, nmax: int) -> np.ndarray:
    """
    用 FFT 卷积精确计算三维离散 shell 简并度 g_q。
    """
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
    """
    球贝塞尔函数 j0(x)=sin(x)/x。
    """
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = x != 0.0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def build_metric_row(
    method: str,
    s: np.ndarray,
    xi_data: np.ndarray,
    xi_std: np.ndarray,
    xi_model: np.ndarray,
    box: str,
    sample: str,
) -> Dict[str, object]:
    """
    统一包装一行指标。
    """
    row = dict(m5.compute_alignment_metrics(s, xi_data, xi_std, xi_model, tag=method))
    row["box"] = box
    row["sample"] = sample
    return row


def cell_kernel_center(s_data: np.ndarray, k_lo: np.ndarray, k_hi: np.ndarray, coeff_prefac: float) -> np.ndarray:
    """
    计算常数 P(k)=P_q 的 radial k-cell 对 xi(r) 的中心值 kernel。

    公式
    ----
    单个 cell 的贡献为

        K_cell(r) = 1/(2π²) ∫_{k_lo}^{k_hi} k² j0(kr) dk
                  = [sin(kr) - kr cos(kr)]_{k_lo}^{k_hi} / (2π² r³)

    在 r -> 0 极限下，

        K_cell(0) = (k_hi^3 - k_lo^3) / (6π²)

    若 cell 满足 mode-count preserving，
    该极限就正好等于 g_q / V。
    """
    s_data = np.asarray(s_data, dtype=float)
    k_lo = np.asarray(k_lo, dtype=float)
    k_hi = np.asarray(k_hi, dtype=float)

    r = s_data[None, :]
    x_lo = np.outer(k_lo, s_data)
    x_hi = np.outer(k_hi, s_data)
    numer = (np.sin(x_hi) - x_hi * np.cos(x_hi)) - (np.sin(x_lo) - x_lo * np.cos(x_lo))
    denom = 2.0 * np.pi**2 * np.maximum(r**3, 1.0e-300)
    out = numer / denom

    # 对可能极接近 r=0 的情况做极限保护；当前数据 r>0，但这里保持公式完整。
    small = s_data < 1.0e-10
    if np.any(small):
        out[:, small] = (k_hi[:, None]**3 - k_lo[:, None]**3) / (6.0 * np.pi**2)

    return coeff_prefac * out


def xi_discrete_and_modecount_cell_for_two_models(
    s_data: np.ndarray,
    gq: np.ndarray,
    k_fund: float,
    k_dense: np.ndarray,
    p_dense_a: np.ndarray,
    p_dense_b: np.ndarray,
    box_size: float,
    q_chunk: int,
) -> Dict[str, np.ndarray]:
    """
    在同一轮 q-shell 扫描中，同时计算：

    1. 原始 delta-shell 全离散
    2. mode-count preserving cell 模型
    """
    volume = float(box_size) ** 3
    xi_a_disc = np.zeros_like(s_data, dtype=np.float64)
    xi_b_disc = np.zeros_like(s_data, dtype=np.float64)
    xi_a_cell = np.zeros_like(s_data, dtype=np.float64)
    xi_b_cell = np.zeros_like(s_data, dtype=np.float64)

    q_nonzero = np.nonzero(gq[1:])[0] + 1
    cumulative_modes = 0.0
    coeff_modecount = 1.0

    for start in range(0, q_nonzero.size, q_chunk):
        q_block = q_nonzero[start : start + q_chunk]
        g_block = gq[q_block].astype(np.float64)
        kvals = k_fund * np.sqrt(q_block.astype(np.float64))
        pvals_a = np.interp(kvals, k_dense, p_dense_a)
        pvals_b = np.interp(kvals, k_dense, p_dense_b)

        j0 = spherical_bessel_j0(np.outer(kvals, s_data))
        xi_a_disc += np.sum((g_block * pvals_a)[:, None] * j0, axis=0) / volume
        xi_b_disc += np.sum((g_block * pvals_b)[:, None] * j0, axis=0) / volume

        n_lo = cumulative_modes + np.concatenate(([0.0], np.cumsum(g_block[:-1])))
        n_hi = cumulative_modes + np.cumsum(g_block)
        cumulative_modes += float(np.sum(g_block))

        k_lo = np.cbrt(6.0 * np.pi**2 * n_lo / volume)
        k_hi = np.cbrt(6.0 * np.pi**2 * n_hi / volume)
        kernel_cell = cell_kernel_center(s_data, k_lo, k_hi, coeff_prefac=coeff_modecount)
        xi_a_cell += np.sum(pvals_a[:, None] * kernel_cell, axis=0)
        xi_b_cell += np.sum(pvals_b[:, None] * kernel_cell, axis=0)

        if start == 0 or (start // q_chunk) % 50 == 0:
            print(f"[INFO] shell-sum progress: {start + q_block.size} / {q_nonzero.size}")

    return {
        "a_disc": xi_a_disc,
        "b_disc": xi_b_disc,
        "a_cell": xi_a_cell,
        "b_cell": xi_b_cell,
    }


def main() -> None:
    """
    执行 mode-count preserving cell 测试。
    """
    print("[INFO] Mission8 all-discrete modecount-cell start")
    all_rows: List[Dict[str, object]] = []
    plot_data: Dict[str, Dict[str, object]] = {}

    for cfg in BOXES:
        print(f"[INFO] ===== box {cfg.tag} =====")
        k_fund = 2.0 * np.pi / cfg.box_size
        qmax = int(np.floor((KMAX_INT / k_fund) ** 2))
        nmax = int(np.floor(KMAX_INT / k_fund))

        data100_raw = m5.load_mock_data(f"{cfg.tag}_fnl100", cfg.pk100_glob, cfg.pcf100_glob, cfg.rid_min, cfg.rid_max)
        data0_raw = m5.load_mock_data(f"{cfg.tag}_fnl0", cfg.pk0_glob, cfg.pcf0_glob, cfg.rid_min, cfg.rid_max)
        if cfg.drop_zero_std:
            data100 = m8finish.filter_zero_std_pk_bins(data100_raw, tag=f"{cfg.tag} fnl100")
            data0 = m8finish.filter_zero_std_pk_bins(data0_raw, tag=f"{cfg.tag} fnl0")
        else:
            data100 = data100_raw
            data0 = data0_raw

        fit100 = m5.fit_best_pk(data100)
        fit0 = m5.fit_best_pk(data0)
        bestfit100 = fit100["bestfit"]
        bestfit0 = fit0["bestfit"]
        print(f"[INFO] {cfg.tag} bestfit100 = {bestfit100}")
        print(f"[INFO] {cfg.tag} bestfit0   = {bestfit0}")

        k_fft = np.geomspace(KMIN_GLOBAL / FFTLOG_PADDING, KMAX_INT * FFTLOG_PADDING, FFTLOG_N)
        p100_fft = m5.build_theory_p0(k_fft, bestfit100)
        p0_fft = m5.build_theory_p0(k_fft, bestfit0)
        xi100_exp = m8finish.evaluate_exp_window_model(
            s_data=data100.scen,
            k_grid=k_fft,
            p0_grid=p100_fft,
            k_fund=k_fund,
            x_power=cfg.x_power,
        )
        xi0_exp = m8finish.evaluate_exp_window_model(
            s_data=data0.scen,
            k_grid=k_fft,
            p0_grid=p0_fft,
            k_fund=k_fund,
            x_power=cfg.x_power,
        )

        print(f"[INFO] {cfg.tag}: compute full-shell degeneracy")
        gq = compute_shell_degeneracy_fft(qmax, nmax)
        k_dense = np.geomspace(k_fund, KMAX_INT, 300000)
        p100_dense = m5.build_theory_p0(k_dense, bestfit100)
        p0_dense = m5.build_theory_p0(k_dense, bestfit0)

        xi_map = xi_discrete_and_modecount_cell_for_two_models(
            s_data=data100.scen,
            gq=gq,
            k_fund=k_fund,
            k_dense=k_dense,
            p_dense_a=p100_dense,
            p_dense_b=p0_dense,
            box_size=cfg.box_size,
            q_chunk=cfg.q_chunk,
        )

        rows_here = [
            build_metric_row(f"{cfg.tag}_fnl100_ExpWindow", data100.scen, data100.xi_mean, data100.xi_std, xi100_exp, cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl100_AllDiscrete", data100.scen, data100.xi_mean, data100.xi_std, xi_map["a_disc"], cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl100_ModeCountCell", data100.scen, data100.xi_mean, data100.xi_std, xi_map["a_cell"], cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl0_ExpWindow", data0.scen, data0.xi_mean, data0.xi_std, xi0_exp, cfg.tag, "fnl0"),
            build_metric_row(f"{cfg.tag}_fnl0_AllDiscrete", data0.scen, data0.xi_mean, data0.xi_std, xi_map["b_disc"], cfg.tag, "fnl0"),
            build_metric_row(f"{cfg.tag}_fnl0_ModeCountCell", data0.scen, data0.xi_mean, data0.xi_std, xi_map["b_cell"], cfg.tag, "fnl0"),
        ]
        all_rows.extend(rows_here)

        plot_data[cfg.tag] = {
            "fnl100": {
                "s": data100.scen,
                "xi_data": data100.xi_mean,
                "xi_std": data100.xi_std,
                "exp": xi100_exp,
                "disc": xi_map["a_disc"],
                "cell": xi_map["a_cell"],
            },
            "fnl0": {
                "s": data0.scen,
                "xi_data": data0.xi_mean,
                "xi_std": data0.xi_std,
                "exp": xi0_exp,
                "disc": xi_map["b_disc"],
                "cell": xi_map["b_cell"],
            },
        }

    keys: List[str] = []
    seen = set()
    for row in all_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with OUT_METRIC_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in keys} for row in all_rows])
    print(f"[INFO] saved metrics: {OUT_METRIC_CSV}")

    fig, axes = plt.subplots(4, 2, figsize=(12.8, 16.1))
    panels = [("1Gpc", "fnl100"), ("1Gpc", "fnl0"), ("3Gpc", "fnl100"), ("3Gpc", "fnl0")]
    for row_idx, (tag, sample) in enumerate(panels):
        item = plot_data[tag][sample]
        s = item["s"]

        ax = axes[row_idx, 0]
        ax.errorbar(s, s**2 * item["xi_data"], yerr=s**2 * item["xi_std"],
                    fmt="o", ms=2.8, capsize=2, color="black", label="Measured")
        ax.plot(s, s**2 * item["exp"], "-", lw=1.5, color="tab:red", label="Exp window")
        ax.plot(s, s**2 * item["disc"], "-", lw=1.6, color="tab:green", label="All discrete")
        ax.plot(s, s**2 * item["cell"], "-", lw=1.8, color="tab:orange", label="Mode-count cell")
        ax.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(f"{tag} {sample}: curves")
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2 \xi_0(r)$")
        ax.legend(fontsize=8)

        axr = axes[row_idx, 1]
        r2_data = s**2 * item["xi_data"]
        r2_std = np.maximum(s**2 * item["xi_std"], 1e-12)
        for arr, color, label, marker in [
            (item["exp"], "tab:red", "Exp", "s"),
            (item["disc"], "tab:green", "AllDisc", "^"),
            (item["cell"], "tab:orange", "ModeCell", "d"),
        ]:
            resid = (r2_data - s**2 * arr) / r2_std
            axr.plot(s, resid, marker + "-", ms=2.7, lw=1.2, color=color, label=label)
        axr.axhline(0.0, color="black", lw=1.0)
        axr.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.axhline(-1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        axr.set_title(f"{tag} {sample}: residuals")
        axr.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        axr.set_ylabel(r"$(Data-Model)/\sigma$")
        axr.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved figure: {OUT_FIG}")

    with OUT_SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("# Mission 8：mode-count preserving k-cell\n\n")
        f.write("本次把原始全离散的 `delta-shell` 改成了**无自由参数**的 `k-cell`：每个 shell 的连续 k 体积严格匹配它的离散模式数 `g_q`。\n\n")
        f.write("其中 `mean_sigma < 0` 表示模型整体偏高，`mean_sigma > 0` 表示模型整体偏低。\n\n")
        f.write("| Box | Sample | Method | mean_abs_sigma | chi2_ndof | mean_sigma |\n")
        f.write("|---|---|---|---:|---:|---:|\n")
        for tag in ["1Gpc", "3Gpc"]:
            for sample in ["fnl100", "fnl0"]:
                for method in ["ExpWindow", "AllDiscrete", "ModeCountCell"]:
                    key = f"{tag}_{sample}_{method}"
                    row = next(row for row in all_rows if row["method"] == key)
                    f.write(
                        f"| {tag} | {sample} | {method} | "
                        f"{float(row['mean_abs_sigma']):.4f} | "
                        f"{float(row['chi2_ndof']):.4f} | "
                        f"{float(row['mean_sigma']):.4f} |\n"
                    )
        f.write("\n## 物理解释\n\n")
        f.write("1. 原始全离散把每个 shell 视为零宽度 delta；mode-count cell 则把每个 shell 视为占据有限相空间体积的 radial cell。\n")
        f.write("2. cell 体积不靠调参，而是由 `g_q` 严格固定，因此这是无自由参数的修正。\n")
        f.write("3. 若该方法优于原始全离散，就说明“delta-shell 过于尖锐，而 mode-count preserving 的有限体积描述更接近真实有限盒统计”。\n")
    print(f"[INFO] saved summary: {OUT_SUMMARY_MD}")
    print("[INFO] Mission8 all-discrete modecount-cell done")


if __name__ == "__main__":
    main()
