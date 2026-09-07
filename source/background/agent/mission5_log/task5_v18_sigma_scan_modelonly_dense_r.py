#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务5 v18：纯模型 sigma 扫描（密集 r 网格）

目的
----
检查仅改变模型中的 sigmas（小尺度阻尼参数）时，
2PCF 在大尺度（r>=150 Mpc/h）是否也会被明显影响。

要求口径
--------
- 只画模型曲线，不叠加测量数据
- r 取值更密集
- 积分使用 baseline 口径：kmin = 2pi/L（按盒长分别设置）
- kmax = 20，保留 FFTLog 内置 log-k taper
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MISSION5_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission5_log"
sys.path.insert(0, MISSION5_DIR)

import task5_ir_window_solution_3gpc_multitype as m5

# 盒长与积分下限
BOX_1GPC = 1000.0
BOX_3GPC = 3000.0
K_FUND_1GPC = 2.0 * np.pi / BOX_1GPC
K_FUND_3GPC = 2.0 * np.pi / BOX_3GPC

# 扫描参数（覆盖 1Gpc/3Gpc 既往 best-fit 附近及更宽范围）
SIGMAS_SCAN = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 2.6, 3.2, 4.0]

# 密集 r 网格（仅模型评估）
R_MIN = 20.0
R_MAX = 500.0
R_N = 1200
R_DENSE = np.linspace(R_MIN, R_MAX, R_N)
R_SPLIT = 150.0

# 使用此前 fnl0 的 box-wise best-fit（固定 fnl_loc, b1, sn0，仅扫 sigmas）
# 来源：mission5 已有 fnl0 PK best-fit 结果
PARAMS_1GPC_BASE = {
    "fnl_loc": 4.9277,
    "b1": 2.8018,
    "sn0": 0.0,
    "sigmas": 2.572251,
}
PARAMS_3GPC_BASE = {
    "fnl_loc": 7.0638,
    "b1": 2.8186,
    "sn0": 0.0,
    "sigmas": 1.135544,
}

OUT_FIG = os.path.join(MISSION5_DIR, "task5_v18_sigma_scan_modelonly_dense_r.png")
OUT_TXT = os.path.join(MISSION5_DIR, "task5_v18_sigma_scan_modelonly_dense_r_summary.txt")

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"


def compute_xi_curve_for_sigma(
    k_grid: np.ndarray,
    r_dense: np.ndarray,
    params_base: Dict[str, float],
    sigma_val: float,
    kmin_box: float,
) -> np.ndarray:
    """给定 sigma 计算模型 xi(r) 并插值到目标 r 网格。"""
    p = dict(params_base)
    p["sigmas"] = float(sigma_val)

    p0 = m5.build_theory_p0(k_grid, p)
    taper = m5.build_log_taper_window(
        k_grid,
        kmin=kmin_box,
        kmax=m5.KMAX_INT,
        frac=m5.EDGE_TAPER_FRAC,
    )
    r_grid, xi_grid = m5.xi_fftlog_from_effective_p0(k_grid, p0 * taper)
    return m5.interp_xi_to_s(r_dense, r_grid, xi_grid)


def summarize_large_scale_spread(
    r_dense: np.ndarray,
    curves_r2xi: np.ndarray,
    sigmas: List[float],
) -> Tuple[float, float, float, float]:
    """量化 r>=R_SPLIT 上的模型包络变化。"""
    mask = r_dense >= R_SPLIT
    x = curves_r2xi[:, mask]

    envelope = np.max(x, axis=0) - np.min(x, axis=0)
    mean_env = float(np.mean(envelope))
    max_env = float(np.max(envelope))

    # 用 sigma=1.0 作为参考，评估最大绝对偏移
    idx_ref = int(np.argmin(np.abs(np.array(sigmas) - 1.0)))
    ref = x[idx_ref]
    max_abs_shift = float(np.max(np.max(np.abs(x - ref[None, :]), axis=1)))

    # 与参考振幅的相对尺度（避免除零）
    ref_amp = float(np.mean(np.abs(ref)))
    rel = float(max_abs_shift / max(ref_amp, 1e-12))

    return mean_env, max_env, max_abs_shift, rel


def main() -> None:
    old_p = float(m5.FIXED_P)

    # 按任务当前统一设置
    m5.FIXED_P = 1.1

    try:
        k_grid = np.geomspace(
            m5.KMIN_GLOBAL / m5.FFTLOG_PADDING,
            m5.KMAX_INT * m5.FFTLOG_PADDING,
            m5.FFTLOG_N,
        )

        curves_1gpc = []
        curves_3gpc = []

        for sig in SIGMAS_SCAN:
            xi1 = compute_xi_curve_for_sigma(k_grid, R_DENSE, PARAMS_1GPC_BASE, sig, K_FUND_1GPC)
            xi3 = compute_xi_curve_for_sigma(k_grid, R_DENSE, PARAMS_3GPC_BASE, sig, K_FUND_3GPC)
            curves_1gpc.append(xi1)
            curves_3gpc.append(xi3)

        curves_1gpc = np.array(curves_1gpc)  # [Nsigma, Nr]
        curves_3gpc = np.array(curves_3gpc)

        r2_1 = (R_DENSE[None, :] ** 2) * curves_1gpc
        r2_3 = (R_DENSE[None, :] ** 2) * curves_3gpc

        # 量化大尺度包络
        mean_env_1, max_env_1, max_shift_1, rel_1 = summarize_large_scale_spread(R_DENSE, r2_1, SIGMAS_SCAN)
        mean_env_3, max_env_3, max_shift_3, rel_3 = summarize_large_scale_spread(R_DENSE, r2_3, SIGMAS_SCAN)

        # 画图：全区间 + 大尺度放大
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6), sharex="col")
        cmap = plt.cm.viridis

        for i, sig in enumerate(SIGMAS_SCAN):
            color = cmap(i / max(len(SIGMAS_SCAN) - 1, 1))
            label = f"sigmas={sig:g}"

            axes[0, 0].plot(R_DENSE, r2_1[i], lw=1.8, color=color, label=label)
            axes[0, 1].plot(R_DENSE, r2_3[i], lw=1.8, color=color, label=label)
            axes[1, 0].plot(R_DENSE, r2_1[i], lw=1.8, color=color)
            axes[1, 1].plot(R_DENSE, r2_3[i], lw=1.8, color=color)

        axes[0, 0].set_title("1Gpc model-only: dense r (20-500)")
        axes[0, 1].set_title("3Gpc model-only: dense r (20-500)")

        axes[0, 0].set_ylabel(r"$r^2\xi_0(r)$")
        axes[1, 0].set_ylabel(r"$r^2\xi_0(r)$")
        axes[1, 0].set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        axes[1, 1].set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")

        # 大尺度放大
        for ax in [axes[1, 0], axes[1, 1]]:
            ax.set_xlim(R_SPLIT, R_MAX)

        # 顶部看全范围
        for ax in [axes[0, 0], axes[0, 1]]:
            ax.set_xlim(R_MIN, R_MAX)
            ax.axvline(R_SPLIT, color="gray", ls="--", lw=1.0, alpha=0.8)

        axes[1, 0].axvline(R_SPLIT, color="gray", ls="--", lw=1.0, alpha=0.8)
        axes[1, 1].axvline(R_SPLIT, color="gray", ls="--", lw=1.0, alpha=0.8)

        # 图例统一放在右上图
        axes[0, 1].legend(fontsize=8, ncol=2, loc="best")

        fig.suptitle(
            "Effect of varying sigmas on model 2PCF (no data, baseline kmin=2pi/L)",
            y=0.995,
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(OUT_FIG, bbox_inches="tight")
        plt.close(fig)

        with open(OUT_TXT, "w", encoding="utf-8") as f:
            f.write("任务5 v18：纯模型 sigmas 扫描（密集 r）\n")
            f.write("====================================\n\n")
            f.write("设置：\n")
            f.write("- 只画模型曲线，不叠加数据\n")
            f.write("- r 网格: 20~500, N=1200（密集）\n")
            f.write(f"- sigmas 扫描: {SIGMAS_SCAN}\n")
            f.write("- 积分口径: kmin=2pi/L（分盒子）, kmax=20, taper保留\n")
            f.write(f"- 固定 p={m5.FIXED_P:.2f}\n\n")

            f.write("模型基准参数（仅 sigmas 变化）：\n")
            f.write(f"- 1Gpc: fnl_loc={PARAMS_1GPC_BASE['fnl_loc']:.4f}, b1={PARAMS_1GPC_BASE['b1']:.4f}, sn0={PARAMS_1GPC_BASE['sn0']:.1f}\n")
            f.write(f"- 3Gpc: fnl_loc={PARAMS_3GPC_BASE['fnl_loc']:.4f}, b1={PARAMS_3GPC_BASE['b1']:.4f}, sn0={PARAMS_3GPC_BASE['sn0']:.1f}\n\n")

            f.write("大尺度(r>=150)包络变化（基于 r^2 xi）：\n")
            f.write(f"- 1Gpc: mean_envelope={mean_env_1:.4e}, max_envelope={max_env_1:.4e}, max_abs_shift_vs_sigma1={max_shift_1:.4e}, rel_shift={rel_1:.4f}\n")
            f.write(f"- 3Gpc: mean_envelope={mean_env_3:.4e}, max_envelope={max_env_3:.4e}, max_abs_shift_vs_sigma1={max_shift_3:.4e}, rel_shift={rel_3:.4f}\n")

        print("[INFO] done")
        print(f"[INFO] fig: {OUT_FIG}")
        print(f"[INFO] summary: {OUT_TXT}")
        print("[INFO] large-scale spread (r>=150)")
        print(f"[INFO] 1Gpc: mean_env={mean_env_1:.4e}, max_env={max_env_1:.4e}, rel_shift={rel_1:.4f}")
        print(f"[INFO] 3Gpc: mean_env={mean_env_3:.4e}, max_env={max_env_3:.4e}, rel_shift={rel_3:.4f}")

    finally:
        m5.FIXED_P = old_p


if __name__ == "__main__":
    main()
