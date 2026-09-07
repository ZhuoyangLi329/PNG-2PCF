#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本名称
--------
task5_v5_1gpc_transfer_test.py

目的
----
验证：把 3Gpc 上得到的窗口参数（tanh_log, param=8）保持不变，
迁移到 1Gpc 盒子时，2PCF 是否仍能被良好建模。

关键设定
--------
1) 数据：1Gpc fnl=100（pk_masscut + pcf_masscut）。
2) PK 拟合：固定 p=1.5（沿用你前面 p 扫描后的推荐值）。
3) 2PCF 比较三条曲线：
   - Baseline: kmin=2*pi/L_1gpc
   - TestA: kmin_global=1e-4, 无 IR 窗口
   - TestB_fixed: window=tanh_log, param=8（直接乘总 P0）

输出
----
- 一张图：r^2xi 曲线 + 残差子图
- 一份 summary.txt：定量指标与结论
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =====================
# 参数区
# =====================

MISSION5_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission5_log"
sys.path.insert(0, MISSION5_DIR)
import task5_ir_window_solution_3gpc_multitype as m5

# 1Gpc 数据路径
PK_1GPC_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat"
PCF_1GPC_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat"

# 1Gpc 盒长
BOX_SIZE_1GPC = 1000.0
K_FUND_1GPC = 2.0 * np.pi / BOX_SIZE_1GPC

# 固定设定（来自前面任务）
P_FIXED = 1.5
KMIN_GLOBAL = 1e-4
KMAX_INT = m5.KMAX_INT
EDGE_TAPER_FRAC = m5.EDGE_TAPER_FRAC

# 固定窗口参数（从 3Gpc 最优迁移）
WINDOW_TYPE = "tanh_log"
WINDOW_PARAM = 8.0
USE_PNG_ONLY_WINDOW = False  # 直接加窗口：对总 P0 乘窗

# 输出
OUT_FIG = os.path.join(MISSION5_DIR, "task5_v5_1gpc_fixed_window_transfer.png")
OUT_SUMMARY = os.path.join(MISSION5_DIR, "task5_v5_1gpc_fixed_window_transfer_summary.txt")

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"


# =====================
# 工具函数
# =====================

def evaluate_unwindowed_xi(
    s_data: np.ndarray,
    k_grid: np.ndarray,
    p0_base: np.ndarray,
    kmin: float,
) -> np.ndarray:
    """
    计算无 IR 窗的 xi（仅由 kmin/kmax 与边界taper控制）。
    """
    taper = m5.build_log_taper_window(k_grid, kmin=kmin, kmax=KMAX_INT, frac=EDGE_TAPER_FRAC)
    p0_eff = p0_base * taper
    r_grid, xi_grid = m5.xi_fftlog_from_effective_p0(k_grid, p0_eff)
    return m5.interp_xi_to_s(s_data, r_grid, xi_grid)


def evaluate_testb_fixed_xi(
    s_data: np.ndarray,
    k_grid: np.ndarray,
    p0_fnl100: np.ndarray,
    p0_fnl0_ref: np.ndarray,
    k_fund: float,
) -> np.ndarray:
    """
    固定窗口参数的 TestB（直接加窗口）。

    注意
    ----
    USE_PNG_ONLY_WINDOW=False 时，仅使用 p0_fnl100 与窗口；
    这里仍保留 p0_fnl0_ref 参数以便后续切换模式时复用。
    """
    ir_window = m5.build_ir_window(k_grid, kf=k_fund, wtype=WINDOW_TYPE, param=WINDOW_PARAM)

    if USE_PNG_ONLY_WINDOW:
        delta_png = p0_fnl100 - p0_fnl0_ref
        p0_ir = p0_fnl0_ref + ir_window * delta_png
    else:
        p0_ir = p0_fnl100 * ir_window

    taper = m5.build_log_taper_window(k_grid, kmin=KMIN_GLOBAL, kmax=KMAX_INT, frac=EDGE_TAPER_FRAC)
    p0_eff = p0_ir * taper

    r_grid, xi_grid = m5.xi_fftlog_from_effective_p0(k_grid, p0_eff)
    return m5.interp_xi_to_s(s_data, r_grid, xi_grid)


# =====================
# 主流程
# =====================

def main() -> None:
    """执行 1Gpc 迁移验证。"""
    old_p = float(m5.FIXED_P)
    m5.FIXED_P = P_FIXED

    try:
        # 1) 数据读取
        data = m5.load_mock_data("1gpc_fnl100", PK_1GPC_GLOB, PCF_1GPC_GLOB, rid_min=1, rid_max=99)

        # 2) PK best-fit
        fit = m5.fit_best_pk(data)
        bestfit = fit["bestfit"]

        # 3) 构建理论 P0
        k_grid = np.geomspace(KMIN_GLOBAL / m5.FFTLOG_PADDING, KMAX_INT * m5.FFTLOG_PADDING, m5.FFTLOG_N)
        p0_100 = m5.build_theory_p0(k_grid, bestfit)

        bestfit_fnl0 = dict(bestfit)
        bestfit_fnl0["fnl_loc"] = 0.0
        p0_fnl0_ref = m5.build_theory_p0(k_grid, bestfit_fnl0)

        s = data.scen
        xi_data = data.xi_mean
        xi_std = data.xi_std

        # 4) 三种曲线
        xi_baseline = evaluate_unwindowed_xi(s, k_grid, p0_100, kmin=K_FUND_1GPC)
        xi_testa = evaluate_unwindowed_xi(s, k_grid, p0_100, kmin=KMIN_GLOBAL)
        xi_testb = evaluate_testb_fixed_xi(s, k_grid, p0_100, p0_fnl0_ref, k_fund=K_FUND_1GPC)

        # 5) 指标
        m_base = m5.compute_alignment_metrics(s, xi_data, xi_std, xi_baseline, "Baseline_1gpc")
        m_a = m5.compute_alignment_metrics(s, xi_data, xi_std, xi_testa, "TestA_1gpc")
        m_b = m5.compute_alignment_metrics(s, xi_data, xi_std, xi_testb, "TestBfixed_1gpc")

        # 6) 出图（上：r2xi；下：残差）
        fig, axes = plt.subplots(2, 1, figsize=(9.4, 8.5), sharex=True)

        ax = axes[0]
        ax.errorbar(
            s,
            s**2 * xi_data,
            yerr=s**2 * xi_std,
            fmt="o",
            ms=3.6,
            capsize=2,
            color="black",
            label=f"Measured mean (N={data.nmock})",
        )
        ax.plot(s, s**2 * xi_baseline, "-", lw=1.8, color="tab:blue", label="Baseline (kmin=2pi/L_1gpc)")
        ax.plot(s, s**2 * xi_testa, "-", lw=1.8, color="tab:orange", label="TestA (kmin=1e-4, no IR window)")
        ax.plot(
            s,
            s**2 * xi_testb,
            "-",
            lw=2.1,
            color="tab:red",
            label=f"TestB fixed ({WINDOW_TYPE}, {WINDOW_PARAM:g})",
        )

        ax.axvline(m5.LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_ylabel(r"$r^2\xi_0(r)$")
        ax.set_title("Task5 v5: 1Gpc transfer test of fixed window parameter")
        ax.legend(fontsize=8.5, ncol=2)

        txt = (
            f"PK best-fit (p={P_FIXED}): fnl={bestfit['fnl_loc']:.2f}, b1={bestfit['b1']:.3f}, sigmas={bestfit['sigmas']:.4f}\n"
            f"Baseline mean|Δ/σ|={m_base['mean_abs_sigma']:.3f}, "
            f"TestB fixed mean|Δ/σ|={m_b['mean_abs_sigma']:.3f}"
        )
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.4,
            bbox=dict(facecolor="white", alpha=0.86, edgecolor="gray"),
        )

        # 残差子图
        ax = axes[1]
        r2_sigma = np.maximum((s**2) * xi_std, 1e-12)
        res_base = (s**2 * (xi_data - xi_baseline)) / r2_sigma
        res_a = (s**2 * (xi_data - xi_testa)) / r2_sigma
        res_b = (s**2 * (xi_data - xi_testb)) / r2_sigma

        ax.axhline(0.0, color="black", lw=1.0)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax.axhline(-1.0, color="gray", lw=0.8, ls="--")
        ax.plot(s, res_base, "o-", ms=3.2, lw=1.3, color="tab:blue", label="Baseline")
        ax.plot(s, res_a, "s-", ms=3.2, lw=1.3, color="tab:orange", label="TestA")
        ax.plot(s, res_b, "^-", ms=3.2, lw=1.3, color="tab:red", label="TestB fixed")

        ax.axvline(m5.LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$(Data-Model)/\sigma$ of $r^2\xi_0$")
        ax.legend(fontsize=8.5)

        fig.tight_layout()
        fig.savefig(OUT_FIG, bbox_inches="tight")
        plt.close(fig)

        # 7) 写总结
        with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
            f.write("任务5 v5：1Gpc 盒子固定窗口参数迁移验证\n")
            f.write("=====================================\n\n")
            f.write("[问题]\n")
            f.write("把 3Gpc 上得到的窗口参数（tanh_log, 8）保持不变，迁移到 1Gpc 后，2PCF 能否良好建模？\n\n")

            f.write("[设置]\n")
            f.write(f"- 数据: 1Gpc fnl100, Nmock={data.nmock}, rid={data.realizations.min()}..{data.realizations.max()}\n")
            f.write(f"- 固定 p={P_FIXED}, PK_FIT_KMAX={m5.PK_FIT_KMAX}\n")
            f.write(f"- 盒长: L=1000, k_f=2pi/L={K_FUND_1GPC:.6e}\n")
            f.write(f"- TestB 固定窗口: {WINDOW_TYPE}(param={WINDOW_PARAM:g}), 直接乘总P0\n")
            f.write(f"- 指标: large-scale mean|Δ/σ|, r>={m5.LARGE_SCALE_MIN}\n\n")

            f.write("[PK best-fit]\n")
            f.write(
                f"- fnl_loc={bestfit['fnl_loc']:.4f}, b1={bestfit['b1']:.4f}, sigmas={bestfit['sigmas']:.6f}\n\n"
            )

            f.write("[2PCF 指标]\n")
            f.write(
                f"- Baseline: mean|Δ/σ|={m_base['mean_abs_sigma']:.4f}, "
                f"chi2/ndof={m_base['chi2_ndof']:.4f}, mean(Δ/σ)={m_base['mean_sigma']:.4f}\n"
            )
            f.write(
                f"- TestA: mean|Δ/σ|={m_a['mean_abs_sigma']:.4f}, "
                f"chi2/ndof={m_a['chi2_ndof']:.4f}, mean(Δ/σ)={m_a['mean_sigma']:.4f}\n"
            )
            f.write(
                f"- TestB fixed: mean|Δ/σ|={m_b['mean_abs_sigma']:.4f}, "
                f"chi2/ndof={m_b['chi2_ndof']:.4f}, mean(Δ/σ)={m_b['mean_sigma']:.4f}\n"
            )
            f.write(
                f"- 相对 Baseline 改进（mean|Δ/σ|）: {m_base['mean_abs_sigma'] - m_b['mean_abs_sigma']:.4f}\n\n"
            )

            # 简单判据说明
            f.write("[判据解释]\n")
            f.write("- 常用经验：mean|Δ/σ| < 1 通常可认为在大尺度上建模可接受；越小越好。\n")
            f.write("- 若 TestB fixed 显著小于 Baseline，说明该固定窗口参数具有跨盒长迁移价值。\n\n")

            f.write("[结论]\n")
            if m_b['mean_abs_sigma'] < 1.0:
                f.write("1) 在 1Gpc 上，固定窗口参数下的 TestB 仍然达到可接受建模精度（mean|Δ/σ|<1）。\n")
            else:
                f.write("1) 在 1Gpc 上，固定窗口参数下的 TestB 尚未达到可接受精度（mean|Δ/σ|>=1）。\n")

            if m_b['mean_abs_sigma'] < m_base['mean_abs_sigma']:
                f.write("2) TestB fixed 相比 Baseline 有改进，说明迁移后仍有效。\n")
            else:
                f.write("2) TestB fixed 未优于 Baseline，说明该参数跨盒长迁移有限。\n")

            f.write("3) 详细曲线形态见图，可进一步检查在哪些 r-bin 仍有系统偏差。\n")

        print("[INFO] done")
        print(f"[INFO] fig: {OUT_FIG}")
        print(f"[INFO] summary: {OUT_SUMMARY}")
        print(f"[INFO] metrics baseline={m_base['mean_abs_sigma']:.4f}, testA={m_a['mean_abs_sigma']:.4f}, testBfixed={m_b['mean_abs_sigma']:.4f}")

    finally:
        m5.FIXED_P = old_p


if __name__ == "__main__":
    main()
