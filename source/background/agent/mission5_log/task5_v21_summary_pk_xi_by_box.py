#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务5 v21：按盒长分图的总结图（1Gpc / 3Gpc）

目标
----
每个盒长输出一张图（共两张）：
1) 子图1：不同 fnl（0/100）的 P(k) 测量均值 vs best-fit 模型。
2) 子图2：不同 fnl（0/100）的 2PCF 测量均值 vs 模型。
   - 模型包含：Baseline + TESTB(exp_power, x(L))
   - x(L) 采用保守版：x(L)=4*(L/1000)

统一口径
--------
- TESTB 采用 direct-P window（与 mission5 当前窗口测试一致）
- kmin_global=1e-4, kmax=20, taper 保留
- PK 拟合：k<=0.08, p=1.1
- 额外输出 r>=150 的量化指标到 summary 文本
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MISSION5_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission5_log"
sys.path.insert(0, MISSION5_DIR)

import task5_ir_window_solution_3gpc_multitype as m5
import task5_v13_1gpc_joint_fnl0_fnl100_testb_scan as v13
import task5_v14_3gpc_joint_fnl0_fnl100_testb_scan as v14

# 保守版 L 依赖窗口参数
X_OF_L = lambda L: 4.0 * (L / 1000.0)

# 评估区间（你目前更关注）
RMIN_EVAL = 150.0

OUT_FIG_1GPC = os.path.join(MISSION5_DIR, "task5_v21_summary_1gpc_pk_xi.png")
OUT_FIG_3GPC = os.path.join(MISSION5_DIR, "task5_v21_summary_3gpc_pk_xi.png")
OUT_TXT = os.path.join(MISSION5_DIR, "task5_v21_summary_pk_xi_by_box.txt")

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"


def metric_rge150(s: np.ndarray, xi_data: np.ndarray, xi_std: np.ndarray, xi_model: np.ndarray) -> Dict[str, float]:
    """在 r>=150 上计算 mean|Δ/σ| 等指标（用于 summary）。"""
    mask = s >= RMIN_EVAL
    r2_data = s**2 * xi_data
    r2_model = s**2 * xi_model
    r2_std = np.maximum(s**2 * xi_std, 1e-12)
    resid = (r2_data[mask] - r2_model[mask]) / r2_std[mask]
    return {
        "mean_abs_sigma": float(np.mean(np.abs(resid))),
        "chi2_ndof": float(np.mean(resid**2)),
        "mean_sigma": float(np.mean(resid)),
    }


def make_one_box_figure(
    box_tag: str,
    Lbox: float,
    data100: m5.MockData,
    data0: m5.MockData,
    fit100: Dict[str, object],
    fit0: Dict[str, object],
    k_grid: np.ndarray,
    p0_100: np.ndarray,
    p0_100_ref0: np.ndarray,
    p0_0: np.ndarray,
    is_1gpc: bool,
    out_png: str,
) -> Dict[str, Dict[str, float]]:
    """为单个盒长画图并返回关键指标。"""
    xL = float(X_OF_L(Lbox))

    # Baseline
    if is_1gpc:
        xi100_base = v13.evaluate_unwindowed_1gpc(data100.scen, k_grid, p0_100, kmin=v13.K_FUND_1GPC)
        xi0_base = v13.evaluate_unwindowed_1gpc(data0.scen, k_grid, p0_0, kmin=v13.K_FUND_1GPC)

        # TESTB exp_power(xL)
        xi100_win = v13.evaluate_testb_1gpc_single_window(
            s_data=data100.scen,
            k_grid=k_grid,
            p0_in=p0_100,
            p0_ref_fnl0_same_other=p0_100_ref0,
            window_type="exp_power",
            window_param=xL,
        )
        xi0_win = v13.evaluate_testb_1gpc_single_window(
            s_data=data0.scen,
            k_grid=k_grid,
            p0_in=p0_0,
            p0_ref_fnl0_same_other=p0_0,
            window_type="exp_power",
            window_param=xL,
        )
    else:
        xi100_base = v14.evaluate_unwindowed_3gpc(data100.scen, k_grid, p0_100, kmin=v14.K_FUND_3GPC)
        xi0_base = v14.evaluate_unwindowed_3gpc(data0.scen, k_grid, p0_0, kmin=v14.K_FUND_3GPC)

        # TESTB exp_power(xL)
        xi100_win = v14.evaluate_testb_3gpc_single_window(
            s_data=data100.scen,
            k_grid=k_grid,
            p0_in=p0_100,
            p0_ref_fnl0_same_other=p0_100_ref0,
            window_type="exp_power",
            window_param=xL,
        )
        xi0_win = v14.evaluate_testb_3gpc_single_window(
            s_data=data0.scen,
            k_grid=k_grid,
            p0_in=p0_0,
            p0_ref_fnl0_same_other=p0_0,
            window_type="exp_power",
            window_param=xL,
        )

    # 量化指标（r>=150）
    m100_base = metric_rge150(data100.scen, data100.xi_mean, data100.xi_std, xi100_base)
    m100_win = metric_rge150(data100.scen, data100.xi_mean, data100.xi_std, xi100_win)
    m0_base = metric_rge150(data0.scen, data0.xi_mean, data0.xi_std, xi0_base)
    m0_win = metric_rge150(data0.scen, data0.xi_mean, data0.xi_std, xi0_win)

    # 画图：两子图
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))

    # 子图1：P(k) mean vs best-fit
    ax = axes[0]
    kplot = np.geomspace(min(data100.kcen.min(), data0.kcen.min()), 0.2, 700)
    p100_plot = m5.build_theory_p0(kplot, fit100["bestfit"])
    p0_plot = m5.build_theory_p0(kplot, fit0["bestfit"])

    ax.errorbar(data100.kcen, data100.p0_mean, yerr=data100.p0_std, fmt="o", ms=2.8, capsize=1.8,
                color="tab:red", alpha=0.85, label="Measured mean fnl100")
    ax.plot(kplot, p100_plot, "-", lw=2.0, color="tab:red", label="Best-fit fnl100")

    ax.errorbar(data0.kcen, data0.p0_mean, yerr=data0.p0_std, fmt="s", ms=2.6, capsize=1.8,
                color="tab:blue", alpha=0.85, label="Measured mean fnl0")
    ax.plot(kplot, p0_plot, "-", lw=2.0, color="tab:blue", label="Best-fit fnl0")

    ax.axvline(0.08, color="gray", ls=":", lw=1.0, alpha=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k\,[h/\mathrm{Mpc}]$")
    ax.set_ylabel(r"$P_0(k)$")
    ax.set_title(f"{box_tag}: P(k) mean vs best-fit")
    ax.legend(fontsize=8)

    # 子图2：2PCF 对比（同一窗口形状 exp_power，x由L决定）
    ax = axes[1]

    # 测量均值：按你要求用虚线连接
    ax.errorbar(data100.scen, data100.scen**2 * data100.xi_mean, yerr=data100.scen**2 * data100.xi_std,
                fmt="o--", ms=2.8, capsize=1.8, color="tab:red", alpha=0.85, label="Measured mean fnl100")
    ax.errorbar(data0.scen, data0.scen**2 * data0.xi_mean, yerr=data0.scen**2 * data0.xi_std,
                fmt="s--", ms=2.6, capsize=1.8, color="tab:blue", alpha=0.85, label="Measured mean fnl0")

    # baseline + window
    ax.plot(data100.scen, data100.scen**2 * xi100_base, ":", lw=1.8, color="tab:red",
            label=f"Baseline fnl100 ({m100_base['mean_abs_sigma']:.3f})")
    ax.plot(data100.scen, data100.scen**2 * xi100_win, "-", lw=2.2, color="tab:red",
            label=f"Window fnl100 ({m100_win['mean_abs_sigma']:.3f})")

    ax.plot(data0.scen, data0.scen**2 * xi0_base, ":", lw=1.8, color="tab:blue",
            label=f"Baseline fnl0 ({m0_base['mean_abs_sigma']:.3f})")
    ax.plot(data0.scen, data0.scen**2 * xi0_win, "-", lw=2.2, color="tab:blue",
            label=f"Window fnl0 ({m0_win['mean_abs_sigma']:.3f})")

    ax.axvline(RMIN_EVAL, color="gray", ls="--", lw=1.0, alpha=0.8)
    ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$r^2\xi_0(r)$")
    ax.set_title(f"{box_tag}: 2PCF (exp_power x={xL:g}, r>=150 focus)")
    ax.legend(fontsize=7.8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    return {
        "xL": {"value": xL},
        "fnl100_baseline": m100_base,
        "fnl100_window": m100_win,
        "fnl0_baseline": m0_base,
        "fnl0_window": m0_win,
    }


def main() -> None:
    # 暂存全局参数，避免影响其他任务脚本
    old_p = float(m5.FIXED_P)
    old_pkfit = float(m5.PK_FIT_KMAX)

    # mission5 当前统一口径
    m5.FIXED_P = float(v14.P_FIXED)          # p=1.1
    m5.PK_FIT_KMAX = float(v14.PK_FIT_KMAX)  # 0.08

    try:
        # 读取数据
        d1_100 = m5.load_mock_data("1gpc_fnl100", v13.PK_1GPC_FNL100_GLOB, v13.PCF_1GPC_FNL100_GLOB, v13.RID_MIN, v13.RID_MAX)
        d1_0 = m5.load_mock_data("1gpc_fnl0", v13.PK_1GPC_FNL0_GLOB, v13.PCF_1GPC_FNL0_GLOB, v13.RID_MIN, v13.RID_MAX)
        d3_100 = m5.load_mock_data("3gpc_fnl100", v14.PK_3GPC_FNL100_GLOB, v14.PCF_3GPC_FNL100_GLOB, v14.RID_MIN, v14.RID_MAX)
        d3_0 = m5.load_mock_data("3gpc_fnl0", v14.PK_3GPC_FNL0_GLOB, v14.PCF_3GPC_FNL0_GLOB, v14.RID_MIN, v14.RID_MAX)

        # best-fit
        f1_100 = v13.fit_best_pk_safe(d1_100)
        f1_0 = v13.fit_best_pk_safe(d1_0)
        f3_100 = v14.fit_best_pk_safe(d3_100)
        f3_0 = v14.fit_best_pk_safe(d3_0)

        # FFTLog k 网格
        k_grid = np.geomspace(v14.KMIN_GLOBAL / m5.FFTLOG_PADDING, v14.KMAX_INT * m5.FFTLOG_PADDING, m5.FFTLOG_N)

        # P0 缓存
        p1_100 = m5.build_theory_p0(k_grid, f1_100["bestfit"])
        p1_0 = m5.build_theory_p0(k_grid, f1_0["bestfit"])
        p3_100 = m5.build_theory_p0(k_grid, f3_100["bestfit"])
        p3_0 = m5.build_theory_p0(k_grid, f3_0["bestfit"])

        b1_ref = dict(f1_100["bestfit"])
        b1_ref["fnl_loc"] = 0.0
        p1_100_ref0 = m5.build_theory_p0(k_grid, b1_ref)

        b3_ref = dict(f3_100["bestfit"])
        b3_ref["fnl_loc"] = 0.0
        p3_100_ref0 = m5.build_theory_p0(k_grid, b3_ref)

        # 作图与指标
        m1 = make_one_box_figure(
            box_tag="1Gpc",
            Lbox=1000.0,
            data100=d1_100,
            data0=d1_0,
            fit100=f1_100,
            fit0=f1_0,
            k_grid=k_grid,
            p0_100=p1_100,
            p0_100_ref0=p1_100_ref0,
            p0_0=p1_0,
            is_1gpc=True,
            out_png=OUT_FIG_1GPC,
        )

        m3 = make_one_box_figure(
            box_tag="3Gpc",
            Lbox=3000.0,
            data100=d3_100,
            data0=d3_0,
            fit100=f3_100,
            fit0=f3_0,
            k_grid=k_grid,
            p0_100=p3_100,
            p0_100_ref0=p3_100_ref0,
            p0_0=p3_0,
            is_1gpc=False,
            out_png=OUT_FIG_3GPC,
        )

        # 汇总文本
        with open(OUT_TXT, "w", encoding="utf-8") as f:
            f.write("任务5 v21：按盒长分图的 PK/2PCF 总结图\n")
            f.write("===================================\n\n")
            f.write("统一设置：\n")
            f.write("- PK拟合：k<=0.08, p=1.1\n")
            f.write("- 2PCF积分：kmin_global=1e-4, kmax=20, taper保留\n")
            f.write("- Window（保守版）：exp_power, x(L)=4*(L/1000)\n")
            f.write("- 指标区间：r>=150\n\n")

            f.write("[1Gpc]\n")
            f.write(f"- x(L)={m1['xL']['value']:.2f}\n")
            f.write(
                f"- fnl100 baseline={m1['fnl100_baseline']['mean_abs_sigma']:.4f}, "
                f"window={m1['fnl100_window']['mean_abs_sigma']:.4f}\n"
            )
            f.write(
                f"- fnl0   baseline={m1['fnl0_baseline']['mean_abs_sigma']:.4f}, "
                f"window={m1['fnl0_window']['mean_abs_sigma']:.4f}\n\n"
            )

            f.write("[3Gpc]\n")
            f.write(f"- x(L)={m3['xL']['value']:.2f}\n")
            f.write(
                f"- fnl100 baseline={m3['fnl100_baseline']['mean_abs_sigma']:.4f}, "
                f"window={m3['fnl100_window']['mean_abs_sigma']:.4f}\n"
            )
            f.write(
                f"- fnl0   baseline={m3['fnl0_baseline']['mean_abs_sigma']:.4f}, "
                f"window={m3['fnl0_window']['mean_abs_sigma']:.4f}\n"
            )

        print("[INFO] done")
        print(f"[INFO] fig1: {OUT_FIG_1GPC}")
        print(f"[INFO] fig2: {OUT_FIG_3GPC}")
        print(f"[INFO] summary: {OUT_TXT}")

    finally:
        m5.FIXED_P = old_p
        m5.PK_FIT_KMAX = old_pkfit


if __name__ == "__main__":
    main()
