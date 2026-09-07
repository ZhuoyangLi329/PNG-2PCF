#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本名称
--------
task5_compare_fastpm_measured_vs_baseline_4panel.py

代码大纲（执行逻辑关系）
------------------------
第 0 部分：参数区
- 定义 1Gpc / 3Gpc、fnl0 / fnl100 的输入路径与输出图路径。

第 1 部分：复用 mission5 的已有函数
- 直接复用：
  - 数据读取：m5.load_mock_data
  - P(k) 拟合：v13.fit_best_pk_safe / v14.fit_best_pk_safe
  - baseline 的 pk->2PCF：v13.evaluate_unwindowed_1gpc / v14.evaluate_unwindowed_3gpc

第 2 部分：主流程
- 读取四组 FastPM 测量数据。
- 对每组样本做 P0 best-fit。
- 用 best-fit P0 通过 baseline 方案得到理论 2PCF。
- 再用当前参数化窗口（TestB）得到对应的 2PCF 理论曲线。

第 3 部分：作图
- 画成 2x2 面板图，对应：
  1) 1Gpc fnl0
  2) 1Gpc fnl100
  3) 3Gpc fnl0
  4) 3Gpc fnl100
- 第一张图：每个面板画两条曲线：
  - 测量平均值：实线 + 误差带
  - baseline 理论：虚线
- 第二张图：每个面板画三条曲线：
  - 测量平均值：实线 + 误差带
  - baseline 理论：虚线
  - 参数化窗口理论：虚线（不同颜色）
- legend 只在标题/legend title 中标明 fnl 与盒长 L，不再塞入多余指标。

说明
----
1. 与前一张图保持一致，纵轴仍使用 r^2 * xi0(r)。
2. 误差带只加在测量曲线上，口径为 sqrt(diag(Cov))，即 realization 样本标准差。
3. 拟合与 baseline 口径沿用 mission5 当前默认：
   - PK_FIT_KMAX = 0.08
   - p = 1.1
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =====================
# 0) 参数区
# =====================

MISSION5_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission5_log"
sys.path.insert(0, MISSION5_DIR)

import task5_ir_window_solution_3gpc_multitype as m5
import task5_v13_1gpc_joint_fnl0_fnl100_testb_scan as v13
import task5_v14_3gpc_joint_fnl0_fnl100_testb_scan as v14

OUT_FIG = os.path.join(
    MISSION5_DIR,
    "task5_compare_fastpm_measured_vs_baseline_4panel.png",
)
OUT_FIG_WITH_WINDOW = os.path.join(
    MISSION5_DIR,
    "task5_compare_fastpm_measured_baseline_window_4panel.png",
)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"


# =====================
# 1) 单盒长单 fnl 的 baseline 计算
# =====================


def build_baseline_xi_for_dataset(
    dataset: m5.MockData,
    bestfit: dict[str, float],
    is_1gpc: bool,
) -> np.ndarray:
    """
    对单个数据集，基于 best-fit P0 构造 baseline 的 2PCF 理论曲线。

    参数
    ----
    dataset : m5.MockData
        当前数据集，包含测量的 s 网格。
    bestfit : dict[str, float]
        P(k) best-fit 参数。
    is_1gpc : bool
        是否为 1Gpc 数据；决定调用哪个 baseline 积分函数。

    返回
    ----
    np.ndarray
        已经插值到 dataset.scen 上的 baseline xi0(r)。
    """
    k_grid = np.geomspace(
        m5.KMIN_GLOBAL / m5.FFTLOG_PADDING,
        m5.KMAX_INT * m5.FFTLOG_PADDING,
        m5.FFTLOG_N,
    )
    p0_model = m5.build_theory_p0(k_grid, bestfit)

    if is_1gpc:
        return v13.evaluate_unwindowed_1gpc(
            dataset.scen,
            k_grid,
            p0_model,
            kmin=v13.K_FUND_1GPC,
        )

    return v14.evaluate_unwindowed_3gpc(
        dataset.scen,
        k_grid,
        p0_model,
        kmin=v14.K_FUND_3GPC,
    )


def build_window_xi_for_dataset(
    dataset: m5.MockData,
    bestfit: dict[str, float],
    p0_model: np.ndarray,
    p0_ref0: np.ndarray,
    is_1gpc: bool,
) -> np.ndarray:
    """
    对单个数据集，基于 mission5 当前参数化窗口构造 TestB 的 2PCF 理论曲线。

    参数
    ----
    dataset : m5.MockData
        当前数据集。
    bestfit : dict[str, float]
        best-fit 参数字典，仅用于判定和记录当前样本。
    p0_model : np.ndarray
        当前 best-fit 对应的理论 P0(k)。
    p0_ref0 : np.ndarray
        对 fnl100 来说，是把 fnl_loc 置零后的参考 P0；
        对 fnl0 来说，直接等于 p0_model 本身。
    is_1gpc : bool
        是否为 1Gpc 盒子。

    返回
    ----
    np.ndarray
        已插值到 dataset.scen 上的窗口模型 xi0(r)。
    """
    _ = bestfit  # 保留接口，便于后续若需扩展到参数记录。
    k_grid = np.geomspace(
        m5.KMIN_GLOBAL / m5.FFTLOG_PADDING,
        m5.KMAX_INT * m5.FFTLOG_PADDING,
        m5.FFTLOG_N,
    )

    if is_1gpc:
        return v13.evaluate_testb_1gpc_single_window(
            s_data=dataset.scen,
            k_grid=k_grid,
            p0_in=p0_model,
            p0_ref_fnl0_same_other=p0_ref0,
            window_type="exp_power",
            window_param=4.0,
        )

    return v14.evaluate_testb_3gpc_single_window(
        s_data=dataset.scen,
        k_grid=k_grid,
        p0_in=p0_model,
        p0_ref_fnl0_same_other=p0_ref0,
        window_type="exp_power",
        window_param=12.0,
    )


def plot_panel_figure(
    out_png: str,
    panels: list[dict[str, object]],
    include_window: bool,
) -> None:
    """
    统一画 2x2 面板图。

    参数
    ----
    out_png : str
        输出图片路径。
    panels : list[dict[str, object]]
        每个面板的绘图数据。
    include_window : bool
        是否在图中加入参数化窗口理论曲线。
    """
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.4), sharex=True, sharey=False)
    axes = axes.ravel()

    for ax, panel in zip(axes, panels):
        data = panel["data"]
        s = data.scen
        y_data = s**2 * data.xi_mean
        e_data = s**2 * data.xi_std
        y_base = s**2 * panel["xi_baseline"]

        ax.fill_between(
            s,
            y_data - e_data,
            y_data + e_data,
            color="0.5",
            alpha=0.20,
            linewidth=0.0,
        )
        ax.plot(
            s,
            y_data,
            "-",
            lw=2.1,
            color="black",
            label="Measured mean",
        )
        ax.plot(
            s,
            y_base,
            "--",
            lw=2.0,
            color="tab:red",
            label="Baseline model",
        )

        if include_window:
            y_win = s**2 * panel["xi_window"]
            ax.plot(
                s,
                y_win,
                "--",
                lw=2.0,
                color="tab:blue",
                label="Window model",
            )

        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2 \xi_0(r)$")
        ax.legend(
            frameon=False,
            fontsize=8.6,
            title=panel["legend_title"],
            title_fontsize=9.4,
        )

    title = "FastPM measured mean vs baseline 2PCF model"
    if include_window:
        title = "FastPM measured mean vs baseline/window 2PCF models"
    fig.suptitle(title, y=0.995)

    if include_window:
        expr = (
            r"$W(k)=\frac{1-\exp[-(k/k_f)^x]}{1-\exp(-1)}\ (k<k_f),\ W(k)=1\ (k\geq k_f)$"
            "\n"
            r"$x(L)=4\times(L/1000)$"
        )
        fig.text(
            0.985,
            0.965,
            expr,
            ha="right",
            va="top",
            fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white", alpha=0.86, edgecolor="0.8"),
        )

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# =====================
# 2) 主流程
# =====================


def main() -> None:
    """
    读取四组 FastPM 数据，构造 baseline 模型，并输出 2x2 面板图。
    """
    old_p = float(m5.FIXED_P)
    old_pkfit = float(m5.PK_FIT_KMAX)

    # 维持 mission5 当前统一口径
    m5.FIXED_P = float(v14.P_FIXED)          # p = 1.1
    m5.PK_FIT_KMAX = float(v14.PK_FIT_KMAX)  # 0.08

    try:
        # 1Gpc 数据
        d1_0 = m5.load_mock_data(
            "1gpc_fnl0",
            v13.PK_1GPC_FNL0_GLOB,
            v13.PCF_1GPC_FNL0_GLOB,
            v13.RID_MIN,
            v13.RID_MAX,
        )
        d1_100 = m5.load_mock_data(
            "1gpc_fnl100",
            v13.PK_1GPC_FNL100_GLOB,
            v13.PCF_1GPC_FNL100_GLOB,
            v13.RID_MIN,
            v13.RID_MAX,
        )

        # 3Gpc 数据
        d3_0 = m5.load_mock_data(
            "3gpc_fnl0",
            v14.PK_3GPC_FNL0_GLOB,
            v14.PCF_3GPC_FNL0_GLOB,
            v14.RID_MIN,
            v14.RID_MAX,
        )
        d3_100 = m5.load_mock_data(
            "3gpc_fnl100",
            v14.PK_3GPC_FNL100_GLOB,
            v14.PCF_3GPC_FNL100_GLOB,
            v14.RID_MIN,
            v14.RID_MAX,
        )

        # P(k) best-fit
        fit1_0 = v13.fit_best_pk_safe(d1_0)
        fit1_100 = v13.fit_best_pk_safe(d1_100)
        fit3_0 = v14.fit_best_pk_safe(d3_0)
        fit3_100 = v14.fit_best_pk_safe(d3_100)

        # 构造统一 k 网格与理论 P0，便于同时生成 baseline / window
        k_grid = np.geomspace(
            m5.KMIN_GLOBAL / m5.FFTLOG_PADDING,
            m5.KMAX_INT * m5.FFTLOG_PADDING,
            m5.FFTLOG_N,
        )

        p1_0 = m5.build_theory_p0(k_grid, fit1_0["bestfit"])
        p1_100 = m5.build_theory_p0(k_grid, fit1_100["bestfit"])
        p3_0 = m5.build_theory_p0(k_grid, fit3_0["bestfit"])
        p3_100 = m5.build_theory_p0(k_grid, fit3_100["bestfit"])

        fit1_100_ref0 = dict(fit1_100["bestfit"])
        fit1_100_ref0["fnl_loc"] = 0.0
        p1_100_ref0 = m5.build_theory_p0(k_grid, fit1_100_ref0)

        fit3_100_ref0 = dict(fit3_100["bestfit"])
        fit3_100_ref0["fnl_loc"] = 0.0
        p3_100_ref0 = m5.build_theory_p0(k_grid, fit3_100_ref0)

        # baseline xi
        xi1_0_base = build_baseline_xi_for_dataset(d1_0, fit1_0["bestfit"], is_1gpc=True)
        xi1_100_base = build_baseline_xi_for_dataset(d1_100, fit1_100["bestfit"], is_1gpc=True)
        xi3_0_base = build_baseline_xi_for_dataset(d3_0, fit3_0["bestfit"], is_1gpc=False)
        xi3_100_base = build_baseline_xi_for_dataset(d3_100, fit3_100["bestfit"], is_1gpc=False)

        # window xi
        xi1_0_win = build_window_xi_for_dataset(d1_0, fit1_0["bestfit"], p1_0, p1_0, is_1gpc=True)
        xi1_100_win = build_window_xi_for_dataset(d1_100, fit1_100["bestfit"], p1_100, p1_100_ref0, is_1gpc=True)
        xi3_0_win = build_window_xi_for_dataset(d3_0, fit3_0["bestfit"], p3_0, p3_0, is_1gpc=False)
        xi3_100_win = build_window_xi_for_dataset(d3_100, fit3_100["bestfit"], p3_100, p3_100_ref0, is_1gpc=False)

        panels = [
            {
                "data": d1_0,
                "xi_baseline": xi1_0_base,
                "xi_window": xi1_0_win,
                "legend_title": "L=1 Gpc/h, fnl=0",
            },
            {
                "data": d1_100,
                "xi_baseline": xi1_100_base,
                "xi_window": xi1_100_win,
                "legend_title": "L=1 Gpc/h, fnl=100",
            },
            {
                "data": d3_0,
                "xi_baseline": xi3_0_base,
                "xi_window": xi3_0_win,
                "legend_title": "L=3 Gpc/h, fnl=0",
            },
            {
                "data": d3_100,
                "xi_baseline": xi3_100_base,
                "xi_window": xi3_100_win,
                "legend_title": "L=3 Gpc/h, fnl=100",
            },
        ]

        plot_panel_figure(OUT_FIG, panels, include_window=False)
        plot_panel_figure(OUT_FIG_WITH_WINDOW, panels, include_window=True)

        print(f"[OK] saved figure: {OUT_FIG}")
        print(f"[OK] saved figure: {OUT_FIG_WITH_WINDOW}")

    finally:
        m5.FIXED_P = old_p
        m5.PK_FIT_KMAX = old_pkfit


if __name__ == "__main__":
    main()
