#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本名称
--------
task5_v14_3gpc_joint_fnl0_fnl100_testb_scan.py

代码大纲（执行逻辑）
--------------------
第 0 部分：参数区
- 定义 3Gpc fnl0/fnl100 数据路径、拟合设置、积分设置、窗口扫描列表与输出路径。

第 1 部分：基础函数
- 提供“安全版 PK 拟合”（自动剔除零方差 k-bin，避免协方差奇异）。
- 提供 3Gpc 版 TestB 评估函数（窗口作用于总 P0，k_f 使用 3Gpc 盒长）。

第 2 部分：联合扫描
- 对每个窗口候选 (wtype, param) 同时评估 fnl100 与 fnl0 的 2PCF 建模指标。
- 记录联合目标：
  1) joint_max = max(metric_fnl100, metric_fnl0)
  2) joint_mean = 0.5*(metric_fnl100 + metric_fnl0)
- 以 joint_max 最小为主目标（兼顾两者），joint_mean 作为并列时次级目标。

第 3 部分：输出
- 图1：窗口候选在 (fnl0_metric, fnl100_metric) 平面的分布（联合权衡图）。
- 图2：最优窗口下的曲线对比（fnl100 与 fnl0 两个子图）。
- 文本：summary，给出最优窗口、基线对照与结论。

说明
----
1) 本脚本严格按你最新要求：kmin_global=1e-4，TESTB 方法寻找“同时适配 fnl0/fnl100”的窗口。
2) TestB 使用“B 方法直接对总 P0 加窗”（不是仅对 PNG 增量项加窗）。
3) FFTLog 的 log-k taper 保留（沿用 mission4/mission5 的稳定设置）。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

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

# 3Gpc 数据（fnl100 / fnl0）
PK_3GPC_FNL100_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat"
PCF_3GPC_FNL100_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat"
PK_3GPC_FNL0_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat"
PCF_3GPC_FNL0_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat"

RID_MIN = 2
RID_MAX = 99

# 盒长与基模
BOX_SIZE_3GPC = 3000.0
K_FUND_3GPC = 2.0 * np.pi / BOX_SIZE_3GPC

# 口径：你明确要求 kmin_global 要小（1e-4）
KMIN_GLOBAL = 1e-4
KMAX_INT = float(m5.KMAX_INT)  # 目前为 20

# PK 拟合口径
PK_FIT_KMAX = 0.08
P_FIXED = 1.1

# TestB 口径：直接对总 P0 加窗
USE_PNG_ONLY_WINDOW = False

# 扫描窗口族（覆盖任务5里常见的几类）
WINDOW_SCANS: List[Tuple[str, List[float]]] = [
    ("exp_power", [1, 2, 3, 4, 6, 8, 10, 12, 16]),
    ("rational", [1, 2, 3, 4, 6, 8, 10, 12, 16]),
    ("tanh_log", [1.5, 2.5, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0]),
    ("logcos", [0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0]),
]

# 输出
OUT_FIG1 = os.path.join(MISSION5_DIR, "task5_v14_3gpc_joint_window_tradeoff.png")
OUT_FIG2 = os.path.join(MISSION5_DIR, "task5_v14_3gpc_joint_window_best_curves.png")
OUT_TXT = os.path.join(MISSION5_DIR, "task5_v14_3gpc_joint_window_summary.txt")

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"


# =====================
# 1) 基础函数
# =====================
def fit_best_pk_safe(data: m5.MockData) -> Dict[str, object]:
    """
    对 P0 均值做安全版 best-fit。

    目的
    ----
    某些输入数据在最小 k-bin 可能出现“样本标准差为 0”的情况，
    直接用于协方差会导致奇异矩阵；这里自动剔除该类 bin。

    参数
    ----
    data : m5.MockData
        输入测量数据结构。

    返回
    ----
    Dict[str, object]
        含 bestfit 字典及拟合网格信息。
    """
    # 先按低-k 拟合上限截取，再剔除零方差/非有限 bin
    mask = data.kcen <= PK_FIT_KMAX
    p0_std = data.p0_std
    mask &= np.isfinite(p0_std) & (p0_std > 0.0)

    if np.sum(mask) < 5:
        raise RuntimeError(f"{data.tag}: 可用拟合点过少")

    kfit = data.kcen[mask]
    p0_mean_fit = data.p0_mean[mask]
    p0_mocks_fit = data.p0_mocks[:, mask]

    # 以下拟合流程与 task5/model.ipynb 保持同口径
    cosmo = m5.build_fiducial_cosmology()
    template = m5.FixedPowerSpectrumTemplate(z=m5.UNIT_Z, fiducial=cosmo)
    theory = m5.PNGTracerPowerSpectrumMultipoles(template=template, mode="b-p")

    theory.init.params["p"].update(fixed=True, value=m5.FIXED_P)
    theory.init.params["sn0"].update(fixed=True, value=m5.FIXED_SN0)
    theory.init.params["sigmas"].update(fixed=False, value=m5.FIXED_SIGMAS)

    dk = float(np.median(np.diff(kfit)))
    observable = m5.TracerPowerSpectrumMultipolesObservable(
        data=p0_mean_fit,
        covariance=[row for row in p0_mocks_fit],
        klim={0: [float(kfit.min()), float(kfit.max()), dk]},
        k=kfit,
        ells=[0],
        theory=theory,
    )
    likelihood = m5.ObservablesGaussianLikelihood(observables=[observable])
    _ = likelihood()

    likelihood.all_params["p"].update(fixed=True, value=m5.FIXED_P)
    likelihood.all_params["sn0"].update(fixed=True, value=m5.FIXED_SN0)
    likelihood.all_params["sigmas"].update(fixed=False, value=m5.FIXED_SIGMAS)

    profiler = m5.MinuitProfiler(likelihood, seed=m5.MINUIT_SEED)
    profiles = profiler.maximize(niterations=m5.MINUIT_NITER)
    bestfit = m5.convert_bestfit_to_float_dict(profiles.bestfit.choice(input=True))

    return {
        "bestfit": bestfit,
        "kfit": kfit,
    }


def evaluate_unwindowed_3gpc(
    s_data: np.ndarray,
    k_grid: np.ndarray,
    p0_base: np.ndarray,
    kmin: float,
) -> np.ndarray:
    """
    无 IR 窗版本（用于 Baseline/TestA 对照）。

    参数
    ----
    s_data : np.ndarray
        目标 s 网格。
    k_grid : np.ndarray
        FFTLog k 网格。
    p0_base : np.ndarray
        原始理论 P0(k)。
    kmin : float
        积分下限。

    返回
    ----
    np.ndarray
        插值到 s_data 的 xi 模型。
    """
    taper = m5.build_log_taper_window(
        k_grid,
        kmin=kmin,
        kmax=KMAX_INT,
        frac=m5.EDGE_TAPER_FRAC,
    )
    p0_eff = p0_base * taper
    r_grid, xi_grid = m5.xi_fftlog_from_effective_p0(k_grid, p0_eff)
    return m5.interp_xi_to_s(s_data, r_grid, xi_grid)


def evaluate_testb_3gpc_single_window(
    s_data: np.ndarray,
    k_grid: np.ndarray,
    p0_in: np.ndarray,
    p0_ref_fnl0_same_other: np.ndarray,
    window_type: str,
    window_param: float,
) -> np.ndarray:
    """
    3Gpc 下的 TestB 单窗口评估。

    口径
    ----
    - 使用 3Gpc 的 k_f = 2pi/L_3gpc
    - B 方法直接对总 P0 加窗（USE_PNG_ONLY_WINDOW=False）
    - 保留 FFTLog log-k taper，积分区间 [kmin_global, kmax]

    参数
    ----
    s_data : np.ndarray
        目标 s 网格。
    k_grid : np.ndarray
        FFTLog k 网格。
    p0_in : np.ndarray
        输入理论 P0(k)（可对应 fnl100 或 fnl0）。
    p0_ref_fnl0_same_other : np.ndarray
        若采用 PNG-only 窗时所需参考项；本脚本默认不使用。
    window_type, window_param : str, float
        窗口类型和参数。

    返回
    ----
    np.ndarray
        插值到 s_data 的 xi 模型。
    """
    ir_window = m5.build_ir_window(
        k_grid,
        kf=K_FUND_3GPC,
        wtype=window_type,
        param=float(window_param),
    )

    if USE_PNG_ONLY_WINDOW:
        delta_png = p0_in - p0_ref_fnl0_same_other
        p0_ir = p0_ref_fnl0_same_other + ir_window * delta_png
    else:
        p0_ir = p0_in * ir_window

    taper = m5.build_log_taper_window(
        k_grid,
        kmin=KMIN_GLOBAL,
        kmax=KMAX_INT,
        frac=m5.EDGE_TAPER_FRAC,
    )
    p0_eff = p0_ir * taper
    r_grid, xi_grid = m5.xi_fftlog_from_effective_p0(k_grid, p0_eff)
    return m5.interp_xi_to_s(s_data, r_grid, xi_grid)


# =====================
# 2) 主流程
# =====================
def main() -> None:
    """执行 3Gpc fnl0+fnl100 的联合窗口扫描。"""
    old_p = float(m5.FIXED_P)
    old_pkfit = float(m5.PK_FIT_KMAX)

    # 临时覆盖到本实验口径
    m5.FIXED_P = float(P_FIXED)
    m5.PK_FIT_KMAX = float(PK_FIT_KMAX)

    try:
        # 1) 读取数据
        data100 = m5.load_mock_data(
            "3gpc_fnl100",
            PK_3GPC_FNL100_GLOB,
            PCF_3GPC_FNL100_GLOB,
            rid_min=RID_MIN,
            rid_max=RID_MAX,
        )
        data0 = m5.load_mock_data(
            "3gpc_fnl0",
            PK_3GPC_FNL0_GLOB,
            PCF_3GPC_FNL0_GLOB,
            rid_min=RID_MIN,
            rid_max=RID_MAX,
        )

        # 2) 分别拟合 fnl100/fnl0 的 P0
        fit100 = fit_best_pk_safe(data100)
        fit0 = fit_best_pk_safe(data0)
        bestfit100 = fit100["bestfit"]
        bestfit0 = fit0["bestfit"]

        # 3) 构建统一高分辨率 k 网格与理论 P0
        k_grid = np.geomspace(
            KMIN_GLOBAL / m5.FFTLOG_PADDING,
            KMAX_INT * m5.FFTLOG_PADDING,
            m5.FFTLOG_N,
        )

        p0_100 = m5.build_theory_p0(k_grid, bestfit100)
        bestfit100_fnl0 = dict(bestfit100)
        bestfit100_fnl0["fnl_loc"] = 0.0
        p0_100_fnl0ref = m5.build_theory_p0(k_grid, bestfit100_fnl0)

        p0_0 = m5.build_theory_p0(k_grid, bestfit0)

        # 4) 先算 baseline / TestA 作为对照
        s100 = data100.scen
        s0 = data0.scen

        xi100_baseline = evaluate_unwindowed_3gpc(s100, k_grid, p0_100, kmin=K_FUND_3GPC)
        xi100_testa = evaluate_unwindowed_3gpc(s100, k_grid, p0_100, kmin=KMIN_GLOBAL)
        m100_baseline = m5.compute_alignment_metrics(s100, data100.xi_mean, data100.xi_std, xi100_baseline, "fnl100_Baseline")
        m100_testa = m5.compute_alignment_metrics(s100, data100.xi_mean, data100.xi_std, xi100_testa, "fnl100_TestA")

        xi0_baseline = evaluate_unwindowed_3gpc(s0, k_grid, p0_0, kmin=K_FUND_3GPC)
        xi0_testa = evaluate_unwindowed_3gpc(s0, k_grid, p0_0, kmin=KMIN_GLOBAL)
        m0_baseline = m5.compute_alignment_metrics(s0, data0.xi_mean, data0.xi_std, xi0_baseline, "fnl0_Baseline")
        m0_testa = m5.compute_alignment_metrics(s0, data0.xi_mean, data0.xi_std, xi0_testa, "fnl0_TestA")

        # 5) 联合扫描窗口
        rows: List[Dict[str, float | str]] = []
        xi100_map: Dict[Tuple[str, float], np.ndarray] = {}
        xi0_map: Dict[Tuple[str, float], np.ndarray] = {}

        for wtype, params in WINDOW_SCANS:
            for p in params:
                pval = float(p)

                xi100 = evaluate_testb_3gpc_single_window(
                    s_data=s100,
                    k_grid=k_grid,
                    p0_in=p0_100,
                    p0_ref_fnl0_same_other=p0_100_fnl0ref,
                    window_type=wtype,
                    window_param=pval,
                )
                xi0 = evaluate_testb_3gpc_single_window(
                    s_data=s0,
                    k_grid=k_grid,
                    p0_in=p0_0,
                    p0_ref_fnl0_same_other=p0_0,
                    window_type=wtype,
                    window_param=pval,
                )

                m100 = m5.compute_alignment_metrics(s100, data100.xi_mean, data100.xi_std, xi100, f"fnl100_{wtype}_{pval:g}")
                m0 = m5.compute_alignment_metrics(s0, data0.xi_mean, data0.xi_std, xi0, f"fnl0_{wtype}_{pval:g}")

                score100 = float(m100["mean_abs_sigma"])
                score0 = float(m0["mean_abs_sigma"])
                joint_max = max(score100, score0)
                joint_mean = 0.5 * (score100 + score0)

                rows.append(
                    {
                        "window_type": wtype,
                        "window_param": pval,
                        "metric_fnl100": score100,
                        "metric_fnl0": score0,
                        "joint_max": joint_max,
                        "joint_mean": joint_mean,
                        "chi2ndof_fnl100": float(m100["chi2_ndof"]),
                        "chi2ndof_fnl0": float(m0["chi2_ndof"]),
                    }
                )
                xi100_map[(wtype, pval)] = xi100
                xi0_map[(wtype, pval)] = xi0

        # 6) 选“同时好”的窗口：先最小化 joint_max，再最小化 joint_mean
        best_row = min(rows, key=lambda r: (float(r["joint_max"]), float(r["joint_mean"])))
        best_wtype = str(best_row["window_type"])
        best_wparam = float(best_row["window_param"])

        xi100_best = xi100_map[(best_wtype, best_wparam)]
        xi0_best = xi0_map[(best_wtype, best_wparam)]

        # 7) 图1：联合权衡图（横轴 fnl0，纵轴 fnl100，越靠左下越好）
        x = np.array([float(r["metric_fnl0"]) for r in rows], dtype=float)
        y = np.array([float(r["metric_fnl100"]) for r in rows], dtype=float)
        c = np.array([float(r["joint_max"]) for r in rows], dtype=float)

        fig, ax = plt.subplots(1, 1, figsize=(8.2, 6.8))
        sc = ax.scatter(x, y, c=c, s=44, cmap="viridis", alpha=0.9)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("joint max metric")

        # 参考线：Baseline / TestA
        ax.scatter([float(m0_baseline["mean_abs_sigma"])], [float(m100_baseline["mean_abs_sigma"])],
                   marker="x", s=90, color="tab:red", label="Baseline")
        ax.scatter([float(m0_testa["mean_abs_sigma"])], [float(m100_testa["mean_abs_sigma"])],
                   marker="+", s=110, color="tab:orange", label="TestA (kmin=1e-4)")

        # 最优窗口点
        ax.scatter([float(best_row["metric_fnl0"])], [float(best_row["metric_fnl100"])],
                   marker="*", s=200, color="black", zorder=6, label=f"Best: {best_wtype}({best_wparam:g})")

        ax.set_xlabel(r"fnl0: large-scale mean$|\Delta/\sigma|$")
        ax.set_ylabel(r"fnl100: large-scale mean$|\Delta/\sigma|$")
        ax.set_title("Task5 v14: 3Gpc joint window scan (fnl0 + fnl100)")
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        fig.savefig(OUT_FIG1, bbox_inches="tight")
        plt.close(fig)

        # 8) 图2：最优窗口曲线对比
        fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.6), sharex=True)

        ax = axes[0]
        ax.errorbar(
            s100,
            s100**2 * data100.xi_mean,
            yerr=s100**2 * data100.xi_std,
            fmt="o--",
            ms=3.3,
            capsize=2,
            color="black",
            label="Measured fnl100",
        )
        ax.plot(s100, s100**2 * xi100_baseline, "-", lw=1.7, color="tab:blue", label="Baseline")
        ax.plot(s100, s100**2 * xi100_testa, "-", lw=1.6, color="tab:orange", label="TestA")
        ax.plot(s100, s100**2 * xi100_best, "-", lw=2.1, color="tab:red", label=f"TestB best: {best_wtype}({best_wparam:g})")
        ax.axvline(m5.LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_ylabel(r"$r^2\xi_0(r)$")
        ax.set_title("3Gpc fnl100")
        ax.legend(fontsize=8.3, ncol=2)

        ax = axes[1]
        ax.errorbar(
            s0,
            s0**2 * data0.xi_mean,
            yerr=s0**2 * data0.xi_std,
            fmt="o--",
            ms=3.3,
            capsize=2,
            color="black",
            label="Measured fnl0",
        )
        ax.plot(s0, s0**2 * xi0_baseline, "-", lw=1.7, color="tab:blue", label="Baseline")
        ax.plot(s0, s0**2 * xi0_testa, "-", lw=1.6, color="tab:orange", label="TestA")
        ax.plot(s0, s0**2 * xi0_best, "-", lw=2.1, color="tab:red", label=f"TestB best: {best_wtype}({best_wparam:g})")
        ax.axvline(m5.LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2\xi_0(r)$")
        ax.set_title("3Gpc fnl0")
        ax.legend(fontsize=8.3, ncol=2)

        fig.tight_layout()
        fig.savefig(OUT_FIG2, bbox_inches="tight")
        plt.close(fig)

        # 9) 文本总结
        with open(OUT_TXT, "w", encoding="utf-8") as f:
            f.write("任务5 v14：3Gpc 联合窗口扫描（fnl0 + fnl100）\n")
            f.write("========================================\n\n")

            f.write("[目标]\n")
            f.write("在 TESTB 方法下寻找一个窗口（同一类型+参数），使 fnl0 与 fnl100 的 2PCF 建模同时较好。\n")
            f.write("本次固定 kmin_global=1e-4，保留 FFTLog taper。\n\n")

            f.write("[设置]\n")
            f.write(f"- 数据: 3Gpc fnl100 N={data100.nmock}, rid={data100.realizations.min()}..{data100.realizations.max()}\n")
            f.write(f"- 数据: 3Gpc fnl0   N={data0.nmock}, rid={data0.realizations.min()}..{data0.realizations.max()}\n")
            f.write(f"- PK_FIT_KMAX={PK_FIT_KMAX}, 固定 p={P_FIXED}\n")
            f.write(f"- 积分: kmin_global={KMIN_GLOBAL:.1e}, kmax={KMAX_INT:.1f}\n")
            f.write(f"- TESTB 口径: direct-P window (use_png_only={USE_PNG_ONLY_WINDOW})\n")
            f.write(f"- 扫描候选数: {len(rows)}\n\n")

            f.write("[PK best-fit]\n")
            f.write(
                f"- fnl100: fnl_loc={bestfit100['fnl_loc']:.4f}, b1={bestfit100['b1']:.4f}, sigmas={bestfit100['sigmas']:.6f}\n"
            )
            f.write(
                f"- fnl0:   fnl_loc={bestfit0['fnl_loc']:.4f}, b1={bestfit0['b1']:.4f}, sigmas={bestfit0['sigmas']:.6f}\n\n"
            )

            f.write("[对照指标: large-scale mean|Δ/σ|]\n")
            f.write(
                f"- fnl100 Baseline={m100_baseline['mean_abs_sigma']:.4f}, TestA={m100_testa['mean_abs_sigma']:.4f}\n"
            )
            f.write(
                f"- fnl0   Baseline={m0_baseline['mean_abs_sigma']:.4f}, TestA={m0_testa['mean_abs_sigma']:.4f}\n\n"
            )

            f.write("[联合最优窗口]\n")
            f.write(f"- window_type={best_wtype}, window_param={best_wparam:g}\n")
            f.write(
                f"- fnl100: mean|Δ/σ|={best_row['metric_fnl100']:.4f}, chi2/ndof={best_row['chi2ndof_fnl100']:.4f}\n"
            )
            f.write(
                f"- fnl0:   mean|Δ/σ|={best_row['metric_fnl0']:.4f}, chi2/ndof={best_row['chi2ndof_fnl0']:.4f}\n"
            )
            f.write(
                f"- 联合目标: joint_max={best_row['joint_max']:.4f}, joint_mean={best_row['joint_mean']:.4f}\n\n"
            )

            # 按联合目标排序，给前5名
            top_rows = sorted(rows, key=lambda r: (float(r["joint_max"]), float(r["joint_mean"])))[:5]
            f.write("[联合目标前5名]\n")
            for i, r in enumerate(top_rows, start=1):
                f.write(
                    f"{i}. {r['window_type']}({float(r['window_param']):g}) | "
                    f"fnl100={float(r['metric_fnl100']):.4f}, fnl0={float(r['metric_fnl0']):.4f}, "
                    f"joint_max={float(r['joint_max']):.4f}, joint_mean={float(r['joint_mean']):.4f}\n"
                )
            f.write("\n")

            # 结论句
            both_good = (float(best_row["metric_fnl100"]) < 1.0) and (float(best_row["metric_fnl0"]) < 1.0)
            if both_good:
                f.write("[结论]\n")
                f.write("找到可行窗口：该窗口可同时把 fnl0 与 fnl100 维持在 mean|Δ/σ|<1 的良好建模区间。\n")
            else:
                f.write("[结论]\n")
                f.write("在本轮扫描参数内，已找到联合最优窗口，但至少一侧尚未进入 <1 区间；可继续细扫参数。\n")

        print("[INFO] done")
        print(f"[INFO] best window = {best_wtype}({best_wparam:g})")
        print(
            "[INFO] best metrics "
            f"fnl100={float(best_row['metric_fnl100']):.4f}, "
            f"fnl0={float(best_row['metric_fnl0']):.4f}, "
            f"joint_max={float(best_row['joint_max']):.4f}"
        )
        print(f"[INFO] fig1: {OUT_FIG1}")
        print(f"[INFO] fig2: {OUT_FIG2}")
        print(f"[INFO] summary: {OUT_TXT}")

    finally:
        # 恢复全局口径，避免影响其它脚本
        m5.FIXED_P = old_p
        m5.PK_FIT_KMAX = old_pkfit


if __name__ == "__main__":
    main()
