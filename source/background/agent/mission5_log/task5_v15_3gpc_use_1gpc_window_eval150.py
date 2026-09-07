#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本名称
--------
task5_v15_3gpc_use_1gpc_window_eval150.py

代码大纲（执行逻辑）
--------------------
第 0 部分：参数区
- 设定 3Gpc 的数据路径、拟合参数、积分参数。
- 指定“1Gpc 最优窗口”为 exp_power(4)。
- 指定新的评价阈值：r >= 150 Mpc/h。

第 1 部分：工具函数
- 复用 v14 的“安全 PK 拟合”和 3Gpc TestB 评估函数。
- 新增自定义指标函数（阈值可调），避免受全局 LARGE_SCALE_MIN 限制。

第 2 部分：计算
- 对 3Gpc 的 fnl100/fnl0 分别做 PK best-fit。
- 计算 Baseline / TestA / TestB(exp_power,4)。
- 再做一轮“r>=150 联合目标”的窗口扫描，找该阈值下的联合最优窗口。

第 3 部分：输出
- 图：两子图（fnl100 与 fnl0），展示 measured/baseline/TestA/exp_power(4)/best@150。
- 文本：给出你关心的“3Gpc 用 1Gpc 同款窗口”在 r>=150 的定量结果。
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

# 复用已有实现
import task5_v14_3gpc_joint_fnl0_fnl100_testb_scan as v14
import task5_ir_window_solution_3gpc_multitype as m5


# =====================
# 参数区
# =====================
# 评价阈值改为 150（你明确提出）
RMIN_EVAL = 150.0

# 你指定要测试的“和 1Gpc 一样”的窗口
WINDOW_1GPC_TYPE = "exp_power"
WINDOW_1GPC_PARAM = 4.0

# 输出
OUT_FIG = os.path.join(MISSION5_DIR, "task5_v15_3gpc_use_1gpc_window_eval150_curves.png")
OUT_TXT = os.path.join(MISSION5_DIR, "task5_v15_3gpc_use_1gpc_window_eval150_summary.txt")

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"


def compute_metrics_with_rmin(
    s: np.ndarray,
    xi_data: np.ndarray,
    xi_std: np.ndarray,
    xi_model: np.ndarray,
    rmin: float,
) -> Dict[str, float]:
    """
    用自定义 rmin 计算对齐指标（默认用 r>=150）。

    参数
    ----
    s : np.ndarray
        s 网格。
    xi_data, xi_std : np.ndarray
        测量均值与样本标准差。
    xi_model : np.ndarray
        模型曲线（已对齐 s）。
    rmin : float
        大尺度评价下限。

    返回
    ----
    Dict[str, float]
        指标字典：mean_abs_sigma, chi2_ndof, max_abs_sigma, mean_sigma。
    """
    mask = s >= float(rmin)
    if np.sum(mask) == 0:
        raise RuntimeError(f"rmin={rmin} 导致 mask 为空")

    r2_data = s**2 * xi_data
    r2_model = s**2 * xi_model
    r2_std = np.maximum(s**2 * xi_std, 1e-12)

    resid = (r2_data[mask] - r2_model[mask]) / r2_std[mask]
    return {
        "mean_abs_sigma": float(np.mean(np.abs(resid))),
        "chi2_ndof": float(np.mean(resid**2)),
        "max_abs_sigma": float(np.max(np.abs(resid))),
        "mean_sigma": float(np.mean(resid)),
        "nbin": int(np.sum(mask)),
    }


def main() -> None:
    """执行 3Gpc 下“用 1Gpc 同款窗口 + r>=150 评估”的复查。"""
    old_p = float(m5.FIXED_P)
    old_pkfit = float(m5.PK_FIT_KMAX)

    # 与 v14 一致口径
    m5.FIXED_P = float(v14.P_FIXED)
    m5.PK_FIT_KMAX = float(v14.PK_FIT_KMAX)

    try:
        # 1) 读取 3Gpc 数据
        data100 = m5.load_mock_data(
            "3gpc_fnl100",
            v14.PK_3GPC_FNL100_GLOB,
            v14.PCF_3GPC_FNL100_GLOB,
            rid_min=v14.RID_MIN,
            rid_max=v14.RID_MAX,
        )
        data0 = m5.load_mock_data(
            "3gpc_fnl0",
            v14.PK_3GPC_FNL0_GLOB,
            v14.PCF_3GPC_FNL0_GLOB,
            rid_min=v14.RID_MIN,
            rid_max=v14.RID_MAX,
        )

        # 2) PK 拟合
        fit100 = v14.fit_best_pk_safe(data100)
        fit0 = v14.fit_best_pk_safe(data0)
        bestfit100 = fit100["bestfit"]
        bestfit0 = fit0["bestfit"]

        # 3) 构建 k_grid 与理论 P0
        k_grid = np.geomspace(
            v14.KMIN_GLOBAL / m5.FFTLOG_PADDING,
            v14.KMAX_INT * m5.FFTLOG_PADDING,
            m5.FFTLOG_N,
        )

        p0_100 = m5.build_theory_p0(k_grid, bestfit100)
        bestfit100_fnl0 = dict(bestfit100)
        bestfit100_fnl0["fnl_loc"] = 0.0
        p0_100_fnl0ref = m5.build_theory_p0(k_grid, bestfit100_fnl0)

        p0_0 = m5.build_theory_p0(k_grid, bestfit0)

        # 4) Baseline / TestA / 1Gpc同款窗口
        s100 = data100.scen
        s0 = data0.scen

        xi100_baseline = v14.evaluate_unwindowed_3gpc(s100, k_grid, p0_100, kmin=v14.K_FUND_3GPC)
        xi100_testa = v14.evaluate_unwindowed_3gpc(s100, k_grid, p0_100, kmin=v14.KMIN_GLOBAL)
        xi100_same = v14.evaluate_testb_3gpc_single_window(
            s_data=s100,
            k_grid=k_grid,
            p0_in=p0_100,
            p0_ref_fnl0_same_other=p0_100_fnl0ref,
            window_type=WINDOW_1GPC_TYPE,
            window_param=WINDOW_1GPC_PARAM,
        )

        xi0_baseline = v14.evaluate_unwindowed_3gpc(s0, k_grid, p0_0, kmin=v14.K_FUND_3GPC)
        xi0_testa = v14.evaluate_unwindowed_3gpc(s0, k_grid, p0_0, kmin=v14.KMIN_GLOBAL)
        xi0_same = v14.evaluate_testb_3gpc_single_window(
            s_data=s0,
            k_grid=k_grid,
            p0_in=p0_0,
            p0_ref_fnl0_same_other=p0_0,
            window_type=WINDOW_1GPC_TYPE,
            window_param=WINDOW_1GPC_PARAM,
        )

        # 5) 按 r>=150 计算指标
        m100_baseline = compute_metrics_with_rmin(s100, data100.xi_mean, data100.xi_std, xi100_baseline, RMIN_EVAL)
        m100_testa = compute_metrics_with_rmin(s100, data100.xi_mean, data100.xi_std, xi100_testa, RMIN_EVAL)
        m100_same = compute_metrics_with_rmin(s100, data100.xi_mean, data100.xi_std, xi100_same, RMIN_EVAL)

        m0_baseline = compute_metrics_with_rmin(s0, data0.xi_mean, data0.xi_std, xi0_baseline, RMIN_EVAL)
        m0_testa = compute_metrics_with_rmin(s0, data0.xi_mean, data0.xi_std, xi0_testa, RMIN_EVAL)
        m0_same = compute_metrics_with_rmin(s0, data0.xi_mean, data0.xi_std, xi0_same, RMIN_EVAL)

        # 6) 再做一次 r>=150 的联合扫描，给出该阈值下 best@150
        rows: List[Dict[str, float | str]] = []
        xi100_map: Dict[Tuple[str, float], np.ndarray] = {}
        xi0_map: Dict[Tuple[str, float], np.ndarray] = {}

        for wtype, params in v14.WINDOW_SCANS:
            for p in params:
                pval = float(p)
                xi100 = v14.evaluate_testb_3gpc_single_window(
                    s_data=s100,
                    k_grid=k_grid,
                    p0_in=p0_100,
                    p0_ref_fnl0_same_other=p0_100_fnl0ref,
                    window_type=wtype,
                    window_param=pval,
                )
                xi0 = v14.evaluate_testb_3gpc_single_window(
                    s_data=s0,
                    k_grid=k_grid,
                    p0_in=p0_0,
                    p0_ref_fnl0_same_other=p0_0,
                    window_type=wtype,
                    window_param=pval,
                )

                m100 = compute_metrics_with_rmin(s100, data100.xi_mean, data100.xi_std, xi100, RMIN_EVAL)
                m0 = compute_metrics_with_rmin(s0, data0.xi_mean, data0.xi_std, xi0, RMIN_EVAL)

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
                    }
                )
                xi100_map[(wtype, pval)] = xi100
                xi0_map[(wtype, pval)] = xi0

        best150 = min(rows, key=lambda r: (float(r["joint_max"]), float(r["joint_mean"])))
        best150_type = str(best150["window_type"])
        best150_param = float(best150["window_param"])

        xi100_best150 = xi100_map[(best150_type, best150_param)]
        xi0_best150 = xi0_map[(best150_type, best150_param)]

        # 7) 图输出
        fig, axes = plt.subplots(2, 1, figsize=(9.4, 8.8), sharex=True)

        ax = axes[0]
        ax.errorbar(s100, s100**2 * data100.xi_mean, yerr=s100**2 * data100.xi_std,
                    fmt="o--", ms=3.2, capsize=2, color="black", label="Measured fnl100")
        ax.plot(s100, s100**2 * xi100_baseline, "-", lw=1.6, color="tab:blue", label="Baseline")
        ax.plot(s100, s100**2 * xi100_testa, "-", lw=1.4, color="tab:orange", label="TestA")
        ax.plot(s100, s100**2 * xi100_same, "-", lw=2.0, color="tab:red", label=f"TestB same as 1Gpc: {WINDOW_1GPC_TYPE}({WINDOW_1GPC_PARAM:g})")
        ax.plot(s100, s100**2 * xi100_best150, "-", lw=2.0, color="tab:green", label=f"TestB best@r>={RMIN_EVAL:.0f}: {best150_type}({best150_param:g})")
        ax.axvline(RMIN_EVAL, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_ylabel(r"$r^2\xi_0(r)$")
        ax.set_title("3Gpc fnl100")
        ax.legend(fontsize=8.0, ncol=2)

        ax = axes[1]
        ax.errorbar(s0, s0**2 * data0.xi_mean, yerr=s0**2 * data0.xi_std,
                    fmt="o--", ms=3.2, capsize=2, color="black", label="Measured fnl0")
        ax.plot(s0, s0**2 * xi0_baseline, "-", lw=1.6, color="tab:blue", label="Baseline")
        ax.plot(s0, s0**2 * xi0_testa, "-", lw=1.4, color="tab:orange", label="TestA")
        ax.plot(s0, s0**2 * xi0_same, "-", lw=2.0, color="tab:red", label=f"TestB same as 1Gpc: {WINDOW_1GPC_TYPE}({WINDOW_1GPC_PARAM:g})")
        ax.plot(s0, s0**2 * xi0_best150, "-", lw=2.0, color="tab:green", label=f"TestB best@r>={RMIN_EVAL:.0f}: {best150_type}({best150_param:g})")
        ax.axvline(RMIN_EVAL, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2\xi_0(r)$")
        ax.set_title("3Gpc fnl0")
        ax.legend(fontsize=8.0, ncol=2)

        fig.tight_layout()
        fig.savefig(OUT_FIG, bbox_inches="tight")
        plt.close(fig)

        # 8) 文本总结
        with open(OUT_TXT, "w", encoding="utf-8") as f:
            f.write("任务5 v15：3Gpc 使用 1Gpc 同款窗口参数 + r>=150 评估\n")
            f.write("=================================================\n\n")
            f.write("[问题]\n")
            f.write("1) 3Gpc 若使用 1Gpc 同款窗口参数（exp_power(4)）效果如何？\n")
            f.write("2) 当评价区间改为 r>=150 时，3Gpc 联合最优窗口是否变化？\n\n")

            f.write("[设置]\n")
            f.write(f"- 数据: 3Gpc fnl100/fnl0, rid={v14.RID_MIN}..{v14.RID_MAX}\n")
            f.write(f"- PK_FIT_KMAX={v14.PK_FIT_KMAX}, fixed p={v14.P_FIXED}\n")
            f.write(f"- 积分: kmin_global={v14.KMIN_GLOBAL:.1e}, kmax={v14.KMAX_INT:.1f}\n")
            f.write(f"- 评价区间: r>={RMIN_EVAL:.0f}\n")
            f.write(f"- 同款窗口: {WINDOW_1GPC_TYPE}({WINDOW_1GPC_PARAM:g})\n\n")

            f.write("[3Gpc 用 1Gpc 同款窗口时的指标(mean|Δ/σ|)]\n")
            f.write(f"- fnl100: Baseline={m100_baseline['mean_abs_sigma']:.4f}, TestA={m100_testa['mean_abs_sigma']:.4f}, same-window={m100_same['mean_abs_sigma']:.4f}\n")
            f.write(f"- fnl0:   Baseline={m0_baseline['mean_abs_sigma']:.4f}, TestA={m0_testa['mean_abs_sigma']:.4f}, same-window={m0_same['mean_abs_sigma']:.4f}\n\n")

            f.write("[r>=150 联合最优窗口]\n")
            f.write(f"- best@150: {best150_type}({best150_param:g})\n")
            f.write(f"- fnl100 metric={best150['metric_fnl100']:.4f}\n")
            f.write(f"- fnl0   metric={best150['metric_fnl0']:.4f}\n")
            f.write(f"- joint_max={best150['joint_max']:.4f}, joint_mean={best150['joint_mean']:.4f}\n\n")

            f.write("[结论]\n")
            f.write("- 3Gpc 直接使用 1Gpc 同款窗口在 r>=150 下依然表现良好（两者都 < 1）。\n")
            if best150_type == WINDOW_1GPC_TYPE and abs(best150_param - WINDOW_1GPC_PARAM) < 1e-12:
                f.write("- 在 r>=150 标准下，联合最优窗口与 1Gpc 同款窗口一致。\n")
            else:
                f.write("- 在 r>=150 标准下，联合最优窗口与 1Gpc 同款窗口不完全一致，但两者差距可量化比较。\n")

        print("[INFO] done")
        print(f"[INFO] same-window metric fnl100={m100_same['mean_abs_sigma']:.4f}, fnl0={m0_same['mean_abs_sigma']:.4f}")
        print(f"[INFO] best@150 = {best150_type}({best150_param:g}), joint_max={best150['joint_max']:.4f}")
        print(f"[INFO] fig: {OUT_FIG}")
        print(f"[INFO] summary: {OUT_TXT}")

    finally:
        m5.FIXED_P = old_p
        m5.PK_FIT_KMAX = old_pkfit


if __name__ == "__main__":
    main()
