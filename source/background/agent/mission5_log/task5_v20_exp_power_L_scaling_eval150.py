#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务5 v20：TESTB 固定窗口形状 exp_power(x)，测试 x 与盒长 L 的关系（评估 r>=150）

代码大纲
--------
1) 读取 1Gpc/3Gpc 的 fnl100 与 fnl0 数据，并分别做 PK best-fit。
2) 仅使用 TESTB 的 exp_power(x) 窗，扫描 x，统一用 r>=150 的 2PCF 指标评估。
3) 重点回答：
   - 1Gpc 的 x=4 放到 3Gpc 为什么不够好？
   - 让 x 与 L 相关（线性示例 x=4*(L/1000)）后，3Gpc 是否改善？
4) 输出：
   - 图1：metric vs x（1Gpc/3Gpc, fnl100/fnl0）
   - 图2：3Gpc 曲线对比（x=4, x=12, x_best）
   - 文本总结（含推荐 x(L) 关系）

说明
----
- 保留 FFTLog taper。
- 积分设置按 mission5 当前口径：kmin_global=1e-4, kmax=20。
- 指标统一使用 large-scale: r>=150 Mpc/h。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MISSION5_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission5_log"
sys.path.insert(0, MISSION5_DIR)

import task5_ir_window_solution_3gpc_multitype as m5
import task5_v13_1gpc_joint_fnl0_fnl100_testb_scan as v13
import task5_v14_3gpc_joint_fnl0_fnl100_testb_scan as v14

# 统一评估口径
RMIN_EVAL = 150.0

# 扫描 exp_power 的 x 参数（覆盖从缓到陡，并延伸到大 x）
X_SCAN = [
    0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
    5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0,
    28.0, 32.0, 40.0, 50.0, 64.0, 80.0, 100.0, 128.0, 160.0, 200.0, 256.0,
]

# 对比的“线性 L 关系”示例：x(L)=4*(L/1000)
L1 = 1000.0
L3 = 3000.0
X_1GPC_ANCHOR = 4.0
X_LINEAR_PROP_3GPC = X_1GPC_ANCHOR * (L3 / L1)  # 12

OUT_FIG1 = os.path.join(MISSION5_DIR, "task5_v20_exp_power_metric_vs_x_eval150.png")
OUT_FIG2 = os.path.join(MISSION5_DIR, "task5_v20_exp_power_3gpc_curves_eval150.png")
OUT_TXT = os.path.join(MISSION5_DIR, "task5_v20_exp_power_L_scaling_eval150_summary.txt")

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"


def compute_metric_rge(
    s: np.ndarray,
    xi_data: np.ndarray,
    xi_std: np.ndarray,
    xi_model: np.ndarray,
    rmin: float,
) -> Dict[str, float]:
    """计算 r>=rmin 上的对齐指标（在 r^2 xi 空间）。"""
    mask = s >= float(rmin)
    if not np.any(mask):
        raise RuntimeError(f"评估区间为空：r>={rmin}")

    r2_data = s**2 * xi_data
    r2_model = s**2 * xi_model
    r2_std = np.maximum(s**2 * xi_std, 1e-12)
    resid = (r2_data[mask] - r2_model[mask]) / r2_std[mask]

    return {
        "mean_abs_sigma": float(np.mean(np.abs(resid))),
        "chi2_ndof": float(np.mean(resid**2)),
        "mean_sigma": float(np.mean(resid)),
        "max_abs_sigma": float(np.max(np.abs(resid))),
        "nbin": int(np.sum(mask)),
    }


def evaluate_exp_x_for_1gpc(
    x: float,
    s100: np.ndarray,
    s0: np.ndarray,
    k_grid: np.ndarray,
    p0_100: np.ndarray,
    p0_100_fnl0ref: np.ndarray,
    p0_0: np.ndarray,
    data100: m5.MockData,
    data0: m5.MockData,
) -> Dict[str, float]:
    """评估 1Gpc 在给定 exp_power(x) 下的 fnl100/fnl0 指标。"""
    xi100 = v13.evaluate_testb_1gpc_single_window(
        s_data=s100,
        k_grid=k_grid,
        p0_in=p0_100,
        p0_ref_fnl0_same_other=p0_100_fnl0ref,
        window_type="exp_power",
        window_param=float(x),
    )
    xi0 = v13.evaluate_testb_1gpc_single_window(
        s_data=s0,
        k_grid=k_grid,
        p0_in=p0_0,
        p0_ref_fnl0_same_other=p0_0,
        window_type="exp_power",
        window_param=float(x),
    )

    m100 = compute_metric_rge(s100, data100.xi_mean, data100.xi_std, xi100, RMIN_EVAL)
    m0 = compute_metric_rge(s0, data0.xi_mean, data0.xi_std, xi0, RMIN_EVAL)

    return {
        "x": float(x),
        "fnl100": float(m100["mean_abs_sigma"]),
        "fnl0": float(m0["mean_abs_sigma"]),
        "joint_max": max(float(m100["mean_abs_sigma"]), float(m0["mean_abs_sigma"])),
        "joint_mean": 0.5 * (float(m100["mean_abs_sigma"]) + float(m0["mean_abs_sigma"])),
    }


def evaluate_exp_x_for_3gpc(
    x: float,
    s100: np.ndarray,
    s0: np.ndarray,
    k_grid: np.ndarray,
    p0_100: np.ndarray,
    p0_100_fnl0ref: np.ndarray,
    p0_0: np.ndarray,
    data100: m5.MockData,
    data0: m5.MockData,
) -> Dict[str, float]:
    """评估 3Gpc 在给定 exp_power(x) 下的 fnl100/fnl0 指标。"""
    xi100 = v14.evaluate_testb_3gpc_single_window(
        s_data=s100,
        k_grid=k_grid,
        p0_in=p0_100,
        p0_ref_fnl0_same_other=p0_100_fnl0ref,
        window_type="exp_power",
        window_param=float(x),
    )
    xi0 = v14.evaluate_testb_3gpc_single_window(
        s_data=s0,
        k_grid=k_grid,
        p0_in=p0_0,
        p0_ref_fnl0_same_other=p0_0,
        window_type="exp_power",
        window_param=float(x),
    )

    m100 = compute_metric_rge(s100, data100.xi_mean, data100.xi_std, xi100, RMIN_EVAL)
    m0 = compute_metric_rge(s0, data0.xi_mean, data0.xi_std, xi0, RMIN_EVAL)

    return {
        "x": float(x),
        "fnl100": float(m100["mean_abs_sigma"]),
        "fnl0": float(m0["mean_abs_sigma"]),
        "joint_max": max(float(m100["mean_abs_sigma"]), float(m0["mean_abs_sigma"])),
        "joint_mean": 0.5 * (float(m100["mean_abs_sigma"]) + float(m0["mean_abs_sigma"])),
    }


def main() -> None:
    # 暂存全局参数，避免影响其他脚本
    old_p = float(m5.FIXED_P)
    old_pkfit = float(m5.PK_FIT_KMAX)

    # 统一 mission5 当前口径
    m5.FIXED_P = float(v14.P_FIXED)          # p=1.1
    m5.PK_FIT_KMAX = float(v14.PK_FIT_KMAX)  # 0.08

    try:
        # 1) 读取 1Gpc 数据
        d1_100 = m5.load_mock_data(
            "1gpc_fnl100",
            v13.PK_1GPC_FNL100_GLOB,
            v13.PCF_1GPC_FNL100_GLOB,
            v13.RID_MIN,
            v13.RID_MAX,
        )
        d1_0 = m5.load_mock_data(
            "1gpc_fnl0",
            v13.PK_1GPC_FNL0_GLOB,
            v13.PCF_1GPC_FNL0_GLOB,
            v13.RID_MIN,
            v13.RID_MAX,
        )

        # 2) 读取 3Gpc 数据
        d3_100 = m5.load_mock_data(
            "3gpc_fnl100",
            v14.PK_3GPC_FNL100_GLOB,
            v14.PCF_3GPC_FNL100_GLOB,
            v14.RID_MIN,
            v14.RID_MAX,
        )
        d3_0 = m5.load_mock_data(
            "3gpc_fnl0",
            v14.PK_3GPC_FNL0_GLOB,
            v14.PCF_3GPC_FNL0_GLOB,
            v14.RID_MIN,
            v14.RID_MAX,
        )

        # 3) best-fit P0
        fit1_100 = v13.fit_best_pk_safe(d1_100)
        fit1_0 = v13.fit_best_pk_safe(d1_0)
        fit3_100 = v14.fit_best_pk_safe(d3_100)
        fit3_0 = v14.fit_best_pk_safe(d3_0)

        b1_100 = fit1_100["bestfit"]
        b1_0 = fit1_0["bestfit"]
        b3_100 = fit3_100["bestfit"]
        b3_0 = fit3_0["bestfit"]

        # 统一 k 网格
        k_grid = np.geomspace(
            v14.KMIN_GLOBAL / m5.FFTLOG_PADDING,
            v14.KMAX_INT * m5.FFTLOG_PADDING,
            m5.FFTLOG_N,
        )

        # 构建 P0 数组（含 fnl100 对应的 fnl=0 参考曲线）
        p1_100 = m5.build_theory_p0(k_grid, b1_100)
        b1_100_ref = dict(b1_100)
        b1_100_ref["fnl_loc"] = 0.0
        p1_100_ref0 = m5.build_theory_p0(k_grid, b1_100_ref)
        p1_0 = m5.build_theory_p0(k_grid, b1_0)

        p3_100 = m5.build_theory_p0(k_grid, b3_100)
        b3_100_ref = dict(b3_100)
        b3_100_ref["fnl_loc"] = 0.0
        p3_100_ref0 = m5.build_theory_p0(k_grid, b3_100_ref)
        p3_0 = m5.build_theory_p0(k_grid, b3_0)

        # 4) 扫描 x
        rows_1: List[Dict[str, float]] = []
        rows_3: List[Dict[str, float]] = []
        for x in X_SCAN:
            rows_1.append(
                evaluate_exp_x_for_1gpc(
                    x, d1_100.scen, d1_0.scen, k_grid,
                    p1_100, p1_100_ref0, p1_0, d1_100, d1_0,
                )
            )
            rows_3.append(
                evaluate_exp_x_for_3gpc(
                    x, d3_100.scen, d3_0.scen, k_grid,
                    p3_100, p3_100_ref0, p3_0, d3_100, d3_0,
                )
            )

        # 各自最优 x
        best1_joint = min(rows_1, key=lambda r: (r["joint_max"], r["joint_mean"]))
        best3_joint = min(rows_3, key=lambda r: (r["joint_max"], r["joint_mean"]))
        best3_fnl100 = min(rows_3, key=lambda r: r["fnl100"])

        x3_joint = float(best3_joint["x"])
        x3_f100 = float(best3_fnl100["x"])

        # 线性关系示例A：x=4*(L/1000)
        x3_linear_prop = float(X_LINEAR_PROP_3GPC)

        # 线性关系示例B：通过 (1000,4) 与 (3000,x3_f100) 的线性插值
        # x(L) = a + b*(L-1000), 其中 a=4, b=(x3_f100-4)/2000
        b_lin = (x3_f100 - X_1GPC_ANCHOR) / (L3 - L1)

        # 便捷函数：按给定 x 取 3Gpc 行
        def pick3(xval: float) -> Dict[str, float]:
            return min(rows_3, key=lambda r: abs(r["x"] - float(xval)))

        r3_x4 = pick3(4.0)
        r3_x12 = pick3(x3_linear_prop)
        r3_xbest = pick3(x3_f100)

        # 5) 图1：metric vs x
        xarr = np.array([r["x"] for r in rows_3], dtype=float)
        y3_100 = np.array([r["fnl100"] for r in rows_3], dtype=float)
        y3_0 = np.array([r["fnl0"] for r in rows_3], dtype=float)
        y3_joint = np.array([r["joint_max"] for r in rows_3], dtype=float)

        y1_100 = np.array([r["fnl100"] for r in rows_1], dtype=float)
        y1_0 = np.array([r["fnl0"] for r in rows_1], dtype=float)
        y1_joint = np.array([r["joint_max"] for r in rows_1], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)

        ax = axes[0]
        ax.plot(xarr, y1_100, "o-", ms=3.6, lw=1.6, color="tab:red", label="1Gpc fnl100")
        ax.plot(xarr, y1_0, "s-", ms=3.2, lw=1.4, color="tab:blue", label="1Gpc fnl0")
        ax.plot(xarr, y1_joint, "^-", ms=3.2, lw=1.4, color="tab:green", label="1Gpc joint_max")
        ax.axvline(4.0, color="gray", ls="--", lw=1.0, alpha=0.8, label="x=4 anchor")
        ax.set_title("1Gpc, r>=150")
        ax.set_xlabel("exp_power parameter x")
        ax.set_ylabel(r"mean$|\Delta/\sigma|$")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.plot(xarr, y3_100, "o-", ms=3.6, lw=1.6, color="tab:red", label="3Gpc fnl100")
        ax.plot(xarr, y3_0, "s-", ms=3.2, lw=1.4, color="tab:blue", label="3Gpc fnl0")
        ax.plot(xarr, y3_joint, "^-", ms=3.2, lw=1.4, color="tab:green", label="3Gpc joint_max")
        ax.axvline(4.0, color="gray", ls="--", lw=1.0, alpha=0.8, label="x=4")
        ax.axvline(x3_linear_prop, color="tab:purple", ls="--", lw=1.0, alpha=0.9, label=f"x=4*(L/1000)={x3_linear_prop:g}")
        ax.axvline(x3_f100, color="black", ls=":", lw=1.2, alpha=0.95, label=f"best x (fnl100)={x3_f100:g}")
        ax.set_title("3Gpc, r>=150")
        ax.set_xlabel("exp_power parameter x")
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(OUT_FIG1, bbox_inches="tight")
        plt.close(fig)

        # 6) 图2：3Gpc 曲线对比（测量均值用虚线）
        # baseline 仅作参照
        xi3_baseline_100 = v14.evaluate_unwindowed_3gpc(d3_100.scen, k_grid, p3_100, kmin=v14.K_FUND_3GPC)
        xi3_baseline_0 = v14.evaluate_unwindowed_3gpc(d3_0.scen, k_grid, p3_0, kmin=v14.K_FUND_3GPC)

        def xi3_for_x(x: float, is_fnl100: bool) -> np.ndarray:
            if is_fnl100:
                return v14.evaluate_testb_3gpc_single_window(
                    s_data=d3_100.scen,
                    k_grid=k_grid,
                    p0_in=p3_100,
                    p0_ref_fnl0_same_other=p3_100_ref0,
                    window_type="exp_power",
                    window_param=float(x),
                )
            return v14.evaluate_testb_3gpc_single_window(
                s_data=d3_0.scen,
                k_grid=k_grid,
                p0_in=p3_0,
                p0_ref_fnl0_same_other=p3_0,
                window_type="exp_power",
                window_param=float(x),
            )

        xi3_100_x4 = xi3_for_x(4.0, True)
        xi3_100_x12 = xi3_for_x(x3_linear_prop, True)
        xi3_100_xbest = xi3_for_x(x3_f100, True)

        xi3_0_x4 = xi3_for_x(4.0, False)
        xi3_0_x12 = xi3_for_x(x3_linear_prop, False)
        xi3_0_xbest = xi3_for_x(x3_f100, False)

        fig, axes = plt.subplots(2, 1, figsize=(9.6, 9.0), sharex=True)

        ax = axes[0]
        ax.errorbar(
            d3_100.scen,
            d3_100.scen**2 * d3_100.xi_mean,
            yerr=d3_100.scen**2 * d3_100.xi_std,
            fmt="o--",
            ms=3.1,
            capsize=2,
            color="black",
            label="Measured mean fnl100",
        )
        ax.plot(d3_100.scen, d3_100.scen**2 * xi3_baseline_100, color="tab:blue", lw=1.5, label="Baseline")
        ax.plot(d3_100.scen, d3_100.scen**2 * xi3_100_x4, color="tab:red", lw=1.8, label="exp_power(4)")
        ax.plot(d3_100.scen, d3_100.scen**2 * xi3_100_x12, color="tab:purple", lw=1.8, label=f"exp_power({x3_linear_prop:g})")
        ax.plot(d3_100.scen, d3_100.scen**2 * xi3_100_xbest, color="tab:green", lw=2.1, label=f"exp_power(best={x3_f100:g})")
        ax.axvline(RMIN_EVAL, color="gray", ls="--", lw=1.0, alpha=0.8)
        ax.set_ylabel(r"$r^2\xi_0(r)$")
        ax.set_title("3Gpc fnl100 (focus r>=150)")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.errorbar(
            d3_0.scen,
            d3_0.scen**2 * d3_0.xi_mean,
            yerr=d3_0.scen**2 * d3_0.xi_std,
            fmt="o--",
            ms=3.1,
            capsize=2,
            color="black",
            label="Measured mean fnl0",
        )
        ax.plot(d3_0.scen, d3_0.scen**2 * xi3_baseline_0, color="tab:blue", lw=1.5, label="Baseline")
        ax.plot(d3_0.scen, d3_0.scen**2 * xi3_0_x4, color="tab:red", lw=1.8, label="exp_power(4)")
        ax.plot(d3_0.scen, d3_0.scen**2 * xi3_0_x12, color="tab:purple", lw=1.8, label=f"exp_power({x3_linear_prop:g})")
        ax.plot(d3_0.scen, d3_0.scen**2 * xi3_0_xbest, color="tab:green", lw=2.1, label=f"exp_power(best={x3_f100:g})")
        ax.axvline(RMIN_EVAL, color="gray", ls="--", lw=1.0, alpha=0.8)
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2\xi_0(r)$")
        ax.set_title("3Gpc fnl0 (reference)")
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(OUT_FIG2, bbox_inches="tight")
        plt.close(fig)

        # 7) 写 summary
        with open(OUT_TXT, "w", encoding="utf-8") as f:
            f.write("任务5 v20：exp_power(x) 的 L 依赖测试（统一 r>=150）\n")
            f.write("==============================================\n\n")
            f.write("设置：\n")
            f.write("- TESTB direct-P window（不使用 PNG-only）\n")
            f.write("- 窗口固定形状：exp_power(x)\n")
            f.write(f"- x 扫描: {X_SCAN}\n")
            f.write("- 评估区间：r>=150 Mpc/h\n")
            f.write("- 积分：kmin_global=1e-4, kmax=20, taper保留\n")
            f.write("- PK拟合：k<=0.08, p=1.1\n\n")

            f.write("[1Gpc 最优（按 joint_max）]\n")
            f.write(
                f"- x={best1_joint['x']:.2f}, fnl100={best1_joint['fnl100']:.4f}, "
                f"fnl0={best1_joint['fnl0']:.4f}, joint_max={best1_joint['joint_max']:.4f}\n\n"
            )

            f.write("[3Gpc 最优]\n")
            f.write(
                f"- 按 fnl100 最优: x={best3_fnl100['x']:.2f}, fnl100={best3_fnl100['fnl100']:.4f}, "
                f"fnl0={best3_fnl100['fnl0']:.4f}, joint_max={best3_fnl100['joint_max']:.4f}\n"
            )
            f.write(
                f"- 按 joint_max 最优: x={best3_joint['x']:.2f}, fnl100={best3_joint['fnl100']:.4f}, "
                f"fnl0={best3_joint['fnl0']:.4f}, joint_max={best3_joint['joint_max']:.4f}\n\n"
            )

            f.write("[关键对比（3Gpc）]\n")
            f.write(
                f"- x=4 (1Gpc同款): fnl100={r3_x4['fnl100']:.4f}, fnl0={r3_x4['fnl0']:.4f}, "
                f"joint_max={r3_x4['joint_max']:.4f}\n"
            )
            f.write(
                f"- x=12 (线性比例 x=4*L/1000): fnl100={r3_x12['fnl100']:.4f}, fnl0={r3_x12['fnl0']:.4f}, "
                f"joint_max={r3_x12['joint_max']:.4f}\n"
            )
            f.write(
                f"- x=best_fnl100={r3_xbest['x']:.2f}: fnl100={r3_xbest['fnl100']:.4f}, fnl0={r3_xbest['fnl0']:.4f}, "
                f"joint_max={r3_xbest['joint_max']:.4f}\n\n"
            )

            f.write("[建议的 x(L) 线性写法]\n")
            f.write("- 示例A（简单比例）: x(L)=4*(L/1000)\n")
            f.write(
                f"- 示例B（锚定 1Gpc=4 且匹配 3Gpc fnl100 最优）: "
                f"x(L)=4+({b_lin:.6e})*(L-1000)\n"
            )
            f.write(f"  该式在 L=3000 时给 x={x3_f100:.2f}\n\n")

            f.write("[结论]\n")
            if r3_x12["fnl100"] <= r3_x4["fnl100"]:
                f.write("- 让 x 随 L 线性增大（至少从 4 到 12）对 3Gpc/fnl100 有帮助。\n")
            else:
                f.write("- 简单线性比例 x=4*(L/1000) 对 3Gpc/fnl100 不足，需要重新标定斜率。\n")
            if np.isclose(x3_f100, max(X_SCAN)):
                f.write("- 注意：最佳 x 落在扫描上边界，说明在 exp_power 族内“更陡”仍在持续改善，有限扫描暂无稳定内部最优。\n")
            f.write("- 但是否“好用”仍以你关心的 fnl100 指标为主，fnl0可作为副约束。\n")

        print("[INFO] done")
        print(f"[INFO] best1_joint x={best1_joint['x']:.2f}, metric={best1_joint['joint_max']:.4f}")
        print(f"[INFO] best3_fnl100 x={best3_fnl100['x']:.2f}, fnl100={best3_fnl100['fnl100']:.4f}")
        print(f"[INFO] best3_joint x={best3_joint['x']:.2f}, joint={best3_joint['joint_max']:.4f}")
        print(f"[INFO] 3Gpc x=4: fnl100={r3_x4['fnl100']:.4f}, fnl0={r3_x4['fnl0']:.4f}")
        print(f"[INFO] 3Gpc x=12: fnl100={r3_x12['fnl100']:.4f}, fnl0={r3_x12['fnl0']:.4f}")
        print(f"[INFO] fig1: {OUT_FIG1}")
        print(f"[INFO] fig2: {OUT_FIG2}")
        print(f"[INFO] summary: {OUT_TXT}")

    finally:
        m5.FIXED_P = old_p
        m5.PK_FIT_KMAX = old_pkfit


if __name__ == "__main__":
    main()
