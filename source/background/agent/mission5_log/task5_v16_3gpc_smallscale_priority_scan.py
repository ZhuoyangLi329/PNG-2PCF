#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务5 v16：3Gpc 小尺度优先的联合窗口扫描（fnl0 + fnl100）

目标
----
在 TESTB 方法下，把窗口参数选择标准从“大尺度优先”改为“小尺度优先”：
- 主评价区间: 50 <= r < 150 (Mpc/h)
- 积分口径保持不变: kmin_global=1e-4, kmax=20, 保留 FFTLog taper

输出
----
- 图: task5_v16_3gpc_smallscale_priority_curves.png
- 文本: task5_v16_3gpc_smallscale_priority_summary.txt
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

import task5_v14_3gpc_joint_fnl0_fnl100_testb_scan as v14
import task5_ir_window_solution_3gpc_multitype as m5

# 主评价区间（小尺度优先）
RMIN_SMALL = 50.0
RMAX_SMALL = 150.0
# 对照区间（大尺度）
RMIN_LARGE = 150.0

# 指定两个已讨论窗口，便于横向比较
WINDOW_1GPC_TYPE = "exp_power"
WINDOW_1GPC_PARAM = 4.0
WINDOW_3GPC_LARGEBEST_TYPE = "tanh_log"
WINDOW_3GPC_LARGEBEST_PARAM = 16.0

OUT_FIG = os.path.join(MISSION5_DIR, "task5_v16_3gpc_smallscale_priority_curves.png")
OUT_TXT = os.path.join(MISSION5_DIR, "task5_v16_3gpc_smallscale_priority_summary.txt")

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"


def compute_metrics_range(
    s: np.ndarray,
    xi_data: np.ndarray,
    xi_std: np.ndarray,
    xi_model: np.ndarray,
    rmin: float,
    rmax: float | None,
) -> Dict[str, float]:
    """在指定 r 区间计算 mean|Δ/σ| 等指标。"""
    if rmax is None:
        mask = s >= float(rmin)
    else:
        mask = (s >= float(rmin)) & (s < float(rmax))

    if np.sum(mask) == 0:
        raise RuntimeError(f"评估区间无数据: rmin={rmin}, rmax={rmax}")

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


def eval_window_pair(
    s100: np.ndarray,
    s0: np.ndarray,
    data100: m5.MockData,
    data0: m5.MockData,
    k_grid: np.ndarray,
    p0_100: np.ndarray,
    p0_100_fnl0ref: np.ndarray,
    p0_0: np.ndarray,
    wtype: str,
    wparam: float,
) -> Dict[str, float]:
    """评估同一窗口在 fnl100/fnl0 上的小尺度与大尺度指标。"""
    xi100 = v14.evaluate_testb_3gpc_single_window(
        s_data=s100,
        k_grid=k_grid,
        p0_in=p0_100,
        p0_ref_fnl0_same_other=p0_100_fnl0ref,
        window_type=wtype,
        window_param=wparam,
    )
    xi0 = v14.evaluate_testb_3gpc_single_window(
        s_data=s0,
        k_grid=k_grid,
        p0_in=p0_0,
        p0_ref_fnl0_same_other=p0_0,
        window_type=wtype,
        window_param=wparam,
    )

    m100_small = compute_metrics_range(s100, data100.xi_mean, data100.xi_std, xi100, RMIN_SMALL, RMAX_SMALL)
    m0_small = compute_metrics_range(s0, data0.xi_mean, data0.xi_std, xi0, RMIN_SMALL, RMAX_SMALL)
    m100_large = compute_metrics_range(s100, data100.xi_mean, data100.xi_std, xi100, RMIN_LARGE, None)
    m0_large = compute_metrics_range(s0, data0.xi_mean, data0.xi_std, xi0, RMIN_LARGE, None)

    return {
        "window_type": wtype,
        "window_param": float(wparam),
        "small_fnl100": float(m100_small["mean_abs_sigma"]),
        "small_fnl0": float(m0_small["mean_abs_sigma"]),
        "small_joint_max": max(float(m100_small["mean_abs_sigma"]), float(m0_small["mean_abs_sigma"])),
        "small_joint_mean": 0.5 * (float(m100_small["mean_abs_sigma"]) + float(m0_small["mean_abs_sigma"])),
        "large_fnl100": float(m100_large["mean_abs_sigma"]),
        "large_fnl0": float(m0_large["mean_abs_sigma"]),
        "large_joint_max": max(float(m100_large["mean_abs_sigma"]), float(m0_large["mean_abs_sigma"])),
    }


def main() -> None:
    old_p = float(m5.FIXED_P)
    old_pkfit = float(m5.PK_FIT_KMAX)

    m5.FIXED_P = float(v14.P_FIXED)
    m5.PK_FIT_KMAX = float(v14.PK_FIT_KMAX)

    try:
        # 1) 数据与 best-fit
        data100 = m5.load_mock_data("3gpc_fnl100", v14.PK_3GPC_FNL100_GLOB, v14.PCF_3GPC_FNL100_GLOB, v14.RID_MIN, v14.RID_MAX)
        data0 = m5.load_mock_data("3gpc_fnl0", v14.PK_3GPC_FNL0_GLOB, v14.PCF_3GPC_FNL0_GLOB, v14.RID_MIN, v14.RID_MAX)

        fit100 = v14.fit_best_pk_safe(data100)
        fit0 = v14.fit_best_pk_safe(data0)
        bestfit100 = fit100["bestfit"]
        bestfit0 = fit0["bestfit"]

        k_grid = np.geomspace(v14.KMIN_GLOBAL / m5.FFTLOG_PADDING, v14.KMAX_INT * m5.FFTLOG_PADDING, m5.FFTLOG_N)
        p0_100 = m5.build_theory_p0(k_grid, bestfit100)
        bestfit100_fnl0 = dict(bestfit100)
        bestfit100_fnl0["fnl_loc"] = 0.0
        p0_100_fnl0ref = m5.build_theory_p0(k_grid, bestfit100_fnl0)
        p0_0 = m5.build_theory_p0(k_grid, bestfit0)

        s100 = data100.scen
        s0 = data0.scen

        # 2) 扫描窗口，按小尺度目标选最优
        rows: List[Dict[str, float | str]] = []
        for wtype, params in v14.WINDOW_SCANS:
            for p in params:
                rows.append(
                    eval_window_pair(
                        s100, s0, data100, data0, k_grid,
                        p0_100, p0_100_fnl0ref, p0_0,
                        wtype, float(p),
                    )
                )

        best_small = min(rows, key=lambda r: (float(r["small_joint_max"]), float(r["small_joint_mean"])))
        best_small_type = str(best_small["window_type"])
        best_small_param = float(best_small["window_param"])

        # 3) 额外比较：1Gpc同款、3Gpc大尺度最优
        same_1gpc = eval_window_pair(
            s100, s0, data100, data0, k_grid,
            p0_100, p0_100_fnl0ref, p0_0,
            WINDOW_1GPC_TYPE, WINDOW_1GPC_PARAM,
        )
        best_large = eval_window_pair(
            s100, s0, data100, data0, k_grid,
            p0_100, p0_100_fnl0ref, p0_0,
            WINDOW_3GPC_LARGEBEST_TYPE, WINDOW_3GPC_LARGEBEST_PARAM,
        )

        # 4) 画曲线图：baseline/TestA/same1gpc/large-best/small-best
        xi100_baseline = v14.evaluate_unwindowed_3gpc(s100, k_grid, p0_100, kmin=v14.K_FUND_3GPC)
        xi100_testa = v14.evaluate_unwindowed_3gpc(s100, k_grid, p0_100, kmin=v14.KMIN_GLOBAL)
        xi100_same = v14.evaluate_testb_3gpc_single_window(s100, k_grid, p0_100, p0_100_fnl0ref, WINDOW_1GPC_TYPE, WINDOW_1GPC_PARAM)
        xi100_largebest = v14.evaluate_testb_3gpc_single_window(s100, k_grid, p0_100, p0_100_fnl0ref, WINDOW_3GPC_LARGEBEST_TYPE, WINDOW_3GPC_LARGEBEST_PARAM)
        xi100_smallbest = v14.evaluate_testb_3gpc_single_window(s100, k_grid, p0_100, p0_100_fnl0ref, best_small_type, best_small_param)

        xi0_baseline = v14.evaluate_unwindowed_3gpc(s0, k_grid, p0_0, kmin=v14.K_FUND_3GPC)
        xi0_testa = v14.evaluate_unwindowed_3gpc(s0, k_grid, p0_0, kmin=v14.KMIN_GLOBAL)
        xi0_same = v14.evaluate_testb_3gpc_single_window(s0, k_grid, p0_0, p0_0, WINDOW_1GPC_TYPE, WINDOW_1GPC_PARAM)
        xi0_largebest = v14.evaluate_testb_3gpc_single_window(s0, k_grid, p0_0, p0_0, WINDOW_3GPC_LARGEBEST_TYPE, WINDOW_3GPC_LARGEBEST_PARAM)
        xi0_smallbest = v14.evaluate_testb_3gpc_single_window(s0, k_grid, p0_0, p0_0, best_small_type, best_small_param)

        fig, axes = plt.subplots(2, 1, figsize=(9.6, 9.0), sharex=True)
        ax = axes[0]
        ax.errorbar(s100, s100**2 * data100.xi_mean, yerr=s100**2 * data100.xi_std,
                    fmt="o--", ms=3.1, capsize=2, color="black", label="Measured fnl100")
        ax.plot(s100, s100**2 * xi100_baseline, "-", lw=1.5, color="tab:blue", label="Baseline")
        ax.plot(s100, s100**2 * xi100_testa, "-", lw=1.3, color="tab:orange", label="TestA")
        ax.plot(s100, s100**2 * xi100_same, "-", lw=1.8, color="tab:red", label="same as 1Gpc: exp_power(4)")
        ax.plot(s100, s100**2 * xi100_largebest, "-", lw=1.8, color="tab:green", label="3Gpc large-best: tanh_log(16)")
        ax.plot(s100, s100**2 * xi100_smallbest, "-", lw=2.2, color="tab:purple", label=f"3Gpc small-best: {best_small_type}({best_small_param:g})")
        ax.axvline(RMAX_SMALL, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.axvline(RMIN_LARGE, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_ylabel(r"$r^2\xi_0(r)$")
        ax.set_title("3Gpc fnl100")
        ax.legend(fontsize=7.8, ncol=2)

        ax = axes[1]
        ax.errorbar(s0, s0**2 * data0.xi_mean, yerr=s0**2 * data0.xi_std,
                    fmt="o--", ms=3.1, capsize=2, color="black", label="Measured fnl0")
        ax.plot(s0, s0**2 * xi0_baseline, "-", lw=1.5, color="tab:blue", label="Baseline")
        ax.plot(s0, s0**2 * xi0_testa, "-", lw=1.3, color="tab:orange", label="TestA")
        ax.plot(s0, s0**2 * xi0_same, "-", lw=1.8, color="tab:red", label="same as 1Gpc: exp_power(4)")
        ax.plot(s0, s0**2 * xi0_largebest, "-", lw=1.8, color="tab:green", label="3Gpc large-best: tanh_log(16)")
        ax.plot(s0, s0**2 * xi0_smallbest, "-", lw=2.2, color="tab:purple", label=f"3Gpc small-best: {best_small_type}({best_small_param:g})")
        ax.axvline(RMAX_SMALL, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.axvline(RMIN_LARGE, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2\xi_0(r)$")
        ax.set_title("3Gpc fnl0")
        ax.legend(fontsize=7.8, ncol=2)

        fig.tight_layout()
        fig.savefig(OUT_FIG, bbox_inches="tight")
        plt.close(fig)

        # 5) 写总结
        with open(OUT_TXT, "w", encoding="utf-8") as f:
            f.write("任务5 v16：3Gpc 小尺度优先窗口扫描（r<150）\n")
            f.write("========================================\n\n")
            f.write(f"主评价区间: {RMIN_SMALL:.0f} <= r < {RMAX_SMALL:.0f}\n")
            f.write(f"对照区间: r >= {RMIN_LARGE:.0f}\n")
            f.write(f"积分口径: kmin_global={v14.KMIN_GLOBAL:.1e}, kmax={v14.KMAX_INT:.1f}, taper保留\n\n")

            f.write("[1Gpc同款窗口 exp_power(4)]\n")
            f.write(f"- small fnl100={same_1gpc['small_fnl100']:.4f}, fnl0={same_1gpc['small_fnl0']:.4f}, joint_max={same_1gpc['small_joint_max']:.4f}\n")
            f.write(f"- large fnl100={same_1gpc['large_fnl100']:.4f}, fnl0={same_1gpc['large_fnl0']:.4f}, joint_max={same_1gpc['large_joint_max']:.4f}\n\n")

            f.write("[3Gpc 大尺度最优窗口 tanh_log(16)]\n")
            f.write(f"- small fnl100={best_large['small_fnl100']:.4f}, fnl0={best_large['small_fnl0']:.4f}, joint_max={best_large['small_joint_max']:.4f}\n")
            f.write(f"- large fnl100={best_large['large_fnl100']:.4f}, fnl0={best_large['large_fnl0']:.4f}, joint_max={best_large['large_joint_max']:.4f}\n\n")

            f.write("[3Gpc 小尺度优先最优窗口]\n")
            f.write(f"- best_small = {best_small_type}({best_small_param:g})\n")
            f.write(f"- small fnl100={best_small['small_fnl100']:.4f}, fnl0={best_small['small_fnl0']:.4f}, joint_max={best_small['small_joint_max']:.4f}\n")
            f.write(f"- large fnl100={best_small['large_fnl100']:.4f}, fnl0={best_small['large_fnl0']:.4f}, joint_max={best_small['large_joint_max']:.4f}\n\n")

            f.write("[结论]\n")
            f.write("- 若更看重小尺度(r<150)，窗口选择会偏向更适配小尺度的参数。\n")
            f.write("- 1Gpc同款窗口可用，但通常不是3Gpc小尺度优先目标下的最优。\n")

        print("[INFO] done")
        print(
            f"[INFO] same(1gpc) small joint_max={same_1gpc['small_joint_max']:.4f}, "
            f"large joint_max={same_1gpc['large_joint_max']:.4f}"
        )
        print(
            f"[INFO] best_small={best_small_type}({best_small_param:g}), "
            f"small joint_max={best_small['small_joint_max']:.4f}, large joint_max={best_small['large_joint_max']:.4f}"
        )
        print(f"[INFO] fig: {OUT_FIG}")
        print(f"[INFO] summary: {OUT_TXT}")

    finally:
        m5.FIXED_P = old_p
        m5.PK_FIT_KMAX = old_pkfit


if __name__ == "__main__":
    main()
