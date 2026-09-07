#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务5 v17：3Gpc 小尺度深扫（TESTB, direct-P window）

目标
----
在 v16 的基础上扩大窗口参数扫描范围，重点检查：
1) 小尺度 joint(fnl0+fnl100) 最优是否明显优于 v16
2) fnl100-only / fnl0-only 的小尺度可达下限分别是多少

固定设置
--------
- kmin_global = 1e-4
- kmax = 20
- 保留 FFTLog taper
- 小尺度区间: 56 <= r < 150 (3Gpc 数据最小 r 为 56)
- 大尺度对照: r >= 150
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

import task5_v14_3gpc_joint_fnl0_fnl100_testb_scan as v14
import task5_ir_window_solution_3gpc_multitype as m5

RMIN_SMALL = 56.0
RMAX_SMALL = 150.0
RMIN_LARGE = 150.0

WINDOW_SCANS_EXT = [
    ("exp_power", [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 40.0]),
    ("rational", [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 40.0]),
    ("tanh_log", [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0]),
    ("logcos", [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00]),
]

OUT_FIG = os.path.join(MISSION5_DIR, "task5_v17_3gpc_smallscale_deep_scan_curves.png")
OUT_TXT = os.path.join(MISSION5_DIR, "task5_v17_3gpc_smallscale_deep_scan_summary.txt")


plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"


def compute_metric(
    s: np.ndarray,
    xi_data: np.ndarray,
    xi_std: np.ndarray,
    xi_model: np.ndarray,
    rmin: float,
    rmax: float | None,
) -> float:
    if rmax is None:
        mask = s >= float(rmin)
    else:
        mask = (s >= float(rmin)) & (s < float(rmax))
    if not np.any(mask):
        raise RuntimeError(f"No bins in range: rmin={rmin}, rmax={rmax}")

    r2_data = s**2 * xi_data
    r2_model = s**2 * xi_model
    r2_std = np.maximum(s**2 * xi_std, 1e-12)
    resid = (r2_data[mask] - r2_model[mask]) / r2_std[mask]
    return float(np.mean(np.abs(resid)))


def eval_pair(
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
) -> Dict[str, float | str]:
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

    s100_small = compute_metric(s100, data100.xi_mean, data100.xi_std, xi100, RMIN_SMALL, RMAX_SMALL)
    s0_small = compute_metric(s0, data0.xi_mean, data0.xi_std, xi0, RMIN_SMALL, RMAX_SMALL)
    s100_large = compute_metric(s100, data100.xi_mean, data100.xi_std, xi100, RMIN_LARGE, None)
    s0_large = compute_metric(s0, data0.xi_mean, data0.xi_std, xi0, RMIN_LARGE, None)

    return {
        "window_type": wtype,
        "window_param": float(wparam),
        "small_fnl100": s100_small,
        "small_fnl0": s0_small,
        "small_joint_max": max(s100_small, s0_small),
        "small_joint_mean": 0.5 * (s100_small + s0_small),
        "large_fnl100": s100_large,
        "large_fnl0": s0_large,
        "large_joint_max": max(s100_large, s0_large),
        "large_joint_mean": 0.5 * (s100_large + s0_large),
    }


def render_single_curve(
    s: np.ndarray,
    k_grid: np.ndarray,
    p0: np.ndarray,
    p0_ref: np.ndarray,
    wtype: str,
    wparam: float,
) -> np.ndarray:
    return v14.evaluate_testb_3gpc_single_window(
        s_data=s,
        k_grid=k_grid,
        p0_in=p0,
        p0_ref_fnl0_same_other=p0_ref,
        window_type=wtype,
        window_param=wparam,
    )


def fmt_win(row: Dict[str, float | str]) -> str:
    return f"{row['window_type']}({float(row['window_param']):g})"


def main() -> None:
    old_p = float(m5.FIXED_P)
    old_pkfit = float(m5.PK_FIT_KMAX)

    m5.FIXED_P = float(v14.P_FIXED)
    m5.PK_FIT_KMAX = float(v14.PK_FIT_KMAX)

    try:
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

        rows: List[Dict[str, float | str]] = []
        for wtype, params in WINDOW_SCANS_EXT:
            for p in params:
                rows.append(eval_pair(s100, s0, data100, data0, k_grid, p0_100, p0_100_fnl0ref, p0_0, wtype, float(p)))

        best_joint = min(rows, key=lambda r: (float(r["small_joint_max"]), float(r["small_joint_mean"])))
        best_100 = min(rows, key=lambda r: (float(r["small_fnl100"]), float(r["small_fnl0"])))
        best_0 = min(rows, key=lambda r: (float(r["small_fnl0"]), float(r["small_fnl100"])))

        # Baseline / TestA 作为对照
        xi100_baseline = v14.evaluate_unwindowed_3gpc(s100, k_grid, p0_100, kmin=v14.K_FUND_3GPC)
        xi100_testa = v14.evaluate_unwindowed_3gpc(s100, k_grid, p0_100, kmin=v14.KMIN_GLOBAL)
        xi0_baseline = v14.evaluate_unwindowed_3gpc(s0, k_grid, p0_0, kmin=v14.K_FUND_3GPC)
        xi0_testa = v14.evaluate_unwindowed_3gpc(s0, k_grid, p0_0, kmin=v14.KMIN_GLOBAL)

        baseline_small_100 = compute_metric(s100, data100.xi_mean, data100.xi_std, xi100_baseline, RMIN_SMALL, RMAX_SMALL)
        baseline_small_0 = compute_metric(s0, data0.xi_mean, data0.xi_std, xi0_baseline, RMIN_SMALL, RMAX_SMALL)
        testa_small_100 = compute_metric(s100, data100.xi_mean, data100.xi_std, xi100_testa, RMIN_SMALL, RMAX_SMALL)
        testa_small_0 = compute_metric(s0, data0.xi_mean, data0.xi_std, xi0_testa, RMIN_SMALL, RMAX_SMALL)

        xi100_joint = render_single_curve(s100, k_grid, p0_100, p0_100_fnl0ref, str(best_joint["window_type"]), float(best_joint["window_param"]))
        xi100_b100 = render_single_curve(s100, k_grid, p0_100, p0_100_fnl0ref, str(best_100["window_type"]), float(best_100["window_param"]))
        xi100_b0 = render_single_curve(s100, k_grid, p0_100, p0_100_fnl0ref, str(best_0["window_type"]), float(best_0["window_param"]))

        xi0_joint = render_single_curve(s0, k_grid, p0_0, p0_0, str(best_joint["window_type"]), float(best_joint["window_param"]))
        xi0_b100 = render_single_curve(s0, k_grid, p0_0, p0_0, str(best_100["window_type"]), float(best_100["window_param"]))
        xi0_b0 = render_single_curve(s0, k_grid, p0_0, p0_0, str(best_0["window_type"]), float(best_0["window_param"]))

        fig, axes = plt.subplots(2, 1, figsize=(10.0, 9.0), sharex=True)

        ax = axes[0]
        ax.errorbar(s100, s100**2 * data100.xi_mean, yerr=s100**2 * data100.xi_std,
                    fmt="o--", ms=3.2, capsize=2, color="black", label="Measured fnl100")
        ax.plot(s100, s100**2 * xi100_baseline, color="tab:blue", lw=1.3, label="Baseline")
        ax.plot(s100, s100**2 * xi100_testa, color="tab:orange", lw=1.3, label="TestA")
        ax.plot(s100, s100**2 * xi100_joint, color="tab:purple", lw=2.2, label=f"small-joint best: {fmt_win(best_joint)}")
        ax.plot(s100, s100**2 * xi100_b100, color="tab:red", lw=1.8, label=f"fnl100-only best: {fmt_win(best_100)}")
        ax.plot(s100, s100**2 * xi100_b0, color="tab:green", lw=1.8, label=f"fnl0-only best: {fmt_win(best_0)}")
        ax.axvspan(RMIN_SMALL, RMAX_SMALL, color="gray", alpha=0.08, label="small-scale eval")
        ax.set_ylabel(r"$r^2\xi_0(r)$")
        ax.set_title("3Gpc fnl100")
        ax.legend(fontsize=7.8, ncol=2)

        ax = axes[1]
        ax.errorbar(s0, s0**2 * data0.xi_mean, yerr=s0**2 * data0.xi_std,
                    fmt="o--", ms=3.2, capsize=2, color="black", label="Measured fnl0")
        ax.plot(s0, s0**2 * xi0_baseline, color="tab:blue", lw=1.3, label="Baseline")
        ax.plot(s0, s0**2 * xi0_testa, color="tab:orange", lw=1.3, label="TestA")
        ax.plot(s0, s0**2 * xi0_joint, color="tab:purple", lw=2.2, label=f"small-joint best: {fmt_win(best_joint)}")
        ax.plot(s0, s0**2 * xi0_b100, color="tab:red", lw=1.8, label=f"fnl100-only best: {fmt_win(best_100)}")
        ax.plot(s0, s0**2 * xi0_b0, color="tab:green", lw=1.8, label=f"fnl0-only best: {fmt_win(best_0)}")
        ax.axvspan(RMIN_SMALL, RMAX_SMALL, color="gray", alpha=0.08, label="small-scale eval")
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2\xi_0(r)$")
        ax.set_title("3Gpc fnl0")
        ax.legend(fontsize=7.8, ncol=2)

        fig.tight_layout()
        fig.savefig(OUT_FIG, bbox_inches="tight")
        plt.close(fig)

        with open(OUT_TXT, "w", encoding="utf-8") as f:
            f.write("任务5 v17：3Gpc 小尺度深扫（TESTB）\n")
            f.write("================================\n\n")
            f.write(f"扫描窗口总数: {len(rows)}\n")
            f.write(f"小尺度区间: {RMIN_SMALL:.0f} <= r < {RMAX_SMALL:.0f}\n")
            f.write(f"对照区间: r >= {RMIN_LARGE:.0f}\n")
            f.write(f"积分口径: kmin_global={v14.KMIN_GLOBAL:.1e}, kmax={v14.KMAX_INT:.1f}, taper保留\n\n")

            f.write("[小尺度对照：Baseline/TestA]\n")
            f.write(f"- Baseline small fnl100={baseline_small_100:.4f}, fnl0={baseline_small_0:.4f}\n")
            f.write(f"- TestA    small fnl100={testa_small_100:.4f}, fnl0={testa_small_0:.4f}\n\n")

            f.write("[small-joint 最优]\n")
            f.write(f"- window={fmt_win(best_joint)}\n")
            f.write(f"- small fnl100={best_joint['small_fnl100']:.4f}, fnl0={best_joint['small_fnl0']:.4f}, joint_max={best_joint['small_joint_max']:.4f}\n")
            f.write(f"- large fnl100={best_joint['large_fnl100']:.4f}, fnl0={best_joint['large_fnl0']:.4f}, joint_max={best_joint['large_joint_max']:.4f}\n\n")

            f.write("[fnl100-only 小尺度最优]\n")
            f.write(f"- window={fmt_win(best_100)}\n")
            f.write(f"- small fnl100={best_100['small_fnl100']:.4f}, fnl0={best_100['small_fnl0']:.4f}\n")
            f.write(f"- large fnl100={best_100['large_fnl100']:.4f}, fnl0={best_100['large_fnl0']:.4f}\n\n")

            f.write("[fnl0-only 小尺度最优]\n")
            f.write(f"- window={fmt_win(best_0)}\n")
            f.write(f"- small fnl100={best_0['small_fnl100']:.4f}, fnl0={best_0['small_fnl0']:.4f}\n")
            f.write(f"- large fnl100={best_0['large_fnl100']:.4f}, fnl0={best_0['large_fnl0']:.4f}\n\n")

            f.write("[结论]\n")
            f.write("- 该深扫用于判断小尺度不匹配是否来自窗口参数范围不足。\n")
            f.write("- 若 fnl0-only 最优仍显著偏高，说明单窗口难以完全吸收 fnl0 的小尺度差异。\n")

        print("[INFO] done")
        print(f"[INFO] best_joint: {fmt_win(best_joint)} | small joint_max={best_joint['small_joint_max']:.4f}")
        print(f"[INFO] best_fnl100: {fmt_win(best_100)} | small fnl100={best_100['small_fnl100']:.4f}")
        print(f"[INFO] best_fnl0:   {fmt_win(best_0)} | small fnl0={best_0['small_fnl0']:.4f}")
        print(f"[INFO] fig: {OUT_FIG}")
        print(f"[INFO] summary: {OUT_TXT}")

    finally:
        m5.FIXED_P = old_p
        m5.PK_FIT_KMAX = old_pkfit


if __name__ == "__main__":
    main()
