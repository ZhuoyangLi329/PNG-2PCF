#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务5 v19：仅测试 TestC（3Gpc, fnl100）是否能解决 2PCF 建模偏差

代码大纲
--------
1) 读取 3Gpc fnl100 的 pk/pcf 样本，并做 best-fit P0(k)
2) 用同一 best-fit P0(k) 计算：
   - Baseline: kmin = 2pi/L（mission4 口径）
   - TestC: 低k离散求和 + 高k FFTLog，扫描 n_split
3) 量化大尺度对齐指标（r>=200，同时给 r>=150 参考）
4) 输出图和 summary，回答“TestC 是否解决问题”

说明
----
- 本脚本不做 TestB（按你的要求暂时抛开窗口法）
- 图中会包含测量均值（虚线）用于判断系统性偏低是否被修复
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
import task5_v14_3gpc_joint_fnl0_fnl100_testb_scan as v14

# 数据与扫描设置
RID_MIN = 2
RID_MAX = 99
NSPLIT_LIST = [2, 3, 4, 5, 6, 8, 10, 12, 16]
RMIN_LIST = [150.0, 200.0]

OUT_FIG = os.path.join(MISSION5_DIR, "task5_v19_testc_3gpc_fnl100_check.png")
OUT_TXT = os.path.join(MISSION5_DIR, "task5_v19_testc_3gpc_fnl100_check_summary.txt")

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"


def compute_metric_range(
    s: np.ndarray,
    xi_data: np.ndarray,
    xi_std: np.ndarray,
    xi_model: np.ndarray,
    rmin: float,
) -> Dict[str, float]:
    """在给定 r>=rmin 上计算对齐指标。"""
    mask = s >= float(rmin)
    if not np.any(mask):
        raise RuntimeError(f"没有可用 bin: r >= {rmin}")

    r2_data = s**2 * xi_data
    r2_model = s**2 * xi_model
    r2_std = np.maximum(s**2 * xi_std, 1e-12)
    resid = (r2_data[mask] - r2_model[mask]) / r2_std[mask]

    return {
        "nbin": int(np.sum(mask)),
        "mean_abs_sigma": float(np.mean(np.abs(resid))),
        "chi2_ndof": float(np.mean(resid**2)),
        "mean_sigma": float(np.mean(resid)),
        "max_abs_sigma": float(np.max(np.abs(resid))),
    }


def fmt_metric(m: Dict[str, float]) -> str:
    """格式化指标，便于写 summary。"""
    return (
        f"mean|Δ/σ|={m['mean_abs_sigma']:.4f}, "
        f"chi2/ndof={m['chi2_ndof']:.4f}, "
        f"mean(Δ/σ)={m['mean_sigma']:.4f}, "
        f"max|Δ/σ|={m['max_abs_sigma']:.4f}, "
        f"Nbin={int(m['nbin'])}"
    )


def main() -> None:
    # 暂存并恢复全局参数，避免影响其他脚本
    old_p = float(m5.FIXED_P)
    old_pkfit = float(m5.PK_FIT_KMAX)

    # 使用当前 mission5 的统一口径
    m5.FIXED_P = float(v14.P_FIXED)       # p=1.1
    m5.PK_FIT_KMAX = float(v14.PK_FIT_KMAX)  # 0.08

    try:
        # 1) 读取 3Gpc fnl100 数据
        data100 = m5.load_mock_data(
            "3gpc_fnl100",
            v14.PK_3GPC_FNL100_GLOB,
            v14.PCF_3GPC_FNL100_GLOB,
            RID_MIN,
            RID_MAX,
        )

        # 2) best-fit P0(k)
        fit = v14.fit_best_pk_safe(data100)
        bestfit = fit["bestfit"]

        k_grid = np.geomspace(
            v14.KMIN_GLOBAL / m5.FFTLOG_PADDING,
            v14.KMAX_INT * m5.FFTLOG_PADDING,
            m5.FFTLOG_N,
        )
        p0_best = m5.build_theory_p0(k_grid, bestfit)

        # 3) Baseline（kmin=2pi/L）
        xi_baseline = m5.evaluate_unwindowed_with_kmin(
            s_data=data100.scen,
            k_grid=k_grid,
            p0_base=p0_best,
            kmin=v14.K_FUND_3GPC,
            tag="Baseline",
        )

        # 4) TestC 扫 n_split
        testc_rows: List[Dict[str, object]] = []
        xi_testc_map: Dict[int, np.ndarray] = {}
        nmodes_map: Dict[int, int] = {}

        for nsp in NSPLIT_LIST:
            xi_c, nmodes = m5.evaluate_testc_hybrid(
                s_data=data100.scen,
                k_grid=k_grid,
                p0_base=p0_best,
                n_split=int(nsp),
            )
            xi_testc_map[int(nsp)] = xi_c
            nmodes_map[int(nsp)] = int(nmodes)

            row: Dict[str, object] = {
                "n_split": int(nsp),
                "k_split": float(nsp) * v14.K_FUND_3GPC,
                "nmodes": int(nmodes),
            }
            for rmin in RMIN_LIST:
                mm = compute_metric_range(data100.scen, data100.xi_mean, data100.xi_std, xi_c, rmin)
                row[f"mean_abs_sigma_r{int(rmin)}"] = mm["mean_abs_sigma"]
                row[f"chi2_ndof_r{int(rmin)}"] = mm["chi2_ndof"]
                row[f"mean_sigma_r{int(rmin)}"] = mm["mean_sigma"]
            testc_rows.append(row)

        # 以 r>=200 的 mean_abs_sigma 作为“是否解决大尺度偏差”的主指标
        best_row = min(testc_rows, key=lambda x: float(x["mean_abs_sigma_r200"]))
        best_nsplit = int(best_row["n_split"])
        xi_testc_best = xi_testc_map[best_nsplit]

        # Baseline 指标
        baseline_m150 = compute_metric_range(data100.scen, data100.xi_mean, data100.xi_std, xi_baseline, 150.0)
        baseline_m200 = compute_metric_range(data100.scen, data100.xi_mean, data100.xi_std, xi_baseline, 200.0)

        # Best TestC 指标
        best_m150 = compute_metric_range(data100.scen, data100.xi_mean, data100.xi_std, xi_testc_best, 150.0)
        best_m200 = compute_metric_range(data100.scen, data100.xi_mean, data100.xi_std, xi_testc_best, 200.0)

        # 5) 画图（数据均值 + Baseline + 各 TestC + best TestC）
        s = data100.scen
        r2_data = s**2 * data100.xi_mean
        r2_std = s**2 * data100.xi_std

        fig, axes = plt.subplots(2, 1, figsize=(9.8, 9.2), sharex=True)

        ax = axes[0]
        ax.errorbar(
            s,
            r2_data,
            yerr=r2_std,
            fmt="o--",
            ms=3.3,
            capsize=2,
            color="black",
            label="Measured mean (fnl100)",
        )
        ax.plot(s, s**2 * xi_baseline, color="tab:blue", lw=2.0, label="Baseline (kmin=2pi/L)")

        # 其他 n_split 先淡色画出
        for nsp in NSPLIT_LIST:
            if nsp == best_nsplit:
                continue
            ax.plot(s, s**2 * xi_testc_map[int(nsp)], color="gray", lw=1.0, alpha=0.35)

        ax.plot(
            s,
            s**2 * xi_testc_best,
            color="tab:red",
            lw=2.4,
            label=f"TestC best (n_split={best_nsplit}, nmodes={nmodes_map[best_nsplit]})",
        )

        ax.axvline(150.0, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.axvline(200.0, color="gray", lw=1.0, ls=":", alpha=0.9)
        ax.set_ylabel(r"$r^2\xi_0(r)$")
        ax.set_title("3Gpc fnl100: Baseline vs TestC (no TestB)")
        ax.legend(fontsize=8.5, ncol=1)

        ax = axes[1]
        resid_base = (r2_data - s**2 * xi_baseline) / np.maximum(r2_std, 1e-12)
        resid_best = (r2_data - s**2 * xi_testc_best) / np.maximum(r2_std, 1e-12)

        ax.axhline(0.0, color="black", lw=1.0)
        ax.plot(s, resid_base, "o-", ms=2.8, lw=1.2, color="tab:blue", label="Baseline")
        ax.plot(s, resid_best, "s-", ms=2.8, lw=1.2, color="tab:red", label=f"TestC best n={best_nsplit}")
        ax.axvline(150.0, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.axvline(200.0, color="gray", lw=1.0, ls=":", alpha=0.9)
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$(Data-Model)/\sigma$")
        ax.legend(fontsize=8.5)

        fig.tight_layout()
        fig.savefig(OUT_FIG, bbox_inches="tight")
        plt.close(fig)

        # 6) 给出“能否解决”的结论
        # 这里采用两个条件：
        # - 对比 baseline，r>=200 的 mean|Δ/σ| 明显下降；
        # - 同时 mean(Δ/σ) 的系统偏低被明显减弱（更接近0）。
        improve_abs = baseline_m200["mean_abs_sigma"] - best_m200["mean_abs_sigma"]
        improve_bias = abs(baseline_m200["mean_sigma"]) - abs(best_m200["mean_sigma"])

        solved = (improve_abs > 0.15) and (improve_bias > 0.15)

        with open(OUT_TXT, "w", encoding="utf-8") as f:
            f.write("任务5 v19：3Gpc fnl100 的 TestC 可行性检查（不使用 TestB）\n")
            f.write("==================================================\n\n")
            f.write("设置：\n")
            f.write("- 数据: 3Gpc fastPM fnl100, rid=2..99\n")
            f.write("- PK 拟合: k<=0.08, p=1.1 固定\n")
            f.write("- 积分: kmax=20, taper保留\n")
            f.write("- Baseline: kmin=2pi/L\n")
            f.write("- TestC: n_split 扫描 %s\n\n" % NSPLIT_LIST)

            f.write("best-fit 参数（fnl100）：\n")
            f.write(f"- fnl_loc={bestfit['fnl_loc']:.6f}, b1={bestfit['b1']:.6f}, sigmas={bestfit['sigmas']:.6f}, p={bestfit['p']:.3f}\n\n")

            f.write("[Baseline 指标]\n")
            f.write(f"- r>=150: {fmt_metric(baseline_m150)}\n")
            f.write(f"- r>=200: {fmt_metric(baseline_m200)}\n\n")

            f.write(f"[TestC 最优: n_split={best_nsplit}, k_split={best_nsplit * v14.K_FUND_3GPC:.6e}, nmodes={nmodes_map[best_nsplit]}]\n")
            f.write(f"- r>=150: {fmt_metric(best_m150)}\n")
            f.write(f"- r>=200: {fmt_metric(best_m200)}\n\n")

            f.write("[TestC 扫描表]\n")
            for row in sorted(testc_rows, key=lambda x: int(x["n_split"])):
                f.write(
                    "- n_split={n:2d}, k_split={k:.6e}, nmodes={m:5d}, "
                    "r>=200: mean|Δ/σ|={a200:.4f}, mean(Δ/σ)={b200:.4f}; "
                    "r>=150: mean|Δ/σ|={a150:.4f}, mean(Δ/σ)={b150:.4f}\n".format(
                        n=int(row["n_split"]),
                        k=float(row["k_split"]),
                        m=int(row["nmodes"]),
                        a200=float(row["mean_abs_sigma_r200"]),
                        b200=float(row["mean_sigma_r200"]),
                        a150=float(row["mean_abs_sigma_r150"]),
                        b150=float(row["mean_sigma_r150"]),
                    )
                )
            f.write("\n")

            f.write("[结论]\n")
            if solved:
                f.write("- 在当前口径下，TestC 对 3Gpc/fnl100 的大尺度建模有明显改进，可视作基本解决。\n")
            else:
                f.write("- 在当前口径下，TestC 对 3Gpc/fnl100 仅有有限改进，未达到“明显解决”水平。\n")
            f.write(
                f"- 定量上（r>=200）：mean|Δ/σ| 改善 {improve_abs:+.4f}，"
                f"|mean(Δ/σ)| 改善 {improve_bias:+.4f}。\n"
            )

        print("[INFO] done")
        print(f"[INFO] best n_split={best_nsplit}, nmodes={nmodes_map[best_nsplit]}")
        print("[INFO] baseline r>=200:", fmt_metric(baseline_m200))
        print("[INFO] testc_best r>=200:", fmt_metric(best_m200))
        print(f"[INFO] fig: {OUT_FIG}")
        print(f"[INFO] summary: {OUT_TXT}")

    finally:
        m5.FIXED_P = old_p
        m5.PK_FIT_KMAX = old_pkfit


if __name__ == "__main__":
    main()
