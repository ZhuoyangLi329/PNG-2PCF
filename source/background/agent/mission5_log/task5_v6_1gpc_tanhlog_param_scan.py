#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本名称
--------
task5_v6_1gpc_tanhlog_param_scan.py

目的
----
回答问题：对于 1Gpc，改变窗口参数（tanh_log 的 beta）能否把 2PCF 建模做得更好？

策略
----
1) 固定此前口径：p=1.5, PK_FIT_KMAX=0.08, B 方法直接对总 P0 乘窗口。
2) 在 1Gpc 数据上扫描 tanh_log 参数 beta（默认 3~16）。
3) 指标使用大尺度 mean|Δ/σ|（r>=200），与 baseline 和 param=8 对比。

输出
----
- 图1: 参数扫描曲线（metric vs beta）
- 图2: baseline / param=8 / best-param 的 2PCF 对比
- 文本: summary
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

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

# 1Gpc 数据
PK_1GPC_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat"
PCF_1GPC_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat"

# 盒长
BOX_SIZE_1GPC = 1000.0
K_FUND_1GPC = 2.0 * np.pi / BOX_SIZE_1GPC

# 固定口径
P_FIXED = 1.5
KMIN_GLOBAL = 1e-4
KMAX_INT = m5.KMAX_INT
EDGE_TAPER_FRAC = m5.EDGE_TAPER_FRAC
USE_PNG_ONLY_WINDOW = False

# 扫描参数（可通过环境变量覆盖）
# 例如：TASK5_V6_BETAS="3,4,5,6,7,8,9,10,12,14,16"
BETAS_ENV = os.getenv("TASK5_V6_BETAS", "").strip()
if BETAS_ENV:
    BETA_LIST = [float(x) for x in BETAS_ENV.split(",")]
else:
    BETA_LIST = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]

OUT_FIG1 = os.path.join(MISSION5_DIR, "task5_v6_1gpc_tanhlog_metric_scan.png")
OUT_FIG2 = os.path.join(MISSION5_DIR, "task5_v6_1gpc_tanhlog_best_vs_fixed8.png")
OUT_SUMMARY = os.path.join(MISSION5_DIR, "task5_v6_1gpc_tanhlog_param_scan_summary.txt")

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
    """无 IR 窗版本，仅由 kmin 控制低 k。"""
    taper = m5.build_log_taper_window(k_grid, kmin=kmin, kmax=KMAX_INT, frac=EDGE_TAPER_FRAC)
    p0_eff = p0_base * taper
    r_grid, xi_grid = m5.xi_fftlog_from_effective_p0(k_grid, p0_eff)
    return m5.interp_xi_to_s(s_data, r_grid, xi_grid)


def evaluate_tanhlog_window_xi(
    s_data: np.ndarray,
    k_grid: np.ndarray,
    p0_fnl100: np.ndarray,
    p0_fnl0_ref: np.ndarray,
    beta: float,
) -> np.ndarray:
    """B 方法：窗口类型固定 tanh_log，仅扫描 beta 参数。"""
    ir_window = m5.build_ir_window(k_grid, kf=K_FUND_1GPC, wtype="tanh_log", param=float(beta))

    if USE_PNG_ONLY_WINDOW:
        delta_png = p0_fnl100 - p0_fnl0_ref
        p0_ir = p0_fnl0_ref + ir_window * delta_png
    else:
        p0_ir = p0_fnl100 * ir_window

    taper = m5.build_log_taper_window(k_grid, kmin=KMIN_GLOBAL, kmax=KMAX_INT, frac=EDGE_TAPER_FRAC)
    p0_eff = p0_ir * taper
    r_grid, xi_grid = m5.xi_fftlog_from_effective_p0(k_grid, p0_eff)
    return m5.interp_xi_to_s(s_data, r_grid, xi_grid)


def main() -> None:
    """执行 1Gpc 的 tanh_log 参数扫描。"""
    old_p = float(m5.FIXED_P)
    m5.FIXED_P = P_FIXED

    try:
        # 1) 数据和 best-fit
        data = m5.load_mock_data("1gpc_fnl100", PK_1GPC_GLOB, PCF_1GPC_GLOB, rid_min=1, rid_max=99)
        fit = m5.fit_best_pk(data)
        bestfit = fit["bestfit"]

        # 2) 理论 P0
        k_grid = np.geomspace(KMIN_GLOBAL / m5.FFTLOG_PADDING, KMAX_INT * m5.FFTLOG_PADDING, m5.FFTLOG_N)
        p0_100 = m5.build_theory_p0(k_grid, bestfit)
        bestfit_fnl0 = dict(bestfit)
        bestfit_fnl0["fnl_loc"] = 0.0
        p0_fnl0_ref = m5.build_theory_p0(k_grid, bestfit_fnl0)

        s = data.scen
        xi_data = data.xi_mean
        xi_std = data.xi_std

        # 3) baseline 与 TestA 参照
        xi_baseline = evaluate_unwindowed_xi(s, k_grid, p0_100, kmin=K_FUND_1GPC)
        xi_testa = evaluate_unwindowed_xi(s, k_grid, p0_100, kmin=KMIN_GLOBAL)
        m_base = m5.compute_alignment_metrics(s, xi_data, xi_std, xi_baseline, "Baseline_1gpc")
        m_a = m5.compute_alignment_metrics(s, xi_data, xi_std, xi_testa, "TestA_1gpc")

        # 4) 扫描 beta
        rows: List[Dict[str, float]] = []
        beta_to_xi: Dict[float, np.ndarray] = {}
        beta_to_metric: Dict[float, float] = {}

        for beta in BETA_LIST:
            b = float(beta)
            xi_model = evaluate_tanhlog_window_xi(s, k_grid, p0_100, p0_fnl0_ref, beta=b)
            metric = m5.compute_alignment_metrics(s, xi_data, xi_std, xi_model, f"tanh_log_{b:g}")

            beta_to_xi[b] = xi_model
            beta_to_metric[b] = float(metric["mean_abs_sigma"])
            rows.append(
                {
                    "beta": b,
                    "mean_abs_sigma": float(metric["mean_abs_sigma"]),
                    "chi2_ndof": float(metric["chi2_ndof"]),
                    "mean_sigma": float(metric["mean_sigma"]),
                }
            )

        # 5) 找最优参数
        best_row = min(rows, key=lambda r: r["mean_abs_sigma"])
        best_beta = float(best_row["beta"])
        fixed8_metric = beta_to_metric.get(8.0, np.nan)

        # 6) 图1：metric 扫描
        xs = np.array(sorted(beta_to_metric.keys()), dtype=float)
        ys = np.array([beta_to_metric[x] for x in xs], dtype=float)

        fig, ax = plt.subplots(1, 1, figsize=(8.6, 5.6))
        ax.plot(xs, ys, "o-", lw=1.8, ms=5.0, color="tab:blue", label=r"TestB tanh\_log scan")
        ax.scatter([best_beta], [best_row["mean_abs_sigma"]], color="tab:red", s=56, zorder=5, label="best beta")
        if np.isfinite(fixed8_metric):
            ax.scatter([8.0], [fixed8_metric], color="tab:green", s=48, zorder=5, label="beta=8")

        ax.axhline(m_base["mean_abs_sigma"], color="gray", lw=1.2, ls="--", label="Baseline")
        ax.axhline(m_a["mean_abs_sigma"], color="tab:orange", lw=1.0, ls="--", label="TestA")

        ax.set_xlabel(r"tanh\_log shape parameter $\beta$")
        ax.set_ylabel(r"large-scale mean$|\Delta/\sigma|$")
        ax.set_title("Task5 v6: 1Gpc tanh_log parameter scan (direct-P window)")
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        fig.savefig(OUT_FIG1, bbox_inches="tight")
        plt.close(fig)

        # 7) 图2：最优 vs fixed8 vs baseline
        xi_best = beta_to_xi[best_beta]
        xi_fix8 = beta_to_xi[8.0] if 8.0 in beta_to_xi else None

        fig, axes = plt.subplots(2, 1, figsize=(9.4, 8.4), sharex=True)

        ax = axes[0]
        ax.errorbar(s, s**2 * xi_data, yerr=s**2 * xi_std, fmt="o", ms=3.5, capsize=2,
                    color="black", label=f"Measured mean (N={data.nmock})")
        ax.plot(s, s**2 * xi_baseline, "-", lw=1.8, color="tab:blue", label="Baseline (kmin=2pi/L_1gpc)")
        if xi_fix8 is not None:
            ax.plot(s, s**2 * xi_fix8, "-", lw=1.8, color="tab:green", label="TestB tanh_log(beta=8)")
        ax.plot(s, s**2 * xi_best, "-", lw=2.1, color="tab:red", label=f"TestB tanh_log(best beta={best_beta:g})")

        ax.axvline(m5.LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_ylabel(r"$r^2\xi_0(r)$")
        ax.set_title("Task5 v6: 1Gpc modeling with tanh_log parameter tuning")
        ax.legend(fontsize=8.5, ncol=2)

        txt = (
            f"PK best-fit (p={P_FIXED}): fnl={bestfit['fnl_loc']:.2f}, b1={bestfit['b1']:.3f}, sigmas={bestfit['sigmas']:.4f}\n"
            f"Baseline={m_base['mean_abs_sigma']:.3f}, beta=8={fixed8_metric:.3f}, best={best_row['mean_abs_sigma']:.3f}"
        )
        ax.text(
            0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8.3,
            bbox=dict(facecolor="white", alpha=0.86, edgecolor="gray"),
        )

        # 残差
        ax = axes[1]
        r2_sigma = np.maximum((s**2) * xi_std, 1e-12)
        res_base = (s**2 * (xi_data - xi_baseline)) / r2_sigma
        res_best = (s**2 * (xi_data - xi_best)) / r2_sigma
        ax.axhline(0.0, color="black", lw=1.0)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax.axhline(-1.0, color="gray", lw=0.8, ls="--")
        ax.plot(s, res_base, "o-", ms=3.0, lw=1.2, color="tab:blue", label="Baseline")
        if xi_fix8 is not None:
            res_fix8 = (s**2 * (xi_data - xi_fix8)) / r2_sigma
            ax.plot(s, res_fix8, "s-", ms=3.0, lw=1.2, color="tab:green", label="beta=8")
        ax.plot(s, res_best, "^-", ms=3.0, lw=1.2, color="tab:red", label=f"best beta={best_beta:g}")

        ax.axvline(m5.LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$(Data-Model)/\sigma$ of $r^2\xi_0$")
        ax.legend(fontsize=8.5)

        fig.tight_layout()
        fig.savefig(OUT_FIG2, bbox_inches="tight")
        plt.close(fig)

        # 8) 文本总结
        with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
            f.write("任务5 v6：1Gpc tanh_log 窗口参数扫描\n")
            f.write("=================================\n\n")
            f.write("[问题]\n")
            f.write("对于 1Gpc，改变窗口参数（tanh_log 的 beta）能否建模得更好？\n\n")

            f.write("[设置]\n")
            f.write(f"- 数据: 1Gpc fnl100, Nmock={data.nmock}, rid={data.realizations.min()}..{data.realizations.max()}\n")
            f.write(f"- 固定 p={P_FIXED}, PK_FIT_KMAX={m5.PK_FIT_KMAX}\n")
            f.write(f"- B 方法: 直接乘总 P0, window=tanh_log(beta)\n")
            f.write(f"- 扫描 beta: {','.join([f'{x:g}' for x in sorted(beta_to_metric.keys())])}\n")
            f.write(f"- 指标: large-scale mean|Δ/σ|, r>={m5.LARGE_SCALE_MIN}\n\n")

            f.write("[PK best-fit]\n")
            f.write(
                f"- fnl_loc={bestfit['fnl_loc']:.4f}, b1={bestfit['b1']:.4f}, sigmas={bestfit['sigmas']:.6f}\n\n"
            )

            f.write("[参考项]\n")
            f.write(
                f"- Baseline: mean|Δ/σ|={m_base['mean_abs_sigma']:.4f}, "
                f"chi2/ndof={m_base['chi2_ndof']:.4f}\n"
            )
            f.write(
                f"- TestA: mean|Δ/σ|={m_a['mean_abs_sigma']:.4f}, "
                f"chi2/ndof={m_a['chi2_ndof']:.4f}\n"
            )
            if np.isfinite(fixed8_metric):
                f.write(f"- TestB(beta=8): mean|Δ/σ|={fixed8_metric:.4f}\n")
            f.write("\n")

            f.write("[扫描结果]\n")
            for r in sorted(rows, key=lambda x: x["beta"]):
                f.write(
                    f"- beta={r['beta']:>4g}: mean|Δ/σ|={r['mean_abs_sigma']:.4f}, "
                    f"chi2/ndof={r['chi2_ndof']:.4f}, mean(Δ/σ)={r['mean_sigma']:.4f}\n"
                )
            f.write("\n")

            f.write("[最优参数]\n")
            f.write(
                f"- best beta={best_beta:g}, mean|Δ/σ|={best_row['mean_abs_sigma']:.4f}, "
                f"chi2/ndof={best_row['chi2_ndof']:.4f}\n"
            )
            if np.isfinite(fixed8_metric):
                f.write(f"- 相对 beta=8 改进量: {fixed8_metric - best_row['mean_abs_sigma']:.4f}\n")
            f.write(f"- 相对 Baseline 改进量: {m_base['mean_abs_sigma'] - best_row['mean_abs_sigma']:.4f}\n\n")

            f.write("[结论]\n")
            if np.isfinite(fixed8_metric) and best_row["mean_abs_sigma"] < fixed8_metric:
                f.write("1) 可以。1Gpc 下调节窗口参数后，确实能比 beta=8 再提升。\n")
            else:
                f.write("1) 在本次扫描范围内，未看到比 beta=8 更明显的提升。\n")
            if best_row["mean_abs_sigma"] < 1.0:
                f.write("2) 最优参数仍保持在 mean|Δ/σ|<1 的可接受建模区间。\n")
            else:
                f.write("2) 最优参数尚未达到 mean|Δ/σ|<1。\n")

        print("[INFO] done")
        print(f"[INFO] fig1: {OUT_FIG1}")
        print(f"[INFO] fig2: {OUT_FIG2}")
        print(f"[INFO] summary: {OUT_SUMMARY}")
        print(
            "[INFO] metrics "
            f"baseline={m_base['mean_abs_sigma']:.4f}, "
            f"testA={m_a['mean_abs_sigma']:.4f}, "
            f"fixed8={fixed8_metric:.4f}, "
            f"best(beta={best_beta:g})={best_row['mean_abs_sigma']:.4f}"
        )

    finally:
        m5.FIXED_P = old_p


if __name__ == "__main__":
    main()

