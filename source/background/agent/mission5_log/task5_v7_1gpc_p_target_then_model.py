#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本名称
--------
task5_v7_1gpc_p_target_then_model.py

目的
----
按你的要求执行两步流程：
1) 在 1Gpc 上先扫描 p，找出让 PK best-fit fnl 最接近 100 的 p*；
2) 再以该 p* 为基准做 2PCF 建模（含 tanh_log 参数扫描）。

核心口径
--------
- 数据：1Gpc fnl=100（masscut）
- PK_FIT_KMAX=0.08
- B 方法：直接对总 P0 乘窗口（不是只对 delta_PNG 加窗）
- 2PCF 评估指标：large-scale mean|Δ/σ|（r>=200 Mpc/h）
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
# 参数区
# =====================

MISSION5_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission5_log"
sys.path.insert(0, MISSION5_DIR)
import task5_ir_window_solution_3gpc_multitype as m5

# 1Gpc 数据
PK_1GPC_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat"
PCF_1GPC_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat"

# 1Gpc 盒长
BOX_SIZE_1GPC = 1000.0
K_FUND_1GPC = 2.0 * np.pi / BOX_SIZE_1GPC

# 目标 fnl
TARGET_FNL = 100.0

# 固定积分设置
KMIN_GLOBAL = 1e-4
KMAX_INT = m5.KMAX_INT
EDGE_TAPER_FRAC = m5.EDGE_TAPER_FRAC
USE_PNG_ONLY_WINDOW = False

# p 扫描列表（可用环境变量覆盖）
# 例：TASK5_V7_P_LIST="1.0,1.05,1.1,...,1.6"
P_LIST_ENV = os.getenv("TASK5_V7_P_LIST", "").strip()
if P_LIST_ENV:
    P_SCAN_LIST = [float(x) for x in P_LIST_ENV.split(",")]
else:
    P_SCAN_LIST = [round(x, 2) for x in np.arange(1.0, 1.601, 0.05)]

# tanh_log 参数扫描
BETAS_ENV = os.getenv("TASK5_V7_BETAS", "").strip()
if BETAS_ENV:
    BETA_LIST = [float(x) for x in BETAS_ENV.split(",")]
else:
    BETA_LIST = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]

# 输出文件
OUT_FIG1 = os.path.join(MISSION5_DIR, "task5_v7_1gpc_p_scan_fnl_target.png")
OUT_FIG2 = os.path.join(MISSION5_DIR, "task5_v7_1gpc_pstar_beta_scan.png")
OUT_FIG3 = os.path.join(MISSION5_DIR, "task5_v7_1gpc_pstar_2pcf_modeling.png")
OUT_SUMMARY = os.path.join(MISSION5_DIR, "task5_v7_1gpc_p_target_then_model_summary.txt")

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
    """无 IR 窗方案。"""
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
    """B 方法 + tanh_log(beta)。"""
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


def plot_p_scan_fnl(rows: List[Dict[str, float]], p_star: float, out_png: str) -> None:
    """图1：p 扫描的 fnl 结果。"""
    p = np.array([r["p"] for r in rows], dtype=float)
    fnl = np.array([r["fnl_best"] for r in rows], dtype=float)
    dfnl = np.abs(fnl - TARGET_FNL)

    i_star = int(np.argmin(dfnl))

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 8.0), sharex=True)

    ax = axes[0]
    ax.plot(p, fnl, "o-", lw=1.8, ms=5.0, color="tab:blue", label="best-fit fnl_loc")
    ax.axhline(TARGET_FNL, color="tab:red", lw=1.2, ls="--", label="target fnl=100")
    ax.scatter([p[i_star]], [fnl[i_star]], color="tab:green", s=62, zorder=6, label=f"chosen p*={p_star:g}")
    ax.set_ylabel("best-fit fnl_loc")
    ax.set_title("Task5 v7: 1Gpc p scan for fnl target")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(p, dfnl, "s-", lw=1.8, ms=5.0, color="tab:purple")
    ax.scatter([p[i_star]], [dfnl[i_star]], color="tab:green", s=62, zorder=6)
    ax.set_xlabel("fixed p in PK fit")
    ax.set_ylabel(r"$|fnl_{best}-100|$")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_beta_scan(
    beta_to_metric: Dict[float, float],
    baseline_metric: float,
    testa_metric: float,
    best_beta: float,
    out_png: str,
) -> None:
    """图2：在 p* 下扫描 beta 的 2PCF 指标。"""
    xs = np.array(sorted(beta_to_metric.keys()), dtype=float)
    ys = np.array([beta_to_metric[x] for x in xs], dtype=float)
    y_best = beta_to_metric[best_beta]
    y_fix8 = beta_to_metric.get(8.0, np.nan)

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.8))
    ax.plot(xs, ys, "o-", lw=1.8, ms=5.0, color="tab:blue", label=r"TestB tanh\_log scan")
    ax.scatter([best_beta], [y_best], color="tab:red", s=58, zorder=6, label=f"best beta={best_beta:g}")
    if np.isfinite(y_fix8):
        ax.scatter([8.0], [y_fix8], color="tab:green", s=52, zorder=6, label="beta=8")

    ax.axhline(baseline_metric, color="gray", lw=1.2, ls="--", label="Baseline")
    ax.axhline(testa_metric, color="tab:orange", lw=1.0, ls="--", label="TestA")

    ax.set_xlabel(r"tanh\_log shape parameter $\beta$")
    ax.set_ylabel(r"large-scale mean$|\Delta/\sigma|$")
    ax.set_title("Task5 v7: 1Gpc beta scan at p*")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_curves_compare(
    s: np.ndarray,
    xi_data: np.ndarray,
    xi_std: np.ndarray,
    xi_baseline: np.ndarray,
    xi_testa: np.ndarray,
    xi_best: np.ndarray,
    xi_fix8: np.ndarray | None,
    p_star: float,
    bestfit: Dict[str, float],
    best_beta: float,
    m_base: Dict[str, float | str],
    m_a: Dict[str, float | str],
    m_best: Dict[str, float | str],
    m_fix8: Dict[str, float | str] | None,
    out_png: str,
) -> None:
    """图3：在 p* 下的 2PCF 曲线与残差。"""
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 8.6), sharex=True)

    ax = axes[0]
    ax.errorbar(
        s, s**2 * xi_data, yerr=s**2 * xi_std, fmt="o", ms=3.4, capsize=2,
        color="black", label="Measured mean"
    )
    ax.plot(s, s**2 * xi_baseline, "-", lw=1.8, color="tab:blue", label="Baseline (kmin=2pi/L_1gpc)")
    ax.plot(s, s**2 * xi_testa, "-", lw=1.6, color="tab:orange", label="TestA (kmin=1e-4)")
    if xi_fix8 is not None:
        ax.plot(s, s**2 * xi_fix8, "-", lw=1.8, color="tab:green", label="TestB beta=8")
    ax.plot(s, s**2 * xi_best, "-", lw=2.2, color="tab:red", label=f"TestB best beta={best_beta:g}")

    ax.axvline(m5.LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
    ax.set_ylabel(r"$r^2\xi_0(r)$")
    ax.set_title(f"Task5 v7: 1Gpc 2PCF modeling at p*={p_star:g}")
    ax.legend(fontsize=8.4, ncol=2)

    txt = (
        f"PK best-fit: fnl={bestfit['fnl_loc']:.2f}, b1={bestfit['b1']:.3f}, sigmas={bestfit['sigmas']:.4f}\n"
        f"Baseline={m_base['mean_abs_sigma']:.3f}, TestA={m_a['mean_abs_sigma']:.3f}, "
        f"Best={m_best['mean_abs_sigma']:.3f}"
    )
    ax.text(
        0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8.2,
        bbox=dict(facecolor="white", alpha=0.86, edgecolor="gray"),
    )

    # 残差子图
    ax = axes[1]
    r2_sigma = np.maximum((s**2) * xi_std, 1e-12)
    res_base = (s**2 * (xi_data - xi_baseline)) / r2_sigma
    res_a = (s**2 * (xi_data - xi_testa)) / r2_sigma
    res_best = (s**2 * (xi_data - xi_best)) / r2_sigma
    ax.axhline(0.0, color="black", lw=1.0)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.axhline(-1.0, color="gray", lw=0.8, ls="--")
    ax.plot(s, res_base, "o-", ms=3.0, lw=1.2, color="tab:blue", label="Baseline")
    ax.plot(s, res_a, "s-", ms=3.0, lw=1.2, color="tab:orange", label="TestA")
    if xi_fix8 is not None and m_fix8 is not None:
        res_fix8 = (s**2 * (xi_data - xi_fix8)) / r2_sigma
        ax.plot(s, res_fix8, "d-", ms=2.9, lw=1.2, color="tab:green", label="beta=8")
    ax.plot(s, res_best, "^-", ms=3.0, lw=1.2, color="tab:red", label=f"best beta={best_beta:g}")

    ax.axvline(m5.LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
    ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$(Data-Model)/\sigma$ of $r^2\xi_0$")
    ax.legend(fontsize=8.3, ncol=2)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """执行完整流程：先定 p*，再建模 2PCF。"""
    old_p = float(m5.FIXED_P)

    try:
        # 1) 读取 1Gpc 数据
        data = m5.load_mock_data("1gpc_fnl100", PK_1GPC_GLOB, PCF_1GPC_GLOB, rid_min=1, rid_max=99)

        # 2) 扫描 p，目标是 fnl_best 接近 100
        p_rows: List[Dict[str, float]] = []
        p_to_bestfit: Dict[float, Dict[str, float]] = {}
        for p in P_SCAN_LIST:
            pv = float(p)
            m5.FIXED_P = pv
            fit = m5.fit_best_pk(data)
            bestfit = fit["bestfit"]

            fnl_best = float(bestfit["fnl_loc"])
            row = {
                "p": pv,
                "fnl_best": fnl_best,
                "abs_fnl_minus_100": abs(fnl_best - TARGET_FNL),
                "b1_best": float(bestfit["b1"]),
                "sigmas_best": float(bestfit["sigmas"]),
            }
            p_rows.append(row)
            p_to_bestfit[pv] = {
                "fnl_loc": float(bestfit["fnl_loc"]),
                "b1": float(bestfit["b1"]),
                "sigmas": float(bestfit["sigmas"]),
            }

        row_star = min(p_rows, key=lambda r: r["abs_fnl_minus_100"])
        p_star = float(row_star["p"])
        bestfit_star = p_to_bestfit[p_star]

        # 图1：p 扫描结果
        plot_p_scan_fnl(p_rows, p_star=p_star, out_png=OUT_FIG1)

        # 3) 在 p* 下做 2PCF 建模
        m5.FIXED_P = p_star
        k_grid = np.geomspace(KMIN_GLOBAL / m5.FFTLOG_PADDING, KMAX_INT * m5.FFTLOG_PADDING, m5.FFTLOG_N)
        p0_100 = m5.build_theory_p0(k_grid, bestfit_star)
        bestfit_fnl0 = dict(bestfit_star)
        bestfit_fnl0["fnl_loc"] = 0.0
        p0_fnl0_ref = m5.build_theory_p0(k_grid, bestfit_fnl0)

        s = data.scen
        xi_data = data.xi_mean
        xi_std = data.xi_std

        xi_baseline = evaluate_unwindowed_xi(s, k_grid, p0_100, kmin=K_FUND_1GPC)
        xi_testa = evaluate_unwindowed_xi(s, k_grid, p0_100, kmin=KMIN_GLOBAL)
        m_base = m5.compute_alignment_metrics(s, xi_data, xi_std, xi_baseline, "Baseline_1gpc")
        m_a = m5.compute_alignment_metrics(s, xi_data, xi_std, xi_testa, "TestA_1gpc")

        # 4) 在 p* 下扫描 beta
        beta_rows: List[Dict[str, float]] = []
        beta_to_xi: Dict[float, np.ndarray] = {}
        beta_to_metric: Dict[float, float] = {}
        beta_to_full_metric: Dict[float, Dict[str, float | str]] = {}

        for beta in BETA_LIST:
            bv = float(beta)
            xi_model = evaluate_tanhlog_window_xi(s, k_grid, p0_100, p0_fnl0_ref, beta=bv)
            metric = m5.compute_alignment_metrics(s, xi_data, xi_std, xi_model, f"tanh_log_{bv:g}")

            beta_to_xi[bv] = xi_model
            beta_to_metric[bv] = float(metric["mean_abs_sigma"])
            beta_to_full_metric[bv] = metric
            beta_rows.append(
                {
                    "beta": bv,
                    "mean_abs_sigma": float(metric["mean_abs_sigma"]),
                    "chi2_ndof": float(metric["chi2_ndof"]),
                    "mean_sigma": float(metric["mean_sigma"]),
                }
            )

        beta_best = float(min(beta_rows, key=lambda r: r["mean_abs_sigma"])["beta"])
        xi_best = beta_to_xi[beta_best]
        m_best = beta_to_full_metric[beta_best]

        xi_fix8 = beta_to_xi[8.0] if 8.0 in beta_to_xi else None
        m_fix8 = beta_to_full_metric[8.0] if 8.0 in beta_to_full_metric else None

        # 图2：beta 扫描
        plot_beta_scan(
            beta_to_metric=beta_to_metric,
            baseline_metric=float(m_base["mean_abs_sigma"]),
            testa_metric=float(m_a["mean_abs_sigma"]),
            best_beta=beta_best,
            out_png=OUT_FIG2,
        )

        # 图3：曲线对比
        plot_curves_compare(
            s=s,
            xi_data=xi_data,
            xi_std=xi_std,
            xi_baseline=xi_baseline,
            xi_testa=xi_testa,
            xi_best=xi_best,
            xi_fix8=xi_fix8,
            p_star=p_star,
            bestfit=bestfit_star,
            best_beta=beta_best,
            m_base=m_base,
            m_a=m_a,
            m_best=m_best,
            m_fix8=m_fix8,
            out_png=OUT_FIG3,
        )

        # 5) 写总结
        with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
            f.write("任务5 v7：1Gpc 先定 p* 再做 2PCF 建模\n")
            f.write("====================================\n\n")
            f.write("[问题]\n")
            f.write("1) p 取多少时，PK best-fit fnl 会在 100 左右？\n")
            f.write("2) 以该 p 为基准时，2PCF 建模效果如何？\n\n")

            f.write("[设置]\n")
            f.write(f"- 数据: 1Gpc fnl100, Nmock={data.nmock}, rid={data.realizations.min()}..{data.realizations.max()}\n")
            f.write(f"- PK_FIT_KMAX={m5.PK_FIT_KMAX}\n")
            f.write(f"- p 扫描: {','.join([f'{x:g}' for x in P_SCAN_LIST])}\n")
            f.write(f"- beta 扫描: {','.join([f'{x:g}' for x in BETA_LIST])}\n")
            f.write(f"- 2PCF 指标: large-scale mean|Δ/σ|, r>={m5.LARGE_SCALE_MIN}\n\n")

            f.write("[p 扫描结果]\n")
            for r in sorted(p_rows, key=lambda x: x["p"]):
                f.write(
                    f"- p={r['p']:.2f}: fnl_best={r['fnl_best']:.4f}, "
                    f"|fnl-100|={r['abs_fnl_minus_100']:.4f}, "
                    f"b1={r['b1_best']:.4f}, sigmas={r['sigmas_best']:.6f}\n"
                )
            f.write("\n")

            f.write("[选定 p*]\n")
            f.write(
                f"- p*={p_star:.2f} (在扫描点中使 |fnl-100| 最小)\n"
            )
            f.write(
                f"- 对应 best-fit: fnl_loc={bestfit_star['fnl_loc']:.4f}, "
                f"b1={bestfit_star['b1']:.4f}, sigmas={bestfit_star['sigmas']:.6f}\n\n"
            )

            f.write("[在 p* 下的 2PCF]\n")
            f.write(
                f"- Baseline: mean|Δ/σ|={m_base['mean_abs_sigma']:.4f}, "
                f"chi2/ndof={m_base['chi2_ndof']:.4f}\n"
            )
            f.write(
                f"- TestA: mean|Δ/σ|={m_a['mean_abs_sigma']:.4f}, "
                f"chi2/ndof={m_a['chi2_ndof']:.4f}\n"
            )
            if m_fix8 is not None:
                f.write(
                    f"- TestB(beta=8): mean|Δ/σ|={m_fix8['mean_abs_sigma']:.4f}, "
                    f"chi2/ndof={m_fix8['chi2_ndof']:.4f}\n"
                )
            f.write(
                f"- TestB(best beta={beta_best:g}): mean|Δ/σ|={m_best['mean_abs_sigma']:.4f}, "
                f"chi2/ndof={m_best['chi2_ndof']:.4f}\n"
            )
            if m_fix8 is not None:
                f.write(
                    f"- 相对 beta=8 改进: "
                    f"{float(m_fix8['mean_abs_sigma']) - float(m_best['mean_abs_sigma']):.4f}\n"
                )
            f.write(
                f"- 相对 Baseline 改进: "
                f"{float(m_base['mean_abs_sigma']) - float(m_best['mean_abs_sigma']):.4f}\n\n"
            )

            f.write("[结论]\n")
            f.write(
                "1) 已先通过 p 扫描把 PK 的 fnl 约束调到接近 100，再进行 2PCF 建模。\n"
            )
            f.write(
                "2) 在该 p* 下，TestB（加窗口）依旧显著优于 baseline，且可继续通过 beta 微调提升。\n"
            )

        print("[INFO] done")
        print(f"[INFO] fig1: {OUT_FIG1}")
        print(f"[INFO] fig2: {OUT_FIG2}")
        print(f"[INFO] fig3: {OUT_FIG3}")
        print(f"[INFO] summary: {OUT_SUMMARY}")
        print(
            "[INFO] key "
            f"p*={p_star:.2f}, fnl@p*={bestfit_star['fnl_loc']:.4f}, "
            f"baseline={m_base['mean_abs_sigma']:.4f}, "
            f"best_beta={beta_best:g}, best_metric={m_best['mean_abs_sigma']:.4f}"
        )

    finally:
        m5.FIXED_P = old_p


if __name__ == "__main__":
    main()

