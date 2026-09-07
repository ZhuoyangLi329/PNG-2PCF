#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本名称
--------
task5_v4_window_param_sensitivity.py

目的
----
验证“B 方法直接加窗口（对总 P0 乘窗口）”对窗口形状参数是否敏感。

核心问题
--------
在固定同一套 PK best-fit 条件下，当窗口类型/参数改变时：
1) 2PCF 与数据匹配指标（large-scale mean|Δ/σ|）变化是否很小？
2) 模型曲线本身（r^2 xi）在大尺度的变化是否明显？

代码大纲（执行逻辑）
--------------------
第0部分：参数区
- 固定 p（默认 1.5），固定数据范围（沿用 mission5 v2：2..80），
  并启用“直接加窗口”（use_png_only_window=False）。

第1部分：读取数据 + PK best-fit
- 读取 fnl100 的 pk/pcf
- 固定 p 后拟合 best-fit PK 参数

第2部分：窗口扫描（直接加窗口）
- 对每个窗口家族（exp_power / rational / tanh_log / logcos）
  和对应参数列表，计算 TestB 曲线和匹配指标。

第3部分：敏感性统计
- 每个家族输出：
  - metric 最小值/最大值/相对波动
  - 曲线对“家族最佳参数曲线”的 RMS 差（单位：sigma）

第4部分：出图
- 图1：每个窗口家族的 metric vs 参数
- 图2：全局最优窗口家族中，不同参数对应的 r^2xi 曲线与相对最佳曲线偏差

第5部分：总结文本
- 写详细 summary.txt，给出是否“参数不敏感”的定量结论。
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

# 固定 p（可通过环境变量覆盖）
P_FIXED = float(os.getenv("TASK5_V4_P", "1.5"))

# 是否直接对总 P0 乘窗口（你当前关心的“直接加窗口”）
USE_PNG_ONLY_WINDOW = False

# 输出文件
OUT_FIG1 = os.path.join(MISSION5_DIR, "task5_v4_window_metric_scan_directP.png")
OUT_FIG2 = os.path.join(MISSION5_DIR, "task5_v4_window_curve_spread_directP.png")
OUT_SUMMARY = os.path.join(MISSION5_DIR, "task5_v4_window_sensitivity_summary.txt")

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"


# =====================
# 1) 小工具函数
# =====================

def family_sensitivity_stats(
    s: np.ndarray,
    xi_std: np.ndarray,
    family_curves: Dict[float, np.ndarray],
    best_param: float,
) -> Dict[str, float]:
    """
    计算某一窗口家族的“参数敏感性”统计。

    指标定义
    --------
    - curve_rms_vs_best_sigma:
      所有参数曲线相对“家族最佳参数曲线”的 RMS 差（单位 sigma）。
    - curve_max_vs_best_sigma:
      所有参数曲线相对“家族最佳参数曲线”的最大偏差（单位 sigma）。

    参数
    ----
    s : np.ndarray
        r 网格。
    xi_std : np.ndarray
        测量 xi 标准差。
    family_curves : Dict[param, xi_model]
        一个家族内各参数对应的模型曲线。
    best_param : float
        该家族最优参数。

    返回
    ----
    Dict[str, float]
        敏感性统计字典。
    """
    mask = s >= m5.LARGE_SCALE_MIN
    r2_sigma = np.maximum((s**2) * xi_std, 1e-12)

    best_curve = family_curves[best_param]
    best_r2 = (s**2) * best_curve

    all_norm = []
    all_abs = []
    for p, xi in family_curves.items():
        dr2 = (s**2) * xi - best_r2
        norm = dr2[mask] / r2_sigma[mask]
        all_norm.append(norm)
        all_abs.append(np.abs(norm))

    arr = np.concatenate(all_norm)
    arr_abs = np.concatenate(all_abs)
    return {
        "curve_rms_vs_best_sigma": float(np.sqrt(np.mean(arr**2))),
        "curve_max_vs_best_sigma": float(np.max(arr_abs)),
    }


# =====================
# 2) 主流程
# =====================

def main() -> None:
    """执行窗口参数敏感性调研。"""

    # 保存并临时修改全局 p
    old_p = float(m5.FIXED_P)
    m5.FIXED_P = P_FIXED

    try:
        # 1) 读取数据并拟合 PK
        data = m5.load_mock_data("fnl100", m5.PK_FNL100_GLOB, m5.PCF_FNL100_GLOB, m5.RID_MIN, m5.RID_MAX)
        fit = m5.fit_best_pk(data)
        bestfit = fit["bestfit"]

        # 2) 构建理论 P0
        k_grid = np.geomspace(m5.KMIN_GLOBAL / m5.FFTLOG_PADDING, m5.KMAX_INT * m5.FFTLOG_PADDING, m5.FFTLOG_N)
        p0_100 = m5.build_theory_p0(k_grid, bestfit)
        bestfit_fnl0 = dict(bestfit)
        bestfit_fnl0["fnl_loc"] = 0.0
        p0_fnl0_ref = m5.build_theory_p0(k_grid, bestfit_fnl0)

        s = data.scen
        xi_data = data.xi_mean
        xi_std = data.xi_std

        # 3) baseline 作为参照
        xi_baseline = m5.evaluate_unwindowed_with_kmin(
            s_data=s,
            k_grid=k_grid,
            p0_base=p0_100,
            kmin=m5.K_FUND,
            tag="Baseline",
        )
        baseline_metric = m5.compute_alignment_metrics(s, xi_data, xi_std, xi_baseline, "baseline")

        # 4) 扫描所有窗口家族
        scan_rows: List[Dict[str, float | str]] = []
        family_to_curves: Dict[str, Dict[float, np.ndarray]] = {}
        family_to_metric: Dict[str, Dict[float, float]] = {}

        for wtype, params in m5.TESTB_WINDOW_SCANS:
            family_to_curves[wtype] = {}
            family_to_metric[wtype] = {}

            for p in params:
                p = float(p)
                xi_model = m5.evaluate_testb_single_window(
                    s_data=s,
                    k_grid=k_grid,
                    p0_fnl100=p0_100,
                    p0_fnl0_same_other_params=p0_fnl0_ref,
                    window_type=wtype,
                    window_param=p,
                    kmin_global=m5.KMIN_GLOBAL,
                    use_png_only_window=USE_PNG_ONLY_WINDOW,
                )

                metric = m5.compute_alignment_metrics(s, xi_data, xi_std, xi_model, f"{wtype}_{p:g}")
                family_to_curves[wtype][p] = xi_model
                family_to_metric[wtype][p] = float(metric["mean_abs_sigma"])

                scan_rows.append(
                    {
                        "window_type": wtype,
                        "window_param": p,
                        "mean_abs_sigma": float(metric["mean_abs_sigma"]),
                        "chi2_ndof": float(metric["chi2_ndof"]),
                        "mean_sigma": float(metric["mean_sigma"]),
                    }
                )

        # 5) 找全局最佳窗口
        best_row = min(scan_rows, key=lambda r: float(r["mean_abs_sigma"]))
        best_wtype = str(best_row["window_type"])
        best_wparam = float(best_row["window_param"])

        # 6) 计算家族敏感性统计
        family_stats: List[Dict[str, float | str]] = []
        for wtype, param_metric in family_to_metric.items():
            params_sorted = sorted(param_metric.keys())
            values = np.array([param_metric[p] for p in params_sorted], dtype=float)
            p_best = float(params_sorted[int(np.argmin(values))])

            sens = family_sensitivity_stats(
                s=s,
                xi_std=xi_std,
                family_curves=family_to_curves[wtype],
                best_param=p_best,
            )

            family_stats.append(
                {
                    "window_type": wtype,
                    "best_param": p_best,
                    "metric_min": float(np.min(values)),
                    "metric_max": float(np.max(values)),
                    "metric_rel_var_percent": float((np.max(values) - np.min(values)) / max(np.min(values), 1e-30) * 100.0),
                    **sens,
                }
            )

        # -----------------
        # 7) 出图1：metric扫描
        # -----------------
        fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.6), sharey=True)
        axes = axes.ravel()

        type_order = [w for w, _ in m5.TESTB_WINDOW_SCANS]
        for ax, wtype in zip(axes, type_order):
            pm = family_to_metric[wtype]
            xs = np.array(sorted(pm.keys()), dtype=float)
            ys = np.array([pm[x] for x in xs], dtype=float)

            ax.plot(xs, ys, "o-", lw=1.8, ms=4.8, color="tab:blue")
            i_best = int(np.argmin(ys))
            ax.scatter([xs[i_best]], [ys[i_best]], color="tab:red", s=42, zorder=5, label="family best")
            ax.set_title(wtype)
            ax.set_xlabel("shape parameter")
            ax.set_ylabel(r"large-scale mean$|\Delta/\sigma|$")
            ax.legend(fontsize=8)

        fig.suptitle(f"Task5 v4 (direct-P window): metric sensitivity scan @ fixed p={P_FIXED}", y=0.995)
        fig.tight_layout()
        fig.savefig(OUT_FIG1, bbox_inches="tight")
        plt.close(fig)

        # -----------------
        # 8) 出图2：全局最优家族的曲线离散
        # -----------------
        fam_curves = family_to_curves[best_wtype]
        fam_params = sorted(fam_curves.keys())
        best_curve = fam_curves[best_wparam]

        fig, axes = plt.subplots(2, 1, figsize=(9.4, 8.4), sharex=True)

        # 上图：曲线本体
        ax = axes[0]
        ax.errorbar(s, s**2 * xi_data, yerr=s**2 * xi_std, fmt="o", ms=3.0, capsize=2,
                    color="black", label="Measured mean")

        cmap = plt.cm.viridis(np.linspace(0.15, 0.95, len(fam_params)))
        for c, p in zip(cmap, fam_params):
            lw = 2.2 if np.isclose(p, best_wparam) else 1.4
            lab = f"param={p:g}"
            if np.isclose(p, best_wparam):
                lab += " (best)"
            ax.plot(s, s**2 * fam_curves[p], "-", color=c, lw=lw, label=lab)

        ax.axvline(m5.LARGE_SCALE_MIN, color="gray", ls="--", lw=1.0)
        ax.set_ylabel(r"$r^2\xi_0(r)$")
        ax.set_title(f"Best family = {best_wtype} (direct-P window), p={P_FIXED}")
        ax.legend(fontsize=8, ncol=2)

        # 下图：相对最佳参数曲线偏差（sigma单位）
        ax = axes[1]
        r2_sigma = np.maximum((s**2) * xi_std, 1e-12)
        for c, p in zip(cmap, fam_params):
            dr = (s**2) * (fam_curves[p] - best_curve) / r2_sigma
            lw = 2.0 if np.isclose(p, best_wparam) else 1.2
            lab = f"param={p:g}"
            if np.isclose(p, best_wparam):
                lab += " (best)"
            ax.plot(s, dr, "-", color=c, lw=lw, label=lab)

        ax.axhline(0.0, color="black", lw=1.0)
        ax.axhline(0.2, color="gray", lw=0.8, ls="--")
        ax.axhline(-0.2, color="gray", lw=0.8, ls="--")
        ax.axvline(m5.LARGE_SCALE_MIN, color="gray", ls="--", lw=1.0)
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$(Model-Model_{best})/\sigma_{data}$")

        fig.tight_layout()
        fig.savefig(OUT_FIG2, bbox_inches="tight")
        plt.close(fig)

        # -----------------
        # 9) 写详细总结
        # -----------------
        with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
            f.write("任务5 v4：B方法（直接加窗口）参数敏感性验证\n")
            f.write("========================================\n\n")
            f.write("[设置]\n")
            f.write(f"- 固定 p = {P_FIXED}\n")
            f.write(f"- 数据范围: realization={m5.RID_MIN}..{m5.RID_MAX}, Nmock={data.nmock}\n")
            f.write(f"- PK_FIT_KMAX = {m5.PK_FIT_KMAX}\n")
            f.write(f"- Baseline: kmin=2pi/L={m5.K_FUND:.6e}, kmax={m5.KMAX_INT}\n")
            f.write(f"- TestB 模式: 直接加窗口（对总 P0 乘 W(k)）\n")
            f.write(f"- 2PCF 指标: large-scale mean|Δ/σ|, r>={m5.LARGE_SCALE_MIN}\n\n")

            f.write("[当前 best-fit PK 参数]\n")
            f.write(
                f"- fnl_loc={bestfit['fnl_loc']:.3f}, b1={bestfit['b1']:.3f}, sigmas={bestfit['sigmas']:.4f}\n\n"
            )

            f.write("[基线对比]\n")
            f.write(
                f"- Baseline mean|Δ/σ| = {baseline_metric['mean_abs_sigma']:.4f}, "
                f"chi2/ndof = {baseline_metric['chi2_ndof']:.4f}, "
                f"mean(Δ/σ) = {baseline_metric['mean_sigma']:.4f}\n"
            )
            f.write(
                f"- TestB 全局最优 = {best_wtype}(param={best_wparam:g}), "
                f"mean|Δ/σ| = {best_row['mean_abs_sigma']:.4f}, "
                f"chi2/ndof = {best_row['chi2_ndof']:.4f}\n"
            )
            f.write(
                f"- 相对 Baseline 改进量（mean|Δ/σ|）= "
                f"{baseline_metric['mean_abs_sigma'] - best_row['mean_abs_sigma']:.4f}\n\n"
            )

            f.write("[窗口家族参数敏感性统计]\n")
            f.write("说明: metric_rel_var_percent 越小，表示该家族对参数越不敏感。\n")
            f.write("      curve_rms_vs_best_sigma 越小，表示曲线对参数扰动越稳。\n\n")

            for st in sorted(family_stats, key=lambda x: float(x['metric_min'])):
                f.write(
                    f"- {st['window_type']}: "
                    f"best_param={st['best_param']}, "
                    f"metric_min={st['metric_min']:.4f}, metric_max={st['metric_max']:.4f}, "
                    f"metric_rel_var={st['metric_rel_var_percent']:.2f}%, "
                    f"curve_rms_vs_best_sigma={st['curve_rms_vs_best_sigma']:.4f}, "
                    f"curve_max_vs_best_sigma={st['curve_max_vs_best_sigma']:.4f}\n"
                )

            # 给出简短结论
            st_best = min(family_stats, key=lambda x: float(x['metric_min']))
            st_robust = min(family_stats, key=lambda x: float(x['metric_rel_var_percent']))

            f.write("\n[结论]\n")
            f.write(
                f"1) 在直接加窗口模式下，全局最优家族是 {best_wtype}(param={best_wparam:g})。\n"
            )
            f.write(
                f"2) 从“参数不敏感”角度看，最稳健家族是 {st_robust['window_type']} "
                f"（metric 相对波动 {st_robust['metric_rel_var_percent']:.2f}%）。\n"
            )
            f.write(
                "3) 若你更看重“性能最优+参数稳健”折中，优先建议查看图2中最佳家族的曲线离散，"
                "如果离散带在大尺度内普遍 <~0.2σ，可认为参数不敏感。\n"
            )

        print("[INFO] done")
        print(f"[INFO] fig1: {OUT_FIG1}")
        print(f"[INFO] fig2: {OUT_FIG2}")
        print(f"[INFO] summary: {OUT_SUMMARY}")

    finally:
        # 恢复全局 p
        m5.FIXED_P = old_p


if __name__ == "__main__":
    main()
