#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本名称
--------
task5_v3_p_scan.py

目标
----
对 PNG 模型参数 p 做小扫描（1.0 ~ 1.6），回答两个问题：
1) 当 p 变化时，PK 拟合得到的 best-fit fnl_loc 会不会更接近 100？
2) 与此同时，2PCF 建模（尤其大尺度）会不会更好？

执行逻辑
--------
1) 复用 mission5 v2 的核心函数（数据读取、PK 拟合、FFTLog、窗口法、指标计算）。
2) 设定 p 扫描列表（默认 1.0, 1.1, ..., 1.6）。
3) 对每个 p：
   - 固定该 p，重新拟合 PK，得到 best-fit(fnl_loc, b1, sigmas)。
   - 计算 baseline 的 2PCF 指标（kmin=2pi/L）。
   - 在多窗口集合中找 TestB 最优窗口，并记录对应 2PCF 指标。
4) 输出：
   - 主结果表（每个 p 一行）
   - 全窗口明细表（每个 p * 每个窗口参数 一行）
   - 图1：best-fit fnl_loc vs p（标注 fnl=100 目标线）
   - 图2：2PCF 指标 vs p（baseline 与 TestB 最优）
   - 文本总结：最接近 fnl=100 的 p、2PCF 最优的 p、两者折中建议。

说明
----
- 该脚本默认沿用 mission5 v2 的其它设置（数据范围 2..80、PK_FIT_KMAX=0.08、窗口库等）。
- 不安装新包，需在 desilike 环境运行。
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =====================
# 0) 路径与参数区
# =====================

MISSION5_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission5_log"
sys.path.insert(0, MISSION5_DIR)

# 复用 mission5 v2 脚本
import task5_ir_window_solution_3gpc_multitype as m5

# p 扫描区间（小扫描）
P_SCAN_LIST = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

# 目标 fnl（当前样本为 fnl=100）
TARGET_FNL = 100.0

# 输出文件
OUT_MAIN_CSV = os.path.join(MISSION5_DIR, "task5_v3_p_scan_results.csv")
OUT_DETAIL_CSV = os.path.join(MISSION5_DIR, "task5_v3_p_scan_window_details.csv")
OUT_FIG_FNL = os.path.join(MISSION5_DIR, "task5_v3_p_scan_fnl_vs_p.png")
OUT_FIG_METRIC = os.path.join(MISSION5_DIR, "task5_v3_p_scan_metric_vs_p.png")
OUT_SUMMARY_TXT = os.path.join(MISSION5_DIR, "task5_v3_p_scan_summary.txt")
OUT_LOG_TXT = os.path.join(MISSION5_DIR, "task5_v3_p_scan_log.txt")

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"


# =====================
# 1) 工具函数
# =====================

def write_rows_csv(rows: List[Dict[str, object]], out_csv: str) -> None:
    """把字典行写成 CSV，自动并集字段。"""
    if not rows:
        raise ValueError(f"rows 为空: {out_csv}")

    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    norm_rows = [{k: row.get(k, "") for k in keys} for row in rows]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(norm_rows)


def plot_fnl_vs_p(main_rows: List[Dict[str, object]], out_png: str) -> None:
    """
    绘制 best-fit fnl_loc 随 p 的变化图。

    上面板：fnl_loc vs p，并画目标线 fnl=100。
    下面板：|fnl_loc-100| vs p。
    """
    p = np.array([float(r["p"]) for r in main_rows])
    fnl = np.array([float(r["bestfit_fnl_loc"]) for r in main_rows])
    dfnl = np.abs(fnl - TARGET_FNL)

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 8.2), sharex=True)

    ax = axes[0]
    ax.plot(p, fnl, "o-", lw=1.8, ms=5, color="tab:blue", label="best-fit fnl_loc")
    ax.axhline(TARGET_FNL, color="tab:red", lw=1.2, ls="--", label="target fnl=100")
    ax.set_ylabel("best-fit fnl_loc")
    ax.set_title("Task5 v3: best-fit fnl_loc vs p")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(p, dfnl, "s-", lw=1.8, ms=5, color="tab:purple")
    ax.set_xlabel("fixed p in PK fit")
    ax.set_ylabel(r"$|fnl_{best}-100|$")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_p(main_rows: List[Dict[str, object]], out_png: str) -> None:
    """
    绘制 2PCF 建模指标随 p 的变化图。

    指标采用大尺度 mean|Δ/σ|（越小越好）：
    - Baseline（kmin=2pi/L）
    - TestB 最优窗口
    """
    p = np.array([float(r["p"]) for r in main_rows])
    m_base = np.array([float(r["baseline_mean_abs_sigma"]) for r in main_rows])
    m_bwin = np.array([float(r["testb_best_mean_abs_sigma"]) for r in main_rows])

    fig, ax = plt.subplots(1, 1, figsize=(8.6, 6.4))
    ax.plot(p, m_base, "o-", lw=1.8, ms=5, color="tab:blue", label="Baseline")
    ax.plot(p, m_bwin, "s-", lw=1.8, ms=5, color="tab:red", label="TestB best-window")
    ax.set_xlabel("fixed p in PK fit")
    ax.set_ylabel(r"large-scale mean$|\Delta/\sigma|$ of $r^2\xi$")
    ax.set_title("Task5 v3: 2PCF metric vs p")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# =====================
# 2) 主流程
# =====================

def main() -> None:
    """执行 p 扫描并输出结果。"""
    logs: List[str] = []
    logs.append("[INFO] ===== Task5 v3 p-scan start =====")
    logs.append(f"[INFO] p scan list = {P_SCAN_LIST}")
    logs.append(f"[INFO] data realization range = [{m5.RID_MIN}, {m5.RID_MAX}]")
    logs.append(f"[INFO] PK_FIT_KMAX = {m5.PK_FIT_KMAX}")

    # 读取 fnl100 数据（用于本次问题）
    data100 = m5.load_mock_data("fnl100", m5.PK_FNL100_GLOB, m5.PCF_FNL100_GLOB, m5.RID_MIN, m5.RID_MAX)
    logs.append(
        f"[INFO] data loaded: fnl100 Nmock={data100.nmock}, "
        f"rid={data100.realizations.min()}..{data100.realizations.max()}"
    )

    # FFTLog k 网格固定一次，减少重复开销
    k_grid = np.geomspace(m5.KMIN_GLOBAL / m5.FFTLOG_PADDING, m5.KMAX_INT * m5.FFTLOG_PADDING, m5.FFTLOG_N)

    # 记录原始全局 p，结束后恢复
    original_fixed_p = float(m5.FIXED_P)

    main_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    try:
        for pval in P_SCAN_LIST:
            pval = float(pval)
            logs.append(f"[INFO] ---- scanning p={pval:.2f} ----")

            # 固定 p 后重新拟合 PK
            m5.FIXED_P = pval
            fit = m5.fit_best_pk(data100)
            bestfit = fit["bestfit"]
            fnl_best = float(bestfit["fnl_loc"])
            b1_best = float(bestfit["b1"])
            sig_best = float(bestfit["sigmas"])

            # 在高分辨率 k 网格评估 P0
            p0_100 = m5.build_theory_p0(k_grid, bestfit)
            bestfit_fnl0 = dict(bestfit)
            bestfit_fnl0["fnl_loc"] = 0.0
            p0_fnl0_ref = m5.build_theory_p0(k_grid, bestfit_fnl0)

            # baseline 指标
            xi_baseline = m5.evaluate_unwindowed_with_kmin(
                s_data=data100.scen,
                k_grid=k_grid,
                p0_base=p0_100,
                kmin=m5.K_FUND,
                tag=f"baseline_p{pval:.2f}",
            )
            baseline_metric = m5.compute_alignment_metrics(
                s=data100.scen,
                xi_data=data100.xi_mean,
                xi_std=data100.xi_std,
                xi_model=xi_baseline,
                tag=f"Baseline_p{pval:.2f}",
            )

            # TestB 多窗口扫描，找最优
            best_row = None
            for wtype, params in m5.TESTB_WINDOW_SCANS:
                for wparam in params:
                    wparam = float(wparam)
                    xi_model = m5.evaluate_testb_single_window(
                        s_data=data100.scen,
                        k_grid=k_grid,
                        p0_fnl100=p0_100,
                        p0_fnl0_same_other_params=p0_fnl0_ref,
                        window_type=wtype,
                        window_param=wparam,
                        kmin_global=m5.KMIN_GLOBAL,
                        use_png_only_window=m5.TESTB_USE_PNG_ONLY_WINDOW,
                    )
                    metric = m5.compute_alignment_metrics(
                        s=data100.scen,
                        xi_data=data100.xi_mean,
                        xi_std=data100.xi_std,
                        xi_model=xi_model,
                        tag=f"TestB_{wtype}_{wparam:g}_p{pval:.2f}",
                    )

                    row = {
                        "p": pval,
                        "bestfit_fnl_loc": fnl_best,
                        "bestfit_b1": b1_best,
                        "bestfit_sigmas": sig_best,
                        "window_type": wtype,
                        "window_param": wparam,
                        **metric,
                    }
                    detail_rows.append(row)

                    if (best_row is None) or (float(metric["mean_abs_sigma"]) < float(best_row["mean_abs_sigma"])):
                        best_row = row

            if best_row is None:
                raise RuntimeError(f"p={pval} 时未找到任何窗口结果")

            main_rows.append(
                {
                    "p": pval,
                    "bestfit_fnl_loc": fnl_best,
                    "abs_fnl_minus_100": abs(fnl_best - TARGET_FNL),
                    "bestfit_b1": b1_best,
                    "bestfit_sigmas": sig_best,
                    "baseline_mean_abs_sigma": float(baseline_metric["mean_abs_sigma"]),
                    "baseline_chi2_ndof": float(baseline_metric["chi2_ndof"]),
                    "baseline_mean_sigma": float(baseline_metric["mean_sigma"]),
                    "testb_best_window_type": best_row["window_type"],
                    "testb_best_window_param": best_row["window_param"],
                    "testb_best_mean_abs_sigma": float(best_row["mean_abs_sigma"]),
                    "testb_best_chi2_ndof": float(best_row["chi2_ndof"]),
                    "testb_best_mean_sigma": float(best_row["mean_sigma"]),
                    "improve_over_baseline": float(baseline_metric["mean_abs_sigma"]) - float(best_row["mean_abs_sigma"]),
                }
            )

            logs.append(
                f"[INFO] p={pval:.2f}: fnl_best={fnl_best:.3f} (|Δ100|={abs(fnl_best-TARGET_FNL):.3f}), "
                f"baseline={baseline_metric['mean_abs_sigma']:.4f}, "
                f"testBbest={best_row['mean_abs_sigma']:.4f} ({best_row['window_type']}, {best_row['window_param']})"
            )

    finally:
        # 恢复默认全局 p，避免影响其它脚本
        m5.FIXED_P = original_fixed_p

    # 保存表格
    write_rows_csv(main_rows, OUT_MAIN_CSV)
    write_rows_csv(detail_rows, OUT_DETAIL_CSV)

    # 画图
    plot_fnl_vs_p(main_rows, OUT_FIG_FNL)
    plot_metric_vs_p(main_rows, OUT_FIG_METRIC)

    # 选出关键 p
    row_closest = min(main_rows, key=lambda r: float(r["abs_fnl_minus_100"]))
    row_best2pcf = min(main_rows, key=lambda r: float(r["testb_best_mean_abs_sigma"]))

    # 一个简单折中分数：
    # score = rank(|fnl-100|) + rank(testB metric)
    # 分数越小越好
    idx_fnl = np.argsort([float(r["abs_fnl_minus_100"]) for r in main_rows])
    idx_xi = np.argsort([float(r["testb_best_mean_abs_sigma"]) for r in main_rows])
    rank_fnl = np.empty(len(main_rows), dtype=int)
    rank_xi = np.empty(len(main_rows), dtype=int)
    rank_fnl[idx_fnl] = np.arange(len(main_rows))
    rank_xi[idx_xi] = np.arange(len(main_rows))

    score_rows = []
    for i, r in enumerate(main_rows):
        score_rows.append((int(rank_fnl[i] + rank_xi[i]), i))
    best_compromise = main_rows[min(score_rows)[1]]

    # 输出总结
    with open(OUT_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("任务5 v3：固定 p 扫描总结\n")
        f.write("==========================\n")
        f.write(f"扫描 p 列表: {P_SCAN_LIST}\n")
        f.write(f"数据范围: realization={m5.RID_MIN}..{m5.RID_MAX}\n")
        f.write(f"PK_FIT_KMAX={m5.PK_FIT_KMAX}\n")
        f.write(f"Baseline 定义: kmin=2pi/L={m5.K_FUND:.6e}\n")
        f.write(f"2PCF 指标: large-scale mean|Δ/σ|, r>={m5.LARGE_SCALE_MIN}\n\n")

        f.write("每个 p 的核心结果：\n")
        for r in main_rows:
            f.write(
                f"- p={r['p']:.2f}: fnl_best={r['bestfit_fnl_loc']:.3f}, "
                f"|fnl-100|={r['abs_fnl_minus_100']:.3f}, "
                f"baseline={r['baseline_mean_abs_sigma']:.4f}, "
                f"testBbest={r['testb_best_mean_abs_sigma']:.4f} "
                f"({r['testb_best_window_type']},{r['testb_best_window_param']})\n"
            )

        f.write("\n关键信息：\n")
        f.write(
            f"1) fnl 最接近 100 的 p: {row_closest['p']:.2f} "
            f"(fnl_best={row_closest['bestfit_fnl_loc']:.3f}, "
            f"|Δ|={row_closest['abs_fnl_minus_100']:.3f})\n"
        )
        f.write(
            f"2) 2PCF 最优（TestB best-window）的 p: {row_best2pcf['p']:.2f} "
            f"(metric={row_best2pcf['testb_best_mean_abs_sigma']:.4f})\n"
        )
        f.write(
            f"3) 折中建议 p: {best_compromise['p']:.2f} "
            f"(fnl偏差={best_compromise['abs_fnl_minus_100']:.3f}, "
            f"2PCF metric={best_compromise['testb_best_mean_abs_sigma']:.4f})\n"
        )

    # 保存扫描日志
    with open(OUT_LOG_TXT, "w", encoding="utf-8") as f:
        for line in logs:
            f.write(line + "\n")

    print("[INFO] ===== Task5 v3 p-scan done =====")
    print(f"[INFO] saved main csv:    {OUT_MAIN_CSV}")
    print(f"[INFO] saved detail csv:  {OUT_DETAIL_CSV}")
    print(f"[INFO] saved fig fnl:     {OUT_FIG_FNL}")
    print(f"[INFO] saved fig metric:  {OUT_FIG_METRIC}")
    print(f"[INFO] saved summary txt: {OUT_SUMMARY_TXT}")
    print(f"[INFO] saved log txt:     {OUT_LOG_TXT}")


if __name__ == "__main__":
    main()
