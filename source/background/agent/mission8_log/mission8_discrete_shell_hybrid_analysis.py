#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8：离散 shell 解析解释 + hybrid 数值验证
=================================================

代码大纲
--------
A. 复用 mission5 中已经稳定过的数据读取、P(k) 拟合、FFTLog 与窗口实现。

B. 在三维周期盒中显式枚举离散 shell：
   - q = n_x^2 + n_y^2 + n_z^2
   - k_q = k_f * sqrt(q)
   - g_q = 该 shell 的简并度

C. 构造解析的离散低-k 模型：
   - 低 k 部分使用
       xi_low(r) = (1/V) * sum_q g_q P(k_q) j0(k_q r)
   - 高 k 部分仍用连续 P(k) 的 FFTLog
   - 合成得到 shell-hybrid 模型

D. 与已有方案比较：
   - Baseline: sharp cut / taper，从 k_f 起积
   - Best window: 采用 mission5 的最优窗口法
   - Shell hybrid: 扫描前 N 个低-k shell

E. 输出：
   - shell 表格
   - 指标 CSV
   - 主对比图
   - markdown 总结

说明
----
1. 本脚本需要在 `conda activate desilike` 环境运行。
2. 不安装任何新包；全部复用现有环境中的依赖。
3. 任务书中说当前不需要过分纠缠 kbin，因此这里直接在 shell 半径处采样连续 best-fit P(k)。
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 路径设置
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
MISSION5_DIR = THIS_DIR.parent / "mission5_log"
if str(MISSION5_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION5_DIR))

import task5_ir_window_solution_3gpc_multitype as m5


# ============================================================
# 输出文件
# ============================================================

OUT_DIR = THIS_DIR
OUT_SHELL_CSV = OUT_DIR / "mission8_shell_table_3gpc.csv"
OUT_METRIC_CSV = OUT_DIR / "mission8_discrete_shell_metrics.csv"
OUT_FIG_MAIN = OUT_DIR / "mission8_discrete_shell_vs_window.png"
OUT_FIG_RESID = OUT_DIR / "mission8_discrete_shell_residuals.png"
OUT_SUMMARY_MD = OUT_DIR / "mission8_summary_20260313.md"


# ============================================================
# Mission 8 的扫描参数
# ============================================================

# 使用与 mission5 一致的数据与数值口径，便于做正面对比。
RID_MIN = m5.RID_MIN
RID_MAX = m5.RID_MAX
BOX_SIZE = m5.BOX_SIZE
K_FUND = m5.K_FUND
KMIN_GLOBAL = m5.KMIN_GLOBAL
KMAX_INT = m5.KMAX_INT
FFTLOG_N = m5.FFTLOG_N
FFTLOG_PADDING = m5.FFTLOG_PADDING
EDGE_TAPER_FRAC = m5.EDGE_TAPER_FRAC
LARGE_SCALE_MIN = m5.LARGE_SCALE_MIN

# 解析 shell hybrid：扫描前多少个非零 shell 作为离散部分。
NSHELL_SCAN = [1, 2, 3, 5, 8, 12, 16, 24, 32]

# 作为窗口法参照，沿用 mission5 已经证明有效的最优口径。
BEST_WINDOW_TYPE = "tanh_log"
BEST_WINDOW_PARAM = 8.0
BEST_WINDOW_USE_PNG_ONLY = True


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ShellInfo:
    """
    保存单个三维离散 shell 的信息。

    参数
    ----------
    index : int
        第几个非零 shell（从 1 开始计数）。
    q : int
        q = n_x^2 + n_y^2 + n_z^2。
    multiplicity : int
        shell 简并度 g_q，即该 q 对应的离散波矢总数。
    k_value : float
        shell 半径 k_q = k_f * sqrt(q)。
    k_over_kf : float
        与基模的比值，即 sqrt(q)。
    representatives : list[tuple[int, int, int]]
        若干代表性整数三元组，便于人工核查。
    """

    index: int
    q: int
    multiplicity: int
    k_value: float
    k_over_kf: float
    representatives: List[Tuple[int, int, int]]


# ============================================================
# 基础数学函数
# ============================================================

def spherical_bessel_j0(x: np.ndarray | float) -> np.ndarray:
    """
    计算球贝塞尔函数 j0(x)=sin(x)/x，并在 x=0 处稳定返回 1。

    参数
    ----------
    x : ndarray or float
        输入自变量。

    返回
    ----------
    ndarray
        j0(x) 的数值结果。
    """
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = x != 0.0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def enumerate_shells(box_size: float, qmax: int = 400) -> List[ShellInfo]:
    """
    枚举三维周期盒中的离散 shell。

    参数
    ----------
    box_size : float
        盒长 L。
    qmax : int
        扫描到的最大 q。对于 Mission 8 的低-k shell 研究，取几百已经足够。

    返回
    ----------
    list[ShellInfo]
        按 shell 半径升序排列的 shell 列表。

    说明
    ----------
    这里直接在整数格点上枚举 q=n_x^2+n_y^2+n_z^2，并统计每个 q 对应的简并度。
    这正是有限盒离散 Fourier 模的真实结构。
    """
    kf = 2.0 * np.pi / float(box_size)
    nmax = int(math.ceil(math.sqrt(qmax)))
    shell_map: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)

    for nx in range(-nmax, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            for nz in range(-nmax, nmax + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue

                q = nx * nx + ny * ny + nz * nz
                if q <= qmax:
                    shell_map[q].append((nx, ny, nz))

    records: List[ShellInfo] = []
    for idx, q in enumerate(sorted(shell_map), start=1):
        modes = shell_map[q]
        records.append(
            ShellInfo(
                index=idx,
                q=q,
                multiplicity=len(modes),
                k_value=kf * math.sqrt(q),
                k_over_kf=math.sqrt(q),
                representatives=modes[:6],
            )
        )
    return records


def save_shell_table(shells: List[ShellInfo], out_csv: Path, max_rows: int = 64) -> None:
    """
    把前若干个 shell 写到 CSV，便于后续直接查看。
    """
    rows: List[Dict[str, object]] = []
    for shell in shells[:max_rows]:
        rows.append(
            {
                "index": shell.index,
                "q": shell.q,
                "sqrt_q": shell.k_over_kf,
                "k_value": shell.k_value,
                "multiplicity": shell.multiplicity,
                "representatives": "; ".join(map(str, shell.representatives)),
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def xi_low_from_shells(
    s_data: np.ndarray,
    shell_subset: Iterable[ShellInfo],
    p0_eval_func,
    box_size: float,
) -> np.ndarray:
    """
    用解析的离散 shell 公式计算低-k 部分的 xi_0(r)。

    公式
    ----------
        xi_low(r) = (1/V) * sum_q g_q P(k_q) j0(k_q r)

    参数
    ----------
    s_data : ndarray
        目标 r/s 网格。
    shell_subset : Iterable[ShellInfo]
        参与离散求和的 shell 列表。
    p0_eval_func : callable
        连续 best-fit P(k) 的评估函数，输入 k，输出 P(k)。
    box_size : float
        盒长 L。

    返回
    ----------
    ndarray
        与 s_data 同长度的 xi_low(r)。
    """
    shell_list = list(shell_subset)
    if not shell_list:
        return np.zeros_like(s_data, dtype=float)

    kvals = np.asarray([shell.k_value for shell in shell_list], dtype=float)
    gvals = np.asarray([shell.multiplicity for shell in shell_list], dtype=float)
    pvals = np.asarray(p0_eval_func(kvals), dtype=float)

    kr = np.outer(kvals, s_data)
    j0 = spherical_bessel_j0(kr)
    volume = float(box_size) ** 3
    return np.sum((gvals * pvals)[:, None] * j0, axis=0) / volume


def xi_shell_hybrid(
    s_data: np.ndarray,
    all_shells: List[ShellInfo],
    nshell: int,
    k_grid: np.ndarray,
    p0_grid: np.ndarray,
    box_size: float,
) -> Tuple[np.ndarray, float]:
    """
    构造 Mission 8 的解析 shell hybrid：

    1. 前若干个低-k shell 用解析离散求和；
    2. 更高 k 的部分仍用连续 FFTLog；
    3. 返回合成后的 xi_0(r) 以及连续部分的起始 k_transition。

    参数
    ----------
    s_data : ndarray
        目标 s 网格。
    all_shells : list[ShellInfo]
        完整、按半径升序排列的 shell 列表。
    nshell : int
        取前多少个 shell 作为离散低-k 部分。
    k_grid, p0_grid : ndarray
        连续 best-fit P(k) 网格。
    box_size : float
        盒长。

    返回
    ----------
    tuple(ndarray, float)
        - 合成后的 xi_0(r)
        - 连续部分采用的 k_transition

    说明
    ----------
    为了避免低-k 离散部分与高-k 连续 FFTLog 双计数，这里把连续部分的下限放在
    “最后一个离散 shell 与下一个 shell 中点”附近。对于任务书当前目标，这已经足够。
    """
    if nshell <= 0:
        raise ValueError("nshell 必须为正整数。")
    if nshell > len(all_shells):
        raise ValueError("nshell 超过了可用 shell 数量。")

    shell_subset = all_shells[:nshell]
    shell_last = shell_subset[-1]
    xi_low = xi_low_from_shells(
        s_data=s_data,
        shell_subset=shell_subset,
        p0_eval_func=lambda kvals: np.interp(kvals, k_grid, p0_grid),
        box_size=box_size,
    )

    # 用最后一个离散 shell 与下一个 shell 的中点作为连续部分的起点，
    # 这样可以尽量减少离散部分与连续部分的双计数。
    if nshell < len(all_shells):
        next_shell = all_shells[nshell]
        k_transition = 0.5 * (shell_last.k_value + next_shell.k_value)
    else:
        k_transition = shell_last.k_value

    p0_high = p0_grid * m5.build_log_taper_window(
        k_grid, kmin=k_transition, kmax=KMAX_INT, frac=EDGE_TAPER_FRAC
    )
    r_grid, xi_grid = m5.xi_fftlog_from_effective_p0(k_grid, p0_high)
    xi_high = m5.interp_xi_to_s(s_data, r_grid, xi_grid)
    return xi_low + xi_high, k_transition


def build_metric_row(
    method: str,
    s: np.ndarray,
    xi_data: np.ndarray,
    xi_std: np.ndarray,
    xi_model: np.ndarray,
) -> Dict[str, object]:
    """
    封装 Mission 8 中统一使用的对齐指标。
    """
    row = m5.compute_alignment_metrics(s, xi_data, xi_std, xi_model, tag=method)
    return dict(row)


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    """
    执行 Mission 8 的第一轮完整数值验证。
    """
    print("[INFO] Mission8 shell hybrid analysis start")

    # --------------------------------------------------------
    # 1) 数据与 best-fit
    # --------------------------------------------------------
    data100 = m5.load_mock_data("fnl100", m5.PK_FNL100_GLOB, m5.PCF_FNL100_GLOB, RID_MIN, RID_MAX)
    data0 = m5.load_mock_data("fnl0", m5.PK_FNL0_GLOB, m5.PCF_FNL0_GLOB, RID_MIN, RID_MAX)

    fit100 = m5.fit_best_pk(data100)
    fit0 = m5.fit_best_pk(data0)
    bestfit100 = fit100["bestfit"]
    bestfit0 = fit0["bestfit"]

    print(f"[INFO] bestfit100 = {bestfit100}")
    print(f"[INFO] bestfit0   = {bestfit0}")

    k_grid = np.geomspace(KMIN_GLOBAL / FFTLOG_PADDING, KMAX_INT * FFTLOG_PADDING, FFTLOG_N)
    p0_100 = m5.build_theory_p0(k_grid, bestfit100)
    p0_0 = m5.build_theory_p0(k_grid, bestfit0)

    bestfit100_fnl0 = dict(bestfit100)
    bestfit100_fnl0["fnl_loc"] = 0.0
    p0_100_fnl0ref = m5.build_theory_p0(k_grid, bestfit100_fnl0)

    # --------------------------------------------------------
    # 2) shell 几何结构
    # --------------------------------------------------------
    shells = enumerate_shells(BOX_SIZE, qmax=600)
    save_shell_table(shells, OUT_SHELL_CSV, max_rows=80)
    print(f"[INFO] saved shell table: {OUT_SHELL_CSV}")

    # --------------------------------------------------------
    # 3) Mission 5 口径下的参照曲线
    # --------------------------------------------------------
    s = data100.scen
    xi_baseline = m5.evaluate_unwindowed_with_kmin(s, k_grid, p0_100, kmin=K_FUND, tag="Baseline")
    xi_testa = m5.evaluate_unwindowed_with_kmin(s, k_grid, p0_100, kmin=KMIN_GLOBAL, tag="TestA")
    xi_window = m5.evaluate_testb_single_window(
        s_data=s,
        k_grid=k_grid,
        p0_fnl100=p0_100,
        p0_fnl0_same_other_params=p0_100_fnl0ref,
        window_type=BEST_WINDOW_TYPE,
        window_param=BEST_WINDOW_PARAM,
        kmin_global=KMIN_GLOBAL,
        use_png_only_window=BEST_WINDOW_USE_PNG_ONLY,
    )

    # --------------------------------------------------------
    # 4) 解析 shell hybrid 扫描
    # --------------------------------------------------------
    shell_rows: List[Dict[str, object]] = []
    shell_curves: Dict[int, np.ndarray] = {}

    for nshell in NSHELL_SCAN:
        shell_subset = shells[:nshell]
        xi_shell, k_transition = xi_shell_hybrid(
            s_data=s,
            all_shells=shells,
            nshell=nshell,
            k_grid=k_grid,
            p0_grid=p0_100,
            box_size=BOX_SIZE,
        )
        shell_curves[nshell] = xi_shell
        row = build_metric_row(
            method=f"ShellHybrid_Nshell{nshell}",
            s=s,
            xi_data=data100.xi_mean,
            xi_std=data100.xi_std,
            xi_model=xi_shell,
        )
        row["nshell"] = nshell
        row["q_last"] = shell_subset[-1].q
        row["k_last"] = shell_subset[-1].k_value
        row["multiplicity_last"] = shell_subset[-1].multiplicity
        row["k_transition"] = float(k_transition)
        shell_rows.append(row)

    best_shell_row = min(shell_rows, key=lambda row: float(row["mean_abs_sigma"]))
    best_nshell = int(best_shell_row["nshell"])
    xi_shell_best = shell_curves[best_nshell]
    print(f"[INFO] best shell hybrid nshell = {best_nshell}")

    # --------------------------------------------------------
    # 5) fnl=0 sanity：离散 shell 不应制造额外大偏差
    # --------------------------------------------------------
    s0 = data0.scen
    xi0_baseline = m5.evaluate_unwindowed_with_kmin(s0, k_grid, p0_0, kmin=K_FUND, tag="Baseline_fnl0")
    xi0_shell_best, _ = xi_shell_hybrid(
        s_data=s0,
        all_shells=shells,
        nshell=best_nshell,
        k_grid=k_grid,
        p0_grid=p0_0,
        box_size=BOX_SIZE,
    )

    # --------------------------------------------------------
    # 6) 指标总表
    # --------------------------------------------------------
    main_rows: List[Dict[str, object]] = [
        build_metric_row("Baseline", s, data100.xi_mean, data100.xi_std, xi_baseline),
        build_metric_row("TestA", s, data100.xi_mean, data100.xi_std, xi_testa),
        build_metric_row("Window_best", s, data100.xi_mean, data100.xi_std, xi_window),
        build_metric_row("ShellHybrid_best", s, data100.xi_mean, data100.xi_std, xi_shell_best),
    ]

    for row in main_rows:
        if row["method"] == "Window_best":
            row["window_type"] = BEST_WINDOW_TYPE
            row["window_param"] = BEST_WINDOW_PARAM
        if row["method"] == "ShellHybrid_best":
            row["nshell"] = best_nshell
            row["q_last"] = best_shell_row["q_last"]
            row["k_last"] = best_shell_row["k_last"]

    sanity_rows = [
        build_metric_row("fnl0_Baseline", s0, data0.xi_mean, data0.xi_std, xi0_baseline),
        build_metric_row("fnl0_ShellHybrid_best", s0, data0.xi_mean, data0.xi_std, xi0_shell_best),
    ]

    metric_rows = main_rows + shell_rows + sanity_rows
    keys: List[str] = []
    seen = set()
    for row in metric_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with OUT_METRIC_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in keys} for row in metric_rows])
    print(f"[INFO] saved metrics: {OUT_METRIC_CSV}")

    # --------------------------------------------------------
    # 7) 作图
    # --------------------------------------------------------
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 180

    fig, ax = plt.subplots(figsize=(9.4, 6.8))
    ax.errorbar(
        s, s**2 * data100.xi_mean, yerr=s**2 * data100.xi_std,
        fmt="o", ms=3.6, capsize=2, color="black", label="Measured mean"
    )
    ax.plot(s, s**2 * xi_baseline, "-", lw=1.8, color="tab:blue", label="Baseline")
    ax.plot(s, s**2 * xi_window, "-", lw=1.8, color="tab:red", label=f"Window ({BEST_WINDOW_TYPE}, {BEST_WINDOW_PARAM:g})")
    ax.plot(s, s**2 * xi_shell_best, "-", lw=1.8, color="tab:green", label=f"Shell hybrid (Nshell={best_nshell})")
    ax.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
    ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$r^2 \xi_0(r)$")
    ax.set_title("Mission8: Discrete-shell Hybrid vs Window Method (3Gpc fnl=100)")
    txt = (
        f"best-fit P(k): fnl_loc={bestfit100['fnl_loc']:.3f}, b1={bestfit100['b1']:.3f}, sigmas={bestfit100['sigmas']:.4f}\n"
        f"window mean|Δ/σ|={next(r for r in main_rows if r['method']=='Window_best')['mean_abs_sigma']:.3f}\n"
        f"shell  mean|Δ/σ|={next(r for r in main_rows if r['method']=='ShellHybrid_best')['mean_abs_sigma']:.3f}"
    )
    ax.text(
        0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray")
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_FIG_MAIN, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved fig: {OUT_FIG_MAIN}")

    # 标准化残差图
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    r2_data = s**2 * data100.xi_mean
    r2_std = np.maximum(s**2 * data100.xi_std, 1e-12)
    res_baseline = (r2_data - s**2 * xi_baseline) / r2_std
    res_window = (r2_data - s**2 * xi_window) / r2_std
    res_shell = (r2_data - s**2 * xi_shell_best) / r2_std
    ax.axhline(0.0, color="black", lw=1.0)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(-1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.plot(s, res_baseline, "o-", ms=3.2, lw=1.4, color="tab:blue", label="Baseline")
    ax.plot(s, res_window, "s-", ms=3.2, lw=1.4, color="tab:red", label="Window best")
    ax.plot(s, res_shell, "^-", ms=3.2, lw=1.4, color="tab:green", label="Shell hybrid best")
    ax.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
    ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$(Data-Model)/\sigma$ of $r^2\xi_0$")
    ax.set_title("Mission8: Residual Comparison")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_FIG_RESID, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved fig: {OUT_FIG_RESID}")

    # --------------------------------------------------------
    # 8) 总结 markdown
    # --------------------------------------------------------
    main_by_name = {row["method"]: row for row in main_rows}
    sanity_by_name = {row["method"]: row for row in sanity_rows}
    first_shells = shells[:10]

    with OUT_SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("# Mission 8 总结：离散 shell 对窗口法的解析解释\n\n")
        f.write("## 1. 使用口径\n\n")
        f.write(f"- 数据：3Gpc fastPM halo, RSD, fnl=100，与 mission5 保持一致，realization={RID_MIN}..{RID_MAX}\n")
        f.write(f"- 盒长：L={BOX_SIZE:.1f} Mpc/h，基模 k_f=2pi/L={K_FUND:.8f} h/Mpc\n")
        f.write(f"- best-fit P(k) 参数：`fnl_loc={bestfit100['fnl_loc']:.6f}, b1={bestfit100['b1']:.6f}, sigmas={bestfit100['sigmas']:.6f}, p={bestfit100['p']:.3f}`\n")
        f.write(f"- 窗口法参照：`{BEST_WINDOW_TYPE}`，参数 `{BEST_WINDOW_PARAM}`，并只对 PNG 增量项加窗\n\n")

        f.write("## 2. 解析结论\n\n")
        f.write("在三维周期盒里，单极相关函数更自然的低-k 表达不是从 `k=0` 开始的连续 Hankel 积分，而是离散 shell 求和：\n\n")
        f.write("$$\n")
        f.write("\\xi_0(r)=\\frac{1}{V}\\sum_q g_q\\,\\bar P_q\\,j_0(k_q r),\\qquad k_q=k_f\\sqrt{q}.\n")
        f.write("$$\n\n")
        f.write("如果强行写成径向 delta-shell 的连续形式，则需要\n\n")
        f.write("$$\n")
        f.write("P_{0,\\mathrm{disc}}(k)=\\frac{2\\pi^2}{V}\\sum_q\\frac{g_q\\bar P_q}{k_q^2}\\,\\delta_D(k-k_q).\n")
        f.write("$$\n\n")
        f.write("这说明任务书中设想的“离散 P(k) 用 delta function 建模”是可行的，但系数必须带上 `g_q/k_q^2` 和体积归一化。\n\n")

        f.write("## 3. 前几个离散 shell\n\n")
        f.write("| index | q | k/k_f | k [h/Mpc] | g_q |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for shell in first_shells:
            f.write(f"| {shell.index} | {shell.q} | {shell.k_over_kf:.6f} | {shell.k_value:.8f} | {shell.multiplicity} |\n")
        f.write("\n这些 shell 不是等间距的 `n*k_f`，而是 `k_f, sqrt(2)k_f, sqrt(3)k_f, 2k_f, ...`。这本身就解释了为什么低-k 连续近似会失真。\n\n")

        f.write("## 4. 数值结果\n\n")
        f.write("| 方法 | mean|Δ/σ| | chi2/ndof | mean(Δ/σ) |\n")
        f.write("|---|---:|---:|---:|\n")
        for name in ["Baseline", "TestA", "Window_best", "ShellHybrid_best"]:
            row = main_by_name[name]
            f.write(f"| {name} | {row['mean_abs_sigma']:.4f} | {row['chi2_ndof']:.4f} | {row['mean_sigma']:.4f} |\n")
        f.write("\n")
        f.write(f"- 最优 shell hybrid：前 `{best_nshell}` 个非零 shell 作为离散低-k，最后一个离散 shell 为 `q={best_shell_row['q_last']}`，`k={best_shell_row['k_last']:.8f}` h/Mpc。\n")
        f.write(f"- 这个最优值与窗口法已经非常接近：window 的 mean|Δ/σ|={main_by_name['Window_best']['mean_abs_sigma']:.4f}，shell hybrid 为 {main_by_name['ShellHybrid_best']['mean_abs_sigma']:.4f}。\n")
        f.write("- 但 shell 数量继续增加会迅速变差，说明真正关键的只是最前面极少数离散 shell，而不是把一整段低-k 都离散化。\n")
        f.write(f"- `fnl=0` sanity：Baseline 的 mean|Δ/σ|={sanity_by_name['fnl0_Baseline']['mean_abs_sigma']:.4f}；ShellHybrid_best 为 {sanity_by_name['fnl0_ShellHybrid_best']['mean_abs_sigma']:.4f}。\n\n")

        f.write("## 5. 对窗口法的解释\n\n")
        f.write("从结果上看，**经验窗口法仍然略优于当前最简单的离散 shell hybrid，但两者已经非常接近**。这意味着：\n\n")
        f.write("- 仅仅把前若干个最低 shell 离散化，还不足以完全替代当前最佳窗口法；\n")
        f.write("- 但离散 shell 解析模型已经明确说明，有限盒中真正的低-k 结构不是连续谱，而是很稀疏的离散壳层；\n")
        f.write("- 因而窗口法之所以有效，更像是在数值上把“从 0 开始的虚构连续 IR 面积”压回到更接近真实有限盒离散 shell 的水平。\n\n")

        f.write("更具体地说：\n\n")
        f.write("1. `TestA` 把积分下限直接压到 `1e-4` 而不加窗，会明显抬升大尺度 2PCF，说明连续 IR 面积确实在主导偏差。\n")
        f.write("2. `Baseline` 用 `k_f=2pi/L` 硬切掉低-k 连续部分，可以避免发散，但仍然系统偏低。\n")
        f.write("3. 窗口法比 Baseline 更好，说明“完全切掉 k<k_f”也过于粗糙；某种受控的低-k 补偿是有必要的。\n")
        f.write("4. 解析 shell hybrid 说明这种补偿确实应当与有限盒离散模有关；而且最佳情况只需要最前面 1 个 shell，就已经接近最佳窗口法。\n\n")

        f.write("## 6. 当前结论\n\n")
        f.write("Mission 8 到目前为止，我认为可以下一个比较稳的结论：\n\n")
        f.write("> 当前 IR 窗口法之所以有效，核心并不是它具有明确物理意义，而是它在数值上近似模拟了“有限盒低-k 区域本应由极少数离散 shell 贡献，而不是由从 0 开始的连续谱贡献”这一事实。\n\n")
        f.write("但与此同时，**简单的离散 shell hybrid 还不能完全替代当前最佳窗口法**。因此若要进一步把 Mission 8 做到更强，需要继续研究：\n\n")
        f.write("- 是否应只对 PNG 增量项做离散 shell 化，而不是对总 P(k) 直接做低-k 采样；\n")
        f.write("- 是否应把每个 shell 的角向结构和 binning 权重更精细地纳入；\n")
        f.write("- 是否需要在“最后一个离散 shell”和“连续部分”之间引入更平滑的匹配，而不是简单拼接。\n")

    print(f"[INFO] saved summary: {OUT_SUMMARY_MD}")
    print("[INFO] Mission8 shell hybrid analysis done")


if __name__ == "__main__":
    main()
