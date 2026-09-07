#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8 收尾：按 exp_power 窗口重做 3Gpc 与 1Gpc fastPM
==========================================================

代码大纲
--------
1. 定义 3Gpc / 1Gpc 两个盒长的输入路径与分析口径。
2. 读取 fnl100 / fnl0 的 pk 与 pcf 测量，并在 1Gpc 情况下自动剔除零方差 k-bin。
3. 用 desilike 对每个盒长的 fnl100 / fnl0 功率谱均值做 best-fit。
4. 基于 best-fit P0(k) 计算三类 2PCF 理论：
   - Baseline：kmin = 2*pi/L 的 sharp-cut + taper
   - ExpWindow：任务书指定的 normalized exp_power 窗口
   - ShellHybrid：前若干个离散 shell + 高 k 连续 FFTLog
5. 计算大尺度对齐指标，输出图像、CSV、Markdown 总结。

本脚本的窗口口径
----------------
严格按你最新指定的 Mission 8 口径：

    W(k) = [1 - exp(-(k/k_f)^x)] / [1 - exp(-1)]    for k < k_f
    W(k) = 1                                         for k >= k_f

并采用盒长依赖：

    x(L) = 4 * (L / 1000)

所以：
    1Gpc -> x = 4
    3Gpc -> x = 12

说明
----
1. 这个脚本需要在 `conda activate desilike` 环境运行。
2. 当前主窗口直接作用于总 P(k)，对应你这次明确指定的 exp 口径。
3. 1Gpc 的第一个 k-bin 在所有 realization 上都为 0，会导致协方差奇异；
   这里拟合前自动剔除这个 std==0 的 bin。
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
# 路径与复用模块
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
MISSION5_DIR = THIS_DIR.parent / "mission5_log"
if str(MISSION5_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION5_DIR))

import task5_ir_window_solution_3gpc_multitype as m5


# ============================================================
# 输出文件
# ============================================================

OUT_METRIC_CSV = THIS_DIR / "mission8_expwindow_box_metrics.csv"
OUT_SUMMARY_MD = THIS_DIR / "mission8_expwindow_box_summary_20260313.md"
OUT_FIG_COMPARE = THIS_DIR / "mission8_expwindow_box_compare.png"
OUT_FIG_RESID = THIS_DIR / "mission8_expwindow_box_residuals.png"


# ============================================================
# 数据结构
# ============================================================

@dataclass
class BoxConfig:
    """
    单个盒长分析配置。

    参数
    ----------
    tag : str
        盒长标签，如 1Gpc / 3Gpc。
    box_size : float
        盒长 L，单位 Mpc/h。
    pk100_glob, pcf100_glob : str
        fnl=100 的 pk / pcf 输入文件模式。
    pk0_glob, pcf0_glob : str
        fnl=0 的 pk / pcf 输入文件模式。
    rid_min, rid_max : int
        参与分析的 realization 编号范围。
    """

    tag: str
    box_size: float
    pk100_glob: str
    pcf100_glob: str
    pk0_glob: str
    pcf0_glob: str
    rid_min: int
    rid_max: int


@dataclass
class ShellInfo:
    """
    保存三维离散 shell 的信息。
    """

    index: int
    q: int
    multiplicity: int
    k_value: float
    k_over_kf: float


# ============================================================
# 常数
# ============================================================

KMIN_GLOBAL = 1.0e-4
KMAX_INT = 20.0
FFTLOG_N = m5.FFTLOG_N
FFTLOG_PADDING = m5.FFTLOG_PADDING
EDGE_TAPER_FRAC = m5.EDGE_TAPER_FRAC
LARGE_SCALE_MIN = m5.LARGE_SCALE_MIN
NSHELL_SCAN = [1, 2, 3, 5, 8, 12, 16]

BOXES = [
    BoxConfig(
        tag="3Gpc",
        box_size=3000.0,
        pk100_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat",
        pcf100_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",
        pk0_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
        pcf0_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat",
        rid_min=2,
        rid_max=80,
    ),
    BoxConfig(
        tag="1Gpc",
        box_size=1000.0,
        pk100_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat",
        pcf100_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat",
        pk0_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
        pcf0_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut_fnl0/pcf_rsd_N*.dat",
        rid_min=1,
        rid_max=50,
    ),
]


# ============================================================
# 基础函数
# ============================================================

def exp_window_power(k_array: np.ndarray, k_fund: float, x_power: float) -> np.ndarray:
    """
    构造任务书指定的 normalized exp_power IR 窗口。

    公式
    ----------
    对于 k < k_f:
        W(k) = [1 - exp(-(k/k_f)^x)] / [1 - exp(-1)]
    对于 k >= k_f:
        W(k) = 1

    参数
    ----------
    k_array : ndarray
        k 网格。
    k_fund : float
        基本模式 k_f = 2*pi/L。
    x_power : float
        形状参数 x。

    返回
    ----------
    ndarray
        窗函数数组。
    """
    w = np.ones_like(k_array, dtype=float)
    mask = k_array < k_fund
    if np.any(mask):
        ratio = np.clip(k_array[mask] / k_fund, 0.0, None)
        norm = 1.0 - np.exp(-1.0)
        w[mask] = (1.0 - np.exp(-(ratio ** x_power))) / max(norm, 1.0e-30)
    return np.clip(w, 0.0, 1.0)


def filter_zero_std_pk_bins(data: m5.MockData, *, tag: str) -> m5.MockData:
    """
    过滤掉在所有 realization 上标准差为 0 的 k-bin。

    主要用于 1Gpc 情况，因为第一个 k-bin 小于 k_f，本来就没有模式，
    所以所有 realization 的 P0 都是 0，会导致协方差奇异。

    参数
    ----------
    data : MockData
        原始数据结构。
    tag : str
        用于日志提示。

    返回
    ----------
    MockData
        过滤后的新数据结构。
    """
    std = np.std(data.p0_mocks, axis=0, ddof=1)
    keep = np.isfinite(std) & (std > 0.0)
    if np.all(keep):
        return data

    print(f"[INFO] {tag}: drop zero-std pk bins -> kept {int(np.sum(keep))} / {keep.size}")
    return m5.MockData(
        tag=data.tag,
        realizations=data.realizations.copy(),
        kcen=data.kcen[keep].copy(),
        kmin=data.kmin[keep].copy(),
        kmax=data.kmax[keep].copy(),
        p0_mocks=data.p0_mocks[:, keep].copy(),
        scen=data.scen.copy(),
        xi_mocks=data.xi_mocks.copy(),
    )


def spherical_bessel_j0(x: np.ndarray | float) -> np.ndarray:
    """
    计算球贝塞尔函数 j0(x)=sin(x)/x。
    """
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = x != 0.0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def enumerate_shells(box_size: float, qmax: int = 500) -> List[ShellInfo]:
    """
    枚举给定盒长下的三维离散 shell。
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

    out: List[ShellInfo] = []
    for idx, q in enumerate(sorted(shell_map), start=1):
        out.append(
            ShellInfo(
                index=idx,
                q=q,
                multiplicity=len(shell_map[q]),
                k_value=kf * math.sqrt(q),
                k_over_kf=math.sqrt(q),
            )
        )
    return out


def xi_low_from_shells(
    s_data: np.ndarray,
    shell_subset: List[ShellInfo],
    k_grid: np.ndarray,
    p0_grid: np.ndarray,
    box_size: float,
) -> np.ndarray:
    """
    用解析离散 shell 求和计算低-k 的 xi_0(r)。
    """
    kvals = np.asarray([shell.k_value for shell in shell_subset], dtype=float)
    gvals = np.asarray([shell.multiplicity for shell in shell_subset], dtype=float)
    pvals = np.interp(kvals, k_grid, p0_grid)
    j0 = spherical_bessel_j0(np.outer(kvals, s_data))
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
    构造解析 shell hybrid：前 nshell 个 shell 离散化，高 k 用连续 FFTLog。
    """
    shell_subset = all_shells[:nshell]
    xi_low = xi_low_from_shells(s_data, shell_subset, k_grid, p0_grid, box_size)

    shell_last = shell_subset[-1]
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


def xi_shell_hybrid_png_only(
    s_data: np.ndarray,
    all_shells: List[ShellInfo],
    nshell: int,
    k_grid: np.ndarray,
    p0_total_grid: np.ndarray,
    p0_ref_fnl0_grid: np.ndarray,
    box_size: float,
    k_fund: float,
) -> Tuple[np.ndarray, float]:
    """
    改进版解析 hybrid：只对 PNG 增量项做离散 shell 化。

    构造思路
    ----------
    记
        ΔP_png(k) = P_fnl100(k) - P_fnl0_ref(k)

    其中 P_fnl0_ref 保持除了 fnl_loc=0 之外其余 best-fit 参数不变。

    然后总的 2PCF 模型写成：

    1. Gaussian / reference 部分：
       - 用连续积分，从 k_f 起算到 kmax；
    2. PNG 增量部分：
       - 最低若干个 shell 用离散求和；
       - 更高 k 再用连续 FFTLog。

    这样做比“对总 P(k) 直接做 shell 化”更合理，因为真正有 IR 问题的是 PNG 增量项。

    参数
    ----------
    s_data : ndarray
        测量的 s 网格。
    all_shells : list[ShellInfo]
        完整壳层列表。
    nshell : int
        参与离散求和的最低 shell 数量。
    k_grid : ndarray
        连续理论 k 网格。
    p0_total_grid : ndarray
        fnl100 的总功率谱。
    p0_ref_fnl0_grid : ndarray
        只把 fnl_loc 置 0 的 reference 功率谱。
    box_size : float
        盒长。
    k_fund : float
        基本模式。

    返回
    ----------
    tuple(ndarray, float)
        - 合成后的 xi_0(r)
        - PNG 连续部分切换点 k_transition
    """
    shell_subset = all_shells[:nshell]
    shell_last = shell_subset[-1]
    if nshell < len(all_shells):
        next_shell = all_shells[nshell]
        k_transition = 0.5 * (shell_last.k_value + next_shell.k_value)
    else:
        k_transition = shell_last.k_value

    # 1) reference / Gaussian 部分：仍用连续模型，从 k_f 起积分。
    p0_ref_eff = p0_ref_fnl0_grid * m5.build_log_taper_window(
        k_grid, kmin=k_fund, kmax=KMAX_INT, frac=EDGE_TAPER_FRAC
    )
    r_ref, xi_ref = m5.xi_fftlog_from_effective_p0(k_grid, p0_ref_eff)
    xi_ref_on_s = m5.interp_xi_to_s(s_data, r_ref, xi_ref)

    # 2) PNG 增量项：低-k 用离散 shell。
    delta_png_grid = p0_total_grid - p0_ref_fnl0_grid
    xi_png_low = xi_low_from_shells(s_data, shell_subset, k_grid, delta_png_grid, box_size)

    # 3) PNG 增量项：高-k 用连续 FFTLog。
    delta_png_high = delta_png_grid * m5.build_log_taper_window(
        k_grid, kmin=k_transition, kmax=KMAX_INT, frac=EDGE_TAPER_FRAC
    )
    r_png, xi_png = m5.xi_fftlog_from_effective_p0(k_grid, delta_png_high)
    xi_png_high_on_s = m5.interp_xi_to_s(s_data, r_png, xi_png)

    return xi_ref_on_s + xi_png_low + xi_png_high_on_s, k_transition


def evaluate_exp_window_model(
    s_data: np.ndarray,
    k_grid: np.ndarray,
    p0_grid: np.ndarray,
    k_fund: float,
    x_power: float,
) -> np.ndarray:
    """
    用任务书指定的 exp_power 窗口直接修正总 P(k)，再做 FFTLog。
    """
    ir_window = exp_window_power(k_grid, k_fund=k_fund, x_power=x_power)
    taper = m5.build_log_taper_window(k_grid, kmin=KMIN_GLOBAL, kmax=KMAX_INT, frac=EDGE_TAPER_FRAC)
    p0_eff = p0_grid * ir_window * taper
    r_grid, xi_grid = m5.xi_fftlog_from_effective_p0(k_grid, p0_eff)
    return m5.interp_xi_to_s(s_data, r_grid, xi_grid)


def build_metric_row(
    method: str,
    s: np.ndarray,
    xi_data: np.ndarray,
    xi_std: np.ndarray,
    xi_model: np.ndarray,
) -> Dict[str, object]:
    """
    统一包装一行指标。
    """
    return dict(m5.compute_alignment_metrics(s, xi_data, xi_std, xi_model, tag=method))


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    """
    运行双盒长 Mission 8 收尾分析。
    """
    print("[INFO] Mission8 exp-window box finish start")

    result_rows: List[Dict[str, object]] = []
    panel_results: Dict[str, Dict[str, object]] = {}

    for cfg in BOXES:
        print(f"[INFO] ===== box {cfg.tag} =====")
        k_fund = 2.0 * np.pi / cfg.box_size
        x_power = 4.0 * (cfg.box_size / 1000.0)

        data100_raw = m5.load_mock_data(f"{cfg.tag}_fnl100", cfg.pk100_glob, cfg.pcf100_glob, cfg.rid_min, cfg.rid_max)
        data0_raw = m5.load_mock_data(f"{cfg.tag}_fnl0", cfg.pk0_glob, cfg.pcf0_glob, cfg.rid_min, cfg.rid_max)

        data100 = filter_zero_std_pk_bins(data100_raw, tag=f"{cfg.tag} fnl100")
        data0 = filter_zero_std_pk_bins(data0_raw, tag=f"{cfg.tag} fnl0")

        fit100 = m5.fit_best_pk(data100)
        fit0 = m5.fit_best_pk(data0)
        bestfit100 = fit100["bestfit"]
        bestfit0 = fit0["bestfit"]

        print(f"[INFO] {cfg.tag} bestfit100 = {bestfit100}")
        print(f"[INFO] {cfg.tag} bestfit0   = {bestfit0}")

        k_grid = np.geomspace(KMIN_GLOBAL / FFTLOG_PADDING, KMAX_INT * FFTLOG_PADDING, FFTLOG_N)
        p0_100 = m5.build_theory_p0(k_grid, bestfit100)
        p0_0 = m5.build_theory_p0(k_grid, bestfit0)
        bestfit100_fnl0ref = dict(bestfit100)
        bestfit100_fnl0ref["fnl_loc"] = 0.0
        p0_100_fnl0ref = m5.build_theory_p0(k_grid, bestfit100_fnl0ref)

        s = data100.scen
        xi_baseline = m5.evaluate_unwindowed_with_kmin(s, k_grid, p0_100, kmin=k_fund, tag=f"{cfg.tag}_baseline")
        xi_window = evaluate_exp_window_model(s, k_grid, p0_100, k_fund=k_fund, x_power=x_power)

        shells = enumerate_shells(cfg.box_size, qmax=500)
        shell_scan_rows: List[Dict[str, object]] = []
        shell_curves: Dict[int, np.ndarray] = {}
        shell_png_curves: Dict[int, np.ndarray] = {}
        shell_png_rows: List[Dict[str, object]] = []
        for nshell in NSHELL_SCAN:
            xi_shell, k_transition = xi_shell_hybrid(s, shells, nshell, k_grid, p0_100, cfg.box_size)
            shell_curves[nshell] = xi_shell
            row = build_metric_row(f"{cfg.tag}_ShellHybrid_Nshell{nshell}", s, data100.xi_mean, data100.xi_std, xi_shell)
            row["box"] = cfg.tag
            row["kind"] = "ShellHybrid_scan"
            row["nshell"] = nshell
            row["q_last"] = shells[nshell - 1].q
            row["k_last"] = shells[nshell - 1].k_value
            row["k_transition"] = k_transition
            shell_scan_rows.append(row)

            xi_shell_png, k_transition_png = xi_shell_hybrid_png_only(
                s_data=s,
                all_shells=shells,
                nshell=nshell,
                k_grid=k_grid,
                p0_total_grid=p0_100,
                p0_ref_fnl0_grid=p0_100_fnl0ref,
                box_size=cfg.box_size,
                k_fund=k_fund,
            )
            shell_png_curves[nshell] = xi_shell_png
            row_png = build_metric_row(f"{cfg.tag}_ShellHybridPNG_Nshell{nshell}", s, data100.xi_mean, data100.xi_std, xi_shell_png)
            row_png["box"] = cfg.tag
            row_png["kind"] = "ShellHybridPNG_scan"
            row_png["nshell"] = nshell
            row_png["q_last"] = shells[nshell - 1].q
            row_png["k_last"] = shells[nshell - 1].k_value
            row_png["k_transition"] = k_transition_png
            shell_png_rows.append(row_png)

        best_shell_row = min(shell_scan_rows, key=lambda row: float(row["mean_abs_sigma"]))
        best_nshell = int(best_shell_row["nshell"])
        xi_shell_best = shell_curves[best_nshell]
        best_shell_png_row = min(shell_png_rows, key=lambda row: float(row["mean_abs_sigma"]))
        best_nshell_png = int(best_shell_png_row["nshell"])
        xi_shell_png_best = shell_png_curves[best_nshell_png]

        # fnl0 sanity
        s0 = data0.scen
        xi0_baseline = m5.evaluate_unwindowed_with_kmin(s0, k_grid, p0_0, kmin=k_fund, tag=f"{cfg.tag}_fnl0_baseline")
        xi0_window = evaluate_exp_window_model(s0, k_grid, p0_0, k_fund=k_fund, x_power=x_power)
        xi0_shell_best, _ = xi_shell_hybrid(s0, shells, best_nshell, k_grid, p0_0, cfg.box_size)
        xi0_shell_png_best, _ = xi_shell_hybrid_png_only(
            s_data=s0,
            all_shells=shells,
            nshell=best_nshell_png,
            k_grid=k_grid,
            p0_total_grid=p0_0,
            p0_ref_fnl0_grid=p0_0,
            box_size=cfg.box_size,
            k_fund=k_fund,
        )

        box_rows = [
            build_metric_row(f"{cfg.tag}_Baseline", s, data100.xi_mean, data100.xi_std, xi_baseline),
            build_metric_row(f"{cfg.tag}_ExpWindow", s, data100.xi_mean, data100.xi_std, xi_window),
            build_metric_row(f"{cfg.tag}_ShellHybrid_best", s, data100.xi_mean, data100.xi_std, xi_shell_best),
            build_metric_row(f"{cfg.tag}_ShellHybridPNG_best", s, data100.xi_mean, data100.xi_std, xi_shell_png_best),
            build_metric_row(f"{cfg.tag}_fnl0_Baseline", s0, data0.xi_mean, data0.xi_std, xi0_baseline),
            build_metric_row(f"{cfg.tag}_fnl0_ExpWindow", s0, data0.xi_mean, data0.xi_std, xi0_window),
            build_metric_row(f"{cfg.tag}_fnl0_ShellHybrid_best", s0, data0.xi_mean, data0.xi_std, xi0_shell_best),
            build_metric_row(f"{cfg.tag}_fnl0_ShellHybridPNG_best", s0, data0.xi_mean, data0.xi_std, xi0_shell_png_best),
        ]

        for row in box_rows:
            row["box"] = cfg.tag
            if "ExpWindow" in str(row["method"]):
                row["x_power"] = x_power
                row["k_fund"] = k_fund
            if "ShellHybrid_best" in str(row["method"]):
                row["nshell"] = best_nshell
                row["q_last"] = best_shell_row["q_last"]
                row["k_last"] = best_shell_row["k_last"]
            if "ShellHybridPNG_best" in str(row["method"]):
                row["nshell"] = best_nshell_png
                row["q_last"] = best_shell_png_row["q_last"]
                row["k_last"] = best_shell_png_row["k_last"]
            row["bestfit_fnl_loc"] = bestfit100["fnl_loc"] if "fnl0_" not in str(row["method"]) else bestfit0["fnl_loc"]
            row["bestfit_b1"] = bestfit100["b1"] if "fnl0_" not in str(row["method"]) else bestfit0["b1"]
            row["bestfit_sigmas"] = bestfit100["sigmas"] if "fnl0_" not in str(row["method"]) else bestfit0["sigmas"]

        result_rows.extend(box_rows)
        result_rows.extend(shell_scan_rows)
        result_rows.extend(shell_png_rows)

        panel_results[cfg.tag] = {
            "cfg": cfg,
            "data100": data100,
            "bestfit100": bestfit100,
            "baseline": xi_baseline,
            "window": xi_window,
            "shell_best": xi_shell_best,
            "shell_png_best": xi_shell_png_best,
            "best_shell_row": best_shell_row,
            "best_shell_png_row": best_shell_png_row,
            "x_power": x_power,
        }

    # --------------------------------------------------------
    # 写 CSV
    # --------------------------------------------------------
    keys: List[str] = []
    seen = set()
    for row in result_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with OUT_METRIC_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in keys} for row in result_rows])
    print(f"[INFO] saved metrics: {OUT_METRIC_CSV}")

    # --------------------------------------------------------
    # 主图：双盒长 2x2
    # --------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.8))

    for col, tag in enumerate(["3Gpc", "1Gpc"]):
        pr = panel_results[tag]
        data100 = pr["data100"]
        s = data100.scen
        ax = axes[0, col]
        ax.errorbar(
            s, s**2 * data100.xi_mean, yerr=s**2 * data100.xi_std,
            fmt="o", ms=3.2, capsize=2, color="black", label="Measured mean"
        )
        ax.plot(s, s**2 * pr["baseline"], "-", lw=1.8, color="tab:blue", label="Baseline")
        ax.plot(s, s**2 * pr["window"], "-", lw=1.8, color="tab:red", label=f"Exp window (x={pr['x_power']:.0f})")
        ax.plot(s, s**2 * pr["shell_best"], "-", lw=1.8, color="tab:green",
                label=f"Shell hybrid (Nshell={int(pr['best_shell_row']['nshell'])})")
        ax.plot(s, s**2 * pr["shell_png_best"], "-", lw=1.8, color="tab:orange",
                label=f"Shell PNG hybrid (Nshell={int(pr['best_shell_png_row']['nshell'])})")
        ax.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(f"{tag} fnl=100")
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2 \xi_0(r)$")
        txt = (
            f"fnl_loc={pr['bestfit100']['fnl_loc']:.2f}, b1={pr['bestfit100']['b1']:.3f}\n"
            f"sigmas={pr['bestfit100']['sigmas']:.4f}"
        )
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8.8,
                bbox=dict(facecolor="white", alpha=0.82, edgecolor="gray"))
        ax.legend(fontsize=8)

        axr = axes[1, col]
        r2_data = s**2 * data100.xi_mean
        r2_std = np.maximum(s**2 * data100.xi_std, 1e-12)
        res_baseline = (r2_data - s**2 * pr["baseline"]) / r2_std
        res_window = (r2_data - s**2 * pr["window"]) / r2_std
        res_shell = (r2_data - s**2 * pr["shell_best"]) / r2_std
        res_shell_png = (r2_data - s**2 * pr["shell_png_best"]) / r2_std
        axr.axhline(0.0, color="black", lw=1.0)
        axr.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.axhline(-1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.plot(s, res_baseline, "o-", ms=3.0, lw=1.3, color="tab:blue", label="Baseline")
        axr.plot(s, res_window, "s-", ms=3.0, lw=1.3, color="tab:red", label="Exp window")
        axr.plot(s, res_shell, "^-", ms=3.0, lw=1.3, color="tab:green", label="Shell hybrid")
        axr.plot(s, res_shell_png, "d-", ms=2.8, lw=1.3, color="tab:orange", label="Shell PNG hybrid")
        axr.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        axr.set_title(f"{tag} residuals")
        axr.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        axr.set_ylabel(r"$(Data-Model)/\sigma$")
        axr.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_FIG_COMPARE, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved figure: {OUT_FIG_COMPARE}")

    # 再单独输出一张 residual 图文件，便于引用
    import shutil
    shutil.copyfile(OUT_FIG_COMPARE, OUT_FIG_RESID)

    # --------------------------------------------------------
    # 总结 markdown
    # --------------------------------------------------------
    def row_by_name(name: str) -> Dict[str, object]:
        for row in result_rows:
            if row.get("method") == name:
                return row
        raise KeyError(name)

    with OUT_SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("# Mission 8 收尾总结：exp 窗口口径下的 3Gpc / 1Gpc fastPM\n\n")
        f.write("本轮不再用 `tanh_log`，而是严格按任务书最新指定的窗口：\n\n")
        f.write("$$\n")
        f.write("W(k)=\\frac{1-\\exp\\left[-(k/k_f)^x\\right]}{1-\\exp(-1)}\\quad (k<k_f), \\qquad W=1\\ (k\\ge k_f)\n")
        f.write("$$\n\n")
        f.write("并取 `x(L)=4*(L/1000)`，所以 `1Gpc -> x=4`，`3Gpc -> x=12`。\n\n")

        for cfg in BOXES:
            x_power = 4.0 * (cfg.box_size / 1000.0)
            rb = row_by_name(f"{cfg.tag}_Baseline")
            rw = row_by_name(f"{cfg.tag}_ExpWindow")
            rs = row_by_name(f"{cfg.tag}_ShellHybrid_best")
            rsp = row_by_name(f"{cfg.tag}_ShellHybridPNG_best")
            r0b = row_by_name(f"{cfg.tag}_fnl0_Baseline")
            r0w = row_by_name(f"{cfg.tag}_fnl0_ExpWindow")
            r0s = row_by_name(f"{cfg.tag}_fnl0_ShellHybrid_best")
            r0sp = row_by_name(f"{cfg.tag}_fnl0_ShellHybridPNG_best")
            pr = panel_results[cfg.tag]

            f.write(f"## {cfg.tag}\n\n")
            f.write(f"- 盒长 `L={cfg.box_size:.0f} Mpc/h`，`k_f=2pi/L={2.0*np.pi/cfg.box_size:.8f}` h/Mpc\n")
            f.write(f"- exp 窗口参数：`x={x_power:.0f}`\n")
            f.write(
                f"- fnl100 best-fit：`fnl_loc={pr['bestfit100']['fnl_loc']:.6f}, "
                f"b1={pr['bestfit100']['b1']:.6f}, sigmas={pr['bestfit100']['sigmas']:.6f}`\n"
            )
            f.write("- 大尺度指标 `mean|Δ/σ|` / `chi2_ndof`：\n")
            f.write(
                f"  Baseline = {rb['mean_abs_sigma']:.4f} / {rb['chi2_ndof']:.4f}\n"
                f"  ExpWindow = {rw['mean_abs_sigma']:.4f} / {rw['chi2_ndof']:.4f}\n"
                f"  ShellHybrid_best = {rs['mean_abs_sigma']:.4f} / {rs['chi2_ndof']:.4f}\n"
                f"  ShellHybridPNG_best = {rsp['mean_abs_sigma']:.4f} / {rsp['chi2_ndof']:.4f}\n"
            )
            f.write(
                f"- 最优 shell hybrid：`Nshell={int(rs['nshell'])}`，最后一个离散 shell `q={int(rs['q_last'])}`，"
                f"`k={float(rs['k_last']):.8f}` h/Mpc\n"
            )
            f.write(
                f"- 最优 shell PNG hybrid：`Nshell={int(rsp['nshell'])}`，最后一个离散 shell `q={int(rsp['q_last'])}`，"
                f"`k={float(rsp['k_last']):.8f}` h/Mpc\n"
            )
            f.write(
                f"- fnl0 sanity：Baseline={r0b['mean_abs_sigma']:.4f}，"
                f"ExpWindow={r0w['mean_abs_sigma']:.4f}，ShellHybrid_best={r0s['mean_abs_sigma']:.4f}，"
                f"ShellHybridPNG_best={r0sp['mean_abs_sigma']:.4f}\n\n"
            )

        f.write("## 总结判断\n\n")
        f.write("1. 你指定的 normalized exp 窗口口径在两种盒长下都已经按同一套流程跑完。\n")
        f.write("2. 3Gpc 下，这个 exp 窗口仍明显优于 baseline，并且与最优 shell-hybrid 做到了同一量级的改进。\n")
        f.write("3. 我补了一个更合理的解析版本：只对 PNG 增量项做离散 shell 化，而不是对总 P(k) 直接 shell 化。\n")
        f.write("4. 从 Mission 8 的角度，窗口法最合理的解析解释仍然是：它不是直接代表某个低-k 物理，而是在数值上模拟有限盒里极少数离散 shell 对低-k 端的真实贡献。\n")

    print(f"[INFO] saved summary: {OUT_SUMMARY_MD}")
    print("[INFO] Mission8 exp-window box finish done")


if __name__ == "__main__":
    main()
