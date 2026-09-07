#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本大纲（执行逻辑）
1) 构建同一组理论 P0(k):
   - 使用 desilike 的 PNGTracerPowerSpectrumMultipoles
   - 参数采用当前任务 3.1 的 best-fit（可在参数区修改）
2) 对每组 (kmin, kmax) 执行两种积分/变换:
   - 我们自写 FFTLog（基于 scipy.fft.fht）
   - mcfit.P2xi
3) 在统一的 r 区间比较两者:
   - 计算 xi0 与 r^2*xi0 的百分比误差（含分母保护）
   - 输出每组统计（mean/median/p90/max）
4) 产出结果文件:
   - CSV: 每组参数与误差统计
   - 图1: MAPE(r^2*xi) 热力图（kmin vs kmax）
   - 图2: 固定 kmax=15 时，误差随 kmin 的变化
   - 图3: 重点案例（kmin=2*pi/L）下，不同 kmax 的误差曲线

说明
- 该脚本是 3.2.1 的“算法一致性验证”专用，不改主模型文件。
- 误差按“百分比误差”定义：
    err(%) = |A - B| / |B| * 100
  其中 B 取 mcfit 结果；为避免 B 接近 0 导致发散，使用分母下限保护。
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fht, fhtoffset

from mcfit import P2xi
from cosmoprimo import Cosmology
from desilike.theories.galaxy_clustering import (
    FixedPowerSpectrumTemplate,
    PNGTracerPowerSpectrumMultipoles,
)


# =====================
# 参数区（可直接改）
# =====================

# 物理与模型参数
BOX_SIZE = 1000.0
UNIT_Z = 1.0
FIXED_P = 1.1
FIXED_SN0 = 0.0

# 当前 3.1 对应 best-fit（可按最新拟合结果更新）
BESTFIT_PARAMS = {
    "fnl_loc": 122.28837057221368,
    "b1": 1.4799074735660578,
    "sigmas": 3.348575085653272,
    "p": 1.1,
    "sn0": 0.0,
}

# k 扫描列表（3.2.1 要求：改变 kmin 与 kmax）
KMIN_LIST = [0.1, 0.05, 0.02, 0.01, 2.0 * np.pi / BOX_SIZE, 0.003, 0.001, 0.0003, 0.0001]
KMAX_LIST = [0.3, 0.5, 1.0, 2.0, 5.0, 8.0, 15.0]

# FFTLog 数值参数
FFTLOG_N = 4096
FFTLOG_PADDING = 4.0
FFTLOG_MU = 0.5
FFTLOG_BIAS = 0.0
EDGE_TAPER_FRAC = 0.06

# 比较区间（统一 r 网格）
R_COMPARE_MIN = 56.0
R_COMPARE_MAX = 380.0
R_COMPARE_N = 250

# 百分比误差分母保护（相对参考曲线幅度）
PCT_DENOM_FLOOR = 1e-6

# 输出目录
OUT_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk-pcf-model"
OUT_CSV = os.path.join(OUT_DIR, "fftlog_vs_mcfit_metrics.csv")
OUT_HEATMAP = os.path.join(OUT_DIR, "fftlog_vs_mcfit_mape_heatmap.png")
OUT_LINE = os.path.join(OUT_DIR, "fftlog_vs_mcfit_kmin_scan_at_kmax15.png")
OUT_CASE = os.path.join(OUT_DIR, "fftlog_vs_mcfit_case_kmin_2pi_over_L.png")


@dataclass
class CompareMetrics:
    """保存单个 (kmin, kmax) 组合的误差统计结果。"""

    kmin: float
    kmax: float
    mape_xi_mean: float
    mape_xi_median: float
    mape_xi_p90: float
    mape_xi_max: float
    mape_r2xi_mean: float
    mape_r2xi_median: float
    mape_r2xi_p90: float
    mape_r2xi_max: float


def build_log_taper_window(k_array: np.ndarray, kmin: float, kmax: float, frac: float) -> np.ndarray:
    """
    在 log(k) 上构建边界平滑窗，确保积分上下限可控并减小截断振铃。

    参数
    ----------
    k_array : ndarray
        对数等间距 k 网格。
    kmin, kmax : float
        积分有效区间 [kmin, kmax]。
    frac : float
        过渡区在 log 区间中的比例。

    返回
    ----------
    ndarray
        窗函数，区间外 0，区间内约 1，边界余弦平滑。
    """
    w = np.zeros_like(k_array, dtype=np.float64)
    lk = np.log(k_array)
    l0 = np.log(kmin)
    l1 = np.log(kmax)

    dl = frac * (l1 - l0)
    if dl <= 0:
        w[(k_array >= kmin) & (k_array <= kmax)] = 1.0
        return w

    left = l0 + dl
    right = l1 - dl

    m = (lk >= l0) & (lk < left)
    w[m] = 0.5 * (1.0 - np.cos(np.pi * (lk[m] - l0) / dl))

    m = (lk >= left) & (lk <= right)
    w[m] = 1.0

    m = (lk > right) & (lk <= l1)
    w[m] = 0.5 * (1.0 + np.cos(np.pi * (lk[m] - right) / dl))
    return w


def build_theory_p0(k_grid: np.ndarray, bestfit: Dict[str, float]) -> np.ndarray:
    """
    在给定 k 网格上计算理论 P0(k)。

    参数
    ----------
    k_grid : ndarray
        理论评估网格（建议对数网格）。
    bestfit : dict
        模型参数字典，至少包含 fnl_loc, b1, sigmas。

    返回
    ----------
    ndarray
        P0(k) 数组。
    """
    cosmo_unit = Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )
    template = FixedPowerSpectrumTemplate(z=UNIT_Z, fiducial=cosmo_unit)
    theory = PNGTracerPowerSpectrumMultipoles(k=k_grid, template=template, mode="b-p")

    theory.init.params["p"].update(fixed=True, value=FIXED_P)
    theory.init.params["sn0"].update(fixed=True, value=FIXED_SN0)
    theory.init.params["sigmas"].update(fixed=False, value=0.0)

    theory(**bestfit)
    return np.array(theory.power[0], dtype=np.float64, copy=True)


def xi_fftlog_custom(k_grid: np.ndarray, p0_grid: np.ndarray, kmin: float, kmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    用“自写 FFTLog”计算 xi0(s)。

    数学映射：
    - xi0(s) = ∫ dk k^2/(2π^2) P0(k) j0(ks)
    - 通过 j0 -> J_{1/2}，转成 μ=1/2 的 Hankel 变换。

    参数
    ----------
    k_grid : ndarray
        对数等间距 k 网格。
    p0_grid : ndarray
        与 k_grid 对应的 P0(k)。
    kmin, kmax : float
        自定义积分上下限。

    返回
    ----------
    tuple(ndarray, ndarray)
        s_grid, xi0_grid。
    """
    window = build_log_taper_window(k_grid, kmin, kmax, EDGE_TAPER_FRAC)
    p0_cut = p0_grid * window

    dln = np.log(k_grid[1] / k_grid[0])
    offset = fhtoffset(dln, mu=FFTLOG_MU, initial=0.0, bias=FFTLOG_BIAS)

    a_in = (k_grid ** 1.5) * p0_cut
    A_out = fht(a_in, dln=dln, mu=FFTLOG_MU, offset=offset, bias=FFTLOG_BIAS)

    n = k_grid.size
    j = np.arange(n)
    j_center = (n - 1) / 2.0
    ln_kc = 0.5 * (np.log(k_grid[0]) + np.log(k_grid[-1]))
    ln_sc = offset - ln_kc
    s_grid = np.exp(ln_sc + (j - j_center) * dln)

    const = np.sqrt(np.pi / 2.0) / (2.0 * np.pi**2)
    xi0_grid = const * A_out / (s_grid ** 1.5)
    return s_grid, xi0_grid


def xi_mcfit_transform(k_grid: np.ndarray, p0_grid: np.ndarray, kmin: float, kmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 mcfit 计算 xi0(s)。

    参数
    ----------
    k_grid : ndarray
        对数等间距 k 网格。
    p0_grid : ndarray
        与 k_grid 对应的 P0(k)。
    kmin, kmax : float
        自定义积分上下限（通过窗函数实现）。

    返回
    ----------
    tuple(ndarray, ndarray)
        s_grid, xi0_grid。
    """
    window = build_log_taper_window(k_grid, kmin, kmax, EDGE_TAPER_FRAC)
    p0_cut = p0_grid * window
    s_grid, xi0_grid = P2xi(k_grid, l=0, lowring=True, q=1.5)(p0_cut, extrap=False)
    return np.asarray(s_grid), np.asarray(xi0_grid)


def percent_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    计算百分比误差 |a-b|/|b|*100，并对分母做幅度下限保护。

    参数
    ----------
    a, b : ndarray
        要比较的两条曲线，长度相同。

    返回
    ----------
    ndarray
        百分比误差数组。
    """
    b_abs = np.abs(b)
    denom_floor = PCT_DENOM_FLOOR * max(np.max(b_abs), 1e-30)
    denom = np.maximum(b_abs, denom_floor)
    return np.abs(a - b) / denom * 100.0


def compute_metrics_for_case(kmin: float, kmax: float, bestfit: Dict[str, float], r_eval: np.ndarray) -> Tuple[CompareMetrics, np.ndarray]:
    """
    对单组 (kmin,kmax) 计算“自写 FFTLog vs mcfit”的误差统计。

    参数
    ----------
    kmin, kmax : float
        当前扫描参数。
    bestfit : dict
        理论参数。
    r_eval : ndarray
        统一比较网格。

    返回
    ----------
    tuple(CompareMetrics, ndarray)
        误差统计对象 + 当前组在 r_eval 上的 r2xi 百分比误差曲线。
    """
    k_grid = np.geomspace(kmin / FFTLOG_PADDING, kmax * FFTLOG_PADDING, FFTLOG_N)
    p0_grid = build_theory_p0(k_grid, bestfit)

    s_fft, xi_fft = xi_fftlog_custom(k_grid, p0_grid, kmin, kmax)
    s_mc, xi_mc = xi_mcfit_transform(k_grid, p0_grid, kmin, kmax)

    # 插值到统一 r 网格，便于一对一比较
    o1 = np.argsort(s_fft)
    o2 = np.argsort(s_mc)
    xi_fft_i = np.interp(r_eval, s_fft[o1], xi_fft[o1])
    xi_mc_i = np.interp(r_eval, s_mc[o2], xi_mc[o2])

    r2_fft = r_eval**2 * xi_fft_i
    r2_mc = r_eval**2 * xi_mc_i

    err_xi = percent_error(xi_fft_i, xi_mc_i)
    err_r2xi = percent_error(r2_fft, r2_mc)

    metrics = CompareMetrics(
        kmin=kmin,
        kmax=kmax,
        mape_xi_mean=float(np.mean(err_xi)),
        mape_xi_median=float(np.median(err_xi)),
        mape_xi_p90=float(np.percentile(err_xi, 90)),
        mape_xi_max=float(np.max(err_xi)),
        mape_r2xi_mean=float(np.mean(err_r2xi)),
        mape_r2xi_median=float(np.median(err_r2xi)),
        mape_r2xi_p90=float(np.percentile(err_r2xi, 90)),
        mape_r2xi_max=float(np.max(err_r2xi)),
    )
    return metrics, err_r2xi


def save_csv(metrics_list: List[CompareMetrics], out_csv: str) -> None:
    """
    将误差统计写入 CSV 文件。

    参数
    ----------
    metrics_list : list[CompareMetrics]
        所有参数组合结果。
    out_csv : str
        CSV 输出路径。
    """
    header = (
        "kmin,kmax,"
        "mape_xi_mean,mape_xi_median,mape_xi_p90,mape_xi_max,"
        "mape_r2xi_mean,mape_r2xi_median,mape_r2xi_p90,mape_r2xi_max"
    )
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for m in metrics_list:
            f.write(
                f"{m.kmin:.8g},{m.kmax:.8g},"
                f"{m.mape_xi_mean:.6g},{m.mape_xi_median:.6g},{m.mape_xi_p90:.6g},{m.mape_xi_max:.6g},"
                f"{m.mape_r2xi_mean:.6g},{m.mape_r2xi_median:.6g},{m.mape_r2xi_p90:.6g},{m.mape_r2xi_max:.6g}\n"
            )


def plot_heatmap(metrics_list: List[CompareMetrics], out_png: str) -> None:
    """
    画 MAPE(r^2*xi) 的热力图（kmin vs kmax）。

    参数
    ----------
    metrics_list : list[CompareMetrics]
        所有参数组合结果。
    out_png : str
        输出图片路径。
    """
    kmins = sorted(set(m.kmin for m in metrics_list))
    kmaxs = sorted(set(m.kmax for m in metrics_list))
    grid = np.full((len(kmins), len(kmaxs)), np.nan, dtype=float)

    idx_min = {v: i for i, v in enumerate(kmins)}
    idx_max = {v: i for i, v in enumerate(kmaxs)}
    for m in metrics_list:
        grid[idx_min[m.kmin], idx_max[m.kmax]] = m.mape_r2xi_mean

    plt.figure(figsize=(9, 6.2))
    im = plt.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, label="Mean Percentage Error of r^2*xi [%]")
    plt.xticks(np.arange(len(kmaxs)), [f"{v:g}" for v in kmaxs], rotation=0)
    plt.yticks(np.arange(len(kmins)), [f"{v:.6g}" for v in kmins], rotation=0)
    plt.xlabel("kmax [h/Mpc]")
    plt.ylabel("kmin [h/Mpc]")
    plt.title("FFTLog(custom) vs mcfit: Mean % Error (r^2*xi)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)


def plot_kmin_scan_line(metrics_list: List[CompareMetrics], out_png: str, fixed_kmax: float = 15.0) -> None:
    """
    画固定 kmax 时，误差随 kmin 变化的折线图（重点看 2*pi/L）。

    参数
    ----------
    metrics_list : list[CompareMetrics]
        所有参数组合结果。
    out_png : str
        输出图片路径。
    fixed_kmax : float
        固定 kmax 值。
    """
    data = [m for m in metrics_list if np.isclose(m.kmax, fixed_kmax)]
    data = sorted(data, key=lambda x: x.kmin)
    x = [m.kmin for m in data]
    y = [m.mape_r2xi_mean for m in data]
    kfund = 2.0 * np.pi / BOX_SIZE

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(x, y, "o-", color="tab:blue", label=f"kmax={fixed_kmax:g}")
    plt.axvline(kfund, color="red", ls="--", lw=1.2, label=f"2pi/L={kfund:.6f}")
    plt.xscale("log")
    plt.xlabel("kmin [h/Mpc]")
    plt.ylabel("Mean Percentage Error of r^2*xi [%]")
    plt.title("kmin scan at fixed kmax=15")
    plt.grid(True, which="both", ls="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)


def plot_case_kmin_fund(err_curves: Dict[float, np.ndarray], r_eval: np.ndarray, out_png: str) -> None:
    """
    画重点案例：kmin=2*pi/L 时，不同 kmax 的误差曲线。

    参数
    ----------
    err_curves : dict[float, ndarray]
        key 为 kmax，value 为 r_eval 上的百分比误差曲线。
    r_eval : ndarray
        r 比较网格。
    out_png : str
        输出图片路径。
    """
    plt.figure(figsize=(9.3, 5.6))
    for kmax in sorted(err_curves.keys()):
        plt.plot(r_eval, err_curves[kmax], lw=1.8, label=f"kmax={kmax:g}")
    plt.xlabel("r [Mpc/h]")
    plt.ylabel("Percentage Error of r^2*xi [%]")
    plt.title("Case Study: kmin=2pi/L, FFTLog(custom) vs mcfit")
    plt.grid(True, ls="--", alpha=0.35)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)


def main() -> None:
    """主函数：执行扫描、汇总误差、输出图表与文本结论。"""
    os.makedirs(OUT_DIR, exist_ok=True)

    r_eval = np.linspace(R_COMPARE_MIN, R_COMPARE_MAX, R_COMPARE_N)
    metrics_list: List[CompareMetrics] = []

    # 重点案例：固定 kmin=2pi/L，保存不同 kmax 的误差曲线
    kfund = 2.0 * np.pi / BOX_SIZE
    fund_err_curves: Dict[float, np.ndarray] = {}

    for kmin in KMIN_LIST:
        for kmax in KMAX_LIST:
            if kmin >= kmax:
                continue
            metrics, err_r2xi = compute_metrics_for_case(kmin, kmax, BESTFIT_PARAMS, r_eval)
            metrics_list.append(metrics)
            if np.isclose(kmin, kfund):
                fund_err_curves[kmax] = err_r2xi
            print(
                f"[kmin={kmin:.6g}, kmax={kmax:.6g}] "
                f"MAPE(r2xi): mean={metrics.mape_r2xi_mean:.4f}% "
                f"p90={metrics.mape_r2xi_p90:.4f}% max={metrics.mape_r2xi_max:.4f}%"
            )

    save_csv(metrics_list, OUT_CSV)
    plot_heatmap(metrics_list, OUT_HEATMAP)
    plot_kmin_scan_line(metrics_list, OUT_LINE, fixed_kmax=15.0)
    if fund_err_curves:
        plot_case_kmin_fund(fund_err_curves, r_eval, OUT_CASE)

    # 打印重点结论：kmax=15 时，kmin=2pi/L 的误差
    subset = [m for m in metrics_list if np.isclose(m.kmax, 15.0)]
    subset = sorted(subset, key=lambda x: x.kmin)
    print("\n=== 重点结论（kmax=15）===")
    for m in subset:
        tag = " <== 2pi/L" if np.isclose(m.kmin, kfund) else ""
        print(
            f"kmin={m.kmin:.6f}: "
            f"Mean%Err(r2xi)={m.mape_r2xi_mean:.4f}, "
            f"P90={m.mape_r2xi_p90:.4f}, Max={m.mape_r2xi_max:.4f}{tag}"
        )

    print("\n输出文件：")
    print(OUT_CSV)
    print(OUT_HEATMAP)
    print(OUT_LINE)
    print(OUT_CASE)


if __name__ == "__main__":
    main()

