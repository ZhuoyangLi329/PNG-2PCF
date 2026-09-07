#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本用途
1) 固定 kmax=20；
2) kmin 从 2*pi/L 开始往下降；
3) 对每个 kmin，直接画 2PCF 曲线对比（自写 FFTLog vs mcfit）；
4) 一张总图里放多个子图，每个子图两条曲线，不再画 relative error。
"""

import os
import math
from typing import Dict, Tuple

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
# 参数区
# =====================
BOX_SIZE = 1000.0
KMAX_FIXED = 20.0
KMIN_LIST = [2.0 * np.pi / BOX_SIZE, 0.003, 0.002, 0.001, 5e-4, 3e-4, 1e-4]

# 使用当前模型参数（与现有 model.ipynb 对齐）
BESTFIT_PARAMS = {
    "fnl_loc": 122.28837057221368,
    "b1": 1.4799074735660578,
    "sigmas": 3.348575085653272,
    "p": 1.1,
    "sn0": 0.0,
}

UNIT_Z = 1.0
FIXED_P = 1.1
FIXED_SN0 = 0.0

FFTLOG_N = 4096
FFTLOG_PADDING = 4.0
FFTLOG_MU = 0.5
FFTLOG_BIAS = 0.0
EDGE_TAPER_FRAC = 0.06

# 画图 r 范围（沿用你前面对 2PCF 的区间）
R_MIN = 56.0
R_MAX = 380.0
R_N = 260

# 为了可读性，用 r^2*xi_0 展示（仍是 2PCF 曲线本体，只做 r^2 加权）
PLOT_R2XI = True

OUT_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk-pcf-model"
OUT_PNG = os.path.join(OUT_DIR, "fftlog_vs_mcfit_kmax20_kminscan_curves.png")


def build_log_taper_window(k_array: np.ndarray, kmin: float, kmax: float, frac: float) -> np.ndarray:
    """构造 log(k) 边界窗，显式控制积分上下限并减小截断振铃。"""
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
    """在指定 k 网格上生成理论 P0(k)。"""
    cosmo = Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )
    template = FixedPowerSpectrumTemplate(z=UNIT_Z, fiducial=cosmo)
    theory = PNGTracerPowerSpectrumMultipoles(k=k_grid, template=template, mode="b-p")
    theory.init.params["p"].update(fixed=True, value=FIXED_P)
    theory.init.params["sn0"].update(fixed=True, value=FIXED_SN0)
    theory.init.params["sigmas"].update(fixed=False, value=0.0)
    theory(**bestfit)
    return np.array(theory.power[0], dtype=np.float64, copy=True)


def xi_custom_fftlog(k_grid: np.ndarray, p0_grid: np.ndarray, kmin: float, kmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """自写 FFTLog 算法计算 xi0(s)。"""
    p0_cut = p0_grid * build_log_taper_window(k_grid, kmin, kmax, EDGE_TAPER_FRAC)
    dln = np.log(k_grid[1] / k_grid[0])
    offset = fhtoffset(dln, mu=FFTLOG_MU, initial=0.0, bias=FFTLOG_BIAS)

    a_in = (k_grid ** 1.5) * p0_cut
    A_out = fht(a_in, dln=dln, mu=FFTLOG_MU, offset=offset, bias=FFTLOG_BIAS)

    n = k_grid.size
    j = np.arange(n)
    jc = (n - 1) / 2.0
    ln_kc = 0.5 * (np.log(k_grid[0]) + np.log(k_grid[-1]))
    s_grid = np.exp((offset - ln_kc) + (j - jc) * dln)

    const = np.sqrt(np.pi / 2.0) / (2.0 * np.pi**2)
    xi_grid = const * A_out / (s_grid ** 1.5)
    return s_grid, xi_grid


def xi_from_mcfit(k_grid: np.ndarray, p0_grid: np.ndarray, kmin: float, kmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """mcfit 计算 xi0(s)。"""
    p0_cut = p0_grid * build_log_taper_window(k_grid, kmin, kmax, EDGE_TAPER_FRAC)
    s_grid, xi_grid = P2xi(k_grid, l=0, lowring=True, q=1.5)(p0_cut, extrap=False)
    return np.asarray(s_grid), np.asarray(xi_grid)


def main() -> None:
    """主流程：固定 kmax，扫描 kmin，输出多子图曲线对比。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    r_eval = np.linspace(R_MIN, R_MAX, R_N)

    # 预先算出每个 kmin 的两条曲线
    curves = {}
    for kmin in KMIN_LIST:
        k_grid = np.geomspace(kmin / FFTLOG_PADDING, KMAX_FIXED * FFTLOG_PADDING, FFTLOG_N)
        p0_grid = build_theory_p0(k_grid, BESTFIT_PARAMS)

        s_c, xi_c = xi_custom_fftlog(k_grid, p0_grid, kmin, KMAX_FIXED)
        s_m, xi_m = xi_from_mcfit(k_grid, p0_grid, kmin, KMAX_FIXED)

        o1 = np.argsort(s_c)
        o2 = np.argsort(s_m)
        xi_c_i = np.interp(r_eval, s_c[o1], xi_c[o1])
        xi_m_i = np.interp(r_eval, s_m[o2], xi_m[o2])

        if PLOT_R2XI:
            y_c = r_eval**2 * xi_c_i
            y_m = r_eval**2 * xi_m_i
            y_label = r"$r^2\xi_0(r)$"
        else:
            y_c = xi_c_i
            y_m = xi_m_i
            y_label = r"$\xi_0(r)$"

        curves[kmin] = (y_c, y_m)
        print(
            f"kmin={kmin:.6f}: "
            f"mean(|custom-mcfit|)={np.mean(np.abs(y_c - y_m)):.6e}, "
            f"max(|custom-mcfit|)={np.max(np.abs(y_c - y_m)):.6e}"
        )

    # 子图布局：尽量紧凑
    n = len(KMIN_LIST)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.5 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    kfund = 2.0 * np.pi / BOX_SIZE
    for i, kmin in enumerate(KMIN_LIST):
        ax = axes[i]
        y_c, y_m = curves[kmin]

        ax.plot(r_eval, y_c, color="tab:red", lw=1.8, label="Custom FFTLog")
        ax.plot(r_eval, y_m, color="tab:blue", lw=1.8, ls="--", label="mcfit")
        title = f"kmin={kmin:.6f}, kmax={KMAX_FIXED:g}"
        if np.isclose(kmin, kfund):
            title += " (2pi/L)"
        ax.set_title(title, fontsize=10)
        ax.grid(True, ls="--", alpha=0.35)
        ax.set_ylabel(y_label)
        ax.set_xlim(R_MIN, R_MAX)

    # 多余子图隐藏
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # 统一 x 标签
    for ax in axes[max(0, (nrows - 1) * ncols):]:
        if ax.has_data():
            ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")

    # 全局图例放在第一个子图
    axes[0].legend(fontsize=9, loc="best")
    fig.suptitle("2PCF Curves Comparison: Custom FFTLog vs mcfit (fixed kmax=20)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=180)
    print(f"\nSaved figure: {OUT_PNG}")


if __name__ == "__main__":
    main()

