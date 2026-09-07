#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本用途（3.2.1 精简检验）
1) 固定 kmax=20；
2) kmin 从 2*pi/L (L=1000) 开始逐步减小；
3) 用同一套 P0(k) 输入，比较：
   - 自写 FFTLog（scipy.fht）
   - mcfit.P2xi
4) 只画 1 张图（含 2 个子图）：
   - 子图1：xi0 的相对误差曲线 [%]
   - 子图2：r^2*xi0 的相对误差曲线 [%]
"""

import os
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

# 当前 best-fit（与现有 model.ipynb 一致）
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

R_MIN = 56.0
R_MAX = 380.0
R_N = 260

OUT_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk-pcf-model"
OUT_PNG = os.path.join(OUT_DIR, "fftlog_vs_mcfit_kmax20_kminscan_relerr.png")


def build_log_taper_window(k_array: np.ndarray, kmin: float, kmax: float, frac: float) -> np.ndarray:
    """在 log(k) 上构造边界余弦窗，保证积分上下限并减小截断振铃。"""
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
    """用 desilike 在指定 k 网格上计算理论 P0(k)。"""
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
    """自写 FFTLog 计算 xi0。"""
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
    """mcfit 计算 xi0。"""
    p0_cut = p0_grid * build_log_taper_window(k_grid, kmin, kmax, EDGE_TAPER_FRAC)
    s_grid, xi_grid = P2xi(k_grid, l=0, lowring=True, q=1.5)(p0_cut, extrap=False)
    return np.asarray(s_grid), np.asarray(xi_grid)


def signed_relative_error_percent(a: np.ndarray, b: np.ndarray, floor_scale: float = 1e-6) -> np.ndarray:
    """
    有符号相对误差 [%]： (a-b)/|b| * 100。
    为避免 b 过小时数值爆炸，对分母加下限。
    """
    b_abs = np.abs(b)
    floor = floor_scale * max(np.max(b_abs), 1e-30)
    denom = np.maximum(b_abs, floor)
    return (a - b) / denom * 100.0


def main() -> None:
    """主流程：扫描 kmin，比较整条曲线相对误差，并画一张含子图的图。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    r_eval = np.linspace(R_MIN, R_MAX, R_N)

    err_xi_dict: Dict[float, np.ndarray] = {}
    err_r2xi_dict: Dict[float, np.ndarray] = {}

    for kmin in KMIN_LIST:
        k_grid = np.geomspace(kmin / FFTLOG_PADDING, KMAX_FIXED * FFTLOG_PADDING, FFTLOG_N)
        p0_grid = build_theory_p0(k_grid, BESTFIT_PARAMS)

        s_c, xi_c = xi_custom_fftlog(k_grid, p0_grid, kmin, KMAX_FIXED)
        s_m, xi_m = xi_from_mcfit(k_grid, p0_grid, kmin, KMAX_FIXED)

        o1 = np.argsort(s_c)
        o2 = np.argsort(s_m)
        xi_c_i = np.interp(r_eval, s_c[o1], xi_c[o1])
        xi_m_i = np.interp(r_eval, s_m[o2], xi_m[o2])

        r2_c = r_eval**2 * xi_c_i
        r2_m = r_eval**2 * xi_m_i

        err_xi = signed_relative_error_percent(xi_c_i, xi_m_i)
        err_r2 = signed_relative_error_percent(r2_c, r2_m)

        err_xi_dict[kmin] = err_xi
        err_r2xi_dict[kmin] = err_r2

        print(
            f"kmin={kmin:.6f}, "
            f"mean|err_xi|={np.mean(np.abs(err_xi)):.4f}%, "
            f"mean|err_r2xi|={np.mean(np.abs(err_r2)):.4f}%"
        )

    kfund = 2.0 * np.pi / BOX_SIZE
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(KMIN_LIST)))

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 8.5), sharex=True)

    for c, kmin in zip(colors, KMIN_LIST):
        label = f"kmin={kmin:.6f}"
        if np.isclose(kmin, kfund):
            label += " (=2pi/L)"
        lw = 2.4 if np.isclose(kmin, kfund) else 1.6

        axes[0].plot(r_eval, err_xi_dict[kmin], color=c, lw=lw, label=label)
        axes[1].plot(r_eval, err_r2xi_dict[kmin], color=c, lw=lw, label=label)

    for ax in axes:
        ax.axhline(0.0, color="black", lw=1.0)
        ax.axhline(1.0, color="red", ls="--", lw=0.9)
        ax.axhline(-1.0, color="red", ls="--", lw=0.9)
        ax.grid(True, ls="--", alpha=0.35)

    axes[0].set_ylabel("Rel. Error of xi0 [%]")
    axes[0].set_title("kmax=20, kmin scan: custom FFTLog vs mcfit")
    axes[1].set_ylabel("Rel. Error of r^2*xi0 [%]")
    axes[1].set_xlabel("r [Mpc/h]")
    axes[1].legend(ncol=2, fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"\nSaved figure: {OUT_PNG}")


if __name__ == "__main__":
    main()

