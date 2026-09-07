#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务5 v22：3Gpc 用 EZmock 协方差 + 当前最佳窗口检查 2PCF 建模

代码大纲
--------
A) 读取 3Gpc fastPM 的 fnl100/fnl0 数据（PK + 2PCF），构建测量均值与误差。
B) 读取现有 cov_mock 下 EZmock 的 PK，作为 external covariance 样本。
C) 在 PK_FIT_KMAX=0.08 下做 best-fit（data=fastPM均值，cov=EZmock样本）。
D) 将 best-fit P0 通过 FFTLog 积分得到 2PCF：
   - Baseline: kmin=2pi/L
   - Window:   TestB exp_power(x(L))，其中 x(L)=4*(L/1000)
   - 参考:     exp_power(12)（同一口径的显式写法）
E) 输出图与 summary，量化 r>=150 / r>=200 指标。

说明
----
- 本脚本只做 3Gpc。
- PK 拟合上限严格使用 0.08。
- 若 EZmock 某些文件缺失/损坏，会自动跳过，只使用可读样本。
"""

from __future__ import annotations

import glob
import os
import re
import sys
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MISSION5_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission5_log"
sys.path.insert(0, MISSION5_DIR)

import task5_ir_window_solution_3gpc_multitype as m5
import task5_v14_3gpc_joint_fnl0_fnl100_testb_scan as v14

# ---------------- 参数区 ----------------
RID_MIN = 2
RID_MAX = 99

PK_3GPC_FNL100_GLOB = v14.PK_3GPC_FNL100_GLOB
PCF_3GPC_FNL100_GLOB = v14.PCF_3GPC_FNL100_GLOB
PK_3GPC_FNL0_GLOB = v14.PK_3GPC_FNL0_GLOB
PCF_3GPC_FNL0_GLOB = v14.PCF_3GPC_FNL0_GLOB

EZCOV_GLOB = "/pscratch/sd/l/lzy/cov_mock/B3000G768Z0N4417983_b0.18d5r270c1.65_seed*/PK_EZmock_B3000G768Z0N4417983_b0.18d5r270c1.65_seed*_RSD.dat"

PK_FIT_KMAX = 0.08
P_FIXED = 1.1

# 当前采用“与 L 有关”的窗口（x(L)=4*(L/1000)）
BEST_WINDOW_TYPE = "exp_power"
BOX_SIZE_3GPC = 3000.0
BEST_WINDOW_PARAM = 4.0 * (BOX_SIZE_3GPC / 1000.0)

# 保守版参考窗口
REF_WINDOW_TYPE = "exp_power"
REF_WINDOW_PARAM = 12.0

OUT_PNG = os.path.join(MISSION5_DIR, "task5_v22_3gpc_ezcov_bestwindow_kmax008.png")
OUT_TXT = os.path.join(MISSION5_DIR, "task5_v22_3gpc_ezcov_bestwindow_kmax008_summary.txt")


plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"


def parse_seed(path: str) -> int:
    m = re.search(r"seed(\d+)_RSD\.dat$", path)
    return int(m.group(1)) if m else -1


def load_ezmock_cov_for_k(k_target: np.ndarray, kfit_mask: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """读取 EZmock PK 并对齐到目标 k 网格，返回用于拟合区间的 covariance mock。"""
    files = sorted(glob.glob(EZCOV_GLOB), key=parse_seed)
    rows = []
    skipped = 0

    for fp in files:
        try:
            arr = np.loadtxt(fp, comments="#")
            if arr.ndim != 2 or arr.shape[1] < 6:
                skipped += 1
                continue

            k_in = arr[:, 0]
            p0_in = arr[:, 5]

            # 与目标网格一致则直取，否则插值对齐
            if len(k_in) >= len(k_target) and np.allclose(k_in[:len(k_target)], k_target, atol=1e-10, rtol=0.0):
                p0 = p0_in[:len(k_target)]
            else:
                p0 = np.interp(k_target, k_in, p0_in)

            row = p0[kfit_mask]
            if np.all(np.isfinite(row)):
                rows.append(row)
            else:
                skipped += 1
        except Exception:
            skipped += 1

    if len(rows) < 20:
        raise RuntimeError(f"可用 EZmock 协方差样本过少: {len(rows)}")

    return np.asarray(rows, dtype=float), len(files), skipped


def fit_best_pk_with_external_cov(data: m5.MockData) -> Dict[str, object]:
    """使用 fastPM 均值 + EZmock covariance 做 PK best-fit。"""
    kcen = data.kcen
    p0_mean = data.p0_mean

    # 拟合口径：k<=0.08，且保留有限正方差 bin（用 fastPM 的 std 仅做质量过滤）
    mask = (kcen <= PK_FIT_KMAX)
    mask &= np.isfinite(data.p0_std) & (data.p0_std > 0)

    if np.sum(mask) < 5:
        raise RuntimeError(f"{data.tag}: 可用拟合点过少")

    kfit = kcen[mask]
    p0_fit = p0_mean[mask]
    cov_rows, n_all, n_skip = load_ezmock_cov_for_k(k_target=kcen, kfit_mask=mask)

    cosmo = m5.build_fiducial_cosmology()
    template = m5.FixedPowerSpectrumTemplate(z=m5.UNIT_Z, fiducial=cosmo)
    theory = m5.PNGTracerPowerSpectrumMultipoles(template=template, mode="b-p")

    theory.init.params["p"].update(fixed=True, value=P_FIXED)
    theory.init.params["sn0"].update(fixed=True, value=m5.FIXED_SN0)
    theory.init.params["sigmas"].update(fixed=False, value=m5.FIXED_SIGMAS)

    dk = float(np.median(np.diff(kfit)))
    observable = m5.TracerPowerSpectrumMultipolesObservable(
        data=p0_fit,
        covariance=[row for row in cov_rows],
        klim={0: [float(kfit.min()), float(kfit.max()), dk]},
        k=kfit,
        ells=[0],
        theory=theory,
    )
    likelihood = m5.ObservablesGaussianLikelihood(observables=[observable])
    _ = likelihood()

    likelihood.all_params["p"].update(fixed=True, value=P_FIXED)
    likelihood.all_params["sn0"].update(fixed=True, value=m5.FIXED_SN0)
    likelihood.all_params["sigmas"].update(fixed=False, value=m5.FIXED_SIGMAS)

    profiler = m5.MinuitProfiler(likelihood, seed=m5.MINUIT_SEED)
    profiles = profiler.maximize(niterations=m5.MINUIT_NITER)
    bestfit = m5.convert_bestfit_to_float_dict(profiles.bestfit.choice(input=True))

    # Minuit 误差（用于图上标注 value ± sigma）
    # desilike 的 profiles.error 是 Samples，参数项可能是长度>1的数组；
    # 这里取绝对值中位数作为稳健的 1σ 代表值，避免单点异常。
    besterr: Dict[str, float] = {}
    if hasattr(profiles, "error"):
        try:
            err_samples = profiles.error
            for name in err_samples.names():
                vals = np.asarray(err_samples[name], dtype=float).ravel()
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    besterr[name] = float(np.nanmedian(np.abs(vals)))
        except Exception:
            besterr = {}

    return {
        "bestfit": bestfit,
        "besterr": besterr,
        "kfit": kfit,
        "n_ezcov_all": int(n_all),
        "n_ezcov_used": int(cov_rows.shape[0]),
        "n_ezcov_skip": int(n_skip),
    }


def metric_range(s: np.ndarray, xi_data: np.ndarray, xi_std: np.ndarray, xi_model: np.ndarray, rmin: float) -> Dict[str, float]:
    m = s >= float(rmin)
    r2d = s**2 * xi_data
    r2m = s**2 * xi_model
    r2s = np.maximum(s**2 * xi_std, 1e-12)
    res = (r2d[m] - r2m[m]) / r2s[m]
    return {
        "mean_abs_sigma": float(np.mean(np.abs(res))),
        "chi2_ndof": float(np.mean(res**2)),
        "mean_sigma": float(np.mean(res)),
        "max_abs_sigma": float(np.max(np.abs(res))),
        "nbin": int(np.sum(m)),
    }


def fmt(m: Dict[str, float]) -> str:
    return (
        f"mean|Δ/σ|={m['mean_abs_sigma']:.4f}, chi2/ndof={m['chi2_ndof']:.4f}, "
        f"mean(Δ/σ)={m['mean_sigma']:.4f}, max|Δ/σ|={m['max_abs_sigma']:.4f}, Nbin={int(m['nbin'])}"
    )


def main() -> None:
    old_p = float(m5.FIXED_P)
    old_pkfit = float(m5.PK_FIT_KMAX)
    m5.FIXED_P = P_FIXED
    m5.PK_FIT_KMAX = PK_FIT_KMAX

    try:
        # 1) 读取数据
        d100 = m5.load_mock_data("3gpc_fnl100", PK_3GPC_FNL100_GLOB, PCF_3GPC_FNL100_GLOB, RID_MIN, RID_MAX)
        d0 = m5.load_mock_data("3gpc_fnl0", PK_3GPC_FNL0_GLOB, PCF_3GPC_FNL0_GLOB, RID_MIN, RID_MAX)

        # 2) 用 EZmock 协方差拟合 best-fit
        fit100 = fit_best_pk_with_external_cov(d100)
        fit0 = fit_best_pk_with_external_cov(d0)
        bf100 = fit100["bestfit"]
        be100 = fit100.get("besterr", {})
        bf0 = fit0["bestfit"]
        be0 = fit0.get("besterr", {})

        # 3) 构建理论 P0
        k_grid = np.geomspace(v14.KMIN_GLOBAL / m5.FFTLOG_PADDING, v14.KMAX_INT * m5.FFTLOG_PADDING, m5.FFTLOG_N)
        p100 = m5.build_theory_p0(k_grid, bf100)
        p0 = m5.build_theory_p0(k_grid, bf0)

        bf100_ref0 = dict(bf100)
        bf100_ref0["fnl_loc"] = 0.0
        p100_ref0 = m5.build_theory_p0(k_grid, bf100_ref0)

        # 4) 2PCF 建模：baseline / best window / conservative window
        xi100_base = v14.evaluate_unwindowed_3gpc(d100.scen, k_grid, p100, kmin=v14.K_FUND_3GPC)
        xi0_base = v14.evaluate_unwindowed_3gpc(d0.scen, k_grid, p0, kmin=v14.K_FUND_3GPC)

        xi100_best = v14.evaluate_testb_3gpc_single_window(d100.scen, k_grid, p100, p100_ref0, BEST_WINDOW_TYPE, BEST_WINDOW_PARAM)
        xi0_best = v14.evaluate_testb_3gpc_single_window(d0.scen, k_grid, p0, p0, BEST_WINDOW_TYPE, BEST_WINDOW_PARAM)

        xi100_ref = v14.evaluate_testb_3gpc_single_window(d100.scen, k_grid, p100, p100_ref0, REF_WINDOW_TYPE, REF_WINDOW_PARAM)
        xi0_ref = v14.evaluate_testb_3gpc_single_window(d0.scen, k_grid, p0, p0, REF_WINDOW_TYPE, REF_WINDOW_PARAM)

        # 5) 指标
        m100_b_150 = metric_range(d100.scen, d100.xi_mean, d100.xi_std, xi100_base, 150.0)
        m100_w_150 = metric_range(d100.scen, d100.xi_mean, d100.xi_std, xi100_best, 150.0)
        m100_r_150 = metric_range(d100.scen, d100.xi_mean, d100.xi_std, xi100_ref, 150.0)

        m0_b_150 = metric_range(d0.scen, d0.xi_mean, d0.xi_std, xi0_base, 150.0)
        m0_w_150 = metric_range(d0.scen, d0.xi_mean, d0.xi_std, xi0_best, 150.0)
        m0_r_150 = metric_range(d0.scen, d0.xi_mean, d0.xi_std, xi0_ref, 150.0)

        m100_b_200 = metric_range(d100.scen, d100.xi_mean, d100.xi_std, xi100_base, 200.0)
        m100_w_200 = metric_range(d100.scen, d100.xi_mean, d100.xi_std, xi100_best, 200.0)
        m0_b_200 = metric_range(d0.scen, d0.xi_mean, d0.xi_std, xi0_base, 200.0)
        m0_w_200 = metric_range(d0.scen, d0.xi_mean, d0.xi_std, xi0_best, 200.0)

        # 6) 画图
        fig, axes = plt.subplots(2, 1, figsize=(10.0, 9.2), sharex=True)

        ax = axes[0]
        ax.errorbar(d100.scen, d100.scen**2 * d100.xi_mean, yerr=d100.scen**2 * d100.xi_std,
                    fmt='o--', ms=3.0, capsize=2, color='black', label='Measured mean fnl100')
        ax.plot(d100.scen, d100.scen**2 * xi100_base, ':', lw=1.8, color='tab:blue', label=f'Baseline ({m100_b_150["mean_abs_sigma"]:.3f})')
        ax.plot(d100.scen, d100.scen**2 * xi100_best, '-', lw=2.1, color='tab:red', label=f'Best window exp_power({BEST_WINDOW_PARAM:g}) ({m100_w_150["mean_abs_sigma"]:.3f})')
        ax.plot(d100.scen, d100.scen**2 * xi100_ref, '--', lw=1.8, color='tab:green', label=f'Ref window exp_power(12) ({m100_r_150["mean_abs_sigma"]:.3f})')
        ax.axvline(150.0, color='gray', ls='--', lw=1.0, alpha=0.8)
        ax.axvline(200.0, color='gray', ls=':', lw=1.0, alpha=0.9)
        ax.set_ylabel(r'$r^2\xi_0(r)$')
        ax.set_title('3Gpc fnl100 (fit k<=0.08, covariance from EZmock)')
        txt100 = (
            f"best-fit PK\n"
            f"fnl_loc={bf100['fnl_loc']:.2f} ± {be100.get('fnl_loc', float('nan')):.2f}\n"
            f"b1={bf100['b1']:.4f} ± {be100.get('b1', float('nan')):.4f}\n"
            f"sigmas={bf100['sigmas']:.4f} ± {be100.get('sigmas', float('nan')):.4f}\n"
            f"kmax_fit={PK_FIT_KMAX:.3f}"
        )
        ax.text(0.02, 0.98, txt100, transform=ax.transAxes, va='top', ha='left', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.82, edgecolor='0.4'))
        ax.legend(fontsize=8.2)

        ax = axes[1]
        ax.errorbar(d0.scen, d0.scen**2 * d0.xi_mean, yerr=d0.scen**2 * d0.xi_std,
                    fmt='o--', ms=3.0, capsize=2, color='black', label='Measured mean fnl0')
        ax.plot(d0.scen, d0.scen**2 * xi0_base, ':', lw=1.8, color='tab:blue', label=f'Baseline ({m0_b_150["mean_abs_sigma"]:.3f})')
        ax.plot(d0.scen, d0.scen**2 * xi0_best, '-', lw=2.1, color='tab:red', label=f'Best window exp_power({BEST_WINDOW_PARAM:g}) ({m0_w_150["mean_abs_sigma"]:.3f})')
        ax.plot(d0.scen, d0.scen**2 * xi0_ref, '--', lw=1.8, color='tab:green', label=f'Ref window exp_power(12) ({m0_r_150["mean_abs_sigma"]:.3f})')
        ax.axvline(150.0, color='gray', ls='--', lw=1.0, alpha=0.8)
        ax.axvline(200.0, color='gray', ls=':', lw=1.0, alpha=0.9)
        ax.set_xlabel(r'$r\,[\mathrm{Mpc}/h]$')
        ax.set_ylabel(r'$r^2\xi_0(r)$')
        ax.set_title('3Gpc fnl0 (fit k<=0.08, covariance from EZmock)')
        txt0 = (
            f"best-fit PK\n"
            f"fnl_loc={bf0['fnl_loc']:.2f} ± {be0.get('fnl_loc', float('nan')):.2f}\n"
            f"b1={bf0['b1']:.4f} ± {be0.get('b1', float('nan')):.4f}\n"
            f"sigmas={bf0['sigmas']:.4f} ± {be0.get('sigmas', float('nan')):.4f}\n"
            f"kmax_fit={PK_FIT_KMAX:.3f}"
        )
        ax.text(0.02, 0.98, txt0, transform=ax.transAxes, va='top', ha='left', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.82, edgecolor='0.4'))
        ax.legend(fontsize=8.2)

        fig.tight_layout()
        fig.savefig(OUT_PNG, bbox_inches='tight')
        plt.close(fig)

        # 7) summary
        with open(OUT_TXT, 'w', encoding='utf-8') as f:
            f.write('任务5 v22：3Gpc EZmock协方差 + 最佳窗口建模检查\n')
            f.write('=============================================\n\n')
            f.write('设置:\n')
            f.write(f'- PK_FIT_KMAX={PK_FIT_KMAX:.3f}\n')
            f.write(f'- p fixed={P_FIXED:.2f}\n')
            f.write('- covariance: EZmock pk samples from cov_mock\n')
            f.write(f'- window(best): {BEST_WINDOW_TYPE}({BEST_WINDOW_PARAM:g})\n')
            f.write(f'- window(ref): {REF_WINDOW_TYPE}({REF_WINDOW_PARAM:g})\n\n')

            f.write('[EZmock covariance usage]\n')
            f.write(f"- fnl100: all={fit100['n_ezcov_all']}, used={fit100['n_ezcov_used']}, skipped={fit100['n_ezcov_skip']}\n")
            f.write(f"- fnl0  : all={fit0['n_ezcov_all']}, used={fit0['n_ezcov_used']}, skipped={fit0['n_ezcov_skip']}\n\n")

            f.write('[best-fit params]\n')
            f.write(f"- fnl100: fnl_loc={bf100['fnl_loc']:.6f}, b1={bf100['b1']:.6f}, sigmas={bf100['sigmas']:.6f}\n")
            f.write(f"- fnl0  : fnl_loc={bf0['fnl_loc']:.6f}, b1={bf0['b1']:.6f}, sigmas={bf0['sigmas']:.6f}\n\n")
            f.write('[best-fit 1σ errors]\n')
            f.write(f"- fnl100: d_fnl_loc={be100.get('fnl_loc', float('nan')):.6f}, d_b1={be100.get('b1', float('nan')):.6f}, d_sigmas={be100.get('sigmas', float('nan')):.6f}\n")
            f.write(f"- fnl0  : d_fnl_loc={be0.get('fnl_loc', float('nan')):.6f}, d_b1={be0.get('b1', float('nan')):.6f}, d_sigmas={be0.get('sigmas', float('nan')):.6f}\n\n")

            f.write('[r>=150]\n')
            f.write(f'- fnl100 baseline: {fmt(m100_b_150)}\n')
            f.write(f'- fnl100 bestwin : {fmt(m100_w_150)}\n')
            f.write(f'- fnl100 refwin  : {fmt(m100_r_150)}\n')
            f.write(f'- fnl0   baseline: {fmt(m0_b_150)}\n')
            f.write(f'- fnl0   bestwin : {fmt(m0_w_150)}\n')
            f.write(f'- fnl0   refwin  : {fmt(m0_r_150)}\n\n')

            f.write('[r>=200]\n')
            f.write(f'- fnl100 baseline: {fmt(m100_b_200)}\n')
            f.write(f'- fnl100 bestwin : {fmt(m100_w_200)}\n')
            f.write(f'- fnl0   baseline: {fmt(m0_b_200)}\n')
            f.write(f'- fnl0   bestwin : {fmt(m0_w_200)}\n')

        print('[INFO] done')
        print(f"[INFO] ezcov used fnl100={fit100['n_ezcov_used']}, fnl0={fit0['n_ezcov_used']}")
        print('[INFO] r>=150 fnl100 baseline/best/ref =',
              f"{m100_b_150['mean_abs_sigma']:.4f}/{m100_w_150['mean_abs_sigma']:.4f}/{m100_r_150['mean_abs_sigma']:.4f}")
        print('[INFO] r>=150 fnl0 baseline/best/ref =',
              f"{m0_b_150['mean_abs_sigma']:.4f}/{m0_w_150['mean_abs_sigma']:.4f}/{m0_r_150['mean_abs_sigma']:.4f}")
        print(f'[INFO] fig: {OUT_PNG}')
        print(f'[INFO] summary: {OUT_TXT}')

    finally:
        m5.FIXED_P = old_p
        m5.PK_FIT_KMAX = old_pkfit


if __name__ == '__main__':
    main()
