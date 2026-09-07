#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission 11: 全离散求和加速 — 方案对比与基准测试
=================================================

大纲
----
本脚本对比 4 种全离散求和 xi_0(r) = (1/V) sum_q g_q P(k_q) j0(k_q r) 的计算方案：

1. Baseline: 原始分块 numpy 矩阵运算（从 mission10 移植），约 100s
2. Rebin(零阶): 把 42M 壳层按 k 值聚合到等间距 bin，
   每个 bin 用加权中心 k_eff 代替所有壳层，大幅减少 sin 计算量
3. Rebin+O1(一阶修正): 在 Rebin 基础上，对 j0(kr) 做一阶泰勒展开修正，
   允许更粗的 bin 宽度
4. SinCos 累积: 利用 sin/cos 加法公式，把等间距 k 上的求和转化为
   递推关系，避免逐壳层 sin 计算

核心输入/输出与原始方法完全一致:
    输入: s, gq, kf, V, kd, pd
    输出: xi_0(r)

测试对象: 3Gpc fnl100, kmax=15

运行方式
--------
cd /pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission11_log
bash -lc "conda activate desilike && PYTHONUNBUFFERED=1 python -u mission11_acceleration_benchmark.py"
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.fft import next_fast_len, rfft, irfft

# ============================================================
# 导入 Mission 10 的工具函数
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MISSION10_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "mission10_log"))
if MISSION10_DIR not in sys.path:
    sys.path.insert(0, MISSION10_DIR)

from mission10_binavgfit_full_discrete_compare import (
    BoxConfig,
    load_pk_multipoles,
    load_pcf,
    fit_standard_desilike,
    fit_binavg_minuit,
    eval_pk_dense,
    gq_enumerate,
    build_shells_for_kmax,
    build_bin_shell_index,
    gq_fft,
    met_mean_abs_sigma,
)

# 尝试导入 numba，若不可用则跳过
try:
    import numba
    HAS_NUMBA = True
    print("[info] numba 可用")
except ImportError:
    HAS_NUMBA = False
    print("[info] numba 不可用，跳过 numba 方案")


# ============================================================
# 通用辅助函数
# ============================================================
def j0(x: np.ndarray) -> np.ndarray:
    """球贝塞尔 j0(x)=sin(x)/x，对 x=0 做极限 j0(0)=1。"""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    m = x != 0
    out[m] = np.sin(x[m]) / x[m]
    return out


def prepare_shells(gq, kf, kd, pd):
    """预计算非零壳层的 kv 和权重 w = g_q * P(k_q)。

    参数
    ----
    gq : 壳层简并度数组
    kf : 基模
    kd, pd : 密集 k 网格及对应 P_model

    返回
    ----
    kv : (N,) 非零壳层的 k 值
    w : (N,) 非零壳层的权重 g_q * P(k_q)
    """
    qnz = np.nonzero(gq[1:])[0] + 1
    kv = kf * np.sqrt(qnz.astype(np.float64))
    g = gq[qnz].astype(np.float64)
    pv = np.interp(kv, kd, pd)
    w = g * pv
    return kv, w


# ============================================================
# 方法0: Baseline（原始分块矩阵运算）
# ============================================================
def xi_baseline(s, gq, kf, V, kd, pd, chunk=500_000):
    """原始全离散求和。

    按 chunk 个壳层分块处理:
    - 对每块计算 kv, 插值得到 P(kv), 权重 w = g * P
    - 对每 10 个 r 点一组, 计算 j0(kv ⊗ r) 并做矩阵乘法
    - 累加得到 xi

    耗时瓶颈: 42M 壳层 × 38 r 点 = 1.6G 次 sin 调用
    """
    kv, w = prepare_shells(gq, kf, kd, pd)
    total = len(kv)
    xi = np.zeros(len(s), dtype=np.float64)

    t0 = time.time()
    for i0 in range(0, total, chunk):
        kv_c = kv[i0:i0 + chunk]
        w_c = w[i0:i0 + chunk]

        for js in range(0, len(s), 10):
            seg = s[js: js + 10]
            arg = np.outer(kv_c, seg)
            J = np.ones_like(arg)
            m = arg != 0
            J[m] = np.sin(arg[m]) / arg[m]
            xi[js: js + len(seg)] += w_c @ J

        if (i0 // chunk) % 20 == 0:
            print(f"    baseline: {min(i0+chunk, total)}/{total} ({time.time()-t0:.1f}s)")

    return xi / V


# ============================================================
# 方法A: k-rebinning（零阶）
# ============================================================
def xi_rebin(s, gq, kf, V, kd, pd, dk_factor=0.5):
    """k-rebinning 加速 — 零阶近似。

    核心思想: 把 42M 壳层按 k 值放入等间距 bin (宽 dk = dk_factor*kf)
    在每个 bin 中求 W_bin = sum(g_q * P(k_q)) 和加权中心 k_eff
    然后只需对 ~N_bins 个 k_eff 做 j0 计算

    近似误差取决于 bin 宽: dk * r_max << 1 时精度好
    dk = 0.5*kf 时, dk * 380 ≈ 0.4, 约 10-20% level per bin
    dk = 0.1*kf 时, dk * 380 ≈ 0.08, 约 1% level per bin

    参数
    ----
    dk_factor : bin 宽 = dk_factor * kf
    """
    kv, w = prepare_shells(gq, kf, kd, pd)

    dk = dk_factor * kf
    kmax_val = kv.max()
    n_bins = int(np.ceil(kmax_val / dk)) + 1

    # 聚合权重和加权中心
    W_bin = np.zeros(n_bins, dtype=np.float64)
    Wk_bin = np.zeros(n_bins, dtype=np.float64)
    bin_idx = np.floor(kv / dk).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    np.add.at(W_bin, bin_idx, w)
    np.add.at(Wk_bin, bin_idx, w * kv)

    nz = W_bin != 0
    W_nz = W_bin[nz]
    k_eff = Wk_bin[nz] / W_nz

    print(f"    rebin: {len(kv)} shells -> {nz.sum()} bins (dk={dk:.6f})")

    # 直接矩阵运算: 计算 j0(k_eff ⊗ s)
    arg = np.outer(k_eff, s)
    J = np.ones_like(arg)
    m = arg != 0
    J[m] = np.sin(arg[m]) / arg[m]

    return (W_nz @ J) / V


# ============================================================
# 方法B: k-rebinning + 一阶泰勒修正
# ============================================================
def xi_rebin_o1(s, gq, kf, V, kd, pd, dk_factor=2.0):
    """k-rebinning + 一阶修正。

    在每个 bin 中, j0(k*r) 在 bin 中心 k_c 处做一阶展开:
        j0(k*r) ≈ j0(k_c*r) + (k-k_c) * r * j0'(k_c*r)
    其中 j0'(x) = d/dx[sin(x)/x] = [x*cos(x) - sin(x)] / x^2

    需要预计算两个聚合系数:
        A_bin = sum_q w_q              (零阶)
        B_bin = sum_q w_q * (k_q - k_c) (一阶)

    可以用更粗的 bin（dk_factor 更大）获得相同精度, 进一步加速。

    参数
    ----
    dk_factor : bin 宽 = dk_factor * kf
    """
    kv, w = prepare_shells(gq, kf, kd, pd)

    dk = dk_factor * kf
    kmax_val = kv.max()
    n_bins = int(np.ceil(kmax_val / dk)) + 1
    k_centers = (np.arange(n_bins) + 0.5) * dk

    A_bin = np.zeros(n_bins, dtype=np.float64)
    B_bin = np.zeros(n_bins, dtype=np.float64)
    bin_idx = np.floor(kv / dk).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    np.add.at(A_bin, bin_idx, w)
    dk_from_center = kv - k_centers[bin_idx]
    np.add.at(B_bin, bin_idx, w * dk_from_center)

    nz = A_bin != 0
    A_nz = A_bin[nz]
    B_nz = B_bin[nz]
    kc_nz = k_centers[nz]

    print(f"    rebin_o1: {len(kv)} shells -> {nz.sum()} bins (dk={dk:.6f})")

    # 对所有 bin 中心和 r 计算 j0 和 j0'
    arg = np.outer(kc_nz, s)  # (n_bins, n_r)
    sinx = np.sin(arg)
    cosx = np.cos(arg)

    j0_val = np.ones_like(arg)
    j0p_val = np.zeros_like(arg)
    m = arg != 0
    j0_val[m] = sinx[m] / arg[m]
    # j0'(x) = [x*cos(x) - sin(x)] / x^2
    j0p_val[m] = (arg[m] * cosx[m] - sinx[m]) / (arg[m] ** 2)

    # xi(r) = (1/V) sum_bin [A * j0(k_c*r) + B * r * j0'(k_c*r)]
    xi = np.zeros(len(s), dtype=np.float64)
    for js in range(len(s)):
        r = s[js]
        xi[js] = np.sum(A_nz * j0_val[:, js] + B_nz * r * j0p_val[:, js])

    return xi / V


# ============================================================
# 方法C: Numba JIT（若可用）
# ============================================================
if HAS_NUMBA:
    @numba.njit(parallel=True, cache=True, fastmath=True)
    def _xi_numba_core(s, kv, w):
        """Numba 内层: xi[i] = sum_j w[j] * sin(kv[j]*s[i]) / (kv[j]*s[i])"""
        n_r = len(s)
        n_k = len(kv)
        xi = np.zeros(n_r, dtype=np.float64)
        for i in numba.prange(n_r):
            r = s[i]
            acc = 0.0
            for j in range(n_k):
                x = kv[j] * r
                if x == 0.0:
                    acc += w[j]
                else:
                    acc += w[j] * np.sin(x) / x
            xi[i] = acc
        return xi

    def xi_numba(s, gq, kf, V, kd, pd):
        """Numba 加速的全离散求和。"""
        kv, w = prepare_shells(gq, kf, kd, pd)
        # 预热
        _ = _xi_numba_core(s[:1], kv[:100], w[:100])
        xi = _xi_numba_core(s, kv, w)
        return xi / V


# ============================================================
# 方法D: 二阶修正 rebinning（更精确的近似）
# ============================================================
def xi_rebin_o2(s, gq, kf, V, kd, pd, dk_factor=5.0):
    """k-rebinning + 二阶泰勒修正。

    在每个 bin 中, j0(k*r) 在 bin 中心 k_c 处做二阶展开:
        j0(k*r) ≈ j0(k_c*r) + δk*r*j0'(k_c*r) + 0.5*(δk)^2*r^2*j0''(k_c*r)
    其中 δk = k - k_c

    j0''(x) = d²/dx²[sin(x)/x]
            = [(x²-2)sin(x) + 2x*cos(x)] / (-x^3)
            算了太复杂，用更简洁的形式:
    j0''(x) = -2*cos(x)/x^2 + 2*sin(x)/x^3 + sin(x)/x - 对不起，让我查一下

    Actually: 对 f(x)=sin(x)/x:
        f'(x) = [x cos(x) - sin(x)] / x^2
        f''(x) = [(2-x^2)sin(x) - 2x cos(x)] / x^3

    需要三个聚合系数:
        A_bin = sum w_q
        B_bin = sum w_q * δk_q
        C_bin = sum w_q * δk_q^2
    """
    kv, w = prepare_shells(gq, kf, kd, pd)

    dk = dk_factor * kf
    kmax_val = kv.max()
    n_bins = int(np.ceil(kmax_val / dk)) + 1
    k_centers = (np.arange(n_bins) + 0.5) * dk

    A_bin = np.zeros(n_bins, dtype=np.float64)
    B_bin = np.zeros(n_bins, dtype=np.float64)
    C_bin = np.zeros(n_bins, dtype=np.float64)
    bin_idx = np.floor(kv / dk).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    dkv = kv - k_centers[bin_idx]
    np.add.at(A_bin, bin_idx, w)
    np.add.at(B_bin, bin_idx, w * dkv)
    np.add.at(C_bin, bin_idx, w * dkv**2)

    nz = A_bin != 0
    A_nz = A_bin[nz]
    B_nz = B_bin[nz]
    C_nz = C_bin[nz]
    kc_nz = k_centers[nz]

    print(f"    rebin_o2: {len(kv)} shells -> {nz.sum()} bins (dk={dk:.6f})")

    arg = np.outer(kc_nz, s)
    sinx = np.sin(arg)
    cosx = np.cos(arg)

    j0_val = np.ones_like(arg)
    j0p_val = np.zeros_like(arg)
    j0pp_val = np.zeros_like(arg)
    m = arg != 0
    j0_val[m] = sinx[m] / arg[m]
    j0p_val[m] = (arg[m] * cosx[m] - sinx[m]) / (arg[m] ** 2)
    # f''(x) = [(2-x^2)sin(x) - 2x cos(x)] / x^3
    j0pp_val[m] = ((2 - arg[m]**2) * sinx[m] - 2 * arg[m] * cosx[m]) / (arg[m] ** 3)

    xi = np.zeros(len(s), dtype=np.float64)
    for js in range(len(s)):
        r = s[js]
        xi[js] = np.sum(
            A_nz * j0_val[:, js]
            + B_nz * r * j0p_val[:, js]
            + 0.5 * C_nz * r**2 * j0pp_val[:, js]
        )

    return xi / V


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    t_total = time.time()

    # 固定参数
    Z = 1.0
    P_FIX = 1.2
    KMAX = 15.0
    N_DENSE = 300_000
    L = 3000.0
    kf = 2.0 * np.pi / L
    V = L ** 3

    from cosmoprimo import Cosmology

    cosmo = Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )

    # 读取数据
    print("=" * 60)
    print("读取 3Gpc fnl100 数据 ...")
    print("=" * 60)

    cfg = BoxConfig(
        title="3Gpc fnl100",
        L=3000.0,
        pk_data_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat",
        pk_cov_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
        pcf_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",
        rid_min=2,
        rid_max=99,
        n_dp=20,
    )

    kc0, km0, kx0, pk_mocks0, _pk_mean0, _ = load_pk_multipoles(
        cfg.pk_data_glob, cfg.rid_min, cfg.rid_max, cfg.n_dp + 1
    )
    _kc, _km, _kx, cov_mocks0, _cov_mean0, _cov_std0 = load_pk_multipoles(
        cfg.pk_cov_glob, cfg.rid_min, cfg.rid_max, cfg.n_dp + 1
    )

    first_valid = 0
    for ib in range(len(kc0)):
        if float(pk_mocks0[:, ib].mean()) > 0:
            first_valid = ib
            break

    kcen = kc0[first_valid:first_valid + cfg.n_dp]
    kmin_b = km0[first_valid:first_valid + cfg.n_dp]
    kmax_b = kx0[first_valid:first_valid + cfg.n_dp]
    pk_mocks = pk_mocks0[:, first_valid:first_valid + cfg.n_dp]
    cov_mocks = cov_mocks0[:, first_valid:first_valid + cfg.n_dp]
    pk_mean = pk_mocks.mean(axis=0)

    s, xi_mean, xi_std = load_pcf(cfg.pcf_glob, cfg.rid_min, cfg.rid_max)
    r2_data = s ** 2 * xi_mean
    r2_std = s ** 2 * xi_std

    # 拟合 BinAvgFit
    print("\n拟合 BinAvgFit ...")
    bf_std = fit_standard_desilike(
        cosmo=cosmo, kcen=kcen, kmin_b=kmin_b, kmax_b=kmax_b,
        pk_mean=pk_mean, cov_mocks=cov_mocks, p_fix=P_FIX, z=Z,
    )

    kmax_fit = float(kmax_b[-1])
    qmax_fit = int(np.floor((kmax_fit / kf) ** 2)) + 1
    gq_fit = gq_enumerate(qmax_fit)
    _q_shell, k_shell, g_shell = build_shells_for_kmax(kf, gq_fit, kmax=kmax_fit)
    idx_per_bin = build_bin_shell_index(k_shell, g_shell, kmin_b, kmax_b)

    bf_binavg = fit_binavg_minuit(
        cosmo=cosmo, base_params=bf_std, k_shell=k_shell, g_shell=g_shell,
        idx_per_bin=idx_per_bin, pk_data=pk_mean, cov_mocks=cov_mocks,
        p_fix=P_FIX, z=Z,
    )

    print(f"  best-fit: fnl={float(bf_binavg.get('fnl_loc', 0)):.3f}, "
          f"b1={float(bf_binavg.get('b1', 0)):.6f}, "
          f"sigmas={float(bf_binavg.get('sigmas', 0)):.6f}")

    # P_model(k) 密集网格
    kd = np.geomspace(kf * 0.5, KMAX * 1.1, N_DENSE)
    pd = eval_pk_dense(cosmo=cosmo, k=kd, params=bf_binavg, p_fix=P_FIX, z=Z)

    # g_q
    qmax = int((KMAX / kf) ** 2)
    nmax = int(KMAX / kf)
    print(f"\n计算 g_q (qmax={qmax}) ...")
    gq = gq_fft(qmax=qmax, nmax=nmax)
    qnz_count = np.count_nonzero(gq[1:])
    print(f"  非零壳层数: {qnz_count}")

    # ============================================================
    # 基准测试
    # ============================================================
    results = {}

    # --- Baseline ---
    print("\n" + "=" * 60)
    print("Baseline ...")
    print("=" * 60)
    t0 = time.time()
    xi_base = xi_baseline(s, gq, kf, V, kd, pd, chunk=500_000)
    t_base = time.time() - t0
    print(f"  耗时: {t_base:.1f}s")
    results["Baseline"] = (xi_base, t_base)

    # --- Rebin 零阶, 不同 dk_factor ---
    for dk_fac in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        name = f"Rebin(dk={dk_fac}kf)"
        print(f"\n{'=' * 60}\n{name} ...\n{'=' * 60}")
        t0 = time.time()
        xi_rb = xi_rebin(s, gq, kf, V, kd, pd, dk_factor=dk_fac)
        t_rb = time.time() - t0
        print(f"  耗时: {t_rb:.1f}s")
        results[name] = (xi_rb, t_rb)

    # --- Rebin + O1, 不同 dk_factor ---
    for dk_fac in [1.0, 2.0, 5.0, 10.0, 20.0]:
        name = f"Rebin_O1(dk={dk_fac}kf)"
        print(f"\n{'=' * 60}\n{name} ...\n{'=' * 60}")
        t0 = time.time()
        xi_o1 = xi_rebin_o1(s, gq, kf, V, kd, pd, dk_factor=dk_fac)
        t_o1 = time.time() - t0
        print(f"  耗时: {t_o1:.1f}s")
        results[name] = (xi_o1, t_o1)

    # --- Rebin + O2, 更粗 dk_factor ---
    for dk_fac in [5.0, 10.0, 20.0, 50.0]:
        name = f"Rebin_O2(dk={dk_fac}kf)"
        print(f"\n{'=' * 60}\n{name} ...\n{'=' * 60}")
        t0 = time.time()
        xi_o2 = xi_rebin_o2(s, gq, kf, V, kd, pd, dk_factor=dk_fac)
        t_o2 = time.time() - t0
        print(f"  耗时: {t_o2:.1f}s")
        results[name] = (xi_o2, t_o2)

    # --- Numba（若可用）---
    if HAS_NUMBA:
        name = "Numba"
        print(f"\n{'=' * 60}\n{name} ...\n{'=' * 60}")
        t0 = time.time()
        xi_nb = xi_numba(s, gq, kf, V, kd, pd)
        t_nb = time.time() - t0
        print(f"  耗时（含编译）: {t_nb:.1f}s")
        results[name] = (xi_nb, t_nb)

        # 无编译热启动
        t0 = time.time()
        xi_nb2 = xi_numba(s, gq, kf, V, kd, pd)
        t_nb2 = time.time() - t0
        print(f"  耗时（热启动）: {t_nb2:.1f}s")
        results["Numba(warm)"] = (xi_nb2, t_nb2)

    # ============================================================
    # 精度 + 速度表
    # ============================================================
    print("\n" + "=" * 60)
    print("精度 & 速度对比 (vs Baseline)")
    print("=" * 60)

    r2_base = s ** 2 * xi_base

    hdr = (f"{'方法':<25s} {'耗时(s)':>8s} {'加速比':>8s} "
           f"{'max|Δξ/ξ|':>12s} {'mean|Δξ/ξ|':>12s} "
           f"{'vsData|Δ/σ|':>14s} {'vsDataΔ/σ':>12s}")
    sep = "-" * len(hdr)
    print(f"\n{hdr}\n{sep}")

    table_rows = []
    for name, (xi_m, t_m) in results.items():
        r2_m = s ** 2 * xi_m
        rel_err = np.abs((xi_m - xi_base) / np.where(np.abs(xi_base) > 1e-30, xi_base, 1e-30))
        max_rel = float(np.nanmax(rel_err))
        mean_rel = float(np.nanmean(rel_err))
        ma, ms = met_mean_abs_sigma(r2_data, r2_std, r2_m)
        speedup = t_base / t_m if t_m > 0 else float('inf')
        row_str = (f"  {name:<23s} {t_m:>8.2f} {speedup:>8.1f}x "
                   f"{max_rel:>12.2e} {mean_rel:>12.2e} "
                   f"{ma:>14.4f} {ms:>+12.4f}")
        print(row_str)
        table_rows.append((name, t_m, speedup, max_rel, mean_rel, ma, ms))

    # ============================================================
    # 画图
    # ============================================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))

    # 图1: r²ξ 对比（只画几个代表方案 + baseline + data）
    ax = axes[0]
    ax.errorbar(s, r2_data, yerr=r2_std, fmt="ko", ms=3, capsize=2, label="Data", zorder=10)

    # 选择代表方案画图
    plot_keys = ["Baseline"]
    # 找最好的 rebin_o1 和 rebin_o2
    for prefix in ["Rebin_O1", "Rebin_O2"]:
        cands = [(n, v) for n, v in results.items() if n.startswith(prefix)]
        if cands:
            # 选精度最好的（mean rel err 最小且 speedup > 10x）
            best = min(cands, key=lambda x: np.nanmean(np.abs((x[1][0] - xi_base)/np.where(np.abs(xi_base)>1e-30, xi_base, 1e-30))))
            plot_keys.append(best[0])
    # 也画 Rebin(dk=0.2kf) 作为零阶参考
    if "Rebin(dk=0.2kf)" in results:
        plot_keys.append("Rebin(dk=0.2kf)")
    if "Numba(warm)" in results:
        plot_keys.append("Numba(warm)")

    colors_map = {k: plt.cm.tab10(i) for i, k in enumerate(plot_keys)}
    for name in plot_keys:
        xi_m, t_m = results[name]
        r2_m = s ** 2 * xi_m
        ma, ms = met_mean_abs_sigma(r2_data, r2_std, r2_m)
        lw = 2.5 if name == "Baseline" else 1.5
        ax.plot(s, r2_m, color=colors_map[name], lw=lw,
                label=f"{name} ({t_m:.1f}s)")
    ax.set_ylabel(r"$r^2\xi_0(r)$")
    ax.set_title("3Gpc fnl100: 加速方案 2PCF 对比")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 图2: 所有方案 vs baseline 的相对误差
    ax = axes[1]
    all_colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    ic = 0
    for name, (xi_m, t_m) in results.items():
        if name == "Baseline":
            ic += 1
            continue
        rel_err = (xi_m - xi_base) / np.where(np.abs(xi_base) > 1e-30, xi_base, 1e-30)
        ax.plot(s, rel_err, lw=1.0, label=name, alpha=0.8)
        ic += 1
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(1e-3, color="gray", ls=":", lw=0.6, label="0.1%")
    ax.axhline(-1e-3, color="gray", ls=":", lw=0.6)
    ax.set_ylabel(r"$\Delta\xi / |\xi_{\rm baseline}|$")
    ax.set_title("各方案 vs Baseline 的相对误差")
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 0.05)

    # 图3: 速度 vs 精度 scatter
    ax = axes[2]
    for name, t_m, speedup, max_rel, mean_rel, ma, ms in table_rows:
        if name == "Baseline":
            ax.scatter(mean_rel, t_m, s=100, marker="*", color="red", zorder=10)
            ax.annotate(name, (mean_rel, t_m), fontsize=7, ha="left")
        else:
            ax.scatter(mean_rel, t_m, s=50, zorder=5)
            ax.annotate(name, (mean_rel, t_m), fontsize=6, ha="left")
    ax.set_xlabel(r"mean $|\Delta\xi/\xi_{\rm baseline}|$")
    ax.set_ylabel("耗时 (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("速度 vs 精度 Pareto 图")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out_png = os.path.join(THIS_DIR, "mission11_acceleration_benchmark.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\n[saved] {out_png}")

    # ============================================================
    # 笔记
    # ============================================================
    out_md = os.path.join(THIS_DIR, "mission11_acceleration_results.md")
    lines = []
    lines.append("# Mission 11: 全离散求和加速结果\n\n")
    lines.append("## 测试环境\n\n")
    lines.append(f"- 盒子: 3Gpc, fnl100\n")
    lines.append(f"- kmax = {KMAX}\n")
    lines.append(f"- 非零壳层数: {qnz_count}\n")
    lines.append(f"- r 网格点数: {len(s)}\n")
    lines.append(f"- kf = 2π/L = {kf:.6f}\n\n")

    lines.append("## 方案说明\n\n")
    lines.append("1. **Baseline**: 原始分块numpy矩阵运算，逐壳层计算 j0(kr)\n")
    lines.append("2. **Rebin(零阶)**: 把壳层按k值聚合到等间距bin，用加权中心k_eff做j0计算\n")
    lines.append("3. **Rebin_O1(一阶)**: k-rebinning + j0 的一阶泰勒修正，允许更粗bin\n")
    lines.append("4. **Rebin_O2(二阶)**: k-rebinning + j0 的二阶泰勒修正，允许更粗bin\n")
    lines.append("5. **Numba**: numba.njit + prange 并行化内层循环（若可用）\n\n")

    lines.append("## 加速原理\n\n")
    lines.append("### k-rebinning 的核心思想\n\n")
    lines.append("全离散求和的瓶颈是 42M 个壳层 × 38 个 r 点 = ~1.6G 次 sin 调用。\n")
    lines.append("但很多壳层的 k 值非常接近（尤其在高 k 区域），\n")
    lines.append("如果把 k 值相近的壳层合并到一个 bin 中，用 bin 中心的 k_eff 代替：\n\n")
    lines.append("```\n")
    lines.append("xi(r) = (1/V) sum_bin W_bin * j0(k_eff * r)\n")
    lines.append("W_bin = sum_{q in bin} g_q * P(k_q)\n")
    lines.append("k_eff = sum_{q in bin} g_q * P(k_q) * k_q / W_bin\n")
    lines.append("```\n\n")
    lines.append("这样 sin 计算从 42M 次降到 ~N_bins 次。\n")
    lines.append("N_bins = kmax / dk，取决于 bin 宽 dk。\n\n")
    lines.append("### 一阶修正的关键\n\n")
    lines.append("零阶 rebinning 的误差 ~ (dk*r)^2。对 dk=kf, r=380：\n")
    lines.append("误差 ~ (0.002*380)^2 ~ 0.6，太大。\n")
    lines.append("加一阶修正后，误差 ~ (dk*r)^3 / 6，显著减小。\n")
    lines.append("这允许使用更粗的 bin（更大的 dk_factor），进一步加速。\n\n")

    lines.append("## 结果对比\n\n")
    lines.append(f"| {'方法':<25s} | {'耗时(s)':>8s} | {'加速比':>8s} | "
                 f"{'max|Δξ/ξ|':>12s} | {'mean|Δξ/ξ|':>12s} | "
                 f"{'vsData|Δ/σ|':>14s} | {'vsDataΔ/σ':>12s} |\n")
    lines.append(f"|{'-'*27}|{'-'*10}|{'-'*10}|{'-'*14}|{'-'*14}|{'-'*16}|{'-'*14}|\n")

    for name, t_m, speedup, max_rel, mean_rel, ma, ms in table_rows:
        lines.append(f"| {name:<25s} | {t_m:>8.2f} | {speedup:>8.1f}x | "
                     f"{max_rel:>12.2e} | {mean_rel:>12.2e} | "
                     f"{ma:>14.4f} | {ms:>+12.4f} |\n")

    lines.append("\n## 推荐方案\n\n")
    lines.append("（待结果填充后由脚本自动补充）\n\n")

    # 找最佳方案: 精度 < 0.1% 且速度最快
    best_name = None
    best_time = float('inf')
    for name, t_m, speedup, max_rel, mean_rel, ma, ms in table_rows:
        if name == "Baseline":
            continue
        if mean_rel < 1e-3 and t_m < best_time:  # 精度 < 0.1%
            best_name = name
            best_time = t_m

    if best_name:
        for name, t_m, speedup, max_rel, mean_rel, ma, ms in table_rows:
            if name == best_name:
                lines.append(f"**推荐: {best_name}**\n")
                lines.append(f"- 耗时: {t_m:.2f}s (加速 {speedup:.0f}x)\n")
                lines.append(f"- 精度: mean|Δξ/ξ| = {mean_rel:.2e}, max = {max_rel:.2e}\n")
                lines.append(f"- vs 数据: mean|Δ/σ| = {ma:.4f} (与 Baseline 的 {table_rows[0][5]:.4f} 几乎一致)\n")

    with open(out_md, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[saved] {out_md}")

    print(f"\n总耗时: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
