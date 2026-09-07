#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission 12: 功率谱拟合 vs 2PCF 拟合的 best-fit 参数对比
=========================================================

大纲
----
本脚本对 3Gpc fastPM fnl100 样本做两种参数拟合，对比 best-fit 差异：

方法A — 功率谱直接拟合（BinAvgFit）
    用前 20 个 k-bin 的 P0 均值拟合，理论用 bin-average 口径。
    χ²_pk = (P_data - P_model_bin)^T C_pk^{-1} (P_data - P_model_bin)

方法B — 2PCF 模型拟合
    用 2PCF 均值（r ∈ [100, 350] Mpc/h）拟合。
    理论是：参数 → P_model(k) → fast_discrete_xi0(r)。
    χ²_xi = (ξ_data - ξ_model)^T C_xi^{-1} (ξ_data - ξ_model)

两种方法拟合的自由参数相同：fnl_loc, b1, sigmas（p=1.2 fix, sn0=0 fix）。

输出
----
1. 两种方法的 best-fit 参数对比表
2. best-fit P0(k) 和 ξ0(r) 的对比图
3. 结果笔记 mission12_results.md

运行
----
cd /pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission12_log
bash -lc "conda activate desilike && PYTHONUNBUFFERED=1 python -u mission12_pk_vs_xi_fit.py"
"""

from __future__ import annotations
import os, sys, glob, re, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.fft import next_fast_len, rfft, irfft
from iminuit import Minuit

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 参数
# ============================================================
BOX_SIZE = 3000.0
K_FUND   = 2 * np.pi / BOX_SIZE
VOL      = BOX_SIZE ** 3
Z        = 1.0
P_FIX    = 1.2
SN0_FIX  = 0.0
KMAX     = 15.0
N_DENSE  = 300_000
DK_FACTOR = 0.1    # k-rebinning bin 宽 = 0.1 * kf

# 功率谱拟合设置
N_KBINS_FIT = 20

# 2PCF 拟合设置
R_FIT_MIN = 100.0
R_FIT_MAX = 350.0

# 数据路径
PK_DATA_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat"
PK_COV_GLOB  = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat"
PCF_DATA_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat"
PCF_COV_GLOB  = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat"
RID_MIN = 2
RID_MAX = 99


# ============================================================
# I/O
# ============================================================
def _rid(fp):
    m = re.search(r"N(\d+)", os.path.basename(fp))
    return int(m.group(1)) if m else -1

def load_pk(pattern, rid_min, rid_max, n_dp, p0_col=5):
    fps = [f for f in sorted(glob.glob(pattern), key=_rid) if rid_min <= _rid(f) <= rid_max]
    if not fps: raise FileNotFoundError(pattern)
    ref = np.loadtxt(fps[0], comments="#")
    kcen = ref[:n_dp, 0]; kmin = ref[:n_dp, 1]; kmax = ref[:n_dp, 2]
    mocks = np.array([np.loadtxt(f, comments="#")[:n_dp, p0_col] for f in fps])
    return kcen, kmin, kmax, mocks, mocks.mean(0), mocks.std(0, ddof=1)

def load_pcf(pattern, rid_min, rid_max, xi_col=3):
    fps = [f for f in sorted(glob.glob(pattern), key=_rid) if rid_min <= _rid(f) <= rid_max]
    if not fps: raise FileNotFoundError(pattern)
    s = np.loadtxt(fps[0], comments="#")[:, 0]
    mocks = np.array([np.loadtxt(f, comments="#")[:, xi_col] for f in fps])
    return s, mocks, mocks.mean(0), mocks.std(0, ddof=1)


# ============================================================
# 数学工具
# ============================================================
def j0(x):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    m = x != 0
    out[m] = np.sin(x[m]) / x[m]
    return out

def gq_fft(qmax, nmax):
    a = np.zeros(qmax + 1, dtype=np.float32)
    a[0] = 1.0
    sq = np.arange(1, nmax + 1, dtype=np.int64) ** 2
    a[sq[sq <= qmax]] = 2.0
    nfft = next_fast_len(3 * qmax + 1)
    fa = rfft(a, n=nfft)
    return np.rint(irfft(fa*fa*fa, n=nfft)[:qmax+1]).astype(np.int64)

def gq_enumerate(qmax):
    nmax = int(np.ceil(np.sqrt(qmax))) + 1
    gq = np.zeros(qmax + 1, dtype=np.int64)
    for nx in range(-nmax, nmax+1):
        for ny in range(-nmax, nmax+1):
            for nz in range(-nmax, nmax+1):
                if nx == ny == nz == 0: continue
                q = nx*nx + ny*ny + nz*nz
                if q <= qmax: gq[q] += 1
    return gq

def precompute_rebin_cache(gq, kf, kmax, dk_factor=0.1):
    qnz = np.nonzero(gq[1:])[0] + 1
    kv = kf * np.sqrt(qnz.astype(np.float64))
    g = gq[qnz].astype(np.float64)
    dk = dk_factor * kf
    n_bins = int(np.ceil(kmax / dk)) + 1
    bin_idx = np.clip((kv / dk).astype(np.int64), 0, n_bins - 1)
    G_bin = np.bincount(bin_idx, weights=g, minlength=n_bins)
    Gk_bin = np.bincount(bin_idx, weights=g * kv, minlength=n_bins)
    nz = G_bin > 0
    return G_bin[nz], Gk_bin[nz] / G_bin[nz]

def fast_discrete_xi0(s, G_nz, k_eff, V, kd, pd):
    W = G_nz * np.interp(k_eff, kd, pd)
    arg = np.outer(k_eff, s)
    J = np.ones_like(arg)
    m = arg != 0
    J[m] = np.sin(arg[m]) / arg[m]
    return (W @ J) / V


# ============================================================
# Main
# ============================================================
def main():
    t_total = time.time()

    from cosmoprimo import Cosmology
    from desilike.theories.galaxy_clustering import (
        FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    )

    cosmo = Cosmology(h=0.6711, Omega_b=0.049, Omega_cdm=0.3175-0.049,
                      sigma8=0.834, n_s=0.9624, engine="class")

    # ============================================================
    # 读取数据
    # ============================================================
    print("=" * 60)
    print("读取数据 ...")
    print("=" * 60)

    kcen, kmin_b, kmax_b, pk_mocks, pk_mean, pk_std_raw = load_pk(
        PK_DATA_GLOB, RID_MIN, RID_MAX, N_KBINS_FIT + 1)

    # 跳过第一个可能为空的 bin
    first_valid = 0
    for ib in range(len(kcen)):
        if pk_mocks[:, ib].mean() > 0:
            first_valid = ib; break
    kcen = kcen[first_valid:first_valid+N_KBINS_FIT]
    kmin_b = kmin_b[first_valid:first_valid+N_KBINS_FIT]
    kmax_b = kmax_b[first_valid:first_valid+N_KBINS_FIT]
    pk_mocks = pk_mocks[:, first_valid:first_valid+N_KBINS_FIT]
    pk_mean = pk_mocks.mean(0)
    pk_std = pk_mocks.std(0, ddof=1)

    _, _, _, cov_pk_mocks, _, _ = load_pk(PK_COV_GLOB, RID_MIN, RID_MAX, N_KBINS_FIT + 1)
    cov_pk_mocks = cov_pk_mocks[:, first_valid:first_valid+N_KBINS_FIT]

    s_all, pcf_mocks_all, xi_mean_all, xi_std_all = load_pcf(PCF_DATA_GLOB, RID_MIN, RID_MAX)
    _, pcf_cov_mocks_all, _, _ = load_pcf(PCF_COV_GLOB, RID_MIN, RID_MAX)

    # 2PCF 拟合范围 mask
    fit_mask = (s_all >= R_FIT_MIN) & (s_all <= R_FIT_MAX)
    s_fit = s_all[fit_mask]
    xi_mean_fit = xi_mean_all[fit_mask]
    xi_mocks_fit = pcf_mocks_all[:, fit_mask]
    xi_cov_mocks_fit = pcf_cov_mocks_all[:, fit_mask]

    print(f"  P(k): {pk_mocks.shape[0]} mocks, {N_KBINS_FIT} k-bins")
    print(f"  ξ(r): {pcf_mocks_all.shape[0]} mocks, fit range [{R_FIT_MIN}, {R_FIT_MAX}] -> {fit_mask.sum()} r-bins")
    print(f"  Covariance: P(k) {cov_pk_mocks.shape[0]} mocks, ξ(r) {pcf_cov_mocks_all.shape[0]} mocks")

    # ============================================================
    # 标准 desilike 拟合 (作为 BinAvgFit 的初始值)
    # ============================================================
    print("\n" + "=" * 60)
    print("标准 desilike 拟合 (获取初始参数) ...")
    print("=" * 60)

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler
    from pypower import PowerSpectrumStatistics

    edges = np.concatenate([kmin_b, [kmax_b[-1]]])
    nm = 4./3.*np.pi*(edges[1:]**3 - edges[:-1]**3)
    data_ps = PowerSpectrumStatistics(edges=edges, modes=kcen,
        power_nonorm=np.array([pk_mean]), nmodes=nm, ells=[0],
        shotnoise_nonorm=0.0, statistic="multipole")
    mock_list = []
    for i in range(cov_pk_mocks.shape[0]):
        t = data_ps.deepcopy()
        t.power_nonorm.flat[...] = np.array([cov_pk_mocks[i]]).ravel()
        mock_list.append(t)

    tmpl = FixedPowerSpectrumTemplate(z=Z, fiducial=cosmo)
    th = PNGTracerPowerSpectrumMultipoles(template=tmpl, mode="b-p")
    th.init.params["p"].update(fixed=True, value=P_FIX)
    th.init.params["sn0"].update(fixed=True, value=SN0_FIX)
    th.init.params["sigmas"].update(fixed=False, value=0.0)

    obs = TracerPowerSpectrumMultipolesObservable(data=data_ps, covariance=mock_list,
        klim={0: [float(kcen.min()), float(kcen.max()), float(kcen[1]-kcen[0])]}, theory=th)
    like = ObservablesGaussianLikelihood(observables=[obs])
    _ = like()
    like.all_params["p"].update(fixed=True, value=P_FIX)
    like.all_params["sn0"].update(fixed=True, value=SN0_FIX)
    like.all_params["sigmas"].update(fixed=False, value=0.0)

    prof = MinuitProfiler(like, seed=66)
    profiles = prof.maximize(niterations=27)
    bf_std = profiles.bestfit.choice(input=True)
    print(f"  StdFit: fnl={float(bf_std['fnl_loc']):.3f}, b1={float(bf_std['b1']):.6f}, sigmas={float(bf_std['sigmas']):.6f}")

    # ============================================================
    # 方法A: BinAvgFit 功率谱拟合
    # ============================================================
    print("\n" + "=" * 60)
    print("方法A: BinAvgFit 功率谱拟合 ...")
    print("=" * 60)

    kmax_fit = float(kmax_b[-1])
    qmax_fit = int(np.floor((kmax_fit / K_FUND)**2)) + 1
    gq_fit_arr = gq_enumerate(qmax_fit)

    # 构建 bin-shell 结构
    q_all_fit = np.nonzero(gq_fit_arr[1:])[0] + 1
    k_all_fit = K_FUND * np.sqrt(q_all_fit.astype(float))
    g_all_fit = gq_fit_arr[q_all_fit].astype(float)

    k_shells_per_bin = []
    g_shells_per_bin = []
    for lo, hi in zip(kmin_b, kmax_b):
        m = (k_all_fit >= lo) & (k_all_fit < hi)
        k_shells_per_bin.append(k_all_fit[m])
        g_shells_per_bin.append(g_all_fit[m])

    all_k_unique = np.unique(np.concatenate(k_shells_per_bin))
    tmpl_fit = FixedPowerSpectrumTemplate(z=Z, fiducial=cosmo)
    th_fit = PNGTracerPowerSpectrumMultipoles(k=all_k_unique, template=tmpl_fit, mode="b-p")
    th_fit.init.params["p"].update(fixed=True, value=P_FIX)
    th_fit.init.params["sn0"].update(fixed=True, value=SN0_FIX)
    th_fit.init.params["sigmas"].update(fixed=False, value=0.0)

    cov_pk = np.cov(cov_pk_mocks, rowvar=False, ddof=1)
    cov_pk_inv = np.linalg.pinv(cov_pk, rcond=1e-10)

    def chi2_pk(fnl_loc, b1, sigmas):
        params = dict(bf_std)
        params.update(fnl_loc=float(fnl_loc), b1=float(b1),
                      sigmas=float(sigmas), p=P_FIX, sn0=SN0_FIX)
        th_fit(**params)
        pk_th = np.array(th_fit.power[0], dtype=float)
        pk_map = dict(zip(all_k_unique, pk_th))
        pk_bin = np.zeros(len(k_shells_per_bin))
        for i, (ks, gs) in enumerate(zip(k_shells_per_bin, g_shells_per_bin)):
            if len(ks) == 0: continue
            pv = np.array([pk_map[k] for k in ks])
            pk_bin[i] = np.sum(gs * pv) / np.sum(gs)
        diff = pk_mean - pk_bin
        return float(diff @ cov_pk_inv @ diff)

    t0 = time.time()
    m_pk = Minuit(chi2_pk,
                  fnl_loc=float(bf_std["fnl_loc"]),
                  b1=float(bf_std["b1"]),
                  sigmas=float(bf_std["sigmas"]))
    m_pk.errordef = 1.0
    m_pk.limits["b1"] = (0, None)
    m_pk.limits["sigmas"] = (0, None)
    m_pk.limits["fnl_loc"] = (-2000, 2000)
    m_pk.migrad()
    t_pk = time.time() - t0

    bf_pk = dict(bf_std)
    bf_pk.update(fnl_loc=float(m_pk.values["fnl_loc"]),
                 b1=float(m_pk.values["b1"]),
                 sigmas=float(m_pk.values["sigmas"]),
                 p=P_FIX, sn0=SN0_FIX)
    print(f"  BinAvgFit(P(k)): fnl={bf_pk['fnl_loc']:.3f}, b1={bf_pk['b1']:.6f}, sigmas={bf_pk['sigmas']:.6f}")
    print(f"  chi2/dof = {m_pk.fmin.fval:.2f}/{N_KBINS_FIT-3} = {m_pk.fmin.fval/(N_KBINS_FIT-3):.3f}")
    print(f"  耗时: {t_pk:.1f}s")

    # ============================================================
    # 准备 k-rebinning 缓存（用于 2PCF 拟合）
    # ============================================================
    print("\n准备 k-rebinning 缓存 ...")
    qmax_fd = int((KMAX / K_FUND)**2)
    nmax_fd = int(KMAX / K_FUND)
    gq_fd = gq_fft(qmax_fd, nmax_fd)
    G_nz, k_eff = precompute_rebin_cache(gq_fd, K_FUND, KMAX, dk_factor=DK_FACTOR)
    print(f"  {len(G_nz)} bins")

    kd = np.geomspace(K_FUND * 0.5, KMAX * 1.1, N_DENSE)

    # 用于快速评估 P_model(k) 的理论对象（在 kd 上）
    tmpl_dense = FixedPowerSpectrumTemplate(z=Z, fiducial=cosmo)
    th_dense = PNGTracerPowerSpectrumMultipoles(k=kd, template=tmpl_dense, mode="b-p")
    th_dense.init.params["p"].update(fixed=True, value=P_FIX)
    th_dense.init.params["sn0"].update(fixed=True, value=SN0_FIX)
    th_dense.init.params["sigmas"].update(fixed=False, value=0.0)

    # ============================================================
    # 方法B: 2PCF 模型拟合
    # ============================================================
    print("\n" + "=" * 60)
    print("方法B: 2PCF 模型拟合 ...")
    print("=" * 60)

    # 2PCF covariance（用 r² * ξ 作为拟合量，与指标一致）
    r2_xi_cov_mocks = s_fit**2 * xi_cov_mocks_fit
    cov_xi = np.cov(r2_xi_cov_mocks, rowvar=False, ddof=1)
    cov_xi_inv = np.linalg.pinv(cov_xi, rcond=1e-10)
    r2_xi_data = s_fit**2 * xi_mean_fit

    n_call = [0]
    def chi2_xi(fnl_loc, b1, sigmas):
        params = dict(bf_std)
        params.update(fnl_loc=float(fnl_loc), b1=float(b1),
                      sigmas=float(sigmas), p=P_FIX, sn0=SN0_FIX)
        # 评估 P_model(k) 在密集网格
        th_dense(**params)
        pd = np.array(th_dense.power[0], dtype=float)
        # 加速版全离散求和
        xi_model_all = fast_discrete_xi0(s_all, G_nz, k_eff, VOL, kd, pd)
        xi_model_fit = xi_model_all[fit_mask]
        r2_xi_model = s_fit**2 * xi_model_fit
        diff = r2_xi_data - r2_xi_model
        n_call[0] += 1
        return float(diff @ cov_xi_inv @ diff)

    print(f"  拟合范围: r ∈ [{R_FIT_MIN}, {R_FIT_MAX}] Mpc/h, {fit_mask.sum()} 个 r-bin")
    print(f"  拟合量: r²ξ₀(r)")

    t0 = time.time()
    m_xi = Minuit(chi2_xi,
                  fnl_loc=float(bf_pk["fnl_loc"]),
                  b1=float(bf_pk["b1"]),
                  sigmas=float(bf_pk["sigmas"]))
    m_xi.errordef = 1.0
    m_xi.limits["b1"] = (0, None)
    m_xi.limits["sigmas"] = (0, None)
    m_xi.limits["fnl_loc"] = (-2000, 2000)
    m_xi.migrad()
    t_xi = time.time() - t0

    bf_xi = dict(bf_std)
    bf_xi.update(fnl_loc=float(m_xi.values["fnl_loc"]),
                 b1=float(m_xi.values["b1"]),
                 sigmas=float(m_xi.values["sigmas"]),
                 p=P_FIX, sn0=SN0_FIX)
    n_dof_xi = fit_mask.sum() - 3
    print(f"  2PCF fit: fnl={bf_xi['fnl_loc']:.3f}, b1={bf_xi['b1']:.6f}, sigmas={bf_xi['sigmas']:.6f}")
    print(f"  chi2/dof = {m_xi.fmin.fval:.2f}/{n_dof_xi} = {m_xi.fmin.fval/n_dof_xi:.3f}")
    print(f"  耗时: {t_xi:.1f}s ({n_call[0]} 次 chi2 调用, {t_xi/n_call[0]:.3f}s/次)")

    # ============================================================
    # 对比
    # ============================================================
    print("\n" + "=" * 60)
    print("Best-fit 参数对比")
    print("=" * 60)
    print(f"  {'方法':<25s} {'fnl_loc':>10s} {'b1':>10s} {'sigmas':>10s} {'chi2/dof':>10s}")
    print(f"  {'-'*67}")
    print(f"  {'StdFit(bin-center)':<25s} {float(bf_std['fnl_loc']):>10.3f} {float(bf_std['b1']):>10.6f} {float(bf_std['sigmas']):>10.6f} {'—':>10s}")
    print(f"  {'BinAvgFit P(k)':<25s} {bf_pk['fnl_loc']:>10.3f} {bf_pk['b1']:>10.6f} {bf_pk['sigmas']:>10.6f} {m_pk.fmin.fval/(N_KBINS_FIT-3):>10.3f}")
    print(f"  {'2PCF fit':<25s} {bf_xi['fnl_loc']:>10.3f} {bf_xi['b1']:>10.6f} {bf_xi['sigmas']:>10.6f} {m_xi.fmin.fval/n_dof_xi:>10.3f}")

    # ============================================================
    # 计算对比曲线
    # ============================================================
    print("\n计算对比曲线 ...")

    # P_model 在画图 k 范围
    k_plot = np.geomspace(kcen.min(), 1.0, 400)
    tmpl_plot = FixedPowerSpectrumTemplate(z=Z, fiducial=cosmo)
    def pk_on_grid(k, params):
        t = PNGTracerPowerSpectrumMultipoles(k=k, template=tmpl_plot, mode="b-p")
        t.init.params["p"].update(fixed=True, value=P_FIX)
        t.init.params["sn0"].update(fixed=True, value=SN0_FIX)
        t.init.params["sigmas"].update(fixed=False, value=float(params.get("sigmas", 0)))
        t(**params)
        return np.array(t.power[0], dtype=float)

    pk_curve_pk = pk_on_grid(k_plot, bf_pk)
    pk_curve_xi = pk_on_grid(k_plot, bf_xi)

    # xi_model from each best-fit
    pd_pk = pk_on_grid(kd, bf_pk)
    pd_xi = pk_on_grid(kd, bf_xi)
    xi_from_pk = fast_discrete_xi0(s_all, G_nz, k_eff, VOL, kd, pd_pk)
    xi_from_xi = fast_discrete_xi0(s_all, G_nz, k_eff, VOL, kd, pd_xi)

    r2_from_pk = s_all**2 * xi_from_pk
    r2_from_xi = s_all**2 * xi_from_xi
    r2_data_all = s_all**2 * xi_mean_all
    r2_err_all = s_all**2 * xi_std_all

    # ============================================================
    # 画图
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (0,0) P(k)
    ax = axes[0, 0]
    ax.errorbar(kcen, pk_mean, yerr=pk_std, fmt="ko", ms=4, capsize=2, label="Data mean")
    ax.loglog(k_plot, pk_curve_pk, "r-", lw=2, label=f"BinAvgFit P(k)  fnl={bf_pk['fnl_loc']:.1f}")
    ax.loglog(k_plot, pk_curve_xi, "b--", lw=2, label=f"2PCF fit  fnl={bf_xi['fnl_loc']:.1f}")
    ax.set_ylabel("P0(k)"); ax.set_title("P0(k): best-fit comparison")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (0,1) r²ξ₀(r)
    ax = axes[0, 1]
    ax.errorbar(s_all, r2_data_all, yerr=r2_err_all, fmt="ko", ms=4, capsize=2, label="Data mean")
    ax.plot(s_all, r2_from_pk, "r-", lw=2, label=f"from P(k) fit  fnl={bf_pk['fnl_loc']:.1f}")
    ax.plot(s_all, r2_from_xi, "b--", lw=2, label=f"from 2PCF fit  fnl={bf_xi['fnl_loc']:.1f}")
    ax.axvspan(R_FIT_MIN, R_FIT_MAX, alpha=0.1, color="blue", label="2PCF fit range")
    ax.set_ylabel(r"$r^2\xi_0(r)$"); ax.set_title(r"$r^2\xi_0(r)$: best-fit comparison")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (1,0) P(k) ratio
    ax = axes[1, 0]
    pk_at_data_pk = pk_on_grid(kcen, bf_pk)
    pk_at_data_xi = pk_on_grid(kcen, bf_xi)
    ax.errorbar(kcen, pk_mean/pk_at_data_pk, yerr=pk_std/np.abs(pk_at_data_pk),
                fmt="ro", ms=4, capsize=2, label="Data / BinAvgFit P(k)")
    ax.errorbar(kcen, pk_mean/pk_at_data_xi, yerr=pk_std/np.abs(pk_at_data_xi),
                fmt="b^", ms=4, capsize=2, alpha=0.7, label="Data / 2PCF fit")
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_ylabel("Data/Model")
    ax.set_xlabel("k [h/Mpc]"); ax.set_title("P(k) Data/Model ratio")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (1,1) ξ residuals
    ax = axes[1, 1]
    ds_pk = (r2_data_all - r2_from_pk) / r2_err_all
    ds_xi = (r2_data_all - r2_from_xi) / r2_err_all
    ax.plot(s_all, ds_pk, "ro-", ms=4, lw=1.5, label=f"from P(k) fit  |D/s|={np.nanmean(np.abs(ds_pk)):.3f}")
    ax.plot(s_all, ds_xi, "b^--", ms=4, lw=1.5, label=f"from 2PCF fit  |D/s|={np.nanmean(np.abs(ds_xi)):.3f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(1, color="gray", ls=":", lw=0.6); ax.axhline(-1, color="gray", ls=":", lw=0.6)
    ax.axvspan(R_FIT_MIN, R_FIT_MAX, alpha=0.1, color="blue")
    ax.set_xlabel(r"$r$ [Mpc/h]"); ax.set_ylabel(r"$(Data-Model)/\sigma$")
    ax.set_title("2PCF residuals"); ax.set_ylim(-3, 3)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("Mission 12: P(k) fit vs 2PCF fit (3Gpc fnl100)", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = os.path.join(THIS_DIR, "mission12_pk_vs_xi_fit.png")
    fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"\n[saved] {out_png}")

    # ============================================================
    # 笔记
    # ============================================================
    out_md = os.path.join(THIS_DIR, "mission12_results.md")
    with open(out_md, "w") as f:
        f.write("# Mission 12: P(k) 拟合 vs 2PCF 拟合的 best-fit 对比\n\n")
        f.write("## 设置\n\n")
        f.write(f"- 数据: 3Gpc fastPM fnl100\n")
        f.write(f"- P(k) 拟合: BinAvgFit, 前 {N_KBINS_FIT} 个 k-bin\n")
        f.write(f"- 2PCF 拟合: r ∈ [{R_FIT_MIN}, {R_FIT_MAX}] Mpc/h, {fit_mask.sum()} 个 r-bin\n")
        f.write(f"- 拟合量: r²ξ₀(r)\n")
        f.write(f"- 自由参数: fnl_loc, b1, sigmas (p={P_FIX} fix, sn0={SN0_FIX} fix)\n")
        f.write(f"- 2PCF model: 参数 → P_model(k) → fast_discrete_xi0(r) (k-rebinning 加速)\n\n")
        f.write("## Best-fit 参数\n\n")
        f.write(f"| 方法 | fnl_loc | b1 | sigmas | chi2/dof |\n")
        f.write(f"|---|---|---|---|---|\n")
        f.write(f"| StdFit(bin-center) | {float(bf_std['fnl_loc']):.3f} | {float(bf_std['b1']):.6f} | {float(bf_std['sigmas']):.6f} | — |\n")
        f.write(f"| BinAvgFit P(k) | {bf_pk['fnl_loc']:.3f} | {bf_pk['b1']:.6f} | {bf_pk['sigmas']:.6f} | {m_pk.fmin.fval/(N_KBINS_FIT-3):.3f} |\n")
        f.write(f"| 2PCF fit | {bf_xi['fnl_loc']:.3f} | {bf_xi['b1']:.6f} | {bf_xi['sigmas']:.6f} | {m_xi.fmin.fval/n_dof_xi:.3f} |\n\n")
        f.write("## 耗时\n\n")
        f.write(f"- P(k) BinAvgFit: {t_pk:.1f}s\n")
        f.write(f"- 2PCF fit: {t_xi:.1f}s ({n_call[0]} calls, {t_xi/n_call[0]:.3f}s/call)\n\n")
        ma_pk = float(np.nanmean(np.abs(ds_pk)))
        ms_pk = float(np.nanmean(ds_pk))
        ma_xi = float(np.nanmean(np.abs(ds_xi)))
        ms_xi = float(np.nanmean(ds_xi))
        f.write("## 2PCF 指标 (全 r 范围)\n\n")
        f.write(f"| 来源 | mean|Δ/σ| | mean(Δ/σ) |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| from P(k) fit | {ma_pk:.4f} | {ms_pk:+.4f} |\n")
        f.write(f"| from 2PCF fit | {ma_xi:.4f} | {ms_xi:+.4f} |\n")
    print(f"[saved] {out_md}")
    print(f"\nTotal: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
