#!/usr/bin/env python3
"""
Mission 9/12: MCMC 对比 P(k) vs 2PCF 后验分布
=============================================
用 emcee 分别对 BinAvgFit P(k) 和 FullDiscrete 2PCF 跑 MCMC，
画 contour plot 对比参数限制。

3Gpc fnl100, EZmock covariance (pk), FastPM covariance (pcf)
单核运行。

自由参数: fnl_loc, b1, sigmas (p=1.2, sn0=0 固定)
"""

import os, glob, re, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import next_fast_len, rfft, irfft
import emcee

OUT = os.path.dirname(os.path.abspath(__file__))
L = 3000.0; kf = 2*np.pi/L; V = L**3; Z = 1.0; P_FIX = 1.2; KMAX = 15.0; N_DP = 20
XI_RMIN = 80.0; XI_RMAX = 380.0

# MCMC 设置
N_WALKERS = 24   # walker 数 (>= 2*ndim)
N_BURN = 300     # burn-in
N_STEPS = 1500   # 采样步数

# ============================================================
# 工具函数（复用）
# ============================================================
def rid(f):
    m = re.search(r'N(\d+)', os.path.basename(f))
    return int(m.group(1)) if m else -1

def load_pk(pat, rmin, rmax, ndp):
    fs = [f for f in sorted(glob.glob(pat), key=rid) if rmin <= rid(f) <= rmax]
    ref = np.loadtxt(fs[0], comments='#')
    v = np.array([np.loadtxt(f, comments='#')[:ndp, 5] for f in fs])
    return ref[:ndp, 0], ref[:ndp, 1], ref[:ndp, 2], v, v.mean(0)

def load_pcf(pat, rmin, rmax):
    fs = [f for f in sorted(glob.glob(pat), key=rid) if rmin <= rid(f) <= rmax]
    s = np.loadtxt(fs[0], comments='#')[:, 0]
    v = np.array([np.loadtxt(f, comments='#')[:, 3] for f in fs])
    return s, v, v.mean(0), v.std(0, ddof=1)

def load_ez(glob_pat, k_target, ndp):
    files = sorted(glob.glob(glob_pat))
    p0_list = []
    for fp in files:
        try:
            arr = np.loadtxt(fp, comments='#')
            if arr.shape[0] < ndp: continue
            if np.allclose(arr[:ndp, 0], k_target, rtol=0, atol=1e-10):
                p0_list.append(arr[:ndp, 5])
            else:
                p0_list.append(np.interp(k_target, arr[:, 0], arr[:, 5]))
        except: pass
    return np.array(p0_list)

def precompute_rebin(gq, kf, kmax, dk_factor=0.1):
    qnz = np.nonzero(gq[1:])[0] + 1
    kv = kf * np.sqrt(qnz.astype(float))
    g = gq[qnz].astype(float)
    dk = dk_factor * kf
    nb = int(np.ceil(kmax / dk)) + 1
    bi = np.clip((kv / dk).astype(int), 0, nb - 1)
    G = np.bincount(bi, weights=g, minlength=nb)
    Gk = np.bincount(bi, weights=g * kv, minlength=nb)
    nz = G > 0
    return G[nz], Gk[nz] / G[nz]

def fast_xi0(s, G, keff, V, kd, pd):
    """加速离散求和 ξ₀(r)"""
    W = G * np.interp(keff, kd, pd)
    arg = np.outer(keff, s)
    J = np.ones_like(arg)
    m = arg != 0
    J[m] = np.sin(arg[m]) / arg[m]
    return (W @ J) / V

# ============================================================
def main():
    T0 = time.time()
    print("=" * 60)
    print("MCMC: P(k) BinAvgFit vs 2PCF FullDiscrete")
    print("=" * 60)

    from cosmoprimo import Cosmology
    from desilike.theories.galaxy_clustering import (
        FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles)
    cosmo = Cosmology(h=0.6711, Omega_b=0.049, Omega_cdm=0.3175-0.049,
                      sigma8=0.834, n_s=0.9624, engine='class')

    # ---- 数据 ----
    print("\n[1] 加载数据...")
    kcen, kmin_b, kmax_b, pk_mocks, pk_mean = load_pk(
        "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat", 2, 99, N_DP)
    s, pcf_mocks_arr, xi_mean, xi_std = load_pcf(
        "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat", 2, 99)

    # P(k) covariance (EZmock)
    print("  EZmock...")
    ez_cov = load_ez("/pscratch/sd/l/lzy/cov_mock/B3000*seed*/PK_EZmock*_RSD.dat", kcen, N_DP)
    print(f"  EZmock: {ez_cov.shape[0]} used")
    pk_cov_mat = np.cov(ez_cov, rowvar=False, ddof=1)
    pk_cov_inv = np.linalg.inv(pk_cov_mat)

    # 2PCF covariance (FastPM mocks, 在拟合范围内)
    fit_mask = (s >= XI_RMIN) & (s <= XI_RMAX)
    s_fit = s[fit_mask]
    n_xi_fit = fit_mask.sum()
    r2_data_fit = (s**2 * xi_mean)[fit_mask]
    # r²ξ 的 covariance
    r2_mocks = s[None, :]**2 * pcf_mocks_arr  # (n_mock, n_r)
    r2_cov = np.cov(r2_mocks[:, fit_mask], rowvar=False, ddof=1)
    r2_cov_inv = np.linalg.inv(r2_cov)
    print(f"  2PCF 拟合范围: r=[{XI_RMIN},{XI_RMAX}], {n_xi_fit} 点")

    # ---- BinAvgFit 壳层索引 ----
    print("\n[2] 壳层索引...")
    qmax_fit = int(np.floor((kmax_b[-1] / kf)**2)) + 1
    nmax = int(np.ceil(np.sqrt(qmax_fit))) + 1
    gq_fit = np.zeros(qmax_fit + 1, dtype=np.int64)
    for nx in range(-nmax, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            for nz in range(-nmax, nmax + 1):
                if nx == ny == nz == 0: continue
                q = nx*nx + ny*ny + nz*nz
                if q <= qmax_fit: gq_fit[q] += 1
    q_all = np.nonzero(gq_fit[1:])[0] + 1
    k_all = kf * np.sqrt(q_all.astype(float))
    g_all = gq_fit[q_all].astype(float)
    ks_bin = []; gs_bin = []
    for lo, hi in zip(kmin_b, kmax_b):
        m = (k_all >= lo) & (k_all < hi)
        ks_bin.append(k_all[m]); gs_bin.append(g_all[m])

    # ---- FullDiscrete rebin cache ----
    print("[3] FullDiscrete rebin cache...")
    from scipy.fft import next_fast_len, rfft, irfft
    qmax_fd = int((KMAX / kf)**2); nmax_fd = int(KMAX / kf)
    a = np.zeros(qmax_fd + 1, dtype=np.float32); a[0] = 1
    sq = np.arange(1, nmax_fd + 1, dtype=np.int64)**2; a[sq[sq <= qmax_fd]] = 2
    nfft = next_fast_len(3 * qmax_fd + 1)
    fa = rfft(a, n=nfft)
    gq_fd = np.rint(irfft(fa**3, n=nfft)[:qmax_fd + 1]).astype(np.int64)
    G_nz, k_eff = precompute_rebin(gq_fd, kf, KMAX)
    print(f"  {len(G_nz)} rebinned shells")

    # ---- P(k) 密集网格 + 理论对象 ----
    kd = np.geomspace(kf * 0.5, KMAX * 1.1, 300000)

    # BinAvgFit 用的理论对象（在 all_k_fit 上评估）
    all_k_fit = np.unique(np.concatenate(ks_bin))
    tmpl_ba = FixedPowerSpectrumTemplate(z=Z, fiducial=cosmo)
    th_ba = PNGTracerPowerSpectrumMultipoles(k=all_k_fit, template=tmpl_ba, mode='b-p')
    th_ba.init.params['p'].update(fixed=True, value=P_FIX)
    th_ba.init.params['sn0'].update(fixed=True, value=0)
    th_ba.init.params['sigmas'].update(fixed=False, value=0)

    # FullDiscrete 用的理论对象（在密集 kd 上评估）
    tmpl_fd = FixedPowerSpectrumTemplate(z=Z, fiducial=cosmo)
    th_fd = PNGTracerPowerSpectrumMultipoles(k=kd, template=tmpl_fd, mode='b-p')
    th_fd.init.params['p'].update(fixed=True, value=P_FIX)
    th_fd.init.params['sn0'].update(fixed=True, value=0)
    th_fd.init.params['sigmas'].update(fixed=False, value=0)

    # ============================================================
    # 定义两个 log-likelihood
    # ============================================================

    # 参数先验范围
    FNL_MIN, FNL_MAX = -200.0, 300.0
    B1_MIN, B1_MAX = 1.0, 5.0
    SIG_MIN, SIG_MAX = 0.0, 20.0

    def log_prior(theta):
        fnl, b1, sigmas = theta
        if FNL_MIN < fnl < FNL_MAX and B1_MIN < b1 < B1_MAX and SIG_MIN < sigmas < SIG_MAX:
            return 0.0
        return -np.inf

    # --- Log-likelihood 1: P(k) BinAvgFit ---
    def log_like_pk(theta):
        fnl, b1, sigmas = theta
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        params = dict(fnl_loc=float(fnl), b1=float(b1), sigmas=float(sigmas),
                      p=P_FIX, sn0=0.0)
        try:
            th_ba(**params)
            pk_arr = np.interp(all_k_fit, th_ba.k, np.array(th_ba.power[0]))
            pk_map = dict(zip(all_k_fit, pk_arr))
            pk_bin = np.zeros(N_DP)
            for i, (ks, gs) in enumerate(zip(ks_bin, gs_bin)):
                if len(ks) == 0: pk_bin[i] = 0
                else:
                    pv = np.array([pk_map[k] for k in ks])
                    pk_bin[i] = np.sum(gs * pv) / np.sum(gs)
            diff = pk_mean - pk_bin
            chi2 = float(diff @ pk_cov_inv @ diff)
            return lp - 0.5 * chi2
        except:
            return -np.inf

    # --- Log-likelihood 2: 2PCF FullDiscrete ---
    def log_like_xi(theta):
        fnl, b1, sigmas = theta
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        params = dict(fnl_loc=float(fnl), b1=float(b1), sigmas=float(sigmas),
                      p=P_FIX, sn0=0.0)
        try:
            th_fd(**params)
            pd_cur = np.array(th_fd.power[0], dtype=float)
            xi_model = fast_xi0(s, G_nz, k_eff, V, kd, pd_cur)
            r2_model = (s**2 * xi_model)[fit_mask]
            diff = r2_data_fit - r2_model
            chi2 = float(diff @ r2_cov_inv @ diff)
            return lp - 0.5 * chi2
        except:
            return -np.inf

    # ============================================================
    # 跑 MCMC
    # ============================================================
    ndim = 3
    # 初始位置：以 BinAvgFit best-fit 为中心加小扰动
    p0_center = np.array([72.4, 2.80, 3.1])

    print(f"\n[4] MCMC P(k) BinAvgFit ({N_WALKERS} walkers, {N_BURN}+{N_STEPS} steps)...")
    t0 = time.time()
    pos_pk = p0_center + 0.1 * np.random.randn(N_WALKERS, ndim) * np.array([3.0, 0.02, 1.0])
    sampler_pk = emcee.EnsembleSampler(N_WALKERS, ndim, log_like_pk)
    state_pk = sampler_pk.run_mcmc(pos_pk, N_BURN, progress=True)
    sampler_pk.reset()
    sampler_pk.run_mcmc(state_pk, N_STEPS, progress=True)
    chain_pk = sampler_pk.get_chain(flat=True)
    print(f"  完成，耗时 {time.time()-t0:.0f}s, 接受率={np.mean(sampler_pk.acceptance_fraction):.3f}")
    print(f"  fnl = {np.mean(chain_pk[:,0]):.2f} ± {np.std(chain_pk[:,0]):.2f}")
    print(f"  b1  = {np.mean(chain_pk[:,1]):.4f} ± {np.std(chain_pk[:,1]):.4f}")
    print(f"  sig = {np.mean(chain_pk[:,2]):.2f} ± {np.std(chain_pk[:,2]):.2f}")

    print(f"\n[5] MCMC 2PCF FullDiscrete ({N_WALKERS} walkers, {N_BURN}+{N_STEPS} steps)...")
    t0 = time.time()
    pos_xi = p0_center + 0.1 * np.random.randn(N_WALKERS, ndim) * np.array([3.0, 0.02, 1.0])
    sampler_xi = emcee.EnsembleSampler(N_WALKERS, ndim, log_like_xi)
    state_xi = sampler_xi.run_mcmc(pos_xi, N_BURN, progress=True)
    sampler_xi.reset()
    sampler_xi.run_mcmc(state_xi, N_STEPS, progress=True)
    chain_xi = sampler_xi.get_chain(flat=True)
    print(f"  完成，耗时 {time.time()-t0:.0f}s, 接受率={np.mean(sampler_xi.acceptance_fraction):.3f}")
    print(f"  fnl = {np.mean(chain_xi[:,0]):.2f} ± {np.std(chain_xi[:,0]):.2f}")
    print(f"  b1  = {np.mean(chain_xi[:,1]):.4f} ± {np.std(chain_xi[:,1]):.4f}")
    print(f"  sig = {np.mean(chain_xi[:,2]):.2f} ± {np.std(chain_xi[:,2]):.2f}")

    # ============================================================
    # 画 contour plot
    # ============================================================
    print("\n[6] 画 contour plot...")

    labels = [r'$f_{\rm NL}$', r'$b_1$', r'$\sigma_s$']

    def contour_2d(ax, chain1, chain2, ix, iy, label1='P(k)', label2='2PCF'):
        """画 2D contour（68% 和 95%）"""
        for chain, color, label in [(chain1, 'red', label1), (chain2, 'blue', label2)]:
            x, y = chain[:, ix], chain[:, iy]
            # 用 histogram2d 做 KDE 近似
            H, xedges, yedges = np.histogram2d(x, y, bins=50, density=True)
            H = H.T  # 转置以匹配 contour 的 (y, x) 约定
            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])
            # 找 68% 和 95% 等高线水平
            H_sorted = np.sort(H.ravel())[::-1]
            H_cum = np.cumsum(H_sorted) / np.sum(H_sorted)
            level_68 = H_sorted[np.searchsorted(H_cum, 0.68)]
            level_95 = H_sorted[np.searchsorted(H_cum, 0.95)]
            ax.contour(xc, yc, H, levels=[level_95, level_68],
                       colors=color, linewidths=[0.8, 1.5])
            ax.contourf(xc, yc, H, levels=[level_68, H.max()*2],
                        colors=[color], alpha=0.15)
            ax.contourf(xc, yc, H, levels=[level_95, level_68],
                        colors=[color], alpha=0.08)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle("P(k) BinAvgFit vs 2PCF FullDiscrete: MCMC posterior", fontsize=14)

    pairs = [(0, 1), (0, 2), (1, 2)]  # (fnl,b1), (fnl,sig), (b1,sig)

    # 对角线：1D 后验
    for i in range(3):
        ax = axes[i, i]
        ax.hist(chain_pk[:, i], bins=50, density=True, color='red', alpha=0.4, label='P(k)')
        ax.hist(chain_xi[:, i], bins=50, density=True, color='blue', alpha=0.4, label='2PCF')
        ax.set_xlabel(labels[i])
        ax.legend(fontsize=8)
        ax.set_yticks([])

    # 下三角：2D contour
    for i in range(3):
        for j in range(i):
            ax = axes[i, j]
            contour_2d(ax, chain_pk, chain_xi, j, i, 'P(k)', '2PCF')
            ax.set_xlabel(labels[j])
            ax.set_ylabel(labels[i])

    # 上三角：隐藏
    for i in range(3):
        for j in range(i + 1, 3):
            axes[i, j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    figpath = os.path.join(OUT, "mission9_mcmc_contour.png")
    fig.savefig(figpath, dpi=150)
    plt.close()
    print(f"  图: {figpath}")

    # 保存 chain
    np.savez(os.path.join(OUT, "mission9_mcmc_chains.npz"),
             chain_pk=chain_pk, chain_xi=chain_xi,
             labels=labels)

    # 汇总
    print(f"\n{'='*60}")
    print("MCMC 后验对比")
    print(f"{'='*60}")
    print(f"{'参数':<10s} {'P(k) mean±std':>20s} {'2PCF mean±std':>20s}")
    for i, name in enumerate(['fnl_loc', 'b1', 'sigmas']):
        pk_m, pk_s = np.mean(chain_pk[:, i]), np.std(chain_pk[:, i])
        xi_m, xi_s = np.mean(chain_xi[:, i]), np.std(chain_xi[:, i])
        print(f"{name:<10s} {pk_m:>8.3f} ± {pk_s:<8.3f} {xi_m:>8.3f} ± {xi_s:<8.3f}")

    print(f"\n完成! 总耗时 {time.time()-T0:.0f}s")

if __name__ == "__main__":
    main()
