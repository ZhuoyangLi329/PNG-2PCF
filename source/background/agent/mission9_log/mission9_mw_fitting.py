#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission 9: Mode-Weighted Fitting 实验
======================================

核心想法：
当前 P(k) 拟合在 bin center 评估模型，但实际测量的 bin-average
是离散模式的 g_q 加权平均。对弯曲的 P(k)（尤其是 PNG 的 1/k²），
这引入系统偏差。

本脚本测试两种校正方案：
(A) 简单版：将 pypower 数据的 k 从 bin_center 改成 k_mw（模式加权k）
(B) 完整版：对每组参数，用离散模式显式计算 bin 内 P_model 的加权平均

同时测试 fnl=0 以确认 -0.086σ 偏差是否与 PNG 有关。

运行: bash -lc "conda activate desilike && PYTHONUNBUFFERED=1 python -u <script>"
"""

from __future__ import annotations
import os, glob, re, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import next_fast_len, rfft, irfft
from collections import defaultdict

OUT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 通用函数
# ============================================================
def rid(f):
    m = re.search(r'N(\d+)', os.path.basename(f))
    return int(m.group(1)) if m else -1

def load_pk(pat, rmin, rmax, ndp, col=5):
    fs = [f for f in sorted(glob.glob(pat), key=rid) if rmin <= rid(f) <= rmax]
    ref = np.loadtxt(fs[0], comments='#')
    v = np.array([np.loadtxt(f, comments='#')[:ndp, col] for f in fs])
    return ref[:ndp,0], ref[:ndp,1], ref[:ndp,2], v, v.mean(0), v.std(0,ddof=1)

def load_pcf(pat, rmin, rmax):
    fs = [f for f in sorted(glob.glob(pat), key=rid) if rmin <= rid(f) <= rmax]
    s = np.loadtxt(fs[0], comments='#')[:,0]
    v = np.array([np.loadtxt(f, comments='#')[:,3] for f in fs])
    return s, v.mean(0), v.std(0,ddof=1)

def j0(x):
    x=np.asarray(x,float); o=np.ones_like(x); m=x!=0; o[m]=np.sin(x[m])/x[m]; return o

def gq_fft(qmax, nmax):
    a=np.zeros(qmax+1,dtype=np.float32); a[0]=1
    sq=np.arange(1,nmax+1,dtype=np.int64)**2; a[sq[sq<=qmax]]=2
    nfft=next_fast_len(3*qmax+1)
    print(f"  FFT: qmax={qmax}, nfft={nfft}")
    fa=rfft(a,n=nfft); return np.rint(irfft(fa**3,n=nfft)[:qmax+1]).astype(np.int64)

def xi_disc(s, gq, kf, V, kd, pd, chunk=500000):
    """全离散求和"""
    qnz=np.nonzero(gq[1:])[0]+1; xi=np.zeros(len(s))
    for i0 in range(0,len(qnz),chunk):
        qb=qnz[i0:i0+chunk]; kv=kf*np.sqrt(qb.astype(float))
        pv=np.interp(kv,kd,pd); w=gq[qb].astype(float)*pv
        for js in range(0,len(s),10):
            je=min(js+10,len(s)); xi[js:je]+=np.dot(w,j0(np.outer(kv,s[js:je])))
    return xi/V

def met(r2d, r2s, r2m):
    ds=(r2d-r2m)/r2s; return np.nanmean(np.abs(ds)), np.nanmean(ds)

def enumerate_shells_for_bins(kf, kmin_arr, kmax_arr, qmax_enum=3000):
    """
    对每个 k-bin，找到其中的离散模式并计算 mode-weighted k。

    参数
    ----------
    kf : float          基模
    kmin_arr, kmax_arr : ndarray  各 bin 的下界和上界
    qmax_enum : int     枚举的最大 q

    返回
    ----------
    k_mw : ndarray      每个 bin 的 mode-weighted 平均 k
    n_modes : ndarray   每个 bin 的总模式数
    shells_per_bin : list[list[tuple(q, g_q, k_q)]]  每个 bin 的 shell 信息
    """
    nmax = int(np.ceil(np.sqrt(qmax_enum))) + 1
    shell_dict = defaultdict(int)
    for nx in range(-nmax, nmax+1):
        for ny in range(-nmax, nmax+1):
            for nz in range(-nmax, nmax+1):
                if nx==0 and ny==0 and nz==0: continue
                q = nx*nx + ny*ny + nz*nz
                if q <= qmax_enum:
                    shell_dict[q] += 1

    nbins = len(kmin_arr)
    k_mw = np.copy(kmin_arr + kmax_arr) / 2  # 默认用 bin center
    n_modes = np.zeros(nbins, dtype=int)
    shells_per_bin = [[] for _ in range(nbins)]

    for q, gq in sorted(shell_dict.items()):
        kq = kf * np.sqrt(q)
        for ib in range(nbins):
            if kmin_arr[ib] <= kq < kmax_arr[ib]:
                shells_per_bin[ib].append((q, gq, kq))
                n_modes[ib] += gq
                break

    for ib in range(nbins):
        if n_modes[ib] > 0:
            weighted_k = sum(gq * kq for _, gq, kq in shells_per_bin[ib])
            k_mw[ib] = weighted_k / n_modes[ib]

    return k_mw, n_modes, shells_per_bin


# ============================================================
# 主程序
# ============================================================
def main():
    T0 = time.time()
    print("=" * 60)
    print("Mission 9: Mode-Weighted Fitting 实验")
    print("=" * 60)

    from cosmoprimo import Cosmology
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler
    from pypower import PowerSpectrumStatistics

    cosmo = Cosmology(h=0.6711, Omega_b=0.049, Omega_cdm=0.3175-0.049,
                      sigma8=0.834, n_s=0.9624, engine='class')

    # ============================================================
    # 测试配置列表
    # ============================================================
    configs = {
        "3Gpc_fnl100": {
            "L": 3000.0,
            "pk_data": "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat",
            "pk_cov": "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
            "pcf": "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",
            "rmin": 2, "rmax": 99, "ndp": 20,
        },
        "3Gpc_fnl0": {
            "L": 3000.0,
            "pk_data": "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
            "pk_cov": "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
            "pcf": "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat",
            "rmin": 2, "rmax": 99, "ndp": 20,
        },
    }

    all_results = []

    for cfg_name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"  {cfg_name}")
        print(f"{'='*60}")

        L = cfg["L"]; kf = 2*np.pi/L; V = L**3
        ndp = cfg["ndp"]

        # 读数据
        kcen, kmin, kmax_b, pk_mk, pk_m, pk_s = load_pk(cfg["pk_data"], cfg["rmin"], cfg["rmax"], ndp)
        _, _, _, cov_mk, _, _ = load_pk(cfg["pk_cov"], cfg["rmin"], cfg["rmax"], ndp)
        s, xi_m, xi_s = load_pcf(cfg["pcf"], cfg["rmin"], cfg["rmax"])
        r2d = s**2 * xi_m; r2s = s**2 * xi_s

        # 计算 mode-weighted k
        print("[1] 枚举离散模式...")
        k_mw, n_modes, shells = enumerate_shells_for_bins(kf, kmin, kmax_b)

        print("\n  bin 对比:")
        print(f"  {'bin':>3s} {'k_cen':>8s} {'k_mw':>8s} {'Δk/k%':>8s} {'N_modes':>8s} {'N_shells':>9s}")
        for ib in range(min(10, ndp)):
            dk = (k_mw[ib] - kcen[ib]) / kcen[ib] * 100 if kcen[ib] > 0 else 0
            print(f"  {ib:>3d} {kcen[ib]:>8.5f} {k_mw[ib]:>8.5f} {dk:>+8.3f} {n_modes[ib]:>8d} {len(shells[ib]):>9d}")

        # 方法 A: 标准拟合（bin center）
        print("\n[2A] 标准拟合 (bin center)...")
        edges = np.concatenate([kmin, [kmax_b[-1]]])
        nmodes_arr = 4/3*np.pi*(edges[1:]**3 - edges[:-1]**3)
        dps = PowerSpectrumStatistics(edges=edges, modes=kcen,
            power_nonorm=np.array([pk_m]), nmodes=nmodes_arr, ells=[0],
            shotnoise_nonorm=0, statistic='multipole')
        ml = []
        for i in range(cov_mk.shape[0]):
            t = dps.deepcopy(); t.power_nonorm.flat[...] = np.array([cov_mk[i]]).ravel()
            ml.append(t)

        tmpl = FixedPowerSpectrumTemplate(z=1.0, fiducial=cosmo)
        th = PNGTracerPowerSpectrumMultipoles(template=tmpl, mode='b-p')
        th.init.params['p'].update(fixed=True, value=1.2)
        th.init.params['sn0'].update(fixed=True, value=0)
        th.init.params['sigmas'].update(fixed=False, value=0)
        obs = TracerPowerSpectrumMultipolesObservable(data=dps, covariance=ml,
            klim={0:[float(kcen.min()),float(kcen.max()),float(kcen[1]-kcen[0])]}, theory=th)
        like = ObservablesGaussianLikelihood(observables=[obs]); _=like()
        like.all_params['p'].update(fixed=True,value=1.2)
        like.all_params['sn0'].update(fixed=True,value=0)
        like.all_params['sigmas'].update(fixed=False,value=0)
        prof = MinuitProfiler(like, seed=66)
        profiles = prof.maximize(niterations=27)
        bf_std = profiles.bestfit.choice(input=True)
        fnl_std = float(bf_std['fnl_loc']); b1_std = float(bf_std['b1'])
        sig_std = float(bf_std['sigmas'])
        print(f"  标准 best-fit: fnl={fnl_std:.2f}, b1={b1_std:.4f}, sig={sig_std:.2f}")

        # 方法 B: Mode-weighted 拟合（用 k_mw 代替 kcen）
        print("\n[2B] Mode-weighted 拟合 (k_mw)...")
        dps_mw = PowerSpectrumStatistics(edges=edges, modes=k_mw,
            power_nonorm=np.array([pk_m]), nmodes=nmodes_arr, ells=[0],
            shotnoise_nonorm=0, statistic='multipole')
        ml_mw = []
        for i in range(cov_mk.shape[0]):
            t = dps_mw.deepcopy(); t.power_nonorm.flat[...] = np.array([cov_mk[i]]).ravel()
            ml_mw.append(t)

        tmpl_mw = FixedPowerSpectrumTemplate(z=1.0, fiducial=cosmo)
        th_mw = PNGTracerPowerSpectrumMultipoles(template=tmpl_mw, mode='b-p')
        th_mw.init.params['p'].update(fixed=True, value=1.2)
        th_mw.init.params['sn0'].update(fixed=True, value=0)
        th_mw.init.params['sigmas'].update(fixed=False, value=0)
        obs_mw = TracerPowerSpectrumMultipolesObservable(data=dps_mw, covariance=ml_mw,
            klim={0:[float(k_mw.min()),float(k_mw.max()),float(k_mw[1]-k_mw[0])]}, theory=th_mw)
        like_mw = ObservablesGaussianLikelihood(observables=[obs_mw]); _=like_mw()
        like_mw.all_params['p'].update(fixed=True,value=1.2)
        like_mw.all_params['sn0'].update(fixed=True,value=0)
        like_mw.all_params['sigmas'].update(fixed=False,value=0)
        prof_mw = MinuitProfiler(like_mw, seed=66)
        profiles_mw = prof_mw.maximize(niterations=27)
        bf_mw = profiles_mw.bestfit.choice(input=True)
        fnl_mw = float(bf_mw['fnl_loc']); b1_mw = float(bf_mw['b1'])
        sig_mw = float(bf_mw['sigmas'])
        print(f"  MW best-fit:   fnl={fnl_mw:.2f}, b1={b1_mw:.4f}, sig={sig_mw:.2f}")
        print(f"  Δfnl = {fnl_mw - fnl_std:+.2f}, Δb1 = {b1_mw - b1_std:+.4f}")

        # 评估 P(k) 并计算 FullDiscrete
        print("\n[3] FullDiscrete 对比...")
        kmax_disc = 15.0
        qmax = int((kmax_disc/kf)**2); nmax = int(kmax_disc/kf)
        gq = gq_fft(qmax, nmax)
        kd = np.geomspace(kf*0.5, kmax_disc*1.1, 300000)

        # 标准 best-fit P(k)
        tmpl_d = FixedPowerSpectrumTemplate(z=1.0, fiducial=cosmo)
        th_d = PNGTracerPowerSpectrumMultipoles(k=kd, template=tmpl_d, mode='b-p')
        th_d.init.params['p'].update(fixed=True,value=1.2)
        th_d.init.params['sn0'].update(fixed=True,value=0)
        th_d.init.params['sigmas'].update(fixed=False,value=0)
        th_d(**bf_std); pd_std = np.array(th_d.power[0], dtype=float)

        # MW best-fit P(k)
        tmpl_d2 = FixedPowerSpectrumTemplate(z=1.0, fiducial=cosmo)
        th_d2 = PNGTracerPowerSpectrumMultipoles(k=kd, template=tmpl_d2, mode='b-p')
        th_d2.init.params['p'].update(fixed=True,value=1.2)
        th_d2.init.params['sn0'].update(fixed=True,value=0)
        th_d2.init.params['sigmas'].update(fixed=False,value=0)
        th_d2(**bf_mw); pd_mw = np.array(th_d2.power[0], dtype=float)

        print("  FullDiscrete (标准 fit)...")
        xi_std = xi_disc(s, gq, kf, V, kd, pd_std)
        print("  FullDiscrete (MW fit)...")
        xi_mw = xi_disc(s, gq, kf, V, kd, pd_mw)

        # 指标
        ma_std, ms_std = met(r2d, r2s, s**2 * xi_std)
        ma_mw, ms_mw = met(r2d, r2s, s**2 * xi_mw)

        print(f"\n  {'方法':<30s} {'mean|Δ/σ|':>10s} {'mean(Δ/σ)':>10s}")
        print(f"  {'-'*54}")
        print(f"  {'FullDisc + 标准fit':<30s} {ma_std:>10.4f} {ms_std:>+10.4f}")
        print(f"  {'FullDisc + MW fit':<30s} {ma_mw:>10.4f} {ms_mw:>+10.4f}")

        all_results.append({
            "cfg": cfg_name,
            "fnl_std": fnl_std, "b1_std": b1_std, "sig_std": sig_std,
            "fnl_mw": fnl_mw, "b1_mw": b1_mw, "sig_mw": sig_mw,
            "ma_std": ma_std, "ms_std": ms_std,
            "ma_mw": ma_mw, "ms_mw": ms_mw,
        })

    # 汇总
    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    md = ["# Mission 9: Mode-Weighted Fitting 结果\n\n"]
    for r in all_results:
        print(f"\n{r['cfg']}:")
        print(f"  标准: fnl={r['fnl_std']:.2f}, b1={r['b1_std']:.4f} → Disc mean(Δ/σ)={r['ms_std']:+.4f}")
        print(f"  MW:   fnl={r['fnl_mw']:.2f}, b1={r['b1_mw']:.4f} → Disc mean(Δ/σ)={r['ms_mw']:+.4f}")
        print(f"  Δfnl={r['fnl_mw']-r['fnl_std']:+.2f}, Δb1={r['b1_mw']-r['b1_std']:+.4f}")

        md.append(f"## {r['cfg']}\n\n")
        md.append(f"| 拟合方式 | fnl | b1 | sigmas | Disc mean\\|Δ/σ\\| | Disc mean(Δ/σ) |\n")
        md.append(f"|---|---|---|---|---|---|\n")
        md.append(f"| 标准 (k_center) | {r['fnl_std']:.2f} | {r['b1_std']:.4f} | {r['sig_std']:.2f} | {r['ma_std']:.4f} | {r['ms_std']:+.4f} |\n")
        md.append(f"| MW (k_mw) | {r['fnl_mw']:.2f} | {r['b1_mw']:.4f} | {r['sig_mw']:.2f} | {r['ma_mw']:.4f} | {r['ms_mw']:+.4f} |\n")
        md.append(f"\nΔfnl = {r['fnl_mw']-r['fnl_std']:+.2f}, Δb1 = {r['b1_mw']-r['b1_std']:+.4f}\n\n")

    md.append(f"\n总耗时: {time.time()-T0:.0f}s\n")
    with open(os.path.join(OUT, "mission9_mw_fitting.md"), 'w') as f:
        f.writelines(md)

    print(f"\n完成! {time.time()-T0:.0f}s")

if __name__ == "__main__":
    main()
