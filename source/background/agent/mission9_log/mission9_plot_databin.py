#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission 9: DataBin 方法结果图
==============================
画 3Gpc fnl100 和 1Gpc fnl100 的 r²ξ₀ 对比图 + 残差图。
方法: Data, Baseline, ExpWindow, FullDiscrete, DataBin
"""

from __future__ import annotations
import os, glob, re, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import next_fast_len, rfft, irfft, fht, fhtoffset
from collections import defaultdict

OUT = os.path.dirname(os.path.abspath(__file__))

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

def xi_disc_custom_p(s, gq, kf, V, p_func, chunk=500000):
    """全离散求和，P(k) 由 p_func 提供"""
    qnz = np.nonzero(gq[1:])[0]+1; xi = np.zeros(len(s))
    total = len(qnz)
    for i0 in range(0, total, chunk):
        qb = qnz[i0:i0+chunk]; kv = kf*np.sqrt(qb.astype(float))
        pv = p_func(kv); w = gq[qb].astype(float)*pv
        for js in range(0, len(s), 10):
            je=min(js+10,len(s)); xi[js:je]+=np.dot(w,j0(np.outer(kv,s[js:je])))
        if (i0//chunk) % 40 == 0:
            print(f"    {min(i0+chunk,total)}/{total}")
    return xi/V

def xi_fftlog(k_grid, p0, kmin, kmax, ir_win=None):
    lk=np.log(k_grid); l0,l1=np.log(kmin),np.log(kmax); dl=0.06*(l1-l0)
    w=np.zeros_like(k_grid)
    left,right=l0+dl,l1-dl
    m=(lk>=l0)&(lk<left); w[m]=0.5*(1-np.cos(np.pi*(lk[m]-l0)/dl))
    m=(lk>=left)&(lk<=right); w[m]=1.0
    m=(lk>right)&(lk<=l1); w[m]=0.5*(1+np.cos(np.pi*(lk[m]-right)/dl))
    if ir_win is not None: w*=ir_win
    dln=np.log(k_grid[1]/k_grid[0]); off=fhtoffset(dln,mu=0.5,initial=0,bias=0)
    A=fht((k_grid**1.5)*p0*w,dln=dln,mu=0.5,offset=off,bias=0)
    n=k_grid.size; jj=np.arange(n)
    sg=np.exp(-(np.log(k_grid[0])+np.log(k_grid[-1]))/2+off+(jj-(n-1)/2)*dln)
    return sg, np.sqrt(np.pi/2)/(2*np.pi**2)*A/sg**1.5

def ir_window_func(k, kf, x):
    """IR参数化窗口（含归一化因子）: W = [1-exp(-(k/kf)^x)] / [1-exp(-1)]"""
    norm = 1.0 - np.exp(-1.0)  # ≈ 0.6321
    w=np.ones_like(k); m=k<kf
    w[m] = (1.0 - np.exp(-(k[m]/kf)**x)) / norm
    return w

def met(r2d, r2s, r2m):
    ds=(r2d-r2m)/r2s; return np.nanmean(np.abs(ds)), np.nanmean(ds)

# ============================================================
def main():
    T0 = time.time()

    from cosmoprimo import Cosmology
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler
    from pypower import PowerSpectrumStatistics

    cosmo = Cosmology(h=0.6711, Omega_b=0.049, Omega_cdm=0.3175-0.049,
                      sigma8=0.834, n_s=0.9624, engine='class')
    Z=1.0; P_FIX=1.2; KMAX=15.0

    configs = [
        ("3Gpc fnl100", 3000.0,
         "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat",
         "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
         "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",
         2, 99, 20, 12.0),
        ("1Gpc fnl100", 1000.0,
         "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat",
         "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
         "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat",
         1, 50, 20, 4.0),
    ]

    # 两个盒子画在一个大图中（2 行 × 2 列）
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle("Mission 9: DataBin method vs existing methods", fontsize=15, y=0.98)

    for idx, (title, L, pk_pat, cov_pat, pcf_pat, rmin, rmax, ndp, ir_x) in enumerate(configs):
        kf = 2*np.pi/L; V = L**3
        print(f"\n{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}")

        # 数据
        _kc,_km,_kx,_pk,_,_ = load_pk(pk_pat, rmin, rmax, ndp+1)
        _,_,_,_cov,_,_ = load_pk(cov_pat, rmin, rmax, ndp+1)
        first_valid = 0
        for ib in range(len(_kc)):
            if _pk[:,ib].mean() > 0: first_valid = ib; break
        kcen=_kc[first_valid:first_valid+ndp]
        kmin_b=_km[first_valid:first_valid+ndp]
        kmax_b=_kx[first_valid:first_valid+ndp]
        pk_mocks=_pk[:,first_valid:first_valid+ndp]
        cov_mocks=_cov[:,first_valid:first_valid+ndp]
        pk_mean=pk_mocks.mean(0)
        s, xi_m, xi_s = load_pcf(pcf_pat, rmin, rmax)
        r2d=s**2*xi_m; r2s=s**2*xi_s

        # 拟合
        print("  拟合...")
        edges=np.concatenate([kmin_b,[kmax_b[-1]]])
        nm=4/3*np.pi*(edges[1:]**3-edges[:-1]**3)
        dps=PowerSpectrumStatistics(edges=edges,modes=kcen,
            power_nonorm=np.array([pk_mean]),nmodes=nm,ells=[0],
            shotnoise_nonorm=0,statistic='multipole')
        ml=[]
        for i in range(cov_mocks.shape[0]):
            t=dps.deepcopy(); t.power_nonorm.flat[...]=np.array([cov_mocks[i]]).ravel(); ml.append(t)
        tmpl=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
        th=PNGTracerPowerSpectrumMultipoles(template=tmpl,mode='b-p')
        th.init.params['p'].update(fixed=True,value=P_FIX)
        th.init.params['sn0'].update(fixed=True,value=0)
        th.init.params['sigmas'].update(fixed=False,value=0)
        obs=TracerPowerSpectrumMultipolesObservable(data=dps,covariance=ml,
            klim={0:[float(kcen.min()),float(kcen.max()),float(kcen[1]-kcen[0])]},theory=th)
        like=ObservablesGaussianLikelihood(observables=[obs]); _=like()
        like.all_params['p'].update(fixed=True,value=P_FIX)
        like.all_params['sn0'].update(fixed=True,value=0)
        like.all_params['sigmas'].update(fixed=False,value=0)
        prof=MinuitProfiler(like,seed=66); pr=prof.maximize(niterations=27)
        bf=pr.bestfit.choice(input=True)
        fnl=float(bf['fnl_loc']); b1=float(bf['b1']); sig=float(bf['sigmas'])
        print(f"  fnl={fnl:.2f}, b1={b1:.4f}, sig={sig:.2f}")

        # P(k) 密集网格
        kd = np.geomspace(kf*0.5, KMAX*1.1, 300000)
        def eval_pk(k_arr):
            tmpl2=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
            th2=PNGTracerPowerSpectrumMultipoles(k=k_arr,template=tmpl2,mode='b-p')
            th2.init.params['p'].update(fixed=True,value=P_FIX)
            th2.init.params['sn0'].update(fixed=True,value=0)
            th2.init.params['sigmas'].update(fixed=False,value=0)
            th2(**bf); return np.array(th2.power[0],dtype=float)
        pd = eval_pk(kd)

        # g_q
        qmax=int((KMAX/kf)**2); nmax=int(KMAX/kf)
        print("  g_q...")
        gq = gq_fft(qmax, nmax)

        # 1) Baseline
        print("  Baseline...")
        kg=np.geomspace(kf/4,KMAX*4,4096); pg=eval_pk(kg)
        sb,xb=xi_fftlog(kg,pg,kf,KMAX)
        xi_baseline=np.interp(s,np.sort(sb),xb[np.argsort(sb)])

        # 2) ExpWindow
        print("  ExpWindow...")
        kgw=np.geomspace(1e-4/4,KMAX*4,4096); pgw=eval_pk(kgw)
        iw=ir_window_func(kgw,kf,ir_x)
        sw,xw=xi_fftlog(kgw,pgw,1e-4,KMAX,ir_win=iw)
        xi_expwin=np.interp(s,np.sort(sw),xw[np.argsort(sw)])

        # 3) FullDiscrete
        print("  FullDiscrete...")
        p_model = lambda k: np.interp(k, kd, pd)
        xi_disc = xi_disc_custom_p(s, gq, kf, V, p_model)

        # 4) DataBin
        print("  DataBin...")
        def p_databin(k_arr):
            p_out = np.interp(k_arr, kd, pd)
            for ib in range(len(kcen)):
                mask = (k_arr >= kmin_b[ib]) & (k_arr < kmax_b[ib])
                p_out[mask] = pk_mean[ib]
            return p_out
        xi_databin = xi_disc_custom_p(s, gq, kf, V, p_databin)

        # 指标
        methods = {
            "Baseline": (s**2*xi_baseline, 'tab:blue', '--'),
            "ExpWindow": (s**2*xi_expwin, 'tab:red', '-'),
            "FullDiscrete": (s**2*xi_disc, 'tab:green', '-'),
            "DataBin": (s**2*xi_databin, 'darkorange', '-'),
        }

        print(f"\n  {'方法':<16s} {'mean|Δ/σ|':>10s} {'mean(Δ/σ)':>10s}")
        print(f"  {'-'*40}")
        for nm, (r2m, _, _) in methods.items():
            ma, ms = met(r2d, r2s, r2m)
            print(f"  {nm:<16s} {ma:>10.4f} {ms:>+10.4f}")

        # 画图
        ax_top = axes[0, idx]
        ax_bot = axes[1, idx]

        # 上图: r²ξ₀
        ax_top.errorbar(s, r2d, yerr=r2s, fmt='ko', ms=3, capsize=2,
                        label='Data mean', zorder=10)
        for nm, (r2m, color, ls) in methods.items():
            ma, ms = met(r2d, r2s, r2m)
            ax_top.plot(s, r2m, color=color, ls=ls, lw=2 if nm=='DataBin' else 1.3,
                        label=f"{nm} ({ma:.3f}, {ms:+.3f})")
        ax_top.set_ylabel(r'$r^2\xi_0(r)$', fontsize=12)
        ax_top.set_title(f"{title}\nfnl={fnl:.1f}, b1={b1:.3f}", fontsize=12)
        ax_top.legend(fontsize=8, title="method (mean|Δ/σ|, mean Δ/σ)")
        ax_top.grid(alpha=0.3)

        # 下图: (Data - Model)/σ
        ax_bot.axhline(0, color='k', lw=0.5)
        ax_bot.axhline(1, color='gray', ls=':', lw=0.5)
        ax_bot.axhline(-1, color='gray', ls=':', lw=0.5)
        for nm, (r2m, color, ls) in methods.items():
            ds = (r2d - r2m) / r2s
            lw = 2.5 if nm == 'DataBin' else 1.2
            ax_bot.plot(s, ds, color=color, ls=ls, lw=lw, marker='o', ms=2 if nm=='DataBin' else 1.5,
                        label=nm, alpha=0.9 if nm=='DataBin' else 0.6)
        ax_bot.set_xlabel(r'$r\,[\mathrm{Mpc}/h]$', fontsize=12)
        ax_bot.set_ylabel(r'$(Data - Model)/\sigma$ of $r^2\xi_0$', fontsize=12)
        ax_bot.legend(fontsize=8)
        ax_bot.grid(alpha=0.3)
        ax_bot.set_ylim(-3, 3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    figpath = os.path.join(OUT, "mission9_databin_results.png")
    fig.savefig(figpath, dpi=150)
    plt.close()
    print(f"\n图已保存: {figpath}")
    print(f"总耗时: {time.time()-T0:.0f}s")

if __name__ == "__main__":
    main()
