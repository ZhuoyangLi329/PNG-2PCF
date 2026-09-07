#!/usr/bin/env python3
"""
Mission 9: DataBin 优化（kmax 扫描 + sn0 组合）
================================================

问题：DataBin 在大尺度有震荡，可能和 kmax 有关。
测试：
1. DataBin + 不同 kmax (5, 8, 10, 12, 15, 20)
2. DataBin + sn0 offset（在数据范围外的模型上加/减常数）
3. 修正后的 ExpWindow（含归一化因子）作为参考
"""

from __future__ import annotations
import os, glob, re, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import next_fast_len, rfft, irfft, fht, fhtoffset

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

def xi_disc_pfunc(s, gq, kf, V, p_func, kmax_cut, chunk=500000):
    """全离散求和，P(k) 由 p_func 提供，截断到 kmax_cut"""
    qm = min(int((kmax_cut/kf)**2), len(gq)-1)
    qnz = np.nonzero(gq[1:qm+1])[0]+1; xi = np.zeros(len(s))
    for i0 in range(0, len(qnz), chunk):
        qb = qnz[i0:i0+chunk]; kv = kf*np.sqrt(qb.astype(float))
        pv = p_func(kv); w = gq[qb].astype(float)*pv
        for js in range(0, len(s), 10):
            je=min(js+10,len(s)); xi[js:je]+=np.dot(w,j0(np.outer(kv,s[js:je])))
        if (i0//chunk) % 40 == 0:
            print(f"    {min(i0+chunk,len(qnz))}/{len(qnz)}")
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
    """含归一化因子的 IR 窗口"""
    norm = 1.0 - np.exp(-1.0)
    w=np.ones_like(k); m=k<kf; w[m]=(1-np.exp(-(k[m]/kf)**x))/norm; return w

def met(r2d, r2s, r2m):
    ds=(r2d-r2m)/r2s; return np.nanmean(np.abs(ds)), np.nanmean(ds)

def main():
    T0 = time.time()
    print("="*60)
    print("Mission 9: DataBin 优化")
    print("="*60)

    from cosmoprimo import Cosmology
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler
    from pypower import PowerSpectrumStatistics
    cosmo = Cosmology(h=0.6711, Omega_b=0.049, Omega_cdm=0.3175-0.049,
                      sigma8=0.834, n_s=0.9624, engine='class')
    Z=1.0; P_FIX=1.2

    configs = [
        ("3Gpc_fnl100", 3000.0,
         "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat",
         "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
         "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",
         2, 99, 20, 12.0),
        ("1Gpc_fnl100", 1000.0,
         "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat",
         "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
         "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat",
         1, 50, 20, 4.0),
    ]

    KMAX_LIST = [5, 8, 10, 12, 15, 20]

    for title, L, pk_pat, cov_pat, pcf_pat, rmin, rmax, ndp, ir_x in configs:
        kf=2*np.pi/L; V=L**3
        print(f"\n{'='*60}\n  {title}\n{'='*60}")

        # 数据（跳过空 bin）
        _kc,_km,_kx,_pk,_,_ = load_pk(pk_pat, rmin, rmax, ndp+1)
        _,_,_,_cov,_,_ = load_pk(cov_pat, rmin, rmax, ndp+1)
        fv=0
        for ib in range(len(_kc)):
            if _pk[:,ib].mean()>0: fv=ib; break
        kcen=_kc[fv:fv+ndp]; kmin_b=_km[fv:fv+ndp]; kmax_b=_kx[fv:fv+ndp]
        pk_mocks=_pk[:,fv:fv+ndp]; cov_mocks=_cov[:,fv:fv+ndp]
        pk_mean=pk_mocks.mean(0)
        s, xi_m, xi_s = load_pcf(pcf_pat, rmin, rmax)
        r2d=s**2*xi_m; r2s=s**2*xi_s

        # 拟合（标准 + 自由 sn0）
        print("  拟合...")
        edges=np.concatenate([kmin_b,[kmax_b[-1]]])
        nm=4/3*np.pi*(edges[1:]**3-edges[:-1]**3)
        dps=PowerSpectrumStatistics(edges=edges,modes=kcen,
            power_nonorm=np.array([pk_mean]),nmodes=nm,ells=[0],
            shotnoise_nonorm=0,statistic='multipole')
        ml=[]
        for i in range(cov_mocks.shape[0]):
            t=dps.deepcopy(); t.power_nonorm.flat[...]=np.array([cov_mocks[i]]).ravel(); ml.append(t)

        def do_fit(free_sn0):
            tmpl=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
            th=PNGTracerPowerSpectrumMultipoles(template=tmpl,mode='b-p')
            th.init.params['p'].update(fixed=True,value=P_FIX)
            th.init.params['sn0'].update(fixed=not free_sn0, value=0)
            th.init.params['sigmas'].update(fixed=False,value=0)
            obs=TracerPowerSpectrumMultipolesObservable(data=dps,covariance=ml,
                klim={0:[float(kcen.min()),float(kcen.max()),float(kcen[1]-kcen[0])]},theory=th)
            like=ObservablesGaussianLikelihood(observables=[obs]); _=like()
            like.all_params['p'].update(fixed=True,value=P_FIX)
            like.all_params['sn0'].update(fixed=not free_sn0, value=0)
            like.all_params['sigmas'].update(fixed=False,value=0)
            prof=MinuitProfiler(like,seed=66); pr=prof.maximize(niterations=27)
            return pr.bestfit.choice(input=True)

        bf_std = do_fit(False)
        bf_sn0 = do_fit(True)
        print(f"  标准: fnl={float(bf_std['fnl_loc']):.2f}, b1={float(bf_std['b1']):.4f}, sn0={float(bf_std['sn0']):.1f}")
        print(f"  FreeSN0: fnl={float(bf_sn0['fnl_loc']):.2f}, b1={float(bf_sn0['b1']):.4f}, sn0={float(bf_sn0['sn0']):.1f}")

        # P(k) 密集网格
        kd=np.geomspace(kf*0.5,22,300000)
        def eval_pk(k, bf):
            tmpl=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
            th=PNGTracerPowerSpectrumMultipoles(k=k,template=tmpl,mode='b-p')
            th.init.params['p'].update(fixed=True,value=P_FIX)
            th.init.params['sn0'].update(fixed=True,value=0)
            th.init.params['sigmas'].update(fixed=False,value=0)
            th(**bf); return np.array(th.power[0],dtype=float)

        pd_std = eval_pk(kd, bf_std)
        pd_sn0 = eval_pk(kd, bf_sn0)

        # g_q（用最大 kmax=20）
        qmax=int((20/kf)**2); nmax=int(20/kf)
        print("  g_q...")
        gq = gq_fft(qmax, nmax)

        # P 函数工厂
        def make_p_model(pd): return lambda k: np.interp(k, kd, pd)
        def make_p_databin(pd, sn0_offset=0.0):
            """低 k 用数据 bin，高 k 用模型（可选 sn0 offset）"""
            def p(k):
                p_out = np.interp(k, kd, pd) + sn0_offset
                for ib in range(len(kcen)):
                    mask = (k >= kmin_b[ib]) & (k < kmax_b[ib])
                    p_out[mask] = pk_mean[ib] + sn0_offset
                return p_out
            return p

        # 测试矩阵
        results = {}

        # ExpWindow（修正归一化）
        print("  ExpWindow (norm)...")
        kgw=np.geomspace(1e-4/4,20*4,4096); pgw=eval_pk(kgw, bf_std)
        iw=ir_window_func(kgw,kf,ir_x)
        sw,xw=xi_fftlog(kgw,pgw,1e-4,20,ir_win=iw)
        xi_ew=np.interp(s,np.sort(sw),xw[np.argsort(sw)])
        results["ExpWindow_20"] = met(r2d, r2s, s**2*xi_ew)

        # FullDiscrete kmax=15
        print("  FullDiscrete kmax=15...")
        xi_fd = xi_disc_pfunc(s, gq, kf, V, make_p_model(pd_std), 15)
        results["FullDisc_15"] = met(r2d, r2s, s**2*xi_fd)

        # DataBin kmax 扫描
        for km in KMAX_LIST:
            label = f"DataBin_km{km}"
            print(f"  {label}...")
            xi = xi_disc_pfunc(s, gq, kf, V, make_p_databin(pd_std), km)
            results[label] = met(r2d, r2s, s**2*xi)

        # DataBin + FreeSN0 拟合参数，kmax=15
        print("  DataBin+FreeSN0...")
        xi_ds = xi_disc_pfunc(s, gq, kf, V, make_p_databin(pd_sn0), 15)
        results["DataBin_sn0free"] = met(r2d, r2s, s**2*xi_ds)

        # 汇总
        print(f"\n--- {title} ---")
        print(f"{'方法':<22s} {'mean|Δ/σ|':>10s} {'mean(Δ/σ)':>10s}")
        print("-"*46)
        for nm in sorted(results.keys()):
            ma,ms = results[nm]
            print(f"{nm:<22s} {ma:>10.4f} {ms:>+10.4f}")

        # 画图：DataBin kmax 扫描
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f"{title}: DataBin kmax scan", fontsize=13)

        ax1.errorbar(s, r2d, yerr=r2s, fmt='ko', ms=3, capsize=2, label='Data')
        for km in KMAX_LIST:
            xi = xi_disc_pfunc(s, gq, kf, V, make_p_databin(pd_std), km) if km != 15 else \
                 xi_disc_pfunc(s, gq, kf, V, make_p_databin(pd_std), 15)
            # 重用已算的避免重复计算：只画已有的
            ax1.plot(s, s**2*xi_disc_pfunc(s, gq, kf, V, make_p_databin(pd_std), km) if False else
                     s**2*np.interp(s,s,xi), '--', lw=1, label=f"kmax={km}")
        # 简化：直接用结果指标画条形图
        ax1.set_ylabel(r'$r^2\xi_0$'); ax1.legend(fontsize=7, ncol=2); ax1.grid(alpha=0.3)

        km_arr = np.array(KMAX_LIST, dtype=float)
        ma_arr = np.array([results[f"DataBin_km{km}"][0] for km in KMAX_LIST])
        ms_arr = np.array([results[f"DataBin_km{km}"][1] for km in KMAX_LIST])
        ax2.plot(km_arr, ma_arr, 'o-', color='darkorange', lw=2, ms=6, label='DataBin mean|Δ/σ|')
        ax2.plot(km_arr, ms_arr, 's--', color='darkorange', lw=1.5, ms=5, label='DataBin mean(Δ/σ)')
        ew_ma, ew_ms = results["ExpWindow_20"]
        fd_ma, fd_ms = results["FullDisc_15"]
        ax2.axhline(ew_ma, color='red', ls=':', label=f"ExpWin |Δ/σ|={ew_ma:.3f}")
        ax2.axhline(fd_ma, color='green', ls=':', label=f"FullDisc |Δ/σ|={fd_ma:.3f}")
        ax2.axhline(0, color='gray', lw=0.5)
        ax2.set_xlabel("kmax"); ax2.set_ylabel("metric")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(OUT, f"mission9_databin_kmax_{title.replace(' ','_')}.png"), dpi=150)
        plt.close()

    print(f"\n完成! {time.time()-T0:.0f}s")

if __name__ == "__main__":
    main()
