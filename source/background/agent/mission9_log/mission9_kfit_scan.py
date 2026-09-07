#!/usr/bin/env python3
"""
Mission 9 支线: 拟合 kmax 扫描 + EZmock covariance
===================================================
测试增大拟合范围（更多 k-bin）是否改善小尺度 2PCF 建模。
使用 EZmock（~1500个）作为 covariance。
只测 3Gpc fnl100。
"""
from __future__ import annotations
import os,glob,re,time
import numpy as np
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import next_fast_len,rfft,irfft,fht,fhtoffset

OUT=os.path.dirname(os.path.abspath(__file__))
L=3000.0;kf=2*np.pi/L;V=L**3;Z=1.0;P_FIX=1.2;KMAX_DISC=15.0

# 拟合 k-bin 数扫描
NFIT_LIST = [20, 30, 40, 50, 60]

def rid(f):
    m=re.search(r'N(\d+)',os.path.basename(f));return int(m.group(1)) if m else -1
def load_pk_fastpm(pat,rmin,rmax,ndp):
    fs=[f for f in sorted(glob.glob(pat),key=rid) if rmin<=rid(f)<=rmax]
    ref=np.loadtxt(fs[0],comments='#')
    v=np.array([np.loadtxt(f,comments='#')[:ndp,5] for f in fs])
    return ref[:ndp,0],ref[:ndp,1],ref[:ndp,2],v,v.mean(0)
def load_pcf(pat,rmin,rmax):
    fs=[f for f in sorted(glob.glob(pat),key=rid) if rmin<=rid(f)<=rmax]
    s=np.loadtxt(fs[0],comments='#')[:,0]
    v=np.array([np.loadtxt(f,comments='#')[:,3] for f in fs])
    return s,v.mean(0),v.std(0,ddof=1)
def load_ezmock_pk(cov_glob, k_target, ndp):
    """加载 EZmock pk 作为 covariance，插值到 k_target 网格"""
    files = sorted(glob.glob(cov_glob))
    p0_list = []
    skipped = 0
    for fp in files:
        try:
            arr = np.loadtxt(fp, comments='#')
            if arr.shape[0] < ndp:
                skipped += 1; continue
            k_in = arr[:ndp, 0]; p0_in = arr[:ndp, 5]
            if np.allclose(k_in, k_target, rtol=0, atol=1e-10):
                p0_list.append(p0_in)
            else:
                p0_list.append(np.interp(k_target, arr[:,0], arr[:,5]))
        except:
            skipped += 1
    print(f"  EZmock: {len(p0_list)} used, {skipped} skipped")
    return np.array(p0_list)

def j0(x):
    x=np.asarray(x,float);o=np.ones_like(x);m=x!=0;o[m]=np.sin(x[m])/x[m];return o
def gq_fft(qmax,nmax):
    a=np.zeros(qmax+1,dtype=np.float32);a[0]=1
    sq=np.arange(1,nmax+1,dtype=np.int64)**2;a[sq[sq<=qmax]]=2
    nfft=next_fast_len(3*qmax+1);fa=rfft(a,n=nfft)
    return np.rint(irfft(fa**3,n=nfft)[:qmax+1]).astype(np.int64)
def xi_disc(s,gq,kf,V,pfunc,chunk=500000):
    qnz=np.nonzero(gq[1:])[0]+1;xi=np.zeros(len(s))
    for i0 in range(0,len(qnz),chunk):
        qb=qnz[i0:i0+chunk];kv=kf*np.sqrt(qb.astype(float))
        pv=pfunc(kv);w=gq[qb].astype(float)*pv
        for js in range(0,len(s),10):
            je=min(js+10,len(s));xi[js:je]+=np.dot(w,j0(np.outer(kv,s[js:je])))
    return xi/V
def met(r2d,r2s,r2m):
    ds=(r2d-r2m)/r2s;return np.nanmean(np.abs(ds)),np.nanmean(ds)
def met_range(r2d,r2s,r2m,s,smin,smax):
    mask=(s>=smin)&(s<=smax)
    ds=(r2d[mask]-r2m[mask])/r2s[mask]
    return np.nanmean(np.abs(ds)),np.nanmean(ds)

def main():
    T0=time.time()
    print("="*60)
    print("Mission 9 支线: 拟合 kmax 扫描 (3Gpc fnl100, EZmock cov)")
    print("="*60)

    from cosmoprimo import Cosmology
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate,PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler
    from pypower import PowerSpectrumStatistics
    cosmo=Cosmology(h=0.6711,Omega_b=0.049,Omega_cdm=0.3175-0.049,sigma8=0.834,n_s=0.9624,engine='class')

    # 加载全部数据（取最大 ndp）
    max_ndp = max(NFIT_LIST)
    print(f"\n[1] 加载数据 (前 {max_ndp} bins)...")
    kcen_all,kmin_all,kmax_all,pk_mk_all,pk_mean_all = load_pk_fastpm(
        "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat",2,99,max_ndp)
    s,xi_m,xi_s = load_pcf(
        "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",2,99)
    r2d=s**2*xi_m; r2s=s**2*xi_s
    print(f"  k range: [{kcen_all[0]:.5f}, {kcen_all[-1]:.5f}]")

    # EZmock covariance
    print("\n[2] 加载 EZmock covariance...")
    ez_glob = "/pscratch/sd/l/lzy/cov_mock/B3000G768Z0N4417983_b0.18d5r270c1.65_seed*/PK_EZmock_B3000G768Z0N4417983_b0.18d5r270c1.65_seed*_RSD.dat"
    ez_cov_all = load_ezmock_pk(ez_glob, kcen_all, max_ndp)

    # g_q
    print("\n[3] g_q...")
    qmax=int((KMAX_DISC/kf)**2);nmax=int(KMAX_DISC/kf)
    gq=gq_fft(qmax,nmax)

    # 扫描不同拟合范围
    results = []
    xi_curves = {}

    for nfit in NFIT_LIST:
        print(f"\n{'='*50}")
        print(f"  N_fit = {nfit} (k_max_fit = {kcen_all[nfit-1]:.4f})")
        print(f"{'='*50}")

        kcen=kcen_all[:nfit]; kmin_b=kmin_all[:nfit]; kmax_b=kmax_all[:nfit]
        pk_mean=pk_mean_all[:nfit]; ez_cov=ez_cov_all[:,:nfit]

        # 拟合（EZmock covariance）
        edges=np.concatenate([kmin_b,[kmax_b[-1]]])
        nm=4/3*np.pi*(edges[1:]**3-edges[:-1]**3)
        dps=PowerSpectrumStatistics(edges=edges,modes=kcen,power_nonorm=np.array([pk_mean]),
            nmodes=nm,ells=[0],shotnoise_nonorm=0,statistic='multipole')
        ml=[]
        for i in range(ez_cov.shape[0]):
            t=dps.deepcopy();t.power_nonorm.flat[...]=np.array([ez_cov[i]]).ravel();ml.append(t)

        tmpl=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
        th=PNGTracerPowerSpectrumMultipoles(template=tmpl,mode='b-p')
        th.init.params['p'].update(fixed=True,value=P_FIX)
        th.init.params['sn0'].update(fixed=True,value=0)
        th.init.params['sigmas'].update(fixed=False,value=0)
        obs=TracerPowerSpectrumMultipolesObservable(data=dps,covariance=ml,
            klim={0:[float(kcen.min()),float(kcen.max()),float(kcen[1]-kcen[0])]},theory=th)
        like=ObservablesGaussianLikelihood(observables=[obs]);_=like()
        like.all_params['p'].update(fixed=True,value=P_FIX)
        like.all_params['sn0'].update(fixed=True,value=0)
        like.all_params['sigmas'].update(fixed=False,value=0)
        prof=MinuitProfiler(like,seed=66);pr=prof.maximize(niterations=27)
        bf=pr.bestfit.choice(input=True)
        fnl_fit=float(bf['fnl_loc']);b1_fit=float(bf['b1']);sig_fit=float(bf['sigmas'])
        print(f"  best-fit: fnl={fnl_fit:.2f}, b1={b1_fit:.4f}, sig={sig_fit:.2f}")

        # P(k) dense + DataBin
        kd=np.geomspace(kf*0.5,KMAX_DISC*1.1,300000)
        tmpl2=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
        th2=PNGTracerPowerSpectrumMultipoles(k=kd,template=tmpl2,mode='b-p')
        th2.init.params['p'].update(fixed=True,value=P_FIX)
        th2.init.params['sn0'].update(fixed=True,value=0)
        th2.init.params['sigmas'].update(fixed=False,value=0)
        th2(**bf);pd=np.array(th2.power[0],dtype=float)

        # DataBin (用当前 nfit 个 bin)
        print(f"  DataBin (nfit={nfit})...")
        def make_pdb(nf,kcn,kmn,kmx,pmn):
            def p(k):
                p_out=np.interp(k,kd,pd)
                for ib in range(nf):
                    mask=(k>=kmn[ib])&(k<kmx[ib]);p_out[mask]=pmn[ib]
                return p_out
            return p
        xi_db=xi_disc(s,gq,kf,V,make_pdb(nfit,kcen,kmin_b,kmax_b,pk_mean))

        r2_db = s**2*xi_db
        xi_curves[f"DataBin_nfit{nfit}"] = r2_db

        # 指标（全范围 + 小尺度 + 大尺度）
        ma_all,ms_all = met(r2d,r2s,r2_db)
        ma_small,ms_small = met_range(r2d,r2s,r2_db,s,50,100)
        ma_large,ms_large = met_range(r2d,r2s,r2_db,s,150,380)

        print(f"  全范围:  mean|Δ/σ|={ma_all:.4f}, mean(Δ/σ)={ms_all:+.4f}")
        print(f"  小尺度:  mean|Δ/σ|={ma_small:.4f}, mean(Δ/σ)={ms_small:+.4f}  (50<r<100)")
        print(f"  大尺度:  mean|Δ/σ|={ma_large:.4f}, mean(Δ/σ)={ms_large:+.4f}  (150<r<380)")

        results.append({
            "nfit":nfit, "kmax_fit":kcen[-1], "fnl":fnl_fit, "b1":b1_fit, "sig":sig_fit,
            "ma_all":ma_all, "ms_all":ms_all,
            "ma_small":ma_small, "ms_small":ms_small,
            "ma_large":ma_large, "ms_large":ms_large,
        })

    # 汇总
    print(f"\n{'='*80}")
    print("汇总 (3Gpc fnl100, EZmock cov, DataBin)")
    print(f"{'='*80}")
    print(f"{'nfit':>5s} {'kmax':>7s} {'fnl':>7s} {'b1':>7s} {'sig':>5s} | {'全|Δ/σ|':>8s} {'小|Δ/σ|':>8s} {'大|Δ/σ|':>8s}")
    print("-"*70)
    for r in results:
        print(f"{r['nfit']:>5d} {r['kmax_fit']:>7.4f} {r['fnl']:>7.2f} {r['b1']:>7.4f} {r['sig']:>5.2f} | "
              f"{r['ma_all']:>8.4f} {r['ma_small']:>8.4f} {r['ma_large']:>8.4f}")

    # 画图
    fig,axes=plt.subplots(2,1,figsize=(14,10))
    fig.suptitle("3Gpc fnl100: DataBin with different fitting k-range (EZmock cov)",fontsize=13)

    ax=axes[0]
    ax.errorbar(s,r2d,yerr=r2s,fmt='ko',ms=3,capsize=2,label='Data',zorder=10)
    colors=plt.cm.viridis(np.linspace(0.2,0.9,len(NFIT_LIST)))
    for i,(nfit,c) in enumerate(zip(NFIT_LIST,colors)):
        r2m=xi_curves[f"DataBin_nfit{nfit}"]
        ma,_=met(r2d,r2s,r2m)
        ax.plot(s,r2m,color=c,lw=1.5,label=f"nfit={nfit} (k<{results[i]['kmax_fit']:.3f}) [{ma:.3f}]")
    ax.set_ylabel(r'$r^2\xi_0$');ax.legend(fontsize=8);ax.grid(alpha=0.3)

    ax=axes[1]
    ax.axhline(0,color='k',lw=0.5)
    ax.axhline(1,color='gray',ls=':',lw=0.5);ax.axhline(-1,color='gray',ls=':',lw=0.5)
    for i,(nfit,c) in enumerate(zip(NFIT_LIST,colors)):
        r2m=xi_curves[f"DataBin_nfit{nfit}"]
        ds=(r2d-r2m)/r2s
        ax.plot(s,ds,color=c,lw=1.5,marker='o',ms=2,label=f"nfit={nfit}")
    ax.axvline(100,color='gray',ls='--',lw=0.8,alpha=0.5)
    ax.set_xlabel(r'$r$ [Mpc/h]');ax.set_ylabel(r'$(Data-Model)/\sigma$')
    ax.set_ylim(-3,3);ax.legend(fontsize=8);ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0,0,1,0.96])
    fp=os.path.join(OUT,"mission9_kfit_scan.png")
    fig.savefig(fp,dpi=150);plt.close()
    print(f"\n图已保存: {fp}\n完成! {time.time()-T0:.0f}s")

if __name__=="__main__":
    main()
