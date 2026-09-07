#!/usr/bin/env python3
"""
Mission 9 支线: fnl=0 + sn0 释放 + kfit 扫描
=============================================
3Gpc fnl=0, EZmock covariance, sn0 自由, 扫描拟合 kmax
"""
from __future__ import annotations
import os,glob,re,time
import numpy as np
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import next_fast_len,rfft,irfft,fht,fhtoffset

OUT=os.path.dirname(os.path.abspath(__file__))
L=3000.0;kf=2*np.pi/L;V=L**3;Z=1.0;P_FIX=1.2;KMAX_DISC=15.0
NFIT_LIST=[20,30,40,50,60]

def rid(f):
    m=re.search(r'N(\d+)',os.path.basename(f));return int(m.group(1)) if m else -1
def load_pk(pat,rmin,rmax,ndp):
    fs=[f for f in sorted(glob.glob(pat),key=rid) if rmin<=rid(f)<=rmax]
    ref=np.loadtxt(fs[0],comments='#')
    v=np.array([np.loadtxt(f,comments='#')[:ndp,5] for f in fs])
    return ref[:ndp,0],ref[:ndp,1],ref[:ndp,2],v,v.mean(0)
def load_pcf(pat,rmin,rmax):
    fs=[f for f in sorted(glob.glob(pat),key=rid) if rmin<=rid(f)<=rmax]
    s=np.loadtxt(fs[0],comments='#')[:,0]
    v=np.array([np.loadtxt(f,comments='#')[:,3] for f in fs])
    return s,v.mean(0),v.std(0,ddof=1)
def load_ez(glob_pat,k_target,ndp):
    files=sorted(glob.glob(glob_pat))
    p0_list=[]
    for fp in files:
        try:
            arr=np.loadtxt(fp,comments='#')
            if arr.shape[0]<ndp:continue
            k_in=arr[:ndp,0];p0_in=arr[:ndp,5]
            if np.allclose(k_in,k_target,rtol=0,atol=1e-10):p0_list.append(p0_in)
            else:p0_list.append(np.interp(k_target,arr[:,0],arr[:,5]))
        except:pass
    print(f"  EZmock: {len(p0_list)} used")
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
    mask=(s>=smin)&(s<=smax);ds=(r2d[mask]-r2m[mask])/r2s[mask]
    return np.nanmean(np.abs(ds)),np.nanmean(ds)

def main():
    T0=time.time()
    print("="*60)
    print("3Gpc fnl=0, sn0 自由, kfit 扫描")
    print("="*60)
    from cosmoprimo import Cosmology
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate,PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler
    from pypower import PowerSpectrumStatistics
    cosmo=Cosmology(h=0.6711,Omega_b=0.049,Omega_cdm=0.3175-0.049,sigma8=0.834,n_s=0.9624,engine='class')

    max_ndp=max(NFIT_LIST)
    kcen_all,kmin_all,kmax_all,pk_mk,pk_mean_all=load_pk(
        "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",2,99,max_ndp)
    s,xi_m,xi_s=load_pcf(
        "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat",2,99)
    r2d=s**2*xi_m;r2s=s**2*xi_s

    ez_glob="/pscratch/sd/l/lzy/cov_mock/B3000G768Z0N4417983_b0.18d5r270c1.65_seed*/PK_EZmock_*_RSD.dat"
    ez_all=load_ez(ez_glob,kcen_all,max_ndp)

    qmax=int((KMAX_DISC/kf)**2);nmax=int(KMAX_DISC/kf)
    print("g_q...")
    gq=gq_fft(qmax,nmax)

    results=[]
    xi_curves={}

    for nfit in NFIT_LIST:
        print(f"\n{'='*50}\n  nfit={nfit} (kmax={kcen_all[nfit-1]:.4f})\n{'='*50}")
        kcen=kcen_all[:nfit];kmin_b=kmin_all[:nfit];kmax_b=kmax_all[:nfit]
        pk_mean=pk_mean_all[:nfit];ez_cov=ez_all[:,:nfit]

        edges=np.concatenate([kmin_b,[kmax_b[-1]]])
        nm=4/3*np.pi*(edges[1:]**3-edges[:-1]**3)
        dps=PowerSpectrumStatistics(edges=edges,modes=kcen,power_nonorm=np.array([pk_mean]),
            nmodes=nm,ells=[0],shotnoise_nonorm=0,statistic='multipole')
        ml=[]
        for i in range(ez_cov.shape[0]):
            t=dps.deepcopy();t.power_nonorm.flat[...]=np.array([ez_cov[i]]).ravel();ml.append(t)

        # 扫描: p=1.1/1.2 × sn0固定/释放
        for p_val, sn0_free, label in [
            (1.2, False, "p1.2_sn0fix"),
            (1.2, True,  "p1.2_sn0free"),
            (1.1, False, "p1.1_sn0fix"),
            (1.1, True,  "p1.1_sn0free"),
        ]:
            tmpl=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
            th=PNGTracerPowerSpectrumMultipoles(template=tmpl,mode='b-p')
            th.init.params['p'].update(fixed=True,value=p_val)
            th.init.params['fnl_loc'].update(fixed=True,value=0)  # fnl 锁定为 0
            th.init.params['sn0'].update(fixed=not sn0_free, value=0)
            th.init.params['sigmas'].update(fixed=False,value=0)
            obs=TracerPowerSpectrumMultipolesObservable(data=dps,covariance=ml,
                klim={0:[float(kcen.min()),float(kcen.max()),float(kcen[1]-kcen[0])]},theory=th)
            like=ObservablesGaussianLikelihood(observables=[obs]);_=like()
            like.all_params['p'].update(fixed=True,value=p_val)
            like.all_params['fnl_loc'].update(fixed=True,value=0)
            like.all_params['sn0'].update(fixed=not sn0_free, value=0)
            like.all_params['sigmas'].update(fixed=False,value=0)
            prof=MinuitProfiler(like,seed=66);pr=prof.maximize(niterations=27)
            bf=pr.bestfit.choice(input=True)
            fnl_f=float(bf['fnl_loc']);b1_f=float(bf['b1']);sig_f=float(bf['sigmas']);sn0_f=float(bf['sn0'])
            print(f"  [{label}] fnl={fnl_f:.2f}, b1={b1_f:.4f}, sig={sig_f:.2f}, sn0={sn0_f:.1f}")

            kd=np.geomspace(kf*0.5,KMAX_DISC*1.1,300000)
            tmpl2=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
            th2=PNGTracerPowerSpectrumMultipoles(k=kd,template=tmpl2,mode='b-p')
            th2.init.params['p'].update(fixed=True,value=p_val)
            th2.init.params['fnl_loc'].update(fixed=True,value=0)
            th2.init.params['sn0'].update(fixed=True,value=0)  # P(k) 不含 sn0
            th2.init.params['sigmas'].update(fixed=False,value=0)
            th2(**bf);pd=np.array(th2.power[0],dtype=float)

            # DataBin
            def make_pdb(nf,kcn,kmn,kmx,pmn):
                def p(k):
                    p_out=np.interp(k,kd,pd)
                    for ib in range(nf):
                        mask=(k>=kmn[ib])&(k<kmx[ib]);p_out[mask]=pmn[ib]
                    return p_out
                return p
            print(f"  DataBin...")
            xi_db=xi_disc(s,gq,kf,V,make_pdb(nfit,kcen,kmin_b,kmax_b,pk_mean))
            r2m=s**2*xi_db
            key=f"nfit{nfit}_{label}"
            xi_curves[key]=r2m

            ma_all,ms_all=met(r2d,r2s,r2m)
            ma_sm,ms_sm=met_range(r2d,r2s,r2m,s,50,100)
            ma_lg,ms_lg=met_range(r2d,r2s,r2m,s,150,380)
            print(f"  全:{ma_all:.4f}, 小:{ma_sm:.4f}, 大:{ma_lg:.4f}")
            results.append({"nfit":nfit,"label":label,"fnl":fnl_f,"b1":b1_f,"sig":sig_f,"sn0":sn0_f,
                           "ma_all":ma_all,"ma_sm":ma_sm,"ma_lg":ma_lg,
                           "ms_all":ms_all,"ms_sm":ms_sm,"ms_lg":ms_lg})

    # 汇总
    print(f"\n{'='*90}")
    print("汇总 (3Gpc fnl=0, EZmock cov)")
    print(f"{'='*90}")
    print(f"{'nfit':>4} {'label':>8} {'fnl':>6} {'b1':>7} {'sig':>5} {'sn0':>6} | {'全':>6} {'小':>6} {'大':>6}")
    for r in results:
        print(f"{r['nfit']:>4} {r['label']:>8} {r['fnl']:>6.1f} {r['b1']:>7.4f} {r['sig']:>5.2f} {r['sn0']:>6.1f} | "
              f"{r['ma_all']:>6.3f} {r['ma_sm']:>6.3f} {r['ma_lg']:>6.3f}")

    # 画图
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,10))
    fig.suptitle("3Gpc fnl=0: DataBin kfit scan (sn0=0 vs sn0free, EZmock cov)",fontsize=13)
    ax1.errorbar(s,r2d,yerr=r2s,fmt='ko',ms=3,capsize=2,label='Data',zorder=10)
    for r in results:
        key=f"nfit{r['nfit']}_{r['label']}"
        ls='-' if r['label']=='sn0=0' else '--'
        ax1.plot(s,xi_curves[key],ls=ls,lw=1.5,
                label=f"nfit={r['nfit']} {r['label']} [{r['ma_all']:.3f}]")
    ax1.set_ylabel(r'$r^2\xi_0$');ax1.legend(fontsize=7,ncol=2);ax1.grid(alpha=0.3)

    ax2.axhline(0,color='k',lw=0.5)
    ax2.axhline(1,color='gray',ls=':',lw=0.5);ax2.axhline(-1,color='gray',ls=':',lw=0.5)
    for r in results:
        key=f"nfit{r['nfit']}_{r['label']}"
        ls='-' if r['label']=='sn0=0' else '--'
        ds=(r2d-xi_curves[key])/r2s
        ax2.plot(s,ds,ls=ls,lw=1.3,marker='.',ms=2,
                label=f"nfit={r['nfit']} {r['label']}")
    ax2.axvline(100,color='gray',ls='--',lw=0.8,alpha=0.5)
    ax2.set_xlabel(r'$r$ [Mpc/h]');ax2.set_ylabel(r'$(Data-Model)/\sigma$')
    ax2.set_ylim(-3,3);ax2.legend(fontsize=7,ncol=2);ax2.grid(alpha=0.3)
    plt.tight_layout(rect=[0,0,1,0.96])
    fp=os.path.join(OUT,"mission9_fnl0_sn0free.png")
    fig.savefig(fp,dpi=150);plt.close()
    print(f"\n图: {fp}\n完成! {time.time()-T0:.0f}s")

if __name__=="__main__":
    main()
