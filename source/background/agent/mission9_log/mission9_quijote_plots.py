#!/usr/bin/env python3
"""
Mission 9: Quijote 2PCF 建模对比图
====================================
每个 fnl 画: Data r²ξ₀ ± σ vs DataBin(20) vs FullDiscrete vs ExpWindow
上排: r²ξ₀ 曲线, 下排: (Data-Model)/σ 残差
"""

from __future__ import annotations
import os, glob, re, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import next_fast_len, rfft, irfft, fht, fhtoffset

OUT = os.path.dirname(os.path.abspath(__file__))
L=1000.0; kf=2*np.pi/L; V=L**3; IR_X=4.0; Z=1.0; P_FIX=1.2; KMAX=15.0
BASE="/pscratch/sd/l/lzy/pks_2pcfs"
INTERP="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission6_log/nbody_validation/interpolated"

FNL_CFGS = [
    (0,   f"{BASE}/pk_fid_*.txt",       f"{BASE}/pcf_fid_*.dat"),
    (50,  f"{BASE}/pk_LCp50_*.txt",     f"{BASE}/pcf_LCp50_*.dat"),
    (100, f"{BASE}/pk_LCp100_*.txt",    f"{BASE}/pcf_LCp100_*.dat"),
]
COV_PAT = f"{BASE}/pk_fid_*.txt"

def rid(f):
    m=re.search(r'(\d+)\.\w+$',os.path.basename(f)); return int(m.group(1)) if m else -1
def load_pk(pat,n=500):
    fs=sorted(glob.glob(pat),key=rid)[:n]
    ref=np.loadtxt(fs[0],comments='#'); nmod=ref[:,4]; valid=nmod>0
    kcen,kmin_b,kmax_b=ref[valid,0],ref[valid,1],ref[valid,2]
    vals=np.array([np.loadtxt(f,comments='#')[valid,5] for f in fs])
    return kcen,kmin_b,kmax_b,vals,vals.mean(0)
def load_pcf(pat,n=500):
    fs=sorted(glob.glob(pat),key=rid)[:n]
    s=np.loadtxt(fs[0],comments='#')[:,0]
    vals=np.array([np.loadtxt(f,comments='#')[:,3] for f in fs])
    return s,vals.mean(0),vals.std(0,ddof=1)
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
def xi_fftlog(kg,p0,kmin,kmax,ir_win=None):
    lk=np.log(kg);l0,l1=np.log(kmin),np.log(kmax);dl=0.06*(l1-l0)
    w=np.zeros_like(kg);left,right=l0+dl,l1-dl
    m=(lk>=l0)&(lk<left);w[m]=0.5*(1-np.cos(np.pi*(lk[m]-l0)/dl))
    m=(lk>=left)&(lk<=right);w[m]=1.0
    m=(lk>right)&(lk<=l1);w[m]=0.5*(1+np.cos(np.pi*(lk[m]-right)/dl))
    if ir_win is not None:w*=ir_win
    dln=np.log(kg[1]/kg[0]);off=fhtoffset(dln,mu=0.5,initial=0,bias=0)
    A=fht((kg**1.5)*p0*w,dln=dln,mu=0.5,offset=off,bias=0)
    n=kg.size;jj=np.arange(n)
    sg=np.exp(-(np.log(kg[0])+np.log(kg[-1]))/2+off+(jj-(n-1)/2)*dln)
    return sg,np.sqrt(np.pi/2)/(2*np.pi**2)*A/sg**1.5
def ir_win_func(k,kf,x):
    norm=1-np.exp(-1);w=np.ones_like(k);m=k<kf;w[m]=(1-np.exp(-(k[m]/kf)**x))/norm;return w
def met(r2d,r2s,r2m):
    ds=(r2d-r2m)/r2s;return np.nanmean(np.abs(ds)),np.nanmean(ds)

def main():
    T0=time.time()
    from cosmoprimo import Cosmology
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate,PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler
    from pypower import PowerSpectrumStatistics
    cosmo=Cosmology(h=0.6711,Omega_b=0.049,Omega_cdm=0.3175-0.049,sigma8=0.834,n_s=0.9624,engine='class')

    qmax=int((KMAX/kf)**2);nmax=int(KMAX/kf)
    print("g_q...")
    gq=gq_fft(qmax,nmax)
    _,_,_,cov_mocks,_=load_pk(COV_PAT)

    n_fnl=len(FNL_CFGS)
    fig,axes=plt.subplots(2,n_fnl,figsize=(6*n_fnl,10))
    if n_fnl==1: axes=axes.reshape(2,1)
    fig.suptitle("Quijote N-body: 2PCF modeling comparison (L=1Gpc)",fontsize=14,y=0.98)

    for col,(fnl_true,pk_pat,pcf_pat) in enumerate(FNL_CFGS):
        print(f"\n--- fnl={fnl_true} ---")
        kcen,kmin_b,kmax_b,pk_mk,pk_mean=load_pk(pk_pat)
        s,xi_m,xi_s=load_pcf(pcf_pat)
        r2d=s**2*xi_m;r2s=s**2*xi_s
        ndp=len(kcen);FIT_NDP=min(20,ndp)

        # 拟合
        fit_kcen=kcen[:FIT_NDP];fit_kmin=kmin_b[:FIT_NDP];fit_kmax=kmax_b[:FIT_NDP]
        fit_pk=pk_mean[:FIT_NDP];fit_cov=cov_mocks[:,:FIT_NDP]
        edges=np.concatenate([fit_kmin,[fit_kmax[-1]]])
        nm=4/3*np.pi*(edges[1:]**3-edges[:-1]**3)
        dps=PowerSpectrumStatistics(edges=edges,modes=fit_kcen,
            power_nonorm=np.array([fit_pk]),nmodes=nm,ells=[0],shotnoise_nonorm=0,statistic='multipole')
        ml=[]
        for i in range(min(fit_cov.shape[0],500)):
            t=dps.deepcopy();t.power_nonorm.flat[...]=np.array([fit_cov[i]]).ravel();ml.append(t)
        tmpl=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
        th=PNGTracerPowerSpectrumMultipoles(template=tmpl,mode='b-p')
        th.init.params['p'].update(fixed=True,value=P_FIX)
        th.init.params['sn0'].update(fixed=True,value=0)
        th.init.params['sigmas'].update(fixed=False,value=0)
        obs=TracerPowerSpectrumMultipolesObservable(data=dps,covariance=ml,
            klim={0:[float(fit_kcen.min()),float(fit_kcen.max()),float(fit_kcen[1]-fit_kcen[0])]},theory=th)
        like=ObservablesGaussianLikelihood(observables=[obs]);_=like()
        like.all_params['p'].update(fixed=True,value=P_FIX)
        like.all_params['sn0'].update(fixed=True,value=0)
        like.all_params['sigmas'].update(fixed=False,value=0)
        prof=MinuitProfiler(like,seed=66);pr=prof.maximize(niterations=27)
        bf=pr.bestfit.choice(input=True)
        fnl_fit=float(bf['fnl_loc']);b1_fit=float(bf['b1'])
        print(f"  fit: fnl={fnl_fit:.1f}, b1={b1_fit:.3f}")

        # P(k) dense
        kd=np.geomspace(kf*0.5,KMAX*1.1,300000)
        tmpl2=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
        th2=PNGTracerPowerSpectrumMultipoles(k=kd,template=tmpl2,mode='b-p')
        th2.init.params['p'].update(fixed=True,value=P_FIX)
        th2.init.params['sn0'].update(fixed=True,value=0)
        th2.init.params['sigmas'].update(fixed=False,value=0)
        th2(**bf);pd=np.array(th2.power[0],dtype=float)

        # FullDiscrete
        print("  FullDiscrete...")
        xi_fd=xi_disc(s,gq,kf,V,lambda k:np.interp(k,kd,pd))

        # DataBin(20)
        print("  DataBin(20)...")
        def pdb(k):
            p_out=np.interp(k,kd,pd)
            for ib in range(FIT_NDP):
                mask=(k>=kmin_b[ib])&(k<kmax_b[ib]);p_out[mask]=pk_mean[ib]
            return p_out
        xi_db=xi_disc(s,gq,kf,V,pdb)

        # ExpWindow
        print("  ExpWindow...")
        kgw=np.geomspace(1e-4/4,KMAX*4,4096)
        tmpl3=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
        th3=PNGTracerPowerSpectrumMultipoles(k=kgw,template=tmpl3,mode='b-p')
        th3.init.params['p'].update(fixed=True,value=P_FIX)
        th3.init.params['sn0'].update(fixed=True,value=0)
        th3.init.params['sigmas'].update(fixed=False,value=0)
        th3(**bf);pgw=np.array(th3.power[0])
        iw=ir_win_func(kgw,kf,IR_X)
        sw,xw=xi_fftlog(kgw,pgw,1e-4,KMAX,ir_win=iw)
        xi_ew=np.interp(s,np.sort(sw),xw[np.argsort(sw)])

        # 画图
        methods={
            "FullDiscrete":(s**2*xi_fd,'green','-'),
            "DataBin(20)":(s**2*xi_db,'darkorange','-'),
            "ExpWindow":(s**2*xi_ew,'red','--'),
        }

        ax_top=axes[0,col]; ax_bot=axes[1,col]

        # 上图: r²ξ₀
        ax_top.errorbar(s,r2d,yerr=r2s,fmt='ko',ms=3,capsize=2,label='Data mean',zorder=10)
        for nm,(r2m,c,ls) in methods.items():
            ma,ms=met(r2d,r2s,r2m)
            ax_top.plot(s,r2m,color=c,ls=ls,lw=2 if 'DataBin' in nm else 1.5,
                       label=f"{nm} ({ma:.3f})")
        ax_top.set_title(f"fnl={fnl_true} (fit={fnl_fit:.1f}, b1={b1_fit:.3f})",fontsize=11)
        ax_top.set_ylabel(r'$r^2\xi_0(r)$')
        ax_top.legend(fontsize=8,title="method (mean|Δ/σ|)")
        ax_top.grid(alpha=0.3)

        # 下图: 残差
        ax_bot.axhline(0,color='k',lw=0.5)
        ax_bot.axhline(1,color='gray',ls=':',lw=0.5)
        ax_bot.axhline(-1,color='gray',ls=':',lw=0.5)
        for nm,(r2m,c,ls) in methods.items():
            ds=(r2d-r2m)/r2s
            ax_bot.plot(s,ds,color=c,ls=ls,lw=2 if 'DataBin' in nm else 1.3,
                       marker='o',ms=2,label=nm)
        ax_bot.set_xlabel(r'$r\,[\mathrm{Mpc}/h]$')
        ax_bot.set_ylabel(r'$(Data-Model)/\sigma$')
        ax_bot.set_ylim(-2.5,2.5)
        ax_bot.legend(fontsize=8)
        ax_bot.grid(alpha=0.3)

    plt.tight_layout(rect=[0,0,1,0.96])
    figpath=os.path.join(OUT,"mission9_quijote_2pcf.png")
    fig.savefig(figpath,dpi=150);plt.close()
    print(f"\n图已保存: {figpath}")

    # 也画 fastPM 的
    print("\n--- FastPM 2PCF 图 ---")
    fastpm_cfgs=[
        ("3Gpc fnl100",3000.0,
         "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat",
         "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
         "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",
         2,99,20,12.0),
        ("1Gpc fnl100",1000.0,
         "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat",
         "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
         "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat",
         1,50,20,4.0),
    ]

    fig2,axes2=plt.subplots(2,2,figsize=(14,10))
    fig2.suptitle("FastPM: 2PCF modeling comparison",fontsize=14,y=0.98)

    for col,(title,Lf,pk_p,cov_p,pcf_p,rmin,rmax,ndpf,ir_xf) in enumerate(fastpm_cfgs):
        kff=2*np.pi/Lf;Vf=Lf**3
        print(f"\n  {title}...")
        # 加载（用通用 load 函数，跳过空 bin）
        # fastPM 格式与 quijote 不同，用专用读取函数
        def load_pk_fm(pat,rmin,rmax,ndp):
            def rid_fm(f):
                m=re.search(r'N(\d+)',os.path.basename(f));return int(m.group(1)) if m else -1
            fs=[f for f in sorted(glob.glob(pat),key=rid_fm) if rmin<=rid_fm(f)<=rmax]
            ref=np.loadtxt(fs[0],comments='#')
            kcen,kmin_b,kmax_b=ref[:ndp,0],ref[:ndp,1],ref[:ndp,2]
            vals=np.array([np.loadtxt(f,comments='#')[:ndp,5] for f in fs])
            # 跳过 P=0 的 bin
            valid=vals.mean(0)>0
            return kcen[valid],kmin_b[valid],kmax_b[valid],vals[:,valid],vals[:,valid].mean(0)
        def load_pcf_fm(pat,rmin,rmax):
            def rid_fm(f):
                m=re.search(r'N(\d+)',os.path.basename(f));return int(m.group(1)) if m else -1
            fs=[f for f in sorted(glob.glob(pat),key=rid_fm) if rmin<=rid_fm(f)<=rmax]
            s=np.loadtxt(fs[0],comments='#')[:,0]
            vals=np.array([np.loadtxt(f,comments='#')[:,3] for f in fs])
            return s,vals.mean(0),vals.std(0,ddof=1)

        kcen,kmin_b,kmax_b,pk_mk,pk_mean=load_pk_fm(pk_p,rmin,rmax,ndpf+1)
        _,_,_,cov_mk,_=load_pk_fm(cov_p,rmin,rmax,ndpf+1)
        sf,xi_mf,xi_sf=load_pcf_fm(pcf_p,rmin,rmax)
        r2df=sf**2*xi_mf;r2sf=sf**2*xi_sf
        ndp_use=len(kcen)

        # 拟合
        edges=np.concatenate([kmin_b,[kmax_b[-1]]])
        nm=4/3*np.pi*(edges[1:]**3-edges[:-1]**3)
        dps=PowerSpectrumStatistics(edges=edges,modes=kcen,
            power_nonorm=np.array([pk_mean]),nmodes=nm,ells=[0],shotnoise_nonorm=0,statistic='multipole')
        ml=[]
        for i in range(cov_mk.shape[0]):
            t=dps.deepcopy();t.power_nonorm.flat[...]=np.array([cov_mk[i]]).ravel();ml.append(t)
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
        fnl_fit=float(bf['fnl_loc']);b1_fit=float(bf['b1'])
        print(f"  fit: fnl={fnl_fit:.1f}, b1={b1_fit:.3f}")

        kd=np.geomspace(kff*0.5,KMAX*1.1,300000)
        tmpl2=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
        th2=PNGTracerPowerSpectrumMultipoles(k=kd,template=tmpl2,mode='b-p')
        th2.init.params['p'].update(fixed=True,value=P_FIX)
        th2.init.params['sn0'].update(fixed=True,value=0)
        th2.init.params['sigmas'].update(fixed=False,value=0)
        th2(**bf);pd=np.array(th2.power[0],dtype=float)

        qmax_f=int((KMAX/kff)**2);nmax_f=int(KMAX/kff)
        gq_f=gq_fft(qmax_f,nmax_f)

        print("  FullDiscrete...")
        xi_fd=xi_disc(sf,gq_f,kff,Vf,lambda k:np.interp(k,kd,pd))
        print("  DataBin...")
        def pdb_fm(k):
            p_out=np.interp(k,kd,pd)
            for ib in range(ndp_use):
                mask=(k>=kmin_b[ib])&(k<kmax_b[ib]);p_out[mask]=pk_mean[ib]
            return p_out
        xi_db=xi_disc(sf,gq_f,kff,Vf,pdb_fm)
        print("  ExpWindow...")
        kgw=np.geomspace(1e-4/4,KMAX*4,4096)
        tmpl3=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
        th3=PNGTracerPowerSpectrumMultipoles(k=kgw,template=tmpl3,mode='b-p')
        th3.init.params['p'].update(fixed=True,value=P_FIX)
        th3.init.params['sn0'].update(fixed=True,value=0)
        th3.init.params['sigmas'].update(fixed=False,value=0)
        th3(**bf);pgw=np.array(th3.power[0])
        iw=ir_win_func(kgw,kff,ir_xf)
        sw,xw=xi_fftlog(kgw,pgw,1e-4,KMAX,ir_win=iw)
        xi_ew=np.interp(sf,np.sort(sw),xw[np.argsort(sw)])

        methods={
            "FullDiscrete":(sf**2*xi_fd,'green','-'),
            "DataBin":(sf**2*xi_db,'darkorange','-'),
            "ExpWindow":(sf**2*xi_ew,'red','--'),
        }
        ax_t=axes2[0,col];ax_b=axes2[1,col]
        ax_t.errorbar(sf,r2df,yerr=r2sf,fmt='ko',ms=3,capsize=2,label='Data',zorder=10)
        for nm,(r2m,c,ls) in methods.items():
            ma,ms=met(r2df,r2sf,r2m)
            ax_t.plot(sf,r2m,color=c,ls=ls,lw=2 if 'DataBin' in nm else 1.5,label=f"{nm} ({ma:.3f})")
        ax_t.set_title(f"{title} (fnl={fnl_fit:.1f}, b1={b1_fit:.3f})",fontsize=11)
        ax_t.set_ylabel(r'$r^2\xi_0$');ax_t.legend(fontsize=8);ax_t.grid(alpha=0.3)

        ax_b.axhline(0,color='k',lw=0.5)
        ax_b.axhline(1,color='gray',ls=':',lw=0.5);ax_b.axhline(-1,color='gray',ls=':',lw=0.5)
        for nm,(r2m,c,ls) in methods.items():
            ds=(r2df-r2m)/r2sf
            ax_b.plot(sf,ds,color=c,ls=ls,lw=2 if 'DataBin' in nm else 1.3,marker='o',ms=2,label=nm)
        ax_b.set_xlabel(r'$r$ [Mpc/h]');ax_b.set_ylabel(r'$(Data-Model)/\sigma$')
        ax_b.set_ylim(-3,3);ax_b.legend(fontsize=8);ax_b.grid(alpha=0.3)

    plt.tight_layout(rect=[0,0,1,0.96])
    figpath2=os.path.join(OUT,"mission9_fastpm_2pcf.png")
    fig2.savefig(figpath2,dpi=150);plt.close()
    print(f"图已保存: {figpath2}")
    print(f"总耗时: {time.time()-T0:.0f}s")

if __name__=="__main__":
    main()
