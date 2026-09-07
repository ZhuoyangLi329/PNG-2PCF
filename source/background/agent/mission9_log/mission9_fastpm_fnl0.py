#!/usr/bin/env python3
"""Mission 9: FastPM fnl=0 验证 DataBin"""
from __future__ import annotations
import os,glob,re,time
import numpy as np
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import next_fast_len,rfft,irfft,fht,fhtoffset

OUT=os.path.dirname(os.path.abspath(__file__))

def rid(f):
    m=re.search(r'N(\d+)',os.path.basename(f));return int(m.group(1)) if m else -1
def load_pk(pat,rmin,rmax,ndp):
    fs=[f for f in sorted(glob.glob(pat),key=rid) if rmin<=rid(f)<=rmax]
    ref=np.loadtxt(fs[0],comments='#')
    v=np.array([np.loadtxt(f,comments='#')[:ndp,5] for f in fs])
    valid=v.mean(0)>0
    return ref[:ndp,0][valid],ref[:ndp,1][valid],ref[:ndp,2][valid],v[:,valid],v[:,valid].mean(0)
def load_pcf(pat,rmin,rmax):
    fs=[f for f in sorted(glob.glob(pat),key=rid) if rmin<=rid(f)<=rmax]
    s=np.loadtxt(fs[0],comments='#')[:,0]
    v=np.array([np.loadtxt(f,comments='#')[:,3] for f in fs])
    return s,v.mean(0),v.std(0,ddof=1)
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
    Z=1.0;P_FIX=1.2;KMAX=15.0

    configs=[
        ("3Gpc fnl0",3000.0,
         "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
         "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
         "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat",
         2,99,20,12.0),
        ("1Gpc fnl0",1000.0,
         "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
         "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
         "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut_fnl0/pcf_rsd_N*.dat",
         1,50,20,4.0),
    ]

    fig,axes=plt.subplots(2,2,figsize=(14,10))
    fig.suptitle("FastPM fnl=0: 2PCF modeling comparison",fontsize=14,y=0.98)

    for col,(title,L,pk_p,cov_p,pcf_p,rmin,rmax,ndp,ir_x) in enumerate(configs):
        kf=2*np.pi/L;V=L**3
        print(f"\n{'='*50}\n  {title}\n{'='*50}")
        kcen,kmin_b,kmax_b,pk_mk,pk_mean=load_pk(pk_p,rmin,rmax,ndp+1)
        _,_,_,cov_mk,_=load_pk(cov_p,rmin,rmax,ndp+1)
        s,xi_m,xi_s=load_pcf(pcf_p,rmin,rmax)
        r2d=s**2*xi_m;r2s=s**2*xi_s
        ndp_use=len(kcen)
        print(f"  pk bins={ndp_use}, pcf s=[{s[0]:.0f},{s[-1]:.0f}]")

        # 拟合
        edges=np.concatenate([kmin_b,[kmax_b[-1]]])
        nm=4/3*np.pi*(edges[1:]**3-edges[:-1]**3)
        dps=PowerSpectrumStatistics(edges=edges,modes=kcen,power_nonorm=np.array([pk_mean]),
            nmodes=nm,ells=[0],shotnoise_nonorm=0,statistic='multipole')
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
        print(f"  fit: fnl={fnl_fit:.2f}, b1={b1_fit:.4f}")

        kd=np.geomspace(kf*0.5,KMAX*1.1,300000)
        tmpl2=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
        th2=PNGTracerPowerSpectrumMultipoles(k=kd,template=tmpl2,mode='b-p')
        th2.init.params['p'].update(fixed=True,value=P_FIX)
        th2.init.params['sn0'].update(fixed=True,value=0)
        th2.init.params['sigmas'].update(fixed=False,value=0)
        th2(**bf);pd=np.array(th2.power[0],dtype=float)

        qmax=int((KMAX/kf)**2);nmax=int(KMAX/kf)
        print("  g_q...")
        gq=gq_fft(qmax,nmax)

        print("  FullDiscrete...")
        xi_fd=xi_disc(s,gq,kf,V,lambda k:np.interp(k,kd,pd))
        print("  DataBin...")
        def pdb(k):
            p_out=np.interp(k,kd,pd)
            for ib in range(ndp_use):
                mask=(k>=kmin_b[ib])&(k<kmax_b[ib]);p_out[mask]=pk_mean[ib]
            return p_out
        xi_db=xi_disc(s,gq,kf,V,pdb)
        print("  ExpWindow...")
        kgw=np.geomspace(1e-4/4,KMAX*4,4096)
        tmpl3=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
        th3=PNGTracerPowerSpectrumMultipoles(k=kgw,template=tmpl3,mode='b-p')
        th3.init.params['p'].update(fixed=True,value=P_FIX)
        th3.init.params['sn0'].update(fixed=True,value=0)
        th3.init.params['sigmas'].update(fixed=False,value=0)
        th3(**bf);pgw=np.array(th3.power[0])
        iw=ir_win_func(kgw,kf,ir_x)
        sw,xw=xi_fftlog(kgw,pgw,1e-4,KMAX,ir_win=iw)
        xi_ew=np.interp(s,np.sort(sw),xw[np.argsort(sw)])

        methods={"FullDiscrete":(s**2*xi_fd,'green','-'),
                 "DataBin":(s**2*xi_db,'darkorange','-'),
                 "ExpWindow":(s**2*xi_ew,'red','--')}

        print(f"\n  {'方法':<16s} {'mean|Δ/σ|':>10s} {'mean(Δ/σ)':>10s}")
        for nm,(r2m,_,_) in methods.items():
            ma,ms=met(r2d,r2s,r2m)
            print(f"  {nm:<16s} {ma:>10.4f} {ms:>+10.4f}")

        ax_t=axes[0,col];ax_b=axes[1,col]
        ax_t.errorbar(s,r2d,yerr=r2s,fmt='ko',ms=3,capsize=2,label='Data',zorder=10)
        for nm,(r2m,c,ls) in methods.items():
            ma,ms=met(r2d,r2s,r2m)
            ax_t.plot(s,r2m,color=c,ls=ls,lw=2 if 'DataBin' in nm else 1.5,
                     label=f"{nm} ({ma:.3f}, {ms:+.3f})")
        ax_t.set_title(f"{title} (fnl={fnl_fit:.1f}, b1={b1_fit:.3f})")
        ax_t.set_ylabel(r'$r^2\xi_0$');ax_t.legend(fontsize=8);ax_t.grid(alpha=0.3)

        ax_b.axhline(0,color='k',lw=0.5)
        ax_b.axhline(1,color='gray',ls=':',lw=0.5);ax_b.axhline(-1,color='gray',ls=':',lw=0.5)
        for nm,(r2m,c,ls) in methods.items():
            ds=(r2d-r2m)/r2s
            ax_b.plot(s,ds,color=c,ls=ls,lw=2 if 'DataBin' in nm else 1.3,marker='o',ms=2,label=nm)
        ax_b.set_xlabel(r'$r$ [Mpc/h]');ax_b.set_ylabel(r'$(Data-Model)/\sigma$')
        ax_b.set_ylim(-3,3);ax_b.legend(fontsize=8);ax_b.grid(alpha=0.3)

    plt.tight_layout(rect=[0,0,1,0.96])
    fp=os.path.join(OUT,"mission9_fastpm_fnl0_2pcf.png")
    fig.savefig(fp,dpi=150);plt.close()
    print(f"\n图已保存: {fp}\n完成! {time.time()-T0:.0f}s")

if __name__=="__main__":
    main()
