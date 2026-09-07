#!/usr/bin/env python3
"""Mission 12: scan p_fix values, compare P(k) vs 2PCF fit (rmin=56, rmax=380, sigmas free)"""
import os,glob,re,time
import numpy as np
from scipy.fft import next_fast_len,rfft,irfft
from iminuit import Minuit

BOX=3000.0;KF=2*np.pi/BOX;VOL=BOX**3;Z=1.0;KMAX=15.0;ND=300000;DKF=0.1
NKFIT=20;RMIN=56.0;RMAX=380.0;R1=2;R2=99

def rid(f):
    m=re.search(r'N(\d+)',os.path.basename(f));return int(m.group(1)) if m else -1

# 读数据（一次性）
def load_all():
    fps_pk=[f for f in sorted(glob.glob('/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat'),key=rid) if R1<=rid(f)<=R2]
    ref=np.loadtxt(fps_pk[0],comments='#')
    n=NKFIT+1
    kc0=ref[:n,0];km0=ref[:n,1];kx0=ref[:n,2]
    pkm0=np.array([np.loadtxt(f,comments='#')[:n,5] for f in fps_pk])
    fps_c=[f for f in sorted(glob.glob('/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat'),key=rid) if R1<=rid(f)<=R2]
    cpk0=np.array([np.loadtxt(f,comments='#')[:n,5] for f in fps_c])
    fv=0
    for i in range(len(kc0)):
        if pkm0[:,i].mean()>0:fv=i;break
    kc=kc0[fv:fv+NKFIT];kmn=km0[fv:fv+NKFIT];kmx=kx0[fv:fv+NKFIT]
    pkm=pkm0[:,fv:fv+NKFIT];cpk=cpk0[:,fv:fv+NKFIT]
    fps_xi=[f for f in sorted(glob.glob('/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat'),key=rid) if R1<=rid(f)<=R2]
    sa=np.loadtxt(fps_xi[0],comments='#')[:,0]
    xim=np.array([np.loadtxt(f,comments='#')[:,3] for f in fps_xi])
    fps_xc=[f for f in sorted(glob.glob('/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat'),key=rid) if R1<=rid(f)<=R2]
    xcov=np.array([np.loadtxt(f,comments='#')[:,3] for f in fps_xc])
    return kc,kmn,kmx,pkm,cpk,sa,xim,xcov

kc,kmn,kmx,pkm,cpk,sa,xim,xcov=load_all()
pkmean=pkm.mean(0)
fm=(sa>=RMIN)&(sa<=RMAX);sf=sa[fm];ximf=xim.mean(0)[fm]

from cosmoprimo import Cosmology
from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate,PNGTracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.profilers import MinuitProfiler
from pypower import PowerSpectrumStatistics
co=Cosmology(h=0.6711,Omega_b=0.049,Omega_cdm=0.3175-0.049,sigma8=0.834,n_s=0.9624,engine='class')

# BinAvgFit 壳层结构（只需算一次）
def gqenum(qm):
    nm=int(np.ceil(np.sqrt(qm)))+1;g=np.zeros(qm+1,dtype=np.int64)
    for x in range(-nm,nm+1):
        for y in range(-nm,nm+1):
            for z in range(-nm,nm+1):
                if x==y==z==0:continue
                q=x*x+y*y+z*z
                if q<=qm:g[q]+=1
    return g

kfm=float(kmx[-1]);qfm=int(np.floor((kfm/KF)**2))+1
gqf=gqenum(qfm);qaf=np.nonzero(gqf[1:])[0]+1;kaf=KF*np.sqrt(qaf.astype(float));gaf=gqf[qaf].astype(float)
ksb=[];gsb=[]
for lo,hi in zip(kmn,kmx):
    m=(kaf>=lo)&(kaf<hi);ksb.append(kaf[m]);gsb.append(gaf[m])
aku=np.unique(np.concatenate(ksb))
Cpk=np.cov(cpk,rowvar=False,ddof=1);Cpki=np.linalg.pinv(Cpk,rcond=1e-10)

# k-rebinning 缓存（一次性）
def gqfft(qm,nm):
    a=np.zeros(qm+1,dtype=np.float32);a[0]=1;sq=np.arange(1,nm+1,dtype=np.int64)**2;a[sq[sq<=qm]]=2
    n=next_fast_len(3*qm+1);fa=rfft(a,n=n);return np.rint(irfft(fa*fa*fa,n=n)[:qm+1]).astype(np.int64)

qmfd=int((KMAX/KF)**2);nmfd=int(KMAX/KF);gqd=gqfft(qmfd,nmfd)
qn=np.nonzero(gqd[1:])[0]+1;kv=KF*np.sqrt(qn.astype(np.float64));gg=gqd[qn].astype(np.float64)
dk=DKF*KF;nb=int(np.ceil(KMAX/dk))+1;bi=np.clip((kv/dk).astype(np.int64),0,nb-1)
Gb=np.bincount(bi,weights=gg,minlength=nb);Gkb=np.bincount(bi,weights=gg*kv,minlength=nb)
nz=Gb>0;Gn=Gb[nz];ke=Gkb[nz]/Gn
kd=np.geomspace(KF*0.5,KMAX*1.1,ND)

r2cm=sf**2*xcov[:,fm];Cxi=np.cov(r2cm,rowvar=False,ddof=1);Cxii=np.linalg.pinv(Cxi,rcond=1e-10)
r2d=sf**2*ximf

# Scan p values
p_list = [1.0, 1.2, 1.4, 1.5, 1.6]

print(f'{"p":>5s} | {"method":>8s} | {"fnl":>10s} {"err":>8s} | {"b1":>10s} {"err":>8s} | {"sigmas":>8s} {"err":>8s} | {"chi2/dof":>10s}')
print('-' * 95)

for PF in p_list:
    # 标准 desilike 初始值
    ed=np.concatenate([kmn,[kmx[-1]]]);nm_=4./3.*np.pi*(ed[1:]**3-ed[:-1]**3)
    dps=PowerSpectrumStatistics(edges=ed,modes=kc,power_nonorm=np.array([pkmean]),nmodes=nm_,ells=[0],shotnoise_nonorm=0.0,statistic='multipole')
    ml=[]
    for i in range(cpk.shape[0]):
        t=dps.deepcopy();t.power_nonorm.flat[...]=np.array([cpk[i]]).ravel();ml.append(t)
    tp=FixedPowerSpectrumTemplate(z=Z,fiducial=co)
    th=PNGTracerPowerSpectrumMultipoles(template=tp,mode='b-p')
    th.init.params['p'].update(fixed=True,value=PF);th.init.params['sn0'].update(fixed=True,value=0)
    th.init.params['sigmas'].update(fixed=False,value=0)
    ob=TracerPowerSpectrumMultipolesObservable(data=dps,covariance=ml,klim={0:[float(kc.min()),float(kc.max()),float(kc[1]-kc[0])]},theory=th)
    lk=ObservablesGaussianLikelihood(observables=[ob]);_=lk()
    lk.all_params['p'].update(fixed=True,value=PF);lk.all_params['sn0'].update(fixed=True,value=0)
    lk.all_params['sigmas'].update(fixed=False,value=0)
    pr=MinuitProfiler(lk,seed=66);ps=pr.maximize(niterations=27)
    bfs=ps.bestfit.choice(input=True)

    # P(k) BinAvgFit
    tf=FixedPowerSpectrumTemplate(z=Z,fiducial=co)
    thf=PNGTracerPowerSpectrumMultipoles(k=aku,template=tf,mode='b-p')
    thf.init.params['p'].update(fixed=True,value=PF);thf.init.params['sn0'].update(fixed=True,value=0)
    thf.init.params['sigmas'].update(fixed=False,value=float(bfs.get('sigmas',0)))

    def c2pk(fnl_loc,b1,sigmas):
        p=dict(bfs);p.update(fnl_loc=float(fnl_loc),b1=float(b1),sigmas=float(sigmas),p=PF,sn0=0)
        thf(**p);pt=np.array(thf.power[0],dtype=float);pm=dict(zip(aku,pt))
        pb=np.zeros(len(ksb))
        for i,(ks,gs) in enumerate(zip(ksb,gsb)):
            if len(ks)==0:continue
            pv=np.array([pm[k] for k in ks]);pb[i]=np.sum(gs*pv)/np.sum(gs)
        d=pkmean-pb;return float(d@Cpki@d)

    mp=Minuit(c2pk,fnl_loc=float(bfs['fnl_loc']),b1=float(bfs['b1']),sigmas=float(bfs['sigmas']))
    mp.errordef=1.0;mp.limits['b1']=(0,None);mp.limits['sigmas']=(0,None);mp.limits['fnl_loc']=(-2000,2000)
    mp.migrad();mp.hesse()
    print(f'{PF:>5.1f} | {"P(k)":>8s} | {mp.values["fnl_loc"]:>10.3f} {mp.errors["fnl_loc"]:>8.3f} | {mp.values["b1"]:>10.6f} {mp.errors["b1"]:>8.6f} | {mp.values["sigmas"]:>8.3f} {mp.errors["sigmas"]:>8.3f} | {mp.fmin.fval/(NKFIT-3):>10.3f}')

    # 2PCF fit
    td2=FixedPowerSpectrumTemplate(z=Z,fiducial=co)
    thd2=PNGTracerPowerSpectrumMultipoles(k=kd,template=td2,mode='b-p')
    thd2.init.params['p'].update(fixed=True,value=PF);thd2.init.params['sn0'].update(fixed=True,value=0)
    thd2.init.params['sigmas'].update(fixed=False,value=float(bfs.get('sigmas',0)))

    def c2xi(fnl_loc,b1,sigmas):
        p=dict(bfs);p.update(fnl_loc=float(fnl_loc),b1=float(b1),sigmas=float(sigmas),p=PF,sn0=0)
        thd2(**p);pd=np.array(thd2.power[0],dtype=float)
        W=Gn*np.interp(ke,kd,pd);arg=np.outer(ke,sa);J=np.ones_like(arg);m=arg!=0;J[m]=np.sin(arg[m])/arg[m]
        xi=(W@J)/VOL;r2m=sf**2*xi[fm];d=r2d-r2m;return float(d@Cxii@d)

    mx=Minuit(c2xi,fnl_loc=float(mp.values['fnl_loc']),b1=float(mp.values['b1']),sigmas=float(mp.values['sigmas']))
    mx.errordef=1.0;mx.limits['b1']=(0,None);mx.limits['sigmas']=(0,None);mx.limits['fnl_loc']=(-2000,2000)
    mx.migrad();mx.hesse()
    ndf=fm.sum()-3
    print(f'{PF:>5.1f} | {"2PCF":>8s} | {mx.values["fnl_loc"]:>10.3f} {mx.errors["fnl_loc"]:>8.3f} | {mx.values["b1"]:>10.6f} {mx.errors["b1"]:>8.6f} | {mx.values["sigmas"]:>8.3f} {mx.errors["sigmas"]:>8.3f} | {mx.fmin.fval/ndf:>10.3f}')
    print('-' * 95)
