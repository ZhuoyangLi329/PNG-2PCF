#!/usr/bin/env python3
"""Mission 12: scan rmin for 2PCF fit, rmax=380 fixed, sigmas free"""
import os,glob,re,time
import numpy as np
from scipy.fft import next_fast_len,rfft,irfft
from iminuit import Minuit

BOX=3000.0;KF=2*np.pi/BOX;VOL=BOX**3;Z=1.0;PF=1.2;KMAX=15.0;ND=300000;DKF=0.1
NKFIT=20;RMAX=380.0;R1=2;R2=99

def rid(f):
    m=re.search(r'N(\d+)',os.path.basename(f));return int(m.group(1)) if m else -1
def ldpk(p,a,b,n,c=5):
    fs=[f for f in sorted(glob.glob(p),key=rid) if a<=rid(f)<=b]
    r=np.loadtxt(fs[0],comments='#');return r[:n,0],r[:n,1],r[:n,2],np.array([np.loadtxt(f,comments='#')[:n,c] for f in fs])
def ldxi(p,a,b,c=3):
    fs=[f for f in sorted(glob.glob(p),key=rid) if a<=rid(f)<=b]
    s=np.loadtxt(fs[0],comments='#')[:,0];return s,np.array([np.loadtxt(f,comments='#')[:,c] for f in fps])
def gqfft(qm,nm):
    a=np.zeros(qm+1,dtype=np.float32);a[0]=1;sq=np.arange(1,nm+1,dtype=np.int64)**2;a[sq[sq<=qm]]=2
    n=next_fast_len(3*qm+1);fa=rfft(a,n=n);return np.rint(irfft(fa*fa*fa,n=n)[:qm+1]).astype(np.int64)
def gqenum(qm):
    nm=int(np.ceil(np.sqrt(qm)))+1;g=np.zeros(qm+1,dtype=np.int64)
    for x in range(-nm,nm+1):
        for y in range(-nm,nm+1):
            for z in range(-nm,nm+1):
                if x==y==z==0:continue
                q=x*x+y*y+z*z
                if q<=qm:g[q]+=1
    return g

# 读数据
kc0,km0,kx0,pkm0=ldpk('/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat',R1,R2,NKFIT+1)
_,_,_,cpk0=ldpk('/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat',R1,R2,NKFIT+1)
fv=0
for i in range(len(kc0)):
    if pkm0[:,i].mean()>0:fv=i;break
kc=kc0[fv:fv+NKFIT];kmn=km0[fv:fv+NKFIT];kmx=kx0[fv:fv+NKFIT]
pkm=pkm0[:,fv:fv+NKFIT];pkmean=pkm.mean(0);cpk=cpk0[:,fv:fv+NKFIT]

# 2PCF 读取（手动避免 ldxi 的 bug）
fps_xi=[f for f in sorted(glob.glob('/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat'),key=rid) if R1<=rid(f)<=R2]
sa=np.loadtxt(fps_xi[0],comments='#')[:,0]
xim=np.array([np.loadtxt(f,comments='#')[:,3] for f in fps_xi])
fps_xc=[f for f in sorted(glob.glob('/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat'),key=rid) if R1<=rid(f)<=R2]
xcov=np.array([np.loadtxt(f,comments='#')[:,3] for f in fps_xc])

from cosmoprimo import Cosmology
from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate,PNGTracerPowerSpectrumMultipoles
co=Cosmology(h=0.6711,Omega_b=0.049,Omega_cdm=0.3175-0.049,sigma8=0.834,n_s=0.9624,engine='class')
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.profilers import MinuitProfiler
from pypower import PowerSpectrumStatistics

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

# BinAvgFit P(k)
print('===== P(k) BinAvgFit (sigmas free) =====')
kfm=float(kmx[-1]);qfm=int(np.floor((kfm/KF)**2))+1
gqf=gqenum(qfm);qaf=np.nonzero(gqf[1:])[0]+1;kaf=KF*np.sqrt(qaf.astype(float));gaf=gqf[qaf].astype(float)
ksb=[];gsb=[]
for lo,hi in zip(kmn,kmx):
    m=(kaf>=lo)&(kaf<hi);ksb.append(kaf[m]);gsb.append(gaf[m])
aku=np.unique(np.concatenate(ksb))
tf=FixedPowerSpectrumTemplate(z=Z,fiducial=co)
thf=PNGTracerPowerSpectrumMultipoles(k=aku,template=tf,mode='b-p')
thf.init.params['p'].update(fixed=True,value=PF);thf.init.params['sn0'].update(fixed=True,value=0)
thf.init.params['sigmas'].update(fixed=False,value=float(bfs.get('sigmas',0)))
Cpk=np.cov(cpk,rowvar=False,ddof=1);Cpki=np.linalg.pinv(Cpk,rcond=1e-10)

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
print(f'  fnl    = {mp.values["fnl_loc"]:.3f} +/- {mp.errors["fnl_loc"]:.3f}')
print(f'  b1     = {mp.values["b1"]:.6f} +/- {mp.errors["b1"]:.6f}')
print(f'  sigmas = {mp.values["sigmas"]:.6f} +/- {mp.errors["sigmas"]:.6f}')

# k-rebinning 缓存
qmfd=int((KMAX/KF)**2);nmfd=int(KMAX/KF);gqd=gqfft(qmfd,nmfd)
qn=np.nonzero(gqd[1:])[0]+1;kv=KF*np.sqrt(qn.astype(np.float64));gg=gqd[qn].astype(np.float64)
dk=DKF*KF;nb=int(np.ceil(KMAX/dk))+1;bi=np.clip((kv/dk).astype(np.int64),0,nb-1)
Gb=np.bincount(bi,weights=gg,minlength=nb);Gkb=np.bincount(bi,weights=gg*kv,minlength=nb)
nz=Gb>0;Gn=Gb[nz];ke=Gkb[nz]/Gn
kd=np.geomspace(KF*0.5,KMAX*1.1,ND)
td=FixedPowerSpectrumTemplate(z=Z,fiducial=co)
thd=PNGTracerPowerSpectrumMultipoles(k=kd,template=td,mode='b-p')
thd.init.params['p'].update(fixed=True,value=PF);thd.init.params['sn0'].update(fixed=True,value=0)
thd.init.params['sigmas'].update(fixed=False,value=float(bfs.get('sigmas',0)))

# Scan rmin
rmin_list = [56, 80, 100, 120, 140, 160, 200, 250]
print(f'\n===== Scan rmin, rmax={RMAX} fixed, sigmas free =====')
print(f'{"rmin":>6s} {"npts":>5s} {"fnl":>10s} {"err_fnl":>10s} {"b1":>10s} {"err_b1":>10s} {"sigmas":>10s} {"err_sig":>10s} {"chi2/dof":>10s}')
print('-' * 85)

# P(k) reference line
print(f'{"P(k)":>6s} {NKFIT:>5d} {mp.values["fnl_loc"]:>10.3f} {mp.errors["fnl_loc"]:>10.3f} {mp.values["b1"]:>10.6f} {mp.errors["b1"]:>10.6f} {mp.values["sigmas"]:>10.3f} {mp.errors["sigmas"]:>10.3f} {mp.fmin.fval/(NKFIT-3):>10.3f}')

# 用上一次的 best-fit 作为下一次的初始值
prev_fnl = float(mp.values['fnl_loc'])
prev_b1 = float(mp.values['b1'])
prev_sig = float(mp.values['sigmas'])

for rmin in rmin_list:
    fm = (sa >= rmin) & (sa <= RMAX)
    sf = sa[fm]
    ximf = xim.mean(0)[fm]
    npts = fm.sum()

    r2cm = sf**2 * xcov[:, fm]
    Cxi = np.cov(r2cm, rowvar=False, ddof=1)
    Cxii = np.linalg.pinv(Cxi, rcond=1e-10)
    r2d = sf**2 * ximf

    def c2xi(fnl_loc, b1, sigmas):
        p = dict(bfs)
        p.update(fnl_loc=float(fnl_loc), b1=float(b1), sigmas=float(sigmas), p=PF, sn0=0)
        thd(**p)
        pd = np.array(thd.power[0], dtype=float)
        W = Gn * np.interp(ke, kd, pd)
        arg = np.outer(ke, sa)
        J = np.ones_like(arg); m = arg != 0; J[m] = np.sin(arg[m]) / arg[m]
        xi = (W @ J) / VOL
        r2m = sf**2 * xi[fm]
        d = r2d - r2m
        return float(d @ Cxii @ d)

    mx = Minuit(c2xi, fnl_loc=prev_fnl, b1=prev_b1, sigmas=prev_sig)
    mx.errordef = 1.0
    mx.limits['b1'] = (0, None); mx.limits['sigmas'] = (0, None); mx.limits['fnl_loc'] = (-2000, 2000)
    mx.migrad(); mx.hesse()

    ndf = npts - 3
    chi2dof = mx.fmin.fval / ndf if ndf > 0 else float('nan')
    print(f'{rmin:>6d} {npts:>5d} {mx.values["fnl_loc"]:>10.3f} {mx.errors["fnl_loc"]:>10.3f} {mx.values["b1"]:>10.6f} {mx.errors["b1"]:>10.6f} {mx.values["sigmas"]:>10.3f} {mx.errors["sigmas"]:>10.3f} {chi2dof:>10.3f}')

    prev_fnl = float(mx.values['fnl_loc'])
    prev_b1 = float(mx.values['b1'])
    prev_sig = float(mx.values['sigmas'])
