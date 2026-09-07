#!/usr/bin/env python3
"""Mission 12: P(k) vs 2PCF 拟合对比 (sigmas 固定)"""
import os, sys, glob, re, time
import numpy as np
from scipy.fft import next_fast_len, rfft, irfft
from iminuit import Minuit

BOX_SIZE=3000.0; K_FUND=2*np.pi/BOX_SIZE; VOL=BOX_SIZE**3
Z=1.0; P_FIX=1.2; SN0_FIX=0.0; KMAX=15.0; N_DENSE=300_000; DK_FACTOR=0.1
N_KBINS_FIT=20; R_FIT_MIN=100.0; R_FIT_MAX=350.0; RID_MIN=2; RID_MAX=99

def _rid(fp):
    m=re.search(r'N(\d+)',os.path.basename(fp))
    return int(m.group(1)) if m else -1
def load_pk(pat,rmin,rmax,ndp,col=5):
    fps=[f for f in sorted(glob.glob(pat),key=_rid) if rmin<=_rid(f)<=rmax]
    ref=np.loadtxt(fps[0],comments='#')
    return ref[:ndp,0],ref[:ndp,1],ref[:ndp,2],np.array([np.loadtxt(f,comments='#')[:ndp,col] for f in fps])
def load_pcf(pat,rmin,rmax,col=3):
    fps=[f for f in sorted(glob.glob(pat),key=_rid) if rmin<=_rid(f)<=rmax]
    s=np.loadtxt(fps[0],comments='#')[:,0]
    return s,np.array([np.loadtxt(f,comments='#')[:,col] for f in fps])
def gq_fft(qmax,nmax):
    a=np.zeros(qmax+1,dtype=np.float32);a[0]=1.0
    sq=np.arange(1,nmax+1,dtype=np.int64)**2;a[sq[sq<=qmax]]=2.0
    nfft=next_fast_len(3*qmax+1);fa=rfft(a,n=nfft)
    return np.rint(irfft(fa*fa*fa,n=nfft)[:qmax+1]).astype(np.int64)
def gq_enumerate(qmax):
    nmax=int(np.ceil(np.sqrt(qmax)))+1
    gq=np.zeros(qmax+1,dtype=np.int64)
    for nx in range(-nmax,nmax+1):
        for ny in range(-nmax,nmax+1):
            for nz in range(-nmax,nmax+1):
                if nx==ny==nz==0:continue
                q=nx*nx+ny*ny+nz*nz
                if q<=qmax:gq[q]+=1
    return gq

# 读数据
kc0,km0,kx0,pk_mocks0=load_pk('/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat',RID_MIN,RID_MAX,N_KBINS_FIT+1)
_,_,_,cov_pk0=load_pk('/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat',RID_MIN,RID_MAX,N_KBINS_FIT+1)
fv=0
for ib in range(len(kc0)):
    if pk_mocks0[:,ib].mean()>0: fv=ib;break
kcen=kc0[fv:fv+N_KBINS_FIT];kmin_b=km0[fv:fv+N_KBINS_FIT];kmax_b=kx0[fv:fv+N_KBINS_FIT]
pk_mocks=pk_mocks0[:,fv:fv+N_KBINS_FIT];pk_mean=pk_mocks.mean(0)
cov_pk=cov_pk0[:,fv:fv+N_KBINS_FIT]
s_all,pcf_mocks=load_pcf('/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat',RID_MIN,RID_MAX)
_,pcf_cov=load_pcf('/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat',RID_MIN,RID_MAX)
fit_mask=(s_all>=R_FIT_MIN)&(s_all<=R_FIT_MAX)
s_fit=s_all[fit_mask]
xi_mean_fit=pcf_mocks.mean(0)[fit_mask]

from cosmoprimo import Cosmology
from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate,PNGTracerPowerSpectrumMultipoles
cosmo=Cosmology(h=0.6711,Omega_b=0.049,Omega_cdm=0.3175-0.049,sigma8=0.834,n_s=0.9624,engine='class')

# 标准desilike获取初始值(sigmas也fix=0)
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.profilers import MinuitProfiler
from pypower import PowerSpectrumStatistics
edges=np.concatenate([kmin_b,[kmax_b[-1]]])
nm=4./3.*np.pi*(edges[1:]**3-edges[:-1]**3)
data_ps=PowerSpectrumStatistics(edges=edges,modes=kcen,power_nonorm=np.array([pk_mean]),nmodes=nm,ells=[0],shotnoise_nonorm=0.0,statistic='multipole')
ml=[]
for i in range(cov_pk.shape[0]):
    t=data_ps.deepcopy();t.power_nonorm.flat[...]=np.array([cov_pk[i]]).ravel();ml.append(t)
tmpl=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
th=PNGTracerPowerSpectrumMultipoles(template=tmpl,mode='b-p')
th.init.params['p'].update(fixed=True,value=P_FIX)
th.init.params['sn0'].update(fixed=True,value=SN0_FIX)
th.init.params['sigmas'].update(fixed=True,value=0.0)
obs=TracerPowerSpectrumMultipolesObservable(data=data_ps,covariance=ml,klim={0:[float(kcen.min()),float(kcen.max()),float(kcen[1]-kcen[0])]},theory=th)
like=ObservablesGaussianLikelihood(observables=[obs]);_=like()
like.all_params['p'].update(fixed=True,value=P_FIX)
like.all_params['sn0'].update(fixed=True,value=SN0_FIX)
like.all_params['sigmas'].update(fixed=True,value=0.0)
prof=MinuitProfiler(like,seed=66);profiles=prof.maximize(niterations=27)
bf_std=profiles.bestfit.choice(input=True)
SIGMAS_FIX=0.0
print(f'初始值: fnl={float(bf_std["fnl_loc"]):.3f}, b1={float(bf_std["b1"]):.6f}, sigmas={SIGMAS_FIX} (fixed)')

# ===== 方法A: BinAvgFit P(k) =====
print('\n===== 方法A: BinAvgFit P(k) (sigmas=0 fixed) =====')
kmax_fit=float(kmax_b[-1]);qmax_fit=int(np.floor((kmax_fit/K_FUND)**2))+1
gq_fit_arr=gq_enumerate(qmax_fit)
q_af=np.nonzero(gq_fit_arr[1:])[0]+1;k_af=K_FUND*np.sqrt(q_af.astype(float));g_af=gq_fit_arr[q_af].astype(float)
ks_bin=[];gs_bin=[]
for lo,hi in zip(kmin_b,kmax_b):
    m=(k_af>=lo)&(k_af<hi);ks_bin.append(k_af[m]);gs_bin.append(g_af[m])
all_ku=np.unique(np.concatenate(ks_bin))
tmpl_f=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
th_f=PNGTracerPowerSpectrumMultipoles(k=all_ku,template=tmpl_f,mode='b-p')
th_f.init.params['p'].update(fixed=True,value=P_FIX)
th_f.init.params['sn0'].update(fixed=True,value=SN0_FIX)
th_f.init.params['sigmas'].update(fixed=True,value=SIGMAS_FIX)
cov_pk_mat=np.cov(cov_pk,rowvar=False,ddof=1)
cov_pk_inv=np.linalg.pinv(cov_pk_mat,rcond=1e-10)

def chi2_pk(fnl_loc,b1):
    params=dict(bf_std);params.update(fnl_loc=float(fnl_loc),b1=float(b1),sigmas=SIGMAS_FIX,p=P_FIX,sn0=SN0_FIX)
    th_f(**params);pk_th=np.array(th_f.power[0],dtype=float)
    pk_map=dict(zip(all_ku,pk_th))
    pk_bin=np.zeros(len(ks_bin))
    for i,(ks,gs) in enumerate(zip(ks_bin,gs_bin)):
        if len(ks)==0:continue
        pv=np.array([pk_map[k] for k in ks]);pk_bin[i]=np.sum(gs*pv)/np.sum(gs)
    d=pk_mean-pk_bin;return float(d@cov_pk_inv@d)

m_pk=Minuit(chi2_pk,fnl_loc=float(bf_std['fnl_loc']),b1=float(bf_std['b1']))
m_pk.errordef=1.0;m_pk.limits['b1']=(0,None);m_pk.limits['fnl_loc']=(-2000,2000)
m_pk.migrad();m_pk.hesse()
print(f'  fnl = {m_pk.values["fnl_loc"]:.3f} +/- {m_pk.errors["fnl_loc"]:.3f}')
print(f'  b1  = {m_pk.values["b1"]:.6f} +/- {m_pk.errors["b1"]:.6f}')
print(f'  chi2/dof = {m_pk.fmin.fval:.2f}/{N_KBINS_FIT-2}')

# ===== k-rebinning缓存 =====
qmax_fd=int((KMAX/K_FUND)**2);nmax_fd=int(KMAX/K_FUND)
gq_fd=gq_fft(qmax_fd,nmax_fd)
qnz=np.nonzero(gq_fd[1:])[0]+1;kv=K_FUND*np.sqrt(qnz.astype(np.float64))
g=gq_fd[qnz].astype(np.float64);dk=DK_FACTOR*K_FUND;n_bins=int(np.ceil(KMAX/dk))+1
bin_idx=np.clip((kv/dk).astype(np.int64),0,n_bins-1)
G_bin=np.bincount(bin_idx,weights=g,minlength=n_bins)
Gk_bin=np.bincount(bin_idx,weights=g*kv,minlength=n_bins)
nz=G_bin>0;G_nz=G_bin[nz];k_eff=Gk_bin[nz]/G_nz
kd=np.geomspace(K_FUND*0.5,KMAX*1.1,N_DENSE)
tmpl_d=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
th_d=PNGTracerPowerSpectrumMultipoles(k=kd,template=tmpl_d,mode='b-p')
th_d.init.params['p'].update(fixed=True,value=P_FIX)
th_d.init.params['sn0'].update(fixed=True,value=SN0_FIX)
th_d.init.params['sigmas'].update(fixed=True,value=SIGMAS_FIX)

# ===== 方法B: 2PCF拟合 =====
print('\n===== 方法B: 2PCF fit (sigmas=0 fixed) =====')
r2_cov_mocks=s_fit**2*pcf_cov[:,fit_mask]
cov_xi=np.cov(r2_cov_mocks,rowvar=False,ddof=1)
cov_xi_inv=np.linalg.pinv(cov_xi,rcond=1e-10)
r2_data=s_fit**2*xi_mean_fit

ncall=[0]
def chi2_xi(fnl_loc,b1):
    params=dict(bf_std);params.update(fnl_loc=float(fnl_loc),b1=float(b1),sigmas=SIGMAS_FIX,p=P_FIX,sn0=SN0_FIX)
    th_d(**params);pd=np.array(th_d.power[0],dtype=float)
    W=G_nz*np.interp(k_eff,kd,pd)
    arg=np.outer(k_eff,s_all);J=np.ones_like(arg);m=arg!=0;J[m]=np.sin(arg[m])/arg[m]
    xi_all=(W@J)/VOL;r2_model=s_fit**2*xi_all[fit_mask]
    d=r2_data-r2_model;ncall[0]+=1;return float(d@cov_xi_inv@d)

t0=time.time()
m_xi=Minuit(chi2_xi,fnl_loc=float(m_pk.values['fnl_loc']),b1=float(m_pk.values['b1']))
m_xi.errordef=1.0;m_xi.limits['b1']=(0,None);m_xi.limits['fnl_loc']=(-2000,2000)
m_xi.migrad();m_xi.hesse()
t_xi=time.time()-t0
n_dof_xi=fit_mask.sum()-2
print(f'  fnl = {m_xi.values["fnl_loc"]:.3f} +/- {m_xi.errors["fnl_loc"]:.3f}')
print(f'  b1  = {m_xi.values["b1"]:.6f} +/- {m_xi.errors["b1"]:.6f}')
print(f'  chi2/dof = {m_xi.fmin.fval:.2f}/{n_dof_xi}')
print(f'  耗时: {t_xi:.1f}s ({ncall[0]} calls)')

print(f'\n{"="*50}')
print(f'汇总 (sigmas={SIGMAS_FIX} fixed)')
print(f'{"="*50}')
print(f'P(k) BinAvgFit:  fnl = {m_pk.values["fnl_loc"]:.3f} +/- {m_pk.errors["fnl_loc"]:.3f},  b1 = {m_pk.values["b1"]:.6f} +/- {m_pk.errors["b1"]:.6f}')
print(f'2PCF fit:        fnl = {m_xi.values["fnl_loc"]:.3f} +/- {m_xi.errors["fnl_loc"]:.3f},  b1 = {m_xi.values["b1"]:.6f} +/- {m_xi.errors["b1"]:.6f}')
print(f'差异:            Dfnl = {m_xi.values["fnl_loc"]-m_pk.values["fnl_loc"]:.3f},  Db1 = {m_xi.values["b1"]-m_pk.values["b1"]:.6f}')
