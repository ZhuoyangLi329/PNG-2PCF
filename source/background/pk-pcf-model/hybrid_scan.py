#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid 方案扫描：低 k 离散求和 + 高 k FFTLog 连续积分
扫描 k_split，与纯 FFTLog baseline 对比
运行：conda activate desilike && python hybrid_scan.py
"""

import os, re, glob, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import next_fast_len, rfft, irfft, fht, fhtoffset
from iminuit import Minuit

# ============================================================
# 参数
# ============================================================
BOX_L    = 3000.0
K_FUND   = 2.0 * np.pi / BOX_L
VOL      = BOX_L ** 3
RID_MIN, RID_MAX = 2, 99
N_DP     = 20
KMAX_FD  = 15.0
N_FFT    = 4096
PADDING  = 4.0

PK_DATA_DIR  = '/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut'
PK_DATA_GLOB = 'pk_rsd_3gpc_fnl100_N*.dat'
PK_COV_DIR   = '/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut'
PK_COV_GLOB  = 'pk_rsd_3gpc_fnl0_N*.dat'
PCF_DIR      = '/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut'
PCF_GLOB     = 'pcf_rsd_3gpc_fnl100_N*.dat'
OUT_PATH     = '/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk-pcf-model/hybrid_ksplit_scan.png'

# ============================================================
# 数据读取
# ============================================================
def parse_rid(fp):
    m = re.search(r'N(\d+)', os.path.basename(fp))
    return int(m.group(1)) if m else -1

def load_pk(data_dir, file_glob, rid_min, rid_max, n_dp=20):
    fps = sorted(glob.glob(os.path.join(data_dir, file_glob)), key=parse_rid)
    fps = [f for f in fps if rid_min <= parse_rid(f) <= rid_max]
    ref = np.loadtxt(fps[0], comments='#')
    kcen=ref[:n_dp,0]; kmin=ref[:n_dp,1]; kmax=ref[:n_dp,2]
    mocks=np.array([np.loadtxt(f,comments='#')[:n_dp,5] for f in fps])
    return kcen, kmin, kmax, mocks, mocks.mean(0), mocks.std(0,ddof=1)

def load_pcf(pcf_dir, pcf_glob, rid_min, rid_max):
    fps = sorted(glob.glob(os.path.join(pcf_dir, pcf_glob)), key=parse_rid)
    fps = [f for f in fps if rid_min <= parse_rid(f) <= rid_max]
    s = np.loadtxt(fps[0], comments='#')[:,0]
    mocks = np.array([np.loadtxt(f,comments='#')[:,3] for f in fps])
    return s, mocks.mean(0), mocks.std(0,ddof=1)

print('读取数据...')
kcen,kmin_b,kmax_b,pk_mocks,pk_mean,pk_std = load_pk(PK_DATA_DIR, PK_DATA_GLOB, RID_MIN, RID_MAX, N_DP)
_,_,_,cov_mocks,_,_ = load_pk(PK_COV_DIR, PK_COV_GLOB, RID_MIN, RID_MAX, N_DP)
s_data, xi_mean, xi_std = load_pcf(PCF_DIR, PCF_GLOB, RID_MIN, RID_MAX)
print(f'  PK={pk_mocks.shape[0]}, PCF r=[{s_data.min():.0f},{s_data.max():.0f}]')

# ============================================================
# desilike / 宇宙学
# ============================================================
from cosmoprimo import Cosmology
from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.profilers import MinuitProfiler
from pypower import PowerSpectrumStatistics

cosmo = Cosmology(h=0.6711,Omega_b=0.049,Omega_cdm=0.3175-0.049,sigma8=0.834,n_s=0.9624,engine='class')

def eval_pk_grid(k_grid, params, p_fix=1.2, z=1.0):
    """在指定 k 网格上评估 desilike PNG 理论 P0(k)。"""
    tmpl = FixedPowerSpectrumTemplate(z=z, fiducial=cosmo)
    th   = PNGTracerPowerSpectrumMultipoles(k=k_grid, template=tmpl, mode='b-p')
    th.init.params['p'].update(fixed=True, value=p_fix)
    th.init.params['sn0'].update(fixed=True, value=0.0)
    th.init.params['sigmas'].update(fixed=False, value=float(params.get('sigmas',0.0)))
    p = dict(params); p['p']=p_fix; p['sn0']=0.0
    th(**p)
    return np.array(th.power[0], dtype=float)

# ============================================================
# 标准拟合（desilike bin center，作为 BinAvgFit 初始值）
# ============================================================
print('标准拟合 (desilike)...')
edges = np.concatenate([kmin_b,[kmax_b[-1]]])
nm    = 4/3*np.pi*(edges[1:]**3-edges[:-1]**3)
data_ps = PowerSpectrumStatistics(edges=edges,modes=kcen,power_nonorm=np.array([pk_mean]),
    nmodes=nm,ells=[0],shotnoise_nonorm=0.0,statistic='multipole')
mock_list=[]
for i in range(cov_mocks.shape[0]):
    t=data_ps.deepcopy(); t.power_nonorm.flat[...]=np.array([cov_mocks[i]]).ravel(); mock_list.append(t)
tmpl=FixedPowerSpectrumTemplate(z=1.0,fiducial=cosmo)
theory=PNGTracerPowerSpectrumMultipoles(template=tmpl,mode='b-p')
theory.init.params['p'].update(fixed=True,value=1.2)
theory.init.params['sn0'].update(fixed=True,value=0.0)
theory.init.params['sigmas'].update(fixed=False,value=0.0)
obs=TracerPowerSpectrumMultipolesObservable(data=data_ps,covariance=mock_list,
    klim={0:[float(kcen.min()),float(kcen.max()),float(kcen[1]-kcen[0])]},theory=theory)
like=ObservablesGaussianLikelihood(observables=[obs]); _=like()
like.all_params['p'].update(fixed=True,value=1.2)
like.all_params['sn0'].update(fixed=True,value=0.0)
like.all_params['sigmas'].update(fixed=False,value=0.0)
prof=MinuitProfiler(like,seed=66); profiles=prof.maximize(niterations=27)
bf_std=profiles.bestfit.choice(input=True)
print(f'  fnl={float(bf_std["fnl_loc"]):.2f}, b1={float(bf_std["b1"]):.4f}, sigmas={float(bf_std["sigmas"]):.4f}')

# ============================================================
# BinAvgFit
# ============================================================
print('BinAvgFit...')

def gq_enumerate(qmax):
    """直接枚举计算小 qmax 的壳层简并度 g_q。"""
    nmax=int(np.ceil(np.sqrt(qmax)))+1
    gq=np.zeros(qmax+1,dtype=np.int64)
    for nx in range(-nmax,nmax+1):
        for ny in range(-nmax,nmax+1):
            for nz in range(-nmax,nmax+1):
                if nx==ny==nz==0: continue
                q=nx*nx+ny*ny+nz*nz
                if q<=qmax: gq[q]+=1
    return gq

kmax_fit = float(kmax_b[-1])
qmax_fit = int(np.floor((kmax_fit/K_FUND)**2))+1
gq_fit   = gq_enumerate(qmax_fit)
q_nz     = np.nonzero(gq_fit[1:])[0]+1
k_all    = K_FUND*np.sqrt(q_nz.astype(float)); g_all=gq_fit[q_nz]

# 每个 bin 内的壳层 k 和 g
k_shells=[]; g_shells=[]
for lo,hi in zip(kmin_b,kmax_b):
    m=(k_all>=lo)&(k_all<hi)
    k_shells.append(k_all[m]); g_shells.append(g_all[m].astype(float))

all_k_fit = np.unique(np.concatenate(k_shells))
tmpl2=FixedPowerSpectrumTemplate(z=1.0,fiducial=cosmo)
th2=PNGTracerPowerSpectrumMultipoles(k=all_k_fit,template=tmpl2,mode='b-p')
th2.init.params['p'].update(fixed=True,value=1.2)
th2.init.params['sn0'].update(fixed=True,value=0.0)
th2.init.params['sigmas'].update(fixed=False,value=0.0)
cov_mat=np.cov(cov_mocks,rowvar=False,ddof=1); cov_inv=np.linalg.pinv(cov_mat,rcond=1e-10)

def chi2_binavg(fnl_loc,b1,sigmas):
    params=dict(bf_std); params.update(fnl_loc=float(fnl_loc),b1=float(b1),sigmas=float(sigmas),p=1.2,sn0=0.0)
    th2(**params)
    pk_all_v=np.interp(all_k_fit,th2.k,np.array(th2.power[0]))
    pk_map=dict(zip(all_k_fit,pk_all_v))
    pk_bin=np.array([
        np.sum(g_shells[i]*np.array([pk_map[k] for k in k_shells[i]]))/np.sum(g_shells[i])
        if len(k_shells[i])>0 else np.nan
        for i in range(len(k_shells))])
    diff=pk_mean-pk_bin; return float(diff@cov_inv@diff)

m_fit=Minuit(chi2_binavg,fnl_loc=float(bf_std['fnl_loc']),b1=float(bf_std['b1']),sigmas=float(bf_std['sigmas']))
m_fit.errordef=1.0; m_fit.limits['b1']=(0,None); m_fit.limits['sigmas']=(0,None); m_fit.limits['fnl_loc']=(-2000,2000)
m_fit.migrad()
bf_binavg=dict(bf_std); bf_binavg.update(
    fnl_loc=float(m_fit.values['fnl_loc']),b1=float(m_fit.values['b1']),
    sigmas=float(m_fit.values['sigmas']),p=1.2,sn0=0.0)
print(f'  fnl={bf_binavg["fnl_loc"]:.2f}, b1={bf_binavg["b1"]:.4f}, sigmas={bf_binavg["sigmas"]:.4f}')

# ============================================================
# 工具函数
# ============================================================
def j0(x):
    out=np.ones_like(x,dtype=float); m=x!=0; out[m]=np.sin(x[m])/x[m]; return out

def gq_fft_fn(qmax, nmax):
    """FFT 卷积计算大 qmax 的 g_q。"""
    a=np.zeros(qmax+1,dtype=np.float32); a[0]=1.0
    sq=np.arange(1,nmax+1,dtype=np.int64)**2; a[sq[sq<=qmax]]=2.0
    nfft=next_fast_len(3*qmax+1); fa=rfft(a,n=nfft)
    return np.rint(irfft(fa*fa*fa,n=nfft)[:qmax+1]).astype(np.int64)

def fftlog_xi0(k_grid, p_grid, kmin, kmax, taper_frac=0.06):
    """FFTLog 计算 xi0，返回排好序的 (s, xi0)。"""
    lk=np.log(k_grid); l0=np.log(kmin); l1=np.log(kmax)
    dl=taper_frac*(l1-l0); w=np.zeros_like(k_grid)
    m=(lk>=l0)&(lk<l0+dl);       w[m]=0.5*(1-np.cos(np.pi*(lk[m]-l0)/dl))
    m=(lk>=l0+dl)&(lk<=l1-dl);   w[m]=1.0
    m=(lk>l1-dl)&(lk<=l1);       w[m]=0.5*(1+np.cos(np.pi*(lk[m]-(l1-dl))/dl))
    p_eff=p_grid*w
    dln=np.log(k_grid[1]/k_grid[0]); mu,bias=0.5,0.0
    offset=fhtoffset(dln,mu=mu,initial=0.0,bias=bias)
    A_out=fht(k_grid**1.5*p_eff,dln=dln,mu=mu,offset=offset,bias=bias)
    n=k_grid.size; j=np.arange(n)
    ln_kc=0.5*(np.log(k_grid[0])+np.log(k_grid[-1]))
    s_grid=np.exp(-ln_kc+offset+(j-(n-1)/2.0)*dln)
    xi0=np.sqrt(np.pi/2.0)/(2*np.pi**2)*A_out/s_grid**1.5
    ord_=np.argsort(s_grid)
    return s_grid[ord_], xi0[ord_]

def hybrid_xi0(s, params, k_split, kmax=KMAX_FD):
    """
    Hybrid 方法：k < k_split 离散求和，k >= k_split FFTLog。

    参数
    ----
    s        : ndarray  r 网格 (Mpc/h)
    params   : dict     best-fit 参数字典
    k_split  : float    切换点 (h/Mpc)
    kmax     : float    最大 k

    返回
    ----
    xi0 : ndarray  与 s 等长的 ξ0(r)
    """
    kf=K_FUND; V=VOL

    # 1) 低 k 离散求和（k_q < k_split）
    q_split  = int((k_split/kf)**2)
    nmax_spl = int(k_split/kf)
    gq_low   = gq_fft_fn(q_split, nmax_spl)
    q_nz_low = np.nonzero(gq_low[1:])[0]+1
    kv = kf*np.sqrt(q_nz_low.astype(float))
    g  = gq_low[q_nz_low].astype(float)
    # 在这些离散 k_q 上插值 P_model
    kd_low = np.geomspace(kf*0.5, k_split*1.5, 20_000)
    pd_low = eval_pk_grid(kd_low, params)
    w = g * np.interp(kv, kd_low, pd_low)
    xi_low = np.zeros(len(s))
    for js in range(0, len(s), 20):
        seg=s[js:js+20]; xi_low[js:js+len(seg)] += w @ j0(np.outer(kv, seg))
    xi_low /= V

    # 2) 高 k FFTLog（k_split 到 kmax）
    k_grid  = np.geomspace(k_split/PADDING, kmax*PADDING, N_FFT)
    pd_high = eval_pk_grid(k_grid, params)
    s_fl, xi_fl = fftlog_xi0(k_grid, pd_high, k_split, kmax)
    xi_high = np.interp(s, s_fl, xi_fl)

    return xi_low + xi_high

# ============================================================
# 计算各 k_split 的 Hybrid 结果
# ============================================================
k_splits = [0.1, 0.3, 0.5, 1.0]
xi_hybrid = {}; timing = {}

print('\nHybrid 扫描...')
for ks in k_splits:
    t0=time.time()
    xi_hybrid[ks] = hybrid_xi0(s_data, bf_binavg, ks)
    timing[ks]    = time.time()-t0
    print(f'  k_split={ks}: {timing[ks]:.1f}s')

# 纯 FFTLog baseline（kmin=kf，不加 IR 窗口）
print('Baseline FFTLog (kmin=kf)...')
k_grid_bl=np.geomspace(K_FUND/4.0, KMAX_FD*4.0, N_FFT)
pd_bl=eval_pk_grid(k_grid_bl, bf_binavg)
s_bl,xi_bl=fftlog_xi0(k_grid_bl, pd_bl, K_FUND, KMAX_FD)
xi_baseline=np.interp(s_data, s_bl, xi_bl)

# ============================================================
# 画图
# ============================================================
r2_data = s_data**2 * xi_mean
r2_err  = s_data**2 * xi_std
colors  = ['tab:blue','tab:orange','tab:green','tab:red']

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(11,10),sharex=True)

ax1.errorbar(s_data,r2_data,yerr=r2_err,fmt='ko',ms=4,capsize=2,label='观测均值',zorder=10)
ds_bl=(r2_data-s_data**2*xi_baseline)/r2_err
ax1.plot(s_data,s_data**2*xi_baseline,'gray',ls=':',lw=1.5,
         label=f'纯 FFTLog baseline (kmin=kf)  |Δ/σ|={np.nanmean(np.abs(ds_bl)):.3f}')
for ks,color in zip(k_splits,colors):
    r2m=s_data**2*xi_hybrid[ks]; ds=(r2_data-r2m)/r2_err
    mabs=np.nanmean(np.abs(ds)); ms=np.nanmean(ds)
    ax1.plot(s_data,r2m,color=color,lw=2,
             label=f'Hybrid k_split={ks} h/Mpc  ({timing[ks]:.0f}s,  |Δ/σ|={mabs:.3f},  Δ/σ={ms:+.3f})')

ax1.set_ylabel(r'$r^2\xi_0(r)$',fontsize=13)
ax1.set_title(
    f'Hybrid 方案：低 k 离散 + 高 k FFTLog\n'
    f'BinAvgFit: fnl={bf_binavg["fnl_loc"]:.1f}, b1={bf_binavg["b1"]:.3f}, sigmas={bf_binavg["sigmas"]:.3f}',
    fontsize=12)
ax1.legend(fontsize=9); ax1.grid(ls='--',alpha=0.35)

ax2.axhline(0,color='k',lw=1)
for lv in [1,-1]: ax2.axhline(lv,color='gray',ls=':',lw=0.8)
for lv in [2,-2]: ax2.axhline(lv,color='orange',ls=':',lw=0.8)
ax2.plot(s_data,ds_bl,'s:',color='gray',ms=3,lw=1,alpha=0.7,label='Baseline FFTLog')
for ks,color in zip(k_splits,colors):
    r2m=s_data**2*xi_hybrid[ks]; ds=(r2_data-r2m)/r2_err
    ax2.plot(s_data,ds,'o-',color=color,ms=3,lw=1.5,label=f'k_split={ks}')
ax2.set_xlabel(r'$r\,[\mathrm{Mpc}/h]$',fontsize=13)
ax2.set_ylabel(r'$(Data-Model)/\sigma$ of $r^2\xi_0$',fontsize=12)
ax2.set_ylim(-3,3); ax2.legend(fontsize=9); ax2.grid(ls='--',alpha=0.35)

plt.tight_layout()
plt.savefig(OUT_PATH,dpi=150,bbox_inches='tight')
print(f'\n图已保存：{OUT_PATH}')

# ============================================================
# 指标汇总
# ============================================================
print('\n===== 指标汇总 =====')
print(f'{"方法":35s}  耗时(s)  mean|Δ/σ|  mean(Δ/σ)')
for ks in k_splits:
    r2m=s_data**2*xi_hybrid[ks]; ds=(r2_data-r2m)/r2_err
    print(f'  Hybrid k_split={ks:<5}             {timing[ks]:5.1f}s    {np.nanmean(np.abs(ds)):.4f}    {np.nanmean(ds):+.4f}')
print(f'  Baseline FFTLog (kmin=kf)         ~0s    {np.nanmean(np.abs(ds_bl)):.4f}    {np.nanmean(ds_bl):+.4f}')
