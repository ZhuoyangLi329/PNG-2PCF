#!/usr/bin/env python3
"""
Mission 9/12: P(k) 拟合 vs 2PCF 拟合 对比
==========================================
任务书 §12：比较从 P(k) best-fit 和从 2PCF best-fit 得到的参数差异。

方法:
1. P(k) 拟合: BinAvgFit（bin-average 口径）→ best-fit params
2. 2PCF 拟合: DataBin+FastDiscrete 建模 → iminuit 最小化 χ²(ξ₀)
   拟合范围: 100-350 Mpc/h
3. 对比两组 best-fit 参数和建模效果

3Gpc fnl100, EZmock covariance
"""
from __future__ import annotations
import os,glob,re,time
import numpy as np
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import next_fast_len,rfft,irfft
from iminuit import Minuit
from collections import defaultdict

OUT=os.path.dirname(os.path.abspath(__file__))
L=3000.0;kf=2*np.pi/L;V=L**3;Z=1.0;P_FIX=1.2;KMAX=15.0
N_DP=20

# 2PCF 拟合范围
XI_RMIN=100.0; XI_RMAX=350.0

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
    return s,v,v.mean(0),v.std(0,ddof=1)
def load_ez(glob_pat,k_target,ndp):
    files=sorted(glob.glob(glob_pat))
    p0_list=[]
    for fp in files:
        try:
            arr=np.loadtxt(fp,comments='#')
            if arr.shape[0]<ndp:continue
            k_in=arr[:ndp,0]
            if np.allclose(k_in,k_target,rtol=0,atol=1e-10):p0_list.append(arr[:ndp,5])
            else:p0_list.append(np.interp(k_target,arr[:,0],arr[:,5]))
        except:pass
    return np.array(p0_list)

def gq_fft(qmax,nmax):
    a=np.zeros(qmax+1,dtype=np.float32);a[0]=1
    sq=np.arange(1,nmax+1,dtype=np.int64)**2;a[sq[sq<=qmax]]=2
    nfft=next_fast_len(3*qmax+1);fa=rfft(a,n=nfft)
    return np.rint(irfft(fa**3,n=nfft)[:qmax+1]).astype(np.int64)

def precompute_rebin(gq,kf,kmax,dk_factor=0.1):
    """k-rebinning 加速缓存"""
    qnz=np.nonzero(gq[1:])[0]+1;kv=kf*np.sqrt(qnz.astype(float));g=gq[qnz].astype(float)
    dk=dk_factor*kf;nb=int(np.ceil(kmax/dk))+1
    bi=np.clip((kv/dk).astype(int),0,nb-1)
    G=np.bincount(bi,weights=g,minlength=nb)
    Gk=np.bincount(bi,weights=g*kv,minlength=nb)
    nz=G>0;return G[nz],Gk[nz]/G[nz]

def fast_xi0(s,G,keff,V,kd,pd):
    """加速离散求和"""
    W=G*np.interp(keff,kd,pd)
    arg=np.outer(keff,s);J=np.ones_like(arg);m=arg!=0;J[m]=np.sin(arg[m])/arg[m]
    return (W@J)/V

def fast_xi0_databin(s,G,keff,V,kd,pd,kmin_b,kmax_b,pk_mean,n_bins):
    """DataBin加速: 低k用数据, 高k用模型"""
    pk_val=np.interp(keff,kd,pd)
    for ib in range(n_bins):
        mask=(keff>=kmin_b[ib])&(keff<kmax_b[ib])
        pk_val[mask]=pk_mean[ib]
    W=G*pk_val
    arg=np.outer(keff,s);J=np.ones_like(arg);m=arg!=0;J[m]=np.sin(arg[m])/arg[m]
    return (W@J)/V

def gq_enumerate(qmax):
    """小 qmax 直接枚举 g_q"""
    nmax=int(np.ceil(np.sqrt(qmax)))+1
    gq=np.zeros(qmax+1,dtype=np.int64)
    for nx in range(-nmax,nmax+1):
        for ny in range(-nmax,nmax+1):
            for nz in range(-nmax,nmax+1):
                if nx==0 and ny==0 and nz==0:continue
                q=nx*nx+ny*ny+nz*nz
                if q<=qmax:gq[q]+=1
    return gq

def build_bin_shells(kf,gq,kmin_b,kmax_b):
    """每个bin内的壳层k和g"""
    q_all=np.nonzero(gq[1:])[0]+1;k_all=kf*np.sqrt(q_all.astype(float));g_all=gq[q_all]
    ks_per_bin=[];gs_per_bin=[]
    for lo,hi in zip(kmin_b,kmax_b):
        m=(k_all>=lo)&(k_all<hi);ks_per_bin.append(k_all[m]);gs_per_bin.append(g_all[m].astype(float))
    return ks_per_bin,gs_per_bin

def met(r2d,r2s,r2m):
    ds=(r2d-r2m)/r2s;return np.nanmean(np.abs(ds)),np.nanmean(ds)

def main():
    T0=time.time()
    print("="*60)
    print("§12: P(k) 拟合 vs 2PCF 拟合")
    print("="*60)

    from cosmoprimo import Cosmology
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate,PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler
    from pypower import PowerSpectrumStatistics
    cosmo=Cosmology(h=0.6711,Omega_b=0.049,Omega_cdm=0.3175-0.049,sigma8=0.834,n_s=0.9624,engine='class')

    # 数据
    print("\n[1] 加载数据...")
    kcen,kmin_b,kmax_b,pk_mk,pk_mean=load_pk(
        "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat",2,99,N_DP)
    s,pcf_mocks,xi_m,xi_s=load_pcf(
        "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",2,99)
    r2d=s**2*xi_m;r2s=s**2*xi_s

    # 2PCF covariance (用 pcf mocks)
    xi_cov=np.cov(pcf_mocks,rowvar=False,ddof=1)  # (nr, nr)
    r2_cov=np.outer(s**2,s**2)*xi_cov  # r²ξ 的 covariance

    # 拟合范围 mask
    fit_mask=(s>=XI_RMIN)&(s<=XI_RMAX)
    s_fit=s[fit_mask]
    r2d_fit=r2d[fit_mask]; r2s_fit=r2s[fit_mask]
    r2_cov_fit=r2_cov[np.ix_(fit_mask,fit_mask)]
    r2_cov_fit_inv=np.linalg.inv(r2_cov_fit)
    print(f"  2PCF 拟合范围: r=[{XI_RMIN},{XI_RMAX}], {fit_mask.sum()} 个点")

    # EZmock pk covariance
    print("  加载 EZmock...")
    ez_cov=load_ez("/pscratch/sd/l/lzy/cov_mock/B3000*seed*/PK_EZmock*_RSD.dat",kcen,N_DP)
    print(f"  EZmock: {ez_cov.shape[0]} used")

    # g_q + rebin cache
    print("\n[2] g_q + rebin cache...")
    qmax=int((KMAX/kf)**2);nmax=int(KMAX/kf)
    gq=gq_fft(qmax,nmax)
    G_nz,k_eff=precompute_rebin(gq,kf,KMAX)

    # BinAvgFit 壳层（拟合区间）
    qmax_fit=int(np.floor((kmax_b[-1]/kf)**2))+1
    gq_fit=gq_enumerate(qmax_fit)
    ks_per_bin,gs_per_bin=build_bin_shells(kf,gq_fit,kmin_b,kmax_b)

    # P(k) 密集网格
    kd=np.geomspace(kf*0.5,KMAX*1.1,300000)

    def eval_pk(k_arr,params):
        tmpl=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
        th=PNGTracerPowerSpectrumMultipoles(k=k_arr,template=tmpl,mode='b-p')
        th.init.params['p'].update(fixed=True,value=P_FIX)
        th.init.params['sn0'].update(fixed=True,value=0)
        th.init.params['sigmas'].update(fixed=False,value=0)
        th(**params);return np.array(th.power[0],dtype=float)

    # BinAvgFit 理论对象（在壳层 k 上）
    all_k_fit=np.unique(np.concatenate(ks_per_bin))
    tmpl_fit=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
    th_fit=PNGTracerPowerSpectrumMultipoles(k=all_k_fit,template=tmpl_fit,mode='b-p')
    th_fit.init.params['p'].update(fixed=True,value=P_FIX)
    th_fit.init.params['sn0'].update(fixed=True,value=0)
    th_fit.init.params['sigmas'].update(fixed=False,value=0)

    # P(k) covariance (EZmock)
    pk_cov_mat=np.cov(ez_cov,rowvar=False,ddof=1)
    pk_cov_inv=np.linalg.inv(pk_cov_mat)

    # ============================================================
    # 方法 1: P(k) BinAvgFit
    # ============================================================
    print("\n[3] P(k) BinAvgFit 拟合...")

    def chi2_pk_binavg(fnl_loc,b1,sigmas):
        params=dict(fnl_loc=float(fnl_loc),b1=float(b1),sigmas=float(sigmas),p=P_FIX,sn0=0.0)
        th_fit(**params)
        pk_all=np.interp(all_k_fit,th_fit.k,np.array(th_fit.power[0]))
        pk_map=dict(zip(all_k_fit,pk_all))
        pk_bin=np.zeros(N_DP)
        for i,(ks,gs) in enumerate(zip(ks_per_bin,gs_per_bin)):
            if len(ks)==0:pk_bin[i]=0
            else:
                pv=np.array([pk_map[k] for k in ks])
                pk_bin[i]=np.sum(gs*pv)/np.sum(gs)
        diff=pk_mean-pk_bin
        return float(diff@pk_cov_inv@diff)

    m_pk=Minuit(chi2_pk_binavg,fnl_loc=80,b1=2.8,sigmas=2.0)
    m_pk.errordef=1.0
    m_pk.limits['b1']=(0,None);m_pk.limits['sigmas']=(0,None);m_pk.limits['fnl_loc']=(-300,300)
    m_pk.migrad()
    bf_pk={'fnl_loc':m_pk.values['fnl_loc'],'b1':m_pk.values['b1'],
           'sigmas':m_pk.values['sigmas'],'p':P_FIX,'sn0':0.0}
    print(f"  P(k) best-fit: fnl={bf_pk['fnl_loc']:.2f}, b1={bf_pk['b1']:.4f}, sig={bf_pk['sigmas']:.2f}")
    print(f"  chi2/dof = {m_pk.fmin.fval:.2f}/{N_DP-3} = {m_pk.fmin.fval/(N_DP-3):.3f}")

    # ============================================================
    # 方法 2: 2PCF DataBin+FastDiscrete 拟合
    # ============================================================
    print(f"\n[4] 2PCF 拟合 (r=[{XI_RMIN},{XI_RMAX}])...")

    # 预缓存 P(k) 评估，避免每次创建新 theory 对象
    tmpl_xi=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
    th_xi=PNGTracerPowerSpectrumMultipoles(k=kd,template=tmpl_xi,mode='b-p')
    th_xi.init.params['p'].update(fixed=True,value=P_FIX)
    th_xi.init.params['sn0'].update(fixed=True,value=0)
    th_xi.init.params['sigmas'].update(fixed=False,value=0)

    n_eval=[0]  # 计数

    def chi2_xi(fnl_loc,b1,sigmas):
        params=dict(fnl_loc=float(fnl_loc),b1=float(b1),sigmas=float(sigmas),p=P_FIX,sn0=0.0)
        th_xi(**params)
        pd_cur=np.array(th_xi.power[0],dtype=float)
        # DataBin: 低k用数据，高k用模型
        xi_model=fast_xi0_databin(s,G_nz,k_eff,V,kd,pd_cur,kmin_b,kmax_b,pk_mean,N_DP)
        r2_model=(s**2*xi_model)[fit_mask]
        diff=r2d_fit-r2_model
        chi2_val=float(diff@r2_cov_fit_inv@diff)
        n_eval[0]+=1
        if n_eval[0]%50==0:
            print(f"    eval #{n_eval[0]}: fnl={fnl_loc:.2f}, b1={b1:.4f}, sig={sigmas:.2f}, chi2={chi2_val:.2f}")
        return chi2_val

    m_xi=Minuit(chi2_xi,fnl_loc=bf_pk['fnl_loc'],b1=bf_pk['b1'],sigmas=bf_pk['sigmas'])
    m_xi.errordef=1.0
    m_xi.limits['b1']=(0,None);m_xi.limits['sigmas']=(0,None);m_xi.limits['fnl_loc']=(-300,300)
    m_xi.migrad()
    bf_xi={'fnl_loc':m_xi.values['fnl_loc'],'b1':m_xi.values['b1'],
           'sigmas':m_xi.values['sigmas'],'p':P_FIX,'sn0':0.0}
    print(f"  2PCF best-fit: fnl={bf_xi['fnl_loc']:.2f}, b1={bf_xi['b1']:.4f}, sig={bf_xi['sigmas']:.2f}")
    print(f"  chi2/dof = {m_xi.fmin.fval:.2f}/{fit_mask.sum()-3}")
    print(f"  总 eval 次数: {n_eval[0]}")

    # ============================================================
    # 比较
    # ============================================================
    print(f"\n{'='*60}")
    print("参数对比")
    print(f"{'='*60}")
    print(f"{'参数':<10s} {'P(k) BinAvgFit':>16s} {'2PCF DataBin':>16s} {'差异':>10s}")
    print("-"*56)
    for p in ['fnl_loc','b1','sigmas']:
        d=bf_xi[p]-bf_pk[p]
        print(f"{p:<10s} {bf_pk[p]:>16.4f} {bf_xi[p]:>16.4f} {d:>+10.4f}")

    # 用两组参数分别建模 2PCF
    print("\n[5] 建模对比...")
    pd_pk=eval_pk(kd,bf_pk)
    pd_xi=eval_pk(kd,bf_xi)
    xi_from_pk=fast_xi0_databin(s,G_nz,k_eff,V,kd,pd_pk,kmin_b,kmax_b,pk_mean,N_DP)
    xi_from_xi=fast_xi0_databin(s,G_nz,k_eff,V,kd,pd_xi,kmin_b,kmax_b,pk_mean,N_DP)

    r2_pk=s**2*xi_from_pk; r2_xi=s**2*xi_from_xi
    ma_pk,ms_pk=met(r2d,r2s,r2_pk)
    ma_xi,ms_xi=met(r2d,r2s,r2_xi)
    print(f"  P(k)拟合→2PCF: mean|Δ/σ|={ma_pk:.4f}, mean(Δ/σ)={ms_pk:+.4f}")
    print(f"  2PCF拟合→2PCF: mean|Δ/σ|={ma_xi:.4f}, mean(Δ/σ)={ms_xi:+.4f}")

    # 画图
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,10))
    fig.suptitle("§12: P(k) fit vs 2PCF fit (3Gpc fnl100, DataBin+FastDiscrete)",fontsize=13)

    ax1.errorbar(s,r2d,yerr=r2s,fmt='ko',ms=3,capsize=2,label='Data',zorder=10)
    ax1.plot(s,r2_pk,'r-',lw=2,label=f"P(k) fit: fnl={bf_pk['fnl_loc']:.1f}, b1={bf_pk['b1']:.3f} [{ma_pk:.3f}]")
    ax1.plot(s,r2_xi,'b--',lw=2,label=f"2PCF fit: fnl={bf_xi['fnl_loc']:.1f}, b1={bf_xi['b1']:.3f} [{ma_xi:.3f}]")
    ax1.axvspan(XI_RMIN,XI_RMAX,alpha=0.08,color='blue',label=f'2PCF fit range [{XI_RMIN:.0f},{XI_RMAX:.0f}]')
    ax1.set_ylabel(r'$r^2\xi_0$');ax1.legend(fontsize=9);ax1.grid(alpha=0.3)

    ax2.axhline(0,color='k',lw=0.5)
    ax2.axhline(1,color='gray',ls=':',lw=0.5);ax2.axhline(-1,color='gray',ls=':',lw=0.5)
    ds_pk=(r2d-r2_pk)/r2s; ds_xi=(r2d-r2_xi)/r2s
    ax2.plot(s,ds_pk,'ro-',ms=3,lw=1.5,label='P(k) fit')
    ax2.plot(s,ds_xi,'b^--',ms=3,lw=1.5,label='2PCF fit')
    ax2.axvspan(XI_RMIN,XI_RMAX,alpha=0.08,color='blue')
    ax2.set_xlabel(r'$r$ [Mpc/h]');ax2.set_ylabel(r'$(Data-Model)/\sigma$')
    ax2.set_ylim(-3,3);ax2.legend(fontsize=9);ax2.grid(alpha=0.3)
    plt.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(os.path.join(OUT,"mission9_pk_vs_xi_fit.png"),dpi=150);plt.close()

    # 保存结果
    md=[f"# §12: P(k) vs 2PCF 拟合对比\n\n"]
    md.append(f"| 参数 | P(k) BinAvgFit | 2PCF DataBin | 差异 |\n|---|---|---|---|\n")
    for p in ['fnl_loc','b1','sigmas']:
        md.append(f"| {p} | {bf_pk[p]:.4f} | {bf_xi[p]:.4f} | {bf_xi[p]-bf_pk[p]:+.4f} |\n")
    md.append(f"\n| 指标 | P(k)→2PCF | 2PCF→2PCF |\n|---|---|---|\n")
    md.append(f"| mean\\|Δ/σ\\| | {ma_pk:.4f} | {ma_xi:.4f} |\n")
    md.append(f"| mean(Δ/σ) | {ms_pk:+.4f} | {ms_xi:+.4f} |\n")
    with open(os.path.join(OUT,"mission9_pk_vs_xi_fit.md"),'w') as f:
        f.writelines(md)

    print(f"\n图: mission9_pk_vs_xi_fit.png\n完成! {time.time()-T0:.0f}s")

if __name__=="__main__":
    main()
