#!/usr/bin/env python3
"""Mission 9: 1Gpc fnl100 kmax 扫描（独立脚本）"""
from __future__ import annotations
import os, glob, re, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import fht, fhtoffset, next_fast_len, rfft, irfft

OUT = os.path.dirname(os.path.abspath(__file__))
L = 1000.0; kf = 2*np.pi/L; V = L**3
KMAX_LIST = [5,8,10,12,15,18,20]
IR_X = 4.0; Z = 1.0; P_FIX = 1.2
N_DP = 20; N_DENSE = 300000

def rid(f):
    m = re.search(r'N(\d+)', os.path.basename(f))
    return int(m.group(1)) if m else -1

def load(pat, rmin, rmax, ndp, col):
    fs = [f for f in sorted(glob.glob(pat), key=rid) if rmin <= rid(f) <= rmax]
    ref = np.loadtxt(fs[0], comments='#')
    v = np.array([np.loadtxt(f, comments='#')[:ndp, col] for f in fs])
    return ref[:ndp, 0], ref[:ndp, 1], ref[:ndp, 2], v, v.mean(0), v.std(0, ddof=1)

def load_pcf(pat, rmin, rmax):
    fs = [f for f in sorted(glob.glob(pat), key=rid) if rmin <= rid(f) <= rmax]
    s = np.loadtxt(fs[0], comments='#')[:, 0]
    v = np.array([np.loadtxt(f, comments='#')[:, 3] for f in fs])
    return s, v.mean(0), v.std(0, ddof=1)

def j0(x):
    x = np.asarray(x, float); o = np.ones_like(x); m = x != 0; o[m] = np.sin(x[m])/x[m]; return o

def gq_fft(qmax, nmax):
    a = np.zeros(qmax+1, dtype=np.float32); a[0]=1
    sq = np.arange(1,nmax+1,dtype=np.int64)**2; a[sq[sq<=qmax]]=2
    nfft = next_fast_len(3*qmax+1)
    print(f"  FFT: qmax={qmax}, nfft={nfft}")
    fa = rfft(a, n=nfft); return np.rint(irfft(fa**3, n=nfft)[:qmax+1]).astype(np.int64)

def xi_disc(s, gq, kf, V, kd, pd, kmax_cut, chunk=500000):
    qm = min(int((kmax_cut/kf)**2), len(gq)-1)
    qnz = np.nonzero(gq[1:qm+1])[0]+1; xi = np.zeros(len(s))
    for i0 in range(0, len(qnz), chunk):
        qb = qnz[i0:i0+chunk]; kv = kf*np.sqrt(qb.astype(float))
        pv = np.interp(kv, kd, pd); w = gq[qb].astype(float)*pv
        for js in range(0, len(s), 10):
            je = min(js+10, len(s)); xi[js:je] += np.dot(w, j0(np.outer(kv, s[js:je])))
    return xi / V

def xi_fftlog(kg, p0, kmin, kmax, ir_win=None):
    lk=np.log(kg); l0,l1=np.log(kmin),np.log(kmax); dl=0.06*(l1-l0)
    w=np.zeros_like(kg)
    left,right=l0+dl,l1-dl
    m=(lk>=l0)&(lk<left); w[m]=0.5*(1-np.cos(np.pi*(lk[m]-l0)/dl))
    m=(lk>=left)&(lk<=right); w[m]=1.0
    m=(lk>right)&(lk<=l1); w[m]=0.5*(1+np.cos(np.pi*(lk[m]-right)/dl))
    if ir_win is not None: w*=ir_win
    dln=np.log(kg[1]/kg[0]); off=fhtoffset(dln,mu=0.5,initial=0,bias=0)
    A=fht((kg**1.5)*p0*w,dln=dln,mu=0.5,offset=off,bias=0)
    n=kg.size; jj=np.arange(n)
    sg=np.exp(-(np.log(kg[0])+np.log(kg[-1]))/2+off+(jj-(n-1)/2)*dln)
    return sg, np.sqrt(np.pi/2)/(2*np.pi**2)*A/sg**1.5

def met(r2d, r2s, r2m):
    ds=(r2d-r2m)/r2s; return np.nanmean(np.abs(ds)), np.nanmean(ds)

def main():
    T0=time.time()
    print("1Gpc fnl100 kmax scan")
    # 注意：1Gpc 第一个 bin (k=0.0035) 有 0 个模式 (k_f=0.0063 > k_max of bin)
    # 必须跳过它，否则 covariance 矩阵奇异
    _kcen,_kmin,_kmax_b,_pk_raw,_,_ = load("/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat", 1,50,N_DP+1,5)
    _,_,_,_cov_raw,_,_ = load("/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat", 1,50,N_DP+1,5)
    # 找到第一个有模式的 bin (P0 均值 > 0)
    first_valid = 0
    for ib in range(len(_kcen)):
        if _pk_raw[:, ib].mean() > 0:
            first_valid = ib; break
    print(f"  跳过前 {first_valid} 个空 bin")
    kcen = _kcen[first_valid:first_valid+N_DP]
    kmin = _kmin[first_valid:first_valid+N_DP]
    kmax_b = _kmax_b[first_valid:first_valid+N_DP]
    _pk = _pk_raw[:, first_valid:first_valid+N_DP]
    pk_m = _pk.mean(0); pk_s = _pk.std(0, ddof=1)
    cov_pk = _cov_raw[:, first_valid:first_valid+N_DP]
    s, xi_m, xi_s = load_pcf("/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat", 1,50)
    r2d=s**2*xi_m; r2s=s**2*xi_s
    print(f"  pk: {_pk.shape[0]}, cov: {cov_pk.shape[0]}, pcf s=[{s[0]:.0f},{s[-1]:.0f}]")

    from cosmoprimo import Cosmology
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler
    from pypower import PowerSpectrumStatistics

    cosmo = Cosmology(h=0.6711, Omega_b=0.049, Omega_cdm=0.3175-0.049, sigma8=0.834, n_s=0.9624, engine='class')
    edges=np.concatenate([kmin,[kmax_b[-1]]]); nm=4/3*np.pi*(edges[1:]**3-edges[:-1]**3)
    dps = PowerSpectrumStatistics(edges=edges, modes=kcen, power_nonorm=np.array([pk_m]),
          nmodes=nm, ells=[0], shotnoise_nonorm=0, statistic='multipole')
    ml=[]
    for i in range(cov_pk.shape[0]):
        t=dps.deepcopy(); t.power_nonorm.flat[...]=np.array([cov_pk[i]]).ravel(); ml.append(t)
    tmpl=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
    th=PNGTracerPowerSpectrumMultipoles(template=tmpl,mode='b-p')
    th.init.params['p'].update(fixed=True,value=P_FIX)
    th.init.params['sn0'].update(fixed=True,value=0)
    th.init.params['sigmas'].update(fixed=False,value=0)
    obs=TracerPowerSpectrumMultipolesObservable(data=dps,covariance=ml,
        klim={0:[float(kcen.min()),float(kcen.max()),float(kcen[1]-kcen[0])]},theory=th)
    like=ObservablesGaussianLikelihood(observables=[obs]); _=like()
    like.all_params['p'].update(fixed=True,value=P_FIX)
    like.all_params['sn0'].update(fixed=True,value=0)
    like.all_params['sigmas'].update(fixed=False,value=0)
    prof=MinuitProfiler(like,seed=66); profiles=prof.maximize(niterations=27)
    bf=profiles.bestfit.choice(input=True)
    fnl=float(bf['fnl_loc']); b1=float(bf['b1']); sig=float(bf['sigmas'])
    print(f"  Best-fit: fnl={fnl:.2f}, b1={b1:.4f}, sig={sig:.2f}")

    kd=np.geomspace(kf*0.5,22,N_DENSE)
    tmpl2=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
    th2=PNGTracerPowerSpectrumMultipoles(k=kd,template=tmpl2,mode='b-p')
    th2.init.params['p'].update(fixed=True,value=P_FIX)
    th2.init.params['sn0'].update(fixed=True,value=0)
    th2.init.params['sigmas'].update(fixed=False,value=0)
    th2(**bf); pd=np.array(th2.power[0],dtype=float)

    qmax=int((20/kf)**2); nmax=int(20/kf)
    print(f"  g_q: qmax={qmax}...")
    gq=gq_fft(qmax,nmax)

    # Baseline + Window (kmax=15)
    kg=np.geomspace(kf/4,15*4,4096)
    tmpl3=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
    th3=PNGTracerPowerSpectrumMultipoles(k=kg,template=tmpl3,mode='b-p')
    th3.init.params['p'].update(fixed=True,value=P_FIX)
    th3.init.params['sn0'].update(fixed=True,value=0)
    th3.init.params['sigmas'].update(fixed=False,value=0)
    th3(**bf); pfg=np.array(th3.power[0])

    sb,xb=xi_fftlog(kg,pfg,kf,15.0)
    xi_bl=np.interp(s,np.sort(sb),xb[np.argsort(sb)])

    kgw=np.geomspace(1e-4/4,15*4,4096)
    tmpl4=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
    th4=PNGTracerPowerSpectrumMultipoles(k=kgw,template=tmpl4,mode='b-p')
    th4.init.params['p'].update(fixed=True,value=P_FIX)
    th4.init.params['sn0'].update(fixed=True,value=0)
    th4.init.params['sigmas'].update(fixed=False,value=0)
    th4(**bf); pfw=np.array(th4.power[0])
    iw=np.ones_like(kgw); m=kgw<kf; iw[m]=1-np.exp(-(kgw[m]/kf)**IR_X)
    sw,xw=xi_fftlog(kgw,pfw,1e-4,15.0,ir_win=iw)
    xi_wn=np.interp(s,np.sort(sw),xw[np.argsort(sw)])

    res = {}
    res["Baseline_15"] = met(r2d, r2s, s**2*xi_bl)
    res["ExpWindow_15"] = met(r2d, r2s, s**2*xi_wn)

    for km in KMAX_LIST:
        print(f"  Disc kmax={km}...")
        t1=time.time()
        xd = xi_disc(s, gq, kf, V, kd, pd, km)
        res[f"Disc_{km}"] = met(r2d, r2s, s**2*xd)
        print(f"    {res[f'Disc_{km}'][0]:.4f}, {res[f'Disc_{km}'][1]:+.4f} ({time.time()-t1:.0f}s)")

    print(f"\n--- 1Gpc fnl100 (fnl={fnl:.1f}, b1={b1:.3f}) ---")
    print(f"{'方法':<20s} {'mean|Δ/σ|':>10s} {'mean(Δ/σ)':>10s}")
    for n,(a,m) in sorted(res.items()):
        print(f"{n:<20s} {a:>10.4f} {m:>+10.4f}")

    md = [f"# 1Gpc fnl100 kmax scan\nfnl={fnl:.2f}, b1={b1:.4f}\n\n"]
    md.append("| 方法 | mean\\|Δ/σ\\| | mean(Δ/σ) |\n|---|---|---|\n")
    for n,(a,m) in sorted(res.items()):
        md.append(f"| {n} | {a:.4f} | {m:+.4f} |\n")
    with open(os.path.join(OUT, "mission9_1gpc_kmax.md"), 'w') as f:
        f.writelines(md)
    print(f"\n完成! {time.time()-T0:.0f}s")

if __name__=="__main__":
    main()
