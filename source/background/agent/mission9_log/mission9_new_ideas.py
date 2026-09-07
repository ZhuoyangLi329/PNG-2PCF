#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission 9: 新方向探索
======================

已知:
- FullDiscrete fnl=0 偏差 ≈ 0 (+0.021σ)，fnl=100 偏差 = -0.085σ
- 偏差来自 PNG 贡献
- MW fitting 过度校正（失败）

新方向:
1. DataBin: 低 k 用测量 P(k) 的 bin 值，高 k 用模型 → 消除模型偏差
2. PNG 缩放: ξ = ξ(fnl=0) + α × [ξ(fnl=bf) - ξ(fnl=0)]，扫描 α
3. 自由 sn0: 释放 shot noise 参数，可能吸收小偏差
4. DataBin + model hybrid: 精确的 per-shell 实现
"""

from __future__ import annotations
import os, glob, re, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import next_fast_len, rfft, irfft, fht, fhtoffset
from collections import defaultdict

OUT = os.path.dirname(os.path.abspath(__file__))

def rid(f):
    m = re.search(r'N(\d+)', os.path.basename(f))
    return int(m.group(1)) if m else -1

def load_pk(pat, rmin, rmax, ndp, col=5):
    fs = [f for f in sorted(glob.glob(pat), key=rid) if rmin <= rid(f) <= rmax]
    ref = np.loadtxt(fs[0], comments='#')
    v = np.array([np.loadtxt(f, comments='#')[:ndp, col] for f in fs])
    return ref[:ndp,0], ref[:ndp,1], ref[:ndp,2], v, v.mean(0), v.std(0,ddof=1)

def load_pcf(pat, rmin, rmax):
    fs = [f for f in sorted(glob.glob(pat), key=rid) if rmin <= rid(f) <= rmax]
    s = np.loadtxt(fs[0], comments='#')[:,0]
    v = np.array([np.loadtxt(f, comments='#')[:,3] for f in fs])
    return s, v.mean(0), v.std(0,ddof=1)

def j0(x):
    x=np.asarray(x,float); o=np.ones_like(x); m=x!=0; o[m]=np.sin(x[m])/x[m]; return o

def gq_fft(qmax, nmax):
    a=np.zeros(qmax+1,dtype=np.float32); a[0]=1
    sq=np.arange(1,nmax+1,dtype=np.int64)**2; a[sq[sq<=qmax]]=2
    nfft=next_fast_len(3*qmax+1)
    print(f"  FFT: qmax={qmax}, nfft={nfft}")
    fa=rfft(a,n=nfft); return np.rint(irfft(fa**3,n=nfft)[:qmax+1]).astype(np.int64)

def xi_disc_custom_p(s, gq, kf, V, p_func, chunk=500000):
    """
    全离散求和，P(k) 由自定义函数 p_func(k_array) 提供。

    参数
    ----------
    p_func : callable  输入 k 数组，返回 P(k) 数组
    """
    qnz = np.nonzero(gq[1:])[0]+1; xi = np.zeros(len(s))
    for i0 in range(0, len(qnz), chunk):
        qb = qnz[i0:i0+chunk]; kv = kf*np.sqrt(qb.astype(float))
        pv = p_func(kv); w = gq[qb].astype(float)*pv
        for js in range(0, len(s), 10):
            je=min(js+10,len(s)); xi[js:je]+=np.dot(w,j0(np.outer(kv,s[js:je])))
        if (i0//chunk) % 40 == 0:
            print(f"    {min(i0+chunk,len(qnz))}/{len(qnz)}")
    return xi/V

def met(r2d, r2s, r2m):
    ds=(r2d-r2m)/r2s; return np.nanmean(np.abs(ds)), np.nanmean(ds)

def xi_fftlog(k_grid, p0, kmin, kmax, ir_win=None):
    lk=np.log(k_grid); l0,l1=np.log(kmin),np.log(kmax); dl=0.06*(l1-l0)
    w=np.zeros_like(k_grid)
    left,right=l0+dl,l1-dl
    m=(lk>=l0)&(lk<left); w[m]=0.5*(1-np.cos(np.pi*(lk[m]-l0)/dl))
    m=(lk>=left)&(lk<=right); w[m]=1.0
    m=(lk>right)&(lk<=l1); w[m]=0.5*(1+np.cos(np.pi*(lk[m]-right)/dl))
    if ir_win is not None: w*=ir_win
    dln=np.log(k_grid[1]/k_grid[0]); off=fhtoffset(dln,mu=0.5,initial=0,bias=0)
    A=fht((k_grid**1.5)*p0*w,dln=dln,mu=0.5,offset=off,bias=0)
    n=k_grid.size; jj=np.arange(n)
    sg=np.exp(-(np.log(k_grid[0])+np.log(k_grid[-1]))/2+off+(jj-(n-1)/2)*dln)
    return sg, np.sqrt(np.pi/2)/(2*np.pi**2)*A/sg**1.5

def ir_window(k, kf, x):
    w=np.ones_like(k); m=k<kf; w[m]=1-np.exp(-(k[m]/kf)**x); return w

# ============================================================
def main():
    T0 = time.time()
    print("="*60)
    print("Mission 9: 新方向探索")
    print("="*60)

    from cosmoprimo import Cosmology
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler
    from pypower import PowerSpectrumStatistics

    cosmo = Cosmology(h=0.6711, Omega_b=0.049, Omega_cdm=0.3175-0.049,
                      sigma8=0.834, n_s=0.9624, engine='class')
    Z=1.0; P_FIX=1.2

    # ============================================================
    # 测试两个盒子
    # ============================================================
    configs = [
        ("3Gpc_fnl100", 3000.0,
         "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat",
         "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
         "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",
         2, 99, 20, 12.0),
        ("1Gpc_fnl100", 1000.0,
         "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat",
         "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
         "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat",
         1, 50, 20, 4.0),
    ]

    all_results = []

    for cfg_name, L, pk_pat, cov_pat, pcf_pat, rmin, rmax, ndp, ir_x in configs:
        kf = 2*np.pi/L; V = L**3
        print(f"\n{'='*60}")
        print(f"  {cfg_name}  (L={L}, kf={kf:.6f})")
        print(f"{'='*60}")

        # 数据
        _kcen,_kmin,_kmax,_pk,_,_ = load_pk(pk_pat, rmin, rmax, ndp+1)
        _,_,_,_cov,_,_ = load_pk(cov_pat, rmin, rmax, ndp+1)
        # 跳过空 bin
        first_valid = 0
        for ib in range(len(_kcen)):
            if _pk[:,ib].mean() > 0:
                first_valid = ib; break
        kcen=_kcen[first_valid:first_valid+ndp]
        kmin_b=_kmin[first_valid:first_valid+ndp]
        kmax_b=_kmax[first_valid:first_valid+ndp]
        pk_mocks=_pk[:,first_valid:first_valid+ndp]
        cov_mocks=_cov[:,first_valid:first_valid+ndp]
        pk_mean=pk_mocks.mean(0); pk_std=pk_mocks.std(0,ddof=1)

        s, xi_m, xi_s = load_pcf(pcf_pat, rmin, rmax)
        r2d=s**2*xi_m; r2s=s**2*xi_s
        kmax_data = kmax_b[-1]  # 数据覆盖的最大 k
        print(f"  data k=[{kcen[0]:.5f},{kcen[-1]:.5f}], s=[{s[0]:.0f},{s[-1]:.0f}]")
        print(f"  跳过 {first_valid} 个空 bin")

        # 标准拟合
        print("\n[1] 标准拟合...")
        edges=np.concatenate([kmin_b,[kmax_b[-1]]])
        nm=4/3*np.pi*(edges[1:]**3-edges[:-1]**3)
        dps=PowerSpectrumStatistics(edges=edges,modes=kcen,
            power_nonorm=np.array([pk_mean]),nmodes=nm,ells=[0],
            shotnoise_nonorm=0,statistic='multipole')
        ml=[]
        for i in range(cov_mocks.shape[0]):
            t=dps.deepcopy(); t.power_nonorm.flat[...]=np.array([cov_mocks[i]]).ravel(); ml.append(t)

        def do_fit(free_sn0=False, label=""):
            tmpl=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
            th=PNGTracerPowerSpectrumMultipoles(template=tmpl,mode='b-p')
            th.init.params['p'].update(fixed=True,value=P_FIX)
            th.init.params['sn0'].update(fixed=not free_sn0, value=0)
            th.init.params['sigmas'].update(fixed=False,value=0)
            obs=TracerPowerSpectrumMultipolesObservable(data=dps,covariance=ml,
                klim={0:[float(kcen.min()),float(kcen.max()),float(kcen[1]-kcen[0])]},theory=th)
            like=ObservablesGaussianLikelihood(observables=[obs]); _=like()
            like.all_params['p'].update(fixed=True,value=P_FIX)
            like.all_params['sn0'].update(fixed=not free_sn0, value=0)
            like.all_params['sigmas'].update(fixed=False,value=0)
            prof=MinuitProfiler(like,seed=66)
            pr=prof.maximize(niterations=27)
            bf=pr.bestfit.choice(input=True)
            print(f"  {label}: fnl={float(bf['fnl_loc']):.2f}, b1={float(bf['b1']):.4f}, "
                  f"sig={float(bf['sigmas']):.2f}, sn0={float(bf['sn0']):.1f}")
            return bf

        bf_std = do_fit(free_sn0=False, label="标准 (sn0=0)")
        bf_sn0 = do_fit(free_sn0=True, label="自由 sn0")

        # fnl=0 的 best-fit (用于 PNG 分离)
        print("\n[2] fnl=0 best-fit...")
        bf_fnl0 = dict(bf_std)
        bf_fnl0['fnl_loc'] = type(bf_std['fnl_loc']).__class__(bf_std['fnl_loc'])
        # 简单方式：直接把 fnl 设为 0
        from copy import deepcopy
        bf_fnl0 = {k: v for k, v in bf_std.items()}
        # 需要创建新的 ParameterArray with value 0
        # 更简单的办法：评估两次 P(k)，分别用 bf_std 和 fnl=0

        # 密集 P(k)
        print("\n[3] P(k) 密集网格...")
        kmax_int = 15.0
        kd = np.geomspace(kf*0.5, kmax_int*1.1, 300000)

        def eval_pk(k_arr, bf_params):
            tmpl=FixedPowerSpectrumTemplate(z=Z,fiducial=cosmo)
            th=PNGTracerPowerSpectrumMultipoles(k=k_arr,template=tmpl,mode='b-p')
            th.init.params['p'].update(fixed=True,value=P_FIX)
            th.init.params['sn0'].update(fixed=True,value=0)
            th.init.params['sigmas'].update(fixed=False,value=0)
            th(**bf_params)
            return np.array(th.power[0],dtype=float)

        pd_std = eval_pk(kd, bf_std)
        pd_sn0 = eval_pk(kd, bf_sn0)

        # P(k) with fnl=0 (其他参数不变)
        bf_nopng = dict(bf_std)
        # 手动构造 fnl=0 版本
        from desilike.parameter import ParameterArray
        bf_nopng_params = {}
        for k, v in bf_std.items():
            if k == 'fnl_loc':
                bf_nopng_params[k] = ParameterArray(0.0, param=v.param) if hasattr(v, 'param') else 0.0
            else:
                bf_nopng_params[k] = v
        pd_nopng = eval_pk(kd, bf_nopng_params)

        # g_q
        print("\n[4] g_q...")
        qmax=int((kmax_int/kf)**2); nmax=int(kmax_int/kf)
        gq = gq_fft(qmax, nmax)

        # ============================================================
        # 方法测试
        # ============================================================
        results = {}

        # 方法 A: 标准 FullDiscrete
        print("\n[5] 方法 A: 标准 FullDiscrete...")
        p_interp_std = lambda k: np.interp(k, kd, pd_std)
        xi_A = xi_disc_custom_p(s, gq, kf, V, p_interp_std)
        results["A.FullDiscrete"] = met(r2d, r2s, s**2*xi_A)

        # 方法 B: 自由 sn0 拟合 + FullDiscrete
        print("\n[6] 方法 B: 自由 sn0 + FullDiscrete...")
        p_interp_sn0 = lambda k: np.interp(k, kd, pd_sn0)
        xi_B = xi_disc_custom_p(s, gq, kf, V, p_interp_sn0)
        results["B.FreeSN0"] = met(r2d, r2s, s**2*xi_B)

        # 方法 C: DataBin hybrid
        # 低 k (数据范围内) 用测量 P 的 bin 平均值，高 k 用模型
        print("\n[7] 方法 C: DataBin hybrid...")
        def p_databin_hybrid(k_arr):
            p_out = np.interp(k_arr, kd, pd_std)  # 默认用模型
            for ib in range(len(kcen)):
                mask = (k_arr >= kmin_b[ib]) & (k_arr < kmax_b[ib])
                p_out[mask] = pk_mean[ib]  # 数据范围内用测量值
            return p_out
        xi_C = xi_disc_custom_p(s, gq, kf, V, p_databin_hybrid)
        results["C.DataBin"] = met(r2d, r2s, s**2*xi_C)

        # 方法 D: PNG 缩放扫描
        # ξ = ξ(fnl=0) + α × ΔP_PNG 的离散求和
        print("\n[8] 方法 D: PNG 缩放...")
        p_interp_nopng = lambda k: np.interp(k, kd, pd_nopng)
        xi_nopng = xi_disc_custom_p(s, gq, kf, V, p_interp_nopng)
        xi_png_only = xi_A - xi_nopng  # PNG 贡献

        alpha_list = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
        best_alpha = 1.0; best_ma = 999
        for alpha in alpha_list:
            xi_scaled = xi_nopng + alpha * xi_png_only
            ma, ms = met(r2d, r2s, s**2*xi_scaled)
            results[f"D.alpha{alpha:.2f}"] = (ma, ms)
            if ma < best_ma:
                best_ma = ma; best_alpha = alpha
            print(f"    α={alpha:.2f}: mean|Δ/σ|={ma:.4f}, mean(Δ/σ)={ms:+.4f}")
        print(f"    最优 α = {best_alpha:.2f}")

        # 方法 E: ExpWindow (参考)
        print("\n[9] 方法 E: ExpWindow (参考)...")
        kgw = np.geomspace(1e-4/4, kmax_int*4, 4096)
        pw = eval_pk(kgw, bf_std)
        iw = ir_window(kgw, kf, ir_x)
        sw, xw = xi_fftlog(kgw, pw, 1e-4, kmax_int, ir_win=iw)
        xi_E = np.interp(s, np.sort(sw), xw[np.argsort(sw)])
        results["E.ExpWindow"] = met(r2d, r2s, s**2*xi_E)

        # 方法 F: DataBin + 模型插值校正
        # 在 bin 内用线性插值而不是常数（使用相邻 bin 的数据推断斜率）
        print("\n[10] 方法 F: DataBin interpolated...")
        def p_databin_interp(k_arr):
            # 在 data 范围内，用 bin 中心值做线性插值
            p_out = np.interp(k_arr, kd, pd_std)  # 默认
            in_range = (k_arr >= kmin_b[0]) & (k_arr <= kmax_b[-1])
            p_out[in_range] = np.interp(k_arr[in_range], kcen, pk_mean)
            return p_out
        xi_F = xi_disc_custom_p(s, gq, kf, V, p_databin_interp)
        results["F.DataBinInterp"] = met(r2d, r2s, s**2*xi_F)

        # 汇总
        print(f"\n--- {cfg_name} 汇总 ---")
        print(f"{'方法':<22s} {'mean|Δ/σ|':>10s} {'mean(Δ/σ)':>10s}")
        print("-"*46)
        for nm in sorted(results.keys()):
            ma, ms = results[nm]
            marker = " ★" if ma < results["A.FullDiscrete"][0] else ""
            print(f"{nm:<22s} {ma:>10.4f} {ms:>+10.4f}{marker}")

        all_results.append((cfg_name, dict(results), float(bf_std['fnl_loc']),
                           float(bf_std['b1']), best_alpha))

    # 保存结果
    md = ["# Mission 9: 新方向探索结果\n\n"]
    for cfg_name, results, fnl, b1, best_a in all_results:
        md.append(f"## {cfg_name} (fnl={fnl:.2f}, b1={b1:.4f})\n\n")
        md.append("| 方法 | mean\\|Δ/σ\\| | mean(Δ/σ) |\n|---|---|---|\n")
        for nm in sorted(results.keys()):
            ma, ms = results[nm]
            md.append(f"| {nm} | {ma:.4f} | {ms:+.4f} |\n")
        md.append(f"\n最优 PNG 缩放因子 α = {best_a:.2f}\n\n")

    with open(os.path.join(OUT, "mission9_new_ideas.md"), 'w') as f:
        f.writelines(md)

    print(f"\n完成! {time.time()-T0:.0f}s")

if __name__ == "__main__":
    main()
