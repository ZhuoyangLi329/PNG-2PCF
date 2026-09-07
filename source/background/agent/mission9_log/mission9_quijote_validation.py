#!/usr/bin/env python3
"""
Mission 9: Quijote N-body validation of DataBin method
=======================================================
在 quijote 1Gpc N-body simulation 的不同 fnl 上验证 DataBin 方法。
fnl = 0, 20, 30, 50, 75, 100

方法对比: DataBin vs FullDiscrete vs ExpWindow(含归一化)

运行: bash -lc "conda activate desilike && PYTHONUNBUFFERED=1 python -u <script>"
"""

from __future__ import annotations
import os, glob, re, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import next_fast_len, rfft, irfft, fht, fhtoffset

OUT = os.path.dirname(os.path.abspath(__file__))

# Quijote: L=1000, kf=2pi/1000
L = 1000.0; kf = 2*np.pi/L; V = L**3
IR_X = 4.0  # x=4*(1000/1000)=4
Z = 1.0; P_FIX = 1.2; KMAX = 15.0

# 数据路径
BASE_DIR = "/pscratch/sd/l/lzy/pks_2pcfs"
INTERP_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission6_log/nbody_validation/interpolated"

# fnl 配置：(label, pk_pattern, pcf_pattern, cov_pk_pattern)
FNL_CONFIGS = {
    0:   ("fid",    f"{BASE_DIR}/pk_fid_*.txt",    f"{BASE_DIR}/pcf_fid_*.dat",    f"{BASE_DIR}/pk_fid_*.txt"),
    20:  ("fnl020", f"{INTERP_DIR}/fnl020/pk_fnl20_*.txt", f"{INTERP_DIR}/fnl020/pcf_fnl20_*.dat", f"{BASE_DIR}/pk_fid_*.txt"),
    30:  ("fnl030", f"{INTERP_DIR}/fnl030/pk_fnl30_*.txt", f"{INTERP_DIR}/fnl030/pcf_fnl30_*.dat", f"{BASE_DIR}/pk_fid_*.txt"),
    50:  ("LCp50",  f"{BASE_DIR}/pk_LCp50_*.txt",  f"{BASE_DIR}/pcf_LCp50_*.dat",  f"{BASE_DIR}/pk_fid_*.txt"),
    75:  ("fnl075", f"{INTERP_DIR}/fnl075/pk_fnl75_*.txt", f"{INTERP_DIR}/fnl075/pcf_fnl75_*.dat", f"{BASE_DIR}/pk_fid_*.txt"),
    100: ("LCp100", f"{BASE_DIR}/pk_LCp100_*.txt", f"{BASE_DIR}/pcf_LCp100_*.dat", f"{BASE_DIR}/pk_fid_*.txt"),
}

# ============================================================
# 工具函数
# ============================================================
def rid(f):
    m = re.search(r'(\d+)\.\w+$', os.path.basename(f))
    return int(m.group(1)) if m else -1

def load_pk_quijote(pat, n_max=500):
    """加载 quijote pk 数据（.txt，7列）"""
    fs = sorted(glob.glob(pat), key=rid)[:n_max]
    if not fs: raise FileNotFoundError(pat)
    ref = np.loadtxt(fs[0], comments='#')
    kcen, kmin_b, kmax_b = ref[:,0], ref[:,1], ref[:,2]
    vals = []
    for f in fs:
        try:
            arr = np.loadtxt(f, comments='#')
            vals.append(arr[:,5])  # P_0 列
        except: pass
    vals = np.array(vals)
    # 跳过空 bin（nmod=0 → P=0）
    ref_nmod = ref[:,4]
    valid = ref_nmod > 0
    return kcen[valid], kmin_b[valid], kmax_b[valid], vals[:,valid], vals[:,valid].mean(0), len(fs)

def load_pcf_quijote(pat, n_max=500):
    """加载 quijote pcf 数据（.dat，4列）"""
    fs = sorted(glob.glob(pat), key=rid)[:n_max]
    if not fs: raise FileNotFoundError(pat)
    ref = np.loadtxt(fs[0], comments='#')
    s = ref[:,0]
    vals = np.array([np.loadtxt(f, comments='#')[:,3] for f in fs[:n_max]])
    return s, vals.mean(0), vals.std(0, ddof=1), len(fs)

def j0(x):
    x=np.asarray(x,float); o=np.ones_like(x); m=x!=0; o[m]=np.sin(x[m])/x[m]; return o

def gq_fft(qmax, nmax):
    a=np.zeros(qmax+1,dtype=np.float32); a[0]=1
    sq=np.arange(1,nmax+1,dtype=np.int64)**2; a[sq[sq<=qmax]]=2
    nfft=next_fast_len(3*qmax+1)
    fa=rfft(a,n=nfft); return np.rint(irfft(fa**3,n=nfft)[:qmax+1]).astype(np.int64)

def xi_disc(s, gq, kf, V, p_func, chunk=500000):
    qnz=np.nonzero(gq[1:])[0]+1; xi=np.zeros(len(s))
    for i0 in range(0,len(qnz),chunk):
        qb=qnz[i0:i0+chunk]; kv=kf*np.sqrt(qb.astype(float))
        pv=p_func(kv); w=gq[qb].astype(float)*pv
        for js in range(0,len(s),10):
            je=min(js+10,len(s)); xi[js:je]+=np.dot(w,j0(np.outer(kv,s[js:je])))
    return xi/V

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

def ir_window_func(k, kf, x):
    norm=1-np.exp(-1); w=np.ones_like(k); m=k<kf; w[m]=(1-np.exp(-(k[m]/kf)**x))/norm; return w

def met(r2d, r2s, r2m):
    ds=(r2d-r2m)/r2s; return np.nanmean(np.abs(ds)), np.nanmean(ds)

# ============================================================
def main():
    T0 = time.time()
    print("="*60)
    print("Quijote N-body Validation: DataBin method")
    print("="*60)

    from cosmoprimo import Cosmology
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler
    from pypower import PowerSpectrumStatistics

    cosmo = Cosmology(h=0.6711, Omega_b=0.049, Omega_cdm=0.3175-0.049,
                      sigma8=0.834, n_s=0.9624, engine='class')

    # 预计算 g_q（所有 fnl 共用）
    qmax = int((KMAX/kf)**2); nmax = int(KMAX/kf)
    print(f"g_q: qmax={qmax}...")
    gq = gq_fft(qmax, nmax)
    print(f"  非零 shell: {np.count_nonzero(gq[1:])}")

    # 结果收集
    all_results = []

    for fnl_true, (label, pk_pat, pcf_pat, cov_pat) in sorted(FNL_CONFIGS.items()):
        print(f"\n{'='*50}")
        print(f"  fnl = {fnl_true} ({label})")
        print(f"{'='*50}")

        # 数据
        try:
            kcen, kmin_b, kmax_b, pk_mocks, pk_mean, npk = load_pk_quijote(pk_pat)
            s, xi_m, xi_s, npcf = load_pcf_quijote(pcf_pat)
        except FileNotFoundError as e:
            print(f"  跳过: {e}")
            continue

        # 限制拟合范围到前 20 个 bin (k~0.06)，避免高 k 模型失效
        FIT_NDP = min(20, len(kcen))
        ndp = len(kcen)  # DataBin 用全部 bin
        r2d = s**2*xi_m; r2s = s**2*xi_s
        print(f"  pk: {npk} files, {ndp} bins, k=[{kcen[0]:.5f},{kcen[-1]:.5f}]")
        print(f"  pcf: {npcf} files, s=[{s[0]:.0f},{s[-1]:.0f}]")

        # Covariance (用 fnl=0 的 500 个)
        _, _, _, cov_mocks, _, ncov = load_pk_quijote(cov_pat)
        print(f"  cov: {ncov} files")

        # 拟合（只用前 FIT_NDP 个 bin，避免高 k 模型失效）
        print(f"  拟合 (前 {FIT_NDP} bins, k<={kcen[FIT_NDP-1]:.4f})...")
        fit_kcen = kcen[:FIT_NDP]; fit_kmin = kmin_b[:FIT_NDP]; fit_kmax = kmax_b[:FIT_NDP]
        fit_pk_mean = pk_mean[:FIT_NDP]
        fit_cov = cov_mocks[:, :FIT_NDP]
        edges = np.concatenate([fit_kmin, [fit_kmax[-1]]])
        nm = 4/3*np.pi*(edges[1:]**3 - edges[:-1]**3)
        dps = PowerSpectrumStatistics(edges=edges, modes=fit_kcen,
            power_nonorm=np.array([fit_pk_mean]), nmodes=nm, ells=[0],
            shotnoise_nonorm=0, statistic='multipole')
        ml = []
        for i in range(min(fit_cov.shape[0], 500)):
            t = dps.deepcopy()
            t.power_nonorm.flat[...] = np.array([fit_cov[i]]).ravel()
            ml.append(t)

        tmpl = FixedPowerSpectrumTemplate(z=Z, fiducial=cosmo)
        th = PNGTracerPowerSpectrumMultipoles(template=tmpl, mode='b-p')
        th.init.params['p'].update(fixed=True, value=P_FIX)
        th.init.params['sn0'].update(fixed=True, value=0)
        th.init.params['sigmas'].update(fixed=False, value=0)
        obs = TracerPowerSpectrumMultipolesObservable(data=dps, covariance=ml,
            klim={0:[float(fit_kcen.min()),float(fit_kcen.max()),float(fit_kcen[1]-fit_kcen[0])]}, theory=th)
        like = ObservablesGaussianLikelihood(observables=[obs]); _=like()
        like.all_params['p'].update(fixed=True, value=P_FIX)
        like.all_params['sn0'].update(fixed=True, value=0)
        like.all_params['sigmas'].update(fixed=False, value=0)
        prof = MinuitProfiler(like, seed=66)
        pr = prof.maximize(niterations=27)
        bf = pr.bestfit.choice(input=True)
        fnl_fit = float(bf['fnl_loc']); b1_fit = float(bf['b1']); sig_fit = float(bf['sigmas'])
        print(f"  best-fit: fnl={fnl_fit:.2f}, b1={b1_fit:.4f}, sig={sig_fit:.2f}")

        # P(k) 密集网格
        kd = np.geomspace(kf*0.5, KMAX*1.1, 300000)
        tmpl2 = FixedPowerSpectrumTemplate(z=Z, fiducial=cosmo)
        th2 = PNGTracerPowerSpectrumMultipoles(k=kd, template=tmpl2, mode='b-p')
        th2.init.params['p'].update(fixed=True, value=P_FIX)
        th2.init.params['sn0'].update(fixed=True, value=0)
        th2.init.params['sigmas'].update(fixed=False, value=0)
        th2(**bf); pd = np.array(th2.power[0], dtype=float)

        # P 函数
        p_model = lambda k: np.interp(k, kd, pd)
        def make_p_databin(n_bins_data=None):
            """DataBin: 前 n_bins_data 个 bin 用测量值，其余用模型"""
            nb = n_bins_data if n_bins_data is not None else ndp
            def p(k):
                p_out = np.interp(k, kd, pd)
                for ib in range(min(nb, ndp)):
                    mask = (k >= kmin_b[ib]) & (k < kmax_b[ib])
                    p_out[mask] = pk_mean[ib]
                return p_out
            return p

        # 计算 xi
        print("  FullDiscrete...")
        xi_fd = xi_disc(s, gq, kf, V, p_model)
        print("  DataBin (all bins)...")
        xi_db = xi_disc(s, gq, kf, V, make_p_databin())
        # DataBin limited: 只替换前 20 个 bin
        print("  DataBin (20 bins)...")
        xi_db20 = xi_disc(s, gq, kf, V, make_p_databin(20))

        # ExpWindow
        print("  ExpWindow...")
        kgw = np.geomspace(1e-4/4, KMAX*4, 4096)
        tmpl3 = FixedPowerSpectrumTemplate(z=Z, fiducial=cosmo)
        th3 = PNGTracerPowerSpectrumMultipoles(k=kgw, template=tmpl3, mode='b-p')
        th3.init.params['p'].update(fixed=True, value=P_FIX)
        th3.init.params['sn0'].update(fixed=True, value=0)
        th3.init.params['sigmas'].update(fixed=False, value=0)
        th3(**bf); pgw = np.array(th3.power[0])
        iw = ir_window_func(kgw, kf, IR_X)
        sw, xw = xi_fftlog(kgw, pgw, 1e-4, KMAX, ir_win=iw)
        xi_ew = np.interp(s, np.sort(sw), xw[np.argsort(sw)])

        # 指标
        ma_fd, ms_fd = met(r2d, r2s, s**2*xi_fd)
        ma_db, ms_db = met(r2d, r2s, s**2*xi_db)
        ma_db20, ms_db20 = met(r2d, r2s, s**2*xi_db20)
        ma_ew, ms_ew = met(r2d, r2s, s**2*xi_ew)

        print(f"\n  {'方法':<20s} {'mean|Δ/σ|':>10s} {'mean(Δ/σ)':>10s}")
        print(f"  {'-'*44}")
        print(f"  {'FullDiscrete':<20s} {ma_fd:>10.4f} {ms_fd:>+10.4f}")
        print(f"  {'DataBin(all)':<20s} {ma_db:>10.4f} {ms_db:>+10.4f}")
        print(f"  {'DataBin(20bins)':<20s} {ma_db20:>10.4f} {ms_db20:>+10.4f}")
        print(f"  {'ExpWindow':<20s} {ma_ew:>10.4f} {ms_ew:>+10.4f}")

        all_results.append({
            "fnl_true": fnl_true, "fnl_fit": fnl_fit, "b1": b1_fit,
            "fd_ma": ma_fd, "fd_ms": ms_fd,
            "db_ma": ma_db, "db_ms": ms_db,
            "db20_ma": ma_db20, "db20_ms": ms_db20,
            "ew_ma": ma_ew, "ew_ms": ms_ew,
        })

    # 汇总
    print(f"\n{'='*70}")
    print("汇总表")
    print(f"{'='*70}")
    print(f"{'fnl':>4s} {'fit':>6s} | {'FD|Δ/σ|':>8s} {'FDΔ/σ':>7s} | {'DB|Δ/σ|':>8s} {'DBΔ/σ':>7s} | {'DB20|Δ/σ|':>9s} {'DB20Δ/σ':>8s} | {'EW|Δ/σ|':>8s} {'EWΔ/σ':>7s}")
    print("-"*100)
    for r in all_results:
        print(f"{r['fnl_true']:>4d} {r['fnl_fit']:>6.1f} | "
              f"{r['fd_ma']:>8.4f} {r['fd_ms']:>+7.4f} | "
              f"{r['db_ma']:>8.4f} {r['db_ms']:>+7.4f} | "
              f"{r['db20_ma']:>9.4f} {r['db20_ms']:>+8.4f} | "
              f"{r['ew_ma']:>8.4f} {r['ew_ms']:>+7.4f}")

    # 画图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Quijote N-body Validation: DataBin vs FullDiscrete vs ExpWindow", fontsize=13)

    fnls = [r['fnl_true'] for r in all_results]
    fd_ma = [r['fd_ma'] for r in all_results]
    db_ma = [r['db_ma'] for r in all_results]
    db20_ma = [r['db20_ma'] for r in all_results]
    ew_ma = [r['ew_ma'] for r in all_results]
    fd_ms = [r['fd_ms'] for r in all_results]
    db_ms = [r['db_ms'] for r in all_results]
    db20_ms = [r['db20_ms'] for r in all_results]
    ew_ms = [r['ew_ms'] for r in all_results]

    ax1.plot(fnls, fd_ma, 'go-', ms=7, lw=2, label='FullDiscrete')
    ax1.plot(fnls, db_ma, 'o--', color='darkorange', ms=5, lw=1, label='DataBin(all)', alpha=0.5)
    ax1.plot(fnls, db20_ma, 'o-', color='darkorange', ms=7, lw=2, label='DataBin(20bins)')
    ax1.plot(fnls, ew_ma, 'rs-', ms=7, lw=2, label='ExpWindow')
    ax1.set_xlabel(r'$f_{\rm NL}^{\rm true}$'); ax1.set_ylabel(r'mean$|\Delta/\sigma|$')
    ax1.legend(); ax1.grid(alpha=0.3); ax1.set_title('Alignment metric')

    ax2.plot(fnls, fd_ms, 'go-', ms=7, lw=2, label='FullDiscrete')
    ax2.plot(fnls, db_ms, 'o--', color='darkorange', ms=5, lw=1, label='DataBin(all)', alpha=0.5)
    ax2.plot(fnls, db20_ms, 'o-', color='darkorange', ms=7, lw=2, label='DataBin(20bins)')
    ax2.plot(fnls, ew_ms, 'rs-', ms=7, lw=2, label='ExpWindow')
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.set_xlabel(r'$f_{\rm NL}^{\rm true}$'); ax2.set_ylabel(r'mean$(\Delta/\sigma)$')
    ax2.legend(); ax2.grid(alpha=0.3); ax2.set_title('Systematic bias')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "mission9_quijote_validation.png"), dpi=150)
    plt.close()

    # MD
    md = ["# Quijote N-body Validation\n\n"]
    md.append("| fnl_true | fnl_fit | b1 | FD mean\\|Δ/σ\\| | FD mean(Δ/σ) | DB mean\\|Δ/σ\\| | DB mean(Δ/σ) | EW mean\\|Δ/σ\\| | EW mean(Δ/σ) |\n")
    md.append("|---|---|---|---|---|---|---|---|---|\n")
    for r in all_results:
        md.append(f"| {r['fnl_true']} | {r['fnl_fit']:.1f} | {r['b1']:.3f} | "
                  f"{r['fd_ma']:.4f} | {r['fd_ms']:+.4f} | "
                  f"{r['db_ma']:.4f} | {r['db_ms']:+.4f} | "
                  f"{r['ew_ma']:.4f} | {r['ew_ms']:+.4f} |\n")
    with open(os.path.join(OUT, "mission9_quijote_validation.md"), 'w') as f:
        f.writelines(md)

    print(f"\n完成! {time.time()-T0:.0f}s")

if __name__ == "__main__":
    main()
