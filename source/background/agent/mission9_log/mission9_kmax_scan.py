#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission 9: kmax 扫描 + 1Gpc 对比
==================================
测试 FullDiscrete 的 bias 如何随 kmax 变化。
同时在 3Gpc fnl100 和 1Gpc fnl100 上测试，验证通用性。

运行: bash -lc "conda activate desilike && PYTHONUNBUFFERED=1 python -u <script>"
"""

from __future__ import annotations
import os, sys, glob, re, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import fht, fhtoffset, next_fast_len, rfft, irfft

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 通用参数
# ============================================================
UNIT_Z = 1.0
FIXED_P = 1.2
FFTLOG_N = 4096
FFTLOG_PADDING = 4.0
EDGE_TAPER_FRAC = 0.06
N_DENSE = 300000
KMAX_SCAN = [5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0]

# 两个盒子配置
CONFIGS = {
    "3Gpc_fnl100": {
        "L": 3000.0, "pk_glob": "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat",
        "cov_glob": "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
        "pcf_glob": "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",
        "rid_min": 2, "rid_max": 99, "n_dp": 20, "ir_x": 12.0, "kmin_global": 1e-4,
    },
    "1Gpc_fnl100": {
        "L": 1000.0,
        "pk_glob": "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat",
        "cov_glob": "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
        "pcf_glob": "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat",
        "rid_min": 1, "rid_max": 50, "n_dp": 20, "ir_x": 4.0, "kmin_global": 1e-4,
    },
}

# ============================================================
# 工具（复用）
# ============================================================
def parse_rid(fp):
    m = re.search(r'N([0-9]+)', os.path.basename(fp))
    return int(m.group(1)) if m else -1

def load_pk(pattern, rmin, rmax, ndp, pcol=5):
    fs = sorted(glob.glob(pattern), key=parse_rid)
    fs = [f for f in fs if rmin <= parse_rid(f) <= rmax]
    if not fs: raise FileNotFoundError(pattern)
    ref = np.loadtxt(fs[0], comments='#')
    kcen, kmin, kmax = ref[:ndp, 0], ref[:ndp, 1], ref[:ndp, 2]
    vals = np.array([np.loadtxt(f, comments='#')[:ndp, pcol] for f in fs])
    return kcen, kmin, kmax, vals, vals.mean(0), vals.std(0, ddof=1)

def load_pcf(pattern, rmin, rmax, xi_col=3):
    fs = sorted(glob.glob(pattern), key=parse_rid)
    fs = [f for f in fs if rmin <= parse_rid(f) <= rmax]
    if not fs: raise FileNotFoundError(pattern)
    s = np.loadtxt(fs[0], comments='#')[:, 0]
    vals = np.array([np.loadtxt(f, comments='#')[:, xi_col] for f in fs])
    return s, vals.mean(0), vals.std(0, ddof=1)

def j0(x):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x); m = x != 0; out[m] = np.sin(x[m]) / x[m]; return out

def compute_gq(qmax, nmax):
    a = np.zeros(qmax + 1, dtype=np.float32)
    a[0] = 1.0
    sq = np.arange(1, nmax + 1, dtype=np.int64) ** 2
    a[sq[sq <= qmax]] = 2.0
    nfft = next_fast_len(3 * qmax + 1)
    print(f"    FFT: qmax={qmax}, nfft={nfft}")
    fa = rfft(a, n=nfft)
    return np.rint(irfft(fa**3, n=nfft)[:qmax + 1]).astype(np.int64)

def xi_disc_partial(s, gq, kf, V, k_dense, p_dense, kmax_cut, chunk=500000):
    """全离散求和，截断到 kmax_cut"""
    qmax_cut = int(np.floor((kmax_cut / kf) ** 2))
    q_nz = np.nonzero(gq[1:min(qmax_cut+1, len(gq))])[0] + 1
    xi = np.zeros(len(s), dtype=np.float64)
    for i0 in range(0, len(q_nz), chunk):
        qb = q_nz[i0:i0+chunk]
        kv = kf * np.sqrt(qb.astype(np.float64))
        pv = np.interp(kv, k_dense, p_dense)
        w = gq[qb].astype(np.float64) * pv
        for js in range(0, len(s), 10):
            je = min(js+10, len(s))
            xi[js:je] += np.dot(w, j0(np.outer(kv, s[js:je])))
    return xi / V

def xi_fftlog(k_grid, p0, kmin, kmax, ir_win=None):
    lk = np.log(k_grid); l0, l1 = np.log(kmin), np.log(kmax)
    dl = EDGE_TAPER_FRAC * (l1 - l0)
    w = np.zeros_like(k_grid)
    if dl > 0:
        left, right = l0 + dl, l1 - dl
        m = (lk >= l0) & (lk < left); w[m] = 0.5*(1-np.cos(np.pi*(lk[m]-l0)/dl))
        m = (lk >= left) & (lk <= right); w[m] = 1.0
        m = (lk > right) & (lk <= l1); w[m] = 0.5*(1+np.cos(np.pi*(lk[m]-right)/dl))
    if ir_win is not None: w *= ir_win
    dln = np.log(k_grid[1]/k_grid[0])
    offset = fhtoffset(dln, mu=0.5, initial=0.0, bias=0.0)
    A = fht((k_grid**1.5)*p0*w, dln=dln, mu=0.5, offset=offset, bias=0.0)
    n = k_grid.size; jj = np.arange(n)
    s_grid = np.exp(-(np.log(k_grid[0])+np.log(k_grid[-1]))/2 + offset + (jj-(n-1)/2)*dln)
    return s_grid, np.sqrt(np.pi/2)/(2*np.pi**2)*A/s_grid**1.5

def ir_window(k, kf, x):
    w = np.ones_like(k); m = k < kf; w[m] = 1-np.exp(-(k[m]/kf)**x); return w

def metrics(r2_data, r2_std, r2_model):
    ds = (r2_data - r2_model) / r2_std
    return np.nanmean(np.abs(ds)), np.nanmean(ds)

# ============================================================
def main():
    T0 = time.time()
    print("=" * 60)
    print("Mission 9: kmax 扫描 (3Gpc + 1Gpc, fnl100)")
    print("=" * 60)

    from cosmoprimo import Cosmology
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler
    from pypower import PowerSpectrumStatistics

    cosmo = Cosmology(h=0.6711, Omega_b=0.049, Omega_cdm=0.3175-0.049,
                      sigma8=0.834, n_s=0.9624, engine='class')

    all_results = {}  # {config_name: {method: (ma, ms)}}

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"配置: {cfg_name}")
        print(f"{'='*60}")

        L = cfg["L"]; kf = 2*np.pi/L; V = L**3

        # 1. 数据
        print("[1] 数据...")
        kcen, kmin, kmax_bins, pk_mocks, pk_mean, pk_std = load_pk(
            cfg["pk_glob"], cfg["rid_min"], cfg["rid_max"], cfg["n_dp"])
        _, _, _, cov_mocks, _, _ = load_pk(
            cfg["cov_glob"], cfg["rid_min"], cfg["rid_max"], cfg["n_dp"])
        s, xi_mean, xi_std = load_pcf(
            cfg["pcf_glob"], cfg["rid_min"], cfg["rid_max"])
        r2d = s**2 * xi_mean; r2s = s**2 * xi_std
        print(f"  pk: {pk_mocks.shape[0]} mocks, pcf: s=[{s[0]:.0f},{s[-1]:.0f}]")

        # 2. 拟合
        print("[2] 拟合...")
        edges = np.concatenate([kmin, [kmax_bins[-1]]])
        nmodes = 4/3*np.pi*(edges[1:]**3 - edges[:-1]**3)
        data_ps = PowerSpectrumStatistics(edges=edges, modes=kcen,
            power_nonorm=np.array([pk_mean]), nmodes=nmodes, ells=[0],
            shotnoise_nonorm=0.0, statistic='multipole')
        mlist = []
        for i in range(cov_mocks.shape[0]):
            tmp = data_ps.deepcopy()
            tmp.power_nonorm.flat[...] = np.array([cov_mocks[i]]).ravel()
            mlist.append(tmp)

        tmpl = FixedPowerSpectrumTemplate(z=UNIT_Z, fiducial=cosmo)
        th = PNGTracerPowerSpectrumMultipoles(template=tmpl, mode='b-p')
        th.init.params['p'].update(fixed=True, value=FIXED_P)
        th.init.params['sn0'].update(fixed=True, value=0.0)
        th.init.params['sigmas'].update(fixed=False, value=0.0)
        obs = TracerPowerSpectrumMultipolesObservable(data=data_ps, covariance=mlist,
            klim={0: [float(kcen.min()), float(kcen.max()), float(kcen[1]-kcen[0])]}, theory=th)
        like = ObservablesGaussianLikelihood(observables=[obs])
        _ = like()
        like.all_params['p'].update(fixed=True, value=FIXED_P)
        like.all_params['sn0'].update(fixed=True, value=0.0)
        like.all_params['sigmas'].update(fixed=False, value=0.0)
        prof = MinuitProfiler(like, seed=66)
        profiles = prof.maximize(niterations=27)
        bestfit = profiles.bestfit.choice(input=True)
        fnl = float(bestfit['fnl_loc']); b1 = float(bestfit['b1']); sig = float(bestfit['sigmas'])
        print(f"  fnl={fnl:.2f}, b1={b1:.4f}, sigmas={sig:.2f}")

        # 3. P(k) 密集网格
        print("[3] P(k) 密集网格...")
        kmax_max = max(KMAX_SCAN) * 1.1
        k_dense = np.geomspace(kf * 0.5, kmax_max, N_DENSE)
        tmpl2 = FixedPowerSpectrumTemplate(z=UNIT_Z, fiducial=cosmo)
        th2 = PNGTracerPowerSpectrumMultipoles(k=k_dense, template=tmpl2, mode='b-p')
        th2.init.params['p'].update(fixed=True, value=FIXED_P)
        th2.init.params['sn0'].update(fixed=True, value=0.0)
        th2.init.params['sigmas'].update(fixed=False, value=0.0)
        th2(**bestfit)
        p_dense = np.array(th2.power[0], dtype=np.float64)

        # 4. g_q（用最大 kmax 计算一次）
        qmax = int(np.floor((max(KMAX_SCAN) / kf) ** 2))
        nmax = int(np.floor(max(KMAX_SCAN) / kf))
        print(f"[4] g_q (qmax={qmax})...")
        gq = compute_gq(qmax, nmax)

        # 5. Baseline 和 ExpWindow（用 kmax=15 作为参考）
        print("[5] Baseline + ExpWindow (kmax=15)...")
        kg = np.geomspace(kf/FFTLOG_PADDING, 15.0*FFTLOG_PADDING, FFTLOG_N)
        tmpl3 = FixedPowerSpectrumTemplate(z=UNIT_Z, fiducial=cosmo)
        th3 = PNGTracerPowerSpectrumMultipoles(k=kg, template=tmpl3, mode='b-p')
        th3.init.params['p'].update(fixed=True, value=FIXED_P)
        th3.init.params['sn0'].update(fixed=True, value=0.0)
        th3.init.params['sigmas'].update(fixed=False, value=0.0)
        th3(**bestfit)
        p_fg = np.array(th3.power[0])

        s_bl, xi_bl = xi_fftlog(kg, p_fg, kf, 15.0)
        xi_baseline = np.interp(s, np.sort(s_bl), xi_bl[np.argsort(s_bl)])

        kg_w = np.geomspace(cfg["kmin_global"]/FFTLOG_PADDING, 15.0*FFTLOG_PADDING, FFTLOG_N)
        tmpl4 = FixedPowerSpectrumTemplate(z=UNIT_Z, fiducial=cosmo)
        th4 = PNGTracerPowerSpectrumMultipoles(k=kg_w, template=tmpl4, mode='b-p')
        th4.init.params['p'].update(fixed=True, value=FIXED_P)
        th4.init.params['sn0'].update(fixed=True, value=0.0)
        th4.init.params['sigmas'].update(fixed=False, value=0.0)
        th4(**bestfit)
        p_fw = np.array(th4.power[0])
        iw = ir_window(kg_w, kf, cfg["ir_x"])
        s_wn, xi_wn = xi_fftlog(kg_w, p_fw, cfg["kmin_global"], 15.0, ir_win=iw)
        xi_window = np.interp(s, np.sort(s_wn), xi_wn[np.argsort(s_wn)])

        results = {}
        results["Baseline_15"] = metrics(r2d, r2s, s**2 * xi_baseline)
        results["ExpWindow_15"] = metrics(r2d, r2s, s**2 * xi_window)

        # 6. kmax 扫描 FullDiscrete
        print("[6] kmax 扫描...")
        for kmax_val in KMAX_SCAN:
            label = f"Disc_kmax{kmax_val:.0f}"
            print(f"  {label}...")
            t1 = time.time()
            xi_d = xi_disc_partial(s, gq, kf, V, k_dense, p_dense, kmax_val)
            results[label] = metrics(r2d, r2s, s**2 * xi_d)
            print(f"    {label}: mean|Δ/σ|={results[label][0]:.4f}, "
                  f"mean(Δ/σ)={results[label][1]:+.4f} ({time.time()-t1:.0f}s)")

        # 也测试 ExpWindow 在 kmax=20 的情况
        print("  ExpWindow_20...")
        kg_w2 = np.geomspace(cfg["kmin_global"]/FFTLOG_PADDING, 20.0*FFTLOG_PADDING, FFTLOG_N)
        tmpl5 = FixedPowerSpectrumTemplate(z=UNIT_Z, fiducial=cosmo)
        th5 = PNGTracerPowerSpectrumMultipoles(k=kg_w2, template=tmpl5, mode='b-p')
        th5.init.params['p'].update(fixed=True, value=FIXED_P)
        th5.init.params['sn0'].update(fixed=True, value=0.0)
        th5.init.params['sigmas'].update(fixed=False, value=0.0)
        th5(**bestfit)
        p_fw2 = np.array(th5.power[0])
        iw2 = ir_window(kg_w2, kf, cfg["ir_x"])
        s_w2, xi_w2 = xi_fftlog(kg_w2, p_fw2, cfg["kmin_global"], 20.0, ir_win=iw2)
        xi_win20 = np.interp(s, np.sort(s_w2), xi_w2[np.argsort(s_w2)])
        results["ExpWindow_20"] = metrics(r2d, r2s, s**2 * xi_win20)

        all_results[cfg_name] = results

        # 打印汇总
        print(f"\n--- {cfg_name} 汇总 (fnl={fnl:.1f}, b1={b1:.3f}) ---")
        print(f"{'方法':<20s} {'mean|Δ/σ|':>10s} {'mean(Δ/σ)':>10s}")
        print("-" * 44)
        for nm, (ma, ms) in sorted(results.items()):
            print(f"{nm:<20s} {ma:>10.4f} {ms:>+10.4f}")

    # 7. 汇总画图
    print(f"\n[7] 画图... ({time.time()-T0:.0f}s)")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("FullDiscrete kmax scan: mean(Δ/σ) vs kmax", fontsize=14)

    for idx, (cfg_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        kmax_arr = np.array(KMAX_SCAN)
        ma_arr = np.array([results[f"Disc_kmax{k:.0f}"][0] for k in KMAX_SCAN])
        ms_arr = np.array([results[f"Disc_kmax{k:.0f}"][1] for k in KMAX_SCAN])

        ax.plot(kmax_arr, ms_arr, 'o-', color='green', lw=2, ms=6, label='FullDiscrete: mean(Δ/σ)')
        ax.plot(kmax_arr, ma_arr, 's--', color='blue', lw=1.5, ms=5, label='FullDiscrete: mean|Δ/σ|')

        # 参考线
        ew15 = results.get("ExpWindow_15", (None, None))
        ew20 = results.get("ExpWindow_20", (None, None))
        if ew15[1] is not None:
            ax.axhline(ew15[1], color='red', ls=':', lw=1, label=f"ExpWindow_15: {ew15[1]:+.3f}")
        if ew20[1] is not None:
            ax.axhline(ew20[1], color='darkred', ls='--', lw=1, label=f"ExpWindow_20: {ew20[1]:+.3f}")
        ax.axhline(0, color='gray', ls='-', lw=0.5)

        ax.set_xlabel("kmax [h/Mpc]"); ax.set_ylabel("metric")
        ax.set_title(cfg_name); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "mission9_kmax_scan.png"), dpi=150); plt.close()

    # 保存 MD
    md = ["# Mission 9: kmax 扫描结果\n\n"]
    for cfg_name, results in all_results.items():
        md.append(f"## {cfg_name}\n\n")
        md.append("| 方法 | mean\\|Δ/σ\\| | mean(Δ/σ) |\n|---|---|---|\n")
        for nm, (ma, ms) in sorted(results.items()):
            md.append(f"| {nm} | {ma:.4f} | {ms:+.4f} |\n")
        md.append("\n")
    md.append(f"\n总耗时: {time.time()-T0:.0f}s\n")
    with open(os.path.join(OUT_DIR, "mission9_kmax_scan.md"), 'w') as f:
        f.writelines(md)

    print(f"\n完成! 总耗时: {time.time()-T0:.0f}s")

if __name__ == "__main__":
    main()
