#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission 9 探索：全离散高估原因诊断 + 通用校正方案（3Gpc fnl100）
================================================================

性能优化版：
- gq_cont 计算完全向量化
- 用 PYTHONUNBUFFERED=1 实时输出
- 离散求和分块更大以减少循环次数

要求：方法必须同时适用于不同 fnl 和 Lbox (1Gpc, 3Gpc)

运行: PYTHONUNBUFFERED=1 conda activate desilike && python mission9_exploration_3gpc.py
"""

from __future__ import annotations
import os, sys, glob, re, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import fht, fhtoffset, next_fast_len, rfft, irfft

# 强制无缓冲输出
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# ============================================================
# 常数
# ============================================================
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BOX_SIZE = 3000.0
K_FUND = 2.0 * np.pi / BOX_SIZE
VOLUME = BOX_SIZE ** 3
KMAX_INT = 15.0
KMIN_GLOBAL = 1e-4
QMAX = int(np.floor((KMAX_INT / K_FUND) ** 2))
NMAX_FFT = int(np.floor(KMAX_INT / K_FUND))
IR_WINDOW_X = 12.0
FFTLOG_N = 4096
FFTLOG_PADDING = 4.0
EDGE_TAPER_FRAC = 0.06

PK_DATA_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat"
PK_COV_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat"
PCF_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat"
RID_MIN, RID_MAX = 2, 99
N_DATAPOINTS = 20
UNIT_Z = 1.0
FIXED_P = 1.2
N_DENSE = 300000

# ============================================================
# 工具函数
# ============================================================
def parse_rid(fp):
    m = re.search(r'N([0-9]+)', os.path.basename(fp))
    return int(m.group(1)) if m else -1

def load_pk(pattern, rid_min, rid_max, n_dp, pcol=5):
    fs = sorted(glob.glob(pattern), key=parse_rid)
    fs = [f for f in fs if rid_min <= parse_rid(f) <= rid_max]
    ref = np.loadtxt(fs[0], comments='#')
    kcen, kmin, kmax = ref[:n_dp, 0], ref[:n_dp, 1], ref[:n_dp, 2]
    vals = np.array([np.loadtxt(f, comments='#')[:n_dp, pcol] for f in fs])
    return kcen, kmin, kmax, vals, vals.mean(0), vals.std(0, ddof=1)

def load_pcf(pattern, rid_min, rid_max, xi_col=3):
    fs = sorted(glob.glob(pattern), key=parse_rid)
    fs = [f for f in fs if rid_min <= parse_rid(f) <= rid_max]
    ref = np.loadtxt(fs[0], comments='#')
    s = ref[:, 0]
    vals = np.array([np.loadtxt(f, comments='#')[:, xi_col] for f in fs])
    return s, vals, vals.mean(0), vals.std(0, ddof=1)

def j0(x):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    m = x != 0
    out[m] = np.sin(x[m]) / x[m]
    return out

def compute_gq(qmax, nmax):
    """FFT 卷积计算 g_q"""
    a = np.zeros(qmax + 1, dtype=np.float32)
    a[0] = 1.0
    sq = np.arange(1, nmax + 1, dtype=np.int64) ** 2
    sq = sq[sq <= qmax]
    a[sq] = 2.0
    nfft = next_fast_len(3 * qmax + 1)
    print(f"  FFT: qmax={qmax}, nfft={nfft} ({nfft*4/1e9:.2f} GB)")
    t0 = time.time()
    fa = rfft(a, n=nfft)
    conv = irfft(fa * fa * fa, n=nfft)
    g = np.rint(conv[:qmax + 1]).astype(np.int64)
    print(f"  FFT 完成，耗时 {time.time()-t0:.1f}s")
    return g

def compute_gq_cont_vectorized(gq, kf):
    """
    向量化计算连续模式密度权重 g_q^cont。
    g_cont(q) = 4πk_q² Δk / kf³，Δk 用 Voronoi cell width。

    参数: gq = g_q 数组, kf = 基模
    返回: gq_cont 数组（与 gq 同长）
    """
    t0 = time.time()
    q_nz = np.nonzero(gq[1:])[0] + 1  # 非零 q 索引
    k_nz = kf * np.sqrt(q_nz.astype(np.float64))
    n = len(q_nz)

    # Voronoi cell width: dk_i = (k_{i+1} - k_{i-1}) / 2
    # 边界处理：第一个 shell 左边界为 0，最后一个 shell 用对称外推
    dk = np.empty(n, dtype=np.float64)
    # 中间部分
    dk[1:-1] = 0.5 * (k_nz[2:] - k_nz[:-2])
    # 第一个 shell: 左边界 = 0
    dk[0] = 0.5 * (k_nz[0] + k_nz[1])
    # 最后一个 shell
    dk[-1] = k_nz[-1] - 0.5 * (k_nz[-2] + k_nz[-1])

    g_cont_vals = 4 * np.pi * k_nz**2 * dk / kf**3

    # 放回完整数组
    gq_cont = np.zeros(len(gq), dtype=np.float64)
    gq_cont[q_nz] = g_cont_vals
    print(f"  gq_cont 向量化计算完成，耗时 {time.time()-t0:.1f}s")
    return gq_cont

def xi_discrete(s, gq, k_dense, p_dense, box, kf, chunk=500000):
    """全离散 shell 求和 xi_0(r)，通过插值 P(k)"""
    V = box ** 3
    xi = np.zeros(len(s), dtype=np.float64)
    q_nz = np.nonzero(gq[1:])[0] + 1
    total = q_nz.size
    t0 = time.time()
    for i0 in range(0, total, chunk):
        i1 = min(i0 + chunk, total)
        qb = q_nz[i0:i1]
        kv = kf * np.sqrt(qb.astype(np.float64))
        pv = np.interp(kv, k_dense, p_dense)
        w = gq[qb].astype(np.float64) * pv
        # 分批计算 j0 避免大矩阵
        for js in range(0, len(s), 10):
            je = min(js + 10, len(s))
            kr = np.outer(kv, s[js:je])
            xi[js:je] += np.dot(w, j0(kr))
        if (i0 // chunk) % 20 == 0:
            print(f"    sum: {i1}/{total} ({time.time()-t0:.0f}s)")
    return xi / V

def xi_fftlog(k_grid, p0, kmin, kmax, ir_win=None):
    """FFTLog 计算 xi_0"""
    lk = np.log(k_grid)
    l0, l1 = np.log(kmin), np.log(kmax)
    dl = EDGE_TAPER_FRAC * (l1 - l0)
    w = np.zeros_like(k_grid)
    if dl > 0:
        left, right = l0 + dl, l1 - dl
        m = (lk >= l0) & (lk < left); w[m] = 0.5*(1-np.cos(np.pi*(lk[m]-l0)/dl))
        m = (lk >= left) & (lk <= right); w[m] = 1.0
        m = (lk > right) & (lk <= l1); w[m] = 0.5*(1+np.cos(np.pi*(lk[m]-right)/dl))
    else:
        w[(k_grid >= kmin) & (k_grid <= kmax)] = 1.0
    if ir_win is not None:
        w *= ir_win
    dln = np.log(k_grid[1] / k_grid[0])
    offset = fhtoffset(dln, mu=0.5, initial=0.0, bias=0.0)
    A = fht((k_grid**1.5) * p0 * w, dln=dln, mu=0.5, offset=offset, bias=0.0)
    n = k_grid.size
    jj = np.arange(n)
    s_grid = np.exp(-(np.log(k_grid[0])+np.log(k_grid[-1]))/2 + offset + (jj - (n-1)/2)*dln)
    return s_grid, np.sqrt(np.pi/2)/(2*np.pi**2) * A / s_grid**1.5

def ir_window(k, kf, x):
    w = np.ones_like(k)
    m = k < kf
    w[m] = 1 - np.exp(-(k[m]/kf)**x)
    return w

def metrics(r2_data, r2_std, r2_model):
    ds = (r2_data - r2_model) / r2_std
    return np.nanmean(np.abs(ds)), np.nanmean(ds), np.nansum(ds**2)/len(ds)

# ============================================================
# 主程序
# ============================================================
def main():
    T0 = time.time()
    print("=" * 60)
    print("Mission 9: 全离散高估诊断 (3Gpc fnl100, kmax=15)")
    print("=" * 60)

    # 1. 读数据
    print("\n[1] 读取数据...")
    kcen, kmin, kmax_bins, pk_mocks, pk_mean, pk_std = load_pk(PK_DATA_GLOB, RID_MIN, RID_MAX, N_DATAPOINTS)
    _, _, _, cov_mocks, _, _ = load_pk(PK_COV_GLOB, RID_MIN, RID_MAX, N_DATAPOINTS)
    s_data, pcf_mocks, xi_mean, xi_std = load_pcf(PCF_GLOB, RID_MIN, RID_MAX)
    r2_data = s_data**2 * xi_mean
    r2_std = s_data**2 * xi_std
    print(f"  pk: {pk_mocks.shape[0]} mocks, pcf: {pcf_mocks.shape[0]} mocks")

    # 2. desilike 拟合
    print("\n[2] desilike 拟合...")
    from cosmoprimo import Cosmology
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler
    from pypower import PowerSpectrumStatistics

    cosmo = Cosmology(h=0.6711, Omega_b=0.049, Omega_cdm=0.3175-0.049,
                      sigma8=0.834, n_s=0.9624, engine='class')
    edges = np.concatenate([kmin, [kmax_bins[-1]]])
    nmodes = 4/3*np.pi*(edges[1:]**3 - edges[:-1]**3)
    data_ps = PowerSpectrumStatistics(edges=edges, modes=kcen,
        power_nonorm=np.array([pk_mean]), nmodes=nmodes, ells=[0],
        shotnoise_nonorm=0.0, statistic='multipole')
    mock_list = []
    for i in range(cov_mocks.shape[0]):
        tmp = data_ps.deepcopy()
        tmp.power_nonorm.flat[...] = np.array([cov_mocks[i]]).ravel()
        mock_list.append(tmp)

    template = FixedPowerSpectrumTemplate(z=UNIT_Z, fiducial=cosmo)
    theory = PNGTracerPowerSpectrumMultipoles(template=template, mode='b-p')
    theory.init.params['p'].update(fixed=True, value=FIXED_P)
    theory.init.params['sn0'].update(fixed=True, value=0.0)
    theory.init.params['sigmas'].update(fixed=False, value=0.0)
    obs = TracerPowerSpectrumMultipolesObservable(data=data_ps, covariance=mock_list,
        klim={0: [float(kcen.min()), float(kcen.max()), float(kcen[1]-kcen[0])]}, theory=theory)
    like = ObservablesGaussianLikelihood(observables=[obs])
    _ = like()
    like.all_params['p'].update(fixed=True, value=FIXED_P)
    like.all_params['sn0'].update(fixed=True, value=0.0)
    like.all_params['sigmas'].update(fixed=False, value=0.0)
    prof = MinuitProfiler(like, seed=66)
    profiles = prof.maximize(niterations=27)
    bestfit = profiles.bestfit.choice(input=True)
    fnl_val = float(bestfit['fnl_loc'])
    b1_val = float(bestfit['b1'])
    sig_val = float(bestfit['sigmas'])
    print(f"  Best-fit: fnl={fnl_val:.2f}, b1={b1_val:.4f}, sigmas={sig_val:.2f}")
    print(f"  拟合耗时: {time.time()-T0:.0f}s")

    # 3. 密集 k 网格 P0(k)
    print("\n[3] P0(k) 密集网格...")
    k_dense = np.geomspace(K_FUND * 0.5, KMAX_INT * 1.1, N_DENSE)
    tmpl = FixedPowerSpectrumTemplate(z=UNIT_Z, fiducial=cosmo)
    th = PNGTracerPowerSpectrumMultipoles(k=k_dense, template=tmpl, mode='b-p')
    th.init.params['p'].update(fixed=True, value=FIXED_P)
    th.init.params['sn0'].update(fixed=True, value=0.0)
    th.init.params['sigmas'].update(fixed=False, value=0.0)
    th(**bestfit)
    p_dense = np.array(th.power[0], dtype=np.float64)
    print(f"  P range: [{p_dense.min():.0f}, {p_dense.max():.0f}]")

    # 4. g_q
    print("\n[4] 计算 g_q (FFT)...")
    gq = compute_gq(QMAX, NMAX_FFT)

    # 5. 连续权重（向量化）
    print("\n[5] 连续权重 g_q^cont...")
    gq_cont = compute_gq_cont_vectorized(gq, K_FUND)

    # 打印前 10 shell 对比
    q_nz = np.nonzero(gq[1:])[0] + 1
    print(f"  非零 shell 数: {q_nz.size}")
    print(f"  {'q':>5} {'k/kf':>8} {'g_q':>8} {'g_cont':>10} {'ratio':>8}")
    for i in range(min(10, len(q_nz))):
        q = q_nz[i]
        ratio = gq[q] / gq_cont[q] if gq_cont[q] > 0 else np.nan
        print(f"  {q:>5} {np.sqrt(q):>8.3f} {gq[q]:>8} {gq_cont[q]:>10.2f} {ratio:>8.3f}")

    # 6. 多方法计算
    print("\n[6] 计算 xi_0...")

    # A. Baseline
    print("  A. Baseline...")
    kg_base = np.geomspace(K_FUND / FFTLOG_PADDING, KMAX_INT * FFTLOG_PADDING, FFTLOG_N)
    tmpl2 = FixedPowerSpectrumTemplate(z=UNIT_Z, fiducial=cosmo)
    th2 = PNGTracerPowerSpectrumMultipoles(k=kg_base, template=tmpl2, mode='b-p')
    th2.init.params['p'].update(fixed=True, value=FIXED_P)
    th2.init.params['sn0'].update(fixed=True, value=0.0)
    th2.init.params['sigmas'].update(fixed=False, value=0.0)
    th2(**bestfit)
    p_base = np.array(th2.power[0])
    s_base, xi_base_raw = xi_fftlog(kg_base, p_base, K_FUND, KMAX_INT)
    xi_base = np.interp(s_data, np.sort(s_base), xi_base_raw[np.argsort(s_base)])

    # B. ExpWindow
    print("  B. ExpWindow...")
    kg_win = np.geomspace(KMIN_GLOBAL / FFTLOG_PADDING, KMAX_INT * FFTLOG_PADDING, FFTLOG_N)
    tmpl3 = FixedPowerSpectrumTemplate(z=UNIT_Z, fiducial=cosmo)
    th3 = PNGTracerPowerSpectrumMultipoles(k=kg_win, template=tmpl3, mode='b-p')
    th3.init.params['p'].update(fixed=True, value=FIXED_P)
    th3.init.params['sn0'].update(fixed=True, value=0.0)
    th3.init.params['sigmas'].update(fixed=False, value=0.0)
    th3(**bestfit)
    p_win = np.array(th3.power[0])
    iw = ir_window(kg_win, K_FUND, IR_WINDOW_X)
    s_win, xi_win_raw = xi_fftlog(kg_win, p_win, KMIN_GLOBAL, KMAX_INT, ir_win=iw)
    xi_win = np.interp(s_data, np.sort(s_win), xi_win_raw[np.argsort(s_win)])

    # C. FullDiscrete
    print("  C. FullDiscrete...")
    xi_disc = xi_discrete(s_data, gq, k_dense, p_dense, BOX_SIZE, K_FUND)

    # D. ContWeight (全部用连续权重)
    print("  D. ContWeight...")
    xi_contw = xi_discrete(s_data, gq_cont, k_dense, p_dense, BOX_SIZE, K_FUND)

    # E. Q_CORR 扫描：低 q 用连续权重，高 q 用原始 g_q
    q_corr_list = [10, 50, 100, 500, 2000]
    xi_qcorr = {}
    for qc in q_corr_list:
        print(f"  E.QCorr{qc}...")
        gq_mixed = gq.astype(np.float64).copy()
        mask = np.arange(len(gq)) <= qc
        gq_mixed[mask] = gq_cont[mask]
        xi_qcorr[qc] = xi_discrete(s_data, gq_mixed, k_dense, p_dense, BOX_SIZE, K_FUND)

    # 7. 指标
    print(f"\n[7] 指标 ({time.time()-T0:.0f}s elapsed):")
    all_r2 = {}
    all_r2["A.Baseline"] = s_data**2 * xi_base
    all_r2["B.ExpWindow"] = s_data**2 * xi_win
    all_r2["C.FullDiscrete"] = s_data**2 * xi_disc
    all_r2["D.ContWeight"] = s_data**2 * xi_contw
    for qc in q_corr_list:
        all_r2[f"E.QCorr{qc}"] = s_data**2 * xi_qcorr[qc]

    print(f"\n{'方法':<22s} {'mean|Δ/σ|':>10s} {'mean(Δ/σ)':>10s} {'χ²/ndof':>10s}")
    print("-" * 56)
    table = []
    for name, r2m in all_r2.items():
        ma, ms, chi = metrics(r2_data, r2_std, r2m)
        table.append((name, ma, ms, chi))
        print(f"{name:<22s} {ma:>10.4f} {ms:>+10.4f} {chi:>10.4f}")

    # 8. 画图
    print("\n[8] 画图...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11))
    fig.suptitle(f"3Gpc fnl100: kmax={KMAX_INT}, fnl={fnl_val:.1f}, b1={b1_val:.3f}", fontsize=13)
    ax1.errorbar(s_data, r2_data, yerr=r2_std, fmt='ko', ms=3, capsize=2, label='Data')
    for nm, c, ls in [("A.Baseline",'blue','--'),("B.ExpWindow",'red','-'),
                       ("C.FullDiscrete",'green','-'),("D.ContWeight",'orange','-.')]:
        ax1.plot(s_data, all_r2[nm], color=c, ls=ls, lw=1.5, label=nm)
    ax1.set_ylabel(r'$r^2\xi_0$'); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    ax2.axhline(0, color='k', lw=0.5)
    for nm, c, ls in [("A.Baseline",'blue','--'),("B.ExpWindow",'red','-'),
                       ("C.FullDiscrete",'green','-'),("D.ContWeight",'orange','-.')]:
        ds = (r2_data - all_r2[nm]) / r2_std
        ax2.plot(s_data, ds, color=c, ls=ls, lw=1.3, marker='o', ms=2, label=nm)
    ax2.set_xlabel(r'$r$ [Mpc/h]'); ax2.set_ylabel(r'$(Data-Model)/\sigma$')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "mission9_methods_3gpc.png"), dpi=150); plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.errorbar(s_data, r2_data, yerr=r2_std, fmt='ko', ms=2, capsize=1, label='Data')
    ax1.plot(s_data, all_r2["B.ExpWindow"], 'r-', lw=2, label='ExpWindow', zorder=10)
    ax1.plot(s_data, all_r2["C.FullDiscrete"], 'g--', lw=1.5, label='FullDiscrete')
    for qc in q_corr_list:
        ax1.plot(s_data, all_r2[f"E.QCorr{qc}"], '--', lw=1, label=f"QCorr{qc}")
    ax1.set_ylabel(r'$r^2\xi_0$'); ax1.legend(fontsize=7, ncol=2); ax1.grid(alpha=0.3)

    qc_arr = np.array(q_corr_list, dtype=float)
    ma_arr = np.array([next(t[1] for t in table if t[0]==f"E.QCorr{qc}") for qc in q_corr_list])
    disc_ma = next(t[1] for t in table if t[0]=="C.FullDiscrete")
    win_ma = next(t[1] for t in table if t[0]=="B.ExpWindow")
    ax2.plot(qc_arr, ma_arr, 'o-', lw=2, color='steelblue', ms=6)
    ax2.axhline(disc_ma, color='g', ls='--', label=f"FullDisc ({disc_ma:.4f})")
    ax2.axhline(win_ma, color='r', ls='--', label=f"ExpWin ({win_ma:.4f})")
    ax2.set_xscale('log'); ax2.set_xlabel("Q_CORR"); ax2.set_ylabel("mean |Δ/σ|")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "mission9_qcorr_3gpc.png"), dpi=150); plt.close()

    # shell 贡献图
    nshow = min(30, len(q_nz))
    r_pick = 200.0
    c_disc = np.zeros(nshow); c_cont = np.zeros(nshow)
    for i in range(nshow):
        q = q_nz[i]; kq = K_FUND*np.sqrt(q)
        pq = np.interp(kq, k_dense, p_dense)
        c_disc[i] = gq[q] * pq * j0(kq*r_pick) / VOLUME
        c_cont[i] = gq_cont[q] * pq * j0(kq*r_pick) / VOLUME
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    x = np.sqrt(q_nz[:nshow].astype(float))
    ax1.bar(x-0.03, r_pick**2*c_disc, width=0.06, color='green', alpha=0.7, label='Discrete g_q')
    ax1.bar(x+0.03, r_pick**2*c_cont, width=0.06, color='orange', alpha=0.7, label='Continuous g_cont')
    ax1.set_xlabel(r'$k/k_f$'); ax1.set_ylabel(r'$r^2\times$ contribution')
    ax1.set_title(f'Per-shell contribution (r={r_pick:.0f})'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(x, np.cumsum(c_disc-c_cont)*r_pick**2, 'k-', lw=2)
    ax2.set_xlabel(r'$k/k_f$'); ax2.set_ylabel(r'$r^2\times$ cumulative excess')
    ax2.set_title('Cumulative (discrete - continuous)'); ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "mission9_shell_contrib_3gpc.png"), dpi=150); plt.close()

    # 保存结论
    md = [f"# Mission 9 (3Gpc fnl100, kmax={KMAX_INT})\n\n"]
    md.append(f"Best-fit: fnl={fnl_val:.2f}, b1={b1_val:.4f}, sigmas={sig_val:.2f}\n\n")
    md.append("## 方法对比\n\n| 方法 | mean\\|Δ/σ\\| | mean(Δ/σ) | χ²/ndof |\n|---|---|---|---|\n")
    for n, ma, ms, chi in table:
        md.append(f"| {n} | {ma:.4f} | {ms:+.4f} | {chi:.4f} |\n")
    md.append(f"\n总耗时: {time.time()-T0:.0f}s\n")
    with open(os.path.join(OUT_DIR, "mission9_exploration_3gpc.md"), 'w') as f:
        f.writelines(md)

    print(f"\n完成! 总耗时: {time.time()-T0:.0f}s")

if __name__ == "__main__":
    main()
