#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission 11: k-rebinning acceleration validation on 1Gpc fastPM + Quijote
=========================================================================

Validate the k-rebinning acceleration method on:
1. 1Gpc fastPM fnl100 and fnl0
2. Quijote N-body (L=1Gpc) with fnl = 0, 20, 30, 50, 75, 100

For each dataset, compare:
- Baseline (original full discrete sum, ~5s for 1Gpc)
- ExactRebin (exact k-rebinning, preserves sum(g*P) per bin)
- CachedRebin (approximate, uses G_bin * P(k_eff))

Covariance:
- 1Gpc fastPM: uses fnl0 fastPM realizations (50 files)
- Quijote: uses fnl=0 (fid) realizations (500 files)

Run:
cd /pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission11_log
bash -lc "conda activate desilike && PYTHONUNBUFFERED=1 python -u mission11_validation_1gpc_quijote.py"
"""

from __future__ import annotations
import os, sys, glob, re, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.fft import next_fast_len, rfft, irfft

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MISSION10_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "mission10_log"))
if MISSION10_DIR not in sys.path:
    sys.path.insert(0, MISSION10_DIR)

from mission10_binavgfit_full_discrete_compare import (
    BoxConfig,
    load_pk_multipoles,
    load_pcf,
    fit_standard_desilike,
    fit_binavg_minuit,
    eval_pk_dense,
    gq_enumerate,
    build_shells_for_kmax,
    build_bin_shell_index,
    gq_fft,
    xi_discrete_multi,
    met_mean_abs_sigma,
)


# ============================================================
# Quijote I/O (from mission10_quijote)
# ============================================================
def _rid_quijote(fp: str) -> int:
    """从 Quijote 文件名提取 realization id: ..._123.txt -> 123"""
    m = re.search(r"(\d+)\.\w+$", os.path.basename(fp))
    return int(m.group(1)) if m else -1


def load_pk_quijote(pat, n_max=500):
    """读取 Quijote pk (7列, col0=kcen col1=kmin col2=kmax col4=nmod col5=P0)"""
    fps = sorted(glob.glob(pat), key=_rid_quijote)[:n_max]
    if not fps:
        raise FileNotFoundError(pat)
    ref = np.loadtxt(fps[0], comments="#")
    nmod = ref[:, 4]
    valid = nmod > 0
    kcen = ref[valid, 0]
    kmin_b = ref[valid, 1]
    kmax_b = ref[valid, 2]
    vals = []
    for f in fps:
        try:
            arr = np.loadtxt(f, comments="#")
            vals.append(arr[valid, 5])
        except Exception:
            continue
    mocks = np.array(vals, dtype=float)
    return kcen, kmin_b, kmax_b, mocks, mocks.mean(axis=0), mocks.std(axis=0, ddof=1), len(mocks)


def load_pcf_quijote(pat, n_max=500):
    """读取 Quijote pcf (4列, col0=s col3=xi0)"""
    fps = sorted(glob.glob(pat), key=_rid_quijote)[:n_max]
    if not fps:
        raise FileNotFoundError(pat)
    ref = np.loadtxt(fps[0], comments="#")
    s = ref[:, 0]
    mocks = np.array([np.loadtxt(f, comments="#")[:, 3] for f in fps], dtype=float)
    return s, mocks.mean(axis=0), mocks.std(axis=0, ddof=1), len(mocks)


# ============================================================
# k-rebinning functions
# ============================================================
def precompute_cache(gq, kf, kmax, dk_factor=0.1):
    """预计算 G_nz 和 k_eff (只依赖 L, kmax)"""
    qnz = np.nonzero(gq[1:])[0] + 1
    kv = kf * np.sqrt(qnz.astype(np.float64))
    g = gq[qnz].astype(np.float64)
    dk = dk_factor * kf
    n_bins = int(np.ceil(kmax / dk)) + 1
    bin_idx = np.clip((kv / dk).astype(np.int64), 0, n_bins - 1)
    G_bin = np.bincount(bin_idx, weights=g, minlength=n_bins)
    Gk_bin = np.bincount(bin_idx, weights=g * kv, minlength=n_bins)
    nz = G_bin > 0
    return G_bin[nz], Gk_bin[nz] / G_bin[nz]


def xi_exact_rebin(s, gq, kf, V, kd, pd, dk_factor=0.1):
    """精确 k-rebinning: 对每个壳层做精确 interp 后聚合"""
    qnz = np.nonzero(gq[1:])[0] + 1
    kv = kf * np.sqrt(qnz.astype(np.float64))
    g = gq[qnz].astype(np.float64)
    pv = np.interp(kv, kd, pd)
    w = g * pv
    dk = dk_factor * kf
    n_bins = int(np.ceil(kv.max() / dk)) + 1
    bin_idx = np.clip((kv / dk).astype(np.int64), 0, n_bins - 1)
    W_bin = np.bincount(bin_idx, weights=w, minlength=n_bins)
    Wk_bin = np.bincount(bin_idx, weights=w * kv, minlength=n_bins)
    nz = W_bin != 0
    W_nz = W_bin[nz]
    k_eff = Wk_bin[nz] / W_nz
    arg = np.outer(k_eff, s)
    J = np.ones_like(arg)
    m = arg != 0
    J[m] = np.sin(arg[m]) / arg[m]
    return (W_nz @ J) / V


def xi_cached_rebin(s, G_nz, k_eff, V, kd, pd):
    """缓存 k-rebinning: 只需 interp(~N_bins 点) + matmul"""
    W = G_nz * np.interp(k_eff, kd, pd)
    arg = np.outer(k_eff, s)
    J = np.ones_like(arg)
    m = arg != 0
    J[m] = np.sin(arg[m]) / arg[m]
    return (W @ J) / V


# ============================================================
# Main
# ============================================================
def main():
    t_total = time.time()

    Z = 1.0; P_FIX = 1.2; KMAX = 15.0; N_DENSE = 300_000

    from cosmoprimo import Cosmology
    cosmo = Cosmology(h=0.6711, Omega_b=0.049, Omega_cdm=0.3175-0.049,
                      sigma8=0.834, n_s=0.9624, engine="class")

    # Quijote 拟合只用前 20 个 k-bin（与 mission9/10 一致）
    QUIJOTE_FIT_NDP = 20

    # ======================================================
    # Part 1: 1Gpc fastPM
    # ======================================================
    fastpm_configs = [
        BoxConfig(title="1Gpc fastPM fnl100", L=1000.0,
            pk_data_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat",
            pk_cov_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
            pcf_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat",
            rid_min=1, rid_max=50, n_dp=20),
        BoxConfig(title="1Gpc fastPM fnl0", L=1000.0,
            pk_data_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
            pk_cov_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
            pcf_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut_fnl0/pcf_rsd_N*.dat",
            rid_min=1, rid_max=50, n_dp=20),
    ]

    # Quijote configs (all L=1Gpc)
    BASE_DIR = "/pscratch/sd/l/lzy/pks_2pcfs"
    INTERP_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission6_log/nbody_validation/interpolated"
    quijote_fnl_configs = {
        0:   ("fid",    f"{BASE_DIR}/pk_fid_*.txt",    f"{BASE_DIR}/pcf_fid_*.dat"),
        20:  ("fnl020", f"{INTERP_DIR}/fnl020/pk_fnl20_*.txt",  f"{INTERP_DIR}/fnl020/pcf_fnl20_*.dat"),
        30:  ("fnl030", f"{INTERP_DIR}/fnl030/pk_fnl30_*.txt",  f"{INTERP_DIR}/fnl030/pcf_fnl30_*.dat"),
        50:  ("LCp50",  f"{BASE_DIR}/pk_LCp50_*.txt",  f"{BASE_DIR}/pcf_LCp50_*.dat"),
        75:  ("fnl075", f"{INTERP_DIR}/fnl075/pk_fnl75_*.txt",  f"{INTERP_DIR}/fnl075/pcf_fnl75_*.dat"),
        100: ("LCp100", f"{BASE_DIR}/pk_LCp100_*.txt", f"{BASE_DIR}/pcf_LCp100_*.dat"),
    }
    quijote_cov_pat = f"{BASE_DIR}/pk_fid_*.txt"

    # Collect all results
    all_results = []

    # g_q and cache for L=1Gpc (shared)
    L_1gpc = 1000.0
    kf_1gpc = 2 * np.pi / L_1gpc
    V_1gpc = L_1gpc ** 3
    kd_1gpc = np.geomspace(kf_1gpc * 0.5, KMAX * 1.1, N_DENSE)

    print("Computing g_q for L=1Gpc ...")
    qmax_1g = int((KMAX / kf_1gpc) ** 2)
    nmax_1g = int(KMAX / kf_1gpc)
    gq_1g = gq_fft(qmax=qmax_1g, nmax=nmax_1g)
    n_shells_1g = np.count_nonzero(gq_1g[1:])
    print(f"  shells: {n_shells_1g}")

    print("Precomputing rebin cache for L=1Gpc ...")
    t0 = time.time()
    G_nz_1g, k_eff_1g = precompute_cache(gq_1g, kf_1gpc, KMAX, dk_factor=0.1)
    t_precomp_1g = time.time() - t0
    print(f"  precomp: {t_precomp_1g:.2f}s, {len(G_nz_1g)} bins")

    # --- Process fastPM 1Gpc ---
    for cfg in fastpm_configs:
        print(f"\n{'='*60}\n{cfg.title}\n{'='*60}")

        kc0, km0, kx0, pk_mocks0, _, _ = load_pk_multipoles(
            cfg.pk_data_glob, cfg.rid_min, cfg.rid_max, cfg.n_dp+1)
        _, _, _, cov_mocks0, _, _ = load_pk_multipoles(
            cfg.pk_cov_glob, cfg.rid_min, cfg.rid_max, cfg.n_dp+1)

        first_valid = 0
        for ib in range(len(kc0)):
            if float(pk_mocks0[:, ib].mean()) > 0:
                first_valid = ib; break
        kcen = kc0[first_valid:first_valid+cfg.n_dp]
        kmin_b = km0[first_valid:first_valid+cfg.n_dp]
        kmax_b = kx0[first_valid:first_valid+cfg.n_dp]
        pk_mocks = pk_mocks0[:, first_valid:first_valid+cfg.n_dp]
        cov_mocks = cov_mocks0[:, first_valid:first_valid+cfg.n_dp]
        pk_mean = pk_mocks.mean(axis=0)

        s, xi_mean, xi_std = load_pcf(cfg.pcf_glob, cfg.rid_min, cfg.rid_max)
        r2_data = s**2 * xi_mean; r2_std = s**2 * xi_std

        print("  [fit] standard ...")
        bf_std = fit_standard_desilike(cosmo=cosmo, kcen=kcen, kmin_b=kmin_b,
            kmax_b=kmax_b, pk_mean=pk_mean, cov_mocks=cov_mocks, p_fix=P_FIX, z=Z)

        print("  [fit] BinAvgFit ...")
        kmax_fit = float(kmax_b[-1])
        qmax_fit = int(np.floor((kmax_fit/kf_1gpc)**2))+1
        gq_fit = gq_enumerate(qmax_fit)
        _, k_shell, g_shell = build_shells_for_kmax(kf_1gpc, gq_fit, kmax=kmax_fit)
        idx_per_bin = build_bin_shell_index(k_shell, g_shell, kmin_b, kmax_b)
        bf = fit_binavg_minuit(cosmo=cosmo, base_params=bf_std, k_shell=k_shell,
            g_shell=g_shell, idx_per_bin=idx_per_bin, pk_data=pk_mean,
            cov_mocks=cov_mocks, p_fix=P_FIX, z=Z)

        fnl_val = float(bf.get('fnl_loc', 0))
        b1_val = float(bf.get('b1', 0))
        print(f"    best-fit: fnl={fnl_val:.3f}, b1={b1_val:.6f}")

        pd = eval_pk_dense(cosmo=cosmo, k=kd_1gpc, params=bf, p_fix=P_FIX, z=Z)

        # Baseline
        print("  [xi] Baseline ...")
        t0 = time.time()
        xi_base = xi_discrete_multi(s=s, gq=gq_1g, kf=kf_1gpc, V=V_1gpc,
            kd=kd_1gpc, pd_list=[pd], databin_edges=None, databin_pk=None, chunk=500_000)["FD0"]
        t_base = time.time() - t0

        # ExactRebin
        print("  [xi] ExactRebin ...")
        t0 = time.time()
        xi_er = xi_exact_rebin(s, gq_1g, kf_1gpc, V_1gpc, kd_1gpc, pd, dk_factor=0.1)
        t_er = time.time() - t0

        # CachedRebin
        print("  [xi] CachedRebin ...")
        t0 = time.time()
        xi_cr = xi_cached_rebin(s, G_nz_1g, k_eff_1g, V_1gpc, kd_1gpc, pd)
        t_cr = time.time() - t0

        r2_base = s**2 * xi_base; r2_er = s**2 * xi_er; r2_cr = s**2 * xi_cr
        ma_b, ms_b = met_mean_abs_sigma(r2_data, r2_std, r2_base)
        ma_e, ms_e = met_mean_abs_sigma(r2_data, r2_std, r2_er)
        ma_c, ms_c = met_mean_abs_sigma(r2_data, r2_std, r2_cr)
        safe_base = np.where(np.abs(xi_base) > 1e-30, xi_base, 1e-30)
        rel_er = np.nanmean(np.abs((xi_er - xi_base) / safe_base))
        rel_cr = np.nanmean(np.abs((xi_cr - xi_base) / safe_base))

        print(f"    Baseline:    {t_base:.1f}s  |D/s|={ma_b:.4f}")
        print(f"    ExactRebin:  {t_er:.2f}s  |D/s|={ma_e:.4f}  rel={rel_er:.2e}  {t_base/t_er:.0f}x")
        print(f"    CachedRebin: {t_cr:.4f}s  |D/s|={ma_c:.4f}  rel={rel_cr:.2e}  {t_base/t_cr:.0f}x")

        all_results.append({
            "title": cfg.title, "type": "fastPM", "fnl_true": 100 if "fnl100" in cfg.title else 0,
            "fnl_fit": fnl_val, "b1_fit": b1_val,
            "t_base": t_base, "t_er": t_er, "t_cr": t_cr,
            "ma_b": ma_b, "ms_b": ms_b, "ma_e": ma_e, "ms_e": ms_e,
            "ma_c": ma_c, "ms_c": ms_c, "rel_er": rel_er, "rel_cr": rel_cr,
            "s": s, "r2_data": r2_data, "r2_std": r2_std,
            "r2_base": r2_base, "r2_er": r2_er, "r2_cr": r2_cr,
        })

    # --- Process Quijote ---
    # Load covariance from fnl=0
    print(f"\n{'='*60}\nLoading Quijote covariance (fnl=0) ...\n{'='*60}")
    kc_cov, km_cov, kx_cov, cov_mocks_q, _, _, n_cov = load_pk_quijote(quijote_cov_pat)
    print(f"  Covariance files: {n_cov}")

    for fnl_true in sorted(quijote_fnl_configs.keys()):
        label, pk_pat, pcf_pat = quijote_fnl_configs[fnl_true]
        title = f"Quijote fnl={fnl_true}"
        print(f"\n{'='*60}\n{title} ({label})\n{'='*60}")

        try:
            kcen_q, kmin_q, kmax_q, pk_mocks_q, pk_mean_q, _, n_pk = load_pk_quijote(pk_pat)
            s_q, xi_mean_q, xi_std_q, n_pcf = load_pcf_quijote(pcf_pat)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        # 只用前 QUIJOTE_FIT_NDP 个 k-bin 拟合（与 mission9/10 一致）
        n_dp_q = min(QUIJOTE_FIT_NDP, len(kcen_q), len(kc_cov))
        kcen_q = kcen_q[:n_dp_q]; kmin_q = kmin_q[:n_dp_q]; kmax_q = kmax_q[:n_dp_q]
        pk_mean_q = pk_mean_q[:n_dp_q]
        cov_q = cov_mocks_q[:, :n_dp_q]

        r2_data_q = s_q**2 * xi_mean_q; r2_std_q = s_q**2 * xi_std_q
        print(f"  pk mocks: {n_pk}, pcf mocks: {n_pcf}, k-bins: {n_dp_q}")

        print("  [fit] standard ...")
        bf_std_q = fit_standard_desilike(cosmo=cosmo, kcen=kcen_q, kmin_b=kmin_q,
            kmax_b=kmax_q, pk_mean=pk_mean_q, cov_mocks=cov_q, p_fix=P_FIX, z=Z)

        print("  [fit] BinAvgFit ...")
        kmax_fit_q = float(kmax_q[-1])
        qmax_fit_q = int(np.floor((kmax_fit_q/kf_1gpc)**2))+1
        gq_fit_q = gq_enumerate(qmax_fit_q)
        _, ks_q, gs_q = build_shells_for_kmax(kf_1gpc, gq_fit_q, kmax=kmax_fit_q)
        idx_q = build_bin_shell_index(ks_q, gs_q, kmin_q, kmax_q)
        bf_q = fit_binavg_minuit(cosmo=cosmo, base_params=bf_std_q, k_shell=ks_q,
            g_shell=gs_q, idx_per_bin=idx_q, pk_data=pk_mean_q,
            cov_mocks=cov_q, p_fix=P_FIX, z=Z)

        fnl_val_q = float(bf_q.get('fnl_loc', 0))
        b1_val_q = float(bf_q.get('b1', 0))
        print(f"    best-fit: fnl={fnl_val_q:.3f}, b1={b1_val_q:.6f}")

        pd_q = eval_pk_dense(cosmo=cosmo, k=kd_1gpc, params=bf_q, p_fix=P_FIX, z=Z)

        # Baseline
        print("  [xi] Baseline ...")
        t0 = time.time()
        xi_base_q = xi_discrete_multi(s=s_q, gq=gq_1g, kf=kf_1gpc, V=V_1gpc,
            kd=kd_1gpc, pd_list=[pd_q], databin_edges=None, databin_pk=None, chunk=500_000)["FD0"]
        t_base_q = time.time() - t0

        # ExactRebin
        print("  [xi] ExactRebin ...")
        t0 = time.time()
        xi_er_q = xi_exact_rebin(s_q, gq_1g, kf_1gpc, V_1gpc, kd_1gpc, pd_q, dk_factor=0.1)
        t_er_q = time.time() - t0

        # CachedRebin
        print("  [xi] CachedRebin ...")
        t0 = time.time()
        xi_cr_q = xi_cached_rebin(s_q, G_nz_1g, k_eff_1g, V_1gpc, kd_1gpc, pd_q)
        t_cr_q = time.time() - t0

        r2b_q = s_q**2 * xi_base_q; r2e_q = s_q**2 * xi_er_q; r2c_q = s_q**2 * xi_cr_q
        ma_b_q, ms_b_q = met_mean_abs_sigma(r2_data_q, r2_std_q, r2b_q)
        ma_e_q, ms_e_q = met_mean_abs_sigma(r2_data_q, r2_std_q, r2e_q)
        ma_c_q, ms_c_q = met_mean_abs_sigma(r2_data_q, r2_std_q, r2c_q)
        safe_q = np.where(np.abs(xi_base_q) > 1e-30, xi_base_q, 1e-30)
        rel_er_q = np.nanmean(np.abs((xi_er_q - xi_base_q) / safe_q))
        rel_cr_q = np.nanmean(np.abs((xi_cr_q - xi_base_q) / safe_q))

        print(f"    Baseline:    {t_base_q:.1f}s  |D/s|={ma_b_q:.4f}")
        print(f"    ExactRebin:  {t_er_q:.2f}s  |D/s|={ma_e_q:.4f}  rel={rel_er_q:.2e}  {t_base_q/t_er_q:.0f}x")
        print(f"    CachedRebin: {t_cr_q:.4f}s  |D/s|={ma_c_q:.4f}  rel={rel_cr_q:.2e}  {t_base_q/t_cr_q:.0f}x")

        all_results.append({
            "title": title, "type": "Quijote", "fnl_true": fnl_true,
            "fnl_fit": fnl_val_q, "b1_fit": b1_val_q,
            "t_base": t_base_q, "t_er": t_er_q, "t_cr": t_cr_q,
            "ma_b": ma_b_q, "ms_b": ms_b_q, "ma_e": ma_e_q, "ms_e": ms_e_q,
            "ma_c": ma_c_q, "ms_c": ms_c_q, "rel_er": rel_er_q, "rel_cr": rel_cr_q,
            "s": s_q, "r2_data": r2_data_q, "r2_std": r2_std_q,
            "r2_base": r2b_q, "r2_er": r2e_q, "r2_cr": r2c_q,
        })

    # ============================================================
    # Plots
    # ============================================================
    # Figure 1: 1Gpc fastPM (2 columns)
    fp_results = [r for r in all_results if r["type"] == "fastPM"]
    if fp_results:
        fig, axes = plt.subplots(2, len(fp_results), figsize=(7*len(fp_results), 10))
        if len(fp_results) == 1:
            axes = axes.reshape(2, 1)
        for col, res in enumerate(fp_results):
            ax_t = axes[0, col]; ax_b = axes[1, col]
            s = res["s"]
            ax_t.errorbar(s, res["r2_data"], yerr=res["r2_std"], fmt="ko", ms=3, capsize=2, label="Data", zorder=10)
            ax_t.plot(s, res["r2_base"], "tab:blue", lw=2,
                      label=f"Baseline ({res['t_base']:.1f}s, |D/s|={res['ma_b']:.3f})")
            ax_t.plot(s, res["r2_er"], "tab:red", lw=1.5, ls="--",
                      label=f"ExactRebin ({res['t_er']:.2f}s, |D/s|={res['ma_e']:.3f})")
            ax_t.plot(s, res["r2_cr"], "tab:green", lw=1.5, ls=":",
                      label=f"CachedRebin ({res['t_cr']:.3f}s, |D/s|={res['ma_c']:.3f})")
            ax_t.set_title(f"{res['title']}\nfnl_fit={res['fnl_fit']:.1f}, b1={res['b1_fit']:.3f}")
            ax_t.set_ylabel(r"$r^2\xi_0(r)$")
            ax_t.legend(fontsize=7); ax_t.grid(alpha=0.3)

            for (nm, r2m, c, ls) in [("Baseline", res["r2_base"], "tab:blue", "-"),
                                       ("ExactRebin", res["r2_er"], "tab:red", "--"),
                                       ("CachedRebin", res["r2_cr"], "tab:green", ":")]:
                ds = (res["r2_data"] - r2m) / res["r2_std"]
                ax_b.plot(s, ds, color=c, ls=ls, lw=1.5, marker="o", ms=2, label=nm)
            ax_b.axhline(0, color="k", lw=0.6)
            ax_b.axhline(1, color="gray", ls=":", lw=0.6); ax_b.axhline(-1, color="gray", ls=":", lw=0.6)
            ax_b.set_xlabel(r"$r$ [Mpc/h]"); ax_b.set_ylabel(r"$(Data-Model)/\sigma$")
            ax_b.legend(fontsize=7); ax_b.grid(alpha=0.3); ax_b.set_ylim(-3, 3)

        fig.suptitle("Mission 11: k-rebinning validation on 1Gpc fastPM", fontsize=13, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out1 = os.path.join(THIS_DIR, "mission11_validation_1gpc_fastpm.png")
        fig.savefig(out1, dpi=150); plt.close(fig)
        print(f"\n[saved] {out1}")

    # Figure 2: Quijote (6 columns, split into 2 rows x 3 cols per figure)
    qj_results = [r for r in all_results if r["type"] == "Quijote"]
    if qj_results:
        n_q = len(qj_results)
        n_row = 2
        n_col = min(3, n_q)
        n_fig = (n_q + n_col - 1) // n_col

        for ifig in range(n_fig):
            batch = qj_results[ifig * n_col: (ifig + 1) * n_col]
            fig, axes = plt.subplots(2, len(batch), figsize=(7*len(batch), 10))
            if len(batch) == 1:
                axes = axes.reshape(2, 1)
            for col, res in enumerate(batch):
                ax_t = axes[0, col]; ax_b = axes[1, col]
                s = res["s"]
                ax_t.errorbar(s, res["r2_data"], yerr=res["r2_std"], fmt="ko", ms=3, capsize=2, label="Data", zorder=10)
                ax_t.plot(s, res["r2_base"], "tab:blue", lw=2,
                          label=f"Baseline ({res['t_base']:.1f}s, |D/s|={res['ma_b']:.3f})")
                ax_t.plot(s, res["r2_er"], "tab:red", lw=1.5, ls="--",
                          label=f"ExactRebin ({res['t_er']:.2f}s, |D/s|={res['ma_e']:.3f})")
                ax_t.plot(s, res["r2_cr"], "tab:green", lw=1.5, ls=":",
                          label=f"CachedRebin ({res['t_cr']:.3f}s, |D/s|={res['ma_c']:.3f})")
                ax_t.set_title(f"{res['title']}\nfnl_fit={res['fnl_fit']:.1f}, b1={res['b1_fit']:.3f}")
                ax_t.set_ylabel(r"$r^2\xi_0(r)$")
                ax_t.legend(fontsize=7); ax_t.grid(alpha=0.3)

                for (nm, r2m, c, ls) in [("Baseline", res["r2_base"], "tab:blue", "-"),
                                           ("ExactRebin", res["r2_er"], "tab:red", "--"),
                                           ("CachedRebin", res["r2_cr"], "tab:green", ":")]:
                    ds = (res["r2_data"] - r2m) / res["r2_std"]
                    ax_b.plot(s, ds, color=c, ls=ls, lw=1.5, marker="o", ms=2, label=nm)
                ax_b.axhline(0, color="k", lw=0.6)
                ax_b.axhline(1, color="gray", ls=":", lw=0.6); ax_b.axhline(-1, color="gray", ls=":", lw=0.6)
                ax_b.set_xlabel(r"$r$ [Mpc/h]"); ax_b.set_ylabel(r"$(Data-Model)/\sigma$")
                ax_b.legend(fontsize=7); ax_b.grid(alpha=0.3); ax_b.set_ylim(-3, 3)

            fnl_range = [r["fnl_true"] for r in batch]
            fig.suptitle(f"Mission 11: k-rebinning validation on Quijote (fnl={fnl_range})", fontsize=13, y=0.99)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            out_q = os.path.join(THIS_DIR, f"mission11_validation_quijote_part{ifig+1}.png")
            fig.savefig(out_q, dpi=150); plt.close(fig)
            print(f"[saved] {out_q}")

    # ============================================================
    # Summary table
    # ============================================================
    print("\n" + "=" * 100)
    print("Summary Table")
    print("=" * 100)
    print(f"{'Dataset':<28s} {'fnl_fit':>8s} {'t_base':>7s} {'t_ER':>7s} {'t_CR':>7s} "
          f"{'|D/s|_B':>8s} {'|D/s|_ER':>9s} {'|D/s|_CR':>9s} "
          f"{'rel_ER':>10s} {'rel_CR':>10s} {'spdup_ER':>9s} {'spdup_CR':>9s}")
    print("-" * 130)
    for r in all_results:
        print(f"{r['title']:<28s} {r['fnl_fit']:>8.1f} {r['t_base']:>7.1f} {r['t_er']:>7.2f} {r['t_cr']:>7.3f} "
              f"{r['ma_b']:>8.4f} {r['ma_e']:>9.4f} {r['ma_c']:>9.4f} "
              f"{r['rel_er']:>10.2e} {r['rel_cr']:>10.2e} "
              f"{r['t_base']/r['t_er']:>9.0f}x {r['t_base']/r['t_cr']:>9.0f}x")

    # Save markdown summary
    out_md = os.path.join(THIS_DIR, "mission11_validation_1gpc_quijote.md")
    lines = []
    lines.append("# Mission 11: k-rebinning validation on 1Gpc fastPM + Quijote\n\n")
    lines.append("## Summary\n\n")
    lines.append(f"| {'Dataset':<28s} | {'fnl_fit':>8s} | {'t_base':>7s} | {'t_ER':>6s} | {'t_CR':>6s} | "
                 f"{'|D/s|_B':>8s} | {'|D/s|_ER':>8s} | {'|D/s|_CR':>8s} | "
                 f"{'rel_ER':>10s} | {'rel_CR':>10s} | {'x_ER':>5s} | {'x_CR':>6s} |\n")
    lines.append(f"|{'-'*30}|{'-'*10}|{'-'*9}|{'-'*8}|{'-'*8}|"
                 f"{'-'*10}|{'-'*10}|{'-'*10}|{'-'*12}|{'-'*12}|{'-'*7}|{'-'*8}|\n")
    for r in all_results:
        lines.append(f"| {r['title']:<28s} | {r['fnl_fit']:>8.1f} | {r['t_base']:>7.1f} | {r['t_er']:>6.2f} | {r['t_cr']:>6.3f} | "
                     f"{r['ma_b']:>8.4f} | {r['ma_e']:>8.4f} | {r['ma_c']:>8.4f} | "
                     f"{r['rel_er']:>10.2e} | {r['rel_cr']:>10.2e} | "
                     f"{r['t_base']/r['t_er']:>5.0f}x | {r['t_base']/r['t_cr']:>6.0f}x |\n")
    lines.append("\n## Notes\n\n")
    lines.append(f"- All boxes L=1Gpc, kmax=15, shells={n_shells_1g}\n")
    lines.append(f"- CachedRebin precompute time: {t_precomp_1g:.2f}s (one-time cost)\n")
    lines.append("- Covariance: fastPM uses fnl0 fastPM (50 files); Quijote uses fnl=0 fid (500 files)\n")
    lines.append("- rel_ER/rel_CR = mean|Dxi/xi_baseline|, measures numerical accuracy vs exact discrete sum\n")
    lines.append("- |D/s| = mean|Delta/sigma| vs measured 2PCF data, measures physical agreement\n")

    with open(out_md, "w") as f:
        f.writelines(lines)
    print(f"\n[saved] {out_md}")
    print(f"Total time: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
