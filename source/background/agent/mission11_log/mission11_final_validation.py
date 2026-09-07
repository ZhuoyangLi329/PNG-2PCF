#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission 11: 全离散求和加速 — 最终验证图
=========================================

对比三种方案在 3Gpc fnl100 和 3Gpc fnl0 上的 2PCF:
1. Baseline (原始全离散, ~69s)
2. ExactRebin (精确k-rebinning, ~1.2s, 58x加速)
3. CachedRebin (缓存k-rebinning, 预计算2.6s + 每次0.08s, ~690x加速)

三种方案的 vs data 指标几乎一致。

运行:
cd /pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission11_log
bash -lc "conda activate desilike && PYTHONUNBUFFERED=1 python -u mission11_final_validation.py"
"""

from __future__ import annotations
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MISSION10_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "mission10_log"))
if MISSION10_DIR not in sys.path:
    sys.path.insert(0, MISSION10_DIR)

from mission10_binavgfit_full_discrete_compare import (
    BoxConfig, load_pk_multipoles, load_pcf,
    fit_standard_desilike, fit_binavg_minuit, eval_pk_dense,
    gq_enumerate, build_shells_for_kmax, build_bin_shell_index,
    gq_fft, xi_discrete_multi, met_mean_abs_sigma,
)


def xi_exact_rebin(s, gq, kf, V, kd, pd, dk_factor=0.1):
    """精确 k-rebinning: 对 42M 壳层做精确 interp 后聚合到 k-bin。

    每个 bin 的权重 W_bin = sum(g_q * P(k_q)) 是精确的（不是 G * P(k_eff)）。
    加权中心 k_eff_bin = sum(g*P*k) / sum(g*P)。

    精度几乎无损，耗时约 1.2s（主要花在 42M 点的 interp 和 bincount）。
    """
    qnz = np.nonzero(gq[1:])[0] + 1
    kv = kf * np.sqrt(qnz.astype(np.float64))
    g = gq[qnz].astype(np.float64)
    pv = np.interp(kv, kd, pd)
    w = g * pv

    dk = dk_factor * kf
    n_bins = int(np.ceil(kv.max() / dk)) + 1
    bin_idx = (kv / dk).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

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
    """缓存 k-rebinning: 使用预计算的 G_nz 和 k_eff。

    每次只需 interp 71k 个点 + 矩阵乘法，耗时约 0.08s。
    W = G_nz * P(k_eff) 是一个近似（假设 P 在 bin 内恒定），
    但 dk=0.1kf 时精度很好（max|Δξ/ξ| ~ 0.04%）。
    """
    p_eff = np.interp(k_eff, kd, pd)
    W = G_nz * p_eff

    arg = np.outer(k_eff, s)
    J = np.ones_like(arg)
    m = arg != 0
    J[m] = np.sin(arg[m]) / arg[m]

    return (W @ J) / V


def precompute_cache(gq, kf, kmax, dk_factor=0.1):
    """预计算 G_nz 和 k_eff（只依赖 L 和 kmax，可缓存到文件）。"""
    qnz = np.nonzero(gq[1:])[0] + 1
    kv = kf * np.sqrt(qnz.astype(np.float64))
    g = gq[qnz].astype(np.float64)

    dk = dk_factor * kf
    n_bins = int(np.ceil(kmax / dk)) + 1
    bin_idx = (kv / dk).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    G_bin = np.bincount(bin_idx, weights=g, minlength=n_bins)
    Gk_bin = np.bincount(bin_idx, weights=g * kv, minlength=n_bins)

    nz = G_bin > 0
    G_nz = G_bin[nz].astype(np.float64)
    k_eff = Gk_bin[nz] / G_nz

    return G_nz, k_eff


def main():
    t_total = time.time()

    Z = 1.0; P_FIX = 1.2; KMAX = 15.0; N_DENSE = 300_000

    from cosmoprimo import Cosmology
    cosmo = Cosmology(h=0.6711, Omega_b=0.049, Omega_cdm=0.3175-0.049,
                      sigma8=0.834, n_s=0.9624, engine="class")

    configs = [
        BoxConfig(title="3Gpc fnl100", L=3000.0,
            pk_data_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat",
            pk_cov_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
            pcf_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",
            rid_min=2, rid_max=99, n_dp=20),
        BoxConfig(title="3Gpc fnl0", L=3000.0,
            pk_data_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
            pk_cov_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
            pcf_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat",
            rid_min=2, rid_max=99, n_dp=20),
    ]

    fig, axes = plt.subplots(2, len(configs), figsize=(7 * len(configs), 10))
    if len(configs) == 1:
        axes = axes.reshape(2, 1)

    gq_cache = {}
    kd_cache = {}
    cache_cache = {}
    all_results = []

    for col, cfg in enumerate(configs):
        print(f"\n{'='*60}\n{cfg.title}\n{'='*60}")

        L = float(cfg.L); kf = 2*np.pi/L; V = L**3

        # 读数据
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
        r2_data = s**2 * xi_mean
        r2_std = s**2 * xi_std

        # BinAvgFit
        print("  [fit] standard ...")
        bf_std = fit_standard_desilike(cosmo=cosmo, kcen=kcen, kmin_b=kmin_b,
            kmax_b=kmax_b, pk_mean=pk_mean, cov_mocks=cov_mocks, p_fix=P_FIX, z=Z)

        print("  [fit] BinAvgFit ...")
        kmax_fit = float(kmax_b[-1])
        qmax_fit = int(np.floor((kmax_fit/kf)**2))+1
        gq_fit = gq_enumerate(qmax_fit)
        _, k_shell, g_shell = build_shells_for_kmax(kf, gq_fit, kmax=kmax_fit)
        idx_per_bin = build_bin_shell_index(k_shell, g_shell, kmin_b, kmax_b)
        bf_binavg = fit_binavg_minuit(cosmo=cosmo, base_params=bf_std, k_shell=k_shell,
            g_shell=g_shell, idx_per_bin=idx_per_bin, pk_data=pk_mean,
            cov_mocks=cov_mocks, p_fix=P_FIX, z=Z)

        fnl_val = float(bf_binavg.get('fnl_loc', 0))
        b1_val = float(bf_binavg.get('b1', 0))
        sigmas_val = float(bf_binavg.get('sigmas', 0))
        print(f"    best-fit: fnl={fnl_val:.3f}, b1={b1_val:.6f}, sigmas={sigmas_val:.6f}")

        # P_model
        if L in kd_cache:
            kd = kd_cache[L]
        else:
            kd = np.geomspace(kf*0.5, KMAX*1.1, N_DENSE)
            kd_cache[L] = kd
        pd = eval_pk_dense(cosmo=cosmo, k=kd, params=bf_binavg, p_fix=P_FIX, z=Z)

        # g_q
        if L in gq_cache:
            gq = gq_cache[L]
        else:
            qmax = int((KMAX/kf)**2); nmax = int(KMAX/kf)
            print(f"  [g_q] computing ...")
            gq = gq_fft(qmax=qmax, nmax=nmax)
            gq_cache[L] = gq

        # 1. Baseline
        print("  [xi] Baseline ...")
        t0 = time.time()
        xi_dict = xi_discrete_multi(s=s, gq=gq, kf=kf, V=V, kd=kd,
            pd_list=[pd], databin_edges=None, databin_pk=None, chunk=500_000)
        xi_base = xi_dict["FD0"]
        t_base = time.time() - t0
        r2_base = s**2 * xi_base
        ma_base, ms_base = met_mean_abs_sigma(r2_data, r2_std, r2_base)
        print(f"    Baseline: {t_base:.1f}s, vs data: |Δ/σ|={ma_base:.4f}, Δ/σ={ms_base:+.4f}")

        # 2. ExactRebin
        print("  [xi] ExactRebin ...")
        t0 = time.time()
        xi_er = xi_exact_rebin(s, gq, kf, V, kd, pd, dk_factor=0.1)
        t_er = time.time() - t0
        r2_er = s**2 * xi_er
        ma_er, ms_er = met_mean_abs_sigma(r2_data, r2_std, r2_er)
        rel_er = np.abs((xi_er - xi_base) / np.where(np.abs(xi_base) > 1e-30, xi_base, 1e-30))
        print(f"    ExactRebin: {t_er:.2f}s, vs data: |Δ/σ|={ma_er:.4f}, Δ/σ={ms_er:+.4f}, "
              f"vs baseline: mean|Δξ/ξ|={rel_er.mean():.2e}")

        # 3. CachedRebin
        print("  [xi] CachedRebin ...")
        if L in cache_cache:
            G_nz, k_eff = cache_cache[L]
            t_precomp = 0
        else:
            t0 = time.time()
            G_nz, k_eff = precompute_cache(gq, kf, KMAX, dk_factor=0.1)
            t_precomp = time.time() - t0
            cache_cache[L] = (G_nz, k_eff)

        t0 = time.time()
        xi_cr = xi_cached_rebin(s, G_nz, k_eff, V, kd, pd)
        t_eval = time.time() - t0
        r2_cr = s**2 * xi_cr
        ma_cr, ms_cr = met_mean_abs_sigma(r2_data, r2_std, r2_cr)
        rel_cr = np.abs((xi_cr - xi_base) / np.where(np.abs(xi_base) > 1e-30, xi_base, 1e-30))
        print(f"    CachedRebin: precomp={t_precomp:.2f}s + eval={t_eval:.4f}s, "
              f"vs data: |Δ/σ|={ma_cr:.4f}, Δ/σ={ms_cr:+.4f}, "
              f"vs baseline: mean|Δξ/ξ|={rel_cr.mean():.2e}")

        all_results.append({
            "title": cfg.title,
            "fnl": fnl_val, "b1": b1_val, "sigmas": sigmas_val,
            "t_base": t_base, "t_er": t_er, "t_cr_pre": t_precomp, "t_cr_eval": t_eval,
            "ma_base": ma_base, "ms_base": ms_base,
            "ma_er": ma_er, "ms_er": ms_er,
            "ma_cr": ma_cr, "ms_cr": ms_cr,
            "mean_rel_er": rel_er.mean(), "mean_rel_cr": rel_cr.mean(),
        })

        # 画图
        ax_top = axes[0, col]; ax_bot = axes[1, col]

        ax_top.errorbar(s, r2_data, yerr=r2_std, fmt="ko", ms=3, capsize=2,
                        label="Data mean", zorder=10)
        ax_top.plot(s, r2_base, "tab:blue", lw=2.0,
                    label=f"Baseline ({t_base:.0f}s, |Δ/σ|={ma_base:.3f})")
        ax_top.plot(s, r2_er, "tab:red", lw=1.5, ls="--",
                    label=f"ExactRebin ({t_er:.1f}s, |Δ/σ|={ma_er:.3f})")
        ax_top.plot(s, r2_cr, "tab:green", lw=1.5, ls=":",
                    label=f"CachedRebin ({t_eval:.2f}s, |Δ/σ|={ma_cr:.3f})")
        ax_top.set_title(f"{cfg.title}\nbest-fit: fnl={fnl_val:.1f}, b1={b1_val:.3f}")
        ax_top.set_ylabel(r"$r^2\xi_0(r)$")
        ax_top.legend(fontsize=8)
        ax_top.grid(alpha=0.3)

        # 残差
        for (name, r2_m, c, ls) in [
            ("Baseline", r2_base, "tab:blue", "-"),
            ("ExactRebin", r2_er, "tab:red", "--"),
            ("CachedRebin", r2_cr, "tab:green", ":"),
        ]:
            ds = (r2_data - r2_m) / r2_std
            ax_bot.plot(s, ds, color=c, ls=ls, lw=1.5, marker="o", ms=2, label=name)
        ax_bot.axhline(0, color="k", lw=0.6)
        ax_bot.axhline(1, color="gray", ls=":", lw=0.6)
        ax_bot.axhline(-1, color="gray", ls=":", lw=0.6)
        ax_bot.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax_bot.set_ylabel(r"$(Data-Model)/\sigma$")
        ax_bot.legend(fontsize=8)
        ax_bot.grid(alpha=0.3)
        ax_bot.set_ylim(-3, 3)

    fig.suptitle("Mission 11: Full Discrete Acceleration\n(BinAvgFit + FullDiscrete, kmax=15)",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(THIS_DIR, "mission11_final_validation.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\n[saved] {out_png}")

    # 结果汇总表
    print("\n" + "=" * 80)
    print("Final Summary")
    print("=" * 80)
    for res in all_results:
        print(f"\n{res['title']}:")
        print(f"  Baseline:    {res['t_base']:.1f}s  vs data |Δ/σ|={res['ma_base']:.4f}")
        print(f"  ExactRebin:  {res['t_er']:.2f}s  vs data |Δ/σ|={res['ma_er']:.4f}  "
              f"vs baseline mean|Δξ/ξ|={res['mean_rel_er']:.2e}  speedup={res['t_base']/res['t_er']:.0f}x")
        print(f"  CachedRebin: {res['t_cr_eval']:.3f}s  vs data |Δ/σ|={res['ma_cr']:.4f}  "
              f"vs baseline mean|Δξ/ξ|={res['mean_rel_cr']:.2e}  speedup={res['t_base']/res['t_cr_eval']:.0f}x")

    print(f"\nTotal time: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
