#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 1Gpc FastPM fnl100 的当前方法 best-fit 2PCF 数组。

代码大纲
--------
1. 读取 1Gpc FastPM fnl100 的 RSD P(k) 与原始盒子 2PCF 文件；
2. 用 fnl0 的 P(k) mocks 构造协方差，与 Mission10/11 保持一致；
3. 先做标准 desilike 拟合，再做当前口径的 BinAvgFit；
4. 用 BinAvgFit 参数评估稠密 P_model(k)；
5. 用 FullDiscrete 的 CachedRebin 加速求和得到 xi_box_model(s)；
6. 保存 npz/json，供 meeting figure 直接读取。

逻辑关系
--------
- `build_cosmology` 定义与 Mission10/11 一致的 cosmoprimo 宇宙学；
- `select_fit_bins` 负责跳过 1Gpc P(k) 中可能为空的最低 bin；
- `precompute_cache` 和 `xi_cached_rebin` 是当前 Pipeline V1 使用的
  FullDiscrete 加速实现；
- `main` 组织数据读取、拟合、理论 2PCF 计算和输出。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# 登录节点运行时限制常见数值库线程数，避免超过任务书要求。
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "8")

import numpy as np
from cosmoprimo import Cosmology


THIS_DIR = Path(__file__).resolve().parent
MISSION10_DIR = THIS_DIR.parent / "mission10_log"
if str(MISSION10_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION10_DIR))

from mission10_binavgfit_full_discrete_compare import (  # noqa: E402
    BoxConfig,
    build_bin_shell_index,
    build_shells_for_kmax,
    eval_pk_dense,
    fit_binavg_minuit,
    fit_standard_desilike,
    gq_enumerate,
    gq_fft,
    load_pcf,
    load_pk_multipoles,
)


OUT_NPZ = THIS_DIR / "fastpm_1gpc_fnl100_current_bestfit_2pcf_model.npz"
OUT_JSON = THIS_DIR / "fastpm_1gpc_fnl100_current_bestfit_2pcf_model.json"

Z = 1.0
P_FIX = 1.2
KMAX_DISCRETE = 15.0
N_DENSE = 300_000
REBIN_DK_FACTOR = 0.1


def build_cosmology() -> Cosmology:
    """
    构建与 Mission10/11 和 task182 一致的 fiducial cosmology。

    返回
    ----
    Cosmology
        cosmoprimo 的宇宙学对象，用于 desilike PNG RSD P(k) 理论。
    """
    return Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )


def select_fit_bins(
    kcen0: np.ndarray,
    kmin0: np.ndarray,
    kmax0: np.ndarray,
    pk_mocks0: np.ndarray,
    cov_mocks0: np.ndarray,
    n_dp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    选择用于拟合的连续 P(k) bins。

    1Gpc 的最低 k bin 可能没有模式，Mission10/11 的做法是先多读一个 bin，
    再从第一个均值为正的 bin 开始取 `n_dp` 个 bin。

    参数
    ----
    kcen0, kmin0, kmax0 : ndarray
        多读一个 bin 后的 k 网格信息。
    pk_mocks0, cov_mocks0 : ndarray
        数据样本和协方差样本的 P(k) mocks。
    n_dp : int
        最终用于拟合的 bin 数。

    返回
    ----
    tuple
        `kcen, kmin_b, kmax_b, pk_mocks, cov_mocks, pk_mean, first_valid`
    """
    first_valid = 0
    for ibin in range(len(kcen0)):
        if float(pk_mocks0[:, ibin].mean()) > 0.0:
            first_valid = ibin
            break

    kcen = kcen0[first_valid : first_valid + n_dp]
    kmin_b = kmin0[first_valid : first_valid + n_dp]
    kmax_b = kmax0[first_valid : first_valid + n_dp]
    pk_mocks = pk_mocks0[:, first_valid : first_valid + n_dp]
    cov_mocks = cov_mocks0[:, first_valid : first_valid + n_dp]
    pk_mean = pk_mocks.mean(axis=0)
    return kcen, kmin_b, kmax_b, pk_mocks, cov_mocks, pk_mean, first_valid


def precompute_cache(
    gq: np.ndarray,
    kf: float,
    kmax: float,
    dk_factor: float = REBIN_DK_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    预计算 CachedRebin 需要的总模式数和有效 k。

    参数
    ----
    gq : ndarray
        FullDiscrete 壳层简并度数组，`gq[q]` 是 q 壳层的模式数。
    kf : float
        基模 `2*pi/L`。
    kmax : float
        离散求和的最大 k。
    dk_factor : float
        CachedRebin 的分箱宽度为 `dk_factor * kf`。

    返回
    ----
    tuple
        `g_nz, k_eff`，分别是非空重分箱的总模式数和有效 k。
    """
    q_nz = np.nonzero(gq[1:])[0] + 1
    k_shell = kf * np.sqrt(q_nz.astype(np.float64))
    g_shell = gq[q_nz].astype(np.float64)

    dk = dk_factor * kf
    n_bins = int(np.ceil(kmax / dk)) + 1
    bin_index = np.clip((k_shell / dk).astype(np.int64), 0, n_bins - 1)

    g_bin = np.bincount(bin_index, weights=g_shell, minlength=n_bins)
    gk_bin = np.bincount(bin_index, weights=g_shell * k_shell, minlength=n_bins)
    nonzero = g_bin > 0.0
    return g_bin[nonzero], gk_bin[nonzero] / g_bin[nonzero]


def xi_cached_rebin(
    s: np.ndarray,
    g_nz: np.ndarray,
    k_eff: np.ndarray,
    volume: float,
    k_dense: np.ndarray,
    p_dense: np.ndarray,
) -> np.ndarray:
    """
    用 CachedRebin 计算 FullDiscrete 近似的 xi0(s)。

    参数
    ----
    s : ndarray
        2PCF 的 separation 网格。
    g_nz, k_eff : ndarray
        `precompute_cache` 输出的非空重分箱信息。
    volume : float
        周期盒体积 `L^3`。
    k_dense, p_dense : ndarray
        best-fit 理论 P(k) 的稠密网格。

    返回
    ----
    ndarray
        与 `s` 同长度的 best-fit `xi_box_model(s)`。
    """
    weights = g_nz * np.interp(k_eff, k_dense, p_dense)
    kr = np.outer(k_eff, s)
    kernel = np.ones_like(kr)
    mask = kr != 0.0
    kernel[mask] = np.sin(kr[mask]) / kr[mask]
    return (weights @ kernel) / volume


def main() -> None:
    """
    主流程：读取数据、拟合、计算当前方法 best-fit 2PCF 并保存。
    """
    t_start = time.time()
    cfg = BoxConfig(
        title="1Gpc FastPM fnl100",
        L=1000.0,
        pk_data_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat",
        pk_cov_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
        pcf_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat",
        rid_min=1,
        rid_max=50,
        n_dp=20,
    )

    box_size = float(cfg.L)
    kfund = 2.0 * np.pi / box_size
    volume = box_size**3
    cosmo = build_cosmology()

    print("[info] loading 1Gpc FastPM fnl100 P(k) and 2PCF ...")
    kcen0, kmin0, kmax0, pk_mocks0, _pk_mean0, _pk_std0 = load_pk_multipoles(
        cfg.pk_data_glob,
        cfg.rid_min,
        cfg.rid_max,
        cfg.n_dp + 1,
    )
    _kc, _km, _kx, cov_mocks0, _cov_mean0, _cov_std0 = load_pk_multipoles(
        cfg.pk_cov_glob,
        cfg.rid_min,
        cfg.rid_max,
        cfg.n_dp + 1,
    )
    kcen, kmin_b, kmax_b, pk_mocks, cov_mocks, pk_mean, first_valid = select_fit_bins(
        kcen0,
        kmin0,
        kmax0,
        pk_mocks0,
        cov_mocks0,
        cfg.n_dp,
    )
    s, xi_mean, xi_std = load_pcf(cfg.pcf_glob, cfg.rid_min, cfg.rid_max)
    print(f"[info] first valid P(k) bin index: {first_valid}")
    print(f"[info] pk mocks: {pk_mocks.shape[0]}, pcf points: {len(s)}")

    print("[fit] standard desilike fit ...")
    bestfit_std = fit_standard_desilike(
        cosmo=cosmo,
        kcen=kcen,
        kmin_b=kmin_b,
        kmax_b=kmax_b,
        pk_mean=pk_mean,
        cov_mocks=cov_mocks,
        p_fix=P_FIX,
        z=Z,
    )
    print(
        "[fit] standard best-fit: "
        f"fnl={float(bestfit_std.get('fnl_loc', np.nan)):.6f}, "
        f"b1={float(bestfit_std.get('b1', np.nan)):.6f}, "
        f"sigmas={float(bestfit_std.get('sigmas', np.nan)):.6f}"
    )

    print("[fit] BinAvgFit ...")
    kmax_fit = float(kmax_b[-1])
    qmax_fit = int(np.floor((kmax_fit / kfund) ** 2)) + 1
    gq_fit = gq_enumerate(qmax_fit)
    _q_shell, k_shell, g_shell = build_shells_for_kmax(kfund, gq_fit, kmax=kmax_fit)
    idx_per_bin = build_bin_shell_index(k_shell, g_shell, kmin_b, kmax_b)
    bestfit_binavg = fit_binavg_minuit(
        cosmo=cosmo,
        base_params=bestfit_std,
        k_shell=k_shell,
        g_shell=g_shell,
        idx_per_bin=idx_per_bin,
        pk_data=pk_mean,
        cov_mocks=cov_mocks,
        p_fix=P_FIX,
        z=Z,
    )
    print(
        "[fit] BinAvgFit best-fit: "
        f"fnl={float(bestfit_binavg.get('fnl_loc', np.nan)):.6f}, "
        f"b1={float(bestfit_binavg.get('b1', np.nan)):.6f}, "
        f"sigmas={float(bestfit_binavg.get('sigmas', np.nan)):.6f}"
    )

    print("[model] evaluating dense P_model(k) ...")
    k_dense = np.geomspace(kfund * 0.5, KMAX_DISCRETE * 1.1, N_DENSE)
    p_dense = eval_pk_dense(
        cosmo=cosmo,
        k=k_dense,
        params=bestfit_binavg,
        p_fix=P_FIX,
        z=Z,
    )

    print("[model] building FullDiscrete CachedRebin operator ...")
    t_cache = time.time()
    qmax = int((KMAX_DISCRETE / kfund) ** 2)
    nmax = int(KMAX_DISCRETE / kfund)
    gq = gq_fft(qmax=qmax, nmax=nmax)
    g_nz, k_eff = precompute_cache(gq, kfund, KMAX_DISCRETE, dk_factor=REBIN_DK_FACTOR)
    cache_sec = time.time() - t_cache
    print(f"[model] cache bins: {len(g_nz)}, cache seconds: {cache_sec:.2f}")

    print("[model] computing xi_box_model(s) ...")
    xi_cached_binavg = xi_cached_rebin(
        s=s,
        g_nz=g_nz,
        k_eff=k_eff,
        volume=volume,
        k_dense=k_dense,
        p_dense=p_dense,
    )

    np.savez(
        OUT_NPZ,
        box_size=np.array(box_size, dtype="f8"),
        kfund=np.array(kfund, dtype="f8"),
        kmax_discrete=np.array(KMAX_DISCRETE, dtype="f8"),
        rebin_dk_factor=np.array(REBIN_DK_FACTOR, dtype="f8"),
        n_dense=np.array(N_DENSE, dtype="i8"),
        first_valid_pk_bin=np.array(first_valid, dtype="i8"),
        k_values=kcen,
        kmin_b=kmin_b,
        kmax_b=kmax_b,
        pk_mean=pk_mean,
        pk_std=pk_mocks.std(axis=0, ddof=1),
        s=s,
        xi_mean=xi_mean,
        xi_std=xi_std,
        k_dense=k_dense,
        p_dense=p_dense,
        g_nz=g_nz,
        k_eff=k_eff,
        xi_cached_binavg=xi_cached_binavg,
        bestfit_std_fnl_loc=np.array(float(bestfit_std.get("fnl_loc", np.nan)), dtype="f8"),
        bestfit_std_b1=np.array(float(bestfit_std.get("b1", np.nan)), dtype="f8"),
        bestfit_std_sigmas=np.array(float(bestfit_std.get("sigmas", np.nan)), dtype="f8"),
        bestfit_binavg_fnl_loc=np.array(float(bestfit_binavg.get("fnl_loc", np.nan)), dtype="f8"),
        bestfit_binavg_b1=np.array(float(bestfit_binavg.get("b1", np.nan)), dtype="f8"),
        bestfit_binavg_sigmas=np.array(float(bestfit_binavg.get("sigmas", np.nan)), dtype="f8"),
        bestfit_binavg_p=np.array(P_FIX, dtype="f8"),
        bestfit_binavg_sn0=np.array(0.0, dtype="f8"),
        cached_rebin_precompute_sec=np.array(cache_sec, dtype="f8"),
        total_sec=np.array(time.time() - t_start, dtype="f8"),
    )

    summary = {
        "case": cfg.title,
        "method": "BinAvgFit + FullDiscrete/CachedRebin",
        "box_size": box_size,
        "kmax_discrete": KMAX_DISCRETE,
        "rebin_dk_factor": REBIN_DK_FACTOR,
        "n_dense": N_DENSE,
        "first_valid_pk_bin": int(first_valid),
        "n_pk_mocks": int(pk_mocks.shape[0]),
        "n_pcf_points": int(len(s)),
        "bestfit_std": {
            "fnl_loc": float(bestfit_std.get("fnl_loc", np.nan)),
            "b1": float(bestfit_std.get("b1", np.nan)),
            "sigmas": float(bestfit_std.get("sigmas", np.nan)),
            "p": P_FIX,
            "sn0": 0.0,
        },
        "bestfit_binavg": {
            "fnl_loc": float(bestfit_binavg.get("fnl_loc", np.nan)),
            "b1": float(bestfit_binavg.get("b1", np.nan)),
            "sigmas": float(bestfit_binavg.get("sigmas", np.nan)),
            "p": P_FIX,
            "sn0": 0.0,
        },
        "outputs": {
            "npz": str(OUT_NPZ),
            "json": str(OUT_JSON),
        },
        "timing_sec": {
            "cached_rebin_precompute": float(cache_sec),
            "total": float(time.time() - t_start),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[OK] saved npz: {OUT_NPZ}")
    print(f"[OK] saved json: {OUT_JSON}")


if __name__ == "__main__":
    main()
