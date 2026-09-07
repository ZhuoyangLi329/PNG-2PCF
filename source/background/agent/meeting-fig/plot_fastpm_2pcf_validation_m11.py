#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meeting-fig: FastPM + Quijote 2PCF validation with Mission11 method
===================================================================

代码大纲
--------
1. 复用 mission10 的拟合部分：
   - 标准 desilike bin-center 拟合
   - BinAvgFit（最终建模口径）
2. 复用 mission11 的数值积分部分：
   - CachedRebin / FastDiscrete
3. 生成两张总图（只保留 PDF）：
   - fastPM: 4 个 top panel + 4 个 sigma residual panel
   - Quijote: 6 个 top panel + 6 个 sigma residual panel

说明
----
- 这里只学习并复用到 mission11 为止，不涉及 mission12 之后的内容。
- 所有 residual 都定义为 `(Data - Model) / sigma`，其中 sigma 来自测量 2PCF 的样本标准差。
- 输出目录只保留会议需要的总图 PDF。
"""

from __future__ import annotations

import glob
import os
import re
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 路径与 mission10 复用函数
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MISSION10_DIR = os.path.join(AGENT_DIR, "mission10_log")
if MISSION10_DIR not in sys.path:
    sys.path.insert(0, MISSION10_DIR)

from mission10_binavgfit_full_discrete_compare import (  # noqa: E402
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
)


# ============================================================
# 全局设置
# ============================================================
OUT_DIR = THIS_DIR
Z = 1.0
P_FIX = 1.2
KMAX = 15.0
N_DENSE = 300_000
DK_FACTOR = 0.1

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "figure.dpi": 150,
})


# ============================================================
# FastPM 配置
# ============================================================
FASTPM_CONFIGS = [
    BoxConfig(
        title="3Gpc fastPM fnl100",
        L=3000.0,
        pk_data_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat",
        pk_cov_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
        pcf_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",
        rid_min=2,
        rid_max=99,
        n_dp=20,
    ),
    BoxConfig(
        title="3Gpc fastPM fnl0",
        L=3000.0,
        pk_data_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
        pk_cov_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
        pcf_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat",
        rid_min=2,
        rid_max=99,
        n_dp=20,
    ),
    BoxConfig(
        title="1Gpc fastPM fnl100",
        L=1000.0,
        pk_data_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat",
        pk_cov_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
        pcf_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat",
        rid_min=1,
        rid_max=50,
        n_dp=20,
    ),
    BoxConfig(
        title="1Gpc fastPM fnl0",
        L=1000.0,
        pk_data_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
        pk_cov_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
        pcf_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut_fnl0/pcf_rsd_N*.dat",
        rid_min=1,
        rid_max=50,
        n_dp=20,
    ),
]


# ============================================================
# Quijote 配置
# ============================================================
QUIJOTE_FIT_NDP = 20
QUIJOTE_BASE_DIR = "/pscratch/sd/l/lzy/pks_2pcfs"
QUIJOTE_INTERP_DIR = (
    "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission6_log/"
    "nbody_validation/interpolated"
)
QUIJOTE_COV_PATTERN = f"{QUIJOTE_BASE_DIR}/pk_fid_*.txt"
QUIJOTE_FNL_CONFIGS = [
    (0, "fid", f"{QUIJOTE_BASE_DIR}/pk_fid_*.txt", f"{QUIJOTE_BASE_DIR}/pcf_fid_*.dat"),
    (20, "fnl020", f"{QUIJOTE_INTERP_DIR}/fnl020/pk_fnl20_*.txt", f"{QUIJOTE_INTERP_DIR}/fnl020/pcf_fnl20_*.dat"),
    (30, "fnl030", f"{QUIJOTE_INTERP_DIR}/fnl030/pk_fnl30_*.txt", f"{QUIJOTE_INTERP_DIR}/fnl030/pcf_fnl30_*.dat"),
    (50, "LCp50", f"{QUIJOTE_BASE_DIR}/pk_LCp50_*.txt", f"{QUIJOTE_BASE_DIR}/pcf_LCp50_*.dat"),
    (75, "fnl075", f"{QUIJOTE_INTERP_DIR}/fnl075/pk_fnl75_*.txt", f"{QUIJOTE_INTERP_DIR}/fnl075/pcf_fnl75_*.dat"),
    (100, "LCp100", f"{QUIJOTE_BASE_DIR}/pk_LCp100_*.txt", f"{QUIJOTE_BASE_DIR}/pcf_LCp100_*.dat"),
]


# ============================================================
# Mission11 的 CachedRebin / FastDiscrete 核心
# ============================================================
def precompute_rebin_cache(gq: np.ndarray, kf: float, kmax: float, dk_factor: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    预计算 CachedRebin 所需的模式总数与有效 k。

    参数
    ----
    gq : ndarray
        壳层简并度数组。
    kf : float
        基模 2π/L。
    kmax : float
        离散求和的最大 k。
    dk_factor : float
        重分箱宽度，定义为 dk = dk_factor * kf。

    返回
    ----
    g_nz, k_eff : tuple[ndarray, ndarray]
        每个非空重分箱里的总模式数与有效 k。
    """
    q_nz = np.nonzero(gq[1:])[0] + 1
    k_shell = kf * np.sqrt(q_nz.astype(np.float64))
    g_shell = gq[q_nz].astype(np.float64)

    dk = dk_factor * kf
    n_bins = int(np.ceil(kmax / dk)) + 1
    bin_index = np.clip((k_shell / dk).astype(np.int64), 0, n_bins - 1)

    g_bin = np.bincount(bin_index, weights=g_shell, minlength=n_bins)
    gk_bin = np.bincount(bin_index, weights=g_shell * k_shell, minlength=n_bins)
    nonzero = g_bin > 0
    return g_bin[nonzero], gk_bin[nonzero] / g_bin[nonzero]


def xi_cached_rebin(s: np.ndarray, g_nz: np.ndarray, k_eff: np.ndarray, volume: float, kd: np.ndarray, pd: np.ndarray) -> np.ndarray:
    """
    用 CachedRebin 方法快速计算 xi0(r)。

    参数
    ----
    s : ndarray
        2PCF 的 r 网格。
    g_nz : ndarray
        每个非空重分箱里的总模式数。
    k_eff : ndarray
        每个非空重分箱的有效 k。
    volume : float
        盒子体积。
    kd, pd : ndarray
        稠密 k 网格及对应的理论 P(k)。

    返回
    ----
    ndarray
        xi0(r)。
    """
    weights = g_nz * np.interp(k_eff, kd, pd)
    kr = np.outer(k_eff, s)
    j0 = np.ones_like(kr)
    mask = kr != 0.0
    j0[mask] = np.sin(kr[mask]) / kr[mask]
    return (weights @ j0) / volume


def ensure_box_cache(box_size: float, cache_by_l: dict[float, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """
    为指定盒长准备 Mission11 的缓存。

    参数
    ----
    box_size : float
        盒长 L。
    cache_by_l : dict
        已有缓存字典，会原地更新。

    返回
    ----
    dict
        对应当前盒长的缓存条目，包含 g_q / kd / g_nz / k_eff。
    """
    if box_size in cache_by_l:
        return cache_by_l[box_size]

    kf = 2.0 * np.pi / box_size
    print(f"  [cache] building for L={box_size:.0f} ...")
    qmax = int((KMAX / kf) ** 2)
    nmax = int(KMAX / kf)
    gq = gq_fft(qmax=qmax, nmax=nmax)
    kd = np.geomspace(kf * 0.5, KMAX * 1.1, N_DENSE)
    g_nz, k_eff = precompute_rebin_cache(gq=gq, kf=kf, kmax=KMAX, dk_factor=DK_FACTOR)
    cache_by_l[box_size] = {
        "gq": gq,
        "kd": kd,
        "g_nz": g_nz,
        "k_eff": k_eff,
    }
    print(f"    shells={np.count_nonzero(gq[1:])}, rebinned bins={len(g_nz)}")
    return cache_by_l[box_size]


# ============================================================
# Quijote I/O（复用 mission11 口径）
# ============================================================
def quijote_rid_from_filename(filepath: str) -> int:
    """
    从 Quijote 文件名末尾提取 realization id。

    参数
    ----
    filepath : str
        文件路径。

    返回
    ----
    int
        realization id；若匹配失败返回 -1。
    """
    match = re.search(r"(\d+)\.\w+$", os.path.basename(filepath))
    return int(match.group(1)) if match else -1


def load_pk_quijote(pattern: str, n_max: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    读取 Quijote 功率谱 realization。

    参数
    ----
    pattern : str
        文件通配路径。
    n_max : int
        最多读取多少个 realization。

    返回
    ----
    kcen, kmin_b, kmax_b : ndarray
        有效的 k-bin 网格。
    mocks : ndarray
        realization 矩阵。
    mean, std : ndarray
        均值与样本标准差。
    int
        实际读取数量。
    """
    filepaths = sorted(glob.glob(pattern), key=quijote_rid_from_filename)[:n_max]
    if not filepaths:
        raise FileNotFoundError(pattern)

    ref = np.loadtxt(filepaths[0], comments="#")
    nmode = ref[:, 4]
    valid = nmode > 0
    kcen = ref[valid, 0]
    kmin_b = ref[valid, 1]
    kmax_b = ref[valid, 2]

    values = []
    for filepath in filepaths:
        try:
            arr = np.loadtxt(filepath, comments="#")
        except Exception:
            continue
        values.append(arr[valid, 5])

    mocks = np.array(values, dtype=float)
    return kcen, kmin_b, kmax_b, mocks, mocks.mean(axis=0), mocks.std(axis=0, ddof=1), len(mocks)


def load_pcf_quijote(pattern: str, n_max: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    读取 Quijote 2PCF realization。

    参数
    ----
    pattern : str
        文件通配路径。
    n_max : int
        最多读取多少个 realization。

    返回
    ----
    s : ndarray
        r 网格。
    mean_xi, std_xi : ndarray
        2PCF 的均值与样本标准差。
    int
        实际读取数量。
    """
    filepaths = sorted(glob.glob(pattern), key=quijote_rid_from_filename)[:n_max]
    if not filepaths:
        raise FileNotFoundError(pattern)

    ref = np.loadtxt(filepaths[0], comments="#")
    s = ref[:, 0]
    mocks = np.array([np.loadtxt(filepath, comments="#")[:, 3] for filepath in filepaths], dtype=float)
    return s, mocks.mean(axis=0), mocks.std(axis=0, ddof=1), len(mocks)


# ============================================================
# 业务函数
# ============================================================
def load_trimmed_fastpm_pk(cfg: BoxConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    读取并裁剪 fastPM 功率谱，跳过前面的空 bin。

    参数
    ----
    cfg : BoxConfig
        单个 fastPM 数据集配置。

    返回
    ----
    kcen, kmin_b, kmax_b : ndarray
        最终用于拟合的 k-bin 网格。
    pk_mocks, cov_mocks : ndarray
        数据 realization 与协方差 realization。
    """
    kcen0, kmin0, kmax0, pk_mocks0, _, _ = load_pk_multipoles(
        cfg.pk_data_glob, cfg.rid_min, cfg.rid_max, cfg.n_dp + 1
    )
    _, _, _, cov_mocks0, _, _ = load_pk_multipoles(
        cfg.pk_cov_glob, cfg.rid_min, cfg.rid_max, cfg.n_dp + 1
    )

    first_valid = 0
    for ib in range(len(kcen0)):
        if float(pk_mocks0[:, ib].mean()) > 0.0:
            first_valid = ib
            break

    sl = slice(first_valid, first_valid + cfg.n_dp)
    return kcen0[sl], kmin0[sl], kmax0[sl], pk_mocks0[:, sl], cov_mocks0[:, sl]


def fit_and_model_dataset(
    title: str,
    box_size: float,
    kcen: np.ndarray,
    kmin_b: np.ndarray,
    kmax_b: np.ndarray,
    pk_mean: np.ndarray,
    cov_mocks: np.ndarray,
    s: np.ndarray,
    xi_mean: np.ndarray,
    xi_std: np.ndarray,
    n_pk: int,
    n_pcf: int,
    cosmo,
    cache_by_l: dict[float, dict[str, np.ndarray]],
    compute_baseline: bool = False,
) -> dict[str, object]:
    """
    对任意单个数据集执行完整的拟合与 2PCF 建模。

    参数
    ----
    title : str
        数据集标题，用于日志和作图。
    box_size : float
        盒长 L。
    kcen, kmin_b, kmax_b : ndarray
        拟合所用的功率谱 k-bin 网格。
    pk_mean : ndarray
        测量功率谱均值。
    cov_mocks : ndarray
        用于构造协方差的 realization 矩阵。
    s, xi_mean, xi_std : ndarray
        2PCF 的 r 网格、均值、标准差。
    n_pk, n_pcf : int
        realization 数量，仅用于记录。
    cosmo :
        cosmoprimo 的 Cosmology 对象。
    cache_by_l : dict
        盒长缓存。
    compute_baseline : bool
        若为 True，则额外计算原始 full discrete sum 的 baseline 曲线。

    返回
    ----
    dict
        作图所需结果：data/model/residual 及拟合信息。
    """
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")
    print(f"  data mocks: pk={n_pk}, pcf={n_pcf}")
    print(f"  fit k-range: [{kcen.min():.5f}, {kcen.max():.5f}] h/Mpc")

    kf = 2.0 * np.pi / box_size
    volume = box_size ** 3
    r2_data = s ** 2 * xi_mean
    r2_err = s ** 2 * xi_std

    print("  [fit] standard desilike ...")
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

    print("  [fit] BinAvgFit ...")
    kmax_fit = float(kmax_b[-1])
    qmax_fit = int(np.floor((kmax_fit / kf) ** 2)) + 1
    gq_fit = gq_enumerate(qmax_fit)
    _, k_shell, g_shell = build_shells_for_kmax(kf, gq_fit, kmax=kmax_fit)
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

    fnl_fit = float(bestfit_binavg["fnl_loc"])
    b1_fit = float(bestfit_binavg["b1"])
    sigmas_fit = float(bestfit_binavg["sigmas"])
    print(f"    best-fit: fnl={fnl_fit:.3f}, b1={b1_fit:.6f}, sigmas={sigmas_fit:.6f}")

    cache = ensure_box_cache(box_size=box_size, cache_by_l=cache_by_l)
    print("  [xi] CachedRebin / FastDiscrete ...")
    pd = eval_pk_dense(cosmo=cosmo, k=cache["kd"], params=bestfit_binavg, p_fix=P_FIX, z=Z)
    t0 = time.time()
    xi_model = xi_cached_rebin(
        s=s,
        g_nz=cache["g_nz"],
        k_eff=cache["k_eff"],
        volume=volume,
        kd=cache["kd"],
        pd=pd,
    )
    dt = time.time() - t0
    r2_model = s ** 2 * xi_model
    ds = (r2_data - r2_model) / r2_err
    mean_abs = float(np.nanmean(np.abs(ds)))
    mean_signed = float(np.nanmean(ds))
    print(f"    xi evaluation: {dt:.4f}s")
    print(f"    mean|Δ/σ|={mean_abs:.4f}, mean(Δ/σ)={mean_signed:+.4f}")

    r2_baseline = None
    ds_baseline = None
    baseline_mean_abs = None
    baseline_mean_signed = None
    if compute_baseline:
        print("  [xi] Baseline full discrete sum ...")
        t0 = time.time()
        xi_baseline = xi_discrete_multi(
            s=s,
            gq=cache["gq"],
            kf=kf,
            V=volume,
            kd=cache["kd"],
            pd_list=[pd],
            databin_edges=None,
            databin_pk=None,
            chunk=500_000,
        )["FD0"]
        print(f"    baseline evaluation: {time.time() - t0:.2f}s")
        r2_baseline = s ** 2 * xi_baseline
        ds_baseline = (r2_data - r2_baseline) / r2_err
        baseline_mean_abs = float(np.nanmean(np.abs(ds_baseline)))
        baseline_mean_signed = float(np.nanmean(ds_baseline))
        print(
            f"    baseline mean|Δ/σ|={baseline_mean_abs:.4f}, "
            f"mean(Δ/σ)={baseline_mean_signed:+.4f}"
        )

    return {
        "title": title,
        "s": s,
        "r2_data": r2_data,
        "r2_err": r2_err,
        "r2_model": r2_model,
        "fnl_fit": fnl_fit,
        "b1_fit": b1_fit,
        "sigmas_fit": sigmas_fit,
        "mean_abs": mean_abs,
        "mean_signed": mean_signed,
        "n_pk": int(n_pk),
        "n_pcf": int(n_pcf),
        "r2_baseline": r2_baseline,
        "ds_baseline": ds_baseline,
        "baseline_mean_abs": baseline_mean_abs,
        "baseline_mean_signed": baseline_mean_signed,
    }


def run_fastpm_dataset(
    cfg: BoxConfig,
    cosmo,
    cache_by_l: dict[float, dict[str, np.ndarray]],
    compute_baseline: bool = False,
) -> dict[str, object]:
    """
    跑单个 fastPM 数据集。

    参数
    ----
    cfg : BoxConfig
        当前 fastPM 配置。
    cosmo :
        cosmoprimo 的 Cosmology 对象。
    cache_by_l : dict
        盒长缓存。
    compute_baseline : bool
        是否额外计算 full discrete sum baseline。

    返回
    ----
    dict
        作图结果。
    """
    kcen, kmin_b, kmax_b, pk_mocks, cov_mocks = load_trimmed_fastpm_pk(cfg)
    pk_mean = pk_mocks.mean(axis=0)
    s, xi_mean, xi_std = load_pcf(cfg.pcf_glob, cfg.rid_min, cfg.rid_max)
    return fit_and_model_dataset(
        title=cfg.title,
        box_size=float(cfg.L),
        kcen=kcen,
        kmin_b=kmin_b,
        kmax_b=kmax_b,
        pk_mean=pk_mean,
        cov_mocks=cov_mocks,
        s=s,
        xi_mean=xi_mean,
        xi_std=xi_std,
        n_pk=pk_mocks.shape[0],
        n_pcf=cfg.rid_max - cfg.rid_min + 1,
        cosmo=cosmo,
        cache_by_l=cache_by_l,
        compute_baseline=compute_baseline,
    )


def run_quijote_dataset(
    fnl_true: int,
    label: str,
    pk_pattern: str,
    pcf_pattern: str,
    cov_mocks_quijote: np.ndarray,
    cosmo,
    cache_by_l: dict[float, dict[str, np.ndarray]],
) -> dict[str, object]:
    """
    跑单个 Quijote 数据集。

    参数
    ----
    fnl_true : int
        数据集真实 fnl。
    label : str
        标签，仅用于日志。
    pk_pattern, pcf_pattern : str
        Quijote 数据路径。
    cov_mocks_quijote : ndarray
        Quijote 的协方差 realization，统一使用 fid(fn0)。
    cosmo :
        cosmoprimo 的 Cosmology 对象。
    cache_by_l : dict
        盒长缓存。

    返回
    ----
    dict
        作图结果。
    """
    title = f"Quijote fnl={fnl_true}"
    print(f"\n[{label}]")

    kcen_all, kmin_all, kmax_all, _, pk_mean_all, _, n_pk = load_pk_quijote(pk_pattern)
    s, xi_mean, xi_std, n_pcf = load_pcf_quijote(pcf_pattern)

    n_dp = min(QUIJOTE_FIT_NDP, len(kcen_all), cov_mocks_quijote.shape[1])
    return fit_and_model_dataset(
        title=title,
        box_size=1000.0,
        kcen=kcen_all[:n_dp],
        kmin_b=kmin_all[:n_dp],
        kmax_b=kmax_all[:n_dp],
        pk_mean=pk_mean_all[:n_dp],
        cov_mocks=cov_mocks_quijote[:, :n_dp],
        s=s,
        xi_mean=xi_mean,
        xi_std=xi_std,
        n_pk=n_pk,
        n_pcf=n_pcf,
        cosmo=cosmo,
        cache_by_l=cache_by_l,
    )


def draw_combined(
    results: list[dict[str, object]],
    out_path: str,
    figure_title: str,
    model_label: str = "BinAvgFit + FastDiscrete",
) -> None:
    """
    把一组 validation 结果画成总图。

    参数
    ----
    results : list[dict]
        某一类数据集的结果列表。
    out_path : str
        输出 PDF 路径。
    figure_title : str
        总图标题。
    """
    n_col = len(results)
    if n_col == 0:
        return

    fig, axes = plt.subplots(
        2,
        n_col,
        figsize=(5.0 * n_col + 1.0, 8.0),
        sharex=False,
        gridspec_kw={"height_ratios": [3.2, 1.5], "hspace": 0.08, "wspace": 0.20},
    )
    if n_col == 1:
        axes = np.array(axes).reshape(2, 1)

    top_axes = axes[0]
    bot_axes = axes[1]

    for idx, result in enumerate(results):
        ax = top_axes[idx]
        ax_res = bot_axes[idx]

        s = result["s"]
        r2_data = result["r2_data"]
        r2_err = result["r2_err"]
        r2_model = result["r2_model"]
        ds = (r2_data - r2_model) / r2_err
        r2_baseline = result.get("r2_baseline")
        ds_baseline = result.get("ds_baseline")

        ax.errorbar(
            s,
            r2_data,
            yerr=r2_err,
            fmt="o",
            ms=3.5,
            lw=0.9,
            capsize=2.0,
            color="black",
            ecolor="black",
            label="Measured mean",
            zorder=10,
        )
        if r2_baseline is not None:
            ax.plot(
                s,
                r2_baseline,
                color="tab:blue",
                lw=1.8,
                ls="--",
                label="Baseline",
                zorder=7,
            )
        ax.plot(
            s,
            r2_model,
            color="#c62828",
            lw=2.2,
            label=model_label,
            zorder=8,
        )
        ax.set_title(result["title"])
        if idx == 0:
            ax.set_ylabel(r"$r^2 \xi_0(r)$")
        ax.grid(alpha=0.28, ls="--")
        ax.tick_params(axis="x", labelbottom=False)

        ax_res.axhline(0.0, color="black", lw=0.9)
        ax_res.axhline(1.0, color="gray", ls=":", lw=0.8)
        ax_res.axhline(-1.0, color="gray", ls=":", lw=0.8)
        ax_res.axhline(2.0, color="#d98c00", ls=":", lw=0.8)
        ax_res.axhline(-2.0, color="#d98c00", ls=":", lw=0.8)
        if ds_baseline is not None:
            ax_res.plot(
                s,
                ds_baseline,
                color="tab:blue",
                lw=1.3,
                ls="--",
                marker="o",
                ms=2.2,
            )
        ax_res.plot(
            s,
            ds,
            color="#c62828",
            lw=1.6,
            marker="o",
            ms=2.8,
        )
        ax_res.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        if idx == 0:
            ax_res.set_ylabel(r"$(D-M)/\sigma$")
        ax_res.grid(alpha=0.28, ls="--")
        ax_res.set_ylim(-3.0, 3.0)

    if figure_title:
        fig.suptitle(figure_title, fontsize=15, y=0.98)
    handles, labels = top_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        ncol=2,
        fontsize=22,
        bbox_to_anchor=(0.5, 0.955 if figure_title else 0.998),
    )
    fig.subplots_adjust(
        left=0.04,
        right=0.995,
        bottom=0.10,
        top=0.88 if figure_title else 0.90,
        wspace=0.20,
        hspace=0.08,
    )
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """
    主函数。

    执行逻辑
    --------
    1. 初始化宇宙学对象；
    2. 跑 fastPM 四组 validation 并输出总图 PDF；
    3. 跑 Quijote 六组 validation 并输出总图 PDF。
    """
    t_start = time.time()

    from cosmoprimo import Cosmology

    cosmo = Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    cache_by_l: dict[float, dict[str, np.ndarray]] = {}

    fastpm_results: list[dict[str, object]] = []
    for cfg in FASTPM_CONFIGS:
        fastpm_results.append(
            run_fastpm_dataset(
                cfg=cfg,
                cosmo=cosmo,
                cache_by_l=cache_by_l,
                compute_baseline=False,
            )
        )

    fastpm_pdf = os.path.join(OUT_DIR, "fastpm_2pcf_validation_mission11.pdf")
    draw_combined(
        results=fastpm_results,
        out_path=fastpm_pdf,
        figure_title="",
        model_label="2PCF model",
    )
    print(f"\n[OK] saved combined figure: {fastpm_pdf}")

    print(f"\n{'=' * 68}\nLoading Quijote covariance (fnl=0)\n{'=' * 68}")
    _, _, _, cov_mocks_quijote, _, _, n_cov = load_pk_quijote(QUIJOTE_COV_PATTERN)
    print(f"  covariance files: {n_cov}")

    quijote_results: list[dict[str, object]] = []
    for fnl_true, label, pk_pattern, pcf_pattern in QUIJOTE_FNL_CONFIGS:
        try:
            quijote_results.append(
                run_quijote_dataset(
                    fnl_true=fnl_true,
                    label=label,
                    pk_pattern=pk_pattern,
                    pcf_pattern=pcf_pattern,
                    cov_mocks_quijote=cov_mocks_quijote,
                    cosmo=cosmo,
                    cache_by_l=cache_by_l,
                )
            )
        except FileNotFoundError as exc:
            print(f"  [skip] {exc}")

    quijote_pdf = os.path.join(OUT_DIR, "quijote_2pcf_validation_mission11.pdf")
    draw_combined(
        results=quijote_results,
        out_path=quijote_pdf,
        figure_title="Quijote Validation with BinAvgFit + FastDiscrete",
    )
    print(f"[OK] saved combined figure: {quijote_pdf}")
    print(f"Total elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
