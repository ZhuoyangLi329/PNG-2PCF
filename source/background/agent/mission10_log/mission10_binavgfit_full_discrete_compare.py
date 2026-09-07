#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission 10: 解决 FullDiscrete vs DataBin 的“测量依赖”问题
=========================================================

目标
----
任务书第 10 问本质是：
    DataBin 方法在低 k 直接用测量的 P_data_bin(i) 去替换 P_model(k_q)，
    所以它不是纯 forward model（没有 P_measure 就没法做）。

本脚本实现一种“按我在 Mission10 笔记里的想法”的替代方案：

    (A) BinAvgFit（拟合阶段改造）:
        用离散模式计数 g_q 对理论 P_model(k) 做 bin-average，
        用 P_model_bin(i) 去拟合测量的 P_data_bin(i)，而不是用 P_model(k_center,i)。

    (B) FullDiscrete（预测阶段不喂数据）:
        用 BinAvgFit 得到的 best-fit 参数产生 P_model(k)，
        再用标准 FullDiscrete 求和计算 ξ0(r)。

这样得到的 ξ0(r) 是纯 forward model（只依赖参数和盒长 L），
但它应当在数值上更接近 DataBin 的效果（因为它从源头消除了 bin-average vs 点值的不自洽）。

输出
----
在同一张图里对比（每个盒子一列，上 r²ξ，下残差）：
    - FullDiscrete（标准拟合参数）
    - DataBin（原方法，低 k 用 P_data_bin）
    - BinAvgFit+FullDiscrete（新方法，不用 P_measure 参与 pk->2pcf）

运行方式（需要 desilike 环境）
------------------------------
bash -lc "conda activate desilike && PYTHONUNBUFFERED=1 python -u mission10_binavgfit_full_discrete_compare.py"

注意
----
1) 本脚本不会删除任何文件。
2) 若出现模块缺失，请先不要安装包，先确认是否在其他 conda 环境里已有。
"""

from __future__ import annotations

import os
import glob
import re
import time
from dataclasses import dataclass

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.fft import next_fast_len, rfft, irfft


OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# I/O 小工具
# ============================================================
def _rid_from_filename(fp: str) -> int:
    """从文件名里提取 realization id: ...N{rid}... -> rid。

    参数
    ----
    fp : str
        文件路径。

    返回
    ----
    int
        rid；若未匹配到返回 -1。
    """
    m = re.search(r"N(\d+)", os.path.basename(fp))
    return int(m.group(1)) if m else -1


def load_pk_multipoles(
    pattern: str,
    rid_min: int,
    rid_max: int,
    n_dp: int,
    p_col: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取一批 pk 文件，并返回前 n_dp 个 bin 的统计量。

    约定：pk 文件列格式与 mission9 代码一致：
        col0=k_center, col1=k_min, col2=k_max, col5=P0

    返回
    ----
    kcen, kmin, kmax : ndarray (n_dp,)
    mocks : ndarray (n_mocks, n_dp)
    mean, std : ndarray (n_dp,)
    """
    fps = [f for f in sorted(glob.glob(pattern), key=_rid_from_filename)
           if rid_min <= _rid_from_filename(f) <= rid_max]
    if not fps:
        raise FileNotFoundError(f"PK 文件未找到: {pattern}")
    ref = np.loadtxt(fps[0], comments="#")
    kcen = ref[:n_dp, 0].copy()
    kmin = ref[:n_dp, 1].copy()
    kmax = ref[:n_dp, 2].copy()
    mocks = np.array([np.loadtxt(f, comments="#")[:n_dp, p_col] for f in fps], dtype=float)
    mean = mocks.mean(axis=0)
    std = mocks.std(axis=0, ddof=1)
    return kcen, kmin, kmax, mocks, mean, std


def load_pcf(
    pattern: str,
    rid_min: int,
    rid_max: int,
    xi_col: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取一批 2PCF 文件，返回 r 与 xi 的均值和标准差。

    返回
    ----
    s : ndarray (n_r,)
    mean_xi : ndarray (n_r,)
    std_xi : ndarray (n_r,)
    """
    fps = [f for f in sorted(glob.glob(pattern), key=_rid_from_filename)
           if rid_min <= _rid_from_filename(f) <= rid_max]
    if not fps:
        raise FileNotFoundError(f"PCF 文件未找到: {pattern}")
    ref = np.loadtxt(fps[0], comments="#")
    s = ref[:, 0].copy()
    mocks = np.array([np.loadtxt(f, comments="#")[:, xi_col] for f in fps], dtype=float)
    return s, mocks.mean(axis=0), mocks.std(axis=0, ddof=1)


# ============================================================
# 数学/数值工具
# ============================================================
def j0(x: np.ndarray) -> np.ndarray:
    """球贝塞尔 j0(x)=sin(x)/x，数值上对 x=0 做极限处理。"""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    m = x != 0
    out[m] = np.sin(x[m]) / x[m]
    return out


def met_mean_abs_sigma(r2_data: np.ndarray, r2_std: np.ndarray, r2_model: np.ndarray) -> tuple[float, float]:
    """返回两个简单指标：mean(|Δ/σ|) 和 mean(Δ/σ)。"""
    ds = (r2_data - r2_model) / r2_std
    return float(np.nanmean(np.abs(ds))), float(np.nanmean(ds))


def gq_fft(qmax: int, nmax: int) -> np.ndarray:
    """用 FFT 卷积精确计算 g_q（壳层简并度）。

    算法同 Mission 8/9：
        a[0]=1, a[n^2]=2 (n>=1)
        g = IFFT( FFT(a)^3 ) 取前 qmax+1

    参数
    ----
    qmax : int
        最大 q。
    nmax : int
        模索引最大值，通常取 floor(kmax/kf)。

    返回
    ----
    gq : ndarray (qmax+1,), int64
        gq[q] = # { (nx,ny,nz) ∈ Z^3, nx^2+ny^2+nz^2=q }，包含正负与排列。
    """
    a = np.zeros(qmax + 1, dtype=np.float32)
    a[0] = 1.0
    sq = (np.arange(1, nmax + 1, dtype=np.int64) ** 2)
    sq = sq[sq <= qmax]
    a[sq] = 2.0

    nfft = next_fast_len(3 * qmax + 1)
    # 估计仅供参考：这里打印的是 real array 的体积，FFT 中间量会更大
    print(f"  [g_q] FFT conv: qmax={qmax}, nfft={nfft} (a ~ {a.nbytes/1e9:.2f} GB)")
    t0 = time.time()
    fa = rfft(a, n=nfft)
    conv = irfft(fa * fa * fa, n=nfft)
    g = np.rint(conv[:qmax + 1]).astype(np.int64)
    print(f"  [g_q] done in {time.time()-t0:.1f}s")
    return g


def gq_enumerate(qmax: int) -> np.ndarray:
    """用直接枚举计算小 qmax 的 g_q。

    适用场景：
    - BinAvgFit 只需要覆盖拟合的最低 ~20 个 k-bin，qmax 很小（3Gpc 约 10^3 量级）。
    - 这种情况下用枚举比 FFT 更快、更省内存。
    """
    nmax = int(np.ceil(np.sqrt(qmax))) + 1
    gq = np.zeros(qmax + 1, dtype=np.int64)
    for nx in range(-nmax, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            for nz in range(-nmax, nmax + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                q = nx * nx + ny * ny + nz * nz
                if q <= qmax:
                    gq[q] += 1
    return gq


def build_shells_for_kmax(kf: float, gq: np.ndarray, kmax: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 g_q 数组转换成 (q_list, k_list, g_list)，只保留 k<=kmax 的非零壳层。"""
    q_nz = np.nonzero(gq[1:])[0] + 1
    k_nz = kf * np.sqrt(q_nz.astype(float))
    m = k_nz <= kmax
    q = q_nz[m]
    k = k_nz[m]
    g = gq[q].astype(np.int64)
    return q, k, g


def build_bin_shell_index(
    k_shell: np.ndarray,
    g_shell: np.ndarray,
    kmin_bin: np.ndarray,
    kmax_bin: np.ndarray,
) -> list[np.ndarray]:
    """对每个 k-bin，找到其中包含的 shell 索引（在 k_shell 数组中的位置）。

    返回
    ----
    idx_per_bin : list[array]
        idx_per_bin[i] 给出第 i 个 bin 中所有 shell 的索引。
    """
    idx_per_bin: list[np.ndarray] = []
    for i in range(len(kmin_bin)):
        m = (k_shell >= kmin_bin[i]) & (k_shell < kmax_bin[i])
        idx_per_bin.append(np.nonzero(m)[0])
    # sanity：理论上每个 bin 应该至少包含一个 shell（否则该 bin 真实测量也可能为空）
    return idx_per_bin


def binavg_pk_from_shells(
    pk_shell: np.ndarray,
    g_shell: np.ndarray,
    idx_per_bin: list[np.ndarray],
) -> np.ndarray:
    """把 P(k_shell) 做成每个 bin 的离散模式加权平均。"""
    out = np.zeros(len(idx_per_bin), dtype=float)
    for i, idx in enumerate(idx_per_bin):
        if idx.size == 0:
            out[i] = np.nan
            continue
        w = g_shell[idx].astype(float)
        out[i] = float(np.sum(w * pk_shell[idx]) / np.sum(w))
    return out


def xi_discrete_multi(
    s: np.ndarray,
    gq: np.ndarray,
    kf: float,
    V: float,
    kd: np.ndarray,
    pd_list: list[np.ndarray],
    databin_edges: np.ndarray | None = None,
    databin_pk: np.ndarray | None = None,
    chunk: int = 500_000,
) -> dict[str, np.ndarray]:
    """一次离散求和同时计算多条 ξ0(r) 曲线，尽量复用 j0(kr) 的计算。

    参数
    ----
    s : ndarray
        r 网格（pcf 文件里的 s）。
    gq : ndarray
        壳层简并度数组，长度 qmax+1。
    kf : float
        基模 2π/L。
    V : float
        体积 L^3。
    kd, pd_list :
        用于插值的密集 k 网格与对应的 P(k) 数组列表。
        pd_list 的每一项对应一条 FullDiscrete 曲线（不同 best-fit）。
    databin_edges, databin_pk :
        若给定，则额外计算一条 DataBin：
            在 kv < databin_edges[-1] 的范围内，把 P(kv) 替换为对应 bin 的 databin_pk[bin]。
        这里的 databin_pk 通常就是测量 pk_mean（与 Mission9 一致）。

    返回
    ----
    dict
        key:
            "FD0","FD1",... 对应 pd_list 的各条全离散；
            "DataBin" 若 databin_* 提供。
        value: xi0(s) ndarray
    """
    qnz = np.nonzero(gq[1:])[0] + 1
    total = len(qnz)
    xi_fd = [np.zeros(len(s), dtype=float) for _ in pd_list]
    xi_db = np.zeros(len(s), dtype=float) if (databin_edges is not None and databin_pk is not None) else None

    t0 = time.time()
    for i0 in range(0, total, chunk):
        qb = qnz[i0:i0 + chunk]
        kv = kf * np.sqrt(qb.astype(float))
        g = gq[qb].astype(float)

        # 多条 FullDiscrete: pv_j = interp(kv, kd, pd_j)
        pv_list = [np.interp(kv, kd, pd) for pd in pd_list]

        # DataBin：在低 k bin 里用数据替换（只在 kv < edge_max 时发生）
        if xi_db is not None:
            pv_db = pv_list[0].copy()  # DataBin 的高 k 外推沿用 FD0（标准 best-fit）
            # bin_idx: [-1,0,...,nbins-1,nbins]，我们只替换 0..nbins-1
            bin_idx = np.searchsorted(databin_edges, kv, side="right") - 1
            m = (bin_idx >= 0) & (bin_idx < len(databin_pk))
            pv_db[m] = databin_pk[bin_idx[m]]

        # 共同权重
        w_fd = [g * pv for pv in pv_list]
        w_stack = np.stack(w_fd, axis=0)  # (n_model, n_k)
        if xi_db is not None:
            w_stack = np.concatenate([w_stack, (g * pv_db)[None, :]], axis=0)

        # 分 r 分块，避免一次 outer 过大
        for js in range(0, len(s), 10):
            seg = s[js: js + 10]
            J = j0(np.outer(kv, seg))  # (n_k, n_rseg)
            contrib = w_stack @ J      # (n_model(+1), n_rseg)
            # 写回
            for im in range(len(pv_list)):
                xi_fd[im][js: js + len(seg)] += contrib[im]
            if xi_db is not None:
                xi_db[js: js + len(seg)] += contrib[-1]

        if (i0 // chunk) % 10 == 0:
            print(f"    [xi] {min(i0 + chunk, total)}/{total} (elapsed {time.time()-t0:.0f}s)")

    out: dict[str, np.ndarray] = {}
    for i, xi in enumerate(xi_fd):
        out[f"FD{i}"] = xi / V
    if xi_db is not None:
        out["DataBin"] = xi_db / V
    return out


# ============================================================
# 业务逻辑：BinAvgFit + 对比绘图
# ============================================================
@dataclass
class BoxConfig:
    """单个盒子（一个数据集）的配置。"""

    title: str
    L: float
    pk_data_glob: str
    pk_cov_glob: str
    pcf_glob: str
    rid_min: int
    rid_max: int
    n_dp: int


def fit_standard_desilike(
    cosmo,
    kcen: np.ndarray,
    kmin_b: np.ndarray,
    kmax_b: np.ndarray,
    pk_mean: np.ndarray,
    cov_mocks: np.ndarray,
    p_fix: float,
    z: float,
):
    """复用 Mission 9 的做法：用 desilike 在 bin center 处拟合 best-fit。"""
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler
    from pypower import PowerSpectrumStatistics

    edges = np.concatenate([kmin_b, [kmax_b[-1]]])
    nm = 4.0 / 3.0 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    data_ps = PowerSpectrumStatistics(
        edges=edges,
        modes=kcen,
        power_nonorm=np.array([pk_mean]),
        nmodes=nm,
        ells=[0],
        shotnoise_nonorm=0.0,
        statistic="multipole",
    )
    mock_list = []
    for i in range(cov_mocks.shape[0]):
        t = data_ps.deepcopy()
        t.power_nonorm.flat[...] = np.array([cov_mocks[i]]).ravel()
        mock_list.append(t)

    tmpl = FixedPowerSpectrumTemplate(z=z, fiducial=cosmo)
    th = PNGTracerPowerSpectrumMultipoles(template=tmpl, mode="b-p")
    th.init.params["p"].update(fixed=True, value=p_fix)
    th.init.params["sn0"].update(fixed=True, value=0.0)
    th.init.params["sigmas"].update(fixed=False, value=0.0)

    obs = TracerPowerSpectrumMultipolesObservable(
        data=data_ps,
        covariance=mock_list,
        klim={0: [float(kcen.min()), float(kcen.max()), float(kcen[1] - kcen[0])]},
        theory=th,
    )

    like = ObservablesGaussianLikelihood(observables=[obs])
    _ = like()
    like.all_params["p"].update(fixed=True, value=p_fix)
    like.all_params["sn0"].update(fixed=True, value=0.0)
    like.all_params["sigmas"].update(fixed=False, value=0.0)

    prof = MinuitProfiler(like, seed=66)
    profiles = prof.maximize(niterations=27)
    return profiles.bestfit.choice(input=True)


def fit_binavg_minuit(
    cosmo,
    base_params: dict,
    k_shell: np.ndarray,
    g_shell: np.ndarray,
    idx_per_bin: list[np.ndarray],
    pk_data: np.ndarray,
    cov_mocks: np.ndarray,
    p_fix: float,
    z: float,
) -> dict:
    """用 iminuit 最小化基于 bin-average theory 的 χ²，得到新的 best-fit。

    这里我们只让 (fnl_loc, b1, sigmas) 三个参数自由，其余参数固定在 base_params 里。
    """
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from iminuit import Minuit

    # 协方差矩阵来自 mocks（与数据定义一致：都是 bin-average 的 P）
    cov = np.cov(cov_mocks, rowvar=False, ddof=1)
    cov_inv = np.linalg.pinv(cov, rcond=1e-10)

    # 只在 shell_k 上评估理论，避免每次都算 300k 的 kd
    tmpl = FixedPowerSpectrumTemplate(z=z, fiducial=cosmo)
    th = PNGTracerPowerSpectrumMultipoles(k=k_shell, template=tmpl, mode="b-p")
    th.init.params["p"].update(fixed=True, value=p_fix)
    th.init.params["sn0"].update(fixed=True, value=0.0)
    th.init.params["sigmas"].update(fixed=False, value=float(base_params.get("sigmas", 0.0)))

    def chi2(fnl_loc: float, b1: float, sigmas: float) -> float:
        # 基于标准 best-fit 的参数字典，只替换 3 个自由参
        params = dict(base_params)
        params["fnl_loc"] = float(fnl_loc)
        params["b1"] = float(b1)
        params["sigmas"] = float(sigmas)
        params["p"] = float(p_fix)
        params["sn0"] = 0.0

        # 评估 P_model(k_shell) -> bin-average
        th(**params)
        pk_shell = np.array(th.power[0], dtype=float)
        pk_bin = binavg_pk_from_shells(pk_shell, g_shell=g_shell, idx_per_bin=idx_per_bin)
        diff = pk_data - pk_bin
        return float(diff @ cov_inv @ diff)

    m = Minuit(
        chi2,
        fnl_loc=float(base_params.get("fnl_loc", 0.0)),
        b1=float(base_params.get("b1", 2.0)),
        sigmas=float(base_params.get("sigmas", 0.0)),
    )
    m.errordef = 1.0
    # 非物理区域的简单限制
    m.limits["b1"] = (0.0, None)
    m.limits["sigmas"] = (0.0, None)
    m.limits["fnl_loc"] = (-2000.0, 2000.0)

    m.migrad()
    if not m.fmin.is_valid:
        print("  [BinAvgFit] WARNING: migrad not valid, still using current values")

    out = dict(base_params)
    out["fnl_loc"] = float(m.values["fnl_loc"])
    out["b1"] = float(m.values["b1"])
    out["sigmas"] = float(m.values["sigmas"])
    out["p"] = float(p_fix)
    out["sn0"] = 0.0
    return out


def eval_pk_dense(
    cosmo,
    k: np.ndarray,
    params: dict,
    p_fix: float,
    z: float,
) -> np.ndarray:
    """在密集 k 网格上计算 P_model(k)。"""
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles

    tmpl = FixedPowerSpectrumTemplate(z=z, fiducial=cosmo)
    th = PNGTracerPowerSpectrumMultipoles(k=k, template=tmpl, mode="b-p")
    th.init.params["p"].update(fixed=True, value=p_fix)
    th.init.params["sn0"].update(fixed=True, value=0.0)
    th.init.params["sigmas"].update(fixed=False, value=float(params.get("sigmas", 0.0)))

    p = dict(params)
    p["p"] = float(p_fix)
    p["sn0"] = 0.0
    th(**p)
    return np.array(th.power[0], dtype=float)


def main() -> None:
    t_start = time.time()

    # ------------------------------------------------------------
    # 固定的建模设置（与 Mission 9 保持一致）
    # ------------------------------------------------------------
    Z = 1.0
    P_FIX = 1.2
    KMAX_DISCRETE = 15.0
    N_DENSE = 300_000

    from cosmoprimo import Cosmology

    cosmo = Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )

    # FastPM tests: 1Gpc/3Gpc, fnl=100/0
    # 说明：
    # - 对 1Gpc，fnl100/fnl0 的文件名不带 "fnlxxx"，通过目录区分；
    # - 对 3Gpc，文件名里带 fnl100/fnl0。
    configs = [
        BoxConfig(
            title="3Gpc fnl100",
            L=3000.0,
            pk_data_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat",
            pk_cov_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
            pcf_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",
            rid_min=2,
            rid_max=99,
            n_dp=20,
        ),
        BoxConfig(
            title="3Gpc fnl0",
            L=3000.0,
            pk_data_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
            pk_cov_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
            pcf_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat",
            rid_min=2,
            rid_max=99,
            n_dp=20,
        ),
        BoxConfig(
            title="1Gpc fnl100",
            L=1000.0,
            pk_data_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat",
            pk_cov_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
            pcf_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat",
            rid_min=1,
            rid_max=50,
            n_dp=20,
        ),
        BoxConfig(
            title="1Gpc fnl0",
            L=1000.0,
            pk_data_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
            pk_cov_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
            pcf_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut_fnl0/pcf_rsd_N*.dat",
            rid_min=1,
            rid_max=50,
            n_dp=20,
        ),
    ]

    n_cfg = len(configs)
    # 2 x N figure：上 r²ξ，下残差；每列一个数据集
    fig, axes = plt.subplots(2, n_cfg, figsize=(6 * n_cfg, 10))
    if n_cfg == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle("Mission 10: BinAvgFit+FullDiscrete vs DataBin vs FullDiscrete (FastPM)", fontsize=15, y=0.98)

    # 缓存：同一 Lbox 的 g_q / kd 重复使用（fnl100 与 fnl0 共用）
    gq_cache: dict[float, np.ndarray] = {}
    kd_cache: dict[float, np.ndarray] = {}

    for col, cfg in enumerate(configs):
        print(f"\n{'='*70}\n{cfg.title}\n{'='*70}")

        L = float(cfg.L)
        kf = 2.0 * np.pi / L
        V = L ** 3

        # --------------------------------------------------------
        # 1) 读入 pk / pcf 数据
        # --------------------------------------------------------
        # 这里与 Mission9 的处理保持一致：多读 1 个点，用 first_valid 跳过“空 bin”
        kc0, km0, kx0, pk_mocks0, pk_mean0, _pk_std0 = load_pk_multipoles(
            cfg.pk_data_glob, cfg.rid_min, cfg.rid_max, cfg.n_dp + 1
        )
        _kc, _km, _kx, cov_mocks0, _cov_mean0, _cov_std0 = load_pk_multipoles(
            cfg.pk_cov_glob, cfg.rid_min, cfg.rid_max, cfg.n_dp + 1
        )

        first_valid = 0
        for ib in range(len(kc0)):
            if float(pk_mocks0[:, ib].mean()) > 0:
                first_valid = ib
                break

        kcen = kc0[first_valid: first_valid + cfg.n_dp]
        kmin_b = km0[first_valid: first_valid + cfg.n_dp]
        kmax_b = kx0[first_valid: first_valid + cfg.n_dp]
        pk_mocks = pk_mocks0[:, first_valid: first_valid + cfg.n_dp]
        cov_mocks = cov_mocks0[:, first_valid: first_valid + cfg.n_dp]
        pk_mean = pk_mocks.mean(axis=0)

        s, xi_mean, xi_std = load_pcf(cfg.pcf_glob, cfg.rid_min, cfg.rid_max)
        r2_data = s**2 * xi_mean
        r2_std = s**2 * xi_std
        print(f"  pk mocks: {pk_mocks.shape[0]}  pcf mocks: (mean/std from {cfg.rid_max-cfg.rid_min+1} files)")

        # DataBin 使用的边界（要求 bins 连续）
        edges = np.concatenate([kmin_b, [kmax_b[-1]]])

        # --------------------------------------------------------
        # 2) 标准 desilike 拟合（bin center）
        # --------------------------------------------------------
        print("  [Fit] 标准 desilike (bin center) ...")
        bf_std = fit_standard_desilike(
            cosmo=cosmo,
            kcen=kcen,
            kmin_b=kmin_b,
            kmax_b=kmax_b,
            pk_mean=pk_mean,
            cov_mocks=cov_mocks,
            p_fix=P_FIX,
            z=Z,
        )
        print(f"    best-fit(std): fnl={float(bf_std.get('fnl_loc', np.nan)):.3f}, "
              f"b1={float(bf_std.get('b1', np.nan)):.6f}, sigmas={float(bf_std.get('sigmas', np.nan)):.6f}")

        # --------------------------------------------------------
        # 3) 新方法：BinAvgFit（bin-average theory）
        # --------------------------------------------------------
        print("  [Fit] BinAvgFit (discrete mode-weighted bin average) ...")
        # 只需要覆盖拟合区间的 qmax，直接枚举计算 g_q 更快
        kmax_fit = float(kmax_b[-1])
        qmax_fit = int(np.floor((kmax_fit / kf) ** 2)) + 1
        gq_fit = gq_enumerate(qmax_fit)
        _q_shell, k_shell, g_shell = build_shells_for_kmax(kf, gq_fit, kmax=kmax_fit)
        idx_per_bin = build_bin_shell_index(k_shell, g_shell, kmin_b, kmax_b)

        bf_binavg = fit_binavg_minuit(
            cosmo=cosmo,
            base_params=bf_std,
            k_shell=k_shell,
            g_shell=g_shell,
            idx_per_bin=idx_per_bin,
            pk_data=pk_mean,
            cov_mocks=cov_mocks,
            p_fix=P_FIX,
            z=Z,
        )
        print(f"    best-fit(binavg): fnl={float(bf_binavg.get('fnl_loc', np.nan)):.3f}, "
              f"b1={float(bf_binavg.get('b1', np.nan)):.6f}, sigmas={float(bf_binavg.get('sigmas', np.nan)):.6f}")

        # --------------------------------------------------------
        # 4) 计算两套 best-fit 的 P_model(k)（密集网格）用于离散求和插值
        # --------------------------------------------------------
        if L in kd_cache:
            kd = kd_cache[L]
        else:
            kd = np.geomspace(kf * 0.5, KMAX_DISCRETE * 1.1, N_DENSE)
            kd_cache[L] = kd
        print("  [Pk] evaluate dense P(k) for std fit ...")
        pd_std = eval_pk_dense(cosmo=cosmo, k=kd, params=bf_std, p_fix=P_FIX, z=Z)
        print("  [Pk] evaluate dense P(k) for binavg fit ...")
        pd_binavg = eval_pk_dense(cosmo=cosmo, k=kd, params=bf_binavg, p_fix=P_FIX, z=Z)

        # --------------------------------------------------------
        # 5) 计算 g_q（用于 ξ0 全离散求和到 kmax=15）
        # --------------------------------------------------------
        if L in gq_cache:
            gq = gq_cache[L]
        else:
            qmax = int((KMAX_DISCRETE / kf) ** 2)
            nmax = int(KMAX_DISCRETE / kf)
            print("  [g_q] compute for FullDiscrete sum ...")
            gq = gq_fft(qmax=qmax, nmax=nmax)
            gq_cache[L] = gq

        # --------------------------------------------------------
        # 6) 一次求和同时得到 3 条曲线：FD(std), FD(binavg), DataBin
        # --------------------------------------------------------
        print("  [xi] FullDiscrete(std), FullDiscrete(binavg), DataBin ...")
        xi_dict = xi_discrete_multi(
            s=s,
            gq=gq,
            kf=kf,
            V=V,
            kd=kd,
            pd_list=[pd_std, pd_binavg],
            databin_edges=edges,
            databin_pk=pk_mean,
            chunk=500_000,
        )
        r2_fd_std = s**2 * xi_dict["FD0"]
        r2_fd_binavg = s**2 * xi_dict["FD1"]
        r2_db = s**2 * xi_dict["DataBin"]

        # --------------------------------------------------------
        # 7) 指标与绘图
        # --------------------------------------------------------
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        ax_top.errorbar(s, r2_data, yerr=r2_std, fmt="ko", ms=3, capsize=2, label="Data mean", zorder=10)

        methods = [
            ("FullDiscrete(std)", r2_fd_std, "tab:green", "-"),
            ("DataBin", r2_db, "darkorange", "-"),
            ("BinAvgFit+FD", r2_fd_binavg, "tab:purple", "-"),
        ]

        print(f"\n  {'Method':<18s} {'mean|Δ/σ|':>10s} {'mean(Δ/σ)':>10s}")
        print(f"  {'-'*44}")
        for name, r2m, _, _ in methods:
            ma, ms = met_mean_abs_sigma(r2_data, r2_std, r2m)
            print(f"  {name:<18s} {ma:>10.4f} {ms:>+10.4f}")

        for name, r2m, c, ls in methods:
            ma, ms = met_mean_abs_sigma(r2_data, r2_std, r2m)
            ax_top.plot(s, r2m, color=c, ls=ls, lw=2.0 if name == "BinAvgFit+FD" else 1.5,
                        label=f"{name} ({ma:.3f}, {ms:+.3f})")

        ax_top.set_title(cfg.title, fontsize=12)
        ax_top.set_ylabel(r"$r^2\xi_0(r)$", fontsize=12)
        ax_top.grid(alpha=0.3)
        ax_top.legend(fontsize=8, title="(mean|Δ/σ|, mean Δ/σ)")

        ax_bot.axhline(0.0, color="k", lw=0.6)
        ax_bot.axhline(1.0, color="gray", ls=":", lw=0.6)
        ax_bot.axhline(-1.0, color="gray", ls=":", lw=0.6)
        for name, r2m, c, ls in methods:
            ds = (r2_data - r2m) / r2_std
            ax_bot.plot(s, ds, color=c, ls=ls, lw=2.0 if name == "BinAvgFit+FD" else 1.2,
                        marker="o", ms=2.0 if name == "BinAvgFit+FD" else 1.5, alpha=0.85)
        ax_bot.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$", fontsize=12)
        ax_bot.set_ylabel(r"$(Data-Model)/\sigma$ of $r^2\xi_0$", fontsize=12)
        ax_bot.grid(alpha=0.3)
        ax_bot.set_ylim(-3, 3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(OUT_DIR, "mission10_fastpm_binavgfit_fd_vs_databin_fulldiscrete.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\n[OK] saved: {out_png}")
    print(f"Total elapsed: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
