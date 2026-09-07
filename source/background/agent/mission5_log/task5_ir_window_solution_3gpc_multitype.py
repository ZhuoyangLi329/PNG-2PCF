#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本名称
--------
task5_ir_window_solution_3gpc.py

代码大纲（执行逻辑）
--------------------
第 0 部分：参数区
- 集中定义任务 5 需要的常数、输入输出路径、拟合区间、FFTLog 参数、扫描参数。

第 1 部分：数据读取与对齐
- 读取 3Gpc fnl=100 / fnl=0 的 pk 与 pcf 文件。
- 在 realization 编号上做交集对齐。
- 计算测量均值与样本标准差，供后续比较。

第 2 部分：P0 best-fit（desilike + Minuit）
- 对测量 P0 均值做低-k 拟合，得到 best-fit 参数。
- 在高分辨率 k 网格缓存 best-fit P0(k)。

第 3 部分：pk -> xi 的 4 种数值方案
- Baseline：k in [k_f, kmax] 的硬截断积分（sharp cut）。
- TestA：k in [kmin_global, kmax]，不加 IR 窗（用于证明问题存在）。
- TestB：k in [kmin_global, kmax]，在 k<k_f 施加 IR 窗。
  默认采用“只对 PNG 增量项加窗”的稳健版本。
- TestC：低-k 离散模式求和 + 高-k 连续 FFTLog（hybrid）。

第 4 部分：指标、图像与稳定性检查
- 指标：large-scale (r>=200) 的 mean|Δ/σ| 与 chi2/ndof。
- 图1：P0 mean vs best-fit。
- 图2：r^2 xi 的 Data/Model 对比。
- 图3：残差 (Data-Model)/σ 对比。
- sanity：fnl=0 下复跑，检查 TestB 是否引入额外偏差。
- 稳定性：TestB 降低 kmin_global 一阶；TestC 扫 k_split。

第 5 部分：实验日志归档
- 输出脚本、PNG、CSV、JSON、TXT 到 mission5_log。

说明
----
1) 本脚本不安装任何新包，默认应在 desilike 环境运行。
2) 所有函数提供中文注释，便于后续复现实验。
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fht, fhtoffset


# =====================
# 0) 参数区
# =====================

# 输入文件（3Gpc，质量统一后的 masscut 样本）
PK_FNL100_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat"
PCF_FNL100_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat"
PK_FNL0_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat"
PCF_FNL0_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat"

# 输出目录（任务要求）
OUT_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission5_log"
os.makedirs(OUT_DIR, exist_ok=True)

# 输出文件
OUT_FIG1_PK = os.path.join(OUT_DIR, "task5_v2_fig1_3gpc_fnl100_pk_mean_vs_bestfit.png")
OUT_FIG2_XI = os.path.join(OUT_DIR, "task5_v2_fig2_3gpc_fnl100_r2xi_compare.png")
OUT_FIG3_RES = os.path.join(OUT_DIR, "task5_v2_fig3_3gpc_fnl100_residual_sigma.png")
OUT_FIG4_SANITY = os.path.join(OUT_DIR, "task5_v2_fig4_3gpc_fnl0_sanity_check.png")

OUT_METRIC_MAIN_CSV = os.path.join(OUT_DIR, "task5_v2_metrics_fnl100_main.csv")
OUT_SCAN_B_CSV = os.path.join(OUT_DIR, "task5_v2_scan_testB_multitype.csv")
OUT_SCAN_C_CSV = os.path.join(OUT_DIR, "task5_v2_scan_testC_ksplit.csv")
OUT_SANITY_CSV = os.path.join(OUT_DIR, "task5_v2_sanity_fnl0_metrics.csv")
OUT_BESTFIT_JSON = os.path.join(OUT_DIR, "task5_v2_bestfit_params.json")
OUT_SUMMARY_TXT = os.path.join(OUT_DIR, "task5_v2_summary.txt")

# realization 范围（默认对齐 mission4 口径，方便直接比较 baseline）
RID_MIN = int(os.getenv("TASK5_RID_MIN", "2"))
RID_MAX = int(os.getenv("TASK5_RID_MAX", "80"))

# 数据列定义
# pk: col0=kcen, col1=kmin, col2=kmax, col5=P0
PK_KCEN_COL = 0
PK_KMIN_COL = 1
PK_KMAX_COL = 2
PK_P0_COL = 5

# pcf: col0=s_cen, col1=s_min, col2=s_max, col3=xi0
PCF_SCEN_COL = 0
PCF_XI0_COL = 3

# 盒长与基础模式
BOX_SIZE = 3000.0
K_FUND = 2.0 * np.pi / BOX_SIZE

# 积分设置（任务清单）
KMIN_GLOBAL = float(os.getenv("TASK5_KMIN_GLOBAL", "1e-4"))
KMAX_INT = 20.0

# P0 拟合设置
# 用户最新要求：将拟合上限从 0.0635 提高到 0.08
PK_FIT_KMAX = 0.08
MINUIT_SEED = 66
MINUIT_NITER = 25

# PNG 理论固定参数（沿用前面任务）
UNIT_Z = 1.0
FIXED_P = 1.1
FIXED_SN0 = 0.0
FIXED_SIGMAS = 0.0

# FFTLog 参数
FFTLOG_N = 4096
FFTLOG_PADDING = 4.0
FFTLOG_MU = 0.5
FFTLOG_BIAS = 0.0

# mission4 同款 log-k 平滑边界窗参数
EDGE_TAPER_FRAC = 0.06

# TestB: 多种窗口类型 + 参数扫描
TESTB_WINDOW_SCANS = [
    ("exp_power", [1, 2, 3, 4, 6, 8, 10]),
    ("rational", [1, 2, 3, 4, 6, 8, 10]),
    ("tanh_log", [1.5, 2.5, 4.0, 6.0, 8.0]),
    ("logcos", [0.3, 0.5, 0.8, 1.0, 1.3]),
]
TESTB_USE_PNG_ONLY_WINDOW = True

# TestC: hybrid 的 k_split = N_split * k_f 扫描
TESTC_NSPLIT_LIST = [3, 4, 6, 8]

# 大尺度区间（量化对齐程度）
LARGE_SCALE_MIN = 200.0

# 稳定性检查
STABILITY_KMIN_FACTOR = 0.1  # 把 kmin_global 再降低 10 倍

# 图像风格
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"


# =====================
# 1) desilike 依赖
# =====================

try:
    from cosmoprimo import Cosmology
    from desilike import setup_logging
    from desilike.theories.galaxy_clustering import (
        FixedPowerSpectrumTemplate,
        PNGTracerPowerSpectrumMultipoles,
    )
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler
except ModuleNotFoundError as exc:
    raise SystemExit(
        "缺少 desilike 依赖，请在 desilike 环境运行。\n"
        f"原始报错: {repr(exc)}"
    )

setup_logging()


# =====================
# 2) 数据结构
# =====================

@dataclass
class MockData:
    """
    保存一个数据集（fnl=100 或 fnl=0）在 realization 对齐后的测量数据。

    参数
    ----
    tag : str
        数据标签，例如 "fnl100"。
    realizations : np.ndarray
        参与统计的 realization 编号。
    kcen, kmin, kmax : np.ndarray
        功率谱网格。
    p0_mocks : np.ndarray
        shape=(Nmock, Nk) 的 P0 样本。
    scen : np.ndarray
        2PCF 的 s 网格。
    xi_mocks : np.ndarray
        shape=(Nmock, Ns) 的 xi0 样本。
    """

    tag: str
    realizations: np.ndarray
    kcen: np.ndarray
    kmin: np.ndarray
    kmax: np.ndarray
    p0_mocks: np.ndarray
    scen: np.ndarray
    xi_mocks: np.ndarray

    @property
    def nmock(self) -> int:
        """mock 数量。"""
        return int(self.p0_mocks.shape[0])

    @property
    def p0_mean(self) -> np.ndarray:
        """P0 均值。"""
        return np.mean(self.p0_mocks, axis=0)

    @property
    def p0_std(self) -> np.ndarray:
        """P0 样本标准差。"""
        return np.std(self.p0_mocks, axis=0, ddof=1)

    @property
    def xi_mean(self) -> np.ndarray:
        """xi0 均值。"""
        return np.mean(self.xi_mocks, axis=0)

    @property
    def xi_std(self) -> np.ndarray:
        """xi0 样本标准差。"""
        return np.std(self.xi_mocks, axis=0, ddof=1)


# =====================
# 3) 基础函数：文件读取与网格检查
# =====================


def parse_realization_id(path: str) -> int:
    """
    从文件名解析 realization 编号。

    参数
    ----
    path : str
        文件路径，要求末尾含 `_Nxx.dat`。

    返回
    ----
    int
        realization 编号。
    """
    m = re.search(r"_N(\d+)\.dat$", os.path.basename(path))
    if m is None:
        raise ValueError(f"无法解析 realization 编号: {path}")
    return int(m.group(1))


def list_realization_files(file_glob: str) -> Dict[int, str]:
    """
    扫描文件并建立 `realization -> path` 映射。

    参数
    ----
    file_glob : str
        glob 匹配模式。

    返回
    ----
    Dict[int, str]
        映射字典。
    """
    files = glob.glob(file_glob)
    if not files:
        raise FileNotFoundError(f"未找到输入文件: {file_glob}")

    mapping: Dict[int, str] = {}
    for fp in files:
        rid = parse_realization_id(fp)
        mapping[rid] = fp
    return mapping


def assert_same_grid(arrays: List[np.ndarray], name: str) -> None:
    """
    检查多个数组是否一致（用于不同 realization 间网格一致性）。

    参数
    ----
    arrays : List[np.ndarray]
        待比较数组列表。
    name : str
        变量名（报错提示使用）。
    """
    ref = arrays[0]
    for i, arr in enumerate(arrays[1:], start=1):
        if not np.allclose(ref, arr):
            raise ValueError(f"{name} 网格不一致：index={i}")


def load_mock_data(tag: str, pk_glob: str, pcf_glob: str, rid_min: int, rid_max: int) -> MockData:
    """
    读取并对齐同一数据集的 pk 与 pcf。

    参数
    ----
    tag : str
        标签，写入返回结构。
    pk_glob, pcf_glob : str
        输入文件匹配模式。
    rid_min, rid_max : int
        realization 取值范围（闭区间）。

    返回
    ----
    MockData
        对齐后的数据结构。
    """
    pk_map = list_realization_files(pk_glob)
    pcf_map = list_realization_files(pcf_glob)

    common = sorted(set(pk_map) & set(pcf_map))
    common = [rid for rid in common if rid_min <= rid <= rid_max]
    if not common:
        raise RuntimeError(f"{tag}: 在范围 [{rid_min},{rid_max}] 内没有共同 realization")

    # 读取 pk
    pk_tables = [np.loadtxt(pk_map[rid], comments="#") for rid in common]
    kcen_list = [arr[:, PK_KCEN_COL] for arr in pk_tables]
    kmin_list = [arr[:, PK_KMIN_COL] for arr in pk_tables]
    kmax_list = [arr[:, PK_KMAX_COL] for arr in pk_tables]
    assert_same_grid(kcen_list, f"{tag}:kcen")
    assert_same_grid(kmin_list, f"{tag}:kmin")
    assert_same_grid(kmax_list, f"{tag}:kmax")

    # 读取 pcf
    pcf_tables = [np.loadtxt(pcf_map[rid], comments="#") for rid in common]
    scen_list = [arr[:, PCF_SCEN_COL] for arr in pcf_tables]
    assert_same_grid(scen_list, f"{tag}:scen")

    return MockData(
        tag=tag,
        realizations=np.asarray(common, dtype=int),
        kcen=kcen_list[0].copy(),
        kmin=kmin_list[0].copy(),
        kmax=kmax_list[0].copy(),
        p0_mocks=np.vstack([arr[:, PK_P0_COL] for arr in pk_tables]),
        scen=scen_list[0].copy(),
        xi_mocks=np.vstack([arr[:, PCF_XI0_COL] for arr in pcf_tables]),
    )


# =====================
# 4) 理论模型：P0 拟合与评估
# =====================


def build_fiducial_cosmology() -> Cosmology:
    """
    构建 UNIT 任务中统一使用的 fiducial cosmology。

    返回
    ----
    Cosmology
        cosmoprimo 宇宙学对象。
    """
    return Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )


def convert_bestfit_to_float_dict(bestfit: Dict[str, object]) -> Dict[str, float]:
    """
    把 desilike 参数对象转为纯 float 字典。

    参数
    ----
    bestfit : Dict[str, object]
        profiler.bestfit.choice(input=True) 的输出。

    返回
    ----
    Dict[str, float]
        仅含 float 的参数字典。
    """
    out: Dict[str, float] = {}
    for key, val in bestfit.items():
        out[key] = float(np.ravel(np.asarray(val))[0])
    return out


def fit_best_pk(data: MockData) -> Dict[str, object]:
    """
    对测量 P0 均值做 best-fit。

    参数
    ----
    data : MockData
        输入数据。

    返回
    ----
    Dict[str, object]
        含 bestfit 参数、拟合区间数组和模型曲线。
    """
    mask = data.kcen <= PK_FIT_KMAX
    if mask.sum() < 5:
        raise RuntimeError(f"{data.tag}: 低-k 可用点太少")

    kfit = data.kcen[mask]
    p0_mean_fit = data.p0_mean[mask]
    p0_mocks_fit = data.p0_mocks[:, mask]

    cosmo = build_fiducial_cosmology()
    template = FixedPowerSpectrumTemplate(z=UNIT_Z, fiducial=cosmo)
    theory = PNGTracerPowerSpectrumMultipoles(template=template, mode="b-p")

    theory.init.params["p"].update(fixed=True, value=FIXED_P)
    theory.init.params["sn0"].update(fixed=True, value=FIXED_SN0)
    theory.init.params["sigmas"].update(fixed=False, value=FIXED_SIGMAS)

    dk = float(np.median(np.diff(kfit)))
    observable = TracerPowerSpectrumMultipolesObservable(
        data=p0_mean_fit,
        covariance=[row for row in p0_mocks_fit],
        klim={0: [float(kfit.min()), float(kfit.max()), dk]},
        k=kfit,
        ells=[0],
        theory=theory,
    )
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    _ = likelihood()

    likelihood.all_params["p"].update(fixed=True, value=FIXED_P)
    likelihood.all_params["sn0"].update(fixed=True, value=FIXED_SN0)
    likelihood.all_params["sigmas"].update(fixed=False, value=FIXED_SIGMAS)

    profiler = MinuitProfiler(likelihood, seed=MINUIT_SEED)
    profiles = profiler.maximize(niterations=MINUIT_NITER)
    bestfit = convert_bestfit_to_float_dict(profiles.bestfit.choice(input=True))

    # 在拟合点上取模型，便于画图
    theory_fit = PNGTracerPowerSpectrumMultipoles(k=kfit, template=template, mode="b-p")
    theory_fit.init.params["p"].update(fixed=True, value=FIXED_P)
    theory_fit.init.params["sn0"].update(fixed=True, value=FIXED_SN0)
    theory_fit.init.params["sigmas"].update(fixed=False, value=FIXED_SIGMAS)
    theory_fit(**bestfit)
    p0_model_fit = np.asarray(theory_fit.power[0], dtype=float)

    return {
        "bestfit": bestfit,
        "kfit": kfit,
        "p0_mean_fit": p0_mean_fit,
        "p0_std_fit": np.std(p0_mocks_fit, axis=0, ddof=1),
        "p0_model_fit": p0_model_fit,
    }


def build_theory_p0(k_grid: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    在给定 k 网格上评估理论 P0(k)。

    参数
    ----
    k_grid : np.ndarray
        目标 k 网格（通常为对数均匀）。
    params : Dict[str, float]
        理论参数字典。

    返回
    ----
    np.ndarray
        P0(k)。
    """
    cosmo = build_fiducial_cosmology()
    template = FixedPowerSpectrumTemplate(z=UNIT_Z, fiducial=cosmo)
    theory = PNGTracerPowerSpectrumMultipoles(k=k_grid, template=template, mode="b-p")
    theory.init.params["p"].update(fixed=True, value=FIXED_P)
    theory.init.params["sn0"].update(fixed=True, value=FIXED_SN0)
    theory.init.params["sigmas"].update(fixed=False, value=FIXED_SIGMAS)
    theory(**params)
    return np.asarray(theory.power[0], dtype=float)


# =====================
# 5) FFTLog 与窗口函数
# =====================


def build_log_taper_window(k_array: np.ndarray, kmin: float, kmax: float, frac: float) -> np.ndarray:
    """
    构造 mission4 同款 log(k) 边界平滑窗：
    - [kmin, kmax] 为有效积分区间；
    - 在低/高边界采用余弦过渡，减少 FFTLog 振铃。

    参数
    ----
    k_array : np.ndarray
        对数等间距 k 网格。
    kmin, kmax : float
        积分有效区间。
    frac : float
        过渡区宽度占总 log 区间的比例。

    返回
    ----
    np.ndarray
        边界平滑窗函数。
    """
    w = np.zeros_like(k_array, dtype=float)
    lk = np.log(k_array)
    l0 = np.log(kmin)
    l1 = np.log(kmax)
    dl = frac * (l1 - l0)

    if dl <= 0:
        w[(k_array >= kmin) & (k_array <= kmax)] = 1.0
        return w

    left = l0 + dl
    right = l1 - dl

    m = (lk >= l0) & (lk < left)
    w[m] = 0.5 * (1.0 - np.cos(np.pi * (lk[m] - l0) / dl))

    m = (lk >= left) & (lk <= right)
    w[m] = 1.0

    m = (lk > right) & (lk <= l1)
    w[m] = 0.5 * (1.0 + np.cos(np.pi * (lk[m] - right) / dl))
    return w


def build_ir_window(k_array: np.ndarray, kf: float, wtype: str, param: float) -> np.ndarray:
    """
    构造多类型 IR 窗函数 W(k)，统一满足：
    - k >= k_f: W=1
    - k -> 0: W->0

    支持窗口类型
    ----------
    exp_power:
        W = [1-exp(-(k/kf)^a)] / [1-exp(-1)]
        参数 param=a (>0)。
    rational:
        W = 2*x^a/(1+x^a), x=k/kf
        参数 param=a (>0)。
    tanh_log:
        W = 0.5*(1+tanh(beta*ln(k/kf)))
        参数 param=beta (>0)。
    logcos:
        在 logk 空间从 k0 到 kf 用余弦过渡：
          k<k0:0,  k0<=k<kf:cosine ramp,  k>=kf:1
        参数 param=delta_decade，表示 k0 = kf * 10^(-delta_decade)。

    参数
    ----
    k_array : np.ndarray
        k 网格。
    kf : float
        基本模式 k_f。
    wtype : str
        窗口类型名。
    param : float
        对应窗口参数。

    返回
    ----
    np.ndarray
        IR 窗函数。
    """
    x = np.maximum(k_array / max(kf, 1e-30), 1e-300)
    w = np.ones_like(k_array, dtype=float)
    m = k_array < kf
    if not np.any(m):
        return w

    p = float(param)
    if wtype == "exp_power":
        norm = 1.0 - np.exp(-1.0)
        w[m] = (1.0 - np.exp(-(x[m] ** p))) / max(norm, 1e-30)

    elif wtype == "rational":
        xa = x[m] ** p
        w[m] = 2.0 * xa / (1.0 + xa)

    elif wtype == "tanh_log":
        w[m] = 0.5 * (1.0 + np.tanh(p * np.log(x[m])))

    elif wtype == "logcos":
        delta = max(p, 1e-6)
        k0 = kf * (10.0 ** (-delta))
        w[m] = 0.0
        mt = (k_array >= k0) & (k_array < kf)
        if np.any(mt):
            lt = np.log(k_array[mt])
            l0 = np.log(k0)
            lf = np.log(kf)
            u = (lt - l0) / max(lf - l0, 1e-30)
            w[mt] = 0.5 * (1.0 - np.cos(np.pi * np.clip(u, 0.0, 1.0)))

    else:
        raise ValueError(f"未知窗口类型: {wtype}")

    w[k_array >= kf] = 1.0
    return np.clip(w, 0.0, 1.0)


def xi_fftlog_from_effective_p0(k_grid: np.ndarray, p0_eff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 FFTLog 计算 xi0(r)。

    参数
    ----
    k_grid : np.ndarray
        对数均匀 k 网格。
    p0_eff : np.ndarray
        已经施加了所有 cut/window 的有效 P0(k)。

    返回
    ----
    Tuple[np.ndarray, np.ndarray]
        (r_grid, xi_grid)。
    """
    dln = np.log(k_grid[1] / k_grid[0])
    offset = fhtoffset(dln, mu=FFTLOG_MU, initial=0.0, bias=FFTLOG_BIAS)

    # j0 Hankel 对应输入：a(k)=k^(3/2) P(k)
    a_in = (k_grid ** 1.5) * p0_eff
    a_out = fht(a_in, dln=dln, mu=FFTLOG_MU, offset=offset, bias=FFTLOG_BIAS)

    n = k_grid.size
    j = np.arange(n)
    jc = (n - 1) / 2.0
    ln_kc = 0.5 * (np.log(k_grid[0]) + np.log(k_grid[-1]))
    r_grid = np.exp((offset - ln_kc) + (j - jc) * dln)

    const = np.sqrt(np.pi / 2.0) / (2.0 * np.pi**2)
    xi_grid = const * a_out / np.maximum(r_grid ** 1.5, 1e-300)
    return r_grid, xi_grid


def interp_xi_to_s(s_data: np.ndarray, r_grid: np.ndarray, xi_grid: np.ndarray) -> np.ndarray:
    """
    将 FFTLog 输出的 xi(r) 插值到测量的 s 网格。

    参数
    ----
    s_data : np.ndarray
        测量 s 网格。
    r_grid, xi_grid : np.ndarray
        FFTLog 输出。

    返回
    ----
    np.ndarray
        与 s_data 同长度的模型 xi。
    """
    order = np.argsort(r_grid)
    return np.interp(s_data, r_grid[order], xi_grid[order])


# =====================
# 6) 指标与表格
# =====================


def compute_alignment_metrics(
    s: np.ndarray,
    xi_data: np.ndarray,
    xi_std: np.ndarray,
    xi_model: np.ndarray,
    tag: str,
) -> Dict[str, float | str]:
    """
    计算 Data/Model 对齐指标。

    指标定义（都在 r^2 xi 上）：
    - mean_abs_sigma = mean(|(Data-Model)/sigma|)
    - chi2_ndof = mean(((Data-Model)/sigma)^2)
    - mean_sigma = mean((Data-Model)/sigma)，用于看系统性偏低/偏高方向

    参数
    ----
    s : np.ndarray
        s 网格。
    xi_data, xi_std : np.ndarray
        测量均值与样本标准差。
    xi_model : np.ndarray
        模型曲线（已插值到 s 网格）。
    tag : str
        指标标签。

    返回
    ----
    Dict[str, float | str]
        指标字典。
    """
    r2_data = s**2 * xi_data
    r2_model = s**2 * xi_model
    r2_std = np.maximum(s**2 * xi_std, 1e-12)

    mask = s >= LARGE_SCALE_MIN
    if np.sum(mask) == 0:
        raise RuntimeError("大尺度 mask 为空，请检查 s 网格")

    resid_sigma = (r2_data[mask] - r2_model[mask]) / r2_std[mask]

    out: Dict[str, float | str] = {
        "method": tag,
        "nbin_large": int(np.sum(mask)),
        "mean_abs_sigma": float(np.mean(np.abs(resid_sigma))),
        "chi2_ndof": float(np.mean(resid_sigma**2)),
        "mean_sigma": float(np.mean(resid_sigma)),
        "max_abs_sigma": float(np.max(np.abs(resid_sigma))),
    }
    return out


def write_dict_rows_csv(rows: List[Dict[str, object]], out_csv: str) -> None:
    """
    把字典行写到 CSV。

    参数
    ----
    rows : List[Dict[str, object]]
        数据行。
    out_csv : str
        输出路径。
    """
    if not rows:
        raise ValueError(f"写 CSV 失败：rows 为空 -> {out_csv}")

    # 不同方案可能附带不同附加字段（如 alpha/n_split），
    # 这里按“首次出现顺序”取并集列名，避免 DictWriter 因字段不齐报错。
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    normalized_rows = [{key: row.get(key, "") for key in keys} for row in rows]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(normalized_rows)


# =====================
# 7) 四种方案：Baseline / TestA / TestB / TestC
# =====================


def evaluate_unwindowed_with_kmin(
    s_data: np.ndarray,
    k_grid: np.ndarray,
    p0_base: np.ndarray,
    kmin: float,
    tag: str,
) -> np.ndarray:
    """
    计算“无 IR 窗”的积分结果（通过 kmin 控制下限）。

    该函数用于：
    - Baseline: kmin = k_f = 2*pi/L（mission4 口径）
    - TestA:    kmin = kmin_global（仅扩展下限，不加 IR 抑制）

    参数
    ----
    s_data : np.ndarray
        测量 s 网格。
    k_grid : np.ndarray
        FFTLog k 网格。
    p0_base : np.ndarray
        原始 best-fit P0(k)。
    kmin : float
        积分下限。
    tag : str
        标签，仅用于调试信息。

    返回
    ----
    np.ndarray
        插值到 s_data 的模型 xi。
    """
    # 使用 mission4 同款 log-k 边界平滑窗，避免硬截断振铃。
    taper = build_log_taper_window(k_grid, kmin=kmin, kmax=KMAX_INT, frac=EDGE_TAPER_FRAC)
    p0_eff = p0_base * taper

    r_grid, xi_grid = xi_fftlog_from_effective_p0(k_grid, p0_eff)
    xi_on_s = interp_xi_to_s(s_data, r_grid, xi_grid)
    _ = tag  # 保留变量，便于后续扩展时加日志
    return xi_on_s


def evaluate_testb_single_window(
    s_data: np.ndarray,
    k_grid: np.ndarray,
    p0_fnl100: np.ndarray,
    p0_fnl0_same_other_params: np.ndarray,
    window_type: str,
    window_param: float,
    kmin_global: float,
    use_png_only_window: bool,
) -> np.ndarray:
    """
    计算 TestB：IR 窗方案（单一窗口类型 + 单一参数）。

    参数
    ----
    s_data : np.ndarray
        测量 s 网格。
    k_grid : np.ndarray
        FFTLog k 网格。
    p0_fnl100 : np.ndarray
        fnl=100 best-fit 对应 P0(k)。
    p0_fnl0_same_other_params : np.ndarray
        把 best-fit 的 fnl_loc 改为 0 后得到的 P0(k)。
    window_type : str
        窗函数类型名。
    window_param : float
        窗函数参数值。
    kmin_global : float
        全局积分下限。
    use_png_only_window : bool
        若 True，仅对 PNG 增量项加窗；否则对总 P0 加窗。

    返回
    ----
    np.ndarray
        插值到 s_data 的模型 xi。
    """
    ir_window = build_ir_window(k_grid, K_FUND, wtype=window_type, param=window_param)

    if use_png_only_window:
        delta_png = p0_fnl100 - p0_fnl0_same_other_params
        p0_ir = p0_fnl0_same_other_params + ir_window * delta_png
    else:
        p0_ir = p0_fnl100 * ir_window

    # 积分范围 [kmin_global, kmax] 使用 mission4 同款边界平滑窗。
    taper = build_log_taper_window(k_grid, kmin=kmin_global, kmax=KMAX_INT, frac=EDGE_TAPER_FRAC)
    p0_eff = p0_ir * taper

    r_grid, xi_grid = xi_fftlog_from_effective_p0(k_grid, p0_eff)
    xi_on_s = interp_xi_to_s(s_data, r_grid, xi_grid)
    return xi_on_s


def generate_discrete_k_modes(kf: float, n_split: int) -> np.ndarray:
    """
    生成 |k| < n_split * kf 的离散模式模长数组。

    参数
    ----
    kf : float
        基本模式。
    n_split : int
        k_split / kf 的整数倍数。

    返回
    ----
    np.ndarray
        所有满足条件的离散 |k|（每个离散向量都计入一次）。
    """
    nmax = int(np.ceil(float(n_split)))
    rng = np.arange(-nmax, nmax + 1, dtype=int)
    nx, ny, nz = np.meshgrid(rng, rng, rng, indexing="ij")

    n2 = nx**2 + ny**2 + nz**2
    nmag = np.sqrt(n2.astype(float))

    mask = (n2 > 0) & (nmag < float(n_split))
    kmag = kf * nmag[mask]
    return kmag.astype(float)


def evaluate_testc_hybrid(
    s_data: np.ndarray,
    k_grid: np.ndarray,
    p0_base: np.ndarray,
    n_split: int,
) -> Tuple[np.ndarray, int]:
    """
    计算 TestC：低-k 离散求和 + 高-k 连续 FFTLog。

    参数
    ----
    s_data : np.ndarray
        测量 s 网格。
    k_grid : np.ndarray
        FFTLog k 网格。
    p0_base : np.ndarray
        best-fit P0(k)。
    n_split : int
        k_split = n_split * kf。

    返回
    ----
    Tuple[np.ndarray, int]
        - xi_total_on_s: 合成后的 xi 曲线（插值到 s_data）
        - nmodes: 低-k 离散求和实际使用的模式个数
    """
    k_split = float(n_split) * K_FUND

    # 低-k 离散模式
    k_modes = generate_discrete_k_modes(K_FUND, n_split=n_split)
    nmodes = int(k_modes.size)

    # 用线性插值把 P0(k) 映射到离散模式点
    p0_modes = np.interp(k_modes, k_grid, p0_base)

    # j0(x)=sin(x)/x，可用 np.sinc(x/pi) 稳定计算
    kr = np.outer(k_modes, s_data)  # shape=(Nmodes, Ns)
    j0 = np.sinc(kr / np.pi)

    # xi_low(r) = (1/V) * sum P(k) * j0(kr)
    volume = BOX_SIZE**3
    xi_low = np.sum(p0_modes[:, None] * j0, axis=0) / volume

    # 高-k 连续部分：从 k_split 到 kmax（mission4 同款边界平滑窗）
    p0_high = p0_base * build_log_taper_window(
        k_grid, kmin=k_split, kmax=KMAX_INT, frac=EDGE_TAPER_FRAC
    )
    r_grid, xi_grid = xi_fftlog_from_effective_p0(k_grid, p0_high)
    xi_high_on_s = interp_xi_to_s(s_data, r_grid, xi_grid)

    xi_total = xi_low + xi_high_on_s
    return xi_total, nmodes


# =====================
# 8) 绘图函数
# =====================


def plot_fig1_pk(
    data: MockData,
    fit: Dict[str, object],
    out_png: str,
) -> None:
    """
    图1：3Gpc fnl100 的 P0 mean vs best-fit。

    参数
    ----
    data : MockData
        测量数据。
    fit : Dict[str, object]
        best-fit 结果。
    out_png : str
        输出图路径。
    """
    bestfit = fit["bestfit"]
    kfit = np.asarray(fit["kfit"])
    p0_mean_fit = np.asarray(fit["p0_mean_fit"])
    p0_std_fit = np.asarray(fit["p0_std_fit"])
    p0_model_fit = np.asarray(fit["p0_model_fit"])

    # 同时画一条更连续的曲线
    k_plot = np.geomspace(float(kfit.min()), 0.3, 600)
    p0_plot = build_theory_p0(k_plot, bestfit)

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 6.4))
    ax.errorbar(
        kfit,
        p0_mean_fit,
        yerr=p0_std_fit,
        fmt="o",
        ms=4,
        capsize=2,
        color="black",
        label=f"Measured mean (N={data.nmock})",
    )
    ax.plot(k_plot, p0_plot, "-", lw=1.8, color="tab:red", label="Best-fit model")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k\,[h/\mathrm{Mpc}]$")
    ax.set_ylabel(r"$P_0(k)$")
    ax.set_title("Task5 Fig1: 3Gpc fnl=100 P0 Mean vs Best-fit")

    ratio = p0_mean_fit / np.maximum(p0_model_fit, 1e-30)
    txt = (
        f"best-fit: fnl_loc={bestfit['fnl_loc']:.3f}, b1={bestfit['b1']:.3f}, sigmas={bestfit['sigmas']:.4f}\n"
        f"mean(Data/Model)={np.mean(ratio):.3f}"
    )
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.86, edgecolor="gray"),
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_fig2_r2xi_compare(
    s: np.ndarray,
    xi_data: np.ndarray,
    xi_std: np.ndarray,
    curves: Dict[str, np.ndarray],
    out_png: str,
) -> None:
    """
    图2：r^2 xi 的 Data/Model 对比（主结果图）。

    参数
    ----
    s : np.ndarray
        s 网格。
    xi_data, xi_std : np.ndarray
        测量均值和标准差。
    curves : Dict[str, np.ndarray]
        各模型曲线字典。
    out_png : str
        输出图路径。
    """
    fig, ax = plt.subplots(1, 1, figsize=(9.2, 6.8))

    ax.errorbar(
        s,
        s**2 * xi_data,
        yerr=s**2 * xi_std,
        fmt="o",
        ms=3.8,
        capsize=2,
        color="black",
        label="Measured mean",
    )

    color_map = {
        "Baseline": "tab:blue",
        "TestA": "tab:orange",
        "TestB_best": "tab:red",
        "TestC_best": "tab:green",
    }
    for name, xi_model in curves.items():
        ax.plot(
            s,
            s**2 * xi_model,
            "-",
            lw=1.8,
            color=color_map.get(name, None),
            label=name,
        )

    ax.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
    ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$r^2\xi_0(r)$")
    ax.set_title("Task5 Fig2: 3Gpc fnl=100, Data vs IR-handled Models")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_fig3_residual_sigma(
    s: np.ndarray,
    xi_data: np.ndarray,
    xi_std: np.ndarray,
    baseline: np.ndarray,
    testb_best: np.ndarray,
    testc_best: np.ndarray,
    out_png: str,
) -> None:
    """
    图3：标准化残差 (Data-Model)/sigma 对比。

    参数
    ----
    s : np.ndarray
        s 网格。
    xi_data, xi_std : np.ndarray
        测量均值和标准差。
    baseline, testb_best, testc_best : np.ndarray
        三条模型曲线。
    out_png : str
        输出图路径。
    """
    r2_data = s**2 * xi_data
    r2_std = np.maximum(s**2 * xi_std, 1e-12)

    res_baseline = (r2_data - s**2 * baseline) / r2_std
    res_testb = (r2_data - s**2 * testb_best) / r2_std
    res_testc = (r2_data - s**2 * testc_best) / r2_std

    fig, ax = plt.subplots(1, 1, figsize=(9.2, 5.8))
    ax.axhline(0.0, color="black", lw=1.0)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(-1.0, color="gray", lw=0.8, ls="--", alpha=0.7)

    ax.plot(s, res_baseline, "o-", ms=3.4, lw=1.4, color="tab:blue", label="Baseline")
    ax.plot(s, res_testb, "s-", ms=3.4, lw=1.4, color="tab:red", label="TestB(best)")
    ax.plot(s, res_testc, "^-", ms=3.4, lw=1.4, color="tab:green", label="TestC(best)")

    ax.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
    ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$(Data-Model)/\sigma$ of $r^2\xi_0$")
    ax.set_title("Task5 Fig3: Residual Comparison")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_fig4_fnl0_sanity(
    s: np.ndarray,
    xi_data: np.ndarray,
    xi_std: np.ndarray,
    xi_baseline: np.ndarray,
    xi_testb: np.ndarray,
    out_png: str,
    best_window_param: float,
) -> None:
    """
    图4：fnl=0 的 sanity check（baseline vs TestB）。

    参数
    ----
    s : np.ndarray
        s 网格。
    xi_data, xi_std : np.ndarray
        fnl0 测量均值和标准差。
    xi_baseline, xi_testb : np.ndarray
        fnl0 模型曲线。
    out_png : str
        输出图路径。
    best_window_param : float
        TestB 最优窗口参数（沿用 fnl100）。
    """
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 6.4))

    ax.errorbar(
        s,
        s**2 * xi_data,
        yerr=s**2 * xi_std,
        fmt="o",
        ms=3.6,
        capsize=2,
        color="black",
        label="Measured fnl=0 mean",
    )
    ax.plot(s, s**2 * xi_baseline, "-", lw=1.8, color="tab:blue", label="Baseline (fnl0)")
    ax.plot(
        s,
        s**2 * xi_testb,
        "-",
        lw=1.8,
        color="tab:red",
        label=f"TestB(best-param={best_window_param:g}, fnl0)",
    )
    ax.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
    ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$r^2\xi_0(r)$")
    ax.set_title("Task5 Fig4: fnl=0 Sanity Check")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# =====================
# 9) 主流程
# =====================


def main() -> None:
    """执行任务 5 全流程。"""
    print("[INFO] ===== Task5: start =====")
    print(f"[INFO] rid range = [{RID_MIN}, {RID_MAX}]")
    print(f"[INFO] box size = {BOX_SIZE:.1f}, k_f = {K_FUND:.6e}")
    print(f"[INFO] kmin_global = {KMIN_GLOBAL:.2e}, kmax = {KMAX_INT:.2f}")

    # -------------------------------------------------
    # 1) 读取 fnl100 / fnl0 数据
    # -------------------------------------------------
    data100 = load_mock_data("fnl100", PK_FNL100_GLOB, PCF_FNL100_GLOB, RID_MIN, RID_MAX)
    data0 = load_mock_data("fnl0", PK_FNL0_GLOB, PCF_FNL0_GLOB, RID_MIN, RID_MAX)

    if not np.allclose(data100.scen, data0.scen):
        raise RuntimeError("fnl100 与 fnl0 的 s 网格不一致，无法做统一 sanity check")

    print(
        f"[INFO] data loaded: fnl100 Nmock={data100.nmock} ({data100.realizations.min()}..{data100.realizations.max()}), "
        f"fnl0 Nmock={data0.nmock} ({data0.realizations.min()}..{data0.realizations.max()})"
    )
    print(f"[INFO] s range = [{data100.scen.min():.1f}, {data100.scen.max():.1f}] with {data100.scen.size} bins")

    # -------------------------------------------------
    # 2) P0 best-fit（fnl100 与 fnl0）
    # -------------------------------------------------
    fit100 = fit_best_pk(data100)
    fit0 = fit_best_pk(data0)
    bestfit100 = fit100["bestfit"]
    bestfit0 = fit0["bestfit"]

    print(f"[INFO] fnl100 bestfit = {bestfit100}")
    print(f"[INFO] fnl0 bestfit   = {bestfit0}")

    # 图1：P0 mean vs best-fit（fnl100）
    plot_fig1_pk(data100, fit100, OUT_FIG1_PK)
    print(f"[INFO] saved fig1: {OUT_FIG1_PK}")

    # -------------------------------------------------
    # 3) 缓存高分辨率 P0(k)
    # -------------------------------------------------
    k_grid = np.geomspace(KMIN_GLOBAL / FFTLOG_PADDING, KMAX_INT * FFTLOG_PADDING, FFTLOG_N)

    # fnl100 best-fit P0
    p0_100 = build_theory_p0(k_grid, bestfit100)

    # “只改变 fnl_loc->0，其余参数不变”的对照 P0（用于 TestB PNG-only window）
    bestfit100_fnl0 = dict(bestfit100)
    bestfit100_fnl0["fnl_loc"] = 0.0
    p0_100_with_fnl0 = build_theory_p0(k_grid, bestfit100_fnl0)

    # fnl0 best-fit P0（用于 sanity）
    p0_0 = build_theory_p0(k_grid, bestfit0)

    # -------------------------------------------------
    # 4) 计算四类方案（fnl100）
    # -------------------------------------------------
    s = data100.scen

    # Baseline（mission4 口径：kmin=2pi/L）
    xi_baseline = evaluate_unwindowed_with_kmin(
        s_data=s,
        k_grid=k_grid,
        p0_base=p0_100,
        kmin=K_FUND,
        tag="Baseline",
    )

    # TestA（仅扩展积分下限，不加 IR 窗）
    xi_testa = evaluate_unwindowed_with_kmin(
        s_data=s,
        k_grid=k_grid,
        p0_base=p0_100,
        kmin=KMIN_GLOBAL,
        tag="TestA",
    )

    # TestB: 多窗口类型扫描
    testb_rows: List[Dict[str, object]] = []
    xi_testb_map: Dict[Tuple[str, float], np.ndarray] = {}
    for wtype, params in TESTB_WINDOW_SCANS:
        for p in params:
            pval = float(p)
            xi_b = evaluate_testb_single_window(
                s_data=s,
                k_grid=k_grid,
                p0_fnl100=p0_100,
                p0_fnl0_same_other_params=p0_100_with_fnl0,
                window_type=wtype,
                window_param=pval,
                kmin_global=KMIN_GLOBAL,
                use_png_only_window=TESTB_USE_PNG_ONLY_WINDOW,
            )
            xi_testb_map[(wtype, pval)] = xi_b

            metric = compute_alignment_metrics(
                s=s,
                xi_data=data100.xi_mean,
                xi_std=data100.xi_std,
                xi_model=xi_b,
                tag=f"TestB_{wtype}_{pval:g}",
            )
            metric["window_type"] = wtype
            metric["window_param"] = pval
            testb_rows.append(metric)

    # 选 TestB 最优窗口（按 mean_abs_sigma 最小）
    best_b_row = min(testb_rows, key=lambda row: float(row["mean_abs_sigma"]))
    best_wtype = str(best_b_row["window_type"])
    best_wparam = float(best_b_row["window_param"])
    xi_testb_best = xi_testb_map[(best_wtype, best_wparam)]

    # TestC: 扫 n_split
    testc_rows: List[Dict[str, object]] = []
    xi_testc_map: Dict[int, np.ndarray] = {}
    for nsp in TESTC_NSPLIT_LIST:
        xi_c, nmodes = evaluate_testc_hybrid(
            s_data=s,
            k_grid=k_grid,
            p0_base=p0_100,
            n_split=int(nsp),
        )
        xi_testc_map[int(nsp)] = xi_c

        metric = compute_alignment_metrics(
            s=s,
            xi_data=data100.xi_mean,
            xi_std=data100.xi_std,
            xi_model=xi_c,
            tag=f"TestC_nsplit{nsp}",
        )
        metric["n_split"] = int(nsp)
        metric["k_split"] = float(nsp) * K_FUND
        metric["nmodes_lowk"] = int(nmodes)
        testc_rows.append(metric)

    best_c_row = min(testc_rows, key=lambda row: float(row["mean_abs_sigma"]))
    best_nsplit = int(best_c_row["n_split"])
    xi_testc_best = xi_testc_map[best_nsplit]

    # -------------------------------------------------
    # 5) 主指标对比表（fnl100）
    # -------------------------------------------------
    main_rows = [
        compute_alignment_metrics(s, data100.xi_mean, data100.xi_std, xi_baseline, "Baseline"),
        compute_alignment_metrics(s, data100.xi_mean, data100.xi_std, xi_testa, "TestA"),
        compute_alignment_metrics(s, data100.xi_mean, data100.xi_std, xi_testb_best, "TestB_best"),
        compute_alignment_metrics(s, data100.xi_mean, data100.xi_std, xi_testc_best, "TestC_best"),
    ]

    # 给最优行补充参数信息
    for row in main_rows:
        if row["method"] == "TestB_best":
            row["window_type"] = best_wtype
            row["window_param"] = best_wparam
        if row["method"] == "TestC_best":
            row["n_split"] = best_nsplit
            row["k_split"] = float(best_nsplit) * K_FUND

    write_dict_rows_csv(main_rows, OUT_METRIC_MAIN_CSV)
    write_dict_rows_csv(testb_rows, OUT_SCAN_B_CSV)
    write_dict_rows_csv(testc_rows, OUT_SCAN_C_CSV)

    print(f"[INFO] saved metrics: {OUT_METRIC_MAIN_CSV}")
    print(f"[INFO] saved scan B:  {OUT_SCAN_B_CSV}")
    print(f"[INFO] saved scan C:  {OUT_SCAN_C_CSV}")

    # -------------------------------------------------
    # 6) 稳定性检查
    # -------------------------------------------------
    # 6.1 TestB 最优参数下，把 kmin_global 再降 10 倍
    kmin_smaller = KMIN_GLOBAL * STABILITY_KMIN_FACTOR
    xi_testb_stability = evaluate_testb_single_window(
        s_data=s,
        k_grid=k_grid,
        p0_fnl100=p0_100,
        p0_fnl0_same_other_params=p0_100_with_fnl0,
        window_type=best_wtype,
        window_param=best_wparam,
        kmin_global=kmin_smaller,
        use_png_only_window=TESTB_USE_PNG_ONLY_WINDOW,
    )

    r2_diff_stability = s**2 * (xi_testb_stability - xi_testb_best)
    r2_sigma = np.maximum(s**2 * data100.xi_std, 1e-12)
    mask_large = s >= LARGE_SCALE_MIN
    stability_rms_sigma = float(np.sqrt(np.mean((r2_diff_stability[mask_large] / r2_sigma[mask_large]) ** 2)))

    # 6.2 TestC 稳定性已通过 n_split 扫描表给出

    # -------------------------------------------------
    # 7) fnl=0 sanity check
    # -------------------------------------------------
    s0 = data0.scen
    xi0_baseline = evaluate_unwindowed_with_kmin(
        s_data=s0,
        k_grid=k_grid,
        p0_base=p0_0,
        kmin=K_FUND,
        tag="Baseline_fnl0",
    )
    # fnl=0 下 PNG 增量项理论上很小，TestB 应近似 baseline；
    # 这里沿用 fnl100 上选出的最优窗口，验证不会明显恶化。
    xi0_testb = evaluate_testb_single_window(
        s_data=s0,
        k_grid=k_grid,
        p0_fnl100=p0_0,
        p0_fnl0_same_other_params=p0_0,
        window_type=best_wtype,
        window_param=best_wparam,
        kmin_global=KMIN_GLOBAL,
        use_png_only_window=TESTB_USE_PNG_ONLY_WINDOW,
    )

    sanity_rows = [
        compute_alignment_metrics(s0, data0.xi_mean, data0.xi_std, xi0_baseline, "fnl0_Baseline"),
        compute_alignment_metrics(s0, data0.xi_mean, data0.xi_std, xi0_testb, "fnl0_TestB_bestWindow"),
    ]
    write_dict_rows_csv(sanity_rows, OUT_SANITY_CSV)
    print(f"[INFO] saved sanity: {OUT_SANITY_CSV}")

    # -------------------------------------------------
    # 8) 绘图输出
    # -------------------------------------------------
    plot_fig2_r2xi_compare(
        s=s,
        xi_data=data100.xi_mean,
        xi_std=data100.xi_std,
        curves={
            "Baseline": xi_baseline,
            "TestA": xi_testa,
            "TestB_best": xi_testb_best,
            "TestC_best": xi_testc_best,
        },
        out_png=OUT_FIG2_XI,
    )

    plot_fig3_residual_sigma(
        s=s,
        xi_data=data100.xi_mean,
        xi_std=data100.xi_std,
        baseline=xi_baseline,
        testb_best=xi_testb_best,
        testc_best=xi_testc_best,
        out_png=OUT_FIG3_RES,
    )

    plot_fig4_fnl0_sanity(
        s=s0,
        xi_data=data0.xi_mean,
        xi_std=data0.xi_std,
        xi_baseline=xi0_baseline,
        xi_testb=xi0_testb,
        out_png=OUT_FIG4_SANITY,
        best_window_param=best_wparam,
    )

    print(f"[INFO] saved fig2: {OUT_FIG2_XI}")
    print(f"[INFO] saved fig3: {OUT_FIG3_RES}")
    print(f"[INFO] saved fig4: {OUT_FIG4_SANITY}")

    # -------------------------------------------------
    # 9) 保存参数与总结
    # -------------------------------------------------
    bestfit_dump = {
        "fnl100_bestfit": bestfit100,
        "fnl100_bestfit_with_fnl0_replaced": bestfit100_fnl0,
        "fnl0_bestfit": bestfit0,
        "task5_constants": {
            "BOX_SIZE": BOX_SIZE,
            "K_FUND": K_FUND,
            "KMIN_GLOBAL": KMIN_GLOBAL,
            "KMAX_INT": KMAX_INT,
            "RID_MIN": RID_MIN,
            "RID_MAX": RID_MAX,
            "TESTB_WINDOW_SCANS": TESTB_WINDOW_SCANS,
            "TESTC_NSPLIT_LIST": TESTC_NSPLIT_LIST,
            "TESTB_USE_PNG_ONLY_WINDOW": TESTB_USE_PNG_ONLY_WINDOW,
        },
        "task5_best_choices": {
            "best_window_type": best_wtype,
            "best_window_param": best_wparam,
            "best_nsplit": best_nsplit,
            "stability_kmin_smaller": kmin_smaller,
            "stability_rms_sigma": stability_rms_sigma,
        },
    }
    with open(OUT_BESTFIT_JSON, "w", encoding="utf-8") as f:
        json.dump(bestfit_dump, f, ensure_ascii=False, indent=2)

    # 把关键指标写入 summary
    main_by_name = {row["method"]: row for row in main_rows}
    sanity_by_name = {row["method"]: row for row in sanity_rows}

    with open(OUT_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("任务5总结（3Gpc, fnl=100: IR数值处理）\n")
        f.write("===================================\n")
        f.write(f"数据范围: realization={RID_MIN}..{RID_MAX}\n")
        f.write(f"样本数量: fnl100 N={data100.nmock}, fnl0 N={data0.nmock}\n")
        f.write(f"盒长: L={BOX_SIZE:.1f}, k_f=2pi/L={K_FUND:.6e}\n")
        f.write(f"积分设置: kmin_global={KMIN_GLOBAL:.2e}, kmax={KMAX_INT:.2f}\n")
        f.write(f"Baseline 口径: kmin=2pi/L={K_FUND:.6e}（mission4 同款边界平滑窗）\n")
        f.write(f"大尺度指标区间: r>={LARGE_SCALE_MIN:.1f}\n\n")

        f.write("[fnl100] best-fit 参数:\n")
        f.write(json.dumps(bestfit100, ensure_ascii=False, indent=2) + "\n\n")

        f.write("[主指标对比: mean|Δ/σ|, chi2/ndof, mean(Δ/σ)]\n")
        for name in ["Baseline", "TestA", "TestB_best", "TestC_best"]:
            row = main_by_name[name]
            f.write(
                f"- {name}: mean|Δ/σ|={row['mean_abs_sigma']:.4f}, "
                f"chi2/ndof={row['chi2_ndof']:.4f}, mean(Δ/σ)={row['mean_sigma']:.4f}\n"
            )

        f.write("\n[TestB 扫描最优]\n")
        f.write(f"- 最优窗口 = {best_wtype}, 参数={best_wparam}\n")
        f.write(
            f"- 指标: mean|Δ/σ|={main_by_name['TestB_best']['mean_abs_sigma']:.4f}, "
            f"chi2/ndof={main_by_name['TestB_best']['chi2_ndof']:.4f}\n"
        )

        f.write("\n[TestC 扫描最优]\n")
        f.write(f"- 最优 n_split = {best_nsplit}, k_split={best_nsplit * K_FUND:.6e}\n")
        f.write(
            f"- 指标: mean|Δ/σ|={main_by_name['TestC_best']['mean_abs_sigma']:.4f}, "
            f"chi2/ndof={main_by_name['TestC_best']['chi2_ndof']:.4f}\n"
        )

        f.write("\n[TestB 稳定性检查]\n")
        f.write(f"- 把 kmin_global 降到 {kmin_smaller:.2e} 后，与最优曲线差异 RMS_sigma={stability_rms_sigma:.4f}\n")

        f.write("\n[fnl0 sanity check]\n")
        f.write(
            "- Baseline: "
            f"mean|Δ/σ|={sanity_by_name['fnl0_Baseline']['mean_abs_sigma']:.4f}, "
            f"chi2/ndof={sanity_by_name['fnl0_Baseline']['chi2_ndof']:.4f}\n"
        )
        f.write(
            "- TestB(best-window): "
            f"mean|Δ/σ|={sanity_by_name['fnl0_TestB_bestWindow']['mean_abs_sigma']:.4f}, "
            f"chi2/ndof={sanity_by_name['fnl0_TestB_bestWindow']['chi2_ndof']:.4f}\n"
        )

        # 简要结论
        b = main_by_name["Baseline"]["mean_abs_sigma"]
        tb = main_by_name["TestB_best"]["mean_abs_sigma"]
        tc = main_by_name["TestC_best"]["mean_abs_sigma"]
        f.write("\n[结论]\n")
        if tb < b:
            f.write("- TestB 相比 Baseline 在大尺度对齐指标上有改进。\n")
        else:
            f.write("- TestB 未显著优于 Baseline，需要进一步调窗函数形式。\n")

        if tc < b:
            f.write("- TestC（hybrid）相对 Baseline 也有改进，说明离散低-k shell 处理有价值。\n")
        else:
            f.write("- TestC 未明显优于 Baseline，需继续优化 k_split 或低-k 建模。\n")

    print(f"[INFO] saved bestfit json: {OUT_BESTFIT_JSON}")
    print(f"[INFO] saved summary:     {OUT_SUMMARY_TXT}")
    print("[INFO] ===== Task5: done =====")


if __name__ == "__main__":
    main()
