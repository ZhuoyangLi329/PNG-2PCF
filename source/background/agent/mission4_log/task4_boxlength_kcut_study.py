#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本名称
--------
task4_boxlength_kcut_study.py

代码大纲（执行逻辑）
--------------------
第 0 部分：参数区
- 集中定义输入/输出路径、拟合区间、FFTLog 数值参数、图像样式等。

第 1 部分：数据读取与整理（任务 4.1 的基础）
- 读取 1Gpc fnl=100 与 3Gpc fnl=100 的 pk/pcf 文件。
- 按 realization 编号做交集对齐，得到每个盒子的：
  - P0(k) mock 矩阵
  - xi0(s) mock 矩阵
  - 对应均值/标准差

第 2 部分：任务 4.1（测量 2PCF 与盒长关系）
- 在“全部可用样本”上比较 1Gpc 与 3Gpc 的测量 2PCF。
- 在“匹配子样本（共同 realization）”上再次比较，避免样本数量差异影响。
- 输出对比图与每个 s-bin 的统计表。

第 3 部分：任务 4.2（k-cut 与盒长关系）
- 用 desilike + Minuit 分别拟合两种盒长的测量 P0，得到 best-fit 参数。
- 对每个盒长采用 kmin=2*pi/L 的积分下限（妥协 cut），固定 kmax 做 FFTLog：
    xi0(s) = int dk k^2/(2pi^2) P0(k) j0(ks)
- 将理论 xi0 与测量均值比较，输出残差指标与图像。

第 4 部分：归档
- 输出 CSV、PNG、TXT 总结到 mission4_log。
- 生成 mission4_log.ipynb（实验报告 Notebook）。

说明
----
1) 本脚本不安装任何新包；默认在 conda 环境 desilike 中运行。
2) 所有函数都有中文注释，便于后续维护和复现实验。
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fht, fhtoffset


# =====================
# 0) 参数区
# =====================

# 输入目录（统一使用质量修补后的 masscut 结果）
PK_1GPC_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat"
PCF_1GPC_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat"
PK_3GPC_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat"
PCF_3GPC_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat"

# 输出目录
OUT_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission4_log"
os.makedirs(OUT_DIR, exist_ok=True)

# 列定义（与现有数据格式一致）
# pk: col0=kcen, col1=kmin, col2=kmax, col5=P0
PK_KCEN_COL = 0
PK_KMIN_COL = 1
PK_KMAX_COL = 2
PK_P0_COL = 5

# pcf: col0=s_cen, col1=s_min, col2=s_max, col3=xi0
PCF_SCEN_COL = 0
PCF_SMIN_COL = 1
PCF_SMAX_COL = 2
PCF_XI0_COL = 3

# 盒长（用于 kmin=2*pi/L）
BOX_SIZE_1GPC = 1000.0
BOX_SIZE_3GPC = 3000.0

# 这轮按你的要求先用前 80 个 realization（不等 3Gpc 全量）
REALIZATION_MIN = 1
REALIZATION_MAX = 80

# P0 拟合区间：沿用任务3思路，聚焦大尺度低-k
# 1Gpc 的前 10 个点约到 k=0.06328；这里按物理 k 上限统一。
PK_FIT_KMAX = 0.0635

# Minuit 设置
MINUIT_SEED = 66
MINUIT_NITER = 25

# PNG 理论固定参数（与任务3保持一致）
UNIT_Z = 1.0
FIXED_P = 1.1
FIXED_SN0 = 0.0
FIXED_SIGMAS = 0.0

# FFTLog 设置（任务4.2）
FFTLOG_N = 4096
FFTLOG_PADDING = 4.0
FFTLOG_MU = 0.5
FFTLOG_BIAS = 0.0
EDGE_TAPER_FRAC = 0.06
K_INT_MAX = 20.0

# 2PCF 大尺度统计区间（用于回答任务问题）
LARGE_SCALE_MIN = 200.0

# 图像风格
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 160
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"


# =====================
# 1) 数据结构与基础函数
# =====================

@dataclass
class BoxConfig:
    """单个盒长数据源的配置。"""

    tag: str
    box_size: float
    pk_glob: str
    pcf_glob: str


@dataclass
class BoxData:
    """
    保存某个盒长在“realization 对齐后”的完整数据。

    成员说明
    --------
    tag : str
        盒长标签，例如 1gpc / 3gpc。
    box_size : float
        盒长 L，单位 Mpc/h。
    realizations : np.ndarray
        参与分析的 realization 编号（升序）。
    kcen/kmin/kmax : np.ndarray
        pk 的 k-bin 网格信息。
    p0_mocks : np.ndarray
        shape = (Nmock, Nk)，每个 realization 的 P0。
    scen/smin/smax : np.ndarray
        pcf 的 s-bin 网格信息。
    xi_mocks : np.ndarray
        shape = (Nmock, Ns)，每个 realization 的 xi0。
    """

    tag: str
    box_size: float
    realizations: np.ndarray
    kcen: np.ndarray
    kmin: np.ndarray
    kmax: np.ndarray
    p0_mocks: np.ndarray
    scen: np.ndarray
    smin: np.ndarray
    smax: np.ndarray
    xi_mocks: np.ndarray

    @property
    def nmock(self) -> int:
        """返回 mock 数量。"""
        return int(self.p0_mocks.shape[0])

    @property
    def p0_mean(self) -> np.ndarray:
        """返回 P0 均值曲线。"""
        return np.mean(self.p0_mocks, axis=0)

    @property
    def p0_std(self) -> np.ndarray:
        """返回 P0 标准差曲线（样本标准差）。"""
        return np.std(self.p0_mocks, axis=0, ddof=1)

    @property
    def xi_mean(self) -> np.ndarray:
        """返回 xi0 均值曲线。"""
        return np.mean(self.xi_mocks, axis=0)

    @property
    def xi_std(self) -> np.ndarray:
        """返回 xi0 标准差曲线（样本标准差）。"""
        return np.std(self.xi_mocks, axis=0, ddof=1)


def parse_realization_id(path: str) -> int:
    """
    从文件名中解析 realization 编号。

    参数
    ----
    path : str
        文件绝对路径。

    返回
    ----
    int
        解析到的 realization 编号。
    """
    m = re.search(r"_N(\d+)\.dat$", os.path.basename(path))
    if m is None:
        raise ValueError(f"无法从文件名解析 realization 编号: {path}")
    return int(m.group(1))


def list_realization_files(file_glob: str) -> Dict[int, str]:
    """
    扫描并建立 realization->文件路径映射。

    参数
    ----
    file_glob : str
        glob 模式。

    返回
    ----
    Dict[int, str]
        字典键是 realization 编号，值是文件路径。
    """
    files = glob.glob(file_glob)
    if not files:
        raise FileNotFoundError(f"未找到文件: glob={file_glob}")

    mapping: Dict[int, str] = {}
    for fp in files:
        rid = parse_realization_id(fp)
        mapping[rid] = fp
    return mapping


def assert_same_grid(arrays: List[np.ndarray], what: str) -> None:
    """
    检查多个数组是否一致，用于保证不同 realization 的 bin 网格统一。

    参数
    ----
    arrays : List[np.ndarray]
        待比较数组列表。
    what : str
        报错提示中的变量名。
    """
    ref = arrays[0]
    for i, a in enumerate(arrays[1:], start=1):
        if not np.allclose(ref, a):
            raise ValueError(f"{what} 在第 {i} 个文件与第 0 个文件不一致")


def load_box_data(config: BoxConfig, keep_realizations: Iterable[int] | None = None) -> BoxData:
    """
    读取一个盒长的数据，并按 realization 做 pk/pcf 交集对齐。

    参数
    ----
    config : BoxConfig
        盒长配置。
    keep_realizations : Iterable[int] | None
        若给定，只保留该集合中的 realization。

    返回
    ----
    BoxData
        对齐后的盒长数据结构。
    """
    pk_map = list_realization_files(config.pk_glob)
    pcf_map = list_realization_files(config.pcf_glob)

    common = sorted(set(pk_map) & set(pcf_map))
    if keep_realizations is not None:
        keep_set = set(int(x) for x in keep_realizations)
        common = [rid for rid in common if rid in keep_set]

    if not common:
        raise RuntimeError(f"{config.tag} 没有可用的共同 realization")

    # 读取 pk
    pk_tables = [np.loadtxt(pk_map[rid], comments="#") for rid in common]
    kcen_list = [t[:, PK_KCEN_COL] for t in pk_tables]
    kmin_list = [t[:, PK_KMIN_COL] for t in pk_tables]
    kmax_list = [t[:, PK_KMAX_COL] for t in pk_tables]
    assert_same_grid(kcen_list, f"{config.tag} kcen")
    assert_same_grid(kmin_list, f"{config.tag} kmin")
    assert_same_grid(kmax_list, f"{config.tag} kmax")

    p0_mocks = np.vstack([t[:, PK_P0_COL] for t in pk_tables])

    # 读取 pcf
    pcf_tables = [np.loadtxt(pcf_map[rid], comments="#") for rid in common]
    scen_list = [t[:, PCF_SCEN_COL] for t in pcf_tables]
    smin_list = [t[:, PCF_SMIN_COL] for t in pcf_tables]
    smax_list = [t[:, PCF_SMAX_COL] for t in pcf_tables]
    assert_same_grid(scen_list, f"{config.tag} scen")
    assert_same_grid(smin_list, f"{config.tag} smin")
    assert_same_grid(smax_list, f"{config.tag} smax")

    xi_mocks = np.vstack([t[:, PCF_XI0_COL] for t in pcf_tables])

    return BoxData(
        tag=config.tag,
        box_size=config.box_size,
        realizations=np.array(common, dtype=int),
        kcen=kcen_list[0].copy(),
        kmin=kmin_list[0].copy(),
        kmax=kmax_list[0].copy(),
        p0_mocks=p0_mocks,
        scen=scen_list[0].copy(),
        smin=smin_list[0].copy(),
        smax=smax_list[0].copy(),
        xi_mocks=xi_mocks,
    )


def subset_box_data(box: BoxData, subset_realizations: Iterable[int]) -> BoxData:
    """
    从 BoxData 中截取 realization 子集，保持网格不变。

    参数
    ----
    box : BoxData
        原始盒长数据。
    subset_realizations : Iterable[int]
        希望保留的 realization 编号。

    返回
    ----
    BoxData
        子集数据。
    """
    subset = set(int(x) for x in subset_realizations)
    idx = [i for i, rid in enumerate(box.realizations) if rid in subset]
    if not idx:
        raise RuntimeError(f"{box.tag} 在指定子集下没有样本")

    return BoxData(
        tag=box.tag,
        box_size=box.box_size,
        realizations=box.realizations[idx].copy(),
        kcen=box.kcen.copy(),
        kmin=box.kmin.copy(),
        kmax=box.kmax.copy(),
        p0_mocks=box.p0_mocks[idx, :].copy(),
        scen=box.scen.copy(),
        smin=box.smin.copy(),
        smax=box.smax.copy(),
        xi_mocks=box.xi_mocks[idx, :].copy(),
    )


def save_measurement_csv(box1: BoxData, box2: BoxData, out_csv: str) -> None:
    """
    保存 1Gpc/3Gpc 测量 2PCF 的逐 bin 对比表。

    参数
    ----
    box1, box2 : BoxData
        两个盒长数据，要求 s 网格一致。
    out_csv : str
        输出 CSV 路径。
    """
    if not np.allclose(box1.scen, box2.scen):
        raise ValueError("保存测量对比表时发现 s 网格不一致")

    s = box1.scen
    xi1, xi2 = box1.xi_mean, box2.xi_mean
    std1, std2 = box1.xi_std, box2.xi_std
    err1 = std1 / np.sqrt(box1.nmock)
    err2 = std2 / np.sqrt(box2.nmock)

    delta = xi1 - xi2
    sigma_delta_mean = np.sqrt(err1**2 + err2**2)
    significance = delta / np.maximum(sigma_delta_mean, 1e-12)

    r2_xi1 = s**2 * xi1
    r2_xi2 = s**2 * xi2

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "s",
            "xi_mean_1gpc",
            "xi_std_1gpc",
            "xi_errmean_1gpc",
            "xi_mean_3gpc",
            "xi_std_3gpc",
            "xi_errmean_3gpc",
            "xi_delta_1minus3",
            "xi_significance_delta",
            "r2xi_mean_1gpc",
            "r2xi_mean_3gpc",
            "r2xi_delta_1minus3",
        ])
        for i in range(s.size):
            writer.writerow([
                f"{s[i]:.6g}",
                f"{xi1[i]:.12e}",
                f"{std1[i]:.12e}",
                f"{err1[i]:.12e}",
                f"{xi2[i]:.12e}",
                f"{std2[i]:.12e}",
                f"{err2[i]:.12e}",
                f"{delta[i]:.12e}",
                f"{significance[i]:.12e}",
                f"{r2_xi1[i]:.12e}",
                f"{r2_xi2[i]:.12e}",
                f"{(r2_xi1[i] - r2_xi2[i]):.12e}",
            ])


def plot_measurement_compare(box1: BoxData, box2: BoxData, out_png: str, title_suffix: str) -> None:
    """
    绘制测量 2PCF 对比图（xi 与 s^2 xi）。

    参数
    ----
    box1, box2 : BoxData
        两个盒长数据。
    out_png : str
        输出图像路径。
    title_suffix : str
        标题后缀，用于区分 all / matched 子样本。
    """
    if not np.allclose(box1.scen, box2.scen):
        raise ValueError("绘图时发现 s 网格不一致")

    s = box1.scen

    xi1, xi2 = box1.xi_mean, box2.xi_mean
    std1, std2 = box1.xi_std, box2.xi_std

    r2_1, r2_2 = s**2 * xi1, s**2 * xi2
    r2_std1, r2_std2 = s**2 * std1, s**2 * std2

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 9.2), sharex=True)

    ax = axes[0]
    ax.plot(s, xi1, "o-", ms=4, lw=1.4, color="tab:blue", label=f"1Gpc mean (N={box1.nmock})")
    ax.fill_between(s, xi1 - std1, xi1 + std1, color="tab:blue", alpha=0.18, label="1Gpc ±1σ")
    ax.plot(s, xi2, "s-", ms=4, lw=1.4, color="tab:orange", label=f"3Gpc mean (N={box2.nmock})")
    ax.fill_between(s, xi2 - std2, xi2 + std2, color="tab:orange", alpha=0.18, label="3Gpc ±1σ")
    ax.set_ylabel(r"$\xi_0(s)$")
    ax.set_title(f"Task 4.1 Measured 2PCF Comparison ({title_suffix})")
    ax.legend(ncol=2, fontsize=9)

    ax = axes[1]
    ax.plot(s, r2_1, "o-", ms=4, lw=1.4, color="tab:blue", label="1Gpc")
    ax.fill_between(s, r2_1 - r2_std1, r2_1 + r2_std1, color="tab:blue", alpha=0.18)
    ax.plot(s, r2_2, "s-", ms=4, lw=1.4, color="tab:orange", label="3Gpc")
    ax.fill_between(s, r2_2 - r2_std2, r2_2 + r2_std2, color="tab:orange", alpha=0.18)
    ax.axhline(0.0, color="k", lw=1.0, alpha=0.6)
    ax.set_xlabel(r"$s\,[\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$s^2\xi_0(s)$")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# =====================
# 2) 任务4.2的建模函数（desilike + FFTLog）
# =====================

# 这些导入仅在 desilike 环境可用；若缺失则给出明确提示。
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
except ModuleNotFoundError as e:
    raise SystemExit(
        "缺少 desilike 相关依赖。请在 conda 环境 desilike 下运行本脚本。"
        f"\n原始报错: {repr(e)}"
    )


# 初始化 desilike 日志（只做一次）
setup_logging()


def convert_bestfit_to_float_dict(bestfit: Dict[str, object]) -> Dict[str, float]:
    """
    把 desilike 返回的 ParameterArray 字典转为纯 float 字典。

    参数
    ----
    bestfit : Dict[str, object]
        profiler.bestfit.choice(input=True) 的输出。

    返回
    ----
    Dict[str, float]
        便于序列化和后续数值计算的参数字典。
    """
    out: Dict[str, float] = {}
    for key, val in bestfit.items():
        # ParameterArray 通常可被 np.asarray 再转 float
        out[key] = float(np.ravel(np.asarray(val))[0])
    return out


def fit_box_pk(box: BoxData) -> Dict[str, object]:
    """
    对单个盒长执行 P0 拟合，输出 best-fit 与模型曲线。

    参数
    ----
    box : BoxData
        输入盒长数据。

    返回
    ----
    Dict[str, object]
        包含 best-fit 参数与绘图所需数组。
    """
    # 1) 选取低-k 拟合区间
    mask = box.kcen <= PK_FIT_KMAX
    if mask.sum() < 5:
        raise RuntimeError(f"{box.tag} 在 k<={PK_FIT_KMAX} 下可用点太少")

    kcen_fit = box.kcen[mask]
    kmin_fit = box.kmin[mask]
    kmax_fit = box.kmax[mask]
    p0_mean_fit = box.p0_mean[mask]
    p0_mocks_fit = box.p0_mocks[:, mask]

    # 2) 理论与似然（任务3同设置）
    cosmo_unit = Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )
    template = FixedPowerSpectrumTemplate(z=UNIT_Z, fiducial=cosmo_unit)
    theory = PNGTracerPowerSpectrumMultipoles(template=template, mode="b-p")

    theory.init.params["p"].update(fixed=True, value=FIXED_P)
    theory.init.params["sn0"].update(fixed=True, value=FIXED_SN0)
    theory.init.params["sigmas"].update(fixed=False, value=FIXED_SIGMAS)

    dk = float(np.median(np.diff(kcen_fit)))
    observable = TracerPowerSpectrumMultipolesObservable(
        # 直接传数组：data 用均值曲线，covariance 用 mock 列表；
        # desilike 会自动估计协方差，并应用 Hartlap/Percival 修正。
        data=p0_mean_fit,
        covariance=[row for row in p0_mocks_fit],
        klim={0: [float(kcen_fit.min()), float(kcen_fit.max()), dk]},
        k=kcen_fit,
        ells=[0],
        theory=theory,
    )
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    _ = likelihood()

    likelihood.all_params["p"].update(fixed=True, value=FIXED_P)
    likelihood.all_params["sn0"].update(fixed=True, value=FIXED_SN0)
    likelihood.all_params["sigmas"].update(fixed=False, value=FIXED_SIGMAS)

    # 3) Minuit 极大似然
    profiler = MinuitProfiler(likelihood, seed=MINUIT_SEED)
    profiles = profiler.maximize(niterations=MINUIT_NITER)
    bestfit_raw = profiles.bestfit.choice(input=True)
    bestfit = convert_bestfit_to_float_dict(bestfit_raw)

    # 4) 生成“拟合网格”与“展示网格”上的模型曲线
    theory_fit = PNGTracerPowerSpectrumMultipoles(k=kcen_fit, template=template, mode="b-p")
    theory_fit.init.params["p"].update(fixed=True, value=FIXED_P)
    theory_fit.init.params["sn0"].update(fixed=True, value=FIXED_SN0)
    theory_fit.init.params["sigmas"].update(fixed=False, value=FIXED_SIGMAS)
    theory_fit(**bestfit)
    p0_model_fit = np.asarray(theory_fit.power[0], dtype=float)

    k_plot = np.geomspace(float(kcen_fit.min()), 0.3, 500)
    theory_plot = PNGTracerPowerSpectrumMultipoles(k=k_plot, template=template, mode="b-p")
    theory_plot.init.params["p"].update(fixed=True, value=FIXED_P)
    theory_plot.init.params["sn0"].update(fixed=True, value=FIXED_SN0)
    theory_plot.init.params["sigmas"].update(fixed=False, value=FIXED_SIGMAS)
    theory_plot(**bestfit)
    p0_model_plot = np.asarray(theory_plot.power[0], dtype=float)

    return {
        "kcen_fit": kcen_fit,
        "p0_mean_fit": p0_mean_fit,
        "p0_std_fit": np.std(p0_mocks_fit, axis=0, ddof=1),
        "bestfit": bestfit,
        "p0_model_fit": p0_model_fit,
        "k_plot": k_plot,
        "p0_model_plot": p0_model_plot,
    }


def build_log_taper_window(k_array: np.ndarray, kmin: float, kmax: float, frac: float) -> np.ndarray:
    """
    构建 log(k) 边界平滑窗，显式控制积分上下限并减小振铃。

    参数
    ----
    k_array : np.ndarray
        对数均匀的 k 网格。
    kmin, kmax : float
        有效积分区间。
    frac : float
        过渡带宽度占 log(kmax/kmin) 的比例。

    返回
    ----
    np.ndarray
        与 k_array 同长度的窗函数。
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


def build_theory_p0(k_grid: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    在给定 k 网格上计算 PNG 理论 P0(k)。

    参数
    ----
    k_grid : np.ndarray
        目标 k 网格。
    params : Dict[str, float]
        理论参数（至少包含 fnl_loc/b1/sigmas/p/sn0）。

    返回
    ----
    np.ndarray
        P0(k) 数组。
    """
    cosmo_unit = Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )
    template = FixedPowerSpectrumTemplate(z=UNIT_Z, fiducial=cosmo_unit)
    theory = PNGTracerPowerSpectrumMultipoles(k=k_grid, template=template, mode="b-p")
    theory.init.params["p"].update(fixed=True, value=FIXED_P)
    theory.init.params["sn0"].update(fixed=True, value=FIXED_SN0)
    theory.init.params["sigmas"].update(fixed=False, value=FIXED_SIGMAS)
    theory(**params)
    return np.asarray(theory.power[0], dtype=float)


def xi_custom_fftlog(k_grid: np.ndarray, p0_grid: np.ndarray, kmin: float, kmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用自写 FFTLog 计算 xi0(r)。

    参数
    ----
    k_grid : np.ndarray
        对数均匀 k 网格。
    p0_grid : np.ndarray
        与 k_grid 对应的 P0(k)。
    kmin, kmax : float
        显式积分上下限。

    返回
    ----
    Tuple[np.ndarray, np.ndarray]
        r_grid, xi0_grid
    """
    p0_cut = p0_grid * build_log_taper_window(k_grid, kmin, kmax, EDGE_TAPER_FRAC)

    dln = np.log(k_grid[1] / k_grid[0])
    offset = fhtoffset(dln, mu=FFTLOG_MU, initial=0.0, bias=FFTLOG_BIAS)

    # j0 对应的 Hankel 输入配比：a_in = k^(3/2) P(k)
    a_in = (k_grid ** 1.5) * p0_cut
    A_out = fht(a_in, dln=dln, mu=FFTLOG_MU, offset=offset, bias=FFTLOG_BIAS)

    n = k_grid.size
    j = np.arange(n)
    jc = (n - 1) / 2.0
    ln_kc = 0.5 * (np.log(k_grid[0]) + np.log(k_grid[-1]))
    r_grid = np.exp((offset - ln_kc) + (j - jc) * dln)

    const = np.sqrt(np.pi / 2.0) / (2.0 * np.pi**2)
    xi_grid = const * A_out / (r_grid ** 1.5)

    return r_grid, xi_grid


def evaluate_theory_xi_on_data_grid(
    s_data: np.ndarray,
    bestfit: Dict[str, float],
    box_size: float,
) -> Dict[str, np.ndarray]:
    """
    在给定盒长下，按 kmin=2*pi/L 构建理论 xi，并插值到测量 s 网格。

    参数
    ----
    s_data : np.ndarray
        测量 2PCF 的 s 网格。
    bestfit : Dict[str, float]
        P0 拟合得到的 best-fit 参数。
    box_size : float
        盒长 L。

    返回
    ----
    Dict[str, np.ndarray]
        包含 k_grid, p0_grid, r_grid, xi_grid, xi_on_s, kmin_cut。
    """
    kmin_cut = 2.0 * np.pi / float(box_size)
    k_grid = np.geomspace(kmin_cut / FFTLOG_PADDING, K_INT_MAX * FFTLOG_PADDING, FFTLOG_N)
    p0_grid = build_theory_p0(k_grid, bestfit)
    r_grid, xi_grid = xi_custom_fftlog(k_grid, p0_grid, kmin_cut, K_INT_MAX)

    order = np.argsort(r_grid)
    xi_on_s = np.interp(s_data, r_grid[order], xi_grid[order])

    return {
        "kmin_cut": np.array([kmin_cut]),
        "k_grid": k_grid,
        "p0_grid": p0_grid,
        "r_grid": r_grid,
        "xi_grid": xi_grid,
        "xi_on_s": xi_on_s,
    }


def compute_residual_metrics(
    s: np.ndarray,
    xi_data: np.ndarray,
    xi_std: np.ndarray,
    nmock: int,
    xi_model: np.ndarray,
    tag: str,
) -> List[Dict[str, float | str]]:
    """
    计算模型-数据差异指标（全区间 + 大尺度区间）。

    参数
    ----
    s : np.ndarray
        s 网格。
    xi_data : np.ndarray
        测量均值。
    xi_std : np.ndarray
        测量样本标准差（mock scatter）。
    nmock : int
        mock 数量。
    xi_model : np.ndarray
        模型曲线（已插值到 s 网格）。
    tag : str
        标签，写入输出表。

    返回
    ----
    List[Dict[str, float | str]]
        每个区间一条记录。
    """
    rows: List[Dict[str, float | str]] = []

    masks = {
        "all": np.ones_like(s, dtype=bool),
        f"s>={LARGE_SCALE_MIN:.0f}": s >= LARGE_SCALE_MIN,
    }

    xi_err_mean = xi_std / np.sqrt(max(nmock, 1))
    r2_data = s**2 * xi_data
    r2_model = s**2 * xi_model
    r2_std = s**2 * xi_std
    r2_err_mean = s**2 * xi_err_mean

    for region, mask in masks.items():
        if mask.sum() == 0:
            continue

        # xi 指标
        d_xi = xi_model[mask] - xi_data[mask]
        rms_xi = float(np.sqrt(np.mean(d_xi**2)))
        rms_xi_over_std = float(np.sqrt(np.mean((d_xi / np.maximum(xi_std[mask], 1e-12)) ** 2)))
        rms_xi_over_errmean = float(np.sqrt(np.mean((d_xi / np.maximum(xi_err_mean[mask], 1e-12)) ** 2)))

        # r^2 xi 指标
        d_r2 = r2_model[mask] - r2_data[mask]
        rms_r2 = float(np.sqrt(np.mean(d_r2**2)))
        rms_r2_over_std = float(np.sqrt(np.mean((d_r2 / np.maximum(r2_std[mask], 1e-12)) ** 2)))
        rms_r2_over_errmean = float(np.sqrt(np.mean((d_r2 / np.maximum(r2_err_mean[mask], 1e-12)) ** 2)))

        rows.append(
            {
                "box": tag,
                "region": region,
                "nbin": int(mask.sum()),
                "rms_xi": rms_xi,
                "rms_xi_over_std": rms_xi_over_std,
                "rms_xi_over_errmean": rms_xi_over_errmean,
                "rms_r2xi": rms_r2,
                "rms_r2xi_over_std": rms_r2_over_std,
                "rms_r2xi_over_errmean": rms_r2_over_errmean,
            }
        )

    return rows


def write_dict_rows_csv(rows: List[Dict[str, object]], out_csv: str) -> None:
    """
    将字典列表写出为 CSV（自动使用第一行键作为表头）。

    参数
    ----
    rows : List[Dict[str, object]]
        数据行。
    out_csv : str
        输出路径。
    """
    if not rows:
        raise ValueError("写 CSV 时 rows 为空")

    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_pk_fit_compare(fit1: Dict[str, object], fit3: Dict[str, object], out_png: str) -> None:
    """
    绘制两种盒长的 P0 拟合结果图。

    参数
    ----
    fit1, fit3 : Dict[str, object]
        由 fit_box_pk 返回的结果字典。
    out_png : str
        输出图像路径。
    """
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 9.0), sharex=False)

    for ax, fit, tag in [(axes[0], fit1, "1Gpc"), (axes[1], fit3, "3Gpc")]:
        kfit = np.asarray(fit["kcen_fit"])
        p0_data = np.asarray(fit["p0_mean_fit"])
        p0_std = np.asarray(fit["p0_std_fit"])
        kplot = np.asarray(fit["k_plot"])
        p0_model_plot = np.asarray(fit["p0_model_plot"])
        p0_model_fit = np.asarray(fit["p0_model_fit"])

        ax.errorbar(kfit, p0_data, yerr=p0_std, fmt="o", ms=4, capsize=2,
                    color="black", label=f"{tag} measured mean")
        ax.plot(kplot, p0_model_plot, color="tab:red", lw=1.8, label=f"{tag} best-fit model")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$P_0(k)$")
        ax.set_title(f"Task 4.2 PK Fit ({tag})")

        ratio = p0_data / np.maximum(p0_model_fit, 1e-30)
        txt = f"mean(Data/Model)={ratio.mean():.3f}"
        ax.text(0.98, 0.06, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
        ax.legend(fontsize=9)

    axes[-1].set_xlabel(r"$k\,[h/\mathrm{Mpc}]$")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_xi_model_compare(
    box1: BoxData,
    box3: BoxData,
    xi_model_1: np.ndarray,
    xi_model_3: np.ndarray,
    kmin1: float,
    kmin3: float,
    out_png: str,
) -> None:
    """
    绘制“模型 vs 测量”的 2PCF 对比图（包含标准化残差）。

    参数
    ----
    box1, box3 : BoxData
        两种盒长数据。
    xi_model_1, xi_model_3 : np.ndarray
        分别插值到数据 s 网格后的理论 xi。
    kmin1, kmin3 : float
        两种盒长使用的 kmin cut。
    out_png : str
        输出图像路径。
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.8), sharex="col")

    # 1Gpc
    s = box1.scen
    r2_data = s**2 * box1.xi_mean
    r2_std = s**2 * box1.xi_std
    r2_model = s**2 * xi_model_1
    resid = (r2_model - r2_data) / np.maximum(r2_std, 1e-12)

    ax = axes[0, 0]
    ax.errorbar(s, r2_data, yerr=r2_std, fmt="o", ms=3.8, capsize=2,
                color="black", label=f"Measured mean (N={box1.nmock})")
    ax.plot(s, r2_model, "-", lw=1.8, color="tab:blue", label="Model from best-fit P0")
    ax.set_ylabel(r"$s^2\xi_0(s)$")
    ax.set_title(f"1Gpc (kmin=2pi/L={kmin1:.6f}, kmax={K_INT_MAX:.1f})")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.axhline(0, color="k", lw=1)
    ax.axhline(1, color="gray", lw=0.8, ls="--")
    ax.axhline(-1, color="gray", lw=0.8, ls="--")
    ax.plot(s, resid, "o-", ms=3.5, lw=1.2, color="tab:blue")
    ax.set_ylabel(r"$(\Delta r^2\xi)/\sigma_{mock}$")
    ax.set_title("1Gpc Residual")

    # 3Gpc
    s = box3.scen
    r2_data = s**2 * box3.xi_mean
    r2_std = s**2 * box3.xi_std
    r2_model = s**2 * xi_model_3
    resid = (r2_model - r2_data) / np.maximum(r2_std, 1e-12)

    ax = axes[1, 0]
    ax.errorbar(s, r2_data, yerr=r2_std, fmt="o", ms=3.8, capsize=2,
                color="black", label=f"Measured mean (N={box3.nmock})")
    ax.plot(s, r2_model, "-", lw=1.8, color="tab:orange", label="Model from best-fit P0")
    ax.set_xlabel(r"$s\,[\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$s^2\xi_0(s)$")
    ax.set_title(f"3Gpc (kmin=2pi/L={kmin3:.6f}, kmax={K_INT_MAX:.1f})")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axhline(0, color="k", lw=1)
    ax.axhline(1, color="gray", lw=0.8, ls="--")
    ax.axhline(-1, color="gray", lw=0.8, ls="--")
    ax.plot(s, resid, "o-", ms=3.5, lw=1.2, color="tab:orange")
    ax.set_xlabel(r"$s\,[\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$(\Delta r^2\xi)/\sigma_{mock}$")
    ax.set_title("3Gpc Residual")

    fig.suptitle("Task 4.2: Model 2PCF (with kmin=2pi/L) vs Measured Mean", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# =====================
# 3) 主流程
# =====================

def main() -> None:
    """执行任务4全部流程，并将结果写入 mission4_log。"""

    # 1) 读取并对齐两种盒长数据
    cfg1 = BoxConfig(tag="1gpc", box_size=BOX_SIZE_1GPC, pk_glob=PK_1GPC_GLOB, pcf_glob=PCF_1GPC_GLOB)
    cfg3 = BoxConfig(tag="3gpc", box_size=BOX_SIZE_3GPC, pk_glob=PK_3GPC_GLOB, pcf_glob=PCF_3GPC_GLOB)

    keep_ids = range(REALIZATION_MIN, REALIZATION_MAX + 1)
    box1_all = load_box_data(cfg1, keep_realizations=keep_ids)
    box3_all = load_box_data(cfg3, keep_realizations=keep_ids)

    print("[INFO] 数据读取完成")
    print(f"  约束区间: realization=[{REALIZATION_MIN}..{REALIZATION_MAX}]")
    print(f"  1Gpc: Nmock={box1_all.nmock}, realization=[{box1_all.realizations.min()}..{box1_all.realizations.max()}]")
    print(f"  3Gpc: Nmock={box3_all.nmock}, realization=[{box3_all.realizations.min()}..{box3_all.realizations.max()}]")

    # 2) 任务4.1: 仅输出一张“匹配 realization”对比图（避免样本数差异干扰）
    common_ids = sorted(set(box1_all.realizations) & set(box3_all.realizations))
    box1_matched = subset_box_data(box1_all, common_ids)
    box3_matched = subset_box_data(box3_all, common_ids)

    png_measure_matched = os.path.join(OUT_DIR, "task41_measurement_compare_matched_1to80.png")
    plot_measurement_compare(
        box1_matched,
        box3_matched,
        png_measure_matched,
        title_suffix=f"matched realizations (N={len(common_ids)})",
    )

    # 3) 任务4.2: P0 拟合（分别对两个盒长）
    print("[INFO] 开始 1Gpc P0 拟合...")
    fit1 = fit_box_pk(box1_all)
    print("[INFO] 开始 3Gpc P0 拟合...")
    fit3 = fit_box_pk(box3_all)

    # 4) 用 kmin=2*pi/L 进行 FFTLog 转换，得到模型 xi
    th1 = evaluate_theory_xi_on_data_grid(box1_all.scen, fit1["bestfit"], BOX_SIZE_1GPC)
    th3 = evaluate_theory_xi_on_data_grid(box3_all.scen, fit3["bestfit"], BOX_SIZE_3GPC)
    kmin1 = float(th1["kmin_cut"][0])
    kmin3 = float(th3["kmin_cut"][0])

    # 5) 计算模型-测量差异指标
    metrics_rows: List[Dict[str, float | str]] = []
    metrics_rows.extend(
        compute_residual_metrics(
            s=box1_all.scen,
            xi_data=box1_all.xi_mean,
            xi_std=box1_all.xi_std,
            nmock=box1_all.nmock,
            xi_model=np.asarray(th1["xi_on_s"]),
            tag="1gpc",
        )
    )
    metrics_rows.extend(
        compute_residual_metrics(
            s=box3_all.scen,
            xi_data=box3_all.xi_mean,
            xi_std=box3_all.xi_std,
            nmock=box3_all.nmock,
            xi_model=np.asarray(th3["xi_on_s"]),
            tag="3gpc",
        )
    )

    # 6) 模型 vs 测量图
    png_xi_cmp = os.path.join(OUT_DIR, "task42_xi_model_vs_measurement_1to80.png")
    plot_xi_model_compare(
        box1=box1_all,
        box3=box3_all,
        xi_model_1=np.asarray(th1["xi_on_s"]),
        xi_model_3=np.asarray(th3["xi_on_s"]),
        kmin1=kmin1,
        kmin3=kmin3,
        out_png=png_xi_cmp,
    )

    # 7) 文本总结（先写定量指标，便于你快速看结论）
    def pick_metric(box: str, region: str, key: str) -> float:
        for row in metrics_rows:
            if row["box"] == box and row["region"] == region:
                return float(row[key])
        raise KeyError((box, region, key))

    # 重点看大尺度 region 的 r^2xi 残差（按 mock scatter 标准化）
    m1 = pick_metric("1gpc", f"s>={LARGE_SCALE_MIN:.0f}", "rms_r2xi_over_std")
    m3 = pick_metric("3gpc", f"s>={LARGE_SCALE_MIN:.0f}", "rms_r2xi_over_std")

    summary_txt = os.path.join(OUT_DIR, "task4_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("任务4总结（自动生成）\n")
        f.write("====================\n")
        f.write(f"1) 4.1 数据规模:\n")
        f.write(f"   - 1Gpc all: Nmock={box1_all.nmock}, realization={box1_all.realizations.min()}..{box1_all.realizations.max()}\n")
        f.write(f"   - 3Gpc all: Nmock={box3_all.nmock}, realization={box3_all.realizations.min()}..{box3_all.realizations.max()}\n")
        f.write(f"   - matched: N={len(common_ids)}, realization={common_ids[0]}..{common_ids[-1]}\n")
        f.write(f"   - 本轮限制: realization={REALIZATION_MIN}..{REALIZATION_MAX}\n")
        f.write("\n")
        f.write("2) 4.2 k-cut 设置:\n")
        f.write(f"   - 1Gpc: kmin=2pi/L={kmin1:.8f}, kmax={K_INT_MAX:.2f}\n")
        f.write(f"   - 3Gpc: kmin=2pi/L={kmin3:.8f}, kmax={K_INT_MAX:.2f}\n")
        f.write("\n")
        f.write("3) 模型与测量差异（大尺度, 指标=RMS[(delta r^2xi)/sigma_mock]）:\n")
        f.write(f"   - 1Gpc: {m1:.4f}\n")
        f.write(f"   - 3Gpc: {m3:.4f}\n")
        if m3 < m1:
            f.write("   - 结论: 3Gpc 在该指标上偏差更小（与任务4问题一致，差异变小）。\n")
        elif m3 > m1:
            f.write("   - 结论: 3Gpc 在该指标上未变小（需进一步检查拟合区间/模型设定）。\n")
        else:
            f.write("   - 结论: 两者在该指标上几乎一致。\n")

        f.write("\n4) 主要输出文件:\n")
        f.write("   - task41_measurement_compare_matched_1to80.png\n")
        f.write("   - task42_xi_model_vs_measurement_1to80.png\n")
        f.write("   - task4_summary.txt\n")
        f.write("\n5) best-fit 参数（用于复现实验）:\n")
        f.write("   - 1Gpc: " + json.dumps(fit1["bestfit"], ensure_ascii=False) + "\n")
        f.write("   - 3Gpc: " + json.dumps(fit3["bestfit"], ensure_ascii=False) + "\n")

    print("[INFO] 任务4分析完成。")
    print(f"[INFO] 输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
