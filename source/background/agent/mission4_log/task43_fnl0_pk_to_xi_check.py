#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本名称
--------
task43_fnl0_pk_to_xi_check.py

代码大纲（执行逻辑）
--------------------
1) 读取 3Gpc fnl=0 的测量数据：
   - P0(k): /3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat
   - xi0(s): /3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat
   并在 realization 上做交集对齐，得到测量均值与 mock 离散。

2) 对测量均值 P0 做 best-fit：
   - 使用 desilike 的 PNGTracerPowerSpectrumMultipoles（与任务4.2一致）
   - 拟合区间使用低-k（k<=0.0635）

3) 用 best-fit P0 做 FFTLog 积分得到 xi0(s)：
   - 固定 kmax=20
   - 扫描一组更小的 kmin（包括 2*pi/L 以及更小数值），检查收敛与大尺度匹配

4) 输出：
   - 一张图：测量均值 vs 不同 kmin 的模型曲线（r^2 xi）
   - 一份总结文本：best-fit 参数 + 大尺度(s>=200)偏差指标与结论
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fht, fhtoffset


# =====================
# 参数区
# =====================
PK_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat"
PCF_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat"

OUT_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission4_log"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "task43_fnl0_xi_kmin_scan.png")
OUT_TXT = os.path.join(OUT_DIR, "task43_summary.txt")

# 可选 realization 范围（默认全量 N2..N99）
# 支持用环境变量临时覆盖，便于先做预览：
#   TASK43_RMIN=2 TASK43_RMAX=50 python task43_fnl0_pk_to_xi_check.py
RID_MIN = int(os.getenv("TASK43_RMIN", "2"))
RID_MAX = int(os.getenv("TASK43_RMAX", "99"))

# 数据列定义
PK_KCEN_COL = 0
PK_KMIN_COL = 1
PK_KMAX_COL = 2
PK_P0_COL = 5

PCF_SCEN_COL = 0
PCF_XI0_COL = 3

# 盒长与积分设置
BOX_SIZE_3GPC = 3000.0
K_INT_MAX = 20.0
KMIN_BOX = 2.0 * np.pi / BOX_SIZE_3GPC
KMIN_LIST = [KMIN_BOX, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 1e-5]

# 拟合设置（与任务4.2保持一致）
PK_FIT_KMAX = 0.0635
MINUIT_SEED = 66
MINUIT_NITER = 25
UNIT_Z = 1.0
FIXED_P = 1.1
FIXED_SN0 = 0.0
FIXED_SIGMAS = 0.0

# FFTLog 参数
FFTLOG_N = 4096
FFTLOG_PADDING = 4.0
FFTLOG_MU = 0.5
FFTLOG_BIAS = 0.0
EDGE_TAPER_FRAC = 0.06

# 大尺度判据
LARGE_SCALE_MIN = 200.0

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"


# =====================
# desilike 依赖
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


@dataclass
class Data3GpcFnl0:
    """保存 3Gpc fnl0 对齐后的测量数据。"""

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
        """P0 测量均值。"""
        return np.mean(self.p0_mocks, axis=0)

    @property
    def p0_std(self) -> np.ndarray:
        """P0 mock 离散。"""
        return np.std(self.p0_mocks, axis=0, ddof=1)

    @property
    def xi_mean(self) -> np.ndarray:
        """xi0 测量均值。"""
        return np.mean(self.xi_mocks, axis=0)

    @property
    def xi_std(self) -> np.ndarray:
        """xi0 mock 离散。"""
        return np.std(self.xi_mocks, axis=0, ddof=1)


def parse_realization_id(path: str) -> int:
    """从文件名提取 realization 编号。"""
    m = re.search(r"_N(\d+)\.dat$", os.path.basename(path))
    if m is None:
        raise ValueError(f"无法解析 realization: {path}")
    return int(m.group(1))


def assert_same_grid(arrays: List[np.ndarray], name: str) -> None:
    """检查网格是否一致。"""
    ref = arrays[0]
    for i, arr in enumerate(arrays[1:], start=1):
        if not np.allclose(ref, arr):
            raise ValueError(f"{name} 网格不一致: index={i}")


def load_data() -> Data3GpcFnl0:
    """
    读取并对齐 fnl0 的 pk/pcf 数据。

    返回
    ----
    Data3GpcFnl0
        对齐后的测量数据结构。
    """
    pk_files = glob.glob(PK_GLOB)
    pcf_files = glob.glob(PCF_GLOB)
    if not pk_files or not pcf_files:
        raise FileNotFoundError("fnl0 的 pk/pcf 数据文件不存在或为空")

    pk_map = {parse_realization_id(fp): fp for fp in pk_files}
    pcf_map = {parse_realization_id(fp): fp for fp in pcf_files}
    common = sorted(set(pk_map) & set(pcf_map))
    common = [rid for rid in common if RID_MIN <= rid <= RID_MAX]
    if not common:
        raise RuntimeError("pk 与 pcf 没有共同 realization")

    pk_tables = [np.loadtxt(pk_map[r], comments="#") for r in common]
    pcf_tables = [np.loadtxt(pcf_map[r], comments="#") for r in common]

    kcen_list = [t[:, PK_KCEN_COL] for t in pk_tables]
    kmin_list = [t[:, PK_KMIN_COL] for t in pk_tables]
    kmax_list = [t[:, PK_KMAX_COL] for t in pk_tables]
    assert_same_grid(kcen_list, "kcen")
    assert_same_grid(kmin_list, "kmin")
    assert_same_grid(kmax_list, "kmax")

    scen_list = [t[:, PCF_SCEN_COL] for t in pcf_tables]
    assert_same_grid(scen_list, "scen")

    return Data3GpcFnl0(
        realizations=np.array(common, dtype=int),
        kcen=kcen_list[0].copy(),
        kmin=kmin_list[0].copy(),
        kmax=kmax_list[0].copy(),
        p0_mocks=np.vstack([t[:, PK_P0_COL] for t in pk_tables]),
        scen=scen_list[0].copy(),
        xi_mocks=np.vstack([t[:, PCF_XI0_COL] for t in pcf_tables]),
    )


def convert_bestfit_to_float_dict(bestfit: Dict[str, object]) -> Dict[str, float]:
    """把 desilike 参数对象转成 float 字典。"""
    out: Dict[str, float] = {}
    for key, val in bestfit.items():
        out[key] = float(np.ravel(np.asarray(val))[0])
    return out


def fit_best_pk(data: Data3GpcFnl0) -> Dict[str, object]:
    """
    对 3Gpc fnl0 的测量均值 P0 做 best-fit。

    参数
    ----
    data : Data3GpcFnl0
        输入数据结构。

    返回
    ----
    Dict[str, object]
        包含 bestfit 字典以及用于绘图的模型曲线。
    """
    mask = data.kcen <= PK_FIT_KMAX
    if mask.sum() < 5:
        raise RuntimeError("低-k 拟合点太少")

    kfit = data.kcen[mask]
    p0_mean_fit = data.p0_mean[mask]
    p0_mocks_fit = data.p0_mocks[:, mask]

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

    # 输出平滑 P0，用于 FFTLog 积分
    k_plot = np.geomspace(float(kfit.min()), K_INT_MAX * FFTLOG_PADDING, 600)
    theory_plot = PNGTracerPowerSpectrumMultipoles(k=k_plot, template=template, mode="b-p")
    theory_plot.init.params["p"].update(fixed=True, value=FIXED_P)
    theory_plot.init.params["sn0"].update(fixed=True, value=FIXED_SN0)
    theory_plot.init.params["sigmas"].update(fixed=False, value=FIXED_SIGMAS)
    theory_plot(**bestfit)
    p0_plot = np.asarray(theory_plot.power[0], dtype=float)

    return {"bestfit": bestfit, "k_plot": k_plot, "p0_plot": p0_plot}


def build_log_taper_window(k_array: np.ndarray, kmin: float, kmax: float, frac: float) -> np.ndarray:
    """构建 log(k) 边界平滑窗，避免硬截断振铃。"""
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


def xi_custom_fftlog(k_grid: np.ndarray, p0_grid: np.ndarray, kmin: float, kmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用自定义 FFTLog 计算 xi0(r)。

    参数
    ----
    k_grid, p0_grid : np.ndarray
        对数均匀 k 网格及其 P0(k)。
    kmin, kmax : float
        积分上下限。

    返回
    ----
    Tuple[np.ndarray, np.ndarray]
        (r_grid, xi_grid)
    """
    p0_cut = p0_grid * build_log_taper_window(k_grid, kmin, kmax, EDGE_TAPER_FRAC)
    dln = np.log(k_grid[1] / k_grid[0])
    offset = fhtoffset(dln, mu=FFTLOG_MU, initial=0.0, bias=FFTLOG_BIAS)

    # j0 Hankel 对应关系：a_in = k^(3/2) P(k)
    a_in = (k_grid ** 1.5) * p0_cut
    a_out = fht(a_in, dln=dln, mu=FFTLOG_MU, offset=offset, bias=FFTLOG_BIAS)

    n = k_grid.size
    j = np.arange(n)
    jc = (n - 1) / 2.0
    ln_kc = 0.5 * (np.log(k_grid[0]) + np.log(k_grid[-1]))
    r_grid = np.exp((offset - ln_kc) + (j - jc) * dln)

    const = np.sqrt(np.pi / 2.0) / (2.0 * np.pi**2)
    xi_grid = const * a_out / (r_grid ** 1.5)
    return r_grid, xi_grid


def evaluate_xi_on_s(bestfit: Dict[str, float], s_data: np.ndarray, kmin: float) -> np.ndarray:
    """
    给定 kmin 与 best-fit 参数，计算并插值 xi 到测量 s 网格。

    参数
    ----
    bestfit : Dict[str, float]
        P0 best-fit 参数。
    s_data : np.ndarray
        测量 s 网格。
    kmin : float
        积分下限。

    返回
    ----
    np.ndarray
        与 s_data 同长度的模型 xi。
    """
    # 在较宽区间上建对数网格，再由 window 控制有效积分范围
    k_grid = np.geomspace(kmin / FFTLOG_PADDING, K_INT_MAX * FFTLOG_PADDING, FFTLOG_N)

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
    theory(**bestfit)
    p0_grid = np.asarray(theory.power[0], dtype=float)

    r_grid, xi_grid = xi_custom_fftlog(k_grid, p0_grid, kmin=kmin, kmax=K_INT_MAX)
    order = np.argsort(r_grid)
    return np.interp(s_data, r_grid[order], xi_grid[order])


def main() -> None:
    """主流程：执行 4.3 分析并输出图与总结。"""
    data = load_data()
    print(f"[INFO] fnl0 data loaded: Nmock={data.nmock}, realization={data.realizations.min()}..{data.realizations.max()}")

    fit = fit_best_pk(data)
    bestfit = fit["bestfit"]
    print("[INFO] best-fit done")

    # 扫描 kmin
    xi_models: Dict[float, np.ndarray] = {}
    for kmin in KMIN_LIST:
        xi_models[kmin] = evaluate_xi_on_s(bestfit, data.scen, kmin=kmin)

    s = data.scen
    xi_data = data.xi_mean
    xi_std = data.xi_std
    mask_large = s >= LARGE_SCALE_MIN

    # 计算大尺度偏差指标（以 s^2 xi 为主，更直观看大尺度）
    metric_rows: List[Tuple[float, float]] = []
    for kmin in KMIN_LIST:
        r2_data = s**2 * xi_data
        r2_model = s**2 * xi_models[kmin]
        r2_std = s**2 * xi_std
        delta = r2_model[mask_large] - r2_data[mask_large]
        norm = np.maximum(r2_std[mask_large], 1e-12)
        rms_over_std = float(np.sqrt(np.mean((delta / norm) ** 2)))
        metric_rows.append((kmin, rms_over_std))

    # 收敛性：比较最小两个 kmin 曲线在大尺度的差异
    kmin_small = min(KMIN_LIST)
    kmin_next = sorted(KMIN_LIST)[1]
    d_conv = np.abs((s**2 * xi_models[kmin_small]) - (s**2 * xi_models[kmin_next]))
    conv_rms = float(np.sqrt(np.mean((d_conv[mask_large]) ** 2)))

    # 出图（仅一张）
    fig, ax = plt.subplots(1, 1, figsize=(9.2, 6.8))
    ax.errorbar(
        s,
        s**2 * xi_data,
        yerr=s**2 * xi_std,
        fmt="o",
        ms=3.4,
        color="black",
        capsize=2,
        label=f"Measured mean (N={data.nmock})",
    )

    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(KMIN_LIST)))
    for c, kmin in zip(colors, KMIN_LIST):
        ax.plot(s, s**2 * xi_models[kmin], "-", lw=1.6, color=c, label=f"model kmin={kmin:.2e}")

    ax.set_xlabel(r"$s\,[\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$s^2\xi_0(s)$")
    ax.set_title("Task 4.3 (3Gpc fnl=0): best-fit PK -> 2PCF with small kmin")
    ax.legend(fontsize=8, ncol=2)

    txt = (
        f"best-fit: fnl_loc={bestfit['fnl_loc']:.3f}, "
        f"b1={bestfit['b1']:.3f}, sigmas={bestfit['sigmas']:.4f}\n"
        f"large-scale metric: RMS[(Δ s^2ξ)/σ_mock], s>={LARGE_SCALE_MIN:.0f}"
    )
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(facecolor="white", alpha=0.86, edgecolor="gray"),
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    # 文本总结
    best_kmin, best_metric = min(metric_rows, key=lambda x: x[1])
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("任务4.3总结（3Gpc fnl=0）\n")
        f.write("========================\n")
        f.write(f"数据规模: Nmock={data.nmock}, realization={data.realizations.min()}..{data.realizations.max()}\n")
        f.write(f"best-fit参数: {bestfit}\n")
        f.write(f"积分设置: kmax={K_INT_MAX:.2f}, kmin扫描={KMIN_LIST}\n")
        f.write("\n大尺度偏差指标（s^2xi, s>=200）:\n")
        for kmin, metric in metric_rows:
            f.write(f"  kmin={kmin:.2e}: RMS[(Δ s^2ξ)/σ_mock]={metric:.4f}\n")
        f.write(f"\n最优kmin(该指标最小): {best_kmin:.2e}, 指标={best_metric:.4f}\n")
        f.write(f"收敛性检查: 两个最小kmin曲线的大尺度RMS差={conv_rms:.4e}\n")

        if best_metric <= 1.0:
            f.write("结论: 在大尺度上，模型与测量均值总体可认为一致（偏差约在1σ内）。\n")
        elif best_metric <= 2.0:
            f.write("结论: 在大尺度上，模型与测量均值基本接近，但仍有可见偏差（约1~2σ）。\n")
        else:
            f.write("结论: 在大尺度上，模型与测量均值仍有显著偏差（>2σ）。\n")

    print(f"[INFO] saved figure: {OUT_PNG}")
    print(f"[INFO] saved summary: {OUT_TXT}")


if __name__ == "__main__":
    main()
