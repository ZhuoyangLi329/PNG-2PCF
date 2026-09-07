#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本名称
--------
task421_pk_bestfit_box_compare.py

代码大纲（执行逻辑）
--------------------
1) 参数区：
   - 设置 1Gpc/3Gpc 的 pk_masscut 输入路径
   - 设置拟合区间、输出图路径
2) 读入测量的 P0：
   - 分别读取两种盒长的所有 realization（1Gpc: N1..50；3Gpc fnl100: N2..99）
   - 计算每个盒子的测量均值与标准差
3) 做 best-fit：
   - 使用 desilike + Minuit，对每个盒子拟合低-k 区间的 P0（monopole）
4) 出图（仅一张图）：
   - 同时画两种盒子的“测量均值曲线 + best-fit 曲线”
   - 图中直接标注 best-fit 参数（fnl_loc, b1, sigmas）
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


# =====================
# 参数区
# =====================
PK_1GPC_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat"
PK_3GPC_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat"

OUT_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission4_log"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "task421_pk_bestfit_box_compare_1to99.png")

# 仅为保证两边口径一致：3Gpc 用 N2..99（当前已齐）；1Gpc 本身只有 N1..50
RMIN_1GPC = 1
RMAX_1GPC = 50
RMIN_3GPC = 2
RMAX_3GPC = 99

# 拟合区间（沿用任务4已有设置）
PK_FIT_KMAX = 0.0635
MINUIT_SEED = 66
MINUIT_NITER = 25

UNIT_Z = 1.0
FIXED_P = 1.1
FIXED_SN0 = 0.0
FIXED_SIGMAS = 0.0

# pk 文件列定义（与当前数据格式一致）
PK_KCEN_COL = 0
PK_KMIN_COL = 1
PK_KMAX_COL = 2
PK_P0_COL = 5

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
class PkBoxData:
    """保存单个盒长的 P0 mock 数据与网格。"""

    tag: str
    realizations: np.ndarray
    kcen: np.ndarray
    kmin: np.ndarray
    kmax: np.ndarray
    p0_mocks: np.ndarray

    @property
    def nmock(self) -> int:
        """mock 数量。"""
        return int(self.p0_mocks.shape[0])

    @property
    def p0_mean(self) -> np.ndarray:
        """测量均值。"""
        return np.mean(self.p0_mocks, axis=0)

    @property
    def p0_std(self) -> np.ndarray:
        """mock 离散标准差。"""
        return np.std(self.p0_mocks, axis=0, ddof=1)


def parse_realization_id(path: str) -> int:
    """从文件名里解析 realization 编号。"""
    m = re.search(r"_N(\d+)\.dat$", os.path.basename(path))
    if m is None:
        raise ValueError(f"无法解析 realization: {path}")
    return int(m.group(1))


def assert_same_grid(arrays: List[np.ndarray], name: str) -> None:
    """检查不同 realization 的网格是否一致。"""
    ref = arrays[0]
    for i, arr in enumerate(arrays[1:], start=1):
        if not np.allclose(ref, arr):
            raise ValueError(f"{name} 网格不一致: index={i}")


def load_pk_box(tag: str, file_glob: str, rmin: int, rmax: int) -> PkBoxData:
    """
    读取某个盒长的 P0 全部 realization，并构建 mock 矩阵。

    参数
    ----
    tag : str
        标签名，例如 1gpc / 3gpc。
    file_glob : str
        文件通配符。
    rmin, rmax : int
        realization 范围。

    返回
    ----
    PkBoxData
        包含网格与 mock 矩阵的数据结构。
    """
    files = glob.glob(file_glob)
    if not files:
        raise FileNotFoundError(f"未找到文件: {file_glob}")

    selected: List[str] = []
    rids: List[int] = []
    for fp in files:
        rid = parse_realization_id(fp)
        if rmin <= rid <= rmax:
            selected.append(fp)
            rids.append(rid)

    if not selected:
        raise RuntimeError(f"{tag} 在给定范围内没有文件: [{rmin}, {rmax}]")

    order = np.argsort(np.array(rids))
    selected = [selected[i] for i in order]
    rids = [rids[i] for i in order]

    tables = [np.loadtxt(fp, comments="#") for fp in selected]
    kcen_list = [t[:, PK_KCEN_COL] for t in tables]
    kmin_list = [t[:, PK_KMIN_COL] for t in tables]
    kmax_list = [t[:, PK_KMAX_COL] for t in tables]
    assert_same_grid(kcen_list, f"{tag}:kcen")
    assert_same_grid(kmin_list, f"{tag}:kmin")
    assert_same_grid(kmax_list, f"{tag}:kmax")

    p0_mocks = np.vstack([t[:, PK_P0_COL] for t in tables])
    return PkBoxData(
        tag=tag,
        realizations=np.array(rids, dtype=int),
        kcen=kcen_list[0].copy(),
        kmin=kmin_list[0].copy(),
        kmax=kmax_list[0].copy(),
        p0_mocks=p0_mocks,
    )


def convert_bestfit_to_float_dict(bestfit: Dict[str, object]) -> Dict[str, float]:
    """把 desilike 的参数对象转成 float 字典。"""
    out: Dict[str, float] = {}
    for k, v in bestfit.items():
        out[k] = float(np.ravel(np.asarray(v))[0])
    return out


def fit_box_pk(box: PkBoxData) -> Dict[str, object]:
    """
    对单个盒长拟合 P0(k)，返回 best-fit 与模型曲线。

    参数
    ----
    box : PkBoxData
        输入盒长数据。

    返回
    ----
    Dict[str, object]
        包含 bestfit、模型曲线、拟合区间数据等。
    """
    mask = box.kcen <= PK_FIT_KMAX
    if mask.sum() < 5:
        raise RuntimeError(f"{box.tag}: 拟合点数太少")

    kcen_fit = box.kcen[mask]
    p0_mean_fit = box.p0_mean[mask]
    p0_mocks_fit = box.p0_mocks[:, mask]

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

    profiler = MinuitProfiler(likelihood, seed=MINUIT_SEED)
    profiles = profiler.maximize(niterations=MINUIT_NITER)
    bestfit = convert_bestfit_to_float_dict(profiles.bestfit.choice(input=True))

    # 拟合网格上的模型（用于参考）
    theory_fit = PNGTracerPowerSpectrumMultipoles(k=kcen_fit, template=template, mode="b-p")
    theory_fit.init.params["p"].update(fixed=True, value=FIXED_P)
    theory_fit.init.params["sn0"].update(fixed=True, value=FIXED_SN0)
    theory_fit.init.params["sigmas"].update(fixed=False, value=FIXED_SIGMAS)
    theory_fit(**bestfit)
    p0_model_fit = np.asarray(theory_fit.power[0], dtype=float)

    # 展示网格上的模型（画图更平滑）
    k_plot = np.geomspace(float(kcen_fit.min()), 0.3, 500)
    theory_plot = PNGTracerPowerSpectrumMultipoles(k=k_plot, template=template, mode="b-p")
    theory_plot.init.params["p"].update(fixed=True, value=FIXED_P)
    theory_plot.init.params["sn0"].update(fixed=True, value=FIXED_SN0)
    theory_plot.init.params["sigmas"].update(fixed=False, value=FIXED_SIGMAS)
    theory_plot(**bestfit)
    p0_model_plot = np.asarray(theory_plot.power[0], dtype=float)

    return {
        "bestfit": bestfit,
        "kcen_fit": kcen_fit,
        "p0_model_fit": p0_model_fit,
        "k_plot": k_plot,
        "p0_model_plot": p0_model_plot,
    }


def format_bestfit_text(tag: str, bestfit: Dict[str, float]) -> str:
    """把 best-fit 参数格式化成图注字符串。"""
    return (
        f"{tag} best-fit\n"
        f"fnl_loc={bestfit['fnl_loc']:.3f}\n"
        f"b1={bestfit['b1']:.3f}\n"
        f"sigmas={bestfit['sigmas']:.4f}"
    )


def main() -> None:
    """主流程：读取数据、拟合、出一张对比图。"""
    box1 = load_pk_box("1Gpc", PK_1GPC_GLOB, RMIN_1GPC, RMAX_1GPC)
    box3 = load_pk_box("3Gpc", PK_3GPC_GLOB, RMIN_3GPC, RMAX_3GPC)

    print(f"[INFO] 1Gpc mocks: {box1.nmock}, realization={box1.realizations.min()}..{box1.realizations.max()}")
    print(f"[INFO] 3Gpc mocks: {box3.nmock}, realization={box3.realizations.min()}..{box3.realizations.max()}")

    fit1 = fit_box_pk(box1)
    fit3 = fit_box_pk(box3)
    print("[INFO] 拟合完成")

    # 仅一张图：叠加测量均值与 best-fit 模型
    fig, ax = plt.subplots(1, 1, figsize=(8.8, 6.6))

    # 1Gpc
    ax.plot(box1.kcen, box1.p0_mean, "o", ms=3.4, color="tab:blue", alpha=0.85, label=f"1Gpc measured mean (N={box1.nmock})")
    ax.fill_between(
        box1.kcen,
        np.maximum(box1.p0_mean - box1.p0_std, 1e-30),
        box1.p0_mean + box1.p0_std,
        color="tab:blue",
        alpha=0.14,
    )
    ax.plot(fit1["k_plot"], fit1["p0_model_plot"], "-", lw=2.0, color="tab:blue", alpha=0.95, label="1Gpc best-fit model")

    # 3Gpc
    ax.plot(box3.kcen, box3.p0_mean, "s", ms=3.2, color="tab:orange", alpha=0.85, label=f"3Gpc measured mean (N={box3.nmock})")
    ax.fill_between(
        box3.kcen,
        np.maximum(box3.p0_mean - box3.p0_std, 1e-30),
        box3.p0_mean + box3.p0_std,
        color="tab:orange",
        alpha=0.14,
    )
    ax.plot(fit3["k_plot"], fit3["p0_model_plot"], "-", lw=2.0, color="tab:orange", alpha=0.95, label="3Gpc best-fit model")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k\,[h/\mathrm{Mpc}]$")
    ax.set_ylabel(r"$P_0(k)$")
    ax.set_title("Task 4.2.1: Measured Mean PK vs Best-fit PK (1Gpc vs 3Gpc)")
    ax.legend(fontsize=9, ncol=2)

    txt1 = format_bestfit_text("1Gpc", fit1["bestfit"])
    txt3 = format_bestfit_text("3Gpc", fit3["bestfit"])
    ax.text(
        0.03,
        0.97,
        txt1,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="tab:blue",
        bbox=dict(facecolor="white", edgecolor="tab:blue", alpha=0.86),
    )
    ax.text(
        0.67,
        0.97,
        txt3,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="tab:orange",
        bbox=dict(facecolor="white", edgecolor="tab:orange", alpha=0.86),
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] saved figure: {OUT_PNG}")


if __name__ == "__main__":
    main()
