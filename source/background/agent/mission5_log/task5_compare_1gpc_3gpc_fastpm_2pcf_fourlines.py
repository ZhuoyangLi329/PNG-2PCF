#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本名称
--------
task5_compare_1gpc_3gpc_fastpm_2pcf_fourlines.py

代码大纲（执行逻辑关系）
------------------------
第 0 部分：参数区
- 统一定义 1Gpc / 3Gpc、fnl0 / fnl100 的 2PCF 输入路径与输出图路径。

第 1 部分：基础函数
- 解析 realization 编号。
- 读取单个 2PCF 文件。
- 批量读取某一组样本，并计算测量均值。

第 2 部分：主流程
- 分别读取四组 FastPM 测量的 2PCF：
  1) 1Gpc fnl0
  2) 1Gpc fnl100
  3) 3Gpc fnl0
  4) 3Gpc fnl100
- 在统一的 r 网格上画四条平均值曲线。

第 3 部分：作图规则
- 不同盒长 L 用不同颜色：
  - 1Gpc: 蓝色
  - 3Gpc: 橙色
- 同一颜色下，不同 fnl 用不同线型：
  - fnl0: 实线
  - fnl100: 虚线
- 为了与前面 mission5 的图保持一致，纵轴使用 r^2 * xi0(r)。
- 误差带使用样本协方差矩阵对角元的平方根：
  error(r) = sqrt(Cov[r, r])
  对当前 realization 集合，这与样本标准差 std 等价，而不是 std/sqrt(N)。

说明
----
1. 本脚本画“测量平均值 + covariance 对角元误差带”的四条 2PCF 曲线。
2. 使用所有可用的 FastPM realization：
   - 1Gpc: 50 个
   - 3Gpc: 98 个（N=2..99）
3. 若不同文件的 r 网格存在微小差异，会插值到第一份文件的网格上再求平均。
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =====================
# 0) 参数区
# =====================

# 1Gpc FastPM
PCF_1GPC_FNL100_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat"
PCF_1GPC_FNL0_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut_fnl0/pcf_rsd_N*.dat"

# 3Gpc FastPM
PCF_3GPC_FNL100_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat"
PCF_3GPC_FNL0_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat"

# 输出图
OUT_FIG = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission5_log/task5_compare_1gpc_3gpc_fastpm_2pcf_fourlines.png"

# realization 编号解析规则
REAL_RE = re.compile(r"_N(\d+)\.dat$")

# 颜色与线型
COLOR_1GPC = "#1f77b4"   # 蓝色
COLOR_3GPC = "#ff7f0e"   # 橙色
LS_FNL0 = "-"
LS_FNL100 = "--"


# =====================
# 1) 基础函数
# =====================


@dataclass
class XiDataset:
    """
    保存一组 2PCF 样本的公共网格与均值信息。

    参数
    ----
    tag : str
        数据标签，例如 '1gpc_fnl0'。
    s : np.ndarray
        2PCF 的 s 中心网格。
    xi : np.ndarray
        所有 realization 的 xi0 样本矩阵，shape=(Nmock, Ns)。
    """

    tag: str
    s: np.ndarray
    xi: np.ndarray

    @property
    def nmock(self) -> int:
        """返回 realization 数量。"""
        return int(self.xi.shape[0])

    @property
    def mean(self) -> np.ndarray:
        """返回 xi0 的测量平均值。"""
        return np.mean(self.xi, axis=0)

    @property
    def std(self) -> np.ndarray:
        """返回 xi0 的样本标准差。"""
        return np.std(self.xi, axis=0, ddof=1)

    @property
    def cov_diag_error(self) -> np.ndarray:
        """返回由样本协方差对角元给出的误差条，即 sqrt(diag(Cov))。"""
        return self.std


def realization_id(path: str) -> int:
    """
    从 2PCF 文件名解析 realization 编号。

    参数
    ----
    path : str
        文件路径。

    返回
    ----
    int
        realization 编号。
    """
    match = REAL_RE.search(os.path.basename(path))
    if match is None:
        raise ValueError(f"无法从文件名解析 realization 编号: {path}")
    return int(match.group(1))


def read_pcf(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    读取单个 2PCF 文件。

    参数
    ----
    path : str
        pcf 文件路径。

    返回
    ----
    tuple[np.ndarray, np.ndarray]
        - s: 2PCF 的距离中心
        - xi0: monopole
    """
    arr = np.loadtxt(path, comments="#")
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"2PCF 文件格式异常: {path}")
    s = np.asarray(arr[:, 0], dtype=float)
    xi0 = np.asarray(arr[:, 3], dtype=float)
    return s, xi0


def load_dataset(tag: str, pattern: str) -> XiDataset:
    """
    批量读取一组 2PCF 文件，并对齐到公共 s 网格后求均值。

    参数
    ----
    tag : str
        数据标签。
    pattern : str
        文件 glob 表达式。

    返回
    ----
    XiDataset
        该组样本的公共网格和 xi 样本矩阵。
    """
    files = sorted(glob.glob(pattern), key=realization_id)
    if not files:
        raise RuntimeError(f"没有匹配到文件: {pattern}")

    s_ref, xi_ref = read_pcf(files[0])
    xi_rows = [xi_ref]

    for fp in files[1:]:
        s, xi = read_pcf(fp)
        if np.allclose(s, s_ref, rtol=0.0, atol=0.0):
            xi_rows.append(xi)
        else:
            xi_rows.append(np.interp(s_ref, s, xi))

    return XiDataset(
        tag=tag,
        s=s_ref,
        xi=np.asarray(xi_rows, dtype=float),
    )


# =====================
# 2) 主流程
# =====================


def main() -> None:
    """
    读取四组 FastPM 2PCF 平均值，并画成一张四曲线对比图。
    """
    data_1gpc_fnl0 = load_dataset("1gpc_fnl0", PCF_1GPC_FNL0_GLOB)
    data_1gpc_fnl100 = load_dataset("1gpc_fnl100", PCF_1GPC_FNL100_GLOB)
    data_3gpc_fnl0 = load_dataset("3gpc_fnl0", PCF_3GPC_FNL0_GLOB)
    data_3gpc_fnl100 = load_dataset("3gpc_fnl100", PCF_3GPC_FNL100_GLOB)

    # 默认四组的 s 网格是一致的；若 3Gpc 稍有差异，则插值到 1Gpc 的网格上统一展示。
    s_plot = data_1gpc_fnl0.s
    y_1gpc_fnl0 = s_plot**2 * data_1gpc_fnl0.mean
    y_1gpc_fnl100 = s_plot**2 * data_1gpc_fnl100.mean
    y_3gpc_fnl0 = s_plot**2 * np.interp(s_plot, data_3gpc_fnl0.s, data_3gpc_fnl0.mean)
    y_3gpc_fnl100 = s_plot**2 * np.interp(s_plot, data_3gpc_fnl100.s, data_3gpc_fnl100.mean)

    e_1gpc_fnl0 = s_plot**2 * data_1gpc_fnl0.cov_diag_error
    e_1gpc_fnl100 = s_plot**2 * data_1gpc_fnl100.cov_diag_error
    e_3gpc_fnl0 = s_plot**2 * np.interp(s_plot, data_3gpc_fnl0.s, data_3gpc_fnl0.cov_diag_error)
    e_3gpc_fnl100 = s_plot**2 * np.interp(s_plot, data_3gpc_fnl100.s, data_3gpc_fnl100.cov_diag_error)

    fig, ax = plt.subplots(figsize=(8.8, 6.0))

    ax.fill_between(
        s_plot,
        y_1gpc_fnl0 - e_1gpc_fnl0,
        y_1gpc_fnl0 + e_1gpc_fnl0,
        color=COLOR_1GPC,
        alpha=0.16,
        linewidth=0.0,
    )
    ax.plot(
        s_plot,
        y_1gpc_fnl0,
        color=COLOR_1GPC,
        lw=2.2,
        ls=LS_FNL0,
        label="L=1 Gpc/h, fnl=0",
    )
    ax.fill_between(
        s_plot,
        y_1gpc_fnl100 - e_1gpc_fnl100,
        y_1gpc_fnl100 + e_1gpc_fnl100,
        color=COLOR_1GPC,
        alpha=0.10,
        linewidth=0.0,
    )
    ax.plot(
        s_plot,
        y_1gpc_fnl100,
        color=COLOR_1GPC,
        lw=2.2,
        ls=LS_FNL100,
        label="L=1 Gpc/h, fnl=100",
    )
    ax.fill_between(
        s_plot,
        y_3gpc_fnl0 - e_3gpc_fnl0,
        y_3gpc_fnl0 + e_3gpc_fnl0,
        color=COLOR_3GPC,
        alpha=0.16,
        linewidth=0.0,
    )
    ax.plot(
        s_plot,
        y_3gpc_fnl0,
        color=COLOR_3GPC,
        lw=2.2,
        ls=LS_FNL0,
        label="L=3 Gpc/h, fnl=0",
    )
    ax.fill_between(
        s_plot,
        y_3gpc_fnl100 - e_3gpc_fnl100,
        y_3gpc_fnl100 + e_3gpc_fnl100,
        color=COLOR_3GPC,
        alpha=0.10,
        linewidth=0.0,
    )
    ax.plot(
        s_plot,
        y_3gpc_fnl100,
        color=COLOR_3GPC,
        lw=2.2,
        ls=LS_FNL100,
        label="L=3 Gpc/h, fnl=100",
    )

    ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$r^2 \xi_0(r)$")
    ax.set_title("FastPM measured mean 2PCF: 1Gpc vs 3Gpc, fnl0 vs fnl100")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.text(
        0.015,
        0.03,
        r"Shaded band = $\sqrt{\mathrm{diag}(\mathrm{Cov})}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="0.8"),
    )

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] saved figure: {OUT_FIG}")
    print(f"[INFO] 1Gpc fnl0 N={data_1gpc_fnl0.nmock}")
    print(f"[INFO] 1Gpc fnl100 N={data_1gpc_fnl100.nmock}")
    print(f"[INFO] 3Gpc fnl0 N={data_3gpc_fnl0.nmock}")
    print(f"[INFO] 3Gpc fnl100 N={data_3gpc_fnl100.nmock}")


if __name__ == "__main__":
    main()
