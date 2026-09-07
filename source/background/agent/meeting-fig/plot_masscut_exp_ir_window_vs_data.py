#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meeting-fig: masscut measured mean + exp IR window model
=======================================================

目标
----
对不同 halo masscut 的 fastPM 样本，使用 mission7 已有的
`model_validation_normexp` best-fit 参数，重算 exp IR window 模型曲线，
并与对应的测量 2PCF 均值和误差棒画在一张大图的多个子图中。

口径
----
- 只保留前 8 个 masscut（去掉 halo 太少的最后 4 个）。
- 每个子图只画：
  - 测量均值 + error bar（黑点）
  - exp IR window 模型（红线）
- 统一使用 `r^2 * xi0(r)`。

输出
----
- `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/meeting-fig/masscut_exp_ir_window_vs_data.pdf`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
AGENT_DIR = THIS_DIR.parent
MISSION7_SCRIPT_DIR = AGENT_DIR / "mission7_masscut" / "scripts"
if str(MISSION7_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION7_SCRIPT_DIR))

from plot_masscut_model_overlay import (  # noqa: E402
    _load_bestfit_params,
    _compute_xi_window_on_rgrid,
    _read_pcf_mean_std,
)


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm")
MASSCUT_ROOT = PROJECT_ROOT / "masscut_scan"
OUT_PDF = THIS_DIR / "masscut_exp_ir_window_vs_data.pdf"
REALIZATION_MIN = 1
REALIZATION_MAX = 50
KEEP_TAGS = [
    "mmin_1e12",
    "mmin_1e13",
    "mmin_1p4e13",
    "mmin_2e13",
    "mmin_3e13",
    "mmin_5e13",
    "mmin_7e13",
    "mmin_1e14",
]

matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "figure.dpi": 150,
})


def main() -> None:
    """
    主函数。

    执行逻辑
    --------
    1. 对每个保留的 masscut，读取已有 best-fit 参数；
    2. 读取对应 pcf 的测量均值；
    3. 用 exp IR window 方法重算模型；
    4. 画成 2x4 大图并保存为 PDF。
    """
    fig, axes = plt.subplots(2, 4, figsize=(20.0, 8.2), sharex=False, sharey=False)
    axes = axes.ravel()

    for ax, tag in zip(axes, KEEP_TAGS):
        bestfit_path = MASSCUT_ROOT / tag / "model_validation_normexp" / "bestfit_params.json"
        pcf_dir = MASSCUT_ROOT / tag / "pcf"

        bestfit = _load_bestfit_params(bestfit_path)
        r, xi_mean, xi_std = _read_pcf_mean_std(pcf_dir, REALIZATION_MIN, REALIZATION_MAX)
        xi_model = _compute_xi_window_on_rgrid(
            bestfit_params=bestfit,
            r_data=r,
            box_size=1000.0,
            kint_min=1e-4,
            kint_max=20.0,
            fftlog_n=4096,
            fftlog_padding=4.0,
            fftlog_mu=0.5,
            fftlog_bias=0.0,
            edge_taper_frac=0.06,
            unit_z=1.0,
        )

        ax.plot(r, r**2 * xi_model, color="#c62828", lw=2.0, label="exp IR window model")
        ax.errorbar(
            r,
            r**2 * xi_mean,
            yerr=r**2 * xi_std,
            fmt="o",
            color="black",
            ecolor="black",
            ms=3.2,
            lw=0.8,
            capsize=1.8,
            label="measured mean",
        )
        ax.set_title(tag)
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2 \xi_0(r)$")
        ax.grid(alpha=0.28, ls="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Masscut: measured mean + exp IR window model", fontsize=15, y=0.985)
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.955))
    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.08, top=0.92, wspace=0.22, hspace=0.26)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"[OK] saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
