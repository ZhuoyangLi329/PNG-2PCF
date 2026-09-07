#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meeting-fig: measured 2PCF means for different masscuts
=======================================================

目标
----
只比较不同 halo masscut 的测量 2PCF 均值，不画 error-bar，不画任何建模曲线。

口径
----
- 默认使用 `r^2 * xi0(r)`，与前面的会图保持一致。
- 默认去掉排序最后 4 个高 mass_min 子样本，因为 halo 数太少。

输出
----
- `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/meeting-fig/masscut_measured_2pcf_means.pdf`
"""

from __future__ import annotations

import os
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

from masscut_model_validate import read_pcf_mean_std  # noqa: E402


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm")
MASSCUT_ROOT = PROJECT_ROOT / "masscut_scan"
OUT_PDF = THIS_DIR / "masscut_measured_2pcf_means.pdf"
REALIZATION_MIN = 1
REALIZATION_MAX = 50
DROP_LAST_N_TAGS = 4
USE_R2 = True

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


def tag_to_mass_value(tag: str) -> float:
    """
    把 `mmin_1p4e13` / `mmin_1e14` 转回 float。

    参数
    ----
    tag : str
        masscut 标签。

    返回
    ----
    float
        mass_min 数值。
    """
    return float(tag.replace("mmin_", "").replace("p", "."))


def discover_masscut_tags(root: Path) -> list[str]:
    """
    自动发现已有完整 masscut 数据的标签，并按 mass_min 排序。

    参数
    ----
    root : Path
        `masscut_scan` 根目录。

    返回
    ----
    list[str]
        标签列表。
    """
    tags: list[str] = []
    for tag_dir in sorted(root.glob("mmin_*")):
        pk_files = list((tag_dir / "pk").glob("pk_rsd_N*.dat"))
        pcf_files = list((tag_dir / "pcf").glob("pcf_rsd_N*.dat"))
        if pk_files and pcf_files:
            tags.append(tag_dir.name)
    tags.sort(key=tag_to_mass_value)
    return tags


def main() -> None:
    """
    主函数。

    执行逻辑
    --------
    1. 自动发现已有完整 masscut 数据的标签；
    2. 去掉排序最后 4 个 halo 太少的样本；
    3. 读取每个样本的测量 2PCF 均值；
    4. 叠加成一张 PDF。
    """
    tags = discover_masscut_tags(MASSCUT_ROOT)
    if DROP_LAST_N_TAGS > 0 and len(tags) > DROP_LAST_N_TAGS:
        tags = tags[:-DROP_LAST_N_TAGS]
    print("usable tags:", tags)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    cmap = plt.get_cmap("viridis")

    for idx, tag in enumerate(tags):
        pcf_dir = MASSCUT_ROOT / tag / "pcf"
        r, xi_mean, _, used_files = read_pcf_mean_std(
            pcf_dir=str(pcf_dir),
            pcf_glob="pcf_rsd_N*.dat",
            realization_min=REALIZATION_MIN,
            realization_max=REALIZATION_MAX,
        )
        y = r**2 * xi_mean if USE_R2 else xi_mean
        color = cmap(idx / max(len(tags) - 1, 1))
        ax.plot(r, y, lw=2.0, color=color, label=f"{tag} (N={len(used_files)})")

    ax.set_title("Measured 2PCF Means for Different Masscuts")
    ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$r^2 \xi_0(r)$" if USE_R2 else r"$\xi_0(r)$")
    ax.grid(alpha=0.28, ls="--")
    ax.legend(frameon=False, ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"[OK] saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
