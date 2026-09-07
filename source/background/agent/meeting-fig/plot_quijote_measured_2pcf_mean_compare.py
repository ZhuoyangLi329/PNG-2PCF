#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meeting-fig: Quijote measured 2PCF mean comparison for fnl=0 and fnl=100
============================================================================

代码大纲
--------
1. 定义输入目录、文件模式和输出路径；
2. 读取 Quijote 的 2PCF realization 文件；
3. 计算 fnl=0 与 fnl=100 的 xi0(r) 均值和标准差；
4. 画出一张 `r^2 * xi0(r)` 的对比图，并用阴影表示 1σ 波动；
5. 同时输出 PDF 和 PNG，方便会议展示或后续插图复用。

逻辑关系
--------
- `read_pcf_mean_std_from_pattern` 负责读取一组 2PCF 文件并返回统计量；
- `main` 负责组装两组数据、统一作图样式并保存结果。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# 一、路径与绘图全局配置
# =========================
DATA_DIR = Path("/pscratch/sd/l/lzy/pks_2pcfs")
THIS_DIR = Path(__file__).resolve().parent
OUT_PDF = THIS_DIR / "quijote_measured_2pcf_mean_fnl0_vs_fnl100.pdf"
OUT_PNG = THIS_DIR / "quijote_measured_2pcf_mean_fnl0_vs_fnl100.png"

# 使用 `r^2 * xi0(r)` 作为主展示量，和项目里其它会图口径保持一致。
USE_R2 = True

DATASETS = [
    {
        "name": "Quijote fnl=0",
        "pattern": "pcf_fid_*.dat",
        "color": "#1f77b4",
    },
    {
        "name": "Quijote fnl=100",
        "pattern": "pcf_LCp100_*.dat",
        "color": "#d62728",
    },
]

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


def parse_realization_id(path: Path) -> int:
    """
    从文件名中提取 realization 编号。

    参数
    ----
    path : Path
        例如 `pcf_fid_123.dat` 或 `pcf_LCp100_456.dat`。

    返回
    ----
    int
        realization 的整数编号。
    """
    match = re.search(r"_(\d+)\.dat$", path.name)
    if match is None:
        raise ValueError(f"无法从文件名解析 realization 编号: {path}")
    return int(match.group(1))


def read_pcf_mean_std_from_pattern(data_dir: Path, pattern: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Path]]:
    """
    读取指定模式的 2PCF 文件，返回 r 网格、均值、标准差与文件列表。

    输入文件格式约定
    ----------------
    每个文件应为四列：
    1. s_cen
    2. s_min
    3. s_max
    4. xi_0

    参数
    ----
    data_dir : Path
        Quijote 2PCF 数据所在目录。
    pattern : str
        文件模式，例如 `pcf_fid_*.dat`。

    返回
    ----
    tuple
        `r_grid, xi_mean, xi_std, files`
    """
    files = sorted(data_dir.glob(pattern), key=parse_realization_id)
    if not files:
        raise FileNotFoundError(f"未找到匹配文件: {data_dir / pattern}")

    xi_all: list[np.ndarray] = []
    r_grid: np.ndarray | None = None

    for file_path in files:
        arr = np.loadtxt(file_path, comments="#")
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError(f"2PCF 文件列数异常: {file_path}")
        if r_grid is None:
            r_grid = arr[:, 0].astype(np.float64)
        elif not np.allclose(arr[:, 0], r_grid, rtol=0.0, atol=1e-12):
            raise ValueError(f"2PCF 的 r 网格不一致: {file_path}")
        xi_all.append(arr[:, 3].astype(np.float64))

    xi_stack = np.asarray(xi_all, dtype=np.float64)
    xi_mean = np.mean(xi_stack, axis=0)
    xi_std = np.std(xi_stack, axis=0, ddof=1)
    assert r_grid is not None
    return r_grid, xi_mean, xi_std, files


def main() -> None:
    """
    主函数。

    执行流程
    --------
    1. 分别读取 fnl=0 和 fnl=100 的 500 个 Quijote 2PCF realization；
    2. 计算均值与标准差；
    3. 在同一张图上画出两条均值曲线；
    4. 用半透明阴影标记各自的 1σ 区间；
    5. 保存为 PDF 和 PNG。
    """
    fig, ax = plt.subplots(figsize=(8.8, 6.0))

    for dataset in DATASETS:
        r_grid, xi_mean, xi_std, files = read_pcf_mean_std_from_pattern(
            DATA_DIR,
            dataset["pattern"],
        )

        if USE_R2:
            y_mean = r_grid ** 2 * xi_mean
            y_low = r_grid ** 2 * (xi_mean - xi_std)
            y_high = r_grid ** 2 * (xi_mean + xi_std)
            ylabel = r"$r^2 \xi_0(r)$"
        else:
            y_mean = xi_mean
            y_low = xi_mean - xi_std
            y_high = xi_mean + xi_std
            ylabel = r"$\xi_0(r)$"

        ax.plot(
            r_grid,
            y_mean,
            lw=2.4,
            color=dataset["color"],
            label=f"{dataset['name']} (N={len(files)})",
        )
        ax.fill_between(
            r_grid,
            y_low,
            y_high,
            color=dataset["color"],
            alpha=0.18,
            linewidth=0.0,
        )

    ax.set_title("Quijote Measured 2PCF Mean: fnl=0 vs fnl=100")
    ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.28, ls="--")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG)
    plt.close(fig)

    print(f"[OK] saved pdf: {OUT_PDF}")
    print(f"[OK] saved png: {OUT_PNG}")


if __name__ == "__main__":
    main()
