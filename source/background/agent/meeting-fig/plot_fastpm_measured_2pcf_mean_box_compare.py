#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meeting-fig: FastPM fnl100 原始盒子 2PCF 均值与当前 best-fit 对比
===================================================================

代码大纲
--------
1. 定义 1Gpc / 3Gpc fnl100 原始周期盒子的 2PCF 输入路径；
2. 读取 realization，计算 xi0(s) 的均值和样本标准差；
3. 读取当前口径 best-fit 2PCF：
   - 1Gpc：本目录缓存的 `BinAvgFit + FullDiscrete/CachedRebin` npz；
   - 3Gpc：task182 RSD 结果中的 `fnl100_xi_cached_binavg`；
4. 在同一个 panel 中画 `s^2 xi0(s)`：均值用 errorbar，best-fit 用曲线；
5. 保存指定 PDF。

逻辑关系
--------
- `parse_realization_id` 负责从文件名提取 realization 编号；
- `read_pcf_mean_std_from_glob` 负责读取一组 2PCF 文件并返回统计量；
- `load_bestfit_models` 负责读取 1Gpc / 3Gpc 当前方法 best-fit 2PCF；
- `plot_fnl100_comparison` 负责画并保存最终图；
- `main` 只组织 fnl100 这一张图。
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 一、路径与全局配置
# ============================================================
THIS_DIR = Path(__file__).resolve().parent

OUT_FNL100 = THIS_DIR / "fastpm_measured_2pcf_mean_fnl100_1gpc_vs_3gpc.pdf"
MODEL_1GPC = THIS_DIR / "fastpm_1gpc_fnl100_current_bestfit_2pcf_model.npz"
MODEL_3GPC = Path(
    "/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task18_outputs/"
    "task182_m11_redshiftspace_repro/task182_m11_redshiftspace_results.npz"
)

USE_R2 = True

DATASETS = [
    {
        "name": "1Gpc mean",
        "model_name": "1Gpc best-fit",
        "pattern": "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat",
        "color": "#1f77b4",
        "marker": "o",
    },
    {
        "name": "3Gpc mean",
        "model_name": "3Gpc best-fit",
        "pattern": "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",
        "color": "#d62728",
        "marker": "s",
    },
]

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "savefig.dpi": 300,
    "figure.dpi": 150,
})


def parse_realization_id(path: Path) -> int:
    """
    从文件名中提取 realization 编号。

    参数
    ----
    path : Path
        例如 `pcf_rsd_N10.dat` 或 `pcf_rsd_3gpc_fnl100_N77.dat`。

    返回
    ----
    int
        realization 的整数编号。
    """
    match = re.search(r"_N(\d+)\.dat$", path.name)
    if match is None:
        raise ValueError(f"无法从文件名解析 realization 编号: {path}")
    return int(match.group(1))


def read_pcf_mean_std_from_glob(pattern: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Path]]:
    """
    读取指定 glob 的 2PCF 文件，返回 r 网格、均值、标准差与文件列表。

    输入文件格式约定
    ----------------
    每个文件应至少有四列：
    1. s_cen
    2. s_min
    3. s_max
    4. xi_0

    参数
    ----
    pattern : str
        例如 `/path/to/pcf_rsd_N*.dat`。

    返回
    ----
    tuple
        `r_grid, xi_mean, xi_std, files`
    """
    files = sorted((Path(fp) for fp in glob.glob(pattern)), key=parse_realization_id)
    if not files:
        raise FileNotFoundError(f"未找到匹配文件: {pattern}")

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


def load_bestfit_models() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    读取 1Gpc / 3Gpc 的当前方法 best-fit 2PCF。

    返回
    ----
    dict
        key 为 `1Gpc` / `3Gpc`，value 为 `(s, xi_model)`。
    """
    if not MODEL_1GPC.exists():
        raise FileNotFoundError(
            f"未找到 1Gpc 当前 best-fit 2PCF 缓存: {MODEL_1GPC}\n"
            "请先运行 build_fastpm_1gpc_fnl100_current_bestfit_2pcf.py"
        )
    if not MODEL_3GPC.exists():
        raise FileNotFoundError(f"未找到 task182 3Gpc best-fit 结果: {MODEL_3GPC}")

    data_1gpc = np.load(MODEL_1GPC)
    data_3gpc = np.load(MODEL_3GPC)
    return {
        "1Gpc": (
            data_1gpc["s"].astype(np.float64),
            data_1gpc["xi_cached_binavg"].astype(np.float64),
        ),
        "3Gpc": (
            data_3gpc["fnl100_s"].astype(np.float64),
            data_3gpc["fnl100_xi_cached_binavg"].astype(np.float64),
        ),
    }


def plot_fnl100_comparison(datasets: list[dict[str, str]], out_path: Path) -> None:
    """
    画 fnl100 原始周期盒子的 1Gpc vs 3Gpc 均值和 best-fit 对比图。

    参数
    ----
    datasets : list[dict]
        两个数据集的配置列表，每项至少包含 `name`, `model_name`, `pattern`, `color`。
    out_path : Path
        输出 PDF 路径。
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.7))
    model_by_box = load_bestfit_models()

    ylabel = r"$r^2 \xi_0(r)$" if USE_R2 else r"$\xi_0(r)$"

    for dataset in datasets:
        r_grid, xi_mean, xi_std, files = read_pcf_mean_std_from_glob(dataset["pattern"])
        box_key = "1Gpc" if "1Gpc" in dataset["name"] else "3Gpc"
        model_s, model_xi = model_by_box[box_key]

        if USE_R2:
            y_mean = r_grid ** 2 * xi_mean
            y_err = r_grid ** 2 * xi_std
            y_model = model_s ** 2 * model_xi
        else:
            y_mean = xi_mean
            y_err = xi_std
            y_model = model_xi

        ax.errorbar(
            r_grid,
            y_mean,
            yerr=y_err,
            fmt=dataset["marker"],
            ms=4.4,
            mew=0.0,
            lw=0.0,
            elinewidth=1.15,
            capsize=2.5,
            capthick=1.0,
            color=dataset["color"],
            label=f"{dataset['name']}",
            alpha=0.92,
            zorder=5,
        )
        ax.plot(
            model_s,
            y_model,
            color=dataset["color"],
            lw=2.3,
            ls="-",
            label=f"{dataset['model_name']}",
            zorder=7,
        )

        print(f"[info] {dataset['name']}: {len(files)} PCF files")

    ax.set_title(r"FastPM fnl=100 original boxes: measured mean vs best-fit", fontsize=15)
    ax.set_xlabel(r"$s\,[h^{-1}{\rm Mpc}]$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(alpha=0.28, ls="--")
    ax.legend(frameon=False, fontsize=13, ncol=2, columnspacing=1.4, handlelength=2.0)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] saved pdf: {out_path}")


def main() -> None:
    """
    主函数。

    执行流程
    --------
    1. 读取 fnl100 的 1Gpc/3Gpc 原始盒子 2PCF 均值；
    2. 读取当前方法 best-fit 2PCF；
    3. 保存目标 PDF。
    """
    plot_fnl100_comparison(
        datasets=DATASETS,
        out_path=OUT_FNL100,
    )


if __name__ == "__main__":
    main()
