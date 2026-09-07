#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastPM 原始盒子 2PCF 均值：1Gpc vs 3Gpc
========================================

代码大纲
--------
1. 读取 FastPM 原始 periodic box 的四组 2PCF 文件：
   - 1Gpc, fnl=0
   - 1Gpc, fnl=100
   - 3Gpc, fnl=0
   - 3Gpc, fnl=100
2. 分别计算每组 realization 的 xi0(r) 均值与均值误差；
3. 对每个 fNL 画两行：
   - 上行：1Gpc 和 3Gpc 的 r^2 xi0(r) 平均值；
   - 下行：r^2 [xi0_1Gpc(r) - xi0_3Gpc(r)]；
4. 保存 PNG、PDF，并写出一个简短 summary 文本。
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent

OUT_PNG = THIS_DIR / "fastpm_original_box_1gpc_vs_3gpc_2pcf_mean_delta.png"
OUT_PDF = THIS_DIR / "fastpm_original_box_1gpc_vs_3gpc_2pcf_mean_delta.pdf"
OUT_SUMMARY = THIS_DIR / "fastpm_original_box_1gpc_vs_3gpc_2pcf_mean_delta_summary.txt"

REAL_RE = re.compile(r"_N(\d+)\.dat$")

DATASETS = {
    "fnl0": {
        "title": r"$f_{\rm NL}=0$",
        "one_gpc": "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut_fnl0/pcf_rsd_N*.dat",
        "three_gpc": "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat",
    },
    "fnl100": {
        "title": r"$f_{\rm NL}=100$",
        "one_gpc": "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat",
        "three_gpc": "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",
    },
}

COLOR_1GPC = "#1f77b4"
COLOR_3GPC = "#d95f02"
COLOR_DELTA = "#222222"


@dataclass(frozen=True)
class XiDataset:
    """保存一组原始盒子 2PCF 样本。"""

    label: str
    pattern: str
    s: np.ndarray
    xi: np.ndarray
    files: tuple[Path, ...]

    @property
    def nmock(self) -> int:
        return int(self.xi.shape[0])

    @property
    def mean(self) -> np.ndarray:
        return np.mean(self.xi, axis=0)

    @property
    def std(self) -> np.ndarray:
        return np.std(self.xi, axis=0, ddof=1)

    @property
    def mean_error(self) -> np.ndarray:
        return self.std / np.sqrt(self.nmock)


def realization_id(path: Path) -> int:
    """从文件名末尾的 _Nxxx.dat 解析 realization 编号。"""
    match = REAL_RE.search(path.name)
    if match is None:
        raise ValueError(f"无法解析 realization 编号: {path}")
    return int(match.group(1))


def read_one_pcf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取单个 2PCF 文件，返回 s 中心和 xi0。"""
    arr = np.loadtxt(path, comments="#")
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"2PCF 文件格式异常: {path}")
    return np.asarray(arr[:, 0], dtype=float), np.asarray(arr[:, 3], dtype=float)


def load_dataset(label: str, pattern: str) -> XiDataset:
    """读取一组 realization，并在必要时插值到第一份文件的 s 网格。"""
    files = tuple(sorted((Path(fp) for fp in glob.glob(pattern)), key=realization_id))
    if not files:
        raise FileNotFoundError(f"没有匹配到 2PCF 文件: {pattern}")

    s_ref, xi_ref = read_one_pcf(files[0])
    xi_rows = [xi_ref]

    for path in files[1:]:
        s, xi = read_one_pcf(path)
        if np.allclose(s, s_ref, rtol=0.0, atol=0.0):
            xi_rows.append(xi)
        else:
            xi_rows.append(np.interp(s_ref, s, xi))

    return XiDataset(
        label=label,
        pattern=pattern,
        s=s_ref,
        xi=np.asarray(xi_rows, dtype=float),
        files=files,
    )


def interp_to(s_target: np.ndarray, s_source: np.ndarray, y_source: np.ndarray) -> np.ndarray:
    """把 3Gpc 曲线插值到 1Gpc 的 s 网格，便于逐点相减。"""
    if np.allclose(s_target, s_source, rtol=0.0, atol=0.0):
        return y_source
    return np.interp(s_target, s_source, y_source)


def plot_column(
    ax_mean: plt.Axes,
    ax_delta: plt.Axes,
    title: str,
    data_1gpc: XiDataset,
    data_3gpc: XiDataset,
) -> dict[str, float | int]:
    """画一个 fNL 列：均值曲线和 1Gpc-3Gpc 差值。"""
    s = data_1gpc.s

    xi1 = data_1gpc.mean
    err1 = data_1gpc.mean_error
    xi3 = interp_to(s, data_3gpc.s, data_3gpc.mean)
    err3 = interp_to(s, data_3gpc.s, data_3gpc.mean_error)

    y1 = s**2 * xi1
    y3 = s**2 * xi3
    e1 = s**2 * err1
    e3 = s**2 * err3
    delta = y1 - y3
    delta_err = np.sqrt(e1**2 + e3**2)

    ax_mean.fill_between(s, y1 - e1, y1 + e1, color=COLOR_1GPC, alpha=0.18, linewidth=0.0)
    ax_mean.plot(s, y1, color=COLOR_1GPC, lw=2.1, label=f"1Gpc, N={data_1gpc.nmock}")
    ax_mean.fill_between(s, y3 - e3, y3 + e3, color=COLOR_3GPC, alpha=0.18, linewidth=0.0)
    ax_mean.plot(s, y3, color=COLOR_3GPC, lw=2.1, label=f"3Gpc, N={data_3gpc.nmock}")
    ax_mean.set_title(title, fontsize=13)
    ax_mean.set_ylabel(r"$r^2\xi_0(r)$")
    ax_mean.grid(alpha=0.24, ls="--")
    ax_mean.legend(frameon=False, fontsize=10, loc="best")

    ax_delta.axhline(0.0, color="0.45", lw=1.0, ls=":")
    ax_delta.fill_between(
        s,
        delta - delta_err,
        delta + delta_err,
        color=COLOR_DELTA,
        alpha=0.16,
        linewidth=0.0,
    )
    ax_delta.plot(s, delta, color=COLOR_DELTA, lw=2.0)
    ax_delta.set_xlabel(r"$r\,[{\rm Mpc}/h]$")
    ax_delta.set_ylabel(r"$r^2(\xi_{\rm 1Gpc}-\xi_{\rm 3Gpc})$")
    ax_delta.grid(alpha=0.24, ls="--")

    return {
        "n_1gpc": data_1gpc.nmock,
        "n_3gpc": data_3gpc.nmock,
        "r_min": float(np.min(s)),
        "r_max": float(np.max(s)),
        "max_abs_delta_r2xi": float(np.max(np.abs(delta))),
        "rms_delta_r2xi": float(np.sqrt(np.mean(delta**2))),
        "mean_abs_delta_over_sem": float(np.mean(np.abs(delta) / np.maximum(delta_err, 1.0e-30))),
    }


def main() -> None:
    """主入口。"""
    matplotlib.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "stix",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "savefig.dpi": 300,
            "figure.dpi": 150,
        }
    )

    loaded = {
        key: (
            load_dataset(f"{key}_1gpc", cfg["one_gpc"]),
            load_dataset(f"{key}_3gpc", cfg["three_gpc"]),
        )
        for key, cfg in DATASETS.items()
    }

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12.2, 7.4),
        sharex="col",
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08, "wspace": 0.25},
    )

    summaries: dict[str, dict[str, float | int]] = {}
    for col, key in enumerate(("fnl0", "fnl100")):
        data_1gpc, data_3gpc = loaded[key]
        summaries[key] = plot_column(
            ax_mean=axes[0, col],
            ax_delta=axes[1, col],
            title=DATASETS[key]["title"],
            data_1gpc=data_1gpc,
            data_3gpc=data_3gpc,
        )

    fig.suptitle("FastPM original periodic boxes: measured 2PCF mean, 1Gpc vs 3Gpc", y=0.965)
    fig.text(0.5, 0.012, "Shaded bands show standard error of the mean.", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.875, hspace=0.08, wspace=0.25)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    with OUT_SUMMARY.open("w", encoding="utf-8") as fout:
        fout.write("FastPM original periodic box 2PCF mean comparison: 1Gpc vs 3Gpc\n")
        fout.write("No cutsky, no subbox. Inputs are original pcf_masscut box measurements.\n\n")
        for key in ("fnl0", "fnl100"):
            cfg = DATASETS[key]
            stat = summaries[key]
            fout.write(f"[{key}]\n")
            fout.write(f"1Gpc glob: {cfg['one_gpc']}\n")
            fout.write(f"3Gpc glob: {cfg['three_gpc']}\n")
            fout.write(f"N_1Gpc = {stat['n_1gpc']}\n")
            fout.write(f"N_3Gpc = {stat['n_3gpc']}\n")
            fout.write(f"r_range = {stat['r_min']:.6g} .. {stat['r_max']:.6g} Mpc/h\n")
            fout.write(f"max_abs_delta_r2xi = {stat['max_abs_delta_r2xi']:.8e}\n")
            fout.write(f"rms_delta_r2xi = {stat['rms_delta_r2xi']:.8e}\n")
            fout.write(f"mean_abs_delta_over_sem = {stat['mean_abs_delta_over_sem']:.8e}\n\n")

    print(f"[OK] saved PNG: {OUT_PNG}")
    print(f"[OK] saved PDF: {OUT_PDF}")
    print(f"[OK] saved summary: {OUT_SUMMARY}")
    for key, stat in summaries.items():
        print(
            f"[INFO] {key}: N_1Gpc={stat['n_1gpc']}, "
            f"N_3Gpc={stat['n_3gpc']}, "
            f"max_abs_delta_r2xi={stat['max_abs_delta_r2xi']:.6e}, "
            f"rms_delta_r2xi={stat['rms_delta_r2xi']:.6e}"
        )


if __name__ == "__main__":
    main()
