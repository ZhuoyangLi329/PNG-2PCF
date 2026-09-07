#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重画 Task4.1 fNL=50 P(k) vs 2PCF 的紧凑会议版 contour。

代码大纲
========
1. 直接读取已经通过收敛检查的两条 emcee post-burn 等权链，不重跑拟合；
2. 完全复用权威主图的参数、范围、GetDist smoothing、颜色和 contour 顺序；
3. 把两个对角 1D posterior 轴的物理高度都压缩为原来的一半，左下 2D
   contour 的位置和尺寸保持不变；
4. 从 fNL marginal 删除约束文字，把两组彩色 fNL 数字移到 contour 右上角；
5. 只输出独立的 7.13meeting PDF 和 provenance JSON，不覆盖原始主图。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
INPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/task4_outputs/quijote_emcee_free_sn0_fnl50_pk_xi_r50"
)
PK_DATA_DIR = Path("/pscratch/sd/l/lzy/pks_2pcfs")
PK_SUMMARY = INPUT_ROOT / "pk_binavg/summary.json"
XI_SUMMARY = INPUT_ROOT / "xi_r50_350/summary.json"
SOURCE_PDF = (
    PROJECT_ROOT
    / "plots/task4/important_4p1_4p2/contours/"
    "4p1_fnl50_pk_kmax0p10_vs_2pcf_r50_350_contour.pdf"
)
OUTPUT_PDF = PROJECT_ROOT / "plots/7.13meeting/fnl50_pk_vs_2pcf_contour.pdf"
OUTPUT_SUMMARY = INPUT_ROOT / "task45_fnl50_meeting_compact_plot.json"

PLOT_PARAM_NAMES = ("fnl_loc", "b1")
COLORS = {"pk": "#2F2F2F", "twopcf": "#C44E52"}
SMOOTH_1D = 0.35
SMOOTH_2D = 0.40
ONE_D_HEIGHT_SCALE = 0.50
FNL_TEXT_FONTSIZE = 19.8
UPPER_RIGHT_HEADROOM = 0.18


def sha256(path: Path) -> str:
    """流式计算输入/输出哈希，便于确认会议图的数据来源。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    """把 Path 和 numpy 容器递归转换成 JSON 可写对象。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 并要求顶层是字典。"""
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def load_samples(summary: dict[str, Any]) -> tuple[np.ndarray, Path]:
    """读取 fNL/b1 两列 post-burn 等权样本并返回源路径。"""
    path = Path(summary["paths"]["postburn_samples"])
    with np.load(path, allow_pickle=False) as data:
        names = [str(name) for name in np.asarray(data["param_names"])]
        samples = np.asarray(data["samples"], dtype="f8")
    indices = [names.index(name) for name in PLOT_PARAM_NAMES]
    selected = samples[:, indices]
    if selected.ndim != 2 or selected.shape[1] != 2 or not np.all(np.isfinite(selected)):
        raise ValueError(f"Invalid posterior samples: {path}")
    return selected, path


def constraint_text(samples: np.ndarray) -> str:
    """把 16/50/84 分位数格式化为图内非对称误差。"""
    q16, q50, q84 = np.quantile(
        np.asarray(samples, dtype="f8"), [0.1586552539, 0.5, 0.8413447461]
    )
    return rf"{q50:.1f}^{{+{q84 - q50:.1f}}}_{{-{q50 - q16:.1f}}}"


def shared_axis_limits(
    sample_sets: list[np.ndarray],
    column: int,
    *,
    include_zero: bool = False,
) -> tuple[float, float]:
    """逐字复用权威图的共享紧凑坐标范围规则。"""
    values = np.concatenate(
        [np.asarray(samples[:, column], dtype="f8") for samples in sample_sets]
    )
    finite = values[np.isfinite(values)]
    lo, hi = np.quantile(finite, [0.0025, 0.9975])
    if include_zero:
        lo = min(float(lo), 0.0)
        hi = max(float(hi), 0.0)
    width = float(hi - lo)
    if not np.isfinite(width) or width <= 0.0:
        width = 1.0
    return float(lo - 0.08 * width), float(hi + 0.08 * width)


def audit_pk_fit_range() -> dict[str, Any]:
    """从 Quijote 原始输入核验第一个/最后一个拟合 bin 的中心与边界。"""
    files = sorted(PK_DATA_DIR.glob("pk_LCp50_*.txt"))
    if len(files) != 500:
        raise ValueError(f"Expected 500 LCp50 P(k) files, found {len(files)}")
    reference = np.loadtxt(files[0], comments="#")
    valid = reference[:, 4] > 0
    selected = reference[valid & (reference[:, 0] <= 0.10)]
    if selected.shape[0] != 31:
        raise ValueError(f"Expected 31 fitted P(k) bins, found {selected.shape[0]}")
    return {
        "reference_file": files[0],
        "n_input_files": len(files),
        "n_fit_bins": int(selected.shape[0]),
        "first_bin": {
            "center_h_mpc": float(selected[0, 0]),
            "lower_edge_h_mpc": float(selected[0, 1]),
            "upper_edge_h_mpc": float(selected[0, 2]),
            "n_modes": int(selected[0, 4]),
        },
        "last_bin": {
            "center_h_mpc": float(selected[-1, 0]),
            "lower_edge_h_mpc": float(selected[-1, 1]),
            "upper_edge_h_mpc": float(selected[-1, 2]),
            "n_modes": int(selected[-1, 4]),
        },
        "box_fundamental_2pi_over_1000_h_mpc": float(2.0 * np.pi / 1000.0),
        "interpretation": (
            "0.008 is the rounded first-bin center (0.00778), not the strict lower edge; "
            "the first-bin lower edge is 0.00628, matching the L=1000 box fundamental mode."
        ),
    }


def halve_axis_height(axis: plt.Axes, *, anchor: str) -> dict[str, list[float]]:
    """把一个 1D posterior 轴的物理高度减半，并记录前后位置。

    左上 fNL marginal 以底边为锚点，仍与 contour 顶边相接；右下 b1
    marginal 以底边为锚点，保留 x 标签位置，并在上方释放 legend 留白。
    """
    original = axis.get_position()
    new_height = original.height * ONE_D_HEIGHT_SCALE
    if anchor == "bottom":
        new_y0 = original.y0
    elif anchor == "top":
        new_y0 = original.y1 - new_height
    else:
        raise ValueError(f"Unknown anchor={anchor!r}")
    axis.set_position([original.x0, new_y0, original.width, new_height])
    updated = axis.get_position()
    return {
        "before": [original.x0, original.y0, original.width, original.height],
        "after": [updated.x0, updated.y0, updated.width, updated.height],
    }


def main() -> None:
    """绘制紧凑会议图并写 provenance。"""
    from getdist import MCSamples, plots

    pk_summary = read_json(PK_SUMMARY)
    xi_summary = read_json(XI_SUMMARY)
    pk_range_audit = audit_pk_fit_range()
    for name, summary in (("pk", pk_summary), ("2pcf", xi_summary)):
        if summary.get("status") != "done":
            raise ValueError(f"{name} source summary is not done")
    pk_samples, pk_samples_path = load_samples(pk_summary)
    xi_samples, xi_samples_path = load_samples(xi_summary)

    names = ["fnl_loc", "b1"]
    labels = [r"f_{\rm NL}", r"b_1"]
    xlim_base = shared_axis_limits([pk_samples, xi_samples], 0, include_zero=True)
    ylim_base = shared_axis_limits([pk_samples, xi_samples], 1)
    # 只扩展上限：把 contour 视觉上推向左下并缩小占比，专门在右上角
    # 留出 fNL 数字文字空间；posterior 和置信水平本身完全不变。
    xlim = (
        float(xlim_base[0]),
        float(xlim_base[1] + UPPER_RIGHT_HEADROOM * (xlim_base[1] - xlim_base[0])),
    )
    ylim = (
        float(ylim_base[0]),
        float(ylim_base[1] + UPPER_RIGHT_HEADROOM * (ylim_base[1] - ylim_base[0])),
    )
    ranges = {"fnl_loc": xlim, "b1": ylim}
    settings = {
        "ignore_rows": 0,
        "fine_bins": 2048,
        "fine_bins_2D": 1024,
        "smooth_scale_1D": SMOOTH_1D,
        "smooth_scale_2D": SMOOTH_2D,
        "boundary_correction_order": 1,
        "mult_bias_correction_order": 1,
    }
    pk_label = r"$P_0(k)$" "\n" r"$0.008\leq k\leq0.10\,h\,{\rm Mpc}^{-1}$"
    xi_label = r"$\xi_0(s)$" "\n" r"$50<s<350\,h^{-1}{\rm Mpc}$"
    pk_mc = MCSamples(
        samples=pk_samples,
        names=names,
        labels=labels,
        label=pk_label,
        ranges=ranges,
        settings=settings,
    )
    xi_mc = MCSamples(
        samples=xi_samples,
        names=names,
        labels=labels,
        label=xi_label,
        ranges=ranges,
        settings=settings,
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 12,
            "axes.linewidth": 1.0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )
    plotter = plots.get_subplot_plotter(width_inch=9.4)
    plotter.settings.axes_fontsize = 13.5
    plotter.settings.lab_fontsize = 17.0
    plotter.settings.legend_fontsize = 17.0
    plotter.settings.legend_frame = False
    plotter.settings.figure_legend_frame = False
    plotter.settings.linewidth = 1.8
    plotter.settings.linewidth_contour = 1.8
    plotter.settings.alpha_filled_add = 0.30
    plotter.settings.num_plot_contours = 2
    plotter.triangle_plot(
        [pk_mc, xi_mc],
        names,
        filled=True,
        contour_colors=[COLORS["pk"], COLORS["twopcf"]],
        contour_lws=[1.8, 1.8],
        legend_labels=[pk_label, xi_label],
        legend_loc="lower left",
        param_limits=ranges,
    )
    axes = plotter.subplots
    fnl_axis = axes[0, 0]
    contour_axis = axes[1, 0]
    b1_axis = axes[1, 1]
    fnl_axis.set_xlim(*xlim)
    contour_axis.set_xlim(*xlim)
    contour_axis.set_ylim(*ylim)
    b1_axis.set_xlim(*ylim)

    # 在任何轴移动前记录 contour，随后再次核验它完全不变。
    contour_before = contour_axis.get_position()
    axis_position_audit = {
        "fnl_1d": halve_axis_height(fnl_axis, anchor="bottom"),
        "b1_1d": halve_axis_height(b1_axis, anchor="bottom"),
    }
    contour_after = contour_axis.get_position()
    contour_delta = np.asarray(
        [
            contour_after.x0 - contour_before.x0,
            contour_after.y0 - contour_before.y0,
            contour_after.width - contour_before.width,
            contour_after.height - contour_before.height,
        ],
        dtype="f8",
    )
    if float(np.max(np.abs(contour_delta))) > 1.0e-14:
        raise RuntimeError("2D contour axis changed while compacting 1D axes")

    # 用户最终要求完全删除 legend；MCSamples label 仅保留内部 provenance。
    removed_legend_count = len(plotter.fig.legends)
    for legend in list(plotter.fig.legends):
        legend.remove()

    # 用户指定：两组 fNL 数字放进 contour 右上角，采用与曲线一致的颜色。
    contour_axis.text(
        0.965,
        0.955,
        rf"$P_0(k):\ f_{{\rm NL}}={constraint_text(pk_samples[:, 0])}$",
        transform=contour_axis.transAxes,
        ha="right",
        va="top",
        fontsize=FNL_TEXT_FONTSIZE,
        color=COLORS["pk"],
        zorder=20,
    )
    contour_axis.text(
        0.965,
        0.865,
        rf"$\xi_0(s):\ f_{{\rm NL}}={constraint_text(xi_samples[:, 0])}$",
        transform=contour_axis.transAxes,
        ha="right",
        va="top",
        fontsize=FNL_TEXT_FONTSIZE,
        color=COLORS["twopcf"],
        zorder=20,
    )

    plotter.fig.patch.set_facecolor("white")
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    plotter.fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
        transparent=False,
    )
    plt.close(plotter.fig)

    manifest = {
        "task": "task45_fnl50_meeting_compact_plot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "done",
        "scientific_scope": {
            "input_fnl": 50.0,
            "plot_parameters": list(PLOT_PARAM_NAMES),
            "marginalized_parameters": ["sigmas", "sn0"],
            "pk_kmax_h_mpc": 0.10,
            "xi_range_mpc_h": [50.0, 350.0],
        },
        "style_changes_only": {
            "one_d_axis_height_scale": ONE_D_HEIGHT_SCALE,
            "both_diagonal_1d_axes_compacted": True,
            "contour_axis_unchanged": bool(np.max(np.abs(contour_delta)) <= 1.0e-14),
            "axis_position_audit": axis_position_audit,
            "fnl_annotations": "contour upper right",
            "fnl_annotation_fontsize": FNL_TEXT_FONTSIZE,
            "upper_right_range_headroom_fraction": UPPER_RIGHT_HEADROOM,
            "base_plot_ranges": {"fnl_loc": xlim_base, "b1": ylim_base},
            "expanded_plot_ranges": {"fnl_loc": xlim, "b1": ylim},
            "legend_removed": True,
            "removed_figure_legend_count": removed_legend_count,
        },
        "constraints": {
            "pk": constraint_text(pk_samples[:, 0]),
            "2pcf": constraint_text(xi_samples[:, 0]),
        },
        "inputs": {
            "source_pdf": SOURCE_PDF,
            "source_pdf_sha256": sha256(SOURCE_PDF),
            "pk_summary": PK_SUMMARY,
            "xi_summary": XI_SUMMARY,
            "pk_samples": pk_samples_path,
            "pk_samples_sha256": sha256(pk_samples_path),
            "xi_samples": xi_samples_path,
            "xi_samples_sha256": sha256(xi_samples_path),
            "pk_fit_range_audit": pk_range_audit,
        },
        "output": {
            "pdf": OUTPUT_PDF,
            "pdf_sha256": sha256(OUTPUT_PDF),
        },
    }
    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as stream:
        json.dump(jsonable(manifest), stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print(f"saved: {OUTPUT_PDF}")
    print(f"saved: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
