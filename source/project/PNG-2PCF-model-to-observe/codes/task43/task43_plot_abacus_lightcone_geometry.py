#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制 Task43 AbacusSummit halo-lightcone 的二维几何示意图。

执行大纲
--------
1. 在 x-y 平面画出边长为 2 Gpc/h 的 AbacusSummit mother box。
2. 在 (-0.99, -0.99) Gpc/h 标出 lightcone observer。
3. 画出穿过 observer 的二维径向截面：survey window 是由命令行红移范围
   和 AbacusSummit c000 comoving distance 确定的四分之一圆环。
4. 添加汇报所需的尺度与边界标注，并只输出 PDF。

注意：这里使用的是穿过 observer 的二维截面，而不是把完整三维点云沿 z
方向积分后的投影。这样能够忠实保留 lightcone 球壳的内外径向边界。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Rectangle, Wedge


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
DEFAULT_OUTPUT = PROJECT_ROOT / "plots" / "task43" / "task43_abacus_box_lightcone_geometry.pdf"

# 图中统一使用 h^-1 Gpc，避免坐标轴出现过长的 Mpc/h 数字。
BOX_MIN = -1.0
BOX_MAX = 1.0
OBSERVER = np.array([-0.99, -0.99], dtype="f8")
DEFAULT_ZMIN = 0.6
DEFAULT_ZMAX = 0.8
BOX_COLOR = "#4b5563"
WINDOW_COLOR = "#2aa7df"
WINDOW_DARK = "#087eae"
INNER_COLOR = "#edae3f"
OBSERVER_COLOR = "#cc365c"
TEXT_SCALE = 1.30


def configure_style() -> None:
    """设置适合横向汇报幻灯片的字体、线宽与 PDF 参数。"""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18.2 * TEXT_SCALE,
            "axes.labelsize": 25.2 * TEXT_SCALE,
            "legend.fontsize": 12.2 * TEXT_SCALE,
            "xtick.labelsize": 22.4 * TEXT_SCALE,
            "ytick.labelsize": 22.4 * TEXT_SCALE,
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def point_on_ray(radius: float, angle_deg: float) -> np.ndarray:
    """返回从 observer 出发、指定极角和半径处的二维点。

    参数
    ----
    radius
        observer-centric 半径，单位为 h^-1 Gpc。
    angle_deg
        从正 x 方向逆时针测量的角度，单位为 degree。

    返回
    ----
    numpy.ndarray
        全局 box 坐标中的 ``[x, y]``。
    """
    angle = np.deg2rad(float(angle_deg))
    return OBSERVER + float(radius) * np.array([np.cos(angle), np.sin(angle)])


def draw_box(ax: plt.Axes) -> None:
    """绘制 2 Gpc/h mother box 与其图内标签。"""
    box = Rectangle(
        (BOX_MIN, BOX_MIN),
        BOX_MAX - BOX_MIN,
        BOX_MAX - BOX_MIN,
        facecolor="white",
        edgecolor=BOX_COLOR,
        linewidth=2.0,
        zorder=0,
    )
    ax.add_patch(box)

    ax.text(
        0.96,
        0.95,
        "Abacus original\n" + r"$2\,h^{-1}{\rm Gpc}$ box",
        ha="right",
        va="top",
        color=BOX_COLOR,
        fontsize=23.8 * TEXT_SCALE,
        fontweight="bold",
    )


def draw_survey_window(
    ax: plt.Axes,
    *,
    zmin: float,
    zmax: float,
    chi_min: float,
    chi_max: float,
) -> None:
    """绘制四分之一圆环形式的 lightcone 二维径向截面。"""
    shell = Wedge(
        center=OBSERVER,
        r=chi_max,
        theta1=0.0,
        theta2=90.0,
        width=chi_max - chi_min,
        facecolor=WINDOW_COLOR,
        edgecolor="none",
        alpha=0.28,
        zorder=2,
    )
    ax.add_patch(shell)

    # 利用内壳层中的留白直接标明当前几何对应的 mock，避免与右上角
    # mother-box 标签混淆。
    ax.text(
        -0.20,
        0.25,
        "Abacus\nlightcone mock\n" + rf"${zmin:.1f}<z_{{\rm obs}}<{zmax:.1f}$",
        ha="center",
        va="center",
        color=WINDOW_DARK,
        fontsize=20.5 * TEXT_SCALE,
        fontweight="bold",
        linespacing=1.00,
        zorder=8,
    )

    # 单独画内外圆弧，避免 Wedge 的统一边线弱化两个物理边界的区别。
    outer_arc = Arc(
        OBSERVER,
        2.0 * chi_max,
        2.0 * chi_max,
        theta1=0.0,
        theta2=90.0,
        color=WINDOW_DARK,
        linewidth=3.0,
        zorder=5,
    )
    inner_arc = Arc(
        OBSERVER,
        2.0 * chi_min,
        2.0 * chi_min,
        theta1=0.0,
        theta2=90.0,
        color=INNER_COLOR,
        linewidth=2.4,
        linestyle="--",
        zorder=5,
    )
    ax.add_patch(outer_arc)
    ax.add_patch(inner_arc)

    # 两条径向边界显示 angular window 的 90-degree opening。
    for direction in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
        inner = OBSERVER + chi_min * direction
        outer = OBSERVER + chi_max * direction
        ax.plot(
            [OBSERVER[0], inner[0]],
            [OBSERVER[1], inner[1]],
            color="#9aa5b3",
            linestyle=":",
            linewidth=1.7,
            zorder=3,
        )
        ax.plot(
            [inner[0], outer[0]],
            [inner[1], outer[1]],
            color=WINDOW_DARK,
            linewidth=3.0,
            zorder=5,
        )

    # 用一条代表性视线标记两个 comoving-distance 边界。
    ray_angle = 22.0
    inner_point = point_on_ray(chi_min, ray_angle)
    outer_point = point_on_ray(chi_max, ray_angle)
    ax.plot(
        [OBSERVER[0], outer_point[0]],
        [OBSERVER[1], outer_point[1]],
        color="#516170",
        linewidth=1.15,
        alpha=0.75,
        zorder=4,
    )
    ax.scatter(
        [inner_point[0], outer_point[0]],
        [inner_point[1], outer_point[1]],
        s=[34, 38],
        color=[INNER_COLOR, WINDOW_DARK],
        edgecolor="white",
        linewidth=0.7,
        zorder=7,
    )
    # observer 位于 mother box 的 (-0.99,-0.99) 角附近。
    ax.scatter(
        [OBSERVER[0]],
        [OBSERVER[1]],
        marker="*",
        s=260,
        color=OBSERVER_COLOR,
        edgecolor="white",
        linewidth=0.9,
        zorder=10,
    )
    ax.annotate(
        "Observer",
        xy=OBSERVER,
        xytext=(-0.77, -0.80),
        color=OBSERVER_COLOR,
        fontsize=23.8 * TEXT_SCALE,
        fontweight="bold",
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "-|>",
            "color": OBSERVER_COLOR,
            "lw": 2.5,
            "mutation_scale": 20,
            "shrinkA": 2,
            "shrinkB": 4,
        },
        zorder=10,
    )


def draw_coordinate_directions(ax: plt.Axes) -> None:
    """在 mother box 左下外侧标出二维截面的正 x、正 y 方向。"""
    origin = np.array([-1.055, -1.055], dtype="f8")
    arrow_style = {
        "arrowstyle": "-|>",
        "color": "#273444",
        "lw": 2.1,
        "mutation_scale": 16,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    x_end = origin + np.array([0.58, 0.0])
    y_end = origin + np.array([0.0, 0.58])
    ax.annotate("", xy=x_end, xytext=origin, arrowprops=arrow_style, zorder=9)
    ax.annotate("", xy=y_end, xytext=origin, arrowprops=arrow_style, zorder=9)
    ax.text(
        x_end[0] + 0.045,
        x_end[1] + 0.012,
        r"$x$",
        ha="left",
        va="bottom",
        fontsize=21.7 * TEXT_SCALE,
        color="#273444",
        fontweight="bold",
        zorder=10,
    )
    ax.text(
        y_end[0] + 0.012,
        y_end[1] + 0.045,
        r"$y$",
        ha="left",
        va="bottom",
        fontsize=21.7 * TEXT_SCALE,
        color="#273444",
        fontweight="bold",
        zorder=10,
    )


def build_figure(
    output: Path,
    *,
    zmin: float = DEFAULT_ZMIN,
    zmax: float = DEFAULT_ZMAX,
) -> None:
    """组装二维示意图并保存为 PDF。

    参数
    ----
    output
        输出 PDF 路径；保存前会自动创建父目录。
    """
    if not 0.0 <= float(zmin) < float(zmax):
        raise ValueError(f"invalid redshift interval: {zmin}, {zmax}")
    from cosmoprimo.fiducial import AbacusSummit

    cosmology = AbacusSummit(0)
    chi_min, chi_max = (
        float(cosmology.comoving_radial_distance(redshift)) / 1000.0
        for redshift in (zmin, zmax)
    )
    configure_style()
    fig = plt.figure(figsize=(7.8, 7.8))
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    draw_box(ax)
    draw_survey_window(
        ax,
        zmin=float(zmin),
        zmax=float(zmax),
        chi_min=chi_min,
        chi_max=chi_max,
    )
    draw_coordinate_directions(ax)

    ax.set_xlim(-1.12, 1.08)
    ax.set_ylim(-1.12, 1.08)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-1.0, 0.0, 1.0])
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(color="#94a3b8", linewidth=0.7, alpha=0.18, zorder=-5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", facecolor="white", transparent=False)
    plt.close(fig)


def main() -> None:
    """解析输出路径并生成二维 Task43 lightcone 几何 PDF。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 PDF 路径。")
    parser.add_argument("--zmin", type=float, default=DEFAULT_ZMIN)
    parser.add_argument("--zmax", type=float, default=DEFAULT_ZMAX)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".pdf":
        raise ValueError("Task43 plots must be saved as PDF files")
    build_figure(args.output, zmin=args.zmin, zmax=args.zmax)
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
