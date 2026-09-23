#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 Task4.3 最终仅保留的两张 matched radial-RIC 主结果。

代码大纲
========
1. 读取同一条 24x50000 P(k)-RIC 长链，以及两条 64x20000 2PCF-RIC
   长链；两条 2PCF 链只切换 covariance（RascalC / jaxpower）。
2. 硬检查 50--350 Mpc/h shell-average、L=2000、fixed 2PCF sn0=0、
   free P(k) sn0、mother-box cutoff 和同一个 factorized radial operator。
3. 对三条链的全部自由参数执行 length/tau>100 与 split<0.05 sigma gate。
4. 用三条 posterior 的并集固定共享 fNL,b1 坐标范围，再按 Task44 PPT
   风格分别画 RascalC 主结果和 both-jaxpower 结果：P(k) 深炭灰且置顶、
   2PCF 红色、无标题/真值线/MAP 星号，彩色 fNL 结果写在左上子图内。
5. 只向 plots/task43 写这两张 PDF；完整链、posterior、共享坐标和输入路径
   写入 outputs 下的机器可读 audit JSON。

这仍是 single ``IC^(rad,rad)`` approximation；不包含两个 density--RIC
cross terms，且 RIC 只进入 mean model，不改 covariance。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import task43_overlay_pk_window_kmin_contour as base  # noqa: E402
import task43_plot_ric_rascalc_pk_vs_2pcf as single  # noqa: E402


JAXPOWER_XI_DIR = (
    PROJECT_ROOT
    / "outputs/task43_outputs/ric_singleterm/fits/"
    "2pcf_jaxpower_ph000_dchi2_nsub200000_long_mcmc20k"
)
JAXPOWER_XI_SUMMARY = JAXPOWER_XI_DIR / "task43_minimal_closure_mcmc_summary.json"
JAXPOWER_XI_SAMPLES = JAXPOWER_XI_DIR / "task43_mcmc_radial_singleterm_samples.npz"
JAXPOWER_COVARIANCE_NAME = (
    "jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_"
    "smoothfftlog_rrdeconv_mesh64_nran100k_ndata50k_pad400_win3600_ds2_"
    "k0001_3000_dk002_p1p0_s50_350_ds10.npz"
)

OUTPUT_RASCALC = (
    PROJECT_ROOT
    / "plots/task43/task43_pk_vs_2pcf_s50_350_2pcf_rascalc_covariance.pdf"
)
OUTPUT_JAXPOWER = (
    PROJECT_ROOT
    / "plots/task43/task43_pk_vs_2pcf_s50_350_both_jaxpower_covariance.pdf"
)
AUDIT = (
    PROJECT_ROOT
    / "outputs/task43_outputs/ric_singleterm/audits/"
    "task43_ric_pk_vs_2pcf_covariance_pair_longchain.json"
)


def validate_jaxpower_xi(summary: dict[str, Any], pk_summary: dict[str, Any]) -> None:
    """检查 jaxpower 2PCF 链与 RascalC 主图除 covariance 外完全同口径。"""
    fit = summary["fit_range"]
    if (float(fit["rmin"]), float(fit["rmax"])) != (50.0, 350.0):
        raise ValueError(f"jaxpower 2PCF fit range 不匹配：{fit}")
    if Path(summary["covariance"]["path"]).name != JAXPOWER_COVARIANCE_NAME:
        raise ValueError("jaxpower 2PCF covariance provenance 不匹配")
    theory = summary["theory"]
    if not (
        float(theory["boxsize"]) == 2000.0
        and float(theory["p_fixed"]) == 1.0
        and theory["sn0_policy"] == "fixed"
        and float(theory["sn0_fixed"]) == 0.0
        and theory["xi_kernel"] == "shell-averaged"
    ):
        raise ValueError("jaxpower 2PCF theory/sn0/shell-average 口径不匹配")
    model = single.radial_model(summary)
    xi_operator = Path(model["radial_singleterm"]["operator"]["path"]).name
    pk_operator = Path(pk_summary["config"]["radial_singleterm_ric"]["path"]).name
    if xi_operator != single.BASE_OPERATOR_NAME or xi_operator != pk_operator:
        raise ValueError("jaxpower 2PCF 与 P(k) 没有使用同一个 baseline RIC operator")
    if model["radial_singleterm"]["extra_global_sigma_w2"] is not False:
        raise ValueError("jaxpower 2PCF radial branch 重复减了 global sigmaW2")


PPT_COLORS = {"pk": "#2F2F2F", "twopcf": "#C44E52"}
PPT_LEGEND_FONTSIZE = 17.0
PPT_FNL_FONTSIZE = 14.0
PPT_FNL_HEADROOM = 1.38
PPT_LEGEND_GAP = 0.018


def fnl_panel_label(observable: str, stats: dict[str, float]) -> str:
    """生成左上 fNL 子图内的一行彩色约束文字。"""
    return rf"${observable}:\ f_{{\rm NL}}={base.constraint_text(stats)}$"


def plot_ppt_corner(
    *,
    xi_samples: np.ndarray,
    pk_samples: np.ndarray,
    xi_fnl: dict[str, float],
    pk_fnl: dict[str, float],
    pk_summary: dict[str, Any],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xi_smin: float = 50.0,
    xi_smax: float = 350.0,
) -> tuple[plt.Figure, np.ndarray, tuple[float, float]]:
    """按用户定稿的 Task44 PPT 风格绘制一张 Task43 两参数 corner 图。

    ``xi_smin`` / ``xi_smax`` 是 2PCF likelihood 的径向 edge；显式传入
    legend，避免替换 posterior 后仍错误显示旧的 ``50<s<350`` 标签。
    """
    from getdist import MCSamples, plots

    names = ["fnl_loc", "b1"]
    parameter_labels = [r"f_{\rm NL}", r"b_1"]
    ranges = {"fnl_loc": xlim, "b1": ylim}
    labels = {
        "pk": (
            r"$P_0(k)$" "\n"
            + rf"${float(pk_summary['data']['k_min']):.3f}\leq k\leq "
            + rf"{float(pk_summary['data']['k_max']):.1f}\,h\,{{\rm Mpc}}^{{-1}}$"
        ),
        "twopcf": (
            r"$\xi_0(s)$" "\n"
            + rf"${float(xi_smin):g}<s<{float(xi_smax):g}\,h^{{-1}}{{\rm Mpc}}$"
        ),
    }
    # GetDist 把第一条 root 画在更高显示层；P(k) 放第一位，避免被 2PCF
    # 填充轮廓遮挡，同时 legend 也保持 P(k) 在前。
    order = ("pk", "twopcf")
    sample_map = {"pk": pk_samples, "twopcf": xi_samples}
    chains = [
        MCSamples(
            samples=sample_map[key],
            names=names,
            labels=parameter_labels,
            label=labels[key],
            ranges=ranges,
            settings={
                "ignore_rows": 0,
                "smooth_scale_1D": base.GETDIST_SMOOTH_1D,
                "smooth_scale_2D": base.GETDIST_SMOOTH_2D,
            },
        )
        for key in order
    ]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 12,
            "axes.linewidth": 1.0,
            "legend.frameon": False,
        }
    )
    plotter = plots.get_subplot_plotter(width_inch=8.8)
    plotter.settings.axes_fontsize = 13.5
    plotter.settings.lab_fontsize = 17.0
    plotter.settings.legend_fontsize = PPT_LEGEND_FONTSIZE
    plotter.settings.legend_frame = False
    plotter.settings.figure_legend_frame = False
    plotter.settings.linewidth = 1.8
    plotter.settings.linewidth_contour = 1.8
    plotter.settings.alpha_filled_add = 0.30
    plotter.settings.num_plot_contours = 2
    plotter.triangle_plot(
        chains,
        names,
        filled=True,
        contour_colors=[PPT_COLORS[key] for key in order],
        contour_lws=[1.8, 1.8],
        legend_labels=[labels[key] for key in order],
        legend_loc="lower left",
        param_limits=ranges,
    )

    figure = plotter.fig
    axes = plotter.subplots
    axes[0, 0].set_xlim(*xlim)
    axes[1, 0].set_xlim(*xlim)
    axes[1, 0].set_ylim(*ylim)
    axes[1, 1].set_xlim(*ylim)

    # 不写死 2x2 corner 的第一象限角点，而是从实际 axes 位置计算；这样
    # legend 放大到 17 后仍稳定落在无子图区域左下角且不压相邻边框。
    blank_corner_x = float(axes[0, 0].get_position().x1)
    blank_corner_y = float(axes[1, 1].get_position().y1)
    legend_anchor = (
        blank_corner_x + PPT_LEGEND_GAP,
        blank_corner_y + PPT_LEGEND_GAP,
    )
    if not figure.legends:
        raise RuntimeError("GetDist 没有创建 figure legend")
    figure.legends[-1].set_bbox_to_anchor(legend_anchor, transform=figure.transFigure)

    # 给 fNL 一维 posterior 增加顶部注释空间，并把两条约束按 contour 颜色
    # 写入子图内部；不再使用空白第一象限里的灰色 constraint box。
    fnl_axis = axes[0, 0]
    ymin, ymax = fnl_axis.get_ylim()
    fnl_axis.set_ylim(ymin, ymin + (ymax - ymin) * PPT_FNL_HEADROOM)
    fnl_axis.text(
        0.035,
        0.955,
        fnl_panel_label(r"P_0(k)", pk_fnl),
        transform=fnl_axis.transAxes,
        ha="left",
        va="top",
        fontsize=PPT_FNL_FONTSIZE,
        color=PPT_COLORS["pk"],
    )
    fnl_axis.text(
        0.035,
        0.825,
        fnl_panel_label(r"\xi_0(s)", xi_fnl),
        transform=fnl_axis.transAxes,
        ha="left",
        va="top",
        fontsize=PPT_FNL_FONTSIZE,
        color=PPT_COLORS["twopcf"],
    )
    return figure, axes, legend_anchor


def plot_variant(
    *,
    xi_summary: dict[str, Any],
    xi_samples: np.ndarray,
    pk_summary: dict[str, Any],
    pk_samples: np.ndarray,
    covariance_short: str,
    output: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> dict[str, Any]:
    """画一张 covariance variant 并返回它的 posterior 摘要。"""
    fit_range = xi_summary["fit_range"]
    xi_fnl = single.summarize(xi_samples[:, 0])
    pk_fnl = single.summarize(pk_samples[:, 0])
    figure, _, legend_anchor = plot_ppt_corner(
        xi_samples=xi_samples,
        pk_samples=pk_samples,
        xi_fnl=xi_fnl,
        pk_fnl=pk_fnl,
        pk_summary=pk_summary,
        xlim=xlim,
        ylim=ylim,
        xi_smin=float(fit_range["rmin"]),
        xi_smax=float(fit_range["rmax"]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.pdf")
    try:
        figure.savefig(
            temporary,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.08,
        )
        temporary.replace(output)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()
    print(f"[write] {output}")
    return {
        "fnl_loc": xi_fnl,
        "b1": single.summarize(xi_samples[:, 1]),
        "covariance_short": covariance_short,
        "legend_anchor_figure_fraction": list(legend_anchor),
    }


def main() -> None:
    """通过全部口径/链门槛后生成最终两图和统一审计。"""
    base.apply_publication_style()
    rascalc_summary = single.read_json(single.XI_SUMMARY)
    jaxpower_summary = single.read_json(JAXPOWER_XI_SUMMARY)
    pk_summary = single.read_json(single.PK_SUMMARY)
    single.validate_inputs(rascalc_summary, pk_summary)
    validate_jaxpower_xi(jaxpower_summary, pk_summary)

    rascalc_samples = single.load_xi_samples(rascalc_summary, single.XI_SAMPLES)
    jaxpower_samples = single.load_xi_samples(jaxpower_summary, JAXPOWER_XI_SAMPLES)
    pk_samples_full = base.load_named_samples(single.PK_SAMPLES, ("fnl_loc", "b1", "sn0"))
    pk_samples = pk_samples_full[:, :2]
    diagnostics = {
        "2pcf_rascalc": single.chain_diagnostics(
            rascalc_samples,
            nwalkers=int(single.radial_model(rascalc_summary)["mcmc"]["nwalkers"]),
        ),
        "2pcf_jaxpower": single.chain_diagnostics(
            jaxpower_samples,
            nwalkers=int(single.radial_model(jaxpower_summary)["mcmc"]["nwalkers"]),
        ),
        "pk_jaxpower": single.chain_diagnostics(
            pk_samples_full,
            nwalkers=int(pk_summary["config"]["nwalkers"]),
        ),
    }
    if not all(row["pass"] for row in diagnostics.values()):
        raise RuntimeError(f"至少一条长链未通过 gate：{diagnostics}")

    all_samples = [rascalc_samples, jaxpower_samples, pk_samples]
    xlim = base.shared_axis_limits(all_samples, 0, include_zero=True)
    ylim = base.shared_axis_limits(all_samples, 1)
    posterior_rascalc = plot_variant(
        xi_summary=rascalc_summary,
        xi_samples=rascalc_samples,
        pk_summary=pk_summary,
        pk_samples=pk_samples,
        covariance_short="RascalC",
        output=OUTPUT_RASCALC,
        xlim=xlim,
        ylim=ylim,
    )
    posterior_jaxpower = plot_variant(
        xi_summary=jaxpower_summary,
        xi_samples=jaxpower_samples,
        pk_summary=pk_summary,
        pk_samples=pk_samples,
        covariance_short="jaxpower",
        output=OUTPUT_JAXPOWER,
        xlim=xlim,
        ylim=ylim,
    )
    pk_posterior = {
        "fnl_loc": single.summarize(pk_samples[:, 0]),
        "b1": single.summarize(pk_samples[:, 1]),
        "sn0": single.summarize(pk_samples_full[:, 2]),
    }

    audit = {
        "task": "task43_plot_ric_covariance_pair_longchain",
        "status": "done",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "figure_contract": {
            "active_pdf_count": 2,
            "model": "radial single-term RIC on both probes",
            "only_changed_quantity_between_pdfs": "2PCF covariance: RascalC versus jaxpower",
            "same_pk_chain": True,
            "same_axis_ranges": True,
            "same_radial_operator": True,
            "density_ric_cross_terms": False,
            "extra_global_sigma_w2": False,
            "covariances_changed_by_ric": False,
        },
        "inputs": {
            "2pcf_rascalc_summary": str(single.XI_SUMMARY),
            "2pcf_rascalc_samples": str(single.XI_SAMPLES),
            "2pcf_jaxpower_summary": str(JAXPOWER_XI_SUMMARY),
            "2pcf_jaxpower_samples": str(JAXPOWER_XI_SAMPLES),
            "pk_summary": str(single.PK_SUMMARY),
            "pk_samples": str(single.PK_SAMPLES),
            "radial_operator_name": single.BASE_OPERATOR_NAME,
        },
        "chain_diagnostics": diagnostics,
        "posterior": {
            "2pcf_rascalc": posterior_rascalc,
            "2pcf_jaxpower": posterior_jaxpower,
            "pk_jaxpower": pk_posterior,
        },
        "shared_plot_ranges": {"fnl_loc": xlim, "b1": ylim},
        "outputs": {
            "main_rascalc": str(OUTPUT_RASCALC),
            "jaxpower": str(OUTPUT_JAXPOWER),
        },
        "plot_settings": {
            "backend": "matplotlib + getdist",
            "smooth_scale_1d": base.GETDIST_SMOOTH_1D,
            "smooth_scale_2d": base.GETDIST_SMOOTH_2D,
            "format": "pdf only",
            "style_reference": "Task44 LRG-all PPT contour style taught by user",
            "title_shown": False,
            "truth_marker_shown": False,
            "map_markers_shown": False,
            "contour_and_legend_order": ["pk", "xi0_2pcf"],
            "pk_color": PPT_COLORS["pk"],
            "twopcf_color": PPT_COLORS["twopcf"],
            "legend_fontsize": PPT_LEGEND_FONTSIZE,
            "legend_loc": "lower left of blank first-quadrant region",
            "legend_covariance_names_shown": False,
            "fnl_annotation_in_panel": True,
            "fnl_annotation_fontsize": PPT_FNL_FONTSIZE,
            "fnl_annotation_headroom": PPT_FNL_HEADROOM,
        },
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(
        json.dumps(base.jsonable(audit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[write] {AUDIT}")


if __name__ == "__main__":
    main()
