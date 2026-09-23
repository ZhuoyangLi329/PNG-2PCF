#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在已审计的 s50-350 both-jaxpower 主图上叠加 P(k)+xi0 联合 contour。

代码大纲
========
1. 读取与现主图完全相同的两条已审计长链（P(k) 24x50000 free-sn0、
   2PCF 64x20000 radial-RIC），以及本轮新的 joint 长链。
2. 复用 pair audit 冻结的共享坐标（fNL, b1）与 Task44 PPT 风格：
   无标题、无真值线、无 MAP 星号、P(k) 深炭灰、2PCF 红、joint 蓝。
3. 三条链的 fNL 结果按各自颜色写进左上子图；joint 画在最上层。
4. 输出单一 PDF（不覆盖已有审计 PDF，文件名带 joint）与机器审计
   JSON（含输入 SHA256、posterior、gate 与增益摘要）。
"""

from __future__ import annotations

import argparse
import hashlib
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

ACTIVE_T43_ROOT = PROJECT_ROOT / "plots" / "outputs" / "task43_outputs"
JOINT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "joint_pkxi_s50_350"

PK_SAMPLES = (
    ACTIVE_T43_ROOT
    / "ric_singleterm/fits/pk/task43_pk_ric_ph000_dchi2_nsub200000_motherbox_long_mcmc50k/"
    "task43_pk_ric_ph000_dchi2_nsub200000_motherbox_long_mcmc50k/"
    "task43_pk_lightcone_task43_pk_ric_ph000_dchi2_nsub200000_motherbox_long_mcmc50k_fit_samples.npz"
)
XI_SAMPLES = (
    ACTIVE_T43_ROOT
    / "ric_singleterm/fits/2pcf_jaxpower_ph000_dchi2_nsub200000_long_mcmc20k/"
    "task43_mcmc_radial_singleterm_samples.npz"
)
JOINT_SAMPLES = JOINT_ROOT / "fits" / "joint" / "samples.npz"
FIT_SUMMARY = JOINT_ROOT / "audits" / "task43_joint_pkxi_fit_summary.json"
PAIR_AUDIT = (
    ACTIVE_T43_ROOT
    / "ric_singleterm/audits/task43_ric_pk_vs_2pcf_covariance_pair_longchain.json"
)
OUTPUT_PDF = PROJECT_ROOT / "plots" / "task43" / "task43_pk_vs_2pcf_joint_s50_350_jaxpower_covariance.pdf"
OUTPUT_AUDIT = JOINT_ROOT / "audits" / "task43_pk_vs_2pcf_joint_contour_audit.json"

PPT_COLORS = {"joint": "#4C72B0", "pk": "#2F2F2F", "twopcf": "#C44E52"}
PPT_LEGEND_FONTSIZE = 17.0
PPT_FNL_FONTSIZE = 14.0
PPT_FNL_HEADROOM = 1.52  # 三行 fNL 文字需要更多顶部空间
PPT_LEGEND_GAP = 0.018
GETDIST_SMOOTH_1D = 0.35
GETDIST_SMOOTH_2D = 0.4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(samples_1d: np.ndarray) -> dict[str, float]:
    q16, q50, q84 = np.percentile(samples_1d, [15.8655, 50.0, 84.1345])
    return {"q16": float(q16), "q50": float(q50), "q84": float(q84), "err_low": float(q50 - q16), "err_high": float(q84 - q50)}


def constraint_text(stats: dict[str, float]) -> str:
    return rf"{stats['q50']:.1f}\,_{{-{stats['err_low']:.1f}}}^{{+{stats['err_high']:.1f}}}"


def fnl_panel_label(observable: str, stats: dict[str, float]) -> str:
    return rf"${observable}:\ f_{{\rm NL}}={constraint_text(stats)}$"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PDF)
    parser.add_argument("--audit", type=Path, default=OUTPUT_AUDIT)
    args = parser.parse_args()

    pair = json.loads(PAIR_AUDIT.read_text(encoding="utf-8"))
    xlim = tuple(float(v) for v in pair["shared_plot_ranges"]["fnl_loc"])
    ylim = tuple(float(v) for v in pair["shared_plot_ranges"]["b1"])

    pk_npz = np.load(PK_SAMPLES, allow_pickle=False)
    pk_samples = np.asarray(pk_npz["samples"], dtype="f8")[:, :2]
    xi_npz = np.load(XI_SAMPLES, allow_pickle=False)
    xi_samples = np.asarray(xi_npz["samples"], dtype="f8")[:, :2]
    joint_npz = np.load(JOINT_SAMPLES, allow_pickle=False)
    joint_samples = np.asarray(joint_npz["samples"], dtype="f8")[:, :2]

    axis_warnings = []
    for name, samples in (("pk", pk_samples), ("xi", xi_samples), ("joint", joint_samples)):
        lo, hi = np.percentile(samples[:, 0], [0.1, 99.9])
        if lo < xlim[0] or hi > xlim[1]:
            # pair-audit 冻结轴本来就是裁剪显示；原主图的审计 pk 链同样
            # 略超轴范围。这里记录为警告，不阻断出图。
            axis_warnings.append({"chain": name, "q01": float(lo), "q999": float(hi), "xlim": list(xlim)})

    from getdist import MCSamples, plots

    names = ["fnl_loc", "b1"]
    parameter_labels = [r"f_{\rm NL}", r"b_1"]
    ranges = {"fnl_loc": xlim, "b1": ylim}
    labels = {
        "joint": r"$P_0(k)+\xi_0(s)$" "\n" r"${\rm joint}$",
        "pk": r"$P_0(k)$" "\n" + rf"$0.006\leq k\leq 0.094\,h\,{{\rm Mpc}}^{{-1}}$",
        "twopcf": r"$\xi_0(s)$" "\n" + rf"$50<s<350\,h^{{-1}}{{\rm Mpc}}$",
    }
    # GetDist 首条 root 在最上层；joint 是本轮新增结果，放第一位保证可见。
    order = ("joint", "pk", "twopcf")
    sample_map = {"joint": joint_samples, "pk": pk_samples, "twopcf": xi_samples}
    chains = [
        MCSamples(
            samples=sample_map[key],
            names=names,
            labels=parameter_labels,
            label=labels[key],
            ranges=ranges,
            settings={"ignore_rows": 0, "smooth_scale_1D": GETDIST_SMOOTH_1D, "smooth_scale_2D": GETDIST_SMOOTH_2D},
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
        contour_lws=[1.8, 1.8, 1.8],
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

    blank_corner_x = float(axes[0, 0].get_position().x1)
    blank_corner_y = float(axes[1, 1].get_position().y1)
    legend_anchor = (blank_corner_x + PPT_LEGEND_GAP, blank_corner_y + PPT_LEGEND_GAP)
    if not figure.legends:
        raise RuntimeError("GetDist 没有创建 figure legend")
    figure.legends[-1].set_bbox_to_anchor(legend_anchor, transform=figure.transFigure)

    fnl_axis = axes[0, 0]
    ymin, ymax = fnl_axis.get_ylim()
    fnl_axis.set_ylim(ymin, ymin + (ymax - ymin) * PPT_FNL_HEADROOM)
    pk_fnl = summarize(pk_samples[:, 0])
    xi_fnl = summarize(xi_samples[:, 0])
    joint_fnl = summarize(joint_samples[:, 0])
    fnl_axis.text(0.035, 0.965, fnl_panel_label(r"P_0(k)", pk_fnl), transform=fnl_axis.transAxes, ha="left", va="top", fontsize=PPT_FNL_FONTSIZE, color=PPT_COLORS["pk"])
    fnl_axis.text(0.035, 0.845, fnl_panel_label(r"\xi_0(s)", xi_fnl), transform=fnl_axis.transAxes, ha="left", va="top", fontsize=PPT_FNL_FONTSIZE, color=PPT_COLORS["twopcf"])
    fnl_axis.text(0.035, 0.725, fnl_panel_label(r"P_0+\xi_0", joint_fnl), transform=fnl_axis.transAxes, ha="left", va="top", fontsize=PPT_FNL_FONTSIZE, color=PPT_COLORS["joint"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.pdf")
    try:
        figure.savefig(temporary, format="pdf", bbox_inches="tight", pad_inches=0.08)
        temporary.replace(args.output)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()
    print(f"[write] {args.output}")

    fit_summary = json.loads(FIT_SUMMARY.read_text(encoding="utf-8"))
    audit = {
        "task": "task43_plot_joint_pk_xi_contours",
        "status": "done",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "figure_contract": {
            "model": "radial single-term RIC on both probes; joint likelihood with measured P-xi cross covariance",
            "marginal_chains": "identical to the audited pair-figure chains (not rerun)",
            "axis_ranges_frozen_from": str(PAIR_AUDIT),
            "joint_on_top_layer": True,
            "style": "Task44 PPT contour style taught by user",
            "format": "pdf only",
            "science_ready": False,
        },
        "posterior": {
            "pk_audited": pair["posterior"]["pk_jaxpower"],
            "xi_audited": pair["posterior"]["2pcf_jaxpower"],
            "joint_new": {
                "fnl_loc": {key: float(v) for key, v in fit_summary["fits"]["joint"]["fnl_loc"].items()},
                "b1": {key: float(v) for key, v in fit_summary["fits"]["joint"]["b1"].items()},
            },
        },
        "improvement": fit_summary.get("improvement"),
        "chain_gates": {
            label: fit_summary["fits"][label]["chain_diagnostics"]["pass"] for label in fit_summary["fits"]
        },
        "axis_range_warnings": axis_warnings,
        "inputs_sha256": {
            str(path): sha256(path) for path in (PK_SAMPLES, XI_SAMPLES, JOINT_SAMPLES, FIT_SUMMARY, PAIR_AUDIT)
        },
        "output_pdf": str(args.output),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[write] {args.audit}")


if __name__ == "__main__":
    main()
