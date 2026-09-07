#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot free-sigmas constraints for all Quijote tags currently available locally.

Rows are grouped by data tag.  Within each group we show P(k) center,
P(k) BinAvgFit, and 2PCF rlist choices.  The three panels show fnl_loc, b1,
sigmas, and optionally sn0 with 1-sigma errorbars.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task4_outputs" / "quijote_lcp50_profiler"
PLOT_DIR = PROJECT_ROOT / "plots" / "task4"

BASE_PARAMETERS = [
    ("fnl_loc", r"$f_{\mathrm{NL}}^{\mathrm{loc}}$"),
    ("b1", r"$b_1$"),
    ("sigmas", r"$\sigma_s\,[h^{-1}{\rm Mpc}]$"),
]
SN0_PARAMETER = ("sn0", r"$s_{n,0}\,[(h^{-1}{\rm Mpc})^3]$")

DEFAULT_INPUT_TEMPLATE = "task42_quijote_lcp50_profiler_freesigmas_single_rmin50_{tag}.json"
DEFAULT_OUT_PDF = PLOT_DIR / "task42_quijote_alltags_freesigmas_constraints.pdf"


def covariance_corrections(nmock: int, ndata: int, nparams: int) -> dict[str, float]:
    """Return Hartlap and Percival factors for mock-estimated covariance."""
    hartlap = (nmock - ndata - 2.0) / (nmock - 1.0)
    a = 2.0 / ((nmock - ndata - 1.0) * (nmock - ndata - 4.0))
    b = (nmock - ndata - 2.0) / ((nmock - ndata - 1.0) * (nmock - ndata - 4.0))
    m1 = (1.0 + b * (ndata - nparams)) / (1.0 + a + b * (nparams + 1.0))
    return {
        "hartlap": float(hartlap),
        "percival_error_factor": float(math.sqrt(m1)),
        "total_error_factor_if_hartlap_not_in_fit": float(math.sqrt(m1 / hartlap)),
    }


def tag_files(input_template: str) -> dict[str, Path]:
    """Build the tag -> JSON path mapping from a filename template."""
    return {
        tag: OUTPUT_DIR / input_template.format(tag=tag)
        for tag in ["fid", "LCp50", "LCp100"]
    }


def load_rows(input_template: str, parameters: list[tuple[str, str]]) -> list[dict[str, object]]:
    """Load all constraints into a common row format."""
    rows: list[dict[str, object]] = []
    for tag, path in tag_files(input_template).items():
        payload = json.loads(path.read_text(encoding="utf-8"))

        for key, label in [("center", "P(k) center"), ("binavg", "P(k) BinAvgFit")]:
            item = payload["pk_reference"][key]
            ndata = int(payload["pk_reference"]["n_fit_bins"])
            nparams = int(ndata - int(item["ndof"]))
            pk_corr = covariance_corrections(
                nmock=int(payload["inputs"]["n_pk_mocks"]),
                ndata=ndata,
                nparams=nparams,
            )
            pk_error_factor = pk_corr["total_error_factor_if_hartlap_not_in_fit"]
            rows.append(
                {
                    "tag": tag,
                    "method": "pk",
                    "label": f"{tag}  {label}",
                    "values": {name: float(item[name]) for name, _ in parameters},
                    "errors": {
                        name: float(item["errors_raw"][name] * pk_error_factor) for name, _ in parameters
                    },
                }
            )

        for item in payload["profiler_results"]:
            rows.append(
                {
                    "tag": tag,
                    "method": "xi",
                    "label": (
                        f"{tag}  2PCF {int(item['s_center_min'])}-{int(item['s_center_max'])}"
                        + rf"  $(N={int(item['ndata'])})$"
                    ),
                    "values": {name: float(item["bestfit"][name]) for name, _ in parameters},
                    "errors": {name: float(item["errors_percival"][name]) for name, _ in parameters},
                }
            )

        rows.append({"tag": tag, "method": "gap", "label": "", "values": {}, "errors": {}})
    return rows[:-1]


def axis_limits(rows: list[dict[str, object]], parameter: str) -> tuple[float, float]:
    """Choose padded x limits that include all visible intervals."""
    lo = []
    hi = []
    for row in rows:
        if row["method"] == "gap":
            continue
        value = row["values"][parameter]
        error = row["errors"][parameter]
        lo.append(value - error)
        hi.append(value + error)
    xmin = float(np.nanmin(lo))
    xmax = float(np.nanmax(hi))
    span = xmax - xmin
    pad = 0.08 * span if span > 0 else 1.0
    return xmin - pad, xmax + pad


def parse_args() -> argparse.Namespace:
    """Parse plot options."""
    parser = argparse.ArgumentParser(description="Plot Task42 all-tag Quijote profiler constraints")
    parser.add_argument(
        "--input-template",
        default=DEFAULT_INPUT_TEMPLATE,
        help=(
            "JSON filename template under outputs/task4_outputs/quijote_lcp50_profiler; "
            "must contain {tag}"
        ),
    )
    parser.add_argument("--output-pdf", default=str(DEFAULT_OUT_PDF), help="output PDF path")
    parser.add_argument("--include-sn0", action="store_true", help="add a fourth sn0 panel")
    parser.add_argument(
        "--title",
        default="",
        help="custom title; default is chosen from include-sn0",
    )
    return parser.parse_args()


def plot() -> None:
    """Create the all-tag forest plot."""
    args = parse_args()
    out_pdf = Path(args.output_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    parameters = list(BASE_PARAMETERS)
    if args.include_sn0:
        parameters.append(SN0_PARAMETER)
    rows = load_rows(args.input_template, parameters)
    y = np.arange(len(rows))[::-1]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1,
        len(parameters),
        figsize=(23.0 if args.include_sn0 else 19.0, 12.5),
        sharey=True,
        gridspec_kw={"wspace": 0.17},
    )
    if len(parameters) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#f7f7f5")

    tag_colors = {"fid": "#3c6e9f", "LCp50": "#bf5b2f", "LCp100": "#417a50"}
    method_markers = {"pk": "s", "xi": "o"}

    for ax, (parameter, title) in zip(axes, parameters):
        ax.set_facecolor("#fbfbfa")
        for idx, row in enumerate(rows):
            yi = y[idx]
            if row["method"] == "gap":
                ax.axhline(yi, color="#a6a6a6", lw=0.9, alpha=0.55, zorder=0)
                continue
            if idx % 2 == 0:
                ax.axhspan(yi - 0.42, yi + 0.42, color="#ececea", alpha=0.45, zorder=0)
            color = tag_colors[row["tag"]]
            marker = method_markers[row["method"]]
            ax.errorbar(
                row["values"][parameter],
                yi,
                xerr=row["errors"][parameter],
                fmt=marker,
                ms=7.0,
                mfc="white",
                mec=color,
                mew=2.0,
                ecolor=color,
                elinewidth=1.9,
                capsize=3.8,
                capthick=1.7,
                alpha=0.95,
                zorder=3,
            )

        ax.set_title(title, pad=12)
        ax.set_xlabel("best fit +/- 1 sigma")
        ax.set_xlim(*axis_limits(rows, parameter))
        ax.set_ylim(-0.8, len(rows) - 0.2)
        ax.grid(axis="x", color="#c9c9c9", lw=0.8, alpha=0.65)
        ax.tick_params(axis="y", length=0)

    visible_labels = [row["label"] for row in rows]
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(visible_labels)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    handles = []
    for tag, color in tag_colors.items():
        handles.append(axes[0].plot([], [], "o", ms=8, mfc="white", mec=color, mew=2, label=tag)[0])
    handles.extend(
        [
            axes[0].plot([], [], "s", ms=8, mfc="white", mec="#555555", mew=2, label="P(k)")[0],
            axes[0].plot([], [], "o", ms=8, mfc="white", mec="#555555", mew=2, label="2PCF")[0],
        ]
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.965), ncol=5, frameon=False)
    title = args.title
    if not title:
        title = (
            "Quijote local-PNG tags: P(k) and 2PCF constraints with free $\\sigma_s$ and $s_{n,0}$"
            if args.include_sn0
            else "Quijote local-PNG tags: P(k) and 2PCF constraints with free $\\sigma_s$"
        )
    fig.suptitle(title, x=0.54, y=0.995, fontsize=17, fontweight="bold")
    fig.text(
        0.54,
        0.035,
        "Each tag uses its own 500-realization sample covariance.  P(k): kcen <= 0.08 h/Mpc.  2PCF: rmax = 350 Mpc/h.",
        ha="center",
        va="center",
        fontsize=11,
        color="#4d4d4d",
    )
    fig.subplots_adjust(left=0.29, right=0.985, top=0.90, bottom=0.085)

    fig.savefig(out_pdf)
    print(out_pdf)


if __name__ == "__main__":
    plot()
