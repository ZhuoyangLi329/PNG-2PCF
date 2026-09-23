#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""9.22meeting-style all-parameter joint contour comparison.

这个脚本严格沿用 Task43 9.18/9.22 meeting 图的画法：

- lower-triangle corner layout；
- diagonal raw-chain histograms；
- off-diagonal filled 68%/95% contours；
- Arial/DejaVu Sans 风格、内向 ticks、统一轴范围；
- PDF only。

比较的两个 joint chain 使用同一个 promoted rawbox BAO-mask 和固定 joint
covariance。原模型的 xi-side parameter ``sigma_s_xi`` 与 GSM 的
``sigma_FOG`` 被放入同一个显示列，但图注明确这两个物理定义不同。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import numpy as np

from task43_plot_rsd_rawbox_pk0_vs_xi0_contours import density_levels, plot_range
from task43_rsd_common import atomic_write_json, sha256_file


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "task432_model_repair"
OUTDIR = PROJECT_ROOT / "9.22meeting" / "task432_linear_gsm_vs_original"
ORIGINAL = OUTPUT_ROOT / "rawbox_baomask_split_sigma_longchain" / "fits" / "joint" / "samples.npz"
GSM = OUTPUT_ROOT / "rawbox_linear_gsm_contours_v2" / "linear_gsm_fog_joint_samples.npz"

PARAMETERS = ("fNL", "b1", "sigma_s_P", "sigma_xi_display", "sn0")
LABELS = {
    "fNL": r"$f_{\rm NL}$",
    "b1": r"$b_1$",
    "sigma_s_P": r"$\sigma_{s,P}\ [h^{-1}{\rm Mpc}]$",
    "sigma_xi_display": r"$\sigma_{\xi/{\rm GSM}}\ [h^{-1}{\rm Mpc}]$",
    "sn0": r"$s_{n0}$",
}
COLORS = {"original": "#2F2F2F", "gsm": "#4C72B0"}
VARIANTS = (
    ("original", "original FullDiscrete split-sigma", COLORS["original"]),
    ("gsm", "linear GSM + sigma_FOG", COLORS["gsm"]),
)


def load_chain(path: Path, *, xi_name: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        samples = np.asarray(payload["samples"], dtype="f8")
        names = [str(value) for value in np.asarray(payload["parameter_names"]).ravel()]
    index = {name: i for i, name in enumerate(names)}
    required = ["fNL", "b1", "sigma_s_P", xi_name, "sn0"]
    if any(name not in index for name in required):
        raise ValueError(f"{path} lacks required parameters {required}; found {names}")
    if samples.ndim != 2 or samples.shape[1] != len(names) or not np.all(np.isfinite(samples)):
        raise ValueError(f"invalid chain shape/content: {path} {samples.shape}")
    return {
        "fNL": samples[:, index["fNL"]],
        "b1": samples[:, index["b1"]],
        "sigma_s_P": samples[:, index["sigma_s_P"]],
        "sigma_xi_display": samples[:, index[xi_name]],
        "sn0": samples[:, index["sn0"]],
    }


def draw_contour(axis: plt.Axes, x: np.ndarray, y: np.ndarray, *, color: str, xlim: tuple[float, float], ylim: tuple[float, float], zorder: int) -> dict[str, float]:
    hist, xedges, yedges = np.histogram2d(
        np.asarray(x, dtype="f8"),
        np.asarray(y, dtype="f8"),
        bins=(170, 160),
        range=(xlim, ylim),
        density=False,
    )
    hist = __import__("scipy.ndimage", fromlist=["gaussian_filter"]).gaussian_filter(
        hist.astype("f8"), sigma=2.5, mode="nearest"
    )
    level95, level68 = density_levels(hist)
    top = float(np.max(hist)) * 1.001
    xcenter = 0.5 * (xedges[:-1] + xedges[1:])
    ycenter = 0.5 * (yedges[:-1] + yedges[1:])
    from matplotlib.colors import to_rgba

    axis.contourf(
        xcenter,
        ycenter,
        hist.T,
        levels=[level95, level68, top],
        colors=[to_rgba(color, 0.10), to_rgba(color, 0.22)],
        antialiased=True,
        zorder=zorder,
    )
    axis.contour(
        xcenter,
        ycenter,
        hist.T,
        levels=[level95, level68],
        colors=color,
        linewidths=[1.2, 1.8],
        zorder=zorder + 1,
    )
    return {"level95": float(level95), "level68": float(level68), "density_max": top / 1.001}


def plot_range_all(left: np.ndarray, right: np.ndarray, *, parameter: str) -> tuple[float, float]:
    values = np.concatenate([np.asarray(left, dtype="f8"), np.asarray(right, dtype="f8")])
    lo, hi = np.quantile(values, [0.001, 0.999])
    if parameter == "fNL":
        lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    if parameter in {"sigma_s_P", "sigma_xi_display", "sn0"}:
        lo = min(0.0, float(lo))
    width = float(hi - lo)
    return float(lo - (0.0 if parameter in {"sigma_s_P", "sigma_xi_display", "sn0"} else 0.06 * width)), float(hi + 0.06 * width)


def posterior_text(samples: np.ndarray) -> str:
    q16, q50, q84 = np.percentile(np.asarray(samples, dtype="f8"), [16.0, 50.0, 84.0])
    return rf"{q50:.2f}_{{-{q50-q16:.2f}}}^{{+{q84-q50:.2f}}}"


def main() -> None:
    output = OUTDIR / "task432_original_vs_linear_gsm_allparams_9.22style.pdf"
    audit_path = output.with_suffix(".json")
    if output.exists() or audit_path.exists():
        raise FileExistsError(f"immutable output exists: {output}")

    original = load_chain(ORIGINAL, xi_name="sigma_s_xi")
    gsm = load_chain(GSM, xi_name="sigma_FOG")
    ranges = {
        name: plot_range_all(original[name], gsm[name], parameter=name)
        for name in PARAMETERS
    }

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10.0,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.pdf")
    contour_audit: dict[str, dict[str, dict[str, float]]] = {}
    with PdfPages(temporary) as pdf:
        figure, axes = plt.subplots(5, 5, figsize=(11.2, 10.8), squeeze=False)
        for irow, yname in enumerate(PARAMETERS):
            for icol, xname in enumerate(PARAMETERS):
                axis = axes[irow, icol]
                if icol > irow:
                    axis.set_axis_off()
                    continue
                axis.set_xlim(*ranges[xname])
                axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
                axis.tick_params(labelsize=9)
                if icol == irow:
                    bins = np.linspace(*ranges[xname], 90)
                    for key, label, color in VARIANTS:
                        axis.hist(original[xname] if key == "original" else gsm[xname], bins=bins, density=True, histtype="step", lw=1.8, color=color)
                    axis.set_yticks([])
                    if xname == "fNL":
                        axis.axvline(0.0, color="0.55", lw=0.8, ls="--")
                else:
                    axis.set_ylim(*ranges[yname])
                    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
                    contour_audit[f"{xname}_vs_{yname}"] = {}
                    for zorder, (key, label, color) in enumerate(VARIANTS):
                        samples = original if key == "original" else gsm
                        contour_audit[f"{xname}_vs_{yname}"][key] = draw_contour(
                            axis,
                            samples[xname],
                            samples[yname],
                            color=color,
                            xlim=ranges[xname],
                            ylim=ranges[yname],
                            zorder=2 + 2 * zorder,
                        )
                    if xname == "fNL":
                        axis.axvline(0.0, color="0.55", lw=0.8, ls="--", zorder=0)
                    if yname == "fNL":
                        axis.axhline(0.0, color="0.55", lw=0.8, ls="--", zorder=0)
                if irow < len(PARAMETERS) - 1:
                    axis.tick_params(labelbottom=False)
                else:
                    axis.set_xlabel(LABELS[xname], fontsize=12)
                if icol == 0 and irow > 0:
                    axis.set_ylabel(LABELS[yname], fontsize=12)
                elif icol > 0 and irow != icol:
                    axis.tick_params(labelleft=False)

        handles = [plt.Line2D([], [], color=color, lw=2.0) for _, _, color in VARIANTS]
        labels = [
            f"original FullDiscrete split-sigma\n$f_{{\\rm NL}}={posterior_text(original['fNL'])}$",
            f"linear GSM + $\\sigma_{{\\rm FOG}}$\n$f_{{\\rm NL}}={posterior_text(gsm['fNL'])}$",
        ]
        figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.965), frameon=False, fontsize=11.0)
        figure.suptitle("Task43 rawbox RSD: original versus linear GSM", fontsize=14.0, y=0.985)
        figure.text(
            0.5,
            0.955,
            r"$50\leq s<350\ h^{-1}{\rm Mpc}$; BAO mask $80$--$120$; same promoted joint covariance",
            ha="center",
            fontsize=10.0,
        )
        figure.text(
            0.5,
            0.012,
            "68% / 95% contours from raw chains, following the 9.18/9.22meeting convention; "
            "xi-side width definitions differ between the two models.",
            ha="center",
            fontsize=8.2,
            color="0.4",
        )
        figure.subplots_adjust(left=0.09, right=0.975, bottom=0.075, top=0.90, wspace=0.075, hspace=0.075)
        pdf.savefig(figure)
        plt.close(figure)

    temporary.replace(output)
    audit = {
        "task": "task432_original_vs_linear_gsm_allparams_9.22style",
        "status": "pass",
        "style_reference": "9.18/9.22meeting allparams lower-triangle corner",
        "parameter_order": list(PARAMETERS),
        "variants": {key: {"label": label, "color": color} for key, label, color in VARIANTS},
        "credible_contours": [0.68, 0.95],
        "ranges": {name: list(value) for name, value in ranges.items()},
        "contour_density_thresholds": contour_audit,
        "chains": {
            "original": {"path": str(ORIGINAL), "sha256": sha256_file(ORIGINAL), "samples": int(original["fNL"].size)},
            "gsm": {"path": str(GSM), "sha256": sha256_file(GSM), "samples": int(gsm["fNL"].size)},
        },
        "definitions": {
            "original_xi_side": "sigma_s_xi from FullDiscrete xi02 split-sigma fit",
            "gsm_xi_side": "sigma_FOG from configuration-space linear GSM",
        },
        "output_pdf": str(output),
        "output_pdf_sha256": sha256_file(output),
    }
    atomic_write_json(audit_path, audit)
    print(json.dumps({"status": "pass", "output": str(output), "audit": str(audit_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
