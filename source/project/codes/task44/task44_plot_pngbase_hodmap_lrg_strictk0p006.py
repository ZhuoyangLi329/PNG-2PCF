#!/usr/bin/env python3
"""Current-contract LRG P0/xi0 best fits and Task43-style posterior plots."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASK43_CODE_DIR = Path(__file__).resolve().parents[1] / "task43"
if str(TASK43_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(TASK43_CODE_DIR))

from task43_plot_rsd_rawbox_pk0_vs_xi0_contours import (  # noqa: E402
    COLORS as TASK43_COLORS,
    draw_contour as task43_draw_contour,
    plot_range as task43_plot_range,
    posterior_text as task43_posterior_text,
)
from task44_pngbase_hodhost_mmin1e13_common import (  # noqa: E402
    CATALOGS,
    P_GAUSSIAN_PRIOR_MEAN,
    P_GAUSSIAN_PRIOR_SIGMA,
    PLOT_ROOT,
    get_spec,
    lrg_pk_conservative_fit_prefix,
    lrg_pk_conservative_gaussianp_fit_prefix,
    lrg_xi_extended_fit_prefix,
    lrg_xi_extended_gaussianp_fit_prefix,
    sha256_file,
)
from task44_pngbase_hodmap_rawbox_common import xi_metadata_path, xi_path  # noqa: E402


COLORS = {"c300": "#4C72B0", "c302": "#C44E52"}
DISPLAY_S_EDGES = np.arange(30.0, 350.0 + 10.0, 10.0, dtype="f8")
DISPLAY_S_CENTERS = 0.5 * (DISPLAY_S_EDGES[:-1] + DISPLAY_S_EDGES[1:])
FIXED_BESTFIT_PDF = PLOT_ROOT / (
    "task44_pngbase_c300_c302_hodmap_lrg_fixedfnl_"
    "pk0_kmin0p006_kmax0p080_xi0_smin30_smax350_bestfit_measurements.pdf"
)
FIXED_CONTOUR_PDF = PLOT_ROOT / (
    "task44_pngbase_c300_c302_hodmap_lrg_fixedfnl_"
    "pk0_kmin0p006_kmax0p080_vs_xi0_smin30_smax350_p_b1_contours.pdf"
)
GAUSSIAN_BESTFIT_PDF = PLOT_ROOT / (
    "task44_pngbase_c300_c302_hodmap_lrg_gaussianp_freefnl_"
    "pk0_kmin0p006_kmax0p080_xi0_smin30_smax350_bestfit_measurements.pdf"
)
GAUSSIAN_CONTOUR_PDF = PLOT_ROOT / (
    "task44_pngbase_c300_c302_hodmap_lrg_gaussianp_freefnl_"
    "pk0_kmin0p006_kmax0p080_vs_xi0_smin30_smax350_fnl_b1_p_contours.pdf"
)
OUTPUTS = (FIXED_BESTFIT_PDF, FIXED_CONTOUR_PDF, GAUSSIAN_BESTFIT_PDF, GAUSSIAN_CONTOUR_PDF)
PRIOR_LABEL = r"$p\sim\mathcal{N}(0.70724,\,0.26947^2)$"


def _prefix(tag: str, probe: str, gaussian: bool) -> Path:
    if gaussian:
        return (
            lrg_pk_conservative_gaussianp_fit_prefix(tag)
            if probe == "pk"
            else lrg_xi_extended_gaussianp_fit_prefix(tag)
        )
    return lrg_pk_conservative_fit_prefix(tag) if probe == "pk" else lrg_xi_extended_fit_prefix(tag)


def load_fit(tag: str, probe: str, gaussian: bool) -> dict[str, Any]:
    prefix = _prefix(tag, probe, gaussian)
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"missing LRG {probe} fit: {npz_path} / {json_path}")
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    if summary.get("status") != "pass" or summary.get("output_npz_sha256") != sha256_file(npz_path):
        raise RuntimeError(f"invalid LRG fit status/hash: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files if name != "summary_json"}
    return {"summary": summary, "arrays": arrays}


def load_full_xi(tag: str) -> dict[str, np.ndarray]:
    path = xi_path(tag, smin=30.0)
    metadata_path = xi_metadata_path(tag, smin=30.0)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "pass" or metadata.get("output_sha256") != sha256_file(path):
        raise RuntimeError(f"invalid full LRG xi status/hash: {path}")
    with np.load(path, allow_pickle=False) as data:
        result = {
            "s": np.asarray(data["s"], dtype="f8"),
            "edges": np.asarray(data["s_edges"], dtype="f8"),
            "xi": np.asarray(data["xi0"], dtype="f8"),
        }
    if not np.array_equal(result["s"], DISPLAY_S_CENTERS) or not np.array_equal(
        result["edges"], DISPLAY_S_EDGES
    ):
        raise RuntimeError(f"unexpected full LRG xi radial contract for {tag}")
    return result


def posterior(fit: dict[str, Any], name: str) -> dict[str, float]:
    return {key: float(value) for key, value in fit["summary"]["mcmc"]["posterior"][name].items()}


def interval(row: dict[str, float], digits: int) -> str:
    return (
        rf"{row['q50']:.{digits}f}_{{-{row['q50'] - row['q16']:.{digits}f}}}"
        rf"^{{+{row['q84'] - row['q50']:.{digits}f}}}"
    )


def flat_samples(fit: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    chain = np.asarray(fit["arrays"]["chain_final_by_step"], dtype="f8")
    flat = chain.reshape(-1, chain.shape[-1])
    available = [str(value) for value in fit["arrays"]["parameter_names"]]
    return np.column_stack([flat[:, available.index(name)] for name in names])


def save_figure(figure: plt.Figure, output: Path, *, tight: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.pdf")
    options = {"format": "pdf", "bbox_inches": "tight", "pad_inches": 0.08} if tight else {"format": "pdf"}
    figure.savefig(temporary, **options)
    plt.close(figure)
    temporary.replace(output)


def fit_box(tag: str, probe: str, fit: dict[str, Any], gaussian: bool) -> str:
    lines = [r"HOD-MAP LRGs, $z=0.5$"]
    if gaussian:
        lines.extend([PRIOR_LABEL, rf"input $f_{{\rm NL}}={get_spec(tag).fnl:g}$"])
    else:
        lines.append(rf"fixed $f_{{\rm NL}}={get_spec(tag).fnl:g}$")
    if probe == "pk":
        lines.extend(
            [
                r"$k_{\min}=0.006\ h\,{\rm Mpc}^{-1}$",
                r"$k_{\max}=0.080\ h\,{\rm Mpc}^{-1}$ (bin centre)",
            ]
        )
    else:
        lines.extend(
            [
                r"$s_{\min}^{\rm fit}=30\ h^{-1}{\rm Mpc}$",
                r"$s_{\max}^{\rm fit}=350\ h^{-1}{\rm Mpc}$",
            ]
        )
    row = posterior(fit, "fnl" if gaussian else "p")
    if gaussian:
        lines.append(rf"$f_{{\rm NL}}={interval(row, 1)}$")
    else:
        lines.append(rf"$p={interval(row, 2)}$")
    return "\n".join(lines)


def make_bestfit(*, gaussian: bool) -> None:
    output = GAUSSIAN_BESTFIT_PDF if gaussian else FIXED_BESTFIT_PDF
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), sharey="row")
    for column, tag in enumerate(CATALOGS):
        pk = load_fit(tag, "pk", gaussian)
        axis = axes[0, column]
        k = np.asarray(pk["arrays"]["coordinate"], dtype="f8")
        data = np.asarray(pk["arrays"]["data"], dtype="f8")
        prediction = np.asarray(pk["arrays"]["prediction_map"], dtype="f8")
        sigma = np.sqrt(np.diag(np.asarray(pk["arrays"]["covariance_final"], dtype="f8")))
        axis.errorbar(k, data, yerr=sigma, fmt="o", ms=3.7, capsize=1.7, color=COLORS[tag], label=r"LRG $P_0$ measurement")
        axis.plot(k, prediction, color="black", lw=1.45, label="MAP best fit")
        axis.set_xscale("log")
        axis.set_title(tag, fontsize=16, fontweight="bold")
        axis.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
        axis.text(
            0.04,
            0.04,
            fit_box(tag, "pk", pk, gaussian),
            transform=axis.transAxes,
            va="bottom",
            fontsize=8.4,
            linespacing=1.13,
            bbox={"facecolor": "white", "edgecolor": "0.72", "alpha": 0.92},
        )

        xi = load_fit(tag, "xi", gaussian)
        axis = axes[1, column]
        s = np.asarray(xi["arrays"]["coordinate"], dtype="f8")
        data = np.asarray(xi["arrays"]["data"], dtype="f8")
        prediction = np.asarray(xi["arrays"]["prediction_map"], dtype="f8")
        sigma = np.sqrt(np.diag(np.asarray(xi["arrays"]["covariance_final"], dtype="f8")))
        full = load_full_xi(tag)
        if not np.array_equal(full["s"], s) or not np.array_equal(full["xi"], data):
            raise RuntimeError(f"{tag} full LRG xi does not equal the fitted 30--350 data vector")
        axis.errorbar(
            s,
            s**2 * data,
            yerr=s**2 * sigma,
            fmt="o",
            ms=4.0,
            capsize=1.9,
            color=COLORS[tag],
            label=r"LRG $\xi_0$ measurement",
        )
        axis.plot(s, s**2 * prediction, color="black", lw=1.5, label="MAP best fit")
        axis.set_xlim(25.0, 355.0)
        axis.set_xlabel(r"$s\ [h^{-1}{\rm Mpc}]$")
        axis.text(
            0.96,
            0.96,
            fit_box(tag, "xi", xi, gaussian),
            transform=axis.transAxes,
            va="top",
            ha="right",
            fontsize=8.4,
            linespacing=1.13,
            bbox={"facecolor": "white", "edgecolor": "0.72", "alpha": 0.92},
        )

    axes[0, 0].set_ylabel(r"$P_0(k)\ [(h^{-1}{\rm Mpc})^3]$")
    axes[1, 0].set_ylabel(r"$s^2\xi_0(s)\ [(h^{-1}{\rm Mpc})^2]$")
    axes[0, 1].legend(frameon=False, loc="upper right", fontsize=9.0)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    order = [labels.index("MAP best fit"), labels.index(r"LRG $\xi_0$ measurement")]
    axes[1, 1].legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, loc="lower left", fontsize=8.8)
    figure.tight_layout()
    save_figure(figure, output)


def set_contour_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11.5,
            "axes.labelsize": 14.0,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def make_contours(*, gaussian: bool) -> None:
    set_contour_style()
    plot_names = ("fNL", "b1", "p") if gaussian else ("p", "b1")
    chain_names = ("fnl", "b1", "p") if gaussian else ("p", "b1")
    labels = {"fNL": r"$f_{\rm NL}$", "b1": r"$b_1$", "p": r"$p$"}
    output = GAUSSIAN_CONTOUR_PDF if gaussian else FIXED_CONTOUR_PDF
    figure = plt.figure(figsize=(17.4, 8.2) if gaussian else (12.4, 5.9))
    subfigures = figure.subfigures(1, len(CATALOGS), wspace=0.02)
    for subfigure, tag in zip(np.atleast_1d(subfigures), CATALOGS, strict=True):
        pk = load_fit(tag, "pk", gaussian)
        xi = load_fit(tag, "xi", gaussian)
        pk_samples = flat_samples(pk, chain_names)
        xi_samples = flat_samples(xi, chain_names)
        ranges = {
            name: task43_plot_range(pk_samples[:, index], xi_samples[:, index], parameter=name)
            for index, name in enumerate(plot_names)
        }
        axes = subfigure.subplots(len(plot_names), len(plot_names), squeeze=False)
        for irow, yname in enumerate(plot_names):
            for icol, xname in enumerate(plot_names):
                axis = axes[irow, icol]
                if icol > irow:
                    axis.set_axis_off()
                    continue
                if icol == irow:
                    bins = np.linspace(*ranges[xname], 90)
                    axis.hist(pk_samples[:, icol], bins=bins, density=True, histtype="step", lw=1.9, color=TASK43_COLORS["pk0"])
                    axis.hist(xi_samples[:, icol], bins=bins, density=True, histtype="step", lw=1.9, color=TASK43_COLORS["xi0"])
                    axis.set_xlim(*ranges[xname])
                    axis.set_yticks([])
                    if gaussian and xname == "fNL":
                        axis.axvline(get_spec(tag).fnl, color="0.5", lw=0.8, ls="--")
                else:
                    task43_draw_contour(axis, xi_samples[:, icol], xi_samples[:, irow], color=TASK43_COLORS["xi0"], xlim=ranges[xname], ylim=ranges[yname], zorder=1)
                    task43_draw_contour(axis, pk_samples[:, icol], pk_samples[:, irow], color=TASK43_COLORS["pk0"], xlim=ranges[xname], ylim=ranges[yname], zorder=3)
                    axis.set(xlim=ranges[xname], ylim=ranges[yname])
                    if gaussian and xname == "fNL":
                        axis.axvline(get_spec(tag).fnl, color="0.5", lw=0.8, ls="--", zorder=0)
                    if gaussian and yname == "fNL":
                        axis.axhline(get_spec(tag).fnl, color="0.5", lw=0.8, ls="--", zorder=0)
                if irow < len(plot_names) - 1:
                    axis.tick_params(labelbottom=False)
                else:
                    axis.set_xlabel(labels[xname])
                if icol == 0 and irow > 0:
                    axis.set_ylabel(labels[yname])
                elif icol > 0 and irow != icol:
                    axis.tick_params(labelleft=False)

        parameter = "fnl" if gaussian else "p"
        symbol = r"f_{\rm NL}" if gaussian else "p"
        handles = [
            plt.Line2D([], [], color=TASK43_COLORS["pk0"], lw=2.0),
            plt.Line2D([], [], color=TASK43_COLORS["xi0"], lw=2.0),
        ]
        legend_labels = [
            rf"$P_0(k):\ {symbol}={task43_posterior_text(posterior(pk, parameter))}$",
            rf"$\xi_0(s):\ {symbol}={task43_posterior_text(posterior(xi, parameter))}$",
        ]
        axes[0, 1].legend(
            handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(0.02, 0.20 if gaussian else 0.12),
            frameon=False,
            fontsize=12.0 if gaussian else 10.8,
        )
        annotation = [rf"$\mathbf{{{tag}}}$ HOD-MAP LRGs, $z=0.5$"]
        annotation.append(PRIOR_LABEL if gaussian else rf"Fiducial $f_{{\rm NL}}={get_spec(tag).fnl:g}$ (fixed)")
        if gaussian:
            annotation.append(rf"Input $f_{{\rm NL}}={get_spec(tag).fnl:g}$")
        annotation.extend(
            [
                r"$k_{\min,\max}^{\rm fit}=0.006,\ 0.080\ h\,{\rm Mpc}^{-1}$",
                r"$s_{\min,\max}^{\rm fit}=30,\ 350\ h^{-1}{\rm Mpc}$",
            ]
        )
        info_axis = axes[0, 2] if gaussian else axes[0, 1]
        info_axis.text(
            0.50,
            0.70 if gaussian else 0.73,
            "\n".join(annotation),
            transform=info_axis.transAxes,
            ha="center",
            va="center",
            fontsize=10.0 if gaussian else 8.8,
            linespacing=1.30,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "0.45", "linewidth": 0.9},
        )
        subfigure.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.96, wspace=0.0, hspace=0.0)
    save_figure(figure, output, tight=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if all(path.is_file() for path in OUTPUTS) and not args.overwrite:
        print("[skip] all four validated-contract LRG plot paths already exist")
        return
    make_bestfit(gaussian=False)
    make_contours(gaussian=False)
    make_bestfit(gaussian=True)
    make_contours(gaussian=True)
    for path in OUTPUTS:
        print(f"[plot] {path}")


if __name__ == "__main__":
    main()
