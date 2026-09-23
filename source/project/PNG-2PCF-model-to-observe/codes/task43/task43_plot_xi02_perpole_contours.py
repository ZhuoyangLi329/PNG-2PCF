#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contour comparison for the xi per-pole smin experiment.

3x3 triangle (fNL, b1, sigma_s) with four chains: xi0-only control (s>=50),
common-mask smin=80 (both poles cut), and per-pole xi0@s50+xi2@s80 / s120.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task43_plot_rsd_rawbox_pk0_vs_xi0_contours import (
    LABELS,
    draw_contour,
    plot_range,
)
from task43_rsd_common import OUTPUT_ROOT, PLOT_ROOT, atomic_write_json, sha256_file


BOXSAFE_ROOT = OUTPUT_ROOT / "lightcone_boxsafe_zobs0p4_0p8"
ELL2_DIR = BOXSAFE_ROOT / "ell2_increment"
PDF_PATH = PLOT_ROOT / "task43_rsd_boxsafe_xi02_perpole_contours.pdf"
PARAMETERS = ("fNL", "b1", "sigma_s")
CHAIN_SPECS = (
    ("control", ELL2_DIR / "perpole_smin" / "fits" / "ell0_s50_control" / "samples.npz", "#2F2F2F", 1),
    ("common80", ELL2_DIR / "fits" / "smin080_ell02" / "samples.npz", "#C44E52", 3),
    ("per80", ELL2_DIR / "perpole_smin" / "fits" / "ell0s50_ell2s80" / "samples.npz", "#4C72B0", 5),
    ("per120", ELL2_DIR / "perpole_smin" / "fits" / "ell0s50_ell2s120" / "samples.npz", "#55A868", 7),
)
LEGEND_LABELS = {
    "control": r"$\xi_0$ only ($s\geq50$)",
    "common80": r"common $s_{\min}=80$ ($\xi_0$ cut too)",
    "per80": r"$\xi_0^{s\geq50}+\xi_2^{s\geq80}$",
    "per120": r"$\xi_0^{s\geq50}+\xi_2^{s\geq120}$",
}


def load_flat(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as d:
        key = "chain_flat" if "chain_flat" in d.files else "chain_by_step"
        chain = np.asarray(d[key], dtype="f8")
    return chain.reshape(-1, chain.shape[-1])[:, :3]


def main() -> None:
    if PDF_PATH.exists() or PDF_PATH.with_suffix(".json").exists():
        raise FileExistsError(f"immutable figure exists: {PDF_PATH}")
    chains = {key: load_flat(path) for key, path, _, _ in CHAIN_SPECS}
    ranges = {
        name: plot_range(
            np.concatenate([c[:, i] for c in chains.values()]),
            np.concatenate([c[:, i] for c in chains.values()]),
            parameter=name,
        )
        for i, name in enumerate(PARAMETERS)
    }

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "font.size": 11.5,
            "axes.labelsize": 13.5,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    figure, axes = plt.subplots(3, 3, figsize=(8.5, 8.1))
    thresholds: dict = {}
    for irow, yname in enumerate(PARAMETERS):
        for icol, xname in enumerate(PARAMETERS):
            axis = axes[irow, icol]
            if icol > irow:
                axis.set_axis_off()
                continue
            if icol == irow:
                bins = np.linspace(*ranges[xname], 90)
                for key, _, color, _ in CHAIN_SPECS:
                    axis.hist(chains[key][:, icol], bins=bins, density=True, histtype="step", lw=1.6, color=color)
                axis.set_xlim(*ranges[xname])
                axis.set_yticks([])
                if xname == "fNL":
                    axis.axvline(0.0, color="0.5", lw=0.8, ls="--")
            else:
                cell: dict = {}
                for key, _, color, zorder in CHAIN_SPECS:
                    cell[key] = draw_contour(
                        axis, chains[key][:, icol], chains[key][:, irow],
                        color=color, xlim=ranges[xname], ylim=ranges[yname], zorder=zorder,
                    )
                thresholds[f"{xname}_vs_{yname}"] = cell
                axis.set(xlim=ranges[xname], ylim=ranges[yname])
                if xname == "fNL":
                    axis.axvline(0.0, color="0.5", lw=0.8, ls="--", zorder=0)
                if yname == "fNL":
                    axis.axhline(0.0, color="0.5", lw=0.8, ls="--", zorder=0)
            if irow < len(PARAMETERS) - 1:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel(LABELS[xname])
            if icol == 0 and irow > 0:
                axis.set_ylabel(LABELS[yname])
            elif icol > 0 and irow != icol:
                axis.tick_params(labelleft=False)

    sigma = {"control": 36.53, "common80": 39.04, "per80": 31.70, "per120": 35.51}
    handles = [plt.Line2D([], [], color=spec[2], lw=2.0) for spec in CHAIN_SPECS]
    labels = [rf"{LEGEND_LABELS[key]}: $\sigma(f_{{\rm NL}})={sigma[key]:.1f}$" for key, _, _, _ in CHAIN_SPECS]
    figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.97, 0.95), frameon=False, fontsize=10.5)
    figure.text(0.69, 0.60, "68% and 95% contours\nboxsafe RSD lightcone\nper-pole $s_{\\min}$ experiment", ha="center", va="center", fontsize=10.0, color="0.35")
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.96, wspace=0.07, hspace=0.07)
    figure.savefig(PDF_PATH, format="pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)

    audit = {
        "task": "task43_plot_xi02_perpole_contours",
        "status": "pass",
        "chains": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path, _, _ in CHAIN_SPECS},
        "sigma_fNL_reference": sigma,
        "plot_ranges": {name: list(v) for name, v in ranges.items()},
        "contour_thresholds": thresholds,
        "output_pdf": str(PDF_PATH),
        "output_pdf_sha256": sha256_file(PDF_PATH),
    }
    atomic_write_json(PDF_PATH.with_suffix(".json"), audit)
    print(json.dumps({"status": "pass", "output": str(PDF_PATH)}))


if __name__ == "__main__":
    main()
