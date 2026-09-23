#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Four-way joint contour triangle: P0+P2 vs xi0+xi2 vs joint vs naive.

The sigma_s row shows the cross-probe tension (P-side ~1.2 vs xi-side ~9.2)
that dominates the joint's parameter compromise.
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
    posterior_text,
)
from task43_rsd_common import OUTPUT_ROOT, PLOT_ROOT, atomic_write_json, sha256_file


BOXSAFE_ROOT = OUTPUT_ROOT / "lightcone_boxsafe_zobs0p4_0p8"
JOINT_DIR = BOXSAFE_ROOT / "joint_p02xi02"
AUDIT = JOINT_DIR / "audits" / "task43_rsd_joint_p02xi02_fit_summary.json"
PDF_PATH = PLOT_ROOT / "task43_rsd_boxsafe_p02xi02_joint_contours.pdf"
PARAMETERS = ("fNL", "b1", "sigma_s")
CHAIN_SPECS = (
    ("p02", JOINT_DIR / "fits" / "p02_marginal" / "samples.npz", "#2F2F2F", 1),
    ("xi02", JOINT_DIR / "fits" / "xi02_marginal" / "samples.npz", "#C44E52", 3),
    ("naive", JOINT_DIR / "fits" / "joint_naive" / "samples.npz", "#B07AA1", 5),
    ("joint", JOINT_DIR / "fits" / "joint" / "samples.npz", "#4C72B0", 7),
)


def main() -> None:
    if PDF_PATH.exists() or PDF_PATH.with_suffix(".json").exists():
        raise FileExistsError(f"immutable figure exists: {PDF_PATH}")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    chains = {}
    for key, path, _, _ in CHAIN_SPECS:
        with np.load(path, allow_pickle=False) as d:
            arr = np.asarray(d["chain_by_step"], dtype="f8")
        chains[key] = arr.reshape(-1, arr.shape[-1])[:, :3]
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
    figure, axes = plt.subplots(3, 3, figsize=(8.6, 8.2))
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

    fnl = {k: audit["results"][n]["posterior"]["fNL"] for k, n in
           (("p02", "p02_marginal"), ("xi02", "xi02_marginal"), ("naive", "joint_naive"), ("joint", "joint"))}
    handles = [plt.Line2D([], [], color=spec[2], lw=2.0) for spec in CHAIN_SPECS]
    labels = [
        rf"$P_0{{+}}P_2:\ f_{{\rm NL}}={posterior_text(fnl['p02'])}$",
        rf"$\xi_0{{+}}\xi_2:\ f_{{\rm NL}}={posterior_text(fnl['xi02'])}$",
        rf"joint naive: $\sigma(f_{{\rm NL}})={fnl['naive']['sigma68']:.1f}$",
        rf"joint: $\ f_{{\rm NL}}={posterior_text(fnl['joint'])}$",
    ]
    figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.97, 0.95), frameon=False, fontsize=11.0)
    figure.text(
        0.70,
        0.50,
        r"$\sigma_s$ tension:" "\n"
        r"$P$ side $1.2\pm0.9$ vs $\xi$ side $9.2\pm1.5\ h^{-1}{\rm Mpc}$" "\n"
        "the joint settles at an unphysical\n"
        "compromise $\sigma_s\simeq2.7$; the nominal\n"
        r"$-26\%$ gain in $\sigma(f_{\rm NL})$ rides on misfit",
        ha="center",
        va="center",
        fontsize=9.5,
        color="0.30",
        linespacing=1.5,
    )
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.96, wspace=0.07, hspace=0.07)
    figure.savefig(PDF_PATH, format="pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)

    out = {
        "task": "task43_plot_p02xi02_joint_contours",
        "status": "pass",
        "chains": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path, _, _ in CHAIN_SPECS},
        "plot_ranges": {name: list(v) for name, v in ranges.items()},
        "contour_thresholds": thresholds,
        "fNL_posteriors": {k: fnl[k] for k in fnl},
        "output_pdf": str(PDF_PATH),
        "output_pdf_sha256": sha256_file(PDF_PATH),
    }
    atomic_write_json(PDF_PATH.with_suffix(".json"), out)
    print(json.dumps({"status": "pass", "output": str(PDF_PATH)}))


if __name__ == "__main__":
    main()
