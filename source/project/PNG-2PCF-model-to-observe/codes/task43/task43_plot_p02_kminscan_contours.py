#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contour comparison of the P2 kmin scan (fix-3).

One 3x3 triangle (fNL, b1, sigma_s) with five chains: P0-only control,
full-bin P0+P2, and kmin2 in {0.015, 0.02, 0.03}.
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
P02 = BOXSAFE_ROOT / "p02_increment"
PDF_PATH = PLOT_ROOT / "task43_rsd_boxsafe_p02_kminscan_contours.pdf"
PARAMETERS = ("fNL", "b1", "sigma_s")
CHAIN_SPECS = (
    ("p0", P02 / "fits" / "p0_control" / "samples.npz", "#2F2F2F", 1),
    ("full", P02 / "fits" / "p02" / "samples.npz", "#C44E52", 3),
    ("k015", P02 / "kminscan" / "fits" / "kmin2_0p015" / "samples.npz", "#4C72B0", 5),
    ("k020", P02 / "kminscan" / "fits" / "kmin2_0p020" / "samples.npz", "#55A868", 7),
    ("k030", P02 / "kminscan" / "fits" / "kmin2_0p030" / "samples.npz", "#B07AA1", 9),
)
LEGEND_LABELS = {
    "p0": "$P_0$ only",
    "full": "$P_0{+}P_2$ (all bins)",
    "k015": "$k_{\\min}^{P_2}=0.015$",
    "k020": "$k_{\\min}^{P_2}=0.020$",
    "k030": "$k_{\\min}^{P_2}=0.030$",
}


def main() -> None:
    if PDF_PATH.exists() or PDF_PATH.with_suffix(".json").exists():
        raise FileExistsError(f"immutable figure exists: {PDF_PATH}")
    kmin_audit = json.loads((P02 / "kminscan" / "audits" / "task43_rsd_boxsafe_p02_kminscan_summary.json").read_text(encoding="utf-8"))
    inc_audit = json.loads((P02 / "audits" / "task43_rsd_boxsafe_p02_increment_summary.json").read_text(encoding="utf-8"))

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
                    axis.hist(chains[key][:, icol], bins=bins, density=True, histtype="step", lw=1.5, color=color)
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

    fnl_rows = {
        "p0": inc_audit["results"]["p0_control"]["posterior"]["fNL"],
        "full": inc_audit["results"]["p02"]["posterior"]["fNL"],
        "k015": kmin_audit["results"]["kmin2_0p015"]["posterior"]["fNL"],
        "k020": kmin_audit["results"]["kmin2_0p020"]["posterior"]["fNL"],
        "k030": kmin_audit["results"]["kmin2_0p030"]["posterior"]["fNL"],
    }
    handles = [plt.Line2D([], [], color=spec[2], lw=2.0) for spec in CHAIN_SPECS]
    labels = [
        rf"{LEGEND_LABELS[key]}: $\sigma(f_{{\rm NL}})={fnl_rows[key]['sigma68']:.1f}$"
        for key, _, _, _ in CHAIN_SPECS
    ]
    figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.97, 0.95), frameon=False, fontsize=11.0)
    figure.text(0.69, 0.62, "68% and 95% contours\nboxsafe RSD lightcone\nfix-3: per-pole $k_{\\min}$ cut", ha="center", va="center", fontsize=10.5, color="0.35")
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.96, wspace=0.07, hspace=0.07)
    figure.savefig(PDF_PATH, format="pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)

    audit = {
        "task": "task43_plot_p02_kminscan_contours",
        "status": "pass",
        "chains": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path, _, _ in CHAIN_SPECS},
        "plot_ranges": {name: list(v) for name, v in ranges.items()},
        "contour_thresholds": thresholds,
        "output_pdf": str(PDF_PATH),
        "output_pdf_sha256": sha256_file(PDF_PATH),
    }
    atomic_write_json(PDF_PATH.with_suffix(".json"), audit)
    print(json.dumps({"status": "pass", "output": str(PDF_PATH)}))


if __name__ == "__main__":
    main()
