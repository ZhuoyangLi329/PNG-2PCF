#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic multi-chain contour figure with median + asymmetric errors labeled.

Standing rule from 2026-09-07: every fit computation gets a contour figure;
each chain is annotated as median_{-low}^{+high} with sigma68 in the legend.
Usage: --config <json> describing chains {key, path, color, label} and output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task43_plot_rsd_rawbox_pk0_vs_xi0_contours import (
    LABELS,
    draw_contour,
    plot_range,
)
from task43_rsd_common import PLOT_ROOT, atomic_write_json, sha256_file


DEFAULT_PARAMETERS = ("fNL", "b1", "sigma_s")


def load_flat(path: Path, ncols: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as d:
        key = next(k for k in ("chain_by_step", "chain_flat", "samples") if k in d.files)
        chain = np.asarray(d[key], dtype="f8")
    return chain.reshape(-1, chain.shape[-1])[:, :ncols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="JSON: {chains:[{key,path,color,label}], output_pdf, annotation}")
    args = parser.parse_args()
    spec = json.loads(args.config.read_text(encoding="utf-8"))
    pdf = PLOT_ROOT / spec["output_pdf"]
    if pdf.exists() or pdf.with_suffix(".json").exists():
        raise FileExistsError(f"immutable figure exists: {pdf}")
    parameters = tuple(spec.get("parameters", DEFAULT_PARAMETERS))
    chains: dict[str, np.ndarray] = {}
    chainspec = []
    for entry in spec["chains"]:
        chains[entry["key"]] = load_flat(Path(entry["path"]), len(parameters))
        chainspec.append((entry["key"], entry["color"], entry["label"]))
    ranges = {
        name: plot_range(
            np.concatenate([c[:, i] for c in chains.values()]),
            np.concatenate([c[:, i] for c in chains.values()]),
            parameter=name,
        )
        for i, name in enumerate(parameters)
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
    n = len(parameters)
    figure, axes = plt.subplots(n, n, figsize=(3.2 * n + 2.0, 3.0 * n + 1.0))
    thresholds: dict[str, Any] = {}
    for irow, yname in enumerate(parameters):
        for icol, xname in enumerate(parameters):
            axis = axes[irow, icol]
            if icol > irow:
                axis.set_axis_off()
                continue
            if icol == irow:
                bins = np.linspace(*ranges[xname], 90)
                for key, color, _ in chainspec:
                    axis.hist(chains[key][:, icol], bins=bins, density=True, histtype="step", lw=1.7, color=color)
                axis.set_xlim(*ranges[xname])
                axis.set_yticks([])
                if xname == "fNL":
                    axis.axvline(0.0, color="0.5", lw=0.8, ls="--")
            else:
                cell: dict[str, Any] = {}
                for zorder, (key, color, _) in enumerate(chainspec, start=1):
                    cell[key] = draw_contour(
                        axis, chains[key][:, icol], chains[key][:, irow],
                        color=color, xlim=ranges[xname], ylim=ranges[yname], zorder=2 * zorder - 1,
                    )
                thresholds[f"{xname}_vs_{yname}"] = cell
                axis.set(xlim=ranges[xname], ylim=ranges[yname])
                if xname == "fNL":
                    axis.axvline(0.0, color="0.5", lw=0.8, ls="--", zorder=0)
                if yname == "fNL":
                    axis.axhline(0.0, color="0.5", lw=0.8, ls="--", zorder=0)
            if irow < n - 1:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel(LABELS[xname])
            if icol == 0 and irow > 0:
                axis.set_ylabel(LABELS[yname])
            elif icol > 0 and irow != icol:
                axis.tick_params(labelleft=False)

    # Legend: median with asymmetric errors and sigma68 for every chain.
    handles, labels = [], []
    for key, color, label in chainspec:
        flat = chains[key]
        q = np.percentile(flat, [16.0, 50.0, 84.0], axis=0)
        med, dlo, dhi = q[1], q[1] - q[0], q[2] - q[1]
        sig68 = 0.5 * (q[2] - q[0])
        text = (
            rf"{label}: $f_{{\rm NL}}={med[0]:.2f}^{{+{dhi[0]:.2f}}}_{{-{dlo[0]:.2f}}}$"
            rf" ($\sigma={sig68[0]:.2f}$)"
        )
        handles.append(plt.Line2D([], [], color=color, lw=2.0))
        labels.append(text)
    figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.97, 0.95), frameon=False, fontsize=10.5)
    if spec.get("annotation"):
        figure.text(0.70, 0.50, spec["annotation"], ha="center", va="center", fontsize=9.5, color="0.30", linespacing=1.5)
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.96, wspace=0.07, hspace=0.07)
    figure.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)

    audit = {
        "task": "task43_plot_joint_contours_generic",
        "status": "pass",
        "config": spec,
        "chains": {key: {"path": str(Path(c["path"])), "sha256": sha256_file(Path(c["path"]))} for c in spec["chains"] for key in [c["key"]]},
        "plot_ranges": {name: list(v) for name, v in ranges.items()},
        "contour_thresholds": thresholds,
        "output_pdf": str(pdf),
        "output_pdf_sha256": sha256_file(pdf),
    }
    atomic_write_json(pdf.with_suffix(".json"), audit)
    print(json.dumps({"status": "pass", "output": str(pdf)}))


if __name__ == "__main__":
    main()
