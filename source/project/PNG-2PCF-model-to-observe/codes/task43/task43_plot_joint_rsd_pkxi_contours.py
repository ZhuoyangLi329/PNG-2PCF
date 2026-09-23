#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-contour triangle for the boxsafe-RSD joint P0(k)+xi0(s) experiment.

Adds the joint posterior (blue, top layer) to the audited P0 (grey) vs xi0
(red) triangle.  Ranges are derived from the three chains of this experiment
with the frozen plot_range helper of the audited contour product; output is
PDF-only and never overwrites an existing product.
"""

from __future__ import annotations

import argparse
import json
import os
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
    posterior_text,
)
from task43_rsd_common import OUTPUT_ROOT, PLOT_ROOT, atomic_write_json, sha256_file


BOXSAFE_ROOT = OUTPUT_ROOT / "lightcone_boxsafe_zobs0p4_0p8"
FITS_DIR = BOXSAFE_ROOT / "joint_pkxi_s50_350" / "fits"
FIT_SUMMARY = BOXSAFE_ROOT / "joint_pkxi_s50_350" / "audits" / "task43_joint_rsd_pkxi_fit_summary.json"
DEFAULT_PDF = PLOT_ROOT / "task43_rsd_boxsafe_x25_pk0_xi0_joint_s50_350_contours.pdf"

PARAMETERS = ("fNL", "b1", "sigma_s")
COLORS = {"pk0": "#2F2F2F", "xi0": "#C44E52", "joint": "#4C72B0"}
ZORDERS = {"xi0": 1, "pk0": 3, "joint": 5}


def load_chain(variant: str) -> np.ndarray:
    path = FITS_DIR / variant / "samples.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        chain = np.asarray(data["chain_by_step"], dtype="f8")
    flat = chain.reshape(-1, chain.shape[-1])
    return flat[:, :3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args()
    audit_path = args.output.with_suffix(".json")
    if args.output.exists() or audit_path.exists():
        raise FileExistsError(f"immutable contour output exists: {args.output} / {audit_path}")

    summary = json.loads(FIT_SUMMARY.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise RuntimeError("joint fit summary is not complete")
    chains = {name: load_chain(name) for name in ("pk_marginal", "xi_marginal", "joint")}
    chain_of = {"pk0": chains["pk_marginal"], "xi0": chains["xi_marginal"], "joint": chains["joint"]}
    posteriors = {
        "pk0": summary["results"]["pk_marginal"]["posterior"]["fNL"],
        "xi0": summary["results"]["xi_marginal"]["posterior"]["fNL"],
        "joint": summary["results"]["joint"]["posterior"]["fNL"],
    }
    ranges = {
        name: plot_range(
            np.concatenate([chains["pk_marginal"][:, i], chains["xi_marginal"][:, i], chains["joint"][:, i]]),
            np.concatenate([chains["pk_marginal"][:, i], chains["xi_marginal"][:, i], chains["joint"][:, i]]),
            parameter=name,
        )
        for i, name in enumerate(PARAMETERS)
    }

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
    figure, axes = plt.subplots(3, 3, figsize=(8.7, 8.2))
    contour_audit: dict[str, Any] = {}
    for irow, yname in enumerate(PARAMETERS):
        for icol, xname in enumerate(PARAMETERS):
            axis = axes[irow, icol]
            if icol > irow:
                axis.set_axis_off()
                continue
            if icol == irow:
                bins = np.linspace(*ranges[xname], 90)
                for label in ("xi0", "pk0", "joint"):
                    axis.hist(
                        chain_of[label][:, icol],
                        bins=bins,
                        density=True,
                        histtype="step",
                        lw=1.9,
                        color=COLORS[label],
                    )
                axis.set_xlim(*ranges[xname])
                axis.set_yticks([])
                if xname == "fNL":
                    axis.axvline(0.0, color="0.5", lw=0.8, ls="--")
            else:
                key = f"{xname}_vs_{yname}"
                contour_audit[key] = {}
                for label in ("xi0", "pk0", "joint"):
                    chain = chain_of[label]
                    contour_audit[key][label] = draw_contour(
                        axis,
                        chain[:, icol],
                        chain[:, irow],
                        color=COLORS[label],
                        xlim=ranges[xname],
                        ylim=ranges[yname],
                        zorder=ZORDERS[label],
                    )
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

    handles = [plt.Line2D([], [], color=COLORS[label], lw=2.0) for label in ("pk0", "xi0", "joint")]
    labels = [
        rf"$P_0(k):\ f_{{\rm NL}}={posterior_text(posteriors['pk0'])}$",
        rf"$\xi_0(s):\ f_{{\rm NL}}={posterior_text(posteriors['xi0'])}$",
        rf"$\mathrm{{joint}}:\ f_{{\rm NL}}={posterior_text(posteriors['joint'])}$",
    ]
    figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.97, 0.94), frameon=False, fontsize=12.5)
    figure.text(0.69, 0.70, "68% and 95% contours", ha="center", va="center", fontsize=11.5, color="0.35")
    annotation = (
        "AbacusSummit halo lightcone\n"
        "Redshift space, $0.6<z_{\\rm obs}<0.8$\n"
        "(boxsafe $0.4<z_{\\rm obs}<0.8$ catalog)\n"
        "Fiducial $f_{\\rm NL}=0$\n"
        "$P_0$: 15 bins, $k_{\\max}=0.10\\ h\\,{\\rm Mpc}^{-1}$\n"
        "$\\xi_0$: formal-GIC, $s=50$--$350\\ h^{-1}{\\rm Mpc}$\n"
        "diagnostic covariance;\nRSD closure: validation$_{\\rm failed}$"
    )
    figure.text(0.71, 0.50, annotation, ha="center", va="center", fontsize=9.5, color="0.30", linespacing=1.45)
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.96, wspace=0.07, hspace=0.07)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp.pdf")
    figure.savefig(temporary, format="pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    temporary.replace(args.output)
    if args.output.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError("contour output does not have a PDF header")

    audit = {
        "task": "task43_plot_joint_rsd_pkxi_contours",
        "status": "pass",
        "scope": "boxsafe RSD lightcone P0 vs formal-GIC xi0 vs joint, ell=0 only, common (fNL,b1,sigma_s)",
        "credible_contours": [0.68, 0.95],
        "parameter_order": list(PARAMETERS),
        "colors": COLORS,
        "plot_ranges": {name: list(value) for name, value in ranges.items()},
        "contour_density_thresholds": contour_audit,
        "sample_counts": {name: int(chain.shape[0]) for name, chain in chains.items()},
        "fNL_posteriors": {label: posteriors[label] for label in posteriors},
        "fit_summary": {"path": str(FIT_SUMMARY), "sha256": sha256_file(FIT_SUMMARY)},
        "output_pdf": str(args.output),
        "output_pdf_sha256": sha256_file(args.output),
    }
    atomic_write_json(audit_path, audit)
    print(json.dumps({"status": "pass", "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
