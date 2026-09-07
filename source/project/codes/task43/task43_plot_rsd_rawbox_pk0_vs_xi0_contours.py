#!/usr/bin/env python3
"""Plot common-parameter P0-vs-xi0 contours for Task 4.3.2 rawboxes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
from scipy.ndimage import gaussian_filter

from task43_rsd_common import atomic_write_json, sha256_file


PARAMETERS = ("fNL", "b1", "sigma_s")
LABELS = {
    "fNL": r"$f_{\rm NL}$",
    "b1": r"$b_1$",
    "sigma_s": r"$\sigma_s\ [h^{-1}{\rm Mpc}]$",
}
COLORS = {"pk0": "#2F2F2F", "xi0": "#C44E52"}


def density_levels(density: np.ndarray) -> tuple[float, float]:
    flat = np.sort(np.asarray(density, dtype="f8").ravel())[::-1]
    cumulative = np.cumsum(flat)
    cumulative /= cumulative[-1]
    level68 = flat[min(int(np.searchsorted(cumulative, 0.68)), flat.size - 1)]
    level95 = flat[min(int(np.searchsorted(cumulative, 0.95)), flat.size - 1)]
    if not 0.0 < level95 < level68 < float(flat[0]):
        raise RuntimeError(f"invalid contour thresholds: {level95}, {level68}, {flat[0]}")
    return float(level95), float(level68)


def plot_range(left: np.ndarray, right: np.ndarray, *, parameter: str) -> tuple[float, float]:
    values = np.concatenate([np.asarray(left, dtype="f8"), np.asarray(right, dtype="f8")])
    lo, hi = np.quantile(values, [0.001, 0.999])
    if parameter == "fNL":
        lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    if parameter == "sigma_s":
        lo = 0.0
    width = float(hi - lo)
    return float(lo - (0.0 if parameter == "sigma_s" else 0.06 * width)), float(hi + 0.06 * width)


def draw_contour(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zorder: int,
) -> dict[str, float]:
    hist, xedges, yedges = np.histogram2d(
        np.asarray(x, dtype="f8"),
        np.asarray(y, dtype="f8"),
        bins=(170, 160),
        range=(xlim, ylim),
        density=False,
    )
    hist = gaussian_filter(hist.astype("f8"), sigma=2.5, mode="nearest")
    level95, level68 = density_levels(hist)
    top = float(np.max(hist)) * 1.001
    xcenter = 0.5 * (xedges[:-1] + xedges[1:])
    ycenter = 0.5 * (yedges[:-1] + yedges[1:])
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
    return {"level95": level95, "level68": level68, "density_max": top / 1.001}


def posterior_text(row: dict[str, Any]) -> str:
    median = float(row["q50"])
    low = median - float(row["q16"])
    high = float(row["q84"]) - median
    return rf"{median:.2f}_{{-{low:.2f}}}^{{+{high:.2f}}}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.output.with_suffix(".json")
    if args.output.exists() or audit_path.exists():
        raise FileExistsError(f"immutable contour output exists: {args.output} / {audit_path}")
    summary = json.loads(args.input_json.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise RuntimeError("input long-chain summary is not complete")
    scope = summary.get("scope", {})
    if scope.get("multipoles") != [0] or "ell=2" not in scope.get("explicitly_excluded", []):
        raise RuntimeError(f"input is not the frozen ell=0-only comparison: {scope}")
    with np.load(args.input_npz, allow_pickle=False) as data:
        forbidden = [name for name in data.files if "xi2" in name.lower() or "pk2" in name.lower()]
        if forbidden:
            raise RuntimeError(f"ell=2 arrays found: {forbidden}")
        pk_chain = np.asarray(data["pk0_chain_by_step"], dtype="f8")
        xi_chain = np.asarray(data["xi0_chain_by_step"], dtype="f8")
    if pk_chain.ndim != 3 or pk_chain.shape[2] != 4 or xi_chain.ndim != 3 or xi_chain.shape[2] != 3:
        raise RuntimeError(f"unexpected chain shapes: P0={pk_chain.shape}, xi0={xi_chain.shape}")
    pk = pk_chain.reshape(-1, 4)[:, :3]
    xi = xi_chain.reshape(-1, 3)
    ranges = {
        name: plot_range(pk[:, index], xi[:, index], parameter=name)
        for index, name in enumerate(PARAMETERS)
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
                axis.hist(pk[:, icol], bins=bins, density=True, histtype="step", lw=1.9, color=COLORS["pk0"])
                axis.hist(xi[:, icol], bins=bins, density=True, histtype="step", lw=1.9, color=COLORS["xi0"])
                axis.set_xlim(*ranges[xname])
                axis.set_yticks([])
                if xname == "fNL":
                    axis.axvline(0.0, color="0.5", lw=0.8, ls="--")
            else:
                key = f"{xname}_vs_{yname}"
                contour_audit[key] = {
                    "xi0": draw_contour(
                        axis,
                        xi[:, icol],
                        xi[:, irow],
                        color=COLORS["xi0"],
                        xlim=ranges[xname],
                        ylim=ranges[yname],
                        zorder=1,
                    ),
                    "pk0": draw_contour(
                        axis,
                        pk[:, icol],
                        pk[:, irow],
                        color=COLORS["pk0"],
                        xlim=ranges[xname],
                        ylim=ranges[yname],
                        zorder=3,
                    ),
                }
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

    pk_fnl = summary["pk0"]["mcmc"]["posterior"]["fNL"]
    xi_fnl = summary["xi0"]["posterior"]["fNL"]
    handles = [
        plt.Line2D([], [], color=COLORS["pk0"], lw=2.0),
        plt.Line2D([], [], color=COLORS["xi0"], lw=2.0),
    ]
    labels = [
        rf"$P_0(k):\ f_{{\rm NL}}={posterior_text(pk_fnl)}$",
        rf"$\xi_0(s):\ f_{{\rm NL}}={posterior_text(xi_fnl)}$",
    ]
    figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.97, 0.94), frameon=False, fontsize=13.0)
    figure.text(0.69, 0.73, "68% and 95% contours", ha="center", va="center", fontsize=11.5, color="0.35")
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.96, wspace=0.07, hspace=0.07)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp.pdf")
    figure.savefig(temporary, format="pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    temporary.replace(args.output)
    if args.output.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError("contour output does not have a PDF header")
    audit = {
        "task": "task43_plot_rsd_rawbox_pk0_vs_xi0_contours",
        "status": "pass",
        "scope": "rawbox P0(kmax=0.10) vs xi0(smin=50), common parameters, ell=0 only",
        "credible_contours": [0.68, 0.95],
        "parameter_order": list(PARAMETERS),
        "chain_shapes": {"pk0": list(pk_chain.shape), "xi0": list(xi_chain.shape)},
        "sample_counts": {"pk0": int(pk.shape[0]), "xi0": int(xi.shape[0])},
        "plot_ranges": {name: list(value) for name, value in ranges.items()},
        "contour_density_thresholds": contour_audit,
        "ell2_arrays": [],
        "input_npz": str(args.input_npz),
        "input_npz_sha256": sha256_file(args.input_npz),
        "input_json": str(args.input_json),
        "input_json_sha256": sha256_file(args.input_json),
        "output_pdf": str(args.output),
        "output_pdf_sha256": sha256_file(args.output),
    }
    atomic_write_json(audit_path, audit)
    print(json.dumps({"status": "pass", "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
