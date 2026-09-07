#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Meeting version of Task45 UltraNest constraints for Quijote LCp50 only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
SUMMARY_ROOT = PROJECT_ROOT / "outputs" / "task4_outputs" / "quijote_ultranest"
OUT_DIR = PROJECT_ROOT / "plots" / "6.18meeting"
OUT_PDF = OUT_DIR / "task45_quijote_fnl50_ultranest_constraints.pdf"

TAG = "LCp50"
DISPLAY_TAG = r"$f_{\mathrm{NL}}=50$"
PARAM_NAMES = ["fnl_loc", "b1", "sigmas"]
PARAM_LABELS = {
    "fnl_loc": r"$f_{\mathrm{NL}}^{\mathrm{loc}}$",
    "b1": r"$b_1$",
    "sigmas": r"$\sigma_s\,[h^{-1}{\rm Mpc}]$",
}
CASES = ["pk_binavg", "xi_r50_350", "xi_r60_350", "xi_r80_350", "xi_r100_350"]
CASE_LABELS = {
    "pk_binavg": r"P(k), kmax=0.08",
    "xi_r50_350": "2PCF 55-345",
    "xi_r60_350": "2PCF 65-345",
    "xi_r80_350": "2PCF 85-345",
    "xi_r100_350": "2PCF 105-345",
}


def load_summaries() -> list[dict[str, object]]:
    """Load LCp50 Task45 summary JSON files."""
    rows = []
    for case in CASES:
        path = SUMMARY_ROOT / f"task45_{TAG}_{case}_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(payload)
    return rows


def padded_limits(values: list[float], lows: list[float], highs: list[float]) -> tuple[float, float]:
    """Return padded axis limits from asymmetric intervals."""
    xmin = float(np.nanmin(np.asarray(values) - np.asarray(lows)))
    xmax = float(np.nanmax(np.asarray(values) + np.asarray(highs)))
    span = xmax - xmin
    pad = 0.08 * span if span > 0.0 else 1.0
    return xmin - pad, xmax + pad


def plot() -> None:
    """Create the meeting PDF."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_summaries()
    y = np.arange(len(rows))[::-1]

    scale = 1.5
    plt.rcParams.update(
        {
            "font.size": 11 * scale,
            "axes.titlesize": 15 * scale,
            "axes.labelsize": 12 * scale,
            "xtick.labelsize": 10 * scale,
            "ytick.labelsize": 9.5 * scale,
            "legend.fontsize": 11 * scale,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 6.4), sharey=True, gridspec_kw={"wspace": 0.26})
    fig.patch.set_facecolor("white")
    color = "#bf5b2f"
    markers = {"pk_binavg": "s"}

    for ax, pname in zip(axes, PARAM_NAMES):
        ax.set_facecolor("white")
        medians: list[float] = []
        err_lows: list[float] = []
        err_highs: list[float] = []

        for i, row in enumerate(rows):
            yi = y[i]
            case = str(row["case"])
            item = row["parameters"][pname]
            median = float(item["median"])
            err_low = float(item["err_low_percival"])
            err_high = float(item["err_high_percival"])
            medians.append(median)
            err_lows.append(err_low)
            err_highs.append(err_high)

            if i % 2 == 0:
                ax.axhspan(yi - 0.42, yi + 0.42, color="#ececea", alpha=0.45, zorder=0)
            ax.errorbar(
                median,
                yi,
                xerr=np.array([[err_low], [err_high]]),
                fmt=markers.get(case, "o"),
                ms=8.5,
                mfc="white",
                mec=color,
                mew=2.2,
                ecolor=color,
                elinewidth=2.2,
                capsize=4.8,
                capthick=2.0,
                alpha=0.96,
                zorder=3,
            )

        ax.set_title(PARAM_LABELS[pname], pad=12)
        ax.set_xlabel("")
        ax.set_xlim(*padded_limits(medians, err_lows, err_highs))
        ax.set_ylim(-0.75, len(rows) - 0.25)
        ax.grid(axis="x", color="#c9c9c9", lw=0.9, alpha=0.65)
        ax.tick_params(axis="both", length=0)

    labels = [f"{DISPLAY_TAG}  {CASE_LABELS[str(row['case'])]}" for row in rows]
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    handles = [
        axes[0].plot([], [], "s", ms=9, mfc="white", mec=color, mew=2.2, label="P(k)")[0],
        axes[0].plot([], [], "o", ms=9, mfc="white", mec=color, mew=2.2, label="2PCF")[0],
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.945), ncol=2, frameon=False)
    fig.suptitle(
        r"Quijote $f_{\mathrm{NL}}=50$: UltraNest posterior constraints",
        x=0.55,
        y=0.995,
        fontsize=18 * scale,
        fontweight="bold",
    )
    fig.text(
        0.55,
        0.045,
        r"P(k): kmax=0.08 h/Mpc.  2PCF: FullDiscrete, rmax = 350 Mpc/h.  $s_{n,0}$ fixed to 0.",
        ha="center",
        va="center",
        fontsize=10.5 * scale,
        color="#4d4d4d",
    )
    fig.subplots_adjust(left=0.30, right=0.985, top=0.82, bottom=0.18)
    fig.savefig(OUT_PDF)
    print(OUT_PDF)


if __name__ == "__main__":
    plot()
