#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Task43 formal-GIC W2 convergence for the L2000 setup."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
SUMMARY_PATH = PROJECT_DIR / "outputs/task43_outputs/summary/task43_formal_gic_w2_convergence_L2000_seed20260703.json"
OUT_PDF = PROJECT_DIR / "plots/task43/task43_formal_gic_w2_convergence_L2000_seed20260703.pdf"


def main() -> None:
    payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    fit_rows = payload["fit_rows"]
    window_rows = payload["window_rows"]
    smins = sorted({int(row["smin"]) for row in fit_rows})
    nsubs = sorted({int(row["nsub"]) for row in fit_rows})

    colors = {50: "#2563eb", 60: "#059669", 80: "#dc2626"}
    markers = {50: "o", 60: "s", 80: "^"}

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=False)
    ax = axes[0]
    for smin in smins:
        rows = sorted([row for row in fit_rows if int(row["smin"]) == smin], key=lambda item: int(item["nsub"]))
        x = np.asarray([row["nsub"] for row in rows], dtype="f8")
        y = np.asarray([row["q50"] for row in rows], dtype="f8")
        yerr_low = y - np.asarray([row["q16"] for row in rows], dtype="f8")
        yerr_high = np.asarray([row["q84"] for row in rows], dtype="f8") - y
        base = rows[0]
        ax.axhline(base["baseline_q50"], color=colors[smin], alpha=0.22, lw=1.2)
        ax.fill_between(
            [min(nsubs) * 0.8, max(nsubs) * 1.25],
            base["baseline_q16"],
            base["baseline_q84"],
            color=colors[smin],
            alpha=0.08,
            linewidth=0,
        )
        ax.errorbar(
            x,
            y,
            yerr=[yerr_low, yerr_high],
            color=colors[smin],
            marker=markers[smin],
            lw=1.4,
            capsize=3,
            label=f"smin={smin}",
        )
    ax.axhline(0.0, color="0.3", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_ylabel(r"$f_{\rm NL}$ posterior")
    ax.set_title("Task43 formal-GIC W2 convergence")
    ax.legend(frameon=False, ncol=3)
    ax.grid(True, which="both", alpha=0.18)

    ax = axes[1]
    rows = sorted(window_rows, key=lambda item: int(item["nsub"]))
    x = np.asarray([row["nsub"] for row in rows], dtype="f8")
    rms_abs = np.asarray([row["w2_vs_200k_rms_abs"] for row in rows], dtype="f8")
    max_abs = np.asarray([row["w2_vs_200k_max_abs"] for row in rows], dtype="f8")
    ax.plot(x, rms_abs, marker="o", color="#7c3aed", label="RMS |W2-W2_200k|")
    ax.plot(x, max_abs, marker="s", color="#ea580c", label="max |W2-W2_200k|")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0e-8)
    ax.set_xlabel("W2 random subsample size")
    ax.set_ylabel("absolute W2 difference")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(frameon=False)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
