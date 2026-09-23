#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnosis figure for the P2 fix paths (kmin cut vs radial-RIC).

One PDF page: P0 and P2 mean data with the baseline P0+P2 model, the
boxsafe-geometry RIC model, and the combined (RIC + kmin2=0.015) model;
bottom row shows the per-bin residual/sigma of the baseline versus the combo.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task43_rsd_boxsafe_p02_increment import COV_NPZ, PAYLOAD_NPZ
from task43_rsd_common import OUTPUT_ROOT, PLOT_ROOT, atomic_write_json, sha256_file


BOXSAFE_ROOT = OUTPUT_ROOT / "lightcone_boxsafe_zobs0p4_0p8"
P02 = BOXSAFE_ROOT / "p02_increment"
PDF_PATH = PLOT_ROOT / "task43_rsd_boxsafe_p02_fix_diagnosis.pdf"
NPHASE = 25
COLORS = {"baseline": "#C44E52", "ric": "#4C72B0", "combo": "#55A868"}


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        return np.asarray(d["data"], dtype="f8"), np.asarray(d["prediction_map"], dtype="f8")


def main() -> None:
    if PDF_PATH.exists() or PDF_PATH.with_suffix(".json").exists():
        raise FileExistsError(f"immutable figure exists: {PDF_PATH}")
    with np.load(PAYLOAD_NPZ, allow_pickle=False) as d:
        k_obs = np.asarray(d["k_obs"], dtype="f8")
        fit_indices = np.asarray(d["fit_bin_indices"], dtype="i8")
    with np.load(COV_NPZ, allow_pickle=False) as d:
        cov_full = np.asarray(d["covariance_full"], dtype="f8")
    ids = np.concatenate([fit_indices, 150 + fit_indices])
    cov30 = cov_full[np.ix_(ids, ids)]

    data_base, pred_base = load(P02 / "fits" / "p02" / "samples.npz")
    _, pred_ric = load(P02 / "ricprobe_boxsafe" / "fits" / "p02_ric" / "samples.npz")
    _, pred_combo = load(P02 / "combo_kmin0p015" / "fits" / "p02_ric_kmin2" / "samples.npz")
    keep = k_obs >= 0.015 - 1.0e-12
    keep_idx = np.flatnonzero(keep)
    # combo vector = [P0 15 bins, P2 kept bins]; its prediction segment covers kept P2 bins only
    pred_combo_full = np.full(30, np.nan)
    pred_combo_full[:15] = pred_combo[:15]
    pred_combo_full[15 + keep_idx] = pred_combo[15:]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "font.size": 11.5,
            "axes.labelsize": 13.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), gridspec_kw={"height_ratios": [2.1, 1.0]})
    for ipole, name in enumerate(("P_0", "P_2")):
        top, bottom = axes[0, ipole], axes[1, ipole]
        seg = slice(15 * ipole, 15 * (ipole + 1))
        sig = np.sqrt(np.diag(cov30[seg, seg]) / NPHASE)
        top.errorbar(k_obs, data_base[seg], yerr=sig, fmt="o", ms=4, color="0.25", zorder=3, label=f"x25 mean ${name}$")
        top.plot(k_obs, pred_base[seg], color=COLORS["baseline"], lw=1.6, label="baseline $P_0{+}P_2$")
        top.plot(k_obs, pred_ric[seg], color=COLORS["ric"], lw=1.6, ls="--", label="+ boxsafe radial-RIC")
        top.plot(k_obs, pred_combo[seg] if ipole == 0 else pred_combo_full[seg], color=COLORS["combo"], lw=1.6, ls=":", label="+ RIC and $k_{\\min}^{P_2}{=}0.015$")
        top.set(xscale="log", ylabel=rf"${name}(k)\ [(h^{{-1}}{{\rm Mpc}})^3]$" if ipole == 0 else rf"${name}(k)$")
        top.legend(frameon=False, fontsize=8.5)
        bottom.axhline(0.0, color="0.5", lw=0.8)
        bottom.plot(k_obs, (data_base[seg] - pred_base[seg]) / sig, "o-", ms=3.5, lw=0.8, color=COLORS["baseline"], label="baseline")
        combo_curve = (data_base[seg] - pred_combo_full[seg]) / sig
        bottom.plot(k_obs if ipole == 0 else k_obs[keep], combo_curve if ipole == 0 else combo_curve[keep], "o-", ms=3.5, lw=0.9, color=COLORS["combo"], label="combo (kept bins)")
        bottom.set(xlabel=r"$k\ [h\,{\rm Mpc}^{-1}]$", ylabel=r"residual / $\sigma_{\rm mean}$", xscale="log")
        bottom.legend(frameon=False, fontsize=8.5)
        if ipole == 1:
            top.axvline(0.015, color="0.4", lw=0.8, ls=":")
            bottom.axvline(0.015, color="0.4", lw=0.8, ls=":")
    axes[0, 0].set_title(r"$P_0$: RIC $\sim$1.3\% correction, absorbed by sn0", fontsize=10.5)
    axes[0, 1].set_title(r"$P_2$: low-$k$ residual band unchanged by RIC; removed by $k_{\min}$ cut", fontsize=10.5)
    figure.suptitle("boxsafe RSD lightcone: fix-1 (radial-RIC) vs fix-3 (per-pole $k_{\\min}$) attribution", fontsize=12.0)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PDF_PATH, format="pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)

    audit = {
        "task": "task43_plot_p02_fix_diagnosis",
        "status": "pass",
        "output_pdf": str(PDF_PATH),
        "output_pdf_sha256": sha256_file(PDF_PATH),
        "inputs": {
            "baseline": str(P02 / "fits" / "p02" / "samples.npz"),
            "ric_boxsafe": str(P02 / "ricprobe_boxsafe" / "fits" / "p02_ric" / "samples.npz"),
            "combo": str(P02 / "combo_kmin0p015" / "fits" / "p02_ric_kmin2" / "samples.npz"),
        },
    }
    atomic_write_json(PDF_PATH.with_suffix(".json"), audit)
    print(json.dumps({"status": "pass", "output": str(PDF_PATH)}))


if __name__ == "__main__":
    main()
