#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Best-fit vs measurement for the xi per-pole smin experiment.

One page, 2x2: left column xi0, right column xi2.  Top: s^2*xi data (x25
mean) with the xi0-only control and the per-pole joint fits; mask lines at
s=50/80/120.  Bottom: per-bin residual/sigma of the s>=80 per-pole joint fit
(the variant that shows the pole tension).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task43_rsd_common import OUTPUT_ROOT, PLOT_ROOT, atomic_write_json, sha256_file


BOXSAFE_ROOT = OUTPUT_ROOT / "lightcone_boxsafe_zobs0p4_0p8"
ELL2_DIR = BOXSAFE_ROOT / "ell2_increment"
SUMMARY_NPZ = ELL2_DIR / "task43_rsd_boxsafe_x25_mean_xi02_s30_350_ds10.npz"
ELL02_COV = (
    BOXSAFE_ROOT / "covariance"
    / "task43_rsd_boxsafe_ph000_jaxpower_rrdeconv_ell02_rsdpoles024_win02468_mesh64_p1_b1cov2p50"
    "_sigmas7p50_nran100k_ndata50k_seed20260817_rrnran300k_rrseed430420_kbox_fkpP010000_s30_350_ds10.npz"
)
PDF_PATH = PLOT_ROOT / "task43_rsd_boxsafe_xi02_perpole_bestfits.pdf"
NPHASE = 25
COLORS = {"control": "#C44E52", "s80": "#4C72B0", "s120": "#55A868", "common": "0.55"}


def load_variant(name: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(ELL2_DIR / "perpole_smin" / "fits" / name / "samples.npz", allow_pickle=False) as d:
        return np.asarray(d["ids"], dtype="i8"), np.asarray(d["prediction_map"], dtype="f8")


def main() -> None:
    if PDF_PATH.exists() or PDF_PATH.with_suffix(".json").exists():
        raise FileExistsError(f"immutable figure exists: {PDF_PATH}")
    audit_path = ELL2_DIR / "perpole_smin" / "audits" / "task43_rsd_boxsafe_xi02_perpole_smin_summary.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    with np.load(SUMMARY_NPZ, allow_pickle=False) as d:
        s = np.asarray(d["s"], dtype="f8")
        mean_xi = np.asarray(d["xi_multipoles_mean"], dtype="f8")
    with np.load(ELL02_COV, allow_pickle=False) as d:
        cov64 = np.asarray(d["covariance_single_realization"], dtype="f8")
    ns = s.size
    sig = {
        ell: np.sqrt(np.diag(cov64[iell * ns:iell * ns + ns, iell * ns:iell * ns + ns]) / NPHASE)
        for iell, ell in enumerate((0, 2))
    }
    with np.load(ELL2_DIR / "fits" / "smin050_ell02" / "samples.npz", allow_pickle=False) as d:
        common_pred = np.asarray(d["model_map"], dtype="f8")
    mask50 = s >= 50.0
    mask80 = s >= 80.0
    mask120 = s >= 120.0
    n50 = int(np.count_nonzero(mask50))

    ids_c, pred_c = load_variant("ell0_s50_control")
    ids_80, pred_80 = load_variant("ell0s50_ell2s80")
    ids_120, pred_120 = load_variant("ell0s50_ell2s120")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "font.size": 11.0,
            "axes.labelsize": 12.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), sharex="col", gridspec_kw={"height_ratios": [2.1, 1.0]})
    # xi0 panel
    top, bottom = axes[0, 0], axes[1, 0]
    top.errorbar(s, s**2 * mean_xi[0], yerr=s**2 * sig[0], fmt="o", ms=4, color="0.25", zorder=2, label="x25 mean $\\xi_0$")
    top.plot(s[mask50], s[mask50] ** 2 * pred_c, color=COLORS["control"], lw=1.6, ls="--", label="$\\xi_0$-only fit")
    top.plot(s[mask50], s[mask50] ** 2 * pred_80[:n50], color=COLORS["s80"], lw=1.6, label="$+\\xi_2^{s\\geq80}$ fit")
    top.plot(s[mask50], s[mask50] ** 2 * pred_120[:n50], color=COLORS["s120"], lw=1.5, ls=":", label="$+\\xi_2^{s\\geq120}$ fit")
    top.axvline(50.0, color="0.4", lw=0.8, ls=":")
    top.set_ylabel(r"$s^2\xi_0(s)$")
    top.legend(frameon=False, fontsize=8.5)
    pte0 = audit["results"]["ell0s50_ell2s80"]["per_pole_mean_goodness"]["ell0"]["pte"]
    top.set_title(rf"$\xi_0$ (mask $s\geq50$): PTE $= {pte0:.1e}$ in the $+{{\xi_2^{{s\geq80}}}}$ fit", fontsize=10.0)
    bottom.axhline(0.0, color="0.5", lw=0.8)
    bottom.plot(s[mask50], (mean_xi[0][mask50] - pred_80[:n50]) / sig[0][mask50], "o-", ms=3.5, lw=0.9, color=COLORS["s80"])
    bottom.set(xlabel=r"$s\ [h^{-1}{\rm Mpc}]$", ylabel=r"residual / $\sigma_{\rm mean}$", xlim=(25, 355))

    # xi2 panel
    top, bottom = axes[0, 1], axes[1, 1]
    top.errorbar(s, s**2 * mean_xi[1], yerr=s**2 * sig[2], fmt="o", ms=4, color="0.25", zorder=2, label="x25 mean $\\xi_2$")
    m80 = int(np.count_nonzero(mask80))
    top.plot(s[mask50], s[mask50] ** 2 * common_pred[n50:], color=COLORS["common"], lw=1.4, ls="--", label="common $s\\geq50$ fit")
    top.plot(s[mask80], s[mask80] ** 2 * pred_80[n50:], color=COLORS["s80"], lw=1.7, label="$\\xi_2^{s\\geq80}$ fit")
    top.plot(s[mask120], s[mask120] ** 2 * pred_120[n50:], color=COLORS["s120"], lw=1.6, ls=":", label="$\\xi_2^{s\\geq120}$ fit")
    for sm in (50.0, 80.0, 120.0):
        top.axvline(sm, color="0.4", lw=0.8, ls=":")
    top.set_ylabel(r"$s^2\xi_2(s)$")
    top.legend(frameon=False, fontsize=8.5)
    pte2 = audit["results"]["ell0s50_ell2s80"]["per_pole_mean_goodness"]["ell2"]["pte"]
    top.set_title(rf"$\xi_2$: PTE $= {pte2:.1e}$ even at $s\geq80$ (masks dotted)", fontsize=10.0)
    bottom.axhline(0.0, color="0.5", lw=0.8)
    bottom.plot(s[mask80], (mean_xi[1][mask80] - pred_80[n50:]) / sig[2][mask80], "o-", ms=3.5, lw=0.9, color=COLORS["s80"])
    bottom.set(xlabel=r"$s\ [h^{-1}{\rm Mpc}]$", ylabel=r"residual / $\sigma_{\rm mean}$", xlim=(25, 355))

    figure.suptitle(r"per-pole $s_{\min}$: $\xi_0$ keeps $s\geq50$, $\xi_2$ enters above 80/120; residuals of the $s\geq80$ joint fit", fontsize=11.5)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(PDF_PATH, format="pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)

    out_audit = {
        "task": "task43_plot_xi02_perpole_bestfits",
        "status": "pass",
        "inputs": {
            "perpole_audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
            "summary": {"path": str(SUMMARY_NPZ), "sha256": sha256_file(SUMMARY_NPZ)},
        },
        "output_pdf": str(PDF_PATH),
        "output_pdf_sha256": sha256_file(PDF_PATH),
    }
    atomic_write_json(PDF_PATH.with_suffix(".json"), out_audit)
    print(json.dumps({"status": "pass", "output": str(PDF_PATH)}))


if __name__ == "__main__":
    main()
