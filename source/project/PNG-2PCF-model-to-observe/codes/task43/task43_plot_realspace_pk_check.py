#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-space rawbox P0(k) best-fit figure WITH error bars and residuals."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import RAWBOX_FIT_EDGES, load_pk_x25
from task43_rsd_common import OUTPUT_ROOT, PLOT_ROOT, atomic_write_json, sha256_file
from task43_rsd_model import DELTA_C, FullDiscreteRSDModel, build_cache


OUT_DIR = OUTPUT_ROOT / "rawbox" / "realspace_check"
PDF_PATH = PLOT_ROOT / "task43_rsd_rawbox_realspace_pk0_check.pdf"
NPHASE = 25


def main() -> None:
    if PDF_PATH.exists() or PDF_PATH.with_suffix(".json").exists():
        raise FileExistsError(f"immutable figure exists: {PDF_PATH}")
    audit = json.loads((OUT_DIR / "audits" / "task43_rsd_rawbox_realspace_pk_check.json").read_text(encoding="utf-8"))
    pk = load_pk_x25(RAWBOX_FIT_EDGES)
    k_obs = np.asarray(pk["k"], dtype="f8")
    data_v = np.mean(np.asarray(pk["pk0_real"], dtype="f8"), axis=0)
    nbar = float(np.mean(np.asarray(pk["nbar"], dtype="f8")))

    # model curve on the mode set (same as the check script)
    exact = FullDiscreteRSDModel(
        build_cache(zeff=0.725, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000"), nmu=64
    )
    kfund = 2.0 * np.pi / 2000.0
    nmax = int(np.ceil(float(np.max(RAWBOX_FIT_EDGES)) / kfund))
    integers = np.arange(-nmax, nmax + 1, dtype="i4")
    nx, ny, nz = np.meshgrid(integers, integers, integers, indexing="ij")
    n2 = (nx.astype("f8") ** 2 + ny.astype("f8") ** 2 + nz.astype("f8") ** 2).ravel()
    kval = kfund * np.sqrt(n2)
    bin_id = np.full(kval.size, -1, dtype="i4")
    for ibin, (lo, hi) in enumerate(RAWBOX_FIT_EDGES):
        bin_id[(kval >= lo) & (kval < hi)] = ibin
    keep = bin_id >= 0
    b_id = bin_id[keep]
    k_modes = kval[keep]
    counts = np.bincount(b_id, minlength=RAWBOX_FIT_EDGES.shape[0]).astype("f8")
    pk_dd = np.interp(np.log(k_modes), np.log(np.asarray(exact.k_eff, dtype="f8")), np.asarray(exact.pk_dd, dtype="f8"))
    alpha = np.interp(np.log(k_modes), np.log(np.asarray(exact.k_eff, dtype="f8")), np.asarray(exact.alpha, dtype="f8"))

    fnl, b1 = audit["map"]["fNL"], audit["map"]["b1"]
    model = np.bincount(b_id, weights=pk_dd * (b1 + fnl * 2.0 * DELTA_C * (b1 - 1.0) * alpha) ** 2, minlength=counts.size) / counts
    total2 = (b1**2 * pk_dd + 1.0 / nbar) ** 2
    sig = np.sqrt(2.0 * np.bincount(b_id, weights=total2, minlength=counts.size) / counts**2) / np.sqrt(NPHASE)
    res = (data_v - model) / sig

    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42, "font.size": 11.5})
    figure, axes = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.0]})
    axes[0].errorbar(k_obs, data_v, yerr=sig, fmt="o", ms=5, color="0.25", capsize=2.5, lw=1.2, label="x25 mean $P_0(k)$ real space")
    axes[0].plot(k_obs, model, color="#C44E52", lw=1.8, label=rf"linear template: $f_{{\rm NL}}={fnl:.1f},\ b_1={b1:.2f}$")
    axes[0].set(xscale="log", yscale="log", ylabel=r"$P_0(k)\ [(h^{-1}{\rm Mpc})^3]$")
    axes[0].legend(frameon=False, fontsize=10)
    axes[0].set_title("rawbox real space P0, frozen 16 bins (k=0.003..0.095), exact parent-mode average", fontsize=11.0)

    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].errorbar(k_obs, res, yerr=1.0, fmt="o", ms=4, color="#C44E52", capsize=2.5, lw=1.0)
    axes[1].set(xscale="log", xlabel=r"$k\ [h\,{\rm Mpc}^{-1}]$", ylabel=r"residual / $\sigma_{\rm mean}$")
    axes[1].set_title(rf"$\chi^2_{{\rm mean}}={audit['chi2_mean']:.1f}/{audit['dof']}$, PTE $={audit['pte_mean']:.1e}$", fontsize=10.5)

    figure.tight_layout()
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PDF_PATH, format="pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)

    out = {
        "task": "task43_plot_realspace_pk_check",
        "status": "pass",
        "audit": {"path": str(OUT_DIR / "audits" / "task43_rsd_rawbox_realspace_pk_check.json")},
        "output_pdf": str(PDF_PATH),
        "output_pdf_sha256": sha256_file(PDF_PATH),
    }
    atomic_write_json(PDF_PATH.with_suffix(".json"), out)
    print(json.dumps({"status": "pass", "output": str(PDF_PATH)}))


if __name__ == "__main__":
    main()
