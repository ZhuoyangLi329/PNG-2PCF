#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-space rawbox xi0 closure figure: data vs minimal model + residuals."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task43_fit_rsd_rawbox_x25 import S_EDGES, load_x25
from task43_rsd_common import OUTPUT_ROOT, PLOT_ROOT, atomic_write_json, sha256_file
from task43_rsd_model import DELTA_C, FullDiscreteRSDModel, build_cache


OUT_DIR = OUTPUT_ROOT / "rawbox" / "realspace_check"
PDF_PATH = PLOT_ROOT / "task43_rsd_rawbox_realspace_xi0_check.pdf"
NPHASE = 25


def main() -> None:
    if PDF_PATH.exists() or PDF_PATH.with_suffix(".json").exists():
        raise FileExistsError(f"immutable figure exists: {PDF_PATH}")
    audit = json.loads((OUT_DIR / "audits" / "task43_rsd_rawbox_realspace_check.json").read_text(encoding="utf-8"))
    xi, metadata_rows, _ = load_x25()
    nbar = float(np.mean([row["nbar_h3_mpc3_from_npz"] for row in metadata_rows]))
    xi0_mean = np.mean(np.asarray(xi["xi0_real"], dtype="f8"), axis=0)
    exact = FullDiscreteRSDModel(
        build_cache(zeff=0.725, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000"), nmu=64
    )
    kernel0 = np.asarray(exact.kernels[0], dtype="f8")
    g = np.asarray(exact.g_nz, dtype="f8")
    pk_dd = np.asarray(exact.pk_dd, dtype="f8")
    alpha = np.asarray(exact.alpha, dtype="f8")
    proj = (g[:, None] * pk_dd[:, None]) * kernel0 / float(exact.volume)

    def model_curve(theta):
        fnl, b1 = theta
        amp = b1 + fnl * 2.0 * DELTA_C * (b1 - 1.0) * alpha
        return (amp**2) @ proj

    def sigma_curve(b1):
        total2 = (b1**2 * pk_dd + 1.0 / nbar) ** 2
        cov_diag = np.einsum("q,qi,qi->i", g * total2, kernel0, kernel0) / float(exact.volume) ** 2
        return np.sqrt(np.maximum(cov_diag, 0.0))

    theta50 = (audit["results"]["smin50"]["fNL"], audit["results"]["smin50"]["b1"])
    theta120 = (audit["results"]["smin120"]["fNL"], audit["results"]["smin120"]["b1"])
    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    sig50 = sigma_curve(theta50[1]) / np.sqrt(NPHASE)
    m50 = model_curve(theta50)

    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42, "font.size": 11.5})
    figure, axes = plt.subplots(2, 1, figsize=(8.6, 6.8), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.0]})
    axes[0].errorbar(centers, centers**2 * xi0_mean, yerr=centers**2 * sig50, fmt="o", ms=4, color="0.25", label="x25 mean $\\xi_0$ (real space)")
    axes[0].plot(centers, centers**2 * m50, color="#C44E52", lw=1.7, label=f"smin=50 fit: $f_{{\\rm NL}}={theta50[0]:.1f}, b_1={theta50[1]:.2f}$")
    axes[0].plot(centers, centers**2 * model_curve(theta120), color="#4C72B0", lw=1.5, ls="--", label=f"smin=120 fit: $f_{{\\rm NL}}={theta120[0]:.1f}, b_1={theta120[1]:.2f}$")
    axes[0].axvline(120.0, color="0.4", lw=0.9, ls=":")
    axes[0].set_ylabel(r"$s^2\xi_0(s)$")
    axes[0].legend(frameon=False, fontsize=9.5)
    axes[0].set_title("rawbox real space, no FoG / Kaiser / quadrupole: linear template + j0 shell kernels", fontsize=11.0)

    res50 = (xi0_mean - m50) / sig50
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].plot(centers, res50, "o-", ms=3.5, lw=0.9, color="#C44E52", label="smin=50 fit residuals")
    r120 = audit["results"]["smin120"]["residual_over_sigma_single"]
    axes[1].plot(centers[centers >= 120], np.asarray(r120) * np.sqrt(NPHASE), "o-", ms=3.5, lw=0.9, color="#4C72B0", label="smin=120 fit residuals")
    axes[1].axvline(120.0, color="0.4", lw=0.9, ls=":")
    axes[1].set(xlabel=r"$s\ [h^{-1}{\rm Mpc}]$", ylabel=r"residual / $\sigma_{\rm mean}$", xlim=(45, 355))
    axes[1].legend(frameon=False, fontsize=9.5)
    axes[1].set_title(rf"smin=50: $\chi^2={audit['results']['smin50']['chi2_mean']:.0f}/28$, PTE $=1.3\times10^{{-121}}$;   smin=120: $\chi^2=24.1/21$, PTE $=0.29$", fontsize=10.5)

    figure.tight_layout()
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PDF_PATH, format="pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)

    out = {
        "task": "task43_plot_realspace_check",
        "status": "pass",
        "audit": {"path": str(OUT_DIR / "audits" / "task43_rsd_rawbox_realspace_check.json")},
        "output_pdf": str(PDF_PATH),
        "output_pdf_sha256": sha256_file(PDF_PATH),
    }
    atomic_write_json(PDF_PATH.with_suffix(".json"), out)
    print(json.dumps({"status": "pass", "output": str(PDF_PATH)}))


if __name__ == "__main__":
    main()
