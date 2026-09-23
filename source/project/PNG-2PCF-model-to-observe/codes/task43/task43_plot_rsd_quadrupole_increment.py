#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for the RSD quadrupole-increment experiments (xi0->xi0+xi2, P0->P0+P2).

Three PDF-only products:
1. xi02 contour triangle, one page per smin in {50, 80}: xi0-only (red) vs
   xi0+xi2 (blue), 68/95% contours;
2. p02 contour triangle: P0-only (grey) vs P0+P2 (blue);
3. best-fit comparison: mean data with Cmean/25 error bars, model curves over
   the fitted bins, and residual/sigma_mean panels showing where the
   quadrupole fits fail.
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
ELL2_DIR = BOXSAFE_ROOT / "ell2_increment"
ELL2_SUMMARY = ELL2_DIR / "task43_rsd_boxsafe_x25_mean_xi02_s30_350_ds10.npz"
ELL2_COV = (
    BOXSAFE_ROOT / "covariance"
    / "task43_rsd_boxsafe_ph000_jaxpower_rrdeconv_ell02_rsdpoles024_win02468_mesh64_p1_b1cov2p50"
    "_sigmas7p50_nran100k_ndata50k_seed20260817_rrnran300k_rrseed430420_kbox_fkpP010000_s30_350_ds10.npz"
)
ELL2_AUDIT = ELL2_DIR / "audits" / "task43_rsd_boxsafe_ell2_increment_summary.json"
P02_DIR = BOXSAFE_ROOT / "p02_increment"
P02_AUDIT = P02_DIR / "audits" / "task43_rsd_boxsafe_p02_increment_summary.json"
P02_COV = (
    Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/rsd_validation/lightcone/pk_covariance")
    / "boxsafe_zobs0p4_0p8_x25_fkpP010000_fnlcov0_b1cov2p50_sigmas7p50"
    / "task43_rsd_lightcone_pk0_cov_ph000_mesh128_kmax0p300_dk0p002.npz"
)
PAYLOAD_NPZ = (
    BOXSAFE_ROOT / "pk_summary"
    / "task43_rsd_boxsafe_lightcone_pk0_x25_kmin0p004291_kmax0p10_l0only_15bin.npz"
)

PARAMETERS = ("fNL", "b1", "sigma_s")
XI_COLORS = {"ell0": "#C44E52", "ell02": "#4C72B0"}
PK_COLORS = {"p0": "#2F2F2F", "p02": "#4C72B0"}
NPHASE = 25

XICON_PDF = PLOT_ROOT / "task43_rsd_boxsafe_xi02_increment_contours.pdf"
PCON_PDF = PLOT_ROOT / "task43_rsd_boxsafe_p02_increment_contours.pdf"
BESTFIT_PDF = PLOT_ROOT / "task43_rsd_boxsafe_quadrupole_bestfits.pdf"


def rc_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11.5,
            "axes.labelsize": 13.5,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def load_flat(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as d:
        key = "chain_flat" if "chain_flat" in d.files else "chain_by_step"
        chain = np.asarray(d[key], dtype="f8")
    return chain.reshape(-1, chain.shape[-1])[:, :3]


def triangle(axes, chains: dict[str, np.ndarray], colors: dict[str, str], ranges: dict[str, tuple[float, float]]) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for irow, yname in enumerate(PARAMETERS):
        for icol, xname in enumerate(PARAMETERS):
            axis = axes[irow, icol]
            if icol > irow:
                axis.set_axis_off()
                continue
            if icol == irow:
                bins = np.linspace(*ranges[xname], 90)
                for label, chain in chains.items():
                    axis.hist(chain[:, icol], bins=bins, density=True, histtype="step", lw=1.9, color=colors[label])
                axis.set_xlim(*ranges[xname])
                axis.set_yticks([])
                if xname == "fNL":
                    axis.axvline(0.0, color="0.5", lw=0.8, ls="--")
            else:
                key = f"{xname}_vs_{yname}"
                audit[key] = {}
                zorder = 1
                for label in reversed(list(chains)):
                    chain = chains[label]
                    audit[key][label] = draw_contour(
                        axis, chain[:, icol], chain[:, irow],
                        color=colors[label], xlim=ranges[xname], ylim=ranges[yname], zorder=zorder,
                    )
                    zorder += 2
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
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    for path in (XICON_PDF, PCON_PDF, BESTFIT_PDF):
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError(f"immutable figure exists: {path}")
    rc_style()
    ell2_audit = json.loads(ELL2_AUDIT.read_text(encoding="utf-8"))
    p02_audit = json.loads(P02_AUDIT.read_text(encoding="utf-8"))
    audits: dict[str, Any] = {}

    # ---- Figure 1: xi triangles, one page per smin ----
    from matplotlib.backends.backend_pdf import PdfPages

    xi_pages = {"smin050": 50.0, "smin080": 80.0}
    with PdfPages(XICON_PDF) as pdf:
        for key, smin in xi_pages.items():
            control = load_flat(ELL2_DIR / "fits" / f"{key}_ell0" / "samples.npz")
            increment = load_flat(ELL2_DIR / "fits" / f"{key}_ell02" / "samples.npz")
            chains = {"ell0": control, "ell02": increment}
            ranges = {
                name: plot_range(
                    np.concatenate([c[:, i] for c in chains.values()]),
                    np.concatenate([c[:, i] for c in chains.values()]),
                    parameter=name,
                )
                for i, name in enumerate(PARAMETERS)
            }
            figure, axes = plt.subplots(3, 3, figsize=(8.4, 8.0))
            audits[f"xi_triangle_{key}"] = triangle(axes, chains, XI_COLORS, ranges)
            row_c = ell2_audit["results"][f"{key}_ell0"]["posterior"]["fNL"]
            row_i = ell2_audit["results"][f"{key}_ell02"]["posterior"]["fNL"]
            handles = [plt.Line2D([], [], color=XI_COLORS[k], lw=2.0) for k in ("ell0", "ell02")]
            labels = [
                rf"$\xi_0\ (s_{{\min}}={smin:.0f}):\ f_{{\rm NL}}={posterior_text(row_c)}$",
                rf"$\xi_0+\xi_2:\ f_{{\rm NL}}={posterior_text(row_i)}$",
            ]
            figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.97, 0.95), frameon=False, fontsize=12.5)
            figure.text(0.70, 0.70, "68% and 95% contours\nformal-GIC, boxsafe RSD lightcone", ha="center", va="center", fontsize=10.5, color="0.35")
            figure.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.96, wspace=0.07, hspace=0.07)
            pdf.savefig(figure, bbox_inches="tight", pad_inches=0.06)
            plt.close(figure)

    # ---- Figure 2: P0 vs P0+P2 triangle ----
    control = load_flat(P02_DIR / "fits" / "p0_control" / "samples.npz")
    increment = load_flat(P02_DIR / "fits" / "p02" / "samples.npz")
    chains = {"p0": control, "p02": increment}
    ranges = {
        name: plot_range(
            np.concatenate([c[:, i] for c in chains.values()]),
            np.concatenate([c[:, i] for c in chains.values()]),
            parameter=name,
        )
        for i, name in enumerate(PARAMETERS)
    }
    figure, axes = plt.subplots(3, 3, figsize=(8.4, 8.0))
    audits["p_triangle"] = triangle(axes, chains, PK_COLORS, ranges)
    row_c = p02_audit["results"]["p0_control"]["posterior"]["fNL"]
    row_i = p02_audit["results"]["p02"]["posterior"]["fNL"]
    handles = [plt.Line2D([], [], color=PK_COLORS[k], lw=2.0) for k in ("p0", "p02")]
    labels = [
        rf"$P_0(k):\ f_{{\rm NL}}={posterior_text(row_c)}$",
        rf"$P_0+P_2:\ f_{{\rm NL}}={posterior_text(row_i)}$",
    ]
    figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.97, 0.95), frameon=False, fontsize=12.5)
    figure.text(0.70, 0.70, "68% and 95% contours\nboxsafe RSD lightcone", ha="center", va="center", fontsize=10.5, color="0.35")
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.96, wspace=0.07, hspace=0.07)
    figure.savefig(PCON_PDF, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)

    # ---- Figure 3: data vs best-fit, two pages (xi, P) ----
    with np.load(ELL2_SUMMARY, allow_pickle=False) as d:
        s_full = np.asarray(d["s"], dtype="f8")
        mean_xi = np.asarray(d["xi_multipoles_mean"], dtype="f8")
    with np.load(ELL2_COV, allow_pickle=False) as d:
        cov64 = np.asarray(d["covariance_single_realization"], dtype="f8")
        s_cov = np.asarray(d["s"], dtype="f8")
    ns = s_full.size

    def xi_model_map(key: str) -> tuple[np.ndarray, np.ndarray]:
        with np.load(ELL2_DIR / "fits" / f"{key}_ell0" / "samples.npz", allow_pickle=False) as d:
            map0 = np.asarray(d["model_map"], dtype="f8")
        with np.load(ELL2_DIR / "fits" / f"{key}_ell02" / "samples.npz", allow_pickle=False) as d:
            map02 = np.asarray(d["model_map"], dtype="f8")
        return map0, map02

    with PdfPages(BESTFIT_PDF) as pdf:
        # xi page at smin=50 (diagnostic where the failure is starkest)
        for key, smin in (("smin050", 50.0), ("smin080", 80.0)):
            mask = s_full >= smin
            nfit = int(np.count_nonzero(mask))
            map0, map02 = xi_model_map(key)
            figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), sharex="col", gridspec_kw={"height_ratios": [2.1, 1.0]})
            sig = {ell: np.sqrt(np.diag(cov64[iell * ns:iell * ns + ns, iell * ns:iell * ns + ns]) / NPHASE) for iell, ell in enumerate((0, 2))}
            for iell, ell in enumerate((0, 2)):
                top, bottom = axes[0, iell], axes[1, iell]
                top.errorbar(s_full, s_full**2 * mean_xi[iell], yerr=s_full**2 * sig[ell], fmt="o", ms=4, color="0.25", zorder=2, label=f"x25 mean $\\xi_{{{ell}}}$")
                if iell == 0:
                    top.plot(s_full[mask], s_full[mask] ** 2 * map0[0:nfit], color=XI_COLORS["ell0"], lw=1.7, ls="--", label="$\\xi_0$ fit")
                top.plot(s_full[mask], s_full[mask] ** 2 * map02[iell * nfit:(iell + 1) * nfit], color=XI_COLORS["ell02"], lw=1.7, label="$\\xi_0+\\xi_2$ fit")
                top.axvline(smin, color="0.4", lw=0.8, ls=":")
                top.set_ylabel(rf"$s^2\xi_{{{ell}}}(s)$")
                top.legend(frameon=False, fontsize=9)
                residual = (mean_xi[iell][mask] - map02[iell * nfit:(iell + 1) * nfit]) / sig[ell][mask]
                bottom.axhline(0.0, color="0.5", lw=0.8)
                bottom.plot(s_full[mask], residual, "o-", ms=3.5, lw=0.8, color=XI_COLORS["ell02"])
                bottom.set(xlabel=r"$s\ [h^{-1}{\rm Mpc}]$", ylabel=r"residual / $\sigma_{\rm mean}$", xlim=(25, 355))
                pte = ell2_audit["results"][f"{key}_ell02"]["per_pole_mean_goodness"][f"ell{ell}"]["pte"]
                top.set_title(rf"$\xi_{{{ell}}}$:  PTE($\xi_0+\xi_2$ fit) $= {pte:.1e}$", fontsize=11.0)
            figure.suptitle(rf"formal-GIC multipoles, $s_{{\min}}={smin:.0f}\ h^{{-1}}{{\rm Mpc}}$ (dotted), boxsafe RSD lightcone", fontsize=12.0)
            figure.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(figure)
            plt.close(figure)

        # P page
        with np.load(PAYLOAD_NPZ, allow_pickle=False) as d:
            k_obs = np.asarray(d["k_obs"], dtype="f8")
            fit_idx = np.asarray(d["fit_bin_indices"], dtype="i8")
        with np.load(P02_COV, allow_pickle=False) as d:
            cov_full = np.asarray(d["covariance_full"], dtype="f8")
        ids = np.concatenate([fit_idx, 150 + fit_idx])
        cov30 = cov_full[np.ix_(ids, ids)]
        with np.load(P02_DIR / "fits" / "p0_control" / "samples.npz", allow_pickle=False) as d:
            pred0 = np.asarray(d["prediction_map"], dtype="f8")
        with np.load(P02_DIR / "fits" / "p02" / "samples.npz", allow_pickle=False) as d:
            data = np.asarray(d["data"], dtype="f8")
            pred02 = np.asarray(d["prediction_map"], dtype="f8")
        figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), gridspec_kw={"height_ratios": [2.1, 1.0]})
        for ipole, (name, color) in enumerate((("P_0", PK_COLORS["p0"]), ("P_2", PK_COLORS["p02"]))):
            top, bottom = axes[0, ipole], axes[1, ipole]
            seg = slice(15 * ipole, 15 * (ipole + 1))
            sig = np.sqrt(np.diag(cov30[seg, seg]) / NPHASE)
            top.errorbar(k_obs, data[seg], yerr=sig, fmt="o", ms=4, color="0.25", zorder=2, label=f"x25 mean ${name}$")
            if ipole == 0:
                top.plot(k_obs, pred0[seg], color=PK_COLORS["p0"], lw=1.6, ls="--", label="$P_0$ fit")
            top.plot(k_obs, pred02[seg], color=PK_COLORS["p02"], lw=1.6, label="$P_0+P_2$ fit")
            if ipole == 0:
                top.set(xscale="log", yscale="log", ylabel=r"$P_0(k)\ [(h^{-1}{\rm Mpc})^3]$")
            else:
                top.set(xscale="log", ylabel=r"$P_2(k)\ [(h^{-1}{\rm Mpc})^3]$")
            top.legend(frameon=False, fontsize=9)
            residual = (data[seg] - pred02[seg]) / sig
            bottom.axhline(0.0, color="0.5", lw=0.8)
            bottom.plot(k_obs, residual, "o-", ms=3.5, lw=0.8, color=PK_COLORS["p02"])
            bottom.set(xlabel=r"$k\ [h\,{\rm Mpc}^{-1}]$", ylabel=r"residual / $\sigma_{\rm mean}$", xscale="log")
            pte = p02_audit["results"]["p02"]["per_pole_mean_goodness"][f"ell{2 * ipole}"]["pte"]
            top.set_title(rf"${name}$:  PTE($P_0+P_2$ fit) $= {pte:.1e}$", fontsize=11.0)
        figure.suptitle(r"boxsafe RSD lightcone, 15 DESI-PNG bins per pole, $k_{\max}=0.10\ h\,{\rm Mpc}^{-1}$", fontsize=12.0)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(figure)
        plt.close(figure)

    audit = {
        "task": "task43_plot_rsd_quadrupole_increment",
        "status": "pass",
        "figures": {
            "xi_contours": {"pdf": str(XICON_PDF), "sha256": sha256_file(XICON_PDF), "pages": ["smin=50", "smin=80"]},
            "p_contours": {"pdf": str(PCON_PDF), "sha256": sha256_file(PCON_PDF)},
            "bestfits": {"pdf": str(BESTFIT_PDF), "sha256": sha256_file(BESTFIT_PDF), "pages": ["xi smin=50", "xi smin=80", "P0/P2"]},
        },
        "inputs": {
            "ell2_audit": str(ELL2_AUDIT),
            "p02_audit": str(P02_AUDIT),
        },
        "contour_thresholds": audits,
    }
    atomic_write_json(PLOT_ROOT / "task43_rsd_quadrupole_increment_figures.json", audit)
    print(json.dumps({"status": "pass", "figures": audit["figures"]}, sort_keys=True))


if __name__ == "__main__":
    main()
