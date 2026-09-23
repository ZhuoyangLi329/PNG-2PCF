#!/usr/bin/env python3
"""9.11-style RSD fNL-b1 corner pages for the EZmock vs jaxpower covariance MCMC.

Layout and typography follow task43_plot_lightcone_joint_baomask_v1.py
(RSD P02 / BAO-masked xi02 / joint page of the 9.11 main result).  One page
per covariance choice: EZmock (empirical) then jaxpower (analytic); both
pages share the same axis ranges so they can be compared side by side.

Usage:
    python task43_plot_ezmock506_covariance_mcmc_vs_jaxpower.py \
        --mcmc-root <.../mcmc_preliminary_N> --nmock N
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from task43_plot_rawbox_joint_baomask_v1 import COLORS
from task43_plot_rsd_rawbox_pk0_vs_xi0_contours import density_levels, plot_range
from task43_rsd_common import PROJECT_ROOT, atomic_write_json, sha256_file

MCMC_ROOT_DEFAULT = (
    PROJECT_ROOT
    / "outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60"
    / "mcmc_preliminary_506"
)
MEETING_DIR = PROJECT_ROOT / "9.18meeting/task43_ezmock_covariance_mcmc_vs_jaxpower"
VARIANTS = (("p02", "p", r"$P_0+P_2$"), ("xi02", "xi", r"$\xi_0+\xi_2$"), ("joint", "joint", "joint"))


def load_chain(root: Path, tag: str) -> dict[str, Any]:
    path = root / f"chain_{tag}.npz"
    with np.load(path, allow_pickle=False) as payload:
        chain = np.asarray(payload["chain"], dtype="f8").reshape(-1, payload["chain"].shape[-1])
    summary = json.loads((root / f"summary_{tag}.json").read_text(encoding="utf-8"))
    if not all(summary["gates"].values()):
        raise RuntimeError(f"failed convergence gates: {tag}")
    return {"chain": chain, "summary": summary, "path": path}


def draw_contour(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zorder: int,
) -> None:
    hist, xedges, yedges = np.histogram2d(
        np.asarray(x, dtype="f8"),
        np.asarray(y, dtype="f8"),
        bins=(170, 160),
        range=(xlim, ylim),
        density=False,
    )
    hist = gaussian_filter(hist.astype("f8"), sigma=2.5, mode="nearest")
    level95, level68 = density_levels(hist)
    xcenter = 0.5 * (xedges[:-1] + xedges[1:])
    ycenter = 0.5 * (yedges[:-1] + yedges[1:])
    axis.contourf(
        xcenter,
        ycenter,
        hist.T,
        levels=[level95, level68, float(np.max(hist)) * 1.001],
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


def interval_text(samples: np.ndarray, maximum_likelihood: float) -> str:
    q16, q84 = np.percentile(samples, [16.0, 84.0])
    if q16 <= maximum_likelihood <= q84:
        return rf"{maximum_likelihood:.2f}_{{-{maximum_likelihood - q16:.2f}}}^{{+{q84 - maximum_likelihood:.2f}}}"
    return rf"{maximum_likelihood:.2f};\ 68\%=[{q16:.2f},{q84:.2f}]"


def triangle_page(
    pdf: PdfPages,
    *,
    cov: str,
    cov_display: str,
    chains: dict[str, Any],
    ranges: dict[str, tuple[float, float]],
    truth_fnl: float = 0.0,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(6.8, 6.0))
    axes[0, 1].set_axis_off()
    fNL_bins = np.linspace(*ranges["fNL"], 90)
    b1_bins = np.linspace(*ranges["b1"], 90)
    for variant, color_key, _ in VARIANTS:
        axes[0, 0].hist(
            chains[f"{cov}_{variant}"]["chain"][:, 0],
            bins=fNL_bins,
            density=True,
            histtype="step",
            lw=1.8,
            color=COLORS[color_key],
        )
        axes[1, 1].hist(
            chains[f"{cov}_{variant}"]["chain"][:, 1],
            bins=b1_bins,
            density=True,
            histtype="step",
            lw=1.8,
            color=COLORS[color_key],
        )
    axes[0, 0].set_xlim(*ranges["fNL"])
    axes[0, 0].set_yticks([])
    axes[0, 0].tick_params(labelbottom=False)
    axes[0, 0].axvline(truth_fnl, color="0.55", lw=0.8, ls="--", zorder=0)
    zorder = 2
    for variant, color_key, _ in VARIANTS:
        draw_contour(
            axes[1, 0],
            chains[f"{cov}_{variant}"]["chain"][:, 0],
            chains[f"{cov}_{variant}"]["chain"][:, 1],
            color=COLORS[color_key],
            xlim=ranges["fNL"],
            ylim=ranges["b1"],
            zorder=zorder,
        )
        zorder += 2
    axes[1, 0].set(xlim=ranges["fNL"], ylim=ranges["b1"])
    axes[1, 0].set_xlabel(r"$f_{\rm NL}$")
    axes[1, 0].set_ylabel(r"$b_1$")
    axes[1, 0].axvline(truth_fnl, color="0.55", lw=0.8, ls="--", zorder=0)
    axes[1, 1].set_xlim(*ranges["b1"])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_xlabel(r"$b_1$")
    handles, legend_labels = [], []
    for variant, color_key, variant_display in VARIANTS:
        entry = chains[f"{cov}_{variant}"]
        handles.append(plt.Line2D([], [], color=COLORS[color_key], lw=2.0))
        legend_labels.append(
            variant_display
            + "\n"
            + rf"$f_{{\rm NL}}={interval_text(entry['chain'][:, 0], entry['summary']['map_theta'][0])}$"
        )
    figure.legend(
        handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.95),
        frameon=False,
        fontsize=17.6,
        labelspacing=0.55,
        handletextpad=0.7,
    )
    figure.suptitle(
        "Task43 lightcone RSD (box-safe 0.4 < z_obs < 0.8; kmax=0.08, smin=50): "
        f"P02, BAO-masked xi02, and joint — {cov_display} covariance",
        fontsize=10.5,
        y=0.985,
    )
    figure.subplots_adjust(left=0.13, right=0.97, bottom=0.10, top=0.93, wspace=0.08, hspace=0.08)
    pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcmc-root", type=Path, default=MCMC_ROOT_DEFAULT)
    parser.add_argument("--nmock", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    mcmc_root = arguments.mcmc_root
    ezlabel = f"ezmock{int(arguments.nmock)}"
    pages = ((ezlabel, f"EZmock-{int(arguments.nmock)} empirical"), ("jaxpower", "jaxpower analytic"))
    output = MEETING_DIR / f"task43_{ezlabel}_vs_jaxpower_rsd_P02xi02_joint_kmax0p08_smin50_baomask80_120_v1.pdf"
    output_json = output.with_suffix(".json")
    if not arguments.force and (output.exists() or output_json.exists()):
        raise FileExistsError(f"immutable plot exists: {output} / {output_json}")
    chains = {
        f"{cov}_{variant}": load_chain(mcmc_root, f"{cov}_{variant}")
        for cov, _ in pages
        for variant, _, _ in VARIANTS
    }
    fNL_all = np.concatenate([entry["chain"][:, 0] for entry in chains.values()])
    b1_all = np.concatenate([entry["chain"][:, 1] for entry in chains.values()])
    ranges = {
        "fNL": plot_range(fNL_all, fNL_all, parameter="fNL"),
        "b1": plot_range(b1_all, b1_all, parameter="b1"),
    }
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10.0,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary) as pdf:
        for cov, cov_display in pages:
            triangle_page(pdf, cov=cov, cov_display=cov_display, chains=chains, ranges=ranges)
    temporary.replace(output)
    if output.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError("output is not a PDF")
    constraints = {}
    for tag, entry in chains.items():
        posterior = entry["summary"]["posterior"]
        map_theta = np.asarray(entry["summary"]["map_theta"], dtype="f8")
        constraints[tag] = {
            "fNL": {
                "maximum_likelihood": float(map_theta[0]),
                **{key: posterior["fNL"][key] for key in ("q16", "q50", "q84", "sigma68")},
            },
            "b1": {
                "maximum_likelihood": float(map_theta[1]),
                **{key: posterior["b1"][key] for key in ("q16", "q50", "q84", "sigma68")},
            },
            "chain_npz": str(entry["path"]),
            "gates": entry["summary"]["gates"],
        }
    atomic_write_json(
        output_json,
        {
            "task": "task43_plot_ezmock506_covariance_mcmc_vs_jaxpower",
            "status": "pass",
            "mcmc_root": str(mcmc_root),
            "ezmock_realizations": int(arguments.nmock),
            "style_reference": "9.11meeting/task43_standard_kmax0p08_smin50/task43_lightcone_real_Pxi_rsd_P02xi02_joint_kmax0p08_smin50_baomask80_120_v1.pdf",
            "pages": [f"EZmock-{int(arguments.nmock)} empirical covariance", "jaxpower analytic covariance"],
            "contour_parameters": ["fNL", "b1"],
            "shared_axis_ranges": {"fNL": list(ranges["fNL"]), "b1": list(ranges["b1"])},
            "data_contract": "P0 13 bins + P2 (kmin 0.015) 9 bins + BAO-masked xi0/xi2 26 bins each; Abacus lightcone x25 mean",
            "constraints": constraints,
            "output_pdf": str(output),
            "output_pdf_sha256": sha256_file(output),
        },
    )
    print(json.dumps({"status": "pass", "output": str(output), "sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
