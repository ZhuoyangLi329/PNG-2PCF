#!/usr/bin/env python3
"""Audit and plot fixed-p=1/free-fNL P0--xi0 results for the PNG-base HOD boxes.

The contour layout and density construction follow the Task43 reference plot
requested by the user.  Only PDF figures are written.  Since P0 and xi0 are
measured from the same realization, differences quoted in independent-error
units are explicitly diagnostic and are never interpreted as a tension test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_CODE_DIR = PROJECT_ROOT / "codes" / "task43"
TASK44_CODE_DIR = PROJECT_ROOT / "codes" / "task44"
for _directory in (TASK43_CODE_DIR, TASK44_CODE_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from task43_plot_rsd_rawbox_pk0_vs_xi0_contours import draw_contour, posterior_text  # noqa: E402
from task44_pngbase_hodmap_rawbox_common import (  # noqa: E402
    BOX_SIZE,
    BOX_VOLUME,
    K_FUND,
    PLOT_ROOT,
    SUMMARY_DIR,
    atomic_write_json,
    get_spec,
    sha256_file,
)
from task44_fit_pngbase_pk_xi_consistency import variant_prefix  # noqa: E402
from task44_plot_pngbase_hodmap_rawbox import get_flat_samples, load_fit, posterior_row  # noqa: E402


REFERENCE_PDF = (
    PROJECT_ROOT
    / "plots/8.21meeting/task43_rsd_boxsafe_zobs0p4_0p8_x25_pk0_vs_xi0_smin50_l0only_contours.pdf"
)
OUTPUT_SUMMARY = SUMMARY_DIR / "task44_pngbase_hodmap_rawbox_fixedp1_freefnl_pk_xi_audit.json"
OUTPUT_FOREST = PLOT_ROOT / "task44_pngbase_hodmap_rawbox_fixedp1_freefnl_pk_xi_summary.pdf"
OUTPUT_CONTOURS = {
    tag: PLOT_ROOT / f"task44_pngbase_{tag}_fixedp1_pk0_vs_xi0_smin30_fnl_b1_contours.pdf"
    for tag in ("c300", "c302")
}
CORRECTED_PLOT_ROOT = PLOT_ROOT / "consistency_diagnostics"
OUTPUT_CORRECTED_FOREST = CORRECTED_PLOT_ROOT / "task44_pngbase_hodmap_rawbox_fixedfnl_pstochcov_pk_xi_smin30_summary.pdf"
OUTPUT_CORRECTED_CONTOURS = {
    tag: CORRECTED_PLOT_ROOT / f"task44_pngbase_{tag}_fixedfnl_pk0_vs_xi0_pstochcov_smin30_p_b1_contours.pdf"
    for tag in ("c300", "c302")
}
OUTPUT_CORRECTED_BESTFIT = {
    tag: CORRECTED_PLOT_ROOT / f"task44_pngbase_{tag}_fixedfnl_pk0_vs_xi0_pstochcov_smin30_bestfit_curves.pdf"
    for tag in ("c300", "c302")
}
COLORS = {"pk": "#2F2F2F", "xi30": "#DD8452"}
LABELS = {"fnl": r"$f_{\rm NL}$", "p": r"$p$", "b1": r"$b_1$"}


def fit_rows(tag: str) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Return only the active P0 and smin=30 xi0 fixed-p fits."""

    return [
        (
            "pk",
            r"$P_0(k)$",
            COLORS["pk"],
            load_fit(tag, "pk", kmin=0.003, fixed_p=1.0),
        ),
        (
            "xi",
            r"$\xi_0(s)$",
            COLORS["xi30"],
            load_fit(tag, "xi", smin=30.0, fixed_p=1.0),
        ),
    ]


def load_diagnostic_fit(tag: str, *, smin: float, fixed_p: bool) -> dict[str, Any]:
    kind = "xi_fixedp1_stochastic_cov" if fixed_p else "xi_stochastic_cov"
    prefix = variant_prefix(tag, kind, smin=smin)
    json_path, npz_path = prefix.with_suffix(".json"), prefix.with_suffix(".npz")
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    if summary.get("status") != "pass" or summary.get("output_npz_sha256") != sha256_file(npz_path):
        raise RuntimeError(f"invalid corrected xi fit: {json_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files if name != "summary_json"}
    return {"summary": summary, "arrays": arrays, "json_path": json_path, "npz_path": npz_path}


def corrected_fit_rows(tag: str) -> list[tuple[str, str, str, dict[str, Any]]]:
    return [
        (
            "pk",
            r"$P_0(k)$",
            COLORS["pk"],
            load_fit(tag, "pk", kmin=0.003, fixed_p=1.0),
        ),
        (
            "xi",
            r"$\xi_0(s)$",
            COLORS["xi30"],
            load_diagnostic_fit(tag, smin=30.0, fixed_p=True),
        ),
    ]


def corrected_freep_fit_rows(tag: str) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Return fixed-baseline-fNL/free-p fits for the requested contours."""

    return [
        (
            "pk",
            r"$P_0(k)$",
            COLORS["pk"],
            load_fit(tag, "pk", kmin=0.003),
        ),
        (
            "xi",
            r"$\xi_0(s)$",
            COLORS["xi30"],
            load_diagnostic_fit(tag, smin=30.0, fixed_p=False),
        ),
    ]


def combined_range(arrays: list[np.ndarray], *, include: float | None = None) -> tuple[float, float]:
    values = np.concatenate([np.asarray(array, dtype="f8") for array in arrays])
    lower, upper = map(float, np.quantile(values, [0.001, 0.999]))
    if include is not None:
        lower, upper = min(lower, float(include)), max(upper, float(include))
    width = upper - lower
    if not width > 0.0:
        width = max(abs(lower), 1.0)
    return lower - 0.07 * width, upper + 0.07 * width


def validate_fit(tag: str, key: str, fit: dict[str, Any]) -> dict[str, Any]:
    summary = fit["summary"]
    expected_names = ["b1", "fnl", "sn0"] if key == "pk" else ["b1", "fnl"]
    actual_names = [str(value) for value in fit["arrays"]["parameter_names"]]
    gates = {
        "fit_status_pass": summary.get("status") == "pass",
        "fixed_p_is_1": bool(np.isclose(float(summary.get("fixed_p")), 1.0, rtol=0.0, atol=1.0e-14)),
        "parameter_names_match": actual_names == expected_names,
        "all_mcmc_gates_pass": bool(all(summary["mcmc"]["gates"].values())),
        "covariance_fixed_point_converged": bool(summary["covariance"]["fixed_point_converged"]),
        "full_periodic_realspace_geometry": summary.get("geometry") == "full periodic real-space cube",
        "volume_is_L_cubed": bool(np.isclose(float(summary["volume_mpc_h3"]), BOX_VOLUME, rtol=0.0, atol=1.0e-6)),
        "kfund_is_2pi_over_L": bool(np.isclose(float(summary["kfund_h_mpc"]), K_FUND, rtol=0.0, atol=1.0e-16)),
    }
    if key == "pk":
        gates.update(
            {
                "primary_kmin_is_0p003": bool(
                    np.isclose(float(summary["fit_range"]["kmin_edge_h_mpc"]), 0.003, rtol=0.0, atol=1.0e-14)
                ),
                "kmax_is_0p10": bool(
                    np.isclose(float(summary["fit_range"]["kmax_contract_h_mpc"]), 0.1, rtol=0.0, atol=1.0e-14)
                ),
                "lowest_k_bin_has_18_modes": bool(summary["gates"]["primary_first_bin_nmodes_18"]),
                "primary_has_49_contiguous_periodic_bins": bool(
                    summary["fit_range"]["nbins"] == 49
                    and summary["gates"]["primary_uses_49_contiguous_periodic_bins"]
                ),
            }
        )
    else:
        expected_smin = 30.0
        gates.update(
            {
                "smin_matches_label": bool(
                    np.isclose(float(summary["fit_range"]["smin_mpc_h"]), expected_smin, rtol=0.0, atol=1.0e-14)
                ),
                "smax_is_350": bool(
                    np.isclose(float(summary["fit_range"]["smax_mpc_h"]), 350.0, rtol=0.0, atol=1.0e-14)
                ),
            }
        )
    if not all(gates.values()):
        raise RuntimeError(f"{tag}/{key} fixed-p fit failed audit: {gates}")
    return gates


def draw_triangle(
    tag: str,
    rows: list[tuple[str, str, str, dict[str, Any]]],
    output: Path,
    *,
    parameter: str = "fnl",
) -> dict[str, Any]:
    if parameter not in ("fnl", "p"):
        raise ValueError(parameter)
    baseline = float(get_spec(tag).fnl)
    chains = {key: get_flat_samples(fit, (parameter, "b1")) for key, _label, _color, fit in rows}
    ranges = {
        parameter: combined_range(
            [chain[:, 0] for chain in chains.values()],
            include=baseline if parameter == "fnl" else None,
        ),
        "b1": combined_range([chain[:, 1] for chain in chains.values()]),
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
    figure, axes = plt.subplots(2, 2, figsize=(7.4, 6.8))
    axes[0, 1].set_axis_off()
    contour_audit: dict[str, Any] = {}
    for index, plotted_parameter in enumerate((parameter, "b1")):
        axis = axes[index, index]
        bins = np.linspace(*ranges[plotted_parameter], 90)
        for key, _label, color, _fit in rows:
            axis.hist(chains[key][:, index], bins=bins, density=True, histtype="step", lw=1.9, color=color)
        axis.set_xlim(*ranges[plotted_parameter])
        axis.set_yticks([])
        axis.set_xlabel(LABELS[plotted_parameter])
        if plotted_parameter == "fnl":
            axis.axvline(baseline, color="0.45", lw=1.0, ls="--")

    axis = axes[1, 0]
    for zorder, (key, _label, color, _fit) in enumerate(rows, start=1):
        contour_audit[key] = draw_contour(
            axis,
            chains[key][:, 0],
            chains[key][:, 1],
            color=color,
            xlim=ranges[parameter],
            ylim=ranges["b1"],
            zorder=2 * zorder,
        )
    if parameter == "fnl":
        axis.axvline(baseline, color="0.45", lw=1.0, ls="--", zorder=0)
    axis.set(xlim=ranges[parameter], ylim=ranges["b1"])
    axis.set_xlabel(LABELS[parameter])
    axis.set_ylabel(LABELS["b1"])

    handles = [plt.Line2D([], [], color=color, lw=2.0) for _key, _label, color, _fit in rows]
    legend_labels = []
    for _key, label, _color, fit in rows:
        symbol = r"f_{\rm NL}" if parameter == "fnl" else "p"
        legend_labels.append(
            rf"{label}: ${symbol}={posterior_text(posterior_row(fit, parameter))}$"
        )
    axes[0, 1].legend(handles, legend_labels, loc="upper left", frameon=False, fontsize=10.8)
    parameter_contract = (
        rf"catalog baseline $f_{{\rm NL}}={baseline:g}$" "\n" r"fixed $p=1$"
        if parameter == "fnl"
        else rf"fixed $f_{{\rm NL}}={baseline:g}$"
    )
    axes[0, 1].text(
        0.02,
        0.44,
        f"{tag} periodic real space\n"
        f"{parameter_contract}\n"
        r"$P_0$: $k_{\min}=0.003$, $k_{\max}=0.100\ h\,{\rm Mpc}^{-1}$ (49 bins)" "\n"
        r"$\xi_0$: $s_{\min}=30$, $s_{\max}=350\ h^{-1}{\rm Mpc}$ (32 bins)" "\n"
        "68% and 95% contours",
        transform=axes[0, 1].transAxes,
        ha="left",
        va="top",
        fontsize=10.4,
        linespacing=1.30,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.45"},
    )
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.11, top=0.97, wspace=0.08, hspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.pdf")
    figure.savefig(temporary, format="pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    temporary.replace(output)
    if output.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError(f"output lacks PDF header: {output}")
    return {
        "output": str(output),
        "sha256": sha256_file(output),
        "credible_contours": [0.68, 0.95],
        "plot_ranges": {name: list(value) for name, value in ranges.items()},
        "density_thresholds": contour_audit,
        "sample_counts": {key: int(chain.shape[0]) for key, chain in chains.items()},
    }


def draw_fixedfnl_bestfit(
    tag: str,
    rows: list[tuple[str, str, str, dict[str, Any]]],
    output: Path,
) -> dict[str, Any]:
    """Draw the MAP curves corresponding exactly to the fixed-fNL p contours."""

    fits = {key: fit for key, _label, _color, fit in rows}
    pk, xi = fits["pk"], fits["xi"]
    baseline = float(get_spec(tag).fnl)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11.0,
            "axes.labelsize": 12.5,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), gridspec_kw={"wspace": 0.24})

    k = np.asarray(pk["arrays"]["coordinate"], dtype="f8")
    pk_data = np.asarray(pk["arrays"]["data"], dtype="f8")
    pk_map = np.asarray(pk["arrays"]["prediction_map"], dtype="f8")
    pk_sigma = np.sqrt(np.diag(np.asarray(pk["arrays"]["covariance_final"], dtype="f8")))
    axes[0].errorbar(
        k,
        pk_data,
        yerr=pk_sigma,
        fmt="o",
        ms=3.4,
        lw=0.7,
        capsize=1.5,
        color="0.25",
        label="measurement",
    )
    axes[0].plot(k, pk_map, color=COLORS["pk"], lw=1.8, label="best fit")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    axes[0].set_ylabel(r"$P_0(k)\ [(h^{-1}{\rm Mpc})^3]$")
    axes[0].set_title(r"$P_0(k)$", fontsize=15)
    axes[0].legend(frameon=False, fontsize=10)

    s = np.asarray(xi["arrays"]["coordinate"], dtype="f8")
    xi_data = np.asarray(xi["arrays"]["data"], dtype="f8")
    xi_map = np.asarray(xi["arrays"]["prediction_map"], dtype="f8")
    xi_sigma = np.sqrt(np.diag(np.asarray(xi["arrays"]["covariance_final"], dtype="f8")))
    axes[1].errorbar(
        s,
        s**2 * xi_data,
        yerr=s**2 * xi_sigma,
        fmt="o",
        ms=3.4,
        lw=0.7,
        capsize=1.5,
        color="0.25",
        label="measurement",
    )
    axes[1].plot(
        s,
        s**2 * xi_map,
        color=COLORS["xi30"],
        lw=1.7,
        label="best fit",
    )
    axes[1].axhline(0.0, color="0.7", lw=0.7)
    axes[1].set_xlabel(r"$s\ [h^{-1}{\rm Mpc}]$")
    axes[1].set_ylabel(r"$s^2\xi_0(s)\ [(h^{-1}{\rm Mpc})^2]$")
    axes[1].set_title(r"$\xi_0(s)$", fontsize=15)
    axes[1].legend(frameon=False, fontsize=10)

    map_rows = {key: fit["summary"]["map"] for key, fit in fits.items()}
    figure.suptitle(tag, fontsize=20, fontweight="bold", y=0.99)
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.13, top=0.82)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.pdf")
    figure.savefig(temporary, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)
    temporary.replace(output)
    if output.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError(f"output lacks PDF header: {output}")
    return {
        "output": str(output),
        "sha256": sha256_file(output),
        "fixed_fnl": baseline,
        "map": map_rows,
        "curves": ["pk", "xi"],
        "layout": "two best-fit panels without annotations or residuals",
    }


def draw_forest(catalogs: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
    labels = [r"$P_0(k)$", r"$\xi_0(s)$"]
    keys = ("pk", "xi")
    y = np.arange(2)[::-1]
    for axis, tag in zip(axes, ("c300", "c302"), strict=True):
        row = catalogs[tag]
        for yy, key in zip(y, keys, strict=True):
            posterior = row["fits"][key]["posterior"]["fnl"]
            axis.errorbar(
                posterior["q50"],
                yy,
                xerr=[[posterior["q50"] - posterior["q16"]], [posterior["q84"] - posterior["q50"]]],
                fmt="o",
                color=COLORS[key if key == "pk" else "xi30"],
                capsize=3,
                ms=6,
            )
        axis.axvline(row["catalog_baseline_fnl"], color="0.45", lw=1.0, ls="--", label="catalog baseline")
        axis.set_yticks(y, labels)
        axis.set_ylim(-0.65, 1.65)
        axis.set_xlabel(r"$f_{\rm NL}$ at fixed $p=1$")
        axis.set_title(rf"{tag}, baseline $f_{{\rm NL}}={row['catalog_baseline_fnl']:g}$")
        axis.grid(axis="x", alpha=0.2)
        axis.legend(frameon=False, fontsize=8.5)
    figure.suptitle(
        "Periodic real-space HOD-MAP boxes\n"
        r"$P_0$: $k_{\min}=0.003$, $k_{\max}=0.100\ h\,{\rm Mpc}^{-1}$; "
        r"$\xi_0$: $s_{\min}=30$, $s_{\max}=350\ h^{-1}{\rm Mpc}$",
        y=0.99,
        fontsize=11.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.45"},
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.78))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.pdf")
    figure.savefig(temporary, format="pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    temporary.replace(output)
    if output.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError(f"output lacks PDF header: {output}")


def draw_freep_forest(catalogs: dict[str, Any], output: Path) -> None:
    """Summarize only the active fixed-fNL/free-p P0 and xi0 results."""

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
    labels = [r"$P_0(k)$", r"$\xi_0(s)$"]
    keys = ("pk", "xi")
    y = np.arange(2)[::-1]
    for axis, tag in zip(axes, ("c300", "c302"), strict=True):
        row = catalogs[tag]
        for yy, key in zip(y, keys, strict=True):
            posterior = row["fits"][key]["posterior"]["p"]
            axis.errorbar(
                posterior["q50"],
                yy,
                xerr=[
                    [posterior["q50"] - posterior["q16"]],
                    [posterior["q84"] - posterior["q50"]],
                ],
                fmt="o",
                color=COLORS[key if key == "pk" else "xi30"],
                capsize=3,
                ms=6,
            )
        axis.set_yticks(y, labels)
        axis.set_ylim(-0.65, 1.65)
        axis.set_xlabel(r"$p$")
        axis.set_title(rf"{tag}, fixed $f_{{\rm NL}}={row['fixed_fnl']:g}$")
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle(
        "Periodic real-space HOD-MAP boxes\n"
        r"$P_0$: $k_{\min}=0.003$, $k_{\max}=0.100\ h\,{\rm Mpc}^{-1}$; "
        r"$\xi_0$: $s_{\min}=30$, $s_{\max}=350\ h^{-1}{\rm Mpc}$",
        y=0.99,
        fontsize=11.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.45"},
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.78))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.pdf")
    figure.savefig(temporary, format="pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    temporary.replace(output)
    if output.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError(f"output lacks PDF header: {output}")


def build_catalog_record(tag: str, rows: list[tuple[str, str, str, dict[str, Any]]]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "catalog_baseline_fnl": float(get_spec(tag).fnl),
        "fixed_p": 1.0,
        "fits": {},
        "comparisons": {},
        "gates": {},
    }
    by_key = {key: fit for key, _label, _color, fit in rows}
    for key, fit in by_key.items():
        gates = validate_fit(tag, key, fit)
        record["gates"][key] = gates
        record["fits"][key] = {
            "posterior": {name: posterior_row(fit, name) for name in ("fnl", "b1")},
            "sn0": posterior_row(fit, "sn0") if key == "pk" else None,
            "map": fit["summary"]["map"],
            "mcmc": fit["summary"]["mcmc"],
            "json": str(fit["json_path"]),
            "json_sha256": sha256_file(fit["json_path"]),
            "npz": str(fit["npz_path"]),
            "npz_sha256": sha256_file(fit["npz_path"]),
        }
        free_fit = load_fit(
            tag,
            "pk" if key == "pk" else "xi",
            kmin=0.003 if key == "pk" else None,
            smin=30.0,
        )
        delta_prediction = np.asarray(fit["arrays"]["prediction_map"], dtype="f8") - np.asarray(
            free_fit["arrays"]["prediction_map"], dtype="f8"
        )
        sigma = np.sqrt(np.diag(np.asarray(fit["arrays"]["covariance_final"], dtype="f8")))
        record["fits"][key]["reparameterization_check"] = {
            "fixed_p1_vs_fixed_baseline_fnl_freep_map_max_abs": float(np.max(np.abs(delta_prediction))),
            "fixed_p1_vs_fixed_baseline_fnl_freep_map_max_abs_sigma": float(np.max(np.abs(delta_prediction / sigma))),
            "map_chi2_difference": float(fit["summary"]["map"]["chi2"] - free_fit["summary"]["map"]["chi2"]),
        }

    pk_fnl = record["fits"]["pk"]["posterior"]["fnl"]
    xi_fnl = record["fits"]["xi"]["posterior"]["fnl"]
    independent_sigma = float(np.hypot(pk_fnl["sigma68"], xi_fnl["sigma68"]))
    delta = float(xi_fnl["q50"] - pk_fnl["q50"])
    record["comparisons"]["xi_minus_pk"] = {
        "delta_fnl": delta,
        "independent_covariance_combined_sigma68": independent_sigma,
        "delta_over_independent_combined_sigma68": float(delta / independent_sigma),
        "cross_covariance_included": False,
        "interpretation": "descriptive only because P0 and xi0 use the same ph000 realization",
    }
    for key in ("pk", "xi"):
        posterior = record["fits"][key]["posterior"]["fnl"]
        record["comparisons"][f"{key}_baseline_pull_sigma68"] = float(
            (posterior["q50"] - record["catalog_baseline_fnl"]) / posterior["sigma68"]
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    targets = (
        OUTPUT_SUMMARY,
        OUTPUT_CORRECTED_FOREST,
        *OUTPUT_CORRECTED_CONTOURS.values(),
        *OUTPUT_CORRECTED_BESTFIT.values(),
    )
    if all(path.is_file() for path in targets) and not args.overwrite:
        print(json.dumps({"status": "skip", "summary": str(OUTPUT_SUMMARY)}, sort_keys=True))
        return
    if not REFERENCE_PDF.is_file():
        raise FileNotFoundError(REFERENCE_PDF)
    for path in targets:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"partial output exists; rerun with --overwrite after audit: {path}")

    catalogs: dict[str, Any] = {}
    corrected_catalogs: dict[str, Any] = {}
    corrected_freep_catalogs: dict[str, Any] = {}
    corrected_plot_records: dict[str, Any] = {}
    corrected_bestfit_records: dict[str, Any] = {}
    for tag in ("c300", "c302"):
        rows = fit_rows(tag)
        catalogs[tag] = build_catalog_record(tag, rows)
        corrected_rows = corrected_fit_rows(tag)
        corrected_catalogs[tag] = {
            "catalog_baseline_fnl": float(get_spec(tag).fnl),
            "fixed_p": 1.0,
            "fits": {
                key: {
                    "posterior": {
                        "fnl": posterior_row(fit, "fnl"),
                        "b1": posterior_row(fit, "b1"),
                    },
                    "map": fit["summary"]["map"],
                    "status": fit["summary"]["status"],
                    "mcmc_gates": fit["summary"]["mcmc"]["gates"],
                    "json": str(fit["json_path"]),
                    "json_sha256": sha256_file(fit["json_path"]),
                    "npz": str(fit["npz_path"]),
                    "npz_sha256": sha256_file(fit["npz_path"]),
                    "covariance_policy": (
                        "P0 self-consistent residual stochastic"
                        if key == "pk"
                        else "xi0 includes primary-P0 residual stochastic power"
                    ),
                }
                for key, _label, _color, fit in corrected_rows
            },
        }
        corrected_freep_rows = corrected_freep_fit_rows(tag)
        corrected_freep_catalogs[tag] = {
            "catalog_baseline_fnl": float(get_spec(tag).fnl),
            "fixed_fnl": float(get_spec(tag).fnl),
            "free_parameters": {"pk": ["b1", "p", "sn0"], "xi": ["b1", "p"]},
            "fits": {
                key: {
                    "posterior": {
                        "p": posterior_row(fit, "p"),
                        "b1": posterior_row(fit, "b1"),
                    },
                    "map": fit["summary"]["map"],
                    "status": fit["summary"]["status"],
                    "mcmc_gates": fit["summary"]["mcmc"]["gates"],
                    "json": str(fit["json_path"]),
                    "json_sha256": sha256_file(fit["json_path"]),
                    "npz": str(fit["npz_path"]),
                    "npz_sha256": sha256_file(fit["npz_path"]),
                }
                for key, _label, _color, fit in corrected_freep_rows
            },
        }
        corrected_plot_records[tag] = draw_triangle(
            tag,
            corrected_freep_rows,
            OUTPUT_CORRECTED_CONTOURS[tag],
            parameter="p",
        )
        corrected_bestfit_records[tag] = draw_fixedfnl_bestfit(
            tag,
            corrected_freep_rows,
            OUTPUT_CORRECTED_BESTFIT[tag],
        )
    draw_freep_forest(corrected_freep_catalogs, OUTPUT_CORRECTED_FOREST)
    global_gates = {
        "all_fit_audits_pass": bool(
            all(all(all(values.values()) for values in row["gates"].values()) for row in catalogs.values())
        ),
        "all_map_reparameterization_shifts_below_1e_4_sigma": bool(
            max(
                fit["reparameterization_check"]["fixed_p1_vs_fixed_baseline_fnl_freep_map_max_abs_sigma"]
                for row in catalogs.values()
                for fit in row["fits"].values()
            )
            < 1.0e-4
        ),
        "pdf_only_outputs": bool(
            all(
                path.suffix == ".pdf" and path.read_bytes()[:5] == b"%PDF-"
                for path in (
                    OUTPUT_CORRECTED_FOREST,
                    *OUTPUT_CORRECTED_CONTOURS.values(),
                    *OUTPUT_CORRECTED_BESTFIT.values(),
                )
            )
        ),
        "all_stochastic_covariance_corrected_fits_pass": bool(
            all(
                fit["status"] == "pass" and all(fit["mcmc_gates"].values())
                for row in (*corrected_catalogs.values(), *corrected_freep_catalogs.values())
                for fit in row["fits"].values()
            )
        ),
    }
    payload = {
        "task": "task44_pngbase_hodmap_rawbox_fixedp1_freefnl_pk_xi",
        "status": "pass" if all(global_gates.values()) else "review",
        "scope": {
            "geometry": "full periodic real-space cube",
            "boxsize_mpc_h": BOX_SIZE,
            "volume_mpc_h3": BOX_VOLUME,
            "kfund_h_mpc": K_FUND,
            "fixed_p": 1.0,
            "fnl_prior": [-500.0, 500.0],
            "pk": {
                "bin_edges_h_mpc": [0.003, 0.101],
                "bin_center_max_h_mpc": 0.100,
                "nbins": 49,
                "selection": "all contiguous native periodic-box bins",
                "free_parameters": ["b1", "fnl", "sn0"],
            },
            "xi": {"fit_range_mpc_h": [30.0, 350.0], "free_parameters": ["b1", "fnl"]},
        },
        "catalogs": catalogs,
        "plots": {
            "stochastic_covariance_corrected_summary": {
                "path": str(OUTPUT_CORRECTED_FOREST),
                "sha256": sha256_file(OUTPUT_CORRECTED_FOREST),
            },
            "stochastic_covariance_corrected_fixedfnl_freep_contours": corrected_plot_records,
            "stochastic_covariance_corrected_fixedfnl_freep_bestfit_curves": corrected_bestfit_records,
        },
        "stochastic_covariance_corrected_catalogs": corrected_catalogs,
        "stochastic_covariance_corrected_fixedfnl_freep_catalogs": corrected_freep_catalogs,
        "global_gates": global_gates,
        "warnings": [
            "P0 and xi0 are measured from the same ph000 realization; independent-error differences are descriptive only.",
            "c300 and c302 share ph000 initial phases and have separately retuned HOD parameters; do not multiply them as independent likelihoods.",
            "The active xi0 result uses smin=30 Mpc/h only; non-active radial-cut chains are excluded from result products.",
            "The original xi0 rows use Poisson-only covariance; stochastic_covariance_corrected_catalogs contains the corrected diagnostic chains.",
        ],
    }
    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(OUTPUT_SUMMARY, payload)
    print(json.dumps({"status": payload["status"], "summary": str(OUTPUT_SUMMARY), "plots": payload["plots"]}, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
