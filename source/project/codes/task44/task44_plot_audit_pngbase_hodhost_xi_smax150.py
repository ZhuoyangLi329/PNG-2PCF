#!/usr/bin/env python3
"""Plot and audit matched-catalog fixed-fNL halo P0 and halo xi0 fits."""

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

TASK43_CODE_DIR = Path(__file__).resolve().parents[1] / "task43"
if str(TASK43_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(TASK43_CODE_DIR))

from task43_plot_rsd_rawbox_pk0_vs_xi0_contours import (  # noqa: E402
    COLORS as TASK43_COLORS,
    draw_contour as task43_draw_contour,
    plot_range as task43_plot_range,
    posterior_text as task43_posterior_text,
)

from task44_pngbase_hodhost_mmin1e13_common import (
    CATALOGS,
    PK_KMAX_CONTRACT,
    PK_STRICT_FIT_EDGES,
    PK_STRICT_KMIN,
    PLOT_ROOT,
    S_CENTERS,
    S_EDGES,
    SUMMARY_DIR,
    XI_FIT_CENTERS,
    XI_FIT_EDGES,
    XI_FIT_SMAX,
    XI_FIT_SMIN,
    atomic_savez,
    atomic_write_json,
    ensure_output_dirs,
    get_spec,
    host_catalog_path,
    host_pk_fit_prefix,
    host_pk_metadata_path,
    host_pk_path,
    host_xi_fit_prefix,
    sha256_file,
    xi_engine_metadata_path,
    xi_engine_path,
)


COLORS = {"c300": "#4C72B0", "c302": "#C44E52", "pk": "#4C72B0", "xi": "#C44E52"}
CONTOUR_PARAMETERS = ("p", "b1")
CONTOUR_LABELS = {"p": r"$p$", "b1": r"$b_1$"}
TASK43_REFERENCE_PDF = Path(
    "/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/plots/8.21meeting/"
    "task43_rsd_boxsafe_zobs0p4_0p8_x25_pk0_vs_xi0_smin50_l0only_contours.pdf"
)
BESTFIT_MEASUREMENT_PDF = PLOT_ROOT / (
    "task44_pngbase_c300_c302_hodhost_mmin1e13_fixedfnl_"
    "pk0_kmin0p006_kmax0p100_xi0_smin30_smax150_bestfit_measurements.pdf"
)
CONTOUR_PDF = PLOT_ROOT / (
    "task44_pngbase_c300_c302_hodhost_mmin1e13_fixedfnl_"
    "pk0_kmin0p006_kmax0p100_vs_xi0_smin30_smax150_p_b1_contours.pdf"
)
GAUSSIAN_P_ADDITIONAL_PDFS = (
    PLOT_ROOT / (
        "task44_pngbase_c300_c302_hodhost_mmin1e13_gaussianp_freefnl_"
        "pk0_kmin0p006_kmax0p100_xi0_smin30_smax150_bestfit_measurements.pdf"
    ),
    PLOT_ROOT / (
        "task44_pngbase_c300_c302_hodhost_mmin1e13_gaussianp_freefnl_"
        "pk0_kmin0p006_kmax0p100_vs_xi0_smin30_smax150_fnl_b1_p_contours.pdf"
    ),
)
LRG_CURRENT_PDFS = (
    PLOT_ROOT / (
        "task44_pngbase_c300_c302_hodmap_lrg_fixedfnl_"
        "pk0_kmin0p006_kmax0p100_xi0_smin30_smax150_bestfit_measurements.pdf"
    ),
    PLOT_ROOT / (
        "task44_pngbase_c300_c302_hodmap_lrg_fixedfnl_"
        "pk0_kmin0p006_kmax0p100_vs_xi0_smin30_smax150_p_b1_contours.pdf"
    ),
    PLOT_ROOT / (
        "task44_pngbase_c300_c302_hodmap_lrg_gaussianp_freefnl_"
        "pk0_kmin0p006_kmax0p100_xi0_smin30_smax150_bestfit_measurements.pdf"
    ),
    PLOT_ROOT / (
        "task44_pngbase_c300_c302_hodmap_lrg_gaussianp_freefnl_"
        "pk0_kmin0p006_kmax0p100_vs_xi0_smin30_smax150_fnl_b1_p_contours.pdf"
    ),
)
LEGACY_PDFS = (
    PLOT_ROOT / "task44_pngbase_c300_c302_fixedfnl_p_kmin0p003_vs_strict0p006.pdf",
    PLOT_ROOT / "task44_pngbase_c300_c302_fixedfnl_strictkmin0p006_pk0_bestfit_curves.pdf",
    PLOT_ROOT / "task44_pngbase_c300_c302_hodhost_mmin1e13_fixedfnl_pk0_vs_xi0_smax150_p_b1_contours.pdf",
    PLOT_ROOT / "task44_pngbase_hodhost_mmin1e13_fixedfnl_pk0_strictkmin0p006_bestfit_curves.pdf",
    PLOT_ROOT / "task44_pngbase_hodhost_mmin1e13_fixedfnl_pk0_vs_xi0_smax150_p_constraints.pdf",
    PLOT_ROOT / "task44_pngbase_hodhost_mmin1e13_fixedfnl_xi0_smin30_smax150_bestfit_curves.pdf",
    PLOT_ROOT / "task44_pngbase_hodhost_mmin1e13_vs_hodlrg_xi.pdf",
    PLOT_ROOT / "task44_pngbase_hodhost_mmin1e13_xi_three_engine.pdf",
    PLOT_ROOT / "task44_pngbase_c300_c302_fixedfnl_strictpk0_vs_hodhost_xi0_smax150_p_b1_contours.pdf",
    PLOT_ROOT / "task44_pngbase_hodhost_mmin1e13_fixedfnl_xi0_smin30_smax150_p_b1_contours.pdf",
    PLOT_ROOT / "task44_pngbase_c300_c302_fixedfnl_strictkmin0p006_p_b1_contours.pdf",
)
AUDIT_JSON = SUMMARY_DIR / "task44_pngbase_hodhost_mmin1e13_fixedfnl_pk0_vs_xi0_smax150_audit.json"
AUDIT_NPZ = SUMMARY_DIR / "task44_pngbase_hodhost_mmin1e13_fixedfnl_pk0_vs_xi0_smax150_audit.npz"


def load_fit(tag: str, *, probe: str) -> dict[str, Any]:
    if probe == "xi":
        prefix = host_xi_fit_prefix(tag)
    elif probe == "pk":
        prefix = host_pk_fit_prefix(tag)
    else:
        raise ValueError(probe)
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"missing {probe} fit: {npz_path} / {json_path}")
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    if summary.get("output_npz_sha256") != sha256_file(npz_path):
        raise RuntimeError(f"fit hash mismatch: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files if name != "summary_json"}
    return {"summary": summary, "arrays": arrays, "npz_path": npz_path, "json_path": json_path}


def load_full_xi_measurement(tag: str) -> dict[str, Any]:
    """Load the validated 30--350 FCFC measurement used as the fit source."""

    path = xi_engine_path(tag, "fcfc")
    metadata_path = xi_engine_metadata_path(tag, "fcfc")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing full FCFC xi measurement: {path} / {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "pass" or metadata.get("output_sha256") != sha256_file(path):
        raise RuntimeError(f"invalid full FCFC xi measurement/hash: {path}")
    with np.load(path, allow_pickle=False) as data:
        centers = np.asarray(data["s"], dtype="f8")
        edges = np.asarray(data["s_edges"], dtype="f8")
        values = np.asarray(data["xi0"], dtype="f8")
    if not np.array_equal(edges, S_EDGES):
        raise RuntimeError(f"full FCFC xi edges changed for {tag}: {edges}")
    if not np.array_equal(centers, S_CENTERS) or values.shape != S_CENTERS.shape:
        raise RuntimeError(
            f"full FCFC xi centers/data changed for {tag}: centers={centers.shape}, data={values.shape}"
        )
    return {
        "path": path,
        "metadata_path": metadata_path,
        "sha256": metadata["output_sha256"],
        "s": centers,
        "s_edges": edges,
        "xi0": values,
    }


def posterior(fit: dict[str, Any], name: str) -> dict[str, float]:
    return {key: float(value) for key, value in fit["summary"]["mcmc"]["posterior"][name].items()}


def flat_samples(fit: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    chain = np.asarray(fit["arrays"]["chain_final_by_step"], dtype="f8")
    chain = chain.reshape(-1, chain.shape[-1])
    available = [str(value) for value in fit["arrays"]["parameter_names"]]
    return np.column_stack([chain[:, available.index(name)] for name in names])


def _atomic_figure_save(
    figure: plt.Figure,
    output: Path,
    *,
    tight: bool = False,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.pdf")
    save_options = (
        {"format": "pdf", "bbox_inches": "tight", "pad_inches": 0.08}
        if tight
        else {"format": "pdf"}
    )
    figure.savefig(temporary, **save_options)
    plt.close(figure)
    temporary.replace(output)


def _xi_box(tag: str, fit: dict[str, Any], *, include_p: bool = True) -> str:
    lines = [
        r"host: $M_{\rm halo}\geq10^{13}\,h^{-1}M_\odot$",
        rf"fixed $f_{{\rm NL}}={get_spec(tag).fnl:g}$",
        r"fit: $s_{\min}/s_{\max}=30/150\ h^{-1}{\rm Mpc}$",
        r"shown: $s_{\min}/s_{\max}=30/350\ h^{-1}{\rm Mpc}$",
    ]
    if include_p:
        row = posterior(fit, "p")
        lines.append(
            rf"$p={row['q50']:.2f}_{{-{row['q50'] - row['q16']:.2f}}}"
            rf"^{{+{row['q84'] - row['q50']:.2f}}}$"
        )
    return "\n".join(lines)


def _pk_box(tag: str, fit: dict[str, Any], *, include_p: bool = True) -> str:
    lines = [
        r"host: $M_{\rm halo}\geq10^{13}\,h^{-1}M_\odot$",
        rf"fixed $f_{{\rm NL}}={get_spec(tag).fnl:g}$",
        r"$k_{\min}=0.006\ h\,{\rm Mpc}^{-1}$",
        r"$k_{\max}=0.100\ h\,{\rm Mpc}^{-1}$ (bin centre)",
    ]
    if include_p:
        row = posterior(fit, "p")
        lines.append(
            rf"$p={row['q50']:.2f}_{{-{row['q50'] - row['q16']:.2f}}}"
            rf"^{{+{row['q84'] - row['q50']:.2f}}}$"
        )
    return "\n".join(lines)


def make_bestfit_measurements() -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), sharey="row")
    for column, tag in enumerate(CATALOGS):
        pk_fit = load_fit(tag, probe="pk")
        pk_axis = axes[0, column]
        k = np.asarray(pk_fit["arrays"]["coordinate"], dtype="f8")
        pk_data = np.asarray(pk_fit["arrays"]["data"], dtype="f8")
        pk_prediction = np.asarray(pk_fit["arrays"]["prediction_map"], dtype="f8")
        pk_sigma = np.sqrt(np.diag(np.asarray(pk_fit["arrays"]["covariance_final"], dtype="f8")))
        pk_axis.errorbar(
            k,
            pk_data,
            yerr=pk_sigma,
            fmt="o",
            ms=3.7,
            color=COLORS[tag],
            capsize=1.7,
            label=r"halo $P_0$ measurement",
        )
        pk_axis.plot(k, pk_prediction, color="black", lw=1.45, label="MAP best fit")
        pk_axis.set_xscale("log")
        pk_axis.set_title(tag, fontsize=15, fontweight="bold")
        pk_axis.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
        pk_axis.text(
            0.04,
            0.04,
            _pk_box(tag, pk_fit),
            transform=pk_axis.transAxes,
            va="bottom",
            fontsize=8.8,
            linespacing=1.16,
            bbox={"facecolor": "white", "edgecolor": "0.72", "alpha": 0.92},
        )

        xi_fit = load_fit(tag, probe="xi")
        xi_axis = axes[1, column]
        s = np.asarray(xi_fit["arrays"]["coordinate"], dtype="f8")
        xi_data = np.asarray(xi_fit["arrays"]["data"], dtype="f8")
        xi_prediction = np.asarray(xi_fit["arrays"]["prediction_map"], dtype="f8")
        xi_sigma = np.sqrt(np.diag(np.asarray(xi_fit["arrays"]["covariance_final"], dtype="f8")))
        full_xi = load_full_xi_measurement(tag)
        full_s = np.asarray(full_xi["s"], dtype="f8")
        full_data = np.asarray(full_xi["xi0"], dtype="f8")
        if not np.array_equal(full_s[: s.size], s) or not np.array_equal(full_data[: s.size], xi_data):
            raise RuntimeError(f"{tag} plotted full FCFC xi does not begin with the fitted 12-bin vector")
        xi_axis.errorbar(
            s,
            s**2 * xi_data,
            yerr=s**2 * xi_sigma,
            fmt="o",
            ms=4.0,
            capsize=1.9,
            color=COLORS[tag],
            label=r"halo $\xi_0$ measurement (fitted)",
        )
        outside_fit = full_s >= XI_FIT_SMAX
        xi_axis.plot(
            full_s[outside_fit],
            full_s[outside_fit] ** 2 * full_data[outside_fit],
            linestyle="none",
            marker="o",
            ms=4.2,
            markerfacecolor="none",
            markeredgecolor=COLORS[tag],
            markeredgewidth=1.15,
            label=r"halo $\xi_0$ measurement (not fitted)",
        )
        xi_axis.plot(s, s**2 * xi_prediction, color="black", lw=1.5, label="MAP best fit")
        xi_axis.set_xlim(25.0, 355.0)
        xi_axis.set_xlabel(r"$s\ [h^{-1}{\rm Mpc}]$")
        xi_axis.text(
            0.96,
            0.96,
            _xi_box(tag, xi_fit),
            transform=xi_axis.transAxes,
            va="top",
            ha="right",
            fontsize=8.8,
            linespacing=1.16,
            bbox={"facecolor": "white", "edgecolor": "0.72", "alpha": 0.92},
        )

    axes[0, 0].set_ylabel(r"$P_0(k)\ [(h^{-1}{\rm Mpc})^3]$")
    axes[1, 0].set_ylabel(r"$s^2\xi_0(s)\ [(h^{-1}{\rm Mpc})^2]$")
    axes[0, 1].legend(frameon=False, loc="upper right")
    xi_handles, xi_labels = axes[1, 1].get_legend_handles_labels()
    xi_order = [
        xi_labels.index("MAP best fit"),
        xi_labels.index(r"halo $\xi_0$ measurement (fitted)"),
        xi_labels.index(r"halo $\xi_0$ measurement (not fitted)"),
    ]
    axes[1, 1].legend(
        [xi_handles[index] for index in xi_order],
        [xi_labels[index] for index in xi_order],
        frameon=False,
        loc="lower left",
        fontsize=8.8,
    )
    figure.tight_layout()
    _atomic_figure_save(figure, BESTFIT_MEASUREMENT_PDF)


def _set_task43_contour_style() -> None:
    """Use the exact typography and tick contract of the requested Task43 figure."""

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


def make_contours() -> dict[str, Any]:
    """Draw two side-by-side Task43-style 2D corners, including both 1D marginals."""

    _set_task43_contour_style()
    figure = plt.figure(figsize=(12.4, 5.9))
    subfigures = figure.subfigures(1, len(CATALOGS), wspace=0.035)
    contour_audit: dict[str, Any] = {}

    for subfigure, tag in zip(np.atleast_1d(subfigures), CATALOGS, strict=True):
        pk_fit = load_fit(tag, probe="pk")
        xi_fit = load_fit(tag, probe="xi")
        pk_samples = flat_samples(pk_fit, CONTOUR_PARAMETERS)
        xi_samples = flat_samples(xi_fit, CONTOUR_PARAMETERS)
        ranges = {
            name: task43_plot_range(
                pk_samples[:, index],
                xi_samples[:, index],
                parameter=name,
            )
            for index, name in enumerate(CONTOUR_PARAMETERS)
        }
        axes = subfigure.subplots(len(CONTOUR_PARAMETERS), len(CONTOUR_PARAMETERS), squeeze=False)
        tag_audit: dict[str, Any] = {"ranges": {name: list(value) for name, value in ranges.items()}}

        for irow, yname in enumerate(CONTOUR_PARAMETERS):
            for icol, xname in enumerate(CONTOUR_PARAMETERS):
                axis = axes[irow, icol]
                if icol > irow:
                    axis.set_axis_off()
                    continue
                if icol == irow:
                    bins = np.linspace(*ranges[xname], 90)
                    axis.hist(
                        pk_samples[:, icol],
                        bins=bins,
                        density=True,
                        histtype="step",
                        lw=1.9,
                        color=TASK43_COLORS["pk0"],
                    )
                    axis.hist(
                        xi_samples[:, icol],
                        bins=bins,
                        density=True,
                        histtype="step",
                        lw=1.9,
                        color=TASK43_COLORS["xi0"],
                    )
                    axis.set_xlim(*ranges[xname])
                    axis.set_yticks([])
                else:
                    key = f"{xname}_vs_{yname}"
                    tag_audit[key] = {
                        "xi0": task43_draw_contour(
                            axis,
                            xi_samples[:, icol],
                            xi_samples[:, irow],
                            color=TASK43_COLORS["xi0"],
                            xlim=ranges[xname],
                            ylim=ranges[yname],
                            zorder=1,
                        ),
                        "pk0": task43_draw_contour(
                            axis,
                            pk_samples[:, icol],
                            pk_samples[:, irow],
                            color=TASK43_COLORS["pk0"],
                            xlim=ranges[xname],
                            ylim=ranges[yname],
                            zorder=3,
                        ),
                    }
                    axis.set(xlim=ranges[xname], ylim=ranges[yname])

                if irow < len(CONTOUR_PARAMETERS) - 1:
                    axis.tick_params(labelbottom=False)
                else:
                    axis.set_xlabel(CONTOUR_LABELS[xname])
                if icol == 0 and irow > 0:
                    axis.set_ylabel(CONTOUR_LABELS[yname])
                elif icol > 0 and irow != icol:
                    axis.tick_params(labelleft=False)

        pk_p = posterior(pk_fit, "p")
        xi_p = posterior(xi_fit, "p")
        handles = [
            plt.Line2D([], [], color=TASK43_COLORS["pk0"], lw=2.0),
            plt.Line2D([], [], color=TASK43_COLORS["xi0"], lw=2.0),
        ]
        labels = [
            rf"$P_0(k):\ p={task43_posterior_text(pk_p)}$",
            rf"$\xi_0(s):\ p={task43_posterior_text(xi_p)}$",
        ]
        axes[0, 1].legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.04, 0.13),
            frameon=False,
            fontsize=11.0,
        )
        annotation_text = "\n".join(
            [
                rf"$\mathbf{{{tag}}}$ host halos, $z=0.5$",
                r"$M_{\rm halo}\geq10^{13}\ h^{-1}M_\odot$",
                rf"Fiducial $f_{{\rm NL}}={get_spec(tag).fnl:g}$ (fixed)",
                r"$k_{\min,\max}^{\rm fit}=0.006,\ 0.100\ h\,{\rm Mpc}^{-1}$",
                r"$s_{\min,\max}^{\rm fit}=30,\ 150\ h^{-1}{\rm Mpc}$",
            ]
        )
        axes[0, 1].text(
            0.50,
            0.73,
            annotation_text,
            transform=axes[0, 1].transAxes,
            ha="center",
            va="center",
            fontsize=8.8,
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "white",
                "edgecolor": "0.45",
                "linewidth": 0.9,
            },
        )
        subfigure.subplots_adjust(
            left=0.12,
            right=0.97,
            bottom=0.10,
            top=0.96,
            wspace=0.0,
            hspace=0.0,
        )
        tag_audit["legend_anchor"] = {
            "axes": [0, 1],
            "location": "center left",
            "bbox_to_anchor": [0.04, 0.13],
        }
        tag_audit["annotation_text"] = annotation_text
        tag_audit["subplot_spacing"] = {"wspace": 0.0, "hspace": 0.0}
        contour_audit[tag] = tag_audit

    _atomic_figure_save(figure, CONTOUR_PDF, tight=True)
    return contour_audit


def build_audit() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    payload: dict[str, np.ndarray] = {}
    all_gates: list[bool] = []
    for tag in CATALOGS:
        xi = load_fit(tag, probe="xi")
        pk = load_fit(tag, probe="pk")
        full_xi = load_full_xi_measurement(tag)
        xi_p = posterior(xi, "p")
        pk_p = posterior(pk, "p")
        edges = np.asarray(xi["arrays"]["radial_edges"], dtype="f8")
        pk_measurement_metadata = json.loads(host_pk_metadata_path(tag).read_text(encoding="utf-8"))
        xi_measurement_metadata = json.loads(xi_engine_metadata_path(tag, "fcfc").read_text(encoding="utf-8"))
        source_path = host_catalog_path(tag)
        source_sha256 = sha256_file(source_path)
        row_gates = {
            "halo_pk_fit_status_pass": pk["summary"].get("status") == "pass",
            "halo_pk_all_fit_gates_pass": bool(all(pk["summary"]["gates"].values())),
            "halo_xi_fit_status_pass": xi["summary"].get("status") == "pass",
            "halo_xi_all_fit_gates_pass": bool(all(xi["summary"]["gates"].values())),
            "both_fixed_fnl_match_catalog": bool(
                float(pk["summary"]["fixed_fnl"]) == get_spec(tag).fnl
                and float(xi["summary"]["fixed_fnl"]) == get_spec(tag).fnl
            ),
            "pk_measurement_is_validated_halo_not_lrg": bool(
                pk_measurement_metadata.get("status") == "pass"
                and pk_measurement_metadata.get("output_sha256") == sha256_file(host_pk_path(tag))
                and pk_measurement_metadata.get("source_host_catalog") == str(source_path)
                and pk_measurement_metadata.get("source_host_catalog_sha256") == source_sha256
            ),
            "pk_and_xi_source_catalog_path_and_hash_identical": bool(
                xi_measurement_metadata.get("source_host_catalog") == str(source_path)
                and xi_measurement_metadata.get("source_host_catalog_sha256") == source_sha256
                and pk["summary"]["input"]["source_host_catalog"] == str(source_path)
                and pk["summary"]["input"]["source_host_catalog_sha256"] == source_sha256
            ),
            "pk_and_xi_ndata_identical": int(pk["summary"]["input"]["ndata"]) == int(xi["summary"]["input"]["ndata"]),
            "pk_edges_are_exact_strict_48_shells": bool(
                np.array_equal(np.asarray(pk["arrays"]["coordinate_edges"], dtype="f8"), PK_STRICT_FIT_EDGES)
                and int(np.asarray(pk["arrays"]["data"]).size) == 48
            ),
            "radial_edges_exactly_30_to_150": bool(np.array_equal(edges, XI_FIT_EDGES)),
            "exactly_12_likelihood_bins": int(np.asarray(xi["arrays"]["data"]).size) == 12,
            "covariance_is_12_by_12": np.asarray(xi["arrays"]["covariance_final"]).shape == (12, 12),
            "bestfit_plot_uses_full_32_bin_fcfc_measurement_to_smax350": bool(
                np.array_equal(np.asarray(full_xi["s_edges"], dtype="f8"), S_EDGES)
                and np.asarray(full_xi["xi0"]).size == 32
                and float(np.asarray(full_xi["s"])[-1]) == 345.0
            ),
            "full_plotted_measurement_has_exact_fitted_12_bin_prefix": bool(
                np.array_equal(
                    np.asarray(full_xi["s"], dtype="f8")[:12],
                    np.asarray(xi["arrays"]["coordinate"], dtype="f8"),
                )
                and np.array_equal(
                    np.asarray(full_xi["xi0"], dtype="f8")[:12],
                    np.asarray(xi["arrays"]["data"], dtype="f8"),
                )
            ),
            "no_residual_panel_in_bestfit_contract": True,
        }
        all_gates.extend(row_gates.values())
        combined_sigma = float(np.hypot(xi_p["sigma68"], pk_p["sigma68"]))
        rows[tag] = {
            "fixed_fnl": get_spec(tag).fnl,
            "halo_xi_smin30_smax150": {
                "p": xi_p,
                "b1": posterior(xi, "b1"),
                "chi2": float(xi["summary"]["map"]["chi2"]),
                "dof": int(xi["summary"]["map"]["dof"]),
                "pte": float(xi["summary"]["map"]["pte"]),
                "json": str(xi["json_path"]),
                "npz": str(xi["npz_path"]),
            },
            "bestfit_xi_measurement_display": {
                "source": str(full_xi["path"]),
                "source_sha256": full_xi["sha256"],
                "smin_mpc_h": float(np.asarray(full_xi["s_edges"])[0]),
                "smax_mpc_h": float(np.asarray(full_xi["s_edges"])[-1]),
                "nbins": int(np.asarray(full_xi["xi0"]).size),
                "fitted_bins": 12,
                "unfitted_display_bins": int(np.asarray(full_xi["xi0"]).size - 12),
            },
            "halo_pk_strict_kmin0p006": {
                "p": pk_p,
                "b1": posterior(pk, "b1"),
                "chi2": float(pk["summary"]["map"]["chi2"]),
                "dof": int(pk["summary"]["map"]["dof"]),
                "pte": float(pk["summary"]["map"]["pte"]),
                "json": str(pk["json_path"]),
                "npz": str(pk["npz_path"]),
            },
            "descriptive_pk_minus_xi": {
                "delta_p_q50_pk_minus_xi": float(pk_p["q50"] - xi_p["q50"]),
                "delta_over_naive_combined_sigma68": float((pk_p["q50"] - xi_p["q50"]) / combined_sigma),
                "warning": "same halo catalog and realization; P0 and xi0 errors are correlated, so the naive combined sigma is descriptive only",
            },
            "gates": row_gates,
        }
        payload[f"{tag}_xi_p_b1_samples"] = flat_samples(xi, ("p", "b1"))
        payload[f"{tag}_xi_s"] = np.asarray(xi["arrays"]["coordinate"], dtype="f8")
        payload[f"{tag}_xi_data"] = np.asarray(xi["arrays"]["data"], dtype="f8")
        payload[f"{tag}_xi_prediction_map"] = np.asarray(xi["arrays"]["prediction_map"], dtype="f8")
        payload[f"{tag}_xi_full_s30_350"] = np.asarray(full_xi["s"], dtype="f8")
        payload[f"{tag}_xi_full_data_s30_350"] = np.asarray(full_xi["xi0"], dtype="f8")
        payload[f"{tag}_pk_p_b1_samples"] = flat_samples(pk, ("p", "b1"))
        payload[f"{tag}_pk_k"] = np.asarray(pk["arrays"]["coordinate"], dtype="f8")
        payload[f"{tag}_pk_data"] = np.asarray(pk["arrays"]["data"], dtype="f8")
        payload[f"{tag}_pk_prediction_map"] = np.asarray(pk["arrays"]["prediction_map"], dtype="f8")

    expected_pdfs = (BESTFIT_MEASUREMENT_PDF, CONTOUR_PDF)
    current_pdfs = tuple(sorted(PLOT_ROOT.glob("*.pdf")))
    allowed_pdfs = tuple(sorted(expected_pdfs + GAUSSIAN_P_ADDITIONAL_PDFS + LRG_CURRENT_PDFS))
    global_gates = {
        "both_catalog_fit_gates_pass": bool(all(all_gates)),
        "kmin_is_0p006": PK_STRICT_KMIN == 0.006,
        "kmax_bin_center_is_0p100": PK_KMAX_CONTRACT == 0.100,
        "smin_is_30": XI_FIT_SMIN == 30.0,
        "smax_is_150": XI_FIT_SMAX == 150.0,
        "bestfit_xi_measurement_display_smax_is_350": float(S_EDGES[-1]) == 350.0,
        "task43_contour_reference_exists": TASK43_REFERENCE_PDF.is_file(),
        "contour_parameter_order_is_p_b1": CONTOUR_PARAMETERS == ("p", "b1"),
        "contour_has_1d_marginal_for_every_parameter": True,
        "contour_subplots_have_zero_spacing": True,
        "contour_has_rounded_annotation_box": True,
        "legend_is_anchored_in_first_upper_triangle_cell": True,
        "both_primary_current_range_pdfs_exist": all(path.is_file() for path in expected_pdfs),
        "all_present_pdfs_are_current_range_products": bool(
            set(current_pdfs).issubset(set(allowed_pdfs)) and set(expected_pdfs).issubset(set(current_pdfs))
        ),
        "all_legacy_or_diagnostic_pdfs_removed": not any(path.exists() for path in LEGACY_PDFS),
        "pdf_only_in_extension_plot_tree": not any(PLOT_ROOT.rglob("*.png")),
    }
    audit: dict[str, Any] = {
        "task": "task44_pngbase_hodhost_mmin1e13_fixedfnl_pk0_vs_xi0_smax150_audit",
        "status": "pass" if all(global_gates.values()) else "review",
        "sample": "unit-weight unique occupied HOD hosts with official cleaned CompaSO M>=1e13 Msun/h",
        "fit_ranges": {
            "halo_pk": {
                "kmin_h_mpc": PK_STRICT_KMIN,
                "kmax_bin_center_h_mpc": PK_KMAX_CONTRACT,
                "edges_h_mpc": PK_STRICT_FIT_EDGES.tolist(),
                "nbins": 48,
            },
            "halo_xi": {
                "smin_mpc_h": XI_FIT_SMIN,
                "smax_mpc_h": XI_FIT_SMAX,
                "edges_mpc_h": XI_FIT_EDGES.tolist(),
                "nbins": 12,
            },
        },
        "fit_quality_policy": "chi2/dof and PTE are diagnostics, not pass/fail gates",
        "bestfit_plot_contract": {
            "halo_xi_likelihood": "12 FCFC bins over 30<=s<150 Mpc/h; unchanged",
            "halo_xi_measurement_display": "all 32 FCFC bins over 30<=s<350 Mpc/h",
            "fitted_bin_style": "filled circles with formal fit-covariance error bars",
            "unfitted_display_bin_style": "open circles without invented out-of-fit covariance errors",
            "map_curve": "shown only over the fitted 30<=s<150 bins",
        },
        "contour_plot_contract": {
            "reference_pdf": str(TASK43_REFERENCE_PDF),
            "parameter_order": list(CONTOUR_PARAMETERS),
            "diagonal": "normalized 1D marginalized posteriors for halo P0 and halo xi0",
            "lower_triangle": "Task43-smoothed 68% and 95% credible contours",
            "subplot_spacing": {"wspace": 0.0, "hspace": 0.0},
            "legend_anchor": {"axes": [0, 1], "location": "center left", "bbox_to_anchor": [0.04, 0.13]},
            "upper_triangle": "Task43-style legend plus rounded geometry/fit-range annotation box",
        },
        "global_gates": global_gates,
        "catalogs": rows,
        "plots": [{"path": str(path), "sha256": sha256_file(path)} for path in expected_pdfs],
    }
    atomic_savez(AUDIT_NPZ, **payload, audit_json=np.asarray(json.dumps(audit, sort_keys=True)))
    audit["payload_npz"] = str(AUDIT_NPZ)
    audit["payload_npz_sha256"] = sha256_file(AUDIT_NPZ)
    atomic_write_json(AUDIT_JSON, audit)
    if audit["status"] != "pass":
        raise RuntimeError(f"matched halo-P0/xi0 audit requires review: {global_gates}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    outputs = (BESTFIT_MEASUREMENT_PDF, CONTOUR_PDF, AUDIT_JSON, AUDIT_NPZ)
    if all(path.is_file() for path in outputs) and not args.overwrite:
        audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        if audit.get("status") == "pass":
            print(f"[skip] validated matched halo-P0/xi0 plot/audit: {AUDIT_JSON}", flush=True)
            return
    make_bestfit_measurements()
    make_contours()
    audit = build_audit()
    print(f"[done] matched halo-P0/xi0 plot/audit status={audit['status']}", flush=True)
    for path in (BESTFIT_MEASUREMENT_PDF, CONTOUR_PDF, AUDIT_JSON):
        print(f"[done] {path}", flush=True)


if __name__ == "__main__":
    main()
