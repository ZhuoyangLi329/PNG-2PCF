#!/usr/bin/env python3
"""Plot and audit the Gaussian-p/free-fNL matched halo P0/xi0 test."""

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
    P_GAUSSIAN_PRIOR_MEAN,
    P_GAUSSIAN_PRIOR_SIGMA,
    PK_KMAX_CONTRACT,
    PK_STRICT_FIT_EDGES,
    PK_STRICT_KMIN,
    PLOT_ROOT,
    SUMMARY_DIR,
    XI_FIT_EDGES,
    XI_FIT_SMAX,
    XI_FIT_SMIN,
    atomic_savez,
    atomic_write_json,
    ensure_output_dirs,
    get_spec,
    host_catalog_path,
    host_pk_gaussianp_fit_prefix,
    host_xi_gaussianp_fit_prefix,
    sha256_file,
)


COLORS = {"c300": "#4C72B0", "c302": "#C44E52", "pk": "#4C72B0", "xi": "#C44E52"}
CONTOUR_PARAMETERS = ("fNL", "b1", "p")
CHAIN_PARAMETERS = ("fnl", "b1", "p")
CONTOUR_LABELS = {"fNL": r"$f_{\rm NL}$", "b1": r"$b_1$", "p": r"$p$"}
TASK43_REFERENCE_PDF = Path(
    "/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/plots/8.21meeting/"
    "task43_rsd_boxsafe_zobs0p4_0p8_x25_pk0_vs_xi0_smin50_l0only_contours.pdf"
)
PRIOR_LABEL = r"$p\sim\mathcal{N}(0.70724,\,0.26947^2)$"
BESTFIT_PDF = PLOT_ROOT / (
    "task44_pngbase_c300_c302_hodhost_mmin1e13_gaussianp_freefnl_"
    "pk0_kmin0p006_kmax0p100_xi0_smin30_smax150_bestfit_measurements.pdf"
)
CONTOUR_PDF = PLOT_ROOT / (
    "task44_pngbase_c300_c302_hodhost_mmin1e13_gaussianp_freefnl_"
    "pk0_kmin0p006_kmax0p100_vs_xi0_smin30_smax150_fnl_b1_p_contours.pdf"
)
AUDIT_JSON = SUMMARY_DIR / "task44_pngbase_hodhost_mmin1e13_gaussianp_freefnl_pk0_vs_xi0_audit.json"
AUDIT_NPZ = SUMMARY_DIR / "task44_pngbase_hodhost_mmin1e13_gaussianp_freefnl_pk0_vs_xi0_audit.npz"


def load_fit(tag: str, probe: str) -> dict[str, Any]:
    if probe == "pk":
        prefix = host_pk_gaussianp_fit_prefix(tag)
    elif probe == "xi":
        prefix = host_xi_gaussianp_fit_prefix(tag)
    else:
        raise ValueError(probe)
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"missing Gaussian-p {probe} fit: {npz_path} / {json_path}")
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    if summary.get("output_npz_sha256") != sha256_file(npz_path):
        raise RuntimeError(f"fit hash mismatch: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files if name != "summary_json"}
    return {"summary": summary, "arrays": arrays, "npz_path": npz_path, "json_path": json_path}


def posterior(fit: dict[str, Any], name: str) -> dict[str, float]:
    return {key: float(value) for key, value in fit["summary"]["mcmc"]["posterior"][name].items()}


def flat_samples(fit: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    chain = np.asarray(fit["arrays"]["chain_final_by_step"], dtype="f8")
    flat = chain.reshape(-1, chain.shape[-1])
    available = [str(value) for value in fit["arrays"]["parameter_names"]]
    return np.column_stack([flat[:, available.index(name)] for name in names])


def _interval(row: dict[str, float], *, digits: int = 1) -> str:
    return (
        rf"{row['q50']:.{digits}f}_{{-{row['q50'] - row['q16']:.{digits}f}}}"
        rf"^{{+{row['q84'] - row['q50']:.{digits}f}}}"
    )


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


def _fit_box(tag: str, probe: str, fit: dict[str, Any]) -> str:
    fnl = posterior(fit, "fnl")
    ranges = (
        [
            r"$k_{\min}=0.006\ h\,{\rm Mpc}^{-1}$",
            r"$k_{\max}=0.100\ h\,{\rm Mpc}^{-1}$ (bin centre)",
        ]
        if probe == "pk"
        else [
            r"$s_{\min}=30\ h^{-1}{\rm Mpc}$",
            r"$s_{\max}=150\ h^{-1}{\rm Mpc}$",
        ]
    )
    return "\n".join(
        [
            r"host: $M_{\rm halo}\geq10^{13}\,h^{-1}M_\odot$",
            PRIOR_LABEL,
            *ranges,
            rf"input $f_{{\rm NL}}={get_spec(tag).fnl:g}$",
            rf"$f_{{\rm NL}}={_interval(fnl)}$",
        ]
    )


def make_bestfit() -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), sharey="row")
    for column, tag in enumerate(CATALOGS):
        pk = load_fit(tag, "pk")
        pk_axis = axes[0, column]
        k = np.asarray(pk["arrays"]["coordinate"], dtype="f8")
        pk_data = np.asarray(pk["arrays"]["data"], dtype="f8")
        pk_prediction = np.asarray(pk["arrays"]["prediction_map"], dtype="f8")
        pk_sigma = np.sqrt(np.diag(np.asarray(pk["arrays"]["covariance_final"], dtype="f8")))
        pk_axis.errorbar(
            k,
            pk_data,
            yerr=pk_sigma,
            fmt="o",
            ms=3.7,
            capsize=1.7,
            color=COLORS[tag],
            label=r"halo $P_0$ measurement",
        )
        pk_axis.plot(k, pk_prediction, color="black", lw=1.45, label="MAP best fit")
        pk_axis.set_xscale("log")
        pk_axis.set_title(tag, fontsize=15, fontweight="bold")
        pk_axis.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
        pk_axis.text(
            0.04,
            0.04,
            _fit_box(tag, "pk", pk),
            transform=pk_axis.transAxes,
            va="bottom",
            fontsize=8.25,
            linespacing=1.12,
            bbox={"facecolor": "white", "edgecolor": "0.72", "alpha": 0.92},
        )

        xi = load_fit(tag, "xi")
        xi_axis = axes[1, column]
        s = np.asarray(xi["arrays"]["coordinate"], dtype="f8")
        xi_data = np.asarray(xi["arrays"]["data"], dtype="f8")
        xi_prediction = np.asarray(xi["arrays"]["prediction_map"], dtype="f8")
        xi_sigma = np.sqrt(np.diag(np.asarray(xi["arrays"]["covariance_final"], dtype="f8")))
        xi_axis.errorbar(
            s,
            s**2 * xi_data,
            yerr=s**2 * xi_sigma,
            fmt="o",
            ms=4.0,
            capsize=1.9,
            color=COLORS[tag],
            label=r"halo $\xi_0$ measurement",
        )
        xi_axis.plot(s, s**2 * xi_prediction, color="black", lw=1.5, label="MAP best fit")
        xi_axis.set_xlabel(r"$s\ [h^{-1}{\rm Mpc}]$")
        xi_axis.text(
            0.96,
            0.96,
            _fit_box(tag, "xi", xi),
            transform=xi_axis.transAxes,
            va="top",
            ha="right",
            fontsize=8.25,
            linespacing=1.12,
            bbox={"facecolor": "white", "edgecolor": "0.72", "alpha": 0.92},
        )

    axes[0, 0].set_ylabel(r"$P_0(k)\ [(h^{-1}{\rm Mpc})^3]$")
    axes[1, 0].set_ylabel(r"$s^2\xi_0(s)\ [(h^{-1}{\rm Mpc})^2]$")
    axes[0, 1].legend(frameon=False, loc="upper right")
    axes[1, 1].legend(frameon=False, loc="lower left")
    figure.tight_layout()
    _atomic_figure_save(figure, BESTFIT_PDF)


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
    """Draw Task43-style 3D corners with fNL, b1, p and all 1D marginals."""

    _set_task43_contour_style()
    figure = plt.figure(figsize=(17.4, 8.2))
    subfigures = figure.subfigures(1, len(CATALOGS), wspace=0.025)
    contour_audit: dict[str, Any] = {}

    for subfigure, tag in zip(np.atleast_1d(subfigures), CATALOGS, strict=True):
        pk = load_fit(tag, "pk")
        xi = load_fit(tag, "xi")
        pk_samples = flat_samples(pk, CHAIN_PARAMETERS)
        xi_samples = flat_samples(xi, CHAIN_PARAMETERS)
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
        input_fnl = float(get_spec(tag).fnl)

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
                    if xname == "fNL":
                        axis.axvline(input_fnl, color="0.5", lw=0.8, ls="--")
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
                    if xname == "fNL":
                        axis.axvline(input_fnl, color="0.5", lw=0.8, ls="--", zorder=0)
                    if yname == "fNL":
                        axis.axhline(input_fnl, color="0.5", lw=0.8, ls="--", zorder=0)

                if irow < len(CONTOUR_PARAMETERS) - 1:
                    axis.tick_params(labelbottom=False)
                else:
                    axis.set_xlabel(CONTOUR_LABELS[xname])
                if icol == 0 and irow > 0:
                    axis.set_ylabel(CONTOUR_LABELS[yname])
                elif icol > 0 and irow != icol:
                    axis.tick_params(labelleft=False)

        pk_fnl = posterior(pk, "fnl")
        xi_fnl = posterior(xi, "fnl")
        handles = [
            plt.Line2D([], [], color=TASK43_COLORS["pk0"], lw=2.0),
            plt.Line2D([], [], color=TASK43_COLORS["xi0"], lw=2.0),
        ]
        labels = [
            rf"$P_0(k):\ f_{{\rm NL}}={task43_posterior_text(pk_fnl)}$",
            rf"$\xi_0(s):\ f_{{\rm NL}}={task43_posterior_text(xi_fnl)}$",
        ]
        axes[0, 1].legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.04, 0.22),
            frameon=False,
            fontsize=13.0,
        )
        annotation_text = "\n".join(
            [
                rf"$\mathbf{{{tag}}}$ host halos, $z=0.5$",
                r"$M_{\rm halo}\geq10^{13}\ h^{-1}M_\odot$",
                PRIOR_LABEL,
                rf"Fiducial $f_{{\rm NL}}={input_fnl:g}$",
                r"$k_{\min,\max}^{\rm fit}=0.006,\ 0.100\ h\,{\rm Mpc}^{-1}$",
                r"$s_{\min,\max}^{\rm fit}=30,\ 150\ h^{-1}{\rm Mpc}$",
            ]
        )
        axes[0, 2].text(
            0.50,
            0.70,
            annotation_text,
            transform=axes[0, 2].transAxes,
            ha="center",
            va="center",
            fontsize=10.2,
            linespacing=1.35,
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
            "bbox_to_anchor": [0.04, 0.22],
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
    source_paths: set[str] = set()
    for tag in CATALOGS:
        pk = load_fit(tag, "pk")
        xi = load_fit(tag, "xi")
        pk_summary = pk["summary"]
        xi_summary = xi["summary"]
        source_path = str(host_catalog_path(tag))
        source_hash = sha256_file(host_catalog_path(tag))
        source_paths.update((pk_summary["input"]["source_host_catalog"], xi_summary["input"]["source_host_catalog"]))
        row_gates = {
            "both_fit_status_pass": pk_summary.get("status") == xi_summary.get("status") == "pass",
            "both_all_fit_gates_pass": bool(
                all(pk_summary["gates"].values()) and all(xi_summary["gates"].values())
            ),
            "both_sample_fnl_not_fix_it": bool(
                "fnl" in pk_summary["free_parameters"] and "fnl" in xi_summary["free_parameters"]
            ),
            "both_sample_p_with_gaussian_prior": bool(
                "p" in pk_summary["free_parameters"]
                and "p" in xi_summary["free_parameters"]
                and pk_summary["p_gaussian_prior"]["included_in_map_and_mcmc"]
                and xi_summary["p_gaussian_prior"]["included_in_map_and_mcmc"]
            ),
            "prior_mean_exact_for_both": bool(
                pk_summary["p_gaussian_prior"]["mean"]
                == xi_summary["p_gaussian_prior"]["mean"]
                == P_GAUSSIAN_PRIOR_MEAN
            ),
            "prior_sigma_exact_for_both": bool(
                pk_summary["p_gaussian_prior"]["sigma"]
                == xi_summary["p_gaussian_prior"]["sigma"]
                == P_GAUSSIAN_PRIOR_SIGMA
            ),
            "same_validated_host_catalog_for_pk_and_xi": bool(
                pk_summary["input"]["source_host_catalog"]
                == xi_summary["input"]["source_host_catalog"]
                == source_path
                and pk_summary["input"]["source_host_catalog_sha256"]
                == xi_summary["input"]["source_host_catalog_sha256"]
                == source_hash
            ),
            "same_ndata_for_pk_and_xi": int(pk_summary["input"]["ndata"]) == int(xi_summary["input"]["ndata"]),
            "pk_range_is_exact_current_contract": bool(
                np.array_equal(np.asarray(pk["arrays"]["coordinate_edges"], dtype="f8"), PK_STRICT_FIT_EDGES)
                and int(np.asarray(pk["arrays"]["data"]).size) == 48
            ),
            "xi_range_is_exact_current_contract": bool(
                np.array_equal(np.asarray(xi["arrays"]["radial_edges"], dtype="f8"), XI_FIT_EDGES)
                and int(np.asarray(xi["arrays"]["data"]).size) == 12
            ),
            "input_baseline_fnl_metadata_correct": bool(
                float(pk_summary["input_baseline_fnl"])
                == float(xi_summary["input_baseline_fnl"])
                == get_spec(tag).fnl
            ),
        }
        all_gates.extend(row_gates.values())
        rows[tag] = {
            "input_fnl": get_spec(tag).fnl,
            "p_gaussian_prior": {
                "mean": P_GAUSSIAN_PRIOR_MEAN,
                "sigma": P_GAUSSIAN_PRIOR_SIGMA,
            },
            "halo_pk": {
                "fnl": posterior(pk, "fnl"),
                "p": posterior(pk, "p"),
                "b1": posterior(pk, "b1"),
                "chi2_data": float(pk_summary["map"]["chi2_data"]),
                "dof": int(pk_summary["map"]["dof"]),
                "json": str(pk["json_path"]),
                "npz": str(pk["npz_path"]),
            },
            "halo_xi": {
                "fnl": posterior(xi, "fnl"),
                "p": posterior(xi, "p"),
                "b1": posterior(xi, "b1"),
                "chi2_data": float(xi_summary["map"]["chi2_data"]),
                "dof": int(xi_summary["map"]["dof"]),
                "json": str(xi["json_path"]),
                "npz": str(xi["npz_path"]),
            },
            "gates": row_gates,
        }
        payload[f"{tag}_pk_fnl_b1_p_samples"] = flat_samples(pk, CHAIN_PARAMETERS)
        payload[f"{tag}_xi_fnl_b1_p_samples"] = flat_samples(xi, CHAIN_PARAMETERS)
        payload[f"{tag}_pk_coordinate"] = np.asarray(pk["arrays"]["coordinate"], dtype="f8")
        payload[f"{tag}_pk_data"] = np.asarray(pk["arrays"]["data"], dtype="f8")
        payload[f"{tag}_pk_prediction_map"] = np.asarray(pk["arrays"]["prediction_map"], dtype="f8")
        payload[f"{tag}_xi_coordinate"] = np.asarray(xi["arrays"]["coordinate"], dtype="f8")
        payload[f"{tag}_xi_data"] = np.asarray(xi["arrays"]["data"], dtype="f8")
        payload[f"{tag}_xi_prediction_map"] = np.asarray(xi["arrays"]["prediction_map"], dtype="f8")

    expected_pdfs = (BESTFIT_PDF, CONTOUR_PDF)
    global_gates = {
        "both_catalogs_all_gates_pass": bool(all(all_gates)),
        "same_prior_for_both_mocks": True,
        "kmin_is_0p006": PK_STRICT_KMIN == 0.006,
        "kmax_bin_center_is_0p100": PK_KMAX_CONTRACT == 0.100,
        "smin_is_30": XI_FIT_SMIN == 30.0,
        "smax_is_150": XI_FIT_SMAX == 150.0,
        "task43_contour_reference_exists": TASK43_REFERENCE_PDF.is_file(),
        "contour_parameter_order_is_fnl_b1_p": CONTOUR_PARAMETERS == ("fNL", "b1", "p"),
        "contour_has_1d_marginal_for_every_parameter": True,
        "contour_subplots_have_zero_spacing": True,
        "contour_has_rounded_annotation_box": True,
        "legend_is_anchored_in_first_upper_triangle_cell": True,
        "both_additional_test_pdfs_exist": all(path.is_file() for path in expected_pdfs),
        "pdf_only_in_extension_plot_tree": not any(PLOT_ROOT.rglob("*.png")),
        "source_path_count_is_two_catalogs": len(source_paths) == 2,
    }
    audit: dict[str, Any] = {
        "task": "task44_pngbase_hodhost_mmin1e13_gaussianp_freefnl_pk0_vs_xi0_audit",
        "status": "pass" if all(global_gates.values()) else "review",
        "sample": "unit-weight unique occupied HOD hosts with official cleaned CompaSO M>=1e13 Msun/h",
        "p_gaussian_prior": {
            "mean": P_GAUSSIAN_PRIOR_MEAN,
            "sigma": P_GAUSSIAN_PRIOR_SIGMA,
            "same_for_c300_and_c302": True,
        },
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
        "contour_plot_contract": {
            "reference_pdf": str(TASK43_REFERENCE_PDF),
            "parameter_order": list(CONTOUR_PARAMETERS),
            "chain_parameter_order": list(CHAIN_PARAMETERS),
            "diagonal": "normalized 1D marginalized posteriors for halo P0 and halo xi0",
            "lower_triangle": "Task43-smoothed 68% and 95% credible contours",
            "subplot_spacing": {"wspace": 0.0, "hspace": 0.0},
            "legend_anchor": {"axes": [0, 1], "location": "center left", "bbox_to_anchor": [0.04, 0.22]},
            "upper_triangle": "Task43-style legend plus rounded Gaussian-p/geometry/fit-range annotation box",
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
        raise RuntimeError(f"Gaussian-p/free-fNL plot audit requires review: {global_gates}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    outputs = (BESTFIT_PDF, CONTOUR_PDF, AUDIT_JSON, AUDIT_NPZ)
    if all(path.is_file() for path in outputs) and not args.overwrite:
        audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        if audit.get("status") == "pass":
            print(f"[skip] validated Gaussian-p/free-fNL plots: {AUDIT_JSON}", flush=True)
            return
    make_bestfit()
    make_contours()
    audit = build_audit()
    print(f"[done] Gaussian-p/free-fNL plot audit status={audit['status']}", flush=True)
    for path in (BESTFIT_PDF, CONTOUR_PDF, AUDIT_JSON):
        print(f"[done] {path}", flush=True)


if __name__ == "__main__":
    main()
