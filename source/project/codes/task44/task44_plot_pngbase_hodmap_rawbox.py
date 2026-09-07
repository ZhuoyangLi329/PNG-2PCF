#!/usr/bin/env python3
"""Audit and plot the Task44 fixed-fNL/free-p PNG-base raw-box fits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from task44_pngbase_hodmap_rawbox_common import (
    BOX_SIZE,
    BOX_VOLUME,
    K_FUND,
    PK_PRIMARY_FIT_EDGES,
    PLOT_ROOT,
    SUMMARY_DIR,
    atomic_savez,
    atomic_write_json,
    ensure_output_dirs,
    fit_prefix,
    get_spec,
    pk_path,
    sha256_file,
)

COLORS = {
    "pk": "#4C72B0",
    "xi30": "#DD8452",
    "c300": "#55A868",
    "c302": "#8172B2",
}


def load_fit(
    tag: str,
    probe: str,
    *,
    kmin: float | None = None,
    smin: float = 50.0,
    fixed_p: float | None = None,
) -> dict[str, Any]:
    prefix = fit_prefix(tag, probe, kmin_edge=kmin, smin=smin, fixed_p=fixed_p)
    json_path, npz_path = prefix.with_suffix(".json"), prefix.with_suffix(".npz")
    if not json_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(f"missing fit: {json_path} / {npz_path}")
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files if name != "summary_json"}
    if summary.get("output_npz_sha256") != sha256_file(npz_path):
        raise RuntimeError(f"fit hash mismatch: {npz_path}")
    return {"summary": summary, "arrays": arrays, "json_path": json_path, "npz_path": npz_path}


def selected_indices(source_edges: np.ndarray, target_edges: np.ndarray) -> np.ndarray:
    output = []
    for edge in np.asarray(target_edges, dtype="f8"):
        found = np.flatnonzero(np.all(np.isclose(source_edges, edge[None, :], rtol=0.0, atol=1.0e-13), axis=1))
        if found.size != 1:
            raise ValueError(f"edge {edge.tolist()} has {found.size} matches")
        output.append(int(found[0]))
    return np.asarray(output, dtype="i8")


def posterior_row(fit: dict[str, Any], name: str) -> dict[str, float]:
    return {key: float(value) for key, value in fit["summary"]["mcmc"]["posterior"][name].items()}


def build_audit() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    rows: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    all_measurement_gates = []
    all_fit_contract_gates = []
    all_kmin_gates = []
    for tag in ("c300", "c302"):
        pk = load_fit(tag, "pk", kmin=0.003)
        pk_without_lowest = load_fit(tag, "pk", kmin=0.005)
        xi30 = load_fit(tag, "xi", smin=30.0)
        xi30_fixedp1 = load_fit(tag, "xi", smin=30.0, fixed_p=1.0)
        with np.load(pk_path(tag, mesh=400, origin="positive"), allow_pickle=False) as primary_measurement:
            source_edges = np.asarray(primary_measurement["k_edges"], dtype="f8")
            indices = selected_indices(source_edges, PK_PRIMARY_FIT_EDGES)
            pk400 = np.asarray(primary_measurement["pk0"], dtype="f8")
        with np.load(pk_path(tag, mesh=400, origin="centered"), allow_pickle=False) as centered_measurement:
            pk_centered = np.asarray(centered_measurement["pk0"], dtype="f8")
        with np.load(pk_path(tag, mesh=512, origin="positive"), allow_pickle=False) as mesh_measurement:
            pk512 = np.asarray(mesh_measurement["pk0"], dtype="f8")
        covariance_pk = np.asarray(pk["arrays"]["covariance_final"], dtype="f8")
        mesh_delta_sigma = (pk512[indices] - pk400[indices]) / np.sqrt(np.diag(covariance_pk))
        origin_difference = pk_centered - pk400
        origin_relative_l2 = float(np.linalg.norm(origin_difference) / np.linalg.norm(pk400))

        p_pk = posterior_row(pk, "p")
        p_pk_without_lowest = posterior_row(pk_without_lowest, "p")
        p_xi30 = posterior_row(xi30, "p")
        fnl_xi30_fixedp1 = posterior_row(xi30_fixedp1, "fnl")
        p_shift = float((p_pk_without_lowest["q50"] - p_pk["q50"]) / p_pk["sigma68"])
        naive_probe_difference = float(
            (p_pk["q50"] - p_xi30["q50"]) / np.hypot(p_pk["sigma68"], p_xi30["sigma68"])
        )
        fixedp_prediction_equivalence = float(
            np.max(np.abs(xi30_fixedp1["arrays"]["prediction_map"] - xi30["arrays"]["prediction_map"]))
        )
        measurement_gates = {
            "origin_translation_relative_l2_below_1e_12": origin_relative_l2 < 1.0e-12,
            "mesh400_512_max_shift_below_0p01sigma": float(np.max(np.abs(mesh_delta_sigma))) < 0.01,
            "first_bin_nmodes_18": bool(pk["summary"]["gates"]["primary_first_bin_nmodes_18"]),
            "primary_has_49_contiguous_periodic_bins": bool(
                pk["arrays"]["coordinate_edges"].shape == (49, 2)
                and np.allclose(
                    pk["arrays"]["coordinate_edges"],
                    PK_PRIMARY_FIT_EDGES,
                    rtol=0.0,
                    atol=1.0e-14,
                )
            ),
        }
        fit_contract_gates = {
            "pk_primary_fit_status_pass": pk["summary"]["status"] == "pass",
            "pk_primary_mcmc_all_gates": bool(pk["summary"]["gates"]["mcmc_all_gates"]),
            "pk_without_lowest_bin_fit_status_pass": pk_without_lowest["summary"]["status"] == "pass",
            "pk_without_lowest_bin_mcmc_all_gates": bool(pk_without_lowest["summary"]["gates"]["mcmc_all_gates"]),
            "xi_smin30_fit_status_pass": xi30["summary"]["status"] == "pass",
            "xi_smin30_mcmc_all_gates": bool(xi30["summary"]["gates"]["mcmc_all_gates"]),
            "xi_fixedp1_smin30_fit_status_pass": xi30_fixedp1["summary"]["status"] == "pass",
            "xi_fixedp1_smin30_mcmc_all_gates": bool(xi30_fixedp1["summary"]["gates"]["mcmc_all_gates"]),
            "fixedp1_freefnl_map_prediction_equivalent_below_1e_7": bool(
                fixedp_prediction_equivalence < 1.0e-7
            ),
        }
        kmin_gates = {"absolute_p_shift_below_1sigma_primary": abs(p_shift) < 1.0}
        all_measurement_gates.extend(measurement_gates.values())
        all_fit_contract_gates.extend(fit_contract_gates.values())
        all_kmin_gates.extend(kmin_gates.values())
        rows[tag] = {
            "fixed_fnl": get_spec(tag).fnl,
            "primary_pk": {
                "p": p_pk,
                "b1": posterior_row(pk, "b1"),
                "sn0": posterior_row(pk, "sn0"),
                "map": pk["summary"]["map"],
                "status": pk["summary"]["status"],
                "json": str(pk["json_path"]),
                "npz": str(pk["npz_path"]),
            },
            "pk_without_lowest_k_bin": {
                "purpose": "sensitivity check only; the primary result includes the lowest-k bin",
                "omitted_k_bin_h_mpc": [0.003, 0.005],
                "p": p_pk_without_lowest,
                "b1": posterior_row(pk_without_lowest, "b1"),
                "sn0": posterior_row(pk_without_lowest, "sn0"),
                "map": pk_without_lowest["summary"]["map"],
                "status": pk_without_lowest["summary"]["status"],
                "p_shift_over_primary_sigma68": p_shift,
                "json": str(pk_without_lowest["json_path"]),
                "npz": str(pk_without_lowest["npz_path"]),
            },
            "xi0_fit": {
                "smin_mpc_h": 30.0,
                "p": p_xi30,
                "b1": posterior_row(xi30, "b1"),
                "map": xi30["summary"]["map"],
                "status": xi30["summary"]["status"],
                "json": str(xi30["json_path"]),
                "npz": str(xi30["npz_path"]),
                "fit_quality_note": "chi2/dof and PTE are reported as diagnostics, not used as a fit veto",
            },
            "xi0_fixedp1_freefnl": {
                "fixed_p": 1.0,
                "fnl_prior": [-500.0, 500.0],
                "smin_mpc_h": 30.0,
                "fnl": fnl_xi30_fixedp1,
                "b1": posterior_row(xi30_fixedp1, "b1"),
                "map": xi30_fixedp1["summary"]["map"],
                "status": xi30_fixedp1["summary"]["status"],
                "json": str(xi30_fixedp1["json_path"]),
                "npz": str(xi30_fixedp1["npz_path"]),
                "map_prediction_equivalence_vs_fixedfnl_freep_max_abs": fixedp_prediction_equivalence,
            },
            "diagnostics": {
                "p_pk_minus_xi_over_naive_independent_sigma": naive_probe_difference,
                "same_catalog_probe_correlation_note": "P0 and xi0 use the same realization; the naive independent sigma is descriptive only",
                "different_fnl_catalog_note": "p is inferred separately at this catalog's fixed baseline fNL; no c300-c302 equality gate is applied",
                "origin_translation": {
                    "max_abs_power": float(np.max(np.abs(origin_difference))),
                    "relative_l2": origin_relative_l2,
                },
                "mesh400_vs_512_primary_bins": {
                    "max_abs_shift_sigma": float(np.max(np.abs(mesh_delta_sigma))),
                    "rms_shift_sigma": float(np.sqrt(np.mean(mesh_delta_sigma**2))),
                },
            },
            "gates": {
                "measurement": measurement_gates,
                "fit_contract": fit_contract_gates,
                "kmin_stability": kmin_gates,
            },
        }
        arrays[f"{tag}_mesh_delta_sigma"] = np.asarray(mesh_delta_sigma, dtype="f8")
        arrays[f"{tag}_origin_difference"] = np.asarray(origin_difference, dtype="f8")
        arrays[f"{tag}_fixedp1_smin30_prediction_minus_freep"] = np.asarray(
            xi30_fixedp1["arrays"]["prediction_map"] - xi30["arrays"]["prediction_map"], dtype="f8"
        )

    engine_audit_path = SUMMARY_DIR / "task44_pngbase_hodmap_xi0_three_engine_s30_350_audit.json"
    engine_audit = json.loads(engine_audit_path.read_text(encoding="utf-8"))
    engine_audit_pass = bool(
        engine_audit.get("status") == "pass"
        and all(engine_audit.get("global_gates", {}).values())
    )
    global_gates = {
        "all_measurement_geometry_estimator_gates_pass": bool(all(all_measurement_gates)),
        "all_fit_contract_and_mcmc_gates_pass": bool(all(all_fit_contract_gates)),
        "all_pk_kmin_shifts_below_1sigma": bool(all(all_kmin_gates)),
        "three_xi_pair_count_engines_agree": engine_audit_pass,
    }
    audit = {
        "task": "task44_pngbase_hodmap_rawbox_fixedfnl_freep_audit",
        "status": "pass" if all(global_gates.values()) else "review",
        "primary_science_status": "P0 uses the 49 contiguous native periodic-box bins with centres 0.004--0.100 h/Mpc; xi0 uses smin=30 Mpc/h only; fixed-fNL/free-p and fixed-p=1/free-fNL per-catalog constraints completed; chi2 is diagnostic; the effect of omitting the lowest-k P0 bin is reported separately",
        "geometry": {
            "type": "full periodic real-space cube",
            "boxsize_mpc_h": BOX_SIZE,
            "volume_mpc_h3": BOX_VOLUME,
            "kfund_h_mpc": K_FUND,
            "single_snapshot_z": 0.5,
            "redshift_evolution_within_catalog": False,
            "coordinates": "Cartesian X/Y/Z in [-L/2,L/2); stored velocities are not applied to real-space positions",
            "no_lightcone_window_random_fkp_gic_ric_aic": True,
            "integral_constraint_note": "the absent periodic-box k=0 mode is a finite-box condition, not DESI radial/angular integral constraint",
        },
        "parameter_contract": {
            "c300_fixed_fnl": 30.0,
            "c302_fixed_fnl": 100.0,
            "pk_free": ["b1", "p", "sn0"],
            "xi_free": ["b1", "p"],
            "xi_fixed_p_test": {"fixed_p": 1.0, "free": ["b1", "fNL"], "fNL_prior": [-500.0, 500.0]},
            "sigma_s_fixed": 0.0,
            "p_inference": "separate for each fixed baseline fNL; no equality or consistency gate between catalogs",
        },
        "fit_quality_policy": "chi2/dof and PTE are retained as descriptive diagnostics and are not pass/fail gates",
        "xi_engine_validation": {
            "audit_json": str(engine_audit_path),
            "status": engine_audit.get("status"),
            "global_gates": engine_audit.get("global_gates"),
            "conclusion": engine_audit.get("conclusion"),
        },
        "global_gates": global_gates,
        "catalogs": rows,
        "independence_warning": "c300 and c302 are both ph000 and must not be multiplied as independent likelihoods",
    }
    return audit, arrays


def get_flat_samples(fit: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    chain = np.asarray(fit["arrays"]["chain_final_by_step"], dtype="f8").reshape(-1, fit["arrays"]["chain_final_by_step"].shape[-1])
    available = [str(value) for value in fit["arrays"]["parameter_names"]]
    return np.column_stack([chain[:, available.index(name)] for name in names])


def make_contour(tag: str, output: Path) -> None:
    from getdist import MCSamples, plots

    pk = load_fit(tag, "pk", kmin=0.003)
    xi30 = load_fit(tag, "xi", smin=30.0)
    samples = []
    fit_rows = (
        (pk, r"P_0(k)", COLORS["pk"]),
        (xi30, r"\xi_0(s)", COLORS["xi30"]),
    )
    for fit, observable, color in fit_rows:
        posterior = posterior_row(fit, "p")
        label = (
            rf"${observable}:\ p={posterior['q50']:.2f}"
            rf"_{{-{posterior['q50'] - posterior['q16']:.2f}}}"
            rf"^{{+{posterior['q84'] - posterior['q50']:.2f}}}$"
        )
        values = get_flat_samples(fit, ("p", "b1"))
        sample = MCSamples(
            samples=values,
            names=["p", "b1"],
            labels=["p", "b_1"],
            label=label,
            ranges={"p": [-5.0, 5.0], "b1": [0.5, 5.0]},
            settings={"smooth_scale_1D": 0.3, "smooth_scale_2D": 0.3},
        )
        samples.append((sample, color))
    plotter = plots.get_subplot_plotter(width_inch=7.0)
    plotter.settings.axes_fontsize = 11
    plotter.settings.lab_fontsize = 13
    plotter.settings.legend_fontsize = 9
    plotter.settings.linewidth = 1.5
    plotter.triangle_plot(
        [item[0] for item in samples],
        params=["p", "b1"],
        filled=True,
        contour_colors=[item[1] for item in samples],
        legend_labels=[item[0].label for item in samples],
    )
    annotation = (
        f"{tag} periodic real space\n"
        rf"fixed $f_{{\rm NL}}={get_spec(tag).fnl:g}$" "\n"
        r"$P_0$: $k_{\min}=0.003$, $k_{\max}=0.100\ h\,{\rm Mpc}^{-1}$ (49 bins)" "\n"
        r"$\xi_0$: $s_{\min}=30$, $s_{\max}=350\ h^{-1}{\rm Mpc}$ (32 bins)"
    )
    plotter.fig.text(
        0.97,
        0.76,
        annotation,
        ha="right",
        va="top",
        fontsize=9.5,
        linespacing=1.25,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.45", "alpha": 0.95},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.pdf")
    plotter.export(str(temporary))
    temporary.replace(output)
    plt.close(plotter.fig)


def make_summary_pdf(audit: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary) as pdf:
        figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), sharex="col", gridspec_kw={"height_ratios": [2.2, 1.0]})
        for column, tag in enumerate(("c300", "c302")):
            fit = load_fit(tag, "pk", kmin=0.003)
            k = fit["arrays"]["coordinate"]
            data = fit["arrays"]["data"]
            model = fit["arrays"]["prediction_map"]
            sigma = np.sqrt(np.diag(fit["arrays"]["covariance_final"]))
            axes[0, column].errorbar(k, data, yerr=sigma, fmt="o", ms=4, color=COLORS["pk"], label=rf"{tag}, fixed $f_{{\rm NL}}={get_spec(tag).fnl:g}$")
            axes[0, column].plot(k, model, color="black", lw=1.4, ls="--", label="MAP model")
            axes[0, column].set_xscale("log")
            axes[0, column].set_yscale("log")
            axes[0, column].set_ylabel(r"$P_0(k)\ [(h^{-1}{\rm Mpc})^3]$")
            axes[0, column].legend(frameon=False, fontsize=9)
            axes[1, column].axhline(0.0, color="0.5", lw=0.8)
            axes[1, column].plot(k, (data - model) / sigma, "o-", color=COLORS["pk"], ms=4, lw=0.8)
            axes[1, column].set_xscale("log")
            axes[1, column].set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
            axes[1, column].set_ylabel(r"residual / $\sigma$")
        figure.suptitle(
            r"$P_0$: $k_{\min}=0.003$, $k_{\max}=0.100\ h\,{\rm Mpc}^{-1}$ (49 bins)",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.45"},
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
        for axis, tag in zip(axes, ("c300", "c302"), strict=True):
            row = audit["catalogs"][tag]["xi0_fixedp1_freefnl"]
            posterior = row["fnl"]
            axis.errorbar(
                posterior["q50"],
                0.0,
                xerr=[[posterior["q50"] - posterior["q16"]], [posterior["q84"] - posterior["q50"]]],
                fmt="D",
                color=COLORS["xi30"],
                capsize=3,
                ms=6,
            )
            axis.axvline(get_spec(tag).fnl, color="0.25", ls="--", lw=1.0, label="catalog baseline")
            axis.set_yticks([0.0], [r"$\xi_0(s)$"])
            axis.set_ylim(-0.7, 0.7)
            axis.set_xlabel(r"$f_{\rm NL}$ at fixed $p=1$")
            axis.set_title(rf"{tag}, baseline $f_{{\rm NL}}={get_spec(tag).fnl:g}$")
            axis.grid(axis="x", alpha=0.2)
            axis.legend(frameon=False, fontsize=8)
        figure.suptitle(
            r"$\xi_0$: $s_{\min}=30$, $s_{\max}=350\ h^{-1}{\rm Mpc}$ (32 bins)",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.45"},
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), sharex="col", gridspec_kw={"height_ratios": [2.2, 1.0]})
        for column, tag in enumerate(("c300", "c302")):
            fit30 = load_fit(tag, "xi", smin=30.0)
            s30 = fit30["arrays"]["coordinate"]
            data30 = fit30["arrays"]["data"]
            model30 = fit30["arrays"]["prediction_map"]
            sigma30 = np.sqrt(np.diag(fit30["arrays"]["covariance_final"]))
            axes[0, column].errorbar(
                s30,
                s30**2 * data30,
                yerr=s30**2 * sigma30,
                fmt="o",
                ms=3.8,
                color="0.25",
                label=rf"{tag} measurement",
            )
            axes[0, column].plot(s30, s30**2 * model30, color=COLORS["xi30"], lw=1.5, label="MAP")
            axes[0, column].set_ylabel(r"$s^2\xi_0(s)\ [(h^{-1}{\rm Mpc})^2]$")
            axes[0, column].legend(frameon=False, fontsize=9)
            axes[1, column].axhline(0.0, color="0.5", lw=0.8)
            axes[1, column].plot(
                s30,
                (data30 - model30) / sigma30,
                "o-",
                color=COLORS["xi30"],
                ms=3.8,
                lw=0.8,
            )
            axes[1, column].set_xlabel(r"$s\ [h^{-1}{\rm Mpc}]$")
            axes[1, column].set_ylabel(r"residual / $\sigma$")
        figure.suptitle(
            r"$\xi_0$: $s_{\min}=30$, $s_{\max}=350\ h^{-1}{\rm Mpc}$ (32 bins)",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.45"},
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        pdf.savefig(figure)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8.5, 4.8))
        entries = []
        for tag in ("c300", "c302"):
            row = audit["catalogs"][tag]
            entries.extend(
                [
                    (f"{tag}  P0  primary", row["primary_pk"]["p"], COLORS[tag], "o"),
                    (f"{tag}  P0  lowest-bin omitted", row["pk_without_lowest_k_bin"]["p"], COLORS[tag], "s"),
                    (f"{tag}  xi0  active", row["xi0_fit"]["p"], COLORS["xi30"], "D"),
                ]
            )
        y = np.arange(len(entries))[::-1]
        for yy, (label, posterior, color, marker) in zip(y, entries, strict=True):
            axis.errorbar(
                posterior["q50"],
                yy,
                xerr=[[posterior["q50"] - posterior["q16"]], [posterior["q84"] - posterior["q50"]]],
                fmt=marker,
                color=color,
                capsize=3,
                ms=6,
            )
        axis.set_yticks(y, [entry[0] for entry in entries])
        axis.set_xlabel(r"effective PNG response parameter $p$")
        axis.grid(axis="x", alpha=0.2)
        figure.suptitle(
            r"Primary: $P_0$ $k_{\min}=0.003$, $k_{\max}=0.100\ h\,{\rm Mpc}^{-1}$; "
            r"$\xi_0$ $s_{\min}=30$, $s_{\max}=350\ h^{-1}{\rm Mpc}$",
            fontsize=10.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.45"},
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
        pdf.savefig(figure)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(10.5, 6.0))
        axis.axis("off")
        lines = [
            "Geometry and estimator audit",
            "",
            rf"Periodic real-space cube: $L={BOX_SIZE:g}\ h^{{-1}}{{\rm Mpc}}$, "
            rf"$V=L^3={BOX_VOLUME:.3e}\ (h^{{-1}}{{\rm Mpc}})^3$, "
            rf"$k_{{\rm fund}}=2\pi/L={K_FUND:.8f}\ h\,{{\rm Mpc}}^{{-1}}$.",
            r"$P_0$: mesh 400, TSC, interlacing 3, compensated; formal measurement ends at $k=0.301$.",
            r"Primary $P_0$: $k_{\min}=0.003$, $k_{\max}=0.100\ h\,{\rm Mpc}^{-1}$, 49 native periodic bins; the sensitivity fit omits only [0.003,0.005).",
            r"Primary $\xi_0$: $s_{\min}=30$, $s_{\max}=350\ h^{-1}{\rm Mpc}$; FCFC periodic DD/analytic-RR$-1$.",
            r"FCFC, pycorr/Corrfunc, and CuCount reproduce the same 32-bin $\xi_0$ curve to $|\Delta\xi_0|<9\times10^{-11}$.",
            r"Single-box Gaussian covariance uses the full $L^3$ volume with no realization-count division.",
            r"$\chi^2/{\rm dof}$ is shown as a diagnostic and is not a pass/fail gate.",
            "",
        ]
        for tag in ("c300", "c302"):
            row = audit["catalogs"][tag]
            pk_map = row["primary_pk"]["map"]
            xi30_map = row["xi0_fit"]["map"]
            diagnostics = row["diagnostics"]
            fixedp = row["xi0_fixedp1_freefnl"]
            lines.extend(
                [
                    rf"{tag}, fixed $f_{{\rm NL}}={row['fixed_fnl']:g}$: "
                    rf"$\chi^2_{{P_0}}/{{\rm dof}}={pk_map['chi2']:.2f}/{int(pk_map['dof'])}$; "
                    rf"$\chi^2_{{\xi_0,s_{{\min}}=30}}/{{\rm dof}}={xi30_map['chi2']:.2f}/{int(xi30_map['dof'])}$.",
                    rf"    fixed $p=1$: $f_{{\rm NL}}(s_{{\min}}=30)={fixedp['fnl']['q50']:.2f}$.",
                    rf"    shift in $p$ after omitting the lowest-$k$ bin: {row['pk_without_lowest_k_bin']['p_shift_over_primary_sigma68']:+.2f}$\sigma$; "
                    rf"mesh 400/512 max shift: {diagnostics['mesh400_vs_512_primary_bins']['max_abs_shift_sigma']:.3g}$\sigma$; "
                    rf"origin relative $L_2$: {diagnostics['origin_translation']['relative_l2']:.3e}.",
                ]
            )
        lines.extend(["", "The two baseline-fNL p posteriors are interpreted separately; no equality gate is applied."])
        axis.text(0.04, 0.96, "\n".join(lines), va="top", ha="left", fontsize=9.8, linespacing=1.30)
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    audit_path = SUMMARY_DIR / "task44_pngbase_hodmap_rawbox_fixedfnl_freep_audit.json"
    payload_path = SUMMARY_DIR / "task44_pngbase_hodmap_rawbox_fixedfnl_freep_audit.npz"
    summary_pdf = PLOT_ROOT / "task44_pngbase_hodmap_rawbox_fixedfnl_freep_summary.pdf"
    contour_paths = {
        tag: PLOT_ROOT / f"task44_pngbase_{tag}_fixedfnl_pk0_vs_xi0_p_b1_contours.pdf"
        for tag in ("c300", "c302")
    }
    targets = (audit_path, payload_path, summary_pdf, *contour_paths.values())
    if all(path.is_file() for path in targets) and not args.overwrite:
        print(f"[skip] complete plot/audit products under {PLOT_ROOT}", flush=True)
        return
    audit, arrays = build_audit()
    atomic_savez(payload_path, **arrays, audit_json=np.asarray(json.dumps(audit, sort_keys=True)))
    audit["payload_npz"] = str(payload_path)
    audit["payload_npz_sha256"] = sha256_file(payload_path)
    for tag, path in contour_paths.items():
        make_contour(tag, path)
    make_summary_pdf(audit, summary_pdf)
    audit["plots"] = {
        "summary": str(summary_pdf),
        "fixedfnl_freep_contours": {tag: str(path) for tag, path in contour_paths.items()},
    }
    atomic_write_json(audit_path, audit)
    print(f"[done] audit {audit_path}", flush=True)
    print(f"[done] summary PDF {summary_pdf}", flush=True)


if __name__ == "__main__":
    main()
