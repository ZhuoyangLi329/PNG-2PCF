#!/usr/bin/env python3
"""Plot and audit the strict-kmin fixed-fNL/free-p P0 fits."""

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
from matplotlib.colors import to_rgba
from scipy.ndimage import gaussian_filter

from task44_pngbase_hodhost_mmin1e13_common import (
    CATALOGS,
    PK_KMAX_CONTRACT,
    PK_STRICT_FIT_EDGES,
    PK_STRICT_KMIN,
    PLOT_ROOT,
    SUMMARY_DIR,
    atomic_savez,
    atomic_write_json,
    ensure_output_dirs,
    get_spec,
    sha256_file,
    strict_fit_prefix,
    strict_pk_metadata_path,
    strict_pk_path,
)
from task44_pngbase_hodmap_rawbox_common import fit_prefix


COLORS = {"c300": "#4C72B0", "c302": "#C44E52", "old": "0.45", "strict": "#C44E52"}
BESTFIT_PDF = PLOT_ROOT / "task44_pngbase_c300_c302_fixedfnl_strictkmin0p006_pk0_bestfit_curves.pdf"
LEGACY_STANDALONE_CONTOUR_PDF = PLOT_ROOT / "task44_pngbase_c300_c302_fixedfnl_strictkmin0p006_p_b1_contours.pdf"
COMPARISON_PDF = PLOT_ROOT / "task44_pngbase_c300_c302_fixedfnl_p_kmin0p003_vs_strict0p006.pdf"
AUDIT_JSON = SUMMARY_DIR / "task44_pngbase_hodmap_strictk0p006_fixedfnl_freep_audit.json"
AUDIT_NPZ = SUMMARY_DIR / "task44_pngbase_hodmap_strictk0p006_fixedfnl_freep_audit.npz"


def load_fit(tag: str, *, strict: bool) -> dict[str, Any]:
    prefix = strict_fit_prefix(tag) if strict else fit_prefix(tag, "pk", kmin_edge=0.003)
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"missing fit: {npz_path} / {json_path}")
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
    chain = chain.reshape(-1, chain.shape[-1])
    available = [str(value) for value in fit["arrays"]["parameter_names"]]
    return np.column_stack([chain[:, available.index(name)] for name in names])


def _atomic_figure_save(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.pdf")
    figure.savefig(temporary)
    plt.close(figure)
    temporary.replace(output)


def _range_box(tag: str, fit: dict[str, Any], *, include_p: bool = True) -> str:
    lines = [
        rf"fixed $f_{{\rm NL}}={get_spec(tag).fnl:g}$",
        r"$k_{\min}=0.006\ h\,{\rm Mpc}^{-1}$",
        r"$k_{\max}=0.100\ h\,{\rm Mpc}^{-1}$ (bin centre)",
        r"last bin: $[0.099,0.101)\ h\,{\rm Mpc}^{-1}$",
    ]
    if include_p:
        row = posterior(fit, "p")
        lines.append(
            rf"$p={row['q50']:.2f}_{{-{row['q50'] - row['q16']:.2f}}}"
            rf"^{{+{row['q84'] - row['q50']:.2f}}}$"
        )
    return "\n".join(lines)


def make_bestfit() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.55), sharey=True)
    for axis, tag in zip(axes, CATALOGS, strict=True):
        fit = load_fit(tag, strict=True)
        k = np.asarray(fit["arrays"]["coordinate"], dtype="f8")
        data = np.asarray(fit["arrays"]["data"], dtype="f8")
        prediction = np.asarray(fit["arrays"]["prediction_map"], dtype="f8")
        sigma = np.sqrt(np.diag(np.asarray(fit["arrays"]["covariance_final"], dtype="f8")))
        axis.errorbar(k, data, yerr=sigma, fmt="o", ms=3.8, color=COLORS[tag], capsize=1.8, label=r"$P_0$ data")
        axis.plot(k, prediction, color="black", lw=1.45, label="MAP best fit")
        axis.set_xscale("log")
        axis.set_title(tag, fontsize=14, fontweight="bold")
        axis.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
        axis.text(
            0.04,
            0.04,
            _range_box(tag, fit),
            transform=axis.transAxes,
            va="bottom",
            fontsize=9.1,
            linespacing=1.2,
            bbox={"facecolor": "white", "edgecolor": "0.72", "alpha": 0.92},
        )
    axes[0].set_ylabel(r"$P_0(k)\ [(h^{-1}{\rm Mpc})^3]$")
    axes[1].legend(frameon=False, loc="upper right")
    figure.tight_layout()
    _atomic_figure_save(figure, BESTFIT_PDF)


def _credible_density(axis: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    xlo, xhi = np.quantile(x, [0.002, 0.998])
    ylo, yhi = np.quantile(y, [0.002, 0.998])
    xpad = 0.08 * (xhi - xlo)
    ypad = 0.08 * (yhi - ylo)
    histogram, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=100,
        range=((xlo - xpad, xhi + xpad), (ylo - ypad, yhi + ypad)),
    )
    density = gaussian_filter(histogram, sigma=1.25)
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered)
    cumulative /= cumulative[-1]
    thresholds = []
    for probability in (0.95, 0.68):
        thresholds.append(float(ordered[min(np.searchsorted(cumulative, probability), ordered.size - 1)]))
    low, high = sorted(thresholds)
    maximum = float(np.max(density))
    if not low < high < maximum:
        raise RuntimeError(f"degenerate contour density levels: {low}, {high}, {maximum}")
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    rgba = to_rgba(color)
    axis.contourf(
        xc,
        yc,
        density.T,
        levels=[low, high, maximum],
        colors=[(*rgba[:3], 0.22), (*rgba[:3], 0.48)],
    )
    axis.contour(xc, yc, density.T, levels=[low, high], colors=[color, color], linewidths=[1.0, 1.35])
    axis.set_xlim(xedges[0], xedges[-1])
    axis.set_ylim(yedges[0], yedges[-1])


def make_contours() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.75))
    for axis, tag in zip(axes, CATALOGS, strict=True):
        fit = load_fit(tag, strict=True)
        samples = flat_samples(fit, ("p", "b1"))
        _credible_density(axis, samples[:, 0], samples[:, 1], COLORS[tag])
        axis.set_title(tag, fontsize=14, fontweight="bold")
        axis.set_xlabel(r"$p$")
        axis.text(
            0.04,
            0.96,
            _range_box(tag, fit) + "\n" + r"marginalized over $s_{n,0}$",
            transform=axis.transAxes,
            va="top",
            fontsize=8.9,
            linespacing=1.18,
            bbox={"facecolor": "white", "edgecolor": "0.72", "alpha": 0.92},
        )
    axes[0].set_ylabel(r"$b_1$")
    figure.tight_layout()
    _atomic_figure_save(figure, LEGACY_STANDALONE_CONTOUR_PDF)


def make_comparison() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharex=False, sharey=True)
    labels = [r"old: $k_{\min}=0.003$", r"strict: $k_{\min}=0.006$"]
    for axis, tag in zip(axes, CATALOGS, strict=True):
        old = posterior(load_fit(tag, strict=False), "p")
        new_fit = load_fit(tag, strict=True)
        new = posterior(new_fit, "p")
        for y, row, color in ((1.0, old, COLORS["old"]), (0.0, new, COLORS["strict"])):
            axis.errorbar(
                row["q50"],
                y,
                xerr=[[row["q50"] - row["q16"]], [row["q84"] - row["q50"]]],
                fmt="o",
                color=color,
                capsize=3,
                ms=6,
            )
        axis.set_yticks([1.0, 0.0], labels)
        axis.set_ylim(-0.65, 1.65)
        axis.set_xlabel(r"$p$")
        axis.set_title(tag, fontsize=14, fontweight="bold")
        axis.grid(axis="x", alpha=0.22)
        axis.text(
            0.04,
            0.95,
            rf"fixed $f_{{\rm NL}}={get_spec(tag).fnl:g}$" + "\n" +
            r"$k_{\max}=0.100\ h\,{\rm Mpc}^{-1}$ (bin centre)" + "\n" +
            r"last bin: $[0.099,0.101)$",
            transform=axis.transAxes,
            va="top",
            fontsize=9.0,
            bbox={"facecolor": "white", "edgecolor": "0.72", "alpha": 0.92},
        )
    figure.tight_layout()
    _atomic_figure_save(figure, COMPARISON_PDF)


def build_audit() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    payload: dict[str, np.ndarray] = {}
    gates: list[bool] = []
    for tag in CATALOGS:
        new = load_fit(tag, strict=True)
        old = load_fit(tag, strict=False)
        new_p = posterior(new, "p")
        old_p = posterior(old, "p")
        new_edges = np.asarray(new["arrays"]["coordinate_edges"], dtype="f8")
        old_edges = np.asarray(old["arrays"]["coordinate_edges"], dtype="f8")
        new_data = np.asarray(new["arrays"]["data"], dtype="f8")
        old_data = np.asarray(old["arrays"]["data"], dtype="f8")
        shared_edges_exact = bool(np.array_equal(new_edges[1:], old_edges[2:]))
        shared_data_exact = bool(np.array_equal(new_data[1:], old_data[2:]))
        measurement_metadata = json.loads(strict_pk_metadata_path(tag).read_text(encoding="utf-8"))
        row_gates = {
            "strict_measurement_pass": measurement_metadata.get("status") == "pass",
            "strict_measurement_hash_valid": measurement_metadata.get("output_sha256") == sha256_file(strict_pk_path(tag)),
            "strict_fit_pass": new["summary"].get("status") == "pass",
            "strict_fit_all_gates_pass": bool(all(new["summary"]["gates"].values())),
            "fixed_fnl_matches_catalog": float(new["summary"]["fixed_fnl"]) == get_spec(tag).fnl,
            "strict_edges_exact": bool(np.array_equal(new_edges, PK_STRICT_FIT_EDGES)),
            "common_remeasured_bins_bitwise_identical_to_old_measurement": shared_edges_exact and shared_data_exact,
        }
        gates.extend(row_gates.values())
        sigma_combined = float(np.hypot(old_p["sigma68"], new_p["sigma68"]))
        rows[tag] = {
            "fixed_fnl": get_spec(tag).fnl,
            "old_kmin0p003": {
                "p": old_p,
                "chi2": float(old["summary"]["map"]["chi2"]),
                "dof": int(old["summary"]["map"]["dof"]),
                "json": str(old["json_path"]),
                "npz": str(old["npz_path"]),
            },
            "strict_kmin0p006": {
                "p": new_p,
                "chi2": float(new["summary"]["map"]["chi2"]),
                "dof": int(new["summary"]["map"]["dof"]),
                "json": str(new["json_path"]),
                "npz": str(new["npz_path"]),
            },
            "change": {
                "delta_p_q50_strict_minus_old": float(new_p["q50"] - old_p["q50"]),
                "delta_p_over_old_sigma68": float((new_p["q50"] - old_p["q50"]) / old_p["sigma68"]),
                "delta_p_over_descriptive_combined_sigma68": float((new_p["q50"] - old_p["q50"]) / sigma_combined),
                "sigma68_ratio_strict_over_old": float(new_p["sigma68"] / old_p["sigma68"]),
                "delta_chi2_strict_minus_old": float(new["summary"]["map"]["chi2"] - old["summary"]["map"]["chi2"]),
                "delta_dof_strict_minus_old": int(new["summary"]["map"]["dof"] - old["summary"]["map"]["dof"]),
                "same_catalog_correlated_fit_warning": "old and strict fits use overlapping modes; combined-sigma shift is descriptive only",
            },
            "remeasurement_crosscheck": {
                "n_common_shells": int(new_edges.shape[0] - 1),
                "common_edges_bitwise_equal": shared_edges_exact,
                "common_P0_values_bitwise_equal": shared_data_exact,
                "common_P0_max_abs_difference": float(np.max(np.abs(new_data[1:] - old_data[2:]))),
            },
            "gates": row_gates,
        }
        payload[f"{tag}_old_p_samples"] = flat_samples(old, ("p",))[:, 0]
        payload[f"{tag}_strict_p_samples"] = flat_samples(new, ("p",))[:, 0]
        payload[f"{tag}_strict_k"] = np.asarray(new["arrays"]["coordinate"], dtype="f8")
        payload[f"{tag}_strict_data"] = new_data
        payload[f"{tag}_strict_prediction_map"] = np.asarray(new["arrays"]["prediction_map"], dtype="f8")

    global_gates = {
        "both_catalog_measurements_fits_and_crosschecks_pass": bool(all(gates)),
        "strict_contract_has_48_shells": int(PK_STRICT_FIT_EDGES.shape[0]) == 48,
        "strict_first_shell_has_6_signed_modes": bool(
            all(int(load_fit(tag, strict=True)["arrays"]["nmodes"][0]) == 6 for tag in CATALOGS)
        ),
        "both_expected_pdfs_exist": all(path.is_file() for path in (BESTFIT_PDF, COMPARISON_PDF)),
        "standalone_pk_contour_removed": not LEGACY_STANDALONE_CONTOUR_PDF.exists(),
        "pdf_only_in_extension_plot_tree": not any(PLOT_ROOT.rglob("*.png")),
    }
    audit: dict[str, Any] = {
        "task": "task44_pngbase_hodmap_strictk0p006_fixedfnl_freep_audit",
        "status": "pass" if all(global_gates.values()) else "review",
        "fit_quality_policy": "chi2/dof and PTE are descriptive diagnostics, not pass/fail gates",
        "mode_change": {
            "old": {
                "first_shells_h_mpc": [[0.003, 0.005], [0.005, 0.007]],
                "first_shell_mode_counts": [18, 14],
                "total_signed_modes": 138984,
            },
            "strict": {
                "physical_rule": "k>=0.006 h/Mpc",
                "first_shell_h_mpc": [0.006, 0.007],
                "first_shell_mode_count": 6,
                "total_signed_modes": 138958,
            },
            "removed_signed_modes": 26,
            "common_shells_after_0p007": 47,
            "important_note": "the strict first shell was remeasured; the old [0.005,0.007) value was not relabelled",
        },
        "k_range": {
            "kmin_h_mpc": PK_STRICT_KMIN,
            "kmax_bin_center_contract_h_mpc": PK_KMAX_CONTRACT,
            "last_bin_h_mpc": PK_STRICT_FIT_EDGES[-1].tolist(),
        },
        "global_gates": global_gates,
        "catalogs": rows,
        "independence_warning": "c300 and c302 are both ph000 and are not multiplied as independent likelihoods",
        "plots": {
            "bestfit_pdf": str(BESTFIT_PDF),
            "bestfit_pdf_sha256": sha256_file(BESTFIT_PDF),
            "old_vs_strict_pdf": str(COMPARISON_PDF),
            "old_vs_strict_pdf_sha256": sha256_file(COMPARISON_PDF),
        },
    }
    atomic_savez(AUDIT_NPZ, **payload, audit_json=np.asarray(json.dumps(audit, sort_keys=True)))
    audit["payload_npz"] = str(AUDIT_NPZ)
    audit["payload_npz_sha256"] = sha256_file(AUDIT_NPZ)
    atomic_write_json(AUDIT_JSON, audit)
    if audit["status"] != "pass":
        raise RuntimeError(f"strict-k audit requires review: {global_gates}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    outputs = (BESTFIT_PDF, COMPARISON_PDF, AUDIT_JSON, AUDIT_NPZ)
    if all(path.is_file() for path in outputs) and not args.overwrite:
        audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        if audit.get("status") == "pass":
            print(f"[skip] validated strict-k plot/audit products: {AUDIT_JSON}", flush=True)
            return
    make_bestfit()
    make_comparison()
    audit = build_audit()
    print(f"[done] strict-k plot/audit status={audit['status']}", flush=True)
    for path in (BESTFIT_PDF, COMPARISON_PDF, AUDIT_JSON):
        print(f"[done] {path}", flush=True)


if __name__ == "__main__":
    main()
