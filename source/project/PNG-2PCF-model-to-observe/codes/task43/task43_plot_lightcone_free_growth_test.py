#!/usr/bin/env python3
"""Plot the fixed- versus free-growth Task 4.3 halo-lightcone diagnostic."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np

from task43_plot_rsd_rawbox_pk0_vs_xi0_contours import draw_contour, plot_range
from task43_rsd_common import PROJECT_ROOT, atomic_write_json, sha256_file
from task43_run_lightcone_joint_baomask_v1 import DEFAULT_ROOT as FIXED_ROOT
from task43_lightcone_free_growth_test import ARRAY_PATH, AUDIT_PATH


OUTPUT_DIR = PROJECT_ROOT / "9.11meeting/freef"
OUTPUT = OUTPUT_DIR / "task43_lightcone_rsd_P02_xi02_joint_fixed_vs_freef_baomask80_120_v1.pdf"
COLORS = {"rsd_p02": "#252525", "rsd_xi02": "#C44E52", "rsd_joint_p02xi02": "#4C72B0"}
DISPLAY = {"rsd_p02": r"$P_0+P_2$", "rsd_xi02": r"$\xi_0+\xi_2$", "rsd_joint_p02xi02": "joint"}
LABELS = {
    "fNL": r"$f_{\rm NL}$",
    "b1": r"$b_1$",
    "sigma_s": r"$\sigma_s\ [h^{-1}{\rm Mpc}]$",
    "f_growth": r"$f$",
}
VARIANTS = tuple(COLORS)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def load_fixed(name: str) -> dict[str, np.ndarray]:
    path = FIXED_ROOT / "fits" / name / "samples.npz"
    summary_path = path.with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "pass" or summary.get("output_npz_sha256") != sha256_file(path):
        raise RuntimeError(f"invalid fixed-growth source {name}")
    return load_npz(path)


def summary_from_samples(samples: np.ndarray, names: list[str], theta_ml: np.ndarray) -> dict[str, dict[str, float]]:
    flat = np.asarray(samples, dtype="f8").reshape(-1, len(names))
    quantiles = np.percentile(flat, [16.0, 50.0, 84.0], axis=0)
    return {
        name: {
            "maximum_likelihood": float(theta_ml[index]),
            "q16": float(quantiles[0, index]),
            "q50": float(quantiles[1, index]),
            "q84": float(quantiles[2, index]),
            "sigma68": float(0.5 * (quantiles[2, index] - quantiles[0, index])),
        }
        for index, name in enumerate(names)
    }


def interval_text(row: dict[str, float], *, digits: int = 2) -> str:
    ml, q16, q84 = row["maximum_likelihood"], row["q16"], row["q84"]
    if q16 <= ml <= q84:
        return f"{ml:.{digits}f} -{ml - q16:.{digits}f}/+{q84 - ml:.{digits}f}"
    return f"{ml:.{digits}f}; 68%=[{q16:.{digits}f},{q84:.{digits}f}]"


def triangle_page(pdf: PdfPages, free: dict[str, dict[str, np.ndarray]], fiducial_growth: float) -> None:
    names = ("fNL", "b1", "sigma_s", "f_growth")
    samples: dict[str, np.ndarray] = {}
    for key, value in free.items():
        parameter_names = value["parameter_names"].tolist()
        flat = value["chain_by_ensemble"].reshape(-1, len(parameter_names))
        samples[key] = np.column_stack([flat[:, parameter_names.index(name)] for name in names])
    step = {key: max(1, sample.shape[0] // 12000) for key, sample in samples.items()}
    plotted = {key: sample[:: step[key]] for key, sample in samples.items()}
    ranges = {
        name: plot_range(
            np.concatenate([sample[:, index] for sample in plotted.values()]),
            np.concatenate([sample[:, index] for sample in plotted.values()]),
            parameter=name,
        )
        for index, name in enumerate(names)
    }
    figure, axes = plt.subplots(4, 4, figsize=(9.3, 8.8), squeeze=False)
    for row, yname in enumerate(names):
        for column, xname in enumerate(names):
            axis = axes[row, column]
            if column > row:
                axis.set_axis_off()
                continue
            if row == column:
                bins = np.linspace(*ranges[xname], 80)
                for key in VARIANTS:
                    axis.hist(plotted[key][:, column], bins=bins, density=True, histtype="step", lw=1.7, color=COLORS[key])
                axis.set_xlim(*ranges[xname])
                axis.set_yticks([])
            else:
                for zorder, key in enumerate(VARIANTS, start=1):
                    draw_contour(
                        axis,
                        plotted[key][:, column],
                        plotted[key][:, row],
                        color=COLORS[key],
                        xlim=ranges[xname],
                        ylim=ranges[yname],
                        zorder=2 * zorder,
                    )
                axis.set(xlim=ranges[xname], ylim=ranges[yname])
            if xname == "fNL":
                axis.axvline(0.0, color="0.55", lw=0.8, ls="--", zorder=0)
            if xname == "f_growth":
                axis.axvline(fiducial_growth, color="#2A9D8F", lw=0.9, ls="--", zorder=0)
            if row > column and yname == "f_growth":
                axis.axhline(fiducial_growth, color="#2A9D8F", lw=0.9, ls="--", zorder=0)
            if row < 3:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel(LABELS[xname])
            if column == 0 and row > 0:
                axis.set_ylabel(LABELS[yname])
            elif column > 0 and row != column:
                axis.tick_params(labelleft=False)
    handles = [plt.Line2D([], [], color=COLORS[key], lw=2.0) for key in VARIANTS]
    legend_labels = []
    for key in VARIANTS:
        names_here = free[key]["parameter_names"].tolist()
        summary = summary_from_samples(free[key]["chain_by_ensemble"], names_here, free[key]["free_theta"])
        legend_labels.append(
            f"{DISPLAY[key]}: fNL={interval_text(summary['fNL'])}; f={interval_text(summary['f_growth'], digits=3)}"
        )
    figure.legend(handles, legend_labels, loc="upper right", bbox_to_anchor=(0.975, 0.905), frameon=False, fontsize=8.0)
    figure.suptitle("Task 4.3 halo lightcone: free-growth posteriors", fontsize=12.0, y=0.99)
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.08, top=0.91, wspace=0.08, hspace=0.08)
    pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def constraints_page(
    pdf: PdfPages,
    fixed: dict[str, dict[str, np.ndarray]],
    free: dict[str, dict[str, np.ndarray]],
    fiducial_growth: float,
) -> dict[str, Any]:
    parameters = ("fNL", "b1", "sigma_s", "f_growth")
    summaries: dict[str, Any] = {}
    for key in VARIANTS:
        fixed_names = fixed[key]["parameter_names"].tolist()
        free_names = free[key]["parameter_names"].tolist()
        summaries[key] = {
            "fixed": summary_from_samples(fixed[key]["chain_by_step"], fixed_names, fixed[key]["theta_maximum_likelihood"]),
            "free": summary_from_samples(free[key]["chain_by_ensemble"], free_names, free[key]["free_theta"]),
        }
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.8))
    ypos = np.arange(len(VARIANTS), dtype="f8")
    for axis, parameter in zip(axes.ravel(), parameters, strict=True):
        for offset, mode, marker in ((-0.10, "fixed", "s"), (0.10, "free", "o")):
            for index, key in enumerate(VARIANTS):
                if parameter == "f_growth" and mode == "fixed":
                    axis.plot(fiducial_growth, ypos[index] + offset, marker=marker, color=COLORS[key], ms=5.0, mfc="white")
                    continue
                row = summaries[key][mode][parameter]
                axis.plot([row["q16"], row["q84"]], [ypos[index] + offset] * 2, color=COLORS[key], lw=1.5, alpha=0.55 if mode == "fixed" else 1.0)
                axis.plot(row["maximum_likelihood"], ypos[index] + offset, marker=marker, color=COLORS[key], ms=5.0, mfc="white" if mode == "fixed" else COLORS[key])
        if parameter == "fNL":
            axis.axvline(0.0, color="0.55", lw=0.8, ls="--")
        if parameter == "f_growth":
            axis.axvline(fiducial_growth, color="#2A9D8F", lw=0.9, ls="--")
        axis.set_xlabel(LABELS[parameter])
        axis.set_yticks(ypos, [DISPLAY[key] for key in VARIANTS])
        axis.invert_yaxis()
        axis.grid(axis="x", color="0.90", lw=0.7)
    handles = [
        plt.Line2D([], [], color="0.35", marker="s", mfc="white", lw=1.2, label="fixed f"),
        plt.Line2D([], [], color="0.35", marker="o", lw=1.2, label="free f"),
    ]
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False)
    figure.suptitle("Maximum-likelihood centers with marginal q16-q84 intervals", fontsize=12.0)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return summaries


def segmented(x: np.ndarray, y: np.ndarray, *, xi: bool) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype="f8")
    y = np.asarray(y, dtype="f8")
    if not xi:
        return x, y
    gap = np.flatnonzero(np.diff(x) > 1.5 * np.median(np.diff(x)))
    if gap.size == 0:
        return x, y
    xo: list[float] = []
    yo: list[float] = []
    for index, (xx, yy) in enumerate(zip(x, y, strict=True)):
        xo.append(float(xx))
        yo.append(float(yy))
        if index in gap:
            xo.append(float("nan"))
            yo.append(float("nan"))
    return np.asarray(xo), np.asarray(yo)


def observable_page(
    pdf: PdfPages,
    arrays: dict[str, np.ndarray],
    *,
    source: str,
    title: str,
) -> None:
    if source == "marginal":
        pkey, xkey = "rsd_p02", "rsd_xi02"
        ppred_fixed = arrays[f"{pkey}_fixed_prediction"]
        ppred_free = arrays[f"{pkey}_free_prediction"]
        xpred_fixed = arrays[f"{xkey}_fixed_prediction"]
        xpred_free = arrays[f"{xkey}_free_prediction"]
    else:
        key = "rsd_joint_p02xi02"
        fixed_prediction = arrays[f"{key}_fixed_prediction"]
        free_prediction = arrays[f"{key}_free_prediction"]
        ppred_fixed, xpred_fixed = fixed_prediction[:26], fixed_prediction[26:]
        ppred_free, xpred_free = free_prediction[:26], free_prediction[26:]
        pkey = xkey = key
    p_data = arrays["rsd_p02_data"]
    x_data = arrays["rsd_xi02_data"]
    p_cov = arrays["rsd_p02_covariance_single"]
    x_cov = arrays["rsd_xi02_covariance_single"]
    entries = (
        (arrays["k0"], p_data[:15], p_cov[:15, :15], ppred_fixed[:15], ppred_free[:15], r"$P_0(k)$", False),
        (arrays["k2"], p_data[15:], p_cov[15:, 15:], ppred_fixed[15:], ppred_free[15:], r"$P_2(k)$", False),
        (arrays["s"], x_data[:26], x_cov[:26, :26], xpred_fixed[:26], xpred_free[:26], r"$\xi_0(s)$", True),
        (arrays["s"], x_data[26:], x_cov[26:, 26:], xpred_fixed[26:], xpred_free[26:], r"$\xi_2(s)$", True),
    )
    figure, axes = plt.subplots(2, 4, figsize=(14.2, 5.8), sharex="col", gridspec_kw={"height_ratios": [2.0, 1.0]})
    for column, (x, data, covariance, fixed_prediction, free_prediction, label, is_xi) in enumerate(entries):
        sigma = np.sqrt(np.diag(covariance))
        scale = x**2 if is_xi else np.ones_like(x)
        if is_xi:
            for row in range(2):
                axes[row, column].axvspan(80.0, 120.0, color="0.93", zorder=-5)
        axes[0, column].errorbar(x, scale * data, yerr=scale * sigma, fmt="o", ms=3.2, color="#252525", ecolor="0.60", capsize=1.3, label="x25 mean; single-realization error")
        xf, yf = segmented(x, scale * fixed_prediction, xi=is_xi)
        xn, yn = segmented(x, scale * free_prediction, xi=is_xi)
        axes[0, column].plot(xf, yf, color="#D55E00", lw=1.5, ls="--", label="fixed-f ML")
        axes[0, column].plot(xn, yn, color="#0072B2", lw=1.7, label="free-f ML")
        axes[0, column].set_title(label)
        axes[1, column].axhline(0.0, color="0.55", lw=0.8)
        xf, yf = segmented(x, (data - fixed_prediction) / sigma, xi=is_xi)
        xn, yn = segmented(x, (data - free_prediction) / sigma, xi=is_xi)
        axes[1, column].plot(xf, yf, "s--", color="#D55E00", ms=2.7, lw=0.9)
        axes[1, column].plot(xn, yn, "o-", color="#0072B2", ms=2.7, lw=0.9)
        axes[1, column].set_ylabel(r"residual / $\sigma_{\rm single}$")
        if is_xi:
            axes[0, column].set_ylabel(r"$s^2\xi_\ell(s)$")
            axes[1, column].set_xlabel(r"$s\ [h^{-1}{\rm Mpc}]$")
        else:
            axes[0, column].set_ylabel(r"$P_\ell(k)\ [(h^{-1}{\rm Mpc})^3]$")
            axes[1, column].set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
            axes[0, column].set_xscale("log")
            axes[1, column].set_xscale("log")
            ticks = [0.02, 0.04, 0.08]
            for row in range(2):
                axes[row, column].xaxis.set_major_locator(FixedLocator(ticks))
                axes[row, column].xaxis.set_minor_formatter(NullFormatter())
            axes[1, column].xaxis.set_major_formatter(FixedFormatter(["0.02", "0.04", "0.08"]))
    axes[0, 0].legend(frameon=False, fontsize=7.2)
    figure.suptitle(title, fontsize=12.0)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def fit_quality_page(pdf: PdfPages, audit: dict[str, Any], fixed_audit: dict[str, Any]) -> None:
    labels = [DISPLAY[key] for key in VARIANTS]
    fixed_chi = [fixed_audit["results"][key]["phase_diagnostics"]["chi2_mean"] / fixed_audit["results"][key]["phase_diagnostics"]["dof"] for key in VARIANTS]
    free_chi = [audit["results"][key]["phase_diagnostics"]["chi2_mean"] / audit["results"][key]["phase_diagnostics"]["dof"] for key in VARIANTS]
    fixed_pass = [fixed_audit["results"][key]["phase_diagnostics"]["fraction_pte_above_0p05"] for key in VARIANTS]
    free_pass = [audit["results"][key]["phase_diagnostics"]["fraction_pte_above_0p05"] for key in VARIANTS]
    delta = [audit["results"][key]["delta_chi2_fixed_minus_free"] for key in VARIANTS]
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.7))
    x = np.arange(3)
    width = 0.34
    axes[0].bar(x - width / 2, fixed_chi, width, color="#D55E00", alpha=0.75, label="fixed f")
    axes[0].bar(x + width / 2, free_chi, width, color="#0072B2", alpha=0.85, label="free f")
    axes[0].axhline(1.0, color="0.35", lw=0.8, ls="--")
    axes[0].set_ylabel(r"phase mean $\chi^2/\mathrm{dof}$")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x - width / 2, fixed_pass, width, color="#D55E00", alpha=0.75)
    axes[1].bar(x + width / 2, free_pass, width, color="#0072B2", alpha=0.85)
    axes[1].axhline(0.05, color="0.35", lw=0.8, ls="--")
    axes[1].set_ylabel("fraction of 25 phases with PTE > 0.05")
    axes[1].set_ylim(0.0, 1.0)
    axes[2].bar(x, delta, 0.55, color=[COLORS[key] for key in VARIANTS])
    axes[2].set_ylabel(r"$\chi^2_{\rm fixed}-\chi^2_{\rm free}$ on x25 mean")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", color="0.91", lw=0.7)
    figure.suptitle("Does freeing the growth rate improve multipole closure?", fontsize=12.0)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def main() -> None:
    output_json = OUTPUT.with_suffix(".json")
    if OUTPUT.exists() or output_json.exists():
        raise FileExistsError(f"immutable plot exists: {OUTPUT} / {output_json}")
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("free-growth chains are not validated")
    if audit["covariance_contract"]["physical"] != "single-realization normalization; divisor=1":
        raise RuntimeError("covariance normalization changed")
    arrays = load_npz(ARRAY_PATH)
    if audit["outputs"]["arrays_sha256"] != sha256_file(ARRAY_PATH):
        raise RuntimeError("free-growth arrays hash changed")
    fixed_audit_path = Path(audit["inputs"]["fixed_audit"])
    fixed_audit = json.loads(fixed_audit_path.read_text(encoding="utf-8"))
    fixed = {key: load_fixed(key) for key in VARIANTS}
    free = {
        key: {
            "chain_by_ensemble": arrays[f"{key}_chain_by_ensemble"],
            "parameter_names": arrays[f"{key}_parameter_names"],
            "free_theta": arrays[f"{key}_free_theta"],
        }
        for key in VARIANTS
    }
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary) as pdf:
        triangle_page(pdf, free, float(audit["fiducial"]["f_growth"]))
        summaries = constraints_page(pdf, fixed, free, float(audit["fiducial"]["f_growth"]))
        observable_page(pdf, arrays, source="marginal", title="Marginal P02 and xi02 fits: fixed versus free growth")
        observable_page(pdf, arrays, source="joint", title="Full-cross joint fit: fixed versus free growth")
        fit_quality_page(pdf, audit, fixed_audit)
    temporary.replace(OUTPUT)
    if OUTPUT.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError("output is not a PDF")
    payload = {
        "task": "task43_plot_lightcone_free_growth_test",
        "status": "pass",
        "figure": str(OUTPUT),
        "figure_sha256": sha256_file(OUTPUT),
        "pages": [
            "free-growth fNL-b1-sigma_s-f contours",
            "fixed-versus-free constraints",
            "marginal fixed-versus-free multipole residuals",
            "joint fixed-versus-free multipole residuals",
            "phase goodness-of-fit and delta-chi2",
        ],
        "center_convention": "maximum likelihood",
        "interval_convention": "marginal q16/q84 from two independently validated ensembles",
        "covariance_contract": "single realization; observed curve alone is the mean of 25 phases",
        "parameter_constraints": summaries,
        "fit_quality": {
            key: {
                "delta_chi2_fixed_minus_free": audit["results"][key]["delta_chi2_fixed_minus_free"],
                "fixed_phase": fixed_audit["results"][key]["phase_diagnostics"],
                "free_phase": audit["results"][key]["phase_diagnostics"],
                "marginal_block_chi2": audit["results"][key]["marginal_block_chi2"],
            }
            for key in VARIANTS
        },
        "input_audit": str(AUDIT_PATH),
        "input_audit_sha256": sha256_file(AUDIT_PATH),
        "input_arrays": str(ARRAY_PATH),
        "input_arrays_sha256": sha256_file(ARRAY_PATH),
    }
    atomic_write_json(output_json, payload)
    print(json.dumps({"status": "pass", "output": str(OUTPUT), "sha256": sha256_file(OUTPUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
