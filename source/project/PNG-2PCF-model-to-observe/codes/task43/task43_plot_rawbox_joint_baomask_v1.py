#!/usr/bin/env python3
"""Plot the strict BAO-masked rawbox marginal and joint chains."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from task43_plot_rsd_rawbox_pk0_vs_xi0_contours import draw_contour, plot_range
from task43_rsd_common import PROJECT_ROOT, atomic_write_json, sha256_file
from task43_run_rawbox_monopole_joint_baomask_v1 import DEFAULT_ROOT as MONOPOLE_ROOT
from task43_run_rawbox_joint_baomask_v1 import DEFAULT_ROOT


AUDIT = DEFAULT_ROOT / "task43_rawbox_standard_joint_baomask80_120_v1.json"
COVARIANCE = DEFAULT_ROOT / "task43_rawbox_standard_joint_baomask80_120_covariance_v1.npz"
MONOPOLE_AUDIT = MONOPOLE_ROOT / "task43_rawbox_monopole_joint_baomask80_120_v1.json"
OUTPUT = PROJECT_ROOT / "9.11meeting" / "task43_standard_kmax0p08_smin50" / "task43_rawbox_real_Pxi_rsd_P02xi02_joint_kmax0p08_smin50_baomask80_120_v1.pdf"
COLORS = {"p": "#252525", "xi": "#C44E52", "joint": "#4C72B0"}


def load_variant(name: str, audit: dict[str, Any], *, source_root: Path | None = None) -> dict[str, Any]:
    if source_root is None:
        source_root = DEFAULT_ROOT
    root = source_root / "fits" / name
    npz_path, json_path = root / "samples.npz", root / "summary.json"
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "pass" or metadata.get("output_npz_sha256") != sha256_file(npz_path):
        raise RuntimeError(f"unvalidated chain: {name}")
    if not all(metadata["result"]["mcmc"]["gates"].values()):
        raise RuntimeError(f"failed convergence gates: {name}")
    with np.load(npz_path, allow_pickle=False) as payload:
        result = {key: np.asarray(payload[key]) for key in payload.files}
    result.update({"metadata": metadata, "npz_path": npz_path, "json_path": json_path})
    expected_names = audit["results"][name]["parameter_names"]
    if result["parameter_names"].tolist() != expected_names:
        raise RuntimeError(f"parameter names changed for {name}")
    return result


def parameter_summary(variant: dict[str, Any], parameter: str) -> dict[str, float]:
    names = variant["parameter_names"].tolist()
    index = names.index(parameter)
    samples = np.asarray(variant["chain_by_step"], dtype="f8").reshape(-1, len(names))[:, index]
    q16, q50, q84 = np.percentile(samples, [16.0, 50.0, 84.0])
    maximum_likelihood = float(np.asarray(variant["theta_maximum_likelihood"], dtype="f8")[index])
    return {
        "maximum_likelihood": maximum_likelihood,
        "q16": float(q16),
        "q50": float(q50),
        "q84": float(q84),
        "sigma68": float(0.5 * (q84 - q16)),
    }


def interval_text(variant: dict[str, Any], parameter: str) -> str:
    summary = parameter_summary(variant, parameter)
    maximum_likelihood, q16, q84 = summary["maximum_likelihood"], summary["q16"], summary["q84"]
    if q16 <= maximum_likelihood <= q84:
        return rf"{maximum_likelihood:.2f}_{{-{maximum_likelihood - q16:.2f}}}^{{+{q84 - maximum_likelihood:.2f}}}"
    return rf"{maximum_likelihood:.2f};\ 68\%=[{q16:.2f},{q84:.2f}]"


def independent_marginal_tension(
    left: dict[str, Any], right: dict[str, Any], parameter: str
) -> dict[str, float | str]:
    left_summary = parameter_summary(left, parameter)
    right_summary = parameter_summary(right, parameter)
    difference = right_summary["q50"] - left_summary["q50"]
    denominator = float(np.hypot(left_summary["sigma68"], right_summary["sigma68"]))
    return {
        "definition": "abs(q50_right-q50_left)/hypot(sigma68_left,sigma68_right); ignores cross-correlation",
        "signed_q50_difference_right_minus_left": difference,
        "quadrature_sigma68": denominator,
        "tension_sigma": abs(difference) / denominator,
    }


def triangle_page(
    pdf: PdfPages,
    variants: tuple[tuple[str, dict[str, Any], str], ...],
    names: tuple[str, ...],
    *,
    title: str,
    truth_fnl: float = 0.0,
    plot_stride: int = 8,
) -> None:
    labels = {
        "fNL": r"$f_{\rm NL}$",
        "b1": r"$b_1$",
        "sigma_s": r"$\sigma_s\ [h^{-1}{\rm Mpc}]$",
    }
    samples = {
        key: np.asarray(variant["chain_by_step"], dtype="f8").reshape(-1, len(variant["parameter_names"]))[:, : len(names)]
        for key, variant, _ in variants
    }
    if int(plot_stride) < 1:
        raise ValueError("plot_stride must be at least one")
    plotted = {key: value[:: int(plot_stride)] for key, value in samples.items()}
    ranges = {
        name: plot_range(
            np.concatenate([value[:, index] for value in plotted.values()]),
            np.concatenate([value[:, index] for value in plotted.values()]),
            parameter=name,
        )
        for index, name in enumerate(names)
    }
    ndim = len(names)
    figure, axes = plt.subplots(ndim, ndim, figsize=(8.4, 7.9) if ndim == 3 else (6.8, 6.0))
    axes = np.atleast_2d(axes)
    for row, yname in enumerate(names):
        for column, xname in enumerate(names):
            axis = axes[row, column]
            if column > row:
                axis.set_axis_off()
                continue
            if row == column:
                bins = np.linspace(*ranges[xname], 90)
                for key, _, _ in variants:
                    axis.hist(
                        plotted[key][:, column],
                        bins=bins,
                        density=True,
                        histtype="step",
                        lw=1.8,
                        color=COLORS[key],
                    )
                axis.set_xlim(*ranges[xname])
                axis.set_yticks([])
            else:
                for zorder, (key, _, _) in enumerate(variants, start=1):
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
                axis.axvline(float(truth_fnl), color="0.55", lw=0.8, ls="--", zorder=0)
            if row < ndim - 1:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel(labels[xname])
            if column == 0 and row > 0:
                axis.set_ylabel(labels[yname])
            elif column > 0 and row != column:
                axis.tick_params(labelleft=False)
    handles = [plt.Line2D([], [], color=COLORS[key], lw=2.0) for key, _, _ in variants]
    legend_labels = []
    for key, variant, display in variants:
        text = rf"{display}: $f_{{\rm NL}}^{{\rm ML}}={interval_text(variant, 'fNL')}$"
        if "sigma_s" in names:
            text += rf", $\sigma_s^{{\rm ML}}={interval_text(variant, 'sigma_s')}$"
        legend_labels.append(text)
    figure.legend(
        handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.965, 0.875),
        frameon=False,
        fontsize=8.8,
    )
    figure.suptitle(title, fontsize=12.0, y=0.985)
    figure.subplots_adjust(left=0.13, right=0.97, bottom=0.10, top=0.93, wspace=0.08, hspace=0.08)
    pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def observable_page(
    pdf: PdfPages,
    xvalues: tuple[np.ndarray, ...],
    data_vectors: tuple[np.ndarray, ...],
    covariance_blocks: tuple[np.ndarray, ...],
    marginal_predictions: tuple[np.ndarray, ...],
    joint_predictions: tuple[np.ndarray, ...],
    labels: tuple[str, ...],
    *,
    title: str,
    xi_flags: tuple[bool, ...],
) -> None:
    ncolumn = len(labels)
    figure, axes = plt.subplots(
        2,
        ncolumn,
        figsize=(3.65 * ncolumn, 6.0),
        sharex="col",
        gridspec_kw={"height_ratios": [2.1, 1.0]},
        squeeze=False,
    )
    for column, (x, data, covariance, marginal, joint, label, is_xi) in enumerate(
        zip(xvalues, data_vectors, covariance_blocks, marginal_predictions, joint_predictions, labels, xi_flags, strict=True)
    ):
        sigma_single = np.sqrt(np.diag(np.asarray(covariance, dtype="f8")))
        scale = np.asarray(x, dtype="f8") ** 2 if is_xi else np.ones_like(x, dtype="f8")
        axes[0, column].errorbar(
            x,
            scale * data,
            yerr=scale * sigma_single,
            fmt="o",
            ms=3.5,
            color="#252525",
            ecolor="0.55",
            capsize=1.5,
            label="x25 mean; single-realization error",
        )
        axes[0, column].plot(x, scale * marginal, color=COLORS["xi"], lw=1.6, label="marginal ML")
        axes[0, column].plot(x, scale * joint, color=COLORS["joint"], lw=1.6, ls="--", label="joint ML")
        axes[0, column].set_title(label)
        axes[0, column].legend(frameon=False, fontsize=7.3)
        axes[1, column].axhline(0.0, color="0.55", lw=0.8)
        axes[1, column].plot(x, (data - marginal) / sigma_single, "o-", color=COLORS["xi"], ms=3.0, lw=0.8)
        axes[1, column].plot(x, (data - joint) / sigma_single, "s--", color=COLORS["joint"], ms=2.8, lw=0.8)
        axes[1, column].set_ylabel(r"residual / $\sigma_{\rm single}$")
        if is_xi:
            axes[0, column].set_ylabel(r"$s^2\xi_\ell(s)$")
            axes[1, column].set_xlabel(r"$s\ [h^{-1}{\rm Mpc}]$")
        else:
            axes[0, column].set_ylabel(r"$P_\ell(k)\ [(h^{-1}{\rm Mpc})^3]$")
            axes[1, column].set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
            axes[0, column].set_xscale("log")
            axes[1, column].set_xscale("log")
    figure.suptitle(title, fontsize=12.0)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def main() -> None:
    output_json = OUTPUT.with_suffix(".json")
    if OUTPUT.exists() or output_json.exists():
        raise FileExistsError(f"immutable plot exists: {OUTPUT} / {output_json}")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or not all(audit["numerical_gates"].values()):
        raise RuntimeError("joint inference audit failed")
    if audit["covariance_contract"]["plot_errorbars"] != "sqrt(diag(C_single)); never divided by sqrt(25)":
        raise RuntimeError("single-realization errorbar contract changed")
    contract = audit.get("fit_contract", {})
    contract_label = f"kmax={float(contract.get('pk_kmax_h_mpc', 0.08)):.2f}, smin={float(contract.get('xi_smin_mpc_h', 50.0)):.0f}"
    variants = {name: load_variant(name, audit) for name in audit["results"]}
    monopole_audit = json.loads(MONOPOLE_AUDIT.read_text(encoding="utf-8"))
    if monopole_audit.get("status") != "pass" or not all(monopole_audit["numerical_gates"].values()):
        raise RuntimeError("monopole joint inference audit failed")
    if monopole_audit["inputs"]["source_covariance_sha256"] != sha256_file(COVARIANCE):
        raise RuntimeError("monopole and four-way covariance inputs differ")
    monopole_variants = {
        name: load_variant(name, monopole_audit, source_root=MONOPOLE_ROOT)
        for name in monopole_audit["results"]
    }
    with np.load(COVARIANCE, allow_pickle=False) as payload:
        if audit["outputs"]["covariance_npz_sha256"] != sha256_file(COVARIANCE):
            raise RuntimeError("covariance hash changed")
        k = np.asarray(payload["k"], dtype="f8")
        centers = np.asarray(payload["s_centers"], dtype="f8")
        mask = np.asarray(payload["xi_mask"], dtype=bool)
        covariance = {key: np.asarray(payload[key], dtype="f8") for key in ("real_pp", "real_xx", "rsd_pp", "rsd_xx")}
    s = centers[mask]
    n_real_p = int(np.asarray(variants["real_p"]["data"]).size)
    n_real_xi = int(np.asarray(variants["real_xi"]["data"]).size)
    n_rsd_p0 = int(np.asarray(monopole_variants["rsd_p0"]["data"]).size)
    n_rsd_p2 = int(np.asarray(variants["rsd_p02"]["data"]).size) - n_rsd_p0
    n_rsd_xi0 = int(np.asarray(monopole_variants["rsd_xi0"]["data"]).size)
    n_rsd_xi2 = int(np.asarray(variants["rsd_xi02"]["data"]).size) - n_rsd_xi0

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
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary) as pdf:
        triangle_page(
            pdf,
            (
                ("p", variants["real_p"], r"$P(k)$"),
                ("xi", variants["real_xi"], r"$\xi(s)$"),
                ("joint", variants["real_joint"], "joint"),
            ),
            ("fNL", "b1"),
            title=f"Rawbox real space: P, BAO-masked xi, and joint ({contract_label})",
        )
        triangle_page(
            pdf,
            (
                ("p", monopole_variants["rsd_p0"], r"$P_0$"),
                ("xi", monopole_variants["rsd_xi0"], r"$\xi_0$"),
                ("joint", monopole_variants["rsd_joint_p0xi0"], "joint"),
            ),
            ("fNL", "b1", "sigma_s"),
            title=f"Rawbox redshift space: P0, BAO-masked xi0, and joint ({contract_label})",
        )
        triangle_page(
            pdf,
            (
                ("p", variants["rsd_p02"], r"$P_0+P_2$"),
                ("xi", variants["rsd_xi02"], r"$\xi_0+\xi_2$"),
                ("joint", variants["rsd_joint"], "joint"),
            ),
            ("fNL", "b1", "sigma_s"),
            title=f"Rawbox redshift space: P02, BAO-masked xi02, and joint ({contract_label})",
        )
        real_p, real_x, real_j = variants["real_p"], variants["real_xi"], variants["real_joint"]
        real_joint_prediction = np.asarray(real_j["prediction_maximum_likelihood"], dtype="f8")
        observable_page(
            pdf,
            (k, s),
            (np.asarray(real_p["data"]), np.asarray(real_x["data"])),
            (covariance["real_pp"], covariance["real_xx"]),
            (np.asarray(real_p["prediction_maximum_likelihood"]), np.asarray(real_x["prediction_maximum_likelihood"])),
            (real_joint_prediction[:n_real_p], real_joint_prediction[n_real_p:]),
            (r"real-space $P(k)$", r"real-space $\xi(s)$"),
            title=f"Rawbox real-space observables ({contract_label}): only the curve is averaged over 25 realizations",
            xi_flags=(False, True),
        )
        rsd_p, rsd_x, rsd_j = variants["rsd_p02"], variants["rsd_xi02"], variants["rsd_joint"]
        p_data = np.asarray(rsd_p["data"], dtype="f8")
        x_data = np.asarray(rsd_x["data"], dtype="f8")
        p_prediction = np.asarray(rsd_p["prediction_maximum_likelihood"], dtype="f8")
        x_prediction = np.asarray(rsd_x["prediction_maximum_likelihood"], dtype="f8")
        joint_prediction = np.asarray(rsd_j["prediction_maximum_likelihood"], dtype="f8")
        observable_page(
            pdf,
            (k, k, s, s),
            (p_data[:n_rsd_p0], p_data[n_rsd_p0:], x_data[:n_rsd_xi0], x_data[n_rsd_xi0:]),
            (
                covariance["rsd_pp"][:n_rsd_p0, :n_rsd_p0],
                covariance["rsd_pp"][n_rsd_p0:, n_rsd_p0:],
                covariance["rsd_xx"][:n_rsd_xi0, :n_rsd_xi0],
                covariance["rsd_xx"][n_rsd_xi0:, n_rsd_xi0:],
            ),
            (p_prediction[:n_rsd_p0], p_prediction[n_rsd_p0:], x_prediction[:n_rsd_xi0], x_prediction[n_rsd_xi0:]),
            (
                joint_prediction[:n_rsd_p0],
                joint_prediction[n_rsd_p0:n_rsd_p0 + n_rsd_p2],
                joint_prediction[n_rsd_p0 + n_rsd_p2:n_rsd_p0 + n_rsd_p2 + n_rsd_xi0],
                joint_prediction[n_rsd_p0 + n_rsd_p2 + n_rsd_xi0:],
            ),
            (r"$P_0(k)$", r"$P_2(k)$", r"$\xi_0(s)$", r"$\xi_2(s)$"),
            title=f"Rawbox RSD observables ({contract_label}): only the curve is averaged over 25 realizations",
            xi_flags=(False, False, True, True),
        )
    temporary.replace(OUTPUT)
    if OUTPUT.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError("output is not a PDF")
    all_variants = {**variants, **monopole_variants}
    parameter_constraints = {
        name: {
            parameter: parameter_summary(variant, parameter)
            for parameter in variant["parameter_names"].tolist()
        }
        for name, variant in all_variants.items()
    }
    real_p_sigma = parameter_constraints["real_p"]["fNL"]["sigma68"]
    real_xi_sigma = parameter_constraints["real_xi"]["fNL"]["sigma68"]
    rsd_p_sigma = parameter_constraints["rsd_p02"]["fNL"]["sigma68"]
    rsd_xi_sigma = parameter_constraints["rsd_xi02"]["fNL"]["sigma68"]
    comparison_metrics = {
        "real_joint_fNL_improvement_over_best_marginal_fraction": 1.0
        - parameter_constraints["real_joint"]["fNL"]["sigma68"] / min(real_p_sigma, real_xi_sigma),
        "rsd_joint_fNL_improvement_over_best_marginal_fraction": 1.0
        - parameter_constraints["rsd_joint"]["fNL"]["sigma68"] / min(rsd_p_sigma, rsd_xi_sigma),
        "rsd_marginal_tensions": {
            parameter: independent_marginal_tension(variants["rsd_p02"], variants["rsd_xi02"], parameter)
            for parameter in ("fNL", "b1", "sigma_s")
        },
        "rsd_monopole_joint_fNL_improvement_over_best_marginal_fraction": 1.0
        - parameter_constraints["rsd_joint_p0xi0"]["fNL"]["sigma68"]
        / min(
            parameter_constraints["rsd_p0"]["fNL"]["sigma68"],
            parameter_constraints["rsd_xi0"]["fNL"]["sigma68"],
        ),
    }
    atomic_write_json(
        output_json,
        {
            "task": "task43_plot_rawbox_joint_baomask_v1p2",
            "status": "pass",
            "pages": [
                "real contours",
                "RSD monopole contours",
                "RSD multipole contours",
                "real observables",
                "RSD observables",
            ],
            "fit_contract": audit.get("fit_contract"),
            "errorbar_contract": "sqrt(diag(C_single)); only observed curves are x25 means",
            "parameter_constraints": parameter_constraints,
            "comparison_metrics": comparison_metrics,
            "input_audit": str(AUDIT),
            "input_audit_sha256": sha256_file(AUDIT),
            "input_covariance": str(COVARIANCE),
            "input_covariance_sha256": sha256_file(COVARIANCE),
            "input_monopole_audit": str(MONOPOLE_AUDIT),
            "input_monopole_audit_sha256": sha256_file(MONOPOLE_AUDIT),
            "output_pdf": str(OUTPUT),
            "output_pdf_sha256": sha256_file(OUTPUT),
        },
    )
    print(json.dumps({"status": "pass", "output": str(OUTPUT), "sha256": sha256_file(OUTPUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
