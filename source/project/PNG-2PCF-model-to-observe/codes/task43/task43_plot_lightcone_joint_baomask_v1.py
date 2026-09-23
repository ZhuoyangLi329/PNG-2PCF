#!/usr/bin/env python3
"""Plot the standard BAO-masked Task 4.3 lightcone P/xi fits."""

from __future__ import annotations

import argparse
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

from task43_plot_rawbox_joint_baomask_v1 import (
    COLORS,
    independent_marginal_tension,
    parameter_summary,
    triangle_page,
)
from task43_rsd_common import PROJECT_ROOT, atomic_write_json, sha256_file
from task43_run_lightcone_joint_baomask_v1 import DEFAULT_ROOT


AUDIT = DEFAULT_ROOT / "task43_lightcone_standard_joint_baomask80_120_v1.json"
COVARIANCE = DEFAULT_ROOT / "task43_lightcone_standard_joint_baomask80_120_covariance_v1.npz"
OUTPUT = PROJECT_ROOT / "9.11meeting" / "task43_standard_kmax0p08_smin50" / "task43_lightcone_real_Pxi_rsd_P02xi02_joint_kmax0p08_smin50_baomask80_120_v1.pdf"


def load_variant(name: str, audit: dict[str, Any]) -> dict[str, Any]:
    root = DEFAULT_ROOT / "fits" / name
    npz_path, json_path = root / "samples.npz", root / "summary.json"
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "pass" or metadata.get("output_npz_sha256") != sha256_file(npz_path):
        raise RuntimeError(f"unvalidated chain: {name}")
    if not all(metadata["result"]["mcmc"]["gates"].values()):
        raise RuntimeError(f"failed convergence gates: {name}")
    with np.load(npz_path, allow_pickle=False) as payload:
        result = {key: np.asarray(payload[key]) for key in payload.files}
    result.update({"metadata": metadata, "npz_path": npz_path, "json_path": json_path})
    if result["parameter_names"].tolist() != audit["results"][name]["parameter_names"]:
        raise RuntimeError(f"parameter names changed for {name}")
    return result


def maximum_likelihood_tension(
    left: dict[str, Any], right: dict[str, Any], parameter: str
) -> dict[str, float | str]:
    left_summary = parameter_summary(left, parameter)
    right_summary = parameter_summary(right, parameter)
    difference = right_summary["maximum_likelihood"] - left_summary["maximum_likelihood"]
    denominator = float(np.hypot(left_summary["sigma68"], right_summary["sigma68"]))
    return {
        "definition": "abs(ML_right-ML_left)/hypot(sigma68_left,sigma68_right); ignores cross-correlation",
        "signed_maximum_likelihood_difference_right_minus_left": difference,
        "quadrature_sigma68": denominator,
        "tension_sigma": abs(difference) / denominator,
    }


def segmented_curve(x: np.ndarray, y: np.ndarray, *, is_xi: bool) -> tuple[np.ndarray, np.ndarray]:
    xvalues = np.asarray(x, dtype="f8")
    yvalues = np.asarray(y, dtype="f8")
    if not is_xi or xvalues.size < 3:
        return xvalues, yvalues
    spacing = float(np.median(np.diff(xvalues)))
    gaps = np.flatnonzero(np.diff(xvalues) > 1.5 * spacing)
    if gaps.size == 0:
        return xvalues, yvalues
    xout, yout = [], []
    for index, (xvalue, yvalue) in enumerate(zip(xvalues, yvalues, strict=True)):
        xout.append(float(xvalue))
        yout.append(float(yvalue))
        if index in gaps:
            xout.append(float("nan"))
            yout.append(float("nan"))
    return np.asarray(xout), np.asarray(yout)


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
    mean_label: str = "x25 mean; single-realization error",
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
    rows = zip(
        xvalues,
        data_vectors,
        covariance_blocks,
        marginal_predictions,
        joint_predictions,
        labels,
        xi_flags,
        strict=True,
    )
    for column, (x, data, covariance, marginal, joint, label, is_xi) in enumerate(rows):
        x = np.asarray(x, dtype="f8")
        data = np.asarray(data, dtype="f8")
        marginal = np.asarray(marginal, dtype="f8")
        joint = np.asarray(joint, dtype="f8")
        sigma_single = np.sqrt(np.diag(np.asarray(covariance, dtype="f8")))
        scale = x**2 if is_xi else np.ones_like(x)
        if is_xi:
            for row in range(2):
                axes[row, column].axvspan(80.0, 120.0, color="0.92", zorder=-5)
        axes[0, column].errorbar(
            x,
            scale * data,
            yerr=scale * sigma_single,
            fmt="o",
            ms=3.5,
            color="#252525",
            ecolor="0.55",
            capsize=1.5,
            label=mean_label,
        )
        xm, ym = segmented_curve(x, scale * marginal, is_xi=is_xi)
        xj, yj = segmented_curve(x, scale * joint, is_xi=is_xi)
        axes[0, column].plot(xm, ym, color=COLORS["xi"], lw=1.6, label="marginal ML")
        axes[0, column].plot(xj, yj, color=COLORS["joint"], lw=1.6, ls="--", label="joint ML")
        axes[0, column].set_title(label)
        axes[1, column].axhline(0.0, color="0.55", lw=0.8)
        xm, ym = segmented_curve(x, (data - marginal) / sigma_single, is_xi=is_xi)
        xj, yj = segmented_curve(x, (data - joint) / sigma_single, is_xi=is_xi)
        axes[1, column].plot(xm, ym, "o-", color=COLORS["xi"], ms=3.0, lw=0.8)
        axes[1, column].plot(xj, yj, "s--", color=COLORS["joint"], ms=2.8, lw=0.8)
        axes[1, column].set_ylabel(r"residual / $\sigma_{\rm single}$")
        if is_xi:
            axes[0, column].set_ylabel(r"$s^2\xi_\ell(s)$")
            axes[1, column].set_xlabel(r"$s\ [h^{-1}{\rm Mpc}]$")
        else:
            axes[0, column].set_ylabel(r"$P_\ell(k)\ [(h^{-1}{\rm Mpc})^3]$")
            axes[1, column].set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
            axes[0, column].set_xscale("log")
            axes[1, column].set_xscale("log")
            if float(np.min(x)) > 0.015:
                ticks = [0.02, 0.04, 0.08]
                for row in range(2):
                    axes[row, column].xaxis.set_major_locator(FixedLocator(ticks))
                    axes[row, column].xaxis.set_minor_formatter(NullFormatter())
                axes[1, column].xaxis.set_major_formatter(FixedFormatter(["0.02", "0.04", "0.08"]))
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(title, fontsize=12.0, y=0.99)
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        frameon=False,
        fontsize=7.3,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def main(*, force: bool = False) -> None:
    output_json = OUTPUT.with_suffix(".json")
    if not force and (OUTPUT.exists() or output_json.exists()):
        raise FileExistsError(f"immutable plot exists: {OUTPUT} / {output_json}")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or not all(audit["numerical_gates"].values()):
        raise RuntimeError("lightcone inference audit failed")
    errorbar_contract = str(audit["covariance_contract"]["plot_errorbars"])
    if not errorbar_contract.startswith("sqrt(diag(C_single));"):
        raise RuntimeError("single-realization errorbar contract changed")
    truth_fnl = float(audit.get("injected_fnl", 0.0))
    mean_count = int(audit.get("mean_realization_count", 25))
    mean_label = str(audit.get("observed_curve_label", f"x{mean_count} mean; single-realization error"))
    plot_label = str(audit.get("plot_label", "Lightcone"))
    contract = audit.get("fit_contract", {})
    contract_label = f"kmax={float(contract.get('pk_kmax_h_mpc', 0.08)):.2f}, smin={float(contract.get('xi_smin_mpc_h', 50.0)):.0f}"
    variants = {name: load_variant(name, audit) for name in audit["results"]}
    with np.load(COVARIANCE, allow_pickle=False) as payload:
        if audit["outputs"]["covariance_npz_sha256"] != sha256_file(COVARIANCE):
            raise RuntimeError("covariance hash changed")
        arrays = {key: np.asarray(payload[key]) for key in payload.files}

    real_s = arrays["real_s"][arrays["real_xi_mask"].astype(bool)]
    rsd_s = arrays["rsd_s"][arrays["rsd_xi_mask"].astype(bool)]
    rsd_k0 = arrays["rsd_k"]
    rsd_k2 = rsd_k0[arrays["rsd_p2_keep_indices"].astype(int)]
    n_real_p = int(np.asarray(arrays["real_k"]).size)
    n_real_xi = int(real_s.size)
    n_rsd_p0 = int(rsd_k0.size)
    n_rsd_p2 = int(rsd_k2.size)
    n_rsd_xi = int(rsd_s.size)
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
                ("p", variants["real_p0"], r"$P_0$"),
                ("xi", variants["real_xi0"], r"$\xi_0$"),
                ("joint", variants["real_joint_p0xi0"], "joint"),
            ),
            ("fNL", "b1"),
            title=f"{plot_label} real space (0.6 < z < 0.8; {contract_label}): P0, BAO-masked xi0, and joint",
            truth_fnl=truth_fnl,
        )
        triangle_page(
            pdf,
            (
                ("p", variants["rsd_p0"], r"$P_0$"),
                ("xi", variants["rsd_xi0"], r"$\xi_0$"),
                ("joint", variants["rsd_joint_p0xi0"], "joint"),
            ),
            ("fNL", "b1"),
            title=f"{plot_label} RSD (box-safe 0.4 < z_obs < 0.8; {contract_label}): P0, BAO-masked xi0, and joint",
            truth_fnl=truth_fnl,
            plot_stride=1,
        )
        triangle_page(
            pdf,
            (
                ("p", variants["rsd_p02"], r"$P_0+P_2$"),
                ("xi", variants["rsd_xi02"], r"$\xi_0+\xi_2$"),
                ("joint", variants["rsd_joint_p02xi02"], "full joint"),
            ),
            ("fNL", "b1"),
            title=f"{plot_label} RSD (box-safe 0.4 < z_obs < 0.8; {contract_label}): P02, BAO-masked xi02, and joint",
            truth_fnl=truth_fnl,
            plot_stride=1,
        )

        real_p, real_x, real_j = variants["real_p0"], variants["real_xi0"], variants["real_joint_p0xi0"]
        real_joint_prediction = np.asarray(real_j["prediction_maximum_likelihood"], dtype="f8")
        observable_page(
            pdf,
            (arrays["real_k"], real_s),
            (real_p["data"], real_x["data"]),
            (arrays["real_pp"], arrays["real_xx"]),
            (real_p["prediction_maximum_likelihood"], real_x["prediction_maximum_likelihood"]),
            (real_joint_prediction[:n_real_p], real_joint_prediction[n_real_p:]),
            (r"real-space $P_0(k)$", r"real-space $\xi_0(s)$"),
            title=f"{plot_label} real-space observables ({contract_label}): arithmetic mean of {mean_count} realizations",
            xi_flags=(False, True),
            mean_label=mean_label,
        )

        rsd_p0, rsd_x0, rsd_j0 = variants["rsd_p0"], variants["rsd_xi0"], variants["rsd_joint_p0xi0"]
        rsd_joint0_prediction = np.asarray(rsd_j0["prediction_maximum_likelihood"], dtype="f8")
        observable_page(
            pdf,
            (rsd_k0, rsd_s),
            (rsd_p0["data"], rsd_x0["data"]),
            (arrays["rsd_pp0"], arrays["rsd_xx0"]),
            (rsd_p0["prediction_maximum_likelihood"], rsd_x0["prediction_maximum_likelihood"]),
            (rsd_joint0_prediction[:n_rsd_p0], rsd_joint0_prediction[n_rsd_p0:]),
            (r"RSD $P_0(k)$", r"RSD $\xi_0(s)$"),
            title=f"{plot_label} RSD monopoles ({contract_label}): arithmetic mean of {mean_count} realizations",
            xi_flags=(False, True),
            mean_label=mean_label,
        )

        rsd_p, rsd_x, rsd_j = variants["rsd_p02"], variants["rsd_xi02"], variants["rsd_joint_p02xi02"]
        p_data = np.asarray(rsd_p["data"], dtype="f8")
        x_data = np.asarray(rsd_x["data"], dtype="f8")
        p_prediction = np.asarray(rsd_p["prediction_maximum_likelihood"], dtype="f8")
        x_prediction = np.asarray(rsd_x["prediction_maximum_likelihood"], dtype="f8")
        joint_prediction = np.asarray(rsd_j["prediction_maximum_likelihood"], dtype="f8")
        observable_page(
            pdf,
            (rsd_k0, rsd_k2, rsd_s, rsd_s),
            (p_data[:n_rsd_p0], p_data[n_rsd_p0:], x_data[:n_rsd_xi], x_data[n_rsd_xi:]),
            (
                arrays["rsd_pp"][:n_rsd_p0, :n_rsd_p0],
                arrays["rsd_pp"][n_rsd_p0:, n_rsd_p0:],
                arrays["rsd_xx"][:n_rsd_xi, :n_rsd_xi],
                arrays["rsd_xx"][n_rsd_xi:, n_rsd_xi:],
            ),
            (p_prediction[:n_rsd_p0], p_prediction[n_rsd_p0:], x_prediction[:n_rsd_xi], x_prediction[n_rsd_xi:]),
            (
                joint_prediction[:n_rsd_p0],
                joint_prediction[n_rsd_p0:n_rsd_p0 + n_rsd_p2],
                joint_prediction[n_rsd_p0 + n_rsd_p2:n_rsd_p0 + n_rsd_p2 + n_rsd_xi],
                joint_prediction[n_rsd_p0 + n_rsd_p2 + n_rsd_xi:],
            ),
            (r"RSD $P_0(k)$", r"RSD $P_2(k)$", r"RSD $\xi_0(s)$", r"RSD $\xi_2(s)$"),
            title=f"{plot_label} RSD multipoles ({contract_label}): arithmetic mean of {mean_count} realizations",
            xi_flags=(False, False, True, True),
            mean_label=mean_label,
        )
    temporary.replace(OUTPUT)
    if OUTPUT.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError("output is not a PDF")

    constraints = {
        name: {
            parameter: parameter_summary(variant, parameter)
            for parameter in variant["parameter_names"].tolist()
        }
        for name, variant in variants.items()
    }
    comparisons: dict[str, Any] = {}
    for group, left, right, joint in (
        ("real", "real_p0", "real_xi0", "real_joint_p0xi0"),
        ("rsd_monopole", "rsd_p0", "rsd_xi0", "rsd_joint_p0xi0"),
        ("rsd_multipole", "rsd_p02", "rsd_xi02", "rsd_joint_p02xi02"),
    ):
        shared = tuple(name for name in variants[left]["parameter_names"].tolist() if name in variants[right]["parameter_names"].tolist())
        comparisons[group] = {
            "joint_fNL_improvement_over_best_marginal_fraction": 1.0
            - constraints[joint]["fNL"]["sigma68"]
            / min(constraints[left]["fNL"]["sigma68"], constraints[right]["fNL"]["sigma68"]),
            "q50_marginal_tensions": {
                parameter: independent_marginal_tension(variants[left], variants[right], parameter)
                for parameter in shared
            },
            "maximum_likelihood_marginal_tensions": {
                parameter: maximum_likelihood_tension(variants[left], variants[right], parameter)
                for parameter in shared
            },
        }
    phase_shape = {
        name: {
            "dof": audit["results"][name]["phase_diagnostics"]["dof"],
            "chi2_mean": audit["results"][name]["phase_diagnostics"]["chi2_mean"],
            "fraction_pte_above_0p05": audit["results"][name]["phase_diagnostics"][
                "fraction_pte_above_0p05"
            ],
        }
        for name in audit["results"]
    }
    phase_shape_key = f"phase_shape_diagnostics_at_{mean_count}_realization_mean_ml_using_csingle"
    atomic_write_json(
        output_json,
        {
            "task": "task43_plot_lightcone_joint_baomask_v1",
            "status": "pass",
            "scope": "within-space P/xi consistency; real and RSD redshift ranges intentionally differ",
            "pages": [
                "real P0/xi0/joint contours",
                "RSD P0/xi0/joint fNL-b1 corner; sigma_s panels excluded",
                "RSD P02/xi02/full-joint fNL-b1 corner; sigma_s panels excluded",
                "real observables",
                "RSD monopole observables",
                "RSD multipole observables",
            ],
            "RSD_contour_parameters": ["fNL", "b1"],
            "RSD_contour_explicitly_excluded_parameters": ["sigma_s", "sn0"],
            "RSD_contour_sampling_contract": "all post-burn samples; no plotting-time thinning",
            "fit_contract": audit.get("fit_contract"),
            "errorbar_contract": errorbar_contract,
            "parameter_constraints": constraints,
            "comparison_metrics": comparisons,
            "scientific_assessment": {
                "classification": "P/xi parameter consistency and RSD phase-level shape closure are reported without a hard-coded verdict",
                "reason": (
                    "See comparison_metrics for P/xi maximum-likelihood tensions and "
                    f"scientific_assessment.{phase_shape_key} for the audited phase-level closure."
                ),
                phase_shape_key: phase_shape,
            },
            "input_audit": str(AUDIT),
            "input_audit_sha256": sha256_file(AUDIT),
            "input_covariance": str(COVARIANCE),
            "input_covariance_sha256": sha256_file(COVARIANCE),
            "output_pdf": str(OUTPUT),
            "output_pdf_sha256": sha256_file(OUTPUT),
        },
    )
    print(json.dumps({"status": "pass", "output": str(OUTPUT), "sha256": sha256_file(OUTPUT)}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    main(force=arguments.force)
