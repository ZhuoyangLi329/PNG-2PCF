#!/usr/bin/env python3
"""Compare standard Task 4.3 lightcone smin=50 and smin=40 fits."""

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
from task43_run_lightcone_joint_baomask_smin40_v1 import DEFAULT_ROOT as NEW_ROOT
from task43_run_lightcone_joint_baomask_v1 import DEFAULT_ROOT as OLD_ROOT


OLD_AUDIT = OLD_ROOT / "task43_lightcone_standard_joint_baomask80_120_v1.json"
NEW_AUDIT = NEW_ROOT / "task43_lightcone_joint_smin40_baomask80_120_v1.json"
OUTPUT = PROJECT_ROOT / "9.11meeting/task43_lightcone_smin40_vs50_baomask80_120_v1.pdf"

COLORS = {
    "p": "#252525",
    "xi50": "#E28B84",
    "xi40": "#B53B45",
    "joint50": "#8DB7D7",
    "joint40": "#356D9C",
}
LINESTYLES = {"p": "-", "xi50": "--", "xi40": "-", "joint50": "--", "joint40": "-"}


def load_chain(root: Path, name: str, expected: dict[str, Any]) -> dict[str, Any]:
    summary_path = root / "fits" / name / "summary.json"
    samples_path = root / "fits" / name / "samples.npz"
    metadata = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        metadata.get("status") != "pass"
        or not all(metadata["result"]["mcmc"]["gates"].values())
        or metadata.get("output_npz_sha256") != sha256_file(samples_path)
        or metadata["result"] != expected
    ):
        raise RuntimeError(f"unvalidated chain: {root.name}/{name}")
    with np.load(samples_path, allow_pickle=False) as payload:
        chain = {key: np.asarray(payload[key]) for key in payload.files}
    chain.update({"metadata": metadata, "summary_path": summary_path, "samples_path": samples_path})
    return chain


def summary(chain: dict[str, Any], parameter: str) -> dict[str, float]:
    names = chain["parameter_names"].tolist()
    index = names.index(parameter)
    samples = np.asarray(chain["chain_by_step"], dtype="f8").reshape(-1, len(names))[:, index]
    q16, q50, q84 = np.percentile(samples, [16.0, 50.0, 84.0])
    return {
        "maximum_likelihood": float(np.asarray(chain["theta_maximum_likelihood"], dtype="f8")[index]),
        "q16": float(q16),
        "q50": float(q50),
        "q84": float(q84),
        "sigma68": float(0.5 * (q84 - q16)),
    }


def interval_text(chain: dict[str, Any], parameter: str) -> str:
    values = summary(chain, parameter)
    return (
        rf"{values['maximum_likelihood']:.2f}; "
        rf"68\%=[{values['q16']:.2f},{values['q84']:.2f}]"
    )


def selected_samples(chain: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    parameters = chain["parameter_names"].tolist()
    indices = [parameters.index(name) for name in names]
    flat = np.asarray(chain["chain_by_step"], dtype="f8").reshape(-1, len(parameters))
    return flat[::20, indices]


def triangle_page(
    pdf: PdfPages,
    variants: tuple[tuple[str, dict[str, Any], str], ...],
    names: tuple[str, ...],
    *,
    title: str,
) -> None:
    axis_labels = {
        "fNL": r"$f_{\rm NL}$",
        "b1": r"$b_1$",
        "sigma_s": r"$\sigma_s\ [h^{-1}{\rm Mpc}]$",
    }
    samples = {key: selected_samples(chain, names) for key, chain, _ in variants}
    ranges = {
        name: plot_range(
            np.concatenate([values[:, index] for values in samples.values()]),
            np.concatenate([values[:, index] for values in samples.values()]),
            parameter=name,
        )
        for index, name in enumerate(names)
    }
    ndim = len(names)
    figure, axes = plt.subplots(ndim, ndim, figsize=(8.7, 8.1) if ndim == 3 else (7.4, 6.5))
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
                        samples[key][:, column],
                        bins=bins,
                        density=True,
                        histtype="step",
                        lw=1.65,
                        ls=LINESTYLES[key],
                        color=COLORS[key],
                    )
                axis.set_xlim(*ranges[xname])
                axis.set_yticks([])
            else:
                for zorder, (key, _, _) in enumerate(variants, start=1):
                    draw_contour(
                        axis,
                        samples[key][:, column],
                        samples[key][:, row],
                        color=COLORS[key],
                        xlim=ranges[xname],
                        ylim=ranges[yname],
                        zorder=2 * zorder,
                    )
                axis.set(xlim=ranges[xname], ylim=ranges[yname])
            if xname == "fNL":
                axis.axvline(0.0, color="0.55", lw=0.8, ls=":", zorder=0)
            if row < ndim - 1:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel(axis_labels[xname])
            if column == 0 and row > 0:
                axis.set_ylabel(axis_labels[yname])
            elif column > 0 and row != column:
                axis.tick_params(labelleft=False)
    handles = [
        plt.Line2D([], [], color=COLORS[key], ls=LINESTYLES[key], lw=2.0)
        for key, _, _ in variants
    ]
    legend_labels = [
        rf"{display}: $f_{{\rm NL}}^{{\rm ML}}$ {interval_text(chain, 'fNL')}"
        for key, chain, display in variants
    ]
    figure.legend(handles, legend_labels, loc="upper right", bbox_to_anchor=(0.975, 0.885), frameon=False, fontsize=7.5)
    figure.suptitle(title, fontsize=12.0, y=0.985)
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.93, wspace=0.08, hspace=0.08)
    pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def forest_page(
    pdf: PdfPages,
    groups: tuple[tuple[str, tuple[tuple[str, dict[str, Any], str], ...]], ...],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 5.8), sharex=False)
    for axis, (title, variants) in zip(axes, groups, strict=True):
        y = np.arange(len(variants), dtype="f8")[::-1]
        all_values = []
        for ypos, (key, chain, label) in zip(y, variants, strict=True):
            values = summary(chain, "fNL")
            all_values.extend([values["q16"], values["q84"], values["maximum_likelihood"]])
            axis.hlines(ypos, values["q16"], values["q84"], color=COLORS[key], lw=2.2)
            axis.plot(values["maximum_likelihood"], ypos, "o", color=COLORS[key], ms=6.0)
            axis.text(
                0.98,
                ypos,
                f"{values['maximum_likelihood']:.2f}  [{values['q16']:.2f}, {values['q84']:.2f}]",
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=7.2,
                color=COLORS[key],
            )
        margin = 0.10 * (max(all_values) - min(all_values))
        axis.set_xlim(min(all_values) - margin, max(all_values) + margin)
        axis.set_yticks(y, [label for _, _, label in variants])
        axis.axvline(0.0, color="0.55", lw=0.9, ls=":")
        axis.set_title(title, fontsize=10.5)
        axis.set_xlabel(r"$f_{\rm NL}$ (point: ML; bar: q16--q84)")
        axis.grid(axis="x", color="0.9", lw=0.6)
    figure.suptitle("Task 4.3 halo lightcone: adding the s=45 Mpc/h bin outside the BAO mask", fontsize=12.5)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def diagnostics_page(pdf: PdfPages, new_audit: dict[str, Any]) -> None:
    changes = new_audit["metrics"]["smin40_vs_smin50"]
    rows = []
    for name in (
        "real_xi0",
        "real_joint_p0xi0",
        "rsd_xi0",
        "rsd_joint_p0xi0",
        "rsd_xi02",
        "rsd_joint_p02xi02",
    ):
        fnl = changes[name]["fNL"]
        result = new_audit["results"][name]
        phase = result["phase_diagnostics"]
        map_fit = result["map"]
        rows.append(
            [
                name,
                f"{fnl['maximum_likelihood_smin40_minus_smin50']:+.2f}",
                f"{fnl['maximum_likelihood_shift_over_smin50_sigma68']:+.2f}",
                f"{fnl['sigma68_smin40_over_smin50']:.3f}",
                f"{map_fit['chi2_observed_x25_mean_with_single_realization_covariance']:.2f}/{map_fit['dof']}",
                f"{map_fit['pte_observed_x25_mean_with_single_realization_covariance']:.3g}",
                f"{phase['chi2_mean']:.1f}",
                f"{phase['fraction_pte_above_0p05']:.2f}",
            ]
        )
    figure, axis = plt.subplots(figsize=(12.8, 5.2))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=(
            "fit",
            r"$\Delta f_{NL}^{ML}$",
            r"$\Delta ML/\sigma_{50}$",
            r"$\sigma_{40}/\sigma_{50}$",
            r"$\chi^2_{mean}/dof$",
            "x25-mean PTE",
            r"mean phase $\chi^2$",
            "phase PTE>0.05",
        ),
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=(0.20, 0.11, 0.12, 0.13, 0.13, 0.10, 0.13, 0.14),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1.0, 1.75)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("0.78")
        if row == 0:
            cell.set_facecolor("0.92")
            cell.set_text_props(weight="bold")
    axis.set_title(
        "smin=40 minus smin=50 diagnostics; all chi-square values use single-realization covariance",
        fontsize=11.5,
        pad=18,
    )
    figure.tight_layout()
    pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def main() -> None:
    output_json = OUTPUT.with_suffix(".json")
    if OUTPUT.exists() or output_json.exists():
        raise FileExistsError(f"immutable output exists: {OUTPUT} / {output_json}")
    old_audit = json.loads(OLD_AUDIT.read_text(encoding="utf-8"))
    new_audit = json.loads(NEW_AUDIT.read_text(encoding="utf-8"))
    for audit, label in ((old_audit, "smin50"), (new_audit, "smin40")):
        if audit.get("status") != "pass" or not all(audit["numerical_gates"].values()):
            raise RuntimeError(f"{label} audit did not pass")
    old = {name: load_chain(OLD_ROOT, name, result) for name, result in old_audit["results"].items()}
    new = {
        name: (old[name] if name in ("real_p0", "rsd_p0", "rsd_p02") else load_chain(NEW_ROOT, name, result))
        for name, result in new_audit["results"].items()
    }

    groups = (
        (
            "Real space: 0.6 < z < 0.8",
            (
                ("p", old["real_p0"], r"$P_0$ (unchanged)"),
                ("xi50", old["real_xi0"], r"$\xi_0$, $s_{min}=50$"),
                ("xi40", new["real_xi0"], r"$\xi_0$, $s_{min}=40$"),
                ("joint50", old["real_joint_p0xi0"], r"joint, $s_{min}=50$"),
                ("joint40", new["real_joint_p0xi0"], r"joint, $s_{min}=40$"),
            ),
        ),
        (
            "RSD monopole: 0.4 < zobs < 0.8",
            (
                ("p", old["rsd_p0"], r"$P_0$ (unchanged)"),
                ("xi50", old["rsd_xi0"], r"$\xi_0$, $s_{min}=50$"),
                ("xi40", new["rsd_xi0"], r"$\xi_0$, $s_{min}=40$"),
                ("joint50", old["rsd_joint_p0xi0"], r"joint, $s_{min}=50$"),
                ("joint40", new["rsd_joint_p0xi0"], r"joint, $s_{min}=40$"),
            ),
        ),
        (
            "RSD multipoles: 0.4 < zobs < 0.8",
            (
                ("p", old["rsd_p02"], r"$P_{0,2}$ (unchanged)"),
                ("xi50", old["rsd_xi02"], r"$\xi_{0,2}$, $s_{min}=50$"),
                ("xi40", new["rsd_xi02"], r"$\xi_{0,2}$, $s_{min}=40$"),
                ("joint50", old["rsd_joint_p02xi02"], r"joint, $s_{min}=50$"),
                ("joint40", new["rsd_joint_p02xi02"], r"joint, $s_{min}=40$"),
            ),
        ),
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9.6,
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
        forest_page(pdf, groups)
        triangle_page(pdf, groups[0][1], ("fNL", "b1"), title=groups[0][0] + ": smin sensitivity")
        triangle_page(pdf, groups[1][1], ("fNL", "b1", "sigma_s"), title=groups[1][0] + ": smin sensitivity")
        triangle_page(pdf, groups[2][1], ("fNL", "b1", "sigma_s"), title=groups[2][0] + ": smin sensitivity")
        diagnostics_page(pdf, new_audit)
    temporary.replace(OUTPUT)
    if OUTPUT.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError("output is not a PDF")

    constraints = {
        group_name: {
            key: {parameter: summary(chain, parameter) for parameter in chain["parameter_names"].tolist()}
            for key, chain, _ in variants
        }
        for group_name, variants in groups
    }
    atomic_write_json(
        output_json,
        {
            "task": "task43_plot_lightcone_smin40_vs50_v1",
            "status": "pass",
            "center_marker": "continuous-optimizer maximum likelihood; never posterior median",
            "interval": "posterior q16 to q84",
            "pages": ["fNL forest", "real contours", "RSD monopole contours", "RSD multipole contours", "diagnostics"],
            "constraints": constraints,
            "comparison_metrics": new_audit["metrics"],
            "covariance_contract": new_audit["covariance_contract"],
            "old_audit": str(OLD_AUDIT),
            "old_audit_sha256": sha256_file(OLD_AUDIT),
            "new_audit": str(NEW_AUDIT),
            "new_audit_sha256": sha256_file(NEW_AUDIT),
            "output_pdf": str(OUTPUT),
            "output_pdf_sha256": sha256_file(OUTPUT),
        },
    )
    print(json.dumps({"status": "pass", "output": str(OUTPUT), "sha256": sha256_file(OUTPUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
