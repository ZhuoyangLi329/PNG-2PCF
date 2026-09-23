#!/usr/bin/env python3
"""Plot paired c302-c000 observable responses and mean-fit posteriors."""

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

from task43_pngbase_pseudolc_common import FINAL_ROOT, PHASES, fit_root
from task43_rsd_common import atomic_write_json, sha256_file


VARIANTS = (
    "real_p0",
    "real_xi0",
    "real_joint_p0xi0",
    "rsd_p0",
    "rsd_xi0",
    "rsd_joint_p0xi0",
    "rsd_p02",
    "rsd_xi02",
    "rsd_joint_p02xi02",
)
JOINT_VARIANTS = ("real_joint_p0xi0", "rsd_joint_p0xi0", "rsd_joint_p02xi02")
COLORS = {"ph000": "#4C72B0", "ph001": "#C44E52", "mean": "#252525"}


def fit_paths(cosmology: str, variant: str) -> tuple[Path, Path]:
    root = fit_root(cosmology) / "fits" / variant
    return root / "samples.npz", root / "summary.json"


def load_fit(cosmology: str, variant: str) -> dict[str, Any]:
    npz_path, json_path = fit_paths(cosmology, variant)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "pass" or metadata.get("output_npz_sha256") != sha256_file(npz_path):
        raise RuntimeError(f"unvalidated fit: {cosmology} {variant}")
    if metadata.get("single_phase_fnl_fit") is not False:
        raise RuntimeError(f"single-phase fit contract changed: {json_path}")
    with np.load(npz_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    arrays.update({"metadata": metadata, "npz_path": npz_path, "json_path": json_path})
    return arrays


def normalized_responses(c000: dict[str, Any], c302: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(c000["phase_data"], dtype="f8")
    right = np.asarray(c302["phase_data"], dtype="f8")
    covariance = np.asarray(c000["covariance_single"], dtype="f8")
    if left.shape != right.shape or left.shape[0] != 2 or covariance.shape != (left.shape[1], left.shape[1]):
        raise RuntimeError("paired response shapes changed")
    sigma = np.sqrt(np.diag(covariance))
    paired = (right - left) / sigma[None, :]
    return paired, np.mean(paired, axis=0)


def response_panel(
    axis: plt.Axes,
    x: np.ndarray,
    paired: np.ndarray,
    mean: np.ndarray,
    title: str,
    *,
    is_xi: bool,
) -> None:
    for index, phase in enumerate(PHASES):
        axis.plot(x, paired[index], "o-", ms=3.0, lw=0.9, color=COLORS[phase], label=phase)
    axis.plot(x, mean, "s-", ms=3.2, lw=1.5, color=COLORS["mean"], label="paired mean")
    axis.axhline(0.0, color="0.6", lw=0.8)
    if is_xi:
        axis.axvspan(80.0, 120.0, color="0.92", zorder=-5)
        axis.set_xlabel(r"$s\ [h^{-1}{\rm Mpc}]$")
    else:
        axis.set_xscale("log")
        axis.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    axis.set_title(title)
    axis.set_ylabel(r"$(O_{100}-O_0)/\sigma_{\rm single}$")
    axis.legend(frameon=False, fontsize=8)


def response_snr(c000: dict[str, Any], c302: dict[str, Any]) -> dict[str, Any]:
    left = np.asarray(c000["phase_data"], dtype="f8")
    right = np.asarray(c302["phase_data"], dtype="f8")
    covariance = np.asarray(c000["covariance_single"], dtype="f8")
    precision = np.linalg.inv(covariance)
    differences = right - left
    mean = np.mean(differences, axis=0)
    return {
        "definition": "sqrt(delta^T C_single^-1 delta), with delta=c302-c000 matched by initial-condition seed",
        "by_phase": {
            phase: float(np.sqrt(max(differences[index] @ precision @ differences[index], 0.0)))
            for index, phase in enumerate(PHASES)
        },
        "paired_mean": float(np.sqrt(max(mean @ precision @ mean, 0.0))),
    }


def posterior_summary(fit: dict[str, Any]) -> dict[str, float]:
    names = fit["parameter_names"].tolist()
    index = names.index("fNL")
    values = np.asarray(fit["chain_by_step"], dtype="f8").reshape(-1, len(names))[:, index]
    q16, q50, q84 = np.percentile(values, [16.0, 50.0, 84.0])
    return {"q16": float(q16), "q50": float(q50), "q84": float(q84), "sigma68": float(0.5 * (q84 - q16))}


def main() -> None:
    output = FINAL_ROOT / "task43_pngbase_pseudolc_paired_fnl_response_kmax0p08_smin50_baomask80_120.pdf"
    output_json = output.with_suffix(".json")
    if output.exists() or output_json.exists():
        raise FileExistsError(f"immutable paired-response output exists: {output} / {output_json}")
    fits = {
        cosmology: {variant: load_fit(cosmology, variant) for variant in VARIANTS}
        for cosmology in ("c000", "c302")
    }
    covariance_path = fit_root("c000") / "task43_pngbase_pseudolc_joint_baomask80_120_covariance.npz"
    with np.load(covariance_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    real_s = arrays["real_s"][arrays["real_xi_mask"].astype(bool)]
    rsd_s = arrays["rsd_s"][arrays["rsd_xi_mask"].astype(bool)]
    rsd_k0 = arrays["rsd_k"]
    rsd_k2 = rsd_k0[arrays["rsd_p2_keep_indices"].astype(int)]
    n_rsd_p0, n_rsd_x = rsd_k0.size, rsd_s.size

    response = {
        variant: normalized_responses(fits["c000"][variant], fits["c302"][variant])
        for variant in ("real_p0", "real_xi0", "rsd_p02", "rsd_xi02")
    }
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
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
        figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
        response_panel(axes[0], arrays["real_k"], *response["real_p0"], r"real-space $P_0$", is_xi=False)
        response_panel(axes[1], real_s, *response["real_xi0"], r"real-space $\xi_0$", is_xi=True)
        figure.suptitle(r"Paired pngbase response: $f_{\rm NL}=100-0$, real-space pseudo-lightcone")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
        plt.close(figure)

        p_pair, p_mean = response["rsd_p02"]
        x_pair, x_mean = response["rsd_xi02"]
        figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
        response_panel(axes[0, 0], rsd_k0, p_pair[:, :n_rsd_p0], p_mean[:n_rsd_p0], r"RSD $P_0$", is_xi=False)
        response_panel(axes[0, 1], rsd_k2, p_pair[:, n_rsd_p0:], p_mean[n_rsd_p0:], r"RSD $P_2$", is_xi=False)
        response_panel(axes[1, 0], rsd_s, x_pair[:, :n_rsd_x], x_mean[:n_rsd_x], r"RSD $\xi_0$", is_xi=True)
        response_panel(axes[1, 1], rsd_s, x_pair[:, n_rsd_x:], x_mean[n_rsd_x:], r"RSD $\xi_2$", is_xi=True)
        figure.suptitle(r"Paired pngbase response: $f_{\rm NL}=100-0$, RSD pseudo-lightcone")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
        plt.close(figure)

        labels = {
            "real_joint_p0xi0": r"real $P_0+\xi_0$",
            "rsd_joint_p0xi0": r"RSD $P_0+\xi_0$",
            "rsd_joint_p02xi02": r"RSD $P_{0,2}+\xi_{0,2}$",
        }
        figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.7), sharey=True)
        for axis, variant in zip(axes, JOINT_VARIANTS, strict=True):
            for cosmology, truth, color in (("c000", 0.0, "#4C72B0"), ("c302", 100.0, "#C44E52")):
                fit = fits[cosmology][variant]
                names = fit["parameter_names"].tolist()
                samples = np.asarray(fit["chain_by_step"], dtype="f8").reshape(-1, len(names))[:, names.index("fNL")]
                axis.hist(samples[::8], bins=70, density=True, histtype="step", lw=1.6, color=color, label=cosmology)
                axis.axvline(truth, color=color, lw=0.9, ls="--")
            axis.set_title(labels[variant])
            axis.set_xlabel(r"$f_{\rm NL}$")
            axis.legend(frameon=False)
        axes[0].set_ylabel("posterior density")
        figure.suptitle("Only the two-realization mean is fit; dashed lines are injected values")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
        plt.close(figure)
    temporary.replace(output)
    if output.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError("paired response output is not a PDF")

    audit_paths = {
        cosmology: fit_root(cosmology) / "task43_pngbase_pseudolc_joint_baomask80_120.json"
        for cosmology in ("c000", "c302")
    }
    summary = {
        "task": "task43_plot_pngbase_paired_response",
        "status": "pass",
        "definition": "paired differences are c302-c000 at equal initial-condition seed",
        "phases": list(PHASES),
        "single_phase_fnl_fits": [],
        "mean_fit_policy": "only ph000/ph001 arithmetic observable means are fit",
        "covariance_policy": "all response normalizations and fits use C_single, never C_single/2",
        "response_snr_csingle": {
            variant: response_snr(fits["c000"][variant], fits["c302"][variant]) for variant in VARIANTS
        },
        "posterior_fnl": {
            cosmology: {variant: posterior_summary(fits[cosmology][variant]) for variant in VARIANTS}
            for cosmology in ("c000", "c302")
        },
        "fit_audits": {
            cosmology: {"path": str(path), "sha256": sha256_file(path)} for cosmology, path in audit_paths.items()
        },
        "output_pdf": str(output),
        "output_pdf_sha256": sha256_file(output),
        "pages": ["real paired response", "RSD paired multipole response", "mean-fit fNL posteriors"],
    }
    atomic_write_json(output_json, summary)
    print(json.dumps({"status": "pass", "output": str(output), "sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()

