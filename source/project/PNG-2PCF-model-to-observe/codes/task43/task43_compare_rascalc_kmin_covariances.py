#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare Task43 RascalC kmin runs with scatter and jaxpower covariance.

The 25-phase sample covariance is rank deficient for 30 bins, so it is never
inverted here.  It is used only for diagonal scatter and for evaluating each
full-rank analytic/RascalC precision matrix on the phase residuals.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.linalg import eigh

from task43_config import PLOT_DIR, SUMMARY_DIR


DEFAULT_RASCALC_ROOT = SUMMARY_DIR.parent / "rascalc_covariance" / "lightcone_kmin"
DEFAULT_JAXPOWERS = (
    SUMMARY_DIR
    / "jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_pad400_win3600_ds2_kbox0031416_3000_dk002_p1p0_s50_350_ds10.npz",
    SUMMARY_DIR
    / "jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_350_ds10.npz",
)
DEFAULT_SCATTER = SUMMARY_DIR / "task43_mean_xi_mmin1p4e13_x25_s50_350_ds10_fkpP010000.npz"
DEFAULT_OUTPUT = SUMMARY_DIR / "task43_rascalc_kmin_covariance_comparison"
DEFAULT_PLOT = PLOT_DIR / "task43_rascalc_kmin_covariance_comparison.pdf"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def correlation(covariance: np.ndarray) -> np.ndarray:
    cov = np.asarray(covariance, dtype="f8")
    sigma = np.sqrt(np.diag(cov))
    return cov / np.outer(sigma, sigma)


def load_covariance(path: Path, label: str | None = None) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    if "covariance_single_realization" not in data.files:
        raise KeyError(f"{path} has no covariance_single_realization; keys={data.files}")
    cov = np.asarray(data["covariance_single_realization"], dtype="f8")
    s = np.asarray(data["s"], dtype="f8")
    s_edges = np.asarray(data["s_edges"], dtype="f8")
    meta = {}
    if "meta_json" in data.files and str(np.asarray(data["meta_json"]).item()):
        meta = json.loads(str(np.asarray(data["meta_json"]).item()))
    if label is None:
        physics = meta.get("physics", {})
        settings = meta.get("settings", {})
        if physics.get("xi_model"):
            label = f"RascalC {physics['xi_model']} ({settings.get('n_loops', '?')} loops)"
        else:
            label = "jaxpower RR-deconv"
    return {"path": path, "label": str(label), "covariance": cov, "s": s, "s_edges": s_edges, "meta": meta}


def discover_rascalc(root: Path) -> list[Path]:
    """Find successful controlled runs with the converged FullDiscrete row first."""

    def key(path: Path) -> tuple[int, int, str]:
        name = path.parent.name
        if "xifull_discrete" in name:
            model_rank = 0
        elif "xicontinuous_boxcut" in name:
            model_rank = 1
        elif "xicontinuous_lowk" in name:
            model_rank = 2
        else:
            model_rank = 3
        loops = 0
        for token in name.split("_"):
            if token.startswith("nloop") and token[5:].isdigit():
                loops = int(token[5:])
                break
        return model_rank, -loops, name

    return sorted(root.glob("*/task43_rascalc_lightcone_covariance.npz"), key=key)


def jaxpower_label(path: Path) -> str:
    name = path.name
    if "kbox0031416" in name:
        return r"jaxpower RR-deconv ($k_{\min}=2\pi/2000$)"
    if "_k0001_" in name:
        return r"jaxpower RR-deconv ($k_{\min}=10^{-4}$)"
    return "jaxpower RR-deconv"


def covariance_diagnostics(cov: np.ndarray) -> dict[str, Any]:
    cov = 0.5 * (np.asarray(cov, dtype="f8") + np.asarray(cov, dtype="f8").T)
    eig = np.linalg.eigvalsh(cov)
    corr = correlation(cov)
    return {
        "min_eigenvalue": float(eig[0]),
        "max_eigenvalue": float(eig[-1]),
        "condition_number": float(np.linalg.cond(cov)),
        "sigma_min": float(np.min(np.sqrt(np.diag(cov)))),
        "sigma_max": float(np.max(np.sqrt(np.diag(cov)))),
        "corr_offdiag_max_abs": float(np.max(np.abs(corr - np.eye(corr.shape[0])))),
    }


def scatter_metrics(cov: np.ndarray, xi: np.ndarray) -> dict[str, Any]:
    """Evaluate a model precision on residuals about the empirical phase mean."""
    residual = xi - np.mean(xi, axis=0)
    precision = np.linalg.inv(cov)
    chi2 = np.einsum("ij,jk,ik->i", residual, precision, residual)
    sample_std = np.std(xi, axis=0, ddof=1)
    model_std = np.sqrt(np.diag(cov))
    ratio = sample_std / model_std
    return {
        "chi2_mean": float(np.mean(chi2)),
        "chi2_min": float(np.min(chi2)),
        "chi2_max": float(np.max(chi2)),
        "expected_chi2_mean_about_sample_mean": float((xi.shape[0] - 1) / xi.shape[0] * xi.shape[1]),
        "sample_std_over_model_sigma_min": float(np.min(ratio)),
        "sample_std_over_model_sigma_median": float(np.median(ratio)),
        "sample_std_over_model_sigma_max": float(np.max(ratio)),
        "sample_std_over_model_sigma": ratio,
    }


def pair_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compare two SPD matrices without selecting only their diagonals."""
    a = np.asarray(left["covariance"], dtype="f8")
    b = np.asarray(right["covariance"], dtype="f8")
    ratio = np.sqrt(np.diag(b) / np.diag(a))
    corr_delta = correlation(b) - correlation(a)
    generalized = eigh(b, a, eigvals_only=True)
    return {
        "left": left["label"],
        "right": right["label"],
        "sigma_right_over_left_min": float(np.min(ratio)),
        "sigma_right_over_left_median": float(np.median(ratio)),
        "sigma_right_over_left_max": float(np.max(ratio)),
        "correlation_max_abs_difference": float(np.max(np.abs(corr_delta))),
        "correlation_rms_difference": float(np.sqrt(np.mean(corr_delta**2))),
        "covariance_relative_frobenius_difference": float(np.linalg.norm(b - a) / np.linalg.norm(a)),
        "generalized_variance_eigenvalue_min": float(np.min(generalized)),
        "generalized_variance_eigenvalue_median": float(np.median(generalized)),
        "generalized_variance_eigenvalue_max": float(np.max(generalized)),
    }


def run_row(item: dict[str, Any], xi: np.ndarray) -> dict[str, Any]:
    meta = item["meta"]
    settings = meta.get("settings", {})
    physics = meta.get("physics", {})
    quality = meta.get("quality", {})
    return {
        "label": item["label"],
        "path": str(item["path"]),
        "kind": "rascalc" if physics.get("xi_model") else "jaxpower",
        "xi_model": physics.get("xi_model"),
        "nrandom_monte_carlo": meta.get("random_monte_carlo", {}).get("n_used"),
        "n_loops": settings.get("n_loops"),
        "alpha_sn": physics.get("shot_noise_rescaling"),
        "r_inv_max": quality.get("r_inv_max"),
        "r_inv_pass_0p05": quality.get("r_inv_pass"),
        "covariance": covariance_diagnostics(item["covariance"]),
        "scatter": scatter_metrics(item["covariance"], xi),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "label",
        "kind",
        "xi_model",
        "nrandom_monte_carlo",
        "n_loops",
        "alpha_sn",
        "r_inv_max",
        "r_inv_pass_0p05",
        "chi2_mean",
        "expected_chi2_mean",
        "sample_std_over_sigma_median",
        "sigma_min",
        "sigma_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label": row["label"],
                    "kind": row["kind"],
                    "xi_model": row["xi_model"],
                    "nrandom_monte_carlo": row["nrandom_monte_carlo"],
                    "n_loops": row["n_loops"],
                    "alpha_sn": row["alpha_sn"],
                    "r_inv_max": row["r_inv_max"],
                    "r_inv_pass_0p05": row["r_inv_pass_0p05"],
                    "chi2_mean": row["scatter"]["chi2_mean"],
                    "expected_chi2_mean": row["scatter"]["expected_chi2_mean_about_sample_mean"],
                    "sample_std_over_sigma_median": row["scatter"]["sample_std_over_model_sigma_median"],
                    "sigma_min": row["covariance"]["sigma_min"],
                    "sigma_max": row["covariance"]["sigma_max"],
                }
            )


def plot_pdf(path: Path, items: list[dict[str, Any]], xi: np.ndarray) -> None:
    s = items[0]["s"]
    sample_std = np.std(xi, axis=0, ddof=1)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(items)))
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.5), sharex=True, constrained_layout=True)
        axes[0].plot(s, sample_std, color="black", lw=2.2, label="25-phase sample std")
        for color, item in zip(colors, items, strict=True):
            sigma = np.sqrt(np.diag(item["covariance"]))
            axes[0].plot(s, sigma, color=color, lw=1.6, label=item["label"])
            axes[1].plot(s, sample_std / sigma, color=color, lw=1.6, label=item["label"])
        axes[0].set_yscale("log")
        axes[0].set_ylabel(r"$\sigma[\xi_0(s)]$")
        axes[1].axhline(1.0, color="black", ls="--", lw=1.0)
        axes[1].set_xlabel(r"$s\ [h^{-1}{\rm Mpc}]$")
        axes[1].set_ylabel("sample std / model sigma")
        axes[0].legend(fontsize=7, ncol=2)
        axes[0].set_title("Task43 covariance closure: diagonal and phase scatter")
        pdf.savefig(fig)
        plt.close(fig)

        reference = items[0]
        ncols = len(items)
        fig, axes = plt.subplots(2, ncols, figsize=(3.7 * ncols, 7.0), constrained_layout=True, squeeze=False)
        for index, (item, color) in enumerate(zip(items, colors, strict=True)):
            corr = correlation(item["covariance"])
            image0 = axes[0, index].imshow(corr, origin="lower", vmin=-1.0, vmax=1.0, cmap="coolwarm")
            axes[0, index].set_title(item["label"], fontsize=8)
            delta = corr - correlation(reference["covariance"])
            limit = max(float(np.max(np.abs(delta))), 1.0e-6)
            image1 = axes[1, index].imshow(delta, origin="lower", vmin=-limit, vmax=limit, cmap="coolwarm")
            axes[1, index].set_title(f"corr - {reference['label']}", fontsize=8)
            fig.colorbar(image0, ax=axes[0, index], fraction=0.046)
            fig.colorbar(image1, ax=axes[1, index], fraction=0.046)
        fig.suptitle("Task43 covariance correlation matrices and differences")
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rascalc", type=Path, nargs="*", default=None)
    parser.add_argument(
        "--jaxpower",
        type=Path,
        nargs="*",
        default=None,
        help="One or more jaxpower covariance products; defaults to box-kmin and low-k products.",
    )
    parser.add_argument("--scatter", type=Path, default=DEFAULT_SCATTER)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT)
    args = parser.parse_args()

    rascalc_paths = discover_rascalc(DEFAULT_RASCALC_ROOT) if args.rascalc is None else list(args.rascalc)
    if not rascalc_paths:
        raise FileNotFoundError(f"no successful RascalC products under {DEFAULT_RASCALC_ROOT}")
    jaxpower_paths = list(DEFAULT_JAXPOWERS) if args.jaxpower is None else list(args.jaxpower)
    if not jaxpower_paths:
        raise ValueError("--jaxpower was supplied without any paths")
    for path in rascalc_paths + jaxpower_paths + [args.scatter]:
        if not path.exists():
            raise FileNotFoundError(path)

    scatter_data = np.load(args.scatter, allow_pickle=False)
    xi = np.asarray(scatter_data["xi0_all"], dtype="f8")
    items = [load_covariance(path) for path in rascalc_paths]
    items.extend(load_covariance(path, label=jaxpower_label(path)) for path in jaxpower_paths)
    reference_edges = items[0]["s_edges"]
    if any(item["covariance"].shape != items[0]["covariance"].shape for item in items):
        raise ValueError("covariance shapes differ")
    if any(not np.allclose(item["s_edges"], reference_edges) for item in items):
        raise ValueError("covariance separation bins differ")
    if xi.shape[1] != items[0]["covariance"].shape[0]:
        raise ValueError(f"scatter shape {xi.shape} does not match covariance")

    rows = [run_row(item, xi) for item in items]
    pairs = [pair_metrics(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]
    payload = {
        "status": "done",
        "task": "task43_compare_rascalc_kmin_covariances",
        "scatter_path": str(args.scatter),
        "nphase": int(xi.shape[0]),
        "nbins": int(xi.shape[1]),
        "warning": "The 25x30 sample covariance is rank deficient and is not inverted; alpha_SN=1 RascalC rows remain diagnostic until calibrated.",
        "rows": rows,
        "pairwise": pairs,
    }
    out_json = args.output_prefix.with_suffix(".json")
    out_csv = args.output_prefix.with_suffix(".csv")
    out_pdf = args.plot_path
    out_npz = args.output_prefix.with_suffix(".npz")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(out_csv, rows)
    plot_pdf(out_pdf, items, xi)
    np.savez_compressed(
        out_npz,
        s=items[0]["s"],
        s_edges=reference_edges,
        labels=np.asarray([item["label"] for item in items]),
        covariances=np.stack([item["covariance"] for item in items]),
        sample_covariance=np.cov(xi, rowvar=False, ddof=1),
        sample_std=np.std(xi, axis=0, ddof=1),
        meta_json=np.asarray(json.dumps(_jsonable(payload), sort_keys=True)),
    )
    print(f"[done] wrote {out_json}")
    for row in rows:
        print(
            f"[row] {row['label']}: chi2={row['scatter']['chi2_mean']:.3f} "
            f"ratio_med={row['scatter']['sample_std_over_model_sigma_median']:.3f} "
            f"R_inv={row['r_inv_max']}"
        )


if __name__ == "__main__":
    main()
