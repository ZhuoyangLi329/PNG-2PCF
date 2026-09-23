#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the fitted Task43 P(k) JAXPower covariance as a correlation matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
DEFAULT_PAYLOAD = (
    PROJECT_ROOT
    / "outputs/task43_outputs/pk_lightcone/summary"
    / "task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_payload.npz"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "plots/task43/task43_pk_jaxpower_covariance_correlation_matrix.pdf"
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "outputs/task43_outputs/summary/task43_pk_jaxpower_covariance_correlation_matrix.json"
)


def correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype="f8")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"covariance must be square, got {covariance.shape}")
    if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-8):
        raise ValueError("covariance is not symmetric")
    diagonal = np.diag(covariance)
    if np.any(diagonal <= 0.0):
        raise ValueError("covariance has a non-positive diagonal entry")
    sigma = np.sqrt(diagonal)
    correlation = covariance / np.outer(sigma, sigma)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    if args.output.suffix.lower() != ".pdf":
        raise ValueError("Task43 plot output must be a PDF")
    if not args.payload.exists():
        raise FileNotFoundError(args.payload)

    with np.load(args.payload, allow_pickle=False) as payload:
        k_obs = np.asarray(payload["k_obs"], dtype="f8")
        k_edges = np.asarray(payload["k_edges"], dtype="f8")
        covariance = np.asarray(payload["covariance"], dtype="f8")
        covariance_file = str(np.asarray(payload["covariance_file"]).item())
        input_summary = json.loads(str(np.asarray(payload["summary_json"]).item()))

    if covariance.shape != (k_obs.size, k_obs.size):
        raise ValueError(
            f"covariance shape {covariance.shape} does not match {k_obs.size} fitted bandpowers"
        )
    if k_edges.shape != (k_obs.size, 2):
        raise ValueError(f"expected k_edges shape {(k_obs.size, 2)}, got {k_edges.shape}")

    correlation = correlation_from_covariance(covariance)
    off_diagonal = correlation.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    maximum_index = np.unravel_index(np.argmax(np.abs(off_diagonal)), off_diagonal.shape)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    boundaries = np.arange(k_obs.size + 1, dtype="f8") - 0.5
    image = axis.pcolormesh(
        boundaries,
        boundaries,
        correlation,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        shading="flat",
        edgecolors=(1.0, 1.0, 1.0, 0.28),
        linewidth=0.35,
    )
    ticks = np.arange(k_obs.size)
    labels = [f"{value:.3f}" for value in k_obs]
    axis.set_xticks(ticks, labels=labels, rotation=55, ha="right", rotation_mode="anchor")
    axis.set_yticks(ticks, labels=labels)
    axis.set_xlim(-0.5, k_obs.size - 0.5)
    axis.set_ylim(-0.5, k_obs.size - 0.5)
    axis.set_aspect("equal")
    axis.set_xlabel(r"$k_j\ [h\,\mathrm{Mpc}^{-1}]$")
    axis.set_ylabel(r"$k_i\ [h\,\mathrm{Mpc}^{-1}]$")
    figure.suptitle(r"Task 4.3 $P_0(k)$ JAXPower covariance correlation", fontsize=12)
    axis.set_title(
        "Single-lightcone Gaussian survey-window covariance; 15 fitted bandpowers",
        fontsize=9,
        pad=8,
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.025, fraction=0.047)
    colorbar.set_label(r"$R_{ij}=C_{ij}/\sqrt{C_{ii}C_{jj}}$")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.close(figure)

    summary = {
        "task": "task43_pk_jaxpower_covariance_correlation_matrix",
        "status": "done",
        "input_payload": str(args.payload.resolve()),
        "input_covariance_file": covariance_file,
        "covariance_source": input_summary.get("covariance", {}).get("source"),
        "covariance_not_divided_by_nphase": input_summary.get("covariance", {}).get(
            "not_divided_by_nphase"
        ),
        "output_pdf": str(args.output.resolve()),
        "matrix_shape": list(covariance.shape),
        "number_of_fit_bandpowers": int(k_obs.size),
        "k_obs_h_mpc": k_obs,
        "k_edges_h_mpc": k_edges,
        "correlation_min": float(np.min(correlation)),
        "correlation_max": float(np.max(correlation)),
        "maximum_absolute_offdiagonal_correlation": float(
            abs(off_diagonal[maximum_index])
        ),
        "maximum_absolute_offdiagonal_pair": {
            "indices": [int(maximum_index[0]), int(maximum_index[1])],
            "k_i_h_mpc": float(k_obs[maximum_index[0]]),
            "k_j_h_mpc": float(k_obs[maximum_index[1]]),
            "correlation": float(correlation[maximum_index]),
        },
        "covariance_minimum_eigenvalue": minimum_eigenvalue,
        "covariance_condition_number": float(np.linalg.cond(covariance)),
        "formula": "R_ij = C_ij / sqrt(C_ii C_jj)",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
