#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xi-only posterior and 9.22meeting contour for the hybrid PNG GSM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_DIR = PROJECT_ROOT / "codes" / "task43"
TASK432_DIR = PROJECT_ROOT / "codes" / "task432"
OUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "task432_model_repair" / "png_velocileptors_gsm"
PLOT_ROOT = PROJECT_ROOT / "plots" / "task43" / "rsd_validation" / "task432_model_repair"
MEETING_ROOT = PROJECT_ROOT / "9.22meeting" / "task432_2pcf_rsd_comparison"

for path in (TASK43_DIR, TASK432_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from task43_rsd_common import rawbox_xi_primary_mask  # noqa: E402
from task432_2pcf_rsd_contours import (  # noqa: E402
    COLORS,
    corner_page,
    load_original,
    overlay_page,
    set_style,
)
from task432_linear_gsm_rawbox import build_models_and_covariance, precision_from_covariance  # noqa: E402
from task432_png_velocileptors_gsm import PNGVelocileptorsGSM  # noqa: E402


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def posterior_summary(samples: dict[str, np.ndarray]) -> dict:
    return {
        name: {
            "q16": float(np.percentile(value, 16.0)),
            "q50": float(np.percentile(value, 50.0)),
            "q84": float(np.percentile(value, 84.0)),
            "sigma68": float(0.5 * (np.percentile(value, 84.0) - np.percentile(value, 16.0))),
        }
        for name, value in samples.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwalkers", type=int, default=16)
    parser.add_argument("--nsteps", type=int, default=900)
    parser.add_argument("--burnin", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()

    products = build_models_and_covariance(xi_angle_mode="continuous", k_switch=0.01)
    cache_path = Path(products["metadata"]["cache"])
    with np.load(cache_path, allow_pickle=False) as payload:
        k = np.asarray(payload["k_eff"], dtype="f8")
        pk = np.asarray(payload["pk_dd"], dtype="f8")
        alpha = np.asarray(payload["alpha"], dtype="f8")
        f_growth = float(np.asarray(payload["f_growth"]).item())
        s_edges = np.asarray(payload["s_edges"], dtype="f8")
    mask = rawbox_xi_primary_mask(0.5 * (s_edges[:-1] + s_edges[1:]))
    data = np.asarray(products["data"]["x"], dtype="f8")
    precision = precision_from_covariance(products["covariances"]["x"])
    model = PNGVelocileptorsGSM(k, pk, alpha, f_growth, s_edges)

    def log_prob(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype="f8")
        if theta[0] <= -500.0 or theta[0] >= 500.0 or theta[1] <= 0.5 or theta[1] >= 5.0:
            return -np.inf
        values = model.evaluate(fnl=float(theta[0]), b1_eulerian=float(theta[1]), nint=600)
        vector = np.concatenate([values[0][mask], values[2][mask]])
        residual = data - vector
        return -0.5 * float(residual @ precision @ residual)

    rng = np.random.default_rng(int(args.seed))
    start = np.asarray([4.8, 2.616], dtype="f8")
    spread = np.asarray([4.0, 0.03], dtype="f8")
    initial = start[None, :] + rng.normal(size=(int(args.nwalkers), 2)) * spread[None, :]
    initial[:, 0] = np.clip(initial[:, 0], -499.0, 499.0)
    initial[:, 1] = np.clip(initial[:, 1], 0.501, 4.999)
    np.random.seed(int(args.seed))
    sampler = emcee.EnsembleSampler(int(args.nwalkers), 2, log_prob)
    sampler.run_mcmc(initial, int(args.nsteps), progress=False, skip_initial_state_check=True)
    chain = sampler.get_chain(discard=int(args.burnin), flat=True)
    logp = sampler.get_log_prob(discard=int(args.burnin), flat=True)
    samples = {"fNL": chain[:, 0], "b1": chain[:, 1], "xi_width": np.zeros(chain.shape[0], dtype="f8")}
    summary = {
        "parameter_names": ["fNL", "b1"],
        "posterior": posterior_summary({"fNL": samples["fNL"], "b1": samples["b1"]}),
        "mcmc": {"nwalkers": int(args.nwalkers), "nsteps": int(args.nsteps), "burnin": int(args.burnin), "seed": int(args.seed)},
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
        "model": "velocileptors GSM plus linear PNG response; EFT/b2/bs/b3/s2fog fixed zero",
    }
    try:
        tau = np.asarray(emcee.autocorr.integrated_time(sampler.get_chain(discard=int(args.burnin)), tol=0), dtype="f8")
        summary["postburn_length_over_tau"] = (sampler.get_chain(discard=int(args.burnin)).shape[0] / tau).tolist()
    except Exception as exc:
        summary["tau_error"] = str(exc)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    chain_path = OUT_ROOT / "task432_png_velocileptors_gsm_xi_chain.npz"
    np.savez_compressed(chain_path, samples=chain, log_prob=logp, parameter_names=np.asarray(["fNL", "b1"]))
    original = load_original(OUT_ROOT.parent / "rawbox_baomask_split_sigma_longchain" / "fits" / "xi_marginal" / "samples.npz")
    gsm_for_plot = {"fNL": samples["fNL"], "b1": samples["b1"], "xi_width": np.zeros_like(samples["fNL"]) + np.nan}

    # The two-parameter hybrid does not have an independent fitted sigma_FOG;
    # create a separate corner page manually and use only fNL/b1 overlay.
    set_style()
    MEETING_ROOT.mkdir(parents=True, exist_ok=True)
    pdf_path = MEETING_ROOT / "task432_2pcf_original_vs_png_velocileptors_gsm_9.22style.pdf"
    temp_pdf = pdf_path.with_name(f".{pdf_path.stem}.{os.getpid()}.tmp.pdf")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(temp_pdf) as pdf:
        corner_page(pdf, original, color=COLORS["original"], title="Task43 rawbox xi RSD: original FullDiscrete", xi_label=r"$\sigma_{s,\xi}\ [h^{-1}{\rm Mpc}]$")
        # Hybrid PNG GSM has no fitted xi-side width; show a 2-parameter page.
        figure, axes = plt.subplots(2, 2, figsize=(6.3, 5.8), squeeze=False)
        axes[0, 0].hist(samples["fNL"], bins=90, density=True, histtype="step", lw=1.8, color=COLORS["gsm"])
        axes[1, 0].hist(samples["fNL"], bins=90, density=True, histtype="step", lw=1.8, color=COLORS["gsm"])
        axes[1, 1].hist(samples["b1"], bins=90, density=True, histtype="step", lw=1.8, color=COLORS["gsm"])
        axes[0, 1].set_axis_off()
        axes[1, 0].set_xlabel(r"$f_{\rm NL}$")
        axes[1, 1].set_xlabel(r"$b_1$")
        axes[1, 0].set_ylabel(r"$b_1$")
        axes[1, 0].set_xlim(np.quantile(samples["fNL"], [0.001, 0.999]))
        axes[1, 1].set_xlim(np.quantile(samples["b1"], [0.001, 0.999]))
        # Add the shared fNL-b1 contours by reusing the production helper.
        from task432_2pcf_rsd_contours import draw_contour
        xlim = (min(np.percentile(original["fNL"], 0.1), np.percentile(samples["fNL"], 0.1)), max(np.percentile(original["fNL"], 99.9), np.percentile(samples["fNL"], 99.9)))
        ylim = (min(np.percentile(original["b1"], 0.1), np.percentile(samples["b1"], 0.1)), max(np.percentile(original["b1"], 99.9), np.percentile(samples["b1"], 99.9)))
        draw_contour(axes[1, 0], original["fNL"], original["b1"], color=COLORS["original"], xlim=xlim, ylim=ylim, zorder=2)
        draw_contour(axes[1, 0], samples["fNL"], samples["b1"], color=COLORS["gsm"], xlim=xlim, ylim=ylim, zorder=4)
        axes[1, 0].set_xlim(*xlim); axes[1, 0].set_ylim(*ylim)
        axes[1, 1].set_axis_off()
        figure.suptitle("Task43 rawbox xi RSD: PNG velocileptors GSM", fontsize=14.0, y=0.98)
        figure.text(0.5, 0.012, "xi0+xi2 only; PNG hybrid has shared fNL/b1, no fitted sigma_FOG", ha="center", fontsize=8.0, color="0.4")
        figure.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.86)
        pdf.savefig(figure)
        plt.close(figure)

    temp_pdf.replace(pdf_path)
    audit = {
        "task": "Task432 xi-only PNG velocileptors GSM posterior",
        "status": "complete",
        "posterior": summary,
        "chain": str(chain_path),
        "pdf": str(pdf_path),
        "scope": "xi0+xi2 only; no P02 and no joint P+xi posterior",
    }
    atomic_json(OUT_ROOT / "task432_png_velocileptors_gsm_xi_summary.json", audit)
    print(json.dumps({"status": "complete", "posterior": summary, "pdf": str(pdf_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
