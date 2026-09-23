#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""9.22meeting-style 2PCF-only RSD contour comparison.

本脚本只比较 xi0+xi2 的 marginal posterior，不使用 P02，也不使用 joint
P+xi posterior 作为主结果。

页面：
1. original FullDiscrete xi02 split-sigma：fNL、b1、sigma_s_xi；
2. linear GSM xi02：fNL、b1、sigma_FOG；
3. 只对定义相同的 fNL、b1 做 overlay。

sigma_s_xi 与 sigma_FOG 不在同一张 overlay contour 中出现，避免把不同
物理定义误认为同一个参数。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import emcee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import numpy as np
from matplotlib.colors import to_rgba
from scipy.ndimage import gaussian_filter

from task43_plot_rsd_rawbox_pk0_vs_xi0_contours import density_levels
from task43_rsd_common import atomic_write_json, sha256_file, rawbox_xi_primary_mask
from task432_linear_gsm_rawbox import (
    LinearGSMModel,
    LinearRadialMomentProvider,
    build_models_and_covariance,
    precision_from_covariance,
)


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "task432_model_repair"
OUTDIR = PROJECT_ROOT / "9.22meeting" / "task432_2pcf_rsd_comparison"
ORIGINAL = OUTPUT_ROOT / "rawbox_baomask_split_sigma_longchain" / "fits" / "xi_marginal" / "samples.npz"

PARAMETERS = ("fNL", "b1", "xi_width")
COLORS = {"original": "#2F2F2F", "gsm": "#4C72B0"}


def set_style() -> None:
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


def jsonable(value: Any) -> Any:
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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_original(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        samples = np.asarray(payload["samples"], dtype="f8")
        names = [str(value) for value in np.asarray(payload["parameter_names"]).ravel()]
    index = {name: i for i, name in enumerate(names)}
    required = ["fNL", "b1", "sigma_s_xi"]
    if any(name not in index for name in required):
        raise ValueError(f"original xi chain lacks {required}: {names}")
    return {
        "fNL": samples[:, index["fNL"]],
        "b1": samples[:, index["b1"]],
        "xi_width": samples[:, index["sigma_s_xi"]],
    }


def posterior_summary(samples: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "q16": float(np.percentile(values, 16.0)),
            "q50": float(np.percentile(values, 50.0)),
            "q84": float(np.percentile(values, 84.0)),
            "sigma68": float(0.5 * (np.percentile(values, 84.0) - np.percentile(values, 16.0))),
        }
        for name, values in samples.items()
    }


def posterior_text(samples: np.ndarray) -> str:
    q16, q50, q84 = np.percentile(np.asarray(samples, dtype="f8"), [16.0, 50.0, 84.0])
    return rf"{q50:.2f}_{{-{q50-q16:.2f}}}^{{+{q84-q50:.2f}}}"


def run_gsm_xi_mcmc(
    *,
    products: dict[str, Any],
    model: LinearGSMModel,
    mask: np.ndarray,
    nwalkers: int = 16,
    nsteps: int = 900,
    burnin: int = 200,
    seed: int = 20260924,
    checkpoint_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Sample only the xi0+xi2 likelihood under the frozen xi covariance."""

    data = np.asarray(products["data"]["x"], dtype="f8")
    precision = precision_from_covariance(products["covariances"]["x"])
    bounds_lo = np.asarray([-500.0, 0.5, 0.0], dtype="f8")
    bounds_hi = np.asarray([500.0, 5.0, 30.0], dtype="f8")

    def log_probability(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype="f8")
        if np.any(theta <= bounds_lo) or np.any(theta >= bounds_hi):
            return -np.inf
        fnl, b1, sigma_fog = theta
        prediction = model.vector(fnl=float(fnl), b1=float(b1), sigma_fog=float(sigma_fog), mask=mask)
        residual = data - prediction
        return -0.5 * float(residual @ precision @ residual)

    rng = np.random.default_rng(int(seed))
    start = np.asarray([0.0, 2.4, 15.0], dtype="f8")
    spread = np.asarray([5.0, 0.04, 1.0], dtype="f8")
    initial = start[None, :] + rng.normal(size=(int(nwalkers), 3)) * spread[None, :]
    initial = np.maximum(initial, bounds_lo[None, :] + 1.0e-7)
    initial = np.minimum(initial, bounds_hi[None, :] - 1.0e-7)
    np.random.seed(int(seed))
    sampler = emcee.EnsembleSampler(int(nwalkers), 3, log_probability)
    state = initial
    completed = 0
    checkpoint_every = 50
    while completed < int(nsteps):
        chunk = min(checkpoint_every, int(nsteps) - completed)
        state = sampler.run_mcmc(state, chunk, progress=False, skip_initial_state_check=True)
        completed += chunk
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{os.getpid()}.tmp")
            np.savez_compressed(
                temporary,
                chain=sampler.get_chain(),
                log_prob=sampler.get_log_prob(),
                completed_steps=np.asarray(completed, dtype="i8"),
            )
            generated = Path(str(temporary) + ".npz") if not temporary.name.endswith(".npz") else temporary
            generated.replace(checkpoint_path)

    flat = sampler.get_chain(discard=int(burnin), flat=True)
    samples = {name: flat[:, index] for index, name in enumerate(("fNL", "b1", "xi_width"))}
    summary = {
        "parameter_names": ["fNL", "b1", "xi_width"],
        "posterior": posterior_summary(samples),
        "mcmc": {"nwalkers": int(nwalkers), "nsteps": int(nsteps), "burnin": int(burnin), "seed": int(seed)},
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
    }
    try:
        tau = np.asarray(emcee.autocorr.integrated_time(sampler.get_chain(discard=int(burnin)), tol=0), dtype="f8")
        summary["postburn_length_over_tau"] = (sampler.get_chain(discard=int(burnin)).shape[0] / tau).tolist()
    except Exception as exc:
        summary["tau_error"] = str(exc)
    return samples, summary


def ranges_for(samples_list: list[dict[str, np.ndarray]]) -> dict[str, tuple[float, float]]:
    result = {}
    for name in PARAMETERS:
        values = np.concatenate([samples[name] for samples in samples_list])
        lo, hi = np.quantile(values, [0.001, 0.999])
        if name == "fNL":
            lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
        if name == "xi_width":
            lo = min(0.0, float(lo))
        width = float(hi - lo)
        result[name] = (float(lo - (0.0 if name == "xi_width" else 0.06 * width)), float(hi + 0.06 * width))
    return result


def draw_contour(axis: plt.Axes, x: np.ndarray, y: np.ndarray, *, color: str, xlim: tuple[float, float], ylim: tuple[float, float], zorder: int) -> dict[str, float]:
    hist, xedges, yedges = np.histogram2d(x, y, bins=(170, 160), range=(xlim, ylim), density=False)
    hist = gaussian_filter(hist.astype("f8"), sigma=2.5, mode="nearest")
    flat = np.sort(hist.ravel())[::-1]
    cumulative = np.cumsum(flat) / np.sum(flat)
    level68 = float(flat[min(int(np.searchsorted(cumulative, 0.68)), flat.size - 1)])
    level95 = float(flat[min(int(np.searchsorted(cumulative, 0.95)), flat.size - 1)])
    top = float(np.max(hist)) * 1.001
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    axis.contourf(xc, yc, hist.T, levels=[level95, level68, top], colors=[to_rgba(color, 0.10), to_rgba(color, 0.22)], zorder=zorder)
    axis.contour(xc, yc, hist.T, levels=[level95, level68], colors=color, linewidths=[1.2, 1.8], zorder=zorder + 1)
    return {"level95": level95, "level68": level68}


def corner_page(pdf: PdfPages, samples: dict[str, np.ndarray], *, color: str, title: str, xi_label: str) -> dict[str, Any]:
    labels = {"fNL": r"$f_{\rm NL}$", "b1": r"$b_1$", "xi_width": xi_label}
    ranges = ranges_for([samples])
    figure, axes = plt.subplots(3, 3, figsize=(8.8, 8.2), squeeze=False)
    audit: dict[str, Any] = {}
    for irow, yname in enumerate(PARAMETERS):
        for icol, xname in enumerate(PARAMETERS):
            axis = axes[irow, icol]
            if icol > irow:
                axis.set_axis_off()
                continue
            axis.set_xlim(*ranges[xname])
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.tick_params(labelsize=9)
            if irow == icol:
                axis.hist(samples[xname], bins=np.linspace(*ranges[xname], 90), density=True, histtype="step", lw=1.8, color=color)
                axis.set_yticks([])
                if xname == "fNL":
                    axis.axvline(0.0, color="0.55", lw=0.8, ls="--")
            else:
                axis.set_ylim(*ranges[yname])
                axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
                audit[f"{xname}_vs_{yname}"] = draw_contour(axis, samples[xname], samples[yname], color=color, xlim=ranges[xname], ylim=ranges[yname], zorder=2)
            if irow < 2:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel(labels[xname], fontsize=13)
            if icol == 0 and irow > 0:
                axis.set_ylabel(labels[yname], fontsize=13)
            elif icol > 0 and irow != icol:
                axis.tick_params(labelleft=False)
    figure.suptitle(title, fontsize=14.0, y=0.98)
    figure.text(0.5, 0.945, r"$50\leq s<350\ h^{-1}{\rm Mpc}$; BAO mask 80--120; xi0+xi2 only", ha="center", fontsize=10.5)
    figure.text(0.5, 0.012, "68% / 95% contours from raw chains, following the 9.18/9.22meeting convention.", ha="center", fontsize=8.0, color="0.4")
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.085, top=0.90, wspace=0.08, hspace=0.08)
    pdf.savefig(figure)
    plt.close(figure)
    return {"ranges": {k: list(v) for k, v in ranges.items()}, "contours": audit}


def overlay_page(pdf: PdfPages, original: dict[str, np.ndarray], gsm: dict[str, np.ndarray]) -> dict[str, Any]:
    parameters = ("fNL", "b1")
    ranges = ranges_for([original, gsm])
    figure, axes = plt.subplots(2, 2, figsize=(6.3, 5.8), squeeze=False)
    audit: dict[str, Any] = {}
    for irow, yname in enumerate(parameters):
        for icol, xname in enumerate(parameters):
            axis = axes[irow, icol]
            if icol > irow:
                axis.set_axis_off()
                continue
            axis.set_xlim(*ranges[xname])
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            if irow == icol:
                bins = np.linspace(*ranges[xname], 90)
                axis.hist(original[xname], bins=bins, density=True, histtype="step", lw=1.8, color=COLORS["original"])
                axis.hist(gsm[xname], bins=bins, density=True, histtype="step", lw=1.8, color=COLORS["gsm"])
                axis.set_yticks([])
            else:
                axis.set_ylim(*ranges[yname])
                audit[f"{xname}_vs_{yname}"] = {
                    "original": draw_contour(axis, original[xname], original[yname], color=COLORS["original"], xlim=ranges[xname], ylim=ranges[yname], zorder=2),
                    "gsm": draw_contour(axis, gsm[xname], gsm[yname], color=COLORS["gsm"], xlim=ranges[xname], ylim=ranges[yname], zorder=4),
                }
            if irow == 1:
                axis.set_xlabel(r"$f_{\rm NL}$" if xname == "fNL" else r"$b_1$", fontsize=13)
            else:
                axis.tick_params(labelbottom=False)
            if icol == 0 and irow == 1:
                axis.set_ylabel(r"$b_1$", fontsize=13)
    handles = [plt.Line2D([], [], color=COLORS["original"], lw=2.0), plt.Line2D([], [], color=COLORS["gsm"], lw=2.0)]
    labels = [f"original: fNL={posterior_text(original['fNL'])}", f"GSM: fNL={posterior_text(gsm['fNL'])}"]
    figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.96), frameon=False, fontsize=10.5)
    figure.suptitle("2PCF-only shared-parameter overlay", fontsize=14.0, y=0.98)
    figure.text(0.5, 0.012, "Only fNL and b1 are overlaid; xi-side nuisance definitions remain separate.", ha="center", fontsize=8.0, color="0.4")
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.86, wspace=0.08, hspace=0.08)
    pdf.savefig(figure)
    plt.close(figure)
    return {"ranges": {k: list(v) for k, v in ranges.items()}, "contours": audit}


def main() -> None:
    output = OUTDIR / "task432_2pcf_original_vs_gsm_9.22style.pdf"
    audit_path = output.with_suffix(".json")
    if output.exists() or audit_path.exists():
        raise FileExistsError(output)
    products = build_models_and_covariance(xi_angle_mode="continuous", k_switch=0.01)
    cache_path = Path(products["metadata"]["cache"])
    with np.load(cache_path, allow_pickle=False) as payload:
        s_edges = np.asarray(payload["s_edges"], dtype="f8")
    mask = rawbox_xi_primary_mask(0.5 * (s_edges[:-1] + s_edges[1:]))
    basis = LinearRadialMomentProvider(cache_path, kmax_gsm=0.5, n_radial=2800)
    gsm_model = LinearGSMModel(basis, s_edges, shell_order=4, angular_order=12, stream_order=8, zmax=8.0)
    gsm_raw, gsm_summary = run_gsm_xi_mcmc(
        products=products,
        model=gsm_model,
        mask=mask,
        checkpoint_path=OUTDIR / "gsm_xi_mcmc_checkpoint.npz",
    )
    original = load_original(ORIGINAL)
    set_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary) as pdf:
        original_audit = corner_page(pdf, original, color=COLORS["original"], title="Task43 rawbox xi RSD: original FullDiscrete", xi_label=r"$\sigma_{s,\xi}\ [h^{-1}{\rm Mpc}]$")
        gsm_audit = corner_page(pdf, gsm_raw, color=COLORS["gsm"], title=r"Task43 rawbox xi RSD: linear GSM + $\sigma_{\rm FOG}$", xi_label=r"$\sigma_{\rm FOG}\ [h^{-1}{\rm Mpc}]$")
        shared_audit = overlay_page(pdf, original, gsm_raw)
    temporary.replace(output)
    audit = {
        "task": "task432_2pcf_original_vs_gsm_9.22style",
        "status": "pass",
        "scope": "xi0+xi2 marginal only; no P02 and no joint P+xi posterior",
        "style_reference": "9.18/9.22meeting lower-triangle corner",
        "original_chain": {"path": str(ORIGINAL), "sha256": sha256_file(ORIGINAL), "samples": int(original["fNL"].size)},
        "gsm_summary": gsm_summary,
        "parameter_definitions": {"original_xi_width": "sigma_s_xi", "gsm_xi_width": "sigma_FOG"},
        "pages": ["original xi02", "GSM xi02", "shared fNL/b1 overlay"],
        "contours": {"original": original_audit, "gsm": gsm_audit, "shared": shared_audit},
        "output_pdf": str(output),
        "output_pdf_sha256": sha256_file(output),
    }
    atomic_write_json(audit_path, audit)
    print(json.dumps({"status": "pass", "output": str(output), "audit": str(audit_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
