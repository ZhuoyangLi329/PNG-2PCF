#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate original-vs-linear-GSM posterior contours for Task 4.3.2.

这个脚本只比较当前 promoted rawbox BAO-mask contract 下的两个 joint fit：

1. original: FullDiscrete P02 + xi02 split-sigma chain already produced by
   ``task432_split_sigma_rawbox.py``;
2. GSM: current P-side model + linear-input configuration-space GSM xi-side,
   with one eBOSS-style ``sigma_FOG`` nuisance.

不比较 9.11 旧 contract，也不把旧 unmasked contour 混进来。输出只有 PDF。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_DIR = PROJECT_ROOT / "codes" / "task43"
TASK432_DIR = PROJECT_ROOT / "codes" / "task432"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "task432_model_repair"
PLOT_ROOT = PROJECT_ROOT / "plots" / "task43" / "rsd_validation" / "task432_model_repair"
ORIGINAL_CHAIN = OUTPUT_ROOT / "rawbox_baomask_split_sigma_longchain" / "fits" / "joint" / "samples.npz"

for _path in (TASK43_DIR, TASK432_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from task43_rsd_common import rawbox_xi_primary_mask  # noqa: E402
from task432_linear_gsm_rawbox import (  # noqa: E402
    DELTA_C,
    LinearGSMModel,
    LinearRadialMomentProvider,
    build_models_and_covariance,
    precision_from_covariance,
)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
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


def posterior_summary(samples: np.ndarray, names: list[str]) -> dict[str, dict[str, float]]:
    quantiles = np.percentile(samples, [16.0, 50.0, 84.0], axis=0)
    return {
        name: {
            "q16": float(quantiles[0, index]),
            "q50": float(quantiles[1, index]),
            "q84": float(quantiles[2, index]),
            "sigma68": float(0.5 * (quantiles[2, index] - quantiles[0, index])),
        }
        for index, name in enumerate(names)
    }


def contour_levels(samples_x: np.ndarray, samples_y: np.ndarray, bins: int = 70) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Return smoothed 2D posterior density and enclosed-probability levels."""

    x = np.asarray(samples_x, dtype="f8")
    y = np.asarray(samples_y, dtype="f8")
    xlo, xhi = np.percentile(x, [0.15, 99.85])
    ylo, yhi = np.percentile(y, [0.15, 99.85])
    if xlo == xhi:
        xlo, xhi = xlo - 1.0, xhi + 1.0
    if ylo == yhi:
        ylo, yhi = ylo - 1.0, yhi + 1.0
    hist, xedges, yedges = np.histogram2d(x, y, bins=int(bins), range=((xlo, xhi), (ylo, yhi)))
    smooth = gaussian_filter(hist.astype("f8"), sigma=1.2)
    if float(np.sum(smooth)) <= 0.0:
        raise RuntimeError("empty posterior histogram")
    smooth /= float(np.sum(smooth))
    sorted_density = np.sort(smooth.ravel())[::-1]
    cumulative = np.cumsum(sorted_density)
    level68 = float(sorted_density[np.searchsorted(cumulative, 0.68, side="left")])
    level95 = float(sorted_density[np.searchsorted(cumulative, 0.95, side="left")])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    return xc, yc, smooth.T, [level95, level68]


def plot_pair(
    axis: Any,
    original: dict[str, np.ndarray],
    gsm: dict[str, np.ndarray],
    xname: str,
    yname: str,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Overlay original and GSM contours with consistent labels."""

    colors = {"original": "#4C72B0", "gsm": "#C44E52"}
    for label, samples in (("original", original), ("gsm", gsm)):
        xgrid, ygrid, density, levels = contour_levels(samples[xname], samples[yname])
        axis.contour(xgrid, ygrid, density, levels=levels, colors=colors[label], linewidths=(1.5, 2.0))
        axis.plot(np.median(samples[xname]), np.median(samples[yname]), marker="o", color=colors[label], ms=4)
    axis.set_xlabel(xlabel or xname)
    axis.set_ylabel(ylabel or yname)
    axis.grid(True, alpha=0.22)


def load_original(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    with np.load(path, allow_pickle=False) as payload:
        samples = np.asarray(payload["samples"], dtype="f8")
        names = [str(value) for value in np.asarray(payload["parameter_names"]).ravel()]
    return {name: samples[:, index] for index, name in enumerate(names)}, names


def run_gsm_mcmc(
    *,
    products: dict[str, Any],
    gsm_model: LinearGSMModel,
    mask: np.ndarray,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    start: np.ndarray,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 100,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample the GSM+sigma_FOG joint likelihood under the frozen covariance."""

    data = np.concatenate([np.asarray(products["data"]["p"], dtype="f8"), np.asarray(products["data"]["x"], dtype="f8")])
    covariance = products["covariances"]["joint"]
    precision = precision_from_covariance(covariance)
    p_model = products["models"]["p"]

    def evaluate(theta: np.ndarray) -> np.ndarray:
        fnl, b1, sigma_p, sigma_fog, sn0 = np.asarray(theta, dtype="f8")
        p_vector = p_model(np.asarray([fnl, b1, sigma_p, 0.0, sn0], dtype="f8"))
        x_vector = gsm_model.vector(fnl=float(fnl), b1=float(b1), sigma_fog=float(sigma_fog), mask=mask)
        return np.concatenate([p_vector, x_vector])

    def log_probability(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype="f8")
        if np.any(theta <= bounds_lo) or np.any(theta >= bounds_hi):
            return -np.inf
        residual = data - evaluate(theta)
        return -0.5 * float(residual @ precision @ residual)

    rng = np.random.default_rng(int(seed))
    spread = np.asarray([3.0, 0.012, 0.45, 0.8, 0.015], dtype="f8")
    initial = np.asarray(start, dtype="f8")[None, :] + rng.normal(size=(int(nwalkers), 5)) * spread[None, :]
    initial = np.maximum(initial, bounds_lo[None, :] + 1.0e-7)
    initial = np.minimum(initial, bounds_hi[None, :] - 1.0e-7)
    np.random.seed(int(seed))
    sampler = emcee.EnsembleSampler(int(nwalkers), 5, log_probability)
    state = initial
    completed_steps = 0
    while completed_steps < int(nsteps):
        chunk = min(int(checkpoint_every), int(nsteps) - completed_steps)
        state = sampler.run_mcmc(state, chunk, progress=False, skip_initial_state_check=True)
        completed_steps += chunk
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{os.getpid()}.tmp")
            np.savez_compressed(
                temporary,
                chain=sampler.get_chain(),
                log_prob=sampler.get_log_prob(),
                last_positions=np.asarray(state.coords, dtype="f8"),
                completed_steps=np.asarray(completed_steps, dtype="i8"),
            )
            # np.savez appends .npz if the temporary path lacks it.
            generated = Path(str(temporary) + ".npz") if not temporary.name.endswith(".npz") else temporary
            generated.replace(checkpoint_path)
    samples = sampler.get_chain(discard=int(burnin), flat=True)
    logp = sampler.get_log_prob(discard=int(burnin), flat=True)
    names = ["fNL", "b1", "sigma_s_P", "sigma_FOG", "sn0"]
    summary = {
        "parameter_names": names,
        "posterior": posterior_summary(samples, names),
        "mcmc": {
            "nwalkers": int(nwalkers),
            "nsteps": int(nsteps),
            "burnin": int(burnin),
            "seed": int(seed),
        },
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
    }
    try:
        tau = np.asarray(emcee.autocorr.integrated_time(sampler.get_chain(discard=int(burnin)), tol=0), dtype="f8")
        summary["postburn_length_over_tau"] = (sampler.get_chain(discard=int(burnin)).shape[0] / tau).tolist()
    except Exception as exc:
        summary["tau_error"] = str(exc)
    return samples, logp, summary


def make_pdf(
    path: Path,
    original: dict[str, np.ndarray],
    gsm: dict[str, np.ndarray],
    original_summary: dict[str, Any],
    gsm_summary: dict[str, Any],
) -> None:
    """Create a PDF-only contour comparison."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary) as pdf:
        figure, axis = plt.subplots(figsize=(7.2, 5.8))
        plot_pair(axis, original, gsm, "fNL", "b1", xlabel="fNL", ylabel="b1")
        axis.plot([], [], color="#4C72B0", lw=2, label="original FullDiscrete split-sigma")
        axis.plot([], [], color="#C44E52", lw=2, label="linear GSM + sigma_FOG")
        axis.legend(frameon=False, loc="best")
        axis.set_title("rawbox BAO-masked joint posterior")
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
        plot_pair(axis=axes[0], original=original, gsm=gsm, xname="b1", yname="sigma_s_P", xlabel="b1", ylabel="P-side sigma")
        # The old xi nuisance and new GSM nuisance have different definitions,
        # but this plot makes the posterior-width change visually explicit.
        old = {"b1": original["b1"], "sigma_FOG": original["sigma_s_xi"]}
        new = {"b1": gsm["b1"], "sigma_FOG": gsm["sigma_FOG"]}
        plot_pair(axis=axes[1], original=old, gsm=new, xname="b1", yname="sigma_FOG", xlabel="b1", ylabel="xi-side width")
        axes[1].set_title("xi-side nuisance: definitions differ")
        axes[0].set_title("P-side nuisance")
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8.5, 6.0))
        lines = [
            "Contract: 50 <= s < 350 Mpc/h; 80 <= s < 120 removed",
            "Original: FullDiscrete P02 + xi02 split-sigma joint chain",
            "New: current P-side + linear-input GSM xi-side + sigma_FOG",
            "Covariance: same fixed promoted rawbox joint covariance",
            "",
            "Original posterior:",
        ]
        for name in ("fNL", "b1", "sigma_s_P", "sigma_s_xi", "sn0"):
            row = original_summary["posterior"][name]
            lines.append(f"  {name:12s} {row['q50']: .5g} +{row['q84']-row['q50']:.4g} -{row['q50']-row['q16']:.4g}")
        lines.append("")
        lines.append("GSM posterior:")
        for name in ("fNL", "b1", "sigma_s_P", "sigma_FOG", "sn0"):
            row = gsm_summary["posterior"][name]
            lines.append(f"  {name:12s} {row['q50']: .5g} +{row['q84']-row['q50']:.4g} -{row['q50']-row['q16']:.4g}")
        lines.extend(
            [
                "",
                f"GSM acceptance fraction: {gsm_summary['acceptance_fraction_mean']:.4f}",
                f"GSM postburn length/tau: {gsm_summary.get('postburn_length_over_tau', 'unavailable')}",
            ]
        )
        axis.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", transform=axis.transAxes)
        axis.axis("off")
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kmax-gsm", type=float, default=0.5)
    parser.add_argument("--nwalkers", type=int, default=24)
    parser.add_argument("--nsteps", type=int, default=1800)
    parser.add_argument("--burnin", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT / "rawbox_linear_gsm_contours")
    args = parser.parse_args()
    if int(args.nwalkers) < 12 or int(args.nsteps) <= int(args.burnin):
        raise ValueError("need at least 12 walkers and nsteps > burnin")

    started = time.perf_counter()
    products = build_models_and_covariance(xi_angle_mode="continuous", k_switch=0.01)
    cache_path = Path(products["metadata"]["cache"])
    with np.load(cache_path, allow_pickle=False) as payload:
        s_edges = np.asarray(payload["s_edges"], dtype="f8")
    mask = rawbox_xi_primary_mask(0.5 * (s_edges[:-1] + s_edges[1:]))
    basis = LinearRadialMomentProvider(cache_path, kmax_gsm=float(args.kmax_gsm), n_radial=2800)
    gsm_model = LinearGSMModel(basis, s_edges, shell_order=4, angular_order=12, stream_order=8, zmax=8.0)
    start = np.asarray([2.3, 2.571, 0.7, 16.9, 0.15], dtype="f8")
    bounds_lo = np.asarray([-500.0, 0.5, 0.0, 0.0, -1.0], dtype="f8")
    bounds_hi = np.asarray([500.0, 5.0, 30.0, 30.0, 1.0], dtype="f8")
    samples, logp, gsm_summary = run_gsm_mcmc(
        products=products,
        gsm_model=gsm_model,
        mask=mask,
        nwalkers=int(args.nwalkers),
        nsteps=int(args.nsteps),
        burnin=int(args.burnin),
        seed=int(args.seed),
        start=start,
        bounds_lo=bounds_lo,
        bounds_hi=bounds_hi,
        checkpoint_path=Path(args.output_root) / "linear_gsm_fog_mcmc_checkpoint.npz",
        checkpoint_every=int(args.checkpoint_every),
    )
    original, original_names = load_original(ORIGINAL_CHAIN)
    original_summary = {
        "parameter_names": original_names,
        "posterior": posterior_summary(
            np.column_stack([original[name] for name in original_names]), original_names
        ),
    }
    gsm = {name: samples[:, index] for index, name in enumerate(["fNL", "b1", "sigma_s_P", "sigma_FOG", "sn0"])}
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_root / "linear_gsm_fog_joint_samples.npz",
        samples=samples,
        log_prob=logp,
        parameter_names=np.asarray(["fNL", "b1", "sigma_s_P", "sigma_FOG", "sn0"]),
    )
    audit = {
        "task": "Task 4.3.2 original-vs-linear-GSM contour comparison",
        "status": "complete",
        "contract": products["metadata"],
        "kmax_gsm_h_mpc": float(args.kmax_gsm),
        "original_chain": str(ORIGINAL_CHAIN),
        "new_chain": str(output_root / "linear_gsm_fog_joint_samples.npz"),
        "original_summary": original_summary,
        "gsm_summary": gsm_summary,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    atomic_json(output_root / "task432_linear_gsm_contours.json", audit)
    make_pdf(
        PLOT_ROOT / "task432_original_vs_linear_gsm_contours.pdf",
        original,
        gsm,
        original_summary,
        gsm_summary,
    )
    print(json.dumps({"status": "complete", "pdf": str(PLOT_ROOT / "task432_original_vs_linear_gsm_contours.pdf"), "new_samples": int(samples.shape[0])}, sort_keys=True))


if __name__ == "__main__":
    main()
