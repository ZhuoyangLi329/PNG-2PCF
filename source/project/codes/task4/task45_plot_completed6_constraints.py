#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot a temporary constraints comparison for the six Task45 30k results that
were already complete when the diagnostic was requested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
OUT_ROOT = PROJECT_ROOT / "outputs" / "task4_outputs"
PLOT_ROOT = PROJECT_ROOT / "plots" / "task4" / "quijote_ultranest_free_sn0_30k"
DIAG_ROOT = PLOT_ROOT / "diagnostics"

PARAM_NAMES = ["fnl_loc", "b1", "sigmas", "sn0"]
PARAM_LABELS = {
    "fnl_loc": r"$f_{\mathrm{NL}}^{\mathrm{loc}}$",
    "b1": r"$b_1$",
    "sigmas": r"$\sigma_s\,[h^{-1}{\rm Mpc}]$",
    "sn0": r"$s_{n,0}$",
}


@dataclass(frozen=True)
class CompletedCase:
    output_label: str
    tag: str
    case: str
    plot_label: str
    color: str
    linestyle: str = "-"

    @property
    def root(self) -> Path:
        return OUT_ROOT / f"quijote_ultranest_{self.output_label}"

    @property
    def stem(self) -> str:
        return f"task45_{self.output_label}_{self.tag}_{self.case}"

    @property
    def summary_path(self) -> Path:
        return self.root / f"{self.stem}_summary.json"

    @property
    def sample_path(self) -> Path:
        return self.root / "samples" / f"{self.stem}_weighted_samples.npz"


CASES = [
    CompletedCase("free_sn0_kmax0p06_30k", "fid", "pk_binavg", r"$P(k)$ $k_{\max}=0.06$", "#4c78a8", "--"),
    CompletedCase("free_sn0_30k", "fid", "pk_binavg", r"$P(k)$ $k_{\max}=0.08$", "#2f4b7c", "-"),
    CompletedCase("free_sn0_kmax0p10_30k", "fid", "pk_binavg", r"$P(k)$ $k_{\max}=0.10$", "#0f6b63", "-."),
    CompletedCase("free_sn0_30k", "fid", "xi_r50_350", r"$\xi(r)$ 55-345", "#b95d36", "-"),
    CompletedCase("free_sn0_30k", "fid", "xi_r60_350", r"$\xi(r)$ 65-345", "#d08c2f", "-"),
    CompletedCase("free_sn0_30k", "fid", "xi_r80_350", r"$\xi(r)$ 85-345", "#8a5fbf", "-"),
]


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype="f8")
    weights = np.asarray(weights, dtype="f8")
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[mask]
    weights = weights[mask]
    if values.size == 0 or weights.sum() <= 0:
        return np.full(len(quantiles), np.nan)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.interp(quantiles, cdf, values)


def contour_levels(hist: np.ndarray, probs: tuple[float, float] = (0.95, 0.68)) -> list[float]:
    flat = np.asarray(hist, dtype="f8").ravel()
    flat = flat[np.isfinite(flat)]
    flat = flat[flat > 0]
    if flat.size == 0:
        return []
    order = np.argsort(flat)[::-1]
    sorted_vals = flat[order]
    cdf = np.cumsum(sorted_vals)
    cdf /= cdf[-1]
    levels = []
    for prob in probs:
        idx = int(np.searchsorted(cdf, prob, side="left"))
        idx = min(idx, sorted_vals.size - 1)
        levels.append(float(sorted_vals[idx]))
    return sorted(levels)


def load_case(case: CompletedCase) -> dict:
    if not case.summary_path.exists():
        raise FileNotFoundError(case.summary_path)
    if not case.sample_path.exists():
        raise FileNotFoundError(case.sample_path)
    summary = json.loads(case.summary_path.read_text(encoding="utf-8"))
    sample = np.load(case.sample_path)
    points = np.asarray(sample["points"], dtype="f8")
    weights = np.asarray(sample["weights"], dtype="f8")
    weights = weights / weights.sum()
    ncall = int(summary["ultranest"]["ncall"])
    if ncall < 30000:
        raise RuntimeError(f"{case.plot_label} has ncall={ncall}, below 30000")
    return {"case": case, "summary": summary, "points": points, "weights": weights}


def parameter_ranges(items: list[dict]) -> dict[str, tuple[float, float]]:
    ranges = {}
    for ipar, name in enumerate(PARAM_NAMES):
        lo_vals = []
        hi_vals = []
        for item in items:
            qlo, qhi = weighted_quantile(item["points"][:, ipar], item["weights"], [0.005, 0.995])
            lo_vals.append(qlo)
            hi_vals.append(qhi)
        lo = float(np.nanmin(lo_vals))
        hi = float(np.nanmax(hi_vals))
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            span = max(abs(lo), abs(hi), 1.0)
        ranges[name] = (lo - 0.08 * span, hi + 0.08 * span)
    return ranges


def plot_corner(items: list[dict], out_pdf: Path) -> None:
    ranges = parameter_ranges(items)
    ndim = len(PARAM_NAMES)
    fig, axes = plt.subplots(ndim, ndim, figsize=(12.5, 12.5))
    fig.patch.set_facecolor("#f7f7f5")

    bins_1d = 80
    bins_2d = 72
    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            ax.set_facecolor("#fbfbfa")
            if i < j:
                ax.axis("off")
                continue
            xname = PARAM_NAMES[j]
            yname = PARAM_NAMES[i]
            ax.set_xlim(*ranges[xname])
            if i > j:
                ax.set_ylim(*ranges[yname])
            for item in items:
                case = item["case"]
                points = item["points"]
                weights = item["weights"]
                if i == j:
                    hist, edges = np.histogram(
                        points[:, i],
                        bins=bins_1d,
                        range=ranges[xname],
                        weights=weights,
                        density=True,
                    )
                    hist = gaussian_filter(hist, sigma=1.0)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    ax.plot(centers, hist, color=case.color, lw=1.8, ls=case.linestyle, label=case.plot_label)
                    q16, q50, q84 = weighted_quantile(points[:, i], weights, [0.16, 0.5, 0.84])
                    ax.axvline(q50, color=case.color, lw=0.9, ls=case.linestyle, alpha=0.55)
                    ax.set_yticks([])
                else:
                    hist, xedges, yedges = np.histogram2d(
                        points[:, j],
                        points[:, i],
                        bins=bins_2d,
                        range=[ranges[xname], ranges[yname]],
                        weights=weights,
                    )
                    hist = gaussian_filter(hist.T, sigma=1.15)
                    levels = contour_levels(hist)
                    if levels:
                        xc = 0.5 * (xedges[:-1] + xedges[1:])
                        yc = 0.5 * (yedges[:-1] + yedges[1:])
                        ax.contour(xc, yc, hist, levels=levels, colors=[case.color], linewidths=[1.05, 1.8], linestyles=case.linestyle)
            if i == ndim - 1:
                ax.set_xlabel(PARAM_LABELS[xname], fontsize=12)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(PARAM_LABELS[yname], fontsize=12)
            elif i > j:
                ax.set_yticklabels([])
            ax.grid(color="#d7d7d3", lw=0.6, alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 0.985), fontsize=10.5)
    fig.suptitle("Completed Task45 30k results: fid parameter constraints", x=0.53, y=0.998, fontsize=16, fontweight="bold")
    fig.text(0.53, 0.025, "Contours show approximate 68% and 95% highest-density regions from weighted posterior histograms.", ha="center", fontsize=10.5, color="#4d4d4d")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.075, top=0.91, hspace=0.06, wspace=0.06)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_forest(items: list[dict], out_pdf: Path) -> None:
    labels = [item["case"].plot_label for item in items]
    y = np.arange(len(items))[::-1]
    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(18.0, 5.5), sharey=True, gridspec_kw={"wspace": 0.18})
    fig.patch.set_facecolor("#f7f7f5")
    for ax, pname in zip(axes, PARAM_NAMES):
        ax.set_facecolor("#fbfbfa")
        ipar = PARAM_NAMES.index(pname)
        lows = []
        highs = []
        for iy, item in zip(y, items):
            case = item["case"]
            q16, q50, q84 = weighted_quantile(item["points"][:, ipar], item["weights"], [0.16, 0.5, 0.84])
            lows.append(q16)
            highs.append(q84)
            ax.errorbar(q50, iy, xerr=np.array([[q50 - q16], [q84 - q50]]), fmt="o", ms=6, mfc="white", mec=case.color, mew=1.8, ecolor=case.color, elinewidth=1.7, capsize=3)
        ax.set_title(PARAM_LABELS[pname], fontsize=13)
        ax.grid(axis="x", color="#cfcfca", lw=0.7, alpha=0.7)
        span = max(highs) - min(lows)
        if span <= 0:
            span = max(abs(max(highs)), 1.0)
        ax.set_xlim(min(lows) - 0.08 * span, max(highs) + 0.08 * span)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.suptitle("Completed Task45 30k results: posterior median and 68% interval", y=0.98, fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.18, right=0.985, top=0.82, bottom=0.14)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    items = [load_case(case) for case in CASES]
    corner_pdf = DIAG_ROOT / "task45_completed6_fid_constraints_corner.pdf"
    forest_pdf = DIAG_ROOT / "task45_completed6_fid_constraints_forest.pdf"
    plot_corner(items, corner_pdf)
    plot_forest(items, forest_pdf)
    manifest = {
        "task": "task45_completed6_constraints",
        "status": "done",
        "cases": [
            {
                "label": item["case"].plot_label,
                "summary": str(item["case"].summary_path),
                "samples": str(item["case"].sample_path),
                "ncall": item["summary"]["ultranest"]["ncall"],
                "niter": item["summary"]["ultranest"]["niter"],
            }
            for item in items
        ],
        "outputs": {"corner_pdf": str(corner_pdf), "forest_pdf": str(forest_pdf)},
    }
    manifest_path = DIAG_ROOT / "task45_completed6_fid_constraints_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[write] {corner_pdf}")
    print(f"[write] {forest_pdf}")
    print(f"[write] {manifest_path}")


if __name__ == "__main__":
    main()
