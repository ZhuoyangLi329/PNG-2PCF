#!/usr/bin/env python3
"""Plot Task43 lightcone fNL-b1 contours from minimal-closure samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


COLORS = {
    "no_gic": "#2563eb",
    "formal_gic": "#dc2626",
}
LABELS = {
    "no_gic": "no GIC",
    "formal_gic": "formal GIC",
}


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def contour_levels(hist: np.ndarray) -> list[float]:
    flat = np.sort(np.asarray(hist, dtype="f8").ravel())[::-1]
    if flat.size == 0 or np.sum(flat) <= 0:
        return []
    cdf = np.cumsum(flat) / np.sum(flat)
    levels = []
    for frac in (0.95, 0.68):
        idx = int(np.searchsorted(cdf, frac))
        level = float(flat[min(idx, flat.size - 1)])
        if level > 0:
            levels.append(level)
    return sorted(set(levels))


def model_entries(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = summary.get("models", [])
    if isinstance(models, dict):
        return {str(key): value for key, value in models.items()}
    return {str(item["model"]): item for item in models}


def load_samples(fit_dir: Path, model: str) -> np.ndarray:
    path = fit_dir / f"task43_mcmc_{model}_samples.npz"
    data = np.load(path, allow_pickle=False)
    return np.asarray(data["samples"], dtype="f8")[:, :2]


def symmetric_limits(values: np.ndarray, center: float = 0.0, pad: float = 0.10) -> tuple[float, float]:
    lo, hi = np.percentile(values, [0.3, 99.7])
    lo = min(float(lo), center)
    hi = max(float(hi), center)
    width = hi - lo
    return lo - pad * width, hi + pad * width


def b1_limits(values: np.ndarray, maps: list[float], pad: float = 0.12) -> tuple[float, float]:
    lo, hi = np.percentile(values, [0.3, 99.7])
    if maps:
        lo = min(float(lo), min(maps))
        hi = max(float(hi), max(maps))
    width = hi - lo
    return lo - pad * width, hi + pad * width


def draw_model_contour(
    ax: plt.Axes,
    samples: np.ndarray,
    *,
    model: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    color = COLORS.get(model, "#0f172a")
    hist, xedges, yedges = np.histogram2d(
        samples[:, 0],
        samples[:, 1],
        bins=160,
        range=[list(xlim), list(ylim)],
        density=False,
    )
    hist = gaussian_filter(hist.astype("f8"), sigma=2.0, mode="nearest")
    xcen = 0.5 * (xedges[:-1] + xedges[1:])
    ycen = 0.5 * (yedges[:-1] + yedges[1:])
    levels = contour_levels(hist)
    if levels:
        linestyles = ["--", "-"] if len(levels) == 2 else ["-"] * len(levels)
        ax.contour(
            xcen,
            ycen,
            hist.T,
            levels=levels,
            colors=color,
            linewidths=2.2,
            linestyles=linestyles,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--models", type=str, default="no_gic,formal_gic")
    parser.add_argument("--title", type=str, default="Task43 lightcone: s=50-350, ds=10")
    parser.add_argument("--truth-fnl", type=float, default=None)
    parser.add_argument("--truth-label", type=str, default=None)
    args = parser.parse_args()

    summary_path = args.fit_dir / "task43_minimal_closure_mcmc_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    entries = model_entries(summary)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]

    samples_by_model = {model: load_samples(args.fit_dir, model) for model in model_names}
    maps = [float(entries[model]["map"]["b1"]) for model in model_names]
    all_samples = np.vstack([samples_by_model[model] for model in model_names])
    xlim = symmetric_limits(all_samples[:, 0], center=0.0)
    ylim = b1_limits(all_samples[:, 1], maps)

    payload_models: dict[str, Any] = {}
    annotation_texts: dict[str, str] = {}
    for model in model_names:
        entry = entries[model]
        label = LABELS.get(model, model)
        fnl = entry["fnl_loc"]
        b1 = entry["b1"]
        map_fnl = float(entry["map"]["fnl_loc"])
        map_b1 = float(entry["map"]["b1"])
        chi2 = float(entry["data"]["chi2_map_total"])
        text = (
            f"{label}\n"
            f"MAP ({map_fnl:.1f}, {map_b1:.3f})\n"
            f"fNL={float(fnl['q50']):.1f}"
            f"-{float(fnl['q50']) - float(fnl['q16']):.1f}"
            f"+{float(fnl['q84']) - float(fnl['q50']):.1f}"
        )
        annotation_texts[model] = text
        payload_models[model] = {
            "label": label,
            "sample_path": str(args.fit_dir / f"task43_mcmc_{model}_samples.npz"),
            "parameter_names": entry.get("parameter_names", ["fnl_loc", "b1"]),
            "posterior": {"fnl_loc": fnl, "b1": b1},
            "map": entry["map"],
            "optimizer": entry.get("optimizer"),
            "chi2_map_total": chi2,
        }
        if "formal_gic" in entry:
            payload_models[model]["formal_gic"] = entry["formal_gic"]

    plot_backend = "getdist"
    try:
        from getdist import MCSamples, plots

        chains = []
        for model in model_names:
            chains.append(
                MCSamples(
                    samples=samples_by_model[model],
                    names=["fnl_loc", "b1"],
                    labels=[r"f_\mathrm{NL}^{\mathrm{loc}}", r"b_1"],
                    label=LABELS.get(model, model),
                    settings={"ignore_rows": 0, "smooth_scale_1D": 0.4, "smooth_scale_2D": 0.45},
                )
            )
        plotter = plots.get_subplot_plotter(width_inch=7.2)
        plotter.settings.axes_fontsize = 16
        plotter.settings.axes_labelsize = 22
        plotter.settings.legend_fontsize = 15
        plotter.settings.linewidth_contour = 2.0
        plotter.triangle_plot(
            chains,
            ["fnl_loc", "b1"],
            filled=True,
            contour_colors=[COLORS.get(model, "#0f172a") for model in model_names],
            legend_labels=[LABELS.get(model, model) for model in model_names],
        )
        fig = plotter.fig
        ax = plotter.subplots[1, 0]
        if plotter.subplots[0, 0] is not None:
            plotter.subplots[0, 0].axvline(0.0, color="#111827", ls="--", lw=1.1, alpha=0.85)
        ax.axvline(0.0, color="#111827", ls="--", lw=1.1, alpha=0.85)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    except Exception:
        plot_backend = "matplotlib_hist2d_smoothed"
        fig, ax = plt.subplots(figsize=(7.2, 5.9), constrained_layout=True)
        for model in model_names:
            draw_model_contour(ax, samples_by_model[model], model=model, xlim=xlim, ylim=ylim)
        ax.axvline(0.0, color="#111827", ls="--", lw=1.25, alpha=0.85)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$f_\mathrm{NL}^{\mathrm{loc}}$", fontsize=17, labelpad=6)
        ax.set_ylabel(r"$b_1$", fontsize=17)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(alpha=0.18)
        ax.set_title(args.title, fontsize=15)
        for model in model_names:
            ax.plot([], [], color=COLORS.get(model, "#0f172a"), lw=2.0, label=LABELS.get(model, model))
        ax.legend(loc="upper right", fontsize=12, frameon=False)

    summary_blocks = []
    for model in model_names:
        entry = entries[model]
        color = COLORS.get(model, "#0f172a")
        map_fnl = float(entry["map"]["fnl_loc"])
        map_b1 = float(entry["map"]["b1"])
        ax.scatter(map_fnl, map_b1, marker="*", s=130, color=color, edgecolor="white", linewidth=0.7, zorder=10)
        fnl = entry["fnl_loc"]
        b1 = entry["b1"]
        summary_blocks.append(
            f"{LABELS.get(model, model)}\n"
            f"MAP: fNL={map_fnl:.1f}, b1={map_b1:.3f}\n"
            f"fNL={float(fnl['q50']):.1f}"
            f" -{float(fnl['q50']) - float(fnl['q16']):.1f}"
            f" +{float(fnl['q84']) - float(fnl['q50']):.1f}\n"
            f"b1={float(b1['q50']):.3f}"
        )
    if plot_backend == "getdist":
        fig.text(
            0.56,
            0.72,
            "\n\n".join(summary_blocks),
            fontsize=12.0,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#d1d5db", "alpha": 0.9},
        )
    else:
        ax.text(
            0.04,
            0.96,
            "\n\n".join(summary_blocks),
            transform=ax.transAxes,
            fontsize=10.5,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#d1d5db", "alpha": 0.9},
        )

    if args.truth_fnl is not None:
        truth = float(args.truth_fnl)
        truth_label = str(args.truth_label) if args.truth_label is not None else f"true fNL={truth:g}"
        for maybe_ax in (ax, plotter.subplots[0, 0] if plot_backend == "getdist" and plotter.subplots[0, 0] is not None else None):
            if maybe_ax is None:
                continue
            maybe_ax.axvline(truth, color="#16a34a", ls="-.", lw=1.4, alpha=0.95)
        ax.plot([], [], color="#16a34a", ls="-.", lw=1.4, label=truth_label)
        ax.legend(loc="best", fontsize=11, frameon=False)

    out_pdf = args.output_prefix.with_suffix(".pdf")
    out_json = args.output_prefix.with_suffix(".json")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=200)
    plt.close(fig)

    theory = dict(summary.get("theory") or {})
    if "zeff" not in theory and summary.get("zeff") is not None:
        theory["zeff"] = summary.get("zeff")

    payload = {
        "status": "done",
        "task": "task43_lightcone_s50_350_ds10_rrdeconv_fnl_b1_contour",
        "summary_path": str(summary_path),
        "fit_range": summary.get("fit_range"),
        "fit_target": summary.get("fit_target"),
        "covariance": summary.get("covariance"),
        "theory": theory,
        "models": payload_models,
        "plot_backend": plot_backend,
        "plot_convention": "getdist-style triangle contour when available; MAP is marked with numeric labels; posterior q50/q16/q84 are recorded separately",
        "truth_fnl": None if args.truth_fnl is None else float(args.truth_fnl),
        "truth_label": args.truth_label,
        "outputs": {"pdf": str(out_pdf), "json": str(out_json)},
    }
    out_json.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_pdf}")
    print(f"[done] wrote {out_json}")


if __name__ == "__main__":
    main()
