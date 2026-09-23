#!/usr/bin/env python3
"""Analyze a manual-tuning parameter scan of EZmock RSD rawbox P02 at production density.

Each scan point uses ntracer = 1,847,000 and four realizations (seeds
433001..433004, shared across points).  Two scan axes are supported:

  rho_c    : rho_c = 1.14 ... 1.20 with pdf_base = 0.25
  pdf_base : pdf_base = 0.25 ... 0.50 (plus 0.375) with rho_c = 1.14

Both axes share the same baseline point (rho_c=1.14, pdf_base=0.25, seeds
433001..433004).  Reports total and shot-noise-subtracted band ratios against
the Abacus rawbox x25 mean, same-seed paired ratios between points (field phase
mostly cancels), and a determinism check of the baseline against the earlier
probe (same seeds).  Writes one JSON and one PDF per axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TUNING_ROOT = PROJECT_ROOT / "outputs/task43_outputs/ezmock_rsd_manual_tuning"
ABACUS_DIR = PROJECT_ROOT / "outputs/task43_outputs/rsd_validation/rawbox/pk"
PROBE_ROOT = TUNING_ROOT / "probe_ntracer1p847M_c1.14_e5_b0.25_v0_x3"
ABACUS_GLOB = "task43_rsd_rawbox_p02_AbacusSummit_base_c000_ph*_mmin1p4e13_mesh400.npz"
EZK_GLOB = "pk02_seed{seed}_mesh400.npz"
SCAN_AXES = {
    "rho_c": {
        "values": (1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.20),
        "label": "scan_ntracer1p847M_c{text}_e5_b0.25_v0_x4",
        "tag": "rhoc_scan_ntracer1p847M_x4",
        "xlabel": r"$\rho_c$",
        "suptitle": "EZmock RSD rawbox rho_c scan at production density: ntracer=1.847M, "
                    "rho_exp=5, pdf_base=0.25, sigma_v=0, fixampT, 4 realizations/point",
    },
    "pdf_base": {
        "values": (0.25, 0.30, 0.35, 0.375, 0.40, 0.45, 0.50),
        "label": "scan_ntracer1p847M_c1.14_e5_b{text}_v0_x4",
        "tag": "pdfbase_scan_ntracer1p847M_x4",
        "xlabel": "pdf_base",
        "suptitle": "EZmock RSD rawbox pdf_base scan at production density: ntracer=1.847M, "
                    "rho_c=1.14, rho_exp=5, sigma_v=0, fixampT, 4 realizations/point",
    },
}
BANDS = ((0.01, 0.03), (0.03, 0.05), (0.05, 0.08))
LOWK_MAX = 0.01
BOX_SIZE = 2000.0
BAND_LABELS = [f"{lo:.2f}-{hi:.2f}" for lo, hi in BANDS] + ["lowk<0.01"]


def value_text(value: float) -> str:
    """Two decimals for the 0.05 grid (0.30, 1.20), three for off-grid (0.375)."""
    text = f"{value:.3f}"
    return text[:-1] if text.endswith("0") else text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", choices=tuple(SCAN_AXES), default="rho_c")
    parser.add_argument("--tuning-root", type=Path, default=TUNING_ROOT)
    parser.add_argument("--abacus-dir", type=Path, default=ABACUS_DIR)
    parser.add_argument("--probe-root", type=Path, default=PROBE_ROOT)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-pdf", type=Path, default=None)
    return parser.parse_args()


def load_abacus(directory: Path) -> dict[str, np.ndarray]:
    paths = sorted(directory.glob(ABACUS_GLOB))
    if not paths:
        raise FileNotFoundError(f"no Abacus rawbox P02 under {directory}")
    k, pk0, pk2, shot0, shot2, ndata = None, [], [], [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as src:
            kk = np.asarray(src["k"], dtype="f8")
            if k is None:
                k = kk
            elif not np.allclose(kk, k, equal_nan=True):
                raise RuntimeError("Abacus rawbox k grids differ between phases")
            pk0.append(np.asarray(src["pk0"], dtype="f8"))
            pk2.append(np.asarray(src["pk2"], dtype="f8"))
            shot0.append(np.asarray(src["shotnoise0"], dtype="f8"))
            shot2.append(np.asarray(src["shotnoise2"], dtype="f8"))
            ndata.append(float(src["ndata"]))
    return {
        "n": len(paths), "k": k, "ndata": float(np.mean(ndata)),
        "pk0": np.asarray(pk0), "pk2": np.asarray(pk2),
        "shot0": float(np.mean(shot0)), "shot2": float(np.mean(shot2)),
    }


def load_point(root: Path) -> dict[str, np.ndarray]:
    paths = sorted(root.glob("measurements/" + EZK_GLOB.format(seed="*")))
    if not paths:
        raise FileNotFoundError(f"no EZmock measurement files under {root}")
    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as src:
            records.append({
                "seed": int(src["seed"]),
                "k": np.asarray(src["k0"], dtype="f8"),
                "pk0": np.asarray(src["pk0"], dtype="f8"),
                "pk2": np.asarray(src["pk2"], dtype="f8"),
                "shot0": float(np.mean(src["shotnoise0"])),
                "shot2": float(np.mean(src["shotnoise2"])),
                "ndata": float(src["ndata"]),
            })
    records.sort(key=lambda item: item["seed"])
    k = records[0]["k"]
    for item in records:
        if not np.allclose(item["k"], k, equal_nan=True):
            raise RuntimeError(f"EZmock k grids differ under {root}")
    return {
        "n": len(records), "k": k, "seeds": np.asarray([item["seed"] for item in records]),
        "pk0": np.asarray([item["pk0"] for item in records]),
        "pk2": np.asarray([item["pk2"] for item in records]),
        "shot0": np.asarray([item["shot0"] for item in records]),
        "shot2": np.asarray([item["shot2"] for item in records]),
        "ndata": np.asarray([item["ndata"] for item in records]),
    }


def interp_to(rows: np.ndarray, k_source: np.ndarray, k_target: np.ndarray) -> np.ndarray:
    if np.allclose(k_source, k_target, equal_nan=True):
        return rows
    out = np.full((rows.shape[0], k_target.size), np.nan)
    for index, row in enumerate(rows):
        mask = np.isfinite(k_source) & np.isfinite(row)
        out[index] = np.interp(k_target, k_source[mask], row[mask], left=np.nan, right=np.nan)
    return out


def band_mask(k: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (k >= lo) & (k < hi)


def band_mean(k: np.ndarray, values: np.ndarray, lo: float, hi: float) -> float:
    mask = band_mask(k, lo, hi) & np.isfinite(values)
    return float(np.mean(values[mask]))


def point_vs_abacus(point: dict, abacus: dict, shot_subtracted: bool) -> dict[str, dict[str, float]]:
    k = abacus["k"]
    out = {"P0": {}, "P2": {}}
    for ell, pk_key, shot_key, ab_key, ab_shot in (
        (0, "pk0", "shot0", "pk0", abacus["shot0"]),
        (2, "pk2", "shot2", "pk2", abacus["shot2"]),
    ):
        rows = interp_to(point[pk_key], point["k"], k)
        ab_mean = np.mean(abacus[ab_key], axis=0)
        ab_values = ab_mean - ab_shot if shot_subtracted else ab_mean
        num = rows - point[shot_key][:, None] if shot_subtracted else rows
        ratios = []
        for row in num:
            ratios.append([band_mean(k, row, lo, hi) / band_mean(k, ab_values, lo, hi) for lo, hi in BANDS])
        ratios = np.asarray(ratios)
        out[f"P{ell}"] = {
            "bands": {label: {"mean": float(ratios[:, i].mean()),
                              "std": float(ratios[:, i].std(ddof=1)),
                              "sem": float(ratios[:, i].std(ddof=1) / np.sqrt(ratios.shape[0]))}
                      for i, label in enumerate([f"{lo:.2f}-{hi:.2f}" for lo, hi in BANDS])},
            "lowk_ratio_of_means": (
                float(np.mean(np.nanmean(num, axis=0)[k < LOWK_MAX]))
                / float(np.mean(ab_values[k < LOWK_MAX]))
            ),
        }
    return out


def paired_ratios(hi_point: dict, lo_point: dict) -> dict[str, dict[str, float]]:
    shared = sorted(set(hi_point["seeds"].tolist()) & set(lo_point["seeds"].tolist()))
    if not shared:
        raise RuntimeError("no shared seeds for paired comparison")
    hi_index = {int(seed): i for i, seed in enumerate(hi_point["seeds"])}
    lo_index = {int(seed): i for i, seed in enumerate(lo_point["seeds"])}
    k = hi_point["k"]
    out = {}
    for ell, pk_key in ((0, "pk0"), (2, "pk2")):
        values = []
        for seed in shared:
            hi_row = hi_point[pk_key][hi_index[seed]]
            lo_row = lo_point[pk_key][lo_index[seed]]
            values.append([band_mean(k, hi_row, lo, hi) / band_mean(k, lo_row, lo, hi) for lo, hi in BANDS])
        values = np.asarray(values)
        out[f"P{ell}"] = {
            "n_paired": len(shared),
            "bands": {label: {"mean": float(values[:, i].mean()),
                              "std": float(values[:, i].std(ddof=1)),
                              "sem": float(values[:, i].std(ddof=1) / np.sqrt(values.shape[0]))}
                      for i, label in enumerate([f"{lo:.2f}-{hi:.2f}" for lo, hi in BANDS])},
        }
    return out


def fmt(bands: dict[str, dict[str, float]]) -> str:
    return ", ".join(f"{label}: {item['mean']:.4f}+-{item['sem']:.4f}" for label, item in bands.items())


def main() -> None:
    args = parse_args()
    scan = SCAN_AXES[args.axis]
    values = scan["values"]
    baseline = values[0]
    out_json = args.out_json or TUNING_ROOT / f"_scan_driver/task43_ezmock_rsd_rawbox_{scan['tag']}.json"
    out_pdf = args.out_pdf or PROJECT_ROOT / f"test_figure/task43_ezmock_rsd_rawbox_{scan['tag']}.pdf"

    abacus = load_abacus(args.abacus_dir)
    points = {}
    for value in values:
        root = args.tuning_root / scan["label"].format(text=value_text(value))
        if not sorted(root.glob("measurements/" + EZK_GLOB.format(seed="*"))):
            continue
        point = load_point(root)
        if point["n"] != 4:
            print(f"[skip] {args.axis}={value_text(value)}: only {point['n']} realizations present (still running?)")
            continue
        points[value] = point
    if baseline not in points:
        raise RuntimeError(f"baseline {args.axis}={value_text(baseline)} point is missing or incomplete")
    available = tuple(value for value in values if value in points)
    if not np.allclose(points[baseline]["k"], abacus["k"], rtol=1e-3, atol=0.0, equal_nan=True):
        raise RuntimeError("EZmock and Abacus k grids differ")

    report = {
        "task": "task43_analyze_ezmock_rsd_rawbox_param_scan",
        "status": "done",
        "scan_axis": args.axis,
        "ntracer": 1847000,
        "seeds": points[baseline]["seeds"].tolist(),
        "n_abacus": abacus["n"],
        "nbar_box": float(np.mean(points[baseline]["ndata"])) / BOX_SIZE ** 3,
        "abacus_shot": {"P0": abacus["shot0"], "P2": abacus["shot2"]},
        "expected_shot_1_over_nbar": BOX_SIZE ** 3 / abacus["ndata"],
        "scan_values_analyzed": [value_text(value) for value in available],
        "points": {},
    }

    for value in available:
        point = points[value]
        report["points"][value_text(value)] = {
            "ndata_mean": float(point["ndata"].mean()),
            "ndata_std": float(point["ndata"].std(ddof=1)),
            "shot0_mean": float(point["shot0"].mean()),
            "shot2_mean": float(point["shot2"].mean()),
            "total": point_vs_abacus(point, abacus, shot_subtracted=False),
            "shot_subtracted": point_vs_abacus(point, abacus, shot_subtracted=True),
            "paired_vs_baseline": paired_ratios(point, points[baseline]) if value != baseline else None,
        }
    for value_hi, value_lo in zip(available[1:], available[:-1]):
        report["points"][value_text(value_hi)]["paired_vs_prev"] = paired_ratios(points[value_hi], points[value_lo])

    probe_root = args.probe_root
    probe_paths = sorted(probe_root.glob("measurements/" + EZK_GLOB.format(seed="*")))
    if probe_paths and all(seed in points[baseline]["seeds"].tolist() for seed in (433001, 433002, 433003)):
        scan_index = {int(seed): i for i, seed in enumerate(points[baseline]["seeds"])}
        max_rel = 0.0
        for path in probe_paths:
            with np.load(path, allow_pickle=False) as src:
                seed = int(src["seed"])
                if seed not in scan_index:
                    continue
                row = scan_index[seed]
                if not np.allclose(np.asarray(src["pk0"], dtype="f8"), points[baseline]["pk0"][row],
                                   rtol=1e-9, atol=0.0, equal_nan=True):
                    max_rel = max(max_rel, float(np.nanmax(np.abs(
                        np.asarray(src["pk0"], dtype="f8") / points[baseline]["pk0"][row] - 1.0))))
        report["probe_determinism_check"] = {"max_rel_pk0_diff": max_rel}

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    colors = ("#1f77b4", "#2ca02c", "#d62728")
    xs = np.asarray(available)
    for ax, ell in ((axes[0], 0), (axes[1], 2)):
        for color, label in zip(colors, BAND_LABELS[:-1]):
            means = [report["points"][value_text(value)]["shot_subtracted"][f"P{ell}"]["bands"][label]["mean"]
                     for value in available]
            sems = [report["points"][value_text(value)]["shot_subtracted"][f"P{ell}"]["bands"][label]["sem"]
                    for value in available]
            ax.errorbar(xs, means, yerr=sems, color=color, marker="o", ms=4, lw=1.8, capsize=3,
                        label=rf"${label}$")
        ax.axhline(1.0, color="#555555", lw=1.2, ls="--")
        ax.set(xlabel=scan["xlabel"], ylabel="EZmock / Abacus (shot-subtracted)",
               title=rf"rawbox $P_{ell}$ band ratio")
        ax.legend(frameon=False, loc="best")
    fig.suptitle(scan["suptitle"])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"[done] json: {out_json}")
    print(f"[done] pdf:  {out_pdf}")
    if "probe_determinism_check" in report:
        print(f"  probe determinism: max |pk0 rel diff| = {report['probe_determinism_check']['max_rel_pk0_diff']:.3e}")
    for value in available:
        item = report["points"][value_text(value)]
        print(f"  {args.axis}={value_text(value)}  ndata={item['ndata_mean']:.0f}  shot0={item['shot0_mean']:.1f}")
        print(f"    P0 total   -> {fmt(item['total']['P0']['bands'])}")
        print(f"    P0 subtr.  -> {fmt(item['shot_subtracted']['P0']['bands'])}")
        print(f"    P2 total   -> {fmt(item['total']['P2']['bands'])}")
        print(f"    P2 subtr.  -> {fmt(item['shot_subtracted']['P2']['bands'])}")
        if item["paired_vs_baseline"] is not None:
            print(f"    paired/{value_text(baseline)} P0 -> {fmt(item['paired_vs_baseline']['P0']['bands'])}")
            print(f"    paired/{value_text(baseline)} P2 -> {fmt(item['paired_vs_baseline']['P2']['bands'])}")


if __name__ == "__main__":
    main()
