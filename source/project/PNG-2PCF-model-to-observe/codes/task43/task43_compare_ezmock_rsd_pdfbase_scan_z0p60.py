#!/usr/bin/env python3
"""pdf_base scan at fixed EZmock snapshot redshift (default z=0.60) vs Abacus lightcone x25.

Top row: P0/P2 curves, one color per pdf_base. Bottom row: P0 and P2 band ratios vs
pdf_base with the 1.0 line, so the value that centers the lightcone amplitude is read
off directly. Writes a band table (with linear 1.0-crossing estimates) to JSON and
only a PDF figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task43_compare_ezmock_rsd_lightcone_pilot_vs_abacus import (  # noqa: E402
    ABACUS_DIR, BANDS, LOWK_MAX, PROJECT_ROOT, TARGET_NDATA,
    interpolate_to, load_abacus_pk, load_ezmock_pk, mean_std_rows,
)
from task43_compare_ezmock_rsd_zscan_vs_abacus import band_ratio  # noqa: E402

ROOT_TEMPLATE = "outputs/task43_outputs/ezmock_rsd_lightcone_x10_fixampT_nzmatch_b{b}_z0.60_pilot"
DEFAULT_POINTS = "0.20:5,0.28:5,0.30:5,0.32:5,0.35:all"
OUT_PDF = PROJECT_ROOT / "test_figure/task43_ezmock_rsd_lightcone_fixampT_nzmatch_z0.60_pdfbase_scan_c1.14_e5_v0_vs_abacus_x25.pdf"
OUT_JSON = PROJECT_ROOT / ("outputs/task43_outputs/ezmock_rsd_lightcone_x10_fixampT_nzmatch_z0.60_pdfbase_scan_diagnostics/"
                           "task43_ezmock_rsd_lightcone_fixampT_nzmatch_z0.60_pdfbase_scan_vs_abacus_x25.json")
BAND_LABELS = tuple(f"{lo:.2f}-{hi:.2f}" for lo, hi in BANDS) + ("k<0.01",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", default=DEFAULT_POINTS,
                        help="Comma-separated pdf_base:limit pairs (limit 'all' uses every realization).")
    parser.add_argument("--root-template", default=ROOT_TEMPLATE)
    parser.add_argument("--z-box", type=float, default=0.60)
    parser.add_argument("--abacus-dir", type=Path, default=ABACUS_DIR)
    parser.add_argument("--out-pdf", type=Path, default=OUT_PDF)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    return parser.parse_args()


def crossing(pdf_base: np.ndarray, ratio: np.ndarray) -> float | None:
    for left in range(ratio.size - 1):
        a, b = ratio[left] - 1.0, ratio[left + 1] - 1.0
        if a == 0.0 or a * b < 0.0:
            return float(np.interp(0.0, [a, b], [pdf_base[left], pdf_base[left + 1]]))
    return None


def main() -> None:
    args = parse_args()
    points = []
    for part in args.points.split(","):
        token, limit = part.strip().split(":")
        points.append((token, None if limit == "all" else int(limit)))
    abacus = load_abacus_pk(args.abacus_dir)
    abacus_mean0 = np.mean(abacus["pk0"], axis=0)
    abacus_mean2 = np.mean(abacus["pk2"], axis=0)

    report: dict[str, object] = {
        "task": "task43_compare_ezmock_rsd_pdfbase_scan_z0p60",
        "status": "done", "z_box": float(args.z_box),
        "abacus_dir": str(args.abacus_dir), "n_abacus": abacus["n"], "points": {},
    }
    curves = {}
    for token, limit in points:
        root = PROJECT_ROOT / args.root_template.format(b=token)
        pilot = load_ezmock_pk(root, limit)
        pk0 = interpolate_to(pilot["pk0"], pilot["k"], abacus["k"])
        pk2 = interpolate_to(pilot["pk2"], pilot["k"], abacus["k"])
        mean0, std0 = mean_std_rows(pk0)
        mean2, std2 = mean_std_rows(pk2)
        curves[token] = {"mean0": mean0, "std0": std0, "mean2": mean2, "std2": std2}
        report["points"][token] = {
            "root": str(root.relative_to(PROJECT_ROOT)), "n_real": pilot["n"], "seeds": pilot["seeds"],
            "ndata_mean": float(np.mean(pilot["ndata"])),
            "ndata_offset": float(np.mean(pilot["ndata"]) / TARGET_NDATA - 1.0),
            "bands": {"P0": band_ratio(abacus["k"], mean0, std0, abacus_mean0, pilot["n"]),
                      "P2": band_ratio(abacus["k"], mean2, std2, abacus_mean2, pilot["n"])},
        }
        p0 = report["points"][token]["bands"]["P0"]
        print(f"[b={token}] n={pilot['n']} ndata={report['points'][token]['ndata_mean']:.0f} "
              f"P0 band={[round(p0[b]['ratio'], 4) for b in BAND_LABELS[:3]]} "
              f"lowk={p0['k<0.01']['ratio']:.4f}", flush=True)

    pdf_base = np.asarray([float(token) for token, _ in points])
    crossings = {}
    for name in ("P0", "P2"):
        for band in BAND_LABELS[:3]:
            ratio = np.asarray([report["points"][token]["bands"][name][band]["ratio"] for token, _ in points])
            crossings[f"{name} {band}"] = crossing(pdf_base, ratio)
    crossings["P0 k<0.01"] = crossing(pdf_base, np.asarray(
        [report["points"][token]["bands"]["P0"]["k<0.01"]["ratio"] for token, _ in points]))
    report["crossing_pdf_base"] = crossings

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)
    ax_p0, ax_p2, ax_b0, ax_b2 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    legend_handles = []
    for index, (token, _) in enumerate(points):
        color = plt.cm.viridis(index / max(len(points) - 1, 1))
        curve = curves[token]
        label = f"EZmock b={token} (x{report['points'][token]['n_real']})"
        for ax, mean, std in ((ax_p0, curve["mean0"], curve["std0"]), (ax_p2, curve["mean2"], curve["std2"])):
            valid = np.isfinite(mean) & (mean > 0.0)
            line, = ax.plot(abacus["k"][valid], mean[valid], color=color, lw=2.0, ls="--", label=label)
            if ax is ax_p0:
                legend_handles.append(line)
            if np.all(np.isfinite(std)):
                ax.fill_between(abacus["k"][valid], (mean - std)[valid], (mean + std)[valid],
                                color=color, alpha=0.12, lw=0)
    for ax, ell, ref, pmin in ((ax_p0, 0, abacus_mean0, None), (ax_p2, 2, abacus_mean2, 1.0e3)):
        valid = np.isfinite(ref) & (ref > 0.0)
        ref_line, = ax.plot(abacus["k"][valid], ref[valid], color="#222222", lw=2.8,
                            label="Abacus lightcone x25 mean")
        tops = [float(np.max(ref[valid]))]
        for token, _ in points:
            series = curves[token][f"mean{ell}"]
            mask = np.isfinite(series) & (series > 0.0)
            if np.any(mask):
                tops.append(float(np.max(series[mask])))
        ax.set(xscale="log", yscale="log", xlabel=r"$k\ [h\,Mpc^{-1}]$", ylabel=rf"$P_{ell}(k)$",
               title=rf"RSD lightcone $P_{ell}$ (pdf_base scan, $z$={args.z_box:g})",
               ylim=(pmin, 1.15 * max(tops)))
    ax_p0.legend(handles=[ref_line] + legend_handles, frameon=False, loc="lower left", fontsize=9)

    for ax, name in ((ax_b0, "P0"), (ax_b2, "P2")):
        labels = BAND_LABELS if name == "P0" else BAND_LABELS[:3]
        for band_index, band in enumerate(labels):
            ys = np.asarray([report["points"][token]["bands"][name][band]["ratio"] for token, _ in points])
            es = np.asarray([report["points"][token]["bands"][name][band]["err"] for token, _ in points])
            ax.errorbar(pdf_base, ys, yerr=es, marker="os^d"[band_index], ms=6, lw=1.8, capsize=3,
                        color=plt.cm.tab10(band_index),
                        label=rf"$k\in[{band}]$" if band != "k<0.01" else r"$k<0.01$")
        ax.axhline(1.0, color="#888888", lw=1.0, ls=":")
        ax.set(xlabel=r"EZmock pdf\_base", ylabel=rf"$P_{name[-1]}$ band ratio (EZ/Abacus)",
               title=rf"{name} band ratio vs pdf\_base ($z$={args.z_box:g})")
        ax.legend(frameon=False, fontsize=9, loc="best")

    fig.suptitle(f"EZmock RSD lightcone pdf_base scan at z={args.z_box:g} vs Abacus lightcone x25 "
                 f"(c000, mmin=1.4e13, 0.4<z_obs<0.8)\n"
                 "rho_c=1.14, rho_exp=5, sigma_v=0, fixampT, n(z)-matched, BOX_SIZE=2000/320^3, mesh256; "
                 "x5 per point, x10 at b=0.35 (seeds 610001-...)", fontsize=11)
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[crossing pdf_base] " + "  ".join(f"{k}={v if v is None else round(v, 4)}" for k, v in crossings.items()))
    print(f"[done] {args.out_pdf}\n[done] {args.out_json}")


if __name__ == "__main__":
    main()
