#!/usr/bin/env python3
"""z-scan of the fixampT n(z)-matched EZmock RSD lightcone (pdf_base fixed) vs Abacus x25.

Top row: P0/P2 curves, one color per snapshot redshift. Bottom row: quad-to-mono
band ratio vs z (does lowering z fix the large-scale quad?) and P0 band ratio vs z
(z is NOT the amplitude knob; pdf_base is). Writes the band table to JSON and only
a PDF figure.
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
    ABACUS_DIR, BANDS, LOWK_MAX, OLD_ROOT, P2_YMIN, PROJECT_ROOT, TARGET_NDATA,
    interpolate_to, load_abacus_pk, load_ezmock_pk, load_lightcone_Z, mean_std_rows,
    nz_ratio,
)

ROOT_TEMPLATE = "outputs/task43_outputs/ezmock_rsd_lightcone_x10_fixampT_nzmatch_b0.20_z{z}_pilot"
COMMON_RANDOM = OLD_ROOT / "common_random/common_random_zobs0p4_0p8_x50.npz"
OUT_PDF = PROJECT_ROOT / "test_figure/task43_ezmock_rsd_lightcone_fixampT_nzmatch_b0.20_zscan_c1.14_e5_v0_x5_vs_abacus_x25.pdf"
OUT_JSON = PROJECT_ROOT / ("outputs/task43_outputs/ezmock_rsd_lightcone_x10_fixampT_nzmatch_b0.20_zscan_diagnostics/"
                           "task43_ezmock_rsd_lightcone_fixampT_nzmatch_b0.20_zscan_x5_vs_abacus_x25.json")
# n2dV pair-weighted mean of the frozen 25-phase-mean Abacus FKP n(z) over 0.4<z<0.8
Z_EFF = 0.6032
COLORS = {0.725: "#d62728", 0.6: "#2ca02c", 0.65: "#ff7f0e", 0.55: "#1f77b4"}
MARKERS = ("o", "s", "^")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--z-values", default="0.725,0.65,0.60,0.55",
                        help="Comma-separated snapshot redshifts (root name uses %g).")
    parser.add_argument("--root-template", default=ROOT_TEMPLATE)
    parser.add_argument("--abacus-dir", type=Path, default=ABACUS_DIR)
    parser.add_argument("--pk-limit", type=int, default=None)
    parser.add_argument("--out-pdf", type=Path, default=OUT_PDF)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--no-nz", action="store_true", help="Skip the n(z) gate (slower).")
    return parser.parse_args()


def band_ratio(k: np.ndarray, mean: np.ndarray, std: np.ndarray, ref_mean: np.ndarray,
               n: int) -> dict[str, dict[str, float]]:
    out = {}
    for lo, hi in BANDS:
        mask = (k >= lo) & (k < hi) & np.isfinite(mean) & np.isfinite(ref_mean)
        ref = float(np.mean(ref_mean[mask]))
        out[f"{lo:.2f}-{hi:.2f}"] = {
            "ratio": float(np.mean(mean[mask]) / ref),
            "err": float(np.mean(std[mask]) / np.sqrt(n) / ref),
        }
    mask = (k < LOWK_MAX) & np.isfinite(mean) & np.isfinite(ref_mean)
    ref = float(np.mean(ref_mean[mask]))
    out["k<0.01"] = {
        "ratio": float(np.mean(mean[mask]) / ref),
        "err": float(np.mean(std[mask]) / np.sqrt(n) / ref),
    }
    return out


def quad_band_ratios(k: np.ndarray, pk0: np.ndarray, pk2: np.ndarray,
                     ref_pk0: np.ndarray, ref_pk2: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for lo, hi in BANDS:
        mask = (k >= lo) & (k < hi)
        q_ez = np.nanmean(pk2[:, mask], axis=1) / np.nanmean(pk0[:, mask], axis=1)
        q_ref = np.nanmean(ref_pk2[:, mask], axis=1) / np.nanmean(ref_pk0[:, mask], axis=1)
        ref = float(np.nanmean(q_ref))
        err = float(np.nanstd(q_ez, ddof=1) / np.sqrt(q_ez.size) / ref) if q_ez.size > 1 else float("nan")
        out[f"{lo:.2f}-{hi:.2f}"] = {"ratio": float(np.nanmean(q_ez) / ref), "err": err,
                                     "ezmock_quad_to_mono": float(np.nanmean(q_ez)),
                                     "abacus_quad_to_mono": ref}
    return out


def main() -> None:
    args = parse_args()
    z_tokens = [part.strip() for part in args.z_values.split(",") if part.strip()]
    z_values = [float(token) for token in z_tokens]
    abacus = load_abacus_pk(args.abacus_dir)
    abacus_mean0 = np.mean(abacus["pk0"], axis=0)
    abacus_mean2 = np.mean(abacus["pk2"], axis=0)

    report: dict[str, object] = {
        "task": "task43_compare_ezmock_rsd_zscan_vs_abacus",
        "status": "done",
        "abacus_dir": str(args.abacus_dir), "n_abacus": abacus["n"],
        "z_eff_lightcone": Z_EFF,
        "points": {},
    }
    curves = {}
    for token, z in zip(z_tokens, z_values):
        root = PROJECT_ROOT / args.root_template.format(z=token)
        pilot = load_ezmock_pk(root, args.pk_limit)
        pk0 = interpolate_to(pilot["pk0"], pilot["k"], abacus["k"])
        pk2 = interpolate_to(pilot["pk2"], pilot["k"], abacus["k"])
        mean0, std0 = mean_std_rows(pk0)
        mean2, std2 = mean_std_rows(pk2)
        curves[token] = {"mean0": mean0, "std0": std0, "mean2": mean2, "std2": std2}
        point = {
            "root": str(root.relative_to(PROJECT_ROOT)),
            "n_real": pilot["n"], "seeds": pilot["seeds"],
            "ndata_mean": float(np.mean(pilot["ndata"])),
            "ndata_offset": float(np.mean(pilot["ndata"]) / TARGET_NDATA - 1.0),
            "bands": {
                "P0": band_ratio(abacus["k"], mean0, std0, abacus_mean0, pilot["n"]),
                "P2": band_ratio(abacus["k"], mean2, std2, abacus_mean2, pilot["n"]),
            },
            "quad_to_mono": quad_band_ratios(abacus["k"], pk0, pk2, abacus["pk0"], abacus["pk2"]),
        }
        if not args.no_nz:
            random_Z = np.asarray(np.load(COMMON_RANDOM, allow_pickle=False)["Z"], dtype="f8")
            point["nz_max_abs_dev"] = nz_ratio(load_lightcone_Z(root), random_Z)["max_abs_dev"]
        report["points"][token] = point
        quad1 = point["quad_to_mono"]["0.01-0.03"]
        print(f"[z={token}] n={pilot['n']} ndata={point['ndata_mean']:.0f} "
              f"P0 band={[round(point['bands']['P0'][b]['ratio'], 4) for b in ('0.01-0.03','0.03-0.05','0.05-0.08')]} "
              f"P0 lowk={point['bands']['P0']['k<0.01']['ratio']:.4f} "
              f"quad band1={quad1['ratio']:.4f} "
              f"nzdev={point.get('nz_max_abs_dev', float('nan')):.4f}", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)
    ax_p0, ax_p2, ax_q, ax_a = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    legend_handles = []
    for index, token in enumerate(z_tokens):
        color = COLORS.get(float(token), plt.cm.tab10(index % 10))
        curve = curves[token]
        label = f"EZmock z={token} (x{report['points'][token]['n_real']})"
        for ax, mean, std in ((ax_p0, curve["mean0"], curve["std0"]), (ax_p2, curve["mean2"], curve["std2"])):
            valid = np.isfinite(mean) & (mean > 0.0)
            line, = ax.plot(abacus["k"][valid], mean[valid], color=color, lw=2.0, ls="--", label=label)
            if ax is ax_p0:
                legend_handles.append(line)
            if np.all(np.isfinite(std)):
                ax.fill_between(abacus["k"][valid], (mean - std)[valid], (mean + std)[valid],
                                color=color, alpha=0.12, lw=0)
    for ax, ell, ref, ymin in ((ax_p0, 0, abacus_mean0, None), (ax_p2, 2, abacus_mean2, P2_YMIN)):
        valid = np.isfinite(ref) & (ref > 0.0)
        ref_line, = ax.plot(abacus["k"][valid], ref[valid], color="#222222", lw=2.8,
                            label="Abacus lightcone x25 mean")
        tops = [float(np.max(ref[valid]))]
        for token in z_tokens:
            series = curves[token][f"mean{ell}"]
            mask = np.isfinite(series) & (series > 0.0)
            if np.any(mask):
                tops.append(float(np.max(series[mask])))
        ax.set(xscale="log", yscale="log", xlabel=r"$k\ [h\,Mpc^{-1}]$", ylabel=rf"$P_{ell}(k)$",
               title=rf"RSD lightcone $P_{ell}$ (z-scan, pdf_base=0.20)",
               ylim=(ymin if ymin is not None else None, 1.15 * max(tops)))
    ax_p0.legend(handles=[ref_line] + legend_handles, frameon=False, loc="lower left", fontsize=9)

    for band_index, (lo, hi) in enumerate(BANDS):
        label = f"{lo:.2f}-{hi:.2f}"
        xs = np.asarray(z_values)
        ys = np.asarray([report["points"][token]["quad_to_mono"][label]["ratio"] for token in z_tokens])
        es = np.asarray([report["points"][token]["quad_to_mono"][label]["err"] for token in z_tokens])
        ax_q.errorbar(xs, ys, yerr=es, marker=MARKERS[band_index], ms=6, lw=1.8, capsize=3,
                      color=plt.cm.tab10(band_index), label=rf"$k\in[{lo:.2f},{hi:.2f}]$")
        ys0 = np.asarray([report["points"][token]["bands"]["P0"][label]["ratio"] for token in z_tokens])
        es0 = np.asarray([report["points"][token]["bands"]["P0"][label]["err"] for token in z_tokens])
        ax_a.errorbar(xs, ys0, yerr=es0, marker=MARKERS[band_index], ms=6, lw=1.8, capsize=3,
                      color=plt.cm.tab10(band_index), label=rf"$k\in[{lo:.2f},{hi:.2f}]$")
    ys_low = np.asarray([report["points"][token]["bands"]["P0"]["k<0.01"]["ratio"] for token in z_tokens])
    es_low = np.asarray([report["points"][token]["bands"]["P0"]["k<0.01"]["err"] for token in z_tokens])
    ax_a.errorbar(np.asarray(z_values), ys_low, yerr=es_low, marker="d", ms=5, lw=1.6, ls=":", capsize=3,
                  color="#777777", label=r"$k<0.01$ (low-k)")
    for ax, ylabel, title in ((ax_q, r"$(P_2/P_0)_{\rm EZ} \,/\, (P_2/P_0)_{\rm Ab}$",
                               "Quad-to-mono band ratio vs z (quad strength)"),
                              (ax_a, r"$P_0$ band ratio (EZ/Abacus)",
                               "P0 band ratio vs z (amplitude response)")):
        ax.axhline(1.0, color="#888888", lw=1.0, ls=":")
        ax.axvline(Z_EFF, color="#bbbbbb", lw=1.2, ls="--")
        ax.set(xlabel="EZmock snapshot redshift $z$", ylabel=ylabel, title=title)
        ax.legend(frameon=False, fontsize=9, loc="best")
    ax_q.text(Z_EFF, 0.03, rf" $z_{{\rm eff}}={Z_EFF:.3f}$", transform=ax_q.get_xaxis_transform(),
              fontsize=8, color="#888888", va="bottom", ha="left")

    fig.suptitle("EZmock RSD lightcone z-scan vs Abacus lightcone x25 (c000, mmin=1.4e13, 0.4<z_obs<0.8)\n"
                 "pdf_base=0.20, rho_c=1.14, rho_exp=5, sigma_v=0, fixampT, n(z)-matched, "
                 "BOX_SIZE=2000/320^3, mesh256; x5 per z (seeds 610001-610005)", fontsize=11)
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {args.out_pdf}\n[done] {args.out_json}")


if __name__ == "__main__":
    main()
