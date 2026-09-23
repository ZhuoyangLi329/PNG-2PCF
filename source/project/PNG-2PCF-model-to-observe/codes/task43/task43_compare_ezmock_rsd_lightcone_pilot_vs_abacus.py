#!/usr/bin/env python3
"""Compare the fixampT n(z)-matched EZmock RSD lightcone pilot (x10) with Abacus lightcone (x25).

Outputs a band-ratio JSON (pilot root diagnostics/) and the 2x2 P0/P2 + xi0/xi2
comparison PDF (test_figure/), styled after the raw-box manual-tuning figure.
An extra P-only 1x2 PDF is produced only when --out-pdf is given.
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
PILOT_ROOT = PROJECT_ROOT / "outputs/task43_outputs/ezmock_rsd_lightcone_x10_fixampT_nzmatch_b0.35_pilot"
OLD_ROOT = PROJECT_ROOT / "outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50"
ABACUS_DIR = PROJECT_ROOT / "outputs/task43_outputs/rsd_validation/lightcone/pk/boxsafe_zobs0p4_0p8_p02_x25_fkpP010000"
ABACUS_GLOB = "task43_rsd_lightcone_p02_ph0*_mesh256_kmax0p300_dk0p002.npz"
ABACUS_XI = PROJECT_ROOT / "outputs/task43_outputs/rsd_validation/lightcone_boxsafe_zobs0p4_0p8/ell2_increment/task43_rsd_boxsafe_x25_mean_xi02_s30_350_ds10.npz"
EZMOCK_GLOB = "*_p02_mesh256.npz"
EZMOCK_XI_GLOB = "*_xi02_s30_350_ds10.npz"
OLD_XI_GLOB = "*_seed61*_xi02_s30_350_ds10.npz"
OUT_JSON = PILOT_ROOT / "diagnostics/task43_ezmock_rsd_lightcone_fixampT_nzmatch_b0.35_pilot_x10_vs_abacus_x25.json"
OUT_PDF_PKXI = PROJECT_ROOT / "test_figure/task43_ezmock_rsd_lightcone_fixampT_nzmatch_c1.14_e5_b0.35_v0_x10_vs_abacus_x25_pkxi_preview.pdf"
BANDS = ((0.01, 0.03), (0.03, 0.05), (0.05, 0.08))
LOWK_MAX = 0.01
P2_YMIN = 1.0e3
XI_BANDS = ((30.0, 80.0), (80.0, 120.0), (120.0, 350.0))
XI_LARGE_S_MIN = 150.0
XI_DELTA_S_MIN = 50.0
NZ_RANGE = (0.4, 0.8)
NZ_BINS = 40
TARGET_NDATA = 591_308


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, default=PILOT_ROOT)
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--abacus-dir", type=Path, default=ABACUS_DIR)
    parser.add_argument("--abacus-xi", type=Path, default=ABACUS_XI)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-pdf", type=Path, default=None,
                        help="Optional extra P-only 1x2 comparison PDF; skipped when omitted.")
    parser.add_argument("--out-pdf-pkxi", type=Path, default=OUT_PDF_PKXI)
    parser.add_argument("--pdf-base", type=float, default=0.35)
    parser.add_argument("--z-box", type=float, default=None,
                        help="EZmock snapshot redshift; appended to the suptitle when given.")
    parser.add_argument("--pk-limit", type=int, default=None,
                        help="Use only the first N pk realizations (lowest seeds); default all.")
    parser.add_argument("--label", default="fixampT n(z)-matched pilot",
                        help="Mock description used in the suptitle.")
    parser.add_argument("--legend-tag", default="pilot",
                        help="Short tag used in the curve legend.")
    return parser.parse_args()


def load_ezmock_pk(root: Path, limit: int | None = None) -> dict[str, np.ndarray]:
    paths = sorted(root.glob("pk_jaxpower/" + EZMOCK_GLOB))
    if not paths:
        raise FileNotFoundError(f"no EZmock P02 files under {root}/pk_jaxpower")
    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as src:
            records.append({
                "seed": int(src["seed"]), "k": np.asarray(src["k0"], dtype="f8"),
                "pk0": np.asarray(src["pk0"], dtype="f8"), "pk2": np.asarray(src["pk2"], dtype="f8"),
                "ndata": int(src["ndata"]), "path": str(path),
            })
    records.sort(key=lambda item: item["seed"])
    if limit is not None:
        records = records[:int(limit)]
    k = records[0]["k"]
    if not all(np.allclose(item["k"], k) for item in records):
        raise RuntimeError("EZmock k grids differ between realizations")
    return {
        "n": len(records), "k": k, "seeds": [item["seed"] for item in records],
        "pk0": np.asarray([item["pk0"] for item in records], dtype="f8"),
        "pk2": np.asarray([item["pk2"] for item in records], dtype="f8"),
        "ndata": np.asarray([item["ndata"] for item in records], dtype="f8"),
        "paths": [item["path"] for item in records],
    }


def load_abacus_pk(directory: Path) -> dict[str, np.ndarray]:
    paths = sorted(directory.glob(ABACUS_GLOB))
    if not paths:
        raise FileNotFoundError(f"no Abacus P02 files under {directory}")
    k, pk0, pk2 = None, [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as src:
            kk = np.asarray(src["k_obs"], dtype="f8")
            if k is None:
                k = kk
            # Per-phase k centers scatter by ~1e-4 relative from mode weighting.
            elif not np.allclose(kk, k, rtol=1e-3, atol=0.0, equal_nan=True):
                raise RuntimeError("Abacus k grids differ between phases")
            pk0.append(np.asarray(src["pk0"], dtype="f8"))
            pk2.append(np.asarray(src["pk2"], dtype="f8"))
    return {"n": len(paths), "k": k,
            "pk0": np.asarray(pk0, dtype="f8"), "pk2": np.asarray(pk2, dtype="f8")}


def load_ezmock_xi(root: Path, pattern: str = EZMOCK_XI_GLOB) -> dict[str, np.ndarray]:
    paths = sorted(root.glob("xi_fcfc/" + pattern))
    if not paths:
        raise FileNotFoundError(f"no EZmock xi files under {root}/xi_fcfc")
    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as src:
            records.append({
                "seed": int(src["seed"]), "s": np.asarray(src["s"], dtype="f8"),
                "xi0": np.asarray(src["xi0"], dtype="f8"), "xi2": np.asarray(src["xi2"], dtype="f8"),
            })
    records.sort(key=lambda item: item["seed"])
    s = records[0]["s"]
    if not all(np.allclose(item["s"], s, equal_nan=True) for item in records):
        raise RuntimeError("EZmock s grids differ between realizations")
    return {
        "n": len(records), "s": s, "seeds": [item["seed"] for item in records],
        "xi0": np.asarray([item["xi0"] for item in records], dtype="f8"),
        "xi2": np.asarray([item["xi2"] for item in records], dtype="f8"),
    }


def load_abacus_xi(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as src:
        s = np.asarray(src["s"], dtype="f8")
        ells = np.asarray(src["ells"], dtype="i4")
        by_phase = np.asarray(src["xi_multipoles_by_phase"], dtype="f8")
    if list(ells) != [0, 2] or by_phase.ndim != 3 or by_phase.shape[1] != 2:
        raise RuntimeError(f"unexpected Abacus xi reference layout: ells={ells} shape={by_phase.shape}")
    return {"n": by_phase.shape[0], "s": s, "xi0": by_phase[:, 0, :], "xi2": by_phase[:, 1, :]}


def mean_std_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and ddof=1 scatter; scatter is NaN (band skipped) for a single realization."""
    mean = np.nanmean(rows, axis=0)
    if rows.shape[0] < 2:
        return mean, np.full(mean.shape, np.nan)
    return mean, np.nanstd(rows, axis=0, ddof=1)


def interpolate_to(rows: np.ndarray, k_source: np.ndarray, k_target: np.ndarray) -> np.ndarray:
    if np.allclose(k_source, k_target, equal_nan=True):
        return rows
    out = np.full((rows.shape[0], k_target.size), np.nan, dtype="f8")
    for index, row in enumerate(rows):
        mask = np.isfinite(k_source) & np.isfinite(row)
        out[index] = np.interp(k_target, k_source[mask], row[mask], left=np.nan, right=np.nan)
    return out


def band_summary(k: np.ndarray, ezmock_mean: np.ndarray, ezmock_std: np.ndarray,
                 abacus_mean: np.ndarray) -> dict[str, dict[str, float]]:
    summary = {}
    for lo, hi in BANDS:
        mask = (k >= lo) & (k < hi) & np.isfinite(ezmock_mean) & np.isfinite(abacus_mean)
        label = f"{lo:.2f}-{hi:.2f}"
        ezmock_value = float(np.mean(ezmock_mean[mask]))
        abacus_value = float(np.mean(abacus_mean[mask]))
        summary[label] = {
            "ezmock_mean": ezmock_value, "abacus_mean": abacus_value,
            "ratio_mean": ezmock_value / abacus_value,
            "ezmock_std": float(np.mean(ezmock_std[mask])),
        }
    return summary


def lowk_summary(k: np.ndarray, ezmock_mean: np.ndarray, abacus_mean: np.ndarray) -> dict[str, float]:
    mask = (k < LOWK_MAX) & np.isfinite(ezmock_mean) & np.isfinite(abacus_mean)
    return {
        "n_bins": int(np.count_nonzero(mask)),
        "ezmock_mean": float(np.mean(ezmock_mean[mask])),
        "abacus_mean": float(np.mean(abacus_mean[mask])),
        "ezmock_max": float(np.max(ezmock_mean[mask])),
        "abacus_max": float(np.max(abacus_mean[mask])),
    }


def xi_summary(s: np.ndarray, ezmock_mean: np.ndarray, ezmock_std: np.ndarray,
               abacus_mean: np.ndarray) -> dict[str, object]:
    finite = np.isfinite(ezmock_mean) & np.isfinite(abacus_mean)
    bands = {}
    for lo, hi in XI_BANDS:
        mask = (s >= lo) & (s < hi) & finite
        label = f"{lo:.0f}-{hi:.0f}"
        ezmock_value = float(np.mean(ezmock_mean[mask]))
        abacus_value = float(np.mean(abacus_mean[mask]))
        bands[label] = {
            "ezmock_mean": ezmock_value, "abacus_mean": abacus_value,
            "delta": ezmock_value - abacus_value,
            "ezmock_std": float(np.mean(ezmock_std[mask])),
        }
    large = (s >= XI_LARGE_S_MIN) & finite
    wide = (s >= XI_DELTA_S_MIN) & finite
    return {
        "bands": bands,
        "large_s_ge_150": {
            "ezmock_mean": float(np.mean(ezmock_mean[large])),
            "abacus_mean": float(np.mean(abacus_mean[large])),
            "delta": float(np.mean(ezmock_mean[large]) - np.mean(abacus_mean[large])),
        },
        "max_abs_delta_s_ge_50": float(np.max(np.abs(ezmock_mean[wide] - abacus_mean[wide]))),
    }


def nz_ratio(data_Z: np.ndarray, random_Z: np.ndarray) -> dict[str, object]:
    edges = np.linspace(NZ_RANGE[0], NZ_RANGE[1], NZ_BINS + 1)
    data_counts, _ = np.histogram(data_Z, bins=edges)
    rand_counts, _ = np.histogram(random_Z, bins=edges)
    keep = rand_counts > 0
    data_shape = data_counts[keep] / data_counts[keep].sum()
    rand_shape = rand_counts[keep] / rand_counts[keep].sum()
    ratio = data_shape / rand_shape
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "bin_centers": centers[keep].tolist(),
        "ratio": ratio.tolist(),
        "max_abs_dev": float(np.max(np.abs(ratio - 1.0))),
        "n_data": int(data_Z.size), "n_random": int(random_Z.size),
    }


def load_lightcone_Z(root: Path) -> np.ndarray:
    paths = sorted(root.glob("lightcone_catalogs/ezmock_m*_zobs0p4_0p8.npz"))
    if not paths:
        raise FileNotFoundError(f"no lightcone catalogs under {root}")
    return np.concatenate([np.asarray(np.load(path, allow_pickle=False)["Z"], dtype="f8") for path in paths])


def plot_pk_panel(ax, ell: int, k: np.ndarray, pilot_mean: np.ndarray, pilot_std: np.ndarray,
                  abacus_mean: np.ndarray, pilot_label: str, ymin: float | None = None) -> None:
    visible = np.concatenate([abacus_mean[np.isfinite(abacus_mean)], pilot_mean[np.isfinite(pilot_mean)]])
    # Fixed positive floor (ymin): non-positive bins (noisy low-k quadrupole modes)
    # are dropped from the line instead of dragging a symlog range.
    log_positive = ymin is not None or np.all(visible > 0.0)
    valid_a = np.isfinite(abacus_mean)
    valid_e = np.isfinite(pilot_mean)
    if log_positive:
        valid_a &= abacus_mean > 0.0
        valid_e &= pilot_mean > 0.0
    ax.plot(k[valid_a], abacus_mean[valid_a], color="#222222", lw=2.5, label="Abacus lightcone x25 mean")
    ax.plot(k[valid_e], pilot_mean[valid_e], color="#d62728", lw=2.5, ls="--", label=pilot_label)
    if np.all(np.isfinite(pilot_std)):
        ax.fill_between(k[valid_e], (pilot_mean - pilot_std)[valid_e], (pilot_mean + pilot_std)[valid_e],
                        color="#d62728", alpha=0.15, lw=0)
    if ymin is not None:
        ax.set(xscale="log", yscale="log", ylim=(ymin, 1.15 * visible.max()))
    elif log_positive:
        ax.set(xscale="log", yscale="log", ylim=(0.85 * visible.min(), 1.15 * visible.max()))
    else:
        pad = 0.15 * (visible.max() - visible.min())
        ax.set(xscale="log", yscale="symlog", ylim=(visible.min() - pad, visible.max() + pad))
    ax.set(xlabel=r"$k\ [h\,Mpc^{-1}]$", ylabel=rf"$P_{ell}(k)$",
           title=rf"RSD lightcone $P_{ell}$")
    ax.legend(frameon=False, loc="lower left")


def plot_xi_panel(ax, ell: int, s: np.ndarray, pilot_mean: np.ndarray, pilot_std: np.ndarray,
                  abacus_mean: np.ndarray, pilot_label: str) -> None:
    scale = s ** 2
    ax.axhline(0.0, color="#888888", lw=1.0, ls=":")
    ax.plot(s, scale * abacus_mean, color="#222222", lw=2.5, label="Abacus lightcone x25 mean")
    ax.plot(s, scale * pilot_mean, color="#d62728", lw=2.5, ls="--", label=pilot_label)
    if np.all(np.isfinite(pilot_std)):
        ax.fill_between(s, scale * (pilot_mean - pilot_std), scale * (pilot_mean + pilot_std),
                        color="#d62728", alpha=0.15, lw=0)
    ax.set(xlabel=r"$s\ [h^{-1}Mpc]$", ylabel=rf"$s^2\xi_{ell}(s)\ [h^{{-2}}Mpc^2]$",
           title=rf"RSD lightcone $s^2\xi_{ell}$")
    ax.legend(frameon=False, loc="best")


def main() -> None:
    args = parse_args()
    pilot = load_ezmock_pk(args.pilot_root, args.pk_limit)
    abacus = load_abacus_pk(args.abacus_dir)
    pilot_pk0 = interpolate_to(pilot["pk0"], pilot["k"], abacus["k"])
    pilot_pk2 = interpolate_to(pilot["pk2"], pilot["k"], abacus["k"])
    pilot_mean0, pilot_std0 = mean_std_rows(pilot_pk0)
    pilot_mean2, pilot_std2 = mean_std_rows(pilot_pk2)
    abacus_mean0, abacus_mean2 = np.mean(abacus["pk0"], axis=0), np.mean(abacus["pk2"], axis=0)

    abacus_xi = load_abacus_xi(args.abacus_xi)
    abacus_xi_mean0, abacus_xi_mean2 = np.mean(abacus_xi["xi0"], axis=0), np.mean(abacus_xi["xi2"], axis=0)
    has_pilot_xi = bool(list(args.pilot_root.glob("xi_fcfc/" + EZMOCK_XI_GLOB)))
    pilot_xi = None
    pilot_xi_mean0 = pilot_xi_mean2 = pilot_xi_std0 = pilot_xi_std2 = None
    if has_pilot_xi:
        pilot_xi = load_ezmock_xi(args.pilot_root)
        if not np.allclose(pilot_xi["s"], abacus_xi["s"], equal_nan=True):
            raise RuntimeError("EZmock and Abacus s grids differ")
        pilot_xi_mean0, pilot_xi_std0 = mean_std_rows(pilot_xi["xi0"])
        pilot_xi_mean2, pilot_xi_std2 = mean_std_rows(pilot_xi["xi2"])
    else:
        print(f"[warn] no xi products under {args.pilot_root}/xi_fcfc; producing P-only output")

    report = {
        "task": "task43_compare_ezmock_rsd_lightcone_pilot_vs_abacus",
        "status": "done",
        "pilot_root": str(args.pilot_root),
        "abacus_dir": str(args.abacus_dir),
        "n_ezmock": pilot["n"], "n_abacus": abacus["n"],
        "ezmock_seeds": pilot["seeds"],
        "k_grid_match": bool(np.allclose(pilot["k"], abacus["k"], rtol=1e-3, atol=0.0, equal_nan=True)),
        "bands": {
            "P0": band_summary(abacus["k"], pilot_mean0, pilot_std0, abacus_mean0),
            "P2": band_summary(abacus["k"], pilot_mean2, pilot_std2, abacus_mean2),
        },
        "lowk_below_0p01": {
            "P0": lowk_summary(abacus["k"], pilot_mean0, abacus_mean0),
            "P2": lowk_summary(abacus["k"], pilot_mean2, abacus_mean2),
        },
        "ndata": {
            "mean": float(np.mean(pilot["ndata"])), "std": float(np.std(pilot["ndata"], ddof=1)),
            "target": TARGET_NDATA, "fractional_offset": float(np.mean(pilot["ndata"]) / TARGET_NDATA - 1.0),
        },
    }

    if pilot_xi is not None:
        report["xi"] = {
            "abacus_ref": str(args.abacus_xi),
            "n_ezmock": pilot_xi["n"], "n_abacus": abacus_xi["n"],
            "ezmock_seeds": pilot_xi["seeds"],
            "s_grid_match": True,
            "bands": {
                "xi0": xi_summary(abacus_xi["s"], pilot_xi_mean0, pilot_xi_std0, abacus_xi_mean0),
                "xi2": xi_summary(abacus_xi["s"], pilot_xi_mean2, pilot_xi_std2, abacus_xi_mean2),
            },
        }

    if args.old_root.is_dir() and list(args.old_root.glob("pk_jaxpower/" + EZMOCK_GLOB)):
        old = load_ezmock_pk(args.old_root)
        old_pk0 = interpolate_to(old["pk0"], old["k"], abacus["k"])
        old_pk2 = interpolate_to(old["pk2"], old["k"], abacus["k"])
        report["before_fixampF"] = {
            "root": str(args.old_root),
            "bands": {
                "P0": band_summary(abacus["k"], np.nanmean(old_pk0, axis=0), np.nanstd(old_pk0, axis=0, ddof=1), abacus_mean0),
                "P2": band_summary(abacus["k"], np.nanmean(old_pk2, axis=0), np.nanstd(old_pk2, axis=0, ddof=1), abacus_mean2),
            },
            "lowk_below_0p01": {
                "P0": lowk_summary(abacus["k"], np.nanmean(old_pk0, axis=0), abacus_mean0),
                "P2": lowk_summary(abacus["k"], np.nanmean(old_pk2, axis=0), abacus_mean2),
            },
            "ndata": {"mean": float(np.mean(old["ndata"]))},
        }
        old_xi = load_ezmock_xi(args.old_root, OLD_XI_GLOB)
        report["before_fixampF"]["xi"] = {
            "n_ezmock": old_xi["n"], "seeds": old_xi["seeds"],
            "bands": {
                "xi0": xi_summary(abacus_xi["s"], np.mean(old_xi["xi0"], axis=0),
                                  np.std(old_xi["xi0"], axis=0, ddof=1), abacus_xi_mean0),
                "xi2": xi_summary(abacus_xi["s"], np.mean(old_xi["xi2"], axis=0),
                                  np.std(old_xi["xi2"], axis=0, ddof=1), abacus_xi_mean2),
            },
        }
        old_Z = load_lightcone_Z(args.old_root)
    else:
        old_Z = None

    random_Z = np.asarray(np.load(OLD_ROOT / "common_random/common_random_zobs0p4_0p8_x50.npz", allow_pickle=False)["Z"], dtype="f8")
    nz_pilot = nz_ratio(load_lightcone_Z(args.pilot_root), random_Z)
    report["nz_vs_common_random"] = {"pilot_max_abs_dev": nz_pilot["max_abs_dev"], "pilot_bins": nz_pilot["bin_centers"], "pilot_ratio": nz_pilot["ratio"]}
    if old_Z is not None:
        report["nz_vs_common_random"]["before_max_abs_dev"] = nz_ratio(old_Z, random_Z)["max_abs_dev"]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    pilot_label_pk = f"EZmock {args.legend_tag} x{pilot['n']} mean"
    suptitle = ("Abacus lightcone vs EZmock " + args.label + ": rho_c=1.14, "
                f"rho_exp=5, pdf_base={args.pdf_base:g}, sigma_v=0"
                + (f", z_box={args.z_box:g}" if args.z_box is not None else ""))
    pk_panels = (
        (0, abacus["k"], pilot_mean0, pilot_std0, abacus_mean0, None),
        (2, abacus["k"], pilot_mean2, pilot_std2, abacus_mean2, P2_YMIN),
    )
    if pilot_xi is not None:
        pilot_label_xi = f"EZmock {args.legend_tag} x{pilot_xi['n']} mean"
        xi_panels = (
            (0, abacus_xi["s"], pilot_xi_mean0, pilot_xi_std0, abacus_xi_mean0),
            (2, abacus_xi["s"], pilot_xi_mean2, pilot_xi_std2, abacus_xi_mean2),
        )
    else:
        xi_panels = None
    if args.out_pdf is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
        for ax, (ell, x, pmean, pstd, amean, pmin) in zip(axes, pk_panels):
            plot_pk_panel(ax, ell, x, pmean, pstd, amean, pilot_label_pk, ymin=pmin)
        fig.suptitle(suptitle)
        args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_pdf)
        plt.close(fig)

    if xi_panels is not None:
        fig_pkxi, axes_pkxi = plt.subplots(2, 2, figsize=(14, 10.6), constrained_layout=True)
        for ax, (ell, x, pmean, pstd, amean, pmin) in zip(axes_pkxi[0], pk_panels):
            plot_pk_panel(ax, ell, x, pmean, pstd, amean, pilot_label_pk, ymin=pmin)
        for ax, (ell, x, pmean, pstd, amean) in zip(axes_pkxi[1], xi_panels):
            plot_xi_panel(ax, ell, x, pmean, pstd, amean, pilot_label_xi)
        fig_pkxi.suptitle(suptitle)
        args.out_pdf_pkxi.parent.mkdir(parents=True, exist_ok=True)
        fig_pkxi.savefig(args.out_pdf_pkxi)
        plt.close(fig_pkxi)
    print(f"[done] json: {args.out_json}")
    if args.out_pdf is not None:
        print(f"[done] pdf:  {args.out_pdf}")
    if xi_panels is not None:
        print(f"[done] pdf:  {args.out_pdf_pkxi}")
    for name, values in report["bands"].items():
        ratios = ", ".join(f"{band}: {item['ratio_mean']:.3f}" for band, item in values.items())
        print(f"  pilot {name} band ratios -> {ratios}")
    if "xi" in report:
        for name, values in report["xi"]["bands"].items():
            deltas = ", ".join(f"{band}: {item['delta']:+.5f}" for band, item in values["bands"].items())
            print(f"  pilot {name} delta -> {deltas}")
            print(f"    {name} large_s>=150 delta = {values['large_s_ge_150']['delta']:+.5f}, "
                  f"max|delta| (s>=50) = {values['max_abs_delta_s_ge_50']:.5f}")
    if "before_fixampF" in report:
        for name, values in report["before_fixampF"]["bands"].items():
            ratios = ", ".join(f"{band}: {item['ratio_mean']:.3f}" for band, item in values.items())
            print(f"  before {name} band ratios -> {ratios}")
        for name, values in report["before_fixampF"]["xi"]["bands"].items():
            deltas = ", ".join(f"{band}: {item['delta']:+.5f}" for band, item in values["bands"].items())
            print(f"  before x10 {name} delta -> {deltas}")
            print(f"    {name} large_s>=150 delta = {values['large_s_ge_150']['delta']:+.5f}")
    print(f"  ndata mean = {report['ndata']['mean']:.0f} ({100.0 * report['ndata']['fractional_offset']:+.2f}% vs target)")
    print(f"  n(z) max |data/random-1| = {report['nz_vs_common_random']['pilot_max_abs_dev']:.4f}")


if __name__ == "__main__":
    main()
