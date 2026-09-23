#!/usr/bin/env python3
"""Density control probe: EZmock fixampT rawbox P0/P2 at pilot density vs tuned density vs Abacus x25.

Tests whether the EZmock rawbox clustering amplitude depends on NUM_TRACER
(1,847,000 = pilot pre-thinning box density vs 1,297,050 = tuning density,
Abacus rawbox 1,295,781), to explain the residual ~10% P0 deficit of the
n(z)-matched lightcone pilot.  Reports total and shot-noise-subtracted band
ratios, plus same-seed paired probe/x15 ratios that cancel field phase noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TUNING_ROOT = PROJECT_ROOT / "outputs/task43_outputs/ezmock_rsd_manual_tuning"
X15_MEAS = TUNING_ROOT / "notebook_rsd_c1.14_e5_b0.25_v0_x15/measurements"
PROBE_MEAS = TUNING_ROOT / "probe_ntracer1p847M_c1.14_e5_b0.25_v0_x3/measurements"
ABACUS_DIR = PROJECT_ROOT / "outputs/task43_outputs/rsd_validation/rawbox/pk"
ABACUS_GLOB = "task43_rsd_rawbox_p02_AbacusSummit_base_c000_ph*_mmin1p4e13_mesh400.npz"
EZK_GLOB = "pk02_seed{seed}_mesh400.npz"
PAIRED_SEEDS = (433001, 433002, 433003)
BANDS = ((0.01, 0.03), (0.03, 0.05), (0.05, 0.08))
LOWK_MAX = 0.01
BOX_SIZE = 2000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x15-meas", type=Path, default=X15_MEAS)
    parser.add_argument("--probe-meas", type=Path, default=PROBE_MEAS)
    parser.add_argument("--abacus-dir", type=Path, default=ABACUS_DIR)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def load_ezmock(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as src:
        return {
            "path": str(path),
            "seed": int(src["seed"]),
            "ndata": float(src["ndata"]),
            "k": np.asarray(src["k0"], dtype="f8"),
            "pk0": np.asarray(src["pk0"], dtype="f8"),
            "pk2": np.asarray(src["pk2"], dtype="f8"),
            "shot0": np.asarray(src["shotnoise0"], dtype="f8"),
            "shot2": np.asarray(src["shotnoise2"], dtype="f8"),
        }


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
        "shot0": np.asarray(shot0), "shot2": np.asarray(shot2),
    }


def interp_to(rows: np.ndarray, k_source: np.ndarray, k_target: np.ndarray) -> np.ndarray:
    if np.allclose(k_source, k_target, equal_nan=True):
        return rows
    out = np.full((rows.shape[0], k_target.size), np.nan)
    for index, row in enumerate(rows):
        mask = np.isfinite(k_source) & np.isfinite(row)
        out[index] = np.interp(k_target, k_source[mask], row[mask], left=np.nan, right=np.nan)
    return out


def band_dict(k: np.ndarray, values: np.ndarray) -> dict[str, float]:
    out = {}
    for lo, hi in BANDS:
        mask = (k >= lo) & (k < hi) & np.isfinite(values)
        out[f"{lo:.2f}-{hi:.2f}"] = float(np.mean(values[mask]))
    mask = (k < LOWK_MAX) & np.isfinite(values)
    out["lowk<0.01"] = float(np.mean(values[mask]))
    return out


def ratio_dict(k: np.ndarray, num: np.ndarray, den: np.ndarray) -> dict[str, float]:
    out = {}
    for lo, hi in BANDS:
        mask = (k >= lo) & (k < hi) & np.isfinite(num) & np.isfinite(den)
        out[f"{lo:.2f}-{hi:.2f}"] = float(np.mean(num[mask]) / np.mean(den[mask]))
    mask = (k < LOWK_MAX) & np.isfinite(num) & np.isfinite(den)
    out["lowk<0.01"] = float(np.mean(num[mask]) / np.mean(den[mask]))
    return out


def fmt(d: dict[str, float]) -> str:
    return ", ".join(f"{band}: {value:.3f}" for band, value in d.items())


def main() -> None:
    args = parse_args()
    out_json = args.out_json or (args.probe_meas.parent / "diagnostics/task43_ezmock_rsd_rawbox_density_probe.json")
    abacus = load_abacus(args.abacus_dir)
    k = abacus["k"]

    x15_paths = sorted(args.x15_meas.glob(EZK_GLOB.format(seed="*")))
    probe_paths = sorted(args.probe_meas.glob(EZK_GLOB.format(seed="*")))
    if not x15_paths or not probe_paths:
        raise FileNotFoundError((args.x15_meas, args.probe_meas))
    x15 = [load_ezmock(path) for path in x15_paths]
    probe = [load_ezmock(path) for path in probe_paths]
    x15_paired = [item for item in x15 if item["seed"] in PAIRED_SEEDS]
    probe_by_seed = {item["seed"]: item for item in probe}

    def stack(items: list[dict], key: str) -> np.ndarray:
        return np.asarray([item[key] for item in items])

    ab_pk0, ab_pk2 = np.mean(abacus["pk0"], axis=0), np.mean(abacus["pk2"], axis=0)
    ab_shot0, ab_shot2 = float(np.mean(abacus["shot0"])), float(np.mean(abacus["shot2"]))
    x15_pk0 = np.nanmean(stack(x15, "pk0"), axis=0)
    x15_pk2 = np.nanmean(stack(x15, "pk2"), axis=0)
    pr_pk0 = np.nanmean(stack(probe, "pk0"), axis=0)
    pr_pk2 = np.nanmean(stack(probe, "pk2"), axis=0)
    x15_shot0 = float(np.mean(stack(x15, "shot0")))
    x15_shot2 = float(np.mean(stack(x15, "shot2")))
    pr_shot0 = float(np.mean(stack(probe, "shot0")))
    pr_shot2 = float(np.mean(stack(probe, "shot2")))

    ndata_x15 = float(np.mean(stack(x15, "ndata")))
    ndata_probe = float(np.mean(stack(probe, "ndata")))

    report = {
        "task": "task43_analyze_ezmock_rsd_rawbox_density_probe",
        "status": "done",
        "x15_meas": str(args.x15_meas), "probe_meas": str(args.probe_meas),
        "abacus_dir": str(args.abacus_dir), "n_abacus": abacus["n"],
        "n_x15": len(x15), "n_probe": len(probe),
        "nbar": {
            "abacus": abacus["ndata"] / BOX_SIZE ** 3,
            "x15": ndata_x15 / BOX_SIZE ** 3,
            "probe": ndata_probe / BOX_SIZE ** 3,
        },
        "shotnoise": {
            "abacus": {"P0": ab_shot0, "P2": ab_shot2},
            "x15": {"P0": x15_shot0, "P2": x15_shot2},
            "probe": {"P0": pr_shot0, "P2": pr_shot2},
            "expected_1_over_nbar": {
                "abacus": BOX_SIZE ** 3 / abacus["ndata"],
                "x15": BOX_SIZE ** 3 / ndata_x15,
                "probe": BOX_SIZE ** 3 / ndata_probe,
            },
        },
        "band_pk0_total": {"abacus": band_dict(k, ab_pk0), "x15": band_dict(k, x15_pk0), "probe": band_dict(k, pr_pk0)},
    }

    for ell, ab_pk, x15_pk, pr_pk, ab_shot, x15_shot, pr_shot in (
        (0, ab_pk0, x15_pk0, pr_pk0, ab_shot0, x15_shot0, pr_shot0),
        (2, ab_pk2, x15_pk2, pr_pk2, ab_shot2, x15_shot2, pr_shot2),
    ):
        label = f"P{ell}"
        report[f"{label}_total"] = {
            "x15_vs_abacus": ratio_dict(k, x15_pk, ab_pk),
            "probe_vs_abacus": ratio_dict(k, pr_pk, ab_pk),
        }
        report[f"{label}_shot_subtracted"] = {
            "x15_vs_abacus": ratio_dict(k, x15_pk - x15_shot, ab_pk - ab_shot),
            "probe_vs_abacus": ratio_dict(k, pr_pk - pr_shot, ab_pk - ab_shot),
        }
        paired = {}
        for item in x15_paired:
            other = probe_by_seed.get(item["seed"])
            if other is None:
                continue
            paired[str(item["seed"])] = ratio_dict(
                k, other["pk0" if ell == 0 else "pk2"], item["pk0" if ell == 0 else "pk2"])
        if paired:
            keys = list(paired[list(paired)[0]].keys())
            report[f"{label}_paired_probe_over_x15"] = {
                "per_seed": paired,
                "mean": {band: float(np.mean([value[band] for value in paired.values()])) for band in keys},
                "std": {band: float(np.std([value[band] for value in paired.values()], ddof=1)) if len(paired) > 1 else 0.0 for band in keys},
            }

    args.out_json = out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"[done] json: {out_json}")
    print(f"  nbar: abacus={report['nbar']['abacus']:.4e} x15={report['nbar']['x15']:.4e} probe={report['nbar']['probe']:.4e} (Mpc/h)^-3")
    print(f"  shot noise P0: abacus={ab_shot0:.1f} x15={x15_shot0:.1f} probe={pr_shot0:.1f}"
          f" | expected 1/nbar: {report['shotnoise']['expected_1_over_nbar']['abacus']:.1f} /"
          f" {report['shotnoise']['expected_1_over_nbar']['x15']:.1f} / {report['shotnoise']['expected_1_over_nbar']['probe']:.1f}")
    for label in ("P0", "P2"):
        for kind in ("total", "shot_subtracted"):
            print(f"  {label} {kind}:")
            print(f"    x15(1.297M) / abacus   -> {fmt(report[f'{label}_{kind}']['x15_vs_abacus'])}")
            print(f"    probe(1.847M) / abacus -> {fmt(report[f'{label}_{kind}']['probe_vs_abacus'])}")
        if f"{label}_paired_probe_over_x15" in report:
            print(f"  {label} paired probe/x15 (same seed, x{len(PAIRED_SEEDS)}):")
            print(f"    mean -> {fmt(report[f'{label}_paired_probe_over_x15']['mean'])}")
            print(f"    std  -> {fmt(report[f'{label}_paired_probe_over_x15']['std'])}")


if __name__ == "__main__":
    main()
