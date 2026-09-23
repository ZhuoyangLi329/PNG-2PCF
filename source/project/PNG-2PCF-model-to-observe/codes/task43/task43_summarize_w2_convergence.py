#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize Task43 formal-GIC W2 nsub convergence for the L2000 setup."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_DIR = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
SUMMARY_DIR = PROJECT_DIR / "outputs" / "task43_outputs" / "summary"
FIT_DIR = PROJECT_DIR / "outputs" / "task43_outputs" / "fits"
OUT_JSON = SUMMARY_DIR / "task43_formal_gic_w2_convergence_L2000_seed20260703.json"
OUT_CSV = SUMMARY_DIR / "task43_formal_gic_w2_convergence_L2000_seed20260703.csv"
NSUBS = (10000, 50000, 200000)
SMINS = (50, 60, 80)


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
    return obj


def window_path(nsub: int) -> Path:
    return SUMMARY_DIR / f"task43_formal_gic_window_mmin1p4e13_x25_ph000_fkpP010000_L2000_nsub{nsub}_seed20260703.npz"


def fit_path(smin: int, nsub: int) -> Path:
    return FIT_DIR / f"formalgic_w2_convergence_L2000_smin{smin}_nsub{nsub}" / "task43_minimal_closure_mcmc_summary.json"


def baseline_path(smin: int) -> Path:
    return FIT_DIR / f"full25_mean_s{smin}_350_ds10_shellavg_besselfftlog_rrdeconv_fitcov_L2000_p1p0_mcmc" / "task43_minimal_closure_mcmc_summary.json"


def load_window(nsub: int) -> dict[str, Any]:
    path = window_path(nsub)
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(np.asarray(data["meta_json"]).item()))
    return {
        "path": path,
        "k_eff": np.asarray(data["k_eff"], dtype="f8"),
        "w2": np.asarray(data["w2"], dtype="f8"),
        "meta": meta,
    }


def model_entry(summary: dict[str, Any], model: str) -> dict[str, Any]:
    for entry in summary["models"]:
        if entry["model"] == model:
            return entry
    raise KeyError(f"model={model} not found")


def fit_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    entry = model_entry(summary, "formal_gic")
    fnl = entry["fnl_loc"]
    sigma = 0.5 * (float(fnl["q84"]) - float(fnl["q16"]))
    return {
        "path": path,
        "q16": float(fnl["q16"]),
        "q50": float(fnl["q50"]),
        "q84": float(fnl["q84"]),
        "sigma": sigma,
        "map_fnl": float(entry["map"]["fnl_loc"]),
        "map_b1": float(entry["map"]["b1"]),
        "chi2_map": float(entry["data"]["chi2_map_total"]),
        "sigma_w2_map": float(entry["formal_gic"]["sigma_w2_map"]),
        "nwalkers": int(entry["mcmc"]["nwalkers"]),
        "nsteps": int(entry["mcmc"]["nsteps"]),
        "burnin": int(entry["mcmc"]["burnin"]),
        "acceptance": float(entry["mcmc"]["mean_acceptance_fraction"]),
    }


def main() -> None:
    windows = {nsub: load_window(nsub) for nsub in NSUBS}
    ref = windows[200000]
    active = (ref["k_eff"] > 0.0) & (ref["k_eff"] <= 0.5)
    window_rows = []
    for nsub, item in windows.items():
        if item["k_eff"].shape != ref["k_eff"].shape or not np.allclose(item["k_eff"], ref["k_eff"], rtol=0.0, atol=1.0e-14):
            raise ValueError(f"k_eff mismatch for nsub={nsub}")
        diff = item["w2"][active] - ref["w2"][active]
        denom = np.maximum(np.abs(ref["w2"][active]), 1.0e-30)
        window_rows.append(
            {
                "nsub": int(nsub),
                "path": str(item["path"]),
                "n_random_total": int(item["meta"].get("n_random_total", -1)),
                "n_subsample": int(item["meta"].get("n_subsample", -1)),
                "coverage": float(item["meta"].get("coverage", np.nan)),
                "theory_boxsize": float(item["meta"].get("theory_boxsize", np.nan)),
                "active_k_count": int(np.sum(active)),
                "w2_active_min": float(np.min(item["w2"][active])),
                "w2_active_max": float(np.max(item["w2"][active])),
                "w2_vs_200k_rms_abs": float(np.sqrt(np.mean(diff**2))),
                "w2_vs_200k_max_abs": float(np.max(np.abs(diff))),
                "w2_vs_200k_rms_rel": float(np.sqrt(np.mean((diff / denom) ** 2))),
                "w2_vs_200k_max_rel": float(np.max(np.abs(diff / denom))),
            }
        )

    fit_rows = []
    for smin in SMINS:
        base = fit_summary(baseline_path(smin))
        for nsub in NSUBS:
            current = fit_summary(fit_path(smin, nsub))
            fit_rows.append(
                {
                    "smin": int(smin),
                    "nsub": int(nsub),
                    "baseline_q16": base["q16"],
                    "baseline_q50": base["q50"],
                    "baseline_q84": base["q84"],
                    "baseline_sigma": base["sigma"],
                    "baseline_sigma_w2_map": base["sigma_w2_map"],
                    "q16": current["q16"],
                    "q50": current["q50"],
                    "q84": current["q84"],
                    "sigma": current["sigma"],
                    "sigma_w2_map": current["sigma_w2_map"],
                    "delta_q50_vs_baseline": current["q50"] - base["q50"],
                    "delta_sigma_vs_baseline": current["sigma"] - base["sigma"],
                    "delta_sigma_w2_map_vs_baseline": current["sigma_w2_map"] - base["sigma_w2_map"],
                    "delta_q50_over_baseline_sigma": (current["q50"] - base["q50"]) / base["sigma"],
                    "chi2_map": current["chi2_map"],
                    "fit_path": str(current["path"]),
                    "baseline_path": str(base["path"]),
                }
            )
    reference_fit_rows = [row for row in fit_rows if row["nsub"] == 200000]
    convergence_summary = {
        "reference_nsub": 200000,
        "max_abs_delta_q50_over_baseline_sigma_at_reference": float(
            max(abs(row["delta_q50_over_baseline_sigma"]) for row in reference_fit_rows)
        ),
        "max_abs_delta_sigma_at_reference": float(max(abs(row["delta_sigma_vs_baseline"]) for row in reference_fit_rows)),
        "max_abs_delta_sigma_w2_map_at_reference": float(
            max(abs(row["delta_sigma_w2_map_vs_baseline"]) for row in reference_fit_rows)
        ),
    }

    payload = {
        "status": "done",
        "task": "task43_formal_gic_w2_convergence_L2000",
        "nsubs": list(NSUBS),
        "smins": list(SMINS),
        "reference_nsub": 200000,
        "convergence_summary": convergence_summary,
        "window_rows": window_rows,
        "fit_rows": fit_rows,
        "notes": [
            "All fits use mean xi and covariance_single_realization.",
            "Baseline is the previous shell-averaged L2000 p1p0 formal_gic summary produced from the old nsub10000 earlydiag W2; that old cache has been removed from the active workspace.",
            "Relative W2 differences can be inflated near W2 zero crossings; use the fNL and sigma_W2_map rows for the practical convergence check.",
        ],
    }
    OUT_JSON.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(fit_rows[0].keys()) if fit_rows else []
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fit_rows)
    print(json.dumps(jsonable(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
