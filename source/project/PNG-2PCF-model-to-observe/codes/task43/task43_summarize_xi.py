#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize Task43 lightcone 2PCF measurements across phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from task43_config import DEFAULT_MANIFEST, SUMMARY_DIR, read_jsonl
from task43_fkp_zeff import path_with_weight_tag


def zeff_from_fkp_summary(path: Path, p0: float) -> float:
    """Read the all-random effective redshift for one P0 from the FKP summary."""
    data = np.load(path, allow_pickle=False)
    p0_values = np.asarray(data["p0_values"], dtype="f8")
    zeff_values = np.asarray(data["zeff_random_auto"], dtype="f8")
    matches = np.flatnonzero(np.isclose(p0_values, float(p0), rtol=0.0, atol=1.0e-10))
    if matches.size != 1:
        raise ValueError(f"could not find unique P0={p0} in {path}; available={p0_values.tolist()}")
    return float(zeff_values[int(matches[0])])


def load_xi_row(row: dict[str, Any], *, p0: float | None, output_tag: str | None) -> dict[str, Any] | None:
    path = path_with_weight_tag(Path(row["xi_path"]), p0=p0, output_tag=output_tag)
    if not path.exists():
        return None
    data = np.load(path)
    required = ("s", "s_edges", "xi0", "DD", "DR", "RR", "zeff")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    xi = np.asarray(data["xi0"], dtype="f8")
    if not np.all(np.isfinite(xi)):
        raise ValueError(f"{path} has non-finite xi0")
    rr = np.asarray(data["RR"], dtype="f8")
    if not np.all(rr > 0):
        raise ValueError(f"{path} has non-positive RR bins")
    return {
        "phase": str(data["phase"]),
        "sim_name": str(data["sim_name"]),
        "path": str(path),
        "s": np.asarray(data["s"], dtype="f8"),
        "s_edges": np.asarray(data["s_edges"], dtype="f8"),
        "xi0": xi,
        "DD": np.asarray(data["DD"], dtype="f8"),
        "DR": np.asarray(data["DR"], dtype="f8"),
        "RR": rr,
        "ndata": int(data["ndata"]),
        "nrandom": int(data["nrandom"]),
        "zeff": float(data["zeff"]),
        "zeff_random_auto": float(data["zeff_random_auto"]) if "zeff_random_auto" in data.files else float(data["zeff"]),
        "zeff_data_auto": float(data["zeff_data_auto"]) if "zeff_data_auto" in data.files else np.nan,
        "zeff_data_random_cross": float(data["zeff_data_random_cross"]) if "zeff_data_random_cross" in data.files else np.nan,
        "zeff_data_mean": float(data["zeff_data_mean"]) if "zeff_data_mean" in data.files else np.nan,
        "zeff_random_mean": float(data["zeff_random_mean"]) if "zeff_random_mean" in data.files else np.nan,
        "zeff_data_weighted_mean": float(data["zeff_data_weighted_mean"]) if "zeff_data_weighted_mean" in data.files else np.nan,
        "zeff_random_weighted_mean": float(data["zeff_random_weighted_mean"]) if "zeff_random_weighted_mean" in data.files else np.nan,
        "data_weight_min": float(data["data_weight_min"]) if "data_weight_min" in data.files else np.nan,
        "data_weight_max": float(data["data_weight_max"]) if "data_weight_max" in data.files else np.nan,
        "random_weight_min": float(data["random_weight_min"]) if "random_weight_min" in data.files else np.nan,
        "random_weight_max": float(data["random_weight_max"]) if "random_weight_max" in data.files else np.nan,
        "p0": float(data["p0"]) if "p0" in data.files else np.nan,
    }


def correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.diag(cov))
    denom = np.outer(sigma, sigma)
    corr = np.zeros_like(cov)
    np.divide(cov, denom, out=corr, where=denom > 0)
    return corr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-prefix", type=Path, default=SUMMARY_DIR / "task43_mean_xi")
    parser.add_argument("--p0", type=float, default=None)
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--fkp-summary", type=Path, default=None)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    loaded = [load_xi_row(row, p0=args.p0, output_tag=args.output_tag) for row in rows]
    missing = [row["phase"] for row, item in zip(rows, loaded, strict=True) if item is None]
    items = [item for item in loaded if item is not None]
    if args.require_all and missing:
        raise FileNotFoundError(f"missing xi for phases: {missing}")
    if not items:
        raise RuntimeError(f"no xi files found in {args.manifest}")

    s = items[0]["s"]
    s_edges = items[0]["s_edges"]
    for item in items[1:]:
        if not np.allclose(item["s"], s) or not np.allclose(item["s_edges"], s_edges):
            raise ValueError(f"inconsistent s bins for {item['phase']}")

    phases = np.array([item["phase"] for item in items])
    paths = np.array([item["path"] for item in items])
    xi_all = np.vstack([item["xi0"] for item in items])
    dd_all = np.vstack([item["DD"] for item in items])
    dr_all = np.vstack([item["DR"] for item in items])
    rr_all = np.vstack([item["RR"] for item in items])
    zeff = np.array([item["zeff"] for item in items], dtype="f8")
    zeff_random_auto = np.array([item["zeff_random_auto"] for item in items], dtype="f8")
    zeff_data_auto = np.array([item["zeff_data_auto"] for item in items], dtype="f8")
    zeff_data_random_cross = np.array([item["zeff_data_random_cross"] for item in items], dtype="f8")
    zeff_data_mean = np.array([item["zeff_data_mean"] for item in items], dtype="f8")
    zeff_random_mean = np.array([item["zeff_random_mean"] for item in items], dtype="f8")
    zeff_data_weighted_mean = np.array([item["zeff_data_weighted_mean"] for item in items], dtype="f8")
    zeff_random_weighted_mean = np.array([item["zeff_random_weighted_mean"] for item in items], dtype="f8")
    data_weight_min = np.array([item["data_weight_min"] for item in items], dtype="f8")
    data_weight_max = np.array([item["data_weight_max"] for item in items], dtype="f8")
    random_weight_min = np.array([item["random_weight_min"] for item in items], dtype="f8")
    random_weight_max = np.array([item["random_weight_max"] for item in items], dtype="f8")
    p0_values = np.array([item["p0"] for item in items], dtype="f8")
    ndata = np.array([item["ndata"] for item in items], dtype="i8")
    nrandom = np.array([item["nrandom"] for item in items], dtype="i8")

    nreal = xi_all.shape[0]
    xi_mean = np.mean(xi_all, axis=0)
    xi_std = np.std(xi_all, axis=0, ddof=1) if nreal > 1 else np.zeros_like(xi_mean)
    xi_sem = xi_std / np.sqrt(max(nreal, 1))
    dd_mean = np.mean(dd_all, axis=0)
    dr_mean = np.mean(dr_all, axis=0)
    rr_mean = np.mean(rr_all, axis=0)
    zeff_phase_mean = float(np.mean(zeff))
    if args.fkp_summary is not None:
        if args.p0 is None:
            raise ValueError("--p0 is required when --fkp-summary is provided")
        zeff_mean = zeff_from_fkp_summary(args.fkp_summary, float(args.p0))
        zeff_source = "fkp_summary_all_random_zeff_random_auto"
    else:
        zeff_mean = zeff_phase_mean
        zeff_source = "phase_mean_zeff"

    if nreal > 1:
        sample_cov_realization = np.cov(xi_all, rowvar=False, ddof=1)
        sample_cov_mean = sample_cov_realization / nreal
    else:
        sample_cov_realization = np.zeros((s.size, s.size), dtype="f8")
        sample_cov_mean = np.zeros((s.size, s.size), dtype="f8")
    sample_corr_realization = correlation_from_covariance(sample_cov_realization)

    output_prefix = args.output_prefix
    if output_prefix == SUMMARY_DIR / "task43_mean_xi" and (args.p0 is not None or args.output_tag):
        output_prefix = path_with_weight_tag(output_prefix, p0=args.p0, output_tag=args.output_tag)
    out_npz = output_prefix.with_suffix(".npz")
    out_json = output_prefix.with_suffix(".json")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        s=s,
        s_edges=s_edges,
        xi0=xi_mean,
        xi0_mean=xi_mean,
        xi0_all=xi_all,
        xi0_std=xi_std,
        xi0_sem=xi_sem,
        DD=dd_mean,
        DR=dr_mean,
        RR=rr_mean,
        DD_all=dd_all,
        DR_all=dr_all,
        RR_all=rr_all,
        diagnostic_scatter_covariance_of_mean=sample_cov_mean,
        diagnostic_scatter_covariance_single_realization=sample_cov_realization,
        diagnostic_scatter_correlation_single_realization=sample_corr_realization,
        phases=phases,
        xi_paths=paths,
        nreal=nreal,
        ndata=ndata,
        nrandom=nrandom,
        zeff=zeff_mean,
        zeff_source=np.asarray(zeff_source),
        zeff_phase_mean=np.array(zeff_phase_mean, dtype="f8"),
        zeff_all=zeff,
        zeff_random_auto_all=zeff_random_auto,
        zeff_data_auto_all=zeff_data_auto,
        zeff_data_random_cross_all=zeff_data_random_cross,
        zeff_data_mean_all=zeff_data_mean,
        zeff_random_mean_all=zeff_random_mean,
        zeff_data_weighted_mean_all=zeff_data_weighted_mean,
        zeff_random_weighted_mean_all=zeff_random_weighted_mean,
        data_weight_min_all=data_weight_min,
        data_weight_max_all=data_weight_max,
        random_weight_min_all=random_weight_min,
        random_weight_max_all=random_weight_max,
        p0_all=p0_values,
        phase="mean",
        sim_name="AbacusSummit_base_c000_ph000-ph024_mean",
        estimator="landy_szalay_mean_over_phases",
        covariance_note=(
            "xi summary does not provide fit covariance; Task43 lightcone fits "
            "must use external jaxpower Gaussian survey-window covariance"
        ),
    )

    summary = {
        "status": "done",
        "manifest": str(args.manifest),
        "output_npz": str(out_npz),
        "nrows": len(rows),
        "nreal": int(nreal),
        "missing_phases": missing,
        "phases": phases.tolist(),
        "s_min": float(np.min(s)),
        "s_max": float(np.max(s)),
        "nbins": int(s.size),
        "zeff_mean": zeff_mean,
        "zeff_source": zeff_source,
        "zeff_phase_mean": zeff_phase_mean,
        "fkp_summary": None if args.fkp_summary is None else str(args.fkp_summary),
        "zeff_min": float(np.min(zeff)),
        "zeff_max": float(np.max(zeff)),
        "zeff_data_auto_mean": float(np.nanmean(zeff_data_auto)),
        "zeff_random_auto_mean": float(np.nanmean(zeff_random_auto)),
        "zeff_data_random_cross_mean": float(np.nanmean(zeff_data_random_cross)),
        "data_weight_min": float(np.nanmin(data_weight_min)),
        "data_weight_max": float(np.nanmax(data_weight_max)),
        "random_weight_min": float(np.nanmin(random_weight_min)),
        "random_weight_max": float(np.nanmax(random_weight_max)),
        "p0_requested": None if args.p0 is None else float(args.p0),
        "output_tag": args.output_tag,
        "ndata_min": int(np.min(ndata)),
        "ndata_max": int(np.max(ndata)),
        "nrandom_min": int(np.min(nrandom)),
        "nrandom_max": int(np.max(nrandom)),
        "xi0_first3": [float(v) for v in xi_mean[:3]],
        "xi0_last3": [float(v) for v in xi_mean[-3:]],
        "diagnostic_scatter": {
            "diagnostic_scatter_covariance_single_realization": "realization scatter only; not a fit covariance",
            "diagnostic_scatter_covariance_of_mean": "realization scatter divided by nreal; not a fit covariance",
            "fit_covariance_required": "external jaxpower Gaussian survey-window covariance",
        },
    }
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
