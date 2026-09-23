#!/usr/bin/env python3
"""Audit the c302 fNL_cov=100 covariance, refit, and final PDF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader

from task43_build_pngbase_c302_fnl100_covariance import phase_fit_covariance, phase_prefix
from task43_pngbase_pseudolc_common import FINAL_ROOT, PHASES, fit_root, fit_root_png_cov, png_covariance_path
from task43_rsd_common import atomic_write_json, sha256_file


FIT_AUDIT_NAME = "task43_pngbase_pseudolc_joint_baomask80_120.json"
PLOT_STEM = (
    "task43_pngbase_c302_pseudolc_real_Pxi_rsd_P02xi02_joint_cov_fnl100_"
    "kmax0p08_smin50_baomask80_120"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    covariance_path = png_covariance_path()
    covariance_meta_path = covariance_path.with_suffix(".json")
    old_fit_path = fit_root("c302") / FIT_AUDIT_NAME
    new_fit_path = fit_root_png_cov() / FIT_AUDIT_NAME
    plot_path = FINAL_ROOT / f"{PLOT_STEM}.pdf"
    plot_meta_path = plot_path.with_suffix(".json")
    output = FINAL_ROOT / "task43_pngbase_c302_covariance_fnl0_vs_fnl100_audit_v2.json"
    if output.exists():
        raise FileExistsError(f"immutable audit exists: {output}")

    covariance_meta, old_fit, new_fit, plot_meta = map(
        load_json, (covariance_meta_path, old_fit_path, new_fit_path, plot_meta_path)
    )
    gates: dict[str, bool] = {
        "covariance_status_pass": covariance_meta.get("status") == "pass",
        "covariance_hash": covariance_meta.get("output_sha256") == sha256_file(covariance_path),
        "covariance_cosmology_c302": covariance_meta.get("cosmology") == "c302",
        "covariance_fnl_exactly_100": float(covariance_meta.get("fnl_cov", np.nan)) == 100.0,
        "covariance_not_divided_by_two": float(covariance_meta.get("covariance_divisor", np.nan)) == 1.0,
        "new_fit_status_pass": new_fit.get("status") == "pass",
        "new_fit_all_numerical_gates": all(new_fit.get("numerical_gates", {}).values()),
        "new_fit_covariance_mode": new_fit.get("covariance_mode") == "c302_fnl100",
        "plot_status_pass": plot_meta.get("status") == "pass",
        "plot_hash": plot_meta.get("output_pdf_sha256") == sha256_file(plot_path),
        "plot_six_pages": len(PdfReader(plot_path).pages) == 6,
        "no_single_phase_fnl_fits": new_fit.get("single_phase_fnl_fits") == [],
    }
    phase_joint: dict[str, list[np.ndarray]] = {"real": [], "rsd": []}
    phase_sources: dict[str, Any] = {"real": {}, "rsd": {}}
    for space in ("real", "rsd"):
        for phase in PHASES:
            phase_metadata_path = phase_prefix(space, phase).with_suffix(".json")
            phase_metadata = load_json(phase_metadata_path)
            gates[f"{space}_{phase}_status_pass"] = phase_metadata.get("status") == "pass"
            gates[f"{space}_{phase}_fnl_exactly_100"] = float(phase_metadata.get("fnl_cov", np.nan)) == 100.0
            gates[f"{space}_{phase}_cosmology_c302"] = phase_metadata.get("cosmology") == "c302"
            joint, source = phase_fit_covariance(space, phase)
            phase_joint[space].append(joint)
            phase_sources[space][phase] = source
    with np.load(covariance_path, allow_pickle=False) as covariance:
        for space in ("real", "rsd"):
            expected = np.mean(np.stack(phase_joint[space]), axis=0)
            actual = np.asarray(covariance[f"{space}_joint"], dtype="f8")
            gates[f"{space}_exact_phase_covariance_arithmetic_mean"] = np.array_equal(actual, expected)
        gates["stored_fnl_cov_exactly_100"] = float(np.asarray(covariance["fnl_cov"]).item()) == 100.0
        gates["stored_covariance_divisor_exactly_one"] = float(np.asarray(covariance["covariance_divisor"]).item()) == 1.0

    posterior_comparison: dict[str, Any] = {}
    fit_data_unchanged = True
    chain_hashes: dict[str, Any] = {}
    for name, result in new_fit["results"].items():
        old_posterior = old_fit["results"][name]["mcmc"]["posterior"]["fNL"]
        new_posterior = result["mcmc"]["posterior"]["fNL"]
        old_samples = fit_root("c302") / "fits" / name / "samples.npz"
        new_samples = fit_root_png_cov() / "fits" / name / "samples.npz"
        with np.load(old_samples, allow_pickle=False) as old_payload, np.load(new_samples, allow_pickle=False) as new_payload:
            fit_data_unchanged &= np.array_equal(old_payload["data"], new_payload["data"])
            fit_data_unchanged &= np.array_equal(old_payload["phase_data"], new_payload["phase_data"])
        chain_hashes[name] = {"old": sha256_file(old_samples), "new": sha256_file(new_samples)}
        posterior_comparison[name] = {
            "old_fnl_cov0": {key: float(old_posterior[key]) for key in ("q16", "q50", "q84", "sigma68")},
            "new_fnl_cov100": {key: float(new_posterior[key]) for key in ("q16", "q50", "q84", "sigma68")},
            "sigma68_ratio_new_over_old": float(new_posterior["sigma68"] / old_posterior["sigma68"]),
        }
    gates["observable_and_phase_data_bitwise_unchanged"] = bool(fit_data_unchanged)
    status = "pass" if all(gates.values()) else "validation_failed"
    audit = {
        "task": "task43_audit_pngbase_c302_fnl100_covariance", "status": status,
        "result": (
            "c302 refit with phase-matched jaxpower disconnected Gaussian covariance evaluated at fNL_cov=100; "
            "the apparent joint gain is substantially reduced relative to the frozen fNL_cov=0 covariance"
        ),
        "scope_boundary": (
            "Includes PNG through the fNL=100 two-point signal entering the disconnected Gaussian covariance. "
            "Does not include a connected primordial trispectrum or other non-Gaussian covariance terms."
        ),
        "covariance_policy": {
            "per_phase": "independent c302 ph000/ph001 data, random, n(z), FKP survey windows and exact split-RR deconvolution",
            "combination": "elementwise arithmetic mean of the two C_single matrices",
            "divisor": 1.0,
            "observed_data": "same bitwise ph000/ph001 arithmetic-mean observables as the old fit",
        },
        "gates": gates,
        "covariance_old_to_new": covariance_meta["old_fnl0_comparison"],
        "posterior_old_to_new": posterior_comparison,
        "joint_improvement_new": new_fit["metrics"],
        "phase_sources": phase_sources,
        "chain_hashes": chain_hashes,
        "inputs": {
            "covariance": str(covariance_path), "covariance_sha256": sha256_file(covariance_path),
            "old_fit": str(old_fit_path), "old_fit_sha256": sha256_file(old_fit_path),
            "new_fit": str(new_fit_path), "new_fit_sha256": sha256_file(new_fit_path),
            "plot": str(plot_path), "plot_sha256": sha256_file(plot_path),
        },
    }
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, audit)
    print(json.dumps({"status": status, "output": str(output), "gates": gates}, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
