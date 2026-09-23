#!/usr/bin/env python3
"""Final immutable audit for the complete Task 4.3.2 controlled RSD validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from task43_build_rsd_formal_gic_window import output_path as formal_window_path
from task43_rsd_common import OUTPUT_ROOT, PHASES, PLOT_ROOT, atomic_write_json, sha256_file
from task43_summarize_rsd_lightcone_x25 import measurement_path


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
AUDIT_ROOT = OUTPUT_ROOT / "audits"
RAWBOX_JSON = OUTPUT_ROOT / "rawbox" / "closure" / "task43_rsd_rawbox_x25_fulldiscrete_lorentzian.json"
RAWBOX_AUDIT = AUDIT_ROOT / "task43_rsd_rawbox_x25_failure_audit.json"
RR_STABILITY_AUDIT = AUDIT_ROOT / "task43_rsd_lightcone_rr100k_vs_rr300k_covariance_stability.json"
PILOT_JSON = OUTPUT_ROOT / "lightcone" / "pilot" / "task43_rsd_lightcone_pilot_x3.json"
SUMMARY_JSON = OUTPUT_ROOT / "lightcone" / "summary" / "task43_rsd_lightcone_x25_mean_xi02_s30_350_ds10.json"
RAW_COVARIANCE = OUTPUT_ROOT / "lightcone" / "covariance" / (
    "task43_rsd_ph000_jaxpower_raw_rsdpoles024_win02468_mesh64_p1_b1cov2p604_"
    "sigmas7p566_nran100k_ndata50k_seed20260816_kbox_fkpP010000_s30_350_ds10.npz"
)
DECONVOLVED_COVARIANCE = OUTPUT_ROOT / "lightcone" / "covariance" / (
    "task43_rsd_ph000_jaxpower_rrdeconv_ell02_rsdpoles024_win02468_mesh64_p1_b1cov2p604_"
    "sigmas7p566_nran100k_ndata50k_seed20260816_rrnran300k_rrseed430320_"
    "kbox_fkpP010000_s30_350_ds10.npz"
)
LIGHTCONE_JSON = OUTPUT_ROOT / "lightcone" / "closure" / (
    "task43_rsd_lightcone_x25_jaxpower_rrdeconv_rrnran300k_fulldiscrete_lorentzian.json"
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def checked_hash(path: Path) -> str:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return sha256_file(path)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def verify_pdf(path: Path) -> dict[str, Any]:
    digest = checked_hash(path)
    with path.open("rb") as stream:
        header = stream.read(5)
    if header != b"%PDF-":
        raise RuntimeError(f"not a PDF product: {path}")
    return {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsub-window", type=int, default=200000)
    parser.add_argument("--window-seed-base", type=int, default=430340)
    parser.add_argument(
        "--output",
        type=Path,
        default=AUDIT_ROOT / "task43_rsd_validation_final_audit_rrnran300k_verified.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable final RSD audit already exists: {args.output}")

    rawbox = read_json(RAWBOX_JSON)
    rawbox_audit = read_json(RAWBOX_AUDIT)
    rr_stability = read_json(RR_STABILITY_AUDIT)
    pilot = read_json(PILOT_JSON)
    summary = read_json(SUMMARY_JSON)
    lightcone = read_json(LIGHTCONE_JSON)
    raw_covariance_meta = read_json(RAW_COVARIANCE.with_suffix(".json"))
    deconvolved_meta = read_json(DECONVOLVED_COVARIANCE.with_suffix(".json"))

    measurement_hashes: dict[str, str] = {}
    for phase in PHASES:
        path = measurement_path(phase)
        metadata = read_json(path.with_suffix(".json"))
        digest = checked_hash(path)
        if metadata.get("status") != "pass" or metadata.get("output_sha256") != digest:
            raise RuntimeError(f"unvalidated lightcone measurement for {phase}")
        measurement_hashes[phase] = digest

    window_hashes: dict[str, str] = {}
    for phase in PHASES:
        path = formal_window_path(phase, int(args.nsub_window), int(args.window_seed_base))
        metadata = read_json(path.with_suffix(".json"))
        digest = checked_hash(path)
        if metadata.get("status") != "pass" or metadata.get("output_sha256") != digest:
            raise RuntimeError(f"unvalidated formal-GIC window for {phase}")
        window_hashes[phase] = digest

    if rawbox.get("status") != "validation_failed" or rawbox_audit.get("status") != "pass":
        raise RuntimeError("rawbox scientific failure or its failure audit is not preserved")
    if pilot.get("status") != "pass" or not all(bool(value) for value in pilot.get("gates", {}).values()):
        raise RuntimeError("x3 lightcone pilot gate did not pass")
    if summary.get("status") != "pass" or int(summary.get("nphase", -1)) != len(PHASES):
        raise RuntimeError("x25 lightcone summary is incomplete")
    if raw_covariance_meta.get("status") not in {"done", "pass"}:
        raise RuntimeError("raw JAXpower covariance did not complete")
    if lightcone.get("status") != "validation_failed":
        raise RuntimeError("top-level Task 4.3.2 status must retain the failed rawbox dependency")
    if lightcone.get("diagnostic_execution_status") != "pass":
        raise RuntimeError("lightcone diagnostic did not complete cleanly")
    if lightcone.get("task44_greenlight") is not False:
        raise RuntimeError("Task44 greenlight must be false")
    followups = lightcone.get("conditional_followups", {})
    if followups.get("radial_ic") != "not_run" or followups.get("hybrid_gsm") != "not_run":
        raise RuntimeError("forbidden post-failure radial-IC/hybrid branch was not marked not_run")

    closure_covariance = project_path(lightcone["covariance_policy"]["path"])
    if closure_covariance.resolve() != DECONVOLVED_COVARIANCE.resolve():
        raise RuntimeError(
            "lightcone closure did not use the frozen 300k-RR primary covariance: "
            f"{closure_covariance} != {DECONVOLVED_COVARIANCE}"
        )
    covariance_hash = checked_hash(DECONVOLVED_COVARIANCE)
    if lightcone["covariance_policy"].get("sha256") != covariance_hash:
        raise RuntimeError("lightcone closure covariance hash does not match the 300k-RR primary")
    if rr_stability.get("status") != "pass":
        raise RuntimeError("100k/300k RR covariance stability audit did not pass")
    if project_path(rr_stability["reference"]).resolve() != DECONVOLVED_COVARIANCE.resolve():
        raise RuntimeError("RR stability audit reference is not the frozen 300k-RR primary")
    if rr_stability.get("reference_sha256") != covariance_hash:
        raise RuntimeError("RR stability audit reference hash mismatch")

    with np.load(DECONVOLVED_COVARIANCE, allow_pickle=False) as covariance_payload:
        covariance = np.asarray(covariance_payload["covariance_single_realization"], dtype="f8")
        ells = tuple(int(value) for value in np.asarray(covariance_payload["ells"]).ravel())
        s = np.asarray(covariance_payload["s"], dtype="f8")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if covariance.shape != (64, 64) or ells != (0, 2) or s.shape != (32,):
        raise RuntimeError(f"bad final covariance contract: {covariance.shape}, {ells}, {s.shape}")
    if not np.all(np.isfinite(covariance)) or float(eigenvalues[0]) <= 0.0:
        raise RuntimeError("final RR-deconvolved covariance is not finite SPD")

    expected_pdfs = (
        PLOT_ROOT / "task43_rsd_rawbox_x25_fulldiscrete_lorentzian.pdf",
        PLOT_ROOT / "task43_rsd_lightcone_pilot_x3.pdf",
        PLOT_ROOT / "task43_rsd_lightcone_x25_mean_xi02.pdf",
        PLOT_ROOT / "task43_rsd_lightcone_x25_jaxpower_rrdeconv_rrnran300k_fulldiscrete_lorentzian.pdf",
    )
    pdf_products = [verify_pdf(path) for path in expected_pdfs]
    png_products = sorted(str(path) for path in PLOT_ROOT.rglob("*.png"))
    if png_products:
        raise RuntimeError(f"PNG hygiene gate failed: {png_products}")

    summary_npz = project_path(summary["outputs"]["summary_npz"])
    summary_hash = checked_hash(summary_npz)
    if summary_hash != summary["outputs"]["summary_sha256"]:
        raise RuntimeError("x25 summary hash mismatch")
    lightcone_npz = project_path(lightcone["output_npz"])
    lightcone_pdf = project_path(lightcone["output_plot_pdf"])
    if checked_hash(lightcone_npz) != lightcone["output_npz_sha256"]:
        raise RuntimeError("lightcone closure NPZ hash mismatch")
    if checked_hash(lightcone_pdf) != lightcone["output_plot_pdf_sha256"]:
        raise RuntimeError("lightcone closure PDF hash mismatch")

    rawbox_posterior = rawbox["mcmc_nominal"]["posterior"]
    lightcone_primary = lightcone["nominal_mean_smin50_xi0"]["formal_gic"]
    payload = {
        "task": "task43_finalize_rsd_validation",
        "status": "validation_failed",
        "execution_status": "pass",
        "classification": "complete controlled fNL=0 RSD validation; failed base closure, no Task44 model greenlight",
        "scientific_decision": {
            "task44_greenlight": False,
            "rawbox_gate": rawbox["status"],
            "lightcone_diagnostic_execution_status": lightcone["diagnostic_execution_status"],
            "lightcone_diagnostic_validation_status": lightcone["diagnostic_validation_status"],
            "radial_ic": "not_run",
            "hybrid_gsm": "not_run",
            "reason": "rawbox mean/phase PTE and scale-stability gates failed after a passing null-bias gate",
        },
        "rawbox_primary": {
            "fnl": rawbox_posterior["fNL"],
            "b1": rawbox_posterior["b1"],
            "sigma_s": rawbox_posterior["sigma_s"],
            "null_abs_median_over_sigma68": rawbox["primary_null_abs_median_over_sigma68_single"],
            "mean_chi2": rawbox["nominal"]["chi2_mean_covariance"],
            "mean_dof": rawbox["nominal"]["dof"],
            "mean_pte": rawbox["nominal"]["pte_mean_covariance"],
            "phase_ensemble": rawbox["phase_ensemble"],
            "scale_stability": rawbox["scale_stability"],
            "gates": rawbox["primary_gates"],
        },
        "lightcone_diagnostic": {
            "fnl": lightcone_primary["fnl_loc"],
            "b1": lightcone_primary["b1"],
            "sigma_s": lightcone_primary["sigma_s"],
            "null_abs_median_over_sigma68": lightcone["primary_null_abs_median_over_sigma68_single"],
            "mean_goodness": lightcone["mean_goodness_primary"],
            "phase_ensemble": lightcone["phase_ensemble_primary"],
            "scale_stability": lightcone["scale_stability"],
            "gates": lightcone["primary_gates"],
        },
        "covariance": {
            "raw_path": str(RAW_COVARIANCE),
            "raw_sha256": checked_hash(RAW_COVARIANCE),
            "deconvolved_path": str(DECONVOLVED_COVARIANCE),
            "deconvolved_sha256": covariance_hash,
            "deconvolved_meta": {
                key: deconvolved_meta.get(key)
                for key in ("status", "warning", "rr_window", "covariance_single_realization", "spd", "scatter_comparison")
            },
            "shape": list(covariance.shape),
            "ells": list(ells),
            "eigenvalue_min": float(eigenvalues[0]),
            "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            "x25_scatter_comparison": lightcone["covariance_policy"]["x25_scatter_comparison"],
            "rr100k_vs_rr300k_stability": {
                "path": str(RR_STABILITY_AUDIT),
                "sha256": checked_hash(RR_STABILITY_AUDIT),
                "status": rr_stability["status"],
                "relative_frobenius_covariance_delta": rr_stability["relative_frobenius_covariance_delta"],
                "rr_window_delta_relative_frobenius": rr_stability["rr_window_delta_relative_frobenius"],
                "sigma_ratio_candidate_over_reference_median": rr_stability["sigma_ratio_median"],
                "sigma_ratio_candidate_over_reference_min": rr_stability["sigma_ratio_min"],
                "sigma_ratio_candidate_over_reference_max": rr_stability["sigma_ratio_max"],
            },
        },
        "x25_measurement_sha256": measurement_hashes,
        "x25_formal_gic_window_sha256": window_hashes,
        "summary": {"path": str(summary_npz), "sha256": summary_hash},
        "pdf_products": pdf_products,
        "png_products": png_products,
        "provenance": {
            "rawbox_closure": str(RAWBOX_JSON),
            "rawbox_failure_audit": str(RAWBOX_AUDIT),
            "rr100k_vs_rr300k_covariance_stability": str(RR_STABILITY_AUDIT),
            "lightcone_pilot": str(PILOT_JSON),
            "lightcone_summary": str(SUMMARY_JSON),
            "lightcone_closure": str(LIGHTCONE_JSON),
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "execution_status": "pass", "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
