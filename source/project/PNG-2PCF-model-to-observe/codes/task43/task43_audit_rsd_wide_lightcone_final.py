#!/usr/bin/env python3
"""Final provenance/contract audit for the wide-redshift Task 4.3.2 monopoles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from task43_rsd_common import OUTPUT_ROOT, atomic_savez, atomic_write_json, sha256_file
from task43_rsd_lightcone_pk0_contract import TASK43_WIDE_LRGALL_FIT_EDGES, WINDOW_THEORY_KMIN


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
WIDE_ROOT = OUTPUT_ROOT / "lightcone_wide_zobs0p4_1p1"
DEFAULT_OUTPUT = OUTPUT_ROOT / "audits/task43_rsd_wide_lightcone_final_audit.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = {
        "manifest_audit": OUTPUT_ROOT / "audits/task43_rsd_wide_lightcone_manifest_audit.json",
        "preflight": OUTPUT_ROOT / "audits/task43_rsd_wide_lightcone_preflight.json",
        "upstream": OUTPUT_ROOT / "audits/task43_rsd_wide_lightcone_upstream_x25_audit.json",
        "xi_summary": WIDE_ROOT / "summary/task43_rsd_wide_lightcone_x25_mean_xi0_s30_350_ds10.npz",
        "xi_summary_json": WIDE_ROOT / "summary/task43_rsd_wide_lightcone_x25_mean_xi0_s30_350_ds10.json",
        "mean_window": WIDE_ROOT / "formal_gic_windows/task43_rsd_wide_formal_gic_x25_equal_phase_mean_nsub200000_seedbase430340.npz",
        "mean_window_json": WIDE_ROOT / "formal_gic_windows/task43_rsd_wide_formal_gic_x25_equal_phase_mean_nsub200000_seedbase430340.json",
        "xi_covariance": WIDE_ROOT / "covariance/task43_rsd_wide_ph000_jaxpower_rrdeconv_ell0_rsdpoles024_win02468_mesh64_p1_b1cov2p604_sigmas7p566_nran100k_ndata50k_seed20260817_rrnran300k_rrseed430420_kbox_fkpP010000_s30_350_ds10.npz",
        "xi_covariance_json": WIDE_ROOT / "covariance/task43_rsd_wide_ph000_jaxpower_rrdeconv_ell0_rsdpoles024_win02468_mesh64_p1_b1cov2p604_sigmas7p566_nran100k_ndata50k_seed20260817_rrnran300k_rrseed430420_kbox_fkpP010000_s30_350_ds10.json",
        "xi_fit": WIDE_ROOT / "closure/task43_rsd_wide_lightcone_x25_xi0_smin50_formalgic_lorentzian.npz",
        "xi_fit_json": WIDE_ROOT / "closure/task43_rsd_wide_lightcone_x25_xi0_smin50_formalgic_lorentzian.json",
        "pk_summary": WIDE_ROOT / "pk_summary/task43_rsd_wide_lightcone_pk0_x25_kmin0p003271970_kmax0p10_l0only_16bin.npz",
        "pk_summary_json": WIDE_ROOT / "pk_summary/task43_rsd_wide_lightcone_pk0_x25_kmin0p003271970_kmax0p10_l0only_16bin.json",
        "comparison": WIDE_ROOT / "comparison/task43_rsd_wide_lightcone_x25_pk0_vs_xi0_smin50_l0only_longchain.npz",
        "comparison_json": WIDE_ROOT / "comparison/task43_rsd_wide_lightcone_x25_pk0_vs_xi0_smin50_l0only_longchain.json",
        "geometry_comparison_json": OUTPUT_ROOT / "audits/task43_rsd_wide_vs_narrow_vs_rawbox_monopoles.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing final wide-lightcone products: {missing}")

    manifest_audit = load_json(paths["manifest_audit"])
    preflight = load_json(paths["preflight"])
    upstream = load_json(paths["upstream"])
    xi_summary_meta = load_json(paths["xi_summary_json"])
    window_meta = load_json(paths["mean_window_json"])
    xi_cov_meta = load_json(paths["xi_covariance_json"])
    xi_fit_meta = load_json(paths["xi_fit_json"])
    pk_meta = load_json(paths["pk_summary_json"])
    comparison = load_json(paths["comparison_json"])
    geometry_comparison = load_json(paths["geometry_comparison_json"])

    with np.load(paths["xi_summary"], allow_pickle=False) as payload:
        xi_ells = tuple(int(value) for value in np.asarray(payload["ells"]).ravel())
        xi_phases = tuple(str(value) for value in np.asarray(payload["phases"]).tolist())
        xi_zeff = float(np.asarray(payload["zeff_mean"]).item())
        xi_stack_shape = tuple(np.asarray(payload["xi_multipoles_by_phase"]).shape)
    with np.load(paths["mean_window"], allow_pickle=False) as payload:
        window_shape = tuple(np.asarray(payload["w2_by_phase"]).shape)
        window_finite = bool(np.all(np.isfinite(payload["w2_by_phase"])))
    with np.load(paths["xi_covariance"], allow_pickle=False) as payload:
        covariance_ells = tuple(int(value) for value in np.asarray(payload["ells"]).ravel())
        xi_covariance = np.asarray(payload["covariance_single_realization"], dtype="f8")
    with np.load(paths["pk_summary"], allow_pickle=False) as payload:
        pk_edges = np.asarray(payload["k_edges"], dtype="f8")
        pk_stack_shape = tuple(np.asarray(payload["pk_stack"]).shape)
        pk_covariance = np.asarray(payload["covariance_single_realization"], dtype="f8")
        volume = float(np.asarray(payload["lightcone_volume"]).item())
        observed_kmin = float(np.asarray(payload["kmin_fit_observed"]).item())
        theory_kmin = float(np.asarray(payload["window_theory_kmin"]).item())
        theory_ells = tuple(int(value) for value in np.unique(payload["theory_ell"]))

    plot_paths = [
        PROJECT_ROOT / "plots/task43/rsd_validation/task43_rsd_wide_zobs0p4_1p1_x25_mean_xi0.pdf",
        PROJECT_ROOT / "plots/task43/rsd_validation/task43_rsd_wide_zobs0p4_1p1_x25_xi0_smin50_formalgic_lorentzian.pdf",
        PROJECT_ROOT / "plots/task43/rsd_validation/task43_rsd_wide_zobs0p4_1p1_x25_pk0_vs_xi0_smin50_l0only_longchain.pdf",
        PROJECT_ROOT / "plots/task43/rsd_validation/task43_rsd_wide_zobs0p4_1p1_x25_pk0_vs_xi0_smin50_l0only_contours.pdf",
        PROJECT_ROOT / "plots/task43/rsd_validation/task43_rsd_wide_vs_narrow_vs_rawbox_monopoles.pdf",
    ]
    png_outputs = sorted(
        str(path)
        for path in (PROJECT_ROOT / "plots/task43/rsd_validation").glob("*wide*.png")
        if path.is_file()
    )
    gates = {
        "manifest": manifest_audit.get("status") == "pass" and manifest_audit.get("observed_redshift_open_interval") == [0.4, 1.1],
        "preflight": preflight.get("status") == "pass" and all(preflight.get("gates", {}).values()),
        "upstream_x25": upstream.get("status") == "pass" and upstream.get("nphase_validated") == 25,
        "xi_summary_hash": (
            xi_summary_meta.get("outputs", {}).get("summary_sha256") == sha256_file(paths["xi_summary"])
        ),
        "xi_monopole_only": xi_ells == (0,) and xi_stack_shape == (25, 1, 32) and len(xi_phases) == 25,
        "wide_zeff": 0.4 < xi_zeff < 1.1 and np.isclose(
            xi_zeff, upstream["x25_summary"]["zeff_equal_phase_mean"], rtol=0.0, atol=1.0e-14
        ),
        "formal_gic_window": bool(
            window_meta.get("status") == "pass"
            and window_meta.get("output_sha256") == sha256_file(paths["mean_window"])
            and window_shape[0] == 25
            and window_finite
            and "lightcone_wide_zobs0p4_1p1" in window_meta.get("phase_window_root", "")
        ),
        "xi_covariance_monopole_spd": bool(
            covariance_ells == (0,)
            and xi_covariance.shape == (32, 32)
            and np.linalg.eigvalsh(0.5 * (xi_covariance + xi_covariance.T))[0] > 0.0
            and xi_cov_meta.get("status") == "done"
        ),
        "xi_fit_wide_monopole": bool(
            xi_fit_meta.get("diagnostic_execution_status") == "pass"
            and xi_fit_meta.get("observed_redshift_open_interval") == [0.4, 1.1]
            and xi_fit_meta.get("model", {}).get("fit_ells_primary") == [0]
            and xi_fit_meta.get("output_npz_sha256") == sha256_file(paths["xi_fit"])
        ),
        "pk_fit16": bool(
            pk_stack_shape == (25, 16)
            and pk_covariance.shape == (16, 16)
            and np.allclose(pk_edges, TASK43_WIDE_LRGALL_FIT_EDGES, rtol=0.0, atol=1.0e-12)
            and np.linalg.eigvalsh(0.5 * (pk_covariance + pk_covariance.T))[0] > 0.0
        ),
        "pk_kmin_separation": bool(
            np.isclose(observed_kmin, 2.0 * math.pi / volume ** (1.0 / 3.0), rtol=0.0, atol=1.0e-15)
            and np.isclose(theory_kmin, WINDOW_THEORY_KMIN, rtol=0.0, atol=1.0e-15)
            and observed_kmin > theory_kmin
        ),
        "pk_window_theory_ell024": theory_ells == (0, 2, 4),
        "pk_summary_hash": pk_meta.get("output_sha256") == sha256_file(paths["pk_summary"]),
        "combined_observed_monopoles": bool(
            comparison.get("scope", {}).get("observed_multipoles") == [0]
            and "0.4<zobs<1.1" in comparison.get("scope", {}).get("geometry", "")
            and comparison.get("output_npz_sha256") == sha256_file(paths["comparison"])
        ),
        "geometry_comparison": geometry_comparison.get("status") == "pass",
        "pdf_only": all(path.is_file() and path.suffix == ".pdf" for path in plot_paths) and not png_outputs,
    }
    status = "pass" if all(gates.values()) else "fail"
    pk_posterior = comparison["pk0"]["mcmc"]["posterior"]
    xi_posterior = comparison["xi0"]["posterior"]
    outcomes = {
        "pk0": {
            "fNL": pk_posterior["fNL"],
            "mean_goodness": {
                "chi2": comparison["pk0"]["nominal"]["chi2_mean_covariance"],
                "dof": comparison["pk0"]["nominal"]["dof"],
                "pte": comparison["pk0"]["nominal"]["pte_mean_covariance"],
            },
        },
        "xi0": {
            "fNL": xi_posterior["fNL"],
            "mean_goodness": comparison["xi0"]["mean_goodness"],
        },
        "comparison": comparison["comparison"],
        "science_validation": comparison["science_validation"],
    }
    payload = {
        "task": "task43_audit_rsd_wide_lightcone_final",
        "status": status,
        "scope": "Task 4.3.2 0.4<zobs<1.1 monopole-only catalog-to-inference closure",
        "gates": gates,
        "outcomes": outcomes,
        "geometry_comparison_diagnostics": geometry_comparison.get("diagnostics_sge50", {}),
        "plot_paths": [str(path) for path in plot_paths],
        "plot_sha256": [sha256_file(path) for path in plot_paths if path.is_file()],
        "unexpected_png_outputs": png_outputs,
        "inputs": {name: str(path) for name, path in paths.items()},
    }
    atomic_savez(
        args.output.with_suffix(".npz"),
        pk0_fnl=np.asarray([pk_posterior["fNL"][key] for key in ("q16", "q50", "q84")]),
        xi0_fnl=np.asarray([xi_posterior["fNL"][key] for key in ("q16", "q50", "q84")]),
        pk0_pte=np.asarray(outcomes["pk0"]["mean_goodness"]["pte"]),
        xi0_pte=np.asarray(outcomes["xi0"]["mean_goodness"]["pte"]),
        observed_kmin=np.asarray(observed_kmin),
        theory_kmin=np.asarray(theory_kmin),
        volume=np.asarray(volume),
        zeff=np.asarray(xi_zeff),
    )
    payload["output_npz"] = str(args.output.with_suffix(".npz"))
    payload["output_npz_sha256"] = sha256_file(args.output.with_suffix(".npz"))
    atomic_write_json(args.output, payload)
    print(json.dumps({"status": status, "gates": gates, "outcomes": outcomes, "output": str(args.output)}, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
