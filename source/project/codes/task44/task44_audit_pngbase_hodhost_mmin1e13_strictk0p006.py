#!/usr/bin/env python3
"""Final cross-product audit for the current-range matched halo P0/xi0 result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from task44_pngbase_hodhost_mmin1e13_common import (
    CATALOGS,
    HALO_MASS_MIN_HMSUN,
    OUTPUT_ROOT,
    PK_STRICT_FIT_EDGES,
    PLOT_ROOT,
    PROJECT_ROOT,
    SUMMARY_DIR,
    XI_FIT_EDGES,
    XI_FIT_SMAX,
    XI_FIT_SMIN,
    atomic_write_json,
    get_spec,
    host_catalog_metadata_path,
    host_catalog_path,
    host_pk_fit_prefix,
    host_pk_metadata_path,
    host_pk_path,
    host_xi_fit_prefix,
    lrg_pk_gaussianp_fit_prefix,
    lrg_xi_fit_prefix,
    lrg_xi_gaussianp_fit_prefix,
    sha256_file,
    strict_fit_prefix,
    xi_engine_metadata_path,
    xi_engine_path,
)


CATALOG_AUDIT = SUMMARY_DIR / "task44_pngbase_hodhost_mmin1e13_catalog_audit.json"
XI_AUDIT = SUMMARY_DIR / "task44_pngbase_hodhost_mmin1e13_xi_three_engine_audit.json"
MATCHED_HALO_FIT_AUDIT = SUMMARY_DIR / "task44_pngbase_hodhost_mmin1e13_fixedfnl_pk0_vs_xi0_smax150_audit.json"
GAUSSIAN_P_FNL_AUDIT = SUMMARY_DIR / "task44_pngbase_hodhost_mmin1e13_gaussianp_freefnl_pk0_vs_xi0_audit.json"
FINAL_AUDIT = SUMMARY_DIR / "task44_pngbase_hodhost_mmin1e13_strictk0p006_final_audit.json"

EXPECTED_PDFS = (
    PLOT_ROOT / (
        "task44_pngbase_c300_c302_hodhost_mmin1e13_fixedfnl_"
        "pk0_kmin0p006_kmax0p100_xi0_smin30_smax150_bestfit_measurements.pdf"
    ),
    PLOT_ROOT / (
        "task44_pngbase_c300_c302_hodhost_mmin1e13_fixedfnl_"
        "pk0_kmin0p006_kmax0p100_vs_xi0_smin30_smax150_p_b1_contours.pdf"
    ),
    PLOT_ROOT / (
        "task44_pngbase_c300_c302_hodhost_mmin1e13_gaussianp_freefnl_"
        "pk0_kmin0p006_kmax0p100_xi0_smin30_smax150_bestfit_measurements.pdf"
    ),
    PLOT_ROOT / (
        "task44_pngbase_c300_c302_hodhost_mmin1e13_gaussianp_freefnl_"
        "pk0_kmin0p006_kmax0p100_vs_xi0_smin30_smax150_fnl_b1_p_contours.pdf"
    ),
    PLOT_ROOT / (
        "task44_pngbase_c300_c302_hodmap_lrg_fixedfnl_"
        "pk0_kmin0p006_kmax0p100_xi0_smin30_smax150_bestfit_measurements.pdf"
    ),
    PLOT_ROOT / (
        "task44_pngbase_c300_c302_hodmap_lrg_fixedfnl_"
        "pk0_kmin0p006_kmax0p100_vs_xi0_smin30_smax150_p_b1_contours.pdf"
    ),
    PLOT_ROOT / (
        "task44_pngbase_c300_c302_hodmap_lrg_gaussianp_freefnl_"
        "pk0_kmin0p006_kmax0p100_xi0_smin30_smax150_bestfit_measurements.pdf"
    ),
    PLOT_ROOT / (
        "task44_pngbase_c300_c302_hodmap_lrg_gaussianp_freefnl_"
        "pk0_kmin0p006_kmax0p100_vs_xi0_smin30_smax150_fnl_b1_p_contours.pdf"
    ),
)

CODE_FILES = (
    PROJECT_ROOT / "codes/task44/task44_pngbase_hodhost_mmin1e13_common.py",
    PROJECT_ROOT / "codes/task44/task44_build_pngbase_hodhost_mmin1e13.py",
    PROJECT_ROOT / "codes/task44/task44_measure_pngbase_hodhost_mmin1e13_xi.py",
    PROJECT_ROOT / "codes/task44/task44_measure_pngbase_hodhost_mmin1e13_pk.py",
    PROJECT_ROOT / "codes/task44/task44_fit_pngbase_hodhost_pk_strictk0p006.py",
    PROJECT_ROOT / "codes/task44/task44_fit_pngbase_hodhost_xi_smax150.py",
    PROJECT_ROOT / "codes/task44/task44_fit_pngbase_hodhost_gaussianp_freefnl.py",
    PROJECT_ROOT / "codes/task44/task44_plot_audit_pngbase_hodhost_xi_smax150.py",
    PROJECT_ROOT / "codes/task44/task44_plot_audit_pngbase_hodhost_gaussianp_freefnl.py",
    PROJECT_ROOT / "codes/task44/task44_plot_pngbase_hodmap_lrg_strictk0p006.py",
    PROJECT_ROOT / "codes/task44/task44_audit_pngbase_hodhost_mmin1e13_strictk0p006.py",
    PROJECT_ROOT / "codes/task44/run_task44_pngbase_hodhost_mmin1e13_xi_cucount_gpu.sbatch",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_fit_summary_and_arrays(prefix: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    json_path = prefix.with_suffix(".json")
    npz_path = prefix.with_suffix(".npz")
    summary = _load_json(json_path)
    if summary.get("output_npz_sha256") != sha256_file(npz_path):
        raise RuntimeError(f"fit hash mismatch: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files if name != "summary_json"}
    return summary, arrays


def build_audit() -> dict[str, Any]:
    catalog_audit = _load_json(CATALOG_AUDIT)
    xi_audit = _load_json(XI_AUDIT)
    matched_halo_fit_audit = _load_json(MATCHED_HALO_FIT_AUDIT)
    gaussian_p_fnl_audit = _load_json(GAUSSIAN_P_FNL_AUDIT)
    rows: dict[str, Any] = {}
    per_catalog_gates: list[bool] = []

    for tag in CATALOGS:
        catalog_metadata = _load_json(host_catalog_metadata_path(tag))
        halo_pk_metadata = _load_json(host_pk_metadata_path(tag))
        halo_pk_fit, halo_pk_arrays = _load_fit_summary_and_arrays(host_pk_fit_prefix(tag))
        halo_xi_fit, halo_xi_arrays = _load_fit_summary_and_arrays(host_xi_fit_prefix(tag))
        lrg_pk_fit, lrg_pk_arrays = _load_fit_summary_and_arrays(strict_fit_prefix(tag))
        lrg_xi_fit, lrg_xi_arrays = _load_fit_summary_and_arrays(lrg_xi_fit_prefix(tag))
        lrg_pk_gaussian, lrg_pk_gaussian_arrays = _load_fit_summary_and_arrays(
            lrg_pk_gaussianp_fit_prefix(tag)
        )
        lrg_xi_gaussian, lrg_xi_gaussian_arrays = _load_fit_summary_and_arrays(
            lrg_xi_gaussianp_fit_prefix(tag)
        )

        engines: dict[str, Any] = {}
        for engine in ("fcfc", "pycorr", "cucount"):
            metadata = _load_json(xi_engine_metadata_path(tag, engine))
            path = xi_engine_path(tag, engine)
            engines[engine] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "metadata_status": metadata.get("status"),
                "backend": metadata.get("engine_metadata", {}).get("backend"),
            }

        with np.load(xi_engine_path(tag, "fcfc"), allow_pickle=False) as data:
            halo_edges = np.asarray(data["s_edges"], dtype="f8")

        p_halo_pk = halo_pk_fit["mcmc"]["posterior"]["p"]
        p_halo_xi = halo_xi_fit["mcmc"]["posterior"]["p"]
        source_host_path = host_catalog_path(tag)
        source_host_sha256 = sha256_file(source_host_path)
        halo_xi_metadata = _load_json(xi_engine_metadata_path(tag, "fcfc"))
        row_gates = {
            "host_catalog_status_pass": catalog_metadata.get("status") == "pass",
            "host_catalog_hash_valid": catalog_metadata.get("output_sha256") == sha256_file(host_catalog_path(tag)),
            "host_mass_cut_is_exactly_1e13": float(catalog_metadata["selection"]["halo_mass_min_hmsun"]) == HALO_MASS_MIN_HMSUN,
            "host_particle_cut_is_Ncleaned_ge_987": int(catalog_metadata["selection"]["particle_n_min"]) == 987,
            "host_sample_declared_occupied_subset_not_all_halos": bool(catalog_metadata["selection"]["not_a_complete_mass_threshold_catalog"]),
            "all_three_xi_products_pass_and_hash": all(
                row["metadata_status"] == "pass" and row["sha256"] == _load_json(xi_engine_metadata_path(tag, engine))["output_sha256"]
                for engine, row in engines.items()
            ),
            "cucount_used_gpu_backend": engines["cucount"]["backend"] == "gpu",
            "halo_xi_measurement_contains_exact_s30_to_s150_prefix": bool(
                np.array_equal(halo_edges[: XI_FIT_EDGES.size], XI_FIT_EDGES)
            ),
            "halo_pk_measurement_status_hash_and_source_pass": bool(
                halo_pk_metadata.get("status") == "pass"
                and halo_pk_metadata.get("output_sha256") == sha256_file(host_pk_path(tag))
                and halo_pk_metadata.get("source_host_catalog") == str(source_host_path)
                and halo_pk_metadata.get("source_host_catalog_sha256") == source_host_sha256
            ),
            "halo_pk_fit_status_and_all_gates_pass": bool(
                halo_pk_fit.get("status") == "pass" and all(halo_pk_fit["gates"].values())
            ),
            "halo_pk_fit_fixed_correct_fnl": float(halo_pk_fit["fixed_fnl"]) == get_spec(tag).fnl,
            "halo_pk_fit_uses_exact_48_edges": bool(
                np.array_equal(np.asarray(halo_pk_arrays["coordinate_edges"], dtype="f8"), PK_STRICT_FIT_EDGES)
                and str(np.asarray(halo_pk_arrays["tracer"]).item()) == "hodhost_mmin1e13"
            ),
            "halo_xi_fit_status_and_all_gates_pass": bool(
                halo_xi_fit.get("status") == "pass" and all(halo_xi_fit["gates"].values())
            ),
            "halo_xi_fit_fixed_correct_fnl": float(halo_xi_fit["fixed_fnl"]) == get_spec(tag).fnl,
            "halo_xi_fit_uses_exact_s30_to_s150_edges": bool(
                np.array_equal(np.asarray(halo_xi_arrays["radial_edges"], dtype="f8"), XI_FIT_EDGES)
                and np.asarray(halo_xi_arrays["data"]).size == 12
            ),
            "halo_pk_and_xi_use_identical_source_catalog": bool(
                halo_xi_metadata.get("source_host_catalog") == str(source_host_path)
                and halo_xi_metadata.get("source_host_catalog_sha256") == source_host_sha256
                and halo_pk_fit["input"]["source_host_catalog"] == str(source_host_path)
                and halo_pk_fit["input"]["source_host_catalog_sha256"] == source_host_sha256
            ),
            "halo_pk_and_xi_ndata_match_selected_hosts": bool(
                int(halo_pk_fit["input"]["ndata"])
                == int(halo_xi_fit["input"]["ndata"])
                == int(catalog_metadata["counts"]["n_selected_hod_hosts"])
            ),
            "lrg_fixed_pk_and_xi_fits_pass": bool(
                lrg_pk_fit.get("status") == "pass"
                and lrg_xi_fit.get("status") == "pass"
                and all(lrg_pk_fit["gates"].values())
                and all(lrg_xi_fit["gates"].values())
            ),
            "lrg_fixed_pk_and_xi_use_current_ranges": bool(
                np.array_equal(np.asarray(lrg_pk_arrays["coordinate_edges"], dtype="f8"), PK_STRICT_FIT_EDGES)
                and np.array_equal(np.asarray(lrg_xi_arrays["radial_edges"], dtype="f8"), XI_FIT_EDGES)
                and np.asarray(lrg_pk_arrays["data"]).size == 48
                and np.asarray(lrg_xi_arrays["data"]).size == 12
            ),
            "lrg_fixed_pk_and_xi_use_catalog_fnl": bool(
                float(lrg_pk_fit["fixed_fnl"]) == float(lrg_xi_fit["fixed_fnl"]) == get_spec(tag).fnl
            ),
            "lrg_gaussianp_freefnl_pk_and_xi_fits_pass": bool(
                lrg_pk_gaussian.get("status") == "pass"
                and lrg_xi_gaussian.get("status") == "pass"
                and all(lrg_pk_gaussian["gates"].values())
                and all(lrg_xi_gaussian["gates"].values())
            ),
            "lrg_gaussianp_freefnl_has_three_contour_parameters": bool(
                all(name in lrg_pk_gaussian["free_parameters"] for name in ("fnl", "b1", "p"))
                and all(name in lrg_xi_gaussian["free_parameters"] for name in ("fnl", "b1", "p"))
                and tuple(str(value) for value in lrg_pk_gaussian_arrays["parameter_names"][:3])
                == ("b1", "fnl", "p")
                and tuple(str(value) for value in lrg_xi_gaussian_arrays["parameter_names"])
                == ("b1", "fnl", "p")
            ),
        }
        per_catalog_gates.extend(row_gates.values())
        xi_engine_row = xi_audit["catalogs"][tag]["engines"]
        rows[tag] = {
            "catalog_baseline_fnl": get_spec(tag).fnl,
            "host_sample": {
                "n_unique_occupied_before_cut": int(catalog_metadata["counts"]["n_unique_hod_hosts_before_mass_cut"]),
                "n_selected": int(catalog_metadata["counts"]["n_selected_hod_hosts"]),
                "selected_fraction": float(
                    catalog_metadata["counts"]["n_selected_hod_hosts"]
                    / catalog_metadata["counts"]["n_unique_hod_hosts_before_mass_cut"]
                ),
                "nbar_h3_mpc3": float(catalog_metadata["counts"]["nbar_selected_h3_mpc3"]),
                "particle_mass_hmsun": float(catalog_metadata["selection"]["particle_mass_hmsun"]),
                "particle_n_min": int(catalog_metadata["selection"]["particle_n_min"]),
                "effective_minimum_mass_hmsun": float(catalog_metadata["selection"]["effective_minimum_mass_hmsun"]),
                "interpretation": catalog_metadata["selection"]["interpretation"],
            },
            "halo_xi_three_engine": {
                "max_abs_xi0_pycorr_vs_fcfc": float(xi_engine_row["pycorr"]["max_abs_xi0_vs_fcfc"]),
                "max_abs_xi0_cucount_vs_fcfc": float(xi_engine_row["cucount"]["max_abs_xi0_vs_fcfc"]),
                "max_abs_raw_DD_pycorr_vs_fcfc": float(xi_engine_row["pycorr"]["max_abs_raw_DD_vs_fcfc"]),
                "max_abs_raw_DD_cucount_vs_fcfc": float(xi_engine_row["cucount"]["max_abs_raw_DD_vs_fcfc"]),
                "engines": engines,
            },
            "halo_pk_fit_kmin0p006_kmax0p100": {
                "fixed_fnl": float(halo_pk_fit["fixed_fnl"]),
                "p": p_halo_pk,
                "b1": halo_pk_fit["mcmc"]["posterior"]["b1"],
                "chi2": float(halo_pk_fit["map"]["chi2"]),
                "dof": int(halo_pk_fit["map"]["dof"]),
                "pte": float(halo_pk_fit["map"]["pte"]),
                "mcmc_gates": halo_pk_fit["mcmc"]["gates"],
            },
            "halo_xi_fit_smin30_smax150": {
                "fixed_fnl": float(halo_xi_fit["fixed_fnl"]),
                "p": p_halo_xi,
                "b1": halo_xi_fit["mcmc"]["posterior"]["b1"],
                "chi2": float(halo_xi_fit["map"]["chi2"]),
                "dof": int(halo_xi_fit["map"]["dof"]),
                "pte": float(halo_xi_fit["map"]["pte"]),
                "smin_mpc_h": XI_FIT_SMIN,
                "smax_mpc_h": XI_FIT_SMAX,
                "mcmc_gates": halo_xi_fit["mcmc"]["gates"],
            },
            "matched_halo_pk_vs_xi": {
                "delta_p_q50_pk_minus_xi": float(p_halo_pk["q50"] - p_halo_xi["q50"]),
                "warning": "same halo catalog and realization; P0 and xi0 errors are correlated",
            },
            "gaussian_p_free_fnl_test": gaussian_p_fnl_audit["catalogs"][tag],
            "lrg_fixed_fnl_test": {
                "pk": {"p": lrg_pk_fit["mcmc"]["posterior"]["p"], "map": lrg_pk_fit["map"]},
                "xi": {"p": lrg_xi_fit["mcmc"]["posterior"]["p"], "map": lrg_xi_fit["map"]},
            },
            "lrg_gaussian_p_free_fnl_test": {
                "pk": {
                    "fnl": lrg_pk_gaussian["mcmc"]["posterior"]["fnl"],
                    "p": lrg_pk_gaussian["mcmc"]["posterior"]["p"],
                },
                "xi": {
                    "fnl": lrg_xi_gaussian["mcmc"]["posterior"]["fnl"],
                    "p": lrg_xi_gaussian["mcmc"]["posterior"]["p"],
                },
            },
            "gates": row_gates,
        }

    pdfs_found = tuple(sorted(PLOT_ROOT.glob("*.pdf")))
    pngs_found = tuple(sorted(PLOT_ROOT.rglob("*.png")))
    global_gates = {
        "catalog_audit_pass": catalog_audit.get("status") == "pass" and all(catalog_audit["gates"].values()),
        "xi_three_engine_audit_pass": xi_audit.get("status") == "pass" and all(xi_audit["global_gates"].values()),
        "matched_halo_pk_xi_fit_audit_pass": bool(
            matched_halo_fit_audit.get("status") == "pass"
            and all(matched_halo_fit_audit["global_gates"].values())
        ),
        "gaussian_p_free_fnl_audit_pass": bool(
            gaussian_p_fnl_audit.get("status") == "pass"
            and all(gaussian_p_fnl_audit["global_gates"].values())
        ),
        "all_per_catalog_product_gates_pass": bool(all(per_catalog_gates)),
        "exactly_eight_current_range_halo_and_lrg_pdfs": pdfs_found == tuple(sorted(EXPECTED_PDFS)),
        "no_png_outputs": len(pngs_found) == 0,
        "no_smin50_products_in_extension": not any("smin50" in str(path) for path in OUTPUT_ROOT.rglob("*")),
        "all_code_files_exist": all(path.is_file() for path in CODE_FILES),
    }
    audit: dict[str, Any] = {
        "task": "task44_pngbase_hodhost_mmin1e13_strictk0p006_final_audit",
        "status": "pass" if all(global_gates.values()) else "review",
        "scope": {
            "halo_pk": "the exact same unique occupied HOD-host catalog used by halo xi; M>=1e13 Msun/h; unit weight; k>=0.006 h/Mpc",
            "halo_xi": "the exact same unique occupied HOD-host catalog used by halo P0; M>=1e13 Msun/h; unit weight; 30<=s<150 Mpc/h",
            "lrg_pk": "HOD-MAP LRG catalog; 0.006<=k and k-bin centre<=0.100 h/Mpc",
            "lrg_xi": "the same HOD-MAP LRG catalog; 30<=s<150 Mpc/h fitted and measurement displayed to 350 Mpc/h",
            "fixed_fnl": {tag: get_spec(tag).fnl for tag in CATALOGS},
            "additional_test": (
                "free fNL with p~N(0.7072363788448802, 0.2694701311625559^2) for both c300 and c302"
            ),
            "separation_fit_edges_mpc_h": XI_FIT_EDGES.tolist(),
            "pk_edges_h_mpc": PK_STRICT_FIT_EDGES.tolist(),
            "output_isolation": str(OUTPUT_ROOT),
        },
        "important_sample_distinction": (
            "The matched halo P0 and 2PCF sample is a mass-cut subset of halos actually occupied in the exact HOD-MAP catalogs; "
            "it is not the complete raw CompaSO all-halo M>=1e13 catalog."
        ),
        "fit_quality_policy": "chi2/dof and PTE are reported diagnostics and are not pass/fail gates",
        "source_audits": {
            "catalog": {"path": str(CATALOG_AUDIT), "sha256": sha256_file(CATALOG_AUDIT)},
            "xi_three_engine": {"path": str(XI_AUDIT), "sha256": sha256_file(XI_AUDIT)},
            "matched_halo_pk_xi": {"path": str(MATCHED_HALO_FIT_AUDIT), "sha256": sha256_file(MATCHED_HALO_FIT_AUDIT)},
            "gaussian_p_free_fnl": {"path": str(GAUSSIAN_P_FNL_AUDIT), "sha256": sha256_file(GAUSSIAN_P_FNL_AUDIT)},
        },
        "global_gates": global_gates,
        "catalogs": rows,
        "plots": [{"path": str(path), "sha256": sha256_file(path)} for path in EXPECTED_PDFS],
        "code": [{"path": str(path), "sha256": sha256_file(path)} for path in CODE_FILES],
        "scheduler": {
            "cucount_array_job": "57408968",
            "reported_state": "COMPLETED for array indices 0 and 1",
        },
        "independence_warning": "c300 and c302 share phase ph000 and are not multiplied as independent likelihoods",
    }
    atomic_write_json(FINAL_AUDIT, audit)
    if audit["status"] != "pass":
        raise RuntimeError(f"final audit requires review: {global_gates}")
    return audit


def main() -> None:
    audit = build_audit()
    print(f"[done] final audit status={audit['status']} path={FINAL_AUDIT}", flush=True)
    for tag, row in audit["catalogs"].items():
        p_pk = row["halo_pk_fit_kmin0p006_kmax0p100"]["p"]
        p_xi = row["halo_xi_fit_smin30_smax150"]["p"]
        xi = row["halo_xi_three_engine"]
        print(
            f"[result] {tag} Nhost={row['host_sample']['n_selected']} "
            f"halo P0 p={p_pk['q50']:.4f} -{p_pk['q50']-p_pk['q16']:.4f}/+{p_pk['q84']-p_pk['q50']:.4f}; "
            f"halo xi p={p_xi['q50']:.4f} -{p_xi['q50']-p_xi['q16']:.4f}/+{p_xi['q84']-p_xi['q50']:.4f}; "
            f"max|delta xi|={max(xi['max_abs_xi0_pycorr_vs_fcfc'], xi['max_abs_xi0_cucount_vs_fcfc']):.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
