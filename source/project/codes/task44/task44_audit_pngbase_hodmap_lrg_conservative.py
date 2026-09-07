#!/usr/bin/env python3
"""Audit the conservative-scale LRG-only Task44 fits and four official PDFs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from task44_pngbase_hodhost_mmin1e13_common import (
    CATALOGS,
    LRG_EXTENDED_XI_FIT_EDGES,
    LRG_EXTENDED_XI_SMAX,
    LRG_EXTENDED_XI_SMIN,
    LRG_CONSERVATIVE_PK_FIT_EDGES,
    LRG_CONSERVATIVE_PK_KMAX,
    P_GAUSSIAN_PRIOR_MEAN,
    P_GAUSSIAN_PRIOR_SIGMA,
    PK_STRICT_KMIN,
    PLOT_ROOT,
    SUMMARY_DIR,
    atomic_write_json,
    get_spec,
    lrg_pk_conservative_fit_prefix,
    lrg_pk_conservative_gaussianp_fit_prefix,
    lrg_xi_extended_fit_prefix,
    lrg_xi_extended_gaussianp_fit_prefix,
    sha256_file,
    strict_pk_metadata_path,
    strict_pk_path,
)
from task44_pngbase_hodmap_rawbox_common import xi_metadata_path, xi_path


EXPECTED_PDFS = (
    PLOT_ROOT
    / "task44_pngbase_c300_c302_hodmap_lrg_fixedfnl_pk0_kmin0p006_kmax0p080_xi0_smin30_smax350_bestfit_measurements.pdf",
    PLOT_ROOT
    / "task44_pngbase_c300_c302_hodmap_lrg_fixedfnl_pk0_kmin0p006_kmax0p080_vs_xi0_smin30_smax350_p_b1_contours.pdf",
    PLOT_ROOT
    / "task44_pngbase_c300_c302_hodmap_lrg_gaussianp_freefnl_pk0_kmin0p006_kmax0p080_xi0_smin30_smax350_bestfit_measurements.pdf",
    PLOT_ROOT
    / "task44_pngbase_c300_c302_hodmap_lrg_gaussianp_freefnl_pk0_kmin0p006_kmax0p080_vs_xi0_smin30_smax350_fnl_b1_p_contours.pdf",
)
AUDIT_JSON = SUMMARY_DIR / "task44_pngbase_hodmap_lrg_kmax0p080_smin30_smax350_audit.json"


def load_fit(prefix: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    json_path = prefix.with_suffix(".json")
    npz_path = prefix.with_suffix(".npz")
    if not json_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(f"missing fit: {json_path} / {npz_path}")
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    if summary.get("output_npz_sha256") != sha256_file(npz_path):
        raise RuntimeError(f"fit hash mismatch: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files if name != "summary_json"}
    return summary, arrays


def posterior(summary: dict[str, Any], name: str) -> dict[str, float]:
    return {key: float(value) for key, value in summary["mcmc"]["posterior"][name].items()}


def build_audit() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    all_catalog_gates: list[bool] = []
    for tag in CATALOGS:
        fixed_pk, fixed_pk_arrays = load_fit(lrg_pk_conservative_fit_prefix(tag))
        fixed_xi, fixed_xi_arrays = load_fit(lrg_xi_extended_fit_prefix(tag))
        gaussian_pk, gaussian_pk_arrays = load_fit(lrg_pk_conservative_gaussianp_fit_prefix(tag))
        gaussian_xi, gaussian_xi_arrays = load_fit(lrg_xi_extended_gaussianp_fit_prefix(tag))
        pk_measurement = strict_pk_path(tag)
        pk_metadata = json.loads(strict_pk_metadata_path(tag).read_text(encoding="utf-8"))
        xi_measurement = xi_path(tag, smin=30.0)
        xi_metadata = json.loads(xi_metadata_path(tag, smin=30.0).read_text(encoding="utf-8"))
        with np.load(xi_measurement, allow_pickle=False) as data:
            full_xi_edges = np.asarray(data["s_edges"], dtype="f8")
            full_xi_centers = np.asarray(data["s"], dtype="f8")

        gates = {
            "all_four_fit_statuses_pass": all(
                fit.get("status") == "pass" for fit in (fixed_pk, fixed_xi, gaussian_pk, gaussian_xi)
            ),
            "all_four_fit_gate_sets_pass": all(
                all(fit["gates"].values()) for fit in (fixed_pk, fixed_xi, gaussian_pk, gaussian_xi)
            ),
            "fixed_fnl_matches_each_catalog": bool(
                float(fixed_pk["fixed_fnl"]) == float(fixed_xi["fixed_fnl"]) == get_spec(tag).fnl
            ),
            "pk_edges_are_exact_38_shell_conservative_contract": bool(
                np.array_equal(fixed_pk_arrays["coordinate_edges"], LRG_CONSERVATIVE_PK_FIT_EDGES)
                and np.array_equal(gaussian_pk_arrays["coordinate_edges"], LRG_CONSERVATIVE_PK_FIT_EDGES)
                and fixed_pk_arrays["data"].size == gaussian_pk_arrays["data"].size == 38
            ),
            "pk_last_bin_center_is_exactly_0p080": bool(
                np.mean(LRG_CONSERVATIVE_PK_FIT_EDGES[-1]) == LRG_CONSERVATIVE_PK_KMAX
            ),
            "xi_edges_are_exact_32_shell_smin30_smax350_contract": bool(
                np.array_equal(fixed_xi_arrays["radial_edges"], LRG_EXTENDED_XI_FIT_EDGES)
                and np.array_equal(gaussian_xi_arrays["radial_edges"], LRG_EXTENDED_XI_FIT_EDGES)
                and fixed_xi_arrays["data"].size == gaussian_xi_arrays["data"].size == 32
            ),
            "pk_and_xi_use_same_lrg_catalog_size": bool(
                int(fixed_pk["input"]["ndata"])
                == int(fixed_xi["input"]["ndata"])
                == int(gaussian_pk["input"]["ndata"])
                == int(gaussian_xi["input"]["ndata"])
                == get_spec(tag).expected_ngal
            ),
            "gaussian_test_samples_fnl_b1_p": bool(
                all(name in gaussian_pk["free_parameters"] for name in ("fnl", "b1", "p"))
                and all(name in gaussian_xi["free_parameters"] for name in ("fnl", "b1", "p"))
            ),
            "gaussian_p_prior_is_exact": bool(
                gaussian_pk["p_gaussian_prior"]["mean"]
                == gaussian_xi["p_gaussian_prior"]["mean"]
                == P_GAUSSIAN_PRIOR_MEAN
                and gaussian_pk["p_gaussian_prior"]["sigma"]
                == gaussian_xi["p_gaussian_prior"]["sigma"]
                == P_GAUSSIAN_PRIOR_SIGMA
            ),
            "validated_pk_measurement_hash": bool(
                pk_metadata.get("status") == "pass"
                and pk_metadata.get("output_sha256") == sha256_file(pk_measurement)
            ),
            "validated_xi_s30_measurement_hash_and_source": bool(
                xi_metadata.get("status") == "pass"
                and xi_metadata.get("output_sha256") == sha256_file(xi_measurement)
                and xi_metadata.get("source_hdf5", xi_metadata.get("input_hdf5")) == get_spec(tag).path
            ),
            "full_xi_measurement_and_fit_contract_is_30_to_350": bool(
                np.array_equal(full_xi_edges, np.arange(30.0, 351.0, 10.0))
                and np.array_equal(full_xi_centers, np.arange(35.0, 350.0, 10.0))
                and np.array_equal(fixed_xi_arrays["coordinate"], full_xi_centers)
                and np.array_equal(gaussian_xi_arrays["coordinate"], full_xi_centers)
            ),
        }
        all_catalog_gates.extend(gates.values())
        rows[tag] = {
            "input_fnl": get_spec(tag).fnl,
            "fixed_fnl": {
                "pk": {"p": posterior(fixed_pk, "p"), "chi2": fixed_pk["map"]["chi2"], "dof": fixed_pk["map"]["dof"]},
                "xi": {"p": posterior(fixed_xi, "p"), "chi2": fixed_xi["map"]["chi2"], "dof": fixed_xi["map"]["dof"]},
            },
            "gaussian_p_free_fnl": {
                "pk": {"fnl": posterior(gaussian_pk, "fnl"), "p": posterior(gaussian_pk, "p")},
                "xi": {"fnl": posterior(gaussian_xi, "fnl"), "p": posterior(gaussian_xi, "p")},
            },
            "gates": gates,
        }

    present_files = tuple(sorted(path for path in PLOT_ROOT.iterdir() if path.is_file()))
    expected_files = tuple(sorted(EXPECTED_PDFS))
    global_gates = {
        "all_catalog_fit_and_measurement_gates_pass": bool(all(all_catalog_gates)),
        "kmin_is_0p006": PK_STRICT_KMIN == 0.006,
        "kmax_is_0p080": LRG_CONSERVATIVE_PK_KMAX == 0.080,
        "smin_is_30": LRG_EXTENDED_XI_SMIN == 30.0,
        "smax_is_350": LRG_EXTENDED_XI_SMAX == 350.0,
        "exactly_four_current_lrg_pdfs_in_official_directory": present_files == expected_files,
        "all_four_pdfs_exist_and_have_pdf_header": all(
            path.is_file() and path.read_bytes()[:4] == b"%PDF" for path in EXPECTED_PDFS
        ),
        "no_png_in_official_directory": not any(PLOT_ROOT.glob("*.png")),
        "no_old_lrg_range_pdf_in_official_directory": not any(
            "hodmap_lrg" in path.name
            and ("kmax0p100" in path.name or "smin50" in path.name or "smax150" in path.name)
            for path in present_files
        ),
    }
    audit = {
        "task": "task44_pngbase_hodmap_lrg_conservative_scale_audit",
        "status": "pass" if all(global_gates.values()) else "review",
        "scope": {
            "sample": "HOD-MAP LRGs only; halo plots and fits are outside this rerun",
            "pk": "0.006 <= k, final bin centre 0.080 h/Mpc; 38 exact periodic shells",
            "xi": "30 <= s < 350 Mpc/h; all 32 native FCFC shells are fitted and displayed",
        },
        "global_gates": global_gates,
        "catalogs": rows,
        "plots": [{"path": str(path), "sha256": sha256_file(path)} for path in EXPECTED_PDFS],
    }
    atomic_write_json(AUDIT_JSON, audit)
    if audit["status"] != "pass":
        raise RuntimeError(f"LRG conservative-scale audit requires review: {global_gates}")
    return audit


def main() -> None:
    audit = build_audit()
    print(f"[done] LRG conservative-scale audit status={audit['status']} path={AUDIT_JSON}")
    for tag, row in audit["catalogs"].items():
        p_pk = row["fixed_fnl"]["pk"]["p"]
        p_xi = row["fixed_fnl"]["xi"]["p"]
        print(f"[result] {tag} fixed-fNL P0 p={p_pk['q50']:.4f}; xi0 p={p_xi['q50']:.4f}")


if __name__ == "__main__":
    main()
