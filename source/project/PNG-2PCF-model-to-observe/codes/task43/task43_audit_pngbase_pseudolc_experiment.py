#!/usr/bin/env python3
"""Final cross-product audit for the paired pngbase pseudo-lightcone run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from task43_pngbase_pseudolc_common import (
    COSMOLOGIES,
    FINAL_ROOT,
    FNL_BY_COSMOLOGY,
    OBSERVER_MPC_H,
    PHASES,
    SEED_BY_PHASE,
    SPACE_WINDOWS,
    catalog_path,
    fit_root,
    fkp_path,
    pk_path,
    random_path,
    xi_path,
)
from task43_rsd_common import atomic_write_json, sha256_file
from task43_run_pngbase_pseudolc_joint import VARIANT_ORDER


def validated(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(".json")
    if not (path.is_file() and metadata_path.is_file()):
        raise FileNotFoundError(f"missing product/metadata: {path} / {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "pass" or metadata.get("output_sha256") != sha256_file(path):
        raise RuntimeError(f"unvalidated product: {path}")
    return metadata


def main() -> None:
    output = FINAL_ROOT / "task43_pngbase_pseudolc_fnl0_fnl100_experiment_audit.json"
    if output.exists():
        raise FileExistsError(f"immutable final audit exists: {output}")
    products: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    for cosmology in COSMOLOGIES:
        for phase in PHASES:
            for space in SPACE_WINDOWS:
                key = f"{cosmology}_{phase}_{space}"
                paths = {
                    "catalog": catalog_path(cosmology, phase, space),
                    "random": random_path(cosmology, phase, space),
                    "fkp": fkp_path(cosmology, phase, space),
                    "pk": pk_path(cosmology, phase, space),
                    "xi": xi_path(cosmology, phase, space),
                }
                metadata = {name: validated(path) for name, path in paths.items()}
                catalog_meta = metadata["catalog"]
                zmin, zmax = SPACE_WINDOWS[space]
                gates[f"{key}_geometry"] = bool(
                    catalog_meta["positive_octant_gate"]
                    and catalog_meta["geometry"]["observer_mpc_h"] == list(OBSERVER_MPC_H)
                    and catalog_meta["z_observed"]["min"] > zmin
                    and catalog_meta["z_observed"]["max"] < zmax
                )
                gates[f"{key}_fnl"] = float(catalog_meta["injected_fnl"]) == FNL_BY_COSMOLOGY[cosmology]
                gates[f"{key}_seed"] = int(catalog_meta["initial_condition_seed"]) == SEED_BY_PHASE[phase]
                xi_meta = metadata["xi"]
                gates[f"{key}_xi_cpu_engine"] = xi_meta.get("engine") == "pycorr/Corrfunc CPU"
                gates[f"{key}_xi_threads_le_8"] = 1 <= int(xi_meta.get("nthreads", 0)) <= 8
                products[key] = {
                    name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()
                }

    covariances: dict[str, dict[str, np.ndarray]] = {}
    fit_audits: dict[str, Any] = {}
    for cosmology in COSMOLOGIES:
        root = fit_root(cosmology)
        audit_path = root / "task43_pngbase_pseudolc_joint_baomask80_120.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        gates[f"{cosmology}_fit_audit_pass"] = audit.get("status") == "pass"
        gates[f"{cosmology}_two_phase_mean_contract"] = bool(
            audit.get("mean_realization_count") == 2
            and audit.get("single_phase_fnl_fits") == []
            and audit["covariance_contract"].get("covariance_divisor") == 1.0
        )
        gates[f"{cosmology}_fit_workers_le_8"] = bool(
            1 <= int(audit.get("mcmc", {}).get("parallel_workers", 0)) <= 8
        )
        mean_gates = []
        for variant in VARIANT_ORDER:
            path = root / "fits" / variant / "samples.npz"
            metadata_path = path.with_name("summary.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            with np.load(path, allow_pickle=False) as payload:
                phase_data = np.asarray(payload["phase_data"], dtype="f8")
                data = np.asarray(payload["data"], dtype="f8")
            mean_gates.append(
                bool(
                    phase_data.shape[0] == 2
                    and np.array_equal(data, np.mean(phase_data, axis=0))
                    and metadata.get("single_phase_fnl_fit") is False
                )
            )
        gates[f"{cosmology}_all_variant_means_exact"] = all(mean_gates)
        covariance_path = root / "task43_pngbase_pseudolc_joint_baomask80_120_covariance.npz"
        with np.load(covariance_path, allow_pickle=False) as payload:
            covariances[cosmology] = {key: np.asarray(payload[key]) for key in payload.files}
        fit_audits[cosmology] = {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "mean_realization_count": audit["mean_realization_count"],
            "phases": audit["phases"],
            "mcmc": audit["mcmc"],
            "metrics": audit["metrics"],
            "posterior_fNL": {
                variant: audit["results"][variant]["mcmc"]["posterior"]["fNL"]
                for variant in VARIANT_ORDER
            },
        }

    gates["c000_c302_covariance_arrays_identical"] = bool(
        covariances["c000"].keys() == covariances["c302"].keys()
        and all(
            np.array_equal(covariances["c000"][key], covariances["c302"][key])
            for key in covariances["c000"]
        )
    )
    unexpected_phase_fits = sorted(
        str(path) for path in (fit_root("c000").parent.parent).glob("**/fits/ph*/samples.npz")
    )
    gates["no_single_phase_fit_products"] = len(unexpected_phase_fits) == 0
    pdfs = sorted(FINAL_ROOT.glob("*.pdf"))
    pngs = sorted(FINAL_ROOT.glob("*.png"))
    expected_pdfs = {
        "task43_pngbase_c000_pseudolc_real_Pxi_rsd_P02xi02_joint_kmax0p08_smin50_baomask80_120.pdf",
        "task43_pngbase_c302_pseudolc_real_Pxi_rsd_P02xi02_joint_kmax0p08_smin50_baomask80_120.pdf",
    }
    gates["two_expected_pdfs"] = {path.name for path in pdfs} == expected_pdfs
    gates["no_png_outputs"] = not pngs
    gates["all_pdfs_have_pdf_header"] = all(path.read_bytes()[:5] == b"%PDF-" for path in pdfs)
    status = "pass" if all(gates.values()) else "fail"
    summary = {
        "task": "task43_audit_pngbase_pseudolc_experiment",
        "status": status,
        "scientific_target": "test c302 P-versus-xi consistency in simple snapshot-shell octant geometry; retain c000 as a supplementary matched baseline",
        "fit_policy": "fit only c000 and c302 ph000/ph001 arithmetic observable means; no phase-level fNL fits",
        "covariance_policy": "Task4.3 single-realization covariance, identical for c000/c302 and never divided by two",
        "primary_consistency_result": {
            "cosmology": "c302",
            "comparison_definition": "P-versus-xi marginal posterior tension ignores their cross-correlation and is therefore diagnostic only",
            "metrics": fit_audits["c302"]["metrics"],
            "posterior_fNL": fit_audits["c302"]["posterior_fNL"],
        },
        "supplementary_baseline": {
            "cosmology": "c000",
            "realization_count": fit_audits["c000"]["mean_realization_count"],
            "phases": fit_audits["c000"]["phases"],
            "posterior_fNL": fit_audits["c000"]["posterior_fNL"],
        },
        "gates": gates,
        "unexpected_single_phase_fit_products": unexpected_phase_fits,
        "products": products,
        "fit_audits": fit_audits,
        "final_pdfs": [{"path": str(path), "sha256": sha256_file(path)} for path in pdfs],
        "final_pngs": [str(path) for path in pngs],
    }
    atomic_write_json(output, summary)
    print(json.dumps({"status": status, "output": str(output), "gates": gates}, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
