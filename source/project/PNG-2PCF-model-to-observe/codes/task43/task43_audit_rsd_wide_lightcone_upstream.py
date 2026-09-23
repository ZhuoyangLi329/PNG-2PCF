#!/usr/bin/env python3
"""Audit the x25 wide-lightcone catalog/random/n(z)/FKP production contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from task43_build_rsd_lightcone_random import repair_float32_redshift_boundaries
from task43_rsd_common import OUTPUT_ROOT, PHASES, atomic_savez, atomic_write_json, sha256_file


WIDE_ROOT = OUTPUT_ROOT / "lightcone_wide_zobs0p4_1p1"
DEFAULT_MANIFEST = OUTPUT_ROOT / "manifests" / "task43_rsd_validation_lightcone_wide_zobs0p4_1p1_x25.jsonl"
DEFAULT_OUTPUT = OUTPUT_ROOT / "audits" / "task43_rsd_wide_lightcone_upstream_x25_audit.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    if tuple(row["phase"] for row in rows) != PHASES:
        raise RuntimeError("wide manifest phase order is not ph000..ph024")

    phase_rows: list[dict[str, Any]] = []
    ndata_values: list[int] = []
    zeff_values: list[float] = []
    veff_values: list[float] = []
    nbar_values: list[np.ndarray] = []
    fkp_values: list[np.ndarray] = []
    z_centers_reference: np.ndarray | None = None
    failures: list[str] = []

    for row in rows:
        phase = str(row["phase"])
        zmin, zmax = float(row["zmin_observed"]), float(row["zmax_observed"])
        catalog_path = Path(row["lightcone_catalog_path"])
        catalog_meta_path = Path(row["lightcone_metadata_path"])
        random_path = Path(row["lightcone_random_path"])
        random_meta_path = Path(row["lightcone_random_metadata_path"])
        fkp_path = Path(row["lightcone_fkp_path"])
        fkp_meta_path = Path(row["lightcone_fkp_metadata_path"])
        required = (catalog_path, catalog_meta_path, random_path, random_meta_path, fkp_path, fkp_meta_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            failures.append(f"{phase}: missing {missing}")
            continue

        catalog_meta = load_json(catalog_meta_path)
        random_meta = load_json(random_meta_path)
        fkp_meta = load_json(fkp_meta_path)
        catalog_hash = sha256_file(catalog_path)
        random_hash = sha256_file(random_path)
        fkp_hash = sha256_file(fkp_path)
        with np.load(catalog_path, allow_pickle=False) as payload:
            catalog_z, catalog_encoding = repair_float32_redshift_boundaries(payload["Z"], zmin=zmin, zmax=zmax)
            ndata = int(catalog_z.size)
        with np.load(random_path, allow_pickle=False) as payload:
            random_z = np.asarray(payload["Z"], dtype="f8")
            random_index = np.asarray(payload["RANDOM_INDEX"], dtype="i2")
            nrandom = int(random_z.size)
            random_positive = bool(
                np.all(np.asarray(payload["X"]) >= 0.0)
                and np.all(np.asarray(payload["Y"]) >= 0.0)
                and np.all(np.asarray(payload["Zcart"]) >= 0.0)
            )
        with np.load(fkp_path, allow_pickle=False) as payload:
            fkp = {key: np.asarray(payload[key]) for key in payload.files}

        z_edges = np.asarray(fkp["z_edges"], dtype="f8")
        z_centers = np.asarray(fkp["z_centers"], dtype="f8")
        volume = np.asarray(fkp["volume_shell"], dtype="f8")
        counts = np.asarray(fkp["data_counts"], dtype="i8")
        random_counts = np.asarray(fkp["random_counts"], dtype="i8")
        nbar = np.asarray(fkp["nbar"], dtype="f8")
        fkp_weight = np.asarray(fkp["fkp_weights"], dtype="f8")
        p0 = float(np.asarray(fkp["p0"]).item())
        zeff = float(np.asarray(fkp["zeff"]).item())
        veff = float(np.asarray(fkp["effective_volume"]).item())
        expected_edges = np.linspace(zmin, zmax, int(round((zmax - zmin) / 0.01)) + 1)
        pair_weight = nbar**2 * fkp_weight**2 * volume
        zeff_recomputed = float(np.sum(z_centers * pair_weight) / np.sum(pair_weight))
        veff_recomputed = float(np.sum((nbar * p0 / (1.0 + nbar * p0)) ** 2 * volume))
        split_sizes = np.bincount(random_index.astype("i8"), minlength=int(row["random_multiplier"]))
        margins = catalog_meta.get("source_coverage_margin_mpc_h", {})
        displacement = float(catalog_meta.get("maximum_abs_displacement_all_sources_mpc_h", np.inf))
        gates = {
            "catalog_status_hash": bool(
                catalog_meta.get("status") == "pass" and catalog_meta.get("output_sha256") == catalog_hash
            ),
            "catalog_strict_redshift": bool(np.all((catalog_z > zmin) & (catalog_z < zmax))),
            "catalog_positive_octant": bool(catalog_meta.get("positive_octant_gate", False)),
            "catalog_no_duplicates": int(catalog_meta.get("duplicate_shell_index_origin_count", -1)) == 0,
            "catalog_source_coverage": bool(
                catalog_meta.get("source_coverage_gate", False)
                and float(margins.get("lower", -np.inf)) > displacement
                and float(margins.get("upper", -np.inf)) > displacement
            ),
            "catalog_rsd_migration_present": int(
                catalog_meta.get("n_migrated_in_from_source_z_outside_window", 0)
            ) > 0,
            "catalog_shell_count": len(catalog_meta.get("shells", ())) == len(row["lightcone_shells"]) == 11,
            "random_status_hash": bool(
                random_meta.get("status") == "pass" and random_meta.get("output_sha256") == random_hash
            ),
            "random_x25": bool(nrandom == 25 * ndata and np.array_equal(split_sizes, np.full(25, ndata))),
            "random_strict_redshift": bool(np.all((random_z > zmin) & (random_z < zmax))),
            "random_positive_octant": random_positive,
            "fkp_status_hash": bool(fkp_meta.get("status") == "pass" and fkp_meta.get("output_sha256") == fkp_hash),
            "fkp_grid": bool(np.array_equal(z_edges, expected_edges) and z_centers.size == 70),
            "fkp_counts": bool(int(np.sum(counts)) == ndata and int(np.sum(random_counts)) == nrandom),
            "nbar_definition": bool(np.allclose(nbar, counts / volume, rtol=2.0e-14, atol=0.0)),
            "fkp_definition": bool(np.allclose(fkp_weight, 1.0 / (1.0 + nbar * p0), rtol=2.0e-14, atol=0.0)),
            "zeff_definition": bool(np.isclose(zeff, zeff_recomputed, rtol=2.0e-14, atol=0.0)),
            "veff_definition": bool(np.isclose(veff, veff_recomputed, rtol=2.0e-14, atol=0.0)),
            "finite_positive": bool(
                np.all(np.isfinite(nbar)) and np.all(nbar > 0.0)
                and np.all(np.isfinite(fkp_weight)) and np.all((fkp_weight > 0.0) & (fkp_weight < 1.0))
                and zmin < zeff < zmax and veff > 0.0
            ),
        }
        if not all(gates.values()):
            failures.append(f"{phase}: failed gates {[key for key, value in gates.items() if not value]}")
        if z_centers_reference is None:
            z_centers_reference = z_centers
        elif not np.array_equal(z_centers_reference, z_centers):
            failures.append(f"{phase}: z-center grid changed")
        ndata_values.append(ndata)
        zeff_values.append(zeff)
        veff_values.append(veff)
        nbar_values.append(nbar)
        fkp_values.append(fkp_weight)
        phase_rows.append(
            {
                "phase": phase,
                "status": "pass" if all(gates.values()) else "fail",
                "gates": gates,
                "ndata": ndata,
                "nrandom": nrandom,
                "zeff": zeff,
                "geometric_volume_mpc3_h3": float(np.sum(volume)),
                "effective_volume_mpc3_h3": veff,
                "n_migrated_in": int(catalog_meta["n_migrated_in_from_source_z_outside_window"]),
                "source_coverage_margin_mpc_h": margins,
                "maximum_abs_displacement_mpc_h": displacement,
                "catalog_redshift_serialization_repair": catalog_encoding,
                "hashes": {"catalog": catalog_hash, "random": random_hash, "fkp": fkp_hash},
            }
        )

    complete = len(phase_rows) == len(PHASES)
    status = "pass" if complete and not failures else "fail"
    summary: dict[str, Any] = {
        "task": "task43_audit_rsd_wide_lightcone_upstream",
        "status": status,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "analysis_scope": "Task 4.3.2 wide lightcone, 0.4 < zobs < 1.1",
        "nphase_expected": len(PHASES),
        "nphase_validated": len(phase_rows),
        "failures": failures,
        "phases": phase_rows,
    }
    if complete:
        ndata_array = np.asarray(ndata_values, dtype="f8")
        zeff_array = np.asarray(zeff_values, dtype="f8")
        veff_array = np.asarray(veff_values, dtype="f8")
        nbar_array = np.stack(nbar_values)
        fkp_array = np.stack(fkp_values)
        summary["x25_summary"] = {
            "ndata_range": [int(np.min(ndata_array)), int(np.max(ndata_array))],
            "ndata_mean": float(np.mean(ndata_array)),
            "zeff_range": [float(np.min(zeff_array)), float(np.max(zeff_array))],
            "zeff_equal_phase_mean": float(np.mean(zeff_array)),
            "effective_volume_range_mpc3_h3": [float(np.min(veff_array)), float(np.max(veff_array))],
            "effective_volume_equal_phase_mean_mpc3_h3": float(np.mean(veff_array)),
            "geometric_volume_mpc3_h3": phase_rows[0]["geometric_volume_mpc3_h3"],
            "nbar_range_h3_mpc3": [float(np.min(nbar_array)), float(np.max(nbar_array))],
            "fkp_weight_range": [float(np.min(fkp_array)), float(np.max(fkp_array))],
        }
        npz_path = args.output.with_suffix(".npz")
        atomic_savez(
            npz_path,
            phases=np.asarray(PHASES),
            z_centers=np.asarray(z_centers_reference),
            ndata=ndata_array.astype("i8"),
            zeff=zeff_array,
            effective_volume=veff_array,
            nbar_by_phase=nbar_array,
            fkp_weight_by_phase=fkp_array,
        )
        summary["summary_npz"] = str(npz_path)
        summary["summary_npz_sha256"] = sha256_file(npz_path)
    atomic_write_json(args.output, summary)
    print(json.dumps({"status": status, "nphase": len(phase_rows), "failures": failures, "output": str(args.output)}, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
