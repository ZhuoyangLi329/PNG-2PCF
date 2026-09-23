#!/usr/bin/env python3
"""Freeze the independent Task 4.3.2 wide-lightcone (0.4 < zobs < 1.1) manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from task43_rsd_common import (
    MASS_THRESHOLD_HMSUN,
    OUTPUT_ROOT,
    P0_FKP,
    P_FIXED,
    PHASES,
    S_EDGES,
    SMIN_SCAN,
    atomic_write_json,
    lightcone_shell_path,
    rawbox_halo_dir,
    sha256_file,
    sim_name,
)


ZMIN_WIDE = 0.4
ZMAX_WIDE = 1.1
ANALYSIS_TAG = "zobs0p4_1p1"
LIGHTCONE_SHELLS_WIDE = (
    "z0.400",
    "z0.450",
    "z0.500",
    "z0.575",
    "z0.650",
    "z0.725",
    "z0.800",
    "z0.875",
    "z0.950",
    "z1.025",
    "z1.100",
)
WIDE_ROOT = OUTPUT_ROOT / "lightcone_wide_zobs0p4_1p1"
DEFAULT_JSONL = OUTPUT_ROOT / "manifests" / "task43_rsd_validation_lightcone_wide_zobs0p4_1p1_x25.jsonl"
DEFAULT_AUDIT = OUTPUT_ROOT / "audits" / "task43_rsd_wide_lightcone_manifest_audit.json"


def build_row(phase: str, phase_index: int) -> dict[str, object]:
    rawbox_paths = sorted(rawbox_halo_dir(phase).glob("halo_info_*.asdf"))
    lightcone_paths = [lightcone_shell_path(phase, shell) for shell in LIGHTCONE_SHELLS_WIDE]
    label = f"{sim_name(phase)}_mmin1p4e13_{ANALYSIS_TAG}"
    return {
        "task": "task43_rsd_validation_lightcone_wide",
        "analysis_scope": "task43.2_wide_lightcone_LRGall_redshift",
        "analysis_tag": ANALYSIS_TAG,
        "phase": phase,
        "phase_index": int(phase_index),
        "sim_name": sim_name(phase),
        "injected_fnl": 0.0,
        "mass_threshold_hmsun": MASS_THRESHOLD_HMSUN,
        "p_fixed": P_FIXED,
        "p0_fkp": P0_FKP,
        "zmin_observed": ZMIN_WIDE,
        "zmax_observed": ZMAX_WIDE,
        "s_edges_mpc_h": S_EDGES.tolist(),
        "smin_scan_mpc_h": list(SMIN_SCAN),
        "rawbox_source_paths": [str(path) for path in rawbox_paths],
        "lightcone_source_paths": [str(path) for path in lightcone_paths],
        "lightcone_shells": list(LIGHTCONE_SHELLS_WIDE),
        "lightcone_catalog_path": str(WIDE_ROOT / "catalogs" / f"task43_rsd_lightcone_{label}.npz"),
        "lightcone_metadata_path": str(WIDE_ROOT / "catalogs" / f"task43_rsd_lightcone_{label}.json"),
        "lightcone_zero_velocity_catalog_path": str(
            WIDE_ROOT / "catalogs_zero_velocity" / f"task43_rsd_lightcone_{label}_vscale0.npz"
        ),
        "lightcone_zero_velocity_metadata_path": str(
            WIDE_ROOT / "catalogs_zero_velocity" / f"task43_rsd_lightcone_{label}_vscale0.json"
        ),
        "lightcone_random_path": str(WIDE_ROOT / "randoms" / f"task43_rsd_random_{label}_x25.npz"),
        "lightcone_random_metadata_path": str(WIDE_ROOT / "randoms" / f"task43_rsd_random_{label}_x25.json"),
        "lightcone_fkp_path": str(WIDE_ROOT / "fkp" / f"task43_rsd_fkp_{label}_dz0p01.npz"),
        "lightcone_fkp_metadata_path": str(WIDE_ROOT / "fkp" / f"task43_rsd_fkp_{label}_dz0p01.json"),
        "lightcone_fkp_data_view_path": str(
            WIDE_ROOT / "fkp_catalogs" / f"task43_rsd_data_{label}_fkpP010000.npz"
        ),
        "lightcone_fkp_random_view_path": str(
            WIDE_ROOT / "fkp_catalogs" / f"task43_rsd_random_{label}_fkpP010000_x25.npz"
        ),
        "lightcone_xi_path": str(WIDE_ROOT / "xi" / f"task43_rsd_xi0_{label}_x25_s30_350_ds10.npz"),
        "random_multiplier": 25,
        "random_radial_policy": "data_observed_redshift_resample",
        "rsd_lightcone_los": "local_radial",
        "rsd_velocity_scale": 1.0,
        "fallback_policy": "official_cleaned_nonzero_posavg_uses_posavg_velavg_origin_modulo_3",
        "legacy_bridge_catalog_path": None,
        "immutability": "independent wide-redshift products; never overwrite narrow Task43 or Task44 products",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    rows = [build_row(phase, index) for index, phase in enumerate(PHASES)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)

    missing = [path for row in rows for path in row["lightcone_source_paths"] if not Path(path).is_file()]
    raw_counts = [len(row["rawbox_source_paths"]) for row in rows]
    output_keys = tuple(key for key in rows[0] if key.startswith("lightcone_") and key.endswith("_path"))
    output_root_gate = all(
        Path(str(row[key])).is_relative_to(WIDE_ROOT)
        for row in rows
        for key in output_keys
        if row[key] is not None and "source" not in key
    )
    status = "pass" if not missing and raw_counts == [34] * len(PHASES) and output_root_gate else "fail"
    audit = {
        "task": "task43_rsd_wide_lightcone_manifest",
        "status": status,
        "manifest_path": str(args.output),
        "manifest_sha256": sha256_file(args.output),
        "nphases": len(rows),
        "phases": list(PHASES),
        "analysis_tag": ANALYSIS_TAG,
        "observed_redshift_open_interval": [ZMIN_WIDE, ZMAX_WIDE],
        "shells": list(LIGHTCONE_SHELLS_WIDE),
        "lightcone_file_count_by_phase": [len(row["lightcone_source_paths"]) for row in rows],
        "rawbox_file_count_by_phase": raw_counts,
        "missing_lightcone_sources": missing,
        "output_root": str(WIDE_ROOT),
        "output_root_gate": output_root_gate,
        "frozen": {
            "mass_threshold_hmsun": MASS_THRESHOLD_HMSUN,
            "p0_fkp": P0_FKP,
            "p_fixed": P_FIXED,
            "s_edges_mpc_h": S_EDGES.tolist(),
            "smin_scan_mpc_h": list(SMIN_SCAN),
            "random_multiplier": 25,
        },
    }
    atomic_write_json(args.audit, audit)
    print(json.dumps({"status": status, "manifest": str(args.output), "audit": str(args.audit)}, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
