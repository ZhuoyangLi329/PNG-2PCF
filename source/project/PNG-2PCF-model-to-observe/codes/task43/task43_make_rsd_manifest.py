#!/usr/bin/env python3
"""Create the immutable-input manifest for Task 4.3.2 RSD validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from task43_rsd_common import (
    LIGHTCONE_SHELLS,
    MASS_THRESHOLD_HMSUN,
    OUTPUT_ROOT,
    P0_FKP,
    P_FIXED,
    PHASES,
    S_EDGES,
    SMIN_SCAN,
    ZMAX,
    ZMIN,
    atomic_write_json,
    lightcone_shell_path,
    rawbox_halo_dir,
    sha256_file,
    sim_name,
)


DEFAULT_JSONL = OUTPUT_ROOT / "manifests" / "task43_rsd_validation_x25.jsonl"
DEFAULT_AUDIT = OUTPUT_ROOT / "audits" / "task43_rsd_manifest_audit.json"


def build_row(phase: str, phase_index: int) -> dict[str, object]:
    rawbox_paths = sorted(rawbox_halo_dir(phase).glob("halo_info_*.asdf"))
    lightcone_paths = [lightcone_shell_path(phase, shell) for shell in LIGHTCONE_SHELLS]
    label = f"{sim_name(phase)}_mmin1p4e13"
    raw_root = OUTPUT_ROOT / "rawbox"
    lightcone_root = OUTPUT_ROOT / "lightcone"
    return {
        "task": "task43_rsd_validation",
        "phase": phase,
        "phase_index": int(phase_index),
        "sim_name": sim_name(phase),
        "injected_fnl": 0.0,
        "mass_threshold_hmsun": MASS_THRESHOLD_HMSUN,
        "p_fixed": P_FIXED,
        "p0_fkp": P0_FKP,
        "zmin_observed": ZMIN,
        "zmax_observed": ZMAX,
        "s_edges_mpc_h": S_EDGES.tolist(),
        "smin_scan_mpc_h": list(SMIN_SCAN),
        "rawbox_source_paths": [str(path) for path in rawbox_paths],
        "lightcone_source_paths": [str(path) for path in lightcone_paths],
        "lightcone_shells": list(LIGHTCONE_SHELLS),
        "rawbox_catalog_path": str(raw_root / "catalogs" / f"task43_rsd_rawbox_{label}.npz"),
        "rawbox_metadata_path": str(raw_root / "catalogs" / f"task43_rsd_rawbox_{label}.json"),
        "rawbox_summary_path": str(raw_root / "summary" / f"task43_rsd_rawbox_{label}_clustering.npz"),
        "lightcone_catalog_path": str(lightcone_root / "catalogs" / f"task43_rsd_lightcone_{label}_zobs0p6_0p8.npz"),
        "lightcone_metadata_path": str(lightcone_root / "catalogs" / f"task43_rsd_lightcone_{label}_zobs0p6_0p8.json"),
        "lightcone_zero_velocity_catalog_path": str(lightcone_root / "catalogs_zero_velocity" / f"task43_rsd_lightcone_{label}_zobs0p6_0p8_vscale0.npz"),
        "lightcone_zero_velocity_metadata_path": str(lightcone_root / "catalogs_zero_velocity" / f"task43_rsd_lightcone_{label}_zobs0p6_0p8_vscale0.json"),
        "lightcone_random_path": str(lightcone_root / "randoms" / f"task43_rsd_random_{label}_zobs0p6_0p8_x25.npz"),
        "lightcone_random_metadata_path": str(lightcone_root / "randoms" / f"task43_rsd_random_{label}_zobs0p6_0p8_x25.json"),
        "lightcone_xi_path": str(lightcone_root / "xi" / f"task43_rsd_xi0_{label}_zobs0p6_0p8_x25_s30_350_ds10.npz"),
        "random_multiplier": 25,
        "random_radial_policy": "data_redshift_resample",
        "rsd_plane_parallel_los_axis": 2,
        "rsd_lightcone_los": "local_radial",
        "rsd_velocity_scale": 1.0,
        "fallback_policy": "official_cleaned_nonzero_posavg_uses_posavg_velavg_origin_modulo_3",
        "immutability": "new rsd_validation products only; never overwrite existing Task43/Task44 products",
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

    missing_raw = [path for row in rows for path in row["rawbox_source_paths"] if not Path(path).is_file()]
    missing_lightcone = [path for row in rows for path in row["lightcone_source_paths"] if not Path(path).is_file()]
    raw_counts = [len(row["rawbox_source_paths"]) for row in rows]
    output_paths = [
        Path(str(row[key]))
        for row in rows
        for key in (
            "rawbox_catalog_path",
            "rawbox_metadata_path",
            "lightcone_catalog_path",
            "lightcone_metadata_path",
            "lightcone_random_path",
            "lightcone_random_metadata_path",
            "lightcone_xi_path",
        )
    ]
    output_root_gate = all(path.is_relative_to(OUTPUT_ROOT) for path in output_paths)
    audit = {
        "task": "task43_rsd_manifest",
        "status": "pass" if not missing_raw and not missing_lightcone and raw_counts == [34] * 25 and output_root_gate else "fail",
        "manifest_path": str(args.output),
        "manifest_sha256": sha256_file(args.output),
        "nrows": len(rows),
        "phases": list(PHASES),
        "rawbox_file_count_by_phase": raw_counts,
        "lightcone_file_count_by_phase": [len(row["lightcone_source_paths"]) for row in rows],
        "missing_rawbox": missing_raw,
        "missing_lightcone": missing_lightcone,
        "output_root_gate": output_root_gate,
        "frozen": {
            "mass_threshold_hmsun": MASS_THRESHOLD_HMSUN,
            "p0_fkp": P0_FKP,
            "p_fixed": P_FIXED,
            "z_observed": [ZMIN, ZMAX],
            "s_edges_mpc_h": S_EDGES.tolist(),
            "smin_scan_mpc_h": list(SMIN_SCAN),
        },
    }
    atomic_write_json(args.audit, audit)
    print(json.dumps({"status": audit["status"], "manifest": str(args.output), "audit": str(args.audit)}, sort_keys=True))
    if audit["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
