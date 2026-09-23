#!/usr/bin/env python3
"""Audit source schemas, headers, environments, and frozen paths for Task43 RSD."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from task43_rsd_common import OUTPUT_ROOT, PHASES, atomic_write_json, header_boxsize, velocity_kms_per_mpc_h


REQUIRED_RAW_FIELDS = {"N", "x_L2com", "v_L2com"}
REQUIRED_LIGHTCONE_FIELDS = {
    "N_interp",
    "pos_interp",
    "vel_interp",
    "pos_avg",
    "vel_avg",
    "redshift_interp",
    "origin",
    "index_halo",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def module_info(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "version": str(getattr(module, "__version__", "unknown")),
        "path": str(getattr(module, "__file__", "")),
    }


def inspect_asdf(path: Path, required: set[str]) -> dict[str, Any]:
    import asdf

    with asdf.open(path, lazy_load=True, memmap=False) as af:
        header = af["header"]
        keys = set(af["data"].keys())
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "data_fields": sorted(keys),
            "missing_required_fields": sorted(required - keys),
            "boxsize_mpc_h": header_boxsize(header),
            "velocity_kms_per_mpc_h": velocity_kms_per_mpc_h(header),
            "redshift_header": float(header.get("Redshift", np.nan)),
            "particle_mass_hmsun": float(header["ParticleMassHMsun"]),
            "lightcone_origins": (
                np.asarray(header["LightConeOrigins"], dtype="f8").reshape(-1, 3).tolist()
                if "LightConeOrigins" in header
                else None
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "audits" / "task43_rsd_preflight.json")
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    if [row["phase"] for row in rows] != list(PHASES):
        raise RuntimeError("manifest phase order is not ph000..ph024")
    missing = []
    for row in rows:
        for key in ("rawbox_source_paths", "lightcone_source_paths"):
            for name in row[key]:
                if not Path(name).is_file():
                    missing.append(name)

    representative_raw = []
    representative_lightcone = []
    for row in rows:
        representative_raw.append(inspect_asdf(Path(row["rawbox_source_paths"][0]), REQUIRED_RAW_FIELDS))
        representative_lightcone.extend(
            inspect_asdf(Path(path), REQUIRED_LIGHTCONE_FIELDS) for path in row["lightcone_source_paths"]
        )

    required_modules = ("asdf", "numpy", "scipy", "cosmoprimo", "jaxpower", "cucount", "pycorr", "emcee", "lsstypes")
    environment = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "modules": {name: module_info(name) for name in required_modules},
    }
    schema_gate = all(not item["missing_required_fields"] for item in representative_raw + representative_lightcone)
    box_gate = all(abs(float(item["boxsize_mpc_h"]) - 2000.0) < 1.0e-10 for item in representative_raw + representative_lightcone)
    velocity_gate = all(50.0 < float(item["velocity_kms_per_mpc_h"]) < 150.0 for item in representative_raw + representative_lightcone)
    environment_gate = all(environment["modules"][name]["available"] for name in required_modules)
    output_root_gate = args.output.is_relative_to(OUTPUT_ROOT)
    status = "pass" if not missing and schema_gate and box_gate and velocity_gate and environment_gate and output_root_gate else "fail"
    payload = {
        "task": "task43_rsd_preflight",
        "status": status,
        "manifest": str(args.manifest),
        "nphases": len(rows),
        "missing_inputs": missing,
        "gates": {
            "schema": schema_gate,
            "boxsize_2000": box_gate,
            "velocity_conversion_plausible": velocity_gate,
            "environment": environment_gate,
            "output_root": output_root_gate,
        },
        "environment": environment,
        "rawbox_representative_by_phase": representative_raw,
        "lightcone_all_shell_headers": representative_lightcone,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"status": status, "output": str(args.output), "gates": payload["gates"]}, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

