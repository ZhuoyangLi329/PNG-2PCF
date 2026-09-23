#!/usr/bin/env python3
"""Record immutable software and source hashes for covariance production."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

from task43_ezmock_covariance_common import CODE_DIR, EZMOCK_BINARY, FCFC_BINARY, LINEAR_PK, SUMMARY_DIR, sha256, write_json


OUTPUT = SUMMARY_DIR / "task43_ezmock_covariance_environment.json"
SCRIPTS = (
    "task43_ezmock_covariance_common.py",
    "task43_make_ezmock_covariance_manifest.py",
    "task43_build_ezmock_covariance_lightcone.py",
    "task43_build_ezmock_common_random.py",
    "task43_prepare_ezmock_common_random_hdf5.py",
    "task43_build_ezmock_common_rr_fcfc.py",
    "task43_audit_ezmock_common_random.py",
    "task43_measure_validation_common50_xi_fcfc.py",
    "task43_audit_validation_common50_xi_ab.py",
    "task43_benchmark_fcfc_common50_split.py",
    "task43_measure_ezmock_covariance_xi_fcfc.py",
    "task43_measure_ezmock_covariance_pk_jaxpower.py",
    "task43_run_ezmock_covariance_login.py",
    "run_task43_ezmock_covariance_login_persistent.sh",
    "task43_summarize_ezmock_covariance.py",
    "task43_finalize_ezmock_covariance.py",
    "run_task43_ezmock_covariance_finalizer_persistent.sh",
    "task43_audit_ezmock_covariance_environment.py",
)


def version(name: str) -> str | None:
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return None


def main() -> None:
    paths = [CODE_DIR / name for name in SCRIPTS] + [EZMOCK_BINARY, FCFC_BINARY, LINEAR_PK]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing: raise FileNotFoundError(missing)
    fcfc_version = subprocess.run([str(FCFC_BINARY), "--version"], check=True, capture_output=True, text=True).stdout
    payload = {
        "task": "task43_audit_ezmock_covariance_environment", "status": "done",
        "python": sys.version, "platform": platform.platform(), "hostname": platform.node(),
        "cpu_count_visible": os.cpu_count(), "affinity_at_audit": sorted(os.sched_getaffinity(0)),
        "packages": {name: version(name) for name in ("numpy", "jax", "jaxlib", "jaxpower", "h5py", "cosmoprimo", "desilike")},
        "fcfc_hdf5_enabled": "HDF5: enabled" in fcfc_version,
        "fcfc_version_output": fcfc_version,
        "sha256": {str(path): sha256(path) for path in paths},
    }
    if not payload["fcfc_hdf5_enabled"]: raise RuntimeError("production FCFC lacks HDF5 support")
    write_json(OUTPUT, payload); print(f"[write] {OUTPUT}")


if __name__ == "__main__": main()
