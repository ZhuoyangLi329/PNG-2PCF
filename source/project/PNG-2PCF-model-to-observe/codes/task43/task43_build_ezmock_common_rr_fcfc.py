#!/usr/bin/env python3
"""Compute and audit the one FCFC RR cache shared by all 1000 mocks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from task43_ezmock_covariance_common import (
    COMMON_RANDOM,
    COMMON_RANDOM_HDF5,
    COMMON_RANDOM_HDF5_META,
    COMMON_RANDOM_SIZE,
    COMMON_RR,
    COMMON_RR_META,
    FCFC_BINARY,
    FCFC_CONFIG_DIR,
    LOG_DIR,
    S_EDGES,
    ensure_dirs,
    sha256,
    write_json,
)
from task43_rawbox_ezmock_common import set_cpu_affinity


CONFIG = FCFC_CONFIG_DIR / "fcfc_common50_ezmock_geometric_shell_RR_s50_350_ds10.conf"
LOG = LOG_DIR / "fcfc_common50_ezmock_geometric_shell_RR_s50_350_ds10.log"
DUMMY_XI = FCFC_CONFIG_DIR / "fcfc_common50_ezmock_geometric_shell_RR_s50_350_ds10_dummy_xi.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_pair(path: Path) -> np.ndarray:
    table = np.loadtxt(path, comments="#", dtype="f8")
    expected = (S_EDGES.size - 1, 3)
    if table.shape != expected or not np.all(np.isfinite(table)) or np.any(table[:, 2] <= 0.0):
        raise ValueError(f"invalid common RR {path}: shape={table.shape}")
    edges = np.concatenate([table[:1, 0], table[:, 1]])
    if not np.array_equal(edges, S_EDGES):
        raise ValueError("common RR separation grid mismatch")
    return table


def validate_existing(source_hash: str, transport_hash: str) -> bool:
    if not COMMON_RR.is_file() or not COMMON_RR_META.is_file():
        return False
    meta = json.loads(COMMON_RR_META.read_text(encoding="utf-8"))
    if not (
        meta.get("status") == "done"
        and meta.get("common_random_npz_sha256") == source_hash
        and meta.get("common_random_hdf5_sha256") == transport_hash
        and int(meta.get("nrandom", -1)) == COMMON_RANDOM_SIZE
    ):
        return False
    load_pair(COMMON_RR)
    return meta.get("rr_sha256") == sha256(COMMON_RR)


def config_text() -> str:
    return f"""CATALOG = '{COMMON_RANDOM_HDF5}'
CATALOG_LABEL = R
CATALOG_TYPE = 2
POSITION = [${{/X}}, ${{/Y}}, ${{/Zcart}}]
WEIGHT = ${{/WEIGHT_FKP}}
COORD_CONVERT = F
DATA_STRUCT = 0
BINNING_SCHEME = 0
PAIR_COUNT = RR
PAIR_COUNT_FILE = '{COMMON_RR}'
CF_ESTIMATOR = RR / RR
CF_OUTPUT_FILE = '{DUMMY_XI}'
SEP_BIN_MIN = {S_EDGES[0]:.17g}
SEP_BIN_MAX = {S_EDGES[-1]:.17g}
SEP_BIN_SIZE = {np.median(np.diff(S_EDGES)):.17g}
OUTPUT_FORMAT = 1
OVERWRITE = 2
VERBOSE = T
"""


def main() -> None:
    args = parse_args()
    if not 1 <= int(args.threads) <= 8:
        raise ValueError("login-node FCFC contract requires 1..8 threads")
    cpus = set_cpu_affinity(int(args.threads))
    ensure_dirs()
    for path in (COMMON_RANDOM, COMMON_RANDOM_HDF5, COMMON_RANDOM_HDF5_META, FCFC_BINARY):
        if not path.is_file():
            raise FileNotFoundError(path)
    transport_meta = json.loads(COMMON_RANDOM_HDF5_META.read_text(encoding="utf-8"))
    source_hash = sha256(COMMON_RANDOM)
    transport_hash = sha256(COMMON_RANDOM_HDF5)
    if transport_meta.get("source_npz_sha256") != source_hash or transport_meta.get("sha256") != transport_hash:
        raise RuntimeError("common random HDF5 provenance gate failed")
    if not args.overwrite and validate_existing(source_hash, transport_hash):
        print(f"[skip verified] {COMMON_RR}")
        return
    # A completed RR must never be counted twice merely because a post-count
    # metadata check was interrupted.  Adopt only when the full native table
    # and the FCFC written-file evidence both pass; otherwise launch FCFC.
    adopt_completed_count = False
    if not args.overwrite and COMMON_RR.is_file() and CONFIG.is_file() and LOG.is_file():
        prior_log = LOG.read_text(encoding="utf-8", errors="replace")
        if "Counting RR pairs" in prior_log and f"Results written to file: `{COMMON_RR}'" in prior_log:
            load_pair(COMMON_RR)
            adopt_completed_count = True
    if adopt_completed_count:
        elapsed = max(0.0, COMMON_RR.stat().st_mtime - CONFIG.stat().st_mtime)
        print(f"[adopt verified completed RR] {COMMON_RR}")
    else:
        CONFIG.write_text(config_text(), encoding="utf-8")
        env = os.environ.copy()
        env.update({
            "OMP_NUM_THREADS": str(int(args.threads)), "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "close",
            "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        })
        started = time.perf_counter()
        with LOG.open("w", encoding="utf-8") as stream:
            subprocess.run([str(FCFC_BINARY), "-c", str(CONFIG)], check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
        elapsed = time.perf_counter() - started
    table = load_pair(COMMON_RR)
    log_text = LOG.read_text(encoding="utf-8", errors="replace")
    if "Counting RR pairs" not in log_text or f"Results written to file: `{COMMON_RR}'" not in log_text:
        raise RuntimeError("FCFC log lacks written-RR evidence")
    meta = {
        "task": "task43_build_ezmock_common_rr_fcfc",
        "status": "done",
        "classification": "one immutable weighted RR cache shared by all 1000 covariance mocks",
        "rr_path": str(COMMON_RR), "rr_sha256": sha256(COMMON_RR),
        "common_random_npz": str(COMMON_RANDOM), "common_random_npz_sha256": source_hash,
        "common_random_hdf5": str(COMMON_RANDOM_HDF5), "common_random_hdf5_sha256": transport_hash,
        "nrandom": COMMON_RANDOM_SIZE, "s_edges": S_EDGES, "rr_normalized": table[:, 2],
        "fcfc_binary": str(FCFC_BINARY), "fcfc_binary_sha256": sha256(FCFC_BINARY),
        "config": str(CONFIG), "config_sha256": sha256(CONFIG), "log": str(LOG),
        "threads": int(args.threads), "cpu_affinity": cpus, "runtime_sec": float(elapsed),
        "adopted_completed_count_after_metadata_interruption": bool(adopt_completed_count),
    }
    write_json(COMMON_RR_META, meta)
    print(f"[write] {COMMON_RR} bins={table.shape[0]} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
