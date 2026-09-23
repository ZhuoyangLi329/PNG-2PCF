#!/usr/bin/env python3
"""Build and verify the immutable common-50 weighted RR cache for Task43 EZmock RSD."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

_INITIAL_CPUS = tuple(sorted(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else ()
import numpy as np
from task43_ezmock_rsd_covariance_common import (
    COMMON_RANDOM, COMMON_RANDOM_HDF5, COMMON_RANDOM_HDF5_META,
    COMMON_RR, COMMON_RR_META, COMMON_RANDOM_META, FCFC_BINARY,
    FCFC_CONFIG_DIR, LOG_DIR, S_EDGES, ensure_dirs, sha256, write_json,
)
from task43_rawbox_ezmock_common import set_cpu_affinity

CONFIG = FCFC_CONFIG_DIR / "fcfc_common50_ezmock_rsd_RR_s30_350_ds10.conf"
LOG = LOG_DIR / "fcfc_common50_ezmock_rsd_RR_s30_350_ds10.log"
DUMMY_XI = FCFC_CONFIG_DIR / "fcfc_common50_ezmock_rsd_RR_dummy_xi.txt"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def apply_affinity(n):
    if not 1 <= int(n) <= 64:
        raise ValueError("FCFC compute contract requires 1..64 threads")
    if not _INITIAL_CPUS:
        return set_cpu_affinity(int(n))
    chosen = list(_INITIAL_CPUS[:int(n)])
    os.sched_setaffinity(0, chosen)
    return chosen

def load_pair(path):
    table = np.loadtxt(path, comments="#", dtype="f8")
    expected = (S_EDGES.size - 1, 3)
    if table.shape != expected or not np.all(np.isfinite(table)) or np.any(table[:, 2] <= 0):
        raise ValueError("invalid RR table shape or values: %s" % (table.shape,))
    edges = np.concatenate([table[:1, 0], table[:, 1]])
    if not np.array_equal(edges, S_EDGES):
        raise ValueError("common RR separation grid mismatch")
    return table

def config_text():
    pos_x, pos_y, pos_z = "$" + "{/X}", "$" + "{/Y}", "$" + "{/Zcart}"
    weight = "$" + "{/WEIGHT_FKP}"
    return (
        "CATALOG = '" + str(COMMON_RANDOM_HDF5) + "'\n"
        "CATALOG_LABEL = R\n"
        "CATALOG_TYPE = 2\n"
        "POSITION = [" + pos_x + ", " + pos_y + ", " + pos_z + "]\n"
        "WEIGHT = " + weight + "\n"
        "COORD_CONVERT = F\n"
        "DATA_STRUCT = 0\n"
        "BINNING_SCHEME = 0\n"
        "PAIR_COUNT = RR\n"
        "PAIR_COUNT_FILE = '" + str(COMMON_RR) + "'\n"
        "CF_ESTIMATOR = RR / RR\n"
        "CF_OUTPUT_FILE = '" + str(DUMMY_XI) + "'\n"
        "SEP_BIN_MIN = %.17g\n" % S_EDGES[0]
        + "SEP_BIN_MAX = %.17g\n" % S_EDGES[-1]
        + "SEP_BIN_SIZE = %.17g\n" % np.median(np.diff(S_EDGES))
        + "OUTPUT_FORMAT = 1\nOVERWRITE = 2\nVERBOSE = T\n"
    )

def main():
    args = parse_args()
    cpus = apply_affinity(int(args.threads))
    ensure_dirs()
    required = (COMMON_RANDOM, COMMON_RANDOM_META, COMMON_RANDOM_HDF5,
                COMMON_RANDOM_HDF5_META, FCFC_BINARY)
    for path in required:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    source_hash = sha256(COMMON_RANDOM)
    h5_hash = sha256(COMMON_RANDOM_HDF5)
    source_meta = json.loads(COMMON_RANDOM_META.read_text(encoding="utf-8"))
    h5_meta = json.loads(COMMON_RANDOM_HDF5_META.read_text(encoding="utf-8"))
    if source_meta.get("sha256") != source_hash or h5_meta.get("source_npz_sha256") != source_hash or h5_meta.get("sha256") != h5_hash:
        raise RuntimeError("common random source/HDF5 provenance gate failed")
    nrandom = int(h5_meta.get("nrandom", -1))
    if nrandom <= 0:
        raise RuntimeError("invalid HDF5 nrandom metadata")
    if (not args.overwrite and COMMON_RR.is_file() and COMMON_RR_META.is_file()):
        meta = json.loads(COMMON_RR_META.read_text(encoding="utf-8"))
        if (meta.get("status") == "done" and meta.get("common_random_npz_sha256") == source_hash
            and meta.get("common_random_hdf5_sha256") == h5_hash and int(meta.get("nrandom", -1)) == nrandom
            and meta.get("rr_sha256") == sha256(COMMON_RR)):
            load_pair(COMMON_RR)
            print("[skip verified] %s" % COMMON_RR)
            return
    CONFIG.write_text(config_text(), encoding="utf-8")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": str(int(args.threads)), "OMP_DYNAMIC": "FALSE",
                "OMP_PROC_BIND": "close", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    t0 = time.perf_counter()
    with LOG.open("w", encoding="utf-8") as stream:
        subprocess.run([str(FCFC_BINARY), "-c", str(CONFIG)], check=True,
                       stdout=stream, stderr=subprocess.STDOUT, env=env)
    table = load_pair(COMMON_RR)
    log_text = LOG.read_text(encoding="utf-8", errors="replace")
    if "Counting RR pairs" not in log_text or "Results written to file" not in log_text:
        raise RuntimeError("FCFC log lacks complete RR-write evidence")
    meta = {
        "task": "task43_build_ezmock_rsd_common_rr_fcfc",
        "status": "done",
        "classification": "immutable_weighted_common50_RR_for_RSD_covariance",
        "rr_path": str(COMMON_RR), "rr_sha256": sha256(COMMON_RR),
        "common_random_npz": str(COMMON_RANDOM), "common_random_npz_sha256": source_hash,
        "common_random_hdf5": str(COMMON_RANDOM_HDF5), "common_random_hdf5_sha256": h5_hash,
        "nrandom": nrandom, "s_edges": S_EDGES, "rr_normalized": table[:, 2],
        "fcfc_binary": str(FCFC_BINARY), "fcfc_binary_sha256": sha256(FCFC_BINARY),
        "config": str(CONFIG), "config_sha256": sha256(CONFIG), "log": str(LOG),
        "threads": int(args.threads), "cpu_affinity": cpus,
        "runtime_sec": float(time.perf_counter() - t0),
    }
    write_json(COMMON_RR_META, meta)
    print("[done] %s bins=%d nrandom=%d elapsed=%.1fs" % (COMMON_RR, table.shape[0], nrandom, meta["runtime_sec"]))

if __name__ == "__main__":
    main()
