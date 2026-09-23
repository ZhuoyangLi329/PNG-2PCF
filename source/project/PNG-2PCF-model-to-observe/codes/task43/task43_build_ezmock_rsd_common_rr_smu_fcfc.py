#!/usr/bin/env python3
"""Build the immutable weighted s-mu RR cache for FCFC xi0+xi2."""

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
    COMMON_RANDOM,
    COMMON_RANDOM_HDF5,
    COMMON_RANDOM_HDF5_META,
    COMMON_RANDOM_META,
    COMMON_RANDOM_SIZE,
    COMMON_RR_SMU,
    COMMON_RR_SMU_META,
    FCFC_BINARY,
    FCFC_CONFIG_DIR,
    LOG_DIR,
    MU_BIN_NUM,
    S_EDGES,
    ensure_dirs,
    sha256,
    write_json,
)
from task43_rawbox_ezmock_common import set_cpu_affinity


CONFIG = FCFC_CONFIG_DIR / "fcfc_common50_ezmock_rsd_RR_smu_s30_350_ds10_mu120.conf"
LOG = LOG_DIR / "fcfc_common50_ezmock_rsd_RR_smu_s30_350_ds10_mu120.log"
DUMMY_XI = FCFC_CONFIG_DIR / "fcfc_common50_ezmock_rsd_RR_smu_dummy.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def apply_affinity(n: int):
    if not 1 <= int(n) <= 64:
        raise ValueError("FCFC compute contract requires 1..64 threads")
    if not _INITIAL_CPUS:
        return set_cpu_affinity(int(n))
    chosen = list(_INITIAL_CPUS[: int(n)])
    if len(chosen) != int(n):
        raise RuntimeError(f"only {len(chosen)} CPUs available, need {n}")
    os.sched_setaffinity(0, chosen)
    return chosen


def config_text() -> str:
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
        "BINNING_SCHEME = 1\n"
        "PAIR_COUNT = RR\n"
        "PAIR_COUNT_FILE = '" + str(COMMON_RR_SMU) + "'\n"
        "CF_ESTIMATOR = RR / RR\n"
        "CF_OUTPUT_FILE = '" + str(DUMMY_XI) + "'\n"
        "SEP_BIN_MIN = %.17g\n" % S_EDGES[0]
        + "SEP_BIN_MAX = %.17g\n" % S_EDGES[-1]
        + "SEP_BIN_SIZE = %.17g\n" % np.median(np.diff(S_EDGES))
        + "MU_BIN_NUM = %d\n" % MU_BIN_NUM
        + "OUTPUT_FORMAT = 0\nOVERWRITE = 2\nVERBOSE = T\n"
    )


def main() -> None:
    args = parse_args()
    cpus = apply_affinity(int(args.threads))
    ensure_dirs()
    required = (
        COMMON_RANDOM,
        COMMON_RANDOM_META,
        COMMON_RANDOM_HDF5,
        COMMON_RANDOM_HDF5_META,
        FCFC_BINARY,
    )
    for path in required:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    source_hash = sha256(COMMON_RANDOM)
    h5_hash = sha256(COMMON_RANDOM_HDF5)
    source_meta = json.loads(COMMON_RANDOM_META.read_text(encoding="utf-8"))
    h5_meta = json.loads(COMMON_RANDOM_HDF5_META.read_text(encoding="utf-8"))
    if (
        source_meta.get("sha256") != source_hash
        or h5_meta.get("source_npz_sha256") != source_hash
        or h5_meta.get("sha256") != h5_hash
    ):
        raise RuntimeError("common random source/HDF5 provenance gate failed")
    if not args.overwrite and COMMON_RR_SMU.is_file() and COMMON_RR_SMU_META.is_file():
        meta = json.loads(COMMON_RR_SMU_META.read_text(encoding="utf-8"))
        if (
            meta.get("status") == "done"
            and meta.get("common_random_npz_sha256") == source_hash
            and meta.get("common_random_hdf5_sha256") == h5_hash
            and int(meta.get("nrandom", -1)) == COMMON_RANDOM_SIZE
            and int(meta.get("mu_bin_num", -1)) == MU_BIN_NUM
            and meta.get("rr_sha256") == sha256(COMMON_RR_SMU)
        ):
            print(f"[skip verified] {COMMON_RR_SMU}")
            return
    CONFIG.write_text(config_text(), encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(int(args.threads)),
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "close",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    started = time.perf_counter()
    with LOG.open("w", encoding="utf-8") as stream:
        subprocess.run([str(FCFC_BINARY), "-c", str(CONFIG)], check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
    log_text = LOG.read_text(encoding="utf-8", errors="replace")
    if "Counting RR pairs" not in log_text or "Results written to file" not in log_text:
        raise RuntimeError("FCFC log lacks complete 2D RR-write evidence")
    if not COMMON_RR_SMU.is_file() or COMMON_RR_SMU.stat().st_size <= 0:
        raise RuntimeError(f"missing 2D RR output {COMMON_RR_SMU}")
    meta = {
        "task": "task43_build_ezmock_rsd_common_rr_smu_fcfc",
        "status": "done",
        "classification": "immutable_weighted_common50_smu_RR_for_RSD_xi0_xi2_covariance",
        "rr_path": str(COMMON_RR_SMU),
        "rr_sha256": sha256(COMMON_RR_SMU),
        "common_random_npz": str(COMMON_RANDOM),
        "common_random_npz_sha256": source_hash,
        "common_random_hdf5": str(COMMON_RANDOM_HDF5),
        "common_random_hdf5_sha256": h5_hash,
        "nrandom": COMMON_RANDOM_SIZE,
        "s_edges": S_EDGES,
        "mu_bin_num": MU_BIN_NUM,
        "fcfc_binary": str(FCFC_BINARY),
        "fcfc_binary_sha256": sha256(FCFC_BINARY),
        "config": str(CONFIG),
        "config_sha256": sha256(CONFIG),
        "log": str(LOG),
        "threads": int(args.threads),
        "cpu_affinity": cpus,
        "runtime_sec": float(time.perf_counter() - started),
    }
    write_json(COMMON_RR_SMU_META, meta)
    print(f"[done] {COMMON_RR_SMU} nrandom={COMMON_RANDOM_SIZE} mu={MU_BIN_NUM} elapsed={meta['runtime_sec']:.1f}s")


if __name__ == "__main__":
    main()
