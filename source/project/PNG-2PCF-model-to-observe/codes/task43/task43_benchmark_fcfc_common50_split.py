#!/usr/bin/env python3
"""Benchmark common50 FCFC DD, DR, and cached-estimator wall times separately."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from task43_ezmock_covariance_common import COMMON_RANDOM_HDF5, COMMON_RR, COMMON_RR_META, FCFC_BINARY, OUTPUT_ROOT, S_EDGES, sha256, write_json
from task43_ezmock_lightcone_common import MANIFEST as VALIDATION_MANIFEST
from task43_measure_ezmock_covariance_xi_fcfc import fifo_writer, load_pair
from task43_ezmock_covariance_common import read_jsonl
from task43_rawbox_ezmock_common import set_cpu_affinity


ROOT = OUTPUT_ROOT / "benchmark_fcfc_common50_ezmock_selection_split"
OUTPUT = ROOT / "task43_fcfc_common50_split_timing_ph100.json"


def env_threads() -> dict[str, str]:
    env = os.environ.copy(); env.update({"OMP_NUM_THREADS": "8", "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "close", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}); return env


def run_with_fifo(config: Path, log: Path, fifo: Path, catalog: Path) -> float:
    writer = mp.get_context("spawn").Process(target=fifo_writer, args=(str(fifo), str(catalog), 200_000)); writer.start(); started = time.perf_counter()
    try:
        with log.open("w", encoding="utf-8") as stream: subprocess.run([str(FCFC_BINARY), "-c", str(config)], check=True, stdout=stream, stderr=subprocess.STDOUT, env=env_threads())
    finally:
        writer.join(timeout=10)
        if writer.is_alive(): writer.terminate(); writer.join()
    if writer.exitcode not in (0, None): raise RuntimeError(f"FIFO writer failed: {writer.exitcode}")
    return time.perf_counter() - started


def main() -> None:
    set_cpu_affinity(8); ROOT.mkdir(parents=True, exist_ok=True)
    if OUTPUT.is_file():
        meta = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if meta.get("status") == "done" and meta.get("common_rr_sha256") == sha256(COMMON_RR): print(f"[skip verified] {OUTPUT}"); return
    rr_meta = json.loads(COMMON_RR_META.read_text(encoding="utf-8")); rr_hash = sha256(COMMON_RR)
    if rr_meta.get("rr_sha256") != rr_hash: raise RuntimeError("RR gate failed")
    row = read_jsonl(VALIDATION_MANIFEST)[0]; data = Path(row["halo_catalog_path"])
    dd, dr = ROOT / "DD_ph100_benchmark_s50_350.txt", ROOT / "DR_ph100_common50_benchmark_s50_350.txt"
    xi = ROOT / "xi_ph100_benchmark_cached.txt"
    with tempfile.TemporaryDirectory(prefix="ph100_", dir=ROOT) as tmp_name:
        tmp = Path(tmp_name)
        fifo_dd = tmp / "data_dd.pipe"; os.mkfifo(fifo_dd)
        cfg_dd = ROOT / "fcfc_ph100_DD_only.conf"; log_dd = ROOT / "fcfc_ph100_DD_only.log"
        cfg_dd.write_text(f"""CATALOG = '{fifo_dd}'
CATALOG_LABEL = D
CATALOG_TYPE = 0
ASCII_SKIP = 0
ASCII_COMMENT = '#'
ASCII_FORMATTER = '%lf %lf %lf %lf'
POSITION = [$1, $2, $3]
WEIGHT = $4
COORD_CONVERT = F
DATA_STRUCT = 0
BINNING_SCHEME = 0
PAIR_COUNT = DD
PAIR_COUNT_FILE = '{dd}'
SEP_BIN_MIN = 50
SEP_BIN_MAX = 350
SEP_BIN_SIZE = 10
OUTPUT_FORMAT = 1
OVERWRITE = 2
VERBOSE = T
""", encoding="utf-8")
        dd_sec = run_with_fifo(cfg_dd, log_dd, fifo_dd, data)
        fifo_dr = tmp / "data_dr.pipe"; os.mkfifo(fifo_dr)
        cfg_dr = ROOT / "fcfc_ph100_DR_only.conf"; log_dr = ROOT / "fcfc_ph100_DR_only.log"
        cfg_dr.write_text(f"""CATALOG = ['{fifo_dr}', '{COMMON_RANDOM_HDF5}']
CATALOG_LABEL = [D, R]
CATALOG_TYPE = [0, 2]
ASCII_SKIP = [0, 0]
ASCII_COMMENT = ['#', '#']
ASCII_FORMATTER = ['%lf %lf %lf %lf', '']
POSITION = [$1, $2, $3, ${{/X}}, ${{/Y}}, ${{/Zcart}}]
WEIGHT = [$4, ${{/WEIGHT_FKP}}]
COORD_CONVERT = F
DATA_STRUCT = 0
BINNING_SCHEME = 0
PAIR_COUNT = DR
PAIR_COUNT_FILE = '{dr}'
SEP_BIN_MIN = 50
SEP_BIN_MAX = 350
SEP_BIN_SIZE = 10
OUTPUT_FORMAT = 1
OVERWRITE = 2
VERBOSE = T
""", encoding="utf-8")
        dr_sec = run_with_fifo(cfg_dr, log_dr, fifo_dr, data)
        # All three pair tables now exist, so FCFC reads them and opens no catalog.
        fifo_cached = tmp / "unused_data.pipe"; os.mkfifo(fifo_cached)
        cfg_cached = ROOT / "fcfc_ph100_cached_estimator.conf"; log_cached = ROOT / "fcfc_ph100_cached_estimator.log"
        cfg_cached.write_text(f"""CATALOG = ['{fifo_cached}', '{COMMON_RANDOM_HDF5}']
CATALOG_LABEL = [D, R]
CATALOG_TYPE = [0, 2]
ASCII_SKIP = [0, 0]
ASCII_COMMENT = ['#', '#']
ASCII_FORMATTER = ['%lf %lf %lf %lf', '']
POSITION = [$1, $2, $3, ${{/X}}, ${{/Y}}, ${{/Zcart}}]
WEIGHT = [$4, ${{/WEIGHT_FKP}}]
COORD_CONVERT = F
DATA_STRUCT = 0
BINNING_SCHEME = 0
PAIR_COUNT = [DD, DR, RR]
PAIR_COUNT_FILE = ['{dd}', '{dr}', '{COMMON_RR}']
CF_ESTIMATOR = (DD - 2 * DR + RR) / RR
CF_OUTPUT_FILE = '{xi}'
SEP_BIN_MIN = 50
SEP_BIN_MAX = 350
SEP_BIN_SIZE = 10
OUTPUT_FORMAT = 1
OVERWRITE = 1
VERBOSE = T
""", encoding="utf-8")
        started = time.perf_counter()
        with log_cached.open("w", encoding="utf-8") as stream: subprocess.run([str(FCFC_BINARY), "-c", str(cfg_cached)], check=True, stdout=stream, stderr=subprocess.STDOUT, env=env_threads())
        cached_sec = time.perf_counter() - started
    dd_table, dr_table, rr_table = load_pair(dd), load_pair(dr), load_pair(COMMON_RR)
    xi_table = np.loadtxt(xi, comments="#", dtype="f8")
    if xi_table.shape != (30, 4) or np.any(rr_table[:, 2] <= 0): raise RuntimeError("split benchmark output gate failed")
    cache_log = log_cached.read_text(encoding="utf-8", errors="replace")
    if not all(f"<R> {path}" in cache_log for path in (dd, dr, COMMON_RR)): raise RuntimeError("cached estimator did not read all pair tables")
    payload = {"task": "task43_benchmark_fcfc_common50_split", "status": "done", "phase": row["phase"], "ndata_catalog": str(data), "nrandom": int(rr_meta["nrandom"]), "s_edges": S_EDGES, "threads": 8, "common_rr_sha256": rr_hash, "timing_sec": {"DD_including_data_fifo_and_tree": dd_sec, "DR_including_data_fifo_and_both_trees": dr_sec, "cached_DD_DR_RR_read_and_estimator": cached_sec, "DD_plus_DR_plus_cached_estimator": dd_sec + dr_sec + cached_sec, "one_time_RR": float(rr_meta["runtime_sec"])}, "paths": {"DD": str(dd), "DR": str(dr), "RR": str(COMMON_RR), "xi": str(xi)}, "normalized_pair_min": {"DD": float(dd_table[:, 2].min()), "DR": float(dr_table[:, 2].min()), "RR": float(rr_table[:, 2].min())}}
    write_json(OUTPUT, payload); print(f"[done] DD={dd_sec:.2f}s DR={dr_sec:.2f}s cache={cached_sec:.2f}s total={dd_sec+dr_sec+cached_sec:.2f}s")


if __name__ == "__main__": main()
