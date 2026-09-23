#!/usr/bin/env python3
"""A/B measure one historical fixed-amplitude validation lightcone with common50 RR."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from task43_ezmock_covariance_common import (
    COMMON_RANDOM_HDF5, COMMON_RR, COMMON_RR_META, FCFC_BINARY, OUTPUT_ROOT,
    S_EDGES, atomic_savez, read_jsonl, sha256, write_json,
)
from task43_ezmock_lightcone_common import MANIFEST as VALIDATION_MANIFEST
from task43_measure_ezmock_covariance_xi_fcfc import fifo_writer, load_pair
from task43_rawbox_ezmock_common import set_cpu_affinity


ROOT = OUTPUT_ROOT / "validation_ab_common50_ezmock_selection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def paths_for(row: dict[str, object]) -> dict[str, Path]:
    token = f"ezmock_{row['phase']}_seed{int(row['seed'])}"
    xi = ROOT / "xi_fcfc" / f"xi0_{token}_fixedamp_common50_s50_350_ds10_fcfc.npz"
    return {
        "xi": xi, "meta": xi.with_suffix(".json"), "text": xi.with_suffix(".txt"),
        "config": ROOT / "configs" / f"fcfc_{token}_common50.conf",
        "log": ROOT / "logs" / f"fcfc_{token}_common50.log",
        "dd": ROOT / "pairs" / f"DD_{token}_s50_350.txt",
        "dr": ROOT / "pairs" / f"DR_{token}_common50_s50_350.txt", "rr": COMMON_RR,
    }


def config_text(fifo: Path, paths: dict[str, Path]) -> str:
    return f"""CATALOG = ['{fifo}', '{COMMON_RANDOM_HDF5}']
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
PAIR_COUNT_FILE = ['{paths['dd']}', '{paths['dr']}', '{paths['rr']}']
CF_ESTIMATOR = (DD - 2 * DR + RR) / RR
CF_OUTPUT_FILE = '{paths['text']}'
SEP_BIN_MIN = 50
SEP_BIN_MAX = 350
SEP_BIN_SIZE = 10
OUTPUT_FORMAT = 1
OVERWRITE = 1
VERBOSE = T
"""


def main() -> None:
    args = parse_args()
    if not 1 <= args.threads <= 8: raise ValueError(args.threads)
    cpus = set_cpu_affinity(args.threads); rows = read_jsonl(VALIDATION_MANIFEST); row = rows[args.index]
    paths = paths_for(row)
    for path in paths.values(): path.parent.mkdir(parents=True, exist_ok=True)
    rr_meta = json.loads(COMMON_RR_META.read_text(encoding="utf-8")); rr_hash = sha256(COMMON_RR)
    if rr_meta.get("rr_sha256") != rr_hash: raise RuntimeError("common RR hash gate failed")
    if paths["xi"].is_file() and paths["meta"].is_file() and not args.overwrite:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        with np.load(paths["xi"], allow_pickle=False) as data:
            if meta.get("common_rr_sha256") == rr_hash and data["xi0"].shape == (30,):
                print(f"[skip verified] {paths['xi']}"); return
    halo_meta = json.loads(Path(str(row["halo_metadata_path"])).read_text(encoding="utf-8"))
    if halo_meta.get("fix_amplitude") is not True: raise RuntimeError("historical validation is not fixed-amplitude")
    pair_cached = {name: paths[name].is_file() for name in ("dd", "dr", "rr")}
    started = time.perf_counter(); tmp_root = ROOT / "tmp"; tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{row['phase']}_", dir=tmp_root) as tmp_name:
        fifo = Path(tmp_name) / "data.pipe"; os.mkfifo(fifo); paths["config"].write_text(config_text(fifo, paths), encoding="utf-8")
        writer = None
        if not pair_cached["dd"] or not pair_cached["dr"]:
            writer = mp.get_context("spawn").Process(target=fifo_writer, args=(str(fifo), str(row["halo_catalog_path"]), 200_000)); writer.start()
        env = os.environ.copy(); env.update({"OMP_NUM_THREADS": str(args.threads), "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "close", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
        try:
            with paths["log"].open("w", encoding="utf-8") as stream:
                subprocess.run([str(FCFC_BINARY), "-c", str(paths["config"])], check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
        finally:
            if writer is not None:
                writer.join(timeout=10)
                if writer.is_alive(): writer.terminate(); writer.join()
        if writer is not None and writer.exitcode not in (0, None): raise RuntimeError(f"FIFO writer exit={writer.exitcode}")
    table = np.loadtxt(paths["text"], comments="#", dtype="f8")
    if table.shape != (30, 4) or not np.array_equal(np.concatenate([table[:1, 1], table[:, 2]]), S_EDGES): raise RuntimeError("common50 A/B xi grid failed")
    dd, dr, rr = (load_pair(paths[name])[:, 2] for name in ("dd", "dr", "rr"))
    log_text = paths["log"].read_text(encoding="utf-8", errors="replace")
    if f"<R> {COMMON_RR}" not in log_text: raise RuntimeError("A/B FCFC did not read common RR")
    atomic_savez(paths["xi"], s=table[:, 0], s_edges=S_EDGES, xi0=table[:, 3], DD=dd, DR=dr, RR=rr, phase=np.asarray(row["phase"]), common_rr_sha256=np.asarray(rr_hash))
    meta = {"task": "task43_measure_validation_common50_xi_fcfc", "status": "done", "classification": "historical_fixedamp_x10_common50_ab_test", "phase": row["phase"], "seed": int(row["seed"]), "fix_amplitude": True, "common_rr_sha256": rr_hash, "rr_cache_read_gate": True, "threads": args.threads, "cpu_affinity": cpus, "runtime_sec": float(time.perf_counter() - started), "old25_xi": row["xi_path"], "output": str(paths["xi"])}
    write_json(paths["meta"], meta); print(f"[done] {row['phase']} common50 A/B elapsed={meta['runtime_sec']:.1f}s")


if __name__ == "__main__": main()
