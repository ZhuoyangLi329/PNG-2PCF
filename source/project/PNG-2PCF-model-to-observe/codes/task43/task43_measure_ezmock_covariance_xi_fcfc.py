#!/usr/bin/env python3
"""Measure one production EZmock xi0 with FCFC and the immutable common RR."""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing as mp
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from task43_ezmock_covariance_common import (
    ABACUS_FKP_SUMMARY,
    COMMON_RANDOM,
    COMMON_RANDOM_HDF5,
    COMMON_RANDOM_HDF5_META,
    COMMON_RANDOM_SIZE,
    COMMON_RR,
    COMMON_RR_META,
    FCFC_BINARY,
    FCFC_CONFIG_DIR,
    FCFC_PAIR_DIR,
    LOG_DIR,
    MANIFEST,
    P0,
    S_EDGES,
    TMP_DIR,
    atomic_savez,
    read_jsonl,
    sha256,
    write_json,
)
from task43_fkp_zeff import load_fkp_summary, total_weight_from_summary
from task43_rawbox_ezmock_common import set_cpu_affinity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--stream-chunk", type=int, default=200_000)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite xi products but keep valid pair caches.")
    return parser.parse_args()


def paths_for(row: dict[str, object]) -> dict[str, Path]:
    token = f"ezmock_{row['phase']}_seed{int(row['seed'])}"
    xi = Path(str(row["xi_path"]))
    return {
        "xi_npz": xi,
        "xi_json": xi.with_suffix(".json"),
        "xi_text": xi.with_suffix(".txt"),
        "config": FCFC_CONFIG_DIR / f"fcfc_{token}_common50_s50_350.conf",
        "log": LOG_DIR / f"fcfc_{token}_common50_s50_350.log",
        "dd": FCFC_PAIR_DIR / f"DD_{token}_s50_350_ds10.txt",
        "dr": FCFC_PAIR_DIR / f"DR_{token}_common50_s50_350_ds10.txt",
        "rr": COMMON_RR,
    }


def load_pair(path: Path) -> np.ndarray:
    table = np.loadtxt(path, comments="#", dtype="f8")
    if table.shape != (S_EDGES.size - 1, 3) or not np.all(np.isfinite(table)):
        raise ValueError(f"invalid FCFC pair table {path}: {table.shape}")
    edges = np.concatenate([table[:1, 0], table[:, 1]])
    if not np.array_equal(edges, S_EDGES):
        raise ValueError(f"FCFC pair grid mismatch in {path}")
    return table


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        position = np.column_stack([
            np.asarray(data["X"], dtype="f8"),
            np.asarray(data["Y"], dtype="f8"),
            np.asarray(data["Zcart"], dtype="f8"),
        ])
        redshift = np.asarray(data["Z"], dtype="f8")
        base = np.asarray(data["WEIGHT"], dtype="f8") if "WEIGHT" in data.files else np.ones(redshift.size, dtype="f8")
    fkp = load_fkp_summary(ABACUS_FKP_SUMMARY)
    weight = total_weight_from_summary(redshift, base, fkp, p0=P0)
    return position, redshift, weight


def fifo_writer(fifo: str, catalog: str, chunk: int) -> None:
    try:
        position, _, weight = load_data(Path(catalog))
        with Path(fifo).open("w", encoding="utf-8", buffering=1024 * 1024) as stream:
            for start in range(0, position.shape[0], int(chunk)):
                stop = min(start + int(chunk), position.shape[0])
                np.savetxt(stream, np.column_stack([position[start:stop], weight[start:stop]]), fmt="%.9g %.9g %.9g %.12g")
    except BrokenPipeError:
        return


def config_text(data_fifo: Path, paths: dict[str, Path]) -> str:
    return f"""CATALOG = ['{data_fifo}', '{COMMON_RANDOM_HDF5}']
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
CF_OUTPUT_FILE = '{paths['xi_text']}'
SEP_BIN_MIN = {S_EDGES[0]:.17g}
SEP_BIN_MAX = {S_EDGES[-1]:.17g}
SEP_BIN_SIZE = {np.median(np.diff(S_EDGES)):.17g}
OUTPUT_FORMAT = 1
OVERWRITE = 1
VERBOSE = T
"""


def validate_common() -> dict[str, object]:
    for path in (COMMON_RANDOM, COMMON_RANDOM_HDF5, COMMON_RANDOM_HDF5_META, COMMON_RR, COMMON_RR_META, FCFC_BINARY):
        if not path.is_file():
            raise FileNotFoundError(path)
    h5meta = json.loads(COMMON_RANDOM_HDF5_META.read_text(encoding="utf-8"))
    rrmeta = json.loads(COMMON_RR_META.read_text(encoding="utf-8"))
    source_hash, h5_hash, rr_hash = sha256(COMMON_RANDOM), sha256(COMMON_RANDOM_HDF5), sha256(COMMON_RR)
    if not (
        h5meta.get("source_npz_sha256") == source_hash
        and h5meta.get("sha256") == h5_hash
        and rrmeta.get("common_random_npz_sha256") == source_hash
        and rrmeta.get("common_random_hdf5_sha256") == h5_hash
        and rrmeta.get("rr_sha256") == rr_hash
    ):
        raise RuntimeError("common random/RR immutable provenance gate failed")
    load_pair(COMMON_RR)
    return {"source_hash": source_hash, "hdf5_hash": h5_hash, "rr_hash": rr_hash, "fkp_hash": sha256(ABACUS_FKP_SUMMARY)}


def validate_existing(row: dict[str, object], paths: dict[str, Path], common: dict[str, object]) -> bool:
    if not paths["xi_npz"].is_file() or not paths["xi_json"].is_file():
        return False
    meta = json.loads(paths["xi_json"].read_text(encoding="utf-8"))
    if not (
        meta.get("status") == "done"
        and meta.get("classification") == "covariance_production_fixampF_common50"
        and meta.get("fix_amplitude") is False
        and int(meta.get("seed", -1)) == int(row["seed"])
        and meta.get("common_rr_sha256") == common["rr_hash"]
        and meta.get("fkp_summary_sha256") == common["fkp_hash"]
    ):
        return False
    with np.load(paths["xi_npz"], allow_pickle=False) as data:
        return data["xi0"].shape == (30,) and np.array_equal(data["s_edges"], S_EDGES) and np.all(np.isfinite(data["xi0"]))


def _measure_one_locked(row: dict[str, object], *, threads: int, chunk: int, overwrite: bool) -> dict[str, object]:
    paths = paths_for(row)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    common = validate_common()
    if not overwrite and validate_existing(row, paths, common):
        print(f"[skip verified] {paths['xi_npz']}")
        return json.loads(paths["xi_json"].read_text(encoding="utf-8"))
    data_path = Path(str(row["halo_catalog_path"])); halo_meta_path = Path(str(row["halo_metadata_path"]))
    if not data_path.is_file() or not halo_meta_path.is_file():
        raise FileNotFoundError(data_path)
    halo_meta = json.loads(halo_meta_path.read_text(encoding="utf-8"))
    if halo_meta.get("fix_amplitude") is not False or int(halo_meta.get("seed", -1)) != int(row["seed"]):
        raise RuntimeError("halo catalog failed FIX_AMPLITUDE=F provenance gate")
    position, redshift, weight = load_data(data_path)
    ndata = int(redshift.size)
    data_diag = {
        "ndata": ndata, "z_min": float(redshift.min()), "z_max": float(redshift.max()),
        "weight_min": float(weight.min()), "weight_max": float(weight.max()),
        "weight_sum": float(weight.sum()), "weight2_sum": float(np.dot(weight, weight)),
        "position_min": position.min(axis=0), "position_max": position.max(axis=0),
    }
    del position, redshift, weight
    pair_cached = {name: paths[name].is_file() for name in ("dd", "dr", "rr")}
    if pair_cached["rr"] is not True:
        raise RuntimeError("common RR vanished before FCFC measurement")

    tmp_root = TMP_DIR / "fcfc_fifo"; tmp_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"{row['phase']}_", dir=tmp_root) as tmp_name:
        fifo = Path(tmp_name) / "data.pipe"; os.mkfifo(fifo)
        paths["config"].write_text(config_text(fifo, paths), encoding="utf-8")
        need_data = not pair_cached["dd"] or not pair_cached["dr"]
        writer = None
        if need_data:
            writer = mp.get_context("spawn").Process(target=fifo_writer, args=(str(fifo), str(data_path), int(chunk)))
            writer.start()
        env = os.environ.copy(); env.update({
            "OMP_NUM_THREADS": str(int(threads)), "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "close",
            "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        })
        try:
            with paths["log"].open("w", encoding="utf-8") as stream:
                subprocess.run([str(FCFC_BINARY), "-c", str(paths["config"])], check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
        finally:
            if writer is not None:
                writer.join(timeout=10)
                if writer.is_alive():
                    writer.terminate(); writer.join()
        if writer is not None and writer.exitcode not in (0, None):
            raise RuntimeError(f"data FIFO writer failed with exit code {writer.exitcode}")
    fcfc_sec = time.perf_counter() - started

    xi_table = np.loadtxt(paths["xi_text"], comments="#", dtype="f8")
    if xi_table.shape != (30, 4) or not np.all(np.isfinite(xi_table)):
        raise ValueError(f"invalid FCFC xi table {xi_table.shape}")
    dd, dr, rr = (load_pair(paths[name])[:, 2] for name in ("dd", "dr", "rr"))
    s_edges = np.concatenate([xi_table[:1, 1], xi_table[:, 2]])
    if not np.array_equal(s_edges, S_EDGES) or np.any(rr <= 0.0):
        raise RuntimeError("FCFC xi output grid/RR gate failed")
    log_text = paths["log"].read_text(encoding="utf-8", errors="replace")
    if f"<R> {COMMON_RR}" not in log_text:
        raise RuntimeError("FCFC did not report reading the immutable common RR cache")
    atomic_savez(
        paths["xi_npz"], s=xi_table[:, 0], s_edges=s_edges, xi0=xi_table[:, 3],
        DD=dd, DR=dr, RR=rr, ndata=np.asarray(ndata, dtype="i8"), nrandom=np.asarray(COMMON_RANDOM_SIZE, dtype="i8"),
        phase=np.asarray(row["phase"]), seed=np.asarray(int(row["seed"]), dtype="i8"), fix_amplitude=np.asarray(False),
        common_random_npz_sha256=np.asarray(common["source_hash"]), common_random_hdf5_sha256=np.asarray(common["hdf5_hash"]),
        common_rr_sha256=np.asarray(common["rr_hash"]), fkp_summary_sha256=np.asarray(common["fkp_hash"]),
    )
    summary = {
        "task": "task43_measure_ezmock_covariance_xi_fcfc", "status": "done",
        "classification": "covariance_production_fixampF_common50", "production_index": int(row["production_index"]),
        "phase": row["phase"], "seed": int(row["seed"]), "fix_amplitude": False,
        "data_catalog": str(data_path), "data": data_diag, "nrandom": COMMON_RANDOM_SIZE,
        "fkp_summary": str(ABACUS_FKP_SUMMARY), "fkp_summary_sha256": common["fkp_hash"], "p0": P0,
        "estimator": "weighted aperiodic FCFC Landy-Szalay (DD-2DR+RR)/RR",
        "s_edges": S_EDGES, "n_xi_bins": 30,
        "common_random_npz_sha256": common["source_hash"], "common_random_hdf5_sha256": common["hdf5_hash"],
        "common_rr": str(COMMON_RR), "common_rr_sha256": common["rr_hash"], "rr_cache_read_gate": True,
        "fcfc_binary": str(FCFC_BINARY), "fcfc_binary_sha256": sha256(FCFC_BINARY),
        "config": str(paths["config"]), "config_sha256": sha256(paths["config"]), "log": str(paths["log"]),
        "pair_counts": {name.upper(): str(paths[name]) for name in ("dd", "dr", "rr")},
        "pair_cache_before": pair_cached, "threads": int(threads), "runtime_sec": float(fcfc_sec),
        "outputs": {"npz": str(paths["xi_npz"]), "json": str(paths["xi_json"]), "text": str(paths["xi_text"])},
    }
    write_json(paths["xi_json"], summary)
    print(f"[done] {row['phase']} xi bins=30 ndata={ndata} commonRR=read elapsed={fcfc_sec:.1f}s")
    return summary


def measure_one(row: dict[str, object], *, threads: int, chunk: int, overwrite: bool) -> dict[str, object]:
    """Serialize duplicate FCFC requests for one manifest row.

    Resumable supervisors may briefly overlap during a restart.  DD/DR, the
    generated FCFC config/log, and the final xi products all have row-stable
    paths, so acquire a row lock before validating or mutating any of them.
    """
    lock_dir = TMP_DIR / "row_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"xi_{row['phase']}_seed{int(row['seed'])}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        return _measure_one_locked(
            row,
            threads=int(threads),
            chunk=int(chunk),
            overwrite=bool(overwrite),
        )


def main() -> None:
    args = parse_args()
    if not 1 <= int(args.threads) <= 12:
        raise ValueError("login-node FCFC contract requires 1..12 threads")
    cpus = set_cpu_affinity(int(args.threads)); print(f"[cpu] affinity={cpus}")
    rows = read_jsonl(args.manifest)
    if not 0 <= int(args.index) < len(rows):
        raise IndexError(args.index)
    measure_one(rows[int(args.index)], threads=int(args.threads), chunk=int(args.stream_chunk), overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
