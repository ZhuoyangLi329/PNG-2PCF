#!/usr/bin/env python3
"""Measure production EZmock RSD xi0 and xi2 with FCFC and common s-mu RR."""

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

_INITIAL_CPUS = tuple(sorted(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else ()
import numpy as np

from task43_ezmock_rsd_covariance_common import (
    ABACUS_FKP_SUMMARY,
    COMMON_RANDOM,
    COMMON_RANDOM_HDF5,
    COMMON_RANDOM_HDF5_META,
    COMMON_RANDOM_SIZE,
    COMMON_RR_SMU,
    COMMON_RR_SMU_META,
    FCFC_BINARY,
    MANIFEST,
    MU_BIN_NUM,
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
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--stream-chunk", type=int, default=200_000)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite xi products but keep valid pair caches.")
    return parser.parse_args()


def classification_for(row: dict[str, object]) -> str:
    return ("RSD_lightcone_%s" % row["flavor"]) if row.get("flavor") else "RSD_lightcone_covariance_fixampF_common50"


def paths_for(row: dict[str, object]) -> dict[str, Path]:
    token = f"ezmock_{row['phase']}_seed{int(row['seed'])}"
    xi = Path(str(row["xi_path"]))
    # Work root from the row's own xi_path (<root>/xi_fcfc/<token>...) so pilot
    # manifests keep configs/logs/DD/DR out of (and never reuse caches of) the
    # frozen production root.  For production rows this is OUTPUT_ROOT as before.
    root = xi.parent.parent
    return {
        "xi_npz": xi,
        "xi_json": xi.with_suffix(".json"),
        "xi_text": xi.with_suffix(".txt"),
        "smu_text": xi.with_name(xi.stem.replace("_xi02_", "_smu_") + ".txt"),
        "config": root / "fcfc_configs" / f"fcfc_{token}_common50_smu_s30_350.conf",
        "log": root / "logs" / f"fcfc_{token}_common50_smu_s30_350.log",
        "dd": root / "fcfc_pairs" / f"DD_{token}_s30_350_ds10_mu{MU_BIN_NUM}.bin",
        "dr": root / "fcfc_pairs" / f"DR_{token}_common50_s30_350_ds10_mu{MU_BIN_NUM}.bin",
        "rr": COMMON_RR_SMU,
    }


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
BINNING_SCHEME = 1
PAIR_COUNT = [DD, DR, RR]
PAIR_COUNT_FILE = ['{paths['dd']}', '{paths['dr']}', '{paths['rr']}']
CF_ESTIMATOR = (DD - 2 * DR + RR) / RR
CF_OUTPUT_FILE = '{paths['smu_text']}'
MULTIPOLE = [0, 2]
MULTIPOLE_FILE = '{paths['xi_text']}'
SEP_BIN_MIN = {S_EDGES[0]:.17g}
SEP_BIN_MAX = {S_EDGES[-1]:.17g}
SEP_BIN_SIZE = {np.median(np.diff(S_EDGES)):.17g}
MU_BIN_NUM = {MU_BIN_NUM}
OUTPUT_FORMAT = 0
OVERWRITE = 1
VERBOSE = T
"""


def load_multipoles(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = np.loadtxt(path, comments="#", dtype="f8")
    if table.ndim == 1:
        table = table[None, :]
    expected = (S_EDGES.size - 1, 5)
    if table.shape != expected or not np.all(np.isfinite(table)):
        raise ValueError(f"invalid FCFC multipole table {path}: {table.shape}")
    s, slow, shigh, xi0, xi2 = table.T
    if not np.allclose(shigh[:-1], slow[1:], atol=1.0e-12, rtol=0.0):
        raise ValueError(f"non-contiguous FCFC s edges in {path}")
    edges = np.concatenate([slow[:1], shigh])
    if not np.array_equal(edges, S_EDGES):
        raise ValueError(f"FCFC s grid mismatch in {path}")
    return s, edges, xi0, xi2, table


def validate_common() -> dict[str, object]:
    for path in (COMMON_RANDOM, COMMON_RANDOM_HDF5, COMMON_RANDOM_HDF5_META, COMMON_RR_SMU, COMMON_RR_SMU_META, FCFC_BINARY):
        if not path.is_file():
            raise FileNotFoundError(path)
    h5meta = json.loads(COMMON_RANDOM_HDF5_META.read_text(encoding="utf-8"))
    rrmeta = json.loads(COMMON_RR_SMU_META.read_text(encoding="utf-8"))
    source_hash, h5_hash, rr_hash = sha256(COMMON_RANDOM), sha256(COMMON_RANDOM_HDF5), sha256(COMMON_RR_SMU)
    if not (
        h5meta.get("source_npz_sha256") == source_hash
        and h5meta.get("sha256") == h5_hash
        and rrmeta.get("common_random_npz_sha256") == source_hash
        and rrmeta.get("common_random_hdf5_sha256") == h5_hash
        and rrmeta.get("rr_sha256") == rr_hash
        and int(rrmeta.get("mu_bin_num", -1)) == MU_BIN_NUM
    ):
        raise RuntimeError("common random/s-mu RR immutable provenance gate failed")
    return {"source_hash": source_hash, "hdf5_hash": h5_hash, "rr_hash": rr_hash, "fkp_hash": sha256(ABACUS_FKP_SUMMARY)}


def validate_existing(row: dict[str, object], paths: dict[str, Path], common: dict[str, object]) -> bool:
    if not paths["xi_npz"].is_file() or not paths["xi_json"].is_file() or not paths["xi_text"].is_file():
        return False
    meta = json.loads(paths["xi_json"].read_text(encoding="utf-8"))
    if not (
        meta.get("status") == "done"
        and meta.get("classification") == classification_for(row)
        and "fix_amplitude" in meta
        and bool(meta["fix_amplitude"]) == bool(row["fix_amplitude"])
        and int(meta.get("seed", -1)) == int(row["seed"])
        and meta.get("common_rr_smu_sha256") == common["rr_hash"]
        and meta.get("fkp_summary_sha256") == common["fkp_hash"]
    ):
        return False
    with np.load(paths["xi_npz"], allow_pickle=False) as data:
        return (
            data["xi0"].shape == (S_EDGES.size - 1,)
            and data["xi2"].shape == (S_EDGES.size - 1,)
            and np.array_equal(data["s_edges"], S_EDGES)
            and np.all(np.isfinite(data["xi0"]))
            and np.all(np.isfinite(data["xi2"]))
        )


def _measure_one_locked(row: dict[str, object], *, threads: int, chunk: int, overwrite: bool) -> dict[str, object]:
    paths = paths_for(row)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    common = validate_common()
    if not overwrite and validate_existing(row, paths, common):
        print(f"[skip verified] {paths['xi_npz']}")
        return json.loads(paths["xi_json"].read_text(encoding="utf-8"))
    data_path = Path(str(row.get("halo_catalog_path", row.get("lightcone_catalog_path"))))
    halo_meta_path = Path(str(row.get("halo_metadata_path", row.get("lightcone_metadata_path"))))
    if not data_path.is_file() or not halo_meta_path.is_file():
        raise FileNotFoundError(data_path)
    halo_meta = json.loads(halo_meta_path.read_text(encoding="utf-8"))
    row_fix_amplitude = bool(row["fix_amplitude"])
    fix_flag = "T" if row_fix_amplitude else "F"
    if ("fix_amplitude" not in halo_meta or bool(halo_meta["fix_amplitude"]) != row_fix_amplitude
            or int(halo_meta.get("seed", -1)) != int(row["seed"])):
        raise RuntimeError(f"halo catalog failed FIX_AMPLITUDE={fix_flag} provenance gate")
    position, redshift, weight = load_data(data_path)
    ndata = int(redshift.size)
    data_diag = {"ndata": ndata, "z_min": float(redshift.min()), "z_max": float(redshift.max()), "weight_min": float(weight.min()), "weight_max": float(weight.max()), "weight_sum": float(weight.sum()), "weight2_sum": float(np.dot(weight, weight)), "position_min": position.min(axis=0), "position_max": position.max(axis=0)}
    del position, redshift, weight
    pair_cached = {name: paths[name].is_file() for name in ("dd", "dr", "rr")}
    if not pair_cached["rr"]:
        raise RuntimeError("common s-mu RR vanished before FCFC measurement")
    tmp_root = TMP_DIR / "fcfc_fifo"
    tmp_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"{row['phase']}_", dir=tmp_root) as tmp_name:
        fifo = Path(tmp_name) / "data.pipe"
        os.mkfifo(fifo)
        paths["config"].write_text(config_text(fifo, paths), encoding="utf-8")
        need_data = not pair_cached["dd"] or not pair_cached["dr"]
        writer = None
        if need_data:
            writer = mp.get_context("spawn").Process(target=fifo_writer, args=(str(fifo), str(data_path), int(chunk)))
            writer.start()
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": str(int(threads)), "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "close", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
        try:
            with paths["log"].open("w", encoding="utf-8") as stream:
                subprocess.run([str(FCFC_BINARY), "-c", str(paths["config"])], check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
        finally:
            if writer is not None:
                writer.join(timeout=10)
                if writer.is_alive():
                    writer.terminate()
                    writer.join()
        if writer is not None and writer.exitcode not in (0, None):
            raise RuntimeError(f"data FIFO writer failed with exit code {writer.exitcode}")
    fcfc_sec = time.perf_counter() - started
    s, s_edges, xi0, xi2, _ = load_multipoles(paths["xi_text"])
    for name in ("dd", "dr", "rr"):
        if not paths[name].is_file() or paths[name].stat().st_size <= 0:
            raise RuntimeError(f"missing FCFC s-mu pair cache {paths[name]}")
    log_text = paths["log"].read_text(encoding="utf-8", errors="replace")
    if f"<R> {COMMON_RR_SMU}" not in log_text:
        raise RuntimeError("FCFC did not report reading the immutable common s-mu RR cache")
    atomic_savez(paths["xi_npz"], s=s, s_edges=s_edges, xi0=xi0, xi2=xi2, ndata=np.asarray(ndata, dtype="i8"), nrandom=np.asarray(COMMON_RANDOM_SIZE, dtype="i8"), phase=np.asarray(row["phase"]), seed=np.asarray(int(row["seed"]), dtype="i8"), fix_amplitude=np.asarray(row_fix_amplitude), ells=np.asarray([0, 2], dtype="i4"), mu_bin_num=np.asarray(MU_BIN_NUM, dtype="i4"), common_random_npz_sha256=np.asarray(common["source_hash"]), common_random_hdf5_sha256=np.asarray(common["hdf5_hash"]), common_rr_smu_sha256=np.asarray(common["rr_hash"]), fkp_summary_sha256=np.asarray(common["fkp_hash"]))
    summary = {"task": "task43_measure_ezmock_rsd_lightcone_xi_fcfc", "status": "done", "classification": classification_for(row), "production_index": int(row["production_index"]), "phase": row["phase"], "seed": int(row["seed"]), "flavor": row.get("flavor"), "fix_amplitude": row_fix_amplitude, "data_catalog": str(data_path), "data": data_diag, "nrandom": COMMON_RANDOM_SIZE, "fkp_summary": str(ABACUS_FKP_SUMMARY), "fkp_summary_sha256": common["fkp_hash"], "p0": P0, "estimator": "weighted aperiodic FCFC Landy-Szalay multipoles from common s-mu RR", "s_edges": S_EDGES, "n_xi_bins": int(S_EDGES.size - 1), "ells": [0, 2], "mu_bin_num": MU_BIN_NUM, "common_random_npz_sha256": common["source_hash"], "common_random_hdf5_sha256": common["hdf5_hash"], "common_rr_smu": str(COMMON_RR_SMU), "common_rr_smu_sha256": common["rr_hash"], "rr_cache_read_gate": True, "fcfc_binary": str(FCFC_BINARY), "fcfc_binary_sha256": sha256(FCFC_BINARY), "config": str(paths["config"]), "config_sha256": sha256(paths["config"]), "log": str(paths["log"]), "pair_counts": {name.upper(): str(paths[name]) for name in ("dd", "dr", "rr")}, "pair_cache_before": pair_cached, "pair_cache_sha256": {name.upper(): sha256(paths[name]) for name in ("dd", "dr", "rr")}, "threads": int(threads), "runtime_sec": float(fcfc_sec), "outputs": {"npz": str(paths["xi_npz"]), "json": str(paths["xi_json"]), "multipole_text": str(paths["xi_text"]), "smu_text": str(paths["smu_text"])}}
    write_json(paths["xi_json"], summary)
    print(f"[done] {row['phase']} xi0+xi2 bins={S_EDGES.size - 1} ndata={ndata} commonRR_smu=read elapsed={fcfc_sec:.1f}s")
    return summary


def measure_one(row: dict[str, object], *, threads: int, chunk: int, overwrite: bool) -> dict[str, object]:
    lock_dir = TMP_DIR / "row_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"xi_{row['phase']}_seed{int(row['seed'])}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        return _measure_one_locked(row, threads=int(threads), chunk=int(chunk), overwrite=bool(overwrite))


def main() -> None:
    args = parse_args()
    if not 1 <= int(args.threads) <= 64:
        raise ValueError("FCFC compute contract requires 1..64 threads")
    requested = int(args.threads)
    if _INITIAL_CPUS:
        cpus = list(_INITIAL_CPUS[:requested])
        if len(cpus) != requested:
            raise RuntimeError(f"only {len(_INITIAL_CPUS)} pre-NumPy CPUs available, need {requested}")
        os.sched_setaffinity(0, cpus)
    else:
        cpus = set_cpu_affinity(requested)
    print(f"[cpu] affinity={cpus}")
    rows = read_jsonl(args.manifest)
    if not 0 <= int(args.index) < len(rows):
        raise IndexError(args.index)
    row = dict(rows[int(args.index)])
    row.setdefault("halo_catalog_path", row.get("lightcone_catalog_path"))
    row.setdefault("halo_metadata_path", row.get("lightcone_metadata_path"))
    if not row["halo_catalog_path"] or not row["halo_metadata_path"]:
        raise KeyError("manifest row lacks lightcone catalog/metadata paths")
    measure_one(row, threads=int(args.threads), chunk=int(args.stream_chunk), overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
