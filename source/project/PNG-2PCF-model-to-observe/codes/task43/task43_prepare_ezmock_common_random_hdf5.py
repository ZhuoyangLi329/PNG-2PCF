#!/usr/bin/env python3
"""Freeze the common50 random as a direct FCFC HDF5 input with FKP weights."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np

from task43_ezmock_covariance_common import (
    ABACUS_FKP_SUMMARY,
    COMMON_RANDOM,
    COMMON_RANDOM_HDF5,
    COMMON_RANDOM_HDF5_META,
    COMMON_RANDOM_META,
    COMMON_RANDOM_SIZE,
    P0,
    sha256,
    write_json,
)
from task43_fkp_zeff import fkp_bin_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate(path: Path, meta_path: Path, source_hash: str) -> bool:
    if not path.is_file() or not meta_path.is_file():
        return False
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("source_npz_sha256") != source_hash or meta.get("status") != "done":
        return False
    with h5py.File(path, "r") as handle:
        return all(
            name in handle
            and handle[name].shape == (COMMON_RANDOM_SIZE,)
            and handle[name].dtype == np.dtype("f4")
            for name in ("X", "Y", "Zcart", "Z", "WEIGHT_FKP")
        )


def main() -> None:
    args = parse_args()
    for path in (COMMON_RANDOM, COMMON_RANDOM_META, ABACUS_FKP_SUMMARY):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_hash = sha256(COMMON_RANDOM)
    if not args.overwrite and validate(COMMON_RANDOM_HDF5, COMMON_RANDOM_HDF5_META, source_hash):
        print(f"[skip verified] {COMMON_RANDOM_HDF5}")
        return

    started = time.perf_counter()
    with np.load(COMMON_RANDOM, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name], dtype="f4") for name in ("X", "Y", "Zcart", "Z")}
    if any(array.shape != (COMMON_RANDOM_SIZE,) for array in arrays.values()):
        raise ValueError("common random source shape mismatch")
    with np.load(ABACUS_FKP_SUMMARY, allow_pickle=False) as fkp:
        z_edges = np.asarray(fkp["z_edges"], dtype="f8")
        nbar = np.asarray(fkp["nbar"], dtype="f8")
    ibin = np.clip(np.searchsorted(z_edges, arrays["Z"], side="right") - 1, 0, nbar.size - 1)
    weight = np.asarray(fkp_bin_weights(nbar, P0)[ibin], dtype="f4")
    if not np.all(np.isfinite(weight)) or np.any(weight <= 0.0):
        raise RuntimeError("invalid FKP weights for common random")

    COMMON_RANDOM_HDF5.parent.mkdir(parents=True, exist_ok=True)
    tmp = COMMON_RANDOM_HDF5.with_name(f".{COMMON_RANDOM_HDF5.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)
    try:
        with h5py.File(tmp, "w") as handle:
            handle.attrs["classification"] = "immutable common50 single-snapshot EZmock-shell FCFC transport"
            handle.attrs["source_npz_sha256"] = source_hash
            handle.attrs["fkp_p0"] = P0
            for name, array in arrays.items():
                handle.create_dataset(name, data=array, chunks=(262_144,))
            handle.create_dataset("WEIGHT_FKP", data=weight, chunks=(262_144,))
            handle.flush()
        tmp.replace(COMMON_RANDOM_HDF5)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    transport_hash = sha256(COMMON_RANDOM_HDF5)
    meta = {
        "task": "task43_prepare_ezmock_common_random_hdf5",
        "status": "done",
        "classification": "immutable common50 single-snapshot EZmock-shell FCFC HDF5 transport; same objects as source NPZ",
        "path": str(COMMON_RANDOM_HDF5),
        "sha256": transport_hash,
        "source_npz": str(COMMON_RANDOM),
        "source_npz_sha256": source_hash,
        "fkp_summary": str(ABACUS_FKP_SUMMARY),
        "p0": P0,
        "nrandom": COMMON_RANDOM_SIZE,
        "datasets": {name: {"shape": list(array.shape), "dtype": str(array.dtype)} for name, array in {**arrays, "WEIGHT_FKP": weight}.items()},
        "weight_min": float(weight.min()),
        "weight_max": float(weight.max()),
        "weight_sum": float(weight.sum(dtype="f8")),
        "weight2_sum": float(np.dot(weight.astype("f8"), weight.astype("f8"))),
        "runtime_sec": float(time.perf_counter() - started),
    }
    write_json(COMMON_RANDOM_HDF5_META, meta)
    print(f"[write] {COMMON_RANDOM_HDF5} n={COMMON_RANDOM_SIZE} sha256={transport_hash} elapsed={meta['runtime_sec']:.1f}s")


if __name__ == "__main__":
    main()
