#!/usr/bin/env python3
"""Prepare the immutable common random HDF5 transport for FCFC."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

from task43_ezmock_rsd_covariance_common import (
    COMMON_RANDOM, COMMON_RANDOM_HDF5, COMMON_RANDOM_HDF5_META, P0_FKP,
    sha256, write_json,
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not COMMON_RANDOM.is_file():
        raise FileNotFoundError(COMMON_RANDOM)
    if COMMON_RANDOM_HDF5.exists() and not args.overwrite:
        raise FileExistsError(COMMON_RANDOM_HDF5)
    t0 = time.perf_counter()
    with np.load(COMMON_RANDOM, allow_pickle=False) as src:
        arrays = {name: np.asarray(src[name], dtype="f4") for name in ("X","Y","Zcart","Z","WEIGHT_FKP")}
    n = int(arrays["Z"].size)
    if any(arr.shape != (n,) for arr in arrays.values()) or not np.all(np.isfinite(np.column_stack(list(arrays.values())))):
        raise RuntimeError("common random arrays are invalid")
    COMMON_RANDOM_HDF5.parent.mkdir(parents=True, exist_ok=True)
    tmp = COMMON_RANDOM_HDF5.with_name(f".{COMMON_RANDOM_HDF5.name}.{__import__('os').getpid()}.tmp")
    tmp.unlink(missing_ok=True)
    with h5py.File(tmp, "w") as handle:
        handle.attrs["classification"] = "immutable common50 EZmock RSD target-nz FCFC transport"
        handle.attrs["source_npz_sha256"] = sha256(COMMON_RANDOM)
        handle.attrs["p0_fkp"] = P0_FKP
        for name, arr in arrays.items():
            handle.create_dataset(name, data=arr, chunks=(262144,))
        handle.flush()
    tmp.replace(COMMON_RANDOM_HDF5)
    meta = {
        "task": "task43_prepare_ezmock_rsd_common_random_hdf5",
        "status": "done",
        "classification": "immutable_common50_target_nz",
        "source_npz": str(COMMON_RANDOM),
        "source_npz_sha256": sha256(COMMON_RANDOM),
        "path": str(COMMON_RANDOM_HDF5),
        "sha256": sha256(COMMON_RANDOM_HDF5),
        "nrandom": n,
        "datasets": {name: {"shape": list(arr.shape), "dtype": str(arr.dtype)} for name, arr in arrays.items()},
        "runtime_sec": float(time.perf_counter()-t0),
    }
    write_json(COMMON_RANDOM_HDF5_META, meta)
    print(f"[done] {COMMON_RANDOM_HDF5} n={n} sha256={meta['sha256']} elapsed={meta['runtime_sec']:.1f}s")

if __name__ == "__main__":
    main()
