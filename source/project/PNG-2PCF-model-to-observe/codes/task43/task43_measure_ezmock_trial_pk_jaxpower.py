#!/usr/bin/env python3
"""Measure one periodic EZmock trial P0(k) with the Task43 jaxpower setup."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
DESI_CLUSTERING_ROOT = Path("/pscratch/sd/l/lzy/desi-clustering")
for _path in (CODE_DIR, DESI_CLUSTERING_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from task43_rawbox_ezmock_common import atomic_savez, set_cpu_affinity, write_json  # noqa: E402


def make_edges(kmin: float, kmax: float, dk: float) -> np.ndarray:
    edges = np.arange(kmin, kmax + 0.5 * dk, dk, dtype="f8")
    if edges[-1] < kmax:
        edges = np.append(edges, kmax)
    edges[0] = kmin
    edges[-1] = kmax
    return np.column_stack([edges[:-1], edges[1:]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boxsize", type=float, default=2000.0)
    parser.add_argument("--meshsize", type=int, default=400)
    parser.add_argument("--kmin", type=float, default=0.001)
    parser.add_argument("--kmax", type=float, default=0.3001)
    parser.add_argument("--dk", type=float, default=0.002)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cpus = set_cpu_affinity(args.threads)
    metadata_path = args.output.with_suffix(".json")
    if args.output.exists() and metadata_path.exists() and not args.overwrite:
        print(f"[skip] {args.output}", flush=True)
        return
    if not args.catalog.exists():
        raise FileNotFoundError(args.catalog)

    import jax

    jax.config.update("jax_enable_x64", True)
    from clustering_statistics import spectrum2_tools

    started = time.perf_counter()
    position = np.loadtxt(args.catalog, comments="#", usecols=(0, 1, 2), dtype="f8")
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(f"invalid catalog shape: {position.shape}")
    # The ASCII writer keeps only a limited number of significant digits, so
    # positions infinitesimally below L can be printed as exactly BOX_SIZE.
    # They are periodically identical to zero and must be wrapped before mesh
    # painting.  Reject only genuinely out-of-domain values.
    tolerance = 1.0e-6 * float(args.boxsize)
    if np.any(position < -tolerance) or np.any(position > args.boxsize + tolerance):
        raise ValueError("EZmock positions are genuinely outside the periodic box")
    n_wrapped = int(np.count_nonzero((position < 0.0) | (position >= args.boxsize)))
    position %= float(args.boxsize)
    data = {"POSITION": position, "INDWEIGHT": np.ones(position.shape[0], dtype="f8")}

    def get_data() -> dict[str, dict[str, np.ndarray]]:
        return {"data": data}

    edges = make_edges(args.kmin, args.kmax, args.dk)
    spectrum = spectrum2_tools.compute_box_mesh2_spectrum(
        get_data,
        mattrs={
            "boxsize": float(args.boxsize),
            "boxcenter": float(args.boxsize) / 2.0,
            "meshsize": int(args.meshsize),
        },
        edges=edges,
        ells=(0,),
        los="z",
    )
    pole = spectrum.get(0)
    arrays = {
        "k": np.asarray(pole.coords("k"), dtype="f8"),
        "k_edges": np.asarray(pole.edges("k"), dtype="f8"),
        "nmodes": np.asarray(pole.values("nmodes"), dtype="f8"),
        "pk0": np.asarray(pole.value(), dtype="f8"),
        "norm": np.asarray(pole.values("norm"), dtype="f8"),
        "num_shotnoise": np.asarray(pole.values("num_shotnoise"), dtype="f8"),
        "shotnoise": np.asarray(pole.values("shotnoise"), dtype="f8"),
    }
    arrays["num_raw_reconstructed"] = arrays["pk0"] * arrays["norm"] + arrays["num_shotnoise"]
    valid = arrays["nmodes"] > 0
    elapsed = time.perf_counter() - started
    atomic_savez(
        args.output,
        **arrays,
        seed=np.asarray(args.seed, dtype="i8"),
        ndata=np.asarray(position.shape[0], dtype="i8"),
        nbar=np.asarray(position.shape[0] / args.boxsize**3, dtype="f8"),
        boxsize=np.asarray(args.boxsize, dtype="f8"),
        meshsize=np.asarray(args.meshsize, dtype="i8"),
    )
    write_json(
        metadata_path,
        {
            "task": "task43_measure_ezmock_trial_pk_jaxpower",
            "status": "done",
            "catalog": str(args.catalog),
            "output": str(args.output),
            "seed": int(args.seed),
            "ndata": int(position.shape[0]),
            "periodic_coordinate_entries_wrapped": n_wrapped,
            "boxsize": float(args.boxsize),
            "meshsize": int(args.meshsize),
            "kmin": float(args.kmin),
            "kmax": float(args.kmax),
            "dk": float(args.dk),
            "n_valid_bins": int(np.count_nonzero(valid)),
            "estimator": "clustering_statistics.spectrum2_tools.compute_box_mesh2_spectrum",
            "engine": "jaxpower CPU",
            "paint": {"resampler": "tsc", "interlacing": 3, "compensate": True},
            "threads_requested": int(args.threads),
            "cpu_affinity": cpus,
            "jax_backend": jax.default_backend(),
            "elapsed_sec": float(elapsed),
        },
    )
    del position
    print(
        f"[done] seed={args.seed} P0 bins={arrays['pk0'].size} "
        f"valid={np.count_nonzero(valid)} elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
