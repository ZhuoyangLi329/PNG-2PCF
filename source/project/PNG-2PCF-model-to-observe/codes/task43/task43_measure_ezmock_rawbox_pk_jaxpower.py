#!/usr/bin/env python3
"""Measure the Task43 mass-matched periodic raw-box P0(k) with jaxpower."""

from __future__ import annotations

import argparse
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

from task43_rawbox_ezmock_common import (  # noqa: E402
    K_FUND,
    PHASES,
    atomic_savez,
    load_catalog,
    pk_metadata_path,
    pk_path,
    set_cpu_affinity,
    write_json,
)


def make_edges(kmin: float, kmax: float, dk: float) -> np.ndarray:
    edges = np.arange(kmin, kmax + 0.5 * dk, dk, dtype="f8")
    if edges[-1] < kmax:
        edges = np.append(edges, kmax)
    edges[0] = kmin
    edges[-1] = kmax
    return np.column_stack([edges[:-1], edges[1:]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--meshsize", type=int, default=400)
    parser.add_argument("--kmin", type=float, default=0.001)
    parser.add_argument("--kmax", type=float, default=0.3001)
    parser.add_argument("--dk", type=float, default=0.002)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cpus = set_cpu_affinity(args.threads)
    output = pk_path(args.phase, args.meshsize)
    output_meta = pk_metadata_path(args.phase, args.meshsize)
    if output.exists() and output_meta.exists() and not args.overwrite:
        print(f"[skip] {output}", flush=True)
        return

    import jax

    jax.config.update("jax_enable_x64", True)
    from clustering_statistics import spectrum2_tools

    started = time.perf_counter()
    position, catalog_meta = load_catalog(args.phase)
    data = {
        "POSITION": position,
        "INDWEIGHT": np.ones(position.shape[0], dtype="f8"),
    }

    def get_data() -> dict[str, dict[str, np.ndarray]]:
        return {"data": data}

    edges = make_edges(args.kmin, args.kmax, args.dk)
    spectrum = spectrum2_tools.compute_box_mesh2_spectrum(
        get_data,
        mattrs={"boxsize": float(catalog_meta["boxsize"]), "boxcenter": float(catalog_meta["boxsize"]) / 2.0, "meshsize": int(args.meshsize)},
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
    # This lsstypes release does not expose a ``num_raw`` data column.  Its
    # public value convention is (num_raw - num_shotnoise) / norm, so retain
    # the exactly reconstructed numerator for provenance.
    arrays["num_raw_reconstructed"] = arrays["pk0"] * arrays["norm"] + arrays["num_shotnoise"]
    elapsed = time.perf_counter() - started
    valid = arrays["nmodes"] > 0
    metadata = {
        "task": "task43_measure_ezmock_rawbox_pk_jaxpower",
        "status": "done",
        "phase": args.phase,
        "catalog_path": str(catalog_meta["catalog_path"]),
        "output_path": str(output),
        "estimator": "clustering_statistics.spectrum2_tools.compute_box_mesh2_spectrum",
        "engine": "jaxpower CPU",
        "space": "real",
        "los": "z (irrelevant for ell=0 real-space periodic box)",
        "ells": [0],
        "meshsize": int(args.meshsize),
        "boxsize": float(catalog_meta["boxsize"]),
        "k_fundamental": K_FUND,
        "kmin_requested": float(args.kmin),
        "kmax_requested": float(args.kmax),
        "dk_requested": float(args.dk),
        "n_bins": int(arrays["pk0"].size),
        "n_valid_bins": int(np.count_nonzero(valid)),
        "n_empty_bins": int(np.count_nonzero(~valid)),
        "valid_k_min": float(np.min(arrays["k"][valid])),
        "valid_k_max": float(np.max(arrays["k"][valid])),
        "shotnoise_mean_valid": float(np.nanmean(arrays["shotnoise"][valid])),
        "paint": {"resampler": "tsc", "interlacing": 3, "compensate": True},
        "threads_requested": int(args.threads),
        "cpu_affinity": cpus,
        "jax_backend": jax.default_backend(),
        "elapsed_sec": elapsed,
    }
    atomic_savez(
        output,
        **arrays,
        phase=np.asarray(args.phase),
        ndata=np.asarray(position.shape[0], dtype="i8"),
        nbar=np.asarray(catalog_meta["nbar"], dtype="f8"),
        boxsize=np.asarray(catalog_meta["boxsize"], dtype="f8"),
        redshift=np.asarray(catalog_meta["redshift"], dtype="f8"),
        meshsize=np.asarray(args.meshsize, dtype="i8"),
        k_fundamental=np.asarray(K_FUND, dtype="f8"),
    )
    write_json(output_meta, metadata)
    print(f"[done] {args.phase} P0 bins={arrays['pk0'].size} valid={np.count_nonzero(valid)} elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
