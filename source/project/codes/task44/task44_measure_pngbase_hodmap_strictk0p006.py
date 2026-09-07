#!/usr/bin/env python3
"""Remeasure galaxy P0 with the strict physical mode cut k>=0.006 h/Mpc.

The old [0.005,0.007) shell is not relabelled.  It is remeasured as
[0.006,0.007), followed by the unchanged old shells through [0.099,0.101).
Every measured shell is checked against explicit signed Fourier-lattice modes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

from task44_pngbase_hodhost_mmin1e13_common import (
    BASE_OUTPUT_ROOT,
    BOX_SIZE,
    BOX_VOLUME,
    CATALOGS,
    K_FUND,
    PK_KMAX_CONTRACT,
    PK_STRICT_FIT_EDGES,
    PK_STRICT_KMIN,
    REDSHIFT,
    atomic_savez,
    atomic_write_json,
    ensure_output_dirs,
    get_spec,
    load_positions,
    set_cpu_affinity,
    sha256_file,
    strict_pk_metadata_path,
    strict_pk_path,
)


DESI_CLUSTERING_ROOT = Path("/pscratch/sd/l/lzy/desi-clustering")


def explicit_lattice_statistics(edges: np.ndarray = PK_STRICT_FIT_EDGES) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.asarray(edges, dtype="f8")
    nmax = int(np.ceil(float(np.max(pairs)) / K_FUND))
    integer = np.arange(-nmax, nmax + 1, dtype="i4")
    nx, ny, nz = np.meshgrid(integer, integer, integer, indexing="ij")
    modes = K_FUND * np.sqrt(
        nx.astype("f8").ravel() ** 2
        + ny.astype("f8").ravel() ** 2
        + nz.astype("f8").ravel() ** 2
    )
    counts = np.empty(pairs.shape[0], dtype="i8")
    means = np.empty(pairs.shape[0], dtype="f8")
    for index, (lower, upper) in enumerate(pairs):
        selected = modes[(modes >= lower) & (modes < upper)]
        if selected.size == 0:
            raise RuntimeError(f"empty explicit lattice shell [{lower},{upper})")
        counts[index] = selected.size
        means[index] = np.mean(selected)
    return counts, means


def _source_catalog_sha256(tag: str) -> str:
    audit = BASE_OUTPUT_ROOT / "summary" / f"task44_pngbase_{tag}_hodmap_catalog_audit.json"
    if not audit.is_file():
        raise FileNotFoundError(audit)
    payload = json.loads(audit.read_text(encoding="utf-8"))
    path = Path(get_spec(tag).path)
    stat = path.stat()
    if payload.get("status") != "pass" or payload.get("path") != str(path):
        raise RuntimeError(f"invalid source catalog audit: {audit}")
    if payload.get("size_bytes") != stat.st_size or payload.get("mtime_ns") != stat.st_mtime_ns:
        raise RuntimeError(f"source HDF5 changed since audit: {path}")
    return str(payload["sha256"])


def _validated_existing(tag: str, mesh: int) -> bool:
    output = strict_pk_path(tag, mesh=mesh)
    metadata_path = strict_pk_metadata_path(tag, mesh=mesh)
    if not output.is_file() or not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(output)


def measure_one(tag: str, *, mesh: int, threads: int, overwrite: bool) -> None:
    ensure_output_dirs()
    output = strict_pk_path(tag, mesh=mesh)
    metadata_path = strict_pk_metadata_path(tag, mesh=mesh)
    if _validated_existing(tag, mesh) and not overwrite:
        print(f"[skip] validated {output}", flush=True)
        return
    if not overwrite and (output.exists() or metadata_path.exists()):
        raise FileExistsError(f"partial/unvalidated strict-P0 output exists: {output} / {metadata_path}")
    cpus = set_cpu_affinity(threads)
    import jax

    jax.config.update("jax_enable_x64", True)
    if str(DESI_CLUSTERING_ROOT) not in sys.path:
        sys.path.insert(0, str(DESI_CLUSTERING_ROOT))
    from clustering_statistics import spectrum2_tools

    source_sha256 = _source_catalog_sha256(tag)
    position, position_metadata = load_positions(tag, origin="positive")
    data = {"POSITION": position, "INDWEIGHT": np.ones(position.shape[0], dtype="f8")}

    def get_data() -> dict[str, dict[str, np.ndarray]]:
        return {"data": data}

    started = time.perf_counter()
    spectrum = spectrum2_tools.compute_box_mesh2_spectrum(
        get_data,
        mattrs={"boxsize": BOX_SIZE, "boxcenter": BOX_SIZE / 2.0, "meshsize": int(mesh)},
        edges=PK_STRICT_FIT_EDGES,
        ells=(0,),
        los="z",
    )
    pole = spectrum.get(0)
    arrays: dict[str, np.ndarray] = {
        "k": np.asarray(pole.coords("k"), dtype="f8"),
        "k_edges": np.asarray(pole.edges("k"), dtype="f8"),
        "nmodes": np.asarray(pole.values("nmodes"), dtype="f8"),
        "pk0": np.asarray(pole.value(), dtype="f8"),
        "norm": np.asarray(pole.values("norm"), dtype="f8"),
        "num_shotnoise": np.asarray(pole.values("num_shotnoise"), dtype="f8"),
        "shotnoise": np.asarray(pole.values("shotnoise"), dtype="f8"),
    }
    arrays["num_raw_reconstructed"] = arrays["pk0"] * arrays["norm"] + arrays["num_shotnoise"]
    exact_counts, exact_kmeans = explicit_lattice_statistics()
    contract = {
        "remeasured_not_sliced_from_old_product": True,
        "strict_first_edge_is_0p006": bool(PK_STRICT_FIT_EDGES[0, 0] == PK_STRICT_KMIN),
        "first_shell_is_0p006_to_0p007": bool(np.array_equal(PK_STRICT_FIT_EDGES[0], [0.006, 0.007])),
        "last_shell_is_0p099_to_0p101": bool(np.array_equal(PK_STRICT_FIT_EDGES[-1], [0.099, 0.101])),
        "kmax_center_contract_is_0p100": bool(PK_KMAX_CONTRACT == 0.100),
        "exactly_48_shells": int(PK_STRICT_FIT_EDGES.shape[0]) == 48,
        "measured_edges_bitwise_equal": bool(np.array_equal(arrays["k_edges"], PK_STRICT_FIT_EDGES)),
        "all_bins_have_modes": bool(np.all(arrays["nmodes"] > 0)),
        "all_values_finite": bool(all(np.all(np.isfinite(value)) for value in arrays.values())),
        "all_mode_counts_match_signed_lattice": bool(
            np.array_equal(np.asarray(arrays["nmodes"], dtype="i8"), exact_counts)
        ),
        "all_mode_mean_k_match_signed_lattice": bool(
            np.allclose(arrays["k"], exact_kmeans, rtol=0.0, atol=1.0e-14)
        ),
        "first_shell_has_six_modes": int(exact_counts[0]) == int(arrays["nmodes"][0]) == 6,
    }
    if not all(contract.values()):
        raise RuntimeError(f"strict P0 measurement contract failed for {tag}: {contract}")
    elapsed = time.perf_counter() - started
    atomic_savez(
        output,
        **arrays,
        explicit_lattice_nmodes=exact_counts,
        explicit_lattice_kmean=exact_kmeans,
        tag=np.asarray(tag),
        realization=np.asarray(get_spec(tag).realization),
        fnl=np.asarray(get_spec(tag).fnl, dtype="f8"),
        ndata=np.asarray(position.shape[0], dtype="i8"),
        nbar=np.asarray(position.shape[0] / BOX_VOLUME, dtype="f8"),
        boxsize=np.asarray(BOX_SIZE, dtype="f8"),
        volume=np.asarray(BOX_VOLUME, dtype="f8"),
        redshift=np.asarray(REDSHIFT, dtype="f8"),
        kfund=np.asarray(K_FUND, dtype="f8"),
        mesh=np.asarray(mesh, dtype="i8"),
        origin=np.asarray("positive"),
    )
    metadata: dict[str, Any] = {
        "task": "task44_measure_pngbase_hodmap_strictk0p006",
        "status": "pass",
        "tag": tag,
        "input_hdf5": get_spec(tag).path,
        "input_hdf5_sha256": source_sha256,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "mode_selection": {
            "physical_rule": "retain signed Fourier lattice modes with k>=0.006 h/Mpc",
            "first_remeasured_shell_h_mpc": [0.006, 0.007],
            "subsequent_shells": "unchanged old shells [0.007,0.009), ..., [0.099,0.101)",
            "kmax_center_h_mpc": PK_KMAX_CONTRACT,
            "last_upper_edge_h_mpc": float(PK_STRICT_FIT_EDGES[-1, 1]),
            "n_shells": int(PK_STRICT_FIT_EDGES.shape[0]),
            "first_shell_signed_mode_count": int(exact_counts[0]),
            "first_shell_exact_mode_k_h_mpc": float(exact_kmeans[0]),
        },
        "geometry": "full periodic real-space cube",
        "position": position_metadata,
        "estimator": "clustering_statistics.spectrum2_tools.compute_box_mesh2_spectrum",
        "engine": "jaxpower CPU",
        "paint": {"resampler": "tsc", "interlacing": 3, "compensate": True},
        "mesh": int(mesh),
        "contract": contract,
        "threads": int(threads),
        "cpu_affinity": cpus,
        "jax_backend": jax.default_backend(),
        "elapsed_sec": elapsed,
    }
    atomic_write_json(metadata_path, metadata)
    print(f"[done] {tag} strict P0 bins={arrays['pk0'].size} elapsed={elapsed:.1f}s path={output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", action="append", choices=tuple(CATALOGS))
    parser.add_argument("--mesh", type=int, default=400)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    tags = list(CATALOGS) if not args.tag else [get_spec(tag).tag for tag in args.tag]
    for tag in tags:
        measure_one(tag, mesh=int(args.mesh), threads=int(args.threads), overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
