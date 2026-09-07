#!/usr/bin/env python3
"""Audit and measure the two Task44 PNG-base HOD-MAP periodic LRG boxes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

from task44_pngbase_hodmap_rawbox_common import (
    BOX_SIZE,
    CATALOGS,
    FCFC_DIR,
    K_FUND,
    LOG_DIR,
    OUTPUT_ROOT,
    PK_FINE_EDGE_PAIRS,
    REDSHIFT,
    S_EDGES,
    SUMMARY_DIR,
    ascii_catalog_path,
    ascii_metadata_path,
    atomic_savez,
    atomic_write_json,
    catalog_audit,
    ensure_output_dirs,
    get_spec,
    load_positions,
    pk_metadata_path,
    pk_path,
    set_cpu_affinity,
    sha256_file,
    xi_metadata_path,
    xi_path,
)


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
DESI_CLUSTERING_ROOT = Path("/pscratch/sd/l/lzy/desi-clustering")
FCFC_BINARY = PROJECT_ROOT / "refcode" / "FCFC-main" / "FCFC_2PT_BOX"


def exact_first_bin() -> tuple[int, float]:
    nmax = int(np.ceil(0.005 / K_FUND))
    integers = np.arange(-nmax, nmax + 1, dtype="i4")
    nx, ny, nz = np.meshgrid(integers, integers, integers, indexing="ij")
    kval = K_FUND * np.sqrt((nx.astype("f8") ** 2 + ny.astype("f8") ** 2 + nz.astype("f8") ** 2).ravel())
    selected = kval[(kval >= 0.003) & (kval < 0.005)]
    return int(selected.size), float(np.mean(selected))


def run_catalog_audit(tags: list[str], *, overwrite: bool) -> None:
    ensure_output_dirs()
    rows = []
    for tag in tags:
        output = SUMMARY_DIR / f"task44_pngbase_{tag}_hodmap_catalog_audit.json"
        if output.exists() and not overwrite:
            row = json.loads(output.read_text(encoding="utf-8"))
            print(f"[skip] {output}", flush=True)
        else:
            row = catalog_audit(tag, include_sha256=True)
            if row["status"] != "pass":
                raise RuntimeError(f"catalog contract failed for {tag}: {row['contract']}")
            atomic_write_json(output, row)
            print(f"[done] {tag} catalog audit sha256={row['sha256'][:12]}", flush=True)
        rows.append(row)
    combined = {
        "task": "task44_pngbase_hodmap_rawbox_catalog_audit",
        "status": "pass" if all(row.get("status") == "pass" for row in rows) else "fail",
        "geometry": "two full periodic real-space LRG boxes; no survey window/random/FKP",
        "boxsize_mpc_h": BOX_SIZE,
        "volume_mpc_h3": BOX_SIZE**3,
        "kfund_h_mpc": K_FUND,
        "catalogs": rows,
    }
    atomic_write_json(SUMMARY_DIR / "task44_pngbase_hodmap_rawbox_catalog_audit.json", combined)


def build_ascii(tag: str, *, overwrite: bool) -> None:
    ensure_output_dirs()
    output = ascii_catalog_path(tag)
    metadata_path = ascii_metadata_path(tag)
    if output.is_file() and metadata_path.is_file() and not overwrite:
        print(f"[skip] {output}", flush=True)
        return
    started = time.perf_counter()
    position, position_meta = load_positions(tag, origin="positive")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    np.savetxt(
        temporary,
        position,
        fmt="%.10f %.10f %.10f",
        header="X Y Z [Mpc/h]; periodic coordinates shifted from [-1000,1000) to [0,2000)",
        comments="# ",
    )
    temporary.replace(output)
    elapsed = time.perf_counter() - started
    metadata = {
        "task": "task44_build_pngbase_hodmap_rawbox_ascii",
        "status": "pass",
        "tag": tag,
        "source_hdf5": get_spec(tag).path,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "position": position_meta,
        "format": "ASCII X Y Z, unit weight implicit",
        "elapsed_sec": elapsed,
    }
    atomic_write_json(metadata_path, metadata)
    print(f"[done] {tag} ASCII rows={position.shape[0]} elapsed={elapsed:.1f}s", flush=True)


def measure_pk(tag: str, *, mesh: int, origin: str, threads: int, overwrite: bool) -> None:
    ensure_output_dirs()
    output = pk_path(tag, mesh=mesh, origin=origin)
    metadata_path = pk_metadata_path(tag, mesh=mesh, origin=origin)
    measurement_edges = PK_FINE_EDGE_PAIRS
    if output.is_file() and metadata_path.is_file() and not overwrite:
        print(f"[skip] {output}", flush=True)
        return
    cpus = set_cpu_affinity(threads)
    import jax

    jax.config.update("jax_enable_x64", True)
    import sys

    if str(DESI_CLUSTERING_ROOT) not in sys.path:
        sys.path.insert(0, str(DESI_CLUSTERING_ROOT))
    from clustering_statistics import spectrum2_tools

    started = time.perf_counter()
    position, position_meta = load_positions(tag, origin=origin)
    data = {
        "POSITION": position,
        "INDWEIGHT": np.ones(position.shape[0], dtype="f8"),
    }

    def get_data() -> dict[str, dict[str, np.ndarray]]:
        return {"data": data}

    spectrum = spectrum2_tools.compute_box_mesh2_spectrum(
        get_data,
        mattrs={
            "boxsize": BOX_SIZE,
            "boxcenter": float(position_meta["boxcenter_mpc_h"]),
            "meshsize": int(mesh),
        },
        edges=measurement_edges,
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
    expected_count, expected_kmean = exact_first_bin()
    contract = {
        "edges_exact": bool(np.allclose(arrays["k_edges"], measurement_edges, rtol=0.0, atol=1.0e-14)),
        "all_bins_have_modes": bool(np.all(arrays["nmodes"] > 0)),
        "all_finite": bool(np.all(np.isfinite(arrays["pk0"]))),
        "first_bin_mode_count": bool(int(arrays["nmodes"][0]) == expected_count == 18),
        "first_bin_mode_mean": bool(np.isclose(arrays["k"][0], expected_kmean, rtol=0.0, atol=1.0e-14)),
    }
    if not all(contract.values()):
        raise RuntimeError(f"P(k) measurement contract failed for {tag}/mesh{mesh}/{origin}: {contract}")
    elapsed = time.perf_counter() - started
    atomic_savez(
        output,
        **arrays,
        tag=np.asarray(tag),
        realization=np.asarray(get_spec(tag).realization),
        fnl=np.asarray(get_spec(tag).fnl, dtype="f8"),
        ndata=np.asarray(position.shape[0], dtype="i8"),
        nbar=np.asarray(position.shape[0] / BOX_SIZE**3, dtype="f8"),
        boxsize=np.asarray(BOX_SIZE, dtype="f8"),
        volume=np.asarray(BOX_SIZE**3, dtype="f8"),
        redshift=np.asarray(REDSHIFT, dtype="f8"),
        kfund=np.asarray(K_FUND, dtype="f8"),
        mesh=np.asarray(mesh, dtype="i8"),
        origin=np.asarray(origin),
    )
    metadata = {
        "task": "task44_measure_pngbase_hodmap_rawbox_pk",
        "status": "pass",
        "tag": tag,
        "input": get_spec(tag).path,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "geometry": "full periodic cube",
        "space": "real",
        "position": position_meta,
        "estimator": "clustering_statistics.spectrum2_tools.compute_box_mesh2_spectrum",
        "engine": "jaxpower CPU",
        "ells": [0],
        "los": "z (irrelevant for real-space ell=0)",
        "mesh": int(mesh),
        "paint": {"resampler": "tsc", "interlacing": 3, "compensate": True},
        "boxsize_mpc_h": BOX_SIZE,
        "volume_mpc_h3": BOX_SIZE**3,
        "kfund_h_mpc": K_FUND,
        "first_bin": {
            "edges_h_mpc": [0.003, 0.005],
            "nmodes": int(arrays["nmodes"][0]),
            "mode_average_k_h_mpc": float(arrays["k"][0]),
            "explicit_lattice_nmodes": expected_count,
            "explicit_lattice_mode_average_k_h_mpc": expected_kmean,
        },
        "n_bins": int(arrays["pk0"].size),
        "shotnoise_mean": float(np.mean(arrays["shotnoise"])),
        "contract": contract,
        "threads": int(threads),
        "cpu_affinity": cpus,
        "jax_backend": jax.default_backend(),
        "elapsed_sec": elapsed,
    }
    atomic_write_json(metadata_path, metadata)
    print(
        f"[done] {tag} P0 mesh={mesh} origin={origin} bins={arrays['pk0'].size} elapsed={elapsed:.1f}s",
        flush=True,
    )


def measure_xi(tag: str, *, threads: int, overwrite: bool) -> None:
    ensure_output_dirs()
    output = xi_path(tag)
    metadata_path = xi_metadata_path(tag)
    if output.is_file() and metadata_path.is_file() and not overwrite:
        print(f"[skip] {output}", flush=True)
        return
    if not FCFC_BINARY.is_file() or not os.access(FCFC_BINARY, os.X_OK):
        raise FileNotFoundError(f"FCFC binary is unavailable: {FCFC_BINARY}")
    ascii_path = ascii_catalog_path(tag)
    ascii_meta_path = ascii_metadata_path(tag)
    if not ascii_path.is_file() or not ascii_meta_path.is_file():
        raise FileNotFoundError(f"build the positive-coordinate ASCII catalog first: {ascii_path}")
    ascii_meta = json.loads(ascii_meta_path.read_text(encoding="utf-8"))
    cpus = set_cpu_affinity(threads)
    conf_path = FCFC_DIR / f"task44_pngbase_{tag}_periodic_xi0_s50_350_ds10.conf"
    pair_path = FCFC_DIR / f"task44_pngbase_{tag}_periodic_DD_s50_350_ds10.bin"
    text_path = output.with_suffix(".txt")
    log_path = LOG_DIR / f"task44_pngbase_{tag}_periodic_xi0_s50_350_ds10.log"
    configuration = f"""CATALOG = '{ascii_path}'
CATALOG_LABEL = D
CATALOG_TYPE = 0
ASCII_SKIP = 1
ASCII_COMMENT = '#'
ASCII_FORMATTER = '%lf %lf %lf'
POSITION = [$1, $2, $3]
BOX_SIZE = {BOX_SIZE:.17g}
DATA_STRUCT = 0
BINNING_SCHEME = 0
PAIR_COUNT = DD
PAIR_COUNT_FILE = '{pair_path}'
CF_ESTIMATOR = DD / @@ - 1
CF_OUTPUT_FILE = '{text_path}'
SEP_BIN_MIN = {S_EDGES[0]:.17g}
SEP_BIN_MAX = {S_EDGES[-1]:.17g}
SEP_BIN_SIZE = {S_EDGES[1] - S_EDGES[0]:.17g}
OVERWRITE = {2 if overwrite else 1}
VERBOSE = T
"""
    conf_path.write_text(configuration, encoding="utf-8")
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(int(threads))
    environment["OMP_DYNAMIC"] = "FALSE"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [str(FCFC_BINARY), "-c", str(conf_path)],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    elapsed = time.perf_counter() - started
    table = np.loadtxt(text_path, comments="#", dtype="f8")
    if table.ndim == 1:
        table = table[None, :]
    expected_shape = (S_EDGES.size - 1, 4)
    if table.shape != expected_shape:
        raise ValueError(f"unexpected FCFC table shape {table.shape}, expected {expected_shape}")
    s, lower, upper, xi0 = table.T
    measured_edges = np.column_stack([lower, upper])
    contract = {
        "edges_exact": bool(np.allclose(measured_edges, np.column_stack([S_EDGES[:-1], S_EDGES[1:]]), rtol=0.0, atol=1.0e-10)),
        "all_finite": bool(np.all(np.isfinite(xi0))),
        "smax_below_half_box": bool(S_EDGES[-1] < BOX_SIZE / 2.0),
        "ascii_sha256": sha256_file(ascii_path) == ascii_meta["output_sha256"],
    }
    if not all(contract.values()):
        raise RuntimeError(f"xi measurement contract failed for {tag}: {contract}")
    atomic_savez(
        output,
        s=np.asarray(s, dtype="f8"),
        s_edges=np.asarray(S_EDGES, dtype="f8"),
        xi0=np.asarray(xi0, dtype="f8"),
        tag=np.asarray(tag),
        realization=np.asarray(get_spec(tag).realization),
        fnl=np.asarray(get_spec(tag).fnl, dtype="f8"),
        ndata=np.asarray(get_spec(tag).expected_ngal, dtype="i8"),
        nbar=np.asarray(get_spec(tag).expected_ngal / BOX_SIZE**3, dtype="f8"),
        boxsize=np.asarray(BOX_SIZE, dtype="f8"),
        volume=np.asarray(BOX_SIZE**3, dtype="f8"),
        redshift=np.asarray(REDSHIFT, dtype="f8"),
        kfund=np.asarray(K_FUND, dtype="f8"),
    )
    metadata = {
        "task": "task44_measure_pngbase_hodmap_rawbox_xi",
        "status": "pass",
        "tag": tag,
        "input_hdf5": get_spec(tag).path,
        "input_ascii": str(ascii_path),
        "input_ascii_sha256": ascii_meta["output_sha256"],
        "output": str(output),
        "output_sha256": sha256_file(output),
        "output_text": str(text_path),
        "configuration": str(conf_path),
        "pair_count": str(pair_path),
        "log": str(log_path),
        "geometry": "full periodic cube",
        "space": "real",
        "estimator": "DD / analytic_RR - 1",
        "engine": "FCFC_2PT_BOX v1.0.1 OpenMP",
        "boxsize_mpc_h": BOX_SIZE,
        "volume_mpc_h3": BOX_SIZE**3,
        "kfund_h_mpc": K_FUND,
        "s_edges_mpc_h": S_EDGES.tolist(),
        "contract": contract,
        "threads": int(threads),
        "cpu_affinity": cpus,
        "elapsed_sec": elapsed,
    }
    atomic_write_json(metadata_path, metadata)
    print(f"[done] {tag} xi0 bins={xi0.size} elapsed={elapsed:.1f}s", flush=True)


def parse_tags(values: list[str] | None) -> list[str]:
    return list(CATALOGS) if not values else [get_spec(value).tag for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "build-ascii", "pk", "xi"))
    parser.add_argument("--tag", action="append", choices=tuple(CATALOGS), help="repeat to select catalogs; default is both")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--mesh", type=int, default=400)
    parser.add_argument("--origin", choices=("positive", "centered"), default="positive")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    tags = parse_tags(args.tag)
    if args.mode == "audit":
        run_catalog_audit(tags, overwrite=bool(args.overwrite))
    elif args.mode == "build-ascii":
        for tag in tags:
            build_ascii(tag, overwrite=bool(args.overwrite))
    elif args.mode == "pk":
        for tag in tags:
            measure_pk(
                tag,
                mesh=int(args.mesh),
                origin=str(args.origin),
                threads=int(args.threads),
                overwrite=bool(args.overwrite),
            )
    elif args.mode == "xi":
        for tag in tags:
            measure_xi(tag, threads=int(args.threads), overwrite=bool(args.overwrite))
    else:  # pragma: no cover
        raise RuntimeError(args.mode)


if __name__ == "__main__":
    main()
