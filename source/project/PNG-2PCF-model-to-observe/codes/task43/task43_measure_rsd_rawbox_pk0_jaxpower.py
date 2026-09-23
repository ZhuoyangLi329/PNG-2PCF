#!/usr/bin/env python3
"""Measure paired real/RSD periodic rawbox P0(k) for Task 4.3.2.

Only the RSD leg is newly painted.  The real-space leg is reused from the
validated Task 4.3.1 rawbox measurement after a bitwise POSITION bridge.
"""

from __future__ import annotations

import argparse
import json
import logging
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


class _JaxCpuOnlyCudaProbeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            record.name == "jax._src.xla_bridge"
            and "Jax plugin configuration error" in message
            and "jax_plugins.xla_cuda12.initialize()" in message
        )


logging.getLogger("jax._src.xla_bridge").addFilter(_JaxCpuOnlyCudaProbeFilter())

CODE_DIR = Path(__file__).resolve().parent
DESI_CLUSTERING_ROOT = Path("/pscratch/sd/l/lzy/desi-clustering")
for _path in (CODE_DIR, DESI_CLUSTERING_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from task43_rsd_common import (  # noqa: E402
    OUTPUT_ROOT,
    PHASES,
    atomic_savez,
    atomic_write_json,
    sha256_file,
)


LEGACY_ROOT = (
    Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
    / "plots/outputs/task43_outputs/ezmock_rawbox_z0p725_mmin1p4e13"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_row(rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    selected = [row for row in rows if row["phase"] == phase]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one manifest row for {phase}, found {len(selected)}")
    return selected[0]


def set_affinity(threads: int) -> list[int]:
    if not 1 <= int(threads) <= 8:
        raise ValueError("--threads must be in [1, 8]")
    available = sorted(os.sched_getaffinity(0))
    selected = available[: int(threads)]
    if len(selected) != int(threads):
        raise RuntimeError(f"requested {threads} CPUs, only {len(available)} available")
    os.sched_setaffinity(0, selected)
    return selected


def make_edges(kmin: float, kmax: float, dk: float) -> np.ndarray:
    edges = np.arange(float(kmin), float(kmax) + 0.5 * float(dk), float(dk), dtype="f8")
    if edges[-1] < float(kmax):
        edges = np.append(edges, float(kmax))
    edges[0], edges[-1] = float(kmin), float(kmax)
    return np.column_stack([edges[:-1], edges[1:]])


def legacy_paths(phase: str) -> tuple[Path, Path]:
    stem = f"AbacusSummit_base_c000_{phase}_z0p725_mmin1p4e13"
    catalog = LEGACY_ROOT / "halo_catalogs" / f"halo_{stem}.npz"
    pk = LEGACY_ROOT / "pk_jaxpower" / f"pk0_{stem}_mesh400.npz"
    return catalog, pk


def output_path(phase: str, meshsize: int) -> Path:
    return OUTPUT_ROOT / "rawbox" / "pk" / (
        f"task43_rsd_rawbox_pk0_AbacusSummit_base_c000_{phase}_mmin1p4e13_mesh{int(meshsize)}.npz"
    )


def spectrum_arrays(spectrum: Any) -> dict[str, np.ndarray]:
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
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--meshsize", type=int, default=400)
    parser.add_argument("--kmin", type=float, default=0.001)
    parser.add_argument("--kmax", type=float, default=0.3001)
    parser.add_argument("--dk", type=float, default=0.002)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if (int(args.meshsize), float(args.kmin), float(args.kmax), float(args.dk)) != (400, 0.001, 0.3001, 0.002):
        raise ValueError("frozen Task4.3 rawbox estimator requires mesh=400, k=0.001..0.3001, dk=0.002")
    cpus = set_affinity(int(args.threads))
    row = select_row(read_jsonl(args.manifest), args.phase)
    output = output_path(args.phase, args.meshsize)
    metadata_path = output.with_suffix(".json")
    if output.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(output):
            print(f"[skip] validated {output}", flush=True)
            return
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"partial or unvalidated output: {output} / {metadata_path}")

    input_path = Path(row["rawbox_catalog_path"])
    legacy_catalog_path, legacy_pk_path = legacy_paths(args.phase)
    if not legacy_catalog_path.is_file() or not legacy_pk_path.is_file():
        raise FileNotFoundError(f"missing Task4.3.1 bridge: {legacy_catalog_path} / {legacy_pk_path}")
    with np.load(input_path, allow_pickle=False) as data:
        position_real = np.asarray(data["POSITION_REAL"])
        position_rsd = np.asarray(data["POSITION_RSD"], dtype="f8")
        boxsize = float(np.asarray(data["boxsize"]).item())
        redshift = float(np.asarray(data["redshift"]).item())
    outside = np.any((position_rsd < 0.0) | (position_rsd >= boxsize), axis=1)
    n_wrapped = int(np.count_nonzero(outside))
    position_rsd = np.mod(position_rsd, boxsize)
    if np.any((position_rsd < 0.0) | (position_rsd >= boxsize)):
        raise RuntimeError("periodic wrapping did not put every RSD coordinate in [0, L)")
    with np.load(legacy_catalog_path, allow_pickle=False) as data:
        legacy_real = np.asarray(data["POSITION"])
    if not np.array_equal(position_real, legacy_real):
        raise RuntimeError(f"real-space POSITION bridge is not bitwise exact for {args.phase}")
    if position_rsd.shape != position_real.shape or position_rsd.ndim != 2 or position_rsd.shape[1] != 3:
        raise ValueError(f"invalid paired POSITION arrays: {position_real.shape}, {position_rsd.shape}")
    with np.load(legacy_pk_path, allow_pickle=False) as data:
        real = {key: np.asarray(data[key]) for key in (
            "k", "k_edges", "nmodes", "pk0", "norm", "num_shotnoise", "shotnoise", "num_raw_reconstructed"
        )}
        legacy_ndata = int(np.asarray(data["ndata"]).item())
        legacy_meshsize = int(np.asarray(data["meshsize"]).item())
    if legacy_ndata != position_real.shape[0] or legacy_meshsize != int(args.meshsize):
        raise RuntimeError("legacy P0 count/mesh does not match the frozen RSD measurement")

    import jax

    jax.config.update("jax_enable_x64", True)
    from clustering_statistics import spectrum2_tools

    payload = {"POSITION": position_rsd, "INDWEIGHT": np.ones(position_rsd.shape[0], dtype="f8")}

    def get_data() -> dict[str, dict[str, np.ndarray]]:
        return {"data": payload}

    started = time.perf_counter()
    spectrum = spectrum2_tools.compute_box_mesh2_spectrum(
        get_data,
        mattrs={"boxsize": boxsize, "boxcenter": boxsize / 2.0, "meshsize": int(args.meshsize)},
        edges=make_edges(args.kmin, args.kmax, args.dk),
        ells=(0,),
        los="z",
    )
    rsd = spectrum_arrays(spectrum)
    elapsed = time.perf_counter() - started
    for key in ("k", "k_edges", "nmodes", "norm", "num_shotnoise", "shotnoise"):
        if not np.allclose(rsd[key], real[key], rtol=0.0, atol=1.0e-12, equal_nan=True):
            raise RuntimeError(f"real/RSD estimator geometry field differs: {key}")
    valid = rsd["nmodes"] > 0.0
    if np.any(~np.isfinite(rsd["pk0"][valid])):
        raise RuntimeError("non-finite RSD P0 in non-empty bins")

    atomic_savez(
        output,
        k=rsd["k"],
        k_edges=rsd["k_edges"],
        nmodes=rsd["nmodes"],
        pk0_real=np.asarray(real["pk0"], dtype="f8"),
        pk0_rsd=rsd["pk0"],
        norm=rsd["norm"],
        num_shotnoise=rsd["num_shotnoise"],
        shotnoise=rsd["shotnoise"],
        num_raw_rsd_reconstructed=rsd["num_raw_reconstructed"],
        phase=np.asarray(args.phase),
        ndata=np.asarray(position_real.shape[0], dtype="i8"),
        nbar=np.asarray(position_real.shape[0] / boxsize**3, dtype="f8"),
        boxsize=np.asarray(boxsize, dtype="f8"),
        redshift=np.asarray(redshift, dtype="f8"),
        meshsize=np.asarray(args.meshsize, dtype="i8"),
        ell=np.asarray(0, dtype="i4"),
    )
    metadata = {
        "task": "task43_measure_rsd_rawbox_pk0_jaxpower",
        "status": "pass",
        "phase": args.phase,
        "observable": "P0(k) only",
        "ells": [0],
        "space_newly_measured": "plane-parallel RSD, LOS=z",
        "real_space_policy": "reuse Task4.3.1 P0 after bitwise POSITION equality gate",
        "estimator": "clustering_statistics.spectrum2_tools.compute_box_mesh2_spectrum",
        "engine": "jaxpower CPU",
        "paint": {"resampler": "tsc", "interlacing": 3, "compensate": True},
        "meshsize": int(args.meshsize),
        "kmin_requested_h_mpc": float(args.kmin),
        "kmax_requested_h_mpc": float(args.kmax),
        "dk_requested_h_mpc": float(args.dk),
        "n_bins": int(rsd["pk0"].size),
        "n_valid_bins": int(np.count_nonzero(valid)),
        "ndata": int(position_real.shape[0]),
        "nbar_h3_mpc3": float(position_real.shape[0] / boxsize**3),
        "periodic_wrapping": {
            "definition": "POSITION_RSD mod boxsize before jaxpower painting",
            "n_objects_outside_half_open_box_before_wrap": n_wrapped,
        },
        "cpu_affinity": cpus,
        "threads_requested": int(args.threads),
        "jax_backend": jax.default_backend(),
        "elapsed_sec": elapsed,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "legacy_real_catalog_path": str(legacy_catalog_path),
        "legacy_real_catalog_sha256": sha256_file(legacy_catalog_path),
        "legacy_real_pk_path": str(legacy_pk_path),
        "legacy_real_pk_sha256": sha256_file(legacy_pk_path),
        "position_bridge": {"array_equal": True, "nrows": int(position_real.shape[0])},
        "output_path": str(output),
        "output_sha256": sha256_file(output),
    }
    atomic_write_json(metadata_path, metadata)
    print(f"[done] {args.phase} P0 RSD bins={rsd['pk0'].size} elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
