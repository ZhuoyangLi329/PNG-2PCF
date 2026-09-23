#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure rawbox plane-parallel P0 AND P2 with the frozen Task4.3 estimator.

Identical to task43_measure_rsd_rawbox_pk0_jaxpower.py (same catalogs, real-
space POSITION bridge, mesh-400, k grid, LOS=z, tsc/interlacing-3/compensated
paint) except ells=(0,2), so pk2 is measured alongside pk0.  The P0 block must
reproduce the audited monopole measurement bitwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task43_measure_rsd_rawbox_pk0_jaxpower import (  # noqa: E402
    legacy_paths,
    make_edges,
    output_path as pk0_output_path,
    read_jsonl,
    select_row,
    set_affinity,
)
from task43_rsd_common import PHASES, atomic_savez, atomic_write_json, sha256_file  # noqa: E402


OUTPUT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/rsd_validation")


def output_path(phase: str, meshsize: int) -> Path:
    return OUTPUT_ROOT / "rawbox" / "pk" / (
        f"task43_rsd_rawbox_p02_AbacusSummit_base_c000_{phase}_mmin1p4e13_mesh{int(meshsize)}.npz"
    )


def spectrum_arrays(spectrum: Any, ell: int) -> dict[str, np.ndarray]:
    pole = spectrum.get(int(ell))
    arrays = {
        "k": np.asarray(pole.coords("k"), dtype="f8"),
        "k_edges": np.asarray(pole.edges("k"), dtype="f8"),
        "nmodes": np.asarray(pole.values("nmodes"), dtype="f8"),
        f"pk{ell}": np.asarray(pole.value(), dtype="f8"),
        f"norm{ell}": np.asarray(pole.values("norm"), dtype="f8"),
        f"num_shotnoise{ell}": np.asarray(pole.values("num_shotnoise"), dtype="f8"),
        f"shotnoise{ell}": np.asarray(pole.values("shotnoise"), dtype="f8"),
    }
    arrays[f"num_raw_reconstructed{ell}"] = arrays[f"pk{ell}"] * arrays[f"norm{ell}"] + arrays[f"num_shotnoise{ell}"]
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--meshsize", type=int, default=400)
    parser.add_argument("--kmin", type=float, default=0.001)
    parser.add_argument("--kmax", type=float, default=0.3001)
    parser.add_argument("--dk", type=float, default=0.002)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if (int(args.meshsize), float(args.kmin), float(args.kmax), float(args.dk)) != (400, 0.001, 0.3001, 0.002):
        raise ValueError("frozen Task4.3 rawbox estimator requires mesh=400, k=0.001..0.3001, dk=0.002")
    if int(args.threads) > 6:
        raise ValueError("login-node hard limit is 6 threads")
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
        raise FileExistsError(f"partial or unvalidated output exists: {output} / {metadata_path}")

    input_path = Path(row["rawbox_catalog_path"])
    legacy_catalog_path, legacy_pk_path = legacy_paths(args.phase)
    with np.load(input_path, allow_pickle=False) as data:
        position_real = np.asarray(data["POSITION_REAL"])
        position_rsd = np.asarray(data["POSITION_RSD"], dtype="f8")
        boxsize = float(np.asarray(data["boxsize"]).item())
        redshift = float(np.asarray(data["redshift"]).item())
    position_rsd = np.mod(position_rsd, boxsize)
    with np.load(legacy_catalog_path, allow_pickle=False) as data:
        if not np.array_equal(position_real, np.asarray(data["POSITION"])):
            raise RuntimeError(f"real-space POSITION bridge failed for {args.phase}")

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
        ells=(0, 2),
        los="z",
    )
    rsd0 = spectrum_arrays(spectrum, 0)
    rsd2 = spectrum_arrays(spectrum, 2)
    def equal_nan(a, b):
        a, b = np.asarray(a, dtype="f8"), np.asarray(b, dtype="f8")
        return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))

    if not equal_nan(rsd0["k"], rsd2["k"]) or not equal_nan(rsd0["k_edges"], rsd2["k_edges"]):
        raise RuntimeError("P0/P2 k grids differ")
    if not equal_nan(rsd0["nmodes"], rsd2["nmodes"]):
        raise RuntimeError("P0/P2 mode counts differ")
    elapsed = time.perf_counter() - started

    # P0 bridge against the audited monopole measurement (bitwise).
    audited = pk0_output_path(args.phase, int(args.meshsize))
    bridge: dict[str, Any] = {"path": str(audited)}
    with np.load(audited, allow_pickle=False) as data:
        aud_pk0 = np.asarray(data["pk0_rsd"], dtype="f8")
        aud_k = np.asarray(data["k"], dtype="f8")
        aud_nmodes = np.asarray(data["nmodes"], dtype="f8")
    if aud_pk0.shape != rsd0["pk0"].shape:
        raise RuntimeError(f"audited P0 shape {aud_pk0.shape} differs from {rsd0['pk0'].shape}")
    def eq_nan(a, b):
        a, b = np.asarray(a, dtype="f8"), np.asarray(b, dtype="f8")
        return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))

    if not eq_nan(aud_k, rsd0["k"]) or not eq_nan(aud_nmodes, rsd0["nmodes"]):
        raise RuntimeError("audited P0 k/nmodes grids differ")
    maxdiff = float(np.nanmax(np.abs(aud_pk0 - rsd0["pk0"])))
    scale = float(np.nanmax(np.abs(aud_pk0)))
    rel_l2 = float(np.linalg.norm(np.nan_to_num(aud_pk0 - rsd0["pk0"])) / max(np.linalg.norm(np.nan_to_num(aud_pk0)), 1.0e-300))
    bridge.update({"array_equal": bool(eq_nan(aud_pk0, rsd0["pk0"])), "max_abs_diff": maxdiff, "scale_max": scale, "rel_l2": rel_l2})
    # ells=(0,2) changes jaxpower's per-bin reduction order vs the audited
    # ells=(0,) run; x64 reassociation noise ~1e-11, far below any statistical
    # scale.  Gate on relative l2 instead of bitwise equality.
    if rel_l2 > 1.0e-10:
        raise RuntimeError(f"P0 bridge failed: rel_l2={rel_l2:.3e} max|diff|={maxdiff:.3e}")
    valid = rsd0["nmodes"] > 0.0
    if np.any(~np.isfinite(rsd0["pk0"][valid])) or np.any(~np.isfinite(rsd2["pk2"][valid])):
        raise RuntimeError("non-finite P0/P2 in non-empty bins")

    atomic_savez(
        output,
        k=rsd0["k"],
        k_edges=rsd0["k_edges"],
        nmodes=rsd0["nmodes"],
        pk0=rsd0["pk0"],
        pk2=rsd2["pk2"],
        norm0=rsd0["norm0"],
        norm2=rsd2["norm2"],
        num_shotnoise0=rsd0["num_shotnoise0"],
        num_shotnoise2=rsd2["num_shotnoise2"],
        shotnoise0=rsd0["shotnoise0"],
        shotnoise2=rsd2["shotnoise2"],
        num_raw_reconstructed0=rsd0["num_raw_reconstructed0"],
        num_raw_reconstructed2=rsd2["num_raw_reconstructed2"],
        phase=np.asarray(args.phase),
        ndata=np.asarray(position_real.shape[0], dtype="i8"),
        nbar=np.asarray(position_real.shape[0] / boxsize**3, dtype="f8"),
        boxsize=np.asarray(boxsize, dtype="f8"),
        redshift=np.asarray(redshift, dtype="f8"),
        meshsize=np.asarray(args.meshsize, dtype="i8"),
        ells=np.asarray([0, 2], dtype="i4"),
    )
    metadata = {
        "task": "task43_measure_rsd_rawbox_p02_jaxpower",
        "status": "pass",
        "phase": args.phase,
        "observable": "P0(k) and P2(k)",
        "ells": [0, 2],
        "space_newly_measured": "plane-parallel RSD, LOS=z",
        "estimator": "clustering_statistics.spectrum2_tools.compute_box_mesh2_spectrum",
        "engine": "jaxpower CPU",
        "paint": {"resampler": "tsc", "interlacing": 3, "compensate": True},
        "meshsize": int(args.meshsize),
        "k_grid": {"kmin": float(args.kmin), "kmax": float(args.kmax), "dk": float(args.dk)},
        "n_bins": int(rsd0["pk0"].size),
        "n_valid_bins": int(np.count_nonzero(valid)),
        "ndata": int(position_real.shape[0]),
        "nbar_h3_mpc3": float(position_real.shape[0] / boxsize**3),
        "p0_bridge_vs_audited": bridge,
        "real_space_policy": "legacy POSITION bitwise equality (inherited gate)",
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
        "output_path": str(output),
        "output_sha256": sha256_file(output),
    }
    atomic_write_json(metadata_path, metadata)
    print(f"[done] {args.phase} P02 RSD bins={rsd0['pk0'].size} elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
