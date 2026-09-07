#!/usr/bin/env python3
"""Build a mass-matched Abacus periodic raw-box catalog for EZmock calibration."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from task43_rawbox_ezmock_common import (
    MASS_THRESHOLD_HMSUN,
    PHASES,
    SNAPSHOT,
    ascii_path,
    atomic_savez,
    catalog_metadata_path,
    catalog_path,
    ensure_dirs,
    halo_info_paths,
    sim_name,
    write_json,
)


def slab_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def build_phase(phase: str, *, chunk_size: int, overwrite: bool) -> dict[str, Any]:
    import asdf

    ensure_dirs()
    output = catalog_path(phase)
    output_meta = catalog_metadata_path(phase)
    output_ascii = ascii_path(phase)
    if output.exists() and output_meta.exists() and output_ascii.exists() and not overwrite:
        print(f"[skip] {phase}: {output}", flush=True)
        return {"status": "skip_existing", "phase": phase, "catalog_path": str(output)}

    paths = halo_info_paths(phase)
    if len(paths) != 34:
        raise RuntimeError(f"expected 34 halo_info slabs for {phase}, found {len(paths)}")

    started = time.perf_counter()
    positions: list[np.ndarray] = []
    particle_counts: list[np.ndarray] = []
    particle_mass: float | None = None
    boxsize: float | None = None
    threshold_n: int | None = None
    n_total = 0
    n_selected = 0
    redshifts: list[float] = []
    slabs: list[dict[str, Any]] = []

    for path in paths:
        slab_started = time.perf_counter()
        with asdf.open(path, lazy_load=True, memmap=False) as af:
            header = af["header"]
            this_particle_mass = float(header["ParticleMassHMsun"])
            this_boxsize = float(header["BoxSizeHMpc"])
            if particle_mass is None:
                particle_mass = this_particle_mass
                threshold_n = int(math.ceil(MASS_THRESHOLD_HMSUN / particle_mass))
            elif not np.isclose(particle_mass, this_particle_mass, rtol=0.0, atol=1.0e-6):
                raise RuntimeError(f"ParticleMassHMsun changed in {path}")
            if boxsize is None:
                boxsize = this_boxsize
            elif not np.isclose(boxsize, this_boxsize, rtol=0.0, atol=1.0e-10):
                raise RuntimeError(f"BoxSizeHMpc changed in {path}")
            if threshold_n is None:
                raise RuntimeError("internal threshold error")

            redshifts.append(float(header.get("Redshift", np.nan)))
            data = af["data"]
            n_array = data["N"]
            x_array = data["x_L2com"]
            nobj = int(len(n_array))
            n_total += nobj
            selected_this_slab = 0
            for start in range(0, nobj, int(chunk_size)):
                stop = min(start + int(chunk_size), nobj)
                n_chunk = np.asarray(n_array[start:stop], dtype="u4")
                keep = n_chunk >= threshold_n
                if not np.any(keep):
                    continue
                x_normalized = np.asarray(x_array[start:stop], dtype="f4")[keep]
                xyz = np.mod((x_normalized.astype("f8") + 0.5) * this_boxsize, this_boxsize).astype("f4")
                positions.append(xyz)
                particle_counts.append(n_chunk[keep])
                selected_this_slab += int(xyz.shape[0])
            n_selected += selected_this_slab
            slabs.append(
                {
                    "path": str(path),
                    "slab": slab_index(path),
                    "n_total": nobj,
                    "n_selected": selected_this_slab,
                    "elapsed_sec": time.perf_counter() - slab_started,
                }
            )
            print(f"[build] {phase} slab={slab_index(path):03d} selected={selected_this_slab}", flush=True)

    if particle_mass is None or boxsize is None or threshold_n is None or not positions:
        raise RuntimeError(f"no selected halos for {phase}")
    position = np.concatenate(positions, axis=0).astype("f4", copy=False)
    n_halo = np.concatenate(particle_counts).astype("u4", copy=False)
    if position.shape[0] != n_selected or n_halo.size != n_selected:
        raise RuntimeError(f"selected count mismatch for {phase}")
    if np.any(position < 0.0) or np.any(position >= boxsize):
        raise ValueError(f"transformed coordinates outside [0, L) for {phase}")

    atomic_savez(
        output,
        POSITION=position,
        N=n_halo,
        phase=np.asarray(phase),
        sim_name=np.asarray(sim_name(phase)),
        snapshot=np.asarray(SNAPSHOT),
        boxsize=np.asarray(boxsize, dtype="f8"),
        redshift=np.asarray(float(np.nanmean(redshifts)), dtype="f8"),
        particle_mass_hmsun=np.asarray(particle_mass, dtype="f8"),
        mass_threshold_hmsun=np.asarray(MASS_THRESHOLD_HMSUN, dtype="f8"),
        threshold_n=np.asarray(threshold_n, dtype="i8"),
        position_field=np.asarray("x_L2com"),
        coordinate_transform=np.asarray("(x_L2com + 0.5) * BoxSizeHMpc mod BoxSizeHMpc"),
    )

    ascii_tmp = output_ascii.with_name(f".{output_ascii.name}.{os.getpid()}.tmp")
    with ascii_tmp.open("w", encoding="utf-8") as stream:
        stream.write(f"# {sim_name(phase)} {SNAPSHOT} Mmin={MASS_THRESHOLD_HMSUN:.8e} N={n_selected}\n")
        np.savetxt(stream, position, fmt="%.8f %.8f %.8f")
    ascii_tmp.replace(output_ascii)

    metadata = {
        "task": "task43_build_ezmock_rawbox_catalog",
        "status": "done",
        "phase": phase,
        "sim_name": sim_name(phase),
        "snapshot": SNAPSHOT,
        "catalog_path": str(output),
        "ascii_path": str(output_ascii),
        "n_slabs": len(paths),
        "n_total": n_total,
        "n_selected": n_selected,
        "nbar": n_selected / boxsize**3,
        "boxsize": boxsize,
        "redshift": float(np.nanmean(redshifts)),
        "redshift_min": float(np.nanmin(redshifts)),
        "redshift_max": float(np.nanmax(redshifts)),
        "particle_mass_hmsun": particle_mass,
        "mass_threshold_hmsun": MASS_THRESHOLD_HMSUN,
        "threshold_n": threshold_n,
        "mass_threshold_actual_hmsun": threshold_n * particle_mass,
        "mass_min_selected_hmsun": int(np.min(n_halo)) * particle_mass,
        "mass_max_selected_hmsun": int(np.max(n_halo)) * particle_mass,
        "position_field": "x_L2com",
        "coordinate_transform": "(x_L2com + 0.5) * BoxSizeHMpc mod BoxSizeHMpc",
        "coordinate_min": np.min(position, axis=0),
        "coordinate_max": np.max(position, axis=0),
        "chunk_size": int(chunk_size),
        "slabs": slabs,
        "elapsed_sec": time.perf_counter() - started,
    }
    write_json(output_meta, metadata)
    print(f"[done] {phase} selected={n_selected} threshold_N={threshold_n} path={output}", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_phase(args.phase, chunk_size=args.chunk_size, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
