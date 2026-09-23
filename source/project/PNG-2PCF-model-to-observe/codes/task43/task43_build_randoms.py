#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build matching random catalogs for Task43 halo lightcone catalogs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from task43_config import DEFAULT_MANIFEST, read_jsonl


def select_rows(rows: list[dict[str, Any]], *, index: int | None, phase: str | None) -> list[dict[str, Any]]:
    if index is not None and phase is not None:
        raise ValueError("provide only one of --index or --phase")
    if index is not None:
        return [rows[int(index)]]
    if phase is not None:
        selected = [row for row in rows if row["phase"] == phase]
        if not selected:
            raise ValueError(f"phase not found in manifest: {phase}")
        return selected
    return rows


def deterministic_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False) >> 1


def distance_grid(zmin: float, zmax: float, ngrid: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Return z and chi(z) grid in Mpc/h."""
    from cosmoprimo.fiducial import AbacusSummit

    zgrid = np.linspace(zmin, zmax, ngrid, dtype="f8")
    cosmo = AbacusSummit(0)
    chigrid = np.asarray(cosmo.comoving_radial_distance(zgrid), dtype="f8")
    return zgrid, chigrid


def sample_positive_octant_directions(rng: np.random.Generator, size: int) -> np.ndarray:
    """Sample directions uniformly on the positive octant of the unit sphere."""
    vec = np.abs(rng.normal(size=(size, 3)))
    norm = np.linalg.norm(vec, axis=1)
    return (vec / norm[:, None]).astype("f8")


def ra_dec_from_xyz(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.linalg.norm(xyz, axis=1)
    ra = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(xyz[:, 2] / radius, -1.0, 1.0)))
    return ra.astype("f8"), dec.astype("f8")


def acquire_lock(lock_path: Path, *, timeout_sec: int = 3600, poll_sec: int = 10) -> None:
    """Acquire a simple directory lock for one output file."""
    start = time.time()
    while True:
        try:
            lock_path.mkdir(parents=True)
            return
        except FileExistsError:
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"timed out waiting for lock: {lock_path}")
            print(f"[wait] lock exists {lock_path}")
            time.sleep(poll_sec)


def release_lock(lock_path: Path) -> None:
    """Release a directory lock."""
    try:
        lock_path.rmdir()
    except FileNotFoundError:
        pass


def build_one(row: dict[str, Any], *, force: bool) -> dict[str, Any]:
    halo_path = Path(row["halo_catalog_path"])
    halo_meta_path = Path(row["halo_metadata_path"])
    out = Path(row["random_catalog_path"])
    meta_out = Path(row["random_metadata_path"])
    if out.exists() and meta_out.exists() and not force:
        print(f"[skip] existing {out}")
        return {"status": "skip_existing", "path": str(out)}
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out.with_suffix(out.suffix + ".lock")
    acquire_lock(lock_path)
    try:
        if out.exists() and meta_out.exists() and not force:
            print(f"[skip] existing {out}")
            return {"status": "skip_existing", "path": str(out)}
        result = _build_one_locked(row, halo_path=halo_path, halo_meta_path=halo_meta_path, out=out, meta_out=meta_out)
    finally:
        release_lock(lock_path)
    return result


def _build_one_locked(
    row: dict[str, Any],
    *,
    halo_path: Path,
    halo_meta_path: Path,
    out: Path,
    meta_out: Path,
) -> dict[str, Any]:
    if not halo_path.exists():
        raise FileNotFoundError(f"missing halo catalog: {halo_path}")
    if not halo_meta_path.exists():
        raise FileNotFoundError(f"missing halo metadata: {halo_meta_path}")

    halo = np.load(halo_path)
    halo_meta = json.loads(halo_meta_path.read_text(encoding="utf-8"))
    if not halo_meta.get("positive_octant_gate", False):
        raise RuntimeError("halo catalog did not pass positive_octant_gate; do not use octant random")

    ndata = int(len(halo["Z"]))
    multiplier = int(row["random_multiplier"])
    nrandom = ndata * multiplier
    zmin = float(row["zmin"])
    zmax = float(row["zmax"])
    selection_tag = str(row.get("selection_tag", "topN"))
    selection_mode = str(row.get("selection_mode", "fixed_count_top_N_interp"))
    radial_policy = str(row.get("random_radial_policy", "uniform_comoving_volume"))
    seed_parts: tuple[object, ...] = (
        "task43",
        "random",
        row["phase"],
        selection_mode,
        selection_tag,
        row.get("target_count"),
        row.get("mass_threshold_hmsun"),
        multiplier,
        radial_policy,
    )
    seed = deterministic_seed(*seed_parts)
    rng = np.random.default_rng(seed)

    zgrid, chigrid = distance_grid(zmin, zmax)
    chi_min = float(chigrid[0])
    chi_max = float(chigrid[-1])
    if radial_policy == "uniform_comoving_volume":
        chi = (chi_min**3 + rng.random(nrandom) * (chi_max**3 - chi_min**3)) ** (1.0 / 3.0)
        z = np.interp(chi, chigrid, zgrid)
    elif radial_policy == "data_redshift_resample":
        data_z = np.asarray(halo["Z"], dtype="f8")
        z = rng.choice(data_z, size=nrandom, replace=True)
        chi = np.interp(z, zgrid, chigrid)
    else:
        raise ValueError(f"unknown random_radial_policy: {radial_policy}")
    direction = sample_positive_octant_directions(rng, nrandom)
    xyz = direction * chi[:, None]
    ra, dec = ra_dec_from_xyz(xyz)

    tmp_out = out.with_name(f"{out.name}.tmp.{os.getpid()}.npz")
    tmp_meta = meta_out.with_name(f"{meta_out.name}.tmp.{os.getpid()}")
    np.savez_compressed(
        tmp_out,
        RA=ra,
        DEC=dec,
        Z=z.astype("f4"),
        X=xyz[:, 0].astype("f4"),
        Y=xyz[:, 1].astype("f4"),
        Zcart=xyz[:, 2].astype("f4"),
        WEIGHT=np.ones(nrandom, dtype="f4"),
        phase=np.asarray(row["phase"]),
        seed=np.array(seed, dtype="i8"),
    )
    meta = {
        "status": "done",
        "task": "task43",
        "phase": row["phase"],
        "sim_name": row["sim_name"],
        "random_policy": f"positive_octant_uniform_angular_{radial_policy}",
        "random_radial_policy": radial_policy,
        "zmin": zmin,
        "zmax": zmax,
        "chi_min": chi_min,
        "chi_max": chi_max,
        "ndata": ndata,
        "nrandom": nrandom,
        "random_multiplier": multiplier,
        "seed": seed,
        "halo_catalog_path": str(halo_path),
        "halo_metadata_path": str(halo_meta_path),
        "output_path": str(out),
        "ra_range": [float(np.min(ra)), float(np.max(ra))],
        "dec_range": [float(np.min(dec)), float(np.max(dec))],
        "z_range": [float(np.min(z)), float(np.max(z))],
        "direction_min": np.min(direction, axis=0).tolist(),
        "direction_max": np.max(direction, axis=0).tolist(),
    }
    tmp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_out.replace(out)
    tmp_meta.replace(meta_out)
    print(f"[done] {row['phase']} nrandom={nrandom} path={out}")
    return {"status": "done", "path": str(out), "metadata_path": str(meta_out)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--phase", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    selected = select_rows(rows, index=args.index, phase=args.phase)
    print(f"[task43] random rows={len(selected)} manifest={args.manifest}")
    for row in selected:
        build_one(row, force=bool(args.force))


if __name__ == "__main__":
    main()
