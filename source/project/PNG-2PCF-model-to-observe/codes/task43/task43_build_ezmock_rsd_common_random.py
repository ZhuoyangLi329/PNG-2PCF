#!/usr/bin/env python3
"""Build the immutable common-50 random catalog for EZmock RSD covariance."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from task43_ezmock_rsd_covariance_common import (
    ABACUS_FKP_DIR, COMMON_RANDOM, COMMON_RANDOM_META, P0_FKP,
    RANDOM_MULTIPLIER, TARGET_NDATA, ZMAX, ZMIN, atomic_savez, sha256, write_json,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndata", type=int, default=TARGET_NDATA)
    parser.add_argument("--multiplier", type=int, default=RANDOM_MULTIPLIER)
    parser.add_argument("--seed", type=int, default=602000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def load_target_nz() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted(ABACUS_FKP_DIR.glob("task43_rsd_fkp_AbacusSummit_base_c000_ph*_mmin1p4e13_zobs0p4_0p8_dz0p01.npz"))
    if len(paths) < 1:
        raise FileNotFoundError(f"no Abacus RSD FKP tables under {ABACUS_FKP_DIR}")
    tables = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            tables.append({
                "z_edges": np.asarray(data["z_edges"], dtype="f8"),
                "volume": np.asarray(data["volume_shell"], dtype="f8"),
                "nbar": np.asarray(data["nbar"], dtype="f8"),
            })
    z_edges = tables[0]["z_edges"]
    volume = tables[0]["volume"]
    nbar = np.mean(np.asarray([item["nbar"] for item in tables]), axis=0)
    if not np.all(np.isfinite(nbar)) or np.any(nbar <= 0.0):
        raise RuntimeError("Abacus mean FKP nbar is not finite and positive")
    for item in tables[1:]:
        if not np.array_equal(item["z_edges"], z_edges) or not np.array_equal(item["volume"], volume):
            raise RuntimeError("Abacus FKP z/volume grids differ between phases")
    fkp = 1.0 / (1.0 + nbar * P0_FKP)
    return z_edges, volume, nbar, fkp

def main() -> None:
    args = parse_args()
    if int(args.ndata) < 100_000 or int(args.multiplier) < 1:
        raise ValueError("ndata/multiplier are too small")
    if COMMON_RANDOM.exists() and not args.overwrite:
        raise FileExistsError(f"common random exists: {COMMON_RANDOM}")
    z_edges, volume, nbar, fkp = load_target_nz()
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    from cosmoprimo.fiducial import AbacusSummit
    cosmo = AbacusSummit(0)
    chi_edges = np.asarray(cosmo.comoving_radial_distance(z_edges), dtype="f8")
    probabilities = nbar * volume
    probabilities = probabilities / probabilities.sum()
    nrandom = int(args.ndata) * int(args.multiplier)
    rng = np.random.default_rng(int(args.seed))
    t0 = time.perf_counter()
    # Store only the final float32 arrays; generate intermediates in chunks.
    redshift = np.empty(nrandom, dtype="f4")
    x = np.empty(nrandom, dtype="f4")
    y = np.empty(nrandom, dtype="f4")
    zcart = np.empty(nrandom, dtype="f4")
    weight_fkp = np.empty(nrandom, dtype="f4")
    random_index = np.repeat(np.arange(int(args.multiplier), dtype="i2"), int(args.ndata))
    chunk = 1_000_000
    for start in range(0, nrandom, chunk):
        stop = min(start + chunk, nrandom)
        count = stop - start
        bins = rng.choice(probabilities.size, size=count, p=probabilities)
        u = rng.random(count)
        r3 = chi_edges[bins] ** 3 + u * (chi_edges[bins + 1] ** 3 - chi_edges[bins] ** 3)
        radius = np.cbrt(r3)
        phi = rng.uniform(0.0, 0.5 * np.pi, size=count)
        mu = rng.uniform(0.0, 1.0, size=count)
        sintheta = np.sqrt(1.0 - mu * mu)
        redshift[start:stop] = np.interp(radius, chi_edges, z_edges).astype("f4")
        x[start:stop] = (radius * sintheta * np.cos(phi)).astype("f4")
        y[start:stop] = (radius * sintheta * np.sin(phi)).astype("f4")
        zcart[start:stop] = (radius * mu).astype("f4")
        weight_fkp[start:stop] = fkp[bins].astype("f4")
    finite_gate = bool(
        np.all(np.isfinite(redshift))
        and np.all(np.isfinite(x))
        and np.all(np.isfinite(y))
        and np.all(np.isfinite(zcart))
        and np.all(np.isfinite(weight_fkp))
        and np.all(weight_fkp > 0.0)
        and np.all(x >= 0.0) and np.all(y >= 0.0) and np.all(zcart >= 0.0)
        and np.all((redshift.astype("f8") > ZMIN) & (redshift.astype("f8") < ZMAX))
    )
    if not finite_gate:
        raise RuntimeError("common random geometry/finiteness gate failed")
    atomic_savez(
        COMMON_RANDOM,
        RA=np.degrees(np.arctan2(y.astype("f8"), x.astype("f8"))).astype("f4"),
        DEC=np.degrees(np.arcsin(np.clip(zcart.astype("f8") / np.maximum(np.sqrt(x*x+y*y+zcart*zcart), 1e-12), -1.0, 1.0))).astype("f4"),
        Z=redshift, X=x, Y=y, Zcart=zcart,
        WEIGHT=np.ones(nrandom, dtype="f4"),
        WEIGHT_FKP=weight_fkp, WEIGHT_TOTAL=weight_fkp,
        RANDOM_INDEX=random_index, seed=np.asarray(int(args.seed), dtype="i8"),
    )
    meta = {
        "task": "task43_build_ezmock_rsd_common_random",
        "status": "done",
        "classification": "immutable_common50_target_nz",
        "ndata_reference": int(args.ndata),
        "random_multiplier": int(args.multiplier),
        "nrandom": nrandom,
        "seed": int(args.seed),
        "zmin": ZMIN, "zmax": ZMAX, "p0_fkp": P0_FKP,
        "positive_octant_gate": True, "finite_gate": finite_gate,
        "nbar_range": [float(nbar.min()), float(nbar.max())],
        "fkp_range": [float(fkp.min()), float(fkp.max())],
        "fkp_sources": [str(path) for path in sorted(ABACUS_FKP_DIR.glob("task43_rsd_fkp_AbacusSummit_base_c000_ph*_mmin1p4e13_zobs0p4_0p8_dz0p01.npz"))],
        "output": str(COMMON_RANDOM),
        "sha256": sha256(COMMON_RANDOM),
        "runtime_sec": float(time.perf_counter() - t0),
    }
    write_json(COMMON_RANDOM_META, meta)
    print(f"[done] common random n={nrandom} sha256={meta['sha256']} elapsed={meta['runtime_sec']:.1f}s")

if __name__ == "__main__":
    main()
