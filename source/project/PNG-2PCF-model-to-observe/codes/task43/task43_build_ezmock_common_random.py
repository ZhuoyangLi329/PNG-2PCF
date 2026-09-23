#!/usr/bin/env python3
"""Build common50 for the measured single-snapshot EZmock geometric shell."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from task43_ezmock_covariance_common import (
    ABACUS_FKP_SUMMARY, COMMON_RANDOM, COMMON_RANDOM_META, COMMON_RANDOM_SEED,
    COMMON_RANDOM_SIZE, P0, ZMAX, ZMIN, atomic_savez, ensure_dirs, sha256,
    write_json,
)
from task43_fkp_zeff import fkp_bin_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=COMMON_RANDOM)
    parser.add_argument("--seed", type=int, default=COMMON_RANDOM_SEED)
    parser.add_argument("--size", type=int, default=COMMON_RANDOM_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate(path: Path, metadata_path: Path, size: int) -> bool:
    if not path.is_file() or not metadata_path.is_file():
        return False
    with np.load(path, allow_pickle=False) as data:
        return (
            data["Z"].shape == (size,)
            and all(data[key].shape == (size,) for key in ("X", "Y", "Zcart", "WEIGHT"))
            and np.all(np.isfinite(data["Z"]))
        )


def main() -> None:
    args = parse_args(); ensure_dirs(); output = args.output; meta_path = output.with_suffix(".json")
    if not args.overwrite and validate(output, meta_path, int(args.size)):
        print(f"[skip verified] {output}")
        return
    if not ABACUS_FKP_SUMMARY.is_file():
        raise FileNotFoundError(ABACUS_FKP_SUMMARY)
    started = time.perf_counter()
    with np.load(ABACUS_FKP_SUMMARY, allow_pickle=False) as data:
        z_edges = np.asarray(data["z_edges"], dtype="f8")
        nbar = np.asarray(data["nbar"], dtype="f8")
        volume = np.asarray(data["volume_shell"], dtype="f8")
    if not np.isclose(z_edges[0], ZMIN, rtol=0.0, atol=1.0e-12) or not np.isclose(
        z_edges[-1], ZMAX, rtol=0.0, atol=1.0e-12
    ) or nbar.shape != volume.shape:
        raise ValueError("unexpected frozen Abacus FKP selection grid")
    from cosmoprimo.fiducial import AbacusSummit
    cosmo = AbacusSummit(0); chi_edges = np.asarray(cosmo.comoving_radial_distance(z_edges), dtype="f8")
    # The measured EZmock is a single homogeneous snapshot with no imposed
    # n(z) evolution.  Its random must therefore have constant comoving nbar:
    # shell probabilities are proportional to volume, not Abacus nbar*volume.
    # The frozen Abacus nbar remains the FKP-weight table only.
    probability = volume / volume.sum()
    rng = np.random.default_rng(int(args.seed)); size = int(args.size)
    ibin = rng.choice(probability.size, size=size, p=probability)
    u = rng.random(size); radius = np.cbrt(chi_edges[ibin] ** 3 + u * (chi_edges[ibin + 1] ** 3 - chi_edges[ibin] ** 3))
    phi = rng.uniform(0.0, 0.5 * np.pi, size=size)
    # Isotropy requires mu=cos(theta)=sin(DEC) to be uniform, not sin(theta).
    costheta = rng.uniform(0.0, 1.0, size=size)
    sintheta = np.sqrt(1.0 - costheta**2)
    x = (radius * sintheta * np.cos(phi)).astype("f4"); y = (radius * sintheta * np.sin(phi)).astype("f4"); zcart = (radius * costheta).astype("f4")
    zgrid = np.linspace(ZMIN, ZMAX, 200_001, dtype="f8"); chigrid = np.asarray(cosmo.comoving_radial_distance(zgrid), dtype="f8")
    redshift = np.interp(radius, chigrid, zgrid).astype("f4")
    # Preserve the strict open shell after float32 storage; otherwise a handful
    # of values arbitrarily round to exactly z=0.6 or z=0.8.
    redshift = np.clip(
        redshift,
        np.nextafter(np.float32(ZMIN), np.float32(ZMAX)),
        np.nextafter(np.float32(ZMAX), np.float32(ZMIN)),
    )
    base = np.ones(size, dtype="f4")
    atomic_savez(output, Z=redshift, X=x, Y=y, Zcart=zcart, WEIGHT=base, random_seed=np.asarray(int(args.seed), dtype="i8"), selection=np.asarray("ezmock_single_snapshot_constant_nbar_times_volume"))
    file_hash = sha256(output); fkp_weights = fkp_bin_weights(nbar, P0); idx = np.clip(np.searchsorted(z_edges, redshift, side="right") - 1, 0, nbar.size - 1); weights = fkp_weights[idx]
    hist = np.histogram(redshift, bins=z_edges)[0]
    metadata = {
        "task": "task43_build_ezmock_common_random", "status": "done",
        "classification": "one immutable common random for all 1000 single-snapshot EZmock covariance mocks",
        "path": str(output), "sha256": file_hash, "seed": int(args.seed), "nrandom": size,
        "selection_policy": "constant comoving nbar matching the measured single-snapshot EZmock; bin probability proportional to octant shell volume and uniform in chi^3",
        "abacus_nbar_role": "FKP weights only; not injected as tracer redshift evolution",
        "angular_policy": "uniform solid angle in positive octant",
        "zmin": ZMIN, "zmax": ZMAX, "z_range": [float(redshift.min()), float(redshift.max())],
        "coordinate_min": [float(x.min()), float(y.min()), float(zcart.min())], "coordinate_max": [float(x.max()), float(y.max()), float(zcart.max())],
        "positive_octant_gate": bool(np.all(x >= 0) and np.all(y >= 0) and np.all(zcart >= 0)),
        "fkp_summary": str(ABACUS_FKP_SUMMARY), "p0": P0, "z_edges": z_edges,
        "target_probability": probability, "realized_counts": hist,
        "weight_min": float(weights.min()), "weight_max": float(weights.max()), "weight_sum": float(weights.sum()), "weight2_sum": float(np.dot(weights, weights)),
        "runtime_sec": float(time.perf_counter() - started),
    }
    if not metadata["positive_octant_gate"] or hist.sum() != size or not np.all(hist > 0):
        raise RuntimeError("common random audit failed")
    write_json(meta_path, metadata)
    print(f"[write] {output} n={size} sha256={file_hash} elapsed={metadata['runtime_sec']:.1f}s")


if __name__ == "__main__":
    main()
