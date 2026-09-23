#!/usr/bin/env python3
"""Create the immutable 1000-row FIX_AMPLITUDE=F covariance manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from task43_ezmock_covariance_common import (
    ATTACH_PARTICLE, BAO_ENHANCE, BOX_SIZE, COMMON_RANDOM, COMMON_RANDOM_META,
    COMMON_RANDOM_MULTIPLIER, DEFAULT_NTRACER, DK, FIX_AMPLITUDE, INVERT_PHASE,
    JOINT_DIMENSION, KMAX, KMIN, MANIFEST, MANIFEST_AUDIT, NGRID, NREAL,
    N_PK_BINS, P0, PDF_BASE, PHASES,
    PK_INTERP_LOG, PK_MESH_PAD, PK_MESHSIZE, RAND_GENERATOR, REDSHIFT, RHO_C,
    RHO_EXP, SEEDS, SIGMA_V, S_EDGES, TARGET_NDATA, ZMAX, ZMIN, ensure_dirs,
    row_paths, write_json, write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntracer", type=int, default=DEFAULT_NTRACER)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_rows(ntracer: int) -> list[dict[str, object]]:
    rows = []
    for index, (phase, seed) in enumerate(zip(PHASES, SEEDS, strict=True)):
        rows.append({
            "task": "task43_ezmock_lightcone_covariance_production",
            "experiment": "x1000_fixampF_common50_s50_350",
            "production_index": index,
            "phase": phase,
            "phase_index": 1000 + index,
            "seed": int(seed),
            "sim_name": f"EZmock_covariance_seed{seed}",
            "boxsize": BOX_SIZE,
            "ngrid": NGRID,
            "ntracer": int(ntracer),
            "redshift_snapshot": REDSHIFT,
            "space_mode": "real",
            "redshift_evolution": False,
            "rsd": False,
            "fix_amplitude": FIX_AMPLITUDE,
            "attach_particle": ATTACH_PARTICLE,
            "rho_c": RHO_C,
            "rho_exp": RHO_EXP,
            "pdf_base": PDF_BASE,
            "sigma_v": SIGMA_V,
            "rand_generator": RAND_GENERATOR,
            "pk_interp_log": PK_INTERP_LOG,
            "invert_phase": INVERT_PHASE,
            "bao_enhance": BAO_ENHANCE,
            "zmin": ZMIN,
            "zmax": ZMAX,
            "observer": [0.0, 0.0, 0.0],
            "geometry": "positive_octant_radial_shell_from_box_corner",
            "selection_mode": "single_snapshot_geometric_shell",
            "target_abacus_mean_count": TARGET_NDATA,
            "random_multiplier": COMMON_RANDOM_MULTIPLIER,
            "random_policy": "one_immutable_common_single_snapshot_ezmock_geometric_shell_selection",
            "fkp_p0": P0,
            "s_edges": S_EDGES,
            "pk_meshsize": PK_MESHSIZE,
            "pk_mesh_pad": PK_MESH_PAD,
            "pk_kmin": KMIN,
            "pk_kmax": KMAX,
            "pk_dk": DK,
            **row_paths(phase, seed),
        })
    return rows


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if MANIFEST.exists() and not args.overwrite:
        raise FileExistsError(f"manifest exists; use --overwrite after auditing abundance: {MANIFEST}")
    rows = build_rows(int(args.ntracer))
    assert len(rows) == NREAL and len({row["seed"] for row in rows}) == NREAL
    assert not ({int(row["seed"]) for row in rows} & set(range(431001, 431011)))
    assert all(row["fix_amplitude"] is False and row["attach_particle"] is True for row in rows)
    assert all(np.array_equal(np.asarray(row["s_edges"]), S_EDGES) for row in rows)
    assert all(row["random_catalog_path"] == str(COMMON_RANDOM) for row in rows)
    write_jsonl(MANIFEST, rows)
    write_json(MANIFEST_AUDIT, {
        "task": "task43_make_ezmock_covariance_manifest",
        "status": "done",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "production covariance; not fixed-amplitude validation",
        "manifest": str(MANIFEST),
        "nreal": NREAL,
        "seed_min": min(SEEDS),
        "seed_max": max(SEEDS),
        "seed_unique": True,
        "validation_seed_overlap": [],
        "fix_amplitude": False,
        "attach_particle": True,
        "ntracer": int(args.ntracer),
        "common_random": str(COMMON_RANDOM),
        "common_random_metadata": str(COMMON_RANDOM_META),
        "common_random_multiplier": COMMON_RANDOM_MULTIPLIER,
        "s_edges": S_EDGES,
        "n_xi_bins": S_EDGES.size - 1,
        "n_pk_bins": N_PK_BINS,
        "joint_dimension": JOINT_DIMENSION,
    })
    print(f"[write] {MANIFEST} rows={len(rows)} ntracer={args.ntracer} FIX_AMPLITUDE=F")


if __name__ == "__main__":
    main()
