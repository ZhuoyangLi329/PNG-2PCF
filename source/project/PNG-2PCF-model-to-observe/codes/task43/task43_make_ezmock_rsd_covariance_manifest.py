#!/usr/bin/env python3
"""Create the immutable Task43 EZmock RSD x1000 manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from task43_ezmock_rsd_covariance_common import (
    ATTACH_PARTICLE, BAO_ENHANCE, BOX_SIZE, DEFAULT_NTRACER, DK, FIX_AMPLITUDE,
    INVERT_PHASE, KMAX, KMIN, MANIFEST, MANIFEST_AUDIT, NGRID, NREAL, P0_FKP,
    PDF_BASE, PK_MESH_PAD, PK_MESHSIZE, PK_INTERP_LOG, RAND_GENERATOR,
    RANDOM_MULTIPLIER, REDSHIFT, RHO_C, RHO_EXP, S_EDGES, SEED_BASE, SIGMA_V,
    ZMAX, ZMIN, ensure_dirs, row_paths, write_json, write_jsonl,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nreal", type=int, default=NREAL)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--ntracer", type=int, default=DEFAULT_NTRACER)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=__import__("pathlib").Path, default=MANIFEST)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if not 1 <= int(args.nreal) <= 1000:
        raise ValueError("nreal must be in [1,1000]")
    ensure_dirs()
    manifest = args.output
    audit_path = manifest.with_suffix(".json")
    if manifest.exists() and not args.overwrite:
        raise FileExistsError(f"manifest exists: {manifest}")
    rows = []
    seeds = [int(args.seed_base) + i + 1 for i in range(int(args.nreal))]
    for index, seed in enumerate(seeds):
        rows.append({
            "task": "task43_ezmock_rsd_lightcone_covariance_production",
            "experiment": "ezmock_rsd_lightcone_zobs0p4_0p8_x1000_fixampF_common50",
            "production_index": index,
            "phase": f"mock{index:04d}",
            "phase_index": index,
            "seed": seed,
            "sim_name": f"EZmock_RSD_covariance_mock{index:04d}",
            "boxsize": BOX_SIZE,
            "ngrid": NGRID,
            "ntracer": int(args.ntracer),
            "redshift_snapshot": REDSHIFT,
            "space_mode": "redshift",
            "redshift_evolution": False,
            "rsd": True,
            "rsd_los": "local_radial",
            "rsd_velocity_scale": 1.0,
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
            "zmin_observed": ZMIN,
            "zmax_observed": ZMAX,
            "observer": [0.0, 0.0, 0.0],
            "geometry": "positive_octant_radial_shell_from_box_corner",
            "selection_mode": "post_rsd_observed_redshift",
            "random_multiplier": RANDOM_MULTIPLIER,
            "random_policy": "immutable_common50_target_nz",
            "p0_fkp": P0_FKP,
            "s_edges": S_EDGES,
            "pk_meshsize": PK_MESHSIZE,
            "pk_mesh_pad": PK_MESH_PAD,
            "pk_kmin": KMIN,
            "pk_kmax": KMAX,
            "pk_dk": DK,
            **row_paths(index, seed),
        })
    write_jsonl(manifest, rows)
    write_json(audit_path, {
        "task": "task43_make_ezmock_rsd_lightcone_manifest",
        "status": "done",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "RSD lightcone covariance production; FIX_AMPLITUDE=F",
        "manifest": str(manifest),
        "nreal": int(args.nreal),
        "seed_base": int(args.seed_base),
        "seed_min": min(seeds),
        "seed_max": max(seeds),
        "seed_unique": len(set(seeds)) == len(seeds),
        "fix_amplitude": FIX_AMPLITUDE,
        "attach_particle": ATTACH_PARTICLE,
        "ntracer": int(args.ntracer),
        "random_multiplier": RANDOM_MULTIPLIER,
        "s_edges": S_EDGES,
        "k_grid": {"kmin": KMIN, "kmax": KMAX, "dk": DK, "meshsize": PK_MESHSIZE, "pad": PK_MESH_PAD},
    })
    print(f"[write] {manifest} rows={len(rows)} ntracer={args.ntracer} FIX_AMPLITUDE=F")

if __name__ == "__main__":
    main()
