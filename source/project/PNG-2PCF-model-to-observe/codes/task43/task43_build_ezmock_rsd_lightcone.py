#!/usr/bin/env python3
"""Build one EZmock RSD lightcone catalog (FIX_AMPLITUDE from the manifest row)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
_INITIAL_CPUS = tuple(sorted(os.sched_getaffinity(0)))
import re
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
from cosmoprimo.fiducial import AbacusSummit

from task43_build_ezmock_rsd_common_random import load_target_nz
from task43_ezmock_rsd_covariance_common import (
    ABACUS_FKP_DIR, ATTACH_PARTICLE, BAO_ENHANCE, BOX_SIZE, EZMOCK_BINARY, INVERT_PHASE,
    LINEAR_PK, MANIFEST, NGRID, OMEGA_M, OMEGA_NU, PK_INTERP_LOG, RAND_GENERATOR,
    REDSHIFT, RHO_C, RHO_EXP, SIGMA_V, PDF_BASE, ZMAX, ZMIN, atomic_savez,
    read_jsonl, sha256, write_json,
)

NZ_THIN_TAG = 43901
FKP_GLOB = "task43_rsd_fkp_AbacusSummit_base_c000_ph*_mmin1p4e13_zobs0p4_0p8_dz0p01.npz"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-rawbox", action="store_true")
    return parser.parse_args()

def set_affinity(threads: int) -> list[int]:
    threads = int(threads)
    if not 1 <= threads <= 8:
        raise ValueError("login production builder requires 1..8 cores")
    available = list(_INITIAL_CPUS)
    if len(available) < threads:
        raise RuntimeError(f"only {len(available)} CPUs available, need {threads}")
    selected = available[:threads]
    os.sched_setaffinity(0, selected)
    return selected

def distance_table() -> tuple[np.ndarray, np.ndarray]:
    cosmo = AbacusSummit(0)
    zgrid = np.linspace(0.0, 1.1, 300001, dtype="f8")
    chigrid = np.asarray(cosmo.comoving_radial_distance(zgrid), dtype="f8")
    if not np.all(np.diff(chigrid) > 0.0):
        raise RuntimeError("non-monotonic distance table")
    return zgrid, chigrid

def box_redshift(row: dict[str, object]) -> float:
    # REDSHIFT_PK stays pinned to the linear-pk table normalization (0.725); the
    # per-row z_box only moves the snapshot epoch EZmock scales growth/velocities to.
    return float(row.get("z_box", REDSHIFT))


def make_config(row: dict[str, object], raw: Path) -> str:
    # FIX_AMPLITUDE=T stays opt-in for explicit pilot flavors; the production
    # manifest has no flavor field and keeps the F-only contract.
    fix_amplitude = bool(row["fix_amplitude"])
    flavor = row.get("flavor")
    z_box = box_redshift(row)
    if (fix_amplitude and not flavor) or bool(row["attach_particle"]) is not True:
        raise ValueError("RSD covariance builder requires FIX_AMPLITUDE=F (T needs an explicit pilot flavor) and ATTACH_PARTICLE=T")
    return f"""BOX_SIZE = {float(row['boxsize']):.17g}
NUM_GRID = {int(row['ngrid'])}
NUM_TRACER = {int(row['ntracer'])}
LINEAR_PK = '{LINEAR_PK}'
REDSHIFT_PK = {REDSHIFT:.17g}
PK_INTERP_LOG = {'T' if PK_INTERP_LOG else 'F'}
RAND_GENERATOR = {int(RAND_GENERATOR)}
RAND_SEED = {int(row['seed'])}
FIX_AMPLITUDE = {'T' if fix_amplitude else 'F'}
INVERT_PHASE = {'T' if INVERT_PHASE else 'F'}
OMEGA_M = {OMEGA_M:.17g}
OMEGA_NU = {OMEGA_NU:.17g}
DE_EOS_W = -1
REDSHIFT = {z_box:.17g}
BAO_ENHANCE = {float(BAO_ENHANCE):.17g}
RHO_CRITICAL = {float(row['rho_c']):.17g}
RHO_EXP = {float(row['rho_exp']):.17g}
PDF_BASE = {float(row['pdf_base']):.17g}
SIGMA_VELOCITY = {float(row['sigma_v']):.17g}
ATTACH_PARTICLE = T
OUTPUT = '{raw}'
OUTPUT_FORMAT = 0
OUTPUT_HEADER = T
OVERWRITE = 2
VERBOSE = T
"""

def build_one(row: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    cpus = set_affinity(int(args.threads))
    raw = Path(str(row["rawbox_catalog_path"]))
    config = Path(str(row["ezmock_config_path"]))
    log = Path(str(row["ezmock_log_path"]))
    lightcone = Path(str(row["lightcone_catalog_path"]))
    metadata_path = Path(str(row["lightcone_metadata_path"]))
    for path in (raw, config, log, lightcone, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    fix_amplitude = bool(row["fix_amplitude"])
    z_box = box_redshift(row)
    if lightcone.exists() and metadata_path.exists() and not args.overwrite:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (meta.get("status") == "done" and meta.get("seed") == int(row["seed"])
                and "fix_amplitude" in meta and bool(meta["fix_amplitude"]) == fix_amplitude
                and abs(float(meta.get("z_box", REDSHIFT)) - z_box) < 1.0e-12):
            print(f"[skip verified] {lightcone}")
            return meta
    if not EZMOCK_BINARY.is_file() or not LINEAR_PK.is_file():
        raise FileNotFoundError((EZMOCK_BINARY, LINEAR_PK))
    config.write_text(make_config(row, raw), encoding="utf-8")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": str(int(args.threads)), "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "close",
                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run([str(EZMOCK_BINARY), "-c", str(config)], check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
    log_text = log.read_text(encoding="utf-8", errors="replace")
    fix_flag = "T" if fix_amplitude else "F"
    if not re.search(rf"FIX_AMPLITUDE\s*=\s*{fix_flag}", log_text) or not re.search(r"ATTACH_PARTICLE\s*=\s*T", log_text):
        raise RuntimeError(f"EZmock log failed FIX_AMPLITUDE={fix_flag}/ATTACH_PARTICLE=T provenance gate")
    data = np.loadtxt(raw, comments="#", usecols=(0,1,2,3,4,5), dtype="f8")
    if data.ndim != 2 or data.shape[1] != 6 or not np.all(np.isfinite(data)):
        raise RuntimeError(f"invalid raw6 output {raw}: {data.shape}")
    position = np.mod(data[:,:3], float(row["boxsize"]))
    velocity = data[:,3:6]
    del data
    radius = np.linalg.norm(position, axis=1)
    direction = position / np.maximum(radius[:,None], 1.0e-12)
    ezfac = (1.0 + z_box) / (100.0 * np.sqrt(OMEGA_M * (1.0 + z_box)**3 + (1.0 - OMEGA_M)))
    vlos = np.einsum("ij,ij->i", velocity, direction)
    displacement = ezfac * vlos
    position_rsd = position + displacement[:,None] * direction
    radius_rsd = np.linalg.norm(position_rsd, axis=1)
    zgrid, chigrid = distance_table()
    zobs = np.interp(radius_rsd, chigrid, zgrid)
    keep = (zobs > float(row["zmin_observed"])) & (zobs < float(row["zmax_observed"])) & np.all(position_rsd >= 0.0, axis=1)
    n_pre_thin = int(np.count_nonzero(keep))
    thin_meta = {"nz_thinning": False, "n_pre_thin": n_pre_thin, "n_post_thin": n_pre_thin}
    if bool(row.get("nz_thinning", False)):
        # Dilute to the frozen Abacus-mean n(z) so data and the immutable common
        # random share the same radial selection; same source as load_target_nz().
        z_edges_nz, _, nbar_mean, _ = load_target_nz()
        acceptance = np.clip(nbar_mean / float(nbar_mean.max()), 0.0, 1.0)
        ibin = np.clip(np.searchsorted(z_edges_nz, zobs, side="right") - 1, 0, acceptance.size - 1)
        probability = acceptance[ibin]
        draws = np.random.default_rng([int(row["seed"]), NZ_THIN_TAG]).random(zobs.size)
        accept = draws < probability
        thin_meta = {
            "nz_thinning": True,
            "acceptance_mean": float(probability[keep].mean()),
            "acceptance_min": float(probability[keep].min()),
            "acceptance_max": float(probability[keep].max()),
            "n_pre_thin": n_pre_thin,
            "n_post_thin": int(np.count_nonzero(keep & accept)),
            "nbar_sources": [str(path) for path in sorted(ABACUS_FKP_DIR.glob(FKP_GLOB))],
            "nbar_mean_sha256": hashlib.sha256(np.ascontiguousarray(nbar_mean, dtype="f8").tobytes()).hexdigest(),
            "nz_thin_tag": NZ_THIN_TAG,
        }
        keep &= accept
    selected = position_rsd[keep].astype("f4")
    selected_real = position[keep].astype("f4")
    selected_z = zobs[keep].astype("f4")
    selected_vlos = vlos[keep].astype("f4")
    selected_disp = displacement[keep].astype("f4")
    if selected.shape[0] < 100000:
        raise RuntimeError(f"unexpectedly sparse RSD lightcone: {selected.shape[0]}")
    atomic_savez(
        lightcone,
        Z=selected_z, X=selected[:,0], Y=selected[:,1], Zcart=selected[:,2],
        X_REAL=selected_real[:,0], Y_REAL=selected_real[:,1], ZCART_REAL=selected_real[:,2],
        VLOS_KMS=selected_vlos, RSD_DISPLACEMENT=selected_disp,
        WEIGHT=np.ones(selected.shape[0], dtype="f4"),
        seed=np.asarray(int(row["seed"]), dtype="i8"), fix_amplitude=np.asarray(fix_amplitude),
        attach_particle=np.asarray(True), ntracer_requested=np.asarray(int(row["ntracer"]), dtype="i8"),
    )
    meta = {
        "task": "task43_build_ezmock_rsd_lightcone",
        "status": "done",
        "classification": ("RSD_lightcone_%s" % row["flavor"]) if row.get("flavor") else "covariance_production_fixampF",
        "production_index": int(row["production_index"]),
        "phase": row["phase"], "seed": int(row["seed"]),
        "flavor": row.get("flavor"),
        "fix_amplitude": fix_amplitude, "attach_particle": True,
        **thin_meta,
        "ntracer_requested": int(row["ntracer"]), "nraw": int(position.shape[0]),
        "selected_count": int(selected.shape[0]),
        "selected_fraction": float(selected.shape[0] / position.shape[0]),
        "z_box": z_box,
        "z_observed": {"min": float(selected_z.min()), "max": float(selected_z.max())},
        "positive_octant_gate": bool(np.all(selected >= 0.0)),
        "rsd_factor_mpc_h_per_kms": float(ezfac),
        "max_abs_rsd_displacement": float(np.max(np.abs(selected_disp))),
        "coordinate_max": selected.max(axis=0), "coordinate_min": selected.min(axis=0),
        "cpu_affinity": cpus, "threads": int(args.threads),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "runtime_sec": float(time.perf_counter() - started),
        "paths": {"config": str(config), "config_sha256": sha256(config), "log": str(log),
                  "rawbox": str(raw), "lightcone": str(lightcone), "manifest": str(args.manifest)}
    }
    if not meta["positive_octant_gate"]:
        raise RuntimeError("positive-octant gate failed")
    write_json(metadata_path, meta)
    if not args.keep_rawbox:
        raw.unlink(missing_ok=True)
    print(f"[done] {row['phase']} seed={row['seed']} n={selected.shape[0]} rss={meta['peak_rss_mib']:.1f}MiB")
    return meta

def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    if not 0 <= int(args.index) < len(rows):
        raise IndexError(args.index)
    build_one(rows[int(args.index)], args)

if __name__ == "__main__":
    main()
