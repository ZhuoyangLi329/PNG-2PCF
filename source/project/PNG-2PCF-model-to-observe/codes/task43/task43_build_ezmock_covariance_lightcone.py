#!/usr/bin/env python3
"""Generate one non-fixed-amplitude EZmock and cut the Task43 lightcone."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from task43_ezmock_covariance_common import (
    ABACUS_FKP_SUMMARY, BAO_ENHANCE, BOX_SIZE, EZMOCK_BINARY, INVERT_PHASE,
    LINEAR_PK, MANIFEST, OMEGA_M_NON_NEUTRINO, OMEGA_NU, PK_INTERP_LOG,
    RAND_GENERATOR, REDSHIFT, S_EDGES, ZMAX, ZMIN, atomic_savez, read_jsonl,
    install_jax_cpu_only_log_filter, sha256, write_json,
)
from task43_rawbox_ezmock_common import set_cpu_affinity

install_jax_cpu_only_log_filter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-rawbox", action="store_true")
    return parser.parse_args()


def bool_token(value: bool) -> str:
    return "T" if bool(value) else "F"


def make_config(row: dict[str, object]) -> str:
    if bool(row["fix_amplitude"]):
        raise ValueError("covariance production requires FIX_AMPLITUDE=F")
    if not bool(row["attach_particle"]):
        raise ValueError("covariance production requires ATTACH_PARTICLE=T")
    return f"""BOX_SIZE = {float(row['boxsize']):.17g}
NUM_GRID = {int(row['ngrid'])}
NUM_TRACER = {int(row['ntracer'])}
LINEAR_PK = '{LINEAR_PK}'
REDSHIFT_PK = {REDSHIFT:.17g}
PK_INTERP_LOG = {bool_token(PK_INTERP_LOG)}
RAND_GENERATOR = {RAND_GENERATOR}
RAND_SEED = {int(row['seed'])}
FIX_AMPLITUDE = F
INVERT_PHASE = {bool_token(INVERT_PHASE)}
OMEGA_M = {OMEGA_M_NON_NEUTRINO:.17g}
OMEGA_NU = {OMEGA_NU:.17g}
DE_EOS_W = -1
REDSHIFT = {REDSHIFT:.17g}
BAO_ENHANCE = {BAO_ENHANCE:.17g}
RHO_CRITICAL = {float(row['rho_c']):.17g}
RHO_EXP = {float(row['rho_exp']):.17g}
PDF_BASE = {float(row['pdf_base']):.17g}
SIGMA_VELOCITY = {float(row['sigma_v']):.17g}
ATTACH_PARTICLE = T
OUTPUT = '{row['rawbox_catalog_path']}'
OUTPUT_FORMAT = 0
OUTPUT_HEADER = T
OVERWRITE = 2
VERBOSE = T
"""


def distance_table() -> tuple[np.ndarray, np.ndarray]:
    from cosmoprimo.fiducial import AbacusSummit
    z = np.linspace(ZMIN, ZMAX, 100_001, dtype="f8")
    chi = np.asarray(AbacusSummit(0).comoving_radial_distance(z), dtype="f8")
    if not np.all(np.diff(chi) > 0):
        raise RuntimeError("non-monotonic AbacusSummit distance table")
    return z, chi


def validate_existing(row: dict[str, object]) -> dict[str, object] | None:
    out = Path(str(row["halo_catalog_path"]))
    meta_path = Path(str(row["halo_metadata_path"]))
    if not out.is_file() or not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not (
        meta.get("status") == "done"
        and meta.get("classification") == "covariance_production_fixampF"
        and meta.get("fix_amplitude") is False
        and int(meta.get("seed", -1)) == int(row["seed"])
        and int(meta.get("ntracer_requested", -1)) == int(row["ntracer"])
    ):
        return None
    with np.load(out, allow_pickle=False) as data:
        if data["X"].shape != (int(meta["selected_count"]),):
            return None
        if bool(np.asarray(data["fix_amplitude"]).item()):
            return None
    return meta


def _build_one_locked(row: dict[str, object], *, threads: int, overwrite: bool, keep_rawbox: bool) -> dict[str, object]:
    out = Path(str(row["halo_catalog_path"])); meta_path = Path(str(row["halo_metadata_path"]))
    config = Path(str(row["ezmock_config_path"])); log = Path(str(row["ezmock_log_path"])); raw = Path(str(row["rawbox_catalog_path"]))
    for path in (out, meta_path, config, log, raw):
        path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = validate_existing(row)
        if existing is not None:
            print(f"[skip verified] {out}")
            return existing
    for path in (EZMOCK_BINARY, LINEAR_PK, ABACUS_FKP_SUMMARY):
        if not path.is_file():
            raise FileNotFoundError(path)
    config_text = make_config(row)
    config.write_text(config_text, encoding="utf-8")
    if "FIX_AMPLITUDE = F" not in config_text or "ATTACH_PARTICLE = T" not in config_text:
        raise RuntimeError("generated config failed F/T production gate")
    env = os.environ.copy(); env.update({
        "OMP_NUM_THREADS": str(int(threads)), "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "close", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run([str(EZMOCK_BINARY), "-c", str(config)], check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
    log_text = log.read_text(encoding="utf-8", errors="replace")
    if not re.search(r"FIX_AMPLITUDE\s*=\s*F", log_text) or not re.search(r"ATTACH_PARTICLE\s*=\s*T", log_text):
        raise RuntimeError("EZmock log failed FIX_AMPLITUDE=F / ATTACH_PARTICLE=T gate")
    position = np.mod(np.loadtxt(raw, comments="#", usecols=(0, 1, 2), dtype="f8"), BOX_SIZE)
    zgrid, chigrid = distance_table(); radius = np.linalg.norm(position, axis=1)
    keep = (radius > chigrid[0]) & (radius < chigrid[-1]); selected = np.asarray(position[keep], dtype="f4"); rsel = radius[keep]
    if selected.shape[0] < 100_000:
        raise RuntimeError(f"sparse production lightcone: {selected.shape[0]}")
    redshift = np.interp(rsel, chigrid, zgrid).astype("f4")
    with np.load(ABACUS_FKP_SUMMARY, allow_pickle=False) as fkp:
        z_edges = np.asarray(fkp["z_edges"], dtype="f8"); volume = np.asarray(fkp["volume_shell"], dtype="f8")
    z_counts = np.histogram(redshift, bins=z_edges)[0].astype("i8")
    atomic_savez(
        out,
        Z=redshift, X=selected[:, 0], Y=selected[:, 1], Zcart=selected[:, 2],
        WEIGHT=np.ones(selected.shape[0], dtype="f4"), phase=np.asarray(row["phase"]),
        seed=np.asarray(int(row["seed"]), dtype="i8"), production_index=np.asarray(int(row["production_index"]), dtype="i4"),
        fix_amplitude=np.asarray(False), attach_particle=np.asarray(True), snapshot_redshift=np.asarray(REDSHIFT, dtype="f8"),
    )
    frac = rsel / np.interp(redshift.astype("f8"), zgrid, chigrid) - 1.0
    meta = {
        "task": "task43_build_ezmock_covariance_lightcone", "status": "done",
        "classification": "covariance_production_fixampF", "phase": row["phase"],
        "production_index": int(row["production_index"]), "seed": int(row["seed"]),
        "fix_amplitude": False, "attach_particle": True, "ntracer_requested": int(row["ntracer"]),
        "nraw_actual": int(position.shape[0]), "selected_count": int(selected.shape[0]),
        "selected_fraction": float(selected.shape[0] / position.shape[0]),
        "nbar_volume_weighted": float(selected.shape[0] / np.sum(volume)),
        "zmin": ZMIN, "zmax": ZMAX, "z_range": [float(redshift.min()), float(redshift.max())],
        "coordinate_min": selected.min(axis=0), "coordinate_max": selected.max(axis=0),
        "positive_octant_gate": bool(np.all(selected >= 0.0)),
        "coord_check_radius_over_chi_minus_one_max_abs": float(np.max(np.abs(frac))),
        "z_edges": z_edges, "z_counts": z_counts, "nbar_z": z_counts / volume,
        "s_edges_contract": S_EDGES,
        "parameters": {key: row[key] for key in ("rho_c", "rho_exp", "pdf_base", "sigma_v", "boxsize", "ngrid", "ntracer")},
        "paths": {"manifest": str(MANIFEST), "config": str(config), "config_sha256": sha256(config), "log": str(log), "lightcone": str(out), "rawbox": str(raw)},
        "runtime_sec": float(time.perf_counter() - started),
    }
    if not meta["positive_octant_gate"] or meta["coord_check_radius_over_chi_minus_one_max_abs"] > 2e-6:
        raise RuntimeError("production lightcone geometry gate failed")
    write_json(meta_path, meta)
    if not keep_rawbox:
        raw.unlink(missing_ok=True)
        meta["paths"]["rawbox_removed_after_verified_lightcone"] = True
        write_json(meta_path, meta)
    print(f"[done] {row['phase']} seed={row['seed']} n={selected.shape[0]} nbar={meta['nbar_volume_weighted']:.8e} FIX=F")
    return meta


def build_one(row: dict[str, object], *, threads: int, overwrite: bool, keep_rawbox: bool) -> dict[str, object]:
    """Serialize duplicate builders for one manifest row.

    A resumable supervisor or an externally resumed worker may briefly request
    the same row twice.  Both EZmock invocations otherwise share one rawbox
    pathname, so the first successful builder can remove that verified
    intermediate while the second is about to read it.  Lock before checking
    the existing lightcone: a follower then revalidates and skips the completed
    atomic products instead of racing on the rawbox.
    """
    raw = Path(str(row["rawbox_catalog_path"]))
    lock_dir = raw.parent / ".row_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{row['phase']}_seed{int(row['seed'])}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        return _build_one_locked(
            row,
            threads=int(threads),
            overwrite=bool(overwrite),
            keep_rawbox=bool(keep_rawbox),
        )


def main() -> None:
    args = parse_args(); cpus = set_cpu_affinity(int(args.threads)); rows = read_jsonl(args.manifest)
    if not 0 <= args.index < len(rows):
        raise IndexError(args.index)
    print(f"[cpu] affinity={cpus}")
    build_one(rows[args.index], threads=int(args.threads), overwrite=bool(args.overwrite), keep_rawbox=bool(args.keep_rawbox))


if __name__ == "__main__":
    main()
