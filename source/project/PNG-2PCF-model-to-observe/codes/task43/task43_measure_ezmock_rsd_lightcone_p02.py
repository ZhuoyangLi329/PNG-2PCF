#!/usr/bin/env python3
"""Measure one Task43 EZmock RSD lightcone P0/P2 realization with common randoms."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

_INITIAL_CPUS = tuple(sorted(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else ()
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "8" if _name == "OMP_NUM_THREADS" else "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
for _path in (PROJECT_ROOT / "codes/task43", Path("/pscratch/sd/l/lzy/desi-clustering")):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from task43_ezmock_rsd_covariance_common import (
    ABACUS_FKP_DIR, COMMON_RANDOM, COMMON_RANDOM_META, DK, KMAX, KMIN,
    MANIFEST, N_PK_FINE, P0_FKP, PK_MESH_PAD, PK_MESHSIZE, TMP_DIR,
    atomic_savez, read_jsonl, sha256, write_json,
)
from task43_pk_common import column_edges, infer_mesh_attrs_from_catalogs, make_k_edges, mesh_attrs_for_jaxpower

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--window", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def apply_affinity(n):
    if not 1 <= int(n) <= 8:
        raise ValueError("login-node jaxpower contract requires 1..8 threads")
    if _INITIAL_CPUS:
        chosen = list(_INITIAL_CPUS[:int(n)])
        os.sched_setaffinity(0, chosen)
        return chosen
    return []

def mean_fkp():
    paths = sorted(ABACUS_FKP_DIR.glob("task43_rsd_fkp_AbacusSummit_base_c000_ph*_mmin1p4e13_zobs0p4_0p8_dz0p01.npz"))
    if not paths:
        raise FileNotFoundError("no Abacus RSD FKP tables in %s" % ABACUS_FKP_DIR)
    z_edges, nbars = None, []
    hashes = []
    for path in paths:
        with np.load(path, allow_pickle=False) as src:
            edges = np.asarray(src["z_edges"], dtype="f8")
            nbar = np.asarray(src["nbar"], dtype="f8")
        if z_edges is None:
            z_edges = edges
        elif not np.array_equal(z_edges, edges):
            raise RuntimeError("Abacus FKP z grids differ")
        nbars.append(nbar)
        hashes.append(sha256(path))
    nbar_mean = np.mean(np.asarray(nbars), axis=0)
    fkp = 1.0 / (1.0 + nbar_mean * P0_FKP)
    if not np.all(np.isfinite(fkp)) or np.any(fkp <= 0):
        raise RuntimeError("invalid mean FKP weights")
    return z_edges, fkp, [str(path) for path in paths], hashes

def load_catalog(path, z_edges, fkp, add_targetid=False):
    with np.load(path, allow_pickle=False) as src:
        redshift = np.asarray(src["Z"], dtype="f8")
        position = np.column_stack([np.asarray(src["X"], dtype="f8"),
                                    np.asarray(src["Y"], dtype="f8"),
                                    np.asarray(src["Zcart"], dtype="f8")])
        base = np.asarray(src["WEIGHT"], dtype="f8") if "WEIGHT" in src.files else np.ones(redshift.size)
        stored = np.asarray(src["WEIGHT_FKP"], dtype="f8") if "WEIGHT_FKP" in src.files else None
    ibin = np.clip(np.searchsorted(z_edges, redshift, side="right") - 1, 0, fkp.size - 1)
    weight_from_z = base * fkp[ibin]
    stored_gate = True if stored is None else bool(np.allclose(stored, fkp[ibin], rtol=2e-7, atol=2e-8))
    stored_mismatch_count = 0 if stored is None else int(np.count_nonzero(np.abs(stored - fkp[ibin]) > 2e-8))
    # The common-random builder freezes the bin weight before writing Z as float32.
    # Use the stored value for randoms; only boundary-rounded points can differ from
    # recomputation by z, and retaining the stored value guarantees immutable RR/P(k).
    weight = stored if stored is not None else weight_from_z
    catalog = {"POSITION": position, "INDWEIGHT": weight, "Z": redshift}
    if add_targetid:
        catalog["TARGETID"] = np.arange(redshift.size, dtype="i8")
    meta = {
        "path": str(path), "sha256": sha256(path), "n_used": int(redshift.size),
        "z_min": float(redshift.min()), "z_max": float(redshift.max()),
        "position_min": position.min(axis=0), "position_max": position.max(axis=0),
        "weight_min": float(weight.min()), "weight_max": float(weight.max()),
        "weight_sum": float(weight.sum()), "weight2_sum": float(np.dot(weight, weight)),
        "stored_fkp_gate": stored_gate, "stored_fkp_mismatch_count": stored_mismatch_count,
    }
    return catalog, meta

def validate_existing(row, common_hash, require_window):
    output = Path(str(row["pk_path"]))
    meta_path = output.with_suffix(".json")
    if not output.is_file() or not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not (meta.get("status") == "done" and "fix_amplitude" in meta
            and bool(meta["fix_amplitude"]) == bool(row["fix_amplitude"])
            and int(meta.get("seed", -1)) == int(row["seed"])
            and meta.get("common_random_sha256") == common_hash
            and (not require_window or meta.get("has_window") is True)):
        return False
    with np.load(output, allow_pickle=False) as src:
        return (src["pk0"].shape == (N_PK_FINE,) and src["pk2"].shape == (N_PK_FINE,)
                and np.all(np.isfinite(src["pk0"])) and np.all(np.isfinite(src["pk2"])))

def measure_locked(row, args, cpus):
    if args.window and int(row["production_index"]) != 0:
        raise ValueError("window is computed once from production index 0 only")
    for path in (COMMON_RANDOM, COMMON_RANDOM_META):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    common_hash = sha256(COMMON_RANDOM)
    random_meta = json.loads(COMMON_RANDOM_META.read_text(encoding="utf-8"))
    if random_meta.get("sha256") != common_hash:
        raise RuntimeError("common random hash gate failed")
    output = Path(str(row["pk_path"]))
    meta_path = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and validate_existing(row, common_hash, args.window):
        print("[skip verified] %s" % output)
        return
    data_path = Path(str(row["lightcone_catalog_path"]))
    data_meta_path = Path(str(row["lightcone_metadata_path"]))
    if not data_path.is_file() or not data_meta_path.is_file():
        raise FileNotFoundError(data_path)
    provenance = json.loads(data_meta_path.read_text(encoding="utf-8"))
    row_fix_amplitude = bool(row["fix_amplitude"])
    fix_flag = "T" if row_fix_amplitude else "F"
    if ("fix_amplitude" not in provenance or bool(provenance["fix_amplitude"]) != row_fix_amplitude
            or int(provenance.get("seed", -1)) != int(row["seed"])):
        raise RuntimeError(f"RSD lightcone failed FIX_AMPLITUDE={fix_flag} provenance gate")

    import jax
    jax.config.update("jax_enable_x64", True)
    from clustering_statistics import spectrum2_tools

    started = time.perf_counter()
    z_edges, fkp, fkp_sources, fkp_hashes = mean_fkp()
    data, data_meta = load_catalog(data_path, z_edges, fkp, add_targetid=False)
    randoms, random_catalog_meta = load_catalog(COMMON_RANDOM, z_edges, fkp, add_targetid=True)
    mesh_meta = infer_mesh_attrs_from_catalogs([randoms], meshsize=PK_MESHSIZE, pad=PK_MESH_PAD)
    requested_edges = make_k_edges(KMIN, KMAX, DK)
    def get_data_randoms():
        return {"data": data, "randoms": randoms}
    spectrum = spectrum2_tools.compute_mesh2_spectrum(
        get_data_randoms, mattrs=mesh_attrs_for_jaxpower(mesh_meta),
        edges=column_edges(requested_edges), ells=(0, 2), los="local",
        optimal_weights=None, norm={"cellsize": 10.0},
    )
    arrays = {}
    for ell in (0, 2):
        pole = spectrum.get(ell)
        arrays["k%d" % ell] = np.asarray(pole.coords("k"), dtype="f8")
        arrays["k_edges%d" % ell] = np.asarray(pole.edges("k"), dtype="f8")
        arrays["pk%d" % ell] = np.asarray(pole.value(), dtype="f8")
        arrays["nmodes%d" % ell] = np.asarray(pole.values("nmodes"), dtype="f8")
        arrays["norm%d" % ell] = np.asarray(pole.values("norm"), dtype="f8")
        arrays["num_shotnoise%d" % ell] = np.asarray(pole.values("num_shotnoise"), dtype="f8")
        arrays["shotnoise%d" % ell] = np.asarray(pole.values("shotnoise"), dtype="f8")
    if arrays["pk0"].shape != (N_PK_FINE,) or arrays["pk2"].shape != (N_PK_FINE,):
        raise RuntimeError("unexpected P0/P2 fine-grid shape")
    if not np.all(np.isfinite(arrays["pk0"])) or not np.all(np.isfinite(arrays["pk2"])):
        raise RuntimeError("non-finite P0/P2")
    window_arrays = {}
    if args.window:
        window = spectrum2_tools.compute_window_mesh2_spectrum(
            get_data_randoms, spectrum=spectrum, optimal_weights=None, method="smooth_mesh"
        )
        raw = window["raw"] if isinstance(window, dict) else window
        window_arrays["window_matrix"] = np.asarray(raw.value(), dtype="f8")
        window_arrays["window_observable_ells"] = np.asarray([0, 2], dtype="i8")
    atomic_savez(
        output, **arrays, **window_arrays,
        k=arrays["k0"], k_edges=arrays["k_edges0"],
        seed=np.asarray(int(row["seed"]), dtype="i8"),
        production_index=np.asarray(int(row["production_index"]), dtype="i4"),
        fix_amplitude=np.asarray(row_fix_amplitude), ndata=np.asarray(data_meta["n_used"], dtype="i8"),
        nrandom=np.asarray(random_catalog_meta["n_used"], dtype="i8"),
        common_random_sha256=np.asarray(common_hash), has_window=np.asarray(bool(args.window)),
    )
    summary = {
        "task": "task43_measure_ezmock_rsd_lightcone_p02",
        "status": "done",
        "classification": ("RSD_lightcone_%s" % row["flavor"]) if row.get("flavor") else "RSD_lightcone_covariance_fixampF_common50",
        "production_index": int(row["production_index"]), "phase": row["phase"],
        "seed": int(row["seed"]), "flavor": row.get("flavor"), "fix_amplitude": row_fix_amplitude,
        "data": data_meta, "random": random_catalog_meta,
        "common_random_sha256": common_hash,
        "fkp_sources": fkp_sources, "fkp_source_sha256": fkp_hashes, "p0_fkp": P0_FKP,
        "estimator": "jaxpower survey Mesh2Spectrum with common random FKP field",
        "los": "local", "ells": [0, 2], "mesh": mesh_meta,
        "k_grid": {"kmin": KMIN, "kmax": KMAX, "dk": DK, "nbin": N_PK_FINE},
        "has_window": bool(args.window), "window_method": "smooth_mesh" if args.window else None,
        "threads": int(args.threads), "cpu_affinity": cpus,
        "runtime_sec": float(time.perf_counter() - started),
        "output": str(output),
    }
    write_json(meta_path, summary)
    print("[done] %s P0/P2 bins=%d ndata=%d nrandom=%d elapsed=%.1fs" %
          (row["phase"], N_PK_FINE, data_meta["n_used"], random_catalog_meta["n_used"], summary["runtime_sec"]))

def main():
    args = parse_args()
    cpus = apply_affinity(int(args.threads))
    rows = read_jsonl(args.manifest)
    if not 0 <= int(args.index) < len(rows):
        raise IndexError(args.index)
    row = rows[int(args.index)]
    lock_dir = TMP_DIR / "row_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / ("pk_%s_seed%d.lock" % (row["phase"], int(row["seed"])))
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        measure_locked(row, args, cpus)

if __name__ == "__main__":
    main()
