#!/usr/bin/env python3
"""Measure one production EZmock P0(k) on the exact 15-bin Task43 MCMC vector."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes/task43"
DESI_CLUSTERING_ROOT = Path("/pscratch/sd/l/lzy/desi-clustering")
for _path in (CODE_DIR, DESI_CLUSTERING_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from task43_ezmock_covariance_common import (  # noqa: E402
    ABACUS_FKP_SUMMARY,
    ABACUS_PK_PAYLOAD,
    COMMON_RANDOM,
    COMMON_RANDOM_META,
    DK,
    KMAX,
    KMIN,
    MANIFEST,
    N_PK_BINS,
    P0,
    PK_MESH_PAD,
    PK_MESHSIZE,
    TMP_DIR,
    atomic_savez,
    install_jax_cpu_only_log_filter,
    read_jsonl,
    sha256,
    write_json,
)
from task43_pk_common import (  # noqa: E402
    column_edges,
    extract_spectrum_arrays,
    extract_window_arrays,
    infer_mesh_attrs_from_catalogs,
    load_fkp_summary,
    load_phase_catalog,
    make_k_edges,
    mesh_attrs_for_jaxpower,
)
from task43_rawbox_ezmock_common import set_cpu_affinity  # noqa: E402

install_jax_cpu_only_log_filter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--window", action="store_true", help="Compute the one smooth geometry window (production index 0 only).")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def target_grid() -> dict[str, np.ndarray]:
    if not ABACUS_PK_PAYLOAD.is_file():
        raise FileNotFoundError(ABACUS_PK_PAYLOAD)
    with np.load(ABACUS_PK_PAYLOAD, allow_pickle=False) as payload:
        out = {
            "k": np.asarray(payload["k_obs"], dtype="f8"),
            "k_edges": np.asarray(payload["k_edges"], dtype="f8"),
            "fine_indices": np.asarray(payload["fit_bin_indices"], dtype="i8"),
        }
    if out["k"].shape != (N_PK_BINS,) or out["k_edges"].shape != (N_PK_BINS, 2) or out["fine_indices"].shape != (N_PK_BINS,):
        raise ValueError("unexpected Task43 Abacus target P(k) grid")
    return out


def validate_existing(row: dict[str, object], require_window: bool, common_hash: str, fkp_hash: str) -> bool:
    out = Path(str(row["pk_path"])); meta_path = out.with_suffix(".json")
    if not out.is_file() or not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not (
        meta.get("status") == "done"
        and meta.get("classification") == "covariance_production_fixampF_common50"
        and meta.get("fix_amplitude") is False
        and int(meta.get("seed", -1)) == int(row["seed"])
        and meta.get("common_random_sha256") == common_hash
        and meta.get("fkp_summary_sha256") == fkp_hash
        and (not require_window or meta.get("has_window") is True)
    ):
        return False
    target = target_grid()
    with np.load(out, allow_pickle=False) as data:
        return (
            data["pk0"].shape == (N_PK_BINS,)
            and np.array_equal(data["k"], target["k"])
            and np.array_equal(data["k_edges"], target["k_edges"])
            and np.all(np.isfinite(data["pk0"]))
            and (not require_window or "window_matrix" in data.files)
        )


def promote_existing_from_fine(
    row: dict[str, object], *, require_window: bool, common_hash: str, fkp_hash: str,
) -> bool:
    """Promote a legacy 13-bin result to the 15-bin MCMC vector exactly.

    Production NPZ files retain the same 150-bin fine jaxpower spectrum used
    by the likelihood selection, so the two extra k<0.1 bins do not require a
    new mesh measurement.
    """
    output = Path(str(row["pk_path"])); meta_path = output.with_suffix(".json")
    if not output.is_file() or not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not (
        meta.get("status") == "done"
        and meta.get("classification") == "covariance_production_fixampF_common50"
        and meta.get("fix_amplitude") is False
        and int(meta.get("seed", -1)) == int(row["seed"])
        and meta.get("common_random_sha256") == common_hash
        and meta.get("fkp_summary_sha256") == fkp_hash
        and (not require_window or meta.get("has_window") is True)
    ):
        return False
    target = target_grid(); idx = target["fine_indices"]
    with np.load(output, allow_pickle=False) as data:
        if (
            data["pk0"].shape == (N_PK_BINS,)
            and np.array_equal(data["k"], target["k"])
            and np.array_equal(data["k_edges"], target["k_edges"])
        ):
            return True
        required = {"k_edges_fine", "pk0_fine", "shotnoise_fine"}
        if not required.issubset(data.files):
            return False
        arrays = {name: np.asarray(data[name]) for name in data.files}
    if (
        arrays["pk0_fine"].shape != (150,)
        or arrays["shotnoise_fine"].shape != (150,)
        or not np.array_equal(arrays["k_edges_fine"][idx], target["k_edges"])
    ):
        return False
    arrays.update({
        "k": target["k"], "k_edges": target["k_edges"],
        "pk0": arrays["pk0_fine"][idx], "shotnoise": arrays["shotnoise_fine"][idx],
        "fine_indices": idx,
    })
    if not np.all(np.isfinite(arrays["pk0"])):
        return False
    atomic_savez(output, **arrays)
    old_nbins = int(meta.get("n_pk_bins", 13))
    meta.update({
        "target_grid_source": str(ABACUS_PK_PAYLOAD),
        "target_grid_source_sha256": sha256(ABACUS_PK_PAYLOAD),
        "fine_indices": idx, "k": target["k"], "k_edges": target["k_edges"],
        "n_pk_bins": N_PK_BINS,
        "vector_promotion": {
            "from_n_pk_bins": old_nbins, "to_n_pk_bins": N_PK_BINS,
            "method": "exact selection from stored 150-bin fine jaxpower spectrum; no remeasurement",
        },
    })
    write_json(meta_path, meta)
    print(f"[promote] {row['phase']} P(k) {old_nbins}->{N_PK_BINS} bins from stored fine spectrum")
    return True


def _measure_one_locked(row: dict[str, object], *, threads: int, window: bool, overwrite: bool) -> dict[str, object]:
    if window and int(row["production_index"]) != 0:
        raise ValueError("smooth window is frozen once from production index 0")
    for path in (ABACUS_FKP_SUMMARY, ABACUS_PK_PAYLOAD, COMMON_RANDOM, COMMON_RANDOM_META):
        if not path.is_file():
            raise FileNotFoundError(path)
    common_hash = sha256(COMMON_RANDOM)
    fkp_hash = sha256(ABACUS_FKP_SUMMARY)
    common_meta = json.loads(COMMON_RANDOM_META.read_text(encoding="utf-8"))
    if common_meta.get("sha256") != common_hash:
        raise RuntimeError("common random source hash gate failed")
    output = Path(str(row["pk_path"])); meta_path = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        promote_existing_from_fine(
            row, require_window=window, common_hash=common_hash, fkp_hash=fkp_hash,
        )
    if not overwrite and validate_existing(row, window, common_hash, fkp_hash):
        print(f"[skip verified] {output}")
        return json.loads(meta_path.read_text(encoding="utf-8"))
    halo_path = Path(str(row["halo_catalog_path"])); halo_meta_path = Path(str(row["halo_metadata_path"]))
    if not halo_path.is_file() or not halo_meta_path.is_file():
        raise FileNotFoundError(halo_path)
    halo_meta = json.loads(halo_meta_path.read_text(encoding="utf-8"))
    if halo_meta.get("fix_amplitude") is not False or int(halo_meta.get("seed", -1)) != int(row["seed"]):
        raise RuntimeError("halo catalog failed FIX_AMPLITUDE=F provenance gate")

    import jax

    jax.config.update("jax_enable_x64", True)
    from clustering_statistics import spectrum2_tools

    started = time.perf_counter()
    fkp = load_fkp_summary(ABACUS_FKP_SUMMARY)
    data_cat, data_meta = load_phase_catalog(
        halo_path, fkp_summary=fkp, p0=P0, max_rows=None, seed=int(row["seed"]) + 1,
        rescale_subsample=False, add_targetid=False,
    )
    random_cat, random_meta = load_phase_catalog(
        COMMON_RANDOM, fkp_summary=fkp, p0=P0, max_rows=None, seed=int(row["seed"]) + 2,
        rescale_subsample=False, add_targetid=True,
    )
    # Random-only bounds freeze identical mesh geometry for every realization;
    # the 400 Mpc/h pad safely contains tiny data extrema differences.
    mesh_meta = infer_mesh_attrs_from_catalogs([random_cat], meshsize=PK_MESHSIZE, pad=PK_MESH_PAD)
    mattrs = mesh_attrs_for_jaxpower(mesh_meta)
    fine_requested = make_k_edges(KMIN, KMAX, DK)

    def get_data_randoms() -> dict[str, dict[str, np.ndarray]]:
        return {"data": data_cat, "randoms": random_cat}

    spectrum = spectrum2_tools.compute_mesh2_spectrum(
        get_data_randoms, mattrs=mattrs, edges=column_edges(fine_requested), ells=(0,),
        los="local", optimal_weights=None, norm={"cellsize": 10.0},
    )
    fine = extract_spectrum_arrays(spectrum, ell=0)
    target = target_grid(); idx = target["fine_indices"]
    if fine["pk0"].shape != (150,) or not np.array_equal(fine["k_edges"][idx], target["k_edges"]):
        raise RuntimeError("jaxpower fine grid cannot be mapped exactly to Task43 15-bin MCMC grid")
    pk0, shotnoise = fine["pk0"][idx], fine["shotnoise"][idx]
    if not np.all(np.isfinite(pk0)):
        raise RuntimeError("non-finite P(k) output")
    window_arrays: dict[str, np.ndarray] = {}
    if window:
        window_result = spectrum2_tools.compute_window_mesh2_spectrum(
            get_data_randoms, spectrum=spectrum, optimal_weights=None, method="smooth"
        )
        window_arrays = extract_window_arrays(window_result)
        if window_arrays["window_matrix"].shape[0] != 150:
            raise RuntimeError("smooth window observable grid mismatch")
    elapsed = time.perf_counter() - started
    atomic_savez(
        output,
        k=target["k"], k_edges=target["k_edges"], pk0=pk0, shotnoise=shotnoise,
        fine_indices=idx, k_obs_fine=fine["k_obs"], k_edges_fine=fine["k_edges"], pk0_fine=fine["pk0"],
        norm_fine=fine["norm"], num_shotnoise_fine=fine["num_shotnoise"], shotnoise_fine=fine["shotnoise"],
        phase=np.asarray(row["phase"]), seed=np.asarray(int(row["seed"]), dtype="i8"),
        production_index=np.asarray(int(row["production_index"]), dtype="i4"), fix_amplitude=np.asarray(False),
        ndata=np.asarray(int(data_meta["n_used"]), dtype="i8"), nrandom=np.asarray(int(random_meta["n_used"]), dtype="i8"),
        has_window=np.asarray(bool(window)), common_random_sha256=np.asarray(common_hash), fkp_summary_sha256=np.asarray(fkp_hash), **window_arrays,
    )
    summary = {
        "task": "task43_measure_ezmock_covariance_pk_jaxpower", "status": "done",
        "classification": "covariance_production_fixampF_common50", "production_index": int(row["production_index"]),
        "phase": row["phase"], "seed": int(row["seed"]), "fix_amplitude": False,
        "data_catalog": str(halo_path), "random_catalog": str(COMMON_RANDOM), "common_random_sha256": common_hash,
        "fkp_summary": str(ABACUS_FKP_SUMMARY), "fkp_summary_sha256": fkp_hash, "p0": P0,
        "estimator": "desi-clustering spectrum2_tools.compute_mesh2_spectrum; shot-noise-subtracted Mesh2SpectrumPole.value()",
        "measurement_effects": "common survey selection/window and FKP weighting are applied by the data-random FKP field",
        "los": "local", "ells": [0], "painting": {"resampler": "tsc", "interlacing": 3, "compensate": True},
        "mesh": mesh_meta, "mesh_contract": "frozen from immutable common random only",
        "fine_grid": {"kmin": KMIN, "kmax": KMAX, "dk": DK, "nbin": 150},
        "target_grid_source": str(ABACUS_PK_PAYLOAD), "target_grid_source_sha256": sha256(ABACUS_PK_PAYLOAD),
        "fine_indices": idx, "k": target["k"], "k_edges": target["k_edges"], "n_pk_bins": N_PK_BINS,
        "has_window": bool(window), "window_policy": "one smooth jaxpower geometry window from production index 0" if window else "reuse index-0 smooth window",
        "data": data_meta, "random": random_meta, "threads_affinity_limit": int(threads), "runtime_sec": float(elapsed),
        "outputs": {"npz": str(output), "json": str(meta_path)},
    }
    write_json(meta_path, summary)
    print(f"[done] {row['phase']} pk bins={N_PK_BINS} ndata={data_meta['n_used']} nrandom={random_meta['n_used']} window={window} elapsed={elapsed:.1f}s")
    return summary


def measure_one(row: dict[str, object], *, threads: int, window: bool, overwrite: bool) -> dict[str, object]:
    """Serialize duplicate jaxpower requests for one manifest row."""
    lock_dir = TMP_DIR / "row_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"pk_{row['phase']}_seed{int(row['seed'])}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        return _measure_one_locked(
            row,
            threads=int(threads),
            window=bool(window),
            overwrite=bool(overwrite),
        )


def main() -> None:
    args = parse_args()
    if not 1 <= int(args.threads) <= 8:
        raise ValueError("login-node jaxpower contract requires 1..8 CPU affinity")
    cpus = set_cpu_affinity(int(args.threads)); print(f"[cpu] affinity={cpus}")
    rows = read_jsonl(args.manifest)
    if not 0 <= int(args.index) < len(rows):
        raise IndexError(args.index)
    measure_one(rows[int(args.index)], threads=int(args.threads), window=bool(args.window), overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
