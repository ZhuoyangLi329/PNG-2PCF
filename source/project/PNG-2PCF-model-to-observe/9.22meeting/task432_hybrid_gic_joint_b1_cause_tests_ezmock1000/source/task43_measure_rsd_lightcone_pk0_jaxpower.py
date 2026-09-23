#!/usr/bin/env python3
"""Measure Task 4.3.2 radial-LOS lightcone P0(k) with the Task 4.3 estimator.

The estimator, mesh/window construction, k grid, and shot-noise convention are
the Task 4.3 real-space lightcone choices.  The only changed input is the
phase-matched RSD data/random catalog and its observed-redshift FKP table.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np

from task43_build_rsd_lightcone_random import fkp_path
from task43_pk_common import (
    column_edges,
    extract_spectrum_arrays,
    extract_window_arrays,
    infer_mesh_attrs_from_catalogs,
    make_k_edges,
    mesh_attrs_for_jaxpower,
)
from task43_rsd_common import PHASES, atomic_savez, atomic_write_json, sha256_file


DEFAULT_MANIFEST = Path(
    "/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/"
    "rsd_validation/manifests/task43_rsd_validation_x25.jsonl"
)
OUTPUT_DIR = Path(
    "/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/"
    "rsd_validation/lightcone/pk"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_row(rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    matches = [row for row in rows if row["phase"] == phase]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {phase}, found {len(matches)}")
    return matches[0]


def fkp_weights(redshift: np.ndarray, summary: dict[str, np.ndarray]) -> np.ndarray:
    edges = np.asarray(summary["z_edges"], dtype="f8")
    values = np.asarray(summary["fkp_weights"], dtype="f8")
    index = np.clip(np.searchsorted(edges, np.asarray(redshift, dtype="f8"), side="right") - 1, 0, values.size - 1)
    return values[index]


def choose_rows(nrows: int, maximum: int | None, seed: int) -> slice | np.ndarray:
    if maximum is None or int(maximum) <= 0 or int(maximum) >= int(nrows):
        return slice(None)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(nrows), size=int(maximum), replace=False))


def load_catalog(
    path: Path,
    *,
    summary: dict[str, np.ndarray],
    role: str,
    maximum: int | None,
    seed: int,
    rescale_subsample: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        n_total = int(np.asarray(payload["Z"]).size)
        choice = choose_rows(n_total, maximum, seed)
        redshift = np.asarray(payload["Z"][choice], dtype="f8")
        position = np.column_stack(
            [
                np.asarray(payload["X"][choice], dtype="f8"),
                np.asarray(payload["Y"][choice], dtype="f8"),
                np.asarray(payload["Zcart"][choice], dtype="f8"),
            ]
        )
        base_weight = np.asarray(payload["WEIGHT"][choice], dtype="f8")
        stored_total = (
            np.asarray(payload["WEIGHT_TOTAL"][choice], dtype="f8")
            if "WEIGHT_TOTAL" in payload.files
            else None
        )
        if role == "random" and "RANDOM_INDEX" in payload.files:
            targetid = np.asarray(payload["RANDOM_INDEX"][choice], dtype="i8") * int(n_total)
            if isinstance(choice, slice):
                targetid += np.arange(n_total, dtype="i8")
            else:
                targetid += np.asarray(choice, dtype="i8")
        else:
            targetid = None
    weight = base_weight * fkp_weights(redshift, summary)
    stored_weight_gate = True
    stored_weight_max_abs = 0.0
    if stored_total is not None:
        stored_weight_max_abs = float(np.max(np.abs(weight - stored_total)))
        stored_weight_gate = bool(np.allclose(weight, stored_total, rtol=1.0e-7, atol=1.0e-8))
        if not stored_weight_gate:
            raise RuntimeError(f"{path}: stored WEIGHT_TOTAL does not match the frozen phase FKP table")
    n_used = int(weight.size)
    scale = float(n_total) / float(n_used) if (bool(rescale_subsample) and n_used != n_total) else 1.0
    weight *= scale
    catalog = {"POSITION": position, "INDWEIGHT": weight, "Z": redshift}
    if targetid is not None:
        catalog["TARGETID"] = targetid
    meta = {
        "path": str(path),
        "sha256": sha256_file(path),
        "role": role,
        "n_total": n_total,
        "n_used": n_used,
        "subsample_seed": int(seed),
        "subsample_applied": bool(n_used != n_total),
        "subsample_weight_rescale": bool(rescale_subsample),
        "subsample_weight_scale": float(scale),
        "weight_sum": float(np.sum(weight)),
        "weight2_sum": float(weight @ weight),
        "weight_min": float(np.min(weight)),
        "weight_max": float(np.max(weight)),
        "z_min": float(np.min(redshift)),
        "z_max": float(np.max(redshift)),
        "stored_weight_total_gate": stored_weight_gate,
        "stored_weight_total_max_abs": stored_weight_max_abs,
    }
    return catalog, meta


def output_prefix(phase: str, *, tag: str, meshsize: int, kmax: float, dk: float) -> Path:
    ktag = f"kmax{float(kmax):.3f}_dk{float(dk):.3f}".replace(".", "p")
    return OUTPUT_DIR / tag / f"task43_rsd_lightcone_pk0_{phase}_mesh{int(meshsize)}_{ktag}"


def measure(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)
    from clustering_statistics import spectrum2_tools

    prefix = output_prefix(
        row["phase"], tag=str(args.tag), meshsize=int(args.meshsize), kmax=float(args.kmax), dk=float(args.dk)
    )
    output = prefix.with_suffix(".npz")
    metadata_path = prefix.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        if output.is_file() and metadata_path.is_file() and not args.overwrite:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(output):
                print(f"[skip] validated {output}", flush=True)
                return metadata
        raise FileExistsError(f"partial/immutable output exists: {output} / {metadata_path}")

    started = time.perf_counter()
    summary_path = fkp_path(row)
    with np.load(summary_path, allow_pickle=False) as payload:
        summary = {key: np.asarray(payload[key]) for key in payload.files}
    if not np.isclose(float(np.asarray(summary["p0"]).item()), 10000.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Task 4.3.2 P0 measurement requires the frozen FKP P0=10000 table")
    phase_index = int(row["phase_index"])
    data, data_meta = load_catalog(
        Path(row["lightcone_catalog_path"]),
        summary=summary,
        role="data",
        maximum=args.max_data,
        seed=int(args.seed) + 1000 * phase_index + 1,
        rescale_subsample=bool(args.rescale_subsample),
    )
    randoms, random_meta = load_catalog(
        Path(row["lightcone_random_path"]),
        summary=summary,
        role="random",
        maximum=args.max_random,
        seed=int(args.seed) + 1000 * phase_index + 2,
        rescale_subsample=bool(args.rescale_subsample),
    )
    mesh_meta = infer_mesh_attrs_from_catalogs(
        [data, randoms], meshsize=int(args.meshsize), pad=float(args.mesh_pad)
    )
    k_edges = make_k_edges(float(args.kmin), float(args.kmax), float(args.dk))

    def get_data_randoms() -> dict[str, dict[str, np.ndarray]]:
        return {"data": data, "randoms": randoms}

    spectrum = spectrum2_tools.compute_mesh2_spectrum(
        get_data_randoms,
        mattrs=mesh_attrs_for_jaxpower(mesh_meta),
        edges=column_edges(k_edges),
        ells=(0,),
        los="local",
        optimal_weights=None,
        norm={"cellsize": float(args.norm_cellsize)},
    )
    arrays = extract_spectrum_arrays(spectrum, ell=0)
    window_arrays: dict[str, np.ndarray] = {}
    window_h5: Path | None = None
    if args.window_method is not None:
        window = spectrum2_tools.compute_window_mesh2_spectrum(
            get_data_randoms,
            spectrum=spectrum,
            optimal_weights=None,
            method=str(args.window_method),
        )
        window_arrays = extract_window_arrays(window)
        window_h5 = prefix.with_name(prefix.name + f"_window_{args.window_method}.h5")
        raw = window["raw"] if isinstance(window, dict) else window
        raw.write(window_h5)

    finite_gate = bool(
        np.all(np.isfinite(arrays["k_obs"]))
        and np.all(np.isfinite(arrays["pk0"]))
        and np.all(np.isfinite(arrays["shotnoise"]))
    )
    if not finite_gate:
        raise RuntimeError("RSD lightcone P0 finite gate failed")
    theory_ell = np.asarray(window_arrays.get("theory_ell", np.empty(0, dtype="i8")), dtype="i8")
    window_ell_gate = bool(args.window_method is None or np.array_equal(np.unique(theory_ell), [0, 2, 4]))
    if not window_ell_gate:
        raise RuntimeError(f"window theory multipoles are {np.unique(theory_ell)}, expected [0,2,4]")

    atomic_savez(
        output,
        **arrays,
        **window_arrays,
        phase=np.asarray(row["phase"]),
        phase_index=np.asarray(phase_index, dtype="i8"),
        p0=np.asarray(float(np.asarray(summary["p0"]).item()), dtype="f8"),
        zeff=np.asarray(float(np.asarray(summary["zeff"]).item()), dtype="f8"),
        volume_shell=np.asarray(summary["volume_shell"], dtype="f8"),
        meshsize=np.asarray(int(args.meshsize), dtype="i8"),
        k_edges_requested=np.asarray(k_edges, dtype="f8"),
        data_n_used=np.asarray(data_meta["n_used"], dtype="i8"),
        random_n_used=np.asarray(random_meta["n_used"], dtype="i8"),
    )
    elapsed = time.perf_counter() - started
    metadata = {
        "task": "task43_measure_rsd_lightcone_pk0_jaxpower",
        "status": "pass",
        "phase": row["phase"],
        "estimator_reference": "Task 4.3 realspace task43_measure_pk_jaxpower.py",
        "estimator": "desi-clustering spectrum2_tools.compute_mesh2_spectrum",
        "space": "redshift",
        "los": "local",
        "observed_ells": [0],
        "window_theory_ells": [int(value) for value in np.unique(theory_ell)],
        "window_method": args.window_method,
        "window_h5": None if window_h5 is None else str(window_h5),
        "mesh": mesh_meta,
        "measurement_k_grid": {"kmin": float(args.kmin), "kmax": float(args.kmax), "dk": float(args.dk)},
        "shotnoise_convention": "Mesh2SpectrumPole.value(): shot-noise-subtracted observed P0",
        "weighting": "phase observed-z WEIGHT_TOTAL=WEIGHT/(1+nbar(z)P0), P0=10000",
        "fkp_summary": str(summary_path),
        "fkp_summary_sha256": sha256_file(summary_path),
        "data": data_meta,
        "random": random_meta,
        "finite_gate": finite_gate,
        "window_ell024_gate": window_ell_gate,
        "cpu_thread_limits": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "elapsed_sec": float(elapsed),
        "output_path": str(output),
    }
    metadata["output_sha256"] = sha256_file(output)
    atomic_write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "status": "pass",
                "phase": row["phase"],
                "ndata": data_meta["n_used"],
                "nrandom": random_meta["n_used"],
                "elapsed_sec": elapsed,
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--tag", default="x25_fkpP010000")
    parser.add_argument("--meshsize", type=int, default=256)
    parser.add_argument("--mesh-pad", type=float, default=400.0)
    parser.add_argument("--kmin", type=float, default=0.001)
    parser.add_argument("--kmax", type=float, default=0.3001)
    parser.add_argument("--dk", type=float, default=0.002)
    parser.add_argument("--norm-cellsize", type=float, default=10.0)
    parser.add_argument("--window-method", choices=("smooth", "exact"), default=None)
    parser.add_argument("--max-data", type=int, default=None)
    parser.add_argument("--max-random", type=int, default=None)
    parser.add_argument("--rescale-subsample", action="store_true")
    parser.add_argument("--seed", type=int, default=430500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    row = select_row(read_jsonl(args.manifest), args.phase)
    measure(row, args)


if __name__ == "__main__":
    main()
