#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure RSD lightcone P0 and P2 with the Task 4.3 estimator contract.

Identical to task43_measure_rsd_lightcone_pk0_jaxpower.py (same catalogs,
FKP weights, seeds, mesh, k grid, local LOS, norm) except that the spectrum is
computed for ells=(0, 2), so the per-phase products carry pk2 alongside pk0
and, for the window phase (ph000), a (2*nk, ntheory) window matrix whose rows
are ordered [P0 bins, P2 bins].  The P0 block must reproduce the audited
monopole measurement bitwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task43_measure_rsd_lightcone_pk0_jaxpower import (  # noqa: E402
    fkp_path,
    load_catalog,
    read_jsonl,
    select_row,
)
from task43_pk_common import (  # noqa: E402
    column_edges,
    extract_spectrum_arrays,
    extract_window_arrays,
    infer_mesh_attrs_from_catalogs,
    make_k_edges,
    mesh_attrs_for_jaxpower,
)
from task43_rsd_common import PHASES, atomic_savez, atomic_write_json, sha256_file  # noqa: E402


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "outputs/task43_outputs/rsd_validation/manifests"
    / "task43_rsd_validation_lightcone_boxsafe_zobs0p4_0p8_x25.jsonl"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs/task43_outputs/rsd_validation/lightcone/pk"
AUDITED_P0_DIR = OUTPUT_DIR / "boxsafe_zobs0p4_0p8_x25_fkpP010000"


def output_prefix(phase: str, *, tag: str, meshsize: int, kmax: float, dk: float) -> Path:
    ktag = f"kmax{float(kmax):.3f}_dk{float(dk):.3f}".replace(".", "p")
    return OUTPUT_DIR / tag / f"task43_rsd_lightcone_p02_{phase}_mesh{int(meshsize)}_{ktag}"


def measure(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)
    from clustering_statistics import spectrum2_tools

    prefix = output_prefix(
        row["phase"], tag=str(args.tag), meshsize=int(args.meshsize), kmax=float(args.kmax), dk=float(args.dk)
    )
    output = prefix.with_suffix(".npz")
    metadata_path = prefix.with_suffix(".json")
    if output.is_file() and metadata_path.is_file() and not args.overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(output):
            print(f"[skip] validated {output}", flush=True)
            return metadata
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"partial output exists: {output} / {metadata_path}")

    started = time.perf_counter()
    summary_path = fkp_path(row)
    with np.load(summary_path, allow_pickle=False) as payload:
        summary = {key: np.asarray(payload[key]) for key in payload.files}
    if not np.isclose(float(np.asarray(summary["p0"]).item()), 10000.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Task 4.3.2 P02 measurement requires the frozen FKP P0=10000 table")
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
        ells=(0, 2),
        los="local",
        optimal_weights=None,
        norm={"cellsize": float(args.norm_cellsize)},
    )
    a0 = extract_spectrum_arrays(spectrum, ell=0)
    a2 = extract_spectrum_arrays(spectrum, ell=2)
    if not np.array_equal(a0["k_obs"], a2["k_obs"]) or not np.array_equal(a0["k_edges"], a2["k_edges"]):
        raise RuntimeError("P0/P2 k grids differ")
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
        expected_rows = 2 * int(a0["k_obs"].size)
        if window_arrays["window_matrix"].shape[0] != expected_rows:
            raise RuntimeError(
                f"window matrix rows {window_arrays['window_matrix'].shape[0]} != 2*nk={expected_rows}"
            )
    finite_gate = bool(
        np.all(np.isfinite(a0["pk0"]))
        and np.all(np.isfinite(a2["pk0"]))
        and np.all(np.isfinite(a0["k_obs"]))
    )
    if not finite_gate:
        raise RuntimeError("RSD lightcone P02 finite gate failed")

    # P0 bridge against the audited monopole-only measurement.
    audited = AUDITED_P0_DIR / (
        f"task43_rsd_lightcone_pk0_{row['phase']}_mesh{int(args.meshsize)}_"
        f"kmax{float(args.kmax):.3f}_dk{float(args.dk):.3f}".replace(".", "p") + ".npz"
    )
    bridge_applicable = str(row.get("sim_name", "")).startswith("AbacusSummit_base_c000_")
    bridge: dict[str, Any] = {"path": str(audited), "applicable": bridge_applicable}
    if bridge_applicable and audited.is_file():
        with np.load(audited, allow_pickle=False) as d:
            audited_pk0 = np.asarray(d["pk0"], dtype="f8")
        if audited_pk0.shape == a0["pk0"].shape:
            bridge["max_abs_diff"] = float(np.max(np.abs(audited_pk0 - a0["pk0"])))
            bridge["rel_l2"] = float(np.linalg.norm(audited_pk0 - a0["pk0"]) / max(np.linalg.norm(audited_pk0), 1e-300))
            if bridge["rel_l2"] > 1.0e-10:
                raise RuntimeError(f"P0 bridge failed: {bridge}")
        else:
            bridge["note"] = "shape mismatch; audited grid differs"
    elif bridge_applicable:
        bridge["note"] = "audited monopole measurement not found"
    else:
        bridge["note"] = "not applicable outside the original AbacusSummit_base_c000 lightcone"

    atomic_savez(
        output,
        k_obs=a0["k_obs"],
        k_edges=a0["k_edges"],
        pk0=a0["pk0"],
        pk2=a2["pk0"],
        norm_ell0=a0["norm"],
        norm_ell2=a2["norm"],
        num_shotnoise_ell0=a0["num_shotnoise"],
        num_shotnoise_ell2=a2["num_shotnoise"],
        shotnoise_ell0=a0["shotnoise"],
        shotnoise_ell2=a2["shotnoise"],
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
        "task": "task43_measure_rsd_lightcone_p02_jaxpower",
        "status": "pass",
        "phase": row["phase"],
        "estimator": "desi-clustering spectrum2_tools.compute_mesh2_spectrum with ells=(0,2)",
        "space": "redshift",
        "los": "local",
        "observed_ells": [0, 2],
        "window_method": args.window_method,
        "window_h5": None if window_h5 is None else str(window_h5),
        "window_rows": "ordered [P0 bins, P2 bins]" if args.window_method is not None else None,
        "mesh": mesh_meta,
        "measurement_k_grid": {"kmin": float(args.kmin), "kmax": float(args.kmax), "dk": float(args.dk)},
        "p0_bridge_vs_audited": bridge,
        "fkp_summary": str(summary_path),
        "fkp_summary_sha256": sha256_file(summary_path),
        "data": data_meta,
        "random": random_meta,
        "finite_gate": finite_gate,
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
                "p0_bridge_rel_l2": bridge.get("rel_l2"),
                "elapsed_sec": elapsed,
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
    parser.add_argument("--tag", default="boxsafe_zobs0p4_0p8_p02_x25_fkpP010000")
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
