#!/usr/bin/env python3
"""CPU pycorr/Corrfunc xi02 for the pngbase split-random pseudo-lightcones."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from task43_measure_rsd_lightcone_xi import load_catalogs, project_multipoles, read_jsonl, select_row
from task43_pngbase_pseudolc_common import PHASES
from task43_rsd_common import S_EDGES, atomic_savez, atomic_write_json, sha256_file


def cached_counter(
    path: Path,
    *,
    positions1: np.ndarray,
    weights1: np.ndarray,
    positions2: np.ndarray | None,
    weights2: np.ndarray | None,
    s_edges: np.ndarray,
    mu_edges: np.ndarray,
    nthreads: int,
) -> Any:
    from pycorr import TwoPointCounter
    from pycorr.twopoint_counter import BaseTwoPointCounter

    size1 = int(positions1.shape[0])
    size2 = size1 if positions2 is None else int(positions2.shape[0])
    if path.is_file():
        counter = BaseTwoPointCounter.load(str(path))
        if (
            str(counter.mode) != "smu"
            or int(counter.size1) != size1
            or int(counter.size2) != size2
            or not np.array_equal(np.asarray(counter.edges[0], dtype="f8"), s_edges)
            or not np.array_equal(np.asarray(counter.edges[1], dtype="f8"), mu_edges)
            or not np.all(np.isfinite(np.asarray(counter.wcounts, dtype="f8")))
        ):
            raise RuntimeError(f"cached pair counter contract changed: {path}")
        print(f"[pair-cache] loaded {path}", flush=True)
        return counter
    kwargs: dict[str, Any] = {
        "mode": "smu",
        "edges": (s_edges, mu_edges),
        "positions1": positions1.T,
        "weights1": weights1,
        "engine": "corrfunc",
        "los": "midpoint",
        "nthreads": int(nthreads),
    }
    if positions2 is not None:
        kwargs.update({"positions2": positions2.T, "weights2": weights2})
    counter = TwoPointCounter(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp-{os.getpid()}.npy")
    counter.save(str(temporary))
    temporary.replace(path)
    print(f"[pair-cache] saved {path}", flush=True)
    return counter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--nmu", type=int, default=40)
    parser.add_argument("--nthreads", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= int(args.nthreads) <= 8:
        raise ValueError("--nthreads must be in [1,8] for login-node execution")
    if int(args.nmu) < 20 or int(args.nmu) % 2:
        raise ValueError("--nmu must be an even integer >=20")
    row = select_row(read_jsonl(args.manifest), args.phase)
    output = Path(row["lightcone_xi_path"])
    metadata_path = output.with_suffix(".json")
    if output.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(output):
            print(f"[skip] validated {output}", flush=True)
            return
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"partial or unvalidated output: {output} / {metadata_path}")

    from pycorr import LandySzalayTwoPointEstimator, __version__ as pycorr_version
    import Corrfunc

    started = time.perf_counter()
    arrays = load_catalogs(row)
    data_xyz = np.asarray(arrays["data_xyz"], dtype="f8")
    data_weight = np.asarray(arrays["data_weight"], dtype="f8")
    random_xyz = np.asarray(arrays["random_xyz"], dtype="f8")
    random_weight = np.asarray(arrays["random_weight"], dtype="f8")
    random_index = np.asarray(arrays["random_index"], dtype="i2")
    indices = sorted(int(value) for value in np.unique(random_index))
    if indices != list(range(int(row["random_multiplier"]))):
        raise RuntimeError(f"unexpected split-random indices: {indices}")
    sizes = [int(np.count_nonzero(random_index == index)) for index in indices]
    if sizes != [data_xyz.shape[0]] * len(indices):
        raise RuntimeError(f"split random blocks do not equal ndata={data_xyz.shape[0]}: {sizes}")

    s_edges = np.asarray(S_EDGES, dtype="f8")
    mu_edges = np.linspace(-1.0, 1.0, int(args.nmu) + 1, dtype="f8")
    s = 0.5 * (s_edges[:-1] + s_edges[1:])
    mu = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    cache_dir = output.parent / "pair_cache" / output.stem
    dd = cached_counter(
        cache_dir / "DD.npy",
        positions1=data_xyz,
        weights1=data_weight,
        positions2=None,
        weights2=None,
        s_edges=s_edges,
        mu_edges=mu_edges,
        nthreads=int(args.nthreads),
    )
    dd_value = np.asarray(dd.wcounts, dtype="f8")
    xi_by_random: list[np.ndarray] = []
    dr_by_random: list[np.ndarray] = []
    rr_by_random: list[np.ndarray] = []
    block_elapsed: list[float] = []
    for index in indices:
        block_started = time.perf_counter()
        mask = random_index == index
        block_position = random_xyz[mask]
        block_weight = random_weight[mask]
        dr = cached_counter(
            cache_dir / f"DR_{index:02d}.npy",
            positions1=data_xyz,
            weights1=data_weight,
            positions2=block_position,
            weights2=block_weight,
            s_edges=s_edges,
            mu_edges=mu_edges,
            nthreads=int(args.nthreads),
        )
        rr = cached_counter(
            cache_dir / f"RR_{index:02d}.npy",
            positions1=block_position,
            weights1=block_weight,
            positions2=None,
            weights2=None,
            s_edges=s_edges,
            mu_edges=mu_edges,
            nthreads=int(args.nthreads),
        )
        correlation = LandySzalayTwoPointEstimator(D1D2=dd, D1R2=dr, R1R2=rr)
        xi = np.asarray(correlation.corr, dtype="f8")
        xi_by_random.append(xi)
        dr_by_random.append(np.asarray(dr.wcounts, dtype="f8"))
        rr_by_random.append(np.asarray(rr.wcounts, dtype="f8"))
        block_elapsed.append(time.perf_counter() - block_started)
        print(
            f"[split-random-smu-cpu] {row['sim_name']} {row['space']} index={index:02d} "
            f"finite={bool(np.all(np.isfinite(xi)))} elapsed={block_elapsed[-1]:.2f}s",
            flush=True,
        )
        del block_position, block_weight, dr, rr, correlation
        gc.collect()

    xi_stack = np.stack(xi_by_random)
    xi_smu = np.mean(xi_stack, axis=0)
    dr_mean = np.mean(np.stack(dr_by_random), axis=0)
    rr_mean = np.mean(np.stack(rr_by_random), axis=0)
    multipoles = project_multipoles(xi_smu, mu_edges, (0, 2))
    rr_radial = np.sum(rr_mean, axis=1)
    scalar_consistency = float(
        np.max(
            np.abs(np.sum(dd_value, axis=1) / float(dd.wnorm) - np.asarray(dd.normalized_wcounts()).sum(axis=1))
        )
    )
    finite_gate = bool(
        np.all(np.isfinite(xi_stack))
        and np.all(np.isfinite(multipoles))
        and np.all(rr_mean > 0.0)
        and scalar_consistency < 1.0e-12
    )
    if not finite_gate:
        raise RuntimeError("pycorr lightcone xi finite/RR/count-normalization gate failed")

    atomic_savez(
        output,
        s=s,
        s_edges=s_edges,
        mu=mu,
        mu_edges=mu_edges,
        ells=np.asarray((0, 2), dtype="i4"),
        xi0=multipoles[0],
        xi2=multipoles[1],
        xi_multipoles=multipoles,
        xi_smu=xi_smu,
        xi_smu_by_random=xi_stack,
        DD_smu=dd_value,
        DR_smu=dr_mean,
        RR_smu=rr_mean,
        RR=rr_radial,
        ndata=np.asarray(data_xyz.shape[0], dtype="i8"),
        nrandom=np.asarray(random_xyz.shape[0], dtype="i8"),
        zeff=arrays["zeff"],
        p0=arrays["p0"],
        phase=np.asarray(row["phase"]),
    )
    metadata = {
        "task": "task43_measure_pngbase_pseudolc_xi_pycorr",
        "status": "pass",
        "cosmology": row["cosmology"],
        "phase": row["phase"],
        "space": row["space"],
        "engine": "pycorr/Corrfunc CPU",
        "pycorr_version": str(pycorr_version),
        "corrfunc_version": str(Corrfunc.__version__),
        "nthreads": int(args.nthreads),
        "estimator": "Landy-Szalay, arithmetic mean over 25 independent RANDOM_INDEX blocks",
        "los": "midpoint",
        "ells": [0, 2],
        "nmu": int(args.nmu),
        "s_edges_mpc_h": s_edges.tolist(),
        "ndata": int(data_xyz.shape[0]),
        "nrandom_total": int(random_xyz.shape[0]),
        "random_block_sizes": sizes,
        "p0": float(arrays["p0"]),
        "zeff": float(arrays["zeff"]),
        "weighting": "WEIGHT_TOTAL=WEIGHT*WEIGHT_FKP from phase observed nbar(z)",
        "radial_random_policy": row["random_radial_policy"],
        "finite_and_rr_positive_gate": finite_gate,
        "scalar_vs_smu_normalized_count_max_abs": scalar_consistency,
        "pair_cache_dir": str(cache_dir),
        "block_elapsed_sec": block_elapsed,
        "data_catalog_path": row["lightcone_catalog_path"],
        "data_catalog_sha256": sha256_file(Path(row["lightcone_catalog_path"])),
        "random_catalog_path": row["lightcone_random_path"],
        "random_catalog_sha256": sha256_file(Path(row["lightcone_random_path"])),
        "output_path": str(output),
        "output_sha256": sha256_file(output),
        "elapsed_sec": time.perf_counter() - started,
    }
    atomic_write_json(metadata_path, metadata)
    print(json.dumps({"status": "pass", "output": str(output), "elapsed_sec": metadata["elapsed_sec"]}, sort_keys=True))


if __name__ == "__main__":
    main()

