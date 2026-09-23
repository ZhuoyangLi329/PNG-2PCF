#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure Task44 xi0 from frozen, independently generated random blocks.

Random generation belongs exclusively to ``task44_generate_random75x.py``.
This program consumes that generator's manifest and HDF5 blocks; it contains
no angular, cap, redshift, NX, or FKP random-generation logic of its own.

DD is counted once.  Each random block has a restartable DR_i and within-block
RR_i counter computed with pycorr/Corrfunc.  Nested M-block estimates use
normalized counts,

    xi_M = (dd - 2 mean_i(dr_i) + mean_i(rr_i)) / mean_i(rr_i).

There are deliberately no RR pairs between distinct blocks.  The result is an
explicit incomplete block-U estimator, not exact pooled-random Landy--Szalay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_NESTED = (3, 10, 25, 50, 75)
MEASUREMENT_SCHEMA = "task44_random75x_xi_block_u_v1"
GENERATOR_SCHEMA = "task44_random75x_v1"


def canonical_digest(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_signature(path: Path, *, sha256: bool = False) -> dict[str, Any]:
    path = path.resolve(strict=True)
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if sha256:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def atomic_savez(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def parse_nested(text: str) -> tuple[int, ...]:
    values = tuple(sorted(set(int(item.strip()) for item in str(text).split(",") if item.strip())))
    if not values or values[0] <= 0:
        raise ValueError("--nested must contain positive integers")
    return values


def build_edges(s_min: float, s_max: float, ds: float) -> np.ndarray:
    if ds <= 0.0 or s_max <= s_min:
        raise ValueError("require ds > 0 and s_max > s_min")
    edges = np.arange(float(s_min), float(s_max) + 0.5 * float(ds), float(ds), dtype="f8")
    if edges.size < 2 or not np.isclose(edges[-1], s_max, rtol=0.0, atol=1.0e-10):
        raise ValueError("the requested s range must contain an integer number of bins")
    edges[0], edges[-1] = float(s_min), float(s_max)
    return edges


def immutable_generator_contract(manifest: dict[str, Any], sample: str) -> dict[str, Any]:
    if manifest.get("schema_version") != GENERATOR_SCHEMA:
        raise ValueError(f"wrong generator schema {manifest.get('schema_version')!r}")
    if sample not in manifest.get("samples", []):
        raise ValueError(f"sample {sample!r} is absent from generator manifest")
    target = manifest.get("targets", {}).get(sample)
    if not isinstance(target, dict):
        raise ValueError(f"generator manifest has no target for {sample}")
    cap_counts = {cap: int(target["cap_counts"][cap]) for cap in ("NGC", "SGC")}
    total = int(target["per_block_total"])
    if total != sum(cap_counts.values()) or total <= 0:
        raise ValueError(f"invalid generator target {target}")
    contract = {
        "schema_version": manifest["schema_version"],
        "num_blocks_contract": int(manifest["num_blocks_contract"]),
        "block_multiplier": float(manifest["block_multiplier"]),
        "sample": sample,
        "cap_counts": cap_counts,
        "per_block_total": total,
        "footprint_path": str(manifest["footprint_path"]),
        "footprint_sha256": str(manifest["footprint_sha256"]),
        "angular_policy": str(manifest["angular_policy"]),
        "cap_policy": str(manifest["cap_policy"]),
        "radial_policy": str(manifest["radial_policy"]),
        "p0": float(manifest["p0"]),
    }
    if contract["num_blocks_contract"] != 75 or contract["block_multiplier"] != 1.0:
        raise ValueError(f"generator is not the frozen 75 independent 1x contract: {contract}")
    return contract


def redshift_mask(z: np.ndarray, zmin: float | None, zmax: float | None) -> np.ndarray:
    """Return the frozen half-open redshift selection used by sub-samples.

    The final upper edge can be made inclusive through ``zmax_inclusive`` in
    the caller's metadata; Task44 source catalogs already satisfy strict
    upper cuts, so the numerical mask remains half-open for every bin and the
    union audit handles the last endpoint explicitly.
    """

    values = np.asarray(z, dtype="f8")
    mask = np.ones(values.size, dtype=bool)
    if zmin is not None:
        mask &= values >= float(zmin)
    if zmax is not None:
        mask &= values < float(zmax)
    return mask


def load_data(
    path: Path,
    *,
    max_data: int | None,
    seed: int,
    zmin: float | None = None,
    zmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        n_source = int(np.asarray(payload["X"]).size)
        z_all = np.asarray(payload["Z"], dtype="f8")
        selected = np.flatnonzero(redshift_mask(z_all, zmin, zmax))
        n_total = int(selected.size)
        if max_data is not None and 0 < int(max_data) < n_total:
            rng = np.random.default_rng(int(seed))
            local = np.sort(rng.choice(n_total, size=int(max_data), replace=False))
            choice: np.ndarray = selected[local]
        else:
            choice = selected
        positions = np.vstack(
            [np.asarray(payload[name][choice], dtype="f8") for name in ("X", "Y", "Zcart")]
        )
        weights = np.asarray(payload["WEIGHT_TOTAL"][choice], dtype="f8")
        meta_text = str(np.asarray(payload["meta_json"]).item()) if "meta_json" in payload.files else ""
    if positions.shape != (3, weights.size) or weights.size == 0:
        raise ValueError(f"invalid data shape from {path}: {positions.shape}, {weights.shape}")
    if not np.isfinite(positions).all() or not np.isfinite(weights).all():
        raise ValueError(f"non-finite data values in {path}")
    return positions, weights, {
        "n_source": n_source,
        "n_total": n_total,
        "n_used": int(weights.size),
        "subsample_applied": bool(weights.size != n_total),
        "subsample_seed": int(seed),
        "zmin": None if zmin is None else float(zmin),
        "zmax": None if zmax is None else float(zmax),
        "redshift_selection": "z >= zmin and z < zmax",
        "meta_json_sha256": hashlib.sha256(meta_text.encode()).hexdigest(),
    }


def pair_paths(root: Path, block_index: int | None = None) -> tuple[Path, ...]:
    if block_index is None:
        return (root / "pairs" / "DD.npy",)
    tag = f"block{int(block_index):03d}"
    parent = root / "pairs" / "blocks"
    return parent / f"{tag}_DR.npy", parent / f"{tag}_RR.npy", parent / f"{tag}.json"


def load_counter(path: Path, *, edges: np.ndarray, size1: int, size2: int) -> Any:
    from pycorr.twopoint_counter import BaseTwoPointCounter

    counter = BaseTwoPointCounter.load(str(path))
    if str(counter.mode) != "s" or not np.array_equal(np.asarray(counter.edges[0], dtype="f8"), edges):
        raise ValueError(f"counter geometry mismatch in {path}")
    if int(counter.size1) != int(size1) or int(counter.size2) != int(size2):
        raise ValueError(f"counter sizes {(counter.size1, counter.size2)} != {(size1, size2)} in {path}")
    if not np.isfinite(np.asarray(counter.wcounts, dtype="f8")).all() or not np.isfinite(float(counter.wnorm)):
        raise ValueError(f"non-finite counter in {path}")
    return counter


def atomic_counter_save(counter: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp-{os.getpid()}-{uuid.uuid4().hex}.npy")
    counter.save(str(tmp))
    os.replace(tmp, path)


def validate_counter_provenance(counter: Any, expected: dict[str, Any], path: Path) -> None:
    stored = dict(getattr(counter, "attrs", {})).get("task44_random75x")
    if stored != expected:
        raise ValueError(f"counter provenance mismatch in {path}: {stored!r} != {expected!r}")


def load_contract(root: Path) -> tuple[dict[str, Any], str]:
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    if contract.get("schema") != MEASUREMENT_SCHEMA:
        raise ValueError(f"wrong measurement contract schema in {root}")
    digest = canonical_digest(contract)
    # Backward-compatible in-memory defaults for the completed all-sample
    # random75x contracts.  These keys are added only after computing the
    # immutable digest, so cached counter provenance remains unchanged.
    if "nrandom_by_block" not in contract:
        contract["nrandom_by_block"] = [
            int(contract["nrandom_per_block"])
        ] * int(contract["nblocks"])
    contract.setdefault("parent_sample", contract["sample"])
    contract.setdefault("tracer", "LRG" if contract["parent_sample"] == "lrgall" else "QSO")
    contract.setdefault("z_selection", {"zmin": None, "zmax": None, "policy": "none"})
    return contract, digest


def validate_generator_record(
    record: dict[str, Any],
    generator: dict[str, Any],
    *,
    sample: str,
    block_index: int,
    random_path: Path,
) -> None:
    expected = {
        "schema_version": generator["schema_version"],
        "status": "complete",
        "sample": sample,
        "block_id": int(block_index),
        "num_blocks_contract": int(generator["num_blocks_contract"]),
        "block_multiplier": 1.0,
        "nrandom": int(generator["per_block_total"]),
        "cap_counts": generator["cap_counts"],
        "p0": float(generator["p0"]),
        "footprint_sha256": generator["footprint_sha256"],
        "angular_policy": generator["angular_policy"],
        "cap_policy": generator["cap_policy"],
        "radial_policy": generator["radial_policy"],
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"generator record mismatch for {key}: {record.get(key)!r} != {value!r}")
    if Path(str(record.get("output", ""))).resolve(strict=True) != random_path.resolve(strict=True):
        raise ValueError("generator sidecar output path does not match --random-block")


def load_random_block(
    path: Path,
    *,
    generator: dict[str, Any],
    sample: str,
    block_index: int,
    zmin: float | None = None,
    zmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = path.resolve(strict=True)
    sidecar = path.with_suffix(".json")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    validate_generator_record(record, generator, sample=sample, block_index=block_index, random_path=path)
    nrandom = int(generator["per_block_total"])
    with h5py.File(path, "r") as h5:
        required = ("X", "Y", "Z", "Zcart", "WEIGHT_TOTAL", "CAP_CODE", "RANDOM_INDEX")
        for name in required:
            if name not in h5:
                raise KeyError(f"missing random dataset {name}: {path}")
            if int(h5[name].shape[0]) != nrandom:
                raise ValueError(f"random dataset {name} has wrong length in {path}")
        z = np.asarray(h5["Z"], dtype="f8")
        selected = redshift_mask(z, zmin, zmax)
        positions = np.vstack(
            [np.asarray(h5[name], dtype="f8")[selected] for name in ("X", "Y", "Zcart")]
        )
        weights = np.asarray(h5["WEIGHT_TOTAL"], dtype="f8")[selected]
        cap_code = np.asarray(h5["CAP_CODE"], dtype="u1")
        random_index = np.asarray(h5["RANDOM_INDEX"], dtype="u1")
    if not np.isfinite(positions).all() or not np.isfinite(weights).all():
        raise ValueError(f"non-finite random values in {path}")
    measured_caps = {"NGC": int(np.sum(cap_code == 0)), "SGC": int(np.sum(cap_code == 1))}
    if measured_caps != generator["cap_counts"]:
        raise ValueError(f"HDF5 cap counts {measured_caps} != {generator['cap_counts']}")
    if not np.all(random_index == int(block_index)):
        raise ValueError(f"RANDOM_INDEX mismatch in {path}")
    signature = {
        "hdf5": file_signature(path),
        "sidecar": file_signature(sidecar, sha256=True),
        "record_output_sha256": record.get("output_sha256"),
        "cap_seeds": record.get("cap_seeds"),
        "nrandom_after_zcut": int(weights.size),
        "zmin": None if zmin is None else float(zmin),
        "zmax": None if zmax is None else float(zmax),
    }
    return positions, weights, signature


def random_block_path(root: Path, parent_sample: str, tracer: str, index: int) -> Path:
    """Return one immutable generator block without materializing a z-cut copy."""

    return (
        Path(root)
        / str(parent_sample)
        / f"task44_ph001_{tracer}_random75x_block{int(index):02d}.h5"
    )


def scan_random_counts(
    *,
    root: Path,
    parent_sample: str,
    tracer: str,
    generator: dict[str, Any],
    nblocks: int,
    zmin: float | None,
    zmax: float | None,
) -> tuple[list[int], list[str]]:
    """Audit parent blocks and record actual per-block counts after the z cut."""

    counts: list[int] = []
    paths: list[str] = []
    for index in range(int(nblocks)):
        path = random_block_path(root, parent_sample, tracer, index).resolve(strict=True)
        record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        validate_generator_record(
            record,
            generator,
            sample=parent_sample,
            block_index=index,
            random_path=path,
        )
        with h5py.File(path, "r") as h5:
            count = int(np.count_nonzero(redshift_mask(np.asarray(h5["Z"], dtype="f8"), zmin, zmax)))
        if count <= 0:
            raise ValueError(f"empty z-cut random block {index}: {path}")
        counts.append(count)
        paths.append(str(path))
    return counts, paths


def command_prepare(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    sample = str(args.sample)
    parent_sample = str(args.parent_sample or sample)
    edges = build_edges(args.s_min, args.s_max, args.ds)
    data_path = Path(args.data).resolve(strict=True)
    manifest_path = Path(args.random_manifest).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generator = immutable_generator_contract(manifest, parent_sample)
    if int(args.nblocks) <= 0 or int(args.nblocks) > int(generator["num_blocks_contract"]):
        raise ValueError("--nblocks must lie within the frozen generator contract")
    if not np.isclose(float(args.p0), float(generator["p0"]), rtol=0.0, atol=0.0):
        raise ValueError("measurement P0 differs from the generator P0")
    positions, weights, data_meta = load_data(
        data_path,
        max_data=args.max_data,
        seed=args.data_seed,
        zmin=args.zmin,
        zmax=args.zmax,
    )
    tracer = str(args.tracer or ("LRG" if parent_sample == "lrgall" else "QSO"))
    random_counts, random_paths = scan_random_counts(
        root=Path(args.random_root),
        parent_sample=parent_sample,
        tracer=tracer,
        generator=generator,
        nblocks=int(args.nblocks),
        zmin=args.zmin,
        zmax=args.zmax,
    )
    density_ratios = np.asarray(random_counts, dtype="f8") / float(weights.size)
    contract = {
        "schema": MEASUREMENT_SCHEMA,
        "task": "task44_random75x_xi",
        "sample": sample,
        "parent_sample": parent_sample,
        "tracer": tracer,
        "realization": str(args.realization),
        "p0": float(args.p0),
        "data": file_signature(data_path),
        "data_meta": data_meta,
        "nblocks": int(args.nblocks),
        "nrandom_by_block": random_counts,
        "random_block_paths": random_paths,
        "random_density_ratio_by_block": density_ratios.tolist(),
        "random_density_ratio_mean": float(np.mean(density_ratios)),
        "random_density_ratio_std": float(np.std(density_ratios)),
        "z_selection": {
            "zmin": None if args.zmin is None else float(args.zmin),
            "zmax": None if args.zmax is None else float(args.zmax),
            "policy": "z >= zmin and z < zmax",
        },
        "s_edges": edges.tolist(),
        "generator_manifest_path": str(manifest_path),
        "generator_contract": generator,
        "generator_contract_sha256": canonical_digest(generator),
        "pair_estimator": {
            "engine": "pycorr-Corrfunc",
            "DD": "one shared count",
            "DR_RR": "one DR and one within-block RR per frozen HDF5 random block",
            "cross_block_RR": False,
            "aggregate": "normalized_pair_count_incomplete_block_U",
        },
    }
    contract_path = root / "contract.json"
    if contract_path.is_file() and not args.force:
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != contract:
            raise RuntimeError(f"existing contract differs in {contract_path}; use a new root or --force")
    atomic_save(root / "data" / "positions.npy", positions)
    atomic_save(root / "data" / "weights.npy", weights)
    atomic_json(contract_path, contract)
    dd_path = pair_paths(root)[0]
    if args.force:
        dd_path.unlink(missing_ok=True)
    if dd_path.is_file():
        load_counter(dd_path, edges=edges, size1=weights.size, size2=weights.size)
        print(f"[skip] valid shared DD {dd_path}", flush=True)
    else:
        from pycorr import TwoPointCounter

        start = time.perf_counter()
        dd = TwoPointCounter(
            mode="s", edges=edges, positions1=positions, weights1=weights,
            engine="corrfunc", nthreads=int(args.nthreads),
        )
        atomic_counter_save(dd, dd_path)
        print(f"[write] shared DD {dd_path} elapsed={time.perf_counter() - start:.1f}s", flush=True)
    print(
        f"[prepared] sample={sample} ndata={weights.size} "
        f"nrandom/block={min(random_counts)}..{max(random_counts)} nblocks={args.nblocks}", flush=True,
    )


def command_block(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    contract, contract_digest = load_contract(root)
    index = int(args.block_index)
    if not 0 <= index < int(contract["nblocks"]):
        raise ValueError(f"block {index} outside prepared range")
    edges = np.asarray(contract["s_edges"], dtype="f8")
    data_positions = np.load(root / "data" / "positions.npy", mmap_mode="r")
    data_weights = np.load(root / "data" / "weights.npy", mmap_mode="r")
    ndata = int(data_weights.size)
    nrandom = int(contract["nrandom_by_block"][index])
    dr_path, rr_path, meta_path = pair_paths(root, index)
    random_path = Path(args.random_block)
    sidecar_signature = file_signature(random_path.with_suffix(".json"), sha256=True)
    random_stat = file_signature(random_path)
    input_signature = {"hdf5": random_stat, "sidecar": sidecar_signature}
    common_provenance = {
        "schema": MEASUREMENT_SCHEMA,
        "contract_sha256": contract_digest,
        "block_index": index,
        "random_sidecar_sha256": str(sidecar_signature["sha256"]),
    }
    dr_provenance = {**common_provenance, "pair": "DR"}
    rr_provenance = {**common_provenance, "pair": "RR"}
    if dr_path.is_file() and rr_path.is_file() and meta_path.is_file() and not args.force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("contract_sha256") != contract_digest or meta.get("random_input_signature") != input_signature:
            raise RuntimeError(f"stale pair-count block metadata {meta_path}")
        dr_cached = load_counter(dr_path, edges=edges, size1=ndata, size2=nrandom)
        rr_cached = load_counter(rr_path, edges=edges, size1=nrandom, size2=nrandom)
        validate_counter_provenance(dr_cached, dr_provenance, dr_path)
        validate_counter_provenance(rr_cached, rr_provenance, rr_path)
        print(f"[skip] valid pair-count block {index:03d}", flush=True)
        return
    if args.force:
        for path in (dr_path, rr_path, meta_path):
            path.unlink(missing_ok=True)

    random_positions, random_weights, audited_signature = load_random_block(
        random_path,
        generator=contract["generator_contract"],
        sample=contract["parent_sample"],
        block_index=index,
        zmin=contract["z_selection"]["zmin"],
        zmax=contract["z_selection"]["zmax"],
    )
    if int(random_weights.size) != nrandom:
        raise RuntimeError(
            f"z-cut random count changed for block {index}: {random_weights.size} != {nrandom}"
        )
    if audited_signature["hdf5"] != random_stat or audited_signature["sidecar"] != sidecar_signature:
        raise RuntimeError("random block changed while it was being loaded")
    from pycorr import TwoPointCounter

    kwargs = {"mode": "s", "edges": edges, "engine": "corrfunc", "nthreads": int(args.nthreads)}
    start = time.perf_counter()
    if dr_path.is_file():
        dr = load_counter(dr_path, edges=edges, size1=ndata, size2=nrandom)
        validate_counter_provenance(dr, dr_provenance, dr_path)
        print(f"[resume] valid DR {dr_path}", flush=True)
    else:
        dr = TwoPointCounter(
            positions1=data_positions, weights1=data_weights,
            positions2=random_positions, weights2=random_weights, **kwargs,
        )
        dr.attrs["task44_random75x"] = dr_provenance
        atomic_counter_save(dr, dr_path)
    del dr
    if rr_path.is_file():
        rr = load_counter(rr_path, edges=edges, size1=nrandom, size2=nrandom)
        validate_counter_provenance(rr, rr_provenance, rr_path)
        print(f"[resume] valid RR {rr_path}", flush=True)
    else:
        rr = TwoPointCounter(positions1=random_positions, weights1=random_weights, **kwargs)
        rr.attrs["task44_random75x"] = rr_provenance
        atomic_counter_save(rr, rr_path)
    del rr
    meta = {
        "schema": MEASUREMENT_SCHEMA,
        "task": "task44_random75x_xi_pair_block",
        "status": "done",
        "sample": contract["sample"],
        "block_index": index,
        "contract_sha256": contract_digest,
        "random_input_signature": input_signature,
        "random_audited_signature": audited_signature,
        "ndata": ndata,
        "nrandom": nrandom,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "elapsed_sec": float(time.perf_counter() - start),
        "cross_block_RR": False,
    }
    atomic_json(meta_path, meta)
    print(f"[done] pair-count block={index:03d} elapsed={meta['elapsed_sec']:.1f}s", flush=True)


def jackknife_random_covariance(dd: np.ndarray, dr: np.ndarray, rr: np.ndarray) -> np.ndarray:
    nblock, nbin = int(dr.shape[0]), int(dd.size)
    if nblock <= 1:
        return np.full((nbin, nbin), np.nan, dtype="f8")
    dr_sum, rr_sum = np.sum(dr, axis=0), np.sum(rr, axis=0)
    leave_one_out = []
    for index in range(nblock):
        dr_mean = (dr_sum - dr[index]) / (nblock - 1)
        rr_mean = (rr_sum - rr[index]) / (nblock - 1)
        if np.any(rr_mean <= 0.0):
            return np.full((nbin, nbin), np.nan, dtype="f8")
        leave_one_out.append((dd - 2.0 * dr_mean + rr_mean) / rr_mean)
    values = np.asarray(leave_one_out, dtype="f8")
    delta = values - np.mean(values, axis=0)
    return (nblock - 1.0) / nblock * (delta.T @ delta)


def command_aggregate(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    contract, contract_digest = load_contract(root)
    edges = np.asarray(contract["s_edges"], dtype="f8")
    ndata = int(contract["data_meta"]["n_used"])
    dd_counter = load_counter(pair_paths(root)[0], edges=edges, size1=ndata, size2=ndata)
    dd = np.asarray(dd_counter.normalized_wcounts(), dtype="f8")
    nested = parse_nested(args.nested)
    if nested[-1] > int(contract["nblocks"]):
        raise ValueError("nested multiplier exceeds prepared nblocks")
    dr_rows, rr_rows = [], []
    for index in range(nested[-1]):
        dr_path, rr_path, meta_path = pair_paths(root, index)
        if not (dr_path.is_file() and rr_path.is_file() and meta_path.is_file()):
            raise FileNotFoundError(f"pair-count block {index:03d} is incomplete")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("contract_sha256") != contract_digest:
            raise RuntimeError(f"contract mismatch in pair-count block {index:03d}")
        nrandom = int(contract["nrandom_by_block"][index])
        dr = load_counter(dr_path, edges=edges, size1=ndata, size2=nrandom)
        rr = load_counter(rr_path, edges=edges, size1=nrandom, size2=nrandom)
        dr_rows.append(np.asarray(dr.normalized_wcounts(), dtype="f8"))
        rr_rows.append(np.asarray(rr.normalized_wcounts(), dtype="f8"))
    dr_by_block = np.asarray(dr_rows, dtype="f8")
    rr_by_block = np.asarray(rr_rows, dtype="f8")
    xi_rows, dr_mean_rows, rr_mean_rows, covariance_rows = [], [], [], []
    for multiplier in nested:
        dr_mean = np.mean(dr_by_block[:multiplier], axis=0)
        rr_mean = np.mean(rr_by_block[:multiplier], axis=0)
        if np.any(rr_mean <= 0.0):
            raise ValueError(f"non-positive RR for M={multiplier}")
        xi = (dd - 2.0 * dr_mean + rr_mean) / rr_mean
        xi_rows.append(xi)
        dr_mean_rows.append(dr_mean)
        rr_mean_rows.append(rr_mean)
        covariance_rows.append(jackknife_random_covariance(dd, dr_by_block[:multiplier], rr_by_block[:multiplier]))
        print(f"[nested] M={multiplier} finite={bool(np.isfinite(xi).all())}", flush=True)
    tag = "_".join(str(value) for value in nested)
    random_tag = f"random{int(nested[-1])}x"
    output = root / "nested" / f"task44_{contract['sample']}_{random_tag}_incomplete_block_u_nested{tag}.npz"
    measurement = root / "measurements" / f"task44_{contract['sample']}_{random_tag}_M{int(nested[-1])}_xi0_s30_350_ds10.npz"
    meta = {
        "schema": MEASUREMENT_SCHEMA,
        "task": "task44_random75x_xi_aggregate",
        "status": "done",
        "sample": contract["sample"],
        "contract_sha256": contract_digest,
        "generator_contract_sha256": contract["generator_contract_sha256"],
        "nested_multipliers": list(nested),
        "estimator": "normalized_pair_count_incomplete_block_U_Landy_Szalay",
        "formula": "xi_M=(dd-2*mean_i(dr_i)+mean_i(rr_i))/mean_i(rr_i)",
        "cross_block_RR": False,
        "not_exact_pooled_random": True,
        "random_covariance": (
            "delete-one-random-block jackknife; separate conditional finite-random term, never "
            "a replacement, average, or divisor for physical Csingle; a downstream likelihood "
            "may add it only under an explicit Ctotal contract"
        ),
        "physical_covariance_policy": "not loaded, averaged, divided, rescaled, or modified",
        "output": str(output),
        "measurement_output": str(measurement),
    }
    atomic_savez(
        output,
        s_edges=edges,
        s=0.5 * (edges[:-1] + edges[1:]),
        # Canonical convergence-diagnostic schema.
        densities=np.asarray(nested, dtype="i8"),
        xi_nested=np.asarray(xi_rows, dtype="f8"),
        block_density=np.asarray(1.0, dtype="f8"),
        block_order=np.arange(dr_by_block.shape[0], dtype="i8"),
        dr_normalized_blocks=dr_by_block,
        rr_normalized_blocks=rr_by_block,
        # Historical/self-describing aliases retained for direct inspection.
        nested_multipliers=np.asarray(nested, dtype="i8"),
        xi0=np.asarray(xi_rows, dtype="f8"),
        dd_normalized=dd,
        dr_normalized_mean=np.asarray(dr_mean_rows, dtype="f8"),
        rr_normalized_mean=np.asarray(rr_mean_rows, dtype="f8"),
        dr_normalized_by_block=dr_by_block,
        rr_normalized_by_block=rr_by_block,
        finite_random_covariance_jackknife=np.asarray(covariance_rows, dtype="f8"),
        meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    atomic_json(output.with_suffix(".json"), meta)
    measurement_meta = {
        "schema": MEASUREMENT_SCHEMA,
        "task": "task44_random75x_xi_measurement",
        "status": "done",
        "sample": contract["sample"],
        "realization": contract["realization"],
        "random_density_multiplier": int(nested[-1]),
        "random_block_count": int(nested[-1]),
        "random_combination": "normalized_pair_count_incomplete_block_U_then_one_Landy_Szalay_ratio",
        "cross_block_RR": False,
        "not_exact_pooled_random": True,
        "contract_sha256": contract_digest,
        "generator_contract_sha256": contract["generator_contract_sha256"],
        "physical_covariance_policy": "not loaded, averaged, divided, rescaled, or modified",
        "output": str(measurement),
    }
    atomic_savez(
        measurement,
        s_edges=edges,
        s=0.5 * (edges[:-1] + edges[1:]),
        xi0=np.asarray(xi_rows[-1], dtype="f8"),
        DD=np.asarray(dd, dtype="f8"),
        DR=np.asarray(dr_mean_rows[-1], dtype="f8"),
        RR=np.asarray(rr_mean_rows[-1], dtype="f8"),
        ndata=np.asarray(ndata, dtype="i8"),
        nrandom=np.asarray(sum(int(v) for v in contract["nrandom_by_block"][: int(nested[-1])]), dtype="i8"),
        random_block_count=np.asarray(int(nested[-1]), dtype="i8"),
        meta_json=np.asarray(json.dumps(measurement_meta, sort_keys=True)),
    )
    atomic_json(measurement.with_suffix(".json"), measurement_meta)
    print(f"[write] {output}", flush=True)
    print(f"[write] {measurement}", flush=True)


def command_status(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    contract, contract_digest = load_contract(root)
    edges = np.asarray(contract["s_edges"], dtype="f8")
    ndata = int(contract["data_meta"]["n_used"])
    complete, missing, invalid = [], [], []
    for index in range(int(contract["nblocks"])):
        dr_path, rr_path, meta_path = pair_paths(root, index)
        if not (dr_path.is_file() and rr_path.is_file() and meta_path.is_file()):
            missing.append(index)
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("contract_sha256") != contract_digest:
                raise ValueError("contract digest")
            nrandom = int(contract["nrandom_by_block"][index])
            load_counter(dr_path, edges=edges, size1=ndata, size2=nrandom)
            load_counter(rr_path, edges=edges, size1=nrandom, size2=nrandom)
        except Exception as exc:
            invalid.append({"index": index, "error": str(exc)})
        else:
            complete.append(index)
    print(json.dumps({
        "sample": contract["sample"], "nblocks": int(contract["nblocks"]),
        "complete": complete, "missing": missing, "invalid": invalid,
        "status": "done" if len(complete) == int(contract["nblocks"]) and not invalid else "incomplete",
    }, indent=2, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Cache data and compute the one shared DD.")
    prepare.add_argument("--sample", required=True, help="Output sub-sample label, e.g. lrg1 or qso2.")
    prepare.add_argument("--parent-sample", choices=("lrgall", "qsoall"), default=None)
    prepare.add_argument("--tracer", choices=("LRG", "QSO"), default=None)
    prepare.add_argument("--realization", default="ph001")
    prepare.add_argument("--data", type=Path, required=True)
    prepare.add_argument("--random-manifest", type=Path, required=True)
    prepare.add_argument("--random-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--p0", type=float, default=6000.0)
    prepare.add_argument("--nblocks", type=int, default=75)
    prepare.add_argument("--zmin", type=float, default=None)
    prepare.add_argument("--zmax", type=float, default=None)
    prepare.add_argument("--max-data", type=int, default=None, help="Smoke/debug only.")
    prepare.add_argument("--data-seed", type=int, default=2026082702)
    prepare.add_argument("--s-min", type=float, default=30.0)
    prepare.add_argument("--s-max", type=float, default=350.0)
    prepare.add_argument("--ds", type=float, default=10.0)
    prepare.add_argument("--nthreads", type=int, default=8)
    prepare.add_argument("--force", action="store_true")
    block = commands.add_parser("block", help="Count DR/RR for one frozen generator HDF5 block.")
    block.add_argument("--output-root", type=Path, required=True)
    block.add_argument("--block-index", type=int, required=True)
    block.add_argument("--random-block", type=Path, required=True)
    block.add_argument("--nthreads", type=int, default=8)
    block.add_argument("--force", action="store_true")
    aggregate = commands.add_parser("aggregate", help="Build nested incomplete block-U xi0 products.")
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--nested", default=",".join(str(value) for value in DEFAULT_NESTED))
    status = commands.add_parser("status", help="Validate prepared pair-count block checkpoints.")
    status.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if getattr(args, "nthreads", 1) <= 0:
        raise ValueError("--nthreads must be positive")
    if args.command == "prepare":
        command_prepare(args)
    elif args.command == "block":
        command_block(args)
    elif args.command == "aggregate":
        command_aggregate(args)
    elif args.command == "status":
        command_status(args)
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
