#!/usr/bin/env python3
"""Measure empirical rawbox pair moments for the xi-only GSM oracle.

Full-catalog real-space xi is counted with Corrfunc.  Velocity moments use a
reproducible uniform anchor sample and a periodic scipy cKDTree.  Independent
anchor blocks expose sampling uncertainty without an unconstrained N**2 loop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from task43_rsd_common import OUTPUT_ROOT, PHASES, S_EDGES, atomic_savez, atomic_write_json, sha256_file


DEFAULT_OUT_DIR = OUTPUT_ROOT / "rawbox" / "xi_rsd_v2" / "stage03_pair_moments"


def pair_moment_edges(rmin: float, rmax: float, dr: float, inner_edge: float) -> np.ndarray:
    """Build a radial grid, optionally merging the low-count origin bins."""

    lower, upper, step, first = map(float, (rmin, rmax, dr, inner_edge))
    if not 0.0 <= lower < upper < 1000.0 or step <= 0.0:
        raise ValueError("invalid radial range")
    if lower == 0.0:
        if not 0.0 < first < upper:
            raise ValueError("inner_edge must lie inside an origin-inclusive radial range")
        tail = np.arange(first, upper + 0.5 * step, step, dtype="f8")
        edges = np.concatenate([np.asarray([0.0]), tail])
    else:
        edges = np.arange(lower, upper + 0.5 * step, step, dtype="f8")
    if not np.isclose(edges[-1], upper, rtol=0.0, atol=1.0e-12):
        raise ValueError("radial range must end on the requested fine grid")
    return edges


@dataclass
class PairSums:
    count: np.ndarray
    radial_sum: np.ndarray
    radial2_sum: np.ndarray
    transverse2_one_sum: np.ndarray
    los_count: np.ndarray
    los_aligned_sum: np.ndarray
    los2_sum: np.ndarray
    mu_abs_sum: np.ndarray
    mu2_sum: np.ndarray
    mapping_real_count: np.ndarray
    mapping_rsd_count: np.ndarray

    @classmethod
    def zeros(
        cls, nradial: int, nmu: int, mapping_shape: tuple[int, int] = (0, 0)
    ) -> "PairSums":
        radial = lambda: np.zeros(int(nradial), dtype="f8")
        angular = lambda: np.zeros((int(nradial), int(nmu)), dtype="f8")
        mapping = lambda: np.zeros(tuple(map(int, mapping_shape)), dtype="f8")
        return cls(
            radial(), radial(), radial(), radial(), angular(), angular(), angular(), angular(), angular(),
            mapping(), mapping(),
        )

    def add(self, other: "PairSums") -> None:
        for name in self.__dataclass_fields__:
            getattr(self, name)[:] += getattr(other, name)


def catalog_path(phase: str) -> Path:
    return OUTPUT_ROOT / "rawbox" / "catalogs" / (
        f"task43_rsd_rawbox_AbacusSummit_base_c000_{phase}_mmin1p4e13.npz"
    )


def validated_catalog(phase: str) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    path = catalog_path(phase)
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing rawbox catalog: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    if metadata.get("status") != "pass" or metadata.get("output_sha256") != digest:
        raise RuntimeError(f"unvalidated rawbox catalog: {path}")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {
            "position_real": np.asarray(payload["POSITION_REAL"], dtype="f8"),
            "position_rsd": np.asarray(payload["POSITION_RSD"], dtype="f8"),
            "velocity_kms": np.asarray(payload["VELOCITY_KMS"], dtype="f8"),
            "rsd_displacement": np.asarray(payload["RSD_DISPLACEMENT_MPC_H"], dtype="f8"),
            "boxsize": np.asarray(payload["boxsize"], dtype="f8"),
            "velocity_conversion": np.asarray(payload["velocity_conversion_kms_per_mpc_h"], dtype="f8"),
        }
    return arrays, metadata, digest


def _bincount(index: np.ndarray, weights: np.ndarray | None, size: int) -> np.ndarray:
    return np.bincount(index, weights=weights, minlength=int(size)).astype("f8", copy=False)


def accumulate_anchor_pairs(
    positions: np.ndarray,
    displacements: np.ndarray,
    anchors: np.ndarray,
    radial_edges: np.ndarray,
    mu_edges: np.ndarray,
    *,
    boxsize: float,
    threads: int,
    query_chunk: int = 16,
    tree: cKDTree | None = None,
    wrapped_positions: np.ndarray | None = None,
    mapping_edges: np.ndarray | None = None,
    mapping_nmu: int = 0,
) -> PairSums:
    """Accumulate oriented anchor-neighbor pair moments in real separation."""

    xyz = np.asarray(positions, dtype="f8")
    velocity = np.asarray(displacements, dtype="f8")
    anchor_ids = np.asarray(anchors, dtype="i8")
    redges = np.asarray(radial_edges, dtype="f8")
    medges = np.asarray(mu_edges, dtype="f8")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or velocity.shape != xyz.shape:
        raise ValueError("positions and displacement vectors must have shape (N,3)")
    if anchor_ids.ndim != 1 or np.any(anchor_ids < 0) or np.any(anchor_ids >= xyz.shape[0]):
        raise ValueError("invalid anchor indices")
    if np.unique(anchor_ids).size != anchor_ids.size:
        raise ValueError("anchor indices must be unique")
    if np.any(np.diff(redges) <= 0.0) or redges[0] < 0.0 or redges[-1] >= 0.5 * boxsize:
        raise ValueError("radial bins must be ordered and remain below half the periodic box")
    if medges[0] != 0.0 or medges[-1] != 1.0 or np.any(np.diff(medges) <= 0.0):
        raise ValueError("mu edges must span [0,1]")

    wrapped = (
        np.mod(xyz, float(boxsize))
        if wrapped_positions is None
        else np.asarray(wrapped_positions, dtype="f8")
    )
    if wrapped.shape != xyz.shape or np.any(wrapped < 0.0) or np.any(wrapped >= float(boxsize)):
        raise ValueError("wrapped positions do not match the catalog and periodic box")
    if tree is None:
        tree = cKDTree(wrapped, boxsize=float(boxsize))
    elif tree.n != xyz.shape[0]:
        raise ValueError("periodic tree size does not match the catalog")
    nradial = redges.size - 1
    nmu = medges.size - 1
    if mapping_edges is None:
        map_edges = np.empty(0, dtype="f8")
        map_shape = (0, 0)
    else:
        map_edges = np.asarray(mapping_edges, dtype="f8")
        if (
            map_edges.ndim != 1
            or map_edges.size < 2
            or np.any(np.diff(map_edges) <= 0.0)
            or map_edges[0] < 0.0
            or map_edges[-1] >= 0.5 * float(boxsize)
            or int(mapping_nmu) < 1
        ):
            raise ValueError("invalid mapped s-mu binning")
        map_shape = (map_edges.size - 1, int(mapping_nmu))
    result = PairSums.zeros(nradial, nmu, map_shape)
    for start in range(0, anchor_ids.size, int(query_chunk)):
        chunk = anchor_ids[start : start + int(query_chunk)]
        neighbor_lists = tree.query_ball_point(
            wrapped[chunk],
            r=float(redges[-1]),
            workers=int(threads),
            return_sorted=False,
        )
        for anchor, neighbors_raw in zip(chunk, neighbor_lists):
            neighbors = np.asarray(neighbors_raw, dtype="i8")
            neighbors = neighbors[neighbors != int(anchor)]
            if neighbors.size == 0:
                continue
            separation = wrapped[neighbors] - wrapped[int(anchor)]
            separation -= float(boxsize) * np.rint(separation / float(boxsize))
            radius2 = np.einsum("ij,ij->i", separation, separation)
            radius = np.sqrt(radius2)
            valid = (radius >= redges[0]) & (radius < redges[-1])
            if not np.any(valid):
                continue
            neighbors = neighbors[valid]
            separation = separation[valid]
            radius = radius[valid]
            radial_bin = np.searchsorted(redges, radius, side="right") - 1
            delta_u = velocity[neighbors] - velocity[int(anchor)]
            radial_u = np.einsum("ij,ij->i", delta_u, separation) / radius
            delta_u2 = np.einsum("ij,ij->i", delta_u, delta_u)
            transverse2_one = 0.5 * (delta_u2 - radial_u**2)
            if np.min(transverse2_one) < -1.0e-9:
                raise RuntimeError("negative transverse velocity square beyond roundoff")
            transverse2_one = np.maximum(transverse2_one, 0.0)

            result.count += _bincount(radial_bin, None, nradial)
            result.radial_sum += _bincount(radial_bin, radial_u, nradial)
            result.radial2_sum += _bincount(radial_bin, radial_u**2, nradial)
            result.transverse2_one_sum += _bincount(radial_bin, transverse2_one, nradial)

            mu = separation[:, 2] / radius
            mu_abs = np.abs(mu)
            mu_bin = np.minimum(np.searchsorted(medges, mu_abs, side="right") - 1, nmu - 1)
            flat_bin = radial_bin * nmu + mu_bin
            orientation = np.where(mu < 0.0, -1.0, 1.0)
            los_aligned = delta_u[:, 2] * orientation
            flat_size = nradial * nmu
            result.los_count += _bincount(flat_bin, None, flat_size).reshape(nradial, nmu)
            result.los_aligned_sum += _bincount(flat_bin, los_aligned, flat_size).reshape(nradial, nmu)
            result.los2_sum += _bincount(flat_bin, delta_u[:, 2] ** 2, flat_size).reshape(nradial, nmu)
            result.mu_abs_sum += _bincount(flat_bin, mu_abs, flat_size).reshape(nradial, nmu)
            result.mu2_sum += _bincount(flat_bin, mu**2, flat_size).reshape(nradial, nmu)

            if map_edges.size:
                real_selected = (radius >= map_edges[0]) & (radius < map_edges[-1])
                if np.any(real_selected):
                    real_radial_bin = np.searchsorted(map_edges, radius[real_selected], side="right") - 1
                    real_mu_bin = np.minimum(
                        np.floor(mu_abs[real_selected] * int(mapping_nmu)).astype("i8"),
                        int(mapping_nmu) - 1,
                    )
                    real_flat = real_radial_bin * int(mapping_nmu) + real_mu_bin
                    result.mapping_real_count += _bincount(
                        real_flat, None, int(np.prod(map_shape))
                    ).reshape(map_shape)

                rsd_separation = separation.copy()
                rsd_separation[:, 2] += delta_u[:, 2]
                rsd_separation -= float(boxsize) * np.rint(rsd_separation / float(boxsize))
                rsd_radius = np.linalg.norm(rsd_separation, axis=1)
                rsd_selected = (rsd_radius >= map_edges[0]) & (rsd_radius < map_edges[-1])
                if np.any(rsd_selected):
                    rsd_mu_abs = np.abs(rsd_separation[rsd_selected, 2]) / rsd_radius[rsd_selected]
                    rsd_radial_bin = np.searchsorted(map_edges, rsd_radius[rsd_selected], side="right") - 1
                    rsd_mu_bin = np.minimum(
                        np.floor(rsd_mu_abs * int(mapping_nmu)).astype("i8"),
                        int(mapping_nmu) - 1,
                    )
                    rsd_flat = rsd_radial_bin * int(mapping_nmu) + rsd_mu_bin
                    result.mapping_rsd_count += _bincount(
                        rsd_flat, None, int(np.prod(map_shape))
                    ).reshape(map_shape)
    return result


def finalize_mapping_counts(
    sums: PairSums,
    *,
    nanchors: int,
    catalog_size: int,
    boxsize: float,
    radial_edges: np.ndarray,
    nmu: int,
) -> dict[str, np.ndarray]:
    shape = (len(radial_edges) - 1, int(nmu))
    if sums.mapping_real_count.shape != shape or sums.mapping_rsd_count.shape != shape:
        raise ValueError("mapped pair counts do not match the requested bins")
    shell_volume = 4.0 * np.pi / 3.0 * (radial_edges[1:] ** 3 - radial_edges[:-1] ** 3)
    expected = (
        float(nanchors)
        * (float(catalog_size) - 1.0)
        * shell_volume[:, None]
        / float(boxsize) ** 3
        / int(nmu)
    )
    expected = np.broadcast_to(expected, shape).copy()
    return {
        "expected_count_uniform": expected,
        "xi_smu_real": sums.mapping_real_count / expected - 1.0,
        "xi_smu_rsd": sums.mapping_rsd_count / expected - 1.0,
    }


def finalize_pair_sums(
    sums: PairSums,
    *,
    nanchors: int,
    catalog_size: int,
    boxsize: float,
    radial_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    count = np.asarray(sums.count, dtype="f8")
    if np.any(count <= 0.0):
        raise RuntimeError("one or more radial pair-moment bins are empty")
    shell_volume = 4.0 * np.pi / 3.0 * (radial_edges[1:] ** 3 - radial_edges[:-1] ** 3)
    expected = float(nanchors) * (float(catalog_size) - 1.0) * shell_volume / float(boxsize) ** 3
    radial_mean = sums.radial_sum / count
    radial_variance = sums.radial2_sum / count - radial_mean**2
    transverse_variance_one = sums.transverse2_one_sum / count
    if np.any(radial_variance < -1.0e-10) or np.any(transverse_variance_one < 0.0):
        raise RuntimeError("measured central pair variance is negative")

    los_count = np.asarray(sums.los_count, dtype="f8")
    valid = los_count > 0.0
    los_mean = np.full_like(los_count, np.nan)
    los_variance = np.full_like(los_count, np.nan)
    mu_abs_mean = np.full_like(los_count, np.nan)
    mu2_mean = np.full_like(los_count, np.nan)
    los_mean[valid] = sums.los_aligned_sum[valid] / los_count[valid]
    los_variance[valid] = sums.los2_sum[valid] / los_count[valid] - los_mean[valid] ** 2
    mu_abs_mean[valid] = sums.mu_abs_sum[valid] / los_count[valid]
    mu2_mean[valid] = sums.mu2_sum[valid] / los_count[valid]
    return {
        "count": count,
        "expected_count_uniform": expected,
        "xi_anchor_sample": count / expected - 1.0,
        "v12_radial": radial_mean,
        "sigma_r2_central": np.maximum(radial_variance, 0.0),
        "sigma_t2_one_component": transverse_variance_one,
        "los_count": los_count,
        "los_aligned_mean": los_mean,
        "los_variance_central": los_variance,
        "mu_abs_mean": mu_abs_mean,
        "mu2_mean": mu2_mean,
    }


def full_catalog_xi_corrfunc(
    positions: np.ndarray,
    radial_edges: np.ndarray,
    *,
    boxsize: float,
    threads: int,
) -> dict[str, np.ndarray]:
    try:
        import Corrfunc
        from Corrfunc.theory import DD
    except ImportError as error:
        raise RuntimeError("Corrfunc is required; source the project cosmodesi environment") from error

    xyz = np.asarray(positions)
    corrfunc_edges = np.asarray(radial_edges, dtype="f8").copy()
    if corrfunc_edges[0] == 0.0:
        # Corrfunc includes N self-pairs when rmin is exactly zero.  The first
        # positive float keeps all distinct pairs while matching an [0,r1) bin.
        corrfunc_edges[0] = np.nextafter(0.0, 1.0)
    counted = DD(
        1,
        int(threads),
        corrfunc_edges,
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        periodic=True,
        boxsize=float(boxsize),
        output_ravg=True,
        isa="fastest",
    )
    count = np.asarray(counted["npairs"], dtype="f8")
    shell_volume = 4.0 * np.pi / 3.0 * (radial_edges[1:] ** 3 - radial_edges[:-1] ** 3)
    expected = xyz.shape[0] * (xyz.shape[0] - 1.0) * shell_volume / float(boxsize) ** 3
    return {
        "count": count,
        "expected_count_uniform": expected,
        "xi": count / expected - 1.0,
        "ravg": np.asarray(counted["ravg"], dtype="f8"),
        "corrfunc_version": np.asarray(str(Corrfunc.__version__)),
    }


def fcfc_bridge(full_count: np.ndarray, radial_edges: np.ndarray, phase: str) -> dict[str, Any]:
    path = OUTPUT_ROOT / "rawbox" / "summary" / (
        f"task43_rsd_rawbox_AbacusSummit_base_c000_{phase}_mmin1p4e13_clustering.npz"
    )
    with np.load(path, allow_pickle=False) as payload:
        fcfc_edges = np.asarray(payload["s_edges"], dtype="f8")
        xi_fcfc = np.asarray(payload["xi0_real"], dtype="f8")
        nbar = float(np.asarray(payload["nbar"]).item())
        ndata = int(np.asarray(payload["ndata"]).item())
        boxsize = float(np.asarray(payload["boxsize"]).item()) if "boxsize" in payload.files else 2000.0
    rows = []
    xi_rebinned = []
    for lower, upper in zip(fcfc_edges[:-1], fcfc_edges[1:]):
        selected = (radial_edges[:-1] >= lower - 1.0e-12) & (radial_edges[1:] <= upper + 1.0e-12)
        if not np.any(selected):
            raise RuntimeError("Corrfunc radial grid cannot be rebinned to the FCFC grid")
        shell_volume = 4.0 * np.pi / 3.0 * (upper**3 - lower**3)
        expected = ndata * (ndata - 1.0) * shell_volume / boxsize**3
        xi_rebinned.append(float(np.sum(full_count[selected]) / expected - 1.0))
        rows.append(int(np.count_nonzero(selected)))
    xi_rebinned_array = np.asarray(xi_rebinned)
    difference = xi_rebinned_array - xi_fcfc
    return {
        "status": "pass" if float(np.max(np.abs(difference))) < 1.0e-8 else "fail",
        "max_abs_corrfunc_minus_fcfc": float(np.max(np.abs(difference))),
        "rms_corrfunc_minus_fcfc": float(np.sqrt(np.mean(difference**2))),
        "fine_bins_per_fcfc_bin": rows,
        "xi_corrfunc_rebinned": xi_rebinned_array,
        "xi_fcfc": xi_fcfc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nanchors", type=int, default=1024)
    parser.add_argument("--nblocks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--rmin", type=float, default=0.0)
    parser.add_argument("--rmax", type=float, default=500.0)
    parser.add_argument("--dr", type=float, default=5.0)
    parser.add_argument("--inner-edge", type=float, default=20.0)
    parser.add_argument("--nmu", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if not 1 <= int(args.threads) <= 32:
        raise ValueError("--threads must be in [1,32]")
    if int(args.nanchors) < int(args.nblocks) or int(args.nanchors) % int(args.nblocks):
        raise ValueError("--nanchors must be a multiple of --nblocks")

    out_npz = args.output_dir / f"task43_rawbox_pair_moments_{args.phase}_v2.npz"
    out_json = out_npz.with_suffix(".json")
    if out_npz.exists() or out_json.exists():
        raise FileExistsError(f"immutable pair-moment output exists: {out_npz} / {out_json}")

    started = time.perf_counter()
    arrays, catalog_metadata, catalog_sha256 = validated_catalog(args.phase)
    position = np.mod(arrays["position_real"], float(arrays["boxsize"]))
    boxsize = float(arrays["boxsize"])
    conversion = float(arrays["velocity_conversion"])
    displacement = arrays["velocity_kms"] / conversion
    expected_rsd_z = np.mod(position[:, 2] + displacement[:, 2], boxsize)
    rsd_error = np.abs(expected_rsd_z - arrays["position_rsd"][:, 2])
    rsd_error = np.minimum(rsd_error, boxsize - rsd_error)
    rsd_bridge_max = float(np.max(rsd_error))
    if rsd_bridge_max > 1.0e-4:
        raise RuntimeError(f"catalog velocity conversion fails RSD bridge: {rsd_bridge_max}")

    radial_edges = pair_moment_edges(args.rmin, args.rmax, args.dr, args.inner_edge)
    mu_edges = np.linspace(0.0, 1.0, int(args.nmu) + 1)
    full_xi = full_catalog_xi_corrfunc(position, radial_edges, boxsize=boxsize, threads=int(args.threads))
    bridge = fcfc_bridge(full_xi["count"], radial_edges, args.phase)
    if bridge["status"] != "pass":
        raise RuntimeError(f"Corrfunc/FCFC real-space xi bridge failed: {bridge}")

    phase_number = int(args.phase[2:])
    rng = np.random.default_rng(int(args.seed) + 1009 * phase_number)
    anchors = rng.choice(position.shape[0], size=int(args.nanchors), replace=False)
    block_size = int(args.nanchors) // int(args.nblocks)
    block_sums: list[PairSums] = []
    block_results: list[dict[str, np.ndarray]] = []
    pooled = PairSums.zeros(radial_edges.size - 1, mu_edges.size - 1)
    for block_index in range(int(args.nblocks)):
        block_anchors = anchors[block_index * block_size : (block_index + 1) * block_size]
        sums = accumulate_anchor_pairs(
            position,
            displacement,
            block_anchors,
            radial_edges,
            mu_edges,
            boxsize=boxsize,
            threads=int(args.threads),
        )
        result = finalize_pair_sums(
            sums,
            nanchors=block_anchors.size,
            catalog_size=position.shape[0],
            boxsize=boxsize,
            radial_edges=radial_edges,
        )
        block_sums.append(sums)
        block_results.append(result)
        pooled.add(sums)
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "block": block_index,
                    "nanchors": int(block_anchors.size),
                    "pairs": int(np.sum(sums.count)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    pooled_result = finalize_pair_sums(
        pooled,
        nanchors=int(args.nanchors),
        catalog_size=position.shape[0],
        boxsize=boxsize,
        radial_edges=radial_edges,
    )

    uncertainty_keys = ("xi_anchor_sample", "v12_radial", "sigma_r2_central", "sigma_t2_one_component")
    block_arrays = {key: np.stack([row[key] for row in block_results]) for key in uncertainty_keys}
    standard_errors = {
        key: np.std(value, axis=0, ddof=1) / np.sqrt(int(args.nblocks))
        for key, value in block_arrays.items()
    }
    minimum_bin_count = int(np.min(pooled.count))
    if minimum_bin_count < 1000:
        raise RuntimeError(f"pair-moment sampling is too sparse: minimum bin count={minimum_bin_count}")

    output: dict[str, np.ndarray] = {
        "radial_edges": radial_edges,
        "radial_centers": 0.5 * (radial_edges[:-1] + radial_edges[1:]),
        "mu_edges": mu_edges,
        "anchor_indices": anchors,
        "full_xi_count": full_xi["count"],
        "full_xi_expected_count_uniform": full_xi["expected_count_uniform"],
        "full_xi_real": full_xi["xi"],
        "full_xi_ravg": full_xi["ravg"],
        "fcfc_xi_real": bridge["xi_fcfc"],
        "corrfunc_xi_real_rebinned_fcfc": bridge["xi_corrfunc_rebinned"],
    }
    for key, value in pooled_result.items():
        output[f"pair_{key}"] = value
    for key, value in block_arrays.items():
        output[f"block_{key}"] = value
        output[f"se_{key}"] = standard_errors[key]
    atomic_savez(out_npz, **output)

    audit = {
        "task": "task43_measure_rawbox_pair_moments_v2",
        "status": "pass",
        "scientific_scope": "empirical same-catalog oracle inputs; not an independent theoretical prediction",
        "phase": args.phase,
        "catalog": {
            "path": str(catalog_path(args.phase)),
            "sha256": catalog_sha256,
            "ndata": int(position.shape[0]),
            "boxsize_mpc_h": boxsize,
            "velocity_conversion_kms_per_mpc_h": conversion,
            "velocity_definition": "VELOCITY_KMS / velocity_conversion_kms_per_mpc_h",
            "rsd_bridge_max_abs_mpc_h": rsd_bridge_max,
            "source_status": catalog_metadata.get("status"),
        },
        "full_density": {
            "engine": "Corrfunc.theory.DD periodic ordered auto-pairs",
            "corrfunc_version": str(full_xi["corrfunc_version"]),
            "fcfc_bridge": {key: value for key, value in bridge.items() if not isinstance(value, np.ndarray)},
        },
        "pair_sampling": {
            "method": "uniform anchors; all periodic neighbors within rmax; disjoint anchor blocks",
            "seed": int(args.seed) + 1009 * phase_number,
            "nanchors": int(args.nanchors),
            "nblocks": int(args.nblocks),
            "anchors_per_block": block_size,
            "minimum_pooled_pairs_per_radial_bin": minimum_bin_count,
            "total_oriented_anchor_pairs": int(np.sum(pooled.count)),
            "uncertainty": "standard error across disjoint anchor blocks",
            "weights": "uniform, matching the rawbox FCFC catalog",
        },
        "moments": {
            "separation": "minimum-image real-space pair separation",
            "v12": "mean (u_neighbor-u_anchor) dot rhat; negative means infall",
            "sigma_r2": "central variance of radial pair displacement",
            "sigma_t2": "one transverse component: half the summed transverse second moment",
            "units": {"v12": "Mpc/h", "variance": "(Mpc/h)^2"},
        },
        "bins": {
            "radial_edges_mpc_h": radial_edges.tolist(),
            "absolute_mu_edges": mu_edges.tolist(),
        },
        "output_npz": str(out_npz),
        "output_npz_sha256": sha256_file(out_npz),
        "code_sha256": sha256_file(Path(__file__)),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "elapsed_sec": float(time.perf_counter() - started),
    }
    atomic_write_json(out_json, audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
