#!/usr/bin/env python3
"""Refine ph000 pair moments and direct pair mapping with 8192 anchors."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import eval_legendre

from task43_measure_rawbox_pair_moments_v2 import (
    PairSums,
    accumulate_anchor_pairs,
    fcfc_bridge,
    finalize_mapping_counts,
    finalize_pair_sums,
    validated_catalog,
)
from task43_rsd_common import OUTPUT_ROOT, S_EDGES, atomic_savez, atomic_write_json, sha256_file
from task43_rsd_rawbox_realspace_finalmetric_v2 import set_affinity


PHASE = "ph000"
NANCHORS = 8192
NBLOCKS = 8
SEED = 20260908
THREADS = 8
MOMENT_MU_EDGES = np.linspace(0.0, 1.0, 13)
MAPPING_MU_EDGES = np.linspace(0.0, 1.0, 121)
DENSITY_SOURCE = OUTPUT_ROOT / "rawbox" / "xi_rsd_v2" / "stage03_pair_moments" / (
    "task43_rawbox_pair_moments_ph000_v2.npz"
)
OUT_NPZ = DENSITY_SOURCE.parent / "task43_rawbox_pair_moments_ph000_a8192_v3.npz"
OUT_JSON = OUT_NPZ.with_suffix(".json")
SUMMARY_NPZ = OUTPUT_ROOT / "rawbox" / "summary" / (
    "task43_rsd_rawbox_AbacusSummit_base_c000_ph000_mmin1p4e13_clustering.npz"
)


def main() -> None:
    if OUT_NPZ.exists() or OUT_JSON.exists():
        raise FileExistsError(f"immutable output exists: {OUT_NPZ} / {OUT_JSON}")
    started = time.perf_counter()
    cpus = set_affinity(THREADS)

    density_json = DENSITY_SOURCE.with_suffix(".json")
    density_metadata = json.loads(density_json.read_text(encoding="utf-8"))
    density_digest = sha256_file(DENSITY_SOURCE)
    if density_metadata.get("status") != "pass" or density_metadata.get("output_npz_sha256") != density_digest:
        raise RuntimeError("full-density source is not validated")
    with np.load(DENSITY_SOURCE, allow_pickle=False) as density:
        radial_edges = np.asarray(density["radial_edges"], dtype="f8")
        full_density = {
            "count": np.asarray(density["full_xi_count"], dtype="f8"),
            "expected_count_uniform": np.asarray(density["full_xi_expected_count_uniform"], dtype="f8"),
            "xi": np.asarray(density["full_xi_real"], dtype="f8"),
            "ravg": np.asarray(density["full_xi_ravg"], dtype="f8"),
        }
    bridge = fcfc_bridge(full_density["count"], radial_edges, PHASE)
    if bridge["status"] != "pass":
        raise RuntimeError(f"reused Corrfunc/FCFC bridge failed: {bridge}")

    arrays, catalog_metadata, catalog_digest = validated_catalog(PHASE)
    boxsize = float(arrays["boxsize"])
    position = np.mod(arrays["position_real"], boxsize)
    displacement = arrays["velocity_kms"] / float(arrays["velocity_conversion"])
    expected_rsd_z = np.mod(position[:, 2] + displacement[:, 2], boxsize)
    rsd_error = np.abs(expected_rsd_z - arrays["position_rsd"][:, 2])
    rsd_error = np.minimum(rsd_error, boxsize - rsd_error)
    rsd_bridge_max = float(np.max(rsd_error))
    if rsd_bridge_max > 1.0e-4:
        raise RuntimeError(f"catalog velocity conversion fails RSD bridge: {rsd_bridge_max}")

    rng = np.random.default_rng(SEED)
    anchors = rng.choice(position.shape[0], size=NANCHORS, replace=False)
    block_size = NANCHORS // NBLOCKS
    tree_started = time.perf_counter()
    tree = cKDTree(position, boxsize=boxsize)
    tree_sec = float(time.perf_counter() - tree_started)
    pooled = PairSums.zeros(radial_edges.size - 1, MOMENT_MU_EDGES.size - 1, (32, 120))
    block_results = []
    block_mapping = []
    block_sums = []
    for block_index in range(NBLOCKS):
        block_anchors = anchors[block_index * block_size : (block_index + 1) * block_size]
        block_started = time.perf_counter()
        sums = accumulate_anchor_pairs(
            position,
            displacement,
            block_anchors,
            radial_edges,
            MOMENT_MU_EDGES,
            boxsize=boxsize,
            threads=THREADS,
            tree=tree,
            wrapped_positions=position,
            mapping_edges=S_EDGES,
            mapping_nmu=120,
        )
        moments = finalize_pair_sums(
            sums,
            nanchors=block_size,
            catalog_size=position.shape[0],
            boxsize=boxsize,
            radial_edges=radial_edges,
        )
        mapped = finalize_mapping_counts(
            sums,
            nanchors=block_size,
            catalog_size=position.shape[0],
            boxsize=boxsize,
            radial_edges=S_EDGES,
            nmu=120,
        )
        block_sums.append(sums)
        block_results.append(moments)
        block_mapping.append(mapped)
        pooled.add(sums)
        print(
            json.dumps(
                {
                    "block": block_index,
                    "nanchors": block_size,
                    "moment_pairs": int(np.sum(sums.count)),
                    "mapped_real_pairs": int(np.sum(sums.mapping_real_count)),
                    "mapped_rsd_pairs": int(np.sum(sums.mapping_rsd_count)),
                    "elapsed_sec": float(time.perf_counter() - block_started),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    pooled_result = finalize_pair_sums(
        pooled,
        nanchors=NANCHORS,
        catalog_size=position.shape[0],
        boxsize=boxsize,
        radial_edges=radial_edges,
    )
    pooled_mapping = finalize_mapping_counts(
        pooled,
        nanchors=NANCHORS,
        catalog_size=position.shape[0],
        boxsize=boxsize,
        radial_edges=S_EDGES,
        nmu=120,
    )
    moment_keys = ("xi_anchor_sample", "v12_radial", "sigma_r2_central", "sigma_t2_one_component")
    block_arrays = {key: np.stack([row[key] for row in block_results]) for key in moment_keys}
    standard_errors = {
        key: np.std(value, axis=0, ddof=1) / np.sqrt(NBLOCKS) for key, value in block_arrays.items()
    }
    block_real_smu = np.stack([row["xi_smu_real"] for row in block_mapping])
    block_rsd_smu = np.stack([row["xi_smu_rsd"] for row in block_mapping])
    se_real_smu = np.std(block_real_smu, axis=0, ddof=1) / np.sqrt(NBLOCKS)
    se_rsd_smu = np.std(block_rsd_smu, axis=0, ddof=1) / np.sqrt(NBLOCKS)
    mu_midpoint = 0.5 * (MAPPING_MU_EDGES[:-1] + MAPPING_MU_EDGES[1:])

    def poles(smu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.mean(smu, axis=-1),
            5.0 * np.mean(smu * eval_legendre(2, mu_midpoint), axis=-1),
        )

    anchor_real_xi0, anchor_real_xi2 = poles(pooled_mapping["xi_smu_real"])
    anchor_rsd_xi0, anchor_rsd_xi2 = poles(pooled_mapping["xi_smu_rsd"])
    block_real_xi0, block_real_xi2 = poles(block_real_smu)
    block_rsd_xi0, block_rsd_xi2 = poles(block_rsd_smu)
    with np.load(SUMMARY_NPZ, allow_pickle=False) as summary:
        full_real_xi0 = np.asarray(summary["xi0_real"], dtype="f8")
        full_real_xi2 = np.asarray(summary["xi2_real"], dtype="f8")
        full_rsd_xi0 = np.asarray(summary["xi0_rsd"], dtype="f8")
        full_rsd_xi2 = np.asarray(summary["xi2_rsd"], dtype="f8")

    minimum_bin_count = int(np.min(pooled.count))
    if minimum_bin_count < 50000:
        raise RuntimeError(f"refined pair-moment sampling is too sparse: {minimum_bin_count}")

    output = {
        "radial_edges": radial_edges,
        "radial_centers": 0.5 * (radial_edges[:-1] + radial_edges[1:]),
        "mu_edges": MOMENT_MU_EDGES,
        "anchor_indices": anchors,
        "full_xi_count": full_density["count"],
        "full_xi_expected_count_uniform": full_density["expected_count_uniform"],
        "full_xi_real": full_density["xi"],
        "full_xi_ravg": full_density["ravg"],
        "fcfc_xi_real": bridge["xi_fcfc"],
        "corrfunc_xi_real_rebinned_fcfc": bridge["xi_corrfunc_rebinned"],
        "mapping_s_edges": S_EDGES,
        "mapping_mu_edges": MAPPING_MU_EDGES,
        "mapping_real_count": pooled.mapping_real_count,
        "mapping_rsd_count": pooled.mapping_rsd_count,
        "mapping_expected_count_uniform": pooled_mapping["expected_count_uniform"],
        "mapping_xi_smu_real": pooled_mapping["xi_smu_real"],
        "mapping_xi_smu_rsd": pooled_mapping["xi_smu_rsd"],
        "mapping_block_xi_smu_real": block_real_smu,
        "mapping_block_xi_smu_rsd": block_rsd_smu,
        "mapping_se_xi_smu_real": se_real_smu,
        "mapping_se_xi_smu_rsd": se_rsd_smu,
        "mapping_xi0_real": anchor_real_xi0,
        "mapping_xi2_real": anchor_real_xi2,
        "mapping_xi0_rsd": anchor_rsd_xi0,
        "mapping_xi2_rsd": anchor_rsd_xi2,
        "mapping_block_xi0_real": block_real_xi0,
        "mapping_block_xi2_real": block_real_xi2,
        "mapping_block_xi0_rsd": block_rsd_xi0,
        "mapping_block_xi2_rsd": block_rsd_xi2,
        "full_fcfc_xi0_real": full_real_xi0,
        "full_fcfc_xi2_real": full_real_xi2,
        "full_fcfc_xi0_rsd": full_rsd_xi0,
        "full_fcfc_xi2_rsd": full_rsd_xi2,
    }
    for key, value in pooled_result.items():
        output[f"pair_{key}"] = value
    for key, value in block_arrays.items():
        output[f"block_{key}"] = value
        output[f"se_{key}"] = standard_errors[key]
    atomic_savez(OUT_NPZ, **output)

    audit = {
        "task": "task43_refine_rawbox_pair_moments_ph000_v3",
        "status": "pass",
        "scientific_scope": "refined empirical oracle inputs plus exact same-anchor pair mapping",
        "phase": PHASE,
        "catalog": {
            "path": str(catalog_metadata.get("output_path", "rawbox catalog sidecar")),
            "sha256": catalog_digest,
            "ndata": int(position.shape[0]),
            "boxsize_mpc_h": boxsize,
            "rsd_bridge_max_abs_mpc_h": rsd_bridge_max,
        },
        "full_density_reuse": {
            "path": str(DENSITY_SOURCE),
            "sha256": density_digest,
            "fcfc_bridge_max_abs": float(bridge["max_abs_corrfunc_minus_fcfc"]),
            "reason": "avoid repeating validated full-catalog Corrfunc on the login node",
        },
        "pair_sampling": {
            "seed": SEED,
            "nanchors": NANCHORS,
            "nblocks": NBLOCKS,
            "anchors_per_block": block_size,
            "minimum_pooled_pairs_per_radial_bin": minimum_bin_count,
            "total_oriented_anchor_pairs": int(np.sum(pooled.count)),
            "tree_build_sec": tree_sec,
            "uncertainty": "standard error across disjoint anchor blocks",
        },
        "direct_mapping": {
            "definition": "same real-space anchor pairs mapped with pair delta-u-z and periodic minimum image",
            "s_edges_mpc_h": S_EDGES.tolist(),
            "nmu": 120,
            "real_xi0_rms_vs_full_fcfc": float(np.sqrt(np.mean((anchor_real_xi0 - full_real_xi0) ** 2))),
            "rsd_xi0_rms_vs_full_fcfc": float(np.sqrt(np.mean((anchor_rsd_xi0 - full_rsd_xi0) ** 2))),
            "real_xi2_rms_vs_full_fcfc": float(np.sqrt(np.mean((anchor_real_xi2 - full_real_xi2) ** 2))),
            "rsd_xi2_rms_vs_full_fcfc": float(np.sqrt(np.mean((anchor_rsd_xi2 - full_rsd_xi2) ** 2))),
        },
        "thread_policy": {
            "maximum_cores": 8,
            "selected_affinity": cpus,
            "blas_threads_expected": 1,
            "tree_query_workers": THREADS,
        },
        "output_npz": str(OUT_NPZ),
        "output_npz_sha256": sha256_file(OUT_NPZ),
        "code_sha256": sha256_file(Path(__file__)),
        "accumulator_code_sha256": sha256_file(
            Path(__file__).with_name("task43_measure_rawbox_pair_moments_v2.py")
        ),
        "elapsed_sec": float(time.perf_counter() - started),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    atomic_write_json(OUT_JSON, audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
