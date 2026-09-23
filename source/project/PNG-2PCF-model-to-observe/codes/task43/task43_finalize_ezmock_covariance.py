#!/usr/bin/env python3
"""Wait for all x1000 rows, then run the Task43 final covariance audits.

This finalizer is deliberately separate from the two production workers.  It
does not generate mocks or measurements.  It waits for the exact manifest
products and for every production process to exit, then runs the existing
environment, abundance and covariance audits.  The final task-document update
remains an explicit post-production step and is never claimed prematurely.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes/task43"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task43_ezmock_covariance_common import (  # noqa: E402
    ABACUS_PK_PAYLOAD,
    ABUNDANCE_AUDIT,
    COMMON_RANDOM,
    COMMON_RANDOM_HDF5,
    COMMON_RANDOM_HDF5_META,
    COMMON_RR,
    COMMON_RR_META,
    DEFAULT_NTRACER,
    JOINT_DIMENSION,
    MANIFEST,
    MANIFEST_AUDIT,
    N_PK_BINS,
    NREAL,
    OUTPUT_ROOT,
    PLOT_DIR,
    S_EDGES,
    SUMMARY_DIR,
    read_jsonl,
    sha256,
    write_json,
)

FINAL_STEM = f"task43_ezmock_lightcone_covariance_final_x{NREAL}_joint{JOINT_DIMENSION}"
FINAL_NPZ = SUMMARY_DIR / f"{FINAL_STEM}.npz"
FINAL_JSON = SUMMARY_DIR / f"{FINAL_STEM}.json"
FINAL_PDF = PLOT_DIR / f"{FINAL_STEM}.pdf"
ENVIRONMENT_AUDIT = SUMMARY_DIR / "task43_ezmock_covariance_environment.json"
FINALIZER_AUDIT = SUMMARY_DIR / "task43_ezmock_covariance_finalizer_audit.json"
COMMON_RANDOM_AUDIT = SUMMARY_DIR / "task43_ezmock_common50_random_independent_audit.json"
VALIDATION_AUDIT = SUMMARY_DIR / "task43_validation_fixedamp_x10_old25_vs_common50_ezmock_selection_xi_ab.json"
PILOT_AUDIT = SUMMARY_DIR / "task43_ezmock_lightcone_covariance_pilot_x50_joint45.json"
PERSISTENT_LOG_DIR = OUTPUT_ROOT / "logs/persistent"
LIGHTCONE_SUPERVISOR_LOG = PERSISTENT_LOG_DIR / "lightcone_xi_supervisor.log"
PK_SUPERVISOR_LOG = PERSISTENT_LOG_DIR / "pk_supervisor.log"

PRODUCTION_PATTERNS = (
    "task43_run_ezmock_covariance_login.py",
    "task43_build_ezmock_covariance_lightcone.py",
    "task43_measure_ezmock_covariance_xi_fcfc.py",
    "task43_measure_ezmock_covariance_pk_jaxpower.py",
    "FCFC_2PT_HDF5",
    "run_task43_ezmock_covariance_login_persistent.sh lightcone_xi",
    "run_task43_ezmock_covariance_login_persistent.sh pk",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true", help="Poll until all rows and workers are complete.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def process_lines() -> list[str]:
    text = subprocess.run(
        ["ps", "-u", str(os.getuid()), "-o", "pid=,args="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    own_pid = os.getpid()
    active = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split(maxsplit=1)
        if fields and fields[0].isdigit() and int(fields[0]) == own_pid:
            continue
        if any(pattern in stripped for pattern in PRODUCTION_PATTERNS):
            active.append(stripped)
    return active


def readiness() -> dict[str, Any]:
    rows = read_jsonl(MANIFEST)
    if len(rows) != NREAL:
        return {"ready": False, "manifest_rows": len(rows), "reason": "manifest_row_count"}
    expected_lightcone = {Path(str(row["halo_catalog_path"])) for row in rows}
    expected_xi = {Path(str(row["xi_path"])) for row in rows}
    expected_pk = {Path(str(row["pk_path"])) for row in rows}
    expected_support = {
        path
        for row in rows
        for path in (
            Path(str(row["halo_metadata_path"])),
            Path(str(row["ezmock_config_path"])),
            Path(str(row["ezmock_log_path"])),
            Path(str(row["xi_path"])).with_suffix(".json"),
            Path(str(row["pk_path"])).with_suffix(".json"),
        )
    }
    missing_lightcone = [str(path) for path in expected_lightcone if not path.is_file()]
    missing_xi = [str(path) for path in expected_xi if not path.is_file()]
    missing_pk = [str(path) for path in expected_pk if not path.is_file()]
    missing_support = [str(path) for path in expected_support if not path.is_file()]
    actual_lightcone = set((OUTPUT_ROOT / "lightcone_catalogs").glob("lightcone_ezmock_ph*_z0p6_0p8.npz"))
    actual_xi = set((OUTPUT_ROOT / "xi_fcfc").glob("xi0_ezmock_ph*_common50_s50_350_ds10_fcfc.npz"))
    actual_pk = set((OUTPUT_ROOT / "pk_jaxpower").glob("pk0_ezmock_ph*_common50_mesh256_kmax0p300_dk0p002.npz"))
    active = process_lines()
    counts = {
        "lightcone": len(actual_lightcone),
        "xi": len(actual_xi),
        "pk": len(actual_pk),
    }
    exact_sets = {
        "lightcone": actual_lightcone == expected_lightcone,
        "xi": actual_xi == expected_xi,
        "pk": actual_pk == expected_pk,
    }
    ready = not any((missing_lightcone, missing_xi, missing_pk, missing_support, active)) and all(exact_sets.values())
    return {
        "ready": bool(ready),
        "manifest_rows": len(rows),
        "counts": counts,
        "exact_manifest_file_sets": exact_sets,
        "missing_counts": {
            "lightcone": len(missing_lightcone),
            "xi": len(missing_xi),
            "pk": len(missing_pk),
            "support": len(missing_support),
        },
        "active_production_processes": active,
    }


def run_checked(script: str, *arguments: str) -> None:
    command = [sys.executable, "-u", str(CODE_DIR / script), *arguments]
    print(f"[finalizer] run {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def cleanup_designated_temporary_files() -> list[str]:
    """Remove only production-owned intermediates after every worker exits."""
    active = process_lines()
    if active:
        raise RuntimeError(f"refusing temporary cleanup with active production processes: {active}")
    removed: list[str] = []
    for root in (
        OUTPUT_ROOT / "tmp/fcfc_fifo",
        OUTPUT_ROOT / "tmp/rawbox_catalogs",
        OUTPUT_ROOT / "tmp/row_locks",
    ):
        root.mkdir(parents=True, exist_ok=True)
        for child in root.iterdir():
            resolved = child.resolve(strict=False)
            if root.resolve() not in resolved.parents:
                raise RuntimeError(f"temporary cleanup escaped designated root: {child}")
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
            removed.append(str(child))
    return removed


def validate_final_products(ready: dict[str, Any], removed_temporary: list[str]) -> dict[str, Any]:
    upstream_paths = (
        COMMON_RANDOM.with_suffix(".json"), COMMON_RANDOM_HDF5_META, COMMON_RR_META,
        COMMON_RANDOM_AUDIT, VALIDATION_AUDIT, PILOT_AUDIT, MANIFEST_AUDIT,
    )
    for path in (
        FINAL_NPZ, FINAL_JSON, FINAL_PDF, ENVIRONMENT_AUDIT, ABUNDANCE_AUDIT,
        LIGHTCONE_SUPERVISOR_LOG, PK_SUPERVISOR_LOG, *upstream_paths,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing/empty final product: {path}")
    summary = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
    abundance = json.loads(ABUNDANCE_AUDIT.read_text(encoding="utf-8"))
    environment = json.loads(ENVIRONMENT_AUDIT.read_text(encoding="utf-8"))
    common_random_meta = json.loads(COMMON_RANDOM.with_suffix(".json").read_text(encoding="utf-8"))
    common_hdf_meta = json.loads(COMMON_RANDOM_HDF5_META.read_text(encoding="utf-8"))
    common_rr_meta = json.loads(COMMON_RR_META.read_text(encoding="utf-8"))
    common_random_audit = json.loads(COMMON_RANDOM_AUDIT.read_text(encoding="utf-8"))
    validation_audit = json.loads(VALIDATION_AUDIT.read_text(encoding="utf-8"))
    pilot_audit = json.loads(PILOT_AUDIT.read_text(encoding="utf-8"))
    manifest_audit = json.loads(MANIFEST_AUDIT.read_text(encoding="utf-8"))
    environment_hashes = environment.get("sha256", {})
    environment_hashes_current = bool(environment_hashes) and all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in environment_hashes.items()
    )
    supervisor_text = {
        "lightcone_xi": LIGHTCONE_SUPERVISOR_LOG.read_text(encoding="utf-8", errors="replace"),
        "pk": PK_SUPERVISOR_LOG.read_text(encoding="utf-8", errors="replace"),
    }
    supervisor_diagnostics = {
        stage: {
            "launch_count": text.count(f"launch stage={stage}"),
            "retry_count": text.count("retry in"),
            "completed": f"complete stage={stage} rows=0:1000" in text,
            "restart_limit_exhausted": "restart limit exhausted" in text,
        }
        for stage, text in supervisor_text.items()
    }
    expected_shapes = {
        "xi_stack": [1000, 30],
        "pk_stack": [1000, 15],
        "joint_stack": [1000, 45],
        "covariance": [45, 45],
        "xi_covariance": [30, 30],
        "pk_covariance": [15, 15],
        "cross": [30, 15],
    }
    with np.load(ABACUS_PK_PAYLOAD, allow_pickle=False) as target_pk:
        target_k = np.asarray(target_pk["k_obs"], dtype="f8")
        target_k_edges = np.asarray(target_pk["k_edges"], dtype="f8")
    with np.load(FINAL_NPZ, allow_pickle=False) as data:
        required_npz_shapes = {
            "s": [30], "s_edges": [31], "k": [15], "k_edges": [15, 2],
            "seeds": [1000], "xi_stack": [1000, 30], "pk_stack": [1000, 15],
            "shotnoise_stack": [1000, 15], "joint_stack": [1000, 45],
            "xi_mean": [30], "xi_std": [30], "pk_mean": [15], "pk_std": [15],
            "shotnoise_mean": [15], "shotnoise_std": [15],
            "covariance": [45, 45], "correlation": [45, 45],
            "xi_covariance": [30, 30], "pk_covariance": [15, 15],
            "xi_pk_cross_covariance": [30, 15],
            "checkpoints": [5], "convergence_covariance": [5, 45, 45],
            "convergence_relative_frobenius": [5], "ndata": [1000],
            "lightcone_runtime_sec": [1000], "xi_runtime_sec": [1000],
            "pk_runtime_sec": [1000], "lightcone_z_min": [1000],
            "lightcone_z_max": [1000], "lightcone_nbar_volume_weighted": [1000],
            "zeff_random_auto": [], "zeff_data_auto": [],
            "zeff_data_random_cross": [], "hartlap_factor": [],
        }
        missing_npz_keys = sorted(set(required_npz_shapes).difference(data.files))
        if missing_npz_keys:
            raise RuntimeError(f"final NPZ missing required arrays: {missing_npz_keys}")
        npz_shapes = {
            key: list(np.asarray(data[key]).shape)
            for key in required_npz_shapes
        }
        covariance = np.asarray(data["covariance"], dtype="f8")
        correlation = np.asarray(data["correlation"], dtype="f8")
        xi_stack = np.asarray(data["xi_stack"], dtype="f8")
        pk_stack = np.asarray(data["pk_stack"], dtype="f8")
        shotnoise_stack = np.asarray(data["shotnoise_stack"], dtype="f8")
        joint_stack = np.asarray(data["joint_stack"], dtype="f8")
        xi_covariance = np.asarray(data["xi_covariance"], dtype="f8")
        pk_covariance = np.asarray(data["pk_covariance"], dtype="f8")
        cross_covariance = np.asarray(data["xi_pk_cross_covariance"], dtype="f8")
        seeds = np.asarray(data["seeds"], dtype="i8")
        checkpoints = np.asarray(data["checkpoints"], dtype="i8")
        convergence_covariance = np.asarray(data["convergence_covariance"], dtype="f8")
        convergence_relative = np.asarray(data["convergence_relative_frobenius"], dtype="f8")
        expected_checkpoints = np.asarray([50, 100, 250, 500, 1000], dtype="i8")
        recomputed_covariance = np.cov(joint_stack, rowvar=False, ddof=1)
        recomputed_correlation = recomputed_covariance / np.outer(
            np.sqrt(np.diag(recomputed_covariance)), np.sqrt(np.diag(recomputed_covariance))
        )
        recomputed_convergence = np.stack([
            np.cov(joint_stack[:count], rowvar=False, ddof=1)
            for count in expected_checkpoints
        ])
        covariance_norm = np.linalg.norm(recomputed_covariance)
        recomputed_convergence_relative = np.asarray([
            np.linalg.norm(item - recomputed_covariance) / covariance_norm
            for item in recomputed_convergence
        ], dtype="f8")
        metadata_arrays = tuple(
            np.asarray(data[key]) for key in (
                "ndata", "lightcone_runtime_sec", "xi_runtime_sec", "pk_runtime_sec",
                "lightcone_z_min", "lightcone_z_max", "lightcone_nbar_volume_weighted",
                "zeff_random_auto", "zeff_data_auto", "zeff_data_random_cross",
            )
        )
        npz_gates = {
            "shapes": npz_shapes == required_npz_shapes,
            "measurement_grids": bool(
                np.array_equal(np.asarray(data["s_edges"], dtype="f8"), S_EDGES)
                and np.array_equal(
                    np.asarray(data["s"], dtype="f8"), 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
                )
                and np.array_equal(np.asarray(data["k"], dtype="f8"), target_k)
                and np.array_equal(np.asarray(data["k_edges"], dtype="f8"), target_k_edges)
            ),
            "covariance_finite": bool(np.all(np.isfinite(covariance))),
            "covariance_symmetric": bool(np.max(np.abs(covariance - covariance.T)) < 1.0e-12),
            "correlation_finite": bool(np.all(np.isfinite(correlation))),
            "correlation_symmetric": bool(np.max(np.abs(correlation - correlation.T)) < 1.0e-12),
            "correlation_unit_diagonal": bool(np.max(np.abs(np.diag(correlation) - 1.0)) < 1.0e-12),
            "stacks_finite": bool(
                np.all(np.isfinite(xi_stack))
                and np.all(np.isfinite(pk_stack))
                and np.all(np.isfinite(shotnoise_stack))
                and np.all(np.isfinite(joint_stack))
            ),
            "joint_stack_consistent": bool(
                np.array_equal(joint_stack[:, :30], xi_stack)
                and np.array_equal(joint_stack[:, 30:], pk_stack)
            ),
            "covariance_blocks_consistent": bool(
                np.array_equal(covariance[:30, :30], xi_covariance)
                and np.array_equal(covariance[30:, 30:], pk_covariance)
                and np.array_equal(covariance[:30, 30:], cross_covariance)
                and np.array_equal(covariance[30:, :30], cross_covariance.T)
            ),
            "moments_consistent": bool(
                np.array_equal(np.asarray(data["xi_mean"]), np.mean(xi_stack, axis=0))
                and np.array_equal(np.asarray(data["xi_std"]), np.std(xi_stack, axis=0, ddof=1))
                and np.array_equal(np.asarray(data["pk_mean"]), np.mean(pk_stack, axis=0))
                and np.array_equal(np.asarray(data["pk_std"]), np.std(pk_stack, axis=0, ddof=1))
                and np.array_equal(
                    np.asarray(data["shotnoise_mean"]), np.mean(shotnoise_stack, axis=0)
                )
                and np.array_equal(
                    np.asarray(data["shotnoise_std"]), np.std(shotnoise_stack, axis=0, ddof=1)
                )
            ),
            "covariance_recomputed": bool(
                np.allclose(covariance, recomputed_covariance, rtol=1.0e-12, atol=1.0e-18)
            ),
            "correlation_recomputed": bool(
                np.allclose(correlation, recomputed_correlation, rtol=1.0e-12, atol=1.0e-14)
            ),
            "convergence_checkpoints_exact": bool(np.array_equal(checkpoints, expected_checkpoints)),
            "convergence_covariances_recomputed": bool(np.allclose(
                convergence_covariance, recomputed_convergence, rtol=1.0e-12, atol=1.0e-18
            )),
            "convergence_relative_recomputed": bool(
                np.allclose(
                    convergence_relative, recomputed_convergence_relative,
                    rtol=1.0e-12, atol=1.0e-14,
                )
            ),
            "metadata_finite": bool(all(np.all(np.isfinite(item)) for item in metadata_arrays)),
            "metadata_physical": bool(
                np.all(np.asarray(data["ndata"], dtype="i8") > 0)
                and np.all(np.asarray(data["lightcone_runtime_sec"], dtype="f8") > 0.0)
                and np.all(np.asarray(data["xi_runtime_sec"], dtype="f8") > 0.0)
                and np.all(np.asarray(data["pk_runtime_sec"], dtype="f8") > 0.0)
                and np.all(np.asarray(data["lightcone_nbar_volume_weighted"], dtype="f8") > 0.0)
            ),
            "seeds_unique": bool(np.unique(seeds).size == 1000),
            "seed_range": bool(seeds.min() == 432001 and seeds.max() == 433000),
            "seeds_ordered_exactly": bool(np.array_equal(seeds, np.arange(432001, 433001, dtype="i8"))),
            "hartlap": bool(abs(float(np.asarray(data["hartlap_factor"]).item()) - 953.0 / 999.0) < 1.0e-15),
        }
        if not all(npz_gates.values()):
            raise RuntimeError(f"final NPZ gates failed: shapes={npz_shapes}, gates={npz_gates}")
    gates = {
        "summary_pass": summary.get("status") == "pass",
        "classification": summary.get("classification") == "final_x1000_covariance",
        "dimensions": (
            summary.get("nreal"), summary.get("n_xi"), summary.get("n_pk"), summary.get("joint_dimension")
        ) == (1000, 30, 15, 45),
        "shapes": summary.get("shapes") == expected_shapes,
        "hartlap": abs(float(summary.get("hartlap_factor", -1.0)) - 953.0 / 999.0) < 1.0e-15,
        "scientific_gate": summary.get("scientific_gate") is True,
        "no_checkpoint_waiver": summary.get("active_checkpoint_waiver") is False,
        "per_row": summary.get("per_row_all_gates_pass") is True,
        "no_png": summary.get("no_png_gate") is True,
        "no_fifo": summary.get("no_stale_fifo_gate") is True,
        "no_active_production": summary.get("no_active_production_process_gate") is True,
        "seed_unique": summary.get("seed_unique") is True,
        "fix_amplitude_false": summary.get("fix_amplitude") is False,
        "smax_350": float(summary.get("smax", -1.0)) == 350.0,
        "redshift_diagnostics": all(
            key in summary.get("redshift_diagnostics", {}).get("frozen_fkp_effective_redshift", {})
            for key in ("zeff_random_auto", "zeff_data_auto", "zeff_data_random_cross")
        ),
        "convergence_checkpoints": summary.get("convergence_checkpoints") == [50, 100, 250, 500, 1000],
        "convergence_values": (
            len(summary.get("convergence_relative_frobenius_to_current", [])) == 5
            and all(
                np.isfinite(float(value)) and float(value) >= 0.0
                for value in summary.get("convergence_relative_frobenius_to_current", [])
            )
            and float(summary.get("convergence_relative_frobenius_to_current", [1.0])[-1]) == 0.0
        ),
        "runtime_diagnostics": all(
            float(summary.get("runtime", {}).get(key, 0.0)) > 0.0
            for key in ("lightcone_sec_mean", "xi_sec_mean", "pk_sec_mean")
        ),
        "storage_diagnostics": all(
            int(summary.get("storage_bytes", {}).get(key, 0)) > 0
            for key in ("lightcone_catalogs", "xi_fcfc", "pk_jaxpower", "production_output_root_total")
        ),
        "covariance_diagnostics": (
            summary.get("covariance_diagnostics", {}).get("finite") is True
            and float(summary.get("covariance_diagnostics", {}).get("symmetry_max_abs", 1.0)) < 1.0e-12
            and summary.get("covariance_psd_numerical_gate") is True
        ),
        "correlation_diagnostics": (
            summary.get("correlation_diagnostics", {}).get("finite") is True
            and float(summary.get("correlation_diagnostics", {}).get("symmetry_max_abs", 1.0)) < 1.0e-12
            and float(summary.get("correlation_diagnostics", {}).get("condition_number", -1.0)) > 0.0
        ),
        "xi_estimator_diagnostics": (
            summary.get("xi_estimator_diagnostics", {}).get("status") == "pass"
            and int(summary.get("xi_estimator_diagnostics", {}).get("audited_rows", -1)) == 1000
            and summary.get("xi_estimator_diagnostics", {}).get("all_rr_arrays_bitwise_equal_common_rr") is True
            and float(summary.get("xi_estimator_diagnostics", {}).get("max_delta_over_bound", 2.0)) <= 1.0
        ),
        "lightcone_catalog_diagnostics": (
            summary.get("lightcone_catalog_diagnostics", {}).get("status") == "pass"
            and int(summary.get("lightcone_catalog_diagnostics", {}).get("audited_rows", -1)) == 1000
            and summary.get("lightcone_catalog_diagnostics", {}).get("selection_redshift_open_interval") == [0.6, 0.8]
            and summary.get("lightcone_catalog_diagnostics", {}).get("stored_redshift_closed_interval_float32")
            == [float(np.float32(0.6)), float(np.float32(0.8))]
            and float(
                summary.get("lightcone_catalog_diagnostics", {})
                .get("coord_check_radius_over_chi_minus_one_max_abs", {})
                .get("max", 1.0)
            ) < 1.0e-6
        ),
        "pk_measurement_diagnostics": (
            summary.get("pk_measurement_diagnostics", {}).get("status") == "pass"
            and int(summary.get("pk_measurement_diagnostics", {}).get("audited_rows", -1)) == 1000
            and int(summary.get("pk_measurement_diagnostics", {}).get("fine_spectrum_bins", -1)) == 150
            and int(summary.get("pk_measurement_diagnostics", {}).get("selected_mcmc_bins", -1)) == 15
            and summary.get("pk_measurement_diagnostics", {}).get("all_shotnoise_finite") is True
            and summary.get("pk_measurement_diagnostics", {}).get("all_target_selections_exact") is True
            and summary.get("pk_measurement_diagnostics", {}).get("all_window_policies_exact") is True
        ),
        "native_fcfc_diagnostics": (
            summary.get("native_fcfc_diagnostics", {}).get("status") == "pass"
            and int(summary.get("native_fcfc_diagnostics", {}).get("audited_rows", -1)) == 1000
            and summary.get("native_fcfc_diagnostics", {}).get("all_native_files_nonempty") is True
            and summary.get("native_fcfc_diagnostics", {}).get("all_native_grids_exact") is True
            and summary.get("native_fcfc_diagnostics", {}).get("all_native_arrays_equal_npz") is True
            and summary.get("native_fcfc_diagnostics", {}).get("all_logs_confirm_shared_rr") is True
        ),
        "production_abundance_diagnostics": (
            summary.get("production_abundance_diagnostics", {}).get("status") == "pass"
            and int(
                summary.get("production_abundance_diagnostics", {}).get("audited_rows", -1)
            ) == 1000
            and int(
                summary.get("production_abundance_diagnostics", {})
                .get("target_abacus_mean_count", -1)
            ) == 331_545
            and float(
                summary.get("production_abundance_diagnostics", {})
                .get("absolute_fractional_gate", -1.0)
            ) == 0.02
            and abs(float(
                summary.get("production_abundance_diagnostics", {}).get("fractional_offset", 1.0)
            )) < 0.02
            and float(
                summary.get("production_abundance_diagnostics", {}).get("std_ddof1", 0.0)
            ) > 0.0
            and np.isfinite(float(
                summary.get("production_abundance_diagnostics", {})
                .get("index_count_correlation", np.nan)
            ))
            and int(
                summary.get("production_abundance_diagnostics", {}).get("edge_block_size", -1)
            ) == 100
        ),
        "abundance_audit": (
            abundance.get("status") == "pass"
            and abundance.get("fix_amplitude") is False
            and abundance.get("attach_particle") is True
            and int(abundance.get("ntracer_frozen", -1)) == DEFAULT_NTRACER
            and int(abundance.get("target_abacus_mean_count", -1)) == 331545
            and abundance.get("indices") == list(range(10))
            and abundance.get("seeds") == list(range(432001, 432011))
            and len(abundance.get("selected_counts", [])) == 10
            and abs(float(abundance.get("fractional_offset", 1.0)))
            < float(abundance.get("absolute_fractional_gate", 0.0))
            and float(abundance.get("absolute_fractional_gate", -1.0)) == 0.02
        ),
        "environment_audit": (
            environment.get("status") == "done"
            and environment.get("fcfc_hdf5_enabled") is True
            and environment_hashes_current
            and all(
                key in environment.get("packages", {})
                for key in ("numpy", "jax", "jaxlib", "h5py", "cosmoprimo", "desilike")
            )
            and bool(environment.get("python"))
            and bool(environment.get("platform"))
        ),
        "common_random_metadata": (
            common_random_meta.get("status") == "done"
            and int(common_random_meta.get("nrandom", -1)) == 16_577_250
            and int(common_random_meta.get("seed", -1)) == 43_250_001
            and float(common_random_meta.get("p0", -1.0)) == 10_000.0
            and float(common_random_meta.get("zmin", -1.0)) == 0.6
            and float(common_random_meta.get("zmax", -1.0)) == 0.8
            and common_random_meta.get("positive_octant_gate") is True
            and "constant comoving nbar" in common_random_meta.get("selection_policy", "")
            and common_random_meta.get("sha256") == sha256(COMMON_RANDOM)
            and float(common_random_meta.get("weight_min", 0.0)) > 0.0
            and float(common_random_meta.get("weight_max", 0.0))
            >= float(common_random_meta.get("weight_min", 1.0))
            and float(common_random_meta.get("weight_sum", 0.0)) > 0.0
            and float(common_random_meta.get("weight2_sum", 0.0)) > 0.0
        ),
        "common_hdf_metadata": (
            common_hdf_meta.get("status") == "done"
            and int(common_hdf_meta.get("nrandom", -1)) == 16_577_250
            and common_hdf_meta.get("source_npz_sha256") == sha256(COMMON_RANDOM)
        ),
        "common_rr_metadata": (
            common_rr_meta.get("status") == "done"
            and int(common_rr_meta.get("nrandom", -1)) == 16_577_250
            and common_rr_meta.get("rr_sha256") == sha256(COMMON_RR)
            and int(common_rr_meta.get("threads", -1)) == 8
            and common_rr_meta.get("cpu_affinity") == list(range(6, 14))
            and common_rr_meta.get("s_edges") == [float(value) for value in S_EDGES]
            and common_rr_meta.get("common_random_npz_sha256") == sha256(COMMON_RANDOM)
            and common_rr_meta.get("common_random_hdf5_sha256") == sha256(COMMON_RANDOM_HDF5)
            and float(common_rr_meta.get("runtime_sec", 0.0)) > 0.0
        ),
        "common_random_independent_audit": (
            common_random_audit.get("status") == "pass"
            and all(common_random_audit.get("gates", {}).values())
        ),
        "common50_measurement_validation": (
            validation_audit.get("status") == "pass"
            and float(validation_audit.get("rms", 1.0)) < 0.06
            and float(validation_audit.get("max_abs", 1.0)) < 0.15
            and validation_audit.get("s_edges") == [float(value) for value in range(50, 351, 10)]
        ),
        "x50_pilot": (
            pilot_audit.get("status") == "pass"
            and (pilot_audit.get("nreal"), pilot_audit.get("n_xi"), pilot_audit.get("n_pk"), pilot_audit.get("joint_dimension"))
            == (50, 30, 15, 45)
            and pilot_audit.get("per_row_all_gates_pass") is True
        ),
        "manifest_audit": (
            manifest_audit.get("status") == "done"
            and int(manifest_audit.get("nreal", -1)) == 1000
            and manifest_audit.get("fix_amplitude") is False
            and manifest_audit.get("attach_particle") is True
            and manifest_audit.get("seed_unique") is True
            and manifest_audit.get("validation_seed_overlap") == []
            and int(manifest_audit.get("seed_min", -1)) == 432001
            and int(manifest_audit.get("seed_max", -1)) == 433000
            and int(manifest_audit.get("ntracer", -1)) == DEFAULT_NTRACER
            and (manifest_audit.get("n_xi_bins"), manifest_audit.get("n_pk_bins"), manifest_audit.get("joint_dimension"))
            == (30, 15, 45)
            and manifest_audit.get("s_edges") == [float(value) for value in S_EDGES]
        ),
        "persistent_supervisors_complete": (
            supervisor_diagnostics["lightcone_xi"]["completed"]
            and supervisor_diagnostics["pk"]["completed"]
            and not supervisor_diagnostics["lightcone_xi"]["restart_limit_exhausted"]
            and not supervisor_diagnostics["pk"]["restart_limit_exhausted"]
            and "cpu=6-15 threads=10 rows=0:1000" in supervisor_text["lightcone_xi"]
            and "cpu=16-17 threads=2 rows=0:1000" in supervisor_text["pk"]
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"final summary gates failed: {gates}")
    png_files = [str(path) for root in (OUTPUT_ROOT, PLOT_DIR) for path in root.rglob("*.png")]
    fifo_files = [
        str(path)
        for path in OUTPUT_ROOT.rglob("*")
        if path.exists() and stat.S_ISFIFO(path.stat().st_mode)
    ]
    if png_files or fifo_files or process_lines():
        raise RuntimeError(
            f"post-summary cleanliness gate failed: png={png_files}, fifo={fifo_files}, active={process_lines()}"
        )
    return {
        "task": "task43_finalize_ezmock_covariance",
        "status": "pass_products_pending_task_doc",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "all x1000 science products independently verified; agent/task.md update remains",
        "readiness": ready,
        "removed_designated_temporary_entries": removed_temporary,
        "gates": gates,
        "npz_gates": npz_gates,
        "npz_shapes": npz_shapes,
        "persistent_supervisor_diagnostics": supervisor_diagnostics,
        "hartlap_factor": 953.0 / 999.0,
        "hartlap_expression": "(1000-45-2)/(1000-1) = 953/999",
        "outputs": {
            "npz": str(FINAL_NPZ),
            "json": str(FINAL_JSON),
            "pdf": str(FINAL_PDF),
            "environment": str(ENVIRONMENT_AUDIT),
            "abundance": str(ABUNDANCE_AUDIT),
        },
        "sha256": {
            str(path): sha256(path)
            for path in (
                FINAL_NPZ,
                FINAL_JSON,
                FINAL_PDF,
                ENVIRONMENT_AUDIT,
                ABUNDANCE_AUDIT,
                MANIFEST,
                COMMON_RANDOM,
                COMMON_RANDOM_HDF5,
                COMMON_RR,
                LIGHTCONE_SUPERVISOR_LOG,
                PK_SUPERVISOR_LOG,
                *upstream_paths,
            )
        },
        "next_required_step": "update agent/task.md with authoritative paths, timing and limitations, then final audit",
    }


def main() -> None:
    args = parse_args()
    poll_seconds = int(args.poll_seconds)
    if poll_seconds < 10 or poll_seconds > 600:
        raise ValueError("--poll-seconds must be in [10,600]")
    last_report = 0.0
    while True:
        ready = readiness()
        if ready["ready"]:
            break
        if not args.wait:
            print(json.dumps(ready, indent=2, sort_keys=True))
            raise SystemExit(4)
        now = time.monotonic()
        if now - last_report >= 600.0 or last_report == 0.0:
            print(f"[finalizer wait] {json.dumps(ready, sort_keys=True)}", flush=True)
            last_report = now
        time.sleep(poll_seconds)
    print(f"[finalizer ready] {json.dumps(ready, sort_keys=True)}", flush=True)
    if FINALIZER_AUDIT.is_file() and not args.overwrite:
        existing = json.loads(FINALIZER_AUDIT.read_text(encoding="utf-8"))
        if existing.get("status") == "pass_products_pending_task_doc":
            print(f"[finalizer skip] {FINALIZER_AUDIT}")
            return
    removed_temporary = cleanup_designated_temporary_files()
    print(f"[finalizer cleanup] removed={removed_temporary}", flush=True)
    run_checked("task43_audit_ezmock_covariance_environment.py")
    run_checked("task43_audit_ezmock_covariance_abundance.py")
    run_checked("task43_summarize_ezmock_covariance.py", "--nreal", "1000", "--overwrite")
    payload = validate_final_products(ready, removed_temporary)
    write_json(FINALIZER_AUDIT, payload)
    print(f"[finalizer pass] {FINALIZER_AUDIT}", flush=True)


if __name__ == "__main__":
    main()
