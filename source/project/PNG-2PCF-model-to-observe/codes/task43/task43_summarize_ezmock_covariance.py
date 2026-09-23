#!/usr/bin/env python3
"""Build pilot/final EZmock joint covariance products and an independent audit."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task43_ezmock_covariance_common import (
    ABACUS_PK_PAYLOAD,
    ABACUS_FKP_SUMMARY,
    ABACUS_XI_SUMMARY,
    ATTACH_PARTICLE,
    BOX_SIZE,
    COMMON_RANDOM,
    COMMON_RANDOM_HDF5,
    COMMON_RANDOM_SIZE,
    COMMON_RR,
    DEFAULT_NTRACER,
    FIX_AMPLITUDE,
    JOINT_DIMENSION,
    MANIFEST,
    N_PK_BINS,
    NREAL,
    OUTPUT_ROOT,
    P0,
    PDF_BASE,
    PLOT_DIR,
    REDSHIFT,
    RHO_C,
    RHO_EXP,
    S_EDGES,
    SIGMA_V,
    SUMMARY_DIR,
    ZMAX,
    ZMIN,
    atomic_savez,
    read_jsonl,
    sha256,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nreal", type=int, default=NREAL)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-active-checkpoint",
        action="store_true",
        help=(
            "For nreal < 1000 only, permit the live production process and its active FIFO "
            "while retaining every per-row, covariance, hash, no-PNG, and estimator audit. "
            "This option is forbidden for the final x1000 audit."
        ),
    )
    return parser.parse_args()


def covariance_diagnostics(covariance: np.ndarray) -> dict[str, object]:
    cov = 0.5 * (np.asarray(covariance, dtype="f8") + np.asarray(covariance, dtype="f8").T)
    eig = np.linalg.eigvalsh(cov); diag = np.diag(cov)
    return {
        "shape": list(cov.shape), "finite": bool(np.all(np.isfinite(cov))),
        "symmetry_max_abs": float(np.max(np.abs(cov - cov.T))),
        "diag_min": float(diag.min()), "diag_max": float(diag.max()),
        "min_eigenvalue": float(eig.min()), "max_eigenvalue": float(eig.max()),
        "condition_number": float(np.linalg.cond(cov)),
    }


def measurement_drift_diagnostics(
    values: np.ndarray,
    coordinate: np.ndarray,
    block_size: int,
) -> dict[str, object]:
    """Record outlier and production-order drift diagnostics without cuts.

    These are deliberately diagnostic-only: an extreme value is reported but
    never used to remove a realization or alter the sample covariance.
    """
    array = np.asarray(values, dtype="f8")
    grid = np.asarray(coordinate, dtype="f8")
    if array.ndim != 2 or grid.shape != (array.shape[1],):
        raise ValueError("measurement drift diagnostic shape mismatch")
    if not 1 <= block_size <= array.shape[0] // 2:
        raise ValueError("invalid measurement drift edge block size")
    first = array[:block_size]
    last = array[-block_size:]
    first_mean = first.mean(axis=0)
    last_mean = last.mean(axis=0)
    edge_standard_error = np.sqrt(
        first.var(axis=0, ddof=1) / block_size
        + last.var(axis=0, ddof=1) / block_size
    )
    edge_z = (last_mean - first_mean) / edge_standard_error
    index = np.arange(array.shape[0], dtype="f8")
    index_centered = index - index.mean()
    values_centered = array - array.mean(axis=0)
    index_correlation = (index_centered @ values_centered) / np.sqrt(
        np.sum(index_centered**2) * np.sum(values_centered**2, axis=0)
    )
    standard_deviation = array.std(axis=0, ddof=1)
    row_standardized_rms = np.sqrt(
        np.mean(((array - array.mean(axis=0)) / standard_deviation) ** 2, axis=1)
    )
    edge_max_index = int(np.argmax(np.abs(edge_z)))
    correlation_max_index = int(np.argmax(np.abs(index_correlation)))
    row_max_index = int(np.argmax(row_standardized_rms))
    finite = bool(
        np.all(np.isfinite(edge_standard_error))
        and np.all(edge_standard_error > 0.0)
        and np.all(np.isfinite(edge_z))
        and np.all(np.isfinite(index_correlation))
        and np.all(np.isfinite(row_standardized_rms))
    )
    return {
        "status": "pass" if finite else "fail",
        "role": "diagnostic_only_no_realization_cuts_no_covariance_modification",
        "audited_rows": int(array.shape[0]),
        "nbins": int(array.shape[1]),
        "edge_block_size": int(block_size),
        "coordinate": grid,
        "first_block_mean": first_mean,
        "last_block_mean": last_mean,
        "last_minus_first_block_mean": last_mean - first_mean,
        "last_minus_first_standard_error_z": edge_z,
        "edge_z_rms": float(np.sqrt(np.mean(edge_z**2))),
        "edge_z_max_abs": float(np.max(np.abs(edge_z))),
        "edge_z_max_abs_bin_index": edge_max_index,
        "edge_z_max_abs_coordinate": float(grid[edge_max_index]),
        "index_correlation": index_correlation,
        "index_correlation_max_abs": float(np.max(np.abs(index_correlation))),
        "index_correlation_max_abs_bin_index": correlation_max_index,
        "index_correlation_max_abs_coordinate": float(grid[correlation_max_index]),
        "row_standardized_rms_max": float(row_standardized_rms[row_max_index]),
        "row_standardized_rms_max_production_index": row_max_index,
        "row_standardized_rms_p99": float(np.quantile(row_standardized_rms, 0.99)),
    }


def xi_pair_count_diagnostics(
    dd: np.ndarray, dr: np.ndarray, rr: np.ndarray, xi0: np.ndarray,
) -> dict[str, object]:
    """Check Landy--Szalay consistency at FCFC's ASCII precision.

    FCFC writes double-precision ASCII columns with ``OFMT_DBL=\"%.10lg\"``.
    Reconstructing xi from those rounded DD/DR/RR columns therefore cannot be
    compared at binary floating-point precision.  The bound below propagates
    a decimal half-ULP (5e-10 relative) through the estimator and also allows
    for rounding of the independently written xi column.
    """
    dd = np.asarray(dd, dtype="f8")
    dr = np.asarray(dr, dtype="f8")
    rr = np.asarray(rr, dtype="f8")
    xi0 = np.asarray(xi0, dtype="f8")
    shape_gate = dd.shape == dr.shape == rr.shape == xi0.shape == (30,)
    finite_gate = bool(all(np.all(np.isfinite(item)) for item in (dd, dr, rr, xi0)))
    count_gate = bool(np.all(dd >= 0.0) and np.all(dr >= 0.0) and np.all(rr > 0.0))
    if not (shape_gate and finite_gate and count_gate):
        return {
            "status": "fail", "shape_gate": shape_gate,
            "finite_gate": finite_gate, "count_gate": count_gate,
        }
    reconstructed = (dd - 2.0 * dr + rr) / rr
    decimal_relative_half_ulp = 5.0e-10
    rounding_bound = (
        decimal_relative_half_ulp
        * (np.abs(dd) + 2.0 * np.abs(dr) + np.abs(rr)) / np.abs(rr)
        + decimal_relative_half_ulp * (np.abs(reconstructed) + np.abs(xi0))
    ) / (1.0 - decimal_relative_half_ulp)
    delta = np.abs(reconstructed - xi0)
    rounding_gate = bool(np.all(delta <= rounding_bound))
    return {
        "status": "pass" if rounding_gate else "fail",
        "shape_gate": shape_gate, "finite_gate": finite_gate,
        "count_gate": count_gate, "rounding_gate": rounding_gate,
        "fcfc_ascii_format": "OFMT_DBL=%.10lg",
        "decimal_relative_half_ulp": decimal_relative_half_ulp,
        "max_abs_reconstructed_minus_xi0": float(np.max(delta)),
        "max_propagated_rounding_bound": float(np.max(rounding_bound)),
        "max_delta_over_bound": float(np.max(delta / rounding_bound)),
    }


def output_paths(nreal: int) -> tuple[Path, Path, Path]:
    kind = "pilot" if nreal < NREAL else "final"
    stem = f"task43_ezmock_lightcone_covariance_{kind}_x{nreal}_joint{JOINT_DIMENSION}"
    return SUMMARY_DIR / f"{stem}.npz", SUMMARY_DIR / f"{stem}.json", PLOT_DIR / f"{stem}.pdf"


def tree_size_bytes(path: Path) -> int:
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def main() -> None:
    args = parse_args(); nreal = int(args.nreal)
    if bool(args.allow_active_checkpoint) and nreal >= NREAL:
        raise ValueError("--allow-active-checkpoint is forbidden for the final x1000 audit")
    if not 2 <= nreal <= NREAL:
        raise ValueError(nreal)
    out_npz, out_json, out_pdf = output_paths(nreal)
    if all(path.is_file() for path in (out_npz, out_json, out_pdf)) and not args.overwrite:
        print(f"[skip] {out_npz}"); return
    for path in (MANIFEST, COMMON_RANDOM, COMMON_RANDOM_HDF5, COMMON_RR, ABACUS_XI_SUMMARY, ABACUS_PK_PAYLOAD):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = read_jsonl(MANIFEST)[:nreal]
    if len(rows) != nreal or FIX_AMPLITUDE is not False:
        raise RuntimeError("manifest/FIX_AMPLITUDE global gate failed")
    target_pk = np.load(ABACUS_PK_PAYLOAD, allow_pickle=False)
    k = np.asarray(target_pk["k_obs"], dtype="f8"); k_edges = np.asarray(target_pk["k_edges"], dtype="f8")
    target_fine_indices = np.asarray(target_pk["fit_bin_indices"], dtype="i8")
    if target_fine_indices.shape != (N_PK_BINS,):
        raise RuntimeError("invalid frozen P(k) fine-bin index vector")
    common_rr_values = np.asarray(np.loadtxt(COMMON_RR, comments="#", dtype="f8")[:, 2], dtype="f8")
    if common_rr_values.shape != (30,) or not np.all(np.isfinite(common_rr_values)) or np.any(common_rr_values <= 0.0):
        raise RuntimeError("invalid immutable common RR vector")
    common_hashes = {"npz": sha256(COMMON_RANDOM), "hdf5": sha256(COMMON_RANDOM_HDF5), "rr": sha256(COMMON_RR), "fkp": sha256(ABACUS_FKP_SUMMARY)}
    xi_stack, pk_stack, shot_stack, ndata = [], [], [], []
    lightcone_z_min, lightcone_z_max, lightcone_nbar = [], [], []
    lightcone_coord_residual = []
    xi_runtime, pk_runtime, build_runtime = [], [], []
    xi_pair_count_audit = []
    native_fcfc_audit = []
    pk_vector_audit = []
    per_row_audit = []
    for expected_index, row in enumerate(rows):
        config = Path(str(row["ezmock_config_path"])); log = Path(str(row["ezmock_log_path"]))
        halo_path = Path(str(row["halo_catalog_path"])); halo_meta_path = Path(str(row["halo_metadata_path"]))
        xi_path = Path(str(row["xi_path"])); pk_path = Path(str(row["pk_path"]))
        paths = (config, log, halo_path, halo_meta_path, xi_path, xi_path.with_suffix(".json"), pk_path, pk_path.with_suffix(".json"))
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"row {expected_index} missing {missing}")
        config_text = config.read_text(encoding="utf-8", errors="replace")
        log_text = log.read_text(encoding="utf-8", errors="replace")
        halo_meta = json.loads(halo_meta_path.read_text(encoding="utf-8"))
        xi_meta = json.loads(xi_path.with_suffix(".json").read_text(encoding="utf-8"))
        pk_meta = json.loads(pk_path.with_suffix(".json").read_text(encoding="utf-8"))
        xi_config_path = Path(str(xi_meta.get("config", "")))
        if not xi_config_path.is_file():
            raise FileNotFoundError(f"row {expected_index} missing FCFC config {xi_config_path}")
        xi_config_text = xi_config_path.read_text(encoding="utf-8", errors="replace")
        pair_count_paths = {
            key: Path(str(xi_meta.get("pair_counts", {}).get(key, "")))
            for key in ("DD", "DR", "RR")
        }
        xi_text_path = Path(str(xi_meta.get("outputs", {}).get("text", "")))
        xi_log_path = Path(str(xi_meta.get("log", "")))
        native_fcfc_paths = (*pair_count_paths.values(), xi_text_path, xi_config_path, xi_log_path)
        gates = {
            "index": int(row["production_index"]) == expected_index,
            "manifest_fixF": row["fix_amplitude"] is False,
            "manifest_attachT": row["attach_particle"] is True,
            "manifest_s_grid": np.array_equal(np.asarray(row["s_edges"], dtype="f8"), S_EDGES),
            "config_fixF": re.search(r"^FIX_AMPLITUDE\s*=\s*F\s*$", config_text, re.MULTILINE) is not None,
            "config_attachT": re.search(r"^ATTACH_PARTICLE\s*=\s*T\s*$", config_text, re.MULTILINE) is not None,
            "log_fixF": re.search(r"FIX_AMPLITUDE\s*=\s*F", log_text) is not None,
            "log_attachT": re.search(r"ATTACH_PARTICLE\s*=\s*T", log_text) is not None,
            "halo_fixF": halo_meta.get("fix_amplitude") is False,
            "halo_attachT": halo_meta.get("attach_particle") is True,
            "xi_fixF": xi_meta.get("fix_amplitude") is False,
            "pk_fixF": pk_meta.get("fix_amplitude") is False,
            "seed": all(int(item.get("seed", -1)) == int(row["seed"]) for item in (halo_meta, xi_meta, pk_meta)),
            "ntracer": int(row["ntracer"]) == DEFAULT_NTRACER and int(halo_meta.get("ntracer_requested", -1)) == DEFAULT_NTRACER,
            "tuned_parameters": all(np.isclose(float(row[key]), expected, rtol=0.0, atol=1e-14) for key, expected in (("rho_c", RHO_C), ("rho_exp", RHO_EXP), ("pdf_base", PDF_BASE), ("sigma_v", SIGMA_V))),
            "fkp": xi_meta.get("fkp_summary") == str(ABACUS_FKP_SUMMARY) and pk_meta.get("fkp_summary") == str(ABACUS_FKP_SUMMARY) and xi_meta.get("fkp_summary_sha256") == common_hashes["fkp"] and pk_meta.get("fkp_summary_sha256") == common_hashes["fkp"] and float(xi_meta.get("p0", -1)) == P0 and float(pk_meta.get("p0", -1)) == P0,
            "common_xi": xi_meta.get("common_random_npz_sha256") == common_hashes["npz"] and xi_meta.get("common_random_hdf5_sha256") == common_hashes["hdf5"] and xi_meta.get("common_rr_sha256") == common_hashes["rr"],
            "common_pk": pk_meta.get("common_random_sha256") == common_hashes["npz"],
            "rr_read": xi_meta.get("rr_cache_read_gate") is True,
            "fcfc_smax350_only": bool(
                re.search(
                    r"^SEP_BIN_MAX\s*=\s*350(?:\.0*)?\s*$",
                    xi_config_text,
                    re.MULTILINE,
                )
                and not re.search(
                    r"^SEP_BIN_MAX\s*=\s*550(?:\.0*)?\s*$",
                    xi_config_text,
                    re.MULTILINE,
                )
            ),
            "one_window_policy": pk_meta.get("has_window") is (expected_index == 0),
            "native_fcfc_paths": bool(
                all(path.is_file() and path.stat().st_size > 0 for path in native_fcfc_paths)
                and pair_count_paths["RR"] == COMMON_RR
                and xi_config_path == Path(str(xi_meta.get("config")))
            ),
        }
        if not gates["native_fcfc_paths"]:
            raise RuntimeError(f"row {expected_index} native FCFC path gate failed")
        native_pair_tables = {
            key: np.asarray(np.loadtxt(path, comments="#", dtype="f8"), dtype="f8")
            for key, path in pair_count_paths.items()
        }
        native_xi_table = np.asarray(np.loadtxt(xi_text_path, comments="#", dtype="f8"), dtype="f8")
        native_pair_grid_gate = all(
            table.shape == (30, 3)
            and np.all(np.isfinite(table))
            and np.array_equal(np.concatenate([table[:1, 0], table[:, 1]]), S_EDGES)
            for table in native_pair_tables.values()
        )
        native_xi_grid_gate = bool(
            native_xi_table.shape == (30, 4)
            and np.all(np.isfinite(native_xi_table))
            and np.array_equal(
                np.concatenate([native_xi_table[:1, 1], native_xi_table[:, 2]]), S_EDGES
            )
        )
        native_log_text = xi_log_path.read_text(encoding="utf-8", errors="replace")
        gates.update({
            "native_fcfc_pair_grid": native_pair_grid_gate,
            "native_fcfc_xi_grid": native_xi_grid_gate,
            "native_fcfc_config_paths": bool(
                all(str(path) in xi_config_text for path in pair_count_paths.values())
            ),
            "native_fcfc_log_rr": f"<R> {COMMON_RR}" in native_log_text,
        })
        with np.load(halo_path, allow_pickle=False) as halo:
            required_halo = {
                "Z", "X", "Y", "Zcart", "WEIGHT", "phase", "seed",
                "production_index", "fix_amplitude", "attach_particle",
                "snapshot_redshift",
            }
            gates["halo_catalog_fields"] = required_halo.issubset(halo.files)
            if not gates["halo_catalog_fields"]:
                raise RuntimeError(f"row {expected_index} halo catalog fields missing")
            redshift = np.asarray(halo["Z"], dtype="f8")
            x = np.asarray(halo["X"], dtype="f8")
            y = np.asarray(halo["Y"], dtype="f8")
            zcart = np.asarray(halo["Zcart"], dtype="f8")
            base_weight = np.asarray(halo["WEIGHT"], dtype="f8")
            halo_catalog_count = int(redshift.size)
            shapes = {item.shape for item in (redshift, x, y, zcart, base_weight)}
            coordinate_min = np.asarray([x.min(), y.min(), zcart.min()], dtype="f8")
            coordinate_max = np.asarray([x.max(), y.max(), zcart.max()], dtype="f8")
            gates.update({
                "halo_catalog_shapes": shapes == {(halo_catalog_count,)},
                "halo_catalog_finite": bool(all(np.all(np.isfinite(item)) for item in (redshift, x, y, zcart, base_weight))),
                "halo_catalog_positive_octant": bool(
                    np.all(coordinate_min >= 0.0) and np.all(coordinate_max < BOX_SIZE)
                ),
                # The selection is open in double precision before Z is cast
                # to float32.  Stored values can round to the float32 images
                # of the nominal boundaries and are therefore checked against
                # that closed representable interval.
                "halo_catalog_redshift_cut": bool(
                    np.all(redshift >= float(np.float32(ZMIN)))
                    and np.all(redshift <= float(np.float32(ZMAX)))
                ),
                "halo_catalog_base_weight": bool(np.all(base_weight == 1.0)),
                "halo_catalog_provenance": bool(
                    str(np.asarray(halo["phase"]).item()) == str(row["phase"])
                    and int(np.asarray(halo["seed"]).item()) == int(row["seed"])
                    and int(np.asarray(halo["production_index"]).item()) == expected_index
                    and not bool(np.asarray(halo["fix_amplitude"]).item())
                    and bool(np.asarray(halo["attach_particle"]).item())
                    and float(np.asarray(halo["snapshot_redshift"]).item()) == REDSHIFT
                ),
                "halo_metadata_geometry": bool(
                    int(halo_meta.get("selected_count", -1)) == halo_catalog_count
                    and halo_meta.get("positive_octant_gate") is True
                    and float(halo_meta.get("zmin", -1.0)) == ZMIN
                    and float(halo_meta.get("zmax", -1.0)) == ZMAX
                    and np.array_equal(np.asarray(halo_meta.get("z_range", []), dtype="f8"), np.asarray([redshift.min(), redshift.max()]))
                    and np.array_equal(np.asarray(halo_meta.get("coordinate_min", []), dtype="f8"), coordinate_min)
                    and np.array_equal(np.asarray(halo_meta.get("coordinate_max", []), dtype="f8"), coordinate_max)
                    and float(halo_meta.get("coord_check_radius_over_chi_minus_one_max_abs", 1.0)) < 1.0e-6
                ),
            })
            lightcone_coord_residual.append(float(halo_meta["coord_check_radius_over_chi_minus_one_max_abs"]))
        with np.load(xi_path, allow_pickle=False) as xi:
            if xi["xi0"].shape != (30,) or not np.array_equal(xi["s_edges"], S_EDGES) or not np.all(np.isfinite(xi["xi0"])):
                raise RuntimeError(f"row {expected_index} xi vector gate failed")
            pair_diagnostics = xi_pair_count_diagnostics(xi["DD"], xi["DR"], xi["RR"], xi["xi0"])
            gates["xi_estimator_rounding"] = pair_diagnostics.get("status") == "pass"
            gates["rr_array_common"] = np.array_equal(np.asarray(xi["RR"], dtype="f8"), common_rr_values)
            gates["ndata_catalog_xi"] = int(np.asarray(xi["ndata"]).item()) == halo_catalog_count
            gates["native_fcfc_npz_consistency"] = bool(
                native_pair_grid_gate
                and native_xi_grid_gate
                and all(
                    np.array_equal(np.asarray(xi[key], dtype="f8"), native_pair_tables[key][:, 2])
                    for key in ("DD", "DR", "RR")
                )
                and np.array_equal(np.asarray(xi["s"], dtype="f8"), native_xi_table[:, 0])
                and np.array_equal(np.asarray(xi["xi0"], dtype="f8"), native_xi_table[:, 3])
            )
            xi_pair_count_audit.append(pair_diagnostics)
            native_fcfc_audit.append({
                key: gates[key] for key in (
                    "native_fcfc_paths", "native_fcfc_pair_grid",
                    "native_fcfc_xi_grid", "native_fcfc_config_paths",
                    "native_fcfc_log_rr", "native_fcfc_npz_consistency",
                )
            })
            xi_stack.append(np.asarray(xi["xi0"], dtype="f8")); ndata.append(int(np.asarray(xi["ndata"]).item()))
        with np.load(pk_path, allow_pickle=False) as pk:
            required_pk = {
                "k", "k_edges", "pk0", "shotnoise", "fine_indices",
                "k_obs_fine", "k_edges_fine", "pk0_fine", "norm_fine",
                "num_shotnoise_fine", "shotnoise_fine", "phase", "seed",
                "production_index", "fix_amplitude", "ndata", "nrandom",
                "has_window", "common_random_sha256", "fkp_summary_sha256",
            }
            gates["pk_npz_fields"] = required_pk.issubset(pk.files)
            if not gates["pk_npz_fields"]:
                raise RuntimeError(f"row {expected_index} P(k) fields missing")
            pk0 = np.asarray(pk["pk0"], dtype="f8")
            shotnoise = np.asarray(pk["shotnoise"], dtype="f8")
            fine_indices = np.asarray(pk["fine_indices"], dtype="i8")
            k_edges_fine = np.asarray(pk["k_edges_fine"], dtype="f8")
            pk0_fine = np.asarray(pk["pk0_fine"], dtype="f8")
            shotnoise_fine = np.asarray(pk["shotnoise_fine"], dtype="f8")
            fine_numeric = tuple(
                np.asarray(pk[key]) for key in (
                    "k_obs_fine", "k_edges_fine", "pk0_fine", "norm_fine",
                    "num_shotnoise_fine", "shotnoise_fine",
                )
            )
            gates.update({
                "pk_npz_provenance": bool(
                    str(np.asarray(pk["phase"]).item()) == str(row["phase"])
                    and int(np.asarray(pk["seed"]).item()) == int(row["seed"])
                    and int(np.asarray(pk["production_index"]).item()) == expected_index
                    and not bool(np.asarray(pk["fix_amplitude"]).item())
                    and int(np.asarray(pk["ndata"]).item()) == halo_catalog_count
                    and int(np.asarray(pk["nrandom"]).item()) == COMMON_RANDOM_SIZE
                    and str(np.asarray(pk["common_random_sha256"]).item()) == common_hashes["npz"]
                    and str(np.asarray(pk["fkp_summary_sha256"]).item()) == common_hashes["fkp"]
                ),
                "pk_fine_shapes": bool(
                    np.asarray(pk["k_obs_fine"]).shape == (150,)
                    and k_edges_fine.shape == (150, 2)
                    and all(item.shape == (150,) for item in (
                        pk0_fine, np.asarray(pk["norm_fine"]),
                        np.asarray(pk["num_shotnoise_fine"]), shotnoise_fine,
                    ))
                ),
                "pk_fine_finite": bool(all(np.all(np.isfinite(item)) for item in fine_numeric)),
                "pk_target_selection": bool(
                    fine_indices.shape == (N_PK_BINS,)
                    and np.array_equal(fine_indices, target_fine_indices)
                    and np.array_equal(k_edges_fine[fine_indices], k_edges)
                    and np.array_equal(pk0, pk0_fine[fine_indices])
                    and np.array_equal(shotnoise, shotnoise_fine[fine_indices])
                ),
                "pk_shotnoise_finite": bool(
                    shotnoise.shape == (N_PK_BINS,) and np.all(np.isfinite(shotnoise))
                ),
                "pk_window_scalar": bool(np.asarray(pk["has_window"]).item()) is (expected_index == 0),
            })
            window_fields = {
                "window_matrix", "window_observable_k", "window_observable_edges",
                "theory_k", "theory_edges", "theory_ell",
                "theory_slice_start", "theory_slice_stop",
            }
            if expected_index == 0:
                gates["pk_window_fields"] = window_fields.issubset(pk.files)
                if gates["pk_window_fields"]:
                    window_matrix = np.asarray(pk["window_matrix"], dtype="f8")
                    theory_k = np.asarray(pk["theory_k"], dtype="f8")
                    window_numeric = tuple(np.asarray(pk[key]) for key in window_fields)
                    gates["pk_window_shape_finite"] = bool(
                        window_matrix.shape == (150, theory_k.size)
                        and np.asarray(pk["window_observable_k"]).shape == (150,)
                        and np.asarray(pk["window_observable_edges"]).shape == (150, 2)
                        and np.asarray(pk["theory_edges"]).shape == (theory_k.size, 2)
                        and np.asarray(pk["theory_ell"]).shape == (theory_k.size,)
                        and np.asarray(pk["theory_slice_start"]).shape == (3,)
                        and np.asarray(pk["theory_slice_stop"]).shape == (3,)
                        and all(np.all(np.isfinite(item)) for item in window_numeric)
                    )
                else:
                    gates["pk_window_shape_finite"] = False
            else:
                gates["pk_window_fields"] = not any(key in pk.files for key in window_fields)
                gates["pk_window_shape_finite"] = gates["pk_window_fields"]
            if pk0.shape != (N_PK_BINS,) or not np.array_equal(pk["k"], k) or not np.array_equal(pk["k_edges"], k_edges) or not np.all(np.isfinite(pk0)):
                raise RuntimeError(f"row {expected_index} P(k) vector gate failed")
            pk_vector_audit.append({
                key: gates[key] for key in (
                    "pk_npz_fields", "pk_npz_provenance", "pk_fine_shapes",
                    "pk_fine_finite", "pk_target_selection",
                    "pk_shotnoise_finite", "pk_window_scalar",
                    "pk_window_fields", "pk_window_shape_finite",
                )
            })
            pk_stack.append(pk0); shot_stack.append(shotnoise)
        if not all(gates.values()):
            raise RuntimeError(f"row {expected_index} audit failed: {gates}")
        build_runtime.append(float(halo_meta["runtime_sec"])); xi_runtime.append(float(xi_meta["runtime_sec"])); pk_runtime.append(float(pk_meta["runtime_sec"]))
        lightcone_z_min.append(float(halo_meta["z_range"][0])); lightcone_z_max.append(float(halo_meta["z_range"][1]))
        lightcone_nbar.append(float(halo_meta["nbar_volume_weighted"]))
        per_row_audit.append(gates)
    seeds = np.asarray([row["seed"] for row in rows], dtype="i8")
    if np.unique(seeds).size != nreal:
        raise RuntimeError("duplicate production seeds")
    xi_stack = np.vstack(xi_stack); pk_stack = np.vstack(pk_stack); shot_stack = np.vstack(shot_stack); ndata = np.asarray(ndata, dtype="i8")
    lightcone_z_min = np.asarray(lightcone_z_min, dtype="f8"); lightcone_z_max = np.asarray(lightcone_z_max, dtype="f8")
    lightcone_nbar = np.asarray(lightcone_nbar, dtype="f8")
    abundance_target = 331_545.0
    abundance_fractional_offset = float(np.mean(ndata) / abundance_target - 1.0)
    block_size = min(100, nreal // 2)
    production_abundance_diagnostics = {
        "status": "pass" if abs(abundance_fractional_offset) < 0.02 else "fail",
        "audited_rows": nreal,
        "target_abacus_mean_count": int(abundance_target),
        "absolute_fractional_gate": 0.02,
        "mean": float(np.mean(ndata)),
        "std_ddof1": float(np.std(ndata, ddof=1)),
        "min": int(np.min(ndata)),
        "max": int(np.max(ndata)),
        "fractional_offset": abundance_fractional_offset,
        "index_count_correlation": float(np.corrcoef(np.arange(nreal, dtype="f8"), ndata)[0, 1]),
        "edge_block_size": block_size,
        "first_block_mean": float(np.mean(ndata[:block_size])),
        "last_block_mean": float(np.mean(ndata[-block_size:])),
        "last_minus_first_block_mean": float(
            np.mean(ndata[-block_size:]) - np.mean(ndata[:block_size])
        ),
    }
    vector_stack = np.column_stack([xi_stack, pk_stack]); covariance = np.cov(vector_stack, rowvar=False, ddof=1)
    sigma = np.sqrt(np.diag(covariance)); correlation = covariance / np.outer(sigma, sigma)
    xi_cov, pk_cov, cross_cov = covariance[:30, :30], covariance[30:, 30:], covariance[:30, 30:]
    checkpoints = [value for value in (50, 100, 250, 500, 1000) if value <= nreal]
    convergence_cov = np.stack([np.cov(vector_stack[:value], rowvar=False, ddof=1) for value in checkpoints])
    denom = np.linalg.norm(covariance)
    convergence_rel_fro = np.asarray([np.linalg.norm(item - covariance) / denom for item in convergence_cov], dtype="f8")
    with np.load(ABACUS_XI_SUMMARY, allow_pickle=False) as ref:
        s = np.asarray(ref["s"], dtype="f8")[:30]; xi_ref = np.asarray(ref["xi0_mean"], dtype="f8")[:30]; xi_ref_std = np.asarray(ref["xi0_std"], dtype="f8")[:30]
    pk_ref = np.asarray(target_pk["pk_mean"], dtype="f8")
    target_pk.close()
    with np.load(ABACUS_FKP_SUMMARY, allow_pickle=False) as fkp:
        p0_values = np.asarray(fkp["p0_values"], dtype="f8")
        p0_matches = np.flatnonzero(np.isclose(p0_values, P0, rtol=0.0, atol=1.0e-12))
        if p0_matches.size != 1:
            raise RuntimeError(f"frozen FKP table has {p0_matches.size} matches for P0={P0}")
        p0_index = int(p0_matches[0])
        effective_redshift = {
            "source": str(ABACUS_FKP_SUMMARY),
            "p0_index": p0_index,
            "zeff_random_auto": float(np.asarray(fkp["zeff_random_auto"], dtype="f8")[p0_index]),
            "zeff_data_auto": float(np.asarray(fkp["zeff_data_auto"], dtype="f8")[p0_index]),
            "zeff_data_random_cross": float(np.asarray(fkp["zeff_data_random_cross"], dtype="f8")[p0_index]),
            "data_weighted_mean_z": float(np.asarray(fkp["data_weighted_mean_z"], dtype="f8")[p0_index]),
            "random_weighted_mean_z": float(np.asarray(fkp["random_weighted_mean_z"], dtype="f8")[p0_index]),
        }
    xi_mean, pk_mean = xi_stack.mean(axis=0), pk_stack.mean(axis=0)
    closure = {
        "xi_mean_residual_rms_in_abacus_sigma": float(np.sqrt(np.mean(((xi_mean - xi_ref) / xi_ref_std) ** 2))),
        "xi_mean_residual_max_abs_in_abacus_sigma": float(np.max(np.abs((xi_mean - xi_ref) / xi_ref_std))),
        "pk_mean_fractional_rms": float(np.sqrt(np.mean(((pk_mean - pk_ref) / pk_ref) ** 2))),
        "pk_mean_fractional_max_abs": float(np.max(np.abs((pk_mean - pk_ref) / pk_ref))),
    }
    measurement_drift = {
        "interpretation": (
            "diagnostic-only look at production-order drift and row-level scatter; "
            "no threshold removes a realization or modifies the covariance"
        ),
        "xi": measurement_drift_diagnostics(xi_stack, s, block_size),
        "pk": measurement_drift_diagnostics(pk_stack, k, block_size),
    }
    hartlap = (nreal - JOINT_DIMENSION - 2) / (nreal - 1)
    diagnostics = covariance_diagnostics(covariance)
    correlation_diagnostics = covariance_diagnostics(correlation)
    xi_estimator_diagnostics = {
        "status": "pass" if all(item.get("status") == "pass" for item in xi_pair_count_audit) else "fail",
        "audited_rows": len(xi_pair_count_audit),
        "estimator": "(DD - 2 DR + RR) / RR",
        "fcfc_ascii_format": "OFMT_DBL=%.10lg",
        "decimal_relative_half_ulp": 5.0e-10,
        "max_abs_reconstructed_minus_xi0": float(max(item["max_abs_reconstructed_minus_xi0"] for item in xi_pair_count_audit)),
        "max_propagated_rounding_bound": float(max(item["max_propagated_rounding_bound"] for item in xi_pair_count_audit)),
        "max_delta_over_bound": float(max(item["max_delta_over_bound"] for item in xi_pair_count_audit)),
        "all_rr_arrays_bitwise_equal_common_rr": bool(all(gates["rr_array_common"] for gates in per_row_audit)),
    }
    lightcone_catalog_diagnostics = {
        "status": "pass" if all(
            all(gates[key] for key in (
                "halo_catalog_fields", "halo_catalog_shapes", "halo_catalog_finite",
                "halo_catalog_positive_octant", "halo_catalog_redshift_cut",
                "halo_catalog_base_weight", "halo_catalog_provenance",
                "halo_metadata_geometry", "ndata_catalog_xi",
            ))
            for gates in per_row_audit
        ) else "fail",
        "audited_rows": len(per_row_audit),
        "box_open_upper_bound": BOX_SIZE,
        "selection_redshift_open_interval": [ZMIN, ZMAX],
        "stored_redshift_closed_interval_float32": [
            float(np.float32(ZMIN)), float(np.float32(ZMAX)),
        ],
        "snapshot_redshift": REDSHIFT,
        "coord_check_radius_over_chi_minus_one_max_abs": {
            "max": float(np.max(lightcone_coord_residual)),
            "mean": float(np.mean(lightcone_coord_residual)),
        },
    }
    pk_measurement_diagnostics = {
        "status": "pass" if all(all(item.values()) for item in pk_vector_audit) else "fail",
        "audited_rows": len(pk_vector_audit),
        "fine_spectrum_bins": 150,
        "selected_mcmc_bins": N_PK_BINS,
        "fine_indices": target_fine_indices,
        "window_rows": [0],
        "all_shotnoise_finite": bool(all(item["pk_shotnoise_finite"] for item in pk_vector_audit)),
        "all_target_selections_exact": bool(all(item["pk_target_selection"] for item in pk_vector_audit)),
        "all_window_policies_exact": bool(all(
            item["pk_window_scalar"] and item["pk_window_fields"] and item["pk_window_shape_finite"]
            for item in pk_vector_audit
        )),
    }
    native_fcfc_diagnostics = {
        "status": "pass" if all(all(item.values()) for item in native_fcfc_audit) else "fail",
        "audited_rows": len(native_fcfc_audit),
        "pair_tables_per_row": ["DD", "DR"],
        "shared_rr": str(COMMON_RR),
        "all_native_files_nonempty": bool(all(item["native_fcfc_paths"] for item in native_fcfc_audit)),
        "all_native_grids_exact": bool(all(
            item["native_fcfc_pair_grid"] and item["native_fcfc_xi_grid"]
            for item in native_fcfc_audit
        )),
        "all_native_arrays_equal_npz": bool(all(
            item["native_fcfc_npz_consistency"] for item in native_fcfc_audit
        )),
        "all_logs_confirm_shared_rr": bool(all(
            item["native_fcfc_log_rr"] for item in native_fcfc_audit
        )),
    }
    runtime = {
        "lightcone_sec_mean": float(np.mean(build_runtime)), "lightcone_sec_sum": float(np.sum(build_runtime)),
        "xi_sec_mean": float(np.mean(xi_runtime)), "xi_sec_sum": float(np.sum(xi_runtime)),
        "pk_sec_mean": float(np.mean(pk_runtime)), "pk_sec_sum": float(np.sum(pk_runtime)),
    }
    storage = {
        name: tree_size_bytes(OUTPUT_ROOT / name)
        for name in (
            "lightcone_catalogs", "xi_fcfc", "pk_jaxpower", "fcfc_pairs",
            "ezmock_configs", "fcfc_configs", "logs",
        )
    }
    storage["production_output_root_total"] = tree_size_bytes(OUTPUT_ROOT)
    png_files = [str(path) for root in (OUTPUT_ROOT, PLOT_DIR) for path in root.rglob("*.png")]
    fifo_files = [str(path) for path in OUTPUT_ROOT.rglob("*") if path.exists() and stat.S_ISFIFO(path.stat().st_mode)]
    process_text = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "pid=,args="], check=True, capture_output=True, text=True).stdout
    active_production_processes = [
        line.strip() for line in process_text.splitlines()
        if any(pattern in line for pattern in ("task43_run_ezmock_covariance_login.py", "task43_build_ezmock_covariance_lightcone.py", "task43_measure_ezmock_covariance_xi_fcfc.py", "task43_measure_ezmock_covariance_pk_jaxpower.py", "FCFC_2PT_HDF5"))
        and "task43_summarize_ezmock_covariance.py" not in line
    ]
    psd_gate = diagnostics["min_eigenvalue"] >= -1.0e-10 * diagnostics["max_eigenvalue"]
    active_checkpoint_waiver = bool(args.allow_active_checkpoint) and nreal < NREAL
    effective_fifo_gate = not fifo_files or active_checkpoint_waiver
    effective_process_gate = not active_production_processes or active_checkpoint_waiver
    scientific_gate = bool(
        all(all(gates.values()) for gates in per_row_audit)
        and xi_estimator_diagnostics["status"] == "pass"
        and lightcone_catalog_diagnostics["status"] == "pass"
        and pk_measurement_diagnostics["status"] == "pass"
        and native_fcfc_diagnostics["status"] == "pass"
        and production_abundance_diagnostics["status"] == "pass"
        and measurement_drift["xi"]["status"] == "pass"
        and measurement_drift["pk"]["status"] == "pass"
    )
    audit_status = "pass" if (
        diagnostics["finite"]
        and diagnostics["symmetry_max_abs"] < 1e-12
        and psd_gate
        and scientific_gate
        and not png_files
        and effective_fifo_gate
        and effective_process_gate
    ) else "fail"
    summary = {
        "task": "task43_summarize_ezmock_covariance", "status": audit_status,
        "classification": "final_x1000_covariance" if nreal == NREAL else f"pilot_x{nreal}_diagnostic_covariance",
        "nreal": nreal, "n_xi": 30, "n_pk": N_PK_BINS, "joint_dimension": JOINT_DIMENSION,
        "manifest": str(MANIFEST), "seed_min": int(seeds.min()), "seed_max": int(seeds.max()), "seed_unique": True,
        "fix_amplitude": False, "attach_particle": ATTACH_PARTICLE, "ntracer": DEFAULT_NTRACER,
        "tuned_parameters": {"rho_c": RHO_C, "rho_exp": RHO_EXP, "pdf_base": PDF_BASE, "sigma_v": SIGMA_V},
        "common_hashes": common_hashes, "fkp_summary": str(ABACUS_FKP_SUMMARY), "p0": P0, "s_edges": S_EDGES, "smax": float(S_EDGES[-1]), "k": k, "k_edges": k_edges,
        "shapes": {
            "xi_stack": list(xi_stack.shape), "pk_stack": list(pk_stack.shape),
            "joint_stack": list(vector_stack.shape), "covariance": list(covariance.shape),
            "xi_covariance": list(xi_cov.shape), "pk_covariance": list(pk_cov.shape),
            "cross": list(cross_cov.shape),
        },
        "covariance_diagnostics": diagnostics,
        "correlation_diagnostics": correlation_diagnostics,
        "xi_estimator_diagnostics": xi_estimator_diagnostics,
        "lightcone_catalog_diagnostics": lightcone_catalog_diagnostics,
        "pk_measurement_diagnostics": pk_measurement_diagnostics,
        "native_fcfc_diagnostics": native_fcfc_diagnostics,
        "production_abundance_diagnostics": production_abundance_diagnostics,
        "measurement_drift_diagnostics": measurement_drift,
        "covariance_psd_numerical_gate": bool(psd_gate),
        "closure_to_abacus_x25_mean": closure,
        "hartlap_factor": float(hartlap), "hartlap_expression": f"({nreal}-{JOINT_DIMENSION}-2)/({nreal}-1)",
        "hartlap_note": "inverse-covariance correction only; covariance itself is not multiplied by this factor",
        "ndata": {"mean": float(ndata.mean()), "std_ddof1": float(ndata.std(ddof=1)), "min": int(ndata.min()), "max": int(ndata.max())},
        "redshift_diagnostics": {
            "frozen_fkp_effective_redshift": effective_redshift,
            "lightcone_z_min": {"min": float(lightcone_z_min.min()), "max": float(lightcone_z_min.max())},
            "lightcone_z_max": {"min": float(lightcone_z_max.min()), "max": float(lightcone_z_max.max())},
            "lightcone_nbar_volume_weighted": {
                "mean": float(lightcone_nbar.mean()), "std_ddof1": float(lightcone_nbar.std(ddof=1)),
                "min": float(lightcone_nbar.min()), "max": float(lightcone_nbar.max()),
            },
        },
        "runtime": runtime, "storage_bytes": storage,
        "convergence_checkpoints": checkpoints, "convergence_relative_frobenius_to_current": convergence_rel_fro,
        "no_png_gate": len(png_files) == 0, "png_files": png_files,
        "no_stale_fifo_gate": len(fifo_files) == 0, "fifo_files": fifo_files,
        "no_active_production_process_gate": len(active_production_processes) == 0,
        "active_production_processes": active_production_processes,
        "active_checkpoint_waiver": active_checkpoint_waiver,
        "effective_no_stale_fifo_gate": bool(effective_fifo_gate),
        "effective_no_active_production_process_gate": bool(effective_process_gate),
        "scientific_gate": scientific_gate,
        "per_row_all_gates_pass": bool(all(all(gates.values()) for gates in per_row_audit)),
        "outputs": {"npz": str(out_npz), "json": str(out_json), "pdf": str(out_pdf)},
    }
    if audit_status != "pass":
        raise RuntimeError(f"summary audit failed: {summary}")
    atomic_savez(
        out_npz, s=s, s_edges=S_EDGES, k=k, k_edges=k_edges, seeds=seeds,
        xi_stack=xi_stack, pk_stack=pk_stack, shotnoise_stack=shot_stack, joint_stack=vector_stack,
        xi_mean=xi_mean, xi_std=xi_stack.std(axis=0, ddof=1), pk_mean=pk_mean, pk_std=pk_stack.std(axis=0, ddof=1),
        shotnoise_mean=shot_stack.mean(axis=0), shotnoise_std=shot_stack.std(axis=0, ddof=1),
        covariance=covariance, correlation=correlation, xi_covariance=xi_cov, pk_covariance=pk_cov, xi_pk_cross_covariance=cross_cov,
        checkpoints=np.asarray(checkpoints, dtype="i8"), convergence_covariance=convergence_cov, convergence_relative_frobenius=convergence_rel_fro,
        ndata=ndata, lightcone_runtime_sec=np.asarray(build_runtime, dtype="f8"), xi_runtime_sec=np.asarray(xi_runtime, dtype="f8"), pk_runtime_sec=np.asarray(pk_runtime, dtype="f8"),
        lightcone_z_min=lightcone_z_min, lightcone_z_max=lightcone_z_max,
        lightcone_nbar_volume_weighted=lightcone_nbar,
        zeff_random_auto=np.asarray(effective_redshift["zeff_random_auto"], dtype="f8"),
        zeff_data_auto=np.asarray(effective_redshift["zeff_data_auto"], dtype="f8"),
        zeff_data_random_cross=np.asarray(effective_redshift["zeff_data_random_cross"], dtype="f8"),
        hartlap_factor=np.asarray(hartlap, dtype="f8"),
    )
    write_json(out_json, summary)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    ax = axes[0, 0]; ax.plot(s, s**2 * xi_ref, color="black", label="Abacus x25"); ax.plot(s, s**2 * xi_mean, color="tab:red", label=f"EZmock x{nreal}"); ax.set(xlabel=r"$s\,[h^{-1}{\rm Mpc}]$", ylabel=r"$s^2\xi_0(s)$"); ax.legend(frameon=False)
    ax = axes[0, 1]; ax.loglog(k, pk_ref, color="black", label="Abacus x25"); ax.loglog(k, pk_mean, color="tab:blue", label=f"EZmock x{nreal}"); ax.set(xlabel=r"$k\,[h\,{\rm Mpc}^{-1}]$", ylabel=r"$P_0(k)\,[(h^{-1}{\rm Mpc})^3]$"); ax.legend(frameon=False)
    ax = axes[1, 0]; image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm", origin="lower"); ax.axvline(29.5, color="k", lw=0.8); ax.axhline(29.5, color="k", lw=0.8); ax.set(xlabel="joint-bin index", ylabel="joint-bin index", title=f"{JOINT_DIMENSION} x {JOINT_DIMENSION} correlation"); fig.colorbar(image, ax=ax, fraction=0.046)
    ax = axes[1, 1]; ax.plot(checkpoints, convergence_rel_fro, marker="o"); ax.set(xlabel="number of realizations", ylabel=r"$\|C_N-C_{N_{\rm max}}\|_F/\|C_{N_{\rm max}}\|_F$", title=f"Hartlap={hartlap:.6f}"); ax.grid(alpha=0.25)
    fig.suptitle(f"Task43 EZmock covariance audit: FIX_AMPLITUDE=F, common50, N={nreal}")
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[pass] {out_npz} covariance={covariance.shape} PDF={out_pdf}")


if __name__ == "__main__":
    main()
