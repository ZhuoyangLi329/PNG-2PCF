#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strictly audit Task43 50--550 measurements and their 50--350 bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from task43_config import PROJECT_ROOT, read_jsonl
from task43_fkp_zeff import path_with_weight_tag


SCAN_ROOT = PROJECT_ROOT / "outputs/task43_outputs/rmax_scan"
DEFAULT_MANIFEST = SCAN_ROOT / "manifests/task43_rmax_scan_mmin1p4e13_x25.jsonl"
ARCHIVE_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "old_doc_codes/task4_task44_cleanup_20260707T061844Z/moved/outputs/task43_outputs"
)
ARCHIVE_HALO_DIR = ARCHIVE_OUTPUT_ROOT / "halo_catalogs"
ARCHIVE_RANDOM_DIR = ARCHIVE_OUTPUT_ROOT / "randoms"
OLD_XI_DIR = ARCHIVE_OUTPUT_ROOT / "xi_cucount"

# This exact covariance is recorded by the current authoritative jaxpower
# radial-RIC long-chain summary below.  Do not replace it with the later
# kbox/fkpNorm diagnostic product merely because that filename is newer.
AUTHORITATIVE_JAXPOWER_FIT_SUMMARY = (
    PROJECT_ROOT
    / "outputs/task43_outputs/ric_singleterm/fits/"
    "2pcf_jaxpower_ph000_dchi2_nsub200000_long_mcmc20k/"
    "task43_minimal_closure_mcmc_summary.json"
)
AUTHORITATIVE_COVARIANCE = (
    PROJECT_ROOT
    / "outputs/task43_outputs/summary/"
    "jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_"
    "rrdeconv_mesh64_nran100k_ndata50k_pad400_win3600_ds2_"
    "k0001_3000_dk002_p1p0_s50_350_ds10.npz"
)
DEFAULT_FKP_SUMMARY = (
    PROJECT_ROOT / "outputs/task43_outputs/summary/task43_fkp_zeff_mmin1p4e13_x25.npz"
)
DEFAULT_OUTPUT = SCAN_ROOT / "audits/task43_rmax_scan_measurement_bridge.json"

EXPECTED_PHASES = tuple(f"ph{index:03d}" for index in range(25))
EXPECTED_NEW_EDGES = np.arange(50.0, 560.0, 10.0, dtype="f8")
EXPECTED_NEW_CENTERS = 0.5 * (EXPECTED_NEW_EDGES[:-1] + EXPECTED_NEW_EDGES[1:])
EXPECTED_OLD_EDGES = EXPECTED_NEW_EDGES[:31]
EXPECTED_OLD_CENTERS = EXPECTED_NEW_CENTERS[:30]
EXPECTED_RANDOM_MULTIPLIER = 25
EXPECTED_WEIGHT_SCHEME = "WEIGHT_TOTAL=WEIGHT*WEIGHT_FKP"
EXPECTED_OUTPUT_TAG = "fkpP010000"
EXPECTED_BACKEND = "jax"
EXPECTED_ENGINE = "cucount_jax"
EXPECTED_ESTIMATOR = "landy_szalay"


def scalar(data: np.lib.npyio.NpzFile, key: str) -> Any:
    """Return one required scalar NPZ value as a native Python object."""
    if key not in data.files:
        raise KeyError(f"missing required key {key!r}")
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"expected scalar key {key!r}, found shape {value.shape}")
    return value.item()


def string_scalar(data: np.lib.npyio.NpzFile, key: str) -> str:
    """Return one scalar string, decoding bytes if necessary."""
    value = scalar(data, key)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def require_keys(data: np.lib.npyio.NpzFile, keys: tuple[str, ...], path: Path) -> None:
    """Raise with the source path when an NPZ is missing required fields."""
    missing = [key for key in keys if key not in data.files]
    if missing:
        raise KeyError(f"{path} missing required keys: {missing}")


def safe_relative_difference(new: np.ndarray, old: np.ndarray) -> float:
    """Return a conservative maximum elementwise relative difference."""
    scale = np.maximum(np.abs(old), np.finfo("f8").tiny)
    return float(np.max(np.abs(new - old) / scale))


def same_resolved_path(left: Path | str, right: Path | str) -> bool:
    """Compare normalized absolute paths without requiring their targets twice."""
    return Path(left).expanduser().resolve(strict=False) == Path(right).expanduser().resolve(strict=False)


def is_within(path: Path, root: Path) -> bool:
    """Return whether an existing path resolves below an existing root."""
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError):
        return False
    return True


def targets_within(path: Path, root: Path) -> bool:
    """Check path containment even when an optional metadata file is absent."""
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError):
        return False
    return True


def read_metadata(path: Path) -> dict[str, Any]:
    """Read one required catalog metadata JSON object."""
    if not path.exists():
        raise FileNotFoundError(f"missing catalog metadata: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"catalog metadata is not a JSON object: {path}")
    return payload


def catalog_rows_from_npz(path: Path) -> int:
    """Fallback row count when archived JSON metadata is unavailable."""
    if not path.exists():
        raise FileNotFoundError(f"missing archived catalog: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "Z" not in data.files:
            raise KeyError(f"{path} has no Z array for fallback row counting")
        return int(np.asarray(data["Z"]).shape[0])


def catalog_expectations(row: dict[str, Any]) -> dict[str, Any]:
    """Load authoritative counts and provenance for one manifest row."""
    phase = str(row["phase"])
    halo_path = Path(row["halo_catalog_path"])
    random_path = Path(row["random_catalog_path"])
    halo_meta_path = Path(row.get("halo_metadata_path", halo_path.with_suffix(".json")))
    random_meta_path = Path(row.get("random_metadata_path", random_path.with_suffix(".json")))

    for label, path in (("halo", halo_path), ("random", random_path)):
        if not path.exists():
            raise FileNotFoundError(f"missing archived {label} catalog for {phase}: {path}")

    archive_paths = {
        "halo_catalog": is_within(halo_path, ARCHIVE_HALO_DIR),
        "halo_metadata": (
            is_within(halo_meta_path, ARCHIVE_HALO_DIR)
            if halo_meta_path.exists()
            else targets_within(halo_meta_path, ARCHIVE_HALO_DIR)
        ),
        "random_catalog": is_within(random_path, ARCHIVE_RANDOM_DIR),
        "random_metadata": (
            is_within(random_meta_path, ARCHIVE_RANDOM_DIR)
            if random_meta_path.exists()
            else targets_within(random_meta_path, ARCHIVE_RANDOM_DIR)
        ),
    }

    if halo_meta_path.exists():
        halo_meta = read_metadata(halo_meta_path)
        ndata = int(halo_meta["selected_count"])
        halo_count_source = "catalog_metadata_json:selected_count"
        halo_meta_pass = bool(halo_meta.get("phase") == phase and halo_meta.get("status") == "done")
    else:
        halo_meta = None
        ndata = catalog_rows_from_npz(halo_path)
        halo_count_source = "catalog_npz:len(Z)"
        halo_meta_pass = True

    if random_meta_path.exists():
        random_meta = read_metadata(random_meta_path)
        nrandom = int(random_meta["nrandom"])
        random_catalog_ndata = int(random_meta["ndata"])
        random_multiplier = int(random_meta["random_multiplier"])
        random_count_source = "catalog_metadata_json:nrandom"
        random_meta_pass = bool(random_meta.get("phase") == phase and random_meta.get("status") == "done")
    else:
        random_meta = None
        nrandom = catalog_rows_from_npz(random_path)
        random_catalog_ndata = ndata
        random_multiplier = int(row.get("random_multiplier", -1))
        random_count_source = "catalog_npz:len(Z)"
        random_meta_pass = True

    row_multiplier = int(row.get("random_multiplier", -1))
    count_relation_pass = bool(
        random_catalog_ndata == ndata
        and row_multiplier == EXPECTED_RANDOM_MULTIPLIER
        and random_multiplier == EXPECTED_RANDOM_MULTIPLIER
        and nrandom == EXPECTED_RANDOM_MULTIPLIER * ndata
    )
    passed = bool(
        all(archive_paths.values()) and halo_meta_pass and random_meta_pass and count_relation_pass
    )
    return {
        "halo_catalog_path": str(halo_path),
        "random_catalog_path": str(random_path),
        "halo_metadata_path": str(halo_meta_path),
        "random_metadata_path": str(random_meta_path),
        "archive_paths": archive_paths,
        "archive_paths_pass": bool(all(archive_paths.values())),
        "ndata": ndata,
        "nrandom": nrandom,
        "random_catalog_ndata": random_catalog_ndata,
        "row_random_multiplier": row_multiplier,
        "metadata_random_multiplier": random_multiplier,
        "halo_count_source": halo_count_source,
        "random_count_source": random_count_source,
        "halo_metadata_pass": halo_meta_pass,
        "random_metadata_pass": random_meta_pass,
        "count_relation_pass": count_relation_pass,
        "pass": passed,
    }


def load_authoritative_covariance(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and strictly validate the current 30-bin bridge covariance."""
    if not path.exists():
        raise FileNotFoundError(f"missing authoritative covariance: {path}")
    with np.load(path, allow_pickle=False) as payload:
        require_keys(payload, ("s", "s_edges", "covariance_single_realization"), path)
        s = np.asarray(payload["s"], dtype="f8")
        s_edges = np.asarray(payload["s_edges"], dtype="f8")
        covariance = np.asarray(payload["covariance_single_realization"], dtype="f8")

    if covariance.shape != (30, 30):
        raise ValueError(f"authoritative covariance is not 30x30: {covariance.shape}")
    if not np.array_equal(s, EXPECTED_OLD_CENTERS) or not np.array_equal(s_edges, EXPECTED_OLD_EDGES):
        raise ValueError(f"authoritative covariance has unexpected radial bins: {path}")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"authoritative covariance has non-finite entries: {path}")
    if not np.array_equal(covariance, covariance.T):
        raise ValueError(f"authoritative covariance is not exactly symmetric: {path}")
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"authoritative covariance is not positive definite: {path}") from exc
    sigma = np.sqrt(np.diag(covariance))
    if not np.all(np.isfinite(sigma)) or not np.all(sigma > 0.0):
        raise ValueError(f"authoritative covariance has invalid diagonal errors: {path}")
    audit = {
        "path": str(path),
        "key": "covariance_single_realization",
        "shape": list(covariance.shape),
        "radial_bins_exact": True,
        "finite": True,
        "exactly_symmetric": True,
        "positive_definite": True,
        "sigma_min": float(np.min(sigma)),
        "sigma_max": float(np.max(sigma)),
    }
    return covariance, cholesky, audit


def validate_new_measurement(
    data: np.lib.npyio.NpzFile,
    *,
    path: Path,
    row: dict[str, Any],
    catalogs: dict[str, Any],
    p0: float,
) -> dict[str, Any]:
    """Apply strict structural, numeric, weighting, and provenance gates."""
    required = (
        "s",
        "s_edges",
        "xi0",
        "DD",
        "DR",
        "RR",
        "ndata",
        "nrandom",
        "phase",
        "sim_name",
        "p0",
        "weighting_meta_json",
        "halo_catalog_path",
        "random_catalog_path",
        "backend",
        "engine",
        "estimator",
        "status",
    )
    require_keys(data, required, path)

    s = np.asarray(data["s"], dtype="f8")
    s_edges = np.asarray(data["s_edges"], dtype="f8")
    arrays = {key: np.asarray(data[key], dtype="f8") for key in ("xi0", "DD", "DR", "RR")}
    array_shapes = {key: list(value.shape) for key, value in arrays.items()}
    shapes_pass = bool(
        s.shape == (50,)
        and s_edges.shape == (51,)
        and all(value.shape == (50,) for value in arrays.values())
    )
    geometry_pass = bool(
        shapes_pass
        and np.array_equal(s_edges, EXPECTED_NEW_EDGES)
        and np.array_equal(s, EXPECTED_NEW_CENTERS)
    )
    finite = {key: bool(np.all(np.isfinite(value))) for key, value in arrays.items()}
    rr_positive = bool(np.all(arrays["RR"] > 0.0))
    numeric_pass = bool(all(finite.values()) and rr_positive)

    ndata = int(scalar(data, "ndata"))
    nrandom = int(scalar(data, "nrandom"))
    count_checks = {
        "ndata_matches_archive": ndata == int(catalogs["ndata"]),
        "nrandom_matches_archive": nrandom == int(catalogs["nrandom"]),
        "nrandom_is_25_times_ndata": nrandom == EXPECTED_RANDOM_MULTIPLIER * ndata,
    }
    counts_pass = bool(catalogs["pass"] and all(count_checks.values()))

    try:
        weighting = json.loads(string_scalar(data, "weighting_meta_json"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid weighting_meta_json in {path}") from exc
    if not isinstance(weighting, dict):
        raise TypeError(f"weighting_meta_json is not an object in {path}")

    measured_halo = Path(string_scalar(data, "halo_catalog_path"))
    measured_random = Path(string_scalar(data, "random_catalog_path"))
    source_checks = {
        "measurement_halo_matches_manifest": same_resolved_path(measured_halo, row["halo_catalog_path"]),
        "measurement_random_matches_manifest": same_resolved_path(measured_random, row["random_catalog_path"]),
        "measurement_halo_is_controlled_archive": is_within(measured_halo, ARCHIVE_HALO_DIR),
        "measurement_random_is_controlled_archive": is_within(measured_random, ARCHIVE_RANDOM_DIR),
        "output_is_isolated_rmax_scan": is_within(path, SCAN_ROOT / "xi_cucount"),
    }
    source_pass = bool(all(source_checks.values()))

    p0_value = float(scalar(data, "p0"))
    weighting_p0 = weighting.get("p0")
    fkp_path = weighting.get("fkp_summary_path")
    invariant_checks = {
        "phase": string_scalar(data, "phase") == str(row["phase"]),
        "sim_name": string_scalar(data, "sim_name") == str(row["sim_name"]),
        "p0": bool(np.isclose(p0_value, float(p0), rtol=0.0, atol=0.0)),
        "weighting_p0": bool(
            weighting_p0 is not None
            and np.isclose(float(weighting_p0), float(p0), rtol=0.0, atol=0.0)
        ),
        "weighting_scheme": weighting.get("scheme") == EXPECTED_WEIGHT_SCHEME,
        "weighting_output_tag": weighting.get("output_tag") == EXPECTED_OUTPUT_TAG,
        "fkp_summary": bool(
            fkp_path is not None and same_resolved_path(str(fkp_path), DEFAULT_FKP_SUMMARY)
        ),
        "backend": string_scalar(data, "backend") == EXPECTED_BACKEND,
        "engine": string_scalar(data, "engine") == EXPECTED_ENGINE,
        "estimator": string_scalar(data, "estimator") == EXPECTED_ESTIMATOR,
        "status": string_scalar(data, "status") == "done",
    }
    invariants_pass = bool(all(invariant_checks.values()))
    passed = bool(geometry_pass and numeric_pass and counts_pass and source_pass and invariants_pass)
    return {
        "s_shape": list(s.shape),
        "s_edges_shape": list(s_edges.shape),
        "array_shapes": array_shapes,
        "geometry_pass": geometry_pass,
        "finite": finite,
        "rr_positive": rr_positive,
        "numeric_pass": numeric_pass,
        "ndata": ndata,
        "nrandom": nrandom,
        "catalogs": catalogs,
        "count_checks": count_checks,
        "counts_pass": counts_pass,
        "source_checks": source_checks,
        "source_pass": source_pass,
        "weighting": weighting,
        "invariant_checks": invariant_checks,
        "invariants_pass": invariants_pass,
        "pass": passed,
    }


def validate_old_reference(data: np.lib.npyio.NpzFile, path: Path) -> dict[str, np.ndarray]:
    """Validate one immutable 30-bin bridge reference before comparison."""
    required = (
        "s",
        "s_edges",
        "xi0",
        "DD",
        "DR",
        "RR",
        "ndata",
        "nrandom",
        "p0",
        "backend",
        "engine",
        "estimator",
    )
    require_keys(data, required, path)
    s = np.asarray(data["s"], dtype="f8")
    s_edges = np.asarray(data["s_edges"], dtype="f8")
    arrays = {key: np.asarray(data[key], dtype="f8") for key in ("xi0", "DD", "DR", "RR")}
    if s.shape != (30,) or s_edges.shape != (31,) or any(value.shape != (30,) for value in arrays.values()):
        raise ValueError(f"unexpected authoritative measurement shapes in {path}")
    if not np.array_equal(s, EXPECTED_OLD_CENTERS) or not np.array_equal(s_edges, EXPECTED_OLD_EDGES):
        raise ValueError(f"unexpected authoritative measurement bins in {path}")
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError(f"non-finite authoritative measurement array in {path}")
    if not np.all(arrays["RR"] > 0.0):
        raise ValueError(f"non-positive authoritative RR bin in {path}")
    return arrays


def bridge_report(
    new: np.lib.npyio.NpzFile,
    old: np.lib.npyio.NpzFile,
    *,
    old_path: Path,
    covariance: np.ndarray,
    cholesky: np.ndarray,
    atol_xi: float,
    rtol_count: float,
    max_xi_sigma: float,
) -> dict[str, Any]:
    """Compare the first 30 new bins with one authoritative old result."""
    old_arrays = validate_old_reference(old, old_path)
    new_xi = np.asarray(new["xi0"], dtype="f8")[:30]
    delta_xi = new_xi - old_arrays["xi0"]
    sigma = np.sqrt(np.diag(covariance))
    whitened = np.linalg.solve(cholesky, delta_xi)
    delta_chi2 = float(np.dot(whitened, whitened))
    max_abs_over_sigma = float(np.max(np.abs(delta_xi) / sigma))

    count_metrics: dict[str, Any] = {}
    count_pass = True
    for key in ("DD", "DR", "RR"):
        new_count = np.asarray(new[key], dtype="f8")[:30]
        old_count = old_arrays[key]
        passed = bool(np.allclose(new_count, old_count, rtol=float(rtol_count), atol=0.0))
        count_pass = count_pass and passed
        count_metrics[key] = {
            "max_abs": float(np.max(np.abs(new_count - old_count))),
            "max_rel": safe_relative_difference(new_count, old_count),
            "pass": passed,
        }

    invariant_keys = ("ndata", "nrandom", "p0", "backend", "engine", "estimator")
    invariants: dict[str, dict[str, Any]] = {}
    invariants_pass = True
    for key in invariant_keys:
        new_value = scalar(new, key)
        old_value = scalar(old, key)
        if key == "p0":
            passed = bool(np.isclose(float(new_value), float(old_value), rtol=0.0, atol=0.0))
        else:
            passed = new_value == old_value
        invariants_pass = invariants_pass and passed
        invariants[key] = {"new": new_value, "old": old_value, "pass": passed}

    geometry_pass = bool(
        np.array_equal(np.asarray(new["s_edges"], dtype="f8")[:31], EXPECTED_OLD_EDGES)
        and np.array_equal(np.asarray(new["s"], dtype="f8")[:30], EXPECTED_OLD_CENTERS)
    )
    xi_atol_pass = bool(np.allclose(new_xi, old_arrays["xi0"], rtol=0.0, atol=float(atol_xi)))
    xi_sigma_pass = bool(max_abs_over_sigma <= float(max_xi_sigma))
    xi_pass = bool(xi_atol_pass and xi_sigma_pass)
    passed = bool(geometry_pass and invariants_pass and count_pass and xi_pass)
    return {
        "geometry_pass": geometry_pass,
        "invariants": invariants,
        "invariants_pass": invariants_pass,
        "counts": count_metrics,
        "counts_pass": count_pass,
        "xi_max_abs": float(np.max(np.abs(delta_xi))),
        "xi_rms": float(np.sqrt(np.mean(delta_xi**2))),
        "xi_max_abs_over_authoritative_cov_sigma": max_abs_over_sigma,
        "xi_delta_chi2_authoritative_covariance": delta_chi2,
        "xi_delta_chi_authoritative_covariance": float(np.sqrt(max(delta_chi2, 0.0))),
        "xi_atol_pass": xi_atol_pass,
        "xi_sigma_pass": xi_sigma_pass,
        "xi_pass": xi_pass,
        "pass": passed,
    }


def authoritative_covariance_provenance(path: Path) -> dict[str, Any]:
    """Record whether the default covariance matches the current long-chain summary."""
    if not AUTHORITATIVE_JAXPOWER_FIT_SUMMARY.exists():
        raise FileNotFoundError(
            f"missing authoritative jaxpower long-chain summary: {AUTHORITATIVE_JAXPOWER_FIT_SUMMARY}"
        )
    summary = json.loads(AUTHORITATIVE_JAXPOWER_FIT_SUMMARY.read_text(encoding="utf-8"))
    recorded = Path(summary["covariance"]["path"])
    if not recorded.is_absolute():
        recorded = PROJECT_ROOT / recorded
    matches = same_resolved_path(path, recorded)
    return {
        "long_chain_summary": str(AUTHORITATIVE_JAXPOWER_FIT_SUMMARY),
        "long_chain_recorded_covariance": str(recorded),
        "requested_covariance_matches_long_chain": matches,
        "default_covariance": str(AUTHORITATIVE_COVARIANCE),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--old-xi-dir", type=Path, default=OLD_XI_DIR)
    parser.add_argument("--old-covariance", type=Path, default=AUTHORITATIVE_COVARIANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--p0", type=float, default=10000.0)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--atol-xi", type=float, default=2.0e-10)
    parser.add_argument("--rtol-count", type=float, default=2.0e-6)
    parser.add_argument("--max-xi-sigma", type=float, default=0.05)
    args = parser.parse_args()

    if args.atol_xi < 0.0 or args.rtol_count < 0.0 or args.max_xi_sigma < 0.0:
        raise ValueError("audit tolerances must be non-negative")
    if not np.isclose(float(args.p0), 10000.0, rtol=0.0, atol=0.0):
        raise ValueError("Task43 rmax scan requires P0=10000")

    rows = read_jsonl(args.manifest)
    phases = [str(row.get("phase")) for row in rows]
    if len(rows) != 25 or tuple(phases) != EXPECTED_PHASES or len(set(phases)) != 25:
        raise ValueError(f"expected ordered unique phases ph000..ph024, found {phases}")

    covariance, cholesky, covariance_audit = load_authoritative_covariance(args.old_covariance)
    covariance_provenance = authoritative_covariance_provenance(args.old_covariance)
    if same_resolved_path(args.old_covariance, AUTHORITATIVE_COVARIANCE) and not covariance_provenance[
        "requested_covariance_matches_long_chain"
    ]:
        raise ValueError("default authoritative covariance disagrees with current jaxpower long-chain summary")

    reports: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        phase = str(row["phase"])
        new_path = path_with_weight_tag(Path(row["xi_path"]), p0=float(args.p0))
        base_name = Path(row["xi_path"]).name
        suffix = "_s50_550_ds10.npz"
        if not base_name.endswith(suffix):
            raise ValueError(f"unexpected rmax-scan xi base name for {phase}: {base_name}")
        old_name = f"{base_name[:-len(suffix)]}.npz"
        old_path = path_with_weight_tag(args.old_xi_dir / old_name, p0=float(args.p0))
        if not new_path.exists():
            missing.append(phase)
            continue
        if not old_path.exists():
            raise FileNotFoundError(f"missing authoritative measurement: {old_path}")

        catalogs = catalog_expectations(row)
        with np.load(new_path, allow_pickle=False) as new, np.load(old_path, allow_pickle=False) as old:
            measurement = validate_new_measurement(
                new,
                path=new_path,
                row=row,
                catalogs=catalogs,
                p0=float(args.p0),
            )
            bridge = bridge_report(
                new,
                old,
                old_path=old_path,
                covariance=covariance,
                cholesky=cholesky,
                atol_xi=float(args.atol_xi),
                rtol_count=float(args.rtol_count),
                max_xi_sigma=float(args.max_xi_sigma),
            )

        report = {
            "phase": phase,
            "new_path": str(new_path),
            "old_path": str(old_path),
            "measurement": measurement,
            "bridge": bridge,
            # Retain concise compatibility fields used by the first smoke audit.
            "geometry_pass": measurement["geometry_pass"] and bridge["geometry_pass"],
            "invariants_pass": measurement["invariants_pass"] and bridge["invariants_pass"],
            "counts_pass": measurement["counts_pass"] and bridge["counts_pass"],
            "xi_max_abs": bridge["xi_max_abs"],
            "xi_max_abs_over_authoritative_cov_sigma": bridge[
                "xi_max_abs_over_authoritative_cov_sigma"
            ],
            "xi_delta_chi2_authoritative_covariance": bridge[
                "xi_delta_chi2_authoritative_covariance"
            ],
            "xi_pass": bridge["xi_pass"],
            "pass": bool(measurement["pass"] and bridge["pass"]),
        }
        reports.append(report)

    if args.require_all and missing:
        raise FileNotFoundError(f"missing new measurements for phases: {missing}")

    checked_pass = bool(reports) and all(item["pass"] for item in reports)
    complete = not missing and len(reports) == 25
    if reports and not checked_pass:
        status = "fail"
    elif complete and checked_pass:
        status = "pass"
    else:
        status = "partial"
    all_pass = bool(complete and checked_pass)
    payload = {
        "status": status,
        "task": "task43_audit_rmax_scan_measurements",
        "manifest": str(args.manifest),
        "authoritative_covariance": str(args.old_covariance),
        "authoritative_covariance_audit": covariance_audit,
        "authoritative_covariance_provenance": covariance_provenance,
        "comparison": "new 50--550 result cropped to its first 30 bins versus old 50--350 result",
        "thresholds": {
            "xi_atol": float(args.atol_xi),
            "pair_count_rtol": float(args.rtol_count),
            "max_xi_abs_over_authoritative_cov_sigma": float(args.max_xi_sigma),
        },
        "expected_measurement": {
            "nbins": 50,
            "s_edges": EXPECTED_NEW_EDGES.tolist(),
            "s_centers": EXPECTED_NEW_CENTERS.tolist(),
            "p0": float(args.p0),
            "weight_scheme": EXPECTED_WEIGHT_SCHEME,
            "backend": EXPECTED_BACKEND,
            "engine": EXPECTED_ENGINE,
            "estimator": EXPECTED_ESTIMATOR,
            "random_multiplier": EXPECTED_RANDOM_MULTIPLIER,
            "catalog_root": str(ARCHIVE_OUTPUT_ROOT),
        },
        "n_checked": len(reports),
        "n_expected": 25,
        "complete": complete,
        "missing_phases": missing,
        "checked_measurements_pass": checked_pass,
        "all_checked_pass": checked_pass,
        "complete_and_all_pass": all_pass,
        "max_xi_abs": max((item["xi_max_abs"] for item in reports), default=None),
        "max_xi_abs_over_cov_sigma": max(
            (item["xi_max_abs_over_authoritative_cov_sigma"] for item in reports), default=None
        ),
        "max_xi_delta_chi2": max(
            (item["xi_delta_chi2_authoritative_covariance"] for item in reports), default=None
        ),
        "phases": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
