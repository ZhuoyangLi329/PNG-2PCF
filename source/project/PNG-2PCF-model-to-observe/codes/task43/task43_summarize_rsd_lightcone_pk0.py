#!/usr/bin/env python3
"""Assemble the x25 RSD-lightcone P0 payload using Task 4.3 k policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from task43_pk_common import covariance_diagnostics
from task43_rsd_common import PHASES, atomic_savez, atomic_write_json, sha256_file
from task43_rsd_lightcone_pk0_contract import (
    FIT_KMAX,
    TASK43_REALSPACE_FIT_EDGES,
    TASK43_REALSPACE_OBSERVED_KMIN,
    TASK43_REALSPACE_VOLUME,
    TASK43_WIDE_LRGALL_FIT_EDGES,
    WINDOW_THEORY_KMIN,
    load_task43_realspace_reference,
    observed_fit_kmin,
)
from task43_summarize_pk_lightcone import apply_fit_bin_policy, fit_mask, read_window


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
DEFAULT_MEASUREMENT_DIR = PROJECT_ROOT / "outputs/task43_outputs/rsd_validation/lightcone/pk/x25_fkpP010000"
DEFAULT_COVARIANCE = (
    PROJECT_ROOT
    / "outputs/task43_outputs/rsd_validation/lightcone/pk_covariance/"
    "x25_fkpP010000_fnlcov0_b1cov2p604_sigmas7p566/"
    "task43_rsd_lightcone_pk0_cov_ph000_mesh128_kmax0p300_dk0p002.npz"
)
DEFAULT_OUTPUT_PREFIX = (
    PROJECT_ROOT
    / "outputs/task43_outputs/rsd_validation/lightcone/pk_summary/"
    "task43_rsd_lightcone_pk0_x25_task43realspace_kmin0p005065716_kmax0p10_l0only"
)


def validated_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing product/metadata: {path} / {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    if metadata.get("status") != "pass" or metadata.get("output_sha256") != digest:
        raise RuntimeError(f"unvalidated product: {path}")
    return metadata


def measurement_path(directory: Path, phase: str) -> Path:
    matches = sorted(directory.glob(f"task43_rsd_lightcone_pk0_{phase}_mesh*_kmax0p300_dk0p002.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one P0 measurement for {phase} in {directory}, found {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-dir", type=Path, default=DEFAULT_MEASUREMENT_DIR)
    parser.add_argument("--window-file", type=Path, default=None)
    parser.add_argument("--covariance", type=Path, default=DEFAULT_COVARIANCE)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--kmax-fit", type=float, default=FIT_KMAX)
    parser.add_argument("--geometry-contract", choices=("task43_narrow", "wide_lrgall", "boxsafe_lrgall"), default="task43_narrow")
    parser.add_argument("--fkp-summary", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output, metadata_path = args.output_prefix.with_suffix(".npz"), args.output_prefix.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        if output.is_file() and metadata_path.is_file() and not args.overwrite:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(output):
                print(f"[skip] validated {output}")
                return
        raise FileExistsError(f"partial/immutable summary output exists: {output} / {metadata_path}")

    paths = [measurement_path(args.measurement_dir, phase) for phase in PHASES]
    measurement_meta = [validated_metadata(path) for path in paths]
    stack, shots, zeff, volumes, k_by_phase = [], [], [], [], []
    k_ref: np.ndarray | None = None
    edges_ref: np.ndarray | None = None
    for phase, path in zip(PHASES, paths, strict=True):
        with np.load(path, allow_pickle=False) as payload:
            if str(np.asarray(payload["phase"]).item()) != phase:
                raise RuntimeError(f"phase label mismatch in {path}")
            k = np.asarray(payload["k_obs"], dtype="f8")
            edges = np.asarray(payload["k_edges"], dtype="f8")
            if k_ref is None:
                k_ref, edges_ref = k, edges
            elif not np.allclose(edges, edges_ref, rtol=0.0, atol=1.0e-13):
                raise RuntimeError(f"measurement k edges differ in {path}")
            k_by_phase.append(k)
            stack.append(np.asarray(payload["pk0"], dtype="f8"))
            shots.append(np.asarray(payload["shotnoise"], dtype="f8"))
            zeff.append(float(np.asarray(payload["zeff"]).item()))
            volumes.append(float(np.sum(np.asarray(payload["volume_shell"], dtype="f8"))))
    assert k_ref is not None and edges_ref is not None
    pk_stack_full = np.stack(stack)
    shot_stack_full = np.stack(shots)
    k_by_phase_array = np.stack(k_by_phase)
    k_coordinate_max_abs_delta = float(np.max(np.abs(k_by_phase_array - k_ref[None, :])))
    if k_coordinate_max_abs_delta > 1.0e-4:
        raise RuntimeError(
            f"phase-dependent mode-averaged k coordinates drift by {k_coordinate_max_abs_delta}, "
            "larger than the Task 4.3 mesh-geometry audit limit"
        )
    if not np.allclose(volumes, volumes[0], rtol=0.0, atol=1.0e-6):
        raise RuntimeError(f"phase lightcone geometry volumes differ: {volumes}")
    volume = float(volumes[0])
    leff = volume ** (1.0 / 3.0)
    kmin_observed = observed_fit_kmin(volume)
    reference = load_task43_realspace_reference()
    if args.geometry_contract == "task43_narrow":
        if not np.isclose(volume, TASK43_REALSPACE_VOLUME, rtol=0.0, atol=1.0e-6):
            raise RuntimeError(
                f"RSD lightcone volume {volume} does not match frozen Task 4.3 realspace {TASK43_REALSPACE_VOLUME}"
            )
        if not np.isclose(kmin_observed, TASK43_REALSPACE_OBSERVED_KMIN, rtol=0.0, atol=1.0e-15):
            raise RuntimeError(
                f"RSD lightcone observed kmin {kmin_observed} does not match frozen Task 4.3 realspace "
                f"{TASK43_REALSPACE_OBSERVED_KMIN}"
            )
    else:
        if args.fkp_summary is None:
            raise ValueError("--fkp-summary is required for the wide_lrgall/boxsafe_lrgall geometry contract")
        with np.load(args.fkp_summary, allow_pickle=False) as fkp_payload:
            fkp_volume = float(np.sum(np.asarray(fkp_payload["volume_shell"], dtype="f8")))
            fkp_edges = np.asarray(fkp_payload["z_edges"], dtype="f8")
        if not np.isclose(volume, fkp_volume, rtol=0.0, atol=1.0e-6):
            raise RuntimeError(f"measurement volume {volume} differs from wide FKP volume {fkp_volume}")
        if args.geometry_contract == "boxsafe_lrgall":
            # box-safe 0.4<zobs<0.8 的 FKP 网格：dz=0.01，共 41 个边
            expected_edges = (0.4, 0.8, 41)
        else:
            expected_edges = (0.4, 1.1, 71)
        if not (
            np.isclose(fkp_edges[0], expected_edges[0])
            and np.isclose(fkp_edges[-1], expected_edges[1])
            and fkp_edges.size == expected_edges[2]
        ):
            raise RuntimeError(
                f"{args.geometry_contract} FKP redshift grid changed: {fkp_edges[[0, -1]]}, size={fkp_edges.size}"
            )
    base_mask = fit_mask(k_ref, edges_ref, pk_stack_full, kmin_fit=kmin_observed, kmax_fit=float(args.kmax_fit))
    mask, bin_policy = apply_fit_bin_policy(
        base_mask,
        k_ref,
        policy="desi_png",
        stride=2,
        pivots=[0.01, 0.02],
        factors=[2, 2],
    )
    expected_edges = (
        TASK43_REALSPACE_FIT_EDGES
        if (args.geometry_contract == "task43_narrow" or args.geometry_contract == "boxsafe_lrgall")
        else TASK43_WIDE_LRGALL_FIT_EDGES
    )
    if not np.allclose(edges_ref[mask], expected_edges, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"Task 4.3-derived fit-bin contract changed: {edges_ref[mask]}")

    window_path = Path(args.window_file) if args.window_file is not None else paths[0]
    window_meta = validated_metadata(window_path)
    window = read_window(window_path)
    if window is None:
        raise RuntimeError(f"no window matrix in {window_path}")
    matrix = np.asarray(window["window_matrix"], dtype="f8")
    if matrix.shape[0] != k_ref.size:
        raise RuntimeError(f"window rows {matrix.shape[0]} != measurement bins {k_ref.size}")
    window_fit = matrix[mask]
    theory_k = np.asarray(window["theory_k"], dtype="f8")
    theory_ell = np.asarray(window["theory_ell"], dtype="i8")
    if not np.array_equal(np.unique(theory_ell), [0, 2, 4]):
        raise RuntimeError(f"window theory multipoles are {np.unique(theory_ell)}, expected [0,2,4]")
    if args.geometry_contract == "task43_narrow" and matrix.shape[1] != reference["window_shape"][1]:
        raise RuntimeError(
            f"RSD window theory size {matrix.shape[1]} differs from Task 4.3 realspace "
            f"{reference['window_shape'][1]}"
        )
    theory_counts = np.unique(theory_ell, return_counts=True)[1]
    if not np.all(theory_counts == theory_counts[0]) or matrix.shape[1] != theory_k.size:
        raise RuntimeError(
            f"window theory blocks are inconsistent: matrix={matrix.shape}, counts={theory_counts}"
        )

    covariance_meta = validated_metadata(Path(args.covariance))
    with np.load(args.covariance, allow_pickle=False) as payload:
        cov_edges = np.asarray(payload["k_edges"], dtype="f8")
        covariance_full = np.asarray(payload["covariance_single_realization"], dtype="f8")
    if not np.allclose(cov_edges, edges_ref, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("covariance and measurement k edges differ")
    covariance = covariance_full[np.ix_(mask, mask)]
    diagnostics = covariance_diagnostics(covariance)
    if diagnostics["min_eigenvalue"] <= 0.0:
        raise RuntimeError(f"selected covariance is not SPD: {diagnostics}")

    pk_stack = pk_stack_full[:, mask]
    shot_stack = shot_stack_full[:, mask]
    pk_mean = np.mean(pk_stack, axis=0)
    scatter_covariance = np.cov(pk_stack, rowvar=False, ddof=1)
    theory_kmin = WINDOW_THEORY_KMIN
    atomic_savez(
        output,
        phases=np.asarray(PHASES),
        k_obs=np.asarray(k_ref[mask], dtype="f8"),
        k_obs_by_phase=np.asarray(k_by_phase_array[:, mask], dtype="f8"),
        k_edges=np.asarray(edges_ref[mask], dtype="f8"),
        fit_bin_mask=np.asarray(mask, dtype=bool),
        fit_bin_indices=np.asarray(np.flatnonzero(mask), dtype="i8"),
        pk_mean=np.asarray(pk_mean, dtype="f8"),
        pk_stack=np.asarray(pk_stack, dtype="f8"),
        pk_scatter_covariance=np.asarray(scatter_covariance, dtype="f8"),
        covariance_single_realization=np.asarray(covariance, dtype="f8"),
        covariance=np.asarray(covariance, dtype="f8"),
        shotnoise_mean=np.asarray(np.mean(shot_stack, axis=0), dtype="f8"),
        shotnoise_stack=np.asarray(shot_stack, dtype="f8"),
        window_matrix=np.asarray(window_fit, dtype="f8"),
        theory_k=theory_k,
        theory_ell=theory_ell,
        theory_edges=np.asarray(window.get("theory_edges", np.empty((0, 2))), dtype="f8"),
        theory_slice_start=np.asarray(window.get("theory_slice_start", np.empty(0)), dtype="i8"),
        theory_slice_stop=np.asarray(window.get("theory_slice_stop", np.empty(0)), dtype="i8"),
        kmin_fit_observed=np.asarray(kmin_observed),
        kmax_fit=np.asarray(float(args.kmax_fit)),
        window_theory_kmin=np.asarray(theory_kmin),
        lightcone_volume=np.asarray(volume),
        lightcone_leff=np.asarray(leff),
        zeff=np.asarray(float(np.mean(zeff))),
        p0=np.asarray(10000.0),
        sn0_scale=np.asarray(1.0e4),
    )
    metadata = {
        "task": "task43_summarize_rsd_lightcone_pk0",
        "status": "pass",
        "nphase": len(PHASES),
        "phases": list(PHASES),
        "scope": "radial-LOS RSD lightcone observed P0 only",
        "geometry_contract": str(args.geometry_contract),
        "task43_realspace_contract": {
            "reference_payload": str(reference["path"]),
            "reference_payload_sha256": sha256_file(reference["path"]),
            "reference_checks": reference["checks"],
            "measurement_grid": "k=0.001..0.3001, dk=0.002",
            "observed_fit_kmin": kmin_observed,
            "observed_fit_kmin_definition": "2pi/(sum volume_shell)^(1/3)",
            "kmax_fit": float(args.kmax_fit),
            "bin_policy": bin_policy,
            "fit_edges": expected_edges.tolist(),
            "fit_nbin": int(expected_edges.shape[0]),
            "window_theory_kmin": theory_kmin,
            "window_theory_kmin_definition": "mother-box 2pi/2000; distinct from observed fit kmin",
        },
        "geometry": {
            "volume": volume,
            "leff": leff,
            "phase_invariant": True,
            "k_coordinate_policy": (
                "Task 4.3 realspace convention: common requested bin edges and ph000 mode-averaged k as the "
                "representative coordinate; retain all phase coordinates for audit"
            ),
            "k_coordinate_max_abs_phase_delta": k_coordinate_max_abs_delta,
            "fkp_geometry_reference": None if args.fkp_summary is None else str(args.fkp_summary),
        },
        "zeff_equal_phase_mean": float(np.mean(zeff)),
        "window": {
            "path": str(window_path),
            "sha256": sha256_file(window_path),
            "metadata": window_meta,
            "matrix_shape_fit": list(window_fit.shape),
            "theory_ells": [0, 2, 4],
            "theory_ell_counts": theory_counts.tolist(),
            "theory_k_range": [float(np.min(theory_k)), float(np.max(theory_k))],
            "wide_geometry_note": (
                None
                if args.geometry_contract == "task43_narrow"
                else "theory-grid column count is regenerated from the wider survey mesh and is not forced to the narrow 981-column window"
            ),
        },
        "covariance": {
            "path": str(args.covariance),
            "sha256": sha256_file(args.covariance),
            "metadata": covariance_meta,
            "diagnostics_selected": diagnostics,
            "quoted_posterior": "single-lightcone covariance; never divided by 25",
            "mean_goodness": "Cmean=Csingle/25",
        },
        "measurements": [str(path) for path in paths],
        "measurement_sha256": [sha256_file(path) for path in paths],
        "measurement_metadata": measurement_meta,
        "output_path": str(output),
    }
    metadata["output_sha256"] = sha256_file(output)
    atomic_write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "status": "pass",
                "kmin_observed": kmin_observed,
                "theory_kmin": theory_kmin,
                "nfit": int(np.count_nonzero(mask)),
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
