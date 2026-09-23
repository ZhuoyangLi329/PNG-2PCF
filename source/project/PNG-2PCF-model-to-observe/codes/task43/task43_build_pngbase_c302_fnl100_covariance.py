#!/usr/bin/env python3
"""Build phase-matched c302 Gaussian jaxpower P/xi covariance at fNL_cov=100."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from task43_apply_rr_smu_deconvolution_to_covariance import edge_pairs
from task43_make_jaxpower_lightcone_covariance import (
    build_theory_poles as build_real_theory,
    infer_mesh_attrs,
    make_edges,
)
from task43_measure_rsd_lightcone_pk0_jaxpower import load_catalog
from task43_pngbase_pseudolc_common import (
    PHASES,
    P0_FKP,
    PNG_COV_ROOT,
    fkp_path,
    manifest_path,
    pk_path,
    png_covariance_path,
    xi_path,
)
from task43_rsd_common import atomic_savez, atomic_write_json, sha256_file
from task43_run_lightcone_joint_baomask_v1 import (
    REAL_P_PAYLOAD,
    RSD_P_PAYLOAD,
    lightcone_xi_primary_mask,
    select_pk_bins,
)
from task44_make_lrg2_jaxpower_covariance import build_theory_poles as build_rsd_theory


FNL_COV = 100.0
B1_COV = 2.5
SIGMA_S_COV = 7.5
P_FIXED = 1.0
MESH_SIZE = 64
MAX_DATA = 50000
MAX_RANDOM = 100000
WINDOW_SMAX = 3600.0
WINDOW_DS = 2.0
S_EDGES = np.arange(30.0, 350.0 + 5.0, 10.0, dtype="f8")


def read_row(space: str, phase: str) -> dict[str, Any]:
    rows = [json.loads(line) for line in manifest_path("c302", space).read_text().splitlines() if line.strip()]
    matches = [row for row in rows if row["phase"] == phase]
    if len(matches) != 1:
        raise RuntimeError(f"expected one c302 {space} {phase} manifest row, found {len(matches)}")
    row = matches[0]
    if float(row["injected_fnl"]) != FNL_COV or row["cosmology"] != "c302":
        raise RuntimeError(f"wrong covariance target in manifest row: {row}")
    return row


def phase_prefix(space: str, phase: str) -> Path:
    return PNG_COV_ROOT / "phase" / f"task43_pngbase_c302_{phase}_{space}_jaxpower_fnlcov100_mesh64"


def covariance_diagnostics(covariance: np.ndarray) -> dict[str, Any]:
    covariance = 0.5 * (np.asarray(covariance, dtype="f8") + np.asarray(covariance, dtype="f8").T)
    diagonal = np.diag(covariance)
    scale = np.sqrt(np.clip(diagonal, 0.0, None))
    correlation = covariance / np.outer(scale, scale)
    eigenvalues = np.linalg.eigvalsh(correlation)
    return {
        "shape": list(covariance.shape),
        "diag_min": float(np.min(diagonal)),
        "diag_max": float(np.max(diagonal)),
        "correlation_eigenvalue_min": float(eigenvalues[0]),
        "correlation_eigenvalue_max": float(eigenvalues[-1]),
        "correlation_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
    }


def fkp_weight(redshift: np.ndarray, summary: dict[str, np.ndarray]) -> np.ndarray:
    edges = np.asarray(summary["z_edges"], dtype="f8")
    values = np.asarray(summary["fkp_weights"], dtype="f8")
    index = np.clip(np.searchsorted(edges, redshift, side="right") - 1, 0, values.size - 1)
    return values[index]


def rr_inverse(row: dict[str, Any], summary: dict[str, np.ndarray], *, space: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Average the 25 measured RR blocks after the jaxpower FKP normalization."""
    from lsstypes.types import Count2, compute_RR2_window
    from pycorr.twopoint_counter import BaseTwoPointCounter

    measured_xi = xi_path("c302", row["phase"], space)
    cache = measured_xi.parent / "pair_cache" / measured_xi.stem
    rr_paths = sorted(cache.glob("RR_*.npy"))
    if len(rr_paths) != 25:
        raise RuntimeError(f"expected 25 cached RR blocks in {cache}, found {len(rr_paths)}")
    with np.load(Path(row["lightcone_random_path"]), allow_pickle=False) as random:
        z = np.asarray(random["Z"], dtype="f8")
        base_weight = np.asarray(random["WEIGHT"], dtype="f8")
        random_index = np.asarray(random["RANDOM_INDEX"], dtype="i8")
    nx = np.asarray(summary["nbar"], dtype="f8")
    nx_index = np.clip(np.searchsorted(summary["z_edges"], z, side="right") - 1, 0, nx.size - 1)
    nx_all = nx[nx_index]
    weight_all = base_weight * fkp_weight(z, summary)
    ndata = int(np.load(Path(row["lightcone_catalog_path"]), allow_pickle=False)["Z"].size)
    scaled_values, block_meta = [], []
    reference_edges: tuple[np.ndarray, np.ndarray] | None = None
    for index, rr_path in enumerate(rr_paths):
        rr = BaseTwoPointCounter.load(str(rr_path))
        mask = random_index == index
        weight, nx_block = weight_all[mask], nx_all[mask]
        nx_w2 = nx_block**2 * weight**2
        fkp_norm = float((np.sum(nx_block) / ndata) * np.sum(nx_w2**2) / np.sum(nx_w2) ** 2)
        scaled_values.append(np.asarray(rr.wcounts, dtype="f8") / (float(rr.wnorm) * fkp_norm))
        block_meta.append({"index": index, "wnorm": float(rr.wnorm), "fkp_norm": fkp_norm})
        current_edges = (np.asarray(rr.edges[0], dtype="f8"), np.asarray(rr.edges[1], dtype="f8"))
        if reference_edges is None:
            reference_edges = current_edges
        elif not all(np.array_equal(lhs, rhs) for lhs, rhs in zip(reference_edges, current_edges, strict=True)):
            raise RuntimeError("RR cache bin edges differ between random blocks")
    assert reference_edges is not None
    rr_value = np.mean(np.stack(scaled_values), axis=0)
    rr = Count2(
        counts=rr_value,
        norm=np.ones_like(rr_value),
        s=0.5 * (reference_edges[0][:-1] + reference_edges[0][1:]),
        s_edges=edge_pairs(reference_edges[0]),
        mu=0.5 * (reference_edges[1][:-1] + reference_edges[1][1:]),
        mu_edges=edge_pairs(reference_edges[1]),
        coords=["s", "mu"],
    )
    ells = (0,) if space == "real" else (0, 2)
    matrix = np.asarray(
        compute_RR2_window(rr, edges=edge_pairs(S_EDGES), ells=ells, ellsin=ells, kind="RR", resolution=1).value(),
        dtype="f8",
    )
    inverse = np.linalg.inv(matrix)
    return inverse, {
        "source": "arithmetic mean of the 25 exact pycorr RR caches used by the c302 measurement",
        "normalization": "per block RR.wcounts / (RR.wnorm * fkp_norm_nx_window)",
        "blocks": block_meta,
        "ells": list(ells),
        "condition_number": float(np.linalg.cond(matrix)),
        "matrix_diag_min": float(np.min(np.diag(matrix))),
        "matrix_diag_max": float(np.max(np.diag(matrix))),
        "cache_paths": [str(path) for path in rr_paths],
    }


def run_phase(space: str, phase: str, *, overwrite: bool) -> None:
    prefix = phase_prefix(space, phase)
    output, metadata_path = prefix.with_suffix(".npz"), prefix.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        if not overwrite and output.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(output):
                print(json.dumps({"status": "resumed", "output": str(output)}), flush=True)
                return
        raise FileExistsError(f"partial or immutable phase covariance exists: {output} / {metadata_path}")

    import jax
    jax.config.update("jax_enable_x64", True)
    from jaxpower import (
        BinMesh2CorrelationPoles,
        FKPField,
        ParticleField,
        compute_fkp2_covariance_window,
        compute_spectrum2_covariance,
        interpolate_window_function,
    )
    from jaxpower.cov2 import matrix_project_to_correlation

    started = time.perf_counter()
    row = read_row(space, phase)
    with np.load(fkp_path("c302", phase, space), allow_pickle=False) as payload:
        summary = {key: np.asarray(payload[key]) for key in payload.files}
    if not np.isclose(float(summary["p0"]), P0_FKP):
        raise RuntimeError("phase FKP table is not P0=10000")
    data, data_meta = load_catalog(
        Path(row["lightcone_catalog_path"]), summary=summary, role="data", maximum=MAX_DATA,
        seed=20261020 + 100 * PHASES.index(phase) + (0 if space == "real" else 10), rescale_subsample=True,
    )
    random, random_meta = load_catalog(
        Path(row["lightcone_random_path"]), summary=summary, role="random", maximum=MAX_RANDOM,
        seed=20261021 + 100 * PHASES.index(phase) + (0 if space == "real" else 10), rescale_subsample=True,
    )
    data_xyz, random_xyz = np.asarray(data["POSITION"]), np.asarray(random["POSITION"])
    mattrs, mesh_meta = infer_mesh_attrs([data_xyz, random_xyz], meshsize=MESH_SIZE, pad=400.0)
    field = FKPField(
        ParticleField(data_xyz, weights=np.asarray(data["INDWEIGHT"]), attrs=mattrs, exchange=False),
        ParticleField(random_xyz, weights=np.asarray(random["INDWEIGHT"]), attrs=mattrs, exchange=False),
    )
    window_ells = (0,) if space == "real" else (0, 2, 4, 6, 8)
    window_bin = BinMesh2CorrelationPoles(
        mattrs, edges=make_edges(0.0, WINDOW_SMAX, WINDOW_DS), ells=window_ells, basis="bessel"
    )
    window = compute_fkp2_covariance_window(field, bin=window_bin, los="local", resampler="cic", interlacing=1)
    coords = np.logspace(-2.0, 8.0, 8192)
    window = window.map(lambda value: interpolate_window_function(value, coords=coords), level=1)
    kmin = 0.001 if space == "real" else 2.0 * np.pi / 2000.0
    k_edges = make_edges(kmin, 3.0001, 0.002)
    nbar_shot = float(np.mean(summary["nbar"]))
    if space == "real":
        poles, theory_meta = build_real_theory(
            k_edges=k_edges, zeff=float(summary["zeff"]), b1_cov=B1_COV, fnl_cov=FNL_COV,
            p_fixed=P_FIXED, sn0_fixed=0.0, nbar_shot=nbar_shot, cosmology="abacus_c000",
        )
    else:
        poles, theory_meta = build_rsd_theory(
            k_edges=k_edges, zeff=float(summary["zeff"]), b1_cov=B1_COV, fnl_cov=FNL_COV,
            p_fixed=P_FIXED, sigma_s_cov=SIGMA_S_COV, nbar_shot=nbar_shot,
            cosmology="abacus_c000", theory_ells=(0, 2, 4), fog_model="lorentzian",
        )
    raw_parts = compute_spectrum2_covariance(window, poles, flags=("smooth", "fftlog"), return_type="list")
    parts = {name: np.asarray(value.value(), dtype="f8") for name, value in zip(("WW", "WS", "SS"), raw_parts, strict=True)}
    data_scale = float(data_meta["n_total"]) / float(data_meta["n_used"])
    covariance_pk = parts["WW"] + parts["WS"] / data_scale + parts["SS"] / data_scale**2
    covariance_pk = 0.5 * (covariance_pk + covariance_pk.T)
    projection = np.asarray(matrix_project_to_correlation(S_EDGES, poles), dtype="f8")
    covariance_xi_raw = projection @ covariance_pk @ projection.T
    rinv, rr_meta = rr_inverse(row, summary, space=space)
    nxi = rinv.shape[0]
    covariance_xi = rinv @ covariance_xi_raw[:nxi, :nxi] @ rinv.T
    metadata = {
        "task": "task43_build_pngbase_c302_fnl100_covariance", "stage": "phase", "status": "pass",
        "cosmology": "c302", "phase": phase, "space": space, "fnl_cov": FNL_COV,
        "gaussian_covariance_scope": "PNG-dependent disconnected Gaussian covariance; no connected PNG trispectrum",
        "theory": theory_meta, "data": data_meta, "random": random_meta, "mesh": mesh_meta,
        "window": {"ells": list(window_ells), "basis": "bessel", "smax": WINDOW_SMAX, "ds": WINDOW_DS,
                   "interpolated": True, "los": "local", "resampler": "cic", "interlacing": 1},
        "rr_deconvolution": rr_meta,
        "k_grid": {"kmin": kmin, "kmax": 3.0001, "dk": 0.002, "nk": int(k_edges.size - 1)},
        "subsample_correction": {"data_count_scale": data_scale, "WW": 1.0, "WS": 1.0 / data_scale,
                                 "SS": 1.0 / data_scale**2},
        "diagnostics": {"xi": covariance_diagnostics(covariance_xi)},
        "inputs": {"manifest": str(manifest_path("c302", space)), "fkp": str(fkp_path("c302", phase, space)),
                   "pk": str(pk_path("c302", phase, space)), "xi": str(xi_path("c302", phase, space))},
        "elapsed_sec": float(time.perf_counter() - started), "output_npz": str(output),
    }
    prefix.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        output, k_edges=k_edges, k_centers=0.5 * (k_edges[:-1] + k_edges[1:]),
        covariance_pk=covariance_pk, projection_matrix=projection, covariance_xi_raw=covariance_xi_raw,
        rr_window_inverse=rinv, covariance_xi=covariance_xi, s_edges=S_EDGES,
        s=0.5 * (S_EDGES[:-1] + S_EDGES[1:]), meta_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    metadata["output_sha256"] = sha256_file(output)
    atomic_write_json(metadata_path, metadata)
    print(json.dumps({"status": "pass", "output": str(output), "elapsed_sec": metadata["elapsed_sec"]}), flush=True)


def fit_edges(payload_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(payload_path, allow_pickle=False) as payload:
        _, _, k, edges = select_pk_bins({key: np.asarray(payload[key]) for key in payload.files}, 0.08)
    return k, edges, np.flatnonzero(k >= 0.015 - 1.0e-12)


def selected_fine_indices(centers: np.ndarray, edges: np.ndarray) -> np.ndarray:
    selected = []
    for lo, hi in edges:
        inside = np.flatnonzero((centers >= lo - 1.0e-12) & (centers < hi - 1.0e-12))
        if inside.size != 1:
            raise RuntimeError(f"fit edge [{lo}, {hi}] contains {inside.size} fine covariance bins")
        selected.append(int(inside[0]))
    return np.asarray(selected, dtype="i8")


def phase_fit_covariance(space: str, phase: str) -> tuple[np.ndarray, dict[str, Any]]:
    path = phase_prefix(space, phase).with_suffix(".npz")
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "pass" or metadata.get("output_sha256") != sha256_file(path):
        raise RuntimeError(f"unvalidated phase covariance: {path}")
    with np.load(path, allow_pickle=False) as payload:
        covariance_pk = np.asarray(payload["covariance_pk"], dtype="f8")
        projection = np.asarray(payload["projection_matrix"], dtype="f8")
        rinv = np.asarray(payload["rr_window_inverse"], dtype="f8")
        centers = np.asarray(payload["k_centers"], dtype="f8")
        s = np.asarray(payload["s"], dtype="f8")
    payload_path = REAL_P_PAYLOAD if space == "real" else RSD_P_PAYLOAD
    k, edges, keep_p2 = fit_edges(payload_path)
    selected = selected_fine_indices(centers, edges)
    mask = lightcone_xi_primary_mask(s, smin=50.0)
    if space == "real":
        p_ids = selected
        x_ids = np.flatnonzero(mask)
    else:
        nper = centers.size
        p_ids = np.concatenate((selected, nper + selected[keep_p2]))
        x_ids = np.concatenate((np.flatnonzero(mask), s.size + np.flatnonzero(mask)))
    nxi = rinv.shape[0]
    c_pp = covariance_pk[np.ix_(p_ids, p_ids)]
    c_xp_full = rinv @ (projection[:nxi] @ covariance_pk[:, p_ids])
    c_xx_full = rinv @ (projection[:nxi] @ covariance_pk @ projection[:nxi].T) @ rinv.T
    c_xp, c_xx = c_xp_full[x_ids], c_xx_full[np.ix_(x_ids, x_ids)]
    joint = np.block([[c_pp, c_xp.T], [c_xp, c_xx]])
    joint = 0.5 * (joint + joint.T)
    diagnostics = covariance_diagnostics(joint)
    if diagnostics["correlation_eigenvalue_min"] <= 1.0e-12:
        raise RuntimeError(f"{space} {phase} joint covariance is not strictly SPD: {diagnostics}")
    return joint, {
        "source_npz": str(path), "source_sha256": sha256_file(path), "selected_fine_indices": selected.tolist(),
        "p_ids": p_ids.tolist(), "x_ids": x_ids.tolist(), "diagnostics": diagnostics,
    }


def assemble(*, overwrite: bool) -> None:
    output, metadata_path = png_covariance_path(), png_covariance_path().with_suffix(".json")
    if output.exists() or metadata_path.exists():
        if not overwrite and output.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(output):
                print(json.dumps({"status": "resumed", "output": str(output)}), flush=True)
                return
        raise FileExistsError(f"partial or immutable assembled covariance exists: {output} / {metadata_path}")
    phase_joint: dict[str, dict[str, np.ndarray]] = {"real": {}, "rsd": {}}
    sources: dict[str, Any] = {"real": {}, "rsd": {}}
    for space in ("real", "rsd"):
        for phase in PHASES:
            phase_joint[space][phase], sources[space][phase] = phase_fit_covariance(space, phase)
    real_joint = np.mean(np.stack([phase_joint["real"][phase] for phase in PHASES]), axis=0)
    rsd_joint = np.mean(np.stack([phase_joint["rsd"][phase] for phase in PHASES]), axis=0)
    real_k, _, _ = fit_edges(REAL_P_PAYLOAD)
    rsd_k, _, keep_p2 = fit_edges(RSD_P_PAYLOAD)
    nreal_p, nrsd_p = real_k.size, rsd_k.size + keep_p2.size
    nreal_x = real_joint.shape[0] - nreal_p
    nrsd_x = (rsd_joint.shape[0] - nrsd_p) // 2
    real_pp, real_px, real_xx = real_joint[:nreal_p, :nreal_p], real_joint[:nreal_p, nreal_p:], real_joint[nreal_p:, nreal_p:]
    rsd_pp, rsd_xp, rsd_xx = rsd_joint[:nrsd_p, :nrsd_p], rsd_joint[nrsd_p:, :nrsd_p], rsd_joint[nrsd_p:, nrsd_p:]
    rsd_pp0, rsd_xp0, rsd_xx0 = rsd_pp[:rsd_k.size, :rsd_k.size], rsd_xp[:nrsd_x, :rsd_k.size], rsd_xx[:nrsd_x, :nrsd_x]
    rsd_joint0 = np.block([[rsd_pp0, rsd_xp0.T], [rsd_xp0, rsd_xx0]])
    with np.load(phase_prefix("real", "ph000").with_suffix(".npz"), allow_pickle=False) as payload:
        s = np.asarray(payload["s"], dtype="f8")
    mask = lightcone_xi_primary_mask(s, smin=50.0)
    arrays = {
        "real_k": real_k, "real_s": s, "real_xi_mask": mask, "real_pp": real_pp, "real_xx": real_xx,
        "real_px": real_px, "real_joint": real_joint, "rsd_k": rsd_k, "rsd_p2_keep_indices": keep_p2,
        "rsd_s": s, "rsd_xi_mask": mask, "rsd_pp": rsd_pp, "rsd_xx": rsd_xx, "rsd_xp": rsd_xp,
        "rsd_joint": rsd_joint, "rsd_pp0": rsd_pp0, "rsd_xx0": rsd_xx0, "rsd_xp0": rsd_xp0,
        "rsd_joint0": rsd_joint0,
    }
    old_path = Path(__file__).parents[2] / "outputs/task43_outputs/pngbase_pseudolc_fnl0_fnl100/fits/c302/kmax0p08_smin50/task43_pngbase_pseudolc_joint_baomask80_120_covariance.npz"
    comparisons = {}
    with np.load(old_path, allow_pickle=False) as old:
        for key in ("real_pp", "real_xx", "real_joint", "rsd_pp", "rsd_xx", "rsd_joint"):
            old_cov, new_cov = np.asarray(old[key], dtype="f8"), arrays[key]
            comparisons[key] = {
                "sigma_ratio_new_over_old_min": float(np.min(np.sqrt(np.diag(new_cov) / np.diag(old_cov)))),
                "sigma_ratio_new_over_old_median": float(np.median(np.sqrt(np.diag(new_cov) / np.diag(old_cov)))),
                "sigma_ratio_new_over_old_max": float(np.max(np.sqrt(np.diag(new_cov) / np.diag(old_cov)))),
                "relative_frobenius_delta": float(np.linalg.norm(new_cov - old_cov) / np.linalg.norm(old_cov)),
            }
    metadata = {
        "task": "task43_build_pngbase_c302_fnl100_covariance", "stage": "assemble", "status": "pass",
        "cosmology": "c302", "fnl_cov": FNL_COV, "phases": list(PHASES),
        "phase_covariance_combination": "elementwise arithmetic mean of ph000 and ph001 C_single",
        "covariance_divisor": 1.0, "mean_observable_covariance_policy": "retain representative single-realization errors; do not divide by 2",
        "gaussian_covariance_scope": "PNG-dependent disconnected Gaussian covariance; no connected PNG trispectrum",
        "fiducial_nuisance_policy": {"b1_cov": B1_COV, "sigma_s_cov_rsd": SIGMA_S_COV, "p_fixed": P_FIXED,
                                      "reason": "held at the frozen Task4.3 covariance values so fnl_cov is the intended theory change"},
        "sources": sources, "diagnostics": {key: covariance_diagnostics(value) for key, value in arrays.items() if key.endswith(("pp", "xx", "joint"))},
        "old_fnl0_comparison": comparisons, "old_covariance": str(old_path), "old_covariance_sha256": sha256_file(old_path),
        "output_npz": str(output),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(output, **arrays, phases=np.asarray(PHASES), fnl_cov=np.asarray(FNL_COV), covariance_divisor=np.asarray(1.0))
    metadata["output_sha256"] = sha256_file(output)
    atomic_write_json(metadata_path, metadata)
    print(json.dumps({"status": "pass", "output": str(output), "comparison": comparisons}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("phase", "assemble", "all"))
    parser.add_argument("--space", choices=("real", "rsd"))
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.stage == "phase":
        if args.space is None or args.phase is None:
            parser.error("phase stage requires --space and --phase")
        run_phase(args.space, args.phase, overwrite=args.overwrite)
    elif args.stage == "assemble":
        assemble(overwrite=args.overwrite)
    else:
        for space in ("real", "rsd"):
            for phase in PHASES:
                run_phase(space, phase, overwrite=args.overwrite)
        assemble(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
