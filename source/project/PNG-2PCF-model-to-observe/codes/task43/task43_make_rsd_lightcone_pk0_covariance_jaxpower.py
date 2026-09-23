#!/usr/bin/env python3
"""Build the Task 4.3-style survey-window covariance for RSD lightcone P0.

This is the real-space Task 4.3 P(k) covariance construction with an RSD
fiducial theory containing input multipoles ell=0,2,4.  It deliberately keeps
the same measured k grid and WW/WS/SS subsample correction as Task 4.3.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np

from task43_build_rsd_lightcone_random import fkp_path
from task43_make_pk_covariance_jaxpower import compute_covariance_mesh2_spectrum_compat
from task43_measure_rsd_lightcone_pk0_jaxpower import (
    DEFAULT_MANIFEST,
    load_catalog,
    read_jsonl,
    select_row,
)
from task43_pk_common import (
    column_edges,
    covariance_diagnostics,
    infer_mesh_attrs_from_catalogs,
    make_k_edges,
    mesh_attrs_for_jaxpower,
)
from task43_rsd_common import BOX_SIZE_MPC_H, P_FIXED, atomic_savez, atomic_write_json, sha256_file
from task43_rsd_lightcone_pk0_contract import TASK43_REALSPACE_FIT_EDGES, TASK43_WIDE_LRGALL_FIT_EDGES
from task43_rsd_model import DELTA_C, build_cache
from task43_theory_template import build_template_arrays, load_task41


OUTPUT_DIR = Path(
    "/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/"
    "rsd_validation/lightcone/pk_covariance"
)


def interp_logk(k: np.ndarray, base_k: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(np.log(np.asarray(k, dtype="f8")), np.log(base_k), values)


def rsd_multipoles(
    k: np.ndarray,
    *,
    template: dict[str, np.ndarray],
    f_growth: float,
    fnl: float,
    b1: float,
    sigma_s: float,
    p_fixed: float,
    nmu: int = 256,
) -> dict[int, np.ndarray]:
    kval = np.asarray(k, dtype="f8")
    alpha = interp_logk(kval, np.asarray(template["k"]), np.asarray(template["alpha"]))
    pk_dd = interp_logk(kval, np.asarray(template["k"]), np.asarray(template["pk_dd"]))
    bphi = 2.0 * DELTA_C * (float(b1) - float(p_fixed))
    amplitude = float(b1) + float(fnl) * bphi * alpha
    mu, wmu = np.polynomial.legendre.leggauss(int(nmu))
    mu2 = mu**2
    damping = 1.0 / (1.0 + 0.5 * (kval[:, None] * mu[None, :] * float(sigma_s)) ** 2) ** 2
    pkmu = pk_dd[:, None] * (amplitude[:, None] + float(f_growth) * mu2[None, :]) ** 2 * damping
    result: dict[int, np.ndarray] = {}
    for ell in (0, 2, 4):
        coeff = np.zeros(ell + 1, dtype="f8")
        coeff[ell] = 1.0
        legendre = np.polynomial.legendre.legval(mu, coeff)
        result[ell] = 0.5 * (2 * ell + 1) * np.sum(wmu[None, :] * pkmu * legendre[None, :], axis=1)
    return result


def build_theory_poles(
    k_edges: np.ndarray,
    *,
    zeff: float,
    fnl_cov: float,
    b1_cov: float,
    sigma_s_cov: float,
    p_fixed: float,
    nbar_shot: float,
) -> tuple[Any, dict[str, Any]]:
    from jaxpower import Mesh2SpectrumPole, Mesh2SpectrumPoles

    task41 = load_task41()
    k = 0.5 * (np.asarray(k_edges[:-1]) + np.asarray(k_edges[1:]))
    template_k = np.geomspace(max(1.0e-5, float(np.min(k_edges[k_edges > 0])) * 0.5), 1.0, 5000)
    template, cosmology_meta = build_template_arrays(task41, template_k, z=float(zeff), cosmology="abacus_c000")
    cache = build_cache(
        zeff=float(zeff), boxsize=BOX_SIZE_MPC_H, kmax=3.0, ells=(0, 2), cosmology="abacus_c000"
    )
    with np.load(cache, allow_pickle=False) as payload:
        f_growth = float(np.asarray(payload["f_growth"]).item())
    pks = rsd_multipoles(
        k,
        template=template,
        f_growth=f_growth,
        fnl=float(fnl_cov),
        b1=float(b1_cov),
        sigma_s=float(sigma_s_cov),
        p_fixed=float(p_fixed),
    )
    shot = 1.0 / float(nbar_shot)
    edge_pairs = column_edges(k_edges)
    poles = []
    for ell in (0, 2, 4):
        signal = np.asarray(pks[ell], dtype="f8")
        shot_array = np.full_like(signal, shot) if ell == 0 else np.zeros_like(signal)
        poles.append(
            Mesh2SpectrumPole(
                k=k,
                k_edges=edge_pairs,
                num_raw=signal + shot_array,
                num_shotnoise=shot_array,
                norm=np.ones_like(signal),
                ell=ell,
            )
        )
    metadata = {
        "model": "continuous local-LOS Kaiser x squared-Lorentzian-FoG PNG multipoles",
        "theory_reference": "Task 4.3 realspace P(k) covariance grid with Task 4.3.2 RSD factor only",
        "theory_ells": [0, 2, 4],
        "zeff": float(zeff),
        "f_growth": f_growth,
        "fnl_cov": float(fnl_cov),
        "b1_cov": float(b1_cov),
        "sigma_s_cov": float(sigma_s_cov),
        "p_fixed": float(p_fixed),
        "fog": "[1 + (k mu sigma_s)^2 / 2]^-2",
        "nbar_shot": float(nbar_shot),
        "shot_noise_1_over_nbar": float(shot),
        "cosmology_meta": cosmology_meta,
        "theory_cache": str(cache),
        "theory_cache_sha256": sha256_file(cache),
        "pk_min_by_ell": {str(ell): float(np.min(pks[ell])) for ell in (0, 2, 4)},
        "pk_max_by_ell": {str(ell): float(np.max(pks[ell])) for ell in (0, 2, 4)},
    }
    return Mesh2SpectrumPoles(poles, ells=(0, 2, 4)), metadata


def output_prefix(*, phase: str, tag: str, meshsize: int, kmax: float, dk: float) -> Path:
    ktag = f"kmax{float(kmax):.3f}_dk{float(dk):.3f}".replace(".", "p")
    return OUTPUT_DIR / tag / f"task43_rsd_lightcone_pk0_cov_{phase}_mesh{int(meshsize)}_{ktag}"


def extract_p0_block(
    matrix: np.ndarray, *, nk: int, theory_ells: tuple[int, ...] = (0, 2, 4)
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract P0 x P0 from JAXpower's ordered multipole covariance."""

    values = np.asarray(matrix, dtype="f8")
    if not theory_ells or int(theory_ells[0]) != 0:
        raise ValueError(f"theory multipoles must start with ell=0, got {theory_ells}")
    if values.shape == (int(nk), int(nk)):
        block = values
        operation = "full covariance already matches P0 bins"
    elif values.shape == (int(nk) * len(theory_ells), int(nk) * len(theory_ells)):
        block = values[: int(nk), : int(nk)]
        operation = "selected first nk x nk block for ell=0 from ordered ell=(0,2,4) covariance"
    else:
        raise ValueError(
            f"unexpected RSD multipole covariance shape {values.shape}; nk={nk}, theory_ells={theory_ells}"
        )
    block = 0.5 * (block + block.T)
    return block, {
        "source_shape": [int(value) for value in values.shape],
        "selected_shape": [int(value) for value in block.shape],
        "theory_ells_order": [int(ell) for ell in theory_ells],
        "operation": operation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--phase", default="ph000")
    parser.add_argument("--tag", default="x25_fkpP010000_fnlcov0_b1cov2p604_sigmas7p566")
    parser.add_argument("--meshsize", type=int, default=128)
    parser.add_argument("--mesh-pad", type=float, default=400.0)
    parser.add_argument("--kmin", type=float, default=0.001)
    parser.add_argument("--kmax", type=float, default=0.3001)
    parser.add_argument("--dk", type=float, default=0.002)
    parser.add_argument("--max-data", type=int, default=50000)
    parser.add_argument("--max-random", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=430520)
    parser.add_argument("--fnl-cov", type=float, default=0.0)
    parser.add_argument("--b1-cov", type=float, default=2.604)
    parser.add_argument("--sigma-s-cov", type=float, default=7.566)
    parser.add_argument("--fit-contract", choices=("task43_narrow", "wide_lrgall", "boxsafe_lrgall"), default="task43_narrow")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.phase not in tuple(row["phase"] for row in read_jsonl(args.manifest)):
        raise ValueError(f"phase {args.phase} not present in manifest")
    row = select_row(read_jsonl(args.manifest), str(args.phase))
    prefix = output_prefix(
        phase=row["phase"], tag=str(args.tag), meshsize=int(args.meshsize), kmax=float(args.kmax), dk=float(args.dk)
    )
    output, metadata_path = prefix.with_suffix(".npz"), prefix.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        if output.is_file() and metadata_path.is_file() and not args.overwrite:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(output):
                print(f"[skip] validated {output}")
                return
        raise FileExistsError(f"partial/immutable covariance output exists: {output} / {metadata_path}")

    import jax

    jax.config.update("jax_enable_x64", True)
    started = time.perf_counter()
    summary_path = fkp_path(row)
    with np.load(summary_path, allow_pickle=False) as payload:
        summary = {key: np.asarray(payload[key]) for key in payload.files}
    data, data_meta = load_catalog(
        Path(row["lightcone_catalog_path"]),
        summary=summary,
        role="data",
        maximum=args.max_data,
        seed=int(args.seed) + 1,
        rescale_subsample=False,
    )
    randoms, random_meta = load_catalog(
        Path(row["lightcone_random_path"]),
        summary=summary,
        role="random",
        maximum=args.max_random,
        seed=int(args.seed) + 2,
        rescale_subsample=False,
    )
    mesh_meta = infer_mesh_attrs_from_catalogs(
        [data, randoms], meshsize=int(args.meshsize), pad=float(args.mesh_pad)
    )
    k_edges = make_k_edges(float(args.kmin), float(args.kmax), float(args.dk))
    nbar_shot = float(np.mean(np.asarray(summary["nbar"], dtype="f8")))
    theory, theory_meta = build_theory_poles(
        k_edges,
        zeff=float(np.asarray(summary["zeff"]).item()),
        fnl_cov=float(args.fnl_cov),
        b1_cov=float(args.b1_cov),
        sigma_s_cov=float(args.sigma_s_cov),
        p_fixed=P_FIXED,
        nbar_shot=nbar_shot,
    )

    def get_data_randoms() -> dict[str, dict[str, np.ndarray]]:
        return {"data": data, "randoms": randoms}

    result = compute_covariance_mesh2_spectrum_compat(
        get_data_randoms, theory=theory, mattrs=mesh_attrs_for_jaxpower(mesh_meta)
    )
    parts = result["raw_parts"]
    if len(parts) != 3:
        raise RuntimeError(f"expected WW/WS/SS covariance pieces, got {len(parts)}")
    raw_parts = {
        name: np.asarray(part.value(), dtype="f8")
        for name, part in zip(("WW", "WS", "SS"), parts, strict=True)
    }
    data_count_scale = float(data_meta["n_total"]) / float(data_meta["n_used"])
    factors = {"WW": 1.0, "WS": 1.0 / data_count_scale, "SS": 1.0 / data_count_scale**2}
    corrected = {name: raw_parts[name] * factors[name] for name in ("WW", "WS", "SS")}
    covariance_full = sum(corrected.values())
    covariance_full = 0.5 * (covariance_full + covariance_full.T)
    k = np.asarray(theory.get(0).coords("k"), dtype="f8")
    edge_pairs = np.asarray(theory.get(0).edges("k"), dtype="f8")
    covariance, p0_block_meta = extract_p0_block(
        covariance_full, nk=int(k.size), theory_ells=(0, 2, 4)
    )
    corrected_p0 = {
        name: extract_p0_block(corrected[name], nk=int(k.size), theory_ells=(0, 2, 4))[0]
        for name in ("WW", "WS", "SS")
    }
    diagnostics = covariance_diagnostics(covariance)
    numerical_psd_tolerance = 1.0e-12 * float(diagnostics["max_eigenvalue"])
    numerical_psd_gate = bool(float(diagnostics["min_eigenvalue"]) >= -numerical_psd_tolerance)
    psd_repair_record: dict[str, Any] = {"applied": False}
    if not numerical_psd_gate:
        # 高条件数下的浮点舍入可产生 ~1e-12 相对量级的负特征值（真实错误应为 O(1) 相对量级）。
        # 对相对违例 <=1e-10 的情况做特征值下限修复并记录元数据；更大违例仍是硬错误。
        relative_violation = -float(diagnostics["min_eigenvalue"]) / float(diagnostics["max_eigenvalue"])
        if relative_violation > 1.0e-10:
            raise RuntimeError(f"full-grid P0 covariance is not numerically positive semidefinite: {diagnostics}")
        symmetric = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        floor = 1.0e-12 * float(eigenvalues.max())
        n_floored = int(np.count_nonzero(eigenvalues < floor))
        eigenvalues = np.maximum(eigenvalues, floor)
        covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
        covariance = 0.5 * (covariance + covariance.T)
        repaired_diagnostics = covariance_diagnostics(covariance)
        psd_repair_record = {
            "applied": True,
            "relative_violation": float(relative_violation),
            "floor_eigenvalue": float(floor),
            "n_floored": n_floored,
            "diagnostics_before": diagnostics,
            "diagnostics_after": repaired_diagnostics,
        }
        print(f"[warn] numerical PSD repair applied: {psd_repair_record['relative_violation']:.3e} relative, n_floored={n_floored}")
        diagnostics = repaired_diagnostics
        numerical_psd_gate = True
    # contract 决定导出的 fit-block：
    # - wide_lrgall: 16-bin（observed kmin=0.00327 < [0.003,0.005] bin 的模式平均 k）
    # - boxsafe_lrgall: 15-bin（kmin=0.00429 > 该 bin 模式平均 k=0.00378，低 k bin 不被支持）
    # - task43_narrow: 15-bin 冻结口径
    fit_edges = (
        TASK43_REALSPACE_FIT_EDGES
        if args.fit_contract in ("task43_narrow", "boxsafe_lrgall")
        else TASK43_WIDE_LRGALL_FIT_EDGES
    )
    fit_indices = []
    for expected in fit_edges:
        matches = np.flatnonzero(np.all(np.isclose(edge_pairs, expected[None, :], rtol=0.0, atol=1.0e-12), axis=1))
        if matches.size != 1:
            raise RuntimeError(f"could not locate Task 4.3 fit edge {expected} in covariance grid")
        fit_indices.append(int(matches[0]))
    fit_indices_array = np.asarray(fit_indices, dtype="i8")
    covariance_fit = covariance[np.ix_(fit_indices_array, fit_indices_array)]
    diagnostics_fit = covariance_diagnostics(covariance_fit)
    if diagnostics_fit["min_eigenvalue"] <= 0.0:
        raise RuntimeError(f"selected P0 fit covariance is not positive definite: {diagnostics_fit}")
    atomic_savez(
        output,
        k_obs=k,
        k_edges=edge_pairs,
        covariance=covariance,
        covariance_single_realization=covariance,
        covariance_full=covariance_full,
        covariance_p0_WW=corrected_p0["WW"],
        covariance_p0_WS=corrected_p0["WS"],
        covariance_p0_SS=corrected_p0["SS"],
        covariance_contract_fit=covariance_fit,
        covariance_contract_fit_indices=fit_indices_array,
        covariance_full_WW=corrected["WW"],
        covariance_full_WS=corrected["WS"],
        covariance_full_SS=corrected["SS"],
        covariance_full_uncorrected_WW=raw_parts["WW"],
        covariance_full_uncorrected_WS=raw_parts["WS"],
        covariance_full_uncorrected_SS=raw_parts["SS"],
        data_count_scale=np.asarray(data_count_scale),
        phase=np.asarray(row["phase"]),
        zeff=np.asarray(float(np.asarray(summary["zeff"]).item())),
        b1_cov=np.asarray(float(args.b1_cov)),
        fnl_cov=np.asarray(float(args.fnl_cov)),
        sigma_s_cov=np.asarray(float(args.sigma_s_cov)),
    )
    metadata = {
        "task": "task43_make_rsd_lightcone_pk0_covariance_jaxpower",
        "status": "pass",
        "phase": row["phase"],
        "covariance_kind": "Task 4.3 jaxpower Gaussian survey-window P0 covariance with RSD ell=0,2,4 theory",
        "grid": {"kmin": float(args.kmin), "kmax": float(args.kmax), "dk": float(args.dk)},
        "mesh": mesh_meta,
        "theory": theory_meta,
        "data": data_meta,
        "random": random_meta,
        "subsample_correction": {
            "data_count_scale": data_count_scale,
            "factors": factors,
            "definition": "WW + WS/N_subscale + SS/N_subscale^2, exactly as Task 4.3 realspace P(k)",
        },
        "p0_block": p0_block_meta,
        "diagnostics": diagnostics,
        "full_grid_numerical_psd": {
            "gate": numerical_psd_gate,
            "absolute_negative_eigenvalue_tolerance": numerical_psd_tolerance,
            "numerical_psd_repair": psd_repair_record,
            "note": "Task 4.3 fits only the frozen k<0.10 sub-block; full 0.001..0.3001 grid may be singular at machine precision",
        },
        "fit_contract": {
            "name": str(args.fit_contract),
            "indices": fit_indices,
            "edges": fit_edges.tolist(),
            "nbin": int(fit_edges.shape[0]),
            "diagnostics": diagnostics_fit,
            "strict_spd_gate": True,
        },
        "quoted_posterior": "single-lightcone covariance; never divided by 25",
        "mean_goodness": "Cmean=Csingle/25",
        "fkp_summary": str(summary_path),
        "fkp_summary_sha256": sha256_file(summary_path),
        "elapsed_sec": float(time.perf_counter() - started),
        "output_path": str(output),
    }
    metadata["output_sha256"] = sha256_file(output)
    atomic_write_json(metadata_path, metadata)
    print(json.dumps({"status": "pass", "diagnostics": diagnostics, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
