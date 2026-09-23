#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compile the boxsafe-RSD single-term RIC operator for the P0 rows.

Variant of task43_make_ric_factorized_operator.py with three changes:
* the kernel is the boxsafe-geometry factorized kernel;
* the pk payload / xi target edges come from the boxsafe products;
* the estimator-transfer fiducial is the RSD Kaiser x Lorentzian-FoG monopole
  at the covariance fiducial (fNL=0, b1=2.5, sigma_s=7.5, p_fixed=1) instead
  of the real-space 2.5^2 P_dd, so the transfer is derived on the same theory
  the RSD estimator responds to.

The produced pk_ric_matrix has the shape of the payload geometry window
(15, 981) and acts on the ell=0 theory columns only, exactly like the audited
real-space operator; P2 rows are outside the single-auto-term scope.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
TASK44_DIR = PROJECT_ROOT / "codes" / "task44"
for _path in (CODE_DIR, TASK44_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from task43_make_ric_factorized_operator import (  # noqa: E402
    build_basis,
    full_discrete_xi_basis_on_shells,
    load_minimal_closure_module,
    robust_relative_rms,
    target_projection_weights,
)
from task43_ric_singleterm import (  # noqa: E402
    atomic_savez,
    component_separation_probability,
    load_kernel,
    piecewise_constant_pk_xi_basis,
    shell_average_j0,
    to_jsonable,
    write_json,
)


BOXSAFE_ROOT = PROJECT_ROOT / "outputs/task43_outputs/rsd_validation/lightcone_boxsafe_zobs0p4_0p8"
DEFAULT_KERNEL_DIR = BOXSAFE_ROOT / "ric_singleterm" / "kernels"
DEFAULT_XI = BOXSAFE_ROOT / "ell2_increment" / "task43_rsd_boxsafe_x25_mean_xi02_s30_350_ds10.npz"
DEFAULT_PK = (
    BOXSAFE_ROOT / "pk_summary"
    / "task43_rsd_boxsafe_lightcone_pk0_x25_kmin0p004291_kmax0p10_l0only_15bin.npz"
)
OUTPUT_DIR = BOXSAFE_ROOT / "ric_singleterm" / "operators"
FIDUCIAL = {"fnl": 0.0, "b1": 2.5, "sigma_s": 7.5, "p_fixed": 1.0}


def rsd_monopole_fiducial(k_dense: np.ndarray, pk_dd_dense: np.ndarray, *, b1: float, sigma_s: float, f_growth: float, nmu: int = 96) -> np.ndarray:
    """Kaiser x Lorentzian-FoG monopole at fNL=0 (no alpha term)."""
    mu, wmu = np.polynomial.legendre.leggauss(int(nmu))
    damping = 1.0 / (1.0 + 0.5 * (np.asarray(k_dense)[:, None] * mu[None, :] * float(sigma_s)) ** 2) ** 2
    m0 = 0.5 * np.sum(wmu[None, :] * damping, axis=1)
    m2 = 0.5 * np.sum(wmu[None, :] * damping * mu[None, :] ** 2, axis=1)
    m4 = 0.5 * np.sum(wmu[None, :] * damping * mu[None, :] ** 4, axis=1)
    return np.asarray(pk_dd_dense) * (float(b1) ** 2 * m0 + 2.0 * float(b1) * float(f_growth) * m2 + float(f_growth) ** 2 * m4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--xi-path", type=Path, default=DEFAULT_XI)
    parser.add_argument("--pk-payload", type=Path, default=DEFAULT_PK)
    parser.add_argument("--boxsize", type=float, default=2000.0)
    parser.add_argument("--kmax", type=float, default=5.0)
    parser.add_argument("--ndense", type=int, default=60000)
    parser.add_argument("--component-batch", type=int, default=256)
    parser.add_argument("--k-batch", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stem = f"task43_ric_factorized_operator_boxsafe_{Path(args.kernel).stem.replace('task43_ric_factorized_boxsafe_', '')}"
    npz_path = Path(args.output_dir) / f"{stem}.npz"
    json_path = Path(args.output_dir) / f"{stem}_audit.json"
    if not args.overwrite and (npz_path.exists() or json_path.exists()):
        raise FileExistsError(f"immutable operator output exists: {npz_path} / {json_path}")

    from task44_rsd_theory import growth_rate_approx

    cache = load_kernel(Path(args.kernel))
    if cache["meta"].get("kernel_kind") != "factorized_positive_octant_radial_auto":
        raise ValueError("该入口只接受 factorized kernel")
    sedges = np.asarray(cache["separation_edges"], dtype="f8")
    outer_exact = np.asarray(cache["outer_counts"], dtype="f8")
    outer_exact /= np.sum(outer_exact)
    component_weight = np.asarray(cache["component_weight"], dtype="f8")
    component_r1 = np.asarray(cache["component_r1"], dtype="f8")
    component_r2 = np.asarray(cache["component_r2"], dtype="f8")
    ncomponent = component_weight.size

    with np.load(args.xi_path, allow_pickle=False) as xi_data:
        target_edges = np.asarray(xi_data["s_edges"], dtype="f8")
        zeff = float(np.asarray(xi_data["zeff_mean"]).item())
    closure = load_minimal_closure_module()
    theory = closure.build_theory_context(zeff, kmax=float(args.kmax), ndense=int(args.ndense), boxsize=float(args.boxsize), cosmology="abacus_c000")
    dense_basis = build_basis(theory)
    f_growth = float(growth_rate_approx(zeff))
    fiducial_dense = rsd_monopole_fiducial(
        theory["k_dense"], dense_basis["pk_dd"],
        b1=FIDUCIAL["b1"], sigma_s=FIDUCIAL["sigma_s"], f_growth=f_growth,
    )

    with np.load(args.pk_payload, allow_pickle=False) as pk_data:
        observed_k_edges = np.asarray(pk_data["k_edges"], dtype="f8")
        theory_k = np.asarray(pk_data["theory_k"], dtype="f8")
        theory_ell = np.asarray(pk_data["theory_ell"], dtype="i8")
        theory_edges = np.asarray(pk_data["theory_edges"], dtype="f8")
        geometry = np.asarray(pk_data["window_matrix"], dtype="f8")
    ell0 = theory_ell == 0
    centers = 0.5 * (sedges[:-1] + sedges[1:])
    j0_observed = shell_average_j0(centers, observed_k_edges[:, 0], observed_k_edges[:, 1]).T
    xi_band = piecewise_constant_pk_xi_basis(centers, theory_edges[ell0])
    row_ratio = np.asarray(cache["outer_row_ratio"], dtype="f8") if "outer_row_ratio" in cache else None
    if row_ratio is None:
        # 重算 analytic outer marginal（kernel 未存 row_ratio 时）
        outer_model = np.zeros_like(outer_exact)
        for start in range(0, ncomponent, int(args.component_batch)):
            stop = min(ncomponent, start + int(args.component_batch))
            h = component_separation_probability(
                component_r1[start:stop], component_r2[start:stop],
                separation_edges=sedges, cosine_edges=cache["cosine_edges"], cosine_cdf=cache["cosine_cdf"],
            )
            outer_model += component_weight[start:stop] @ h
        outer_model /= np.sum(outer_model)
        row_ratio = np.zeros_like(outer_exact)
        valid = outer_model > 0.0
        row_ratio[valid] = outer_exact[valid] / outer_model[valid]
    target_weight = target_projection_weights(sedges, target_edges, row_ratio)

    pk_direct_ell0 = np.zeros((observed_k_edges.shape[0], int(np.sum(ell0))), dtype="f8")
    raw_c_s_q = np.zeros((centers.size, int(np.sum(ell0))), dtype="f8")
    j0_with_ratio = j0_observed * row_ratio[None, :]
    print(f"[components] n={ncomponent} batch={args.component_batch}", flush=True)
    for start in range(0, ncomponent, int(args.component_batch)):
        stop = min(ncomponent, start + int(args.component_batch))
        h = component_separation_probability(
            component_r1[start:stop], component_r2[start:stop],
            separation_edges=sedges, cosine_edges=cache["cosine_edges"], cosine_cdf=cache["cosine_cdf"],
        )
        weight = component_weight[start:stop]
        inner_pk_band = h @ xi_band
        weighted_inner_pk = weight[:, None] * inner_pk_band
        outer_pk = h @ j0_with_ratio.T
        pk_direct_ell0 += outer_pk.T @ weighted_inner_pk
        raw_c_s_q += (h.T * row_ratio[:, None]) @ weighted_inner_pk

    pair_volume = float(cache["meta"]["fkp_fourier_normalisation"]["pair_volume"])
    pk_direct_ell0 *= pair_volume
    pk_hankel_ell0 = pair_volume * (j0_observed @ raw_c_s_q)
    pk_ric_raw = np.zeros_like(geometry)
    pk_hankel_raw = np.zeros_like(geometry)
    pk_ric_raw[:, ell0] = pk_direct_ell0
    pk_hankel_raw[:, ell0] = pk_hankel_ell0

    geometry_factorized = np.zeros_like(geometry)
    geometry_factorized[:, ell0] = pair_volume * ((j0_observed * outer_exact[None, :]) @ xi_band)
    fiducial_vector = np.zeros_like(theory_k)
    fiducial_vector[ell0] = theory["task41"].interp_logk(theory_k[ell0], theory["k_dense"], fiducial_dense)
    fiducial_vector[theory_k < 2.0 * np.pi / float(args.boxsize)] = 0.0
    geometry_fiducial = geometry @ fiducial_vector
    factorized_fiducial = geometry_factorized @ fiducial_vector
    estimator_transfer = np.ones_like(geometry_fiducial)
    nonzero = factorized_fiducial != 0.0
    estimator_transfer[nonzero] = geometry_fiducial[nonzero] / factorized_fiducial[nonzero]
    pk_ric = estimator_transfer[:, None] * pk_ric_raw
    pk_hankel = estimator_transfer[:, None] * pk_hankel_raw

    common = (theory_k >= 2.0 * np.pi / float(args.boxsize)) & (theory_k <= 0.12) & ell0
    matrix_mask = np.broadcast_to(common[None, :], geometry.shape)
    hankel_diff = pk_ric[:, ell0] - pk_hankel[:, ell0]
    audit = {
        "task": "task43_make_ric_factorized_operator_boxsafe",
        "status": "done",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": {
            "approximation": "single IC^(rad,rad) auto term, ell=0 theory columns, P0 observed rows",
            "geometry": "boxsafe 0.4<zobs<0.8 radial-LOS lightcone (RSD selection)",
            "p2_rows": "outside single-auto-term scope (pair-(s,mu) machinery)",
        },
        "fiducial": {**FIDUCIAL, "f_growth": f_growth, "note": "RSD Kaiser x Lorentzian-FoG monopole; matches the jaxpower covariance fiducial"},
        "inputs": {"kernel": str(args.kernel), "xi": str(args.xi_path), "pk_payload": str(args.pk_payload)},
        "operator_shapes": {"pk_ric_matrix": list(pk_ric.shape)},
        "tests": {
            "hankel_consistency": {
                "relative_frobenius": float(np.linalg.norm(hankel_diff) / max(np.linalg.norm(pk_ric[:, ell0]), 1e-300)),
            },
            "geometry_window_reconstruction": {
                "relative_rms": robust_relative_rms(geometry_factorized, geometry, matrix_mask),
                "fiducial_relative_rms_before_transfer": float(
                    np.linalg.norm(factorized_fiducial - geometry_fiducial) / np.linalg.norm(geometry_fiducial)
                ),
                "fiducial_relative_rms_after_transfer": float(
                    np.linalg.norm(estimator_transfer * factorized_fiducial - geometry_fiducial)
                    / np.linalg.norm(geometry_fiducial)
                ),
                "transfer_policy": "fixed fNL=0,b1=2.5,sigma_s=7.5 RSD monopole; no data fit",
            },
            "matrix_norm_ratio_vs_geometry_window": float(
                np.linalg.norm(pk_ric[:, ell0]) / np.linalg.norm(geometry[:, ell0])
            ),
        },
        "paths": {"operator_npz": str(npz_path), "audit_json": str(json_path)},
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    atomic_savez(
        npz_path,
        kernel_path=np.asarray(str(args.kernel)),
        pk_theory_k=theory_k,
        pk_theory_ell=theory_ell,
        pk_ric_matrix=np.asarray(pk_ric, dtype="f8"),
        pk_ric_matrix_hankel=np.asarray(pk_hankel, dtype="f8"),
        pk_geometry_factorized_matrix=np.asarray(geometry_factorized, dtype="f8"),
        pk_estimator_transfer=np.asarray(estimator_transfer, dtype="f8"),
        meta_json=np.asarray(json.dumps(to_jsonable(audit), sort_keys=True)),
    )
    write_json(json_path, audit)
    print(f"[write] {npz_path}")
    print(f"[audit] hankel_rel={audit['tests']['hankel_consistency']['relative_frobenius']:.3e} "
          f"geometry_rel={audit['tests']['geometry_window_reconstruction']['relative_rms']:.3e} "
          f"norm_ratio={audit['tests']['matrix_norm_ratio_vs_geometry_window']:.3e}")


if __name__ == "__main__":
    main()
