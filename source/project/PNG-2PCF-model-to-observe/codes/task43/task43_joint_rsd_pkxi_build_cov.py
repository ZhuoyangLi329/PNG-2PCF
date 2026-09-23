#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the boxsafe-RSD joint P0(k)+xi0(s) covariance from frozen audits.

No new jaxpower run is needed: the audited Task44 builder already persisted the
corrected 4497-mode P-pole covariance (``covariance_pk``) and the P->xi
projection operator (``projection_matrix``) of the exact same run that produced
the audited RR-deconvolved xi0 covariance.  This script recombines those frozen
arrays into the joint 45x45 covariance:

    C_pp = covariance_pk[sel15, sel15]                       (15x15, bridge-gated)
    C_xx = audited rrdeconv ell0 covariance, s>=50           (30x30, bitwise)
    C_px = R^-1 @ (J_ell0 @ covariance_pk[:, sel15])[s>=50]  (30x15)

Pre-registered gates (failure exits 2 and writes nothing):

* internal consistency: J C_pk J^T must reproduce the persisted
  covariance_xi_multipoles (relFro < 1e-10);
* deconvolution reproduction: R^-1 (J C_pk J^T)_ell0 R^-T must reproduce the
  audited rrdeconv covariance_single_realization (relFro < 1e-8);
* pp bridge vs the audited P0 payload 15x15 covariance: median sigma ratio in
  [0.95, 1.05] AND relFro < 0.10 (mesh64 kfund-grid run vs the mesh128 payload
  run; the Task43 realspace joint experiment passed the same gate at 1.041/0.067).

PSD policy follows the realspace joint lesson: relative eigenvalue floors and
relative pinv rcond silently delete the xi-side modes (the joint spectrum spans
~17 decades); only an absolute floor (1e-30) repair is allowed and the raw
spectrum is kept untouched when already SPD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from task43_rsd_common import OUTPUT_ROOT, atomic_savez, atomic_write_json, sha256_file


BOXSAFE_ROOT = OUTPUT_ROOT / "lightcone_boxsafe_zobs0p4_0p8"
RAW_NPZ = (
    BOXSAFE_ROOT / "covariance"
    / "task43_rsd_boxsafe_ph000_jaxpower_raw_xi0_rsdpoles024_win02468_mesh64_p1_b1cov2p50"
    "_sigmas7p50_nran100k_ndata50k_seed20260817_kbox_fkpP010000_s30_350_ds10.npz"
)
DECONV_NPZ = (
    BOXSAFE_ROOT / "covariance"
    / "task43_rsd_boxsafe_ph000_jaxpower_rrdeconv_ell0_rsdpoles024_win02468_mesh64_p1_b1cov2p50"
    "_sigmas7p50_nran100k_ndata50k_seed20260817_rrnran300k_rrseed430420_kbox_fkpP010000_s30_350_ds10.npz"
)
PAYLOAD_NPZ = (
    BOXSAFE_ROOT / "pk_summary"
    / "task43_rsd_boxsafe_lightcone_pk0_x25_kmin0p004291_kmax0p10_l0only_15bin.npz"
)
JOINT_COV_DIR = BOXSAFE_ROOT / "joint_pkxi_s50_350" / "covariance"
JOINT_COV_NPZ = JOINT_COV_DIR / "task43_rsd_joint_cov_boxsafe_mesh64_s50_350.npz"

GATE_INTERNAL_RELFRO = 1.0e-10
GATE_DECONV_RELFRO = 1.0e-8
GATE_PP_SIGMA_RATIO = (0.95, 1.05)
GATE_PP_RELFRO = 0.10
PSD_ABS_FLOOR = 1.0e-30


def rel_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1.0e-300))


def spd_repair(matrix: np.ndarray, *, floor: float) -> tuple[np.ndarray, dict[str, Any]]:
    sym = 0.5 * (matrix + matrix.T)
    evals, evecs = np.linalg.eigh(sym)
    raw_min = float(evals[0])
    if raw_min > 0.0:
        return sym, {"raw_min_eigenvalue": raw_min, "repaired": False, "n_floored": 0}
    floored = np.maximum(evals, float(floor))
    out = (evecs * floored[None, :]) @ evecs.T
    out = 0.5 * (out + out.T)
    return out, {
        "raw_min_eigenvalue": raw_min,
        "repaired": True,
        "floor_eigenvalue": float(floor),
        "n_floored": int(np.count_nonzero(evals < floor)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=JOINT_COV_NPZ)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out_npz = args.output
    out_json = out_npz.with_suffix(".json")
    if (out_npz.exists() or out_json.exists()) and not args.overwrite:
        raise FileExistsError(f"immutable joint covariance output exists: {out_npz}")

    for path in (RAW_NPZ, DECONV_NPZ, PAYLOAD_NPZ):
        if not path.is_file():
            raise FileNotFoundError(path)

    audit: dict[str, Any] = {
        "task": "task43_joint_rsd_pkxi_build_cov",
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in (("raw", RAW_NPZ), ("rrdeconv", DECONV_NPZ), ("pk_payload", PAYLOAD_NPZ))
        },
    }

    with np.load(RAW_NPZ, allow_pickle=False) as data:
        cov_pk = np.asarray(data["covariance_pk"], dtype="f8")
        proj = np.asarray(data["projection_matrix"], dtype="f8")
        cov_xi_multi = np.asarray(data["covariance_xi_multipoles"], dtype="f8")
        raw_meta = json.loads(str(np.asarray(data["meta_json"]).item()))
    with np.load(DECONV_NPZ, allow_pickle=False) as data:
        rinv = np.asarray(data["rr_window_inverse"], dtype="f8")
        aud_xi32 = np.asarray(data["covariance_single_realization"], dtype="f8")
        s_full = np.asarray(data["s"], dtype="f8")
    with np.load(PAYLOAD_NPZ, allow_pickle=False) as data:
        k_obs = np.asarray(data["k_obs"], dtype="f8")
        k_edges = np.asarray(data["k_edges"], dtype="f8")
        payload_cov = np.asarray(data["covariance_single_realization"], dtype="f8")

    k_grid = raw_meta["k_grid"]
    theory_ells = [int(ell) for ell in raw_meta["theory"]["ells"]]
    nper = int(k_grid["nk"])
    if cov_pk.shape != (nper * len(theory_ells),) * 2 or proj.shape != (3 * 32, cov_pk.shape[0]):
        raise RuntimeError(f"unexpected frozen shapes: cov_pk={cov_pk.shape}, proj={proj.shape}")

    # Gate 1: the persisted fine covariance reproduces the persisted xi projection.
    internal = proj @ cov_pk @ proj.T
    internal_relfro = rel_frobenius(internal, cov_xi_multi)
    if internal_relfro > GATE_INTERNAL_RELFRO:
        raise RuntimeError(
            f"internal consistency gate failed: J C_pk J^T vs covariance_xi_multipoles relFro={internal_relfro:.3e}"
        )

    # Gate 2: R^-1 (J C_pk J^T)_ell0 R^-T reproduces the audited xi0 covariance.
    raw_ell0 = cov_xi_multi[:32, :32]
    xx_recomputed = rinv @ raw_ell0 @ rinv.T
    deconv_relfro = rel_frobenius(xx_recomputed, aud_xi32)
    if deconv_relfro > GATE_DECONV_RELFRO:
        raise RuntimeError(
            f"deconvolution reproduction gate failed: relFro={deconv_relfro:.3e}; "
            "the audited deconvolution is not a plain R^-1 C R^-T on the ell0 block"
        )

    # Select the 15 P0 payload bins on the fine ell0 grid: the fine grid is
    # uniform (kmin=kfund, dk=0.002) and shifted by ~0.0014 h/Mpc relative to
    # the measurement grid, so each payload bin contains exactly one fine bin.
    edges = float(k_grid["kmin"]) + float(k_grid["dk"]) * np.arange(nper + 1, dtype="f8")
    centers = 0.5 * (edges[:-1] + edges[1:])
    selected: list[int] = []
    for lo, hi in k_edges:
        inside = np.flatnonzero((centers >= float(lo) - 1.0e-12) & (centers < float(hi) - 1.0e-12))
        if inside.size != 1:
            raise RuntimeError(f"payload bin [{lo},{hi}] contains {inside.size} fine-grid bins")
        selected.append(int(inside[0]))
    sel = np.asarray(selected, dtype="i8")
    center_offsets = np.abs(centers[sel] - k_obs)
    if float(np.max(center_offsets)) > 1.5e-3:
        raise RuntimeError(f"fine-bin center offset too large: {center_offsets}")

    c_pp = cov_pk[np.ix_(sel, sel)]
    sigma_ratio = np.sqrt(np.diag(c_pp) / np.diag(payload_cov))
    pp_relfro = rel_frobenius(c_pp, payload_cov)
    pp_gate = bool(
        GATE_PP_SIGMA_RATIO[0] <= float(np.median(sigma_ratio)) <= GATE_PP_SIGMA_RATIO[1]
        and pp_relfro < GATE_PP_RELFRO
    )

    mask = s_full >= 50.0
    if int(np.count_nonzero(mask)) != 30:
        raise RuntimeError(f"expected 30 xi fit bins for s>=50, got {int(np.count_nonzero(mask))}")
    c_xx = aud_xi32[np.ix_(mask, mask)]
    c_px = (rinv @ (proj[:32, :] @ cov_pk[:, sel]))[mask, :]

    joint = np.block([[c_pp, c_px.T], [c_px, c_xx]])
    joint_final, spd_meta = spd_repair(joint, floor=PSD_ABS_FLOOR)
    cross_sigma = np.sqrt(np.clip(np.diag(c_pp), 0.0, None))
    xi_sigma = np.sqrt(np.clip(np.diag(c_xx), 0.0, None))
    cross_corr = c_px / np.outer(xi_sigma, cross_sigma)
    eig = np.linalg.eigvalsh(joint_final)

    audit.update(
        {
            "status": "done" if pp_gate else "pp_bridge_gate_failed",
            "gates": {
                "internal_projection_relfro": {
                    "value": internal_relfro,
                    "threshold": GATE_INTERNAL_RELFRO,
                    "pass": True,
                },
                "deconvolution_reproduction_relfro": {
                    "value": deconv_relfro,
                    "threshold": GATE_DECONV_RELFRO,
                    "pass": True,
                },
                "pp_bridge": {
                    "sigma_ratio_median": float(np.median(sigma_ratio)),
                    "sigma_ratio_per_bin": [float(v) for v in sigma_ratio],
                    "relfro": pp_relfro,
                    "thresholds": {"sigma_ratio": list(GATE_PP_SIGMA_RATIO), "relfro": GATE_PP_RELFRO},
                    "pass": pp_gate,
                },
            },
            "fine_grid_selection": {
                "k_grid": k_grid,
                "theory_ells": theory_ells,
                "selected_fine_indices": [int(v) for v in sel],
                "fine_centers": [float(v) for v in centers[sel]],
                "payload_k_obs": [float(v) for v in k_obs],
                "center_offsets": [float(v) for v in center_offsets],
            },
            "joint_covariance": {
                "shape": [45, 45],
                "spd": spd_meta,
                "eigenvalues": {
                    "min": float(eig[0]),
                    "max": float(eig[-1]),
                    "condition": float(eig[-1] / max(eig[0], 1.0e-300)),
                },
                "cross_block": {
                    "corr_abs_max": float(np.max(np.abs(cross_corr))),
                    "corr_abs_median": float(np.median(np.abs(cross_corr))),
                },
                "psd_policy": "raw kept when SPD; absolute floor 1e-30 otherwise; no relative floors",
            },
            "outputs": {"npz": str(out_npz), "npz_sha256": None, "json": str(out_json)},
        }
    )
    if not pp_gate:
        # Nothing is written on a failed gate; the escalation path is an aligned
        # builder rerun, which must be a separate decision.
        audit["escalation_note"] = (
            "pp bridge failed: consider rerunning task44_make_lrg2_jaxpower_covariance on a "
            "measurement-aligned k grid (kmin=0.001) and bridging the xi block instead"
        )
        print(json.dumps({"status": audit["status"], "gates": audit["gates"]}, sort_keys=True))
        raise SystemExit(2)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        out_npz,
        k_obs=k_obs,
        k_edges=k_edges,
        s=s_full[mask],
        c_pp=c_pp,
        c_px=c_px,
        c_xx=c_xx,
        joint=joint_final,
        joint_raw=joint,
        selected_fine_indices=sel,
        fine_centers=centers[sel],
        sigma_ratio_payload=sigma_ratio,
        raw_meta_json=np.asarray(json.dumps(raw_meta, sort_keys=True)),
    )
    audit["outputs"]["npz_sha256"] = sha256_file(out_npz)
    atomic_write_json(out_json, audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "internal_relfro": internal_relfro,
                "deconv_relfro": deconv_relfro,
                "pp_sigma_ratio_median": float(np.median(sigma_ratio)),
                "pp_relfro": pp_relfro,
                "cross_corr_abs_max": audit["joint_covariance"]["cross_block"]["corr_abs_max"],
                "joint_eigmin": float(eig[0]),
                "output": str(out_npz),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
