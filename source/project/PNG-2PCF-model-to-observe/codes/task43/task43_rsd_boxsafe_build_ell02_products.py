#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the boxsafe ell=(0,2) xi products for the quadrupole-increment test.

Stage ``data`` reprojects xi0/xi2 from the persisted per-phase xi(s,mu)
(DD/DR/RR in 32x40 bins) with the exact measurement quadrature and writes a
load_inputs-compatible ell02 summary; the xi0 bridge must reproduce the stored
xi_multipoles bitwise.  Stage ``deconv`` invokes the audited RR(s,mu)
deconvolution script with ells=ellsin=0,2 on the frozen 96x96 raw covariance,
producing the 64x64 ell02 covariance beside the existing audited ell0 file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task43_measure_rsd_lightcone_xi import project_multipoles  # noqa: E402
from task43_rsd_common import OUTPUT_ROOT, PHASES, atomic_savez, atomic_write_json, sha256_file  # noqa: E402


BOXSAFE_ROOT = OUTPUT_ROOT / "lightcone_boxsafe_zobs0p4_0p8"
XI_DIR = BOXSAFE_ROOT / "xi"
RAW_NPZ = (
    BOXSAFE_ROOT / "covariance"
    / "task43_rsd_boxsafe_ph000_jaxpower_raw_xi0_rsdpoles024_win02468_mesh64_p1_b1cov2p50"
    "_sigmas7p50_nran100k_ndata50k_seed20260817_kbox_fkpP010000_s30_350_ds10.npz"
)
RR_NPZ = (
    BOXSAFE_ROOT / "covariance"
    / "task43_rsd_boxsafe_ph000_rr_smu_nran300k_s30_350_ds10_nmu40_seed430420.npz"
)
RR_FKP_NORM = 3.2497076301174904e-10
ELLL0_DECONV_NPZ = (
    BOXSAFE_ROOT / "covariance"
    / "task43_rsd_boxsafe_ph000_jaxpower_rrdeconv_ell0_rsdpoles024_win02468_mesh64_p1_b1cov2p50"
    "_sigmas7p50_nran100k_ndata50k_seed20260817_rrnran300k_rrseed430420_kbox_fkpP010000_s30_350_ds10.npz"
)
ELL02_DECONV_PREFIX = (
    BOXSAFE_ROOT / "covariance"
    / "task43_rsd_boxsafe_ph000_jaxpower_rrdeconv_ell02_rsdpoles024_win02468_mesh64_p1_b1cov2p50"
    "_sigmas7p50_nran100k_ndata50k_seed20260817_rrnran300k_rrseed430420_kbox_fkpP010000_s30_350_ds10"
)
ELL2_DIR = BOXSAFE_ROOT / "ell2_increment"
SUMMARY_NPZ = ELL2_DIR / "task43_rsd_boxsafe_x25_mean_xi02_s30_350_ds10.npz"

XI0_BRIDGE_ATOL = 1.0e-14
ELLL0_BLOCK_RELFRO_GATE = 0.05


def stage_data() -> None:
    if SUMMARY_NPZ.exists() or SUMMARY_NPZ.with_suffix(".json").exists():
        raise FileExistsError(f"immutable ell02 summary exists: {SUMMARY_NPZ}")
    stack, rrs, zeffs, s_ref, edges_ref = [], [], [], None, None
    bridge_max = 0.0
    for phase in PHASES:
        path = XI_DIR / f"task43_rsd_xi0_AbacusSummit_base_c000_{phase}_mmin1p4e13_zobs0p4_0p8_x25_s30_350_ds10.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as d:
            xi_smu = np.asarray(d["xi_smu"], dtype="f8")
            mu_edges = np.asarray(d["mu_edges"], dtype="f8")
            stored = np.asarray(d["xi_multipoles"], dtype="f8")
            s = np.asarray(d["s"], dtype="f8")
            s_edges = np.asarray(d["s_edges"], dtype="f8")
            rr = np.asarray(d["RR"], dtype="f8")
            zeff = float(np.asarray(d["zeff"]).item())
        if s_ref is None:
            s_ref, edges_ref = s, s_edges
        elif not np.array_equal(s, s_ref):
            raise RuntimeError(f"s grid changed for {phase}")
        multipoles = project_multipoles(xi_smu, mu_edges, (0, 2))
        bridge = float(np.max(np.abs(multipoles[0] - stored[0])))
        bridge_max = max(bridge_max, bridge)
        if bridge > XI0_BRIDGE_ATOL:
            raise RuntimeError(f"xi0 reprojection bridge failed for {phase}: {bridge:.3e}")
        if not np.all(np.isfinite(multipoles)):
            raise RuntimeError(f"non-finite reprojected multipoles for {phase}")
        stack.append(multipoles)
        rrs.append(rr)
        zeffs.append(zeff)
    xi_by_phase = np.stack(stack)  # (25, 2, 32)
    rr_by_phase = np.stack(rrs)
    zeff_by_phase = np.asarray(zeffs, dtype="f8")
    ELL2_DIR.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        SUMMARY_NPZ,
        phases=np.asarray(PHASES),
        ells=np.asarray([0, 2], dtype="i4"),
        s=s_ref,
        s_edges=edges_ref,
        xi_multipoles_by_phase=xi_by_phase,
        xi_multipoles_mean=np.mean(xi_by_phase, axis=0),
        RR_by_phase=rr_by_phase,
        zeff_by_phase=zeff_by_phase,
        zeff_mean=np.asarray(float(np.mean(zeff_by_phase)), dtype="f8"),
    )
    meta = {
        "task": "task43_rsd_boxsafe_build_ell02_products",
        "stage": "data",
        "status": "pass",
        "definition": "xi0/xi2 reprojected from persisted per-phase xi(s,mu) with the exact measurement quadrature",
        "xi0_bridge_max_abs": bridge_max,
        "xi0_bridge_atol": XI0_BRIDGE_ATOL,
        "phases": list(PHASES),
        "input_files": "xi/task43_rsd_xi0_*_zobs0p4_0p8_x25_s30_350_ds10.npz (25 phases, hash-frozen by the audited xi summary)",
        "output_npz": str(SUMMARY_NPZ),
        "output_npz_sha256": sha256_file(SUMMARY_NPZ),
    }
    atomic_write_json(SUMMARY_NPZ.with_suffix(".json"), meta)
    print(json.dumps({"stage": "data", "status": "pass", "xi0_bridge_max_abs": bridge_max}))


def stage_deconv() -> None:
    out_npz = ELL02_DECONV_PREFIX.with_suffix(".npz")
    out_json = ELL02_DECONV_PREFIX.with_suffix(".json")
    if out_npz.exists() or out_json.exists():
        raise FileExistsError(f"immutable ell02 deconvolution exists: {out_npz}")
    cmd = [
        sys.executable,
        str(CODE_DIR / "task43_apply_rr_smu_deconvolution_to_covariance.py"),
        "--covariance-path", str(RAW_NPZ),
        "--rr-smu-path", str(RR_NPZ),
        "--output-prefix", str(ELL02_DECONV_PREFIX),
        "--xi-summary", str(SUMMARY_NPZ),
        "--rr-window-kind", "RR",
        "--resolution", "1",
        "--ells", "0,2",
        "--ellsin", "0,2",
        "--rr-fkp-norm", repr(RR_FKP_NORM),
    ]
    print("[deconv] launching audited deconvolution script", flush=True)
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

    with np.load(out_npz, allow_pickle=False) as d:
        cov64 = np.asarray(d["covariance_single_realization"], dtype="f8")
        s = np.asarray(d["s"], dtype="f8")
        ells = tuple(int(v) for v in np.asarray(d["ells"]).ravel())
    with np.load(ELLL0_DECONV_NPZ, allow_pickle=False) as d:
        cov32 = np.asarray(d["covariance_single_realization"], dtype="f8")
    if cov64.shape != (64, 64) or ells != (0, 2):
        raise RuntimeError(f"unexpected ell02 covariance: shape={cov64.shape}, ells={ells}")
    eigenvalues = np.linalg.eigvalsh(0.5 * (cov64 + cov64.T))
    if float(eigenvalues[0]) <= 0.0:
        raise RuntimeError(f"ell02 covariance not SPD: eigmin={eigenvalues[0]:.3e}")
    ns = s.size
    block00 = cov64[:ns, :ns]
    rel = float(np.linalg.norm(block00 - cov32) / np.linalg.norm(cov32))
    ratio = np.sqrt(np.diag(block00) / np.diag(cov32))
    audit = {
        "task": "task43_rsd_boxsafe_build_ell02_products",
        "stage": "deconv",
        "status": "pass" if rel < ELLL0_BLOCK_RELFRO_GATE else "ell0_block_bridge_failed",
        "ell0_block_vs_audited": {
            "relfro": rel,
            "sigma_ratio_median": float(np.median(ratio)),
            "gate": ELLL0_BLOCK_RELFRO_GATE,
        },
        "ell02_covariance": {
            "shape": [64, 64],
            "eigenvalue_min": float(eigenvalues[0]),
            "eigenvalue_max": float(eigenvalues[-1]),
            "condition": float(eigenvalues[-1] / eigenvalues[0]),
        },
        "output_npz": str(out_npz),
        "output_npz_sha256": sha256_file(out_npz),
        "raw_covariance_sha256": sha256_file(RAW_NPZ),
        "rr_smu_sha256": sha256_file(RR_NPZ),
        "rr_fkp_norm": RR_FKP_NORM,
        "ells": [0, 2],
        "ellsin": [0, 2],
    }
    audit_path = ELL2_DIR / "audits" / "task43_rsd_boxsafe_ell02_deconv_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(audit_path, audit)
    print(json.dumps({k: audit[k] for k in ("stage", "status", "ell0_block_vs_audited", "ell02_covariance")}, sort_keys=True))
    if audit["status"] != "pass":
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("data", "deconv"))
    args = parser.parse_args()
    if args.stage == "data":
        stage_data()
    else:
        stage_deconv()


if __name__ == "__main__":
    main()
