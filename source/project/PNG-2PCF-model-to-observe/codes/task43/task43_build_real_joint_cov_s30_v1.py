#!/usr/bin/env python3
"""Build the Task 4.3 real-space P0+xi0 covariance on 30 <= s < 350.

The expensive jaxpower covariance calculation is reused from the audited
s50 joint product.  Only its analytic P-to-xi projection is extended to the
two lower-separation bins.  The existing s30 RR inverse is then applied to
the full 32-bin xi block and to the P-xi cross block.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from jaxpower.cov2 import matrix_project_to_correlation

from task43_make_rawbox_z0p725_mmin1p3e13_gaussian_covariance import correlation_from_covariance, nearest_spd
from task43_rsd_common import PROJECT_ROOT, atomic_write_json, sha256_file


FINE_COVARIANCE = (
    PROJECT_ROOT
    / "outputs/task43_outputs/joint_pkxi_s50_350/covariance/fine"
    / "task43_joint_finecov_ph000_mesh64_kmin0p001.npz"
)
STANDARD_JOINT = (
    PROJECT_ROOT
    / "outputs/task43_outputs/joint_pkxi_s50_350/covariance"
    / "task43_joint_cov_64_s50_350.npz"
)
PK_PAYLOAD = (
    PROJECT_ROOT
    / "plots/outputs/task43_outputs/pk_lightcone/summary"
    / "task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_payload.npz"
)
RRDECONV_S30 = (
    PROJECT_ROOT
    / "outputs/task43_outputs/rmin_scan/covariance"
    / "jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_"
    "rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_pad400_win3600_ds2_"
    "k0001_3000_dk002_p1p0_s30_350_ds10.npz"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs/task43_outputs/joint_pkxi_s30_350_v1"
OUTPUT_NPZ = OUTPUT_ROOT / "covariance/task43_joint_cov_64_s30_350_v1.npz"
OUTPUT_JSON = OUTPUT_NPZ.with_suffix(".json")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def covariance_bridge(test: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    sigma_ratio = np.sqrt(np.diag(test) / np.diag(reference))
    return {
        "sigma_ratio_min": float(np.min(sigma_ratio)),
        "sigma_ratio_median": float(np.median(sigma_ratio)),
        "sigma_ratio_max": float(np.max(sigma_ratio)),
        "relative_frobenius_delta": float(np.linalg.norm(test - reference) / np.linalg.norm(reference)),
        "correlation_delta_absmax": float(
            np.max(np.abs(correlation_from_covariance(test) - correlation_from_covariance(reference)))
        ),
    }


def projection_matrix(k_edges: np.ndarray, k_centers: np.ndarray, s_edges: np.ndarray) -> np.ndarray:
    class Pole:
        def coords(self, name: str) -> np.ndarray:
            if name != "k":
                raise KeyError(name)
            return k_centers

        def edges(self, name: str) -> np.ndarray:
            if name != "k":
                raise KeyError(name)
            return k_edges

    class Theory:
        @staticmethod
        def items(level: int = 1) -> list[tuple[dict[str, int], Pole]]:
            if level != 1:
                raise ValueError(level)
            return [({"ells": 0}, Pole())]

    return np.asarray(matrix_project_to_correlation(s_edges, Theory()), dtype="f8")


def correlation_eigenvalue_min(covariance: np.ndarray) -> float:
    return float(np.min(np.linalg.eigvalsh(correlation_from_covariance(covariance))))


def main() -> None:
    if OUTPUT_NPZ.exists() or OUTPUT_JSON.exists():
        raise FileExistsError(f"immutable output exists: {OUTPUT_NPZ} / {OUTPUT_JSON}")
    for path in (FINE_COVARIANCE, STANDARD_JOINT, PK_PAYLOAD, RRDECONV_S30):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(FINE_COVARIANCE, allow_pickle=False) as payload:
        k_edges = np.asarray(payload["k_edges"], dtype="f8")
        k_centers = np.asarray(payload["k_centers"], dtype="f8")
        fine_covariance = np.asarray(payload["fine_cov_corrected"], dtype="f8")
        standard_projection = np.asarray(payload["projection_matrix"], dtype="f8")
        standard_s_edges = np.asarray(payload["s_edges"], dtype="f8")
    s_edges = np.arange(30.0, 350.0 + 10.0, 10.0, dtype="f8")
    s = 0.5 * (s_edges[:-1] + s_edges[1:])
    projection = projection_matrix(k_edges, k_centers, s_edges)
    projection_overlap = projection[2:] - standard_projection
    projection_overlap_relative_l2 = float(np.linalg.norm(projection_overlap) / np.linalg.norm(standard_projection))
    projection_overlap_absmax = float(np.max(np.abs(projection_overlap)))
    if not np.array_equal(s_edges[2:], standard_s_edges):
        raise RuntimeError("s30 and standard s50 grids do not align")

    with np.load(PK_PAYLOAD, allow_pickle=False) as payload:
        k_obs = np.asarray(payload["k_obs"], dtype="f8")
    selected = np.asarray([int(np.argmin(np.abs(k_centers - value))) for value in k_obs], dtype="i8")
    selected_delta = np.abs(k_centers[selected] - k_obs)
    if float(np.max(selected_delta)) > 1.0e-3 or np.unique(selected).size != selected.size:
        raise RuntimeError("P0 fit bins do not map uniquely to fine covariance bins")

    with np.load(RRDECONV_S30, allow_pickle=False) as payload:
        rr_inverse = np.asarray(payload["rr_window_inverse"], dtype="f8")
        audited_xx = np.asarray(payload["covariance_single_realization"], dtype="f8")
        rr_s = np.asarray(payload["s"], dtype="f8")
    if not np.array_equal(rr_s, s):
        raise RuntimeError("s30 RR-deconvolution grid changed")

    c_pp = fine_covariance[np.ix_(selected, selected)]
    c_px = fine_covariance[selected, :] @ projection.T @ rr_inverse.T
    c_xx_raw = rr_inverse @ (projection @ fine_covariance @ projection.T) @ rr_inverse.T
    c_xx, xx_spd = nearest_spd(c_xx_raw, floor_fraction=1.0e-10)
    joint = np.block([[c_pp, c_px], [c_px.T, c_xx]])
    joint = 0.5 * (joint + joint.T)

    with np.load(STANDARD_JOINT, allow_pickle=False) as payload:
        standard_pp = np.asarray(payload["c_pp"], dtype="f8")
        standard_px = np.asarray(payload["c_px"], dtype="f8")
    xx_bridge = covariance_bridge(c_xx, audited_xx)
    pp_relative_l2 = float(np.linalg.norm(c_pp - standard_pp) / np.linalg.norm(standard_pp))
    px_overlap_relative_l2 = float(np.linalg.norm(c_px[:, 2:] - standard_px) / np.linalg.norm(standard_px))

    selected_mask = (s >= 40.0) & (s < 350.0) & ~((s >= 80.0) & (s < 120.0))
    selected_ids = np.concatenate((np.arange(k_obs.size), k_obs.size + np.flatnonzero(selected_mask)))
    selected_joint = joint[np.ix_(selected_ids, selected_ids)]
    gates = {
        "projection_reproduces_standard_s50_relative_l2_below_1e_12": projection_overlap_relative_l2 < 1.0e-12,
        "p_block_reproduces_standard_relative_l2_below_1e_14": pp_relative_l2 < 1.0e-14,
        "xi_bridge_sigma_ratio_median_in_0p95_1p05": 0.95 <= xx_bridge["sigma_ratio_median"] <= 1.05,
        "xi_bridge_relative_frobenius_below_0p10": xx_bridge["relative_frobenius_delta"] < 0.10,
        "full_joint_strict_correlation_spd": correlation_eigenvalue_min(joint) > 1.0e-12,
        "smin40_baomask_joint_strict_correlation_spd": correlation_eigenvalue_min(selected_joint) > 1.0e-12,
        "smin40_baomask_has_27_xi_bins": int(np.count_nonzero(selected_mask)) == 27,
    }
    status = "done" if all(gates.values()) else "validation_failed"
    metadata = {
        "task": "task43_build_real_joint_cov_s30_v1",
        "status": status,
        "method": (
            "reuse the frozen mesh64 fine P covariance, extend the exact jaxpower shell projection to s30, "
            "then apply the audited 32-bin RR inverse to Cxx and Cpx"
        ),
        "inputs": {
            "fine_covariance": str(FINE_COVARIANCE),
            "fine_covariance_sha256": sha256_file(FINE_COVARIANCE),
            "standard_joint": str(STANDARD_JOINT),
            "standard_joint_sha256": sha256_file(STANDARD_JOINT),
            "pk_payload": str(PK_PAYLOAD),
            "pk_payload_sha256": sha256_file(PK_PAYLOAD),
            "rrdeconv_s30": str(RRDECONV_S30),
            "rrdeconv_s30_sha256": sha256_file(RRDECONV_S30),
        },
        "projection_validation": {
            "standard_s50_overlap_relative_l2": projection_overlap_relative_l2,
            "standard_s50_overlap_absolute_max": projection_overlap_absmax,
        },
        "bridges": {
            "c_pp_vs_standard_joint": {"relative_l2": pp_relative_l2},
            "c_xx_vs_audited_s30_rrdeconv": xx_bridge,
            "c_px_s50_overlap_vs_standard_joint": {
                "relative_l2": px_overlap_relative_l2,
                "note": "not a gate because the s30 RR inverse legitimately mixes the two added bins",
            },
        },
        "spd": {
            "c_xx": xx_spd,
            "full_joint_correlation_eigenvalue_min": correlation_eigenvalue_min(joint),
            "smin40_baomask_joint_correlation_eigenvalue_min": correlation_eigenvalue_min(selected_joint),
        },
        "selection_check": {
            "selected_smin40_baomask_centers": s[selected_mask].tolist(),
            "n_xi": int(np.count_nonzero(selected_mask)),
            "joint_shape": list(selected_joint.shape),
        },
        "gates": gates,
        "output_npz": str(OUTPUT_NPZ),
    }
    if status != "done":
        raise RuntimeError(json.dumps(to_jsonable(metadata), indent=2))

    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_NPZ.with_name(f".{OUTPUT_NPZ.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        joint_covariance=joint,
        c_pp=c_pp,
        c_px=c_px,
        c_xx=c_xx,
        pk_k_obs=k_obs,
        xi_s=s,
        xi_s_edges=s_edges,
        projection_matrix=projection,
        selected_fine_indices=selected,
        meta_json=np.asarray(json.dumps(to_jsonable(metadata), sort_keys=True)),
    )
    temporary.replace(OUTPUT_NPZ)
    metadata["output_npz_sha256"] = sha256_file(OUTPUT_NPZ)
    atomic_write_json(OUTPUT_JSON, metadata)
    print(json.dumps({"status": status, "output": str(OUTPUT_NPZ), "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
