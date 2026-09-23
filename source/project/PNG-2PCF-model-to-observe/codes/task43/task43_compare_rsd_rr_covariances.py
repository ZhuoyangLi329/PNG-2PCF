#!/usr/bin/env python3
"""Compare two immutable RR-deconvolved Task 4.3.2 covariances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from task43_rsd_common import atomic_savez, atomic_write_json, sha256_file


def correlation(covariance: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.diag(covariance))
    return covariance / np.outer(sigma, sigma)


def load(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"s", "s_edges", "ells", "covariance_single_realization", "rr_window_matrix"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise KeyError(f"{path} is missing {missing}")
        result = {key: np.asarray(payload[key]) for key in required}
        result["meta"] = (
            json.loads(str(np.asarray(payload["meta_json"]).item()))
            if "meta_json" in payload.files and str(np.asarray(payload["meta_json"]).item())
            else {}
        )
    covariance = np.asarray(result["covariance_single_realization"], dtype="f8")
    if covariance.shape[0] != covariance.shape[1] or not np.all(np.isfinite(covariance)):
        raise RuntimeError(f"invalid covariance in {path}")
    result["covariance_single_realization"] = 0.5 * (covariance + covariance.T)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    output_npz = args.output_prefix.with_suffix(".npz")
    output_json = args.output_prefix.with_suffix(".json")
    if output_npz.exists() or output_json.exists():
        raise FileExistsError(f"immutable covariance comparison output exists: {output_npz} / {output_json}")

    reference = load(args.reference)
    candidate = load(args.candidate)
    for key in ("s", "s_edges", "ells"):
        if not np.array_equal(reference[key], candidate[key]):
            raise RuntimeError(f"covariance grids differ in {key}")
    cov_ref = reference["covariance_single_realization"]
    cov_cand = candidate["covariance_single_realization"]
    if cov_ref.shape != cov_cand.shape:
        raise RuntimeError(f"covariance shapes differ: {cov_ref.shape} vs {cov_cand.shape}")
    sigma_ref = np.sqrt(np.diag(cov_ref))
    sigma_cand = np.sqrt(np.diag(cov_cand))
    sigma_ratio = sigma_cand / sigma_ref
    corr_delta = correlation(cov_cand) - correlation(cov_ref)
    window_ref = np.asarray(reference["rr_window_matrix"], dtype="f8")
    window_cand = np.asarray(candidate["rr_window_matrix"], dtype="f8")
    window_delta = window_cand - window_ref
    relative_frobenius = float(np.linalg.norm(cov_cand - cov_ref) / np.linalg.norm(cov_ref))
    payload = {
        "task": "task43_compare_rsd_rr_covariances",
        "status": "pass",
        "classification": "RR Monte-Carlo stability diagnostic",
        "reference": str(args.reference),
        "reference_sha256": sha256_file(args.reference),
        "candidate": str(args.candidate),
        "candidate_sha256": sha256_file(args.candidate),
        "ells": [int(value) for value in np.asarray(reference["ells"]).ravel()],
        "shape": list(cov_ref.shape),
        "sigma_ratio_candidate_over_reference": sigma_ratio.tolist(),
        "sigma_ratio_median": float(np.median(sigma_ratio)),
        "sigma_ratio_min": float(np.min(sigma_ratio)),
        "sigma_ratio_max": float(np.max(sigma_ratio)),
        "relative_frobenius_covariance_delta": relative_frobenius,
        "correlation_delta_absmax": float(np.max(np.abs(corr_delta))),
        "rr_window_delta_absmax": float(np.max(np.abs(window_delta))),
        "rr_window_delta_relative_frobenius": float(np.linalg.norm(window_delta) / np.linalg.norm(window_ref)),
        "reference_meta": reference["meta"],
        "candidate_meta": candidate["meta"],
        "output_npz": str(output_npz),
    }
    atomic_savez(
        output_npz,
        s=np.asarray(reference["s"], dtype="f8"),
        s_edges=np.asarray(reference["s_edges"], dtype="f8"),
        ells=np.asarray(reference["ells"], dtype="i8"),
        sigma_ratio=sigma_ratio,
        covariance_delta=cov_cand - cov_ref,
        correlation_delta=corr_delta,
        rr_window_delta=window_delta,
        meta_json=np.asarray(json.dumps(payload, sort_keys=True)),
    )
    payload["output_npz_sha256"] = sha256_file(output_npz)
    atomic_write_json(output_json, payload)
    print(json.dumps({"status": "pass", "output": str(output_json)}, sort_keys=True))


if __name__ == "__main__":
    main()
