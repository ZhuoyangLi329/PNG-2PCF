#!/usr/bin/env python3
"""Freeze a complete EZmock prefix and build interim Task43 covariances.

This branch is intentionally independent of the live x1000 production.  It
reads only the requested manifest prefix, validates the completed xi/P(k)
products, and writes immutable covariance/fit inputs under a separate interim
directory.  The historical default remains x61 for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes/task43"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task43_ezmock_covariance_common import (  # noqa: E402
    ABACUS_PK_PAYLOAD,
    ABACUS_XI_SUMMARY,
    MANIFEST,
    S_EDGES,
    atomic_savez,
    read_jsonl,
    sha256,
    write_json,
)
from task43_finite_mock_corrections import covariance_corrections  # noqa: E402


DEFAULT_NMOCK = 61
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/task43_outputs/ezmock_covariance_interim_x61_hartlap_percival_20260721"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmock", type=int, default=DEFAULT_NMOCK)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def diagnostics(covariance: np.ndarray) -> dict[str, Any]:
    cov = 0.5 * (np.asarray(covariance, dtype="f8") + np.asarray(covariance, dtype="f8").T)
    eig = np.linalg.eigvalsh(cov)
    # xi and P(k) have radically different units.  Testing the unscaled joint
    # covariance directly can therefore report a tiny negative eigenvalue from
    # roundoff even when the congruent correlation matrix is well conditioned
    # and strictly positive definite.  Use the correlation matrix for the PD
    # gate while retaining raw-covariance eigenvalues as diagnostics.
    sigma = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sigma, sigma)
    corr = 0.5 * (corr + corr.T)
    corr_eig = np.linalg.eigvalsh(corr)
    return {
        "shape": list(cov.shape),
        "condition_number": float(np.linalg.cond(cov)),
        "minimum_eigenvalue": float(eig[0]),
        "maximum_eigenvalue": float(eig[-1]),
        "correlation_condition_number": float(np.linalg.cond(corr)),
        "correlation_minimum_eigenvalue": float(corr_eig[0]),
        "correlation_maximum_eigenvalue": float(corr_eig[-1]),
        "positive_definite": bool(corr_eig[0] > 0.0),
        "positive_definite_test_basis": "correlation_matrix",
        "symmetry_max_abs": float(np.max(np.abs(cov - cov.T))),
    }


def main() -> None:
    args = parse_args()
    nmock = int(args.nmock)
    if nmock <= 49 or nmock > 1000:
        raise ValueError("finite-mock joint45 snapshot requires 50 <= nmock <= 1000")
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else (
            DEFAULT_OUTPUT_ROOT
            if nmock == DEFAULT_NMOCK
            else PROJECT_ROOT / f"outputs/task43_outputs/ezmock_covariance_interim_x{nmock}_hartlap_percival_20260721"
        )
    )
    covariance_npz = output_root / f"covariance/task43_ezmock_interim_x{nmock}_covariance_joint45.npz"
    xi_covariance_npz = output_root / f"fit_inputs/task43_ezmock_interim_x{nmock}_xi30_covariance.npz"
    pk_payload_npz = output_root / f"fit_inputs/task43_ezmock_interim_x{nmock}_pk15_fit_payload.npz"
    freeze_json = output_root / f"audit/task43_ezmock_interim_x{nmock}_freeze_audit.json"

    rows = read_jsonl(MANIFEST)[:nmock]
    if len(rows) != nmock:
        raise RuntimeError(f"manifest contains only {len(rows)} rows")
    with np.load(ABACUS_PK_PAYLOAD, allow_pickle=False) as target:
        target_pk = {key: np.asarray(target[key]) for key in target.files}
    target_k = np.asarray(target_pk["k_obs"], dtype="f8")
    target_k_edges = np.asarray(target_pk["k_edges"], dtype="f8")
    if target_k.shape != (15,) or target_k_edges.shape != (15, 2):
        raise RuntimeError("authoritative Task43 P(k) payload is not the 15-bin kmax=0.10 payload")

    xi_stack: list[np.ndarray] = []
    pk_stack: list[np.ndarray] = []
    frozen_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for index, row in enumerate(rows):
        expected_phase = f"ph{1000 + index:04d}"
        expected_seed = 432001 + index
        if (
            int(row["production_index"]) != index
            or str(row["phase"]) != expected_phase
            or int(row["seed"]) != expected_seed
            or row["fix_amplitude"] is not False
        ):
            raise RuntimeError(f"manifest row {index} does not match the frozen sequence")
        xi_path = Path(str(row["xi_path"]))
        pk_path = Path(str(row["pk_path"]))
        for path in (xi_path, xi_path.with_suffix(".json"), pk_path, pk_path.with_suffix(".json")):
            if not path.is_file():
                raise FileNotFoundError(path)
            source_hashes[str(path)] = sha256(path)
        xi_meta = json.loads(xi_path.with_suffix(".json").read_text(encoding="utf-8"))
        pk_meta = json.loads(pk_path.with_suffix(".json").read_text(encoding="utf-8"))
        if xi_meta.get("fix_amplitude") is not False or pk_meta.get("fix_amplitude") is not False:
            raise RuntimeError(f"row {index} is not FIX_AMPLITUDE=F")
        if int(xi_meta["seed"]) != expected_seed or int(pk_meta["seed"]) != expected_seed:
            raise RuntimeError(f"row {index} seed mismatch")
        with np.load(xi_path, allow_pickle=False) as xi:
            vector_xi = np.asarray(xi["xi0"], dtype="f8")
            if vector_xi.shape != (30,) or not np.array_equal(xi["s_edges"], S_EDGES):
                raise RuntimeError(f"row {index} xi grid mismatch")
            if bool(np.asarray(xi["fix_amplitude"]).item()) is not False:
                raise RuntimeError(f"row {index} xi FIX_AMPLITUDE mismatch")
        with np.load(pk_path, allow_pickle=False) as pk:
            vector_pk = np.asarray(pk["pk0"], dtype="f8")
            if (
                vector_pk.shape != (15,)
                or not np.array_equal(pk["k"], target_k)
                or not np.array_equal(pk["k_edges"], target_k_edges)
            ):
                raise RuntimeError(f"row {index} P(k) grid mismatch")
            if bool(np.asarray(pk["fix_amplitude"]).item()) is not False:
                raise RuntimeError(f"row {index} P(k) FIX_AMPLITUDE mismatch")
        if not np.all(np.isfinite(vector_xi)) or not np.all(np.isfinite(vector_pk)):
            raise RuntimeError(f"row {index} contains non-finite values")
        xi_stack.append(vector_xi)
        pk_stack.append(vector_pk)
        frozen_rows.append(
            {
                "production_index": index,
                "phase": expected_phase,
                "seed": expected_seed,
                "xi_path": str(xi_path),
                "pk_path": str(pk_path),
                "xi_sha256": source_hashes[str(xi_path)],
                "pk_sha256": source_hashes[str(pk_path)],
            }
        )

    xi_array = np.vstack(xi_stack)
    pk_array = np.vstack(pk_stack)
    joint = np.column_stack([xi_array, pk_array])
    covariance = np.cov(joint, rowvar=False, ddof=1)
    xi_covariance = covariance[:30, :30]
    pk_covariance = covariance[30:, 30:]
    cross_covariance = covariance[:30, 30:]
    for name, cov in (("joint", covariance), ("xi", xi_covariance), ("pk", pk_covariance)):
        if not diagnostics(cov)["positive_definite"]:
            raise RuntimeError(f"{name} covariance is not positive definite")

    corrections = {
        "xi_only": covariance_corrections(nmock=nmock, ndata=30, nparams=2),
        "pk_only": covariance_corrections(nmock=nmock, ndata=15, nparams=3),
        "joint_optional": covariance_corrections(nmock=nmock, ndata=45, nparams=3),
    }
    created = datetime.now(timezone.utc).isoformat()
    meta = {
        "task": "task43_build_interim_ezmock_covariance",
        "status": "pass_interim_diagnostic",
        "created_utc": created,
        "classification": (
            "interim finite-mock diagnostic; xi Hartlap is small and the covariance is expected "
            "to remain noisy until the x1000 production is complete"
        ),
        "frozen_nmock": nmock,
        "frozen_phase_range": ["ph1000", f"ph{999 + nmock:04d}"],
        "frozen_seed_range": [432001, 432000 + nmock],
        "manifest": str(MANIFEST),
        "manifest_sha256": sha256(MANIFEST),
        "fix_amplitude": False,
        "data_contract": {
            "xi": {"ndata": 30, "s_edges": S_EDGES, "s_centers": 0.5 * (S_EDGES[:-1] + S_EDGES[1:])},
            "pk": {"ndata": 15, "k": target_k, "k_edges": target_k_edges, "kmax_fit": 0.1},
            "joint_dimension": 45,
        },
        "corrections": corrections,
        "diagnostics": {
            "joint": diagnostics(covariance),
            "xi": diagnostics(xi_covariance),
            "pk": diagnostics(pk_covariance),
        },
        "frozen_rows": frozen_rows,
        "authoritative_data": {
            "xi": str(ABACUS_XI_SUMMARY),
            "xi_sha256": sha256(ABACUS_XI_SUMMARY),
            "pk": str(ABACUS_PK_PAYLOAD),
            "pk_sha256": sha256(ABACUS_PK_PAYLOAD),
        },
        "outputs": {
            "covariance": str(covariance_npz),
            "xi_covariance": str(xi_covariance_npz),
            "pk_fit_payload": str(pk_payload_npz),
            "freeze_audit": str(freeze_json),
        },
    }
    atomic_savez(
        covariance_npz,
        s_edges=S_EDGES,
        s=0.5 * (S_EDGES[:-1] + S_EDGES[1:]),
        k=target_k,
        k_edges=target_k_edges,
        phases=np.asarray([item["phase"] for item in frozen_rows]),
        seeds=np.asarray([item["seed"] for item in frozen_rows], dtype="i8"),
        xi_stack=xi_array,
        pk_stack=pk_array,
        joint_stack=joint,
        covariance=covariance,
        xi_covariance=xi_covariance,
        pk_covariance=pk_covariance,
        xi_pk_cross_covariance=cross_covariance,
        meta_json=np.asarray(json.dumps(jsonable(meta), sort_keys=True)),
    )
    atomic_savez(
        xi_covariance_npz,
        s=0.5 * (S_EDGES[:-1] + S_EDGES[1:]),
        s_edges=S_EDGES,
        covariance_single_realization=xi_covariance,
        nmock=np.asarray(nmock, dtype="i8"),
        meta_json=np.asarray(json.dumps(jsonable(meta), sort_keys=True)),
    )
    pk_summary = json.loads(str(np.asarray(target_pk["summary_json"]).item()))
    pk_summary["covariance"] = {
        "source": "EZmock FIX_AMPLITUDE=F sample covariance",
        "nmock": nmock,
        "path": str(covariance_npz),
        "block": "pk_covariance",
        "finite_mock_corrections": corrections["pk_only"],
    }
    target_pk["covariance"] = pk_covariance
    target_pk["covariance_file"] = np.asarray(str(covariance_npz))
    target_pk["summary_json"] = np.asarray(json.dumps(jsonable(pk_summary), sort_keys=True))
    atomic_savez(pk_payload_npz, **target_pk)
    write_json(freeze_json, jsonable(meta))
    print(f"[pass] froze {nmock} paired realizations: {covariance_npz}")
    print(json.dumps(corrections, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
