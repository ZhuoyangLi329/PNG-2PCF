#!/usr/bin/env python3
"""Prepare Task44-compatible Task4.3.2 light-cone covariance catalogs.

The measured catalogs remain immutable.  This adapter adds the explicitly
required NX and WEIGHT_TOTAL columns from the frozen phase FKP table so the
existing Task44 JAXpower window-covariance implementation can be reused without
changing its scientific logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from task43_build_rsd_lightcone_random import fkp_path
from task43_rsd_common import OUTPUT_ROOT, PHASES, atomic_savez, atomic_write_json, sha256_file


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_row(rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    selected = [row for row in rows if row["phase"] == phase]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one manifest row for {phase}, found {len(selected)}")
    return selected[0]


def output_paths(row: dict[str, Any]) -> tuple[Path, Path, Path]:
    phase = str(row["phase"])
    if row.get("analysis_tag"):
        root = Path(row["lightcone_random_path"]).parent.parent / "covariance_inputs"
        stem = f"task43_rsd_covinput_{row['sim_name']}_{row['analysis_tag']}"
        return root / f"{stem}_data.npz", root / f"{stem}_random_x25.npz", root / f"{stem}.json"
    root = OUTPUT_ROOT / "lightcone" / "covariance_inputs"
    stem = f"task43_rsd_covinput_AbacusSummit_base_c000_{phase}_zobs0p6_0p8"
    return root / f"{stem}_data.npz", root / f"{stem}_random_x25.npz", root / f"{stem}.json"


def values_by_redshift(redshift: np.ndarray, z_edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    index = np.clip(
        np.searchsorted(np.asarray(z_edges, dtype="f8"), np.asarray(redshift, dtype="f8"), side="right") - 1,
        0,
        np.asarray(values).size - 1,
    )
    return np.asarray(values, dtype="f8")[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    args = parser.parse_args()
    row = select_row(read_jsonl(args.manifest), args.phase)
    data_output, random_output, metadata_output = output_paths(row)
    products = (data_output, random_output, metadata_output)
    if all(path.is_file() for path in products):
        metadata = json.loads(metadata_output.read_text(encoding="utf-8"))
        if (
            metadata.get("status") == "pass"
            and metadata.get("data_output_sha256") == sha256_file(data_output)
            and metadata.get("random_output_sha256") == sha256_file(random_output)
        ):
            print(json.dumps({"status": "pass", "phase": args.phase, "reused": True}, sort_keys=True))
            return
    if any(path.exists() for path in products):
        raise FileExistsError(f"partial or unvalidated covariance adapters: {[str(path) for path in products]}")

    summary_path = fkp_path(row)
    with np.load(summary_path, allow_pickle=False) as summary:
        z_edges = np.asarray(summary["z_edges"], dtype="f8")
        nbar = np.asarray(summary["nbar"], dtype="f8")
        fkp = np.asarray(summary["fkp_weights"], dtype="f8")
        p0 = float(np.asarray(summary["p0"]).item())
        zeff = float(np.asarray(summary["zeff"]).item())

    with np.load(row["lightcone_catalog_path"], allow_pickle=False) as data:
        data_z = np.asarray(data["Z"], dtype="f4")
        data_nx = values_by_redshift(data_z, z_edges, nbar).astype("f4")
        data_weight = values_by_redshift(data_z, z_edges, fkp).astype("f4")
        atomic_savez(
            data_output,
            compressed=False,
            X=np.asarray(data["X"], dtype="f4"),
            Y=np.asarray(data["Y"], dtype="f4"),
            Zcart=np.asarray(data["Zcart"], dtype="f4"),
            Z=data_z,
            NX=data_nx,
            WEIGHT_TOTAL=data_weight,
        )

    with np.load(row["lightcone_random_path"], allow_pickle=False) as random:
        random_z = np.asarray(random["Z"], dtype="f4")
        random_nx = values_by_redshift(random_z, z_edges, nbar).astype("f4")
        random_weight = values_by_redshift(random_z, z_edges, fkp).astype("f4")
        stored_weight = np.asarray(random["WEIGHT_TOTAL"], dtype="f4")
        weight_difference = float(np.max(np.abs(random_weight.astype("f8") - stored_weight.astype("f8"))))
        if not np.allclose(random_weight, stored_weight, rtol=1.0e-7, atol=1.0e-8):
            raise RuntimeError(f"random FKP adapter mismatch: max_abs={weight_difference}")
        atomic_savez(
            random_output,
            compressed=False,
            X=np.asarray(random["X"], dtype="f4"),
            Y=np.asarray(random["Y"], dtype="f4"),
            Zcart=np.asarray(random["Zcart"], dtype="f4"),
            Z=random_z,
            NX=random_nx,
            WEIGHT_TOTAL=random_weight,
        )

    with np.load(data_output, allow_pickle=False) as data_check, np.load(random_output, allow_pickle=False) as random_check:
        gates = {
            "data_columns_finite": bool(all(np.all(np.isfinite(data_check[key])) for key in data_check.files)),
            "random_columns_finite": bool(all(np.all(np.isfinite(random_check[key])) for key in random_check.files)),
            "data_nx_positive": bool(np.all(data_check["NX"] > 0.0)),
            "random_nx_positive": bool(np.all(random_check["NX"] > 0.0)),
            "data_fkp_identity": bool(
                np.allclose(data_check["WEIGHT_TOTAL"], 1.0 / (1.0 + p0 * data_check["NX"]), rtol=2.0e-7, atol=1.0e-8)
            ),
            "random_fkp_identity": bool(
                np.allclose(random_check["WEIGHT_TOTAL"], 1.0 / (1.0 + p0 * random_check["NX"]), rtol=2.0e-7, atol=1.0e-8)
            ),
        }
        ndata = int(data_check["Z"].size)
        nrandom = int(random_check["Z"].size)
        data_weight_range = [float(np.min(data_check["WEIGHT_TOTAL"])), float(np.max(data_check["WEIGHT_TOTAL"]))]
        random_weight_range = [float(np.min(random_check["WEIGHT_TOTAL"])), float(np.max(random_check["WEIGHT_TOTAL"]))]
    status = "pass" if all(gates.values()) and nrandom == 25 * ndata else "fail"
    metadata = {
        "task": "task43_prepare_rsd_covariance_inputs",
        "status": status,
        "phase": args.phase,
        "purpose": "immutable Task44 JAXpower covariance compatibility adapter",
        "weighting": "NX from frozen phase nbar(z); WEIGHT_TOTAL=1/(1+P0*NX)",
        "p0": p0,
        "zeff": zeff,
        "ndata": ndata,
        "nrandom": nrandom,
        "random_multiplier": float(nrandom) / float(ndata),
        "data_weight_range": data_weight_range,
        "random_weight_range": random_weight_range,
        "random_stored_weight_max_abs_difference": weight_difference,
        "gates": gates,
        "source_data": row["lightcone_catalog_path"],
        "source_data_sha256": sha256_file(Path(row["lightcone_catalog_path"])),
        "source_random": row["lightcone_random_path"],
        "source_random_sha256": sha256_file(Path(row["lightcone_random_path"])),
        "fkp_summary": str(summary_path),
        "fkp_summary_sha256": sha256_file(summary_path),
        "data_output": str(data_output),
        "data_output_sha256": sha256_file(data_output),
        "random_output": str(random_output),
        "random_output_sha256": sha256_file(random_output),
    }
    atomic_write_json(metadata_output, metadata)
    print(json.dumps({"status": status, "phase": args.phase, "data": str(data_output), "random": str(random_output)}, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
