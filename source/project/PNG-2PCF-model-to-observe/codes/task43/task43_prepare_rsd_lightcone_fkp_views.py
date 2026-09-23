#!/usr/bin/env python3
"""Create immutable FKP-weighted analysis views of Task4.3.2 catalogs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from task43_build_rsd_lightcone_random import fkp_path
from task43_rsd_common import PHASES, atomic_savez, atomic_write_json, sha256_file


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_row(rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    matches = [row for row in rows if row["phase"] == phase]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {phase}, found {len(matches)}")
    return matches[0]


def view_paths(row: dict[str, Any]) -> tuple[Path, Path]:
    if row.get("lightcone_fkp_data_view_path") and row.get("lightcone_fkp_random_view_path"):
        return Path(str(row["lightcone_fkp_data_view_path"])), Path(str(row["lightcone_fkp_random_view_path"]))
    root = Path(row["lightcone_random_path"]).parent.parent / "fkp_catalogs"
    label = f"{row['sim_name']}_zobs0p6_0p8_fkpP010000"
    return root / f"task43_rsd_data_{label}.npz", root / f"task43_rsd_random_{label}_x25.npz"


def weights(redshift: np.ndarray, z_edges: np.ndarray, nbar: np.ndarray, p0: float) -> tuple[np.ndarray, np.ndarray]:
    index = np.clip(np.searchsorted(z_edges, redshift, side="right") - 1, 0, nbar.size - 1)
    nx = nbar[index]
    return nx, 1.0 / (1.0 + nx * float(p0))


def write_view(
    *,
    source: Path,
    output: Path,
    role: str,
    z_edges: np.ndarray,
    nbar: np.ndarray,
    p0: float,
    summary_path: Path,
) -> dict[str, Any]:
    metadata_path = output.with_suffix(".json")
    source_hash = sha256_file(source)
    if output.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("status") == "pass"
            and metadata.get("source_sha256") == source_hash
            and metadata.get("output_sha256") == sha256_file(output)
        ):
            return metadata
        raise FileExistsError(f"unvalidated FKP view exists: {output}")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"partial FKP view exists: {output} / {metadata_path}")
    with np.load(source, allow_pickle=False) as payload:
        redshift = np.asarray(payload["Z"], dtype="f8")
        base_weight = np.asarray(payload["WEIGHT"], dtype="f8")
        nx, weight_fkp = weights(redshift, z_edges, nbar, p0)
        save: dict[str, Any] = {
            key: np.asarray(payload[key])
            for key in ("RA", "DEC", "Z", "X", "Y", "Zcart", "WEIGHT", "phase")
            if key in payload.files
        }
        for key in ("RANDOM_INDEX", "realization", "origin_code", "origin_index", "shell_index"):
            if key in payload.files:
                save[key] = np.asarray(payload[key])
    total_weight = base_weight * weight_fkp
    meta_json = {
        "task": "task43_prepare_rsd_lightcone_fkp_views",
        "status": "pass",
        "role": role,
        "source": str(source),
        "source_sha256": source_hash,
        "fkp_summary": str(summary_path),
        "p0": float(p0),
        "nrows": int(redshift.size),
        "weight_definition": "WEIGHT_TOTAL=WEIGHT/(1+NX*P0)",
    }
    save.update(
        {
            "NX": nx.astype("f4"),
            "WEIGHT_FKP": weight_fkp.astype("f4"),
            "WEIGHT_TOTAL": total_weight.astype("f4"),
            "meta_json": np.asarray(json.dumps(meta_json, sort_keys=True)),
        }
    )
    atomic_savez(output, **save)
    metadata = dict(meta_json)
    metadata.update(
        {
            "nx_range_h3_mpc3": [float(np.min(nx)), float(np.max(nx))],
            "weight_total_range": [float(np.min(total_weight)), float(np.max(total_weight))],
            "weight_sum": float(np.sum(total_weight)),
            "weight2_sum": float(np.dot(total_weight, total_weight)),
            "output": str(output),
            "output_sha256": sha256_file(output),
        }
    )
    atomic_write_json(metadata_path, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    args = parser.parse_args()
    row = select_row(read_jsonl(args.manifest), args.phase)
    summary_path = fkp_path(row)
    with np.load(summary_path, allow_pickle=False) as summary:
        z_edges = np.asarray(summary["z_edges"], dtype="f8")
        nbar = np.asarray(summary["nbar"], dtype="f8")
        p0 = float(np.asarray(summary["p0"]).item())
    data_output, random_output = view_paths(row)
    data_meta = write_view(
        source=Path(row["lightcone_catalog_path"]),
        output=data_output,
        role="data",
        z_edges=z_edges,
        nbar=nbar,
        p0=p0,
        summary_path=summary_path,
    )
    random_meta = write_view(
        source=Path(row["lightcone_random_path"]),
        output=random_output,
        role="random",
        z_edges=z_edges,
        nbar=nbar,
        p0=p0,
        summary_path=summary_path,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "phase": row["phase"],
                "data": data_meta["output"],
                "random": random_meta["output"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
