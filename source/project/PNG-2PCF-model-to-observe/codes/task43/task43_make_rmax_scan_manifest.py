#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the isolated Task43 rmax-scan manifest without copying catalogs.

The original Task43 catalogs were moved into the controlled July-2026
archive.  This manifest points at those immutable inputs and sends only the
new 50--550 Mpc/h measurements to ``outputs/task43_outputs/rmax_scan``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from task43_config import PROJECT_ROOT, read_jsonl, write_jsonl


SOURCE_MANIFEST = PROJECT_ROOT / "outputs/task43_outputs/manifests/task43_mmin1p4e13_x25.jsonl"
ARCHIVE_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "old_doc_codes/task4_task44_cleanup_20260707T061844Z/moved/outputs/task43_outputs"
)
SCAN_ROOT = PROJECT_ROOT / "outputs/task43_outputs/rmax_scan"
DEFAULT_OUTPUT = SCAN_ROOT / "manifests/task43_rmax_scan_mmin1p4e13_x25.jsonl"


def resolve_archived_input(path: Path) -> tuple[Path, str]:
    """Resolve an active Task43 catalog path to the controlled archive."""
    path = path.expanduser()
    if path.exists():
        return path.resolve(), "active"
    active_root = PROJECT_ROOT / "outputs/task43_outputs"
    try:
        relative = path.resolve(strict=False).relative_to(active_root.resolve())
    except ValueError as exc:
        raise FileNotFoundError(f"missing non-Task43 input with no archive mapping: {path}") from exc
    archived = ARCHIVE_OUTPUT_ROOT / relative
    if not archived.exists():
        raise FileNotFoundError(f"missing both active and archived input: {path}; archive candidate={archived}")
    return archived.resolve(), "controlled_archive_20260707"


def output_base_for_row(row: dict[str, Any]) -> Path:
    """Return the untagged output path; the measurement appends fkpP010000."""
    old_stem = Path(row["xi_path"]).stem
    return (SCAN_ROOT / "xi_cucount" / f"{old_stem}_s50_550_ds10.npz").resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite existing manifest without --force: {args.output}")

    source_rows = read_jsonl(args.source_manifest)
    if len(source_rows) != 25:
        raise ValueError(f"expected 25 phases, found {len(source_rows)} in {args.source_manifest}")
    expected_phases = [f"ph{i:03d}" for i in range(25)]
    phases = [str(row["phase"]) for row in source_rows]
    if phases != expected_phases:
        raise ValueError(f"manifest phase order mismatch: {phases}")

    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for row in source_rows:
        new = dict(row)
        halo, halo_location = resolve_archived_input(Path(row["halo_catalog_path"]))
        random, random_location = resolve_archived_input(Path(row["random_catalog_path"]))
        new["halo_catalog_path"] = str(halo)
        new["random_catalog_path"] = str(random)
        for key in ("halo_metadata_path", "random_metadata_path"):
            if key in row:
                resolved, _ = resolve_archived_input(Path(row[key]))
                new[key] = str(resolved)
        new["xi_path"] = str(output_base_for_row(row))
        new["experiment"] = "task43_jaxpower_only_rmax_scan"
        new["measurement_edges_mpc_h"] = {"min": 50.0, "max": 550.0, "step": 10.0}
        new["fit_rmax_edges_mpc_h"] = [350.0, 400.0, 450.0, 500.0, 550.0]
        new["source_manifest"] = str(args.source_manifest.resolve())
        new["input_location"] = {
            "halo": halo_location,
            "random": random_location,
            "catalogs_copied": False,
        }
        rows.append(new)
        provenance.append(
            {
                "phase": new["phase"],
                "halo": str(halo),
                "random": str(random),
                "xi_base": new["xi_path"],
            }
        )

    write_jsonl(args.output, rows)
    audit = {
        "status": "done",
        "task": "task43_make_rmax_scan_manifest",
        "source_manifest": str(args.source_manifest.resolve()),
        "output_manifest": str(args.output.resolve()),
        "nrows": len(rows),
        "phases": phases,
        "catalog_policy": "reference controlled archive in place; do not copy or move",
        "measurement": {
            "backend": "cucount.jax",
            "estimator": "weighted Landy-Szalay",
            "p0": 10000.0,
            "s_edges": "50..550 inclusive, ds=10 Mpc/h",
            "nbins": 50,
        },
        "rows": provenance,
    }
    audit_path = args.output.with_suffix(".json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
