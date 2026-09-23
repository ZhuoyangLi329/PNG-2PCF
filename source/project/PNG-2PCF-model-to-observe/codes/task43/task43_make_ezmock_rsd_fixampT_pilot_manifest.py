#!/usr/bin/env python3
"""Build the x10 fixampT n(z)-matched pilot manifest from the frozen production x10 manifest.

Only per-realization paths are rewritten into the pilot root; `random_catalog_path` /
`random_metadata_path` keep pointing at the frozen common-50 random so the P0/P2 and
xi provenance gates stay valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
PRODUCTION_ROOT = PROJECT_ROOT / "outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50"
SOURCE_MANIFEST = PRODUCTION_ROOT / "manifests/task43_ezmock_rsd_covariance_x10_fixampF_common50.jsonl"
PILOT_ROOT = PROJECT_ROOT / "outputs/task43_outputs/ezmock_rsd_lightcone_x10_fixampT_nzmatch_pilot"
PILOT_MANIFEST = PILOT_ROOT / "manifests/task43_ezmock_rsd_x10_fixampT_nzmatch.jsonl"

REWRITE_KEYS = (
    "ezmock_config_path", "ezmock_log_path", "rawbox_catalog_path",
    "lightcone_catalog_path", "lightcone_metadata_path", "pk_path", "xi_path",
)
FROZEN_KEYS = ("random_catalog_path", "random_metadata_path")
NREAL = 10
NTRACER = 1_847_000
FLAVOR = "pilot_fixampT_nzmatch"
EXPERIMENT = "ezmock_rsd_lightcone_zobs0p4_0p8_x10_fixampT_nzmatch_pilot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite_row(row: dict[str, object], root: Path, pdf_base: float | None,
                z_box: float | None) -> dict[str, object]:
    out = dict(row)
    prefix = str(PRODUCTION_ROOT) + os.sep
    for key in REWRITE_KEYS:
        value = str(row[key])
        if not value.startswith(prefix):
            raise RuntimeError(f"{key} is not under the production root: {value}")
        out[key] = str(root) + value[len(str(PRODUCTION_ROOT)):]
    for key in FROZEN_KEYS:
        value = str(row[key])
        if not value.startswith(prefix):
            raise RuntimeError(f"{key} should be a frozen production path: {value}")
        out[key] = value
    out["ntracer"] = NTRACER
    if pdf_base is not None:
        out["pdf_base"] = float(pdf_base)
    if z_box is not None:
        out["z_box"] = float(z_box)
    out["fix_amplitude"] = True
    out["flavor"] = FLAVOR
    out["nz_thinning"] = True
    out["experiment"] = EXPERIMENT
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=PILOT_MANIFEST)
    parser.add_argument("--root", type=Path, default=PILOT_ROOT)
    parser.add_argument("--pdf-base", type=float, default=None)
    parser.add_argument("--z-box", type=float, default=None,
                        help="EZmock snapshot redshift (defaults to the frozen 0.725).")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"pilot manifest exists: {args.output}")
    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != NREAL:
        raise RuntimeError(f"expected {NREAL} source rows, found {len(rows)}")
    pilot_rows = [rewrite_row(row, args.root, args.pdf_base, args.z_box) for row in rows]
    seeds = [int(row["seed"]) for row in pilot_rows]
    if len(set(seeds)) != NREAL or any(int(row["ntracer"]) != NTRACER for row in pilot_rows):
        raise RuntimeError(f"pilot seed/ntracer gate failed: {seeds}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in pilot_rows), encoding="utf-8")
    audit = {
        "task": "task43_make_ezmock_rsd_fixampT_pilot_manifest",
        "status": "done",
        "source_manifest": str(args.source),
        "source_sha256": sha256(args.source),
        "n_rows": len(pilot_rows),
        "seeds": seeds,
        "flavor": FLAVOR,
        "ntracer": NTRACER,
        "pdf_base": float(pilot_rows[0]["pdf_base"]),
        "z_box": float(pilot_rows[0].get("z_box", 0.725)),
        "fix_amplitude": True,
        "nz_thinning": True,
        "frozen_random_paths": {key: str(pilot_rows[0][key]) for key in FROZEN_KEYS},
        "pilot_root": str(args.root),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {args.output} rows={len(pilot_rows)} seeds={seeds[0]}..{seeds[-1]} "
          f"pdf_base={audit['pdf_base']:g} z_box={audit['z_box']:g} sha256={audit['output_sha256']}")


if __name__ == "__main__":
    main()
