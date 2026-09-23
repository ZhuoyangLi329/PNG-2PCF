#!/usr/bin/env python3
"""Build the x1000 fixampF n(z)-matched production manifest for the RSD lightcone covariance re-run.

Rewrites the per-realization paths of the frozen x1000 fixampF manifest into the new
production root and applies the recalibrated knobs (pdf_base=0.30, z_box=0.60,
ntracer=1_847_000 with deterministic n(z) thinning).  FIX_AMPLITUDE stays F (natural
Gaussian mode amplitudes) as required by the covariance contract.  The immutable
common-50 random paths keep pointing at the original root so the P0/P2 and xi
provenance gates keep validating the frozen inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
OLD_ROOT = PROJECT_ROOT / "outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50"
SOURCE_MANIFEST = OLD_ROOT / "manifests/task43_ezmock_rsd_covariance_x1000_fixampF_common50.jsonl"
ROOT = PROJECT_ROOT / ("outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50"
                       "_nzmatch_b0.30_z0.60")
MANIFEST = ROOT / "manifests/task43_ezmock_rsd_covariance_x1000_fixampF_nzmatch_b0.30_z0.60.jsonl"

REWRITE_KEYS = (
    "ezmock_config_path", "ezmock_log_path", "rawbox_catalog_path",
    "lightcone_catalog_path", "lightcone_metadata_path", "pk_path", "xi_path",
)
FROZEN_KEYS = ("random_catalog_path", "random_metadata_path")
NREAL = 1000
SEED_START = 600001
NTRACER = 1_847_000
PDF_BASE = 0.30
Z_BOX = 0.60
EXPERIMENT = "ezmock_rsd_lightcone_zobs0p4_0p8_x1000_fixampF_nzmatch_b0.30_z0.60_common50"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite_row(row: dict[str, object], root: Path) -> dict[str, object]:
    out = dict(row)
    prefix = str(OLD_ROOT) + os.sep
    for key in REWRITE_KEYS:
        value = str(row[key])
        if not value.startswith(prefix):
            raise RuntimeError(f"{key} is not under the old production root: {value}")
        out[key] = str(root) + value[len(str(OLD_ROOT)):]
    for key in FROZEN_KEYS:
        value = str(row[key])
        if not value.startswith(prefix):
            raise RuntimeError(f"{key} should be a frozen old-root path: {value}")
        out[key] = value
    out["ntracer"] = NTRACER
    out["pdf_base"] = PDF_BASE
    out["z_box"] = Z_BOX
    out["redshift_snapshot"] = Z_BOX
    out["fix_amplitude"] = False
    out.pop("flavor", None)
    out["nz_thinning"] = True
    out["experiment"] = EXPERIMENT
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"production manifest exists: {args.output}")
    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != NREAL:
        raise RuntimeError(f"expected {NREAL} source rows, found {len(rows)}")
    out_rows = [rewrite_row(row, args.root) for row in rows]
    seeds = [int(row["seed"]) for row in out_rows]
    if len(set(seeds)) != NREAL or seeds != list(range(SEED_START, SEED_START + NREAL)):
        raise RuntimeError(f"production seed gate failed: {seeds[0]}..{seeds[-1]}")
    if any(int(row["ntracer"]) != NTRACER for row in out_rows):
        raise RuntimeError("production ntracer gate failed")
    if any(bool(row["fix_amplitude"]) for row in out_rows):
        raise RuntimeError("production rows must be FIX_AMPLITUDE=F")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in out_rows), encoding="utf-8")
    audit = {
        "task": "task43_make_ezmock_rsd_production_manifest",
        "status": "done",
        "source_manifest": str(args.source),
        "source_sha256": sha256(args.source),
        "n_rows": len(out_rows),
        "seeds": {"first": seeds[0], "last": seeds[-1], "n_unique": len(set(seeds))},
        "ntracer": NTRACER,
        "pdf_base": PDF_BASE,
        "z_box": Z_BOX,
        "fix_amplitude": False,
        "nz_thinning": True,
        "flavor": None,
        "frozen_random_paths": {key: str(out_rows[0][key]) for key in FROZEN_KEYS},
        "production_root": str(args.root),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] {args.output} rows={len(out_rows)} seeds={seeds[0]}..{seeds[-1]} "
          f"pdf_base={PDF_BASE:g} z_box={Z_BOX:g} fixamp=F sha256={audit['output_sha256']}")


if __name__ == "__main__":
    main()
