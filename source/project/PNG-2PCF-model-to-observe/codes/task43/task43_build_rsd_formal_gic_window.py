#!/usr/bin/env python3
"""Build one phase-specific formal-GIC random-pair window for Task 4.3.2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from task43_build_rsd_lightcone_random import fkp_path
from task43_rsd_common import OUTPUT_ROOT, PHASES, atomic_write_json, sha256_file


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK44_DIR = PROJECT_ROOT / "codes" / "task44"
if str(TASK44_DIR) not in sys.path:
    sys.path.insert(0, str(TASK44_DIR))

from task44_fit_lrg2_rsd import build_theory_context, load_or_build_w2  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_row(rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    matches = [row for row in rows if row["phase"] == phase]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {phase}, found {len(matches)}")
    return matches[0]


DEFAULT_WINDOW_ROOT = OUTPUT_ROOT / "lightcone" / "formal_gic_windows"


def output_path(phase: str, nsub: int, seed_base: int, *, root: Path | None = None) -> Path:
    output_root = DEFAULT_WINDOW_ROOT if root is None else Path(root)
    return output_root / (
        f"task43_rsd_formal_gic_{phase}_nsub{int(nsub)}_seed{int(seed_base) + PHASES.index(phase)}.npz"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--nsub", type=int, default=200000)
    parser.add_argument("--seed-base", type=int, default=430340)
    parser.add_argument("--nthreads", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= int(args.nthreads) <= 8:
        raise ValueError("--nthreads must be in 1..8")
    if int(args.nsub) <= 0:
        raise ValueError("--nsub must be positive for the frozen formal-GIC window")

    row = select_row(read_jsonl(args.manifest), str(args.phase))
    random_path = Path(row["lightcone_random_path"])
    zeff_path = fkp_path(row)
    window_root = (
        Path(row["lightcone_random_path"]).parent.parent / "formal_gic_windows"
        if row.get("analysis_tag")
        else DEFAULT_WINDOW_ROOT
    )
    output = output_path(str(args.phase), int(args.nsub), int(args.seed_base), root=window_root)
    sidecar = output.with_suffix(".json")
    seed = int(args.seed_base) + PHASES.index(str(args.phase))
    if output.is_file() and sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(output):
            print(json.dumps({"status": "pass", "phase": args.phase, "reused": True, "output": str(output)}, sort_keys=True))
            return
        raise FileExistsError(f"unvalidated formal-GIC window exists: {output}")
    if sidecar.exists():
        raise FileExistsError(f"orphan formal-GIC sidecar exists: {sidecar}")

    with np.load(zeff_path, allow_pickle=False) as payload:
        zeff = float(np.asarray(payload["zeff"]).item())
        p0 = float(np.asarray(payload["p0"]).item())
    theory = build_theory_context(
        zeff,
        kmax=5.0,
        ndense=60000,
        boxsize=2000.0,
        cosmology="abacus_c000",
    )
    window = load_or_build_w2(
        theory=theory,
        random_path=random_path,
        window_path=output,
        nsub=int(args.nsub),
        seed=seed,
        nthreads=int(args.nthreads),
        p0=p0,
        boxsize=2000.0,
        force=False,
        trust_cache=False,
    )
    metadata = {
        "task": "task43_build_rsd_formal_gic_window",
        "status": "pass",
        "phase": str(args.phase),
        "classification": "phase-specific formal-GIC random-pair window",
        "random_path": str(random_path),
        "random_sha256": sha256_file(random_path),
        "zeff_path": str(zeff_path),
        "zeff": zeff,
        "p0": p0,
        "nsub": int(args.nsub),
        "seed": seed,
        "nthreads": int(args.nthreads),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "window_meta": window["meta"],
        "output": str(output),
        "output_root": str(window_root),
        "output_sha256": sha256_file(output),
    }
    atomic_write_json(sidecar, metadata)
    print(json.dumps({"status": "pass", "phase": args.phase, "reused": False, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
