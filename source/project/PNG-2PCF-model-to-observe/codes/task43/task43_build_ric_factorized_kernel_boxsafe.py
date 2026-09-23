#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the factorized radial single-term kernel for the boxsafe RSD geometry.

Thin adapter of task43_build_ric_factorized_kernel.py: identical machinery
(the boxsafe random generator is verifiably separable, "uniform positive-
octant solid angle; observed data-z resample"), with the boxsafe manifest row,
boxsafe random catalog and boxsafe FKP summary plugged in.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task43_ric_singleterm import (  # noqa: E402
    atomic_savez,
    exact_outer_pair_histogram,
    fkp_effective_normalisation,
    load_fkp_arrays,
    load_random_subsample,
    positive_octant_cosine_cdf,
    radial_pair_components,
    radial_statistics,
    to_jsonable,
    write_json,
)


BOXSAFE_ROOT = PROJECT_ROOT / "outputs/task43_outputs/rsd_validation/lightcone_boxsafe_zobs0p4_0p8"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "outputs/task43_outputs/rsd_validation/manifests"
    / "task43_rsd_validation_lightcone_boxsafe_zobs0p4_0p8_x25.jsonl"
)
DEFAULT_FKP = BOXSAFE_ROOT / "fkp/task43_rsd_fkp_AbacusSummit_base_c000_ph000_mmin1p4e13_zobs0p4_0p8_dz0p01.npz"
OUTPUT_DIR = BOXSAFE_ROOT / "ric_singleterm" / "kernels"


def manifest_random_path(manifest: Path, phase: str) -> tuple[Path, dict]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [row for row in rows if row["phase"] == phase]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {phase}, found {len(matches)}")
    row = matches[0]
    path = Path(row["lightcone_random_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fkp-summary", type=Path, default=DEFAULT_FKP)
    parser.add_argument("--phase", type=str, default="ph000")
    parser.add_argument("--p0", type=float, default=10000.0)
    parser.add_argument("--radial-width", type=float, default=2.0)
    parser.add_argument("--nsub", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--kernel-ds", type=float, default=2.0)
    parser.add_argument("--smax", type=float, default=3400.0)
    parser.add_argument("--nthreads", type=int, default=8)
    parser.add_argument("--sobol-power", type=int, default=22)
    parser.add_argument("--cosine-histogram-bins", type=int, default=262144)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if int(args.nthreads) > 8:
        raise ValueError("登录节点 exact RR 最多使用 8 threads")

    width = f"{float(args.radial_width):g}".replace(".", "p")
    stem = (
        f"task43_ric_factorized_boxsafe_{args.phase}_dchi{width}_nsub{int(args.nsub)}_"
        f"sobol2p{int(args.sobol_power)}_ds2_seed{int(args.seed)}"
    )
    npz_path = Path(args.output_dir) / f"{stem}.npz"
    json_path = Path(args.output_dir) / f"{stem}.json"
    if npz_path.exists() and json_path.exists() and not args.overwrite:
        print(f"[skip] {npz_path}")
        return
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f"partial kernel output exists: {npz_path} / {json_path}")

    random_path, row = manifest_random_path(Path(args.manifest), str(args.phase))
    fkp = load_fkp_arrays(Path(args.fkp_summary))
    random = load_random_subsample(
        random_path,
        fkp_summary=fkp,
        p0=float(args.p0),
        n_subsample=int(args.nsub),
        seed=int(args.seed),
    )
    radial = radial_statistics(random, float(args.radial_width))
    components = radial_pair_components(radial["radial_probability"], radial["radial_mean_chi"])
    angular = positive_octant_cosine_cdf(
        sobol_power=int(args.sobol_power),
        histogram_bins=int(args.cosine_histogram_bins),
        seed=int(args.seed) + 17,
    )
    outer_edges = np.arange(0.0, float(args.smax) + 0.5 * float(args.kernel_ds), float(args.kernel_ds), dtype="f8")
    outer_counts, outer_meta = exact_outer_pair_histogram(
        random,
        separation_edges=outer_edges,
        nthreads=int(args.nthreads),
    )
    meta = {
        "task": "task43_build_ric_factorized_kernel_boxsafe",
        "status": "done",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kernel_kind": "factorized_positive_octant_radial_auto",
        "adaptation": "boxsafe RSD geometry; identical separable machinery; randoms in REDSHIFT-space selection (z_obs resample)",
        "inputs": {
            "manifest": str(args.manifest),
            "random_catalog": str(random_path),
            "random_subsample": random.metadata,
            "fkp_summary": str(args.fkp_summary),
            "outer_counts": outer_meta,
        },
        "fkp_fourier_normalisation": fkp_effective_normalisation(fkp, float(args.p0)),
        "paths": {"kernel_npz": str(npz_path), "metadata_json": str(json_path)},
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    atomic_savez(
        npz_path,
        separation_edges=outer_edges,
        outer_counts=outer_counts,
        source_indices=np.asarray(random.source_indices, dtype="i8"),
        **radial,
        **components,
        **angular,
        meta_json=np.asarray(json.dumps(to_jsonable(meta), sort_keys=True)),
    )
    write_json(json_path, meta)
    print(f"[write] {npz_path} radial_bins={radial['radial_probability'].size} components={components['component_weight'].size}")


if __name__ == "__main__":
    main()
