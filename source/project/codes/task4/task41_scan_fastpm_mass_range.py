#!/usr/bin/env python3
"""Scan the original FastPM halo masses used by Task4.1 inputs.

Task4.1 downstream ASCII catalogs keep only X Y Z VX VY VZ, so the halo mass
range has to be recovered from the original reduced catalogs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
INPUT_BASE = Path("/global/cfs/cdirs/desi/mocks/UNIT/fastpm_3gpc_fnl/reduced_catalogs")
MEAN_XI_NPZ = (
    PROJECT_ROOT
    / "outputs"
    / "task171_outputs"
    / "box_mean_current_norsd_fnl100"
    / "task171_current_fnl100_box_vs_cutsky_mean_xi.npz"
)
OUTDIR = PROJECT_ROOT / "outputs" / "task41_outputs" / "fastpm_mass_range"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan FastPM halo mass ranges for Task4.1.")
    parser.add_argument("--fnl-tag", default="fnl100", choices=["fnl0", "fnl100"])
    parser.add_argument("--mass-min", type=float, default=1.3e13)
    parser.add_argument("--mass-max", type=float, default=math.inf)
    parser.add_argument(
        "--realizations-from",
        type=Path,
        default=MEAN_XI_NPZ,
        help="NPZ with a 'realizations' array; defaults to Task4.1 fnl100 xi input.",
    )
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument(
        "--assume-sorted",
        action="store_true",
        help="Stop after the first mass below mass-min; use only for mass-sorted catalogs.",
    )
    return parser.parse_args()


def load_realizations(path: Path) -> list[int]:
    data = np.load(path)
    return [int(x) for x in data["realizations"]]


def input_path(fnl_tag: str, realization: int) -> Path:
    return INPUT_BASE / fnl_tag / f"halos_fastpm_N{realization}.gz"


def scan_one(path: Path, mass_min: float, mass_max: float, assume_sorted: bool) -> dict[str, object]:
    n_total = 0
    n_selected = 0
    min_selected = math.inf
    max_selected = -math.inf
    first_mass = None
    monotonic_prefix_ok = True
    previous_mass = math.inf
    stopped_after_below_mass = False
    t0 = time.monotonic()

    with gzip.open(path, "rt") as fin:
        for line in fin:
            if not line.strip():
                continue
            n_total += 1
            cols = line.split()
            mass = float(cols[6])
            if first_mass is None:
                first_mass = mass
            if n_total <= 10000 and mass > previous_mass:
                monotonic_prefix_ok = False
            previous_mass = mass

            if mass_min <= mass <= mass_max:
                n_selected += 1
                if mass < min_selected:
                    min_selected = mass
                if mass > max_selected:
                    max_selected = mass
            elif assume_sorted and mass < mass_min:
                stopped_after_below_mass = True
                break

    return {
        "input_path": str(path),
        "n_total_scanned": n_total,
        "n_selected": n_selected,
        "mass_min_selected": min_selected if n_selected else None,
        "mass_max_selected": max_selected if n_selected else None,
        "first_mass": first_mass,
        "monotonic_prefix_10000_ok": monotonic_prefix_ok,
        "assume_sorted": assume_sorted,
        "stopped_after_below_mass": stopped_after_below_mass,
        "elapsed_sec": time.monotonic() - t0,
    }


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    realizations = load_realizations(args.realizations_from)

    rows = []
    for realization in realizations:
        path = input_path(args.fnl_tag, realization)
        result = scan_one(path, args.mass_min, args.mass_max, args.assume_sorted)
        result["fnl_tag"] = args.fnl_tag
        result["realization"] = realization
        rows.append(result)
        print(
            f"[done] {args.fnl_tag} N{realization:03d}: "
            f"selected={result['n_selected']} "
            f"M=[{result['mass_min_selected']}, {result['mass_max_selected']}] "
            f"scanned={result['n_total_scanned']} elapsed={result['elapsed_sec']:.1f}s",
            flush=True,
        )

    selected_rows = [row for row in rows if row["n_selected"]]
    summary = {
        "fnl_tag": args.fnl_tag,
        "realizations_from": str(args.realizations_from),
        "n_realizations": len(realizations),
        "realizations": realizations,
        "mass_min_cut": args.mass_min,
        "mass_max_cut": args.mass_max if math.isfinite(args.mass_max) else "inf",
        "assume_sorted": args.assume_sorted,
        "total_selected": int(sum(int(row["n_selected"]) for row in rows)),
        "global_mass_min_selected": min(float(row["mass_min_selected"]) for row in selected_rows),
        "global_mass_max_selected": max(float(row["mass_max_selected"]) for row in selected_rows),
        "all_monotonic_prefix_10000_ok": all(bool(row["monotonic_prefix_10000_ok"]) for row in rows),
        "all_stopped_after_below_mass": all(bool(row["stopped_after_below_mass"]) for row in rows)
        if args.assume_sorted
        else False,
        "per_realization": rows,
    }

    suffix = "sortedscan" if args.assume_sorted else "fullscan"
    stem = f"task41_fastpm_{args.fnl_tag}_mass_range_mmin1p3e13_{suffix}"
    json_path = args.outdir / f"{stem}.json"
    csv_path = args.outdir / f"{stem}.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = [
        "fnl_tag",
        "realization",
        "n_total_scanned",
        "n_selected",
        "mass_min_selected",
        "mass_max_selected",
        "first_mass",
        "monotonic_prefix_10000_ok",
        "assume_sorted",
        "stopped_after_below_mass",
        "elapsed_sec",
        "input_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"[summary] global Mmin={summary['global_mass_min_selected']}")
    print(f"[summary] global Mmax={summary['global_mass_max_selected']}")
    print(f"[summary] json={json_path}")
    print(f"[summary] csv={csv_path}")


if __name__ == "__main__":
    main()
