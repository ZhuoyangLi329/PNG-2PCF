#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a Task43 weighted RR pair-kernel formal-GIC window cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from task43_fit_minimal_closure import build_theory_context, load_or_build_formal_gic_window


def zeff_from_fkp_summary(path: Path, p0: float) -> float:
    """Read the random-auto zeff for a requested P0 from the FKP summary."""
    data = np.load(path, allow_pickle=False)
    p0_values = np.asarray(data["p0_values"], dtype="f8")
    zeff_values = np.asarray(data["zeff_random_auto"], dtype="f8")
    matches = np.flatnonzero(np.isclose(p0_values, float(p0), rtol=0.0, atol=1.0e-10))
    if matches.size != 1:
        raise ValueError(f"could not find unique P0={p0} in {path}; available={p0_values.tolist()}")
    return float(zeff_values[int(matches[0])])


def choose_zeff(args: argparse.Namespace) -> tuple[float, dict[str, Any]]:
    """Use xi-summary zeff when available, otherwise fall back to FKP summary."""
    if args.xi_path is not None and args.xi_path.exists():
        data = np.load(args.xi_path, allow_pickle=False)
        return float(data["zeff"]), {"source": "xi_summary", "path": str(args.xi_path)}
    if args.fkp_summary is not None:
        return zeff_from_fkp_summary(args.fkp_summary, float(args.p0)), {
            "source": "fkp_summary",
            "path": str(args.fkp_summary),
            "p0": float(args.p0),
        }
    raise ValueError("provide an existing --xi-path or --fkp-summary/--p0 to determine zeff")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi-path", type=Path, default=None)
    parser.add_argument("--fkp-summary", type=Path, required=True)
    parser.add_argument("--p0", type=float, required=True)
    parser.add_argument("--random-path", type=Path, required=True)
    parser.add_argument("--window-path", type=Path, required=True)
    parser.add_argument("--nsub", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--nthreads", type=int, default=32)
    parser.add_argument("--kmax", type=float, default=5.0)
    parser.add_argument("--ndense", type=int, default=60000)
    parser.add_argument("--theory-boxsize", type=float, default=2000.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    zeff, zeff_meta = choose_zeff(args)
    theory = build_theory_context(
        zeff,
        kmax=float(args.kmax),
        ndense=int(args.ndense),
        boxsize=float(args.theory_boxsize),
    )
    window = load_or_build_formal_gic_window(
        theory=theory,
        window_path=args.window_path,
        random_path=args.random_path,
        fkp_summary_path=args.fkp_summary,
        p0=float(args.p0),
        n_subsample=int(args.nsub),
        seed=int(args.seed),
        nthreads=int(args.nthreads),
        force=bool(args.force),
    )
    meta = dict(window["meta"])
    meta["zeff"] = float(zeff)
    meta["zeff_meta"] = zeff_meta
    meta["theory_boxsize"] = float(args.theory_boxsize)
    meta["status"] = "done"
    json_path = args.window_path.with_suffix(".json")
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
