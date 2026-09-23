#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the Task43 phase manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from task43_config import (
    DEFAULT_MANIFEST,
    DEFAULT_RANDOM_MULTIPLIER,
    DEFAULT_TARGET_COUNT,
    HALO_CATALOG_DIR,
    PHASES,
    RANDOM_DIR,
    READ_SHELLS,
    SELECTION_MODE,
    SPACE_MODE,
    XI_DIR,
    ZMAX,
    ZMIN,
    ensure_task43_dirs,
    phase_sim_name,
    shell_info_path,
    write_jsonl,
)


def _format_mass_label(mass_threshold_hmsun: float) -> str:
    mantissa, exponent = f"{float(mass_threshold_hmsun):.6e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p")
    return f"mmin{mantissa}e{int(exponent)}"


def _path_label(
    target_count: int | None,
    selection_tag: str,
    selection_mode: str,
    mass_threshold_hmsun: float | None,
) -> str:
    if selection_mode == "fixed_mass_threshold_ninterp":
        if selection_tag:
            return selection_tag
        if mass_threshold_hmsun is None:
            raise ValueError("--mass-threshold-hmsun is required for fixed_mass_threshold_ninterp")
        return _format_mass_label(float(mass_threshold_hmsun))
    if selection_tag == "topN":
        if target_count is None:
            raise ValueError("--target-count is required for fixed-count selections")
        return f"topN{target_count}"
    if target_count is None:
        raise ValueError("--target-count is required for fixed-count selections")
    return f"{selection_tag}_topN{target_count}"


def build_rows(
    target_count: int | None,
    random_multiplier: int,
    *,
    selection_mode: str,
    selection_tag: str,
    n_zshells: int,
    mass_threshold_hmsun: float | None,
    random_radial_policy: str | None,
) -> list[dict[str, object]]:
    """Build one manifest row per AbacusSummit phase."""
    rows: list[dict[str, object]] = []
    label = _path_label(target_count, selection_tag, selection_mode, mass_threshold_hmsun)
    selection_z_edges = None
    if n_zshells > 1 and selection_mode != "fixed_mass_threshold_ninterp":
        selection_z_edges = [ZMIN + (ZMAX - ZMIN) * i / n_zshells for i in range(n_zshells + 1)]
    if random_radial_policy is None:
        random_radial_policy = "data_redshift_resample" if selection_mode == "fixed_mass_threshold_ninterp" else "uniform_comoving_volume"
    for iphase, phase in enumerate(PHASES):
        phase_label = phase
        sim_name = phase_sim_name(phase)
        shell_paths = [shell_info_path(phase, shell) for shell in READ_SHELLS]
        row = {
            "task": "task43",
            "phase": phase_label,
            "phase_index": iphase,
            "sim_name": sim_name,
            "zmin": ZMIN,
            "zmax": ZMAX,
            "read_shells": list(READ_SHELLS),
            "shell_paths": [str(path) for path in shell_paths],
            "selection_mode": selection_mode,
            "selection_tag": selection_tag,
            "target_count": None if target_count is None else int(target_count),
            "mass_threshold_hmsun": None if mass_threshold_hmsun is None else float(mass_threshold_hmsun),
            "space_mode": SPACE_MODE,
            "random_multiplier": int(random_multiplier),
            "random_radial_policy": random_radial_policy,
            "halo_catalog_path": str(HALO_CATALOG_DIR / f"halo_lightcone_{sim_name}_z0p6_0p8_{label}.npz"),
            "halo_metadata_path": str(HALO_CATALOG_DIR / f"halo_lightcone_{sim_name}_z0p6_0p8_{label}.json"),
            "random_catalog_path": str(RANDOM_DIR / f"random_{sim_name}_z0p6_0p8_{label}_x{random_multiplier}.npz"),
            "random_metadata_path": str(RANDOM_DIR / f"random_{sim_name}_z0p6_0p8_{label}_x{random_multiplier}.json"),
            "xi_path": str(XI_DIR / f"xi0_{sim_name}_z0p6_0p8_{label}_x{random_multiplier}.npz"),
        }
        if selection_z_edges is not None:
            row["selection_n_zshells"] = int(n_zshells)
            row["selection_z_edges"] = selection_z_edges
        row["missing_shell_paths"] = [str(path) for path in shell_paths if not path.exists()]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--random-multiplier", type=int, default=DEFAULT_RANDOM_MULTIPLIER)
    parser.add_argument("--selection-mode", type=str, default=SELECTION_MODE)
    parser.add_argument("--selection-tag", type=str, default="topN")
    parser.add_argument("--n-zshells", type=int, default=1)
    parser.add_argument("--mass-threshold-hmsun", type=float, default=None)
    parser.add_argument("--random-radial-policy", type=str, default=None)
    args = parser.parse_args()

    ensure_task43_dirs()
    target_count = None if str(args.selection_mode) == "fixed_mass_threshold_ninterp" else int(args.target_count)
    rows = build_rows(
        target_count=target_count,
        random_multiplier=args.random_multiplier,
        selection_mode=str(args.selection_mode),
        selection_tag=str(args.selection_tag),
        n_zshells=int(args.n_zshells),
        mass_threshold_hmsun=args.mass_threshold_hmsun,
        random_radial_policy=args.random_radial_policy,
    )
    write_jsonl(args.output, rows)
    missing = sum(bool(row["missing_shell_paths"]) for row in rows)
    print(f"[task43] wrote {len(rows)} rows to {args.output}")
    print(f"[task43] rows_with_missing_shells={missing}")
    if missing:
        for row in rows:
            if row["missing_shell_paths"]:
                print(f"[missing] {row['phase']} {row['missing_shell_paths']}")


if __name__ == "__main__":
    main()
