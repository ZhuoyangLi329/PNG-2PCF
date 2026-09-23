#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared constants for Task43 AbacusSummit halo lightcone tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
LIGHTCONE_ROOT = Path("/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/halo_light_cones")
SIM_PREFIX = "AbacusSummit_base_c000"

PHASES = tuple(f"ph{i:03d}" for i in range(25))
ZMIN = 0.6
ZMAX = 0.8
READ_SHELLS = ("z0.575", "z0.650", "z0.725", "z0.800")
S_EDGES = np.arange(50.0, 360.0, 10.0, dtype="f8")
# AbacusSummit base_c000 boxes used by the current Task43 lightcone closure
# tests are 2 Gpc/h on a side.  Older diagnostic code used 3000 here, which
# produces a different fundamental-mode grid and must not be the Task43 default.
BOX_SIZE = 2000.0
K_FUND = 2.0 * np.pi / BOX_SIZE

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
HALO_CATALOG_DIR = OUTPUT_ROOT / "halo_catalogs"
RANDOM_DIR = OUTPUT_ROOT / "randoms"
XI_DIR = OUTPUT_ROOT / "xi_cucount"
RASCALC_DIR = OUTPUT_ROOT / "rascalc_covariance"
FIT_DIR = OUTPUT_ROOT / "fits"
SUMMARY_DIR = OUTPUT_ROOT / "summary"
PLOT_DIR = PROJECT_ROOT / "plots" / "task43"
LOG_DIR = PROJECT_ROOT / "codes" / "logs" / "task43"

DEFAULT_MANIFEST = MANIFEST_DIR / "task43_phases.jsonl"
DEFAULT_TARGET_COUNT = 200_000
DEFAULT_RANDOM_MULTIPLIER = 50
SELECTION_MODE = "fixed_count_top_N_interp"
SPACE_MODE = "real"
OBSERVER_ORIGIN_POLICY = "first_LightConeOrigins_triplet"


def phase_sim_name(phase: str) -> str:
    """Return the AbacusSummit simulation directory name for a phase."""
    return f"{SIM_PREFIX}_{phase}"


def phase_dir(phase: str) -> Path:
    """Return the public halo-lightcone directory for a phase."""
    return LIGHTCONE_ROOT / phase_sim_name(phase)


def shell_info_path(phase: str, shell: str) -> Path:
    """Return the lc_halo_info.asdf path for one phase and redshift shell."""
    return phase_dir(phase) / shell / "lc_halo_info.asdf"


def ensure_task43_dirs() -> None:
    """Create Task43 output directories."""
    for path in (
        MANIFEST_DIR,
        HALO_CATALOG_DIR,
        RANDOM_DIR,
        XI_DIR,
        RASCALC_DIR,
        FIT_DIR,
        SUMMARY_DIR,
        PLOT_DIR,
        LOG_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
