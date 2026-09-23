#!/usr/bin/env python3
"""Write immutable two-phase manifests for the paired pngbase experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from task43_pngbase_pseudolc_common import (
    COSMOLOGIES,
    FNL_BY_COSMOLOGY,
    MASS_THRESHOLD_HMSUN,
    OBSERVER_MPC_H,
    OUTPUT_ROOT,
    P0_FKP,
    PHASES,
    RANDOM_MULTIPLIER,
    SEED_BY_PHASE,
    SNAPSHOT_REDSHIFT,
    SPACE_WINDOWS,
    catalog_path,
    fkp_path,
    manifest_path,
    pk_tag,
    random_path,
    sim_name,
    xi_path,
)


def row(cosmology: str, phase: str, space: str) -> dict[str, Any]:
    data = catalog_path(cosmology, phase, space)
    random = random_path(cosmology, phase, space)
    fkp = fkp_path(cosmology, phase, space)
    xi = xi_path(cosmology, phase, space)
    zmin, zmax = SPACE_WINDOWS[space]
    return {
        "analysis_scope": "task43_pngbase_paired_snapshot_shell_pseudolightcone",
        "analysis_tag": f"pngbase_pseudolc_{space}",
        "cosmology": cosmology,
        "space": space,
        "phase": phase,
        "phase_index": PHASES.index(phase),
        "sim_name": sim_name(cosmology, phase),
        "initial_condition_seed": SEED_BY_PHASE[phase],
        "injected_fnl": FNL_BY_COSMOLOGY[cosmology],
        "snapshot_redshift": SNAPSHOT_REDSHIFT,
        "observer_mpc_h": list(OBSERVER_MPC_H),
        "mass_threshold_hmsun": MASS_THRESHOLD_HMSUN,
        "zmin_observed": zmin,
        "zmax_observed": zmax,
        "p0_fkp": P0_FKP,
        "p_fixed": 1.0,
        "random_multiplier": RANDOM_MULTIPLIER,
        "random_radial_policy": "phase-matched observed-z resampling; uniform positive-octant angles",
        "lightcone_catalog_path": str(data),
        "lightcone_metadata_path": str(data.with_suffix(".json")),
        "lightcone_random_path": str(random),
        "lightcone_random_metadata_path": str(random.with_suffix(".json")),
        "lightcone_fkp_path": str(fkp),
        "lightcone_fkp_metadata_path": str(fkp.with_suffix(".json")),
        "lightcone_xi_path": str(xi),
        "p02_tag": pk_tag(cosmology, space),
        "fit_policy": "only the arithmetic ph000/ph001 observable mean is fit; no single-phase fNL fits",
        "covariance_policy": "Task4.3 single-realization covariance; never divided by two",
    }


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    summary: dict[str, Any] = {
        "task": "task43_make_pngbase_pseudolc_manifests",
        "status": "pass",
        "manifests": {},
        "n_catalogs": 0,
        "fit_policy": "two cosmology-level fits only; each data vector is the ph000/ph001 arithmetic mean",
    }
    for cosmology in COSMOLOGIES:
        for space in SPACE_WINDOWS:
            rows = [row(cosmology, phase, space) for phase in PHASES]
            path = manifest_path(cosmology, space)
            atomic_write(path, "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows))
            summary["manifests"][f"{cosmology}_{space}"] = str(path)
            summary["n_catalogs"] += len(rows)
    atomic_write(
        OUTPUT_ROOT / "manifests/task43_pngbase_pseudolc_manifest_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

