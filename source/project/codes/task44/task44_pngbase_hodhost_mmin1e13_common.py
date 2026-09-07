#!/usr/bin/env python3
"""Frozen paths and numerical contracts for the Task44 halo-clustering extension.

This extension keeps the existing Task44 products immutable.  Its halo
sample is the intersection of the exact HOD host IDs and the officially
cleaned CompaSO halos with M = N_cleaned * ParticleMassHMsun >= 1e13 Msun/h.
Both the legacy galaxy P0 cross-check and the matched host-halo P0 fit use a
strict physical mode cut k >= 0.006 h/Mpc while retaining the previous
upper-bin convention [0.099, 0.101).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from task44_pngbase_hodmap_rawbox_common import (
    BOX_SIZE,
    BOX_VOLUME,
    CATALOGS,
    K_FUND,
    PK_PRIMARY_FIT_EDGES,
    PROJECT_ROOT,
    REDSHIFT,
    atomic_savez,
    atomic_write_json,
    get_spec,
    load_positions,
    set_cpu_affinity,
    sha256_file,
    to_jsonable,
)


EXTENSION_LABEL = "hodhost_mmin1e13_strictk0p006"
BASE_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task44_outputs" / "pngbase_hodmap_rawbox_z0p500"
BASE_PLOT_ROOT = PROJECT_ROOT / "plots" / "task44" / "pngbase_hodmap_rawbox_z0p500"
OUTPUT_ROOT = BASE_OUTPUT_ROOT / EXTENSION_LABEL
PLOT_ROOT = BASE_PLOT_ROOT / EXTENSION_LABEL
CATALOG_DIR = OUTPUT_ROOT / "catalogs"
XI_ENGINE_DIR = OUTPUT_ROOT / "xi_engines"
PK_DIR = OUTPUT_ROOT / "pk"
FIT_DIR = OUTPUT_ROOT / "fits"
SUMMARY_DIR = OUTPUT_ROOT / "summary"
LOG_DIR = OUTPUT_ROOT / "logs"

RAW_ABACUS_ROOT = Path("/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit")
HALO_MASS_MIN_HMSUN = 1.0e13
S_EDGES = np.arange(30.0, 350.0 + 10.0, 10.0, dtype="f8")
S_CENTERS = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
XI_FIT_SMIN = 30.0
XI_FIT_SMAX = 150.0
XI_FIT_EDGES = np.arange(XI_FIT_SMIN, XI_FIT_SMAX + 10.0, 10.0, dtype="f8")
XI_FIT_CENTERS = 0.5 * (XI_FIT_EDGES[:-1] + XI_FIT_EDGES[1:])
LRG_EXTENDED_XI_SMIN = 30.0
LRG_EXTENDED_XI_SMAX = 350.0
LRG_EXTENDED_XI_FIT_EDGES = np.arange(
    LRG_EXTENDED_XI_SMIN, LRG_EXTENDED_XI_SMAX + 10.0, 10.0, dtype="f8"
)
LRG_EXTENDED_XI_FIT_CENTERS = 0.5 * (
    LRG_EXTENDED_XI_FIT_EDGES[:-1] + LRG_EXTENDED_XI_FIT_EDGES[1:]
)
LRG_CONSERVATIVE_XI_SMIN = 50.0
LRG_CONSERVATIVE_XI_SMAX = 150.0
LRG_CONSERVATIVE_XI_FIT_EDGES = np.arange(
    LRG_CONSERVATIVE_XI_SMIN, LRG_CONSERVATIVE_XI_SMAX + 10.0, 10.0, dtype="f8"
)
LRG_CONSERVATIVE_XI_FIT_CENTERS = 0.5 * (
    LRG_CONSERVATIVE_XI_FIT_EDGES[:-1] + LRG_CONSERVATIVE_XI_FIT_EDGES[1:]
)

# The old periodic fit bins are
# [0.003,0.005), [0.005,0.007), [0.007,0.009), ..., [0.099,0.101).
# A strict k >= 0.006 cut bisects the second bin.  Remeasure that shell as
# [0.006,0.007), then retain every old shell from [0.007,0.009) onward.
PK_STRICT_KMIN = 0.006
PK_KMAX_CONTRACT = 0.100
P_GAUSSIAN_PRIOR_MEAN = 0.7072363788448802
P_GAUSSIAN_PRIOR_SIGMA = 0.2694701311625559
PK_STRICT_FIT_EDGES = np.vstack(
    [np.asarray([[0.006, 0.007]], dtype="f8"), np.asarray(PK_PRIMARY_FIT_EDGES[2:], dtype="f8")]
)
LRG_CONSERVATIVE_PK_KMAX = 0.080
LRG_CONSERVATIVE_PK_FIT_EDGES = PK_STRICT_FIT_EDGES[
    np.mean(PK_STRICT_FIT_EDGES, axis=1) <= LRG_CONSERVATIVE_PK_KMAX + 1.0e-13
].copy()


def ensure_output_dirs() -> None:
    for path in (
        OUTPUT_ROOT,
        PLOT_ROOT,
        CATALOG_DIR,
        XI_ENGINE_DIR,
        PK_DIR,
        FIT_DIR,
        SUMMARY_DIR,
        LOG_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def raw_halo_files(tag: str) -> tuple[Path, ...]:
    spec = get_spec(tag)
    root = RAW_ABACUS_ROOT / spec.realization / "halos" / "z0.500" / "halo_info"
    paths = tuple(sorted(root.glob("halo_info_*.asdf")))
    if len(paths) != 17:
        raise RuntimeError(f"{tag} requires 17 halo_info ASDF shards, found {len(paths)} under {root}")
    return paths


def host_catalog_path(tag: str) -> Path:
    get_spec(tag)
    return CATALOG_DIR / f"task44_pngbase_{tag}_hodhost_mmin1e13_z0p500_realspace_periodic.npz"


def host_catalog_metadata_path(tag: str) -> Path:
    return host_catalog_path(tag).with_suffix(".json")


def host_ascii_path(tag: str) -> Path:
    get_spec(tag)
    return CATALOG_DIR / f"task44_pngbase_{tag}_hodhost_mmin1e13_z0p500_realspace_periodic_xyz.txt"


def host_ascii_metadata_path(tag: str) -> Path:
    return host_ascii_path(tag).with_suffix(".json")


def xi_engine_path(tag: str, engine: str) -> Path:
    get_spec(tag)
    if engine not in ("fcfc", "pycorr", "cucount"):
        raise ValueError(engine)
    return XI_ENGINE_DIR / engine / f"task44_pngbase_{tag}_hodhost_mmin1e13_xi0_s30_350_{engine}.npz"


def xi_engine_metadata_path(tag: str, engine: str) -> Path:
    return xi_engine_path(tag, engine).with_suffix(".json")


def strict_pk_path(tag: str, *, mesh: int = 400) -> Path:
    get_spec(tag)
    return PK_DIR / f"task44_pngbase_{tag}_hodmap_lrg_z0p500_pk0_strictk0p006_mesh{int(mesh)}.npz"


def strict_pk_metadata_path(tag: str, *, mesh: int = 400) -> Path:
    return strict_pk_path(tag, mesh=mesh).with_suffix(".json")


def host_pk_path(tag: str, *, mesh: int = 400) -> Path:
    get_spec(tag)
    return PK_DIR / f"task44_pngbase_{tag}_hodhost_mmin1e13_z0p500_pk0_strictk0p006_mesh{int(mesh)}.npz"


def host_pk_metadata_path(tag: str, *, mesh: int = 400) -> Path:
    return host_pk_path(tag, mesh=mesh).with_suffix(".json")


def strict_fit_prefix(tag: str) -> Path:
    get_spec(tag)
    label = "pk0_strictkmin0p006_kcentermax0p100_fixedfnl_freep"
    return FIT_DIR / tag / label / f"task44_pngbase_{tag}_{label}"


def lrg_pk_gaussianp_fit_prefix(tag: str) -> Path:
    get_spec(tag)
    label = "hodmap_lrg_pk0_strictkmin0p006_kcentermax0p100_gaussianp_freefnl"
    return FIT_DIR / tag / label / f"task44_pngbase_{tag}_{label}"


def lrg_pk_conservative_fit_prefix(tag: str) -> Path:
    get_spec(tag)
    label = "hodmap_lrg_pk0_strictkmin0p006_kcentermax0p080_fixedfnl_freep"
    return FIT_DIR / tag / label / f"task44_pngbase_{tag}_{label}"


def lrg_pk_conservative_gaussianp_fit_prefix(tag: str) -> Path:
    get_spec(tag)
    label = "hodmap_lrg_pk0_strictkmin0p006_kcentermax0p080_gaussianp_freefnl"
    return FIT_DIR / tag / label / f"task44_pngbase_{tag}_{label}"


def lrg_xi_fit_prefix(tag: str) -> Path:
    get_spec(tag)
    label = "hodmap_lrg_xi0_smin30_smax150_fixedfnl_freep"
    return FIT_DIR / tag / label / f"task44_pngbase_{tag}_{label}"


def lrg_xi_gaussianp_fit_prefix(tag: str) -> Path:
    get_spec(tag)
    label = "hodmap_lrg_xi0_smin30_smax150_gaussianp_freefnl"
    return FIT_DIR / tag / label / f"task44_pngbase_{tag}_{label}"


def lrg_xi_extended_fit_prefix(tag: str) -> Path:
    get_spec(tag)
    label = "hodmap_lrg_xi0_smin30_smax350_fixedfnl_freep"
    return FIT_DIR / tag / label / f"task44_pngbase_{tag}_{label}"


def lrg_xi_extended_gaussianp_fit_prefix(tag: str) -> Path:
    get_spec(tag)
    label = "hodmap_lrg_xi0_smin30_smax350_gaussianp_freefnl"
    return FIT_DIR / tag / label / f"task44_pngbase_{tag}_{label}"


def lrg_xi_conservative_fit_prefix(tag: str) -> Path:
    get_spec(tag)
    label = "hodmap_lrg_xi0_smin50_smax150_fixedfnl_freep"
    return FIT_DIR / tag / label / f"task44_pngbase_{tag}_{label}"


def lrg_xi_conservative_gaussianp_fit_prefix(tag: str) -> Path:
    get_spec(tag)
    label = "hodmap_lrg_xi0_smin50_smax150_gaussianp_freefnl"
    return FIT_DIR / tag / label / f"task44_pngbase_{tag}_{label}"


def host_pk_fit_prefix(tag: str) -> Path:
    get_spec(tag)
    label = "hodhost_mmin1e13_pk0_strictkmin0p006_kcentermax0p100_fixedfnl_freep"
    return FIT_DIR / tag / label / f"task44_pngbase_{tag}_{label}"


def host_pk_gaussianp_fit_prefix(tag: str) -> Path:
    get_spec(tag)
    label = "hodhost_mmin1e13_pk0_strictkmin0p006_kcentermax0p100_gaussianp_freefnl"
    return FIT_DIR / tag / label / f"task44_pngbase_{tag}_{label}"


def host_xi_fit_prefix(tag: str) -> Path:
    get_spec(tag)
    label = "hodhost_mmin1e13_xi0_smin30_smax150_fixedfnl_freep"
    return FIT_DIR / tag / label / f"task44_pngbase_{tag}_{label}"


def host_xi_gaussianp_fit_prefix(tag: str) -> Path:
    get_spec(tag)
    label = "hodhost_mmin1e13_xi0_smin30_smax150_gaussianp_freefnl"
    return FIT_DIR / tag / label / f"task44_pngbase_{tag}_{label}"


def load_host_catalog(tag: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = host_catalog_path(tag)
    metadata_path = host_catalog_metadata_path(tag)
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing host catalog: {path} / {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "pass" or metadata.get("output_sha256") != sha256_file(path):
        raise RuntimeError(f"host catalog metadata/hash validation failed: {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    position = np.asarray(arrays["position"], dtype="f8")
    ndata = int(np.asarray(arrays["ndata"]).item())
    if position.shape != (ndata, 3):
        raise RuntimeError(f"invalid position shape in {path}: {position.shape} != {(ndata, 3)}")
    if np.any(position < 0.0) or np.any(position >= BOX_SIZE):
        raise RuntimeError(f"positive periodic host coordinates leave [0,L): {path}")
    return arrays, metadata


def analytic_rr(edges: np.ndarray = S_EDGES) -> np.ndarray:
    values = np.asarray(edges, dtype="f8")
    if values[-1] >= BOX_SIZE / 2.0:
        raise ValueError("analytic spherical-shell RR requires smax < L/2")
    return 4.0 * np.pi / 3.0 * (values[1:] ** 3 - values[:-1] ** 3) / BOX_VOLUME


__all__ = [
    "BASE_OUTPUT_ROOT",
    "BASE_PLOT_ROOT",
    "BOX_SIZE",
    "BOX_VOLUME",
    "CATALOGS",
    "CATALOG_DIR",
    "FIT_DIR",
    "HALO_MASS_MIN_HMSUN",
    "K_FUND",
    "LOG_DIR",
    "LRG_CONSERVATIVE_PK_FIT_EDGES",
    "LRG_CONSERVATIVE_PK_KMAX",
    "LRG_EXTENDED_XI_FIT_CENTERS",
    "LRG_EXTENDED_XI_FIT_EDGES",
    "LRG_EXTENDED_XI_SMAX",
    "LRG_EXTENDED_XI_SMIN",
    "LRG_CONSERVATIVE_XI_FIT_CENTERS",
    "LRG_CONSERVATIVE_XI_FIT_EDGES",
    "LRG_CONSERVATIVE_XI_SMAX",
    "LRG_CONSERVATIVE_XI_SMIN",
    "OUTPUT_ROOT",
    "PK_DIR",
    "PK_KMAX_CONTRACT",
    "P_GAUSSIAN_PRIOR_MEAN",
    "P_GAUSSIAN_PRIOR_SIGMA",
    "PK_STRICT_FIT_EDGES",
    "PK_STRICT_KMIN",
    "PLOT_ROOT",
    "PROJECT_ROOT",
    "REDSHIFT",
    "S_CENTERS",
    "S_EDGES",
    "SUMMARY_DIR",
    "XI_ENGINE_DIR",
    "XI_FIT_CENTERS",
    "XI_FIT_EDGES",
    "XI_FIT_SMAX",
    "XI_FIT_SMIN",
    "analytic_rr",
    "atomic_savez",
    "atomic_write_json",
    "ensure_output_dirs",
    "get_spec",
    "host_ascii_metadata_path",
    "host_ascii_path",
    "host_catalog_metadata_path",
    "host_catalog_path",
    "host_pk_fit_prefix",
    "host_pk_gaussianp_fit_prefix",
    "host_pk_metadata_path",
    "host_pk_path",
    "host_xi_fit_prefix",
    "host_xi_gaussianp_fit_prefix",
    "load_host_catalog",
    "lrg_pk_conservative_fit_prefix",
    "lrg_pk_conservative_gaussianp_fit_prefix",
    "lrg_xi_conservative_fit_prefix",
    "lrg_xi_conservative_gaussianp_fit_prefix",
    "lrg_xi_extended_fit_prefix",
    "lrg_xi_extended_gaussianp_fit_prefix",
    "load_positions",
    "raw_halo_files",
    "set_cpu_affinity",
    "sha256_file",
    "strict_fit_prefix",
    "strict_pk_metadata_path",
    "strict_pk_path",
    "to_jsonable",
    "xi_engine_metadata_path",
    "xi_engine_path",
]
