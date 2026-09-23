#!/usr/bin/env python3
"""Frozen paths and geometry for the paired pngbase pseudo-lightcones."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
ABACUS_ROOT = Path("/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit")
OUTPUT_ROOT = PROJECT_ROOT / "outputs/task43_outputs/pngbase_pseudolc_fnl0_fnl100"
FINAL_ROOT = PROJECT_ROOT / "9.11meeting/task43_pngbase_pseudolc_fnl0_fnl100_kmax0p08_smin50"
PNG_COV_ROOT = OUTPUT_ROOT / "covariance" / "c302_fnl100"

COSMOLOGIES = ("c000", "c302")
PHASES = ("ph000", "ph001")
FNL_BY_COSMOLOGY = {"c000": 0.0, "c302": 100.0}
SEED_BY_PHASE = {"ph000": 12321, "ph001": 12421}

SNAPSHOT_REDSHIFT = 0.725
SNAPSHOT_TAG = "z0.725"
BOX_SIZE_MPC_H = 2000.0
OBSERVER_MPC_H = (-990.0, -990.0, -990.0)
MASS_THRESHOLD_HMSUN = 1.4e13
P0_FKP = 10000.0
RANDOM_MULTIPLIER = 25
SPACE_WINDOWS = {"real": (0.6, 0.8), "rsd": (0.4, 0.8)}


def validate_case(cosmology: str, phase: str | None = None, space: str | None = None) -> None:
    if cosmology not in COSMOLOGIES:
        raise ValueError(f"unknown cosmology {cosmology!r}; expected one of {COSMOLOGIES}")
    if phase is not None and phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
    if space is not None and space not in SPACE_WINDOWS:
        raise ValueError(f"unknown space {space!r}; expected one of {tuple(SPACE_WINDOWS)}")


def sim_name(cosmology: str, phase: str) -> str:
    validate_case(cosmology, phase)
    return f"Abacus_pngbase_{cosmology}_{phase}"


def halo_dir(cosmology: str, phase: str) -> Path:
    return ABACUS_ROOT / sim_name(cosmology, phase) / "halos" / SNAPSHOT_TAG / "halo_info"


def catalog_path(cosmology: str, phase: str, space: str) -> Path:
    validate_case(cosmology, phase, space)
    ztag = "zgeom0p6_0p8" if space == "real" else "zobs0p4_0p8"
    return OUTPUT_ROOT / "catalogs" / space / (
        f"task43_pngbase_pseudolc_{sim_name(cosmology, phase)}_{space}_{ztag}_mmin1p4e13.npz"
    )


def random_path(cosmology: str, phase: str, space: str) -> Path:
    ztag = "zgeom0p6_0p8" if space == "real" else "zobs0p4_0p8"
    return OUTPUT_ROOT / "randoms" / space / (
        f"task43_pngbase_pseudolc_random_{sim_name(cosmology, phase)}_{space}_{ztag}_x25.npz"
    )


def fkp_path(cosmology: str, phase: str, space: str) -> Path:
    ztag = "zgeom0p6_0p8" if space == "real" else "zobs0p4_0p8"
    return OUTPUT_ROOT / "fkp" / space / (
        f"task43_pngbase_pseudolc_fkp_{sim_name(cosmology, phase)}_{space}_{ztag}_dz0p01.npz"
    )


def xi_path(cosmology: str, phase: str, space: str) -> Path:
    ztag = "zgeom0p6_0p8" if space == "real" else "zobs0p4_0p8"
    return OUTPUT_ROOT / "xi" / space / (
        f"task43_pngbase_pseudolc_xi02_{sim_name(cosmology, phase)}_{space}_{ztag}_x25_s30_350_ds10.npz"
    )


def pk_tag(cosmology: str, space: str) -> str:
    validate_case(cosmology, space=space)
    return f"pngbase_pseudolc/{space}/{cosmology}_fkpP010000"


def pk_measurement_dir(cosmology: str, space: str) -> Path:
    return PROJECT_ROOT / "outputs/task43_outputs/rsd_validation/lightcone/pk" / pk_tag(cosmology, space)


def pk_path(cosmology: str, phase: str, space: str) -> Path:
    return pk_measurement_dir(cosmology, space) / (
        f"task43_rsd_lightcone_p02_{phase}_mesh256_kmax0p300_dk0p002.npz"
    )


def manifest_path(cosmology: str, space: str) -> Path:
    validate_case(cosmology, space=space)
    return OUTPUT_ROOT / "manifests" / f"task43_pngbase_pseudolc_{cosmology}_{space}_x25.jsonl"


def fit_root(cosmology: str) -> Path:
    validate_case(cosmology)
    return OUTPUT_ROOT / "fits" / cosmology / "kmax0p08_smin50"


def png_covariance_path() -> Path:
    return PNG_COV_ROOT / "task43_pngbase_c302_fnl100_covariance_kmax0p08_smin50_baomask80_120.npz"


def fit_root_png_cov() -> Path:
    return OUTPUT_ROOT / "fits" / "c302_cov_fnl100" / "kmax0p08_smin50"
