#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared constants for Task44 lightcone redshift-space tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")

TRACER = "LRG"
SAMPLE = "lrg2"
REALIZATION = "ph000_HODv4"
ZMIN = 0.6
ZMAX = 0.8
P0_DEFAULT = 10000.0

LRG_BINS: dict[str, dict[str, Any]] = {
    "lrg1": {"label": "LRG1", "tracer": "LRG", "zmin": 0.4, "zmax": 0.6, "p_fixed": 0.7072},
    "lrg2": {"label": "LRG2", "tracer": "LRG", "zmin": 0.6, "zmax": 0.8, "p_fixed": 1.1878},
    "lrg3": {"label": "LRG3", "tracer": "LRG", "zmin": 0.8, "zmax": 1.1, "p_fixed": 1.7967},
    "lrgall": {"label": "LRGxLRG", "tracer": "LRG", "zmin": 0.4, "zmax": 1.1, "p_fixed": 1.0},
}

QSO_BINS: dict[str, dict[str, Any]] = {
    # The three QSO bins are pre-registered for the random20x P(k)-2PCF
    # consistency experiment.  Their object-weight-squared sums are nearly
    # balanced; p is intentionally held fixed across redshift by user request.
    "qso1": {"label": "QSO1", "tracer": "QSO", "zmin": 0.8, "zmax": 1.5, "p_fixed": 1.6},
    "qso2": {"label": "QSO2", "tracer": "QSO", "zmin": 1.5, "zmax": 2.1, "p_fixed": 1.6},
    "qso3": {"label": "QSO3", "tracer": "QSO", "zmin": 2.1, "zmax": 3.5, "p_fixed": 1.6},
    "qsoall": {"label": "QSO", "tracer": "QSO", "zmin": 0.8, "zmax": 3.5, "p_fixed": 1.6},
}

SAMPLE_CONFIGS: dict[str, dict[str, Any]] = {**LRG_BINS, **QSO_BINS}
P_FIXED_DEFAULT = float(SAMPLE_CONFIGS[SAMPLE]["p_fixed"])

REALIZATION_CONFIGS: dict[str, dict[str, Any]] = {
    "ph000_HODv4": {
        "label": "ph000_HODv4",
        "lsscat_dir": Path("/pscratch/sd/s/siyizhao/fihobi/lc_test/lc_LRG_fnl100_base-A_HODv4/LSScat"),
        "lsscat_dirs": {
            "LRG": Path("/pscratch/sd/s/siyizhao/fihobi/lc_test/lc_LRG_fnl100_base-A_HODv4/LSScat"),
            "QSO": Path("/pscratch/sd/s/siyizhao/fihobi/lc_test/lc_QSO_fnl100_base-A_HODv4/LSScat"),
        },
    },
    "ph001": {
        "label": "ph001",
        "lsscat_dir": Path("/pscratch/sd/s/siyizhao/primordial-ng-lab/bphi_hod/lc_mock/ph001/LRG/LSScat"),
        "lsscat_dirs": {
            "LRG": Path("/pscratch/sd/s/siyizhao/primordial-ng-lab/bphi_hod/lc_mock/ph001/LRG/LSScat"),
            "QSO": Path("/pscratch/sd/s/siyizhao/primordial-ng-lab/bphi_hod/lc_mock/ph001/QSO/LSScat"),
        },
    },
}
LRG_HODV4_ROOT = REALIZATION_CONFIGS["ph000_HODv4"]["lsscat_dir"].parent
LSSCAT_DIR = REALIZATION_CONFIGS["ph000_HODv4"]["lsscat_dir"]
CAPS = ("NGC", "SGC")
RANDOM_INDICES = tuple(range(10))

S_EDGES = np.arange(50.0, 360.0, 10.0, dtype="f8")
BOX_SIZE = 2000.0
K_FUND = 2.0 * np.pi / BOX_SIZE

# CUTSKY configs in the mock directory use these background parameters.
OMEGA_M = 0.315192
OMEGA_L = 0.684808
C_OVER_100 = 2997.92458  # c / (100 km/s/Mpc), in Mpc/h.

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task44_outputs"
CATALOG_DIR = OUTPUT_ROOT / "catalogs"
XI_DIR = OUTPUT_ROOT / "xi"
COV_DIR = OUTPUT_ROOT / "covariance"
FIT_DIR = OUTPUT_ROOT / "fits"
SUMMARY_DIR = OUTPUT_ROOT / "summary"
PLOT_DIR = PROJECT_ROOT / "plots" / "task44"
LOG_DIR = PROJECT_ROOT / "codes" / "logs" / "task44"

DEFAULT_DATA_CATALOG = CATALOG_DIR / "task44_ph000_HODv4_LRG2_data_zobs_fkpP010000.npz"
DEFAULT_RANDOM_CATALOG = CATALOG_DIR / "task44_ph000_HODv4_LRG2_randoms0-9_zobs_fkpP010000.npz"
DEFAULT_ZEFF = SUMMARY_DIR / "task44_ph000_HODv4_LRG2_zeff_zobs_fkpP010000.npz"
DEFAULT_XI = XI_DIR / "task44_ph000_HODv4_LRG2_xi0_s50_350_ds10_zobs_fkpP010000.npz"


def ensure_task44_dirs() -> None:
    for path in (CATALOG_DIR, XI_DIR, COV_DIR, FIT_DIR, SUMMARY_DIR, PLOT_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def normalize_sample(name: str) -> str:
    key = str(name).strip().lower()
    if key not in SAMPLE_CONFIGS:
        raise ValueError(f"unknown Task44 sample {name!r}; choices={sorted(SAMPLE_CONFIGS)}")
    return key


def normalize_lrg_bin(name: str) -> str:
    return normalize_sample(name)


def normalize_realization(name: str) -> str:
    key = str(name).strip()
    if key not in REALIZATION_CONFIGS:
        raise ValueError(f"unknown Task44 realization {name!r}; choices={sorted(REALIZATION_CONFIGS)}")
    return key


def realization_config(name: str = REALIZATION, *, sample: str | None = None, tracer: str | None = None) -> dict[str, Any]:
    key = normalize_realization(name)
    config = dict(REALIZATION_CONFIGS[key])
    config["name"] = key
    tracer_name = str(tracer or (sample_tracer(sample) if sample is not None else TRACER))
    lsscat_dirs = config.get("lsscat_dirs")
    if isinstance(lsscat_dirs, dict) and tracer_name in lsscat_dirs:
        config["lsscat_dir"] = Path(lsscat_dirs[tracer_name])
    config["tracer"] = tracer_name
    return config


def sample_config(name: str) -> dict[str, Any]:
    key = normalize_sample(name)
    config = dict(SAMPLE_CONFIGS[key])
    config["name"] = key
    return config


def lrg_bin_config(name: str) -> dict[str, Any]:
    return sample_config(name)


def sample_label(name: str) -> str:
    return str(sample_config(name)["label"])


def lrg_bin_label(name: str) -> str:
    return sample_label(name)


def sample_tracer(name: str | None) -> str:
    if name is None:
        return TRACER
    return str(sample_config(name).get("tracer", TRACER))


def format_float_tag(value: float) -> str:
    return f"{float(value):.6g}".replace(".", "p").replace("-", "m")


def format_p_fixed_tag(value: float) -> str:
    return "p" + format_float_tag(float(value))


def catalog_output_paths(
    sample: str,
    *,
    realization: str = REALIZATION,
    p0: float = P0_DEFAULT,
    z_label: str = "zobs",
    random_indices: tuple[int, ...] = RANDOM_INDICES,
    smoke_tag: str | None = None,
) -> tuple[Path, Path]:
    label = sample_label(sample)
    rtag = str(realization_config(realization, sample=sample)["label"])
    ptag = format_p0_tag(float(p0))
    suffix = "" if not smoke_tag else f"_{smoke_tag}"
    random_tag = compact_random_tag(tuple(random_indices))
    stem = f"task44_{rtag}_{label}"
    data_path = CATALOG_DIR / f"{stem}_data_{z_label}_{ptag}{suffix}.npz"
    random_path = CATALOG_DIR / f"{stem}_{random_tag}_{z_label}_{ptag}{suffix}.npz"
    return data_path, random_path


def zeff_output_path(sample: str, *, realization: str = REALIZATION, p0: float = P0_DEFAULT, z_label: str = "zobs") -> Path:
    label = sample_label(sample)
    rtag = str(realization_config(realization, sample=sample)["label"])
    ptag = format_p0_tag(float(p0))
    return SUMMARY_DIR / f"task44_{rtag}_{label}_zeff_{z_label}_{ptag}.npz"


def xi_output_path(
    sample: str,
    *,
    realization: str = REALIZATION,
    p0: float = P0_DEFAULT,
    z_label: str = "zobs",
    s_min: float = 50.0,
    s_max: float = 350.0,
    ds: float = 10.0,
) -> Path:
    label = sample_label(sample)
    rtag = str(realization_config(realization, sample=sample)["label"])
    ptag = format_p0_tag(float(p0))
    return XI_DIR / f"task44_{rtag}_{label}_xi0_s{float(s_min):.0f}_{float(s_max):.0f}_ds{float(ds):.0f}_{z_label}_{ptag}.npz"


def rr_smu_prefix(sample: str, *, nran: int = 100000, seed: int = 20260706) -> Path:
    return COV_DIR / f"task44_{normalize_sample(sample)}_rr_smu_window_nran{int(nran / 1000)}k_s50_350_ds10_nmu20_seed{int(seed)}"


def covariance_prefix(
    sample: str,
    *,
    rrdeconv: bool,
    p_fixed: float | None = None,
    b1_cov: float | None = None,
    sigma_s_cov: float | None = None,
    k_grid_tag: str = "kbox",
) -> Path:
    """Return the Task44 xi covariance prefix.

    The ``p...`` tag is the PNG-bias ``p_fixed`` value, not the FKP ``P0``.
    """

    suffix = "rrdeconv_" if rrdeconv else ""
    p_value = float(sample_config(sample)["p_fixed"] if p_fixed is None else p_fixed)
    cov_tag = format_p_fixed_tag(p_value)
    if b1_cov is not None:
        cov_tag += f"_b1cov{format_float_tag(float(b1_cov))}"
    if sigma_s_cov is not None:
        cov_tag += f"_sigmas{format_float_tag(float(sigma_s_cov))}"
    return COV_DIR / (
        f"task44_{normalize_sample(sample)}_fnl0_jaxpower_covariance_"
        f"rsdpoles024_win02468_bessel_interp_smoothfftlog_{suffix}"
        f"mesh64_nran100k_ndata50k_pad400_win3600_ds2_{str(k_grid_tag)}_3000_dk002_{cov_tag}_s50_350_ds10"
    )


def formal_gic_window_path(sample: str, *, p0: float = P0_DEFAULT, z_label: str = "zobs", nsub: int = 200000, seed: int = 20260704) -> Path:
    ptag = format_p0_tag(float(p0))
    return SUMMARY_DIR / f"task44_{normalize_sample(sample)}_formal_gic_window_{z_label}_{ptag}_L2000_nsub{int(nsub)}_seed{int(seed)}.npz"


def fit_output_dir(sample: str, *, smin: float, p_fixed: float) -> Path:
    return FIT_DIR / f"{normalize_sample(sample)}_rsd_monopole_smin{int(smin)}_{format_p_fixed_tag(float(p_fixed))}_sigmas_rsdpoles024_repairedcov"


def ls_data_path(cap: str, *, realization: str = REALIZATION, sample: str = SAMPLE, tracer: str | None = None) -> Path:
    tracer_name = str(tracer or sample_tracer(sample))
    return Path(realization_config(realization, sample=sample, tracer=tracer_name)["lsscat_dir"]) / f"{tracer_name}_{cap}_clustering.dat.h5"


def ls_random_path(cap: str, random_index: int, *, realization: str = REALIZATION, sample: str = SAMPLE, tracer: str | None = None) -> Path:
    tracer_name = str(tracer or sample_tracer(sample))
    return Path(realization_config(realization, sample=sample, tracer=tracer_name)["lsscat_dir"]) / f"{tracer_name}_{cap}_{int(random_index)}_clustering.ran.h5"


def format_p0_tag(p0: float) -> str:
    value = float(p0)
    if value.is_integer():
        return f"fkpP0{int(value)}"
    return "fkpP0" + f"{value:.6g}".replace(".", "p").replace("-", "m")


def compact_random_tag(indices: list[int] | tuple[int, ...]) -> str:
    values = sorted(int(v) for v in indices)
    if not values:
        return "randomsnone"
    if values == list(range(values[0], values[-1] + 1)):
        return f"randoms{values[0]}-{values[-1]}"
    return "randoms" + "_".join(str(v) for v in values)


def ez(redshift: np.ndarray | float) -> np.ndarray:
    z = np.asarray(redshift, dtype="f8")
    return np.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_L)


def comoving_distance_mpc_h(redshift: np.ndarray, *, ngrid: int = 8192) -> np.ndarray:
    """Return comoving radial distance in Mpc/h using the CUTSKY cosmology."""
    z = np.asarray(redshift, dtype="f8")
    if z.size == 0:
        return z.copy()
    zmax = max(float(np.max(z)), 1.0e-8)
    grid = np.linspace(0.0, zmax, int(ngrid), dtype="f8")
    inv_e = 1.0 / ez(grid)
    # Cumulative trapezoid without requiring scipy at import time.
    integ = np.empty_like(grid)
    integ[0] = 0.0
    integ[1:] = np.cumsum(0.5 * (inv_e[1:] + inv_e[:-1]) * np.diff(grid))
    return C_OVER_100 * np.interp(z, grid, integ)


def rdz_to_xyz(ra: np.ndarray, dec: np.ndarray, redshift: np.ndarray) -> np.ndarray:
    dist = comoving_distance_mpc_h(np.asarray(redshift, dtype="f8"))
    rar = np.deg2rad(np.asarray(ra, dtype="f8"))
    decr = np.deg2rad(np.asarray(dec, dtype="f8"))
    return np.column_stack(
        [
            dist * np.cos(decr) * np.cos(rar),
            dist * np.cos(decr) * np.sin(rar),
            dist * np.sin(decr),
        ]
    ).astype("f8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
