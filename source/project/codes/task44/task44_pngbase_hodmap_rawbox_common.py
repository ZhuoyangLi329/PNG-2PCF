#!/usr/bin/env python3
"""Shared immutable contract for the Task44 PNG-base HOD-MAP raw boxes.

The two inputs are full periodic, real-space LRG catalogs at z=0.5.  This
module deliberately contains no lightcone/FKP/random/window concepts: the
geometry is the parent 2 Gpc/h cube and the only infrared cutoff is its
fundamental mode.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task44_outputs" / "pngbase_hodmap_rawbox_z0p500"
PLOT_ROOT = PROJECT_ROOT / "plots" / "task44" / "pngbase_hodmap_rawbox_z0p500"
CATALOG_DIR = OUTPUT_ROOT / "catalogs"
PK_DIR = OUTPUT_ROOT / "pk"
XI_DIR = OUTPUT_ROOT / "xi"
THEORY_DIR = OUTPUT_ROOT / "theory"
FIT_DIR = OUTPUT_ROOT / "fits"
SUMMARY_DIR = OUTPUT_ROOT / "summary"
LOG_DIR = OUTPUT_ROOT / "logs"
FCFC_DIR = OUTPUT_ROOT / "fcfc"

BOX_SIZE = 2000.0
BOX_VOLUME = BOX_SIZE**3
BOX_CENTER_INPUT = 0.0
BOX_CENTER_POSITIVE = BOX_SIZE / 2.0
REDSHIFT = 0.5
K_FUND = 2.0 * np.pi / BOX_SIZE
DELTA_C = 1.686
SN0_SCALE = 1.0e4

# Fine measurement bins include the physical parent-mode bin and extend well
# beyond the fit range for estimator/shot-noise diagnostics.
PK_FINE_EDGES = np.arange(0.003, 0.301 + 1.0e-12, 0.002, dtype="f8")
PK_FINE_EDGE_PAIRS = np.column_stack([PK_FINE_EDGES[:-1], PK_FINE_EDGES[1:]])
# The periodic-box likelihood follows the measured periodic-box grid itself.
# ``kmax=0.1`` means that the selected bin centres satisfy k <= 0.100; the
# final selected shell is [0.099, 0.101).  This gives 49 contiguous bins.
_PK_FINE_CENTERS = np.mean(PK_FINE_EDGE_PAIRS, axis=1)
PK_PERIODIC_PRIMARY_FIT_EDGES = PK_FINE_EDGE_PAIRS[_PK_FINE_CENTERS <= 0.100 + 1.0e-13].copy()
PK_PERIODIC_WITHOUT_LOWEST_EDGES = PK_PERIODIC_PRIMARY_FIT_EDGES[1:].copy()

# Historical provenance only.  These bins copied the Task43 lightcone
# selection and are not a valid primary selection for the periodic boxes.
PK_LEGACY_LIGHTCONE_TAIL_EDGES = np.asarray(
    [
        [0.005, 0.007],
        [0.007, 0.009],
        [0.009, 0.011],
        [0.013, 0.015],
        [0.017, 0.019],
        [0.021, 0.023],
        [0.029, 0.031],
        [0.037, 0.039],
        [0.045, 0.047],
        [0.053, 0.055],
        [0.061, 0.063],
        [0.069, 0.071],
        [0.077, 0.079],
        [0.085, 0.087],
        [0.093, 0.095],
    ],
    dtype="f8",
)
PK_LEGACY_LIGHTCONE_MATCHED_EDGES = np.vstack(
    [np.asarray([[0.003, 0.005]], dtype="f8"), PK_LEGACY_LIGHTCONE_TAIL_EDGES]
)

# Authoritative alias retained for callers that only need the primary
# periodic-box contract.
PK_PRIMARY_FIT_EDGES = PK_PERIODIC_PRIMARY_FIT_EDGES
S_EDGES = np.arange(50.0, 350.0 + 10.0, 10.0, dtype="f8")
S_CENTERS = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])


def xi_edges(smin: float = 50.0) -> np.ndarray:
    """Return one of the two audited xi radial contracts."""
    value = float(smin)
    if np.isclose(value, 50.0, rtol=0.0, atol=1.0e-13):
        return np.asarray(S_EDGES, dtype="f8").copy()
    if np.isclose(value, 30.0, rtol=0.0, atol=1.0e-13):
        return np.arange(30.0, 350.0 + 10.0, 10.0, dtype="f8")
    raise ValueError(f"xi smin must be 30 or 50 Mpc/h, got {smin}")


@dataclass(frozen=True)
class CatalogSpec:
    tag: str
    realization: str
    fnl: float
    path: str
    expected_ngal: int


CATALOGS: dict[str, CatalogSpec] = {
    "c300": CatalogSpec(
        tag="c300",
        realization="Abacus_pngbase_c300_ph000",
        fnl=30.0,
        path=(
            "/pscratch/sd/s/siyizhao/desi-dr2-hod/loa-v2_HODv4/mocks_base-A/"
            "Abacus_pngbase_c300_ph000/LRG/0p500/LRG_hodMAP_realspace_clustering.dat.h5"
        ),
        expected_ngal=4_193_179,
    ),
    "c302": CatalogSpec(
        tag="c302",
        realization="Abacus_pngbase_c302_ph000",
        fnl=100.0,
        path=(
            "/pscratch/sd/s/siyizhao/desi-dr2-hod/loa-v2_HODv4/mocks_base-A/"
            "Abacus_pngbase_c302_ph000/LRG/0p500/LRG_hodMAP_realspace_clustering.dat.h5"
        ),
        expected_ngal=4_189_015,
    ),
}


def get_spec(tag: str) -> CatalogSpec:
    try:
        return CATALOGS[str(tag)]
    except KeyError as exc:
        raise ValueError(f"unknown catalog tag {tag!r}; choose {tuple(CATALOGS)}") from exc


def ensure_output_dirs() -> None:
    for path in (OUTPUT_ROOT, PLOT_ROOT, CATALOG_DIR, PK_DIR, XI_DIR, THEORY_DIR, FIT_DIR, SUMMARY_DIR, LOG_DIR, FCFC_DIR):
        path.mkdir(parents=True, exist_ok=True)


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return str(value)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def set_cpu_affinity(threads: int) -> list[int]:
    if not 1 <= int(threads) <= 8:
        raise ValueError(f"threads must be in [1,8], got {threads}")
    available = sorted(os.sched_getaffinity(0))
    selected = available[: int(threads)]
    if len(selected) != int(threads):
        raise RuntimeError(f"requested {threads} CPUs but only {len(available)} are available")
    os.sched_setaffinity(0, selected)
    return selected


def catalog_audit(tag: str, *, include_sha256: bool = False) -> dict[str, Any]:
    spec = get_spec(tag)
    path = Path(spec.path)
    stat = path.stat()
    with h5py.File(path, "r") as h5:
        fields = {
            name: {"shape": list(h5[name].shape), "dtype": str(h5[name].dtype)}
            for name in sorted(h5.keys())
        }
        attrs = {name: to_jsonable(value) for name, value in h5.attrs.items()}
        ndata = int(h5["X"].shape[0])
        contract = {
            "all_fields_same_length": all(int(h5[name].shape[0]) == ndata for name in h5.keys()),
            "expected_ngal": ndata == spec.expected_ngal,
            "boxsize": np.isclose(float(h5.attrs["BOXSIZE"]), BOX_SIZE, rtol=0.0, atol=1.0e-12),
            "redshift": np.isclose(float(h5.attrs["ZSNAP"]), REDSHIFT, rtol=0.0, atol=1.0e-12),
            "realization": str(h5.attrs["REALIZATION"]) == spec.realization,
            "tracer": str(h5.attrs["TRACER_TYPE"]) == "LRG",
        }
    payload: dict[str, Any] = {
        "task": "task44_pngbase_hodmap_rawbox_catalog_audit",
        "status": "pass" if all(contract.values()) else "fail",
        "tag": tag,
        "spec": asdict(spec),
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ndata": ndata,
        "nbar_h3_mpc3": float(ndata / BOX_VOLUME),
        "attrs": attrs,
        "fields": fields,
        "contract": contract,
        "geometry": {
            "type": "full periodic cube",
            "space": "real",
            "boxsize_mpc_h": BOX_SIZE,
            "volume_mpc_h3": BOX_VOLUME,
            "kfund_h_mpc": K_FUND,
        },
    }
    if include_sha256:
        payload["sha256"] = sha256_file(path)
    return payload


def load_positions(tag: str, *, origin: str = "positive") -> tuple[np.ndarray, dict[str, Any]]:
    spec = get_spec(tag)
    path = Path(spec.path)
    with h5py.File(path, "r") as h5:
        position = np.column_stack(
            [np.asarray(h5[name], dtype="f8") for name in ("X", "Y", "Z")]
        )
    if position.shape != (spec.expected_ngal, 3):
        raise ValueError(f"unexpected position shape for {tag}: {position.shape}")
    if not np.all(np.isfinite(position)):
        raise ValueError(f"non-finite positions in {path}")
    tolerance = 2.0e-8
    if np.any(position < -BOX_SIZE / 2.0 - tolerance) or np.any(position >= BOX_SIZE / 2.0 + tolerance):
        raise ValueError(f"centered coordinates leave [-L/2,L/2) for {tag}")
    if origin == "positive":
        position = np.mod(position + BOX_SIZE / 2.0, BOX_SIZE)
        boxcenter = BOX_CENTER_POSITIVE
    elif origin == "centered":
        # The mod makes the one possible +L/2 boundary convention explicit.
        position = np.mod(position + BOX_SIZE / 2.0, BOX_SIZE) - BOX_SIZE / 2.0
        boxcenter = BOX_CENTER_INPUT
    else:
        raise ValueError(f"unknown origin {origin!r}; choose positive or centered")
    metadata = {
        "tag": tag,
        "source": str(path),
        "origin": origin,
        "boxcenter_mpc_h": boxcenter,
        "boxsize_mpc_h": BOX_SIZE,
        "ndata": int(position.shape[0]),
        "nbar_h3_mpc3": float(position.shape[0] / BOX_VOLUME),
        "coordinate_min": np.min(position, axis=0).tolist(),
        "coordinate_max": np.max(position, axis=0).tolist(),
    }
    return np.asarray(position, dtype="f8"), metadata


def ascii_catalog_path(tag: str) -> Path:
    return CATALOG_DIR / f"task44_pngbase_{tag}_hodmap_lrg_z0p500_realspace_periodic_xyz.txt"


def ascii_metadata_path(tag: str) -> Path:
    return ascii_catalog_path(tag).with_suffix(".json")


def pk_path(tag: str, *, mesh: int, origin: str = "positive") -> Path:
    suffix = "" if origin == "positive" else f"_{origin}"
    return PK_DIR / f"task44_pngbase_{tag}_hodmap_lrg_z0p500_realspace_pk0_mesh{int(mesh)}{suffix}.npz"


def pk_metadata_path(tag: str, *, mesh: int, origin: str = "positive") -> Path:
    return pk_path(tag, mesh=mesh, origin=origin).with_suffix(".json")


def xi_path(tag: str, *, smin: float = 50.0) -> Path:
    value = float(smin)
    if np.isclose(value, 50.0, rtol=0.0, atol=1.0e-13):
        return XI_DIR / f"task44_pngbase_{tag}_hodmap_lrg_z0p500_realspace_xi0_s50_350_ds10.npz"
    if np.isclose(value, 30.0, rtol=0.0, atol=1.0e-13):
        return (
            OUTPUT_ROOT
            / "xi_engine_validation"
            / "fcfc"
            / f"task44_pngbase_{tag}_xi0_s30_350_ds10_fcfc.npz"
        )
    raise ValueError(f"xi smin must be 30 or 50 Mpc/h, got {smin}")


def xi_metadata_path(tag: str, *, smin: float = 50.0) -> Path:
    return xi_path(tag, smin=smin).with_suffix(".json")


def theory_path(kmax: float = 5.0, *, smin: float = 50.0) -> Path:
    ktag = f"{float(kmax):g}".replace(".", "p")
    stag = "" if np.isclose(float(smin), 50.0, rtol=0.0, atol=1.0e-13) else f"_smin{float(smin):g}"
    xi_edges(smin)
    return THEORY_DIR / f"task44_abacus_c000_z0p500_L2000_fulldiscrete_kmax{ktag}{stag}.npz"


def fit_prefix(
    tag: str,
    probe: str,
    *,
    kmin_edge: float | None = None,
    smin: float = 50.0,
    fixed_p: float | None = None,
) -> Path:
    if probe == "pk":
        if kmin_edge is None:
            raise ValueError("P(k) fit prefix requires kmin_edge")
        if np.isclose(float(kmin_edge), 0.003, rtol=0.0, atol=1.0e-13):
            binning_label = "periodic_contiguous49_kcentermax0p100"
        elif np.isclose(float(kmin_edge), 0.005, rtol=0.0, atol=1.0e-13):
            binning_label = "periodic_contiguous48_withoutlowest_kcentermax0p100"
        else:
            raise ValueError("P(k) kmin edge must be 0.003 or 0.005 h/Mpc")
        if fixed_p is None:
            parameter_label = "fixedfnl_freep"
        else:
            ptag = f"{float(fixed_p):g}".replace("-", "m").replace(".", "p")
            parameter_label = f"fixedp{ptag}_freefnl"
        label = f"pk0_{binning_label}_{parameter_label}"
    elif probe == "xi":
        xi_edges(smin)
        stag = f"{float(smin):g}".replace(".", "p")
        if fixed_p is None:
            parameter_label = "fixedfnl_freep"
        else:
            ptag = f"{float(fixed_p):g}".replace("-", "m").replace(".", "p")
            parameter_label = f"fixedp{ptag}_freefnl"
        label = f"xi0_smin{stag}_smax350_{parameter_label}"
    else:
        raise ValueError(f"unknown probe {probe!r}")
    return FIT_DIR / tag / label / f"task44_pngbase_{tag}_{label}"


def legacy_lightcone_matched_fit_prefix(
    tag: str,
    *,
    include_lowest: bool = True,
    fixed_p: float | None = None,
) -> Path:
    """Return the frozen path of the superseded sparse-bin P0 fits.

    This helper exists only so audits can retain provenance without ever
    confusing the old lightcone-matched selection with the periodic primary.
    """
    ktag = "0p003" if include_lowest else "0p005"
    if fixed_p is None:
        parameter_label = "fixedfnl_freep"
    else:
        ptag = f"{float(fixed_p):g}".replace("-", "m").replace(".", "p")
        parameter_label = f"fixedp{ptag}_freefnl"
    label = f"pk0_kmin{ktag}_kmax0p10_{parameter_label}"
    return FIT_DIR / tag / label / f"task44_pngbase_{tag}_{label}"
