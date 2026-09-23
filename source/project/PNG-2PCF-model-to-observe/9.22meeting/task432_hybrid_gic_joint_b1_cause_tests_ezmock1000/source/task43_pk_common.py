#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for Task43 lightcone-halo P(k) closure tests."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from task43_config import OUTPUT_ROOT, PLOT_DIR, PROJECT_ROOT, SUMMARY_DIR
from task43_fkp_zeff import load_fkp_summary, p0_tag, total_weight_from_summary


DESI_CLUSTERING_ROOT = Path("/pscratch/sd/l/lzy/desi-clustering")
PK_ROOT = OUTPUT_ROOT / "pk_lightcone"
PK_MEASURE_DIR = PK_ROOT / "measurements"
PK_WINDOW_DIR = PK_ROOT / "windows"
PK_COV_DIR = PK_ROOT / "covariance"
PK_SUMMARY_DIR = PK_ROOT / "summary"
PK_FIT_DIR = PK_ROOT / "fits"
PK_PLOT_DIR = PLOT_DIR / "pk_lightcone"

DEFAULT_FKP_SUMMARY = SUMMARY_DIR / "task43_fkp_zeff_mmin1p4e13_x25.npz"
DEFAULT_MANIFEST_MMIN1P4 = OUTPUT_ROOT / "manifests" / "task43_mmin1p4e13_x25.jsonl"

P_FIXED = 1.0
SIGMAS_FIXED = 0.0
SN0_SCALE = 1.0e4
DELTA_C = 1.686
K_MAX_FIT = 0.08


def lightcone_fit_kmin_from_fkp_summary(fkp_summary: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    """Return observed-bin kmin from the lightcone effective volume.

    This is deliberately not the Abacus base-box fundamental mode.  Task43 is a
    positive-octant shell lightcone, whose survey volume is smaller than the
    2 Gpc/h periodic box.
    """
    if "volume_shell" not in fkp_summary:
        raise KeyError("FKP summary lacks volume_shell; cannot infer Task43 lightcone fit kmin")
    volume = float(np.sum(np.asarray(fkp_summary["volume_shell"], dtype="f8")))
    if not volume > 0.0:
        raise ValueError(f"non-positive lightcone volume: {volume}")
    leff = volume ** (1.0 / 3.0)
    kmin = 2.0 * math.pi / leff
    return float(kmin), {"volume_eff": volume, "leff_volume": float(leff), "kmin_eff": float(kmin)}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_pk_dirs() -> None:
    for path in (PK_ROOT, PK_MEASURE_DIR, PK_WINDOW_DIR, PK_COV_DIR, PK_SUMMARY_DIR, PK_FIT_DIR, PK_PLOT_DIR):
        ensure_dir(path)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    return str(obj)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def phase_index(phase: str) -> int:
    phase = str(phase)
    if not phase.startswith("ph"):
        raise ValueError(f"bad phase label: {phase}")
    return int(phase[2:])


def select_rows(rows: list[dict[str, Any]], *, index: int | None = None, phase: str | None = None) -> list[dict[str, Any]]:
    if index is not None and phase is not None:
        raise ValueError("provide only one of index or phase")
    if index is not None:
        return [rows[int(index)]]
    if phase is not None:
        selected = [row for row in rows if row["phase"] == phase]
        if not selected:
            raise ValueError(f"phase not found in manifest: {phase}")
        return selected
    return rows


def p0_output_tag(p0: float, tag: str | None = None) -> str:
    if tag:
        return str(tag)
    return f"mmin1p4e13_x25_{p0_tag(float(p0))}"


def make_k_edges(kmin: float = 1.0e-3, kmax: float = 0.3001, dk: float = 0.002) -> np.ndarray:
    edges = np.arange(float(kmin), float(kmax) + 0.5 * float(dk), float(dk), dtype="f8")
    if edges[-1] < float(kmax):
        edges = np.append(edges, float(kmax))
    edges[0] = float(kmin)
    edges[-1] = float(kmax)
    return edges


def column_edges(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype="f8")
    return np.column_stack([edges[:-1], edges[1:]])


def load_phase_catalog(
    path: Path,
    *,
    fkp_summary: dict[str, np.ndarray],
    p0: float,
    max_rows: int | None = None,
    seed: int = 0,
    rescale_subsample: bool = False,
    add_targetid: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load one Task43 data/random catalog into the jaxpower dictionary form."""
    with np.load(path, allow_pickle=False) as data:
        z_all = np.asarray(data["Z"], dtype="f8")
        n_total = int(z_all.size)
        if max_rows is not None and 0 < int(max_rows) < n_total:
            rng = np.random.default_rng(int(seed))
            choice = np.sort(rng.choice(n_total, size=int(max_rows), replace=False))
        else:
            choice = slice(None)
        z = np.asarray(data["Z"][choice], dtype="f8")
        base_weight = np.asarray(data["WEIGHT"][choice], dtype="f8") if "WEIGHT" in data.files else np.ones(z.size, dtype="f8")
        position = np.column_stack(
            [
                np.asarray(data["X"][choice], dtype="f8"),
                np.asarray(data["Y"][choice], dtype="f8"),
                np.asarray(data["Zcart"][choice], dtype="f8"),
            ]
        )
        if isinstance(choice, slice):
            targetid = np.arange(n_total, dtype="i8")
        else:
            targetid = np.asarray(choice, dtype="i8")
    weight = total_weight_from_summary(z, base_weight, fkp_summary, p0=float(p0))
    n_used = int(z.size)
    scale = float(n_total) / float(n_used) if (rescale_subsample and n_used > 0) else 1.0
    weight = weight * scale
    catalog = {
        "POSITION": np.asarray(position, dtype="f8"),
        "INDWEIGHT": np.asarray(weight, dtype="f8"),
        "Z": np.asarray(z, dtype="f8"),
    }
    if add_targetid:
        catalog["TARGETID"] = np.asarray(targetid, dtype="i8")
    meta = {
        "path": str(path),
        "n_total": n_total,
        "n_used": n_used,
        "subsample_seed": int(seed),
        "subsample_applied": bool(n_used != n_total),
        "subsample_weight_rescale": bool(rescale_subsample),
        "subsample_weight_scale": float(scale),
        "weight_sum": float(np.sum(weight)),
        "weight_min": float(np.min(weight)),
        "weight_max": float(np.max(weight)),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "position_min": np.min(position, axis=0).tolist(),
        "position_max": np.max(position, axis=0).tolist(),
    }
    return catalog, meta


def infer_mesh_attrs_from_catalogs(catalogs: list[dict[str, np.ndarray]], *, meshsize: int, pad: float) -> dict[str, Any]:
    mins = np.min(np.vstack([np.min(cat["POSITION"], axis=0) for cat in catalogs]), axis=0)
    maxs = np.max(np.vstack([np.max(cat["POSITION"], axis=0) for cat in catalogs]), axis=0)
    center = 0.5 * (mins + maxs)
    span = maxs - mins
    boxsize = float(np.max(span) + 2.0 * float(pad))
    return {
        "boxsize": [boxsize] * 3,
        "boxcenter": [float(v) for v in center],
        "meshsize": [int(meshsize)] * 3,
        "catalog_min": [float(v) for v in mins],
        "catalog_max": [float(v) for v in maxs],
        "catalog_span": [float(v) for v in span],
        "pad": float(pad),
    }


def mesh_attrs_for_jaxpower(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "boxsize": np.asarray(meta["boxsize"], dtype="f8"),
        "boxcenter": np.asarray(meta["boxcenter"], dtype="f8"),
        "meshsize": np.asarray(meta["meshsize"], dtype="i8"),
    }


def extract_spectrum_arrays(spectrum: Any, ell: int = 0) -> dict[str, np.ndarray]:
    pole = spectrum.get(int(ell))
    return {
        "k_obs": np.asarray(pole.coords("k"), dtype="f8"),
        "k_edges": np.asarray(pole.edges("k"), dtype="f8"),
        "pk0": np.asarray(pole.value(), dtype="f8"),
        "norm": np.asarray(pole.values("norm"), dtype="f8"),
        "num_shotnoise": np.asarray(pole.values("num_shotnoise"), dtype="f8"),
        "shotnoise": np.asarray(pole.values("shotnoise"), dtype="f8"),
    }


def extract_window_arrays(window: Any) -> dict[str, np.ndarray]:
    raw = window["raw"] if isinstance(window, dict) else window
    theory_k, theory_edges, theory_ell = [], [], []
    starts, stops = [], []
    cursor = 0
    for label, pole in raw.theory.items():
        kvals = np.asarray(pole.coords("k"), dtype="f8")
        edges = np.asarray(pole.edges("k"), dtype="f8")
        ell = int(label["ells"])
        theory_k.append(kvals)
        theory_edges.append(edges)
        theory_ell.append(np.full(kvals.size, ell, dtype="i8"))
        starts.append(cursor)
        cursor += kvals.size
        stops.append(cursor)
    obs = raw.observable.get(0)
    return {
        "window_matrix": np.asarray(raw.value(), dtype="f8"),
        "window_observable_k": np.asarray(obs.coords("k"), dtype="f8"),
        "window_observable_edges": np.asarray(obs.edges("k"), dtype="f8"),
        "theory_k": np.concatenate(theory_k).astype("f8"),
        "theory_edges": np.vstack(theory_edges).astype("f8"),
        "theory_ell": np.concatenate(theory_ell).astype("i8"),
        "theory_slice_start": np.asarray(starts, dtype="i8"),
        "theory_slice_stop": np.asarray(stops, dtype="i8"),
    }


def covariance_diagnostics(cov: np.ndarray) -> dict[str, Any]:
    cov = np.asarray(cov, dtype="f8")
    sym = 0.5 * (cov + cov.T)
    diag = np.diag(sym)
    eig = np.linalg.eigvalsh(sym)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = sym / np.sqrt(np.outer(diag, diag))
    return {
        "shape": [int(v) for v in sym.shape],
        "diag_min": float(np.min(diag)),
        "diag_max": float(np.max(diag)),
        "sigma_min": float(np.sqrt(np.min(diag))),
        "sigma_max": float(np.sqrt(np.max(diag))),
        "min_eigenvalue": float(np.min(eig)),
        "max_eigenvalue": float(np.max(eig)),
        "condition_number": float(np.linalg.cond(sym)),
        "corr_offdiag_max_abs": float(np.nanmax(np.abs(corr - np.eye(corr.shape[0])))),
    }
