#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for Task44 LRG P(k) tests."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from task44_config import (
    CATALOG_DIR,
    FIT_DIR,
    OUTPUT_ROOT,
    P0_DEFAULT,
    PLOT_DIR,
    PROJECT_ROOT,
    SAMPLE_CONFIGS,
    SUMMARY_DIR,
    catalog_output_paths,
    compact_random_tag,
    format_p0_tag,
    lrg_bin_config,
    lrg_bin_label,
    normalize_sample,
    normalize_realization,
    zeff_output_path,
)


DESI_CLUSTERING_ROOT = Path("/pscratch/sd/l/lzy/desi-clustering")
PK_ROOT = OUTPUT_ROOT / "pk_lrg2"
PK_MEASURE_DIR = PK_ROOT / "measurements"
PK_COV_DIR = PK_ROOT / "covariance"
PK_SUMMARY_DIR = PK_ROOT / "summary"
PK_FIT_DIR = PK_ROOT / "fits"
PK_PLOT_DIR = PLOT_DIR / "pk_lrg2"

SN0_SCALE = 1.0e4
K_MAX_FIT_DEFAULT = 0.10
P_FIXED_LRG2 = 1.1878


def parse_random_indices(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("random index list is empty")
    return values


def p_fixed_for_sample(sample: str) -> float:
    return float(lrg_bin_config(sample)["p_fixed"])


def pk_root(sample: str) -> Path:
    return OUTPUT_ROOT / f"pk_{normalize_sample(sample)}"


def pk_paths(sample: str) -> dict[str, Path]:
    root = pk_root(sample)
    return {
        "root": root,
        "measurements": root / "measurements",
        "covariance": root / "covariance",
        "summary": root / "summary",
        "fits": root / "fits",
        "plots": PLOT_DIR / f"pk_{normalize_sample(sample)}",
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_pk_dirs() -> None:
    for path in (PK_ROOT, PK_MEASURE_DIR, PK_COV_DIR, PK_SUMMARY_DIR, PK_FIT_DIR, PK_PLOT_DIR):
        ensure_dir(path)


def ensure_sample_pk_dirs(sample: str) -> dict[str, Path]:
    paths = pk_paths(sample)
    for path in paths.values():
        ensure_dir(path)
    return paths


def default_lrg_data_path(
    sample: str,
    p0: float = P0_DEFAULT,
    *,
    realization: str = "ph000_HODv4",
    random_indices: tuple[int, ...] = tuple(range(10)),
) -> Path:
    data_path, _ = catalog_output_paths(
        sample,
        realization=normalize_realization(realization),
        p0=float(p0),
        random_indices=tuple(random_indices),
    )
    return data_path


def default_lrg_random_path(
    sample: str,
    p0: float = P0_DEFAULT,
    *,
    realization: str = "ph000_HODv4",
    random_indices: tuple[int, ...] = tuple(range(10)),
) -> Path:
    _, random_path = catalog_output_paths(
        sample,
        realization=normalize_realization(realization),
        p0=float(p0),
        random_indices=tuple(random_indices),
    )
    return random_path


def default_lrg_zeff_path(
    sample: str,
    p0: float = P0_DEFAULT,
    *,
    realization: str = "ph000_HODv4",
    random_indices: tuple[int, ...] = tuple(range(10)),
) -> Path:
    path = zeff_output_path(sample, realization=normalize_realization(realization), p0=float(p0))
    if tuple(random_indices) != tuple(range(10)):
        ptag = format_p0_tag(float(p0))
        label = lrg_bin_label(sample)
        # Keep the historical Task44 naming for non-contiguous random selections.
        path = SUMMARY_DIR / (
            f"task44_{normalize_realization(realization)}_{label}_zeff_zobs_"
            f"{ptag}_{compact_random_tag(tuple(random_indices))}.npz"
        )
    return path


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


def task44_pk_tag(sample: str, realization: str, p0: float, tag: str | None = None) -> str:
    if tag:
        return str(tag)
    return f"{str(realization)}_{lrg_bin_label(sample)}_{format_p0_tag(float(p0))}".replace("-", "m")


def make_k_edges(kmin: float = 0.001, kmax: float = 0.3001, dk: float = 0.002) -> np.ndarray:
    edges = np.arange(float(kmin), float(kmax) + 0.5 * float(dk), float(dk), dtype="f8")
    if edges[-1] < float(kmax):
        edges = np.append(edges, float(kmax))
    edges[0] = float(kmin)
    edges[-1] = float(kmax)
    return edges


def column_edges(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype="f8")
    return np.column_stack([edges[:-1], edges[1:]])


def load_npz_catalog_for_jaxpower(
    path: Path,
    *,
    max_rows: int | None = None,
    seed: int = 0,
    rescale_subsample: bool = False,
    add_targetid: bool = False,
    zmin: float | None = None,
    zmax: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        # The frozen random75x generator already writes the same physical
        # columns as the historical NPZ catalog.  Read it directly so P(k)
        # block measurements do not need 150 redundant multi-GB NPZ copies.
        import h5py

        with h5py.File(path, "r") as data:
            required = ("Z", "NX", "WEIGHT_TOTAL", "X", "Y", "Zcart")
            missing = [name for name in required if name not in data]
            if missing:
                raise KeyError(f"{path} misses required jaxpower columns: {missing}")
            n_source = int(data["Z"].shape[0])
            if zmin is None and zmax is None:
                selected: np.ndarray | slice = slice(None)
                n_total = n_source
            else:
                z_source = np.asarray(data["Z"], dtype="f8")
                selected = np.flatnonzero(
                    (z_source >= (-np.inf if zmin is None else float(zmin)))
                    & (z_source < (np.inf if zmax is None else float(zmax)))
                )
                n_total = int(selected.size)
            if max_rows is not None and 0 < int(max_rows) < n_total:
                rng = np.random.default_rng(int(seed))
                local = np.sort(rng.choice(n_total, size=int(max_rows), replace=False))
                choice = local if isinstance(selected, slice) else selected[local]
            else:
                choice = selected
            z = np.asarray(data["Z"][choice], dtype="f8")
            nx = np.asarray(data["NX"][choice], dtype="f8")
            weight = np.asarray(data["WEIGHT_TOTAL"][choice], dtype="f8")
            pos = np.column_stack(
                [
                    np.asarray(data["X"][choice], dtype="f8"),
                    np.asarray(data["Y"][choice], dtype="f8"),
                    np.asarray(data["Zcart"][choice], dtype="f8"),
                ]
            )
            source_attrs = {str(key): to_jsonable(value) for key, value in data.attrs.items()}
        source_format = "hdf5_random75x"
    else:
        with np.load(path, allow_pickle=False) as data:
            n_source = int(np.asarray(data["Z"]).size)
            z_source = np.asarray(data["Z"], dtype="f8")
            selected = np.flatnonzero(
                (z_source >= (-np.inf if zmin is None else float(zmin)))
                & (z_source < (np.inf if zmax is None else float(zmax)))
            )
            n_total = int(selected.size)
            if max_rows is not None and 0 < int(max_rows) < n_total:
                rng = np.random.default_rng(int(seed))
                local = np.sort(rng.choice(n_total, size=int(max_rows), replace=False))
                choice = selected[local]
            else:
                choice = selected
            z = np.asarray(data["Z"][choice], dtype="f8")
            nx = np.asarray(data["NX"][choice], dtype="f8")
            weight = np.asarray(data["WEIGHT_TOTAL"][choice], dtype="f8")
            pos = np.column_stack(
                [
                    np.asarray(data["X"][choice], dtype="f8"),
                    np.asarray(data["Y"][choice], dtype="f8"),
                    np.asarray(data["Zcart"][choice], dtype="f8"),
                ]
            )
        source_attrs = {}
        source_format = "npz"
    targetid = (
        np.arange(n_source, dtype="i8")
        if isinstance(choice, slice)
        else np.asarray(choice, dtype="i8")
    )
    n_used = int(z.size)
    scale = float(n_total) / float(n_used) if (rescale_subsample and n_used > 0) else 1.0
    weight = weight * scale
    catalog: dict[str, np.ndarray] = {
        "POSITION": np.asarray(pos, dtype="f8"),
        "INDWEIGHT": np.asarray(weight, dtype="f8"),
        "Z": np.asarray(z, dtype="f8"),
        "NX": np.asarray(nx, dtype="f8"),
        "WEIGHT_FKP": np.asarray(weight, dtype="f8"),
    }
    if add_targetid:
        catalog["TARGETID"] = targetid
    meta = {
        "path": str(path),
        "format": source_format,
        "source_attrs": source_attrs,
        "source_file_size_bytes": int(path.stat().st_size),
        "source_file_mtime_ns": int(path.stat().st_mtime_ns),
        "n_source": n_source,
        "n_total": n_total,
        "n_used": n_used,
        "subsample_seed": int(seed),
        "subsample_applied": bool(n_used != n_total),
        "subsample_weight_rescale": bool(rescale_subsample),
        "subsample_weight_scale": float(scale),
        "zmin_selection": None if zmin is None else float(zmin),
        "zmax_selection": None if zmax is None else float(zmax),
        "z_selection_policy": "z >= zmin and z < zmax",
        "weight_sum": float(np.sum(weight)),
        "weight_min": float(np.min(weight)),
        "weight_max": float(np.max(weight)),
        "nx_mean_used": float(np.mean(nx)),
        "nx_min_used": float(np.min(nx)),
        "nx_max_used": float(np.max(nx)),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "position_min": np.min(pos, axis=0).tolist(),
        "position_max": np.max(pos, axis=0).tolist(),
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


def frozen_mesh_attrs_from_json(
    path: Path,
    catalogs: list[dict[str, np.ndarray]],
    *,
    meshsize: int,
) -> dict[str, Any]:
    """Load one audited mesh geometry and verify all current particles fit.

    Independent random blocks have slightly different coordinate extrema.  A
    per-block inferred mesh would therefore change the FFT geometry together
    with the Monte-Carlo random realization.  Random75x measurements instead
    reuse one frozen mesh and fail if any current particle lies outside it.
    """

    path = Path(path).resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mesh = payload.get("mesh", payload)
    boxsize = np.asarray(mesh["boxsize"], dtype="f8")
    boxcenter = np.asarray(mesh["boxcenter"], dtype="f8")
    stored_meshsize = np.asarray(mesh["meshsize"], dtype="i8")
    if boxsize.shape != (3,) or boxcenter.shape != (3,) or stored_meshsize.shape != (3,):
        raise ValueError(f"invalid frozen mesh vectors in {path}")
    requested_meshsize = np.full(3, int(meshsize), dtype="i8")
    current_min = np.min(
        np.vstack([np.min(np.asarray(cat["POSITION"]), axis=0) for cat in catalogs]), axis=0
    )
    current_max = np.max(
        np.vstack([np.max(np.asarray(cat["POSITION"]), axis=0) for cat in catalogs]), axis=0
    )
    lower = boxcenter - 0.5 * boxsize
    upper = boxcenter + 0.5 * boxsize
    if np.any(current_min < lower) or np.any(current_max > upper):
        raise ValueError(
            f"particles exceed frozen mesh {path}: current={current_min.tolist()}..{current_max.tolist()} "
            f"box={lower.tolist()}..{upper.tolist()}"
        )
    result = {str(key): to_jsonable(value) for key, value in mesh.items()}
    result.update(
        {
            "boxsize": boxsize.tolist(),
            "boxcenter": boxcenter.tolist(),
            "meshsize": requested_meshsize.tolist(),
            "template_meshsize": stored_meshsize.tolist(),
            "frozen_template_json": str(path),
            "current_catalog_min": current_min.tolist(),
            "current_catalog_max": current_max.tolist(),
            "lower_margin": (current_min - lower).tolist(),
            "upper_margin": (upper - current_max).tolist(),
        }
    )
    return result


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
    sym = 0.5 * (np.asarray(cov, dtype="f8") + np.asarray(cov, dtype="f8").T)
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


def lrg_volume_kmin_from_catalog_meta(data_meta: dict[str, Any]) -> tuple[float, dict[str, float]]:
    n_total = float(data_meta["n_total"])
    nx_mean = float(data_meta["nx_mean_used"])
    if not (n_total > 0.0 and nx_mean > 0.0):
        raise ValueError(f"cannot infer volume kmin from n_total={n_total}, nx_mean={nx_mean}")
    volume = n_total / nx_mean
    leff = volume ** (1.0 / 3.0)
    kmin = 2.0 * math.pi / leff
    return float(kmin), {"volume_n_over_nx": float(volume), "leff_volume": float(leff), "kmin_eff": float(kmin)}


def default_lrg2_data_path(p0: float = P0_DEFAULT) -> Path:
    return CATALOG_DIR / f"task44_ph000_HODv4_LRG2_data_zobs_{format_p0_tag(float(p0))}.npz"


def default_lrg2_random_path(p0: float = P0_DEFAULT) -> Path:
    return CATALOG_DIR / f"task44_ph000_HODv4_LRG2_randoms0-9_zobs_{format_p0_tag(float(p0))}.npz"


def default_lrg2_zeff_path(p0: float = P0_DEFAULT) -> Path:
    return SUMMARY_DIR / f"task44_ph000_HODv4_LRG2_zeff_zobs_{format_p0_tag(float(p0))}.npz"
