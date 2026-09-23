#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FKP weighting and effective-redshift helpers for Task43."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from task43_config import DEFAULT_MANIFEST, PROJECT_ROOT, SUMMARY_DIR, ZMAX, ZMIN, read_jsonl


REF_ZEFF_SCRIPT = PROJECT_ROOT / "refcode" / "eff_z" / "test_zeff.py"
DEFAULT_P0_VALUES = (10000.0, 0.0, 5000.0, 20000.0)
DEFAULT_ZSTEP = 0.01
DEFAULT_SKY_FRACTION = 1.0 / 8.0


def format_float_tag(value: float) -> str:
    """Return a compact filename-safe float tag."""
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}".replace(".", "p").replace("-", "m")


def p0_tag(p0: float | None) -> str:
    """Return the standard FKP P0 tag used by Task43 output files."""
    if p0 is None:
        return "unweighted"
    return f"fkpP0{format_float_tag(float(p0))}"


def path_with_weight_tag(path: Path, *, p0: float | None, output_tag: str | None = None) -> Path:
    """Append a weight tag to an output path while preserving its suffix."""
    tag = output_tag.strip() if output_tag else p0_tag(p0)
    if not tag or tag == "unweighted":
        return path
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def load_compute_zeff() -> Any:
    """Import the reference `compute_zeff` implementation without copying it."""
    spec = importlib.util.spec_from_file_location("task43_ref_compute_zeff", REF_ZEFF_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {REF_ZEFF_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_zeff


def get_fiducial_cosmology(name: str = "DESI") -> Any:
    """Return the cosmology used for z-volume and zeff calculations."""
    from cosmoprimo.fiducial import AbacusSummit, DESI

    key = str(name).strip().lower()
    if key == "desi":
        return DESI()
    if key in {"abacus", "abacussummit", "abacussummit_base_c000"}:
        return AbacusSummit(0)
    raise ValueError(f"unknown cosmology {name!r}; use DESI or AbacusSummit")


def z_edges_from_range(zmin: float, zmax: float, zstep: float) -> np.ndarray:
    """Build closed z-bin edges matching the reference zeff zstep convention."""
    if zmax <= zmin:
        raise ValueError("zmax must exceed zmin")
    if zstep <= 0.0:
        raise ValueError("zstep must be positive")
    return np.arange(float(zmin), float(zmax) + 0.5 * float(zstep), float(zstep), dtype="f8")


def shell_volumes(z_edges: np.ndarray, *, cosmo: Any, sky_fraction: float) -> np.ndarray:
    """Comoving shell volumes for the Task43 octant-like footprint."""
    distance = cosmo.get_background().comoving_radial_distance(np.asarray(z_edges, dtype="f8"))
    return float(sky_fraction) * (4.0 * np.pi / 3.0) * (distance[1:] ** 3 - distance[:-1] ** 3)


def load_z_and_base_weight(path: Path, *, max_rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load redshift and the catalog's base WEIGHT column."""
    data = np.load(path, allow_pickle=False)
    nrows = len(data["Z"]) if max_rows is None else min(int(max_rows), len(data["Z"]))
    z = np.asarray(data["Z"][:nrows], dtype="f8")
    if "WEIGHT" in data.files:
        weight = np.asarray(data["WEIGHT"][:nrows], dtype="f8")
    else:
        weight = np.ones(nrows, dtype="f8")
    return z, weight


def build_random_nbar(
    rows: list[dict[str, Any]],
    *,
    z_edges: np.ndarray,
    cosmo: Any,
    sky_fraction: float = DEFAULT_SKY_FRACTION,
    max_random_per_catalog: int | None = None,
) -> dict[str, np.ndarray]:
    """Estimate data nbar(z) from the all-phase random catalogs."""
    hist_random = np.zeros(len(z_edges) - 1, dtype="f8")
    multiplier_sum = 0.0
    nrandom_total = 0
    for row in rows:
        random_path = Path(row["random_catalog_path"])
        z, weight = load_z_and_base_weight(random_path, max_rows=max_random_per_catalog)
        hist_random += np.histogram(z, bins=z_edges, weights=weight)[0]
        multiplier_sum += float(row.get("random_multiplier", 1.0))
        nrandom_total += int(z.size)

    volume = shell_volumes(z_edges, cosmo=cosmo, sky_fraction=sky_fraction)
    if not np.all(volume > 0.0):
        raise ValueError("non-positive shell volume in nbar estimate")
    if multiplier_sum <= 0.0:
        raise ValueError("non-positive summed random multiplier")
    nbar = hist_random / (multiplier_sum * volume)
    return {
        "hist_random_weighted": hist_random,
        "volume_shell": volume,
        "nbar": nbar,
        "nrandom_total_used": np.array(nrandom_total, dtype="i8"),
        "random_multiplier_sum": np.array(multiplier_sum, dtype="f8"),
    }


def load_fkp_summary(path: Path) -> dict[str, np.ndarray]:
    """Load an FKP/zeff NPZ summary as a plain dict of arrays."""
    data = np.load(path, allow_pickle=False)
    required = ("z_edges", "nbar")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"{path} missing FKP summary keys: {missing}")
    return {key: np.asarray(data[key]) for key in data.files}


def fkp_bin_weights(nbar: np.ndarray, p0: float) -> np.ndarray:
    """Compute FKP weights per z-bin."""
    return 1.0 / (1.0 + np.asarray(nbar, dtype="f8") * float(p0))


def total_weight_from_summary(
    z: np.ndarray,
    base_weight: np.ndarray,
    fkp_summary: dict[str, np.ndarray] | None,
    *,
    p0: float | None,
) -> np.ndarray:
    """Return `WEIGHT * WEIGHT_FKP` for one catalog."""
    base = np.asarray(base_weight, dtype="f8")
    if fkp_summary is None or p0 is None:
        return base
    z_edges = np.asarray(fkp_summary["z_edges"], dtype="f8")
    nbar = np.asarray(fkp_summary["nbar"], dtype="f8")
    per_bin = fkp_bin_weights(nbar, float(p0))
    idx = np.searchsorted(z_edges, np.asarray(z, dtype="f8"), side="right") - 1
    idx = np.clip(idx, 0, per_bin.size - 1)
    return base * per_bin[idx]


def compute_zeff_diagnostics(
    data_z: np.ndarray,
    data_weight: np.ndarray,
    random_z: np.ndarray,
    random_weight: np.ndarray,
    *,
    cosmo: Any,
    zrange: tuple[float, float],
) -> dict[str, float]:
    """Compute Task43 zeff diagnostics using the reference zeff routine."""
    compute_zeff = load_compute_zeff()
    data_z = np.asarray(data_z, dtype="f8")
    random_z = np.asarray(random_z, dtype="f8")
    data_weight = np.asarray(data_weight, dtype="f8")
    random_weight = np.asarray(random_weight, dtype="f8")
    return {
        "zeff_data_auto": float(compute_zeff(data_z, data_weight, cosmo=cosmo, zrange=zrange)),
        "zeff_random_auto": float(compute_zeff(random_z, random_weight, cosmo=cosmo, zrange=zrange)),
        "zeff_data_random_cross": float(
            compute_zeff(data_z, data_weight, random_z, random_weight, cosmo=cosmo, zrange=zrange)
        ),
        "data_mean_z": float(np.mean(data_z)),
        "random_mean_z": float(np.mean(random_z)),
        "data_weighted_mean_z": float(np.average(data_z, weights=data_weight)),
        "random_weighted_mean_z": float(np.average(random_z, weights=random_weight)),
    }


def concatenate_catalogs(
    rows: list[dict[str, Any]],
    *,
    key: str,
    fkp_summary: dict[str, np.ndarray],
    p0: float,
    max_rows_per_catalog: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate z and total weights across all rows for one manifest key."""
    zs: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for row in rows:
        z, base_weight = load_z_and_base_weight(Path(row[key]), max_rows=max_rows_per_catalog)
        total_weight = total_weight_from_summary(z, base_weight, fkp_summary, p0=float(p0))
        zs.append(z)
        weights.append(total_weight)
    return np.concatenate(zs), np.concatenate(weights)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--p0", type=float, nargs="+", default=list(DEFAULT_P0_VALUES))
    parser.add_argument("--zmin", type=float, default=ZMIN)
    parser.add_argument("--zmax", type=float, default=ZMAX)
    parser.add_argument("--zstep", type=float, default=DEFAULT_ZSTEP)
    parser.add_argument("--cosmology", choices=["DESI", "AbacusSummit"], default="DESI")
    parser.add_argument("--sky-fraction", type=float, default=DEFAULT_SKY_FRACTION)
    parser.add_argument("--max-data-per-catalog", type=int, default=None)
    parser.add_argument("--max-random-per-catalog", type=int, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    cosmo = get_fiducial_cosmology(args.cosmology)
    z_edges = z_edges_from_range(float(args.zmin), float(args.zmax), float(args.zstep))
    nbar_payload = build_random_nbar(
        rows,
        z_edges=z_edges,
        cosmo=cosmo,
        sky_fraction=float(args.sky_fraction),
        max_random_per_catalog=args.max_random_per_catalog,
    )
    fkp_summary = {"z_edges": z_edges, "nbar": nbar_payload["nbar"]}

    p0_values = np.asarray(args.p0, dtype="f8")
    fkp_weights = np.vstack([fkp_bin_weights(nbar_payload["nbar"], p0) for p0 in p0_values])
    zeff_rows: list[dict[str, float]] = []
    for p0 in p0_values:
        data_z, data_w = concatenate_catalogs(
            rows,
            key="halo_catalog_path",
            fkp_summary=fkp_summary,
            p0=float(p0),
            max_rows_per_catalog=args.max_data_per_catalog,
        )
        random_z, random_w = concatenate_catalogs(
            rows,
            key="random_catalog_path",
            fkp_summary=fkp_summary,
            p0=float(p0),
            max_rows_per_catalog=args.max_random_per_catalog,
        )
        diag = compute_zeff_diagnostics(
            data_z,
            data_w,
            random_z,
            random_w,
            cosmo=cosmo,
            zrange=(float(args.zmin), float(args.zmax)),
        )
        diag.update(
            {
                "p0": float(p0),
                "data_weight_min": float(np.min(data_w)),
                "data_weight_max": float(np.max(data_w)),
                "random_weight_min": float(np.min(random_w)),
                "random_weight_max": float(np.max(random_w)),
                "ndata_used": float(data_z.size),
                "nrandom_used": float(random_z.size),
            }
        )
        zeff_rows.append(diag)

    output = args.output
    if output is None:
        output = SUMMARY_DIR / f"task43_fkp_zeff_{args.manifest.stem}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "status": "done",
        "manifest": str(args.manifest),
        "method": "all-25 random nbar(z), FKP weights, refcode/eff_z/test_zeff.py compute_zeff",
        "ref_zeff_script": str(REF_ZEFF_SCRIPT),
        "cosmology": str(args.cosmology),
        "zmin": float(args.zmin),
        "zmax": float(args.zmax),
        "zstep": float(args.zstep),
        "sky_fraction": float(args.sky_fraction),
        "nrows": int(len(rows)),
        "max_data_per_catalog": args.max_data_per_catalog,
        "max_random_per_catalog": args.max_random_per_catalog,
    }
    np.savez_compressed(
        output,
        z_edges=z_edges,
        z_centers=0.5 * (z_edges[:-1] + z_edges[1:]),
        nbar=nbar_payload["nbar"],
        volume_shell=nbar_payload["volume_shell"],
        hist_random_weighted=nbar_payload["hist_random_weighted"],
        p0_values=p0_values,
        fkp_weights=fkp_weights,
        zeff_random_auto=np.asarray([row["zeff_random_auto"] for row in zeff_rows], dtype="f8"),
        zeff_data_auto=np.asarray([row["zeff_data_auto"] for row in zeff_rows], dtype="f8"),
        zeff_data_random_cross=np.asarray([row["zeff_data_random_cross"] for row in zeff_rows], dtype="f8"),
        data_mean_z=np.asarray([row["data_mean_z"] for row in zeff_rows], dtype="f8"),
        random_mean_z=np.asarray([row["random_mean_z"] for row in zeff_rows], dtype="f8"),
        data_weighted_mean_z=np.asarray([row["data_weighted_mean_z"] for row in zeff_rows], dtype="f8"),
        random_weighted_mean_z=np.asarray([row["random_weighted_mean_z"] for row in zeff_rows], dtype="f8"),
        data_weight_min=np.asarray([row["data_weight_min"] for row in zeff_rows], dtype="f8"),
        data_weight_max=np.asarray([row["data_weight_max"] for row in zeff_rows], dtype="f8"),
        random_weight_min=np.asarray([row["random_weight_min"] for row in zeff_rows], dtype="f8"),
        random_weight_max=np.asarray([row["random_weight_max"] for row in zeff_rows], dtype="f8"),
        meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    json_path = output.with_suffix(".json")
    summary = dict(meta)
    summary.update(
        {
            "output_npz": str(output),
            "nbar_min": float(np.min(nbar_payload["nbar"])),
            "nbar_max": float(np.max(nbar_payload["nbar"])),
            "nbar_mean": float(np.mean(nbar_payload["nbar"])),
            "p0_results": zeff_rows,
        }
    )
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
