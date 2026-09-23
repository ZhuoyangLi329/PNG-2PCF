#!/usr/bin/env python3
"""Build phase-matched x25 lightcone randoms and an FKP nbar summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from task43_rsd_common import P0_FKP, PHASES, ZMAX, ZMIN, atomic_savez, atomic_write_json, sha256_file


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_row(rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    matches = [row for row in rows if row["phase"] == phase]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {phase}, found {len(matches)}")
    return matches[0]


def deterministic_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) >> 1


def fkp_path(row: dict[str, Any]) -> Path:
    if row.get("lightcone_fkp_path"):
        return Path(str(row["lightcone_fkp_path"]))
    root = Path(row["lightcone_random_path"]).parent.parent / "fkp"
    return root / f"task43_rsd_fkp_{row['sim_name']}_zobs0p6_0p8_dz0p01.npz"


def repair_float32_redshift_boundaries(
    redshift: np.ndarray,
    *,
    zmin: float = ZMIN,
    zmax: float = ZMAX,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover the strict pre-serialization cut from a float32 catalog view.

    The light-cone builder applies the strict cut in float64 and records that
    range in its JSON, but stores ``Z`` as float32.  A valid value immediately
    below (for example) 0.8 can therefore round to float32(0.8), whose float64
    value is slightly above 0.8.  Clip only those one-ULP serialization cases
    back to the nearest strict float64 boundary.
    """

    stored = np.asarray(redshift)
    values = stored.astype("f8")
    lower_storage = float(np.float32(zmin))
    upper_storage = float(np.float32(zmax))
    storage_gate = bool(
        np.all(np.isfinite(values))
        and np.all(values >= lower_storage)
        and np.all(values <= upper_storage)
    )
    if not storage_gate:
        raise RuntimeError(
            "stored observed redshifts exceed the float32 image of the strict selection window"
        )
    repaired = np.clip(
        values,
        np.nextafter(float(zmin), float(zmax)),
        np.nextafter(float(zmax), float(zmin)),
    )
    changed = repaired != values
    return repaired, {
        "stored_dtype": str(stored.dtype),
        "stored_allowed_range": [lower_storage, upper_storage],
        "n_boundary_repaired": int(np.count_nonzero(changed)),
        "n_lower_repaired": int(np.count_nonzero(values <= float(zmin))),
        "n_upper_repaired": int(np.count_nonzero(values >= float(zmax))),
        "max_abs_repair": float(np.max(np.abs(repaired - values))) if values.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    args = parser.parse_args()
    row = select_row(read_jsonl(args.manifest), args.phase)
    zmin = float(row.get("zmin_observed", ZMIN))
    zmax = float(row.get("zmax_observed", ZMAX))
    p0_fkp = float(row.get("p0_fkp", P0_FKP))
    if not (0.0 <= zmin < zmax):
        raise ValueError(f"invalid observed-redshift interval {(zmin, zmax)}")
    data_path = Path(row["lightcone_catalog_path"])
    data_metadata_path = Path(row["lightcone_metadata_path"])
    output = Path(row["lightcone_random_path"])
    metadata_path = Path(row["lightcone_random_metadata_path"])
    summary_path = fkp_path(row)
    summary_metadata_path = Path(str(row.get("lightcone_fkp_metadata_path", summary_path.with_suffix(".json"))))
    products = (output, metadata_path, summary_path, summary_metadata_path)
    if all(path.is_file() for path in products):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summary_metadata = json.loads(summary_metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("status") == "pass"
            and summary_metadata.get("status") == "pass"
            and metadata.get("output_sha256") == sha256_file(output)
            and summary_metadata.get("output_sha256") == sha256_file(summary_path)
        ):
            print(f"[skip] validated random and FKP summary for {row['phase']}")
            return
    if any(path.exists() for path in products):
        raise FileExistsError(f"partial or unvalidated random/FKP products: {[str(path) for path in products]}")
    if not data_path.is_file() or not data_metadata_path.is_file():
        raise FileNotFoundError(f"missing validated data catalog: {data_path} / {data_metadata_path}")
    data_metadata = json.loads(data_metadata_path.read_text(encoding="utf-8"))
    if data_metadata.get("status") != "pass" or not data_metadata.get("positive_octant_gate", False):
        raise RuntimeError("input lightcone catalog did not pass its physical geometry gates")

    started = time.perf_counter()
    with np.load(data_path, allow_pickle=False) as data:
        data_z, data_z_encoding = repair_float32_redshift_boundaries(data["Z"], zmin=zmin, zmax=zmax)
    ndata = int(data_z.size)
    multiplier = int(row["random_multiplier"])
    if multiplier != 25:
        raise ValueError("Task4.3.2 random multiplier is frozen to 25")
    nrandom = ndata * multiplier
    source_z = data_metadata.get("z_observed", {})
    source_range_gate = bool(
        float(source_z.get("min", -np.inf)) > zmin
        and float(source_z.get("max", np.inf)) < zmax
    )
    if not source_range_gate:
        raise RuntimeError("catalog metadata does not verify the strict pre-serialization redshift cut")
    if not (np.all(np.isfinite(data_z)) and np.all((data_z > zmin) & (data_z < zmax))):
        raise RuntimeError("input observed redshifts violate the strict selection window")

    from cosmoprimo.fiducial import AbacusSummit

    cosmo = AbacusSummit(0)
    nzbin = int(round((zmax - zmin) / 0.01))
    if nzbin < 1 or not np.isclose(zmin + 0.01 * nzbin, zmax, rtol=0.0, atol=1.0e-12):
        raise ValueError("observed-redshift span must be an integer multiple of dz=0.01")
    z_edges = np.linspace(zmin, zmax, nzbin + 1, dtype="f8")
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    chi_edges = np.asarray(cosmo.comoving_radial_distance(z_edges), dtype="f8")
    volume_shell = (4.0 * np.pi / 3.0 / 8.0) * np.diff(chi_edges**3)
    data_counts = np.histogram(data_z, bins=z_edges)[0]
    nbar = data_counts.astype("f8") / volume_shell
    fkp_by_bin = 1.0 / (1.0 + nbar * p0_fkp)
    data_bin = np.clip(np.searchsorted(z_edges, data_z, side="right") - 1, 0, nbar.size - 1)
    data_weight_fkp = fkp_by_bin[data_bin]
    effective_volume = float(np.sum((nbar * p0_fkp / (1.0 + nbar * p0_fkp)) ** 2 * volume_shell))
    pair_weight = nbar**2 * fkp_by_bin**2 * volume_shell
    zeff = float(np.sum(z_centers * pair_weight) / np.sum(pair_weight))

    seed = deterministic_seed(
        "task43.2", row["phase"], "observed-z-resample", multiplier, p0_fkp, zmin, zmax
    )
    rng = np.random.default_rng(seed)
    random_z = rng.choice(data_z, size=nrandom, replace=True).astype("f4")
    # Preserve compact float32 storage while keeping the serialized random
    # catalog strictly inside the cut after the same boundary rounding.
    random_z_f8 = random_z.astype("f8")
    random_lower_mask = random_z_f8 <= zmin
    random_upper_mask = random_z_f8 >= zmax
    if np.any(random_lower_mask):
        random_z[random_lower_mask] = np.nextafter(np.float32(zmin), np.float32(np.inf))
    if np.any(random_upper_mask):
        random_z[random_upper_mask] = np.nextafter(np.float32(zmax), np.float32(-np.inf))
    random_boundary_repaired = int(np.count_nonzero(random_lower_mask) + np.count_nonzero(random_upper_mask))
    zgrid = np.linspace(zmin, zmax, 200_001, dtype="f8")
    chigrid = np.asarray(cosmo.comoving_radial_distance(zgrid), dtype="f8")
    radius = np.interp(random_z.astype("f8"), zgrid, chigrid)
    phi = rng.uniform(0.0, 0.5 * np.pi, size=nrandom)
    mu = rng.uniform(0.0, 1.0, size=nrandom)
    sintheta = np.sqrt(1.0 - mu**2)
    x = (radius * sintheta * np.cos(phi)).astype("f4")
    y = (radius * sintheta * np.sin(phi)).astype("f4")
    zcart = (radius * mu).astype("f4")
    random_bin = np.clip(np.searchsorted(z_edges, random_z, side="right") - 1, 0, nbar.size - 1)
    random_weight_fkp = fkp_by_bin[random_bin].astype("f4")
    random_counts = np.histogram(random_z, bins=z_edges)[0]
    random_index = np.repeat(np.arange(multiplier, dtype="i2"), ndata)
    input_hash = sha256_file(data_path)
    positive_octant_gate = bool(np.all(x >= 0.0) and np.all(y >= 0.0) and np.all(zcart >= 0.0))
    # NumPy may cast the Python scalar boundary to float32 when comparing a
    # float32 array.  Promote the stored values back to float64 so a value such
    # as float32(0.6000000238) is correctly recognized as strictly above 0.6.
    random_z_gate = random_z.astype("f8")
    strict_redshift_gate = bool(np.all((random_z_gate > zmin) & (random_z_gate < zmax)))
    if not positive_octant_gate or not strict_redshift_gate:
        raise RuntimeError("random catalog geometry gate failed before writing outputs")

    atomic_savez(
        output,
        RA=np.degrees(phi).astype("f4"),
        DEC=np.degrees(np.arcsin(mu)).astype("f4"),
        Z=random_z,
        X=x,
        Y=y,
        Zcart=zcart,
        WEIGHT=np.ones(nrandom, dtype="f4"),
        WEIGHT_FKP=random_weight_fkp,
        WEIGHT_TOTAL=random_weight_fkp,
        RANDOM_INDEX=random_index,
        phase=np.asarray(row["phase"]),
        seed=np.asarray(seed, dtype="i8"),
    )
    atomic_savez(
        summary_path,
        z_edges=z_edges,
        z_centers=z_centers,
        chi_edges=chi_edges,
        volume_shell=volume_shell,
        data_counts=data_counts.astype("i8"),
        random_counts=random_counts.astype("i8"),
        nbar=nbar,
        fkp_weights=fkp_by_bin,
        p0=np.asarray(p0_fkp, dtype="f8"),
        zeff=np.asarray(zeff, dtype="f8"),
        effective_volume=np.asarray(effective_volume, dtype="f8"),
        data_weight_sum=np.asarray(np.sum(data_weight_fkp), dtype="f8"),
        data_weight2_sum=np.asarray(np.dot(data_weight_fkp, data_weight_fkp), dtype="f8"),
        random_weight_sum=np.asarray(np.sum(random_weight_fkp, dtype="f8"), dtype="f8"),
        random_weight2_sum=np.asarray(np.dot(random_weight_fkp.astype("f8"), random_weight_fkp.astype("f8")), dtype="f8"),
    )
    normalized_random = random_counts.astype("f8") / multiplier
    fractional_hist_residual = (normalized_random - data_counts) / np.maximum(data_counts, 1)
    random_metadata = {
        "task": "task43_build_rsd_lightcone_random",
        "status": "pass",
        "phase": row["phase"],
        "random_policy": "uniform positive-octant solid angle; observed data-z resample",
        "random_radial_policy": row["random_radial_policy"],
        "ndata": ndata,
        "nrandom": nrandom,
        "random_multiplier": multiplier,
        "seed": seed,
        "p0": p0_fkp,
        "observed_redshift_open_interval": [zmin, zmax],
        "positive_octant_gate": positive_octant_gate,
        "strict_redshift_gate": strict_redshift_gate,
        "strict_redshift_gate_dtype": "stored float32 values promoted to float64 before comparison",
        "data_redshift_serialization_repair": data_z_encoding,
        "data_metadata_source_range_gate": source_range_gate,
        "random_redshift_boundary_repaired": random_boundary_repaired,
        "max_abs_fractional_hist_residual_after_dividing_x25": float(np.max(np.abs(fractional_hist_residual))),
        "data_catalog_path": str(data_path),
        "data_catalog_sha256": input_hash,
        "fkp_summary_path": str(summary_path),
        "output_path": str(output),
        "output_sha256": sha256_file(output),
        "elapsed_sec": time.perf_counter() - started,
    }
    summary_metadata = {
        "task": "task43_build_rsd_lightcone_random_fkp",
        "status": "pass",
        "phase": row["phase"],
        "definition": "data observed-z counts divided by positive-octant fiducial shell volume",
        "zstep": 0.01,
        "p0": p0_fkp,
        "observed_redshift_open_interval": [zmin, zmax],
        "geometric_volume_mpc3_h3": float(np.sum(volume_shell)),
        "nbar_range_h3_mpc3": [float(np.min(nbar)), float(np.max(nbar))],
        "fkp_weight_range": [float(np.min(fkp_by_bin)), float(np.max(fkp_by_bin))],
        "zeff_pair_weighted": zeff,
        "effective_volume_mpc3_h3": effective_volume,
        "output_path": str(summary_path),
        "output_sha256": sha256_file(summary_path),
    }
    atomic_write_json(metadata_path, random_metadata)
    atomic_write_json(summary_metadata_path, summary_metadata)
    print(
        json.dumps(
            {"status": "pass", "phase": row["phase"], "ndata": ndata, "nrandom": nrandom, "zeff": zeff},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
