#!/usr/bin/env python3
"""Measure Task4.3.2 lightcone xi0/xi2 with split-random GPU cucount."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from task43_build_rsd_lightcone_random import fkp_path
from task43_rsd_common import PHASES, S_EDGES, atomic_savez, atomic_write_json, sha256_file


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_row(rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    matches = [row for row in rows if row["phase"] == phase]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {phase}, found {len(matches)}")
    return matches[0]


def project_multipoles(xi_smu: np.ndarray, mu_edges: np.ndarray, ells: tuple[int, ...]) -> np.ndarray:
    xi_smu = np.asarray(xi_smu, dtype="f8")
    if xi_smu.ndim != 2 or xi_smu.shape[1] != len(mu_edges) - 1:
        raise ValueError("xi(s,mu) shape does not match mu edges")
    projected = []
    for ell in ells:
        coeff = np.zeros(ell + 1, dtype="f8")
        coeff[ell] = 1.0
        icoeff = np.polynomial.legendre.legint(coeff)
        integral = np.polynomial.legendre.legval(mu_edges[1:], icoeff) - np.polynomial.legendre.legval(
            mu_edges[:-1], icoeff
        )
        projected.append(0.5 * (2 * ell + 1) * np.sum(xi_smu * integral[None, :], axis=1))
    return np.asarray(projected, dtype="f8")


def fkp_weights(redshift: np.ndarray, summary: dict[str, np.ndarray]) -> np.ndarray:
    z_edges = np.asarray(summary["z_edges"], dtype="f8")
    per_bin = np.asarray(summary["fkp_weights"], dtype="f8")
    index = np.clip(np.searchsorted(z_edges, np.asarray(redshift, dtype="f8"), side="right") - 1, 0, per_bin.size - 1)
    return per_bin[index]


def load_catalogs(row: dict[str, Any]) -> dict[str, np.ndarray]:
    summary_path = fkp_path(row)
    with np.load(summary_path, allow_pickle=False) as payload:
        summary = {key: np.asarray(payload[key]) for key in payload.files}
    with np.load(row["lightcone_catalog_path"], allow_pickle=False) as data:
        data_xyz = np.column_stack([data["X"], data["Y"], data["Zcart"]]).astype("f4")
        data_z = np.asarray(data["Z"], dtype="f8")
        data_base = np.asarray(data["WEIGHT"], dtype="f8")
    with np.load(row["lightcone_random_path"], allow_pickle=False) as random:
        random_xyz = np.column_stack([random["X"], random["Y"], random["Zcart"]]).astype("f4")
        random_z = np.asarray(random["Z"], dtype="f8")
        random_base = np.asarray(random["WEIGHT"], dtype="f8")
        random_index = np.asarray(random["RANDOM_INDEX"], dtype="i2")
        stored_total = np.asarray(random["WEIGHT_TOTAL"], dtype="f8")
    data_weight = data_base * fkp_weights(data_z, summary)
    random_weight = random_base * fkp_weights(random_z, summary)
    if not np.allclose(random_weight, stored_total, rtol=1.0e-7, atol=1.0e-8):
        raise RuntimeError("stored random WEIGHT_TOTAL does not match the frozen FKP summary")
    return {
        "data_xyz": data_xyz,
        "data_z": data_z,
        "data_weight": data_weight.astype("f4"),
        "random_xyz": random_xyz,
        "random_z": random_z,
        "random_weight": random_weight.astype("f4"),
        "random_index": random_index,
        "zeff": np.asarray(summary["zeff"], dtype="f8"),
        "p0": np.asarray(summary["p0"], dtype="f8"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--nmu", type=int, default=40)
    parser.add_argument("--ells", default="0,2", help="Comma-separated observed multipoles; supported: 0 or 0,2")
    args = parser.parse_args()
    if int(args.nmu) < 20 or int(args.nmu) % 2:
        raise ValueError("--nmu must be an even integer >=20")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise RuntimeError("GPU cucount requested but CUDA_VISIBLE_DEVICES is empty")
    ells = tuple(int(value) for value in str(args.ells).split(",") if value.strip())
    if ells not in ((0,), (0, 2)):
        raise ValueError("--ells must be either 0 or 0,2")
    row = select_row(read_jsonl(args.manifest), args.phase)
    output = Path(row["lightcone_xi_path"])
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        if output.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(output):
                print(f"[skip] validated {output}")
                return
        raise FileExistsError(f"partial or unvalidated lightcone xi output: {output} / {metadata_path}")

    started = time.perf_counter()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax

    jax.config.update("jax_enable_x64", True)
    devices = jax.devices()
    if not devices or devices[0].platform != "gpu":
        raise RuntimeError(f"Task4.3.2 production cucount requires GPU; found {devices}")
    from cucount.jax import BinAttrs, MeshAttrs, Particles, WeightAttrs
    from cucount.types import count2
    from lsstypes import Count2Correlation

    arrays = load_catalogs(row)
    data_xyz = arrays["data_xyz"]
    data_weight = arrays["data_weight"]
    random_xyz = arrays["random_xyz"]
    random_weight = arrays["random_weight"]
    random_index = arrays["random_index"]
    indices = sorted(int(value) for value in np.unique(random_index))
    if indices != list(range(int(row["random_multiplier"]))):
        raise RuntimeError(f"unexpected split-random indices: {indices}")
    expected_size = data_xyz.shape[0]
    sizes = [int(np.count_nonzero(random_index == index)) for index in indices]
    if sizes != [expected_size] * len(indices):
        raise RuntimeError(f"split random block sizes do not equal ndata={expected_size}: {sizes}")

    s_edges = np.asarray(S_EDGES, dtype="f8")
    mu_edges = np.linspace(-1.0, 1.0, int(args.nmu) + 1, dtype="f8")
    battrs = BinAttrs(s=s_edges, mu=(mu_edges, "midpoint"))
    wattrs = WeightAttrs()
    data_particles = Particles(data_xyz, weights=data_weight, exchange=True)
    dd_mattrs = MeshAttrs(data_particles, data_particles, battrs=battrs, periodic=False)
    dd = count2(data_particles, data_particles, battrs=battrs, mattrs=dd_mattrs, wattrs=wattrs)["weight"]
    dd_value = np.asarray(dd.value(), dtype="f8")
    xi_by_random: list[np.ndarray] = []
    dr_by_random: list[np.ndarray] = []
    rr_by_random: list[np.ndarray] = []
    s: np.ndarray | None = None
    mu: np.ndarray | None = None
    block_elapsed: list[float] = []
    for index in indices:
        block_started = time.perf_counter()
        mask = random_index == index
        random_particles = Particles(random_xyz[mask], weights=random_weight[mask], exchange=True)
        mattrs = MeshAttrs(data_particles, random_particles, battrs=battrs, periodic=False)
        dr = count2(data_particles, random_particles, battrs=battrs, mattrs=mattrs, wattrs=wattrs)["weight"]
        rr = count2(random_particles, random_particles, battrs=battrs, mattrs=mattrs, wattrs=wattrs)["weight"]
        correlation = Count2Correlation(estimator="landyszalay", DD=dd, DS=dr, SD=dr, SS=rr, RR=rr)
        xi = np.asarray(correlation.value(), dtype="f8")
        jax.block_until_ready(xi)
        if s is None:
            s = np.asarray(correlation.coords("s"), dtype="f8")
            mu = np.asarray(correlation.coords("mu"), dtype="f8")
        xi_by_random.append(xi)
        dr_by_random.append(np.asarray(dr.value(), dtype="f8"))
        rr_by_random.append(np.asarray(rr.value(), dtype="f8"))
        block_elapsed.append(time.perf_counter() - block_started)
        print(
            f"[split-random-smu] {row['phase']} index={index:02d} "
            f"finite={bool(np.all(np.isfinite(xi)))} elapsed={block_elapsed[-1]:.2f}s",
            flush=True,
        )
        del random_particles, mattrs, dr, rr, correlation
        gc.collect()
    if s is None or mu is None:
        raise RuntimeError("no split-random measurements were produced")
    xi_stack = np.stack(xi_by_random)
    xi_smu = np.mean(xi_stack, axis=0)
    dr_mean = np.mean(np.stack(dr_by_random), axis=0)
    rr_mean = np.mean(np.stack(rr_by_random), axis=0)
    multipoles = project_multipoles(xi_smu, mu_edges, ells)
    xi0 = multipoles[ells.index(0)]
    rr_radial = np.sum(rr_mean, axis=1)
    finite_gate = bool(
        np.all(np.isfinite(xi_stack))
        and np.all(np.isfinite(xi0))
        and np.all(np.isfinite(multipoles))
        and np.all(rr_mean > 0.0)
    )
    if not finite_gate:
        raise RuntimeError("lightcone xi finite/RR gate failed")

    save_payload: dict[str, Any] = {
        "s": s,
        "s_edges": s_edges,
        "mu": mu,
        "mu_edges": mu_edges,
        "ells": np.asarray(ells, dtype="i4"),
        "xi0": xi0,
        "xi_multipoles": multipoles,
        "xi_smu": xi_smu,
        "xi_smu_by_random": xi_stack,
        "DD_smu": dd_value,
        "DR_smu": dr_mean,
        "RR_smu": rr_mean,
        "RR": rr_radial,
        "ndata": np.asarray(data_xyz.shape[0], dtype="i8"),
        "nrandom": np.asarray(random_xyz.shape[0], dtype="i8"),
        "zeff": arrays["zeff"],
        "p0": arrays["p0"],
        "phase": np.asarray(row["phase"]),
    }
    if 2 in ells:
        save_payload["xi2"] = multipoles[ells.index(2)]
    atomic_savez(
        output,
        **save_payload,
    )
    metadata = {
        "task": "task43_measure_rsd_lightcone_xi",
        "status": "pass",
        "phase": row["phase"],
        "engine": "cucount.jax GPU",
        "devices": [str(device) for device in devices],
        "estimator": "Landy-Szalay, mean over 25 independent RANDOM_INDEX blocks",
        "los": "midpoint",
        "ells": list(ells),
        "nmu": int(args.nmu),
        "s_edges_mpc_h": s_edges.tolist(),
        "ndata": int(data_xyz.shape[0]),
        "nrandom_total": int(random_xyz.shape[0]),
        "random_block_sizes": sizes,
        "p0": float(arrays["p0"]),
        "zeff": float(arrays["zeff"]),
        "weighting": "WEIGHT_TOTAL=WEIGHT*WEIGHT_FKP from phase observed nbar(z)",
        "radial_random_policy": row["random_radial_policy"],
        "finite_and_rr_positive_gate": finite_gate,
        "rr_radial_definition": "RR=sum_mu(mean split-random RR_smu); retained for Task44-compatible fit views",
        "block_elapsed_sec": block_elapsed,
        "data_catalog_path": row["lightcone_catalog_path"],
        "data_catalog_sha256": sha256_file(Path(row["lightcone_catalog_path"])),
        "random_catalog_path": row["lightcone_random_path"],
        "random_catalog_sha256": sha256_file(Path(row["lightcone_random_path"])),
        "fkp_summary_path": str(fkp_path(row)),
        "output_path": str(output),
        "output_sha256": sha256_file(output),
        "elapsed_sec": time.perf_counter() - started,
    }
    atomic_write_json(metadata_path, metadata)
    print(json.dumps({"status": "pass", "phase": row["phase"], "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
