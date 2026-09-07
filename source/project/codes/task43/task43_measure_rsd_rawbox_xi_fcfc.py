#!/usr/bin/env python3
"""Measure paired real/RSD periodic rawbox xi0 and xi2 with FCFC."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from task43_rsd_common import PHASES, PROJECT_ROOT, atomic_savez, atomic_write_json, sha256_file


FCFC_BINARY = PROJECT_ROOT / "refcode" / "FCFC-main" / "FCFC_2PT_BOX"
LEGACY_ROOT = PROJECT_ROOT / "plots" / "outputs" / "task43_outputs" / "ezmock_rawbox_z0p725_mmin1p4e13"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_row(rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    matches = [row for row in rows if row["phase"] == phase]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {phase}, found {len(matches)}")
    return matches[0]


def legacy_ascii_path(row: dict[str, Any]) -> Path:
    return LEGACY_ROOT / "fcfc_catalogs" / (
        f"halo_{row['sim_name']}_z0p725_mmin1p4e13_xyz.txt"
    )


def old_xi_path(row: dict[str, Any]) -> Path:
    return LEGACY_ROOT / "xi_fcfc" / (
        f"xi0_{row['sim_name']}_z0p725_mmin1p4e13_s50_550_ds10_fcfc.npz"
    )


def affinity(threads: int) -> list[int]:
    if int(threads) < 1 or int(threads) > 8:
        raise ValueError("--threads must be between 1 and 8")
    available = sorted(os.sched_getaffinity(0))
    cpus = available[: int(threads)]
    if len(cpus) != int(threads):
        raise RuntimeError(f"only {len(available)} CPUs are available")
    os.sched_setaffinity(0, cpus)
    return cpus


def ensure_rsd_ascii(row: dict[str, Any], position: np.ndarray) -> tuple[Path, Path]:
    root = Path(row["rawbox_catalog_path"]).parent.parent / "fcfc_catalogs"
    path = root / f"{row['phase']}_rsd_xyz.txt"
    metadata_path = path.with_suffix(".json")
    input_path = Path(row["rawbox_catalog_path"])
    input_hash = sha256_file(input_path)
    if path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("status") == "pass"
            and metadata.get("input_sha256") == input_hash
            and metadata.get("output_sha256") == sha256_file(path)
        ):
            return path, metadata_path
        raise FileExistsError(f"unvalidated existing FCFC catalog: {path}")
    if path.exists() or metadata_path.exists():
        raise FileExistsError(f"partial FCFC catalog output: {path} / {metadata_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(f"# {row['sim_name']} Task4.3.2 plane-parallel RSD N={position.shape[0]}\n")
            np.savetxt(stream, position, fmt="%.8f %.8f %.8f")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {
        "task": "task43_measure_rsd_rawbox_xi_fcfc",
        "status": "pass",
        "phase": row["phase"],
        "space": "rsd",
        "nrows": int(position.shape[0]),
        "input_path": str(input_path),
        "input_sha256": input_hash,
        "output_path": str(path),
        "output_sha256": sha256_file(path),
    }
    atomic_write_json(metadata_path, metadata)
    return path, metadata_path


def run_fcfc(
    row: dict[str, Any],
    *,
    space: str,
    catalog: Path,
    ndata: int,
    threads: int,
    nmu: int,
    smin: float,
    smax: float,
    ds: float,
    resume_partial: bool,
) -> dict[str, Any]:
    root = Path(row["rawbox_catalog_path"]).parent.parent
    config = root / "fcfc_configs" / f"{row['phase']}_{space}_s30_350_ds10_mu{nmu}.conf"
    pair = root / "fcfc_pairs" / f"DD_{row['phase']}_{space}_s30_350_ds10_mu{nmu}.bin"
    smu_text = root / "fcfc_text" / f"xi_smu_{row['phase']}_{space}_s30_350_ds10_mu{nmu}.txt"
    multipole_text = root / "fcfc_text" / f"xi02_{row['phase']}_{space}_s30_350_ds10_mu{nmu}.txt"
    log = root / "logs" / f"fcfc_{row['phase']}_{space}_s30_350_ds10_mu{nmu}.log"
    for parent in {config.parent, pair.parent, smu_text.parent, log.parent}:
        parent.mkdir(parents=True, exist_ok=True)
    products = (config, pair, smu_text, multipole_text, log)

    def read_multipoles() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        table = np.loadtxt(multipole_text, comments="#", dtype="f8")
        if table.ndim == 1:
            table = table[None, :]
        nbins = int(round((float(smax) - float(smin)) / float(ds)))
        if table.shape != (nbins, 5):
            raise ValueError(f"unexpected FCFC multipole shape {table.shape}; see {multipole_text}")
        s, slow, shigh, xi0, xi2 = table.T
        if not np.all(np.isfinite(table)) or not np.allclose(shigh[:-1], slow[1:], atol=1.0e-12, rtol=0.0):
            raise RuntimeError(f"invalid FCFC multipoles in {multipole_text}")
        return s, slow, shigh, xi0, xi2

    complete_existing = all(path.is_file() and path.stat().st_size > 0 for path in products)
    if complete_existing:
        try:
            s, slow, shigh, xi0, xi2 = read_multipoles()
        except Exception:
            complete_existing = False
        else:
            return {
                "s": s,
                "s_edges": np.concatenate([slow[:1], shigh]),
                "xi0": xi0,
                "xi2": xi2,
                "elapsed_sec": None,
                "reused_validated_leg": True,
                "catalog_path": str(catalog),
                "config_path": str(config),
                "pair_path": str(pair),
                "smu_text_path": str(smu_text),
                "multipole_text_path": str(multipole_text),
                "log_path": str(log),
                "ndata": int(ndata),
            }
    existing = [path for path in products if path.exists()]
    if existing:
        if not resume_partial:
            raise FileExistsError(f"partial FCFC outputs already exist: {[str(path) for path in existing]}")
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        quarantine = root / "superseded_partial" / f"{row['phase']}_{space}_{stamp}_{os.getpid()}"
        quarantine.mkdir(parents=True, exist_ok=False)
        for path in existing:
            path.replace(quarantine / path.name)
        print(f"[quarantine] moved partial {row['phase']} {space} products to {quarantine}", flush=True)
    config_text = f"""CATALOG = '{catalog}'
CATALOG_LABEL = D
CATALOG_TYPE = 0
ASCII_SKIP = 1
ASCII_COMMENT = '#'
ASCII_FORMATTER = '%lf %lf %lf'
POSITION = [$1, $2, $3]
BOX_SIZE = 2000
DATA_STRUCT = 0
BINNING_SCHEME = 1
PAIR_COUNT = DD
PAIR_COUNT_FILE = '{pair}'
CF_ESTIMATOR = DD / @@ - 1
CF_OUTPUT_FILE = '{smu_text}'
MULTIPOLE = [0, 2]
MULTIPOLE_FILE = '{multipole_text}'
SEP_BIN_MIN = {float(smin):.17g}
SEP_BIN_MAX = {float(smax):.17g}
SEP_BIN_SIZE = {float(ds):.17g}
MU_BIN_NUM = {int(nmu)}
OUTPUT_FORMAT = 0
OVERWRITE = 0
VERBOSE = T
"""
    config.write_text(config_text, encoding="utf-8")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(int(threads))
    env["OMP_DYNAMIC"] = "FALSE"
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(
            [str(FCFC_BINARY), "-c", str(config)],
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
        )
    elapsed = time.perf_counter() - started
    s, slow, shigh, xi0, xi2 = read_multipoles()
    return {
        "s": s,
        "s_edges": np.concatenate([slow[:1], shigh]),
        "xi0": xi0,
        "xi2": xi2,
        "elapsed_sec": elapsed,
        "reused_validated_leg": False,
        "catalog_path": str(catalog),
        "config_path": str(config),
        "pair_path": str(pair),
        "smu_text_path": str(smu_text),
        "multipole_text_path": str(multipole_text),
        "log_path": str(log),
        "ndata": int(ndata),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nmu", type=int, default=120)
    parser.add_argument("--smin", type=float, default=30.0)
    parser.add_argument("--smax", type=float, default=350.0)
    parser.add_argument("--ds", type=float, default=10.0)
    parser.add_argument("--resume-partial", action="store_true")
    args = parser.parse_args()
    if int(args.nmu) < 20:
        raise ValueError("--nmu must be >=20")
    if (float(args.smin), float(args.smax), float(args.ds)) != (30.0, 350.0, 10.0):
        raise ValueError("Task4.3.2 frozen rawbox measurement requires s=30..350 with ds=10")
    cpus = affinity(int(args.threads))
    if not (FCFC_BINARY.is_file() and os.access(FCFC_BINARY, os.X_OK)):
        raise FileNotFoundError(FCFC_BINARY)
    row = select_row(read_jsonl(args.manifest), args.phase)
    output = Path(row["rawbox_summary_path"])
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        if output.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(output):
                print(f"[skip] validated {output}")
                return
        raise FileExistsError(f"partial or unvalidated output: {output} / {metadata_path}")

    with np.load(row["rawbox_catalog_path"], allow_pickle=False) as data:
        position_real = np.asarray(data["POSITION_REAL"], dtype="f8")
        position_rsd = np.asarray(data["POSITION_RSD"], dtype="f8")
        nbar = float(position_real.shape[0] / float(data["boxsize"]) ** 3)
    if position_real.shape != position_rsd.shape or position_real.shape[1] != 3:
        raise ValueError("rawbox real/RSD arrays are not paired")
    real_catalog = legacy_ascii_path(row)
    if not real_catalog.is_file():
        raise FileNotFoundError(real_catalog)
    rsd_catalog, rsd_catalog_metadata = ensure_rsd_ascii(row, position_rsd)
    real = run_fcfc(
        row,
        space="real",
        catalog=real_catalog,
        ndata=position_real.shape[0],
        threads=int(args.threads),
        nmu=int(args.nmu),
        smin=float(args.smin),
        smax=float(args.smax),
        ds=float(args.ds),
        resume_partial=bool(args.resume_partial),
    )
    real_runtime = "reused" if real["elapsed_sec"] is None else f"{real['elapsed_sec']:.2f}s"
    print(f"[fcfc] {row['phase']} real elapsed={real_runtime}", flush=True)
    rsd = run_fcfc(
        row,
        space="rsd",
        catalog=rsd_catalog,
        ndata=position_rsd.shape[0],
        threads=int(args.threads),
        nmu=int(args.nmu),
        smin=float(args.smin),
        smax=float(args.smax),
        ds=float(args.ds),
        resume_partial=bool(args.resume_partial),
    )
    rsd_runtime = "reused" if rsd["elapsed_sec"] is None else f"{rsd['elapsed_sec']:.2f}s"
    print(f"[fcfc] {row['phase']} rsd elapsed={rsd_runtime}", flush=True)
    if not np.array_equal(real["s_edges"], rsd["s_edges"]):
        raise RuntimeError("real/RSD FCFC separation edges differ")

    bridge: dict[str, Any] = {"path": str(old_xi_path(row)), "available": old_xi_path(row).is_file()}
    if old_xi_path(row).is_file():
        with np.load(old_xi_path(row), allow_pickle=False) as old:
            old_s = np.asarray(old["s"], dtype="f8")
            old_xi0 = np.asarray(old["xi0"], dtype="f8")
        indices = [int(np.flatnonzero(np.isclose(old_s, value, atol=1.0e-12, rtol=0.0))[0]) for value in real["s"] if value >= 50.0]
        current = np.asarray(real["xi0"], dtype="f8")[real["s"] >= 50.0]
        delta = current - old_xi0[indices]
        bridge.update(
            {
                "n_overlap": int(delta.size),
                "max_abs_xi0": float(np.max(np.abs(delta))),
                "rms_xi0": float(np.sqrt(np.mean(delta**2))),
            }
        )

    atomic_savez(
        output,
        s=np.asarray(real["s"], dtype="f8"),
        s_edges=np.asarray(real["s_edges"], dtype="f8"),
        xi0_real=np.asarray(real["xi0"], dtype="f8"),
        xi2_real=np.asarray(real["xi2"], dtype="f8"),
        xi0_rsd=np.asarray(rsd["xi0"], dtype="f8"),
        xi2_rsd=np.asarray(rsd["xi2"], dtype="f8"),
        phase=np.asarray(row["phase"]),
        ndata=np.asarray(position_real.shape[0], dtype="i8"),
        nbar=np.asarray(nbar, dtype="f8"),
        boxsize=np.asarray(2000.0, dtype="f8"),
        redshift=np.asarray(0.725, dtype="f8"),
        ells=np.asarray([0, 2], dtype="i4"),
    )
    metadata = {
        "task": "task43_measure_rsd_rawbox_xi_fcfc",
        "status": "pass",
        "phase": row["phase"],
        "engine": "FCFC_2PT_BOX",
        "estimator": "DD / analytic_RR - 1",
        "periodic": True,
        "los": "z",
        "ells": [0, 2],
        "nmu": int(args.nmu),
        "s_edges_mpc_h": real["s_edges"].tolist(),
        "ndata": int(position_real.shape[0]),
        "nbar_h3_mpc3": nbar,
        "threads": int(args.threads),
        "cpu_affinity": cpus,
        "rawbox_catalog_path": row["rawbox_catalog_path"],
        "rsd_ascii_metadata_path": str(rsd_catalog_metadata),
        "real": {key: value for key, value in real.items() if not isinstance(value, np.ndarray)},
        "rsd": {key: value for key, value in rsd.items() if not isinstance(value, np.ndarray)},
        "legacy_real_xi0_bridge": bridge,
        "output_path": str(output),
        "output_sha256": sha256_file(output),
    }
    atomic_write_json(metadata_path, metadata)
    print(json.dumps({"status": "pass", "phase": row["phase"], "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
