#!/usr/bin/env python3
"""Measure the Task43 mass-matched periodic raw-box xi0(s) with FCFC."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from task43_rawbox_ezmock_common import (
    CONFIG_DIR,
    LOG_DIR,
    PAIR_DIR,
    PHASES,
    ascii_path,
    atomic_savez,
    catalog_metadata_path,
    set_cpu_affinity,
    sim_name,
    write_json,
    xi_metadata_path,
    xi_path,
)


FCFC_BINARY = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/refcode/FCFC-main/FCFC_2PT_BOX")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--smin", type=float, default=50.0)
    parser.add_argument("--smax", type=float, default=550.0)
    parser.add_argument("--ds", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cpus = set_cpu_affinity(args.threads)
    output = xi_path(args.phase)
    output_meta = xi_metadata_path(args.phase)
    if output.exists() and output_meta.exists() and not args.overwrite:
        print(f"[skip] {output}", flush=True)
        return
    if not FCFC_BINARY.is_file() or not os.access(FCFC_BINARY, os.X_OK):
        raise FileNotFoundError(f"FCFC binary is missing or not executable: {FCFC_BINARY}")

    catalog_meta_path = catalog_metadata_path(args.phase)
    if not catalog_meta_path.exists() or not ascii_path(args.phase).exists():
        raise FileNotFoundError(f"missing built catalog for {args.phase}")
    import json

    catalog_meta = json.loads(catalog_meta_path.read_text(encoding="utf-8"))
    conf_path = CONFIG_DIR / f"{args.phase}_s50_550_ds10.conf"
    pair_path = PAIR_DIR / f"DD_{args.phase}_s50_550_ds10.bin"
    xi_text = output.with_suffix(".txt")
    log_path = LOG_DIR / f"fcfc_{args.phase}_s50_550_ds10.log"
    for path in (conf_path.parent, pair_path.parent, xi_text.parent, log_path.parent):
        path.mkdir(parents=True, exist_ok=True)

    conf = f"""CATALOG = '{ascii_path(args.phase)}'
CATALOG_LABEL = D
CATALOG_TYPE = 0
ASCII_SKIP = 1
ASCII_COMMENT = '#'
ASCII_FORMATTER = '%lf %lf %lf'
POSITION = [$1, $2, $3]
BOX_SIZE = {float(catalog_meta['boxsize']):.17g}
DATA_STRUCT = 0
BINNING_SCHEME = 0
PAIR_COUNT = DD
PAIR_COUNT_FILE = '{pair_path}'
CF_ESTIMATOR = DD / @@ - 1
CF_OUTPUT_FILE = '{xi_text}'
SEP_BIN_MIN = {float(args.smin):.17g}
SEP_BIN_MAX = {float(args.smax):.17g}
SEP_BIN_SIZE = {float(args.ds):.17g}
OVERWRITE = {2 if args.overwrite else 1}
VERBOSE = T
"""
    conf_path.write_text(conf, encoding="utf-8")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["OMP_DYNAMIC"] = "FALSE"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run([str(FCFC_BINARY), "-c", str(conf_path)], check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    elapsed = time.perf_counter() - started

    table = np.loadtxt(xi_text, comments="#", dtype="f8")
    if table.ndim == 1:
        table = table[None, :]
    if table.shape != (int(round((args.smax - args.smin) / args.ds)), 4):
        raise ValueError(f"unexpected FCFC xi table shape: {table.shape}")
    s, s_lower, s_upper, xi0 = table.T
    if not np.allclose(s_upper[:-1], s_lower[1:], rtol=0.0, atol=1.0e-10):
        raise ValueError("non-contiguous FCFC separation bins")
    s_edges = np.concatenate([s_lower[:1], s_upper])
    if not np.all(np.isfinite(xi0)):
        raise ValueError("non-finite FCFC xi0")

    metadata = {
        "task": "task43_measure_ezmock_rawbox_xi_fcfc",
        "status": "done",
        "phase": args.phase,
        "sim_name": sim_name(args.phase),
        "catalog_path": str(ascii_path(args.phase)),
        "output_path": str(output),
        "output_text": str(xi_text),
        "configuration_path": str(conf_path),
        "pair_count_path": str(pair_path),
        "log_path": str(log_path),
        "engine": "FCFC_2PT_BOX v1.0.1 OpenMP",
        "estimator": "DD / analytic_RR - 1",
        "periodic": True,
        "space": "real",
        "boxsize": float(catalog_meta["boxsize"]),
        "ndata": int(catalog_meta["n_selected"]),
        "smin": float(args.smin),
        "smax": float(args.smax),
        "ds": float(args.ds),
        "n_bins": int(xi0.size),
        "threads_requested": int(args.threads),
        "cpu_affinity": cpus,
        "elapsed_sec": elapsed,
    }
    atomic_savez(
        output,
        s=s,
        s_edges=s_edges,
        xi0=xi0,
        phase=np.asarray(args.phase),
        ndata=np.asarray(catalog_meta["n_selected"], dtype="i8"),
        nbar=np.asarray(catalog_meta["nbar"], dtype="f8"),
        boxsize=np.asarray(catalog_meta["boxsize"], dtype="f8"),
        redshift=np.asarray(catalog_meta["redshift"], dtype="f8"),
        engine=np.asarray("FCFC_2PT_BOX"),
        estimator=np.asarray("DD / analytic_RR - 1"),
    )
    write_json(output_meta, metadata)
    print(f"[done] {args.phase} xi0 bins={xi0.size} elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
