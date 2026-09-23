#!/usr/bin/env python3
"""Resumable serial login-node runner for Task43 EZmock covariance rows."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

from task43_ezmock_covariance_common import MANIFEST, NREAL, SUMMARY_DIR, read_jsonl, write_json
from task43_rawbox_ezmock_common import set_cpu_affinity


HERE = Path(__file__).resolve().parent
PROGRESS_STEM = "task43_ezmock_covariance_production_progress"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=NREAL, help="Exclusive index.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--stages", choices=("all", "lightcone", "lightcone_xi", "xi", "pk"), default="all")
    parser.add_argument("--no-window", action="store_true", help="Do not request the one index-0 smooth window.")
    parser.add_argument(
        "--wait-for-lightcone", action="store_true",
        help="For a P(k)-only worker, wait for each atomically completed lightcone instead of failing if it is not ready yet.",
    )
    return parser.parse_args()


def command(script: str, index: int, threads: int, *extra: str) -> list[str]:
    return [sys.executable, str(HERE / script), "--index", str(index), "--threads", str(threads), *extra]


def wait_for_lightcone(row: dict[str, object], poll_sec: float = 5.0) -> None:
    catalog = Path(str(row["halo_catalog_path"]))
    metadata = Path(str(row["halo_metadata_path"]))
    announced = False
    while True:
        if catalog.is_file() and metadata.is_file():
            try:
                import json

                meta = json.loads(metadata.read_text(encoding="utf-8"))
                if (
                    meta.get("status") == "done"
                    and meta.get("fix_amplitude") is False
                    and int(meta.get("seed", -1)) == int(row["seed"])
                ):
                    return
            except (OSError, ValueError, TypeError):
                pass
        if not announced:
            print(f"[wait lightcone] {row['phase']} seed={row['seed']}", flush=True)
            announced = True
        time.sleep(float(poll_sec))


def main() -> None:
    args = parse_args()
    if os.environ.get("TASK43_EZCOV_SUPERVISED") != "1":
        raise RuntimeError(
            "x1000 production runner must be launched by "
            "run_task43_ezmock_covariance_login_persistent.sh"
        )
    if not 1 <= int(args.threads) <= 12:
        raise ValueError("login-node production contract requires 1..12 cores per worker")
    required_modules = ("cosmoprimo",) if args.stages in ("lightcone", "lightcone_xi") else (("jax", "jaxpower", "cosmoprimo") if args.stages in ("all", "pk") else ())
    missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing_modules:
        raise RuntimeError(
            f"missing production environment modules {missing_modules}; launch through run_task43_ezmock_covariance_login.sh"
        )
    os.environ.update({
        "JAX_PLATFORMS": "cpu", "JAX_PLATFORM_NAME": "cpu", "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    })
    xla = os.environ.get("XLA_FLAGS", "")
    thread_flags = f" --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={int(args.threads)} inter_op_parallelism_threads=1"
    if "intra_op_parallelism_threads" not in xla:
        os.environ["XLA_FLAGS"] = xla + thread_flags
    cpus = set_cpu_affinity(int(args.threads))
    rows = read_jsonl(args.manifest)
    start, stop = int(args.start), int(args.stop)
    if not 0 <= start < stop <= len(rows):
        raise ValueError((start, stop, len(rows)))
    completed = []
    progress = SUMMARY_DIR / f"{PROGRESS_STEM}_{args.stages}.json"
    run_started = time.perf_counter()
    for index in range(start, stop):
        row_started = time.perf_counter(); row = rows[index]
        print(f"[row {index:04d}/{stop - 1:04d}] {row['phase']} seed={row['seed']} FIX_AMPLITUDE=F", flush=True)
        if args.stages in ("all", "lightcone", "lightcone_xi"):
            subprocess.run(command("task43_build_ezmock_covariance_lightcone.py", index, args.threads), check=True)
        if args.stages in ("all", "xi", "lightcone_xi"):
            subprocess.run(command("task43_measure_ezmock_covariance_xi_fcfc.py", index, args.threads), check=True)
        if args.stages in ("all", "pk"):
            if args.wait_for_lightcone:
                wait_for_lightcone(row)
            extra = ("--window",) if index == 0 and not args.no_window else ()
            subprocess.run(command("task43_measure_ezmock_covariance_pk_jaxpower.py", index, args.threads, *extra), check=True)
        elapsed = time.perf_counter() - row_started
        completed.append(index)
        write_json(progress, {
            "task": "task43_run_ezmock_covariance_login", "status": "running" if index + 1 < stop else "requested_range_done",
            "manifest": str(args.manifest), "requested_range": [start, stop], "stages": args.stages,
            "threads": int(args.threads), "cpu_affinity": cpus, "last_completed_index": index,
            "wait_for_lightcone": bool(args.wait_for_lightcone),
            "last_completed_phase": row["phase"], "completed_this_invocation": completed,
            "last_row_sec": float(elapsed), "invocation_elapsed_sec": float(time.perf_counter() - run_started),
        })
        print(f"[row done] index={index} elapsed={elapsed:.1f}s", flush=True)
    print(f"[range done] {start}:{stop} elapsed={time.perf_counter() - run_started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
