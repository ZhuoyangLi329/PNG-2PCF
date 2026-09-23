#!/usr/bin/env python3
"""Restartable login-node runner for EZmock RSD lightcone production."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes/task43"
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/manifests/task43_ezmock_rsd_covariance_x1000_fixampF_common50.jsonl"
DEFAULT_LOG = PROJECT_ROOT / "outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/logs/login_runner.jsonl"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--stage", choices=("build", "pk", "all"), default="all")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--keep-rawbox", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    return p.parse_args()

def main():
    args = parse_args()
    if not 1 <= int(args.threads) <= 8:
        raise ValueError("login runner requires 1..8 threads")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = max(0, int(args.start))
    stop = len(rows) if args.stop is None else min(len(rows), int(args.stop))
    if not start < stop:
        raise ValueError("empty index range")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    script_build = CODE_DIR / "task43_build_ezmock_rsd_lightcone.py"
    script_pk = CODE_DIR / "task43_measure_ezmock_rsd_lightcone_p02.py"
    with args.log.open("a", encoding="utf-8") as log:
        for index in range(start, stop):
            row = rows[index]
            result = {"index": index, "phase": row["phase"], "seed": int(row["seed"]),
                      "stage": args.stage, "started_utc": time.time()}
            try:
                if args.stage in ("build", "all"):
                    cmd = [sys.executable, str(script_build), "--manifest", str(args.manifest),
                           "--index", str(index), "--threads", str(args.threads)]
                    if args.keep_rawbox:
                        cmd.append("--keep-rawbox")
                    if args.overwrite:
                        cmd.append("--overwrite")
                    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)
                if args.stage in ("pk", "all"):
                    cmd = [sys.executable, str(script_pk), "--manifest", str(args.manifest),
                           "--index", str(index), "--threads", str(args.threads)]
                    if args.overwrite:
                        cmd.append("--overwrite")
                    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)
                result["status"] = "done"
            except subprocess.CalledProcessError as exc:
                result["status"] = "failed"
                result["returncode"] = int(exc.returncode)
                log.write(json.dumps(result, sort_keys=True) + "\n")
                log.flush()
                raise
            result["finished_utc"] = time.time()
            result["elapsed_sec"] = result["finished_utc"] - result["started_utc"]
            log.write(json.dumps(result, sort_keys=True) + "\n")
            log.flush()
            print("[runner] index=%d phase=%s status=done elapsed=%.1fs" %
                  (index, row["phase"], result["elapsed_sec"]), flush=True)

if __name__ == "__main__":
    main()
