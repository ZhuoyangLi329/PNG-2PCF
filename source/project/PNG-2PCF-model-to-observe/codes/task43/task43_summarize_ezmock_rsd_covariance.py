#!/usr/bin/env python3
"""Finalize and audit the Task43 EZmock RSD covariance products."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from task43_ezmock_rsd_covariance_common import (
    COMMON_RANDOM, COMMON_RANDOM_HDF5, COMMON_RR_SMU, MANIFEST, NREAL, N_PK_FINE,
    S_EDGES, SUMMARY_DIR, read_jsonl, sha256, write_json,
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--output", type=Path, default=SUMMARY_DIR / "ezmock_rsd_covariance_summary.npz")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    rows = read_jsonl(args.manifest)
    if len(rows) != NREAL:
        raise RuntimeError("manifest length %d != expected %d" % (len(rows), NREAL))
    if not (COMMON_RANDOM.is_file() and COMMON_RANDOM_HDF5.is_file() and COMMON_RR_SMU.is_file()):
        raise FileNotFoundError("common random/HDF5/s-mu RR cache missing")
    p0, p2, xi0, xi2, ndata = [], [], [], [], []
    missing = []
    for row in rows:
        for key, collection in (("pk_path", None), ("xi_path", None)):
            path = Path(row[key])
            if not path.is_file():
                missing.append(str(path))
        if missing:
            continue
        with np.load(row["pk_path"], allow_pickle=False) as src:
            a0, a2 = np.asarray(src["pk0"], dtype="f8"), np.asarray(src["pk2"], dtype="f8")
            if a0.shape != (N_PK_FINE,) or a2.shape != (N_PK_FINE,):
                raise RuntimeError("P grid mismatch at %s" % row["pk_path"])
            p0.append(a0); p2.append(a2)
        with np.load(row["xi_path"], allow_pickle=False) as src:
            a0 = np.asarray(src["xi0"], dtype="f8")
            a2 = np.asarray(src["xi2"], dtype="f8")
            if a0.shape != (S_EDGES.size - 1,) or a2.shape != (S_EDGES.size - 1,):
                raise RuntimeError("xi0/xi2 grid mismatch at %s" % row["xi_path"])
            xi0.append(a0); xi2.append(a2)
        with np.load(row["lightcone_catalog_path"], allow_pickle=False) as src:
            ndata.append(int(np.asarray(src["Z"]).size))
    if missing:
        raise RuntimeError("missing %d products; first=%s" % (len(missing), missing[0]))
    p0, p2, xi0, xi2, ndata = np.asarray(p0), np.asarray(p2), np.asarray(xi0), np.asarray(xi2), np.asarray(ndata)
    if not (np.isfinite(p0).all() and np.isfinite(p2).all() and np.isfinite(xi0).all() and np.isfinite(xi2).all()):
        raise RuntimeError("non-finite covariance input")
    mean0, mean2, meanxi0, meanxi2 = p0.mean(0), p2.mean(0), xi0.mean(0), xi2.mean(0)
    cov0 = np.cov(p0, rowvar=False, ddof=1)
    cov2 = np.cov(p2, rowvar=False, ddof=1)
    covxi0 = np.cov(xi0, rowvar=False, ddof=1)
    covxi2 = np.cov(xi2, rowvar=False, ddof=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    tmp = args.output.with_name("." + args.output.name + ".tmp.npz")
    np.savez_compressed(tmp, p0=p0, p2=p2, xi0=xi0, xi2=xi2, mean_p0=mean0, mean_p2=mean2,
                        mean_xi0=meanxi0, mean_xi2=meanxi2, cov_p0=cov0, cov_p2=cov2,
                        cov_xi0=covxi0, cov_xi2=covxi2,
                        ndata=ndata, k=np.asarray(np.load(rows[0]["pk_path"])["k"]),
                        s_edges=S_EDGES)
    tmp.replace(args.output)
    meta = {
        "task": "task43_summarize_ezmock_rsd_covariance",
        "status": "done", "nreal": int(len(rows)), "n_pk_bins": int(N_PK_FINE),
        "n_xi_bins": int(S_EDGES.size - 1), "fix_amplitude": False,
        "ndata_mean": float(ndata.mean()), "ndata_std": float(ndata.std(ddof=1)),
        "ndata_min": int(ndata.min()), "ndata_max": int(ndata.max()),
        "common_random_sha256": sha256(COMMON_RANDOM),
        "common_random_hdf5_sha256": sha256(COMMON_RANDOM_HDF5),
        "common_rr_smu_sha256": sha256(COMMON_RR_SMU), "output": str(args.output),
    }
    write_json(args.output.with_suffix(".json"), meta)
    print("[done] nreal=%d ndata=%.1f +/- %.1f output=%s" %
          (len(rows), ndata.mean(), ndata.std(ddof=1), args.output))

if __name__ == "__main__":
    main()
