#!/usr/bin/env python3
"""Replot the stored l=0 rawbox comparison without rerunning inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import make_plot
from task43_rsd_common import atomic_write_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.output.with_suffix(".json")
    if args.output.exists() or audit_path.exists():
        raise FileExistsError(f"immutable replot exists: {args.output} / {audit_path}")
    with np.load(args.input, allow_pickle=False) as data:
        pk = {"k": np.asarray(data["k"], dtype="f8")}
        xi = {
            "s": np.asarray(data["s"], dtype="f8"),
            "mean": np.asarray(data["xi0_rsd_mean"], dtype="f8"),
            "prediction": np.asarray(data["xi0_model_map"], dtype="f8"),
            "covariance": np.asarray(data["xi0_covariance_single"], dtype="f8"),
            "chain": np.asarray(data["xi0_chain_by_step"], dtype="f8"),
        }
        pk_mean = np.asarray(data["pk0_rsd_mean"], dtype="f8")
        pk_prediction = np.asarray(data["pk0_model_map"], dtype="f8")
        pk_covariance = np.asarray(data["pk0_covariance_single"], dtype="f8")
        pk_chain = np.asarray(data["pk0_chain_by_step"], dtype="f8")
        forbidden = [name for name in data.files if "xi2" in name.lower() or "pk2" in name.lower()]
    if forbidden:
        raise RuntimeError(f"ell=2 arrays found in l=0-only source: {forbidden}")
    make_plot(args.output, pk, pk_mean, pk_prediction, pk_covariance, pk_chain, xi)
    atomic_write_json(
        audit_path,
        {
            "task": "task43_replot_rsd_rawbox_pk0_vs_xi0_l0only",
            "status": "pass",
            "scope": "rawbox P0(k) vs xi0(smin=50), ell=0 only; no inference rerun",
            "input": str(args.input),
            "input_sha256": sha256_file(args.input),
            "output_pdf": str(args.output),
            "output_pdf_sha256": sha256_file(args.output),
            "pages": 2,
            "sigma_s_display_range_mpc_h": [0.0, 15.0],
            "ell2_arrays": [],
        },
    )
    print(json.dumps({"status": "pass", "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
