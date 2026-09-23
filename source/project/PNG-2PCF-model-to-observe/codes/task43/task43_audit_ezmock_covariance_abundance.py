#!/usr/bin/env python3
"""Audit and freeze the ten-row FIX_AMPLITUDE=F abundance pilot."""

from __future__ import annotations

import json

import numpy as np

from task43_ezmock_covariance_common import (
    ABUNDANCE_AUDIT,
    DEFAULT_NTRACER,
    MANIFEST,
    TARGET_NDATA,
    read_jsonl,
    write_json,
)


def main() -> None:
    rows = read_jsonl(MANIFEST)[:10]
    metadata = []
    for row in rows:
        path = row["halo_metadata_path"]
        meta = json.loads(open(path, encoding="utf-8").read())
        if not (
            meta.get("status") == "done"
            and meta.get("fix_amplitude") is False
            and int(meta.get("seed", -1)) == int(row["seed"])
            and int(meta.get("ntracer_requested", -1)) == DEFAULT_NTRACER
        ):
            raise RuntimeError(f"abundance pilot provenance gate failed: {path}")
        metadata.append(meta)
    counts = np.asarray([meta["selected_count"] for meta in metadata], dtype="i8")
    mean = float(counts.mean()); fraction = mean / TARGET_NDATA - 1.0
    payload = {
        "task": "task43_audit_ezmock_covariance_abundance", "status": "pass" if abs(fraction) < 0.02 else "fail",
        "classification": "ten-realization FIX_AMPLITUDE=F abundance pilot",
        "manifest": str(MANIFEST), "indices": list(range(10)),
        "seeds": [int(row["seed"]) for row in rows], "ntracer_frozen": DEFAULT_NTRACER,
        "target_abacus_mean_count": TARGET_NDATA, "selected_counts": counts,
        "mean": mean, "std_ddof1": float(counts.std(ddof=1)), "fractional_offset": fraction,
        "absolute_fractional_gate": 0.02, "fix_amplitude": False, "attach_particle": True,
    }
    if payload["status"] != "pass":
        raise RuntimeError(f"abundance gate failed: offset={fraction:+.4%}")
    write_json(ABUNDANCE_AUDIT, payload)
    print(f"[pass] {ABUNDANCE_AUDIT} mean={mean:.1f} offset={fraction:+.4%}")


if __name__ == "__main__":
    main()
