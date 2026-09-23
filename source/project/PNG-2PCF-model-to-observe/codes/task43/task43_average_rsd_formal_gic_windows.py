#!/usr/bin/env python3
"""Average the 25 phase-specific formal-GIC windows for the x25 mean fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from task43_build_rsd_formal_gic_window import DEFAULT_WINDOW_ROOT, output_path
from task43_rsd_common import OUTPUT_ROOT, PHASES, atomic_savez, atomic_write_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsub", type=int, default=200000)
    parser.add_argument("--seed-base", type=int, default=430340)
    parser.add_argument("--phase-window-root", type=Path, default=DEFAULT_WINDOW_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "lightcone" / "formal_gic_windows" /
        "task43_rsd_formal_gic_x25_equal_phase_mean_nsub200000_seedbase430340.npz",
    )
    args = parser.parse_args()
    sidecar = args.output.with_suffix(".json")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError(f"immutable averaged formal-GIC output exists: {args.output} / {sidecar}")

    k_reference: np.ndarray | None = None
    windows: list[np.ndarray] = []
    paths: list[str] = []
    hashes: list[str] = []
    phase_metadata: list[dict] = []
    for phase in PHASES:
        path = output_path(
            phase, int(args.nsub), int(args.seed_base), root=args.phase_window_root
        )
        meta_path = path.with_suffix(".json")
        if not path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"missing phase formal-GIC window: {path} / {meta_path}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        digest = sha256_file(path)
        if metadata.get("status") != "pass" or metadata.get("output_sha256") != digest:
            raise RuntimeError(f"unvalidated phase formal-GIC window: {path}")
        with np.load(path, allow_pickle=False) as payload:
            k_eff = np.asarray(payload["k_eff"], dtype="f8")
            w2 = np.asarray(payload["w2"], dtype="f8")
        if k_reference is None:
            k_reference = k_eff
        elif not np.array_equal(k_reference, k_eff):
            raise RuntimeError(f"formal-GIC k grid differs for {phase}")
        if w2.shape != k_eff.shape or not np.all(np.isfinite(w2)):
            raise RuntimeError(f"invalid formal-GIC W2 for {phase}")
        windows.append(w2)
        paths.append(str(path))
        hashes.append(digest)
        phase_metadata.append(metadata)
    if k_reference is None:
        raise RuntimeError("no formal-GIC windows loaded")

    stack = np.stack(windows)
    mean = np.mean(stack, axis=0)
    scatter = np.std(stack, axis=0, ddof=1)
    embedded = {
        "method": "task43_x25_equal_phase_mean_formal_gic_window",
        "formula": "W2_mean(k) = mean_phase W2_phase(k); formal-GIC correction is linear in W2",
        "phases": list(PHASES),
        "nphase": len(PHASES),
        "nsub_per_phase": int(args.nsub),
        "seed_base": int(args.seed_base),
        "phase_window_root": str(args.phase_window_root),
        "input_paths": paths,
        "input_sha256": hashes,
        "trust_cache_required": True,
    }
    atomic_savez(
        args.output,
        k_eff=k_reference,
        w2=mean,
        w2_by_phase=stack,
        w2_phase_scatter=scatter,
        phases=np.asarray(PHASES),
        meta_json=np.asarray(json.dumps(embedded, sort_keys=True)),
    )
    metadata = {
        "task": "task43_average_rsd_formal_gic_windows",
        "status": "pass",
        "classification": "equal-phase window for the unweighted x25 mean measurement",
        **embedded,
        "phase_zeff": [float(item["zeff"]) for item in phase_metadata],
        "w2_phase_scatter_absmax": float(np.max(scatter)),
        "w2_phase_scatter_relative_l2": float(np.linalg.norm(scatter) / max(np.linalg.norm(mean), 1.0e-30)),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    atomic_write_json(sidecar, metadata)
    print(json.dumps({"status": "pass", "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
