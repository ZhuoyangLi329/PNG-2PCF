#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the 50--550 Task43 weighted RR(s,mu) deconvolution window.

This deliberately reproduces the authoritative Task43 RR sampling policy:
ph000, 100,000 randoms, seed 20260702, FKP P0=10000 and 20 signed-mu
midpoint bins.  The first 30 separation bins are bridged to the old window.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from task43_config import PROJECT_ROOT, read_jsonl
from task43_fkp_zeff import load_fkp_summary, total_weight_from_summary


SCAN_ROOT = PROJECT_ROOT / "outputs/task43_outputs/rmax_scan"
DEFAULT_MANIFEST = SCAN_ROOT / "manifests/task43_rmax_scan_mmin1p4e13_x25.jsonl"
DEFAULT_FKP = PROJECT_ROOT / "outputs/task43_outputs/summary/task43_fkp_zeff_mmin1p4e13_x25.npz"
DEFAULT_OUTPUT = SCAN_ROOT / "covariance/task43_rr_smu_ph000_nran100k_seed20260702_s50_550_ds10_nmu20"
OLD_RR = (
    PROJECT_ROOT
    / "outputs/task43_outputs/summary/task43_rr_smu_window_smoke_ph000_nran100k_s50_350_ds10_nmu20_midpoint.npz"
)
AUTHORITATIVE_RR_FKP_NORM = 4.89250917556916e-10


def choose_rows(n_total: int, n_used: int, seed: int) -> np.ndarray | slice:
    if n_used <= 0 or n_used >= n_total:
        return slice(None)
    return np.sort(np.random.default_rng(int(seed)).choice(n_total, size=int(n_used), replace=False))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fkp-summary", type=Path, default=DEFAULT_FKP)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase-index", type=int, default=0)
    parser.add_argument("--max-random", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--p0", type=float, default=10000.0)
    parser.add_argument("--s-min", type=float, default=50.0)
    parser.add_argument("--s-max", type=float, default=550.0)
    parser.add_argument("--s-step", type=float, default=10.0)
    parser.add_argument("--nmu", type=int, default=20)
    parser.add_argument("--nthreads", type=int, default=8)
    parser.add_argument("--old-rr", type=Path, default=OLD_RR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 1 <= int(args.nthreads) <= 8:
        raise ValueError("--nthreads must be in [1,8] on the login node")
    if int(args.phase_index) != 0:
        raise ValueError("the fixed Task43 covariance window must use ph000/phase-index 0")
    out_npz = args.output_prefix.with_suffix(".npz")
    out_json = args.output_prefix.with_suffix(".json")
    if (out_npz.exists() or out_json.exists()) and not args.force:
        if out_npz.exists() and out_json.exists():
            print(f"[skip] existing RR window {out_npz}")
            return
        raise FileExistsError(f"partial output exists; inspect before using --force: {out_npz}, {out_json}")

    from pycorr import TwoPointCorrelationFunction

    rows = read_jsonl(args.manifest)
    row = rows[int(args.phase_index)]
    if str(row["phase"]) != "ph000":
        raise ValueError(f"manifest row 0 is not ph000: {row['phase']}")
    random_path = Path(row["random_catalog_path"])
    random = np.load(random_path, allow_pickle=False)
    fkp = load_fkp_summary(args.fkp_summary)
    z = np.asarray(random["Z"], dtype="f8")
    base_weight = np.asarray(random["WEIGHT"], dtype="f8") if "WEIGHT" in random.files else np.ones(z.size)
    weight_all = total_weight_from_summary(z, base_weight, fkp, p0=float(args.p0))
    choice = choose_rows(z.size, int(args.max_random), int(args.seed))
    xyz = np.column_stack(
        [
            np.asarray(random["X"][choice], dtype="f8"),
            np.asarray(random["Y"][choice], dtype="f8"),
            np.asarray(random["Zcart"][choice], dtype="f8"),
        ]
    )
    weight = np.asarray(weight_all[choice], dtype="f8")
    s_edges = np.arange(
        float(args.s_min), float(args.s_max) + 0.5 * float(args.s_step), float(args.s_step), dtype="f8"
    )
    mu_edges = np.linspace(-1.0, 1.0, int(args.nmu) + 1, dtype="f8")
    # 该构造器最初只服务 50--550 的 50-bin rmax scan；rmin scan 需要
    # 30--350 的 32 bins。这里保留相同算法和归一化，只把径向 bin 数泛化，
    # 从而避免复制一个几乎相同的 RR 实现。
    ns = int(s_edges.size - 1)
    nmu = int(mu_edges.size - 1)
    if ns < 1 or nmu < 1:
        raise ValueError(f"RR 网格必须至少各含一个 bin，得到 ns={ns}, nmu={nmu}")

    result = TwoPointCorrelationFunction(
        "smu",
        (s_edges, mu_edges),
        data_positions1=xyz,
        data_weights1=weight,
        randoms_positions1=xyz,
        randoms_weights1=weight,
        position_type="pos",
        engine="corrfunc",
        nthreads=int(args.nthreads),
    )
    rr_counts = np.asarray(result.R1R2.wcounts, dtype="f8")
    pycorr_rr_wnorm = float(result.R1R2.wnorm)
    # The accepted Task43 RR cache used the ordered-pair convention sum(w)^2,
    # whereas the current pycorr release reports sum(w)^2-sum(w^2).  Preserve
    # the accepted convention so the 350-bin bridge is normalization-identical.
    rr_norm_scalar = float(np.sum(weight)) ** 2
    if args.old_rr.exists():
        old_for_norm = np.load(args.old_rr, allow_pickle=False)
        old_norm_values = np.asarray(old_for_norm["rr_norm"], dtype="f8")
        if not np.allclose(old_norm_values, old_norm_values.flat[0], rtol=0.0, atol=0.0):
            raise RuntimeError("authoritative RR cache does not have a scalar normalization")
        if not np.isclose(rr_norm_scalar, old_norm_values.flat[0], rtol=1.0e-12, atol=1.0e-6):
            raise RuntimeError("sum(w)^2 does not reproduce the authoritative RR normalization")
        rr_norm_scalar = float(old_norm_values.flat[0])
    rr_norm = np.full_like(rr_counts, rr_norm_scalar, dtype="f8")
    rr_value = rr_counts / rr_norm
    if (
        rr_counts.shape != (ns, nmu)
        or not np.all(np.isfinite(rr_counts))
        or np.any(rr_counts < 0.0)
        or not np.all(np.sum(rr_counts, axis=1) > 0.0)
    ):
        raise RuntimeError(f"invalid RR counts shape/values: {rr_counts.shape}")

    bridge: dict[str, Any] = {"available": False}
    if args.old_rr.exists():
        old = np.load(args.old_rr, allow_pickle=False)
        old_edges = np.asarray(old["s_edges"], dtype="f8")
        old_counts = np.asarray(old["rr_counts"], dtype="f8")
        old_norm = np.asarray(old["rr_norm"], dtype="f8")
        # 通过精确 edge 匹配定位旧区间，兼容旧 rmax scan 的前缀桥接和
        # 新 rmin scan 的内部切片桥接，例如新 30--350 中的旧 50--350。
        matches = np.flatnonzero(np.isclose(s_edges, old_edges[0], rtol=0.0, atol=1.0e-12))
        if matches.size != 1:
            raise RuntimeError(f"新 RR 网格中找不到旧起始 edge={old_edges[0]}")
        start = int(matches[0])
        stop = start + int(old_counts.shape[0])
        if stop >= s_edges.size or not np.array_equal(s_edges[start : stop + 1], old_edges):
            raise RuntimeError("新 RR 网格不完整包含旧 RR edge 区间")
        new_counts = rr_counts[start:stop]
        new_norm = rr_norm[start:stop]
        delta = new_counts - old_counts
        bridge = {
            "available": True,
            "old_rr": str(args.old_rr),
            "new_bin_slice": [start, stop],
            "s_edges_exact": True,
            "mu_edges_exact": bool(np.array_equal(mu_edges, np.asarray(old["mu_edges"], dtype="f8"))),
            "rr_norm_max_abs": float(np.max(np.abs(new_norm - old_norm))),
            "rr_counts_max_abs": float(np.max(np.abs(delta))),
            "rr_counts_max_rel": float(np.max(np.abs(delta) / np.maximum(np.abs(old_counts), np.finfo("f8").tiny))),
            "exact": bool(np.array_equal(new_counts, old_counts) and np.array_equal(new_norm, old_norm)),
            "allclose": bool(np.allclose(new_counts, old_counts, rtol=1.0e-12, atol=1.0e-10)),
        }
        if not (bridge["s_edges_exact"] and bridge["mu_edges_exact"] and bridge["allclose"]):
            raise RuntimeError(f"RR bridge failed: {bridge}")

    meta = {
        "status": "done",
        "task": "task43_build_rmax_scan_rr_smu",
        "manifest": str(args.manifest),
        "phase": str(row["phase"]),
        "random_path": str(random_path),
        "fkp_summary": str(args.fkp_summary),
        "weighting": "WEIGHT_TOTAL=WEIGHT*WEIGHT_FKP",
        "p0": float(args.p0),
        "n_random_total": int(z.size),
        "n_random_used": int(weight.size),
        "seed": int(args.seed),
        "seed_policy": "authoritative Task43 RR seed; separate from jaxpower random seed=subsample_seed+2",
        "nthreads": int(args.nthreads),
        "s_edges": s_edges,
        "mu_edges": mu_edges,
        "sumw": float(np.sum(weight)),
        "sumw2": float(np.sum(weight**2)),
        "rr_norm_policy": "authoritative ordered-pair sum(w)^2 convention",
        "rr_norm_scalar": rr_norm_scalar,
        "pycorr_rr_wnorm_no_self": pycorr_rr_wnorm,
        "rr_counts_sum": float(np.sum(rr_counts)),
        "rr_counts_min": float(np.min(rr_counts)),
        "rr_counts_max": float(np.max(rr_counts)),
        "rr_zero_cells": int(np.count_nonzero(rr_counts == 0.0)),
        "rr_rows_with_zero_cells": np.flatnonzero(np.any(rr_counts == 0.0, axis=1)),
        "rr_row_sum_min": float(np.min(np.sum(rr_counts, axis=1))),
        "authoritative_rr_fkp_norm": AUTHORITATIVE_RR_FKP_NORM,
        "rr_fkp_norm_source": (
            "preserved from the accepted Task43 50--350 RR-deconvolved covariance; geometry/FKP selection unchanged"
        ),
        "bridge_to_350": bridge,
    }
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        rr_value=rr_value,
        rr_counts=rr_counts,
        rr_norm=rr_norm,
        s_edges=s_edges,
        s_edge_pairs=np.column_stack([s_edges[:-1], s_edges[1:]]),
        mu_edges=mu_edges,
        s=0.5 * (s_edges[:-1] + s_edges[1:]),
        mu=0.5 * (mu_edges[:-1] + mu_edges[1:]),
        meta_json=np.asarray(json.dumps(jsonable(meta), sort_keys=True)),
    )
    meta["output_npz"] = str(out_npz)
    out_json.write_text(json.dumps(jsonable(meta), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(jsonable(meta), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
