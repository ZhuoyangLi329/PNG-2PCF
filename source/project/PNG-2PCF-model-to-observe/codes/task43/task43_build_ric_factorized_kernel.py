#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构造 Task4.3 正八分体 factorized radial single-term kernel。

代码大纲
========
1. 固定读取与原测量一致的 weighted random 子样本。
2. 利用 Task4.3 random 生成式 ``W(r,Omega)=W_rad(r) W_octant(Omega)`` 的
   精确可分离性，把昂贵且重尾的四点 MC 改写成 radial-bin pair 求和。
3. 用一次 Sobol QMC 只计算正八分体方向夹角 CDF；每个 radial pair 的
   separation distribution 随后由变量变换解析获得。
4. outer RR marginal 仍读取已经完成的 pycorr exact counts，用于消除有限
   radial-bin/mean-radius approximation 对 survey pair distribution 的小误差。
5. cache 只保存足以重建 operator 的低维 ingredients，不保存巨大稠密四点
   tensor；这是论文 W_rad,rad(s,Delta) 在当前可分离几何下的等价压缩表示。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task43_ric_singleterm import (  # noqa: E402
    DEFAULT_FKP_SUMMARY,
    DEFAULT_MANIFEST,
    RIC_CACHE_DIR,
    atomic_savez,
    exact_outer_pair_histogram,
    fkp_effective_normalisation,
    load_fkp_arrays,
    load_random_subsample,
    manifest_row,
    positive_octant_cosine_cdf,
    radial_pair_components,
    radial_statistics,
    resolve_archived_path,
    to_jsonable,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """定义 factorized kernel production 参数。"""
    parser = argparse.ArgumentParser(description="Build factorized Task43 radial single-term kernel.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fkp-summary", type=Path, default=DEFAULT_FKP_SUMMARY)
    parser.add_argument("--phase", type=str, default="ph000")
    parser.add_argument("--p0", type=float, default=10000.0)
    parser.add_argument("--radial-width", type=float, default=2.0)
    parser.add_argument("--nsub", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--outer-counts-from",
        type=Path,
        default=None,
        help="Reuse an existing exact-RR cache; if omitted, compute exact RR for this phase/subsample.",
    )
    parser.add_argument("--kernel-ds", type=float, default=2.0)
    parser.add_argument("--smax", type=float, default=3400.0)
    parser.add_argument("--nthreads", type=int, default=8)
    parser.add_argument("--sobol-power", type=int, default=22)
    parser.add_argument("--cosine-histogram-bins", type=int, default=262144)
    parser.add_argument("--output-dir", type=Path, default=RIC_CACHE_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_stem(args: argparse.Namespace) -> str:
    """构造包含 phase/dchi/nsub/QMC 精度的可追踪文件名。"""
    width = f"{float(args.radial_width):g}".replace(".", "p")
    return (
        f"task43_ric_factorized_{args.phase}_dchi{width}_nsub{int(args.nsub)}_"
        f"sobol2p{int(args.sobol_power)}_ds2_seed{int(args.seed)}"
    )


def main() -> None:
    """生成 radial ingredients、角向 CDF，并复用 exact outer RR 落盘。"""
    args = parse_args()
    if float(args.radial_width) <= 0.0:
        raise ValueError("factorized radial kernel 要求 radial-width>0；global limit 在 operator 中解析测试")
    stem = output_stem(args)
    npz_path = Path(args.output_dir) / f"{stem}.npz"
    json_path = Path(args.output_dir) / f"{stem}.json"
    if npz_path.exists() and json_path.exists() and not args.overwrite:
        print(f"[skip] {npz_path}")
        return

    row = manifest_row(Path(args.manifest), str(args.phase))
    random_path, path_meta = resolve_archived_path(row["random_catalog_path"])
    fkp = load_fkp_arrays(Path(args.fkp_summary))
    random = load_random_subsample(
        random_path,
        fkp_summary=fkp,
        p0=float(args.p0),
        n_subsample=int(args.nsub),
        seed=int(args.seed),
    )
    radial = radial_statistics(random, float(args.radial_width))
    components = radial_pair_components(radial["radial_probability"], radial["radial_mean_chi"])
    angular = positive_octant_cosine_cdf(
        sobol_power=int(args.sobol_power),
        histogram_bins=int(args.cosine_histogram_bins),
        seed=int(args.seed) + 17,
    )
    if args.outer_counts_from is not None:
        with np.load(args.outer_counts_from, allow_pickle=False) as outer:
            outer_edges = np.asarray(outer["separation_edges"], dtype="f8")
            outer_counts = np.asarray(outer["outer_counts"], dtype="f8")
            outer_indices = np.asarray(outer["source_indices"], dtype="i8")
        if outer_indices.shape != random.source_indices.shape or not np.array_equal(outer_indices, random.source_indices):
            raise ValueError("--outer-counts-from 使用的 random source_indices 与当前 nsub/seed 不同")
        outer_meta = {"method": "reused_exact_pycorr", "path": str(args.outer_counts_from)}
    else:
        if int(args.nthreads) > 8:
            raise ValueError("登录节点 exact RR 最多使用 8 threads")
        outer_edges = np.arange(
            0.0,
            float(args.smax) + 0.5 * float(args.kernel_ds),
            float(args.kernel_ds),
            dtype="f8",
        )
        outer_counts, outer_meta = exact_outer_pair_histogram(
            random,
            separation_edges=outer_edges,
            nthreads=int(args.nthreads),
        )
    if not np.allclose(np.diff(outer_edges), 2.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("当前正式 factorized operator 固定复用 ds=2 Mpc/h outer RR")

    meta = {
        "task": "task43_build_ric_factorized_kernel",
        "status": "done",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kernel_kind": "factorized_positive_octant_radial_auto",
        "scientific_scope": {
            "measured_scheme": "data_redshift_resample (shuffled)",
            "modelled_scheme": f"binned radial, Delta chi={float(args.radial_width):g} Mpc/h",
            "approximation": "single IC^(rad,rad) auto term",
            "cross_terms": False,
            "extra_global_sigma_w2": False,
        },
        "factorization": {
            "validity": "Task43 random generator independently samples data-resampled radial z and uniform positive-octant direction",
            "source_code": "codes/task43/task43_build_randoms.py::_build_one_locked",
            "formula": "sum_ab p_ab H_ab(s_outer) H_ab(Delta_inner)",
            "radial_bins": int(radial["radial_probability"].size),
            "radial_pair_components": int(components["component_weight"].size),
            "component_weight_sum": float(np.sum(components["component_weight"])),
            "sobol_power": int(args.sobol_power),
            "sobol_samples": int(np.asarray(angular["sample_size"]).item()),
            "cosine_histogram_bins": int(args.cosine_histogram_bins),
        },
        "inputs": {
            "manifest": str(args.manifest),
            "phase_row": row,
            "random_provenance": path_meta,
            "random_subsample": random.metadata,
            "fkp_summary": str(args.fkp_summary),
            "outer_counts": outer_meta,
        },
        "fkp_fourier_normalisation": fkp_effective_normalisation(fkp, float(args.p0)),
        "paths": {"kernel_npz": str(npz_path), "metadata_json": str(json_path)},
    }
    atomic_savez(
        npz_path,
        separation_edges=outer_edges,
        outer_counts=outer_counts,
        source_indices=np.asarray(random.source_indices, dtype="i8"),
        **radial,
        **components,
        **angular,
        meta_json=np.asarray(json.dumps(to_jsonable(meta), sort_keys=True)),
    )
    write_json(json_path, meta)
    print(
        f"[write] {npz_path} radial_bins={radial['radial_probability'].size} "
        f"components={components['component_weight'].size}",
        flush=True,
    )
    print(f"[write] {json_path}", flush=True)


if __name__ == "__main__":
    main()
