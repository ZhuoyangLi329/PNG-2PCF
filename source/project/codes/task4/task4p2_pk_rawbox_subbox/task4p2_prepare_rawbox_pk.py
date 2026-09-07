#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task4.2 P(k): 标准化 Task4.2 rawbox no-RSD P(k)。

代码大纲：
1. 从 Task47 rawbox 2PCF summary 读取同一批 realization。
2. 读取已有 Task18 no-RSD rawbox P(k) ASCII 文件。
3. 选择指定 observed kmin 且 kmax<=0.08 的 bins；默认 observed kmin 为 2pi/L_box。
4. 写出 mean/cov/mocks 和机器可读 summary，供 Task4.2 P(k) fit 使用。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from task4p2_pk_common import (
    FNL_TAG,
    K_MAX_FIT,
    K_MIN_MODEL,
    OUTPUT_ROOT,
    PARAM_NAMES,
    atomic_savez,
    hartlap_precision,
    load_rawbox_pk_file,
    rawbox_pk_path,
    rawbox_realizations_from_task47,
    select_fit_bins,
    task47_rawbox_npz,
    to_jsonable,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Prepare Task4.2 P(k) rawbox P(k) mean/cov payload.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "rawbox", help="输出目录。")
    parser.add_argument(
        "--fit-kmin-observed",
        type=float,
        default=K_MIN_MODEL,
        help="fit 数据点的 observed kmin；按 kcen >= 此值选 bin，默认 2pi/L_box。",
    )
    parser.add_argument("--strict", action="store_true", help="缺任何 realization 的 P(k) 都报错。")
    return parser.parse_args()


def main() -> None:
    """主入口。"""
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_npz = out_dir / "task4p2_rawbox_fnl100_pk_kmax0p08_payload.npz"
    out_json = out_dir / "task4p2_rawbox_fnl100_pk_kmax0p08_summary.json"

    realizations = rawbox_realizations_from_task47()
    rows = []
    used_realizations = []
    source_files = []
    kcen = kmin = kmax = kavg = nmode = None
    missing = []

    for realization in realizations:
        path = rawbox_pk_path(int(realization))
        if not path.exists():
            missing.append(str(path))
            if args.strict:
                raise FileNotFoundError(path)
            continue
        payload = load_rawbox_pk_file(path)
        if kcen is None:
            kcen = payload["kcen"]
            kmin = payload["kmin"]
            kmax = payload["kmax"]
            kavg = payload["kavg"]
            nmode = payload["nmode"]
        rows.append(payload["pk0"])
        used_realizations.append(int(realization))
        source_files.append(str(path))

    if not rows:
        raise RuntimeError("没有可用 rawbox P(k) 文件")

    assert kcen is not None and kmin is not None and kmax is not None and kavg is not None and nmode is not None
    mocks_all = np.vstack(rows).astype("f8")
    fit_kmin_observed = float(args.fit_kmin_observed)
    fit_indices = np.nonzero((kcen >= fit_kmin_observed - 1.0e-12) & (kmax <= K_MAX_FIT + 1.0e-12))[0].astype("i8")
    if fit_indices.size == 0:
        raise RuntimeError(f"没有 rawbox bins 满足 kcen >= {fit_kmin_observed} 且 kmax <= {K_MAX_FIT}")
    mocks = mocks_all[:, fit_indices]
    mean = np.mean(mocks, axis=0)
    cov = np.cov(mocks, rowvar=False, ddof=1)
    _precision, cov_meta = hartlap_precision(cov, nmock=mocks.shape[0], nparams=len(PARAM_NAMES))

    atomic_savez(
        out_npz,
        kcen=np.asarray(kcen[fit_indices], dtype="f8"),
        kmin=np.asarray(kmin[fit_indices], dtype="f8"),
        kmax=np.asarray(kmax[fit_indices], dtype="f8"),
        kavg=np.asarray(kavg[fit_indices], dtype="f8"),
        nmode=np.asarray(nmode[fit_indices], dtype="f8"),
        pk_mean=mean,
        pk_cov=cov,
        pk_std=np.std(mocks, axis=0, ddof=1),
        pk_mocks=mocks,
        realizations=np.asarray(used_realizations, dtype="i8"),
        source_files=np.asarray(source_files),
        fit_indices=np.asarray(fit_indices, dtype="i8"),
        source_task47_rawbox_npz=np.asarray(str(task47_rawbox_npz())),
        fnl_tag=np.asarray(FNL_TAG),
        fit_kmin_observed=np.asarray(fit_kmin_observed, dtype="f8"),
        rawbox_theory_kmin=np.asarray(K_MIN_MODEL, dtype="f8"),
        rawbox_mode_kfund=np.asarray(K_MIN_MODEL, dtype="f8"),
        kmin_model=np.asarray(K_MIN_MODEL, dtype="f8"),
        kmin_observed=np.asarray(fit_kmin_observed, dtype="f8"),
        parent_kfund=np.asarray(K_MIN_MODEL, dtype="f8"),
    )

    summary = {
        "task": "task4p2_prepare_rawbox_pk",
        "status": "done",
        "fnl_tag": FNL_TAG,
        "source_task47_rawbox_npz": str(task47_rawbox_npz()),
        "output_npz": str(out_npz),
        "n_requested_realizations": int(realizations.size),
        "n_used_realizations": int(len(used_realizations)),
        "missing_count": int(len(missing)),
        "missing": missing[:20],
        "k_policy": {
            "fit_kmin_observed": float(fit_kmin_observed),
            "rawbox_theory_kmin": float(K_MIN_MODEL),
            "rawbox_mode_kfund": float(K_MIN_MODEL),
            "legacy_kmin_model": float(K_MIN_MODEL),
            "legacy_kmin_observed": float(fit_kmin_observed),
            "legacy_parent_kfund": float(K_MIN_MODEL),
            "kmax_fit": float(K_MAX_FIT),
            "selection": "observed bins with kcen >= fit_kmin_observed and kmax <= 0.08; rawbox theory uses discrete parent-box modes inside selected bins",
            "nbin": int(fit_indices.size),
            "kmin_first": float(kmin[fit_indices][0]),
            "kcen_first": float(kcen[fit_indices][0]),
            "kmax_last": float(kmax[fit_indices][-1]),
        },
        "covariance": cov_meta,
        "paths": {
            "payload_npz": str(out_npz),
            "summary_json": str(out_json),
        },
    }
    write_json(out_json, to_jsonable(summary))
    print(f"[write] {out_npz}")
    print(f"[write] {out_json}")


if __name__ == "__main__":
    main()
