#!/usr/bin/env python3
"""用 FCFC weighted Landy--Szalay 测量 Task43 Abacus/EZmock lightcone 2PCF。

执行大纲
--------
1. 从给定 manifest 选择一个 data/random catalog，并加载冻结的 Abacus
   FKP nbar(z) 表，计算 P0=10000 的 WEIGHT_TOTAL。
2. 通过两个临时 FIFO 把 ``x y z weight`` 流式交给 FCFC，避免为每个
   8-million-row random 永久保存数百 MB 的 ASCII 副本。
3. 用非周期 survey ``FCFC_2PT`` 计算 DD/DR/RR 和 weighted
   Landy--Szalay，固定 s_edges=50..550、ds=10 Mpc/h。
4. 保存 xi、归一化 pair counts、权重/zeff/资源/provenance 到 NPZ+JSON。

输入 catalog 本身不会被修改；FIFO 在每次测量结束后自动清理。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from task43_ezmock_lightcone_common import (
    ABACUS_FKP_SUMMARY,
    ABACUS_RMAX_MANIFEST,
    FCFC_BINARY,
    FCFC_CONFIG_DIR,
    FCFC_PAIR_DIR,
    LOG_DIR,
    MANIFEST,
    OUTPUT_ROOT,
    P0,
    S_EDGES,
    XI_DIR,
    atomic_savez,
    read_jsonl,
    write_json,
)
from task43_fkp_zeff import (
    compute_zeff_diagnostics,
    get_fiducial_cosmology,
    load_fkp_summary,
    total_weight_from_summary,
)
from task43_rawbox_ezmock_common import set_cpu_affinity


ABACUS_XI_DIR = OUTPUT_ROOT / "abacus_reference" / "xi_fcfc"
ABACUS_CONFIG_DIR = OUTPUT_ROOT / "abacus_reference" / "fcfc_configs"
ABACUS_PAIR_DIR = OUTPUT_ROOT / "abacus_reference" / "fcfc_pairs"
ABACUS_LOG_DIR = OUTPUT_ROOT / "abacus_reference" / "logs"
TMP_ROOT = OUTPUT_ROOT / "tmp" / "fcfc_fifo"


def parse_args() -> argparse.Namespace:
    """解析样本、manifest 行、线程数和可选 debug subsampling。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=("ezmock", "abacus"), required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--p0", type=float, default=P0)
    parser.add_argument("--fkp-summary", type=Path, default=ABACUS_FKP_SUMMARY)
    parser.add_argument("--stream-chunk", type=int, default=200_000)
    parser.add_argument("--max-data", type=int, default=None, help="仅用于 parser/debug smoke。")
    parser.add_argument("--max-random", type=int, default=None, help="仅用于 parser/debug smoke。")
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def phase_token(row: dict[str, Any], *, sample: str) -> str:
    """构造不会互相覆盖的输出 stem。"""
    phase = str(row["phase"])
    if sample == "ezmock":
        return f"ezmock_{phase}_seed{int(row['seed'])}"
    return f"AbacusSummit_base_c000_{phase}"


def output_paths(
    row: dict[str, Any],
    *,
    sample: str,
    output_tag: str | None,
) -> dict[str, Path]:
    """返回 FCFC config/pairs/log/xi 的完整路径。"""
    token = phase_token(row, sample=sample)
    suffix = f"_{output_tag}" if output_tag else ""
    if sample == "ezmock":
        xi = Path(row["xi_path"]) if output_tag is None else XI_DIR / f"xi0_{token}{suffix}.npz"
        config_dir, pair_dir, log_dir = FCFC_CONFIG_DIR, FCFC_PAIR_DIR, LOG_DIR
    else:
        xi = ABACUS_XI_DIR / f"xi0_{token}_z0p6_0p8_mmin1p4e13_x25_s50_550_ds10_fcfc{suffix}.npz"
        config_dir, pair_dir, log_dir = ABACUS_CONFIG_DIR, ABACUS_PAIR_DIR, ABACUS_LOG_DIR
    return {
        "xi_npz": xi,
        "xi_json": xi.with_suffix(".json"),
        "xi_text": xi.with_suffix(".txt"),
        "config": config_dir / f"fcfc_{token}{suffix}.conf",
        "dd": pair_dir / f"DD_{token}{suffix}.txt",
        "dr": pair_dir / f"DR_{token}{suffix}.txt",
        "rr": pair_dir / f"RR_{token}{suffix}.txt",
        "log": log_dir / f"fcfc_{token}{suffix}.log",
    }


def load_catalog_arrays(
    path: Path,
    *,
    fkp_summary_path: Path,
    p0: float,
    max_rows: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取位置、红移和 WEIGHT_TOTAL；可选截取只允许 debug 使用。"""
    with np.load(path, allow_pickle=False) as data:
        n_total = int(np.asarray(data["Z"]).size)
        stop = n_total if max_rows is None else min(n_total, int(max_rows))
        choice = slice(0, stop)
        position = np.column_stack(
            [
                np.asarray(data["X"][choice], dtype="f8"),
                np.asarray(data["Y"][choice], dtype="f8"),
                np.asarray(data["Zcart"][choice], dtype="f8"),
            ]
        )
        redshift = np.asarray(data["Z"][choice], dtype="f8")
        base = (
            np.asarray(data["WEIGHT"][choice], dtype="f8")
            if "WEIGHT" in data.files
            else np.ones(stop, dtype="f8")
        )
    fkp_summary = load_fkp_summary(fkp_summary_path)
    weight = total_weight_from_summary(redshift, base, fkp_summary, p0=float(p0))
    return position, redshift, weight


def fifo_writer(
    fifo_path: str,
    catalog_path: str,
    fkp_summary_path: str,
    p0: float,
    max_rows: int | None,
    chunk_size: int,
) -> None:
    """子进程：把 NPZ catalog 按块写成 FCFC 所需四列 ASCII 流。"""
    try:
        position, _, weight = load_catalog_arrays(
            Path(catalog_path),
            fkp_summary_path=Path(fkp_summary_path),
            p0=float(p0),
            max_rows=max_rows,
        )
        with Path(fifo_path).open("w", encoding="utf-8", buffering=1024 * 1024) as stream:
            for start in range(0, position.shape[0], int(chunk_size)):
                stop = min(start + int(chunk_size), position.shape[0])
                block = np.column_stack([position[start:stop], weight[start:stop]])
                np.savetxt(stream, block, fmt="%.9g %.9g %.9g %.12g")
    except BrokenPipeError:
        # 主 FCFC 进程失败时 FIFO 读端会提前关闭，子进程无需再写 traceback。
        return


def fcfc_config(fifo_data: Path, fifo_random: Path, paths: dict[str, Path], *, overwrite: bool) -> str:
    """构造 survey FCFC weighted Landy--Szalay 配置。"""
    return f"""CATALOG = ['{fifo_data}', '{fifo_random}']
CATALOG_LABEL = [D, R]
CATALOG_TYPE = [0, 0]
ASCII_SKIP = [0, 0]
ASCII_COMMENT = ['#', '#']
ASCII_FORMATTER = ['%lf %lf %lf %lf', '%lf %lf %lf %lf']
POSITION = [$1, $2, $3, $1, $2, $3]
WEIGHT = [$4, $4]
COORD_CONVERT = F
DATA_STRUCT = 0
BINNING_SCHEME = 0
PAIR_COUNT = [DD, DR, RR]
PAIR_COUNT_FILE = ['{paths['dd']}', '{paths['dr']}', '{paths['rr']}']
CF_ESTIMATOR = (DD - 2 * DR + RR) / RR
CF_OUTPUT_FILE = '{paths['xi_text']}'
SEP_BIN_MIN = {S_EDGES[0]:.17g}
SEP_BIN_MAX = {S_EDGES[-1]:.17g}
SEP_BIN_SIZE = {np.median(np.diff(S_EDGES)):.17g}
OUTPUT_FORMAT = 1
OVERWRITE = {2 if overwrite else 1}
VERBOSE = T
"""


def load_fcfc_table(path: Path, *, expected_columns: int) -> np.ndarray:
    """读取并验证 FCFC ASCII 结果。"""
    table = np.loadtxt(path, comments="#", dtype="f8")
    if table.shape != (S_EDGES.size - 1, expected_columns):
        raise ValueError(f"unexpected FCFC table shape {table.shape} in {path}")
    if not np.all(np.isfinite(table)):
        raise ValueError(f"non-finite FCFC output in {path}")
    return table


def measure_one(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """流式执行一个 realization 的 FCFC 测量并保存结构化结果。"""
    paths = output_paths(row, sample=str(args.sample), output_tag=args.output_tag)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    if paths["xi_npz"].exists() and paths["xi_json"].exists() and not args.overwrite:
        print(f"[skip] {paths['xi_npz']}")
        return json.loads(paths["xi_json"].read_text(encoding="utf-8"))

    data_path = Path(row["halo_catalog_path"])
    random_path = Path(row["random_catalog_path"])
    for path in (data_path, random_path, args.fkp_summary, FCFC_BINARY):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    started = time.perf_counter()
    # 先独立计算权重与 zeff 元数据；正式 pair counts 仍通过 FIFO 读取同一函数。
    data_pos, data_z, data_w = load_catalog_arrays(
        data_path,
        fkp_summary_path=args.fkp_summary,
        p0=float(args.p0),
        max_rows=args.max_data,
    )
    random_pos, random_z, random_w = load_catalog_arrays(
        random_path,
        fkp_summary_path=args.fkp_summary,
        p0=float(args.p0),
        max_rows=args.max_random,
    )
    zeff = compute_zeff_diagnostics(
        data_z,
        data_w,
        random_z,
        random_w,
        cosmo=get_fiducial_cosmology("DESI"),
        zrange=(float(row["zmin"]), float(row["zmax"])),
    )
    ndata, nrandom = int(data_pos.shape[0]), int(random_pos.shape[0])
    del data_pos, random_pos, data_z, random_z, data_w, random_w

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{args.sample}_{row['phase']}_", dir=TMP_ROOT) as tmp_name:
        tmp = Path(tmp_name)
        fifo_data, fifo_random = tmp / "data.pipe", tmp / "random.pipe"
        os.mkfifo(fifo_data)
        os.mkfifo(fifo_random)
        paths["config"].write_text(
            fcfc_config(fifo_data, fifo_random, paths, overwrite=bool(args.overwrite)),
            encoding="utf-8",
        )
        # FCFC 在 OVERWRITE=1 时会直接读取已有 pair-count 文件，并且不会
        # 打开这些 pair 所不需要的 catalog。恢复中断任务时必须只启动真正
        # 会被读取的 FIFO writer，否则无人打开的 writer 会阻塞并被误判失败。
        pair_cached_before = {
            key: bool(paths[key].is_file() and not args.overwrite) for key in ("dd", "dr", "rr")
        }
        need_data_fifo = not pair_cached_before["dd"] or not pair_cached_before["dr"]
        need_random_fifo = not pair_cached_before["dr"] or not pair_cached_before["rr"]

        # ``spawn`` 避免 cosmoprimo/JAX 已初始化线程后再 fork 的死锁风险。
        context = mp.get_context("spawn")
        writers: list[mp.Process] = []
        writer_labels: list[str] = []
        if need_data_fifo:
            writers.append(context.Process(
                target=fifo_writer,
                args=(
                    str(fifo_data),
                    str(data_path),
                    str(args.fkp_summary),
                    float(args.p0),
                    args.max_data,
                    int(args.stream_chunk),
                ),
            ))
            writer_labels.append("data")
        if need_random_fifo:
            writers.append(context.Process(
                target=fifo_writer,
                args=(
                    str(fifo_random),
                    str(random_path),
                    str(args.fkp_summary),
                    float(args.p0),
                    args.max_random,
                    int(args.stream_chunk),
                ),
            ))
            writer_labels.append("random")
        for process in writers:
            process.start()
        fcfc_started = time.perf_counter()
        env = os.environ.copy()
        env.update(
            {
                "OMP_NUM_THREADS": str(args.threads),
                "OMP_DYNAMIC": "FALSE",
                "OMP_PROC_BIND": "close",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        try:
            with paths["log"].open("w", encoding="utf-8") as log:
                subprocess.run(
                    [str(FCFC_BINARY), "-c", str(paths["config"])],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
        finally:
            for process in writers:
                process.join(timeout=60)
                if process.is_alive():
                    process.terminate()
                    process.join()
        if any(process.exitcode not in (0, None) for process in writers):
            raise RuntimeError(
                f"FIFO writer failed: {dict(zip(writer_labels, [process.exitcode for process in writers], strict=True))}"
            )
        fcfc_sec = time.perf_counter() - fcfc_started

    xi_table = load_fcfc_table(paths["xi_text"], expected_columns=4)
    pair_tables = {
        key: load_fcfc_table(paths[key], expected_columns=3) for key in ("dd", "dr", "rr")
    }
    s, slow, shigh, xi0 = xi_table.T
    s_edges = np.concatenate([slow[:1], shigh])
    if not np.array_equal(s_edges, S_EDGES):
        raise ValueError(f"FCFC s_edges mismatch: {s_edges}")
    dd, dr, rr = (pair_tables[key][:, 2] for key in ("dd", "dr", "rr"))
    if not np.all(rr > 0.0):
        raise RuntimeError("FCFC returned non-positive normalized RR")

    debug_subsample = args.max_data is not None or args.max_random is not None
    atomic_savez(
        paths["xi_npz"],
        s=s,
        s_edges=s_edges,
        xi0=xi0,
        DD=dd,
        DR=dr,
        RR=rr,
        ndata=np.asarray(ndata, dtype="i8"),
        nrandom=np.asarray(nrandom, dtype="i8"),
        phase=np.asarray(row["phase"]),
        sample=np.asarray(args.sample),
        p0=np.asarray(float(args.p0), dtype="f8"),
        zeff=np.asarray(float(zeff["zeff_random_auto"]), dtype="f8"),
        data_weight_min=np.asarray(float(zeff.get("data_weight_min", np.nan)), dtype="f8"),
        random_weight_min=np.asarray(float(zeff.get("random_weight_min", np.nan)), dtype="f8"),
        debug_subsample=np.asarray(debug_subsample),
    )
    summary = {
        "task": "task43_measure_lightcone_xi_fcfc",
        "status": "done",
        "sample": str(args.sample),
        "phase": str(row["phase"]),
        "seed": None if "seed" not in row else int(row["seed"]),
        "classification": "debug_subsample" if debug_subsample else "full_input_validation_measurement",
        "catalogs": {"data": str(data_path), "random": str(random_path)},
        "fkp_summary": str(args.fkp_summary),
        "p0": float(args.p0),
        "weight_policy": "WEIGHT_TOTAL=WEIGHT*WEIGHT_FKP from frozen Abacus nbar(z)",
        "ndata": ndata,
        "nrandom": nrandom,
        "estimator": "FCFC_2PT weighted Landy-Szalay (DD-2DR+RR)/RR",
        "geometry": "aperiodic positive-octant radial shell",
        "s_edges": s_edges,
        "zeff": zeff,
        "fcfc": {
            "binary": str(FCFC_BINARY),
            "config": str(paths["config"]),
            "pair_count_files": {key.upper(): str(paths[key]) for key in ("dd", "dr", "rr")},
            "ascii_transport": (
                "pair-count cache only; no catalog FIFO opened"
                if not writer_labels
                else f"temporary FIFO for {writer_labels}; no persistent expanded catalog"
            ),
            "pair_count_cache_before": {key.upper(): value for key, value in pair_cached_before.items()},
            "threads": int(args.threads),
        },
        "output": {"npz": str(paths["xi_npz"]), "json": str(paths["xi_json"]), "text": str(paths["xi_text"])},
        "runtime": {"fcfc_and_stream_sec": float(fcfc_sec), "total_sec": float(time.perf_counter() - started)},
    }
    write_json(paths["xi_json"], summary)
    print(
        f"[done] sample={args.sample} phase={row['phase']} ndata={ndata} "
        f"nrandom={nrandom} elapsed={summary['runtime']['total_sec']:.1f}s"
    )
    return summary


def main() -> None:
    """选择 manifest 行、限制 CPU affinity 并测量 2PCF。"""
    args = parse_args()
    cpus = set_cpu_affinity(int(args.threads))
    manifest = args.manifest
    if manifest is None:
        manifest = MANIFEST if args.sample == "ezmock" else ABACUS_RMAX_MANIFEST
    rows = read_jsonl(manifest)
    if not 0 <= int(args.index) < len(rows):
        raise IndexError(f"index {args.index} outside manifest length {len(rows)}")
    print(f"[cpu] affinity={cpus}")
    print(f"[manifest] {manifest}")
    measure_one(rows[int(args.index)], args)


if __name__ == "__main__":
    main()
