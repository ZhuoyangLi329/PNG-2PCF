#!/usr/bin/env python3
"""
批量计算 cov_mock 目录下 EZmock RSD 位置文件的功率谱（POWSPEC）。

执行逻辑：
1) 扫描 /pscratch/sd/l/lzy/cov_mock 下的 seed 目录；
2) 对每个目录读取 EZmock_*_RSD.dat（位置文件）；
3) 输出同目录 PK_EZmock_*_RSD.dat；
4) 支持 Slurm array 按任务号切片并行。

说明：
- 只处理 seed=10*i+5 的目录；
- 缺失位置文件会自动跳过；
- 输出覆盖由 OVERWRITE 控制（默认覆盖，便于重算）。
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


# ======================== 参数区（可用环境变量覆盖） ========================
POWSPEC_BIN = Path(os.getenv("POWSPEC_BIN", "/global/homes/l/lzy/softdir/powerspec/POWSPEC"))

BASE_DIR = Path(os.getenv("COVMOCK_BASE_DIR", "/pscratch/sd/l/lzy/cov_mock"))
DIR_GLOB = os.getenv("COVMOCK_DIR_GLOB", "B3000G768Z0N4417983_b0.18d5r270c1.65_seed*")

BOX_SIZE = int(os.getenv("COVMOCK_BOX_SIZE", "3000"))
GRID_SIZE = int(os.getenv("COVMOCK_GRID_SIZE", "512"))

KMIN = float(os.getenv("COVMOCK_KMIN", "0.002"))
KMAX = float(os.getenv("COVMOCK_KMAX", "0.3"))
DK = float(os.getenv("COVMOCK_DK", "0.003"))

OMP_NUM_THREADS = int(os.getenv("OMP_NUM_THREADS", "4"))
OVERWRITE = os.getenv("COVMOCK_OVERWRITE", "1") == "1"

PARTICLE_ASSIGN = 1
INTERLACE = "F"
MULTIPOLE_LIST = "[0]"
# ========================================================================


def seed_from_dirname(path: Path) -> int:
    m = re.search(r"seed(\d+)$", path.name)
    return int(m.group(1)) if m else -1


def is_seed_10i_plus_5(seed: int) -> bool:
    return seed >= 5 and (seed - 5) % 10 == 0


def split_for_array(items: list[Path]) -> list[Path]:
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    task_count = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
    if task_count <= 1:
        return items
    return [p for i, p in enumerate(items) if i % task_count == task_id]


def run_one(data_file: Path, out_file: Path) -> str:
    if out_file.exists() and (not OVERWRITE):
        print(f"[SKIP] exists -> {out_file}")
        return "skip"

    cmd = [
        str(POWSPEC_BIN),
        "-d", str(data_file),
        "--data-formatter", "%lf %lf %lf",
        "-p", f"[($1+{BOX_SIZE})%{BOX_SIZE},($2+{BOX_SIZE})%{BOX_SIZE},($3+{BOX_SIZE})%{BOX_SIZE}]",
        "-s", "T",
        "-B", str(BOX_SIZE),
        "-G", str(GRID_SIZE),
        "-n", str(PARTICLE_ASSIGN),
        "-i", INTERLACE,
        "-l", MULTIPOLE_LIST,
        "-k", str(KMIN),
        "-K", str(KMAX),
        "-b", str(DK),
        "-w", "1",
        "-a", str(out_file),
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)

    subprocess.run(cmd, check=True, env=env)
    return "ok"


def main() -> None:
    if not POWSPEC_BIN.exists():
        raise FileNotFoundError(f"POWSPEC binary not found: {POWSPEC_BIN}")

    dirs = sorted(BASE_DIR.glob(DIR_GLOB), key=lambda p: seed_from_dirname(p))
    dirs = [d for d in dirs if d.is_dir()]
    dirs = [d for d in dirs if is_seed_10i_plus_5(seed_from_dirname(d))]

    if not dirs:
        raise RuntimeError(f"No seed dirs found under {BASE_DIR} with glob={DIR_GLOB}")

    my_dirs = split_for_array(dirs)
    print(f"[INFO] total dirs={len(dirs)}, this task handles={len(my_dirs)}")
    print(f"[INFO] settings: B={BOX_SIZE}, G={GRID_SIZE}, k=[{KMIN},{KMAX}], dk={DK}, overwrite={OVERWRITE}")

    ok = skip = fail = noinp = 0

    for d in my_dirs:
        seed = seed_from_dirname(d)
        in_file = d / f"EZmock_B3000G768Z0N4417983_b0.18d5r270c1.65_seed{seed}_RSD.dat"
        out_file = d / f"PK_EZmock_B3000G768Z0N4417983_b0.18d5r270c1.65_seed{seed}_RSD.dat"

        if not in_file.exists():
            print(f"[SKIP] missing input: {in_file}")
            noinp += 1
            continue

        try:
            print(f"[RUN ] seed={seed}")
            st = run_one(in_file, out_file)
            if st == "ok":
                ok += 1
            else:
                skip += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[ERR ] seed={seed}: {exc}")

    print("\n=== SUMMARY ===")
    print(f"ok={ok}")
    print(f"skip={skip}")
    print(f"missing_input={noinp}")
    print(f"fail={fail}")


if __name__ == "__main__":
    main()
