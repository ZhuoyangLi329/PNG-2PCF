#!/usr/bin/env python3
"""
简化版功率谱脚本（POWSPEC）。

执行逻辑：
1) 在参数区设置输入/输出目录与 k-bin 参数
2) 按 realization 循环调用 POWSPEC
"""

import os
from pathlib import Path


# ======================== 参数区（可被环境变量覆盖） ========================
POWSPEC_BIN = os.getenv("POWSPEC_BIN", "/global/homes/l/lzy/softdir/powerspec/POWSPEC")

project_root = Path(os.getenv("FASTPM_PROJECT_ROOT", "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm"))
REALIZATION_START = int(os.getenv("FASTPM_REALIZATION_START", "1"))
REALIZATION_END = int(os.getenv("FASTPM_REALIZATION_END", "50"))

INPUT_TEMPLATE = os.getenv(
    "FASTPM_INPUT_TEMPLATE",
    str(project_root / "position" / "pos_RSD_N{realization}.txt"),
)
OUTPUT_DIR = Path(os.getenv("FASTPM_PK_OUTPUT_DIR", str(project_root / "pk")))
OUTPUT_TEMPLATE = os.getenv(
    "FASTPM_PK_OUTPUT_TEMPLATE",
    str(OUTPUT_DIR / "pk_rsd_N{realization}.dat"),
)

BOX_SIZE = float(os.getenv("FASTPM_BOX_SIZE", "1000"))
GRID_SIZE = int(os.getenv("FASTPM_GRID_SIZE", "512"))

KMIN = float(os.getenv("FASTPM_KMIN", "0.002"))
KMAX = float(os.getenv("FASTPM_KMAX", "0.3"))
DK = float(os.getenv("FASTPM_DK", "0.003"))
# ======================================================================


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for realization in range(REALIZATION_START, REALIZATION_END + 1):
    data_file = INPUT_TEMPLATE.format(realization=realization)
    out_file = OUTPUT_TEMPLATE.format(realization=realization)

    if not os.path.exists(data_file):
        print(f"[SKIP] N{realization}: input not found -> {data_file}")
        continue

    cmd = (
        f'{POWSPEC_BIN} '
        f'-d {data_file} '
        f'--data-formatter "%lf %lf %lf" '
        f'-p \'[($1+{BOX_SIZE})%{BOX_SIZE},($2+{BOX_SIZE})%{BOX_SIZE},($3+{BOX_SIZE})%{BOX_SIZE}]\' '
        f'-s T -B {BOX_SIZE} -G {GRID_SIZE} -n 1 -i F -l \'[0]\' '
        f'-k {KMIN} -K {KMAX} -b {DK} '
        f'-a {out_file}'
    )
    print(f"[RUN ] N{realization}")
    os.system(cmd)
