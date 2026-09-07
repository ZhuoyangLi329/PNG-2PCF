#!/usr/bin/env python3
"""
简化版 2PCF 脚本（FCFC_2PT_BOX）。

执行逻辑：
1) 在参数区设置输入/输出目录、s-mu bins 和 OMP 线程数
2) 按 realization 循环调用 FCFC_2PT_BOX
"""

import os
from pathlib import Path


# ======================== 参数区（可被环境变量覆盖） ========================
FCFC_BIN = os.getenv("FCFC_BIN", "/global/homes/l/lzy/softdir/FCFC/FCFC_2PT_BOX")

project_root = Path(os.getenv("FASTPM_PROJECT_ROOT", "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm"))
REALIZATION_START = int(os.getenv("FASTPM_REALIZATION_START", "1"))
REALIZATION_END = int(os.getenv("FASTPM_REALIZATION_END", "50"))

INPUT_TEMPLATE = os.getenv(
    "FASTPM_INPUT_TEMPLATE",
    str(project_root / "position" / "pos_RSD_N{realization}.txt"),
)
OUTPUT_DIR = Path(os.getenv("FASTPM_PCF_OUTPUT_DIR", str(project_root / "pcf")))
OUTPUT_BASE_TEMPLATE = os.getenv(
    "FASTPM_PCF_OUTPUT_TEMPLATE",
    str(OUTPUT_DIR / "pcf_rsd_N{realization}"),
)

BOX_SIZE = float(os.getenv("FASTPM_BOX_SIZE", "1000"))
BOX_NORM = int(os.getenv("FASTPM_BOX_NORM", "1"))

S_MIN = float(os.getenv("FASTPM_S_MIN", "50"))
S_MAX = float(os.getenv("FASTPM_S_MAX", "380"))
S_STEP = float(os.getenv("FASTPM_S_STEP", "12"))
MU_NUM = int(os.getenv("FASTPM_MU_NUM", "60"))

RANDOM_SEED_MODE = int(os.getenv("FASTPM_RANDOM_SEED_MODE", "0"))
MULTIPOLE_LIST = os.getenv("FASTPM_MULTIPOLE_LIST", "[0]")
OMP_NUM_THREADS = int(os.getenv("FASTPM_OMP_NUM_THREADS", "32"))
# ======================================================================


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for realization in range(REALIZATION_START, REALIZATION_END + 1):
    data_file = INPUT_TEMPLATE.format(realization=realization)
    out_base = OUTPUT_BASE_TEMPLATE.format(realization=realization)

    if not os.path.exists(data_file):
        print(f"[SKIP] N{realization}: input not found -> {data_file}")
        continue

    cmd = (
        f"OMP_NUM_THREADS={OMP_NUM_THREADS} {FCFC_BIN} "
        f"-i {data_file} "
        f"-l D "
        f"-f '%lf %lf %lf' "
        f"-x '[$1,$2,$3]' "
        f"-b {BOX_SIZE} -B {BOX_NORM} "
        f"-p DD "
        f"-P {out_base}.dat.dd "
        f"-e 'DD/@@-1' "
        f"-E {out_base}.dat.xi2d "
        f"-m '{MULTIPOLE_LIST}' "
        f"-M {out_base}.dat "
        f"--s-min {S_MIN} --s-max {S_MAX} --s-step {S_STEP} "
        f"--mu-num {MU_NUM} -S {RANDOM_SEED_MODE}"
    )
    print(f"[RUN ] N{realization}")
    os.system(cmd)
