#!/usr/bin/env python3
"""
批量读取 FastPM FOF halo，并导出 RSD 后位置文本。

执行逻辑：
1) 读取参数区（目录、realization 范围、质量阈值等）
2) 对每个 realization 读取 Position/Velocity/Length
3) 计算质量掩码并筛选 halo
4) 用 Header 的 RSDFactor 计算 RSD 后位置
5) 以 float32 txt 输出 pos_RSD_N{realization}.txt
"""

from __future__ import annotations

import os
from pathlib import Path

import bigfile
import numpy as np


# ======================== 参数区（可被环境变量覆盖） ========================
project_root = Path(os.getenv("FASTPM_PROJECT_ROOT", "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm"))
fofs_base_dir = Path(os.getenv("FASTPM_FOFS_BASE_DIR", str(project_root / "fofs")))
output_dir = Path(os.getenv("FASTPM_OUTPUT_DIR", str(project_root / "position")))

realization_start = int(os.getenv("FASTPM_REALIZATION_START", "1"))
realization_end = int(os.getenv("FASTPM_REALIZATION_END", "50"))

snapshot_dir_name = os.getenv("FASTPM_SNAPSHOT_DIR", "fof_0.5000")
linking_length_dir_name = os.getenv("FASTPM_LINKING_LENGTH_DIR", "LL-0.200")

M_PART = float(os.getenv("FASTPM_M_PART", "9.9775e9"))
# 统一默认质量范围到项目基准脚本 position/read_fastpm.py
# 如需临时改动，可通过 FASTPM_MASS_MIN / FASTPM_MASS_MAX 覆盖。
mass_min = float(os.getenv("FASTPM_MASS_MIN", "1.4e13"))
mass_max = float(os.getenv("FASTPM_MASS_MAX", "1e16"))

los_axis = int(os.getenv("FASTPM_LOS_AXIS", "2"))
output_name_template = os.getenv("FASTPM_OUTPUT_TEMPLATE", "pos_RSD_N{realization}.txt")
txt_fmt = os.getenv("FASTPM_TXT_FMT", "%.7e")
# ======================================================================


def load_header_meta(fof_root: Path) -> tuple[float, float]:
    """读取 Header 的 BoxSize 和 RSDFactor。"""
    bf_root = bigfile.BigFile(str(fof_root))
    attrs = bf_root["Header"].attrs
    boxsize = float(np.atleast_1d(attrs["BoxSize"])[0])
    rsd_factor = float(np.atleast_1d(attrs["RSDFactor"])[0])
    return boxsize, rsd_factor


def apply_rsd(pos: np.ndarray, vel: np.ndarray, rsd_factor: float, boxsize: float, axis: int) -> np.ndarray:
    """在 LOS 方向应用 RSD，并做周期回卷。"""
    pos_rsd = pos.copy()
    pos_rsd[:, axis] = np.mod(pos_rsd[:, axis] + vel[:, axis] * rsd_factor, boxsize)
    return pos_rsd


def process_one_realization(realization: int) -> tuple[int, int]:
    """处理一个 realization，返回 (total_halos, kept_halos)。"""
    fof_root = fofs_base_dir / f"fof_N{realization}" / snapshot_dir_name
    halo_path = fof_root / linking_length_dir_name
    if not halo_path.exists():
        raise FileNotFoundError(f"halo path not found: {halo_path}")

    boxsize, rsd_factor = load_header_meta(fof_root)

    bf_halo = bigfile.BigFile(str(halo_path))
    pos = bf_halo["Position"][:]
    vel = bf_halo["Velocity"][:]
    length = bf_halo["Length"][:]

    total_halos = int(pos.shape[0])
    mass = length.astype(np.float64) * M_PART
    mask = (mass >= mass_min) & (mass <= mass_max)
    kept_halos = int(mask.sum())
    if kept_halos == 0:
        raise RuntimeError(f"N{realization}: no halo passed mass cut.")

    pos_sel = pos[mask].astype(np.float32, copy=False)
    vel_sel = vel[mask].astype(np.float32, copy=False)
    pos_rsd = apply_rsd(pos_sel, vel_sel, rsd_factor, boxsize, los_axis).astype(np.float32, copy=False)

    if pos_rsd.ndim != 2 or pos_rsd.shape[1] != 3:
        raise ValueError(f"N{realization}: invalid output shape {pos_rsd.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name_template.format(realization=realization)
    np.savetxt(out_path, pos_rsd, fmt=txt_fmt)
    print(f"N{realization}: total={total_halos}, kept={kept_halos}, saved={out_path}")
    return total_halos, kept_halos


def main() -> None:
    success = 0
    failed = 0
    for realization in range(realization_start, realization_end + 1):
        try:
            process_one_realization(realization)
            success += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] N{realization}: {exc}")
            failed += 1
    print(f"Done. success={success}, failed={failed}")


if __name__ == "__main__":
    main()
