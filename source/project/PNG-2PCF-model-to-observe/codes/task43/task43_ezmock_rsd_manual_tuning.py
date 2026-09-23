#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task43 EZmock 红移空间手动调参脚本。

代码大纲：
1. 读取 AbacusSummit base_c000 的线性功率谱，并初始化 EZmock。
2. 对给定的 rho_c、rho_exp、pdf_base、sigma_v 参数生成固定振幅 RSD mock。
3. 使用 EZmock 官方 rsd_fac 将 z 方向速度映射成平行视线红移空间坐标。
4. 使用 FCFC 周期盒估计 xi0、xi2。
5. 使用仓库现有 jaxpower 周期盒 estimator 估计 P0、P2。
6. 对每组参数的多个 realization 求均值，并可选与 Abacus RSD 目标比较。
7. 每组参数只输出 PDF 以外的机器可读 NPZ/JSON；本脚本本身不生成 PNG。

重要说明：
- 旧手调 notebook 使用 apply_rsd=False，因此旧参数是 real-space 标定结果。
- 本脚本的 sigma_v 是 EZmock 局域随机速度散布参数；rsd_fac 是官方速度到
  z-space 坐标的转换因子，两者物理作用不同，不能混成一个参数。
- 这是 rawbox plane-parallel RSD 调参脚本，先隔离 RSD 模型本身；通过后再把
  参数带入 lightcone 和 survey-window covariance production。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
EZROOT = ROOT / "EZmock/EZmock-1.0.0"
EZBINARY = EZROOT / "EZmock"
LINEAR_PK = ROOT / "outputs/task43_outputs/ezmock_calibration_rawbox_z0p725_mmin1p4e13/linear_pk/abacus_c000_linear_matter_pk_z0p725_desilike_cosmoprimo.dat"
# 历史 EZmock 标定产物实际位于 plots/outputs；优先使用规范 outputs，找不到时回退。
if not LINEAR_PK.is_file():
    LINEAR_PK = ROOT / "plots/outputs/task43_outputs/ezmock_calibration_rawbox_z0p725_mmin1p4e13/linear_pk/abacus_c000_linear_matter_pk_z0p725_desilike_cosmoprimo.dat"
FCFC_BINARY = ROOT / "refcode/FCFC-main/FCFC_2PT_BOX"
OUTPUT_ROOT = ROOT / "outputs/task43_outputs/ezmock_rsd_manual_tuning"

BOX_SIZE = 2000.0
REDSHIFT = 0.725
OMEGA_M = 0.3137721026737606
NGRID = 320
NTRACER = 1_297_050
NREAL_DEFAULT = 3
THREADS_DEFAULT = 8
S_MIN, S_MAX, DS = 30.0, 350.0, 10.0
MU_BINS = 120
KMIN, KMAX, DK = 0.001, 0.3001, 0.002
PK_MESHSIZE = 400

# 默认值来自旧 real-space 标定，只作为 RSD 调参的起点，不是新的 RSD 最优值。
DEFAULT_PARAMS = (1.14, 5.0, 0.25, 0.0)
# 以下开关可由 notebook 覆盖，用于探索不同生成与测量口径。
FIX_AMPLITUDE = True
ATTACH_PARTICLE = True
RAND_GENERATOR = 1
INVERT_PHASE = False
PK_INTERP_LOG = True
BAO_ENHANCE = 0.0


def parse_args() -> argparse.Namespace:
    """解析命令行参数，允许主人快速替换四个 EZmock 经验参数。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rho-c", type=float, default=DEFAULT_PARAMS[0])
    p.add_argument("--rho-exp", type=float, default=DEFAULT_PARAMS[1])
    p.add_argument("--pdf-base", type=float, default=DEFAULT_PARAMS[2])
    p.add_argument("--sigma-v", type=float, default=DEFAULT_PARAMS[3])
    p.add_argument("--nreal", type=int, default=NREAL_DEFAULT)
    p.add_argument("--seed-base", type=int, default=433000)
    p.add_argument("--threads", type=int, default=THREADS_DEFAULT)
    p.add_argument("--ntracer", type=int, default=NTRACER)
    p.add_argument("--target-xi", type=Path, default=None, help="可选 Abacus RSD xi02 NPZ")
    p.add_argument("--target-pk", type=Path, default=None, help="可选 Abacus RSD P02 NPZ")
    p.add_argument("--output-label", default=None)
    return p.parse_args()


def set_affinity(threads: int) -> list[int]:
    """把当前进程限制在指定数量的 CPU 核，避免登录节点超额占用。"""
    if not 1 <= int(threads) <= 8:
        raise ValueError("RSD 手调登录节点线程数必须在 1..8")
    available = sorted(os.sched_getaffinity(0))
    cpus = available[: int(threads)]
    if len(cpus) != int(threads):
        raise RuntimeError(f"可用 CPU 只有 {len(available)} 个，无法申请 {threads} 个")
    os.sched_setaffinity(0, cpus)
    return cpus


def make_edges(vmin: float, vmax: float, step: float) -> np.ndarray:
    """构造 FCFC/jaxpower 使用的等宽区间边界二维数组。"""
    edges = np.arange(vmin, vmax + 0.5 * step, step, dtype="f8")
    if edges[-1] < vmax:
        edges = np.append(edges, vmax)
    edges[0], edges[-1] = vmin, vmax
    return np.column_stack((edges[:-1], edges[1:]))


def rsd_factor() -> float:
    """计算 EZmock 官方示例使用的平行视线速度到 z-space 位移因子。"""
    e_z = np.sqrt(OMEGA_M * (1.0 + REDSHIFT) ** 3 + (1.0 - OMEGA_M))
    return (1.0 + REDSHIFT) / (100.0 * e_z)


def number_tag(value: float) -> str:
    """把浮点参数转成安全的输出目录标签。"""
    return f"{float(value):.8g}".replace("-", "m").replace("+", "").replace(".", "p")


def build_ezmock(seed: int) -> dict[str, int]:
    """准备一个 EZmock 命令行任务描述。

    EZmock 的 Python Cython 扩展在不同 conda 环境中可能没有安装或与当前
    Python ABI 不匹配；仓库原版手调 notebook 使用已经验证的命令行二进制。
    因此这里不直接 import ``EZmock.EZmock``，而是返回 seed，随后由
    :func:`write_rsd_catalog` 写配置并调用同一个官方二进制。
    """
    if not EZBINARY.is_file():
        raise FileNotFoundError(f"EZmock binary not found: {EZBINARY}")
    if not LINEAR_PK.is_file():
        raise FileNotFoundError(f"linear P(k) not found: {LINEAR_PK}")
    return {"seed": int(seed)}


def write_rsd_catalog(ez: dict[str, int], path: Path, params: tuple[float, float, float, float], ntracer: int) -> None:
    """用仓库已跑通的 EZmock CLI 生成速度，再外部应用 plane-parallel RSD。

    EZmock CLI 输出 ``x,y,z,vx,vy,vz`` 六列；本函数按官方示例的
    ``z_rsd = z + rsd_fac * vz`` 只沿 z 方向移动，并周期性 wrap 到 box。
    raw 六列文件、配置和日志保留在同一手调目录，便于核查参数和复现。
    """
    rho_c, rho_exp, pdf_base, sigma_v = params
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = int(ez["seed"])
    raw = path.with_name(path.stem + "_raw6.dat")
    config = path.with_name(path.stem + ".ezmock.conf")
    log = path.with_name(path.stem + ".ezmock.log")
    config.write_text(f"""BOX_SIZE = {BOX_SIZE:.17g}
NUM_GRID = {int(NGRID)}
NUM_TRACER = {int(ntracer)}
LINEAR_PK = '{LINEAR_PK}'
REDSHIFT_PK = {REDSHIFT:.17g}
PK_INTERP_LOG = {'T' if PK_INTERP_LOG else 'F'}
RAND_GENERATOR = {int(RAND_GENERATOR)}
RAND_SEED = {seed}
FIX_AMPLITUDE = {'T' if FIX_AMPLITUDE else 'F'}
INVERT_PHASE = {'T' if INVERT_PHASE else 'F'}
OMEGA_M = {OMEGA_M:.17g}
OMEGA_NU = 0.0014197664745152646
DE_EOS_W = -1
REDSHIFT = {REDSHIFT:.17g}
BAO_ENHANCE = {float(BAO_ENHANCE):.17g}
RHO_CRITICAL = {float(rho_c):.17g}
RHO_EXP = {float(rho_exp):.17g}
PDF_BASE = {float(pdf_base):.17g}
SIGMA_VELOCITY = {float(sigma_v):.17g}
ATTACH_PARTICLE = {'T' if ATTACH_PARTICLE else 'F'}
OUTPUT = '{raw}'
OUTPUT_FORMAT = 0
OUTPUT_HEADER = T
OVERWRITE = 2
VERBOSE = T
""", encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": str(max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))),
        "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "close",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run([str(EZBINARY), "-c", str(config)], check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
    data = np.loadtxt(raw, comments="#", usecols=(0, 1, 2, 3, 4, 5), dtype="f8")
    if data.ndim != 2 or data.shape[1] != 6 or data.shape[0] <= 0 or not np.all(np.isfinite(data)):
        raise RuntimeError(f"invalid EZmock CLI six-column output: {raw} shape={data.shape}")
    position = np.mod(data[:, :3], BOX_SIZE)
    position[:, 2] = np.mod(position[:, 2] + rsd_factor() * data[:, 5], BOX_SIZE)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(f"# EZmock plane-parallel RSD catalog seed={seed} rsd_fac={rsd_factor():.17g}\n")
        np.savetxt(stream, position, fmt="%.8f %.8f %.8f")
    temporary.replace(path)

def measure_fcfc(path: Path, outdir: Path, stem: str, threads: int) -> dict[str, np.ndarray]:
    """用 FCFC 周期盒 estimator 测量 xi(s,mu) 及 xi0/xi2。"""
    config = outdir / f"{stem}.conf"
    smu = outdir / f"{stem}_smu.txt"
    poles = outdir / f"{stem}_xi02.txt"
    pair = outdir / f"{stem}_DD.bin"
    config.write_text(f"""CATALOG = '{path}'
CATALOG_LABEL = D
CATALOG_TYPE = 0
ASCII_SKIP = 1
ASCII_COMMENT = '#'
ASCII_FORMATTER = '%lf %lf %lf'
POSITION = [$1, $2, $3]
BOX_SIZE = {BOX_SIZE:.17g}
DATA_STRUCT = 0
BINNING_SCHEME = 1
PAIR_COUNT = DD
PAIR_COUNT_FILE = '{pair}'
CF_ESTIMATOR = DD / @@ - 1
CF_OUTPUT_FILE = '{smu}'
MULTIPOLE = [0, 2]
MULTIPOLE_FILE = '{poles}'
SEP_BIN_MIN = {S_MIN:.17g}
SEP_BIN_MAX = {S_MAX:.17g}
SEP_BIN_SIZE = {DS:.17g}
MU_BIN_NUM = {MU_BINS}
OUTPUT_FORMAT = 0
OVERWRITE = 1
VERBOSE = T
""", encoding="utf-8")
    env = os.environ.copy(); env.update({"OMP_NUM_THREADS": str(int(threads)), "OMP_DYNAMIC": "FALSE"})
    log = outdir / f"{stem}.log"
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run([str(FCFC_BINARY), "-c", str(config)], check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
    table = np.loadtxt(poles, comments="#", dtype="f8")
    if table.ndim == 1: table = table[None, :]
    if table.shape != (int((S_MAX - S_MIN) / DS), 5):
        raise RuntimeError(f"FCFC multipole shape 错误: {table.shape}")
    return {"s": table[:, 0], "s_edges": np.concatenate(([table[0, 1]], table[:, 2])), "xi0": table[:, 3], "xi2": table[:, 4]}


def measure_jaxpower(path: Path, seed: int, outdir: Path) -> dict[str, np.ndarray]:
    """在独立 helper 进程中用仓库 jaxpower 测量 RSD P0/P2。

    独立进程沿用原版 notebook 的启动方式，避免 notebook 当前 kernel 的
    用户 site-packages 污染 mpi4py；helper 返回完整 P0/P2 数组供 notebook 画图。
    """
    helper = ROOT / "codes/task43/task43_measure_ezmock_rsd_p02_jaxpower.py"
    output = outdir / f"pk02_seed{int(seed)}_mesh{int(PK_MESHSIZE)}.npz"
    meta = output.with_suffix(".json")
    outdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"JAX_PLATFORMS": "cpu", "JAX_PLATFORM_NAME": "cpu", "CUDA_VISIBLE_DEVICES": ""})
    python_override = env.get("TASK43_JAX_PYTHON")
    command = [python_override or "python", "-u", str(helper),
               "--catalog", str(path), "--output", str(output),
               "--boxsize", str(BOX_SIZE), "--meshsize", str(PK_MESHSIZE),
               "--kmin", str(KMIN), "--kmax", str(KMAX), "--dk", str(DK),
               "--threads", str(max(1, int(env.get("OMP_NUM_THREADS", "1")))),
               "--seed", str(int(seed))]
    if not output.is_file() or not meta.is_file():
        if python_override:
            runner = command
        else:
            # A notebook kernel may be desilike (Python 3.11), while the
            # clustering_statistics/lsstypes stack is supplied by NERSC's
            # cosmodesi environment.  Source it inside the child shell.
            env_script = Path("/global/common/software/desi/users/adematti/cosmodesi_environment.sh")
            if not env_script.is_file():
                runner = [sys.executable, *command[1:]]
            else:
                shell = (
                    f"unset PYTHONHOME PYTHONPATH; "
                    f"source {shlex.quote(str(env_script))} main; "
                    f"exec {shlex.join(command)}"
                )
                runner = ["bash", "-lc", shell]
        subprocess.run(runner, cwd=str(ROOT), check=True, env=env)
    with np.load(output, allow_pickle=False) as data:
        result = {key: np.asarray(data[key]) for key in data.files}
    for ell in (0, 2):
        values = np.asarray(result[f"pk{ell}"], dtype="f8")
        modes = np.asarray(result[f"nmodes{ell}"], dtype="f8")
        valid = (modes > 0.0) & np.isfinite(values)
        if values.ndim != 1 or not np.any(valid):
            raise RuntimeError(f"jaxpower P{ell} has no valid modes: {output}")
        # 空 bin 的 NaN 是 jaxpower 的正常输出；保留它们，绘图时自动跳过。
        values = values.copy(); values[~valid] = np.nan
        result[f"pk{ell}"] = values
        result[f"valid{ell}"] = valid
    result["k0"] = result["k0"].astype("f8")
    result["k2"] = result["k2"].astype("f8")
    return result

def load_target(path: Path | None) -> dict[str, np.ndarray] | None:
    """读取可选目标 NPZ，兼容仓库常见的 mean、all 和 rsd 命名。"""
    if path is None: return None
    with np.load(path, allow_pickle=False) as data:
        result = {key: np.asarray(data[key]) for key in data.files}
    return result


def pick_target_array(target: dict[str, np.ndarray] | None, names: tuple[str, ...]) -> np.ndarray | None:
    """从 Abacus 目标 NPZ 中按兼容名称取出目标数组。"""
    if target is None:
        return None
    for name in names:
        if name in target:
            return np.asarray(target[name], dtype="f8")
    return None


def compare_to_targets(xi_stack: np.ndarray, pk0_stack: np.ndarray, pk2_stack: np.ndarray, target_xi: dict[str, np.ndarray] | None, target_pk: dict[str, np.ndarray] | None) -> dict[str, float]:
    """计算 EZmock 均值相对 Abacus RSD 目标的简单 RMS 诊断。"""
    metrics: dict[str, float] = {}
    xi_mean = xi_stack.mean(axis=0); pk0_mean = pk0_stack.mean(axis=0); pk2_mean = pk2_stack.mean(axis=0)
    for label, value, names in (("xi0", xi_mean[:, 0], ("xi0_rsd_mean", "xi0_rsd", "xi0_mean")), ("xi2", xi_mean[:, 1], ("xi2_rsd_mean", "xi2_rsd", "xi2_mean"))):
        ref = pick_target_array(target_xi, names)
        if ref is not None and ref.shape == value.shape:
            metrics[f"{label}_mean_absolute_rms"] = float(np.sqrt(np.mean((value - ref) ** 2)))
    for label, value, names in (("pk0", pk0_mean, ("pk0_rsd_mean", "pk0_rsd", "pk0_mean", "pk0")), ("pk2", pk2_mean, ("pk2_rsd_mean", "pk2_rsd", "pk2_mean", "pk2"))):
        ref = pick_target_array(target_pk, names)
        if ref is not None and ref.shape == value.shape:
            metrics[f"{label}_mean_absolute_rms"] = float(np.sqrt(np.mean((value - ref) ** 2)))
    return metrics


def main() -> None:
    """执行多 realization RSD 参数试验并写出汇总结果。"""
    args = parse_args()
    if args.nreal < 1 or args.nreal > 25: raise ValueError("nreal 必须在 1..25")
    cpus = set_affinity(args.threads)
    if not EZBINARY.exists() or not LINEAR_PK.exists() or not FCFC_BINARY.exists():
        raise FileNotFoundError("缺少 EZmock、linear P(k) 或 FCFC 可执行文件")
    params = (args.rho_c, args.rho_exp, args.pdf_base, args.sigma_v)
    label = args.output_label or f"rsd_c{number_tag(args.rho_c)}_e{number_tag(args.rho_exp)}_b{number_tag(args.pdf_base)}_v{number_tag(args.sigma_v)}_n{args.nreal}"
    root = OUTPUT_ROOT / label; catdir, measdir = root / "catalogs", root / "measurements"; catdir.mkdir(parents=True, exist_ok=True); measdir.mkdir(parents=True, exist_ok=True)
    xi_rows, pk0_rows, pk2_rows, seeds = [], [], [], []
    started = time.perf_counter()
    for i in range(args.nreal):
        seed = int(args.seed_base) + i + 1; seeds.append(seed)
        catalog = catdir / f"ezmock_rsd_seed{seed}.dat"
        ez = build_ezmock(seed)
        write_rsd_catalog(ez, catalog, params, args.ntracer)
        xi = measure_fcfc(catalog, measdir, f"fcfc_seed{seed}", args.threads)
        pk = measure_jaxpower(catalog, seed, measdir)
        xi_rows.append(np.column_stack((xi["xi0"], xi["xi2"])))
        pk0_rows.append(pk["pk0"]); pk2_rows.append(pk["pk2"])
        print(f"[done] realization={i+1}/{args.nreal} seed={seed}", flush=True)
    xi_stack = np.asarray(xi_rows); pk0_stack = np.asarray(pk0_rows); pk2_stack = np.asarray(pk2_rows)
    target_xi = load_target(args.target_xi); target_pk = load_target(args.target_pk)
    target_metrics = compare_to_targets(xi_stack, pk0_stack, pk2_stack, target_xi, target_pk)
    summary_npz = root / f"{label}_summary.npz"; summary_json = summary_npz.with_suffix(".json")
    np.savez_compressed(summary_npz, seeds=np.asarray(seeds), s=xi["s"], xi0_all=xi_stack[:,0], xi2_all=xi_stack[:,1], xi0_mean=xi_stack[:,0].mean(0), xi2_mean=xi_stack[:,1].mean(0), k=pk["k0"], pk0_all=pk0_stack, pk2_all=pk2_stack, pk0_mean=pk0_stack.mean(0), pk2_mean=pk2_stack.mean(0), params=np.asarray(params), rsd_fac=np.asarray(rsd_factor()))
    payload = {"task":"task43_ezmock_rsd_manual_tuning", "status":"done", "classification":"fixed_amplitude_manual_rsd_calibration", "space":"plane_parallel_redshift_space", "fix_amplitude":True, "attach_particle":True, "nreal":args.nreal, "seeds":seeds, "parameters":{"rho_c":args.rho_c,"rho_exp":args.rho_exp,"pdf_base":args.pdf_base,"sigma_v":args.sigma_v}, "generation_options":{"fix_amplitude":FIX_AMPLITUDE,"attach_particle":ATTACH_PARTICLE,"rand_generator":RAND_GENERATOR,"invert_phase":INVERT_PHASE,"pk_interp_log":PK_INTERP_LOG,"bao_enhance":BAO_ENHANCE}, "rsd_fac":rsd_factor(), "redshift":REDSHIFT, "boxsize":BOX_SIZE, "ntracer":args.ntracer, "xi_grid":{"smin":S_MIN,"smax":S_MAX,"ds":DS,"nmu":MU_BINS}, "pk_grid":{"kmin":KMIN,"kmax":KMAX,"dk":DK,"meshsize":PK_MESHSIZE}, "cpu_affinity":cpus, "elapsed_sec":time.perf_counter()-started, "summary_npz":str(summary_npz), "target_xi":str(args.target_xi) if args.target_xi else None, "target_pk":str(args.target_pk) if args.target_pk else None, "target_metrics":target_metrics}
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[summary] {summary_npz}", flush=True)


if __name__ == "__main__":
    main()