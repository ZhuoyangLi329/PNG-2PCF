#!/usr/bin/env python3
"""
Mission 6: Quijote N-body 样本上的参数化窗口验证（重做版）。

========================
代码大纲（执行逻辑关系）
========================
1. 组织原始输入数据
   - 原始目录：/pscratch/sd/l/lzy/pks_2pcfs
   - 已有真实测量的三个 fnl 节点：0, 50, 100
   - 对每个 realization、每个 k/r bin、每一列数据，在 fnl 维度上做三节点 cubic spline 插值，
     生成目标 fnl=20, 30, 75 的 pk / 2PCF 文件。

2. 读取某个 fnl 的 500 个 realization，构建均值与协方差样本
   - pk：读取前 N 个 k-bin，构造 P0(k) 的均值和 realization 样本矩阵；
   - pcf：读取 xi0(r) 的 realization，构造均值和标准差；
   - covariance 口径按任务书要求，直接使用“当前 fnl 自己的 500 个 realization”。

3. 调用 desilike 对平均功率谱做 best-fit
   - 理论模型：PNGTracerPowerSpectrumMultipoles
   - 固定参数：p=1.2，sn0=0；sigmas 保持可拟合
   - observable 的 data 用当前 fnl 的平均 P0
   - observable 的 covariance 用当前 fnl 的 realization 列表

4. 用 best-fit P0(k) 经过 FFTLog 计算 xi0(r)
   - baseline：kmin = k_f = 2pi/L，只保留数值 taper
   - window：使用 mission5 当前最佳参数化窗口
     W(k) = [1 - exp(-(k/k_f)^x)] / [1 - exp(-1)]  (k < k_f), W=1 (k >= k_f)
   - 盒长固定为 1Gpc，因此默认 x = 4

5. 与 2PCF 的测量平均值对比，保存结果
   - 每个 fnl 输出：
     bestfit_params.json
     metrics.json
     pk_fit.png
     r2xi_compare.png
   - 全部 fnl 结束后输出：
     summary_metrics.json
     summary_metrics.tsv
     summary_window_metrics.png
     interpolation_manifest.json

========================
重要说明
========================
1. 这里的“cubic 插值”使用 scipy.interpolate.CubicSpline。
   原因是任务只给了 3 个已知 fnl 节点（0/50/100），常见的 interp1d(kind='cubic')
   需要更多节点；CubicSpline 在 3 个节点下仍可定义分段三次样条，因此这里采用该口径。
2. 本脚本不会改写原始 /pscratch/sd/l/lzy/pks_2pcfs，只会把插值结果写到 mission6_log 下。
3. 本脚本依赖 desilike / cosmoprimo / pypower / scipy / matplotlib，请在可用环境运行。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from scipy.interpolate import CubicSpline


# =====================
# 一、数据规格与通用工具
# =====================


@dataclass(frozen=True)
class DatasetSpec:
    """
    描述某一个 fnl 数据集的输入位置与命名模式。

    参数
    ----------
    fnl : int
        当前数据集对应的 fNL 数值。
    tag : str
        用于目录命名和输出整理的短标签，例如 'fnl020'。
    pk_pattern : str
        功率谱文件 glob 模式。
    pcf_pattern : str
        2PCF 文件 glob 模式。
    root_dir : str
        该数据集所在目录。
    source_kind : str
        数据来源说明，'original' 或 'interpolated'。
    """

    fnl: int
    tag: str
    pk_pattern: str
    pcf_pattern: str
    root_dir: str
    source_kind: str


@dataclass(frozen=True)
class PkEnsemble:
    """
    保存某个 fnl 的功率谱 realization 集合。

    属性
    ----------
    kcen, kmin, kmax : ndarray
        参与拟合的 k 网格信息。
    p0_mocks : ndarray
        shape = (Nmock, Nk) 的 realization 样本矩阵。
    p0_mean : ndarray
        realization 平均值，shape = (Nk,)。
    used_files : list[str]
        实际读取到的文件列表。
    """

    kcen: np.ndarray
    kmin: np.ndarray
    kmax: np.ndarray
    p0_mocks: np.ndarray
    p0_mean: np.ndarray
    used_files: List[str]


def parse_realization_id(filepath: str) -> int:
    """
    从文件名中解析 realization 编号。

    支持的命名例子：
    - pk_fid_0.txt
    - pk_LCp50_127.txt
    - pk_fnl20_499.txt
    - pcf_fnl75_42.dat
    """
    name = os.path.basename(filepath)
    match = re.search(r"_([0-9]+)\.(?:txt|dat)$", name)
    return int(match.group(1)) if match else -1


def fnl_tag(fnl: int) -> str:
    """
    把整数 fnl 转成统一目录标签，便于排序和输出。
    """
    return f"fnl{fnl:03d}"


def ensure_parent(path: Path) -> None:
    """
    确保某个文件路径的父目录存在。
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def save_tsv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    """
    把简单表格写成 TSV，避免额外依赖 pandas。
    """
    ensure_parent(path)
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(x) for x in row))
    path.write_text("\n".join(lines) + "\n")


# =====================
# 二、Mission 6.1：逐 realization 的 cubic 插值
# =====================


def original_dataset_specs(source_dir: str) -> Dict[int, DatasetSpec]:
    """
    返回原始真实测量的三个基础 fnl 节点规格。
    """
    return {
        0: DatasetSpec(
            fnl=0,
            tag=fnl_tag(0),
            pk_pattern="pk_fid_*.txt",
            pcf_pattern="pcf_fid_*.dat",
            root_dir=source_dir,
            source_kind="original",
        ),
        50: DatasetSpec(
            fnl=50,
            tag=fnl_tag(50),
            pk_pattern="pk_LCp50_*.txt",
            pcf_pattern="pcf_LCp50_*.dat",
            root_dir=source_dir,
            source_kind="original",
        ),
        100: DatasetSpec(
            fnl=100,
            tag=fnl_tag(100),
            pk_pattern="pk_LCp100_*.txt",
            pcf_pattern="pcf_LCp100_*.dat",
            root_dir=source_dir,
            source_kind="original",
        ),
    }


def dataset_spec_for_fnl(fnl: int, source_dir: str, interp_root: str) -> DatasetSpec:
    """
    根据 fnl 返回对应的数据规格。

    规则
    ----
    - 0/50/100：直接指向原始测量目录；
    - 20/30/75：指向 mission6 生成的插值目录。
    """
    originals = original_dataset_specs(source_dir)
    if fnl in originals:
        return originals[fnl]

    tag = fnl_tag(fnl)
    return DatasetSpec(
        fnl=fnl,
        tag=tag,
        pk_pattern=f"pk_fnl{fnl}_*.txt",
        pcf_pattern=f"pcf_fnl{fnl}_*.dat",
        root_dir=str(Path(interp_root) / tag),
        source_kind="interpolated",
    )


def _list_realization_files(root_dir: str, pattern: str) -> List[str]:
    """
    按 realization 编号排序列出文件。

    这里会自动排除 FCFC 的附属文件，例如 .dd / .xi2d。
    """
    files = sorted(glob.glob(os.path.join(root_dir, pattern)), key=parse_realization_id)
    return [fp for fp in files if not (fp.endswith(".dd") or fp.endswith(".xi2d"))]


def _load_triplet_arrays(filepaths: List[str]) -> np.ndarray:
    """
    读取三个基础 fnl 节点对应的数组，并检查形状一致。

    返回
    ----------
    ndarray
        shape = (3, nrow, ncol)
    """
    arrays = [np.loadtxt(fp, comments="#") for fp in filepaths]
    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) != 1:
        raise ValueError(f"基础插值节点文件形状不一致：{filepaths} -> {shapes}")
    return np.asarray(arrays, dtype=np.float64)


def _write_interpolated_file(path: Path, arr: np.ndarray) -> None:
    """
    将插值后的数组写到磁盘。

    说明
    ----
    - 输出不带 header，保持与原始输入的最小兼容格式一致；
    - 使用较高精度，避免后续读取时损失数值信息。
    """
    ensure_parent(path)
    np.savetxt(path, arr, fmt="%.12e")


def build_interpolated_datasets(
    *,
    source_dir: str,
    interp_root: str,
    realization_min: int,
    realization_max: int,
    targets: Iterable[int],
) -> Dict[str, Any]:
    """
    为目标 fnl 批量生成逐 realization 的插值数据。

    参数
    ----------
    source_dir : str
        原始输入目录，即 /pscratch/sd/l/lzy/pks_2pcfs。
    interp_root : str
        插值输出根目录。
    realization_min, realization_max : int
        需要处理的 realization 范围，闭区间。
    targets : Iterable[int]
        需要生成的目标 fnl 列表，例如 [20, 30, 75]。

    返回
    ----------
    dict
        插值汇总信息，用于后续写 manifest。
    """
    base_specs = original_dataset_specs(source_dir)
    base_fnls = np.asarray(sorted(base_specs.keys()), dtype=np.float64)

    pk_files_by_fnl: Dict[int, List[str]] = {}
    pcf_files_by_fnl: Dict[int, List[str]] = {}
    for fnl, spec in base_specs.items():
        pk_files = _list_realization_files(spec.root_dir, spec.pk_pattern)
        pcf_files = _list_realization_files(spec.root_dir, spec.pcf_pattern)
        pk_files_by_fnl[fnl] = pk_files
        pcf_files_by_fnl[fnl] = pcf_files

    # 基本完整性检查：三套基础数据的 realization 编号必须一致。
    target_ids = list(range(realization_min, realization_max + 1))
    for fnl, files in pk_files_by_fnl.items():
        ids = [parse_realization_id(fp) for fp in files]
        missing = [rid for rid in target_ids if rid not in ids]
        if missing:
            raise RuntimeError(f"[PK] 基础节点 fnl={fnl} 缺失 realization，例如：{missing[:10]}")
    for fnl, files in pcf_files_by_fnl.items():
        ids = [parse_realization_id(fp) for fp in files]
        missing = [rid for rid in target_ids if rid not in ids]
        if missing:
            raise RuntimeError(f"[PCF] 基础节点 fnl={fnl} 缺失 realization，例如：{missing[:10]}")

    manifest: Dict[str, Any] = {
        "source_dir": source_dir,
        "interp_root": interp_root,
        "realization_min": realization_min,
        "realization_max": realization_max,
        "base_fnls": [int(x) for x in base_fnls.tolist()],
        "targets": [],
    }

    for target_fnl in targets:
        if target_fnl in base_specs:
            continue

        target_dir = Path(interp_root) / fnl_tag(target_fnl)
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INTERP] 开始生成 fnl={target_fnl} -> {target_dir}")
        sample_pk_shape = None
        sample_pcf_shape = None

        for rid in range(realization_min, realization_max + 1):
            pk_triplet = [
                os.path.join(source_dir, f"pk_fid_{rid}.txt"),
                os.path.join(source_dir, f"pk_LCp50_{rid}.txt"),
                os.path.join(source_dir, f"pk_LCp100_{rid}.txt"),
            ]
            pcf_triplet = [
                os.path.join(source_dir, f"pcf_fid_{rid}.dat"),
                os.path.join(source_dir, f"pcf_LCp50_{rid}.dat"),
                os.path.join(source_dir, f"pcf_LCp100_{rid}.dat"),
            ]

            pk_stack = _load_triplet_arrays(pk_triplet)
            pcf_stack = _load_triplet_arrays(pcf_triplet)

            # 沿 fnl 方向做 cubic spline 插值，axis=0 表示 3 个节点这一维。
            pk_spline = CubicSpline(base_fnls, pk_stack, axis=0)
            pcf_spline = CubicSpline(base_fnls, pcf_stack, axis=0)

            pk_interp = np.asarray(pk_spline(float(target_fnl)), dtype=np.float64)
            pcf_interp = np.asarray(pcf_spline(float(target_fnl)), dtype=np.float64)

            # 物理网格列应当在不同 fnl 下保持一致。为了抑制三次样条的极小数值扰动，
            # 这里直接把这些“坐标列”替换成 fid 的原始值，确保后续文件严格同网格。
            pk_interp[:, 0:5] = pk_stack[0, :, 0:5]
            pcf_interp[:, 0:3] = pcf_stack[0, :, 0:3]

            pk_out = target_dir / f"pk_fnl{target_fnl}_{rid}.txt"
            pcf_out = target_dir / f"pcf_fnl{target_fnl}_{rid}.dat"
            _write_interpolated_file(pk_out, pk_interp)
            _write_interpolated_file(pcf_out, pcf_interp)

            if sample_pk_shape is None:
                sample_pk_shape = list(pk_interp.shape)
            if sample_pcf_shape is None:
                sample_pcf_shape = list(pcf_interp.shape)

        manifest["targets"].append(
            {
                "fnl": int(target_fnl),
                "tag": fnl_tag(target_fnl),
                "realizations": int(realization_max - realization_min + 1),
                "pk_shape": sample_pk_shape,
                "pcf_shape": sample_pcf_shape,
                "output_dir": str(target_dir),
            }
        )
        print(f"[INTERP] fnl={target_fnl} 完成")

    return manifest


# =====================
# 三、Mission 6.2：读取数据、拟合 P(k)、计算 xi(r)
# =====================


def read_single_pk_file(
    filepath: str,
    n_datapoints: int,
    k_cen_col: int,
    k_min_col: int,
    k_max_col: int,
    p0_col: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    读取单个功率谱文件，并截取前 n_datapoints 个 k-bin。

    参数
    ----------
    filepath : str
        输入文件路径。
    n_datapoints : int
        只读取前多少个 k-bin 进入拟合。
    k_cen_col, k_min_col, k_max_col, p0_col : int
        功率谱文件中对应列的索引。

    返回
    ----------
    tuple(ndarray, ndarray, ndarray, ndarray)
        kcen, kmin, kmax, p0
    """
    arr = np.loadtxt(filepath, comments="#")
    sl = slice(0, n_datapoints)
    return arr[sl, k_cen_col], arr[sl, k_min_col], arr[sl, k_max_col], arr[sl, p0_col]


def load_pk_ensemble(
    *,
    spec: DatasetSpec,
    realization_min: int,
    realization_max: int,
    n_datapoints: int,
    k_cen_col: int,
    k_min_col: int,
    k_max_col: int,
    p0_col: int,
) -> PkEnsemble:
    """
    批量读取某个 fnl 的功率谱 realization，构造 data/covariance 需要的均值与样本矩阵。
    """
    files = _list_realization_files(spec.root_dir, spec.pk_pattern)
    selected = [fp for fp in files if realization_min <= parse_realization_id(fp) <= realization_max]
    if not selected:
        raise FileNotFoundError(
            f"未找到可用功率谱文件：root={spec.root_dir}, pattern={spec.pk_pattern}"
        )

    ref_kcen, ref_kmin, ref_kmax, _ = read_single_pk_file(
        selected[0], n_datapoints, k_cen_col, k_min_col, k_max_col, p0_col
    )

    p0_list: List[np.ndarray] = []
    for fp in selected:
        kcen, kmin, kmax, p0 = read_single_pk_file(
            fp, n_datapoints, k_cen_col, k_min_col, k_max_col, p0_col
        )
        if not (
            np.allclose(kcen, ref_kcen, rtol=0.0, atol=1e-12)
            and np.allclose(kmin, ref_kmin, rtol=0.0, atol=1e-12)
            and np.allclose(kmax, ref_kmax, rtol=0.0, atol=1e-12)
        ):
            raise ValueError(f"功率谱 k 网格不一致：{fp}")
        p0_list.append(p0)

    p0_mocks = np.asarray(p0_list, dtype=np.float64)
    p0_mean = np.mean(p0_mocks, axis=0)

    print(f"[PK][fnl={spec.fnl}] 样本数 = {p0_mocks.shape[0]}  k-bin = {p0_mocks.shape[1]}")
    return PkEnsemble(
        kcen=ref_kcen,
        kmin=ref_kmin,
        kmax=ref_kmax,
        p0_mocks=p0_mocks,
        p0_mean=p0_mean,
        used_files=selected,
    )


def filter_singular_pk_bins(pk: PkEnsemble) -> PkEnsemble:
    """
    移除协方差中方差为 0 或非有限的 k-bin，避免 desilike 初始化时出现奇异矩阵。
    """
    std = np.std(pk.p0_mocks, axis=0, ddof=1)
    finite = np.all(np.isfinite(pk.p0_mocks), axis=0) & np.isfinite(std)
    nonzero = ~np.isclose(std, 0.0, rtol=0.0, atol=0.0)
    keep = finite & nonzero

    if not np.all(keep):
        print(
            f"[PK] 剔除奇异/无效 k-bin: drop={np.where(~keep)[0].tolist()} "
            f"keep={int(np.sum(keep))}/{keep.size}"
        )

    if int(np.sum(keep)) < 3:
        raise RuntimeError("过滤奇异 k-bin 后剩余点数过少，无法拟合。")

    return PkEnsemble(
        kcen=pk.kcen[keep],
        kmin=pk.kmin[keep],
        kmax=pk.kmax[keep],
        p0_mocks=pk.p0_mocks[:, keep],
        p0_mean=pk.p0_mean[keep],
        used_files=list(pk.used_files),
    )


def build_pypower_data_and_mocks(
    *,
    kcen: np.ndarray,
    kmin: np.ndarray,
    kmax: np.ndarray,
    p0_mean_data: np.ndarray,
    p0_cov_mocks: np.ndarray,
):
    """
    把 data 与 covariance realization 封装成 desilike 可识别的 pypower 统计对象。

    返回
    ----------
    tuple
        data_ps, mock_ps_list
    """
    from pypower import PowerSpectrumStatistics

    edges = np.concatenate([kmin, [kmax[-1]]])
    nmodes = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    data_ps = PowerSpectrumStatistics(
        edges=edges,
        modes=kcen,
        power_nonorm=np.asarray([p0_mean_data], dtype="f8"),
        nmodes=nmodes,
        ells=[0],
        shotnoise_nonorm=0.0,
        statistic="multipole",
    )

    mock_ps_list = []
    for i in range(p0_cov_mocks.shape[0]):
        tmp = data_ps.deepcopy()
        tmp.power_nonorm.flat[...] = np.asarray([p0_cov_mocks[i]], dtype="f8").ravel()
        mock_ps_list.append(tmp)
    return data_ps, mock_ps_list


def read_pcf_mean_std(
    *,
    spec: DatasetSpec,
    realization_min: int,
    realization_max: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    读取某个 fnl 的 2PCF realization，返回 r 网格、均值、标准差与文件列表。
    """
    files = _list_realization_files(spec.root_dir, spec.pcf_pattern)
    selected = [fp for fp in files if realization_min <= parse_realization_id(fp) <= realization_max]
    if not selected:
        raise FileNotFoundError(
            f"未找到可用 2PCF 文件：root={spec.root_dir}, pattern={spec.pcf_pattern}"
        )

    xi_list: List[np.ndarray] = []
    r_grid = None
    for fp in selected:
        arr = np.loadtxt(fp, comments="#")
        if r_grid is None:
            r_grid = arr[:, 0]
        elif not np.allclose(arr[:, 0], r_grid, rtol=0.0, atol=1e-12):
            raise ValueError(f"2PCF r 网格不一致：{fp}")
        xi_list.append(arr[:, 3])

    xi_stack = np.asarray(xi_list, dtype=np.float64)
    xi_mean = np.mean(xi_stack, axis=0)
    xi_std = np.std(xi_stack, axis=0, ddof=1)
    assert r_grid is not None
    return r_grid, xi_mean, xi_std, selected


# =====================
# 四、FFTLog 与窗口函数
# =====================


def build_log_taper_window(k_array: np.ndarray, kmin: float, kmax: float, frac: float) -> np.ndarray:
    """
    在对数 k 空间构造平滑边界窗，减弱硬截断引起的振铃。
    """
    w = np.zeros_like(k_array, dtype=np.float64)
    lk = np.log(k_array)
    l0 = np.log(kmin)
    l1 = np.log(kmax)
    dl = frac * (l1 - l0)

    if dl <= 0:
        w[(k_array >= kmin) & (k_array <= kmax)] = 1.0
        return w

    left = l0 + dl
    right = l1 - dl

    m = (lk >= l0) & (lk < left)
    w[m] = 0.5 * (1.0 - np.cos(np.pi * (lk[m] - l0) / dl))

    m = (lk >= left) & (lk <= right)
    w[m] = 1.0

    m = (lk > right) & (lk <= l1)
    w[m] = 0.5 * (1.0 + np.cos(np.pi * (lk[m] - right) / dl))
    return w


def build_ir_param_window(k_array: np.ndarray, k_fund: float, x_power: float) -> np.ndarray:
    """
    构造 mission5 当前最佳参数化 IR 窗口。

    公式
    ----
    W(k) = [1 - exp(-(k/k_f)^x)] / [1 - exp(-1)],   k < k_f
         = 1,                                       k >= k_f

    这里加上分母归一化，是为了保证在 k -> k_f^- 时与高 k 段连续拼接。
    """
    w = np.ones_like(k_array, dtype=np.float64)
    mask = k_array < k_fund
    if np.any(mask):
        ratio = np.clip(k_array[mask] / k_fund, 0.0, None)
        norm = 1.0 - np.exp(-1.0)
        w[mask] = (1.0 - np.exp(-(ratio ** x_power))) / max(norm, 1e-30)
    return np.clip(w, 0.0, 1.0)


def xi0_from_p0_fftlog(
    *,
    k_grid: np.ndarray,
    p0_grid: np.ndarray,
    kmin: float,
    kmax: float,
    mu: float,
    bias: float,
    taper_frac: float,
    use_param_ir_window: bool,
    k_fund: float | None,
    x_power: float | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    用 FFTLog 计算 xi0(r)。

    参数
    ----------
    k_grid, p0_grid : ndarray
        高分辨率对数 k 网格及理论 P0(k)。
    kmin, kmax : float
        当前数值积分的上下限。
    mu, bias : float
        scipy.fft.fht / fhtoffset 需要的 FFTLog 参数。
    taper_frac : float
        边界平滑窗在 log(k) 区间中所占比例。
    use_param_ir_window : bool
        是否额外乘 mission5 的参数化 IR 窗口。
    k_fund, x_power : float | None
        参数化窗口所需参数。

    返回
    ----------
    tuple
        r_grid, xi0, window_taper, window_ir
    """
    from scipy.fft import fht, fhtoffset

    window_taper = build_log_taper_window(k_grid, kmin, kmax, taper_frac)

    if use_param_ir_window:
        if k_fund is None or x_power is None:
            raise ValueError("use_param_ir_window=True 时必须提供 k_fund 与 x_power")
        window_ir = build_ir_param_window(k_grid, k_fund=k_fund, x_power=x_power)
    else:
        window_ir = np.ones_like(k_grid, dtype=np.float64)

    p0_eff = p0_grid * window_taper * window_ir

    dln = float(np.log(k_grid[1] / k_grid[0]))
    offset = float(fhtoffset(dln, mu=mu, initial=0.0, bias=bias))
    a_in = (k_grid ** 1.5) * p0_eff
    a_out = fht(a_in, dln=dln, mu=mu, offset=offset, bias=bias)

    n = k_grid.size
    j = np.arange(n)
    j_center = (n - 1) / 2.0
    ln_kc = 0.5 * (np.log(k_grid[0]) + np.log(k_grid[-1]))
    ln_r = -ln_kc + offset + (j - j_center) * dln
    r_grid = np.exp(ln_r)

    coef = np.sqrt(np.pi / 2.0) / (2.0 * np.pi**2)
    xi0 = coef * a_out / (r_grid ** 1.5)
    return r_grid, xi0, window_taper, window_ir


def _to_jsonable_params(bestfit_params: Dict[str, Any]) -> Dict[str, float]:
    """
    把 desilike 的 ParameterArray 风格输出转成普通 float 字典，方便写 JSON。
    """
    out: Dict[str, float] = {}
    for key, value in bestfit_params.items():
        try:
            out[key] = float(np.asarray(value))
        except Exception:
            out[key] = float(value)
    return out


# =====================
# 五、单个 fnl 的完整验证流程
# =====================


def run_single_validation(
    *,
    spec: DatasetSpec,
    output_dir: Path,
    realization_min: int,
    realization_max: int,
    n_datapoints: int,
    k_cen_col: int,
    k_min_col: int,
    k_max_col: int,
    p0_col: int,
    kint_min: float,
    kint_max: float,
    box_size: float,
    fftlog_n: int,
    fftlog_padding: float,
    fftlog_mu: float,
    fftlog_bias: float,
    edge_taper_frac: float,
    unit_z: float,
    fixed_p: float,
    fixed_sn0: float,
    fixed_sigmas: float,
    minuit_seed: int,
    minuit_niter: int,
    r_large_min: float,
    pk_plot_kmax: float,
    pk_plot_npts: int,
) -> Dict[str, Any]:
    """
    对单个 fnl 完成完整验证。

    返回
    ----------
    dict
        这个 fnl 的汇总信息，会被写入全局 summary。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取当前 fnl 的 P(k) realization，并用同一批样本做 covariance。
    pk = load_pk_ensemble(
        spec=spec,
        realization_min=realization_min,
        realization_max=realization_max,
        n_datapoints=n_datapoints,
        k_cen_col=k_cen_col,
        k_min_col=k_min_col,
        k_max_col=k_max_col,
        p0_col=p0_col,
    )
    pk = filter_singular_pk_bins(pk)
    data_ps, mock_ps_list = build_pypower_data_and_mocks(
        kcen=pk.kcen,
        kmin=pk.kmin,
        kmax=pk.kmax,
        p0_mean_data=pk.p0_mean,
        p0_cov_mocks=pk.p0_mocks,
    )

    # 2) desilike best-fit
    from cosmoprimo import Cosmology
    from desilike import setup_logging
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.profilers import MinuitProfiler
    from desilike.theories.galaxy_clustering import (
        FixedPowerSpectrumTemplate,
        PNGTracerPowerSpectrumMultipoles,
    )

    setup_logging()

    cosmo_unit = Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )

    template = FixedPowerSpectrumTemplate(z=unit_z, fiducial=cosmo_unit)
    theory = PNGTracerPowerSpectrumMultipoles(template=template, mode="b-p")
    theory.init.params["p"].update(fixed=True, value=fixed_p)
    theory.init.params["sn0"].update(fixed=True, value=fixed_sn0)
    theory.init.params["sigmas"].update(fixed=False, value=fixed_sigmas)

    observable = TracerPowerSpectrumMultipolesObservable(
        data=data_ps,
        covariance=mock_ps_list,
        klim={0: [float(pk.kcen.min()), float(pk.kcen.max()), float(pk.kcen[1] - pk.kcen[0])]},
        theory=theory,
    )
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    _ = likelihood()
    likelihood.all_params["p"].update(fixed=True, value=fixed_p)
    likelihood.all_params["sn0"].update(fixed=True, value=fixed_sn0)
    likelihood.all_params["sigmas"].update(fixed=False, value=fixed_sigmas)

    print(f"[FIT][fnl={spec.fnl}] 开始 Minuit profiling...")
    profiler = MinuitProfiler(likelihood, seed=minuit_seed)
    profiles = profiler.maximize(niterations=minuit_niter)
    print(profiles.to_stats(tablefmt="pretty"))

    bestfit_params_raw = profiles.bestfit.choice(input=True)
    bestfit_params = _to_jsonable_params(bestfit_params_raw)
    ensure_parent(output_dir / "bestfit_params.json")
    (output_dir / "bestfit_params.json").write_text(
        json.dumps(bestfit_params, indent=2, sort_keys=True)
    )

    # 3) 画 P(k) 的均值与 best-fit
    _ = theory(**bestfit_params_raw)
    model_p0_fitgrid = np.asarray(theory.power[0], dtype=np.float64)

    k_plot = np.geomspace(float(pk.kcen.min()), pk_plot_kmax, pk_plot_npts)
    template_plot = FixedPowerSpectrumTemplate(z=unit_z, fiducial=cosmo_unit)
    theory_plot = PNGTracerPowerSpectrumMultipoles(k=k_plot, template=template_plot, mode="b-p")
    theory_plot.init.params["p"].update(fixed=True, value=fixed_p)
    theory_plot.init.params["sn0"].update(fixed=True, value=fixed_sn0)
    theory_plot.init.params["sigmas"].update(fixed=False, value=fixed_sigmas)
    theory_plot(**bestfit_params_raw)
    model_p0_plot = np.asarray(theory_plot.power[0], dtype=np.float64)

    import matplotlib.pyplot as plt

    err_p0 = np.std(pk.p0_mocks, axis=0, ddof=1)

    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax1.errorbar(
        pk.kcen,
        pk.p0_mean,
        yerr=err_p0,
        fmt="o",
        ms=4,
        capsize=2,
        color="black",
        label=f"Measured mean P0 (N={pk.p0_mocks.shape[0]})",
    )
    ax1.loglog(k_plot, model_p0_plot, "-", lw=2.0, color="tab:red", label="Best-fit model")
    ax1.set_xlim(float(pk.kcen.min()), pk_plot_kmax)
    ax1.set_ylabel("P0(k)")
    ax1.set_title(f"N-body fnl={spec.fnl}: mean P0 vs best-fit")
    ax1.grid(True, which="both", ls="-", color="0.85")
    ax1.legend(frameon=False)

    ax2 = plt.subplot(2, 1, 2)
    ratio = pk.p0_mean / model_p0_fitgrid
    ratio_err = err_p0 / np.maximum(np.abs(model_p0_fitgrid), 1e-30)
    ax2.errorbar(pk.kcen, ratio, yerr=ratio_err, fmt="o", ms=4, capsize=2, color="tab:blue")
    ax2.axhline(1.0, color="red", ls="--", lw=1.0)
    ax2.set_xscale("log")
    ax2.set_xlim(float(pk.kcen.min()), pk_plot_kmax)
    ax2.set_xlabel("k [h/Mpc]")
    ax2.set_ylabel("Data/Model")
    ax2.grid(True, which="both", ls="-", color="0.85")

    plt.tight_layout()
    plt.savefig(output_dir / "pk_fit.png", dpi=180)
    plt.close()

    # 4) 在高分辨率 k 网格上评估理论并做 FFTLog
    k_fund = 2.0 * np.pi / box_size
    x_power = 4.0 * (box_size / 1000.0)

    def build_png_theory_on_kgrid(k_grid: np.ndarray) -> np.ndarray:
        """
        在指定 k 网格上评估 best-fit 的 P0(k)。
        """
        template_fft = FixedPowerSpectrumTemplate(z=unit_z, fiducial=cosmo_unit)
        theory_fft = PNGTracerPowerSpectrumMultipoles(k=k_grid, template=template_fft, mode="b-p")
        theory_fft.init.params["p"].update(fixed=True, value=fixed_p)
        theory_fft.init.params["sn0"].update(fixed=True, value=fixed_sn0)
        theory_fft.init.params["sigmas"].update(fixed=False, value=fixed_sigmas)
        theory_fft(**bestfit_params_raw)
        return np.asarray(theory_fft.power[0], dtype=np.float64)

    baseline_kmin = k_fund
    k_grid_baseline = np.geomspace(
        baseline_kmin / fftlog_padding, kint_max * fftlog_padding, fftlog_n
    )
    k_grid_window = np.geomspace(
        kint_min / fftlog_padding, kint_max * fftlog_padding, fftlog_n
    )
    p0_model_baseline = build_png_theory_on_kgrid(k_grid_baseline)
    p0_model_window = build_png_theory_on_kgrid(k_grid_window)

    r_base, xi_base, _, _ = xi0_from_p0_fftlog(
        k_grid=k_grid_baseline,
        p0_grid=p0_model_baseline,
        kmin=baseline_kmin,
        kmax=kint_max,
        mu=fftlog_mu,
        bias=fftlog_bias,
        taper_frac=edge_taper_frac,
        use_param_ir_window=False,
        k_fund=k_fund,
        x_power=x_power,
    )
    r_win, xi_win, _, _ = xi0_from_p0_fftlog(
        k_grid=k_grid_window,
        p0_grid=p0_model_window,
        kmin=kint_min,
        kmax=kint_max,
        mu=fftlog_mu,
        bias=fftlog_bias,
        taper_frac=edge_taper_frac,
        use_param_ir_window=True,
        k_fund=k_fund,
        x_power=x_power,
    )

    # 5) 读取 2PCF 测量，与 baseline/window 同时比较
    r_data, xi_mean, xi_std, used_pcf = read_pcf_mean_std(
        spec=spec,
        realization_min=realization_min,
        realization_max=realization_max,
    )

    ord_b = np.argsort(r_base)
    ord_w = np.argsort(r_win)
    xi_model_base = np.interp(r_data, r_base[ord_b], xi_base[ord_b])
    xi_model_win = np.interp(r_data, r_win[ord_w], xi_win[ord_w])

    r2_xi_data = r_data**2 * xi_mean
    r2_xi_err = r_data**2 * xi_std
    r2_xi_base = r_data**2 * xi_model_base
    r2_xi_win = r_data**2 * xi_model_win

    sigma_floor = np.nanmedian(r2_xi_err[r2_xi_err > 0]) * 1e-6 if np.any(r2_xi_err > 0) else 1e-12
    r2_xi_err_safe = np.where(r2_xi_err > 0, r2_xi_err, sigma_floor)
    delta_sigma_base = (r2_xi_data - r2_xi_base) / r2_xi_err_safe
    delta_sigma_win = (r2_xi_data - r2_xi_win) / r2_xi_err_safe

    def stats_on_mask(x: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        计算某一掩膜下的 |delta/sigma| 统计量。
        """
        xx = np.asarray(x[mask], dtype=np.float64)
        return {
            "mean_abs": float(np.nanmean(np.abs(xx))),
            "max_abs": float(np.nanmax(np.abs(xx))),
            "count": int(np.sum(mask)),
        }

    mask_all = np.isfinite(delta_sigma_win)
    mask_large = (r_data >= r_large_min) & mask_all

    metrics = {
        "fnl": int(spec.fnl),
        "tag": spec.tag,
        "source_kind": spec.source_kind,
        "inputs": {
            "root_dir": spec.root_dir,
            "pk_pattern": spec.pk_pattern,
            "pcf_pattern": spec.pcf_pattern,
            "realization_min": int(realization_min),
            "realization_max": int(realization_max),
            "n_datapoints": int(n_datapoints),
            "n_pk_files": int(len(pk.used_files)),
            "n_pcf_files": int(len(used_pcf)),
        },
        "fftlog": {
            "box_size": float(box_size),
            "k_fund": float(k_fund),
            "kint_min": float(kint_min),
            "kint_max": float(kint_max),
            "ir_x": float(x_power),
        },
        "fit": {
            "bestfit_params": bestfit_params,
        },
        "delta_sigma": {
            "baseline_all": stats_on_mask(delta_sigma_base, mask_all),
            "window_all": stats_on_mask(delta_sigma_win, mask_all),
            "baseline_large": stats_on_mask(delta_sigma_base, mask_large),
            "window_large": stats_on_mask(delta_sigma_win, mask_large),
            "r_large_min": float(r_large_min),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    # 6) 保存 2PCF 对比图
    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax1.errorbar(
        r_data,
        r2_xi_data,
        yerr=r2_xi_err,
        fmt="o",
        ms=4,
        capsize=2,
        color="black",
        label=r"Measured $r^2\xi_0$ mean",
    )
    ax1.plot(r_data, r2_xi_base, "-", lw=1.8, color="tab:blue", label="Model baseline")
    ax1.plot(
        r_data,
        r2_xi_win,
        "-",
        lw=2.0,
        color="tab:red",
        label=f"Model window (x={x_power:.1f})",
    )
    ax1.set_ylabel(r"$r^2\xi_0(r)$")
    ax1.set_title(f"N-body fnl={spec.fnl}: baseline vs parameterized window")
    ax1.grid(True, ls="--", alpha=0.35)
    ax1.legend(frameon=False)

    ax2 = plt.subplot(2, 1, 2)
    ax2.axhline(0.0, color="black", ls="-", lw=1.0)
    ax2.axhline(1.0, color="red", ls="--", lw=1.0)
    ax2.axhline(-1.0, color="red", ls="--", lw=1.0)
    ax2.plot(r_data, delta_sigma_base, "o-", ms=3, lw=1.0, color="tab:blue", alpha=0.8, label="baseline")
    ax2.plot(r_data, delta_sigma_win, "o-", ms=3, lw=1.0, color="tab:red", alpha=0.9, label="window")
    ax2.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$")
    ax2.set_ylabel(r"$(Data-Model)/\sigma$ of $r^2\xi_0$")
    ax2.grid(True, ls="--", alpha=0.35)
    ax2.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(output_dir / "r2xi_compare.png", dpi=180)
    plt.close()

    print(
        f"[DONE][fnl={spec.fnl}] "
        f"baseline_large_mean={metrics['delta_sigma']['baseline_large']['mean_abs']:.3f} "
        f"window_large_mean={metrics['delta_sigma']['window_large']['mean_abs']:.3f}"
    )
    return metrics


# =====================
# 六、汇总输出
# =====================


def save_summary(work_dir: Path, all_metrics: List[Dict[str, Any]]) -> None:
    """
    保存全部 fnl 的汇总 JSON / TSV / 总览图。
    """
    summary_json = work_dir / "summary_metrics.json"
    summary_json.write_text(json.dumps(all_metrics, indent=2, sort_keys=True))

    rows: List[List[Any]] = []
    for item in sorted(all_metrics, key=lambda x: x["fnl"]):
        fit = item["fit"]["bestfit_params"]
        ds = item["delta_sigma"]
        rows.append(
            [
                item["fnl"],
                item["tag"],
                item["source_kind"],
                fit.get("fnl_loc", "nan"),
                fit.get("b1", "nan"),
                fit.get("sigmas", "nan"),
                ds["baseline_all"]["mean_abs"],
                ds["window_all"]["mean_abs"],
                ds["baseline_large"]["mean_abs"],
                ds["window_large"]["mean_abs"],
                ds["baseline_large"]["max_abs"],
                ds["window_large"]["max_abs"],
            ]
        )

    save_tsv(
        work_dir / "summary_metrics.tsv",
        [
            "fnl",
            "tag",
            "source_kind",
            "bestfit_fnl_loc",
            "bestfit_b1",
            "bestfit_sigmas",
            "baseline_all_mean_abs",
            "window_all_mean_abs",
            "baseline_large_mean_abs",
            "window_large_mean_abs",
            "baseline_large_max_abs",
            "window_large_max_abs",
        ],
        rows,
    )

    import matplotlib.pyplot as plt

    fnls = np.asarray([item["fnl"] for item in sorted(all_metrics, key=lambda x: x["fnl"])], dtype=float)
    baseline_large = np.asarray(
        [item["delta_sigma"]["baseline_large"]["mean_abs"] for item in sorted(all_metrics, key=lambda x: x["fnl"])],
        dtype=float,
    )
    window_large = np.asarray(
        [item["delta_sigma"]["window_large"]["mean_abs"] for item in sorted(all_metrics, key=lambda x: x["fnl"])],
        dtype=float,
    )
    baseline_all = np.asarray(
        [item["delta_sigma"]["baseline_all"]["mean_abs"] for item in sorted(all_metrics, key=lambda x: x["fnl"])],
        dtype=float,
    )
    window_all = np.asarray(
        [item["delta_sigma"]["window_all"]["mean_abs"] for item in sorted(all_metrics, key=lambda x: x["fnl"])],
        dtype=float,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(fnls, baseline_large, "o-", color="tab:blue", lw=1.8, label="baseline, large-scale")
    plt.plot(fnls, window_large, "o-", color="tab:red", lw=2.0, label="window, large-scale")
    plt.plot(fnls, baseline_all, "s--", color="tab:blue", alpha=0.55, label="baseline, all-scale")
    plt.plot(fnls, window_all, "s--", color="tab:red", alpha=0.55, label="window, all-scale")
    plt.xlabel("input fnl")
    plt.ylabel(r"mean $|(Data-Model)/\sigma|$")
    plt.title("Mission 6 summary: parameterized window on Quijote N-body samples")
    plt.grid(True, ls="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(work_dir / "summary_window_metrics.png", dpi=180)
    plt.close()


# =====================
# 七、主程序入口
# =====================


def build_argparser() -> argparse.ArgumentParser:
    """
    构造命令行参数解析器。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-dir",
        default="/pscratch/sd/l/lzy/pks_2pcfs",
        help="原始 N-body pk / 2PCF 输入目录",
    )
    ap.add_argument(
        "--work-dir",
        default="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission6_log/nbody_validation",
        help="Mission 6 的输出根目录",
    )
    ap.add_argument(
        "--fnls",
        default="0,20,30,50,75,100",
        help="需要验证的 fnl 列表，逗号分隔",
    )
    ap.add_argument(
        "--interp-targets",
        default="20,30,75",
        help="需要通过 cubic spline 生成的目标 fnl 列表，逗号分隔",
    )

    ap.add_argument("--realization-min", type=int, default=0)
    ap.add_argument("--realization-max", type=int, default=499)

    ap.add_argument("--n-datapoints", type=int, default=20)
    ap.add_argument("--kcen-col", type=int, default=0)
    ap.add_argument("--kmin-col", type=int, default=1)
    ap.add_argument("--kmax-col", type=int, default=2)
    ap.add_argument("--p0-col", type=int, default=5)

    ap.add_argument("--kint-min", type=float, default=1e-4)
    ap.add_argument("--kint-max", type=float, default=20.0)
    ap.add_argument("--box-size", type=float, default=1000.0)
    ap.add_argument("--fftlog-n", type=int, default=4096)
    ap.add_argument("--fftlog-padding", type=float, default=4.0)
    ap.add_argument("--fftlog-mu", type=float, default=0.5)
    ap.add_argument("--fftlog-bias", type=float, default=0.0)
    ap.add_argument("--edge-taper-frac", type=float, default=0.06)

    ap.add_argument("--unit-z", type=float, default=1.0)
    ap.add_argument("--fixed-p", type=float, default=1.2)
    ap.add_argument("--fixed-sn0", type=float, default=0.0)
    ap.add_argument("--fixed-sigmas", type=float, default=0.0)
    ap.add_argument("--minuit-seed", type=int, default=66)
    ap.add_argument("--minuit-niter", type=int, default=27)

    ap.add_argument("--r-large-min", type=float, default=200.0)
    ap.add_argument("--pk-plot-kmax", type=float, default=1.0)
    ap.add_argument("--pk-plot-npts", type=int, default=400)
    return ap


def main() -> None:
    """
    主程序：
    1) 先生成插值数据；
    2) 再对所有指定 fnl 逐个完成验证；
    3) 最后输出总表。
    """
    args = build_argparser().parse_args()

    work_dir = Path(args.work_dir)
    interp_root = work_dir / "interpolated"
    validation_root = work_dir / "validation"
    work_dir.mkdir(parents=True, exist_ok=True)
    validation_root.mkdir(parents=True, exist_ok=True)

    requested_fnls = [int(x) for x in args.fnls.split(",") if x.strip()]
    interp_targets = [int(x) for x in args.interp_targets.split(",") if x.strip()]

    run_config = {
        "source_dir": args.source_dir,
        "work_dir": str(work_dir),
        "requested_fnls": requested_fnls,
        "interp_targets": interp_targets,
        "realization_min": args.realization_min,
        "realization_max": args.realization_max,
        "n_datapoints": args.n_datapoints,
        "fixed_p": args.fixed_p,
        "box_size": args.box_size,
    }
    (work_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True))

    interpolation_manifest = build_interpolated_datasets(
        source_dir=args.source_dir,
        interp_root=str(interp_root),
        realization_min=args.realization_min,
        realization_max=args.realization_max,
        targets=interp_targets,
    )
    (work_dir / "interpolation_manifest.json").write_text(
        json.dumps(interpolation_manifest, indent=2, sort_keys=True)
    )

    all_metrics: List[Dict[str, Any]] = []
    for fnl in requested_fnls:
        spec = dataset_spec_for_fnl(fnl, args.source_dir, str(interp_root))
        metrics = run_single_validation(
            spec=spec,
            output_dir=validation_root / spec.tag,
            realization_min=args.realization_min,
            realization_max=args.realization_max,
            n_datapoints=args.n_datapoints,
            k_cen_col=args.kcen_col,
            k_min_col=args.kmin_col,
            k_max_col=args.kmax_col,
            p0_col=args.p0_col,
            kint_min=args.kint_min,
            kint_max=args.kint_max,
            box_size=args.box_size,
            fftlog_n=args.fftlog_n,
            fftlog_padding=args.fftlog_padding,
            fftlog_mu=args.fftlog_mu,
            fftlog_bias=args.fftlog_bias,
            edge_taper_frac=args.edge_taper_frac,
            unit_z=args.unit_z,
            fixed_p=args.fixed_p,
            fixed_sn0=args.fixed_sn0,
            fixed_sigmas=args.fixed_sigmas,
            minuit_seed=args.minuit_seed,
            minuit_niter=args.minuit_niter,
            r_large_min=args.r_large_min,
            pk_plot_kmax=args.pk_plot_kmax,
            pk_plot_npts=args.pk_plot_npts,
        )
        all_metrics.append(metrics)

    save_summary(work_dir, all_metrics)
    print(f"[MISSION6] 全部完成，输出根目录：{work_dir}")


if __name__ == "__main__":
    main()
