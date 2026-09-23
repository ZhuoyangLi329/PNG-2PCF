#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task44 拟合入口的数值与缓存防护。

大纲：
1. validate_covariance 检查实际 likelihood 使用的完整矩阵，不用 pinv 掩盖负谱。
2. make_fit_contract 对参数、输入、代码和已加载理论建立稳定摘要。
3. reusable_fit 在复用之前检查合同、成功状态和输出哈希；旧缓存仍保留，
   但不把缺乏来源信息的旧文件当成本次拟合已完成。
本模块不改变模型、先验或 covariance；不自动覆盖、删除任何缓存。
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any
import numpy as np


def file_digest(path: str | Path) -> str:
    """分块读取指定文件并返回 SHA256，不扫描目录，缺失输入明确报错。"""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> Any:
    """将配置与 numpy 对象规范化；数组记录值摘要、shape 和 dtype。"""
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('object array cannot identify a numerical fit input')
        x = np.ascontiguousarray(value)
        return {'shape': list(x.shape), 'dtype': str(x.dtype),
                'sha256': hashlib.sha256(x.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return canonical(value.item())
    if isinstance(value, Path):
        return str(value.resolve())
    return value


def validate_covariance(covariance: np.ndarray, *, rcond: float = 1.0e-10) -> dict[str, Any]:
    """检查满秩 Gaussian likelihood 的协方差；返回谱诊断，失败则 ValueError。

    参数 covariance 为待求逆的矩阵，rcond 为后续 pinv 所用的相对截断。
    接受浮点舍入级不对称，但拒绝非有限值、明显不对称、非正定或
    将被 pinv 丢弃的模式；有意使用降秩 likelihood 必须另建显式合同。
    """
    c = np.asarray(covariance, dtype='f8')
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] == 0:
        raise ValueError('covariance must be a non-empty square matrix')
    if not np.all(np.isfinite(c)):
        raise ValueError('covariance contains non-finite values')
    if not np.isfinite(rcond) or not 0 <= rcond < 1:
        raise ValueError('rcond must be finite and in [0,1)')
    scale = max(float(np.max(np.abs(c))), np.finfo(float).tiny)
    asym = float(np.max(np.abs(c - c.T))) / scale
    if asym > 1.0e-10:
        raise ValueError(f'covariance is not symmetric: relative asymmetry={asym:.3g}')
    sym = (c + c.T) * 0.5
    eig = np.linalg.eigvalsh(sym)
    cutoff = float(rcond * max(float(eig[-1]), 0.0))
    if eig[0] <= 0.0:
        raise ValueError(f'covariance is not positive definite: eigmin={eig[0]:.8g}')
    if eig[0] <= cutoff:
        raise ValueError(f'covariance loses rank at rcond={rcond}: eigmin={eig[0]:.8g}, cutoff={cutoff:.8g}')
    try:
        np.linalg.cholesky(sym)
    except np.linalg.LinAlgError as exc:
        raise ValueError('covariance failed Cholesky factorization') from exc
    return {'spd_hard_gate': True, 'effective_rank': int(len(eig)),
            'min_eigenvalue': float(eig[0]), 'max_eigenvalue': float(eig[-1]),
            'condition_number': float(eig[-1] / eig[0]),
            'relative_asymmetry': asym, 'rank_rcond': float(rcond)}


def make_fit_contract(*, config: dict[str, Any], inputs: dict[str, str | Path],
                      code: list[str | Path], theory: dict[str, Any]) -> dict[str, Any]:
    """生成一次拟合的精确复用合同；包含实际加载的理论而不只信任文件名。

    config 应包括先验、尺度、采样参数、nbar 与固定参数；inputs 包括
    测量、理论缓存及条件 covariance 的源拟合；code 列出直接科学依赖。
    """
    versions = {}
    for name in ('numpy', 'scipy', 'emcee'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not-installed'
    payload = {'schema': 'task44_fit_contract_v1', 'config': canonical(config),
               'inputs': {k: {'path': str(Path(v).resolve()), 'sha256': file_digest(v)}
                          for k, v in sorted(inputs.items())},
               'code': {str(Path(p).resolve()): file_digest(p) for p in code},
               'loaded_theory': canonical(theory), 'runtime_versions': versions}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return {'sha256': hashlib.sha256(encoded.encode()).hexdigest(), 'payload': payload}


def reusable_fit(prefix: str | Path, contract: dict[str, Any], *, overwrite: bool) -> bool:
    """只有两文件完整、合同一致、status=pass 且 NPZ 哈希通过时才复用。

    overwrite=True 表示调用者明确要求重新计算；本函数本身不做任何写入。
    旧结果/半成品/失配合同保留原地并报错，提示用新输出或显式重算。
    """
    prefix = Path(prefix)
    js, nz = prefix.with_suffix('.json'), prefix.with_suffix('.npz')
    if overwrite or not (js.exists() or nz.exists()):
        return False
    reason = None
    if not (js.is_file() and nz.is_file()):
        reason = 'incomplete JSON/NPZ pair'
    else:
        try:
            summary = json.loads(js.read_text(encoding='utf-8'))
            if summary.get('cache_contract') != contract:
                reason = 'missing or changed input/config/code contract'
            elif summary.get('status') != 'pass':
                reason = f"saved fit status is {summary.get('status')!r}"
            elif summary.get('output_npz_sha256') != file_digest(nz):
                reason = 'output NPZ checksum mismatch'
            elif Path(summary.get('output_npz', '')).resolve() != nz.resolve():
                reason = 'output NPZ path mismatch'
        except (OSError, ValueError, TypeError) as exc:
            reason = f'unreadable cache: {exc}'
    if reason:
        raise RuntimeError(f'Cannot reuse {prefix}: {reason}; preserve this result and use a new output path, or explicitly request --overwrite.')
    return True
