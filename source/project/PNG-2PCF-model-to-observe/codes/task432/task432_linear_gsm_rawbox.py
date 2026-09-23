#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 4.3.2: large-scale linear-input GSM rawbox diagnostic.

本脚本实现一个不含 EFT 的、配置空间 Gaussian Streaming Model (GSM)
诊断，专门用于检验当前 rawbox 的大尺度 xi RSD closure。

科学定义
--------
1. P-side 完全复用当前 Task43 的低-k P02 模型、P-only sigma_s_P 和 sn0。
2. xi-side 不再把 k=3--5 h/Mpc 的 FullDiscrete Kaiser x FoG 当作 RSD
   theory，而是先由线性 real-space P(k) 计算
   xi_real(r)、v12(r)、sigma_parallel^2(r)、sigma_perpendicular^2(r)，
   再使用 pair-conserving Gaussian streaming integral。
3. fNL/b1 使用当前 PNG contract 共享，p=1，f_growth 固定。
4. pure 模式没有 xi-side 新的科学参数；optional ``--with-fog`` 模式只
   增加一个 eBOSS-style 的 sigma_FOG，作为配置空间 pairwise variance
   的 phenomenological 宽化，不是 EFT counterterm。
5. k_max_GSM 只截断线性 real-space/velocity-moment 积分，扫描范围由命令行
   指定，默认包含 0.15、0.25、0.50、0.75、1.0、2.0 h/Mpc。

文献依据
--------
- Scoccimarro 2004, arXiv:astro-ph/0407214: pair conservation 与 pairwise
  velocity PDF。
- Reid & White 2011, arXiv:1105.4165: scale-dependent Gaussian streaming
  mapping，并展示 linear-theory moments 输入的 GSM。
- Wang, Reid & White 2013, arXiv:1306.1804: CLPT moments + GSM。
- eBOSS LRG/ELG configuration-space analyses, arXiv:1909.07742、
  2007.08993、2007.09009、2007.09004: CLPT-GS/CLPT-GSRSD 在大尺度
  xi_ell 上的实际使用方式与 sigma_FOG nuisance。

边界
----
- 这是独立 task432 diagnostic，不覆盖 Task43 production 文件。
- 输出只写 task432_model_repair 子树；绘图只生成 PDF，不生成 PNG。
- 当前 linear velocity moments 是 leading-order tracer/velocity moments，
  不是完整 CLPT，也没有把 EFT、b2、bs2 等复杂参数加入 likelihood。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares
from scipy.special import spherical_jn


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_DIR = PROJECT_ROOT / "codes" / "task43"
TASK432_DIR = PROJECT_ROOT / "codes" / "task432"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "task432_model_repair"
PLOT_ROOT = PROJECT_ROOT / "plots" / "task43" / "rsd_validation" / "task432_model_repair"

for _path in (TASK43_DIR, TASK432_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from task43_rsd_common import rawbox_xi_primary_mask  # noqa: E402
from task432_split_sigma_rawbox import build_models_and_covariance  # noqa: E402
from task43_xi_gsm import gsm_points_fixed  # noqa: E402


DELTA_C = 1.686
P_FIXED = 1.0


def jsonable(value: Any) -> Any:
    """递归转换 numpy/path 对象，保证 scan audit 可以稳定写入 JSON。"""

    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写入 JSON，避免中断时留下半个结果文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class LinearRadialMomentProvider:
    """把 linear GSM 的 radial moments 提供给现有 pair-conserving integrator。

    ``xi``、``v12`` 和两个 velocity variance 使用线性 k-space sums 预先
    变换到 radial grid。对每个 likelihood 参数点，只需组合

        b(k) = b1 + q * alpha(k),
        q    = fNL * 2 * Delta_c * (b1 - p),

    的三个 xi basis 和两个 v12 basis，不需要重复构造二维 k-r 数组。
    这样可以把 k_max_GSM scan 用于实际 MAP diagnostic，而不是只做公式
    层面的 smoke test。
    """

    def __init__(
        self,
        cache_path: Path,
        *,
        kmax_gsm: float,
        radial_min: float = 0.5,
        radial_max: float = 1400.0,
        n_radial: int = 2800,
        radial_chunk: int = 128,
    ) -> None:
        if not np.isfinite(kmax_gsm) or kmax_gsm <= 0.0:
            raise ValueError("kmax_gsm must be positive")
        if not (0.0 < radial_min < radial_max) or int(n_radial) < 100:
            raise ValueError("invalid radial grid")

        with np.load(Path(cache_path), allow_pickle=False) as payload:
            k_all = np.asarray(payload["k_eff"], dtype="f8")
            g_nz_all = np.asarray(payload["g_nz"], dtype="f8")
            pk_all = np.asarray(payload["pk_dd"], dtype="f8")
            alpha_all = np.asarray(payload["alpha"], dtype="f8")
            self.f_growth = float(np.asarray(payload["f_growth"]).item())
            self.volume = float(np.asarray(payload["volume"]).item())

        keep = k_all <= float(kmax_gsm) * (1.0 + 1.0e-12)
        if int(np.count_nonzero(keep)) < 5:
            raise ValueError(f"kmax_gsm={kmax_gsm} leaves too few theory modes")

        self.k = k_all[keep]
        self.g_nz = g_nz_all[keep]
        self.pk = pk_all[keep]
        self.alpha = alpha_all[keep]
        self.kmax_gsm = float(kmax_gsm)
        self.n_modes = int(self.k.size)
        self.radial_grid = np.linspace(float(radial_min), float(radial_max), int(n_radial), dtype="f8")

        # The mode weights are exactly the same leading-order weights used by
        # task43_xi_linear_rsd.py.  Keeping this convention is more important
        # than replacing the frozen periodic cache with a new continuum cache.
        weighted = self.g_nz * self.pk
        weighted_over_k = weighted / self.k
        weighted_over_k2 = weighted / self.k**2
        weighted_alpha = weighted * self.alpha
        weighted_alpha_over_k = weighted_alpha / self.k
        weighted_alpha2 = weighted * self.alpha**2

        n = self.radial_grid.size
        self.xi_basis = np.empty((3, n), dtype="f8")
        self.v_basis = np.empty((2, n), dtype="f8")
        psi_parallel = np.empty(n, dtype="f8")
        psi_transverse = np.empty(n, dtype="f8")

        for start in range(0, n, int(radial_chunk)):
            stop = min(start + int(radial_chunk), n)
            radius = self.radial_grid[start:stop]
            kr = self.k[:, None] * radius[None, :]
            j0 = spherical_jn(0, kr)
            j1 = spherical_jn(1, kr)
            j1_over_kr = j1 / kr

            self.xi_basis[0, start:stop] = (weighted[:, None] * j0).sum(axis=0) / self.volume
            self.xi_basis[1, start:stop] = (weighted_alpha[:, None] * j0).sum(axis=0) / self.volume
            self.xi_basis[2, start:stop] = (weighted_alpha2[:, None] * j0).sum(axis=0) / self.volume

            self.v_basis[0, start:stop] = (
                -2.0 * self.f_growth * (weighted_over_k[:, None] * j1).sum(axis=0) / self.volume
            )
            self.v_basis[1, start:stop] = (
                -2.0 * self.f_growth * (weighted_alpha_over_k[:, None] * j1).sum(axis=0) / self.volume
            )

            psi_parallel[start:stop] = (
                self.f_growth**2
                * (weighted_over_k2[:, None] * (j0 - 2.0 * j1_over_kr)).sum(axis=0)
                / self.volume
            )
            psi_transverse[start:stop] = (
                self.f_growth**2
                * (weighted_over_k2[:, None] * j1_over_kr).sum(axis=0)
                / self.volume
            )

        sigma_u2 = self.f_growth**2 * float(np.sum(weighted_over_k2)) / (3.0 * self.volume)
        self.sigma_r2 = 2.0 * (sigma_u2 - psi_parallel)
        self.sigma_t2 = 2.0 * (sigma_u2 - psi_transverse)
        # Roundoff at r close to zero can produce tiny negative values.  A
        # negative physical variance is not accepted; only a numerical floor
        # at the scale of double precision is repaired here.
        variance_floor = max(abs(sigma_u2) * 1.0e-12, 1.0e-12)
        if np.min(self.sigma_r2) < -variance_floor or np.min(self.sigma_t2) < -variance_floor:
            raise RuntimeError(
                f"linear velocity moments became negative for kmax_gsm={kmax_gsm}: "
                f"min_parallel={np.min(self.sigma_r2)}, min_transverse={np.min(self.sigma_t2)}"
            )
        self.sigma_r2 = np.maximum(self.sigma_r2, variance_floor)
        self.sigma_t2 = np.maximum(self.sigma_t2, variance_floor)
        self.sigma_u2 = sigma_u2

        self._metadata = {
            "kmax_gsm_h_mpc": self.kmax_gsm,
            "n_modes": self.n_modes,
            "k_min_h_mpc": float(self.k.min()),
            "k_max_used_h_mpc": float(self.k.max()),
            "radial_min_mpc_h": float(self.radial_grid[0]),
            "radial_max_mpc_h": float(self.radial_grid[-1]),
            "n_radial": int(self.radial_grid.size),
            "sigma_u2_one_component": float(self.sigma_u2),
            "moment_definition": "leading linear moments; no EFT or nonlinear bias operators",
        }

    def metadata(self) -> dict[str, Any]:
        """返回机器可读的 moment-cache 说明。"""

        return dict(self._metadata)

    def provider(self, *, fnl: float, b1: float) -> "LinearRadialMoments":
        """按当前 PNG amplitude 组合一个可插值的 radial moment provider。"""

        q = float(fnl) * 2.0 * DELTA_C * (float(b1) - P_FIXED)
        xi = (
            float(b1) ** 2 * self.xi_basis[0]
            + 2.0 * float(b1) * q * self.xi_basis[1]
            + q**2 * self.xi_basis[2]
        )
        v_numerator = float(b1) * self.v_basis[0] + q * self.v_basis[1]
        v12 = v_numerator / (1.0 + xi)
        if np.any(1.0 + xi <= 0.0) or not np.all(np.isfinite(xi)):
            raise ValueError("linear PNG real-space pair density is non-positive")
        return LinearRadialMoments(
            radial_grid=self.radial_grid,
            xi_values=xi,
            v12_values=v12,
            sigma_r2_values=self.sigma_r2,
            sigma_t2_values=self.sigma_t2,
            metadata={"fnl": float(fnl), "b1": float(b1), **self.metadata()},
        )


class LinearRadialMoments:
    """Array-backed provider compatible with task43_xi_gsm.gsm_points_fixed."""

    def __init__(
        self,
        *,
        radial_grid: np.ndarray,
        xi_values: np.ndarray,
        v12_values: np.ndarray,
        sigma_r2_values: np.ndarray,
        sigma_t2_values: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        self.radial_grid = np.asarray(radial_grid, dtype="f8")
        self.xi_values = np.asarray(xi_values, dtype="f8")
        self.v12_values = np.asarray(v12_values, dtype="f8")
        self.sigma_r2_values = np.asarray(sigma_r2_values, dtype="f8")
        self.sigma_t2_values = np.asarray(sigma_t2_values, dtype="f8")
        self._metadata = dict(metadata)
        if any(value.shape != self.radial_grid.shape for value in (self.xi_values, self.v12_values, self.sigma_r2_values, self.sigma_t2_values)):
            raise ValueError("linear moment arrays do not share the radial grid")
        if np.any(1.0 + self.xi_values <= 0.0) or np.any(self.sigma_r2_values <= 0.0) or np.any(self.sigma_t2_values <= 0.0):
            raise ValueError("linear moment provider is non-physical on its radial grid")

    def at_los_array(
        self, transverse: np.ndarray, real_parallel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate xi, LOS mean displacement, and LOS variance at pair points."""

        perpendicular, los = np.broadcast_arrays(
            np.asarray(transverse, dtype="f8"), np.asarray(real_parallel, dtype="f8")
        )
        if np.any(perpendicular < 0.0) or not np.all(np.isfinite(perpendicular)) or not np.all(np.isfinite(los)):
            raise ValueError("invalid GSM pair coordinates")
        radius = np.hypot(perpendicular, los)
        # GSM tails can reach a formally tiny r even when the observed s is
        # large.  The radial support is deliberately broad; clipping only
        # protects the numerical tail and is audited through zmax convergence.
        radius_eval = np.clip(radius, self.radial_grid[0], self.radial_grid[-1])
        xi = np.interp(radius_eval, self.radial_grid, self.xi_values)
        v12 = np.interp(radius_eval, self.radial_grid, self.v12_values)
        sigma_r2 = np.interp(radius_eval, self.radial_grid, self.sigma_r2_values)
        sigma_t2 = np.interp(radius_eval, self.radial_grid, self.sigma_t2_values)
        mu_real = np.divide(los, radius, out=np.zeros_like(radius), where=radius > 0.0)
        mean = mu_real * v12
        variance = mu_real**2 * sigma_r2 + (1.0 - mu_real**2) * sigma_t2
        return xi, mean, variance


class LinearGSMModel:
    """Shell-averaged xi0/xi2 model built from linear radial moments."""

    def __init__(
        self,
        basis: LinearRadialMomentProvider,
        s_edges: np.ndarray,
        *,
        shell_order: int = 4,
        angular_order: int = 12,
        stream_order: int = 8,
        zmax: float = 8.0,
    ) -> None:
        self.basis = basis
        self.s_edges = np.asarray(s_edges, dtype="f8")
        self.shell_order = int(shell_order)
        self.angular_order = int(angular_order)
        self.stream_order = int(stream_order)
        self.zmax = float(zmax)
        if self.s_edges.ndim != 1 or np.any(np.diff(self.s_edges) <= 0.0):
            raise ValueError("invalid xi shell edges")
        if min(self.shell_order, self.angular_order, self.stream_order) < 2 or self.zmax <= 0.0:
            raise ValueError("invalid GSM quadrature setting")
        radial_nodes, radial_weights = np.polynomial.legendre.leggauss(self.shell_order)
        lo = self.s_edges[:-1, None]
        hi = self.s_edges[1:, None]
        self.shell_radius = 0.5 * (hi - lo) * radial_nodes[None, :] + 0.5 * (hi + lo)
        raw_weights = 0.5 * (hi - lo) * radial_weights[None, :] * self.shell_radius**2
        self.shell_weights = raw_weights / ((hi**3 - lo**3) / 3.0)
        self.mu_nodes, self.mu_weights = np.polynomial.legendre.leggauss(self.angular_order)
        self.shell_scale = float(np.sqrt(max(np.max(basis.sigma_r2), np.max(basis.sigma_t2))))

    def evaluate(
        self,
        *,
        fnl: float,
        b1: float,
        sigma_fog: float = 0.0,
        return_full: bool = False,
    ) -> dict[int, np.ndarray] | np.ndarray:
        if not np.isfinite(sigma_fog) or sigma_fog < 0.0:
            raise ValueError("sigma_fog must be finite and non-negative")
        provider = self.basis.provider(fnl=float(fnl), b1=float(b1))
        integration_scale = float(np.sqrt(self.shell_scale**2 + float(sigma_fog) ** 2))
        radii = self.shell_radius.ravel()
        mu = np.broadcast_to(self.mu_nodes[None, :], (radii.size, self.mu_nodes.size)).ravel()
        radius_grid = np.broadcast_to(radii[:, None], (radii.size, self.mu_nodes.size)).ravel()
        transverse = radius_grid * np.sqrt(np.maximum(1.0 - mu**2, 0.0))
        parallel = radius_grid * mu
        values = gsm_points_fixed(
            transverse,
            parallel,
            provider,
            integration_scale=integration_scale,
            zmax=self.zmax,
            quadrature_order=self.stream_order,
            extra_pair_variance=float(sigma_fog) ** 2,
            boxsize=None,
            periodic_images=0,
            chunk_size=2048,
        ).reshape(radii.size, self.mu_nodes.size)
        # xi_0 = (1/2) * integral_{-1}^{1} xi(s,mu) dmu.
        # The missing 1/2 would double the monopole while leaving the
        # quadrupole normalization unchanged, so keep this explicit here.
        xi0_nodes = 0.5 * (values @ self.mu_weights)
        xi2_nodes = values @ (5.0 * self.mu_weights * 0.5 * (3.0 * self.mu_nodes**2 - 1.0))
        xi0 = np.sum(self.shell_weights * xi0_nodes.reshape(self.shell_radius.shape), axis=1)
        xi2 = np.sum(self.shell_weights * xi2_nodes.reshape(self.shell_radius.shape), axis=1)
        result = {0: xi0, 2: xi2}
        if return_full:
            return result
        return result

    def vector(self, *, fnl: float, b1: float, sigma_fog: float = 0.0, mask: np.ndarray | None = None) -> np.ndarray:
        values = self.evaluate(fnl=fnl, b1=b1, sigma_fog=sigma_fog)
        if mask is None:
            return np.concatenate([np.asarray(values[0]), np.asarray(values[2])])
        return np.concatenate([np.asarray(values[0])[mask], np.asarray(values[2])[mask]])


def precision_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """在相关矩阵空间求逆，避免 P/xi 量纲跨度损坏小 xi 特征值。"""

    covariance = 0.5 * (np.asarray(covariance, dtype="f8") + np.asarray(covariance, dtype="f8").T)
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (correlation + correlation.T))
    floor = max(float(eigenvalues[-1]) * 1.0e-14, 1.0e-300)
    inv_corr = (eigenvectors * np.where(eigenvalues >= floor, 1.0 / np.maximum(eigenvalues, floor), 0.0)[None, :]) @ eigenvectors.T
    return inv_corr / np.outer(scale, scale)


def map_fit(
    label: str,
    data: np.ndarray,
    covariance: np.ndarray,
    evaluate: Callable[[np.ndarray], np.ndarray],
    bounds: tuple[np.ndarray, np.ndarray],
    starts: list[np.ndarray],
) -> dict[str, Any]:
    """执行 deterministic MAP fit，先避免 kmax scan 被长 MCMC 放大。"""

    precision = precision_from_covariance(covariance)

    def residual(theta: np.ndarray) -> np.ndarray:
        delta = np.asarray(data, dtype="f8") - np.asarray(evaluate(theta), dtype="f8")
        # 对正定 covariance 使用对称平方根，避免 Cholesky 顺序影响诊断。
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (precision + precision.T))
        floor = max(float(eigenvalues[-1]) * 1.0e-14, 1.0e-300)
        return (eigenvectors * np.sqrt(np.maximum(eigenvalues, floor))[None, :]).T @ delta

    solutions = [
        least_squares(
            residual,
            np.asarray(start, dtype="f8"),
            bounds=bounds,
            max_nfev=300,
            xtol=1.0e-8,
            ftol=1.0e-8,
            gtol=1.0e-8,
        )
        for start in starts
    ]
    best = min(solutions, key=lambda item: float(item.fun @ item.fun))
    theta = np.asarray(best.x, dtype="f8")
    return {
        "label": label,
        "map_theta": theta.tolist(),
        "map_chi2": float(best.fun @ best.fun),
        "optimizer_status": int(best.status),
        "optimizer_message": str(best.message),
        "n_function_evaluations": int(best.nfev),
    }


def evaluate_scan_point(
    *,
    products: dict[str, Any],
    gsm_model: LinearGSMModel,
    mask: np.ndarray,
    with_fog: bool,
) -> dict[str, Any]:
    """对一个 kmax_GSM 构造 P-only、xi-only 和 joint MAP。"""

    data_p = np.asarray(products["data"]["p"], dtype="f8")
    data_x = np.asarray(products["data"]["x"], dtype="f8")
    data_joint = np.concatenate([data_p, data_x])
    cov = products["covariances"]
    p_model = products["models"]["p"]

    def evaluate_p(theta: np.ndarray) -> np.ndarray:
        if with_fog:
            fnl, b1, sigma_p, _sigma_fog, sn0 = np.asarray(theta, dtype="f8")
        else:
            fnl, b1, sigma_p, sn0 = np.asarray(theta, dtype="f8")
        return np.asarray(p_model(np.asarray([fnl, b1, sigma_p, 0.0, sn0], dtype="f8")), dtype="f8")

    def evaluate_x(theta: np.ndarray) -> np.ndarray:
        theta_array = np.asarray(theta, dtype="f8")
        fnl, b1 = theta_array[:2]
        if with_fog:
            # xi marginal uses [fNL,b1,sigma_FOG], while joint uses
            # [fNL,b1,sigma_s_P,sigma_FOG,sn0].
            sigma_fog = theta_array[2] if theta_array.size == 3 else theta_array[3]
        else:
            sigma_fog = 0.0
        return gsm_model.vector(fnl=float(fnl), b1=float(b1), sigma_fog=float(sigma_fog), mask=mask)

    def evaluate_joint(theta: np.ndarray) -> np.ndarray:
        return np.concatenate([evaluate_p(theta), evaluate_x(theta)])

    if with_fog:
        bounds_p = (np.asarray([-500.0, 0.5, 0.0, 0.0, -1.0]), np.asarray([500.0, 5.0, 30.0, 30.0, 1.0]))
        bounds_x = (np.asarray([-500.0, 0.5, 0.0]), np.asarray([500.0, 5.0, 30.0]))
        bounds_joint = bounds_p
        p_starts = [np.asarray([0.0, 2.55, 1.0, 5.0, 0.0]), np.asarray([-20.0, 2.5, 2.0, 8.0, 0.1]), np.asarray([20.0, 2.6, 1.0, 3.0, -0.1])]
        x_starts = [np.asarray([0.0, 2.55, 5.0]), np.asarray([-20.0, 2.5, 8.0]), np.asarray([20.0, 2.6, 3.0])]
        joint_starts = p_starts
    else:
        bounds_p = (np.asarray([-500.0, 0.5, 0.0, -1.0]), np.asarray([500.0, 5.0, 30.0, 1.0]))
        bounds_x = (np.asarray([-500.0, 0.5]), np.asarray([500.0, 5.0]))
        bounds_joint = bounds_p
        p_starts = [np.asarray([0.0, 2.55, 1.0, 0.0]), np.asarray([-20.0, 2.5, 2.0, 0.1]), np.asarray([20.0, 2.6, 1.0, -0.1])]
        x_starts = [np.asarray([0.0, 2.55]), np.asarray([-20.0, 2.5]), np.asarray([20.0, 2.6])]
        joint_starts = p_starts

    fits = {
        "p_marginal": map_fit("p_marginal", data_p, cov["p"], evaluate_p, bounds_p, p_starts),
        "xi_marginal": map_fit("xi_marginal", data_x, cov["x"], evaluate_x, bounds_x, x_starts),
        "joint": map_fit("joint", data_joint, cov["joint"], evaluate_joint, bounds_joint, joint_starts),
    }
    return fits


def make_pdf(path: Path, scan_rows: list[dict[str, Any]], *, with_fog: bool) -> None:
    """生成 kmax scan PDF，不输出 PNG。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.pdf")
    kmax = np.asarray([row["kmax_gsm_h_mpc"] for row in scan_rows], dtype="f8")
    joint_chi2 = np.asarray([row["fits"]["joint"]["map_chi2"] for row in scan_rows], dtype="f8")
    joint_b1 = np.asarray([row["fits"]["joint"]["map_theta"][1] for row in scan_rows], dtype="f8")
    xi_b1 = np.asarray([row["fits"]["xi_marginal"]["map_theta"][1] for row in scan_rows], dtype="f8")
    p_b1 = np.asarray([row["fits"]["p_marginal"]["map_theta"][1] for row in scan_rows], dtype="f8")

    with PdfPages(temporary) as pdf:
        figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
        axes[0].plot(kmax, joint_chi2, marker="o", label="joint GSM")
        axes[0].set_xscale("log")
        axes[0].set_xlabel(r"$k_{\max}^{\rm GSM}\ [h\,\mathrm{Mpc}^{-1}]$")
        axes[0].set_ylabel(r"MAP $\chi^2$")
        axes[0].set_title("linear-input GSM cutoff scan")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(frameon=False)

        axes[1].plot(kmax, p_b1, marker="o", label="P marginal")
        axes[1].plot(kmax, xi_b1, marker="o", label="xi GSM marginal")
        axes[1].plot(kmax, joint_b1, marker="o", label="joint GSM")
        axes[1].set_xscale("log")
        axes[1].set_xlabel(r"$k_{\max}^{\rm GSM}\ [h\,\mathrm{Mpc}^{-1}]$")
        axes[1].set_ylabel(r"MAP $b_1$")
        axes[1].set_title("b1 cutoff stability")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(frameon=False)
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8.5, 5.0))
        axis.text(
            0.02,
            0.98,
            "Model: linear-input GSM\n"
            f"sigma_FOG enabled: {with_fog}\n"
            "Data: rawbox BAO-mask, 50 <= s < 350, 80 <= s < 120 removed\n"
            "P-side: current Task43 P02 model; xi-side: GSM moments only",
            transform=axis.transAxes,
            va="top",
            family="monospace",
        )
        axis.axis("off")
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    temporary.replace(path)


def main() -> None:
    """运行 linear GSM cutoff scan，并写入 JSON/PDF。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kmax-list",
        type=str,
        default="0.15,0.25,0.50,0.75,1.0,2.0",
        help="comma-separated k_max_GSM values in h/Mpc",
    )
    parser.add_argument("--with-fog", action="store_true", help="enable one eBOSS-style sigma_FOG nuisance")
    parser.add_argument("--shell-order", type=int, default=4)
    parser.add_argument("--angular-order", type=int, default=12)
    parser.add_argument("--stream-order", type=int, default=8)
    parser.add_argument("--zmax", type=float, default=8.0)
    parser.add_argument("--radial-n", type=int, default=2800)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT / "rawbox_linear_gsm")
    parser.add_argument("--smoke", action="store_true", help="use smaller radial/quadrature settings for a quick check")
    args = parser.parse_args()

    started = time.perf_counter()
    kmax_list = [float(item) for item in str(args.kmax_list).split(",") if item.strip()]
    if not kmax_list or any(value <= 0.0 or value > 3.0 for value in kmax_list):
        raise ValueError("kmax values must lie in (0, 3] h/Mpc for the frozen cache")
    if args.smoke:
        args.shell_order = min(int(args.shell_order), 2)
        args.angular_order = min(int(args.angular_order), 6)
        args.stream_order = min(int(args.stream_order), 4)
        args.radial_n = min(int(args.radial_n), 800)
        args.zmax = min(float(args.zmax), 6.0)

    products = build_models_and_covariance(xi_angle_mode="continuous", k_switch=0.01)
    cache_path = Path(products["metadata"]["cache"])
    # The frozen theory cache gives the actual xi shell edges.  Avoid deriving
    # them from data-vector length because the BAO mask removes bins.
    with np.load(cache_path, allow_pickle=False) as payload:
        s_edges = np.asarray(payload["s_edges"], dtype="f8")
    mask = rawbox_xi_primary_mask(0.5 * (s_edges[:-1] + s_edges[1:]))

    rows: list[dict[str, Any]] = []
    for kmax_gsm in kmax_list:
        basis = LinearRadialMomentProvider(
            cache_path,
            kmax_gsm=float(kmax_gsm),
            n_radial=int(args.radial_n),
        )
        gsm_model = LinearGSMModel(
            basis,
            s_edges,
            shell_order=int(args.shell_order),
            angular_order=int(args.angular_order),
            stream_order=int(args.stream_order),
            zmax=float(args.zmax),
        )
        fits = evaluate_scan_point(products=products, gsm_model=gsm_model, mask=mask, with_fog=bool(args.with_fog))
        rows.append(
            {
                "kmax_gsm_h_mpc": float(kmax_gsm),
                "moment_cache": basis.metadata(),
                "quadrature": {
                    "shell_order": int(args.shell_order),
                    "angular_order": int(args.angular_order),
                    "stream_order": int(args.stream_order),
                    "zmax": float(args.zmax),
                    "radial_n": int(args.radial_n),
                },
                "fits": fits,
            }
        )
        print(json.dumps({"kmax_gsm": kmax_gsm, "joint_chi2": fits["joint"]["map_chi2"]}, sort_keys=True))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    audit = {
        "task": "Task 4.3.2 rawbox linear-input GSM kmax scan",
        "status": "smoke_complete" if args.smoke else "complete",
        "literature_contract": {
            "streaming_model": "Scoccimarro 2004 pair conservation; Reid & White 2011 GSM",
            "eBOSS_configuration_space_precedent": "CLPT-GS/CLPT-GSRSD; Alam et al. 2020 mock challenge fit 32--160 h^-1 Mpc",
            "eft": False,
        },
        "data_contract": products["metadata"],
        "model_contract": {
            "shared_parameters": ["fNL", "b1"],
            "p_side_parameters": ["sigma_s_P", "sn0"],
            "xi_side_parameters": ["sigma_FOG"] if args.with_fog else [],
            "p_fixed": P_FIXED,
            "sigma_fog_units": "Mpc/h displacement; added as sigma_FOG^2 to LOS pair variance",
            "kmax_scan_h_mpc": kmax_list,
        },
        "scan_rows": rows,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    atomic_json(output_root / "audits" / ("task432_linear_gsm_kmax_scan_fog.json" if args.with_fog else "task432_linear_gsm_kmax_scan.json"), audit)
    make_pdf(
        PLOT_ROOT / ("task432_linear_gsm_kmax_scan_fog.pdf" if args.with_fog else "task432_linear_gsm_kmax_scan.pdf"),
        rows,
        with_fog=bool(args.with_fog),
    )
    print(json.dumps({"status": audit["status"], "output_root": str(output_root), "n_kmax": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
