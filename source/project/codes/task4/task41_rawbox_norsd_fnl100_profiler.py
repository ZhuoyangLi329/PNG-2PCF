#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码大纲（执行逻辑关系）：

1. 读取输入：
   - fnl100 no-RSD raw periodic box 的 mean xi / covariance / realization 列表；
   - 同一批 realization 的 real-space P(k) 文件。
2. 构建 real-space PNG tracer P(k) 理论：
   - 用 desilike/cosmoprimo 的 FixedPowerSpectrumTemplate 得到 P_dd(k) 和 alpha(k)；
   - 自己显式写 no-RSD 公式 P_h(k) = [b1 + 2 delta_c (b1-p) fnl alpha(k)]^2 P_dd(k) + sn0。
3. 重新计算 p=1.1 口径下的 P(k) BinAvgFit 参考点。
4. 构建 FullDiscrete/CachedRebin 算子，把任意 (fnl_loc, b1) 映射为 xi_model(s)。
5. 对若干 r-bin 选择运行 2PCF profiler：
   - 数据为 ensemble mean xi；
   - 默认 covariance 为 C_sample，用于单个 3Gpc raw box 的 parameter-constraint test；
   - 可选 covariance 为 C_sample / Nmock，只用于 theory-mean diagnostic；
   - inverse covariance 使用 Hartlap 修正；
   - 参数 error 使用 Percival correction 放大。
6. 写出 JSON/NPZ summary，供后续 MCMC 使用。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from iminuit import Minuit
from scipy.fft import irfft, next_fast_len, rfft

from cosmoprimo import Cosmology
try:
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate
except ImportError:
    FixedPowerSpectrumTemplate = None


# ======================== 路径与固定参数 ========================
PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task4_outputs" / "rawbox_norsd_fnl100_profiler_p1p1"

PK_DIR = PROJECT_ROOT / "outputs" / "task18_outputs" / "fnl100_box_realspace_pk" / "pk"
XI_MEAN_NPZ = (
    PROJECT_ROOT
    / "outputs"
    / "task171_outputs"
    / "box_mean_current_norsd_fnl100"
    / "task171_current_fnl100_box_vs_cutsky_mean_xi.npz"
)

OUT_JSON_NAME = "task41_rawbox_norsd_fnl100_profiler_summary.json"
OUT_NPZ_NAME = "task41_rawbox_norsd_fnl100_profiler_arrays.npz"

BOX_SIZE = 3000.0
K_FUND = 2.0 * np.pi / BOX_SIZE
VOLUME = BOX_SIZE**3
UNIT_Z = 1.0

P_FIXED = 1.1
SN0_FIXED = 0.0

N_PK_FIT = 20
K_CEN_COL = 0
K_MIN_COL = 1
K_MAX_COL = 2
P_COL = 5

KMAX_DISCRETE = 15.0
N_DENSE_DEFAULT = 300_000
REBIN_DK_FACTOR = 0.1

N_PARAMS = 2
# =================================================================


def ensure_output_tree() -> None:
    """创建输出目录。"""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def realization_label(realization: int) -> str:
    """把 realization 编号转成文件名使用的 `Nxxx` 格式。"""
    return f"N{int(realization):03d}"


def load_xi_mean_payload() -> dict[str, np.ndarray]:
    """
    读取 fnl100 no-RSD raw box 的 mean xi 与 covariance。

    返回：
    - realizations: realization 编号
    - s: 2PCF 分离距离 bin 中心
    - xi_mean: ensemble mean xi
    - cov_sample: 单个 realization 的 sample covariance
    """
    data = np.load(XI_MEAN_NPZ)
    return {
        "realizations": data["realizations"].astype(int),
        "s": data["s"].astype("f8"),
        "xi_mean": data["mean_box"].astype("f8"),
        "cov_sample": data["cov_box"].astype("f8"),
    }


def load_pk_ensemble(realizations: np.ndarray) -> dict[str, np.ndarray]:
    """
    读取同一批 realization 的 P(k) 文件，用于 P(k) BinAvgFit reference。
    """
    arrays = []
    kcen = kmin = kmax = None
    missing: list[str] = []

    for realization in realizations:
        path = PK_DIR / f"pk_realspace_fnl100_{realization_label(int(realization))}.dat"
        if not path.exists():
            missing.append(str(path))
            continue
        data = np.loadtxt(path, comments="#")
        if kcen is None:
            kcen = data[:N_PK_FIT, K_CEN_COL].copy()
            kmin = data[:N_PK_FIT, K_MIN_COL].copy()
            kmax = data[:N_PK_FIT, K_MAX_COL].copy()
        arrays.append(data[:N_PK_FIT, P_COL].copy())

    if missing:
        raise FileNotFoundError("缺少 P(k) 文件：\n" + "\n".join(missing[:10]))
    if not arrays:
        raise RuntimeError(f"没有从 {PK_DIR} 读取到 P(k) 文件")

    mocks = np.vstack(arrays).astype("f8")
    return {
        "kcen": kcen,
        "kmin": kmin,
        "kmax": kmax,
        "mocks": mocks,
        "mean": mocks.mean(axis=0),
        "cov": np.cov(mocks, rowvar=False, ddof=1),
    }


def build_cosmology() -> Cosmology:
    """构建与 task18/Mission10 相同的 cosmoprimo 宇宙学。"""
    return Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )


def build_template_arrays(k_template: np.ndarray, z: float) -> dict[str, np.ndarray]:
    """
    从 desilike 固定模板提取 real-space 公式需要的 P_dd(k) 和 alpha(k)。
    """
    cosmo = build_cosmology()
    if FixedPowerSpectrumTemplate is not None:
        template = FixedPowerSpectrumTemplate(z=z, fiducial=cosmo, k=k_template)
        template()
        kin = np.asarray(template.k, dtype="f8")
        pk_dd = np.asarray(template.pk_dd, dtype="f8")
    else:
        # Newer desilike builds renamed the fixed template API.  For this
        # no-RSD real-space model we only need the linear matter P(k), which
        # cosmoprimo can provide directly on the same k grid.
        kin = np.asarray(k_template, dtype="f8")
        pk_dd = np.asarray(cosmo.get_fourier().pk_interpolator(non_linear=False, of="delta_m")(kin, z=z), dtype="f8")
    pk_prim = cosmo.get_primordial(mode="scalar").pk_interpolator()(kin)
    pphi_prim = 9.0 / 25.0 * 2.0 * np.pi**2 / kin**3 * pk_prim / cosmo.h**3
    alpha = 1.0 / np.sqrt(pk_dd / pphi_prim)
    return {"k": kin, "pk_dd": pk_dd, "alpha": alpha}


def interp_logk(k_query: np.ndarray, k_base: np.ndarray, y_base: np.ndarray) -> np.ndarray:
    """在 log10(k) 上插值，保持低 k 端平滑。"""
    return np.interp(np.log10(k_query), np.log10(k_base), y_base)


def evaluate_realspace_png_pk(
    k_query: np.ndarray,
    template_arrays: dict[str, np.ndarray],
    *,
    fnl_loc: float,
    b1: float,
    p_fixed: float,
    sn0: float,
) -> np.ndarray:
    """
    计算 no-RSD halo/tracer P(k)。

    参数：
    - k_query: 要评估的 k 网格
    - template_arrays: `build_template_arrays()` 输出
    - fnl_loc, b1: 本轮 profiler 的自由参数
    - p_fixed: 固定的 PNG bias 参数
    - sn0: 固定 shot noise

    返回：
    - P_h(k_query)
    """
    alpha = interp_logk(k_query, template_arrays["k"], template_arrays["alpha"])
    pk_dd = interp_logk(k_query, template_arrays["k"], template_arrays["pk_dd"])
    delta_c = 1.686
    bphi = 2.0 * delta_c * (float(b1) - float(p_fixed))
    bias = float(b1) + bphi * float(fnl_loc) * alpha
    return bias**2 * pk_dd + float(sn0)


def gq_enumerate(qmax: int) -> np.ndarray:
    """直接枚举小 q 区间的壳层简并度，用于 P(k) BinAvgFit。"""
    nmax = int(np.ceil(np.sqrt(qmax))) + 1
    gq = np.zeros(qmax + 1, dtype=np.int64)
    for nx in range(-nmax, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            for nz in range(-nmax, nmax + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                q = nx * nx + ny * ny + nz * nz
                if q <= qmax:
                    gq[q] += 1
    return gq


def build_bin_shell_index(
    kf: float,
    gq: np.ndarray,
    kmin_bin: np.ndarray,
    kmax_bin: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    为每个 P(k) bin 找到落入其中的离散 k-shell 及其简并度。
    """
    q_all = np.nonzero(gq[1:])[0] + 1
    k_all = kf * np.sqrt(q_all.astype("f8"))
    g_all = gq[q_all].astype("f8")
    k_shells: list[np.ndarray] = []
    g_shells: list[np.ndarray] = []
    for lo, hi in zip(kmin_bin, kmax_bin):
        mask = (k_all >= lo) & (k_all < hi)
        k_shells.append(k_all[mask])
        g_shells.append(g_all[mask])
    return k_shells, g_shells


def compute_binavg_pk(
    template_arrays: dict[str, np.ndarray],
    k_shells: list[np.ndarray],
    g_shells: list[np.ndarray],
    *,
    fnl_loc: float,
    b1: float,
) -> np.ndarray:
    """计算与测量 P(k) bin 定义一致的理论 bin-average P(k)。"""
    all_k = np.unique(np.concatenate(k_shells))
    pk_all = evaluate_realspace_png_pk(
        all_k,
        template_arrays,
        fnl_loc=fnl_loc,
        b1=b1,
        p_fixed=P_FIXED,
        sn0=SN0_FIXED,
    )
    pk_map = dict(zip(all_k, pk_all))

    out = np.zeros(len(k_shells), dtype="f8")
    for i, (ks, gs) in enumerate(zip(k_shells, g_shells)):
        values = np.array([pk_map[k] for k in ks], dtype="f8")
        out[i] = np.sum(gs * values) / np.sum(gs)
    return out


def fit_pk_reference(pk_data: dict[str, np.ndarray], template_arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """
    重新计算 p=1.1 口径下的 P(k) center fit 与 BinAvgFit reference。
    """
    cov_inv = np.linalg.pinv(pk_data["cov"], rcond=1e-10)
    kcen = pk_data["kcen"]
    pk_mean = pk_data["mean"]

    def chi2_center(fnl_loc: float, b1: float) -> float:
        model = evaluate_realspace_png_pk(
            kcen,
            template_arrays,
            fnl_loc=fnl_loc,
            b1=b1,
            p_fixed=P_FIXED,
            sn0=SN0_FIXED,
        )
        diff = pk_mean - model
        return float(diff @ cov_inv @ diff)

    m_center = Minuit(chi2_center, fnl_loc=100.0, b1=2.7)
    m_center.errordef = 1.0
    m_center.limits["fnl_loc"] = (-500.0, 500.0)
    m_center.limits["b1"] = (0.1, None)
    m_center.migrad()

    qmax_fit = int(np.floor((float(pk_data["kmax"][-1]) / K_FUND) ** 2)) + 1
    gq_fit = gq_enumerate(qmax_fit)
    k_shells, g_shells = build_bin_shell_index(K_FUND, gq_fit, pk_data["kmin"], pk_data["kmax"])

    def chi2_binavg(fnl_loc: float, b1: float) -> float:
        model = compute_binavg_pk(
            template_arrays,
            k_shells,
            g_shells,
            fnl_loc=fnl_loc,
            b1=b1,
        )
        diff = pk_mean - model
        return float(diff @ cov_inv @ diff)

    m_bin = Minuit(
        chi2_binavg,
        fnl_loc=float(m_center.values["fnl_loc"]),
        b1=float(m_center.values["b1"]),
    )
    m_bin.errordef = 1.0
    m_bin.limits["fnl_loc"] = (-500.0, 500.0)
    m_bin.limits["b1"] = (0.1, None)
    m_bin.migrad()

    return {
        "center": {
            "fnl_loc": float(m_center.values["fnl_loc"]),
            "b1": float(m_center.values["b1"]),
            "chi2": float(m_center.fval),
            "ndof": int(N_PK_FIT - N_PARAMS),
            "valid": bool(m_center.fmin.is_valid),
        },
        "binavg": {
            "fnl_loc": float(m_bin.values["fnl_loc"]),
            "b1": float(m_bin.values["b1"]),
            "chi2": float(m_bin.fval),
            "ndof": int(N_PK_FIT - N_PARAMS),
            "valid": bool(m_bin.fmin.is_valid),
        },
    }


def gq_fft(qmax: int, nmax: int) -> np.ndarray:
    """FFT 卷积精确计算大 q 区间的壳层简并度。"""
    a = np.zeros(qmax + 1, dtype=np.float32)
    a[0] = 1.0
    sq = np.arange(1, nmax + 1, dtype=np.int64) ** 2
    a[sq[sq <= qmax]] = 2.0
    nfft = next_fast_len(3 * qmax + 1)
    fa = rfft(a, n=nfft)
    return np.rint(irfft(fa * fa * fa, n=nfft)[: qmax + 1]).astype(np.int64)


def precompute_rebin_cache(
    gq: np.ndarray,
    kf: float,
    kmax: float,
    dk_factor: float = REBIN_DK_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """预计算 FullDiscrete/CachedRebin 的 `G_bin` 和 `k_eff`。"""
    qnz = np.nonzero(gq[1:])[0] + 1
    kv = kf * np.sqrt(qnz.astype("f8"))
    g = gq[qnz].astype("f8")
    dk = dk_factor * kf
    nbins = int(np.ceil(kmax / dk)) + 1
    bin_idx = np.clip((kv / dk).astype(np.int64), 0, nbins - 1)
    g_bin = np.bincount(bin_idx, weights=g, minlength=nbins)
    gk_bin = np.bincount(bin_idx, weights=g * kv, minlength=nbins)
    mask = g_bin > 0
    return g_bin[mask].astype("f8"), gk_bin[mask] / g_bin[mask]


def fast_discrete_xi(
    s: np.ndarray,
    g_nz: np.ndarray,
    k_eff: np.ndarray,
    kd: np.ndarray,
    pd: np.ndarray,
) -> np.ndarray:
    """用 CachedRebin 快速计算 FullDiscrete xi(r)。"""
    weight = g_nz * interp_logk(k_eff, kd, pd)
    arg = np.outer(k_eff, s)
    kernel = np.ones_like(arg)
    mask = arg != 0.0
    kernel[mask] = np.sin(arg[mask]) / arg[mask]
    return (weight @ kernel) / VOLUME


def covariance_corrections(nmock: int, ndata: int, nparams: int) -> dict[str, float]:
    """计算 Hartlap 与 Percival correction。"""
    if nmock <= ndata + 4:
        raise ValueError(f"Nmock={nmock} 对 Ndata={ndata} 太少，无法稳定做 covariance correction")
    hartlap = (nmock - ndata - 2.0) / (nmock - 1.0)
    a = 2.0 / ((nmock - ndata - 1.0) * (nmock - ndata - 4.0))
    b = (nmock - ndata - 2.0) / ((nmock - ndata - 1.0) * (nmock - ndata - 4.0))
    m1 = (1.0 + b * (ndata - nparams)) / (1.0 + a + b * (nparams + 1.0))
    return {
        "hartlap": float(hartlap),
        "percival_m1_variance": float(m1),
        "percival_error_factor": float(math.sqrt(m1)),
        "A": float(a),
        "B": float(b),
    }


def build_bin_selections(s: np.ndarray) -> dict[str, np.ndarray]:
    """定义本轮 profiler 使用的 r-bin 子集。"""
    all_idx = np.arange(s.size)
    return {
        "largest20": all_idx[-20:],
        "largest18": all_idx[-18:],
        "largest16": all_idx[-16:],
        "largest14": all_idx[-14:],
        "broad_step20_85_345": all_idx[::2],
    }


def run_xi_profiler(
    *,
    selection_name: str,
    indices: np.ndarray,
    xi_payload: dict[str, np.ndarray],
    template_arrays: dict[str, np.ndarray],
    g_nz: np.ndarray,
    k_eff: np.ndarray,
    k_dense: np.ndarray,
    pk_reference: dict[str, object],
    covariance_policy: str,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """
    对一个 r-bin 选择运行 2PCF profiler。

    返回：
    - summary 字典
    - best-fit xi_model，长度等于该 selection 的数据点数
    """
    s_sel = xi_payload["s"][indices]
    data_sel = xi_payload["xi_mean"][indices]
    cov_individual = xi_payload["cov_sample"][np.ix_(indices, indices)]
    nmock = int(xi_payload["realizations"].size)
    ndata = int(indices.size)

    corrections = covariance_corrections(nmock, ndata, N_PARAMS)
    if covariance_policy == "mean":
        covariance_used = cov_individual / float(nmock)
    elif covariance_policy == "single":
        covariance_used = cov_individual
    else:
        raise ValueError(f"unknown covariance_policy={covariance_policy}")

    cov_inv_raw = np.linalg.pinv(covariance_used, rcond=1e-10)
    cov_inv = corrections["hartlap"] * cov_inv_raw
    cond = float(np.linalg.cond(covariance_used))

    def model_xi(fnl_loc: float, b1: float) -> np.ndarray:
        pd = evaluate_realspace_png_pk(
            k_dense,
            template_arrays,
            fnl_loc=fnl_loc,
            b1=b1,
            p_fixed=P_FIXED,
            sn0=SN0_FIXED,
        )
        return fast_discrete_xi(s_sel, g_nz, k_eff, k_dense, pd)

    def chi2(fnl_loc: float, b1: float) -> float:
        diff = data_sel - model_xi(fnl_loc, b1)
        return float(diff @ cov_inv @ diff)

    ref = pk_reference["binavg"]
    ref_fnl = float(ref["fnl_loc"])
    ref_b1 = float(ref["b1"])
    xi_ref = model_xi(ref_fnl, ref_b1)
    ref_diff = data_sel - xi_ref
    ref_chi2 = float(ref_diff @ cov_inv @ ref_diff)

    m = Minuit(chi2, fnl_loc=ref_fnl, b1=ref_b1)
    m.errordef = 1.0
    m.limits["fnl_loc"] = (-500.0, 500.0)
    m.limits["b1"] = (0.1, None)
    t0 = time.time()
    m.migrad()
    m.hesse()
    elapsed = time.time() - t0

    best_fnl = float(m.values["fnl_loc"])
    best_b1 = float(m.values["b1"])
    err_fnl_raw = float(m.errors["fnl_loc"])
    err_b1_raw = float(m.errors["b1"])
    err_fac = corrections["percival_error_factor"]
    err_fnl = err_fnl_raw * err_fac
    err_b1 = err_b1_raw * err_fac
    xi_best = model_xi(best_fnl, best_b1)

    delta_fnl = best_fnl - float(ref["fnl_loc"])
    delta_b1 = best_b1 - float(ref["b1"])
    ndof = ndata - N_PARAMS
    summary = {
        "selection": selection_name,
        "ndata": ndata,
        "s_min": float(s_sel.min()),
        "s_max": float(s_sel.max()),
        "indices": indices.astype(int).tolist(),
        "nmock": nmock,
        "covariance": {
            "policy": covariance_policy,
            "policy_description": (
                "C_sample / Nmock for ensemble-mean systematic test"
                if covariance_policy == "mean"
                else "C_sample for single-survey parameter-constraint test"
            ),
            "condition_number": cond,
            **corrections,
        },
        "bestfit": {
            "fnl_loc": best_fnl,
            "b1": best_b1,
            "p": P_FIXED,
            "sn0": SN0_FIXED,
            "chi2": float(m.fval),
            "ndof": int(ndof),
            "chi2_per_dof": float(m.fval / ndof) if ndof > 0 else float("nan"),
            "valid": bool(m.fmin.is_valid),
            "has_accurate_covar": bool(m.fmin.has_accurate_covar),
        },
        "errors": {
            "fnl_loc_raw": err_fnl_raw,
            "b1_raw": err_b1_raw,
            "fnl_loc_percival": err_fnl,
            "b1_percival": err_b1,
        },
        "reference_delta": {
            "reference": "pk_binavg_p1p1",
            "delta_fnl_loc": delta_fnl,
            "delta_b1": delta_b1,
            "pull_fnl_loc": float(delta_fnl / err_fnl) if err_fnl > 0 else float("nan"),
            "pull_b1": float(delta_b1 / err_b1) if err_b1 > 0 else float("nan"),
            "chi2_at_reference": ref_chi2,
            "chi2_per_dof_at_reference": float(ref_chi2 / ndof) if ndof > 0 else float("nan"),
        },
        "timing_sec": elapsed,
    }
    return summary, xi_best, xi_ref


def evaluate_acceptance(results: list[dict[str, object]]) -> dict[str, object]:
    """根据 goal 文档中的最低标准生成机器可读 pass/fail 标记。"""
    by_name = {item["selection"]: item for item in results}
    fid = by_name["largest20"]
    fid_err = fid["errors"]["fnl_loc_percival"]
    fid_fnl = fid["bestfit"]["fnl_loc"]

    fid_pass = (
        bool(fid["bestfit"]["valid"])
        and abs(fid["reference_delta"]["pull_fnl_loc"]) < 1.0
        and abs(fid["reference_delta"]["pull_b1"]) < 1.0
    )

    drift_rows = []
    drift_pass = True
    for name in ["largest18", "largest16", "largest14", "broad_step20_85_345"]:
        row = by_name[name]
        drift = float(row["bestfit"]["fnl_loc"] - fid_fnl)
        drift_sigma = float(drift / fid_err) if fid_err > 0 else float("nan")
        if name != "broad_step20_85_345" and abs(drift_sigma) > 0.5:
            drift_pass = False
        drift_rows.append({"selection": name, "delta_from_fiducial": drift, "delta_in_fid_sigma": drift_sigma})

    return {
        "fiducial_reference_pull_pass": bool(fid_pass),
        "r_range_drift_pass": bool(drift_pass),
        "overall_pass": bool(fid_pass and drift_pass),
        "drift_relative_to_largest20": drift_rows,
    }


def main() -> None:
    """主入口。"""
    global KMAX_DISCRETE

    parser = argparse.ArgumentParser(description="Task4.1 fnl100 no-RSD raw-box 2PCF profiler validation")
    parser.add_argument("--n-dense", type=int, default=N_DENSE_DEFAULT, help="dense k grid size for P(k) interpolation")
    parser.add_argument("--kmax-discrete", type=float, default=KMAX_DISCRETE, help="FullDiscrete kmax in h/Mpc")
    parser.add_argument(
        "--covariance-policy",
        choices=("mean", "single"),
        default="single",
        help="single uses C_sample for the main likelihood; mean uses C_sample/Nmock only as a diagnostic",
    )
    parser.add_argument(
        "--output-label",
        default="",
        help="optional suffix inserted before .json/.npz, e.g. kmax20",
    )
    args = parser.parse_args()
    KMAX_DISCRETE = float(args.kmax_discrete)

    if args.output_label:
        out_json = OUTPUT_ROOT / f"task41_rawbox_norsd_fnl100_profiler_summary_{args.output_label}.json"
        out_npz = OUTPUT_ROOT / f"task41_rawbox_norsd_fnl100_profiler_arrays_{args.output_label}.npz"
    else:
        out_json = OUTPUT_ROOT / OUT_JSON_NAME
        out_npz = OUTPUT_ROOT / OUT_NPZ_NAME

    ensure_output_tree()
    print("[read] xi payload")
    xi_payload = load_xi_mean_payload()
    print(f"[read] Nmock={xi_payload['realizations'].size}, s bins={xi_payload['s'].size}")

    print("[read] P(k) ensemble")
    pk_data = load_pk_ensemble(xi_payload["realizations"])

    print("[theory] build template arrays")
    k_template = np.geomspace(min(1e-4, K_FUND / 2.0), max(1.0, KMAX_DISCRETE * 1.2), 4000)
    template_arrays = build_template_arrays(k_template, z=UNIT_Z)

    print("[reference] fit P(k) reference with p=1.1")
    pk_reference = fit_pk_reference(pk_data, template_arrays)
    print("[reference]", pk_reference["binavg"])

    print("[full-discrete] precompute gq/cache")
    qmax_fd = int((KMAX_DISCRETE / K_FUND) ** 2)
    nmax_fd = int(KMAX_DISCRETE / K_FUND)
    t0 = time.time()
    gq_fd = gq_fft(qmax_fd, nmax_fd)
    g_nz, k_eff = precompute_rebin_cache(gq_fd, K_FUND, KMAX_DISCRETE, dk_factor=REBIN_DK_FACTOR)
    cache_sec = time.time() - t0
    print(f"[full-discrete] cache bins={len(k_eff)}, sec={cache_sec:.2f}")

    k_dense = np.geomspace(K_FUND * 0.5, KMAX_DISCRETE * 1.1, int(args.n_dense))
    selections = build_bin_selections(xi_payload["s"])

    result_rows = []
    npz_arrays: dict[str, np.ndarray] = {
        "s_all": xi_payload["s"],
        "xi_mean_all": xi_payload["xi_mean"],
        "cov_sample_all": xi_payload["cov_sample"],
        "realizations": xi_payload["realizations"],
        "k_dense": k_dense,
    }

    for name, indices in selections.items():
        print(f"[profiler] {name}: N={indices.size}, s=[{xi_payload['s'][indices].min()}, {xi_payload['s'][indices].max()}]")
        row, xi_best, xi_ref = run_xi_profiler(
            selection_name=name,
            indices=indices,
            xi_payload=xi_payload,
            template_arrays=template_arrays,
            g_nz=g_nz,
            k_eff=k_eff,
            k_dense=k_dense,
            pk_reference=pk_reference,
            covariance_policy=args.covariance_policy,
        )
        print(
            f"  fnl={row['bestfit']['fnl_loc']:.3f} +/- {row['errors']['fnl_loc_percival']:.3f}, "
            f"pull={row['reference_delta']['pull_fnl_loc']:.3f}, chi2/dof={row['bestfit']['chi2_per_dof']:.3f}"
        )
        result_rows.append(row)
        npz_arrays[f"{name}_indices"] = indices.astype("i8")
        npz_arrays[f"{name}_s"] = xi_payload["s"][indices]
        npz_arrays[f"{name}_xi_data"] = xi_payload["xi_mean"][indices]
        npz_arrays[f"{name}_xi_model_best"] = xi_best
        npz_arrays[f"{name}_xi_model_pk_reference"] = xi_ref

    acceptance = evaluate_acceptance(result_rows)
    summary = {
        "task": "task41_rawbox_norsd_fnl100_profiler",
        "status": "done",
        "inputs": {
            "xi_mean_npz": str(XI_MEAN_NPZ),
            "pk_dir": str(PK_DIR),
            "nmock": int(xi_payload["realizations"].size),
            "realization_first": int(xi_payload["realizations"][0]),
            "realization_last": int(xi_payload["realizations"][-1]),
        },
        "fixed_parameters": {
            "p": P_FIXED,
            "sn0": SN0_FIXED,
            "box_size": BOX_SIZE,
            "kfund": K_FUND,
            "unit_z": UNIT_Z,
            "kmax_discrete": KMAX_DISCRETE,
            "rebin_dk_factor": REBIN_DK_FACTOR,
            "n_dense": int(args.n_dense),
            "covariance_policy": args.covariance_policy,
        },
        "free_parameters": ["fnl_loc", "b1"],
        "pk_reference": pk_reference,
        "profiler_results": result_rows,
        "acceptance": acceptance,
        "outputs": {
            "json": str(out_json),
            "npz": str(out_npz),
        },
        "timing": {
            "cache_sec": cache_sec,
        },
    }

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(out_npz, **npz_arrays)
    print(f"[write] {out_json}")
    print(f"[write] {out_npz}")
    print(f"[acceptance] {acceptance}")


if __name__ == "__main__":
    main()
