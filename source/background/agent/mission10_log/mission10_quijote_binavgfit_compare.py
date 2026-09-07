#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission 10: Quijote 不同 fNL 样本上验证 BinAvgFit+FullDiscrete
=============================================================

本脚本的目标
-----------
你要求“再看看 quijote simulation 的不同 fnl 样本的效果”。这里我们在 Quijote 1Gpc N-body
数据上，对不同 fnl（0,20,30,50,75,100）比较三种方法的 2PCF 建模效果：

1) FullDiscrete(std fit)
   - 先用 desilike 的标准口径拟合 P(k)：理论用 bin center 的 P_model(k_center)
   - 再用 FullDiscrete 离散求和得到 xi0(r)

2) DataBin(20)
   - 与 Mission 9 一致：在离散求和中，把“拟合范围内的前 20 个 bin”用测量的 P_data_bin 替换
   - 注意：DataBin 不是纯 forward model（依赖 P_measure）

3) BinAvgFit+FullDiscrete (new)
   - 先做 BinAvgFit：拟合阶段把理论也变成“离散模式 g_q 加权的 bin-average”
     用 P_model_bin(i) 去拟合 P_data_bin(i)，从源头消除 bin-average vs 点值的不自洽
   - 再用标准 FullDiscrete 离散求和（求和阶段不喂 P_measure）

输出
----
1) 两张图（低/高 fnl 分组，便于看清楚）：
   - mission10_quijote_binavgfit_compare_lowfnl.png  (fnl=0,20,30)
   - mission10_quijote_binavgfit_compare_highfnl.png (fnl=50,75,100)

2) 一份 markdown 表格（包含 best-fit 与指标）：
   - mission10_quijote_binavgfit_results.md

运行方式（需要 desilike 环境）
------------------------------
bash -lc "conda activate desilike && PYTHONUNBUFFERED=1 python -u mission10_quijote_binavgfit_compare.py"

注意
----
1) 本脚本不会删除任何文件。
2) 若出现缺包报错，请先不要安装包，先确认是否在其他 conda 环境里已有。
"""

from __future__ import annotations

import os
import glob
import re
import time
from dataclasses import dataclass

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.fft import next_fast_len, rfft, irfft


OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Quijote box
LBOX = 1000.0
K_FUND = 2.0 * np.pi / LBOX
VOLUME = LBOX**3

# Modeling setup (match Mission 9)
Z = 1.0
P_FIX = 1.2
KMAX_DISCRETE = 15.0
N_DENSE = 300_000
FIT_NDP = 20  # 只用前 20 个 k-bin 拟合（与 mission9_quijote_validation.py 一致）

# Data path
BASE_DIR = "/pscratch/sd/l/lzy/pks_2pcfs"
INTERP_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission6_log/nbody_validation/interpolated"

# fnl 配置：fnl_true -> (label, pk_pattern, pcf_pattern, cov_pk_pattern)
FNL_CONFIGS = {
    0: ("fid", f"{BASE_DIR}/pk_fid_*.txt", f"{BASE_DIR}/pcf_fid_*.dat", f"{BASE_DIR}/pk_fid_*.txt"),
    20: (
        "fnl020",
        f"{INTERP_DIR}/fnl020/pk_fnl20_*.txt",
        f"{INTERP_DIR}/fnl020/pcf_fnl20_*.dat",
        f"{BASE_DIR}/pk_fid_*.txt",
    ),
    30: (
        "fnl030",
        f"{INTERP_DIR}/fnl030/pk_fnl30_*.txt",
        f"{INTERP_DIR}/fnl030/pcf_fnl30_*.dat",
        f"{BASE_DIR}/pk_fid_*.txt",
    ),
    50: ("LCp50", f"{BASE_DIR}/pk_LCp50_*.txt", f"{BASE_DIR}/pcf_LCp50_*.dat", f"{BASE_DIR}/pk_fid_*.txt"),
    75: (
        "fnl075",
        f"{INTERP_DIR}/fnl075/pk_fnl75_*.txt",
        f"{INTERP_DIR}/fnl075/pcf_fnl75_*.dat",
        f"{BASE_DIR}/pk_fid_*.txt",
    ),
    100: ("LCp100", f"{BASE_DIR}/pk_LCp100_*.txt", f"{BASE_DIR}/pcf_LCp100_*.dat", f"{BASE_DIR}/pk_fid_*.txt"),
}


# ============================================================
# I/O：Quijote 文件读入
# ============================================================
def _rid_quijote(fp: str) -> int:
    """Quijote 文件名结尾是 ..._{rid}.txt 或 ..._{rid}.dat，这里取最后的数字作为 rid。"""
    m = re.search(r"(\\d+)\\.\\w+$", os.path.basename(fp))
    return int(m.group(1)) if m else -1


def load_pk_quijote(pat: str, n_max: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """加载 Quijote pk 数据（.txt，7列）。

    返回
    ----
    kcen, kmin, kmax : ndarray (n_bin_valid,)
        只保留 nmod>0 的有效 bin。
    mocks : ndarray (n_file, n_bin_valid)
    mean, std : ndarray (n_bin_valid,)
    n_file : int
    """
    fps = sorted(glob.glob(pat), key=_rid_quijote)[:n_max]
    if not fps:
        raise FileNotFoundError(pat)
    ref = np.loadtxt(fps[0], comments="#")
    nmod = ref[:, 4]
    valid = nmod > 0
    kcen, kmin_b, kmax_b = ref[valid, 0], ref[valid, 1], ref[valid, 2]
    vals = []
    for f in fps:
        try:
            arr = np.loadtxt(f, comments="#")
            vals.append(arr[valid, 5])  # P0
        except Exception:
            continue
    mocks = np.array(vals, dtype=float)
    mean = mocks.mean(axis=0)
    std = mocks.std(axis=0, ddof=1)
    return kcen, kmin_b, kmax_b, mocks, mean, std, len(mocks)


def load_pcf_quijote(pat: str, n_max: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """加载 Quijote pcf 数据（.dat，4列）。"""
    fps = sorted(glob.glob(pat), key=_rid_quijote)[:n_max]
    if not fps:
        raise FileNotFoundError(pat)
    ref = np.loadtxt(fps[0], comments="#")
    s = ref[:, 0]
    mocks = np.array([np.loadtxt(f, comments="#")[:, 3] for f in fps], dtype=float)
    return s, mocks.mean(axis=0), mocks.std(axis=0, ddof=1), len(mocks)


# ============================================================
# 数学/数值工具
# ============================================================
def j0(x: np.ndarray) -> np.ndarray:
    """球贝塞尔 j0(x)=sin(x)/x，x=0 用极限值 1。"""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    m = x != 0
    out[m] = np.sin(x[m]) / x[m]
    return out


def met_mean_abs_sigma(r2_data: np.ndarray, r2_std: np.ndarray, r2_model: np.ndarray) -> tuple[float, float]:
    """返回 mean(|Δ/σ|) 与 mean(Δ/σ)。"""
    ds = (r2_data - r2_model) / r2_std
    return float(np.nanmean(np.abs(ds))), float(np.nanmean(ds))


def gq_fft(qmax: int, nmax: int) -> np.ndarray:
    """FFT 卷积计算 g_q（Mission 8/9 同款）。"""
    a = np.zeros(qmax + 1, dtype=np.float32)
    a[0] = 1.0
    sq = (np.arange(1, nmax + 1, dtype=np.int64) ** 2)
    sq = sq[sq <= qmax]
    a[sq] = 2.0
    nfft = next_fast_len(3 * qmax + 1)
    fa = rfft(a, n=nfft)
    conv = irfft(fa * fa * fa, n=nfft)
    return np.rint(conv[: qmax + 1]).astype(np.int64)


def gq_enumerate(qmax: int) -> np.ndarray:
    """直接枚举计算小 qmax 的 g_q（用于拟合阶段的 bin-average）。"""
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


def build_shells_for_kmax(kf: float, gq: np.ndarray, kmax: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 g_q 转成壳层列表 (q,k,g)，并截断到 k<=kmax。"""
    q_nz = np.nonzero(gq[1:])[0] + 1
    k_nz = kf * np.sqrt(q_nz.astype(float))
    m = k_nz <= kmax
    q = q_nz[m]
    k = k_nz[m]
    g = gq[q].astype(np.int64)
    return q, k, g


def build_bin_shell_index(
    k_shell: np.ndarray,
    kmin_bin: np.ndarray,
    kmax_bin: np.ndarray,
) -> list[np.ndarray]:
    """对每个 bin，返回包含该 bin 的 shell 索引。"""
    idx_per_bin: list[np.ndarray] = []
    for i in range(len(kmin_bin)):
        m = (k_shell >= kmin_bin[i]) & (k_shell < kmax_bin[i])
        idx_per_bin.append(np.nonzero(m)[0])
    return idx_per_bin


def binavg_pk_from_shells(pk_shell: np.ndarray, g_shell: np.ndarray, idx_per_bin: list[np.ndarray]) -> np.ndarray:
    """由 P(k_shell) 得到每个 bin 的 g_q 加权平均。"""
    out = np.zeros(len(idx_per_bin), dtype=float)
    for i, idx in enumerate(idx_per_bin):
        if idx.size == 0:
            out[i] = np.nan
            continue
        w = g_shell[idx].astype(float)
        out[i] = float(np.sum(w * pk_shell[idx]) / np.sum(w))
    return out


def xi_discrete_multi(
    s: np.ndarray,
    gq: np.ndarray,
    kf: float,
    V: float,
    kd: np.ndarray,
    pd_list: list[np.ndarray],
    databin_edges: np.ndarray,
    databin_pk: np.ndarray,
    chunk: int = 500_000,
) -> dict[str, np.ndarray]:
    """一次离散求和同时输出：

    - FD0/FD1/...：pd_list 对应的 FullDiscrete
    - DataBin：在低 k 的前 len(databin_pk) 个 bin 内替换为 databin_pk（高 k 用 FD0 的 P_model）
    """
    qnz = np.nonzero(gq[1:])[0] + 1
    total = len(qnz)
    xi_fd = [np.zeros(len(s), dtype=float) for _ in pd_list]
    xi_db = np.zeros(len(s), dtype=float)

    t0 = time.time()
    for i0 in range(0, total, chunk):
        qb = qnz[i0 : i0 + chunk]
        kv = kf * np.sqrt(qb.astype(float))
        g = gq[qb].astype(float)

        pv_list = [np.interp(kv, kd, pd) for pd in pd_list]
        pv_db = pv_list[0].copy()
        # 只替换前 N_fit=len(databin_pk) 个 bin
        bin_idx = np.searchsorted(databin_edges, kv, side="right") - 1
        m = (bin_idx >= 0) & (bin_idx < len(databin_pk))
        pv_db[m] = databin_pk[bin_idx[m]]

        w_fd = [g * pv for pv in pv_list]
        w_stack = np.stack(w_fd + [g * pv_db], axis=0)  # (n_model+1, n_k)

        for js in range(0, len(s), 10):
            seg = s[js : js + 10]
            J = j0(np.outer(kv, seg))
            contrib = w_stack @ J
            for im in range(len(pv_list)):
                xi_fd[im][js : js + len(seg)] += contrib[im]
            xi_db[js : js + len(seg)] += contrib[-1]

        if (i0 // chunk) % 20 == 0:
            print(f"    [xi] {min(i0 + chunk, total)}/{total} (elapsed {time.time()-t0:.0f}s)")

    out: dict[str, np.ndarray] = {}
    for i, xi in enumerate(xi_fd):
        out[f"FD{i}"] = xi / V
    out["DataBin"] = xi_db / V
    return out


# ============================================================
# 拟合（标准 desilike / BinAvgFit）
# ============================================================
def fit_standard_desilike(
    cosmo,
    kcen: np.ndarray,
    kmin_b: np.ndarray,
    kmax_b: np.ndarray,
    pk_mean: np.ndarray,
    cov_mocks: np.ndarray,
) -> dict:
    """标准 desilike 拟合：理论在 bin center 评估。"""
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler
    from pypower import PowerSpectrumStatistics

    edges = np.concatenate([kmin_b, [kmax_b[-1]]])
    nm = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    data_ps = PowerSpectrumStatistics(
        edges=edges,
        modes=kcen,
        power_nonorm=np.array([pk_mean]),
        nmodes=nm,
        ells=[0],
        shotnoise_nonorm=0.0,
        statistic="multipole",
    )
    mock_list = []
    for i in range(cov_mocks.shape[0]):
        t = data_ps.deepcopy()
        t.power_nonorm.flat[...] = np.array([cov_mocks[i]]).ravel()
        mock_list.append(t)

    tmpl = FixedPowerSpectrumTemplate(z=Z, fiducial=cosmo)
    th = PNGTracerPowerSpectrumMultipoles(template=tmpl, mode="b-p")
    th.init.params["p"].update(fixed=True, value=P_FIX)
    th.init.params["sn0"].update(fixed=True, value=0.0)
    th.init.params["sigmas"].update(fixed=False, value=0.0)

    obs = TracerPowerSpectrumMultipolesObservable(
        data=data_ps,
        covariance=mock_list,
        klim={0: [float(kcen.min()), float(kcen.max()), float(kcen[1] - kcen[0])]},
        theory=th,
    )
    like = ObservablesGaussianLikelihood(observables=[obs])
    _ = like()
    like.all_params["p"].update(fixed=True, value=P_FIX)
    like.all_params["sn0"].update(fixed=True, value=0.0)
    like.all_params["sigmas"].update(fixed=False, value=0.0)

    prof = MinuitProfiler(like, seed=66)
    profiles = prof.maximize(niterations=27)
    return profiles.bestfit.choice(input=True)


def fit_binavg_minuit(
    cosmo,
    base_params: dict,
    k_shell: np.ndarray,
    g_shell: np.ndarray,
    idx_per_bin: list[np.ndarray],
    pk_data: np.ndarray,
    cov_mocks: np.ndarray,
) -> dict:
    """BinAvgFit：用 iminuit 最小化 bin-average theory 的 χ²。"""
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from iminuit import Minuit

    cov = np.cov(cov_mocks, rowvar=False, ddof=1)
    cov_inv = np.linalg.pinv(cov, rcond=1e-10)

    tmpl = FixedPowerSpectrumTemplate(z=Z, fiducial=cosmo)
    th = PNGTracerPowerSpectrumMultipoles(k=k_shell, template=tmpl, mode="b-p")
    th.init.params["p"].update(fixed=True, value=P_FIX)
    th.init.params["sn0"].update(fixed=True, value=0.0)
    th.init.params["sigmas"].update(fixed=False, value=float(base_params.get("sigmas", 0.0)))

    def chi2(fnl_loc: float, b1: float, sigmas: float) -> float:
        p = dict(base_params)
        p["fnl_loc"] = float(fnl_loc)
        p["b1"] = float(b1)
        p["sigmas"] = float(sigmas)
        p["p"] = float(P_FIX)
        p["sn0"] = 0.0
        th(**p)
        pk_shell = np.array(th.power[0], dtype=float)
        pk_bin = binavg_pk_from_shells(pk_shell, g_shell=g_shell, idx_per_bin=idx_per_bin)
        diff = pk_data - pk_bin
        return float(diff @ cov_inv @ diff)

    m = Minuit(
        chi2,
        fnl_loc=float(base_params.get("fnl_loc", 0.0)),
        b1=float(base_params.get("b1", 2.0)),
        sigmas=float(base_params.get("sigmas", 0.0)),
    )
    m.errordef = 1.0
    m.limits["b1"] = (0.0, None)
    m.limits["sigmas"] = (0.0, None)
    m.limits["fnl_loc"] = (-2000.0, 2000.0)
    m.migrad()

    out = dict(base_params)
    out["fnl_loc"] = float(m.values["fnl_loc"])
    out["b1"] = float(m.values["b1"])
    out["sigmas"] = float(m.values["sigmas"])
    out["p"] = float(P_FIX)
    out["sn0"] = 0.0
    return out


def eval_pk_dense(cosmo, k: np.ndarray, params: dict) -> np.ndarray:
    """在密集 k 网格上评估 P_model(k)。"""
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles

    tmpl = FixedPowerSpectrumTemplate(z=Z, fiducial=cosmo)
    th = PNGTracerPowerSpectrumMultipoles(k=k, template=tmpl, mode="b-p")
    th.init.params["p"].update(fixed=True, value=P_FIX)
    th.init.params["sn0"].update(fixed=True, value=0.0)
    th.init.params["sigmas"].update(fixed=False, value=float(params.get("sigmas", 0.0)))

    p = dict(params)
    p["p"] = float(P_FIX)
    p["sn0"] = 0.0
    th(**p)
    return np.array(th.power[0], dtype=float)


# ============================================================
# 结果记录
# ============================================================
@dataclass
class OneResult:
    fnl_true: int
    fnl_std: float
    fnl_binavg: float
    b1_std: float
    b1_binavg: float
    sig_std: float
    sig_binavg: float
    fd_ma: float
    fd_ms: float
    db_ma: float
    db_ms: float
    new_ma: float
    new_ms: float


def _write_results_md(results: list[OneResult], out_path: str) -> None:
    """把结果写成 markdown 表格。"""
    lines = []
    lines.append("# Mission 10: Quijote BinAvgFit+FullDiscrete 结果表")
    lines.append("")
    lines.append("说明：")
    lines.append("- 盒长 L=1000 Mpc/h，kmax(离散求和)=15。")
    lines.append("- 拟合只用前 20 个有效 k-bin（与 Mission 9 保持一致）。")
    lines.append("- DataBin 也只替换前 20 个 bin（避免高 k 阶梯函数引入振荡伪影）。")
    lines.append("")
    lines.append("| fnl_true | fnl_fit(std) | fnl_fit(binavg) | b1(std) | b1(binavg) | sigmas(std) | sigmas(binavg) | FD mean\\|Δ/σ\\| | FD mean(Δ/σ) | DB20 mean\\|Δ/σ\\| | DB20 mean(Δ/σ) | New mean\\|Δ/σ\\| | New mean(Δ/σ) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.fnl_true:d} | {r.fnl_std:.2f} | {r.fnl_binavg:.2f} | {r.b1_std:.4f} | {r.b1_binavg:.4f} | "
            f"{r.sig_std:.2f} | {r.sig_binavg:.2f} | "
            f"{r.fd_ma:.4f} | {r.fd_ms:+.4f} | {r.db_ma:.4f} | {r.db_ms:+.4f} | {r.new_ma:.4f} | {r.new_ms:+.4f} |"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================
# 主程序
# ============================================================
def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("Mission 10: Quijote BinAvgFit+FullDiscrete validation")
    print("=" * 60)

    from cosmoprimo import Cosmology

    cosmo = Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )

    # 预计算：g_q（离散求和）与 kd（插值网格）
    kd = np.geomspace(K_FUND * 0.5, KMAX_DISCRETE * 1.1, N_DENSE)
    qmax = int((KMAX_DISCRETE / K_FUND) ** 2)
    nmax = int(KMAX_DISCRETE / K_FUND)
    print(f"[pre] g_q: qmax={qmax} (may take ~1s)")
    gq = gq_fft(qmax=qmax, nmax=nmax)
    print(f"[pre] nonzero shells: {np.count_nonzero(gq[1:])}")

    # 画图：低/高 fnl 两张图
    low_set = [0, 20, 30]
    high_set = [50, 75, 100]

    fig_low, axes_low = plt.subplots(2, 3, figsize=(18, 10))
    fig_low.suptitle("Quijote (L=1Gpc): fnl=0/20/30", fontsize=14, y=0.98)
    fig_high, axes_high = plt.subplots(2, 3, figsize=(18, 10))
    fig_high.suptitle("Quijote (L=1Gpc): fnl=50/75/100", fontsize=14, y=0.98)

    results: list[OneResult] = []

    for fnl_true, (label, pk_pat, pcf_pat, cov_pat) in sorted(FNL_CONFIGS.items()):
        print(f"\n{'='*50}")
        print(f"fnl_true={fnl_true} ({label})")
        print(f"{'='*50}")

        # 读数据
        kcen_all, kmin_all, kmax_all, pk_mocks_all, pk_mean_all, _pk_std_all, npk = load_pk_quijote(pk_pat)
        s, xi_mean, xi_std, npcf = load_pcf_quijote(pcf_pat)
        # 协方差：用 fid 的 500 个（与 Mission 9 一致）
        _kc_cov, _kmin_cov, _kmax_cov, cov_mocks_all, _cov_mean, _cov_std, ncov = load_pk_quijote(cov_pat)

        # 只用前 FIT_NDP 个有效 bin 拟合
        fit_ndp = min(FIT_NDP, len(kcen_all))
        kcen = kcen_all[:fit_ndp]
        kmin_b = kmin_all[:fit_ndp]
        kmax_b = kmax_all[:fit_ndp]
        pk_mean = pk_mean_all[:fit_ndp]
        cov_mocks = cov_mocks_all[:, :fit_ndp]

        print(f"  pk files={npk}, valid_bins={len(kcen_all)}, fit_bins={fit_ndp}, cov_files={ncov}")
        print(f"  k range (fit): [{kcen[0]:.5f}, {kcen[-1]:.5f}]")
        r2_data = s**2 * xi_mean
        r2_std = s**2 * xi_std

        # 标准拟合
        print("  [fit] standard desilike ...")
        bf_std = fit_standard_desilike(
            cosmo=cosmo,
            kcen=kcen,
            kmin_b=kmin_b,
            kmax_b=kmax_b,
            pk_mean=pk_mean,
            cov_mocks=cov_mocks,
        )

        # BinAvgFit：需要拟合区间内所有 shell
        print("  [fit] BinAvgFit (bin-average theory) ...")
        kmax_fit = float(kmax_b[-1])
        qmax_fit = int(np.floor((kmax_fit / K_FUND) ** 2)) + 1
        gq_fit = gq_enumerate(qmax_fit)
        _q_shell, k_shell, g_shell = build_shells_for_kmax(K_FUND, gq_fit, kmax=kmax_fit)
        idx_per_bin = build_bin_shell_index(k_shell, kmin_b, kmax_b)
        bf_binavg = fit_binavg_minuit(
            cosmo=cosmo,
            base_params=bf_std,
            k_shell=k_shell,
            g_shell=g_shell,
            idx_per_bin=idx_per_bin,
            pk_data=pk_mean,
            cov_mocks=cov_mocks,
        )

        print(f"    best-fit(std):    fnl={float(bf_std.get('fnl_loc', np.nan)):.2f}, "
              f"b1={float(bf_std.get('b1', np.nan)):.4f}, sig={float(bf_std.get('sigmas', np.nan)):.2f}")
        print(f"    best-fit(binavg): fnl={float(bf_binavg.get('fnl_loc', np.nan)):.2f}, "
              f"b1={float(bf_binavg.get('b1', np.nan)):.4f}, sig={float(bf_binavg.get('sigmas', np.nan)):.2f}")

        # 评估 P(k)（密集网格）
        print("  [pk] evaluate dense P(k) ...")
        pd_std = eval_pk_dense(cosmo, kd, bf_std)
        pd_binavg = eval_pk_dense(cosmo, kd, bf_binavg)

        # DataBin：只替换前 fit_ndp 个 bin；这里用 fit bins 的 edges
        edges_fit = np.concatenate([kmin_b, [kmax_b[-1]]])

        # 计算 xi
        print("  [xi] FullDiscrete(std), BinAvgFit+FD, DataBin(20) ...")
        xi = xi_discrete_multi(
            s=s,
            gq=gq,
            kf=K_FUND,
            V=VOLUME,
            kd=kd,
            pd_list=[pd_std, pd_binavg],
            databin_edges=edges_fit,
            databin_pk=pk_mean,
        )
        r2_fd = s**2 * xi["FD0"]
        r2_new = s**2 * xi["FD1"]
        r2_db = s**2 * xi["DataBin"]

        # 指标
        fd_ma, fd_ms = met_mean_abs_sigma(r2_data, r2_std, r2_fd)
        db_ma, db_ms = met_mean_abs_sigma(r2_data, r2_std, r2_db)
        new_ma, new_ms = met_mean_abs_sigma(r2_data, r2_std, r2_new)
        print(f"  metrics: FD={fd_ma:.4f},{fd_ms:+.4f}  DB20={db_ma:.4f},{db_ms:+.4f}  NEW={new_ma:.4f},{new_ms:+.4f}")

        results.append(
            OneResult(
                fnl_true=int(fnl_true),
                fnl_std=float(bf_std.get("fnl_loc", np.nan)),
                fnl_binavg=float(bf_binavg.get("fnl_loc", np.nan)),
                b1_std=float(bf_std.get("b1", np.nan)),
                b1_binavg=float(bf_binavg.get("b1", np.nan)),
                sig_std=float(bf_std.get("sigmas", np.nan)),
                sig_binavg=float(bf_binavg.get("sigmas", np.nan)),
                fd_ma=fd_ma,
                fd_ms=fd_ms,
                db_ma=db_ma,
                db_ms=db_ms,
                new_ma=new_ma,
                new_ms=new_ms,
            )
        )

        # 画图位置选择
        if fnl_true in low_set:
            fig = fig_low
            axes = axes_low
            col = low_set.index(fnl_true)
        else:
            fig = fig_high
            axes = axes_high
            col = high_set.index(fnl_true)

        ax_t = axes[0, col]
        ax_b = axes[1, col]

        ax_t.errorbar(s, r2_data, yerr=r2_std, fmt="ko", ms=3, capsize=2, label="Data mean", zorder=10)
        # 三条模型曲线
        ax_t.plot(s, r2_fd, color="tab:green", lw=1.4, label=f"FullDiscrete ({fd_ma:.3f},{fd_ms:+.3f})")
        ax_t.plot(s, r2_db, color="darkorange", lw=1.8, label=f"DataBin(20) ({db_ma:.3f},{db_ms:+.3f})")
        ax_t.plot(s, r2_new, color="tab:purple", lw=2.0, label=f"BinAvgFit+FD ({new_ma:.3f},{new_ms:+.3f})")
        ax_t.set_title(f"fnl={fnl_true}", fontsize=12)
        ax_t.set_ylabel(r"$r^2\xi_0(r)$")
        ax_t.grid(alpha=0.3)
        ax_t.legend(fontsize=8, title="(mean|Δ/σ|, mean Δ/σ)")

        # 残差
        ax_b.axhline(0.0, color="k", lw=0.6)
        ax_b.axhline(1.0, color="gray", ls=":", lw=0.6)
        ax_b.axhline(-1.0, color="gray", ls=":", lw=0.6)
        for name, r2m, c, lw, a in [
            ("FD", r2_fd, "tab:green", 1.2, 0.7),
            ("DB20", r2_db, "darkorange", 1.6, 0.8),
            ("NEW", r2_new, "tab:purple", 2.0, 0.9),
        ]:
            ds = (r2_data - r2m) / r2_std
            ax_b.plot(s, ds, color=c, lw=lw, marker="o", ms=2, alpha=a, label=name)
        ax_b.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax_b.set_ylabel(r"$(Data-Model)/\sigma$ of $r^2\xi_0$")
        ax_b.grid(alpha=0.3)
        ax_b.set_ylim(-3, 3)
        ax_b.legend(fontsize=8)

    # 保存图
    fig_low.tight_layout(rect=[0, 0, 1, 0.96])
    fig_high.tight_layout(rect=[0, 0, 1, 0.96])
    out_low = os.path.join(OUT_DIR, "mission10_quijote_binavgfit_compare_lowfnl.png")
    out_high = os.path.join(OUT_DIR, "mission10_quijote_binavgfit_compare_highfnl.png")
    fig_low.savefig(out_low, dpi=150)
    fig_high.savefig(out_high, dpi=150)
    plt.close(fig_low)
    plt.close(fig_high)
    print(f"\n[OK] saved: {out_low}")
    print(f"[OK] saved: {out_high}")

    # 保存 markdown 表
    out_md = os.path.join(OUT_DIR, "mission10_quijote_binavgfit_results.md")
    _write_results_md(results, out_md)
    print(f"[OK] saved: {out_md}")

    print(f"Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

