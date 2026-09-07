#!/usr/bin/env python3
"""
任务5 v8: 比较 fnl0 下 1Gpc 与 3Gpc 的 2PCF 测量均值。

输出:
- /pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission5_log/fnl0_1gpc_vs_3gpc_2pcf_mean_compare.png
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PCF_1GPC_GLOB = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut_fnl0/pcf_rsd_N*.dat"
PCF_3GPC_GLOB = "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat"
OUTFIG = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission5_log/fnl0_1gpc_vs_3gpc_2pcf_mean_compare.png"

REAL_RE = re.compile(r"_N(\d+)\.dat$")


@dataclass
class XiDataset:
    s: np.ndarray
    xi: np.ndarray  # shape: (Nmock, Ns)

    @property
    def nmock(self) -> int:
        return self.xi.shape[0]

    @property
    def mean(self) -> np.ndarray:
        return np.mean(self.xi, axis=0)

    @property
    def std(self) -> np.ndarray:
        return np.std(self.xi, axis=0, ddof=1)

    @property
    def sem(self) -> np.ndarray:
        return self.std / np.sqrt(self.nmock)


def realization_id(path: str) -> int:
    m = REAL_RE.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"Cannot parse realization id from: {path}")
    return int(m.group(1))


def read_pcf(path: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    s = arr[:, 0]
    xi0 = arr[:, 3]
    return s, xi0


def load_dataset(pattern: str) -> XiDataset:
    files = sorted(glob.glob(pattern), key=realization_id)
    if not files:
        raise RuntimeError(f"No files matched: {pattern}")

    s_ref, xi_ref = read_pcf(files[0])
    rows = [xi_ref]

    for fp in files[1:]:
        s, xi = read_pcf(fp)
        if np.allclose(s, s_ref, rtol=0.0, atol=0.0):
            rows.append(xi)
        else:
            rows.append(np.interp(s_ref, s, xi))

    xi_mat = np.vstack(rows)
    return XiDataset(s=s_ref, xi=xi_mat)


def main() -> None:
    data_1gpc = load_dataset(PCF_1GPC_GLOB)
    data_3gpc = load_dataset(PCF_3GPC_GLOB)

    s = data_1gpc.s
    mean1 = data_1gpc.mean
    mean3 = data_3gpc.mean
    sem1 = data_1gpc.sem
    sem3 = data_3gpc.sem

    r2 = s**2
    y1 = r2 * mean1
    y3 = r2 * mean3
    e1 = r2 * sem1
    e3 = r2 * sem3

    delta = y1 - y3
    e_delta = np.sqrt(e1**2 + e3**2)

    sig = np.divide(
        delta,
        e_delta,
        out=np.zeros_like(delta),
        where=e_delta > 0,
    )

    mask_ls = s >= 200.0
    mean_abs_sig_ls = float(np.mean(np.abs(sig[mask_ls])))
    max_abs_sig_ls = float(np.max(np.abs(sig[mask_ls])))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2]},
    )

    ax = axes[0]
    ax.plot(s, y1, color="#1f77b4", lw=2.0, label=f"1Gpc fnl0 mean (N={data_1gpc.nmock})")
    ax.fill_between(s, y1 - e1, y1 + e1, color="#1f77b4", alpha=0.20, linewidth=0.0, label="1Gpc mean ± SEM")

    ax.plot(s, y3, color="#d62728", lw=2.0, label=f"3Gpc fnl0 mean (N={data_3gpc.nmock})")
    ax.fill_between(s, y3 - e3, y3 + e3, color="#d62728", alpha=0.20, linewidth=0.0, label="3Gpc mean ± SEM")

    ax.set_ylabel(r"$r^2\,\xi_0(r)$")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.set_title("fnl=0: 1Gpc vs 3Gpc measured mean 2PCF (with mean uncertainty)")

    ax2 = axes[1]
    ax2.axhline(0.0, color="black", lw=1.0, alpha=0.7)
    ax2.plot(s, delta, color="#2ca02c", lw=2.0, label=r"$\Delta=r^2(\bar\xi_{1Gpc}-\bar\xi_{3Gpc})$")
    ax2.fill_between(s, -e_delta, e_delta, color="gray", alpha=0.25, linewidth=0.0, label=r"combined $\pm1\sigma_{\rm mean}$")
    ax2.set_ylabel(r"$\Delta r^2\xi_0$")
    ax2.set_xlabel(r"$r\ [h^{-1}{\rm Mpc}]$")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best", frameon=False, fontsize=9)

    info = (
        "1Gpc mass cut for fnl0 re-run: 1.4e13 <= Mh <= 1e16 (h^-1 Msun)\n"
        + rf"large scale (r>=200): mean|Delta/sigma_mean|={mean_abs_sig_ls:.2f}, max={max_abs_sig_ls:.2f}"
    )
    ax2.text(
        0.01,
        0.98,
        info,
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="0.8"),
    )

    fig.tight_layout()
    fig.savefig(OUTFIG, dpi=180)
    plt.close(fig)

    print(f"[OK ] 1Gpc mocks: {data_1gpc.nmock}")
    print(f"[OK ] 3Gpc mocks: {data_3gpc.nmock}")
    print(f"[OK ] saved figure: {OUTFIG}")
    print(f"[METRIC] r>=200 mean|Delta/sigma_mean| = {mean_abs_sig_ls:.4f}")
    print(f"[METRIC] r>=200 max|Delta/sigma_mean|  = {max_abs_sig_ls:.4f}")


if __name__ == "__main__":
    main()
