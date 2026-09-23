#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 4.3.2 RSD model and operator audit.

代码执行大纲
------------
1. 加载 Task43 真实 rawbox RSD theory cache，不修改原 cache。
2. 检查 shell-averaged j2 kernel 的数值积分与解析 primitive 是否一致。
3. 检查 FullDiscreteRSDModel 的 Gauss--Legendre nmu 收敛。
4. 比较低-k 区域的 exact lattice angular operator 与 continuous-angle
   FullDiscrete operator，量化 rawbox xi0/xi2 可能受到的有限盒角向效应。
5. 输出 JSON audit 和 PDF 诊断图到 task432_model_repair，绝不覆盖 4.3
   既有结果，也不生成 PNG。

这个脚本只做数学/数值诊断，不进行 MCMC，不改变 production likelihood。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
OUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "task432_model_repair"
PLOT_ROOT = PROJECT_ROOT / "plots" / "task43" / "rsd_validation" / "task432_model_repair"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task43_rawbox_numerics import (  # noqa: E402
    continuous_rsd_poles,
    exact_lattice_modes,
    mode_operators,
    shell_kernel as analytic_shell_kernel,
)
from task43_rsd_model import (  # noqa: E402
    DELTA_C,
    FullDiscreteRSDModel,
    build_cache,
    shell_jell_kernel,
)


def jsonable(value: Any) -> Any:
    """递归转换 numpy/path 对象，保证 audit 可以稳定写入 JSON。"""

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
    """原子写入 JSON，避免中断时留下半个 audit。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def kernel_audit(s_edges: np.ndarray) -> dict[str, Any]:
    """比较固定 24 点 Gauss shell kernel 与解析 shell primitive。"""

    k = np.linspace(1.0e-5, 3.0, 400, dtype="f8")
    reference = analytic_shell_kernel(k, s_edges, 2)
    numerical = shell_jell_kernel(k, s_edges, 2, nquad=24)
    delta = numerical - reference
    relative = np.abs(delta) / np.maximum(np.abs(reference), 1.0e-14)
    return {
        "k_range_h_mpc": [float(k.min()), float(k.max())],
        "n_k": int(k.size),
        "nquad": 24,
        "max_abs": float(np.max(np.abs(delta))),
        "max_relative_with_floor": float(np.max(relative)),
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1.0e-300)),
        "status": "pass" if float(np.max(relative)) < 1.0e-7 else "investigate",
        "note": "The comparison uses the same s shell edges as the RSD theory cache.",
    }


def nmu_audit(cache_path: Path) -> dict[str, Any]:
    """比较不同 Gauss--Legendre nmu 对 P0/P2 theory poles 的影响。"""

    trials = ((0.0, 2.55, 1.0), (0.0, 2.55, 8.0), (100.0, 2.55, 8.0))
    reference_model = FullDiscreteRSDModel(cache_path, nmu=192)
    rows: list[dict[str, Any]] = []
    for fnl, b1, sigma_s in trials:
        reference = reference_model.evaluate(fnl=fnl, b1=b1, sigma_s=sigma_s, p_fixed=1.0)
        for nmu in (32, 64, 96, 128):
            model = FullDiscreteRSDModel(cache_path, nmu=nmu)
            current = model.evaluate(fnl=fnl, b1=b1, sigma_s=sigma_s, p_fixed=1.0)
            for ell in (0, 2):
                delta = np.asarray(current[ell]) - np.asarray(reference[ell])
                rows.append(
                    {
                        "fnl": fnl,
                        "b1": b1,
                        "sigma_s": sigma_s,
                        "nmu": nmu,
                        "ell": ell,
                        "max_abs": float(np.max(np.abs(delta))),
                        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(reference[ell]), 1.0e-300)),
                    }
                )
    max_rel = max(row["relative_l2"] for row in rows)
    return {
        "reference_nmu": 192,
        "rows": rows,
        "max_relative_l2": float(max_rel),
        "status": "pass" if max_rel < 1.0e-7 else "investigate",
    }


def lattice_angular_audit(cache_path: Path) -> dict[str, Any]:
    """量化 exact lattice angular 与 continuous-angle xi 的低-k 差异。"""

    with np.load(cache_path, allow_pickle=False) as data:
        k_eff = np.asarray(data["k_eff"], dtype="f8")
        pk_dd_cache = np.asarray(data["pk_dd"], dtype="f8")
        alpha_cache = np.asarray(data["alpha"], dtype="f8")
        g_nz = np.asarray(data["g_nz"], dtype="f8")
        s_edges = np.asarray(data["s_edges"], dtype="f8")
        volume = float(np.asarray(data["volume"]).item())
        boxsize = float(np.asarray(data["boxsize"]).item())
        growth = float(np.asarray(data["f_growth"]).item())

    rows: list[dict[str, Any]] = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    for k_switch in (0.005, 0.01, 0.02, 0.05):
        modes = exact_lattice_modes(boxsize, np.asarray([[0.0, k_switch]], dtype="f8"))
        power = np.interp(np.log(modes.k), np.log(k_eff), pk_dd_cache)
        alpha = np.interp(np.log(modes.k), np.log(k_eff), alpha_cache)
        operator_p, operator_x = mode_operators(modes, s_edges, ells=(0, 2))
        for fnl, b1, sigma_s in ((0.0, 2.55, 1.0), (0.0, 2.55, 8.0), (100.0, 2.55, 8.0)):
            bphi = 2.0 * DELTA_C * (b1 - 1.0)
            amplitude = b1 + fnl * bphi * alpha
            damping = 1.0 / (1.0 + 0.5 * (modes.k * modes.mu * sigma_s) ** 2) ** 2
            signal = power * (amplitude + growth * modes.mu**2) ** 2 * damping
            exact = operator_x @ signal
            poles = continuous_rsd_poles(modes.k, power, amplitude, growth, sigma_s)
            continuous = np.concatenate(
                [
                    np.sum(poles[ell][:, None] * analytic_shell_kernel(modes.k, s_edges, ell), axis=0)
                    / volume
                    for ell in (0, 2)
                ]
            )
            delta = exact - continuous
            nbin = centers.size
            key = f"k{str(k_switch).replace('.', 'p')}_fnl{int(fnl)}_sig{str(sigma_s).replace('.', 'p')}"
            curves[key] = {
                "centers": centers.copy(),
                "delta_xi0": delta[:nbin].copy(),
                "delta_xi2": delta[nbin:].copy(),
            }
            for ell, block in ((0, delta[:nbin]), (2, delta[nbin:])):
                ref_block = continuous[0 if ell == 0 else nbin : nbin if ell == 0 else 2 * nbin]
                rows.append(
                    {
                        "k_switch": k_switch,
                        "n_modes": int(modes.k.size),
                        "fnl": fnl,
                        "b1": b1,
                        "sigma_s": sigma_s,
                        "ell": ell,
                        "max_abs": float(np.max(np.abs(block))),
                        "relative_l2": float(np.linalg.norm(block) / max(np.linalg.norm(ref_block), 1.0e-300)),
                    }
                )

    xi2_rows = [row for row in rows if row["ell"] == 2]
    max_xi2 = max(row["relative_l2"] for row in xi2_rows)
    return {
        "k_switches": [0.005, 0.01, 0.02, 0.05],
        "rows": rows,
        "max_xi2_relative_l2": float(max_xi2),
        "curves": curves,
        "status": "diagnostic_only",
        "note": (
            "This is an operator mismatch diagnostic: P-side rawbox P2 uses exact parent modes, "
            "whereas the current xi FullDiscrete model uses continuous angular shell moments."
        ),
    }


def make_pdf(path: Path, kernel: dict[str, Any], lattice: dict[str, Any]) -> None:
    """生成只含 PDF 的数学诊断图。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary) as pdf:
        figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
        k = np.linspace(1.0e-5, 3.0, 400)
        # kernel error is summarized as a horizontal diagnostic envelope
        axes[0].axhline(kernel["max_relative_with_floor"], color="#C44E52", label="max relative error")
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlim(k.min(), k.max())
        axes[0].set_ylim(1.0e-16, max(1.0e-6, kernel["max_relative_with_floor"] * 10.0))
        axes[0].set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
        axes[0].set_ylabel("relative shell-kernel error")
        axes[0].set_title("j2 shell quadrature audit")
        axes[0].legend(frameon=False)
        axes[0].grid(True, which="both", alpha=0.25)

        for row in lattice["rows"]:
            if row["ell"] != 2 or row["fnl"] != 0.0 or row["sigma_s"] != 8.0:
                continue
            axes[1].scatter(row["k_switch"], row["relative_l2"], color="#4C72B0", s=35)
            axes[1].annotate(f"{row['n_modes']} modes", (row["k_switch"], row["relative_l2"]), fontsize=8)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel(r"$k_{\rm switch}\ [h\,\mathrm{Mpc}^{-1}]$")
        axes[1].set_ylabel(r"relative $\Delta\xi_2$ norm")
        axes[1].set_title("exact lattice angle vs continuous angle")
        axes[1].grid(True, which="both", alpha=0.25)
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        figure, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
        for key, curve in lattice["curves"].items():
            if "_fnl0_sig8" not in key:
                continue
            label = key.split("_fnl")[0]
            axes[0].plot(curve["centers"], curve["delta_xi0"], lw=1.0, label=label)
            axes[1].plot(curve["centers"], curve["delta_xi2"], lw=1.0, label=label)
        axes[0].set_ylabel(r"$\Delta\xi_0(s)$")
        axes[1].set_ylabel(r"$\Delta\xi_2(s)$")
        axes[1].set_xlabel(r"$s\ [h^{-1}\mathrm{Mpc}]$")
        axes[0].set_title("exact-lattice low-k correction for representative sigma_s=8")
        axes[0].legend(frameon=False, ncol=2, fontsize=8)
        for axis in axes:
            axis.axhline(0.0, color="0.5", lw=0.8)
            axis.grid(True, alpha=0.25)
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    temporary.replace(path)


def main() -> None:
    """执行理论 cache、kernel、nmu 和 finite-lattice angle audit。"""

    started = time.perf_counter()
    output_dir = OUT_ROOT / "audits"
    plot_dir = PLOT_ROOT
    cache_path = build_cache(zeff=0.725, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000")
    with np.load(cache_path, allow_pickle=False) as payload:
        s_edges = np.asarray(payload["s_edges"], dtype="f8")
    kernel = kernel_audit(s_edges)
    nmu = nmu_audit(cache_path)
    lattice = lattice_angular_audit(cache_path)
    audit = {
        "task": "Task 4.3.2 RSD model audit",
        "status": "complete",
        "theory_cache": str(cache_path),
        "kernel_audit": kernel,
        "nmu_audit": nmu,
        "finite_lattice_angular_audit": lattice,
        "interpretation": {
            "j2_shell_quadrature": "not the leading discrepancy in this audit if status=pass",
            "shared_sigma": "must remain baseline until exact-angle/operator tests are complete",
            "next_step": "build matched rawbox shared-vs-split sigma diagnostic and exact-angle xi variant",
        },
        "elapsed_sec": float(time.perf_counter() - started),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "task432_rsd_model_audit.json", audit)
    make_pdf(plot_dir / "task432_rsd_model_audit.pdf", kernel, lattice)
    print(json.dumps({"status": "complete", "audit": str(output_dir / 'task432_rsd_model_audit.json'), "pdf": str(plot_dir / 'task432_rsd_model_audit.pdf')}, sort_keys=True))


if __name__ == "__main__":
    main()
