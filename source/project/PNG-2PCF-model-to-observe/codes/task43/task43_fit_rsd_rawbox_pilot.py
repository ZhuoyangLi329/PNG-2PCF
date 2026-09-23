#!/usr/bin/env python3
"""Three-phase numerical pilot for the FullDiscrete rawbox RSD model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.special import spherical_jn
from scipy.stats import chi2 as chi2_distribution

from task43_rsd_common import PLOT_ROOT, PROJECT_ROOT, S_EDGES, atomic_savez, atomic_write_json


TASK44_DIR = PROJECT_ROOT / "codes" / "task44"
if str(TASK44_DIR) not in sys.path:
    sys.path.insert(0, str(TASK44_DIR))

from task44_rsd_theory import DELTA_C, build_theory_context  # noqa: E402


def shell_kernel(k: np.ndarray, edges: np.ndarray, ell: int, *, nquad: int = 24) -> np.ndarray:
    k = np.asarray(k, dtype="f8")
    lo, hi = np.asarray(edges[:-1], dtype="f8"), np.asarray(edges[1:], dtype="f8")
    if int(ell) == 0:
        kval = k[:, None]
        shell_volume_no4pi = (hi**3 - lo**3)[None, :] / 3.0
        upper = np.sin(kval * hi[None, :]) - kval * hi[None, :] * np.cos(kval * hi[None, :])
        lower = np.sin(kval * lo[None, :]) - kval * lo[None, :] * np.cos(kval * lo[None, :])
        return (upper - lower) / (kval**3 * shell_volume_no4pi)
    nodes, weights = np.polynomial.legendre.leggauss(int(nquad))
    output = np.empty((k.size, lo.size), dtype="f8")
    for ibin, (left, right) in enumerate(zip(lo, hi, strict=True)):
        radius = 0.5 * (right - left) * nodes + 0.5 * (right + left)
        radial_weight = 0.5 * (right - left) * weights * radius**2
        average = spherical_jn(int(ell), np.outer(k, radius)) @ radial_weight
        output[:, ibin] = ((-1.0) ** (int(ell) // 2)) * average / ((right**3 - left**3) / 3.0)
    return output


class RawboxModel:
    def __init__(self, *, redshift: float = 0.725, kmax: float = 3.0, nmu: int = 96) -> None:
        self.context = build_theory_context(
            float(redshift), kmax=float(kmax), ndense=20_000, boxsize=2000.0, cosmology="abacus_c000"
        )
        task41 = self.context["task41"]
        k = np.asarray(self.context["k_eff"], dtype="f8")
        template = self.context["template"]
        self.k = k
        self.pk_dd = task41.interp_logk(k, template["k"], template["pk_dd"])
        self.alpha = task41.interp_logk(k, template["k"], template["alpha"])
        self.g_nz = np.asarray(self.context["g_nz"], dtype="f8")
        self.volume = float(self.context["volume"])
        self.f_growth = float(self.context["f_growth"])
        self.kernel = {0: shell_kernel(k, S_EDGES, 0), 2: shell_kernel(k, S_EDGES, 2)}
        self.mu, self.wmu = np.polynomial.legendre.leggauss(int(nmu))
        self.mu2 = self.mu**2
        self.leg2 = 0.5 * (3.0 * self.mu2 - 1.0)

    def __call__(self, theta: np.ndarray) -> dict[int, np.ndarray]:
        fnl, b1, sigma_s = map(float, theta)
        bphi = 2.0 * DELTA_C * (b1 - 1.0)
        amplitude = b1 + fnl * bphi * self.alpha
        k_mu_sigma2 = (self.k[:, None] * self.mu[None, :] * sigma_s) ** 2
        damping = 1.0 / (1.0 + 0.5 * k_mu_sigma2) ** 2
        pk_mu = self.pk_dd[:, None] * (amplitude[:, None] + self.f_growth * self.mu2[None, :]) ** 2 * damping
        pk0 = 0.5 * np.sum(self.wmu[None, :] * pk_mu, axis=1)
        pk2 = 2.5 * np.sum(self.wmu[None, :] * pk_mu * self.leg2[None, :], axis=1)
        return {
            ell: ((self.g_nz * pole) @ self.kernel[ell]) / self.volume
            for ell, pole in ((0, pk0), (2, pk2))
        }


def fit_one(
    model: RawboxModel,
    data: dict[int, np.ndarray],
    sigma_single: dict[int, np.ndarray],
    *,
    smin: float,
    ells: tuple[int, ...],
    nphase: int,
) -> dict[str, Any]:
    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    mask = centers >= float(smin)
    vector = np.concatenate([data[ell][mask] for ell in ells])
    sigma = np.concatenate([sigma_single[ell][mask] for ell in ells])
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise RuntimeError("pilot phase scatter contains non-positive errors")

    def residual(theta: np.ndarray) -> np.ndarray:
        prediction = model(theta)
        return (vector - np.concatenate([prediction[ell][mask] for ell in ells])) / sigma

    starts = (
        np.array([0.0, 2.0, 4.0]),
        np.array([-100.0, 2.5, 8.0]),
        np.array([100.0, 1.7, 1.0]),
    )
    solutions = [least_squares(residual, start, bounds=([-500.0, 0.5, 0.0], [500.0, 5.0, 30.0])) for start in starts]
    solution = min(solutions, key=lambda item: float(np.dot(item.fun, item.fun)))
    chi2_single = float(np.dot(solution.fun, solution.fun))
    chi2_mean = float(nphase * chi2_single)
    dof = int(vector.size - 3)
    fisher = solution.jac.T @ solution.jac
    parameter_covariance_single = np.linalg.pinv(fisher, rcond=1.0e-12)
    parameter_sigma_single = np.sqrt(np.clip(np.diag(parameter_covariance_single), 0.0, np.inf))
    return {
        "smin_mpc_h": float(smin),
        "ells": list(ells),
        "success": bool(solution.success),
        "message": str(solution.message),
        "theta": {name: float(value) for name, value in zip(("fNL", "b1", "sigma_s"), solution.x, strict=True)},
        "sigma_single_diagonal_pilot": {
            name: float(value)
            for name, value in zip(("fNL", "b1", "sigma_s"), parameter_sigma_single, strict=True)
        },
        "fnl_center_over_sigma_single": float(abs(solution.x[0]) / parameter_sigma_single[0]),
        "chi2_single_diagonal": chi2_single,
        "chi2_mean_diagonal": chi2_mean,
        "dof": dof,
        "pte_single_diagonal": float(chi2_distribution.sf(chi2_single, dof)),
        "pte_mean_diagonal": float(chi2_distribution.sf(chi2_mean, dof)),
        "residual_rms_sigma_single": float(np.sqrt(np.mean(solution.fun**2))),
        "at_parameter_boundary": bool(
            np.any(np.isclose(solution.x, [-500.0, 0.5, 0.0], atol=[1.0, 0.01, 0.01]))
            or np.any(np.isclose(solution.x, [500.0, 5.0, 30.0], atol=[1.0, 0.01, 0.01]))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", nargs="+", default=["ph000", "ph001", "ph002"])
    args = parser.parse_args()
    phases = tuple(args.phases)
    output_root = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "rawbox" / "pilot"
    output = output_root / f"task43_rsd_rawbox_pilot_x{len(phases)}_fulldiscrete_lorentzian.npz"
    metadata_path = output.with_suffix(".json")
    plot_path = PLOT_ROOT / f"task43_rsd_rawbox_pilot_x{len(phases)}_fulldiscrete_lorentzian.pdf"
    if any(path.exists() for path in (output, metadata_path, plot_path)):
        raise FileExistsError("pilot outputs already exist; immutable rerun requires a new tag")
    started = time.perf_counter()
    stacks = {key: [] for key in ("xi0_real", "xi2_real", "xi0_rsd", "xi2_rsd")}
    input_paths = []
    bridge_max = []
    for phase in phases:
        path = (
            PROJECT_ROOT
            / "outputs/task43_outputs/rsd_validation/rawbox/summary"
            / f"task43_rsd_rawbox_AbacusSummit_base_c000_{phase}_mmin1p4e13_clustering.npz"
        )
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as payload:
            if not np.array_equal(np.asarray(payload["s_edges"], dtype="f8"), S_EDGES):
                raise RuntimeError(f"separation edges changed in {path}")
            for key in stacks:
                stacks[key].append(np.asarray(payload[key], dtype="f8"))
        input_paths.append(str(path))
        bridge_max.append(float(metadata["legacy_real_xi0_bridge"]["max_abs_xi0"]))
    stack = {key: np.stack(value) for key, value in stacks.items()}
    means = {key: np.mean(value, axis=0) for key, value in stack.items()}
    sigma_single = {
        0: np.std(stack["xi0_rsd"], axis=0, ddof=1),
        2: np.std(stack["xi2_rsd"], axis=0, ddof=1),
    }
    model = RawboxModel()
    fits = []
    for smin in (30.0, 50.0, 80.0, 100.0, 120.0):
        for ells in ((0,), (0, 2)):
            fits.append(
                fit_one(
                    model,
                    {0: means["xi0_rsd"], 2: means["xi2_rsd"]},
                    sigma_single,
                    smin=smin,
                    ells=ells,
                    nphase=len(phases),
                )
            )
    nominal = next(item for item in fits if item["smin_mpc_h"] == 50.0 and item["ells"] == [0])
    prediction = model(np.array([nominal["theta"][key] for key in ("fNL", "b1", "sigma_s")]))
    status = "pass" if (
        all(item["success"] and not item["at_parameter_boundary"] for item in fits)
        and max(bridge_max) == 0.0
        and all(np.all(np.isfinite(value)) for value in stack.values())
    ) else "fail"
    atomic_savez(
        output,
        s=0.5 * (S_EDGES[:-1] + S_EDGES[1:]),
        s_edges=S_EDGES,
        phases=np.asarray(phases),
        xi0_real=stack["xi0_real"],
        xi2_real=stack["xi2_real"],
        xi0_rsd=stack["xi0_rsd"],
        xi2_rsd=stack["xi2_rsd"],
        xi0_rsd_mean=means["xi0_rsd"],
        xi2_rsd_mean=means["xi2_rsd"],
        xi0_rsd_sigma_single=sigma_single[0],
        xi2_rsd_sigma_single=sigma_single[2],
        nominal_xi0_model=prediction[0],
        nominal_xi2_model=prediction[2],
    )

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex="col")
    for column, ell in enumerate((0, 2)):
        mean = means[f"xi{ell}_rsd"]
        sem = sigma_single[ell] / np.sqrt(len(phases))
        axes[0, column].errorbar(centers, centers**2 * mean, yerr=centers**2 * sem, fmt="o", ms=3, label=f"x{len(phases)} mean")
        axes[0, column].plot(centers, centers**2 * prediction[ell], lw=1.5, label="FullDiscrete RSD model")
        axes[0, column].set_ylabel(rf"$s^2\xi_{ell}(s)$")
        axes[0, column].legend(frameon=False, fontsize=9)
        residual = (mean - prediction[ell]) / sigma_single[ell]
        axes[1, column].axhline(0.0, color="0.4", lw=0.8)
        axes[1, column].plot(centers, residual, "o-", ms=3, lw=0.8)
        axes[1, column].set(xlabel=r"$s\,[h^{-1}{\rm Mpc}]$", ylabel=r"residual / $\sigma_{\rm single}$")
    fig.suptitle("Task 4.3.2 rawbox RSD three-phase numerical pilot")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    metadata = {
        "task": "task43_fit_rsd_rawbox_pilot",
        "status": status,
        "classification": "x3 numerical pilot only; diagonal phase scatter is not the production covariance",
        "phases": list(phases),
        "nphase": len(phases),
        "model": "FullDiscrete Kaiser x squared-Lorentzian FoG, p_fixed=1",
        "covariance_policy": {
            "fit_weights": "diagonal x3 phase scatter C_single",
            "posterior_error": "inverse local Fisher using C_single",
            "mean_goodness_of_fit": "chi2_mean=nphase*chi2_single",
            "production_claim": False,
        },
        "legacy_estimator_bridge_max_abs": max(bridge_max),
        "fits": fits,
        "nominal_diagnostic": nominal,
        "input_paths": input_paths,
        "output_path": str(output),
        "plot_path": str(plot_path),
        "elapsed_sec": time.perf_counter() - started,
    }
    atomic_write_json(metadata_path, metadata)
    print(json.dumps({"status": status, "nominal": nominal, "plot": str(plot_path)}, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
