#!/usr/bin/env python3
"""Fit RSD-lightcone P0 and compare with the frozen xi0(smin=50) result.

The P0 forward model follows Task 4.3 realspace: continuous multipole theory on
the window input grid, the jaxpower geometry convolution, observed kmin from
the lightcone volume, and a distinct mother-box theory cutoff.  Only the
Kaiser x squared-Lorentzian FoG factor and free sigma_s are added for RSD.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.interpolate import CubicSpline

from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import (
    PARAMETERS,
    fit_map,
    run_mcmc,
    set_affinity,
)
from task43_plot_rsd_rawbox_pk0_vs_xi0_contours import (
    COLORS,
    LABELS,
    draw_contour,
    plot_range,
    posterior_text,
)
from task43_rsd_common import (
    BOX_SIZE_MPC_H,
    OUTPUT_ROOT,
    PHASES,
    PLOT_ROOT,
    P_FIXED,
    atomic_savez,
    atomic_write_json,
    sha256_file,
)
from task43_rsd_lightcone_pk0_contract import (
    WINDOW_THEORY_KMIN,
    observed_fit_kmin,
)
from task43_rsd_model import DELTA_C, build_cache
from task43_theory_template import build_template_arrays, load_task41


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
DEFAULT_PK_PAYLOAD = (
    OUTPUT_ROOT
    / "lightcone/pk_summary/"
    "task43_rsd_lightcone_pk0_x25_task43realspace_kmin0p005065716_kmax0p10_l0only.npz"
)
DEFAULT_XI_PREFIX = (
    OUTPUT_ROOT
    / "lightcone/closure/"
    "task43_rsd_lightcone_x25_jaxpower_rrdeconv_rrnran300k_fulldiscrete_lorentzian"
)
DEFAULT_OUTPUT_PREFIX = (
    OUTPUT_ROOT
    / "lightcone/comparison/"
    "task43_rsd_lightcone_x25_pk0_vs_xi0_smin50_task43realspacewindow_l0only_longchain"
)
DEFAULT_FIT_PDF = (
    PLOT_ROOT / "task43_rsd_lightcone_x25_pk0_vs_xi0_smin50_task43realspacewindow_l0only_longchain.pdf"
)
DEFAULT_CONTOUR_PDF = (
    PLOT_ROOT / "task43_rsd_lightcone_x25_pk0_vs_xi0_smin50_task43realspacewindow_l0only_contours.pdf"
)
SN0_SCALE = 1.0e4


def interp_logk(k: np.ndarray, base_k: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(np.log(np.asarray(k, dtype="f8")), np.log(base_k), values)


class WindowConvolvedPk0Model:
    """Cubic-in-sigma surrogate after the full ell=0,2,4 window convolution."""

    def __init__(self, payload: Path, *, sigma_step: float = 0.05, nmu: int = 96) -> None:
        with np.load(payload, allow_pickle=False) as data:
            self.window = np.asarray(data["window_matrix"], dtype="f8")
            self.k_obs = np.asarray(data["k_obs"], dtype="f8")
            self.theory_k = np.asarray(data["theory_k"], dtype="f8")
            self.theory_ell = np.asarray(data["theory_ell"], dtype="i8")
            self.zeff = float(np.asarray(data["zeff"]).item())
            self.theory_kmin = float(np.asarray(data["window_theory_kmin"]).item())
        if self.window.shape != (self.k_obs.size, self.theory_k.size):
            raise RuntimeError(f"unexpected Task 4.3 window shape {self.window.shape}")
        if not np.array_equal(np.unique(self.theory_ell), [0, 2, 4]):
            raise RuntimeError(f"window input multipoles are {np.unique(self.theory_ell)}")
        self.nmu = int(nmu)
        self.mu, self.wmu = np.polynomial.legendre.leggauss(self.nmu)
        self.mu2 = self.mu**2
        self.task41 = load_task41()
        k_template = np.geomspace(1.0e-5, 20.0, 20000)
        self.template, self.cosmology_meta = build_template_arrays(
            self.task41, k_template, z=self.zeff, cosmology="abacus_c000"
        )
        self.alpha = interp_logk(self.theory_k, self.template["k"], self.template["alpha"])
        self.pk_dd = interp_logk(self.theory_k, self.template["k"], self.template["pk_dd"])
        cache = build_cache(
            zeff=self.zeff,
            boxsize=BOX_SIZE_MPC_H,
            kmax=3.0,
            ells=(0, 2),
            cosmology="abacus_c000",
        )
        self.theory_cache = cache
        with np.load(cache, allow_pickle=False) as data:
            self.f_growth = float(np.asarray(data["f_growth"]).item())
        self.support = self.theory_k >= self.theory_kmin - 1.0e-15
        self.sigma_grid = np.arange(0.0, 30.0 + 0.5 * float(sigma_step), float(sigma_step), dtype="f8")
        convolved = np.empty((self.sigma_grid.size, self.window.shape[0], 6), dtype="f8")
        for index, sigma_s in enumerate(self.sigma_grid):
            convolved[index] = self.window @ self._theory_basis(float(sigma_s))
        self.spline = CubicSpline(self.sigma_grid, convolved, axis=0)
        shot_vector = np.zeros(self.theory_k.size, dtype="f8")
        shot_vector[(self.theory_ell == 0) & self.support] = SN0_SCALE
        self.shot_response = self.window @ shot_vector

    def _theory_basis(self, sigma_s: float) -> np.ndarray:
        basis = np.zeros((self.theory_k.size, 6), dtype="f8")
        for ell in (0, 2, 4):
            selected = self.theory_ell == ell
            k = self.theory_k[selected]
            alpha = self.alpha[selected]
            pk_dd = self.pk_dd[selected]
            coeff = np.zeros(ell + 1, dtype="f8")
            coeff[ell] = 1.0
            legendre = np.polynomial.legendre.legval(self.mu, coeff)
            damping = 1.0 / (1.0 + 0.5 * (k[:, None] * self.mu[None, :] * float(sigma_s)) ** 2) ** 2
            prefactor = 0.5 * (2 * ell + 1)
            moment0 = prefactor * np.sum(self.wmu[None, :] * legendre[None, :] * damping, axis=1)
            moment2 = prefactor * np.sum(
                self.wmu[None, :] * legendre[None, :] * damping * self.mu2[None, :], axis=1
            )
            moment4 = prefactor * np.sum(
                self.wmu[None, :] * legendre[None, :] * damping * self.mu2[None, :] ** 2, axis=1
            )
            basis[selected, 0] = pk_dd * moment0
            basis[selected, 1] = pk_dd * alpha * moment0
            basis[selected, 2] = pk_dd * alpha**2 * moment0
            basis[selected, 3] = pk_dd * moment2
            basis[selected, 4] = pk_dd * alpha * moment2
            basis[selected, 5] = pk_dd * moment4
        basis[~self.support] = 0.0
        return basis

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        fnl, b1, sigma_s, sn0 = map(float, np.asarray(theta, dtype="f8")[:4])
        q = fnl * 2.0 * DELTA_C * (b1 - P_FIXED)
        f = self.f_growth
        coefficients = np.asarray([b1 * b1, 2.0 * b1 * q, q * q, 2.0 * b1 * f, 2.0 * q * f, f * f])
        return np.asarray(self.spline(sigma_s), dtype="f8") @ coefficients + sn0 * self.shot_response

    def direct(self, theta: np.ndarray, *, nmu: int | None = None) -> np.ndarray:
        fnl, b1, sigma_s, sn0 = map(float, np.asarray(theta, dtype="f8")[:4])
        nquad = self.nmu if nmu is None else int(nmu)
        mu, wmu = np.polynomial.legendre.leggauss(nquad)
        mu2 = mu**2
        bphi = 2.0 * DELTA_C * (b1 - P_FIXED)
        amplitude = b1 + fnl * bphi * self.alpha
        theory = np.zeros(self.theory_k.size, dtype="f8")
        for ell in (0, 2, 4):
            selected = self.theory_ell == ell
            k = self.theory_k[selected]
            coeff = np.zeros(ell + 1, dtype="f8")
            coeff[ell] = 1.0
            legendre = np.polynomial.legendre.legval(mu, coeff)
            damping = 1.0 / (1.0 + 0.5 * (k[:, None] * mu[None, :] * sigma_s) ** 2) ** 2
            pkmu = self.pk_dd[selected, None] * (amplitude[selected, None] + self.f_growth * mu2[None, :]) ** 2 * damping
            theory[selected] = 0.5 * (2 * ell + 1) * np.sum(wmu[None, :] * pkmu * legendre[None, :], axis=1)
        theory[(self.theory_ell == 0)] += sn0 * SN0_SCALE
        theory[~self.support] = 0.0
        return self.window @ theory

    def validate(self) -> dict[str, Any]:
        trials = (
            np.asarray([0.0, 2.55, 8.0, 0.0]),
            np.asarray([-75.0, 2.2, 3.37, 0.2]),
            np.asarray([80.0, 2.8, 12.43, -0.3]),
            np.asarray([15.0, 2.5, 0.07, 0.1]),
            np.asarray([-20.0, 2.6, 29.93, -0.1]),
        )
        rows = []
        for theta in trials:
            fast = self.evaluate(theta)
            direct = self.direct(theta)
            direct_hi = self.direct(theta, nmu=192)
            scale = max(1.0, float(np.linalg.norm(direct)))
            rows.append(
                {
                    "theta": theta.tolist(),
                    "surrogate_relative_l2": float(np.linalg.norm(fast - direct) / scale),
                    "nmu96_vs_192_relative_l2": float(np.linalg.norm(direct - direct_hi) / max(1.0, np.linalg.norm(direct_hi))),
                }
            )
        maximum = max(max(row["surrogate_relative_l2"], row["nmu96_vs_192_relative_l2"]) for row in rows)
        return {
            "status": "pass" if maximum < 1.0e-8 else "fail",
            "max_relative_l2": float(maximum),
            "trials": rows,
        }


def load_pk(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "pass" or metadata.get("output_sha256") != sha256_file(path):
        raise RuntimeError(f"unvalidated P0 payload: {path}")
    with np.load(path, allow_pickle=False) as data:
        result = {key: np.asarray(data[key]) for key in data.files}
    nfit = int(np.asarray(result["k_obs"]).size)
    if result["pk_stack"].shape != (25, nfit) or result["covariance_single_realization"].shape != (nfit, nfit):
        raise RuntimeError(f"P0 payload does not satisfy its x25/{nfit}-bin contract")
    volume = float(np.asarray(result["lightcone_volume"]).item())
    if not np.isclose(
        float(result["kmin_fit_observed"]), observed_fit_kmin(volume), rtol=0.0, atol=1.0e-15
    ):
        raise RuntimeError("observed fit kmin no longer equals 2pi/V^(1/3)")
    if not np.isclose(
        float(result["window_theory_kmin"]), WINDOW_THEORY_KMIN, rtol=0.0, atol=1.0e-15
    ):
        raise RuntimeError("window theory kmin no longer matches the Task 4.3 mother box")
    result.update(
        {
            "path": path,
            "metadata_path": metadata_path,
            "sha256": sha256_file(path),
            "metadata_sha256": sha256_file(metadata_path),
            "metadata": metadata,
        }
    )
    return result


def load_xi(prefix: Path) -> dict[str, Any]:
    json_path, npz_path = prefix.with_suffix(".json"), prefix.with_suffix(".npz")
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    if summary.get("nominal_mean_smin50_xi0", {}).get("formal_gic") is None:
        raise RuntimeError("lightcone xi closure lacks formal-GIC xi0,smin=50")
    if summary.get("output_npz_sha256") != sha256_file(npz_path):
        raise RuntimeError("lightcone xi closure NPZ hash gate failed")
    with np.load(npz_path, allow_pickle=False) as data:
        s = np.asarray(data["s"], dtype="f8")
        xi_mean_full = np.asarray(data["xi_multipoles_mean"], dtype="f8")[0]
        covariance_full = np.asarray(data["covariance_single_realization"], dtype="f8")
        prediction = np.asarray(data["formal_gic_model_map"], dtype="f8")
        chain = np.asarray(data["formal_gic_chain_flat"], dtype="f8")
    mask = s >= 50.0
    nbin = s.size
    covariance = covariance_full[:nbin, :nbin][np.ix_(mask, mask)]
    if prediction.shape != (int(np.count_nonzero(mask)),) or chain.ndim != 2 or chain.shape[1] != 3:
        raise RuntimeError(f"unexpected xi products prediction={prediction.shape}, chain={chain.shape}")
    fit = summary["nominal_mean_smin50_xi0"]["formal_gic"]
    posterior = {}
    for source, target in (("fnl_loc", "fNL"), ("b1", "b1"), ("sigma_s", "sigma_s")):
        row = dict(fit[source])
        row["sigma68"] = 0.5 * (float(row["q84"]) - float(row["q16"]))
        posterior[target] = row
    return {
        "json_path": json_path,
        "npz_path": npz_path,
        "json_sha256": sha256_file(json_path),
        "npz_sha256": sha256_file(npz_path),
        "s": s[mask],
        "mean": xi_mean_full[mask],
        "prediction": prediction,
        "covariance": covariance,
        "chain": chain,
        "posterior": posterior,
        "convergence": summary["mcmc_convergence"]["formal_gic"]["gates"],
        "mean_goodness": summary["mean_goodness_primary"],
        "source_status": summary["status"],
    }


def make_fit_plot(
    path: Path,
    *,
    pk: dict[str, Any],
    pk_prediction: np.ndarray,
    pk_chain: np.ndarray,
    xi: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary) as pdf:
        figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex="col", gridspec_kw={"height_ratios": [2.2, 1.0]})
        pk_mean = np.asarray(pk["pk_mean"], dtype="f8")
        pk_sigma_mean = np.sqrt(np.diag(pk["covariance_single_realization"]) / len(PHASES))
        axes[0, 0].errorbar(pk["k_obs"], pk_mean, yerr=pk_sigma_mean, fmt="o", ms=4, color=COLORS["pk0"], label=r"RSD lightcone $P_0$, x25 mean")
        axes[0, 0].plot(pk["k_obs"], pk_prediction, color=COLORS["pk0"], lw=1.6, ls="--", label="best fit")
        axes[0, 0].set(xscale="log", yscale="log", ylabel=r"$P_0(k)\ [(h^{-1}{\rm Mpc})^3]$")
        axes[0, 0].legend(frameon=False, fontsize=9)
        axes[1, 0].axhline(0.0, color="0.5", lw=0.8)
        axes[1, 0].plot(pk["k_obs"], (pk_mean - pk_prediction) / pk_sigma_mean, "o-", color=COLORS["pk0"], ms=4, lw=0.8)
        axes[1, 0].set(xscale="log", xlabel=r"$k\ [h\,{\rm Mpc}^{-1}]$", ylabel=r"residual / $\sigma_{\rm mean}$")

        s = np.asarray(xi["s"], dtype="f8")
        xi_sigma_mean = np.sqrt(np.diag(xi["covariance"]) / len(PHASES))
        axes[0, 1].errorbar(s, s**2 * xi["mean"], yerr=s**2 * xi_sigma_mean, fmt="o", ms=4, color=COLORS["xi0"], label=r"RSD lightcone $\xi_0$, x25 mean")
        axes[0, 1].plot(s, s**2 * xi["prediction"], color=COLORS["xi0"], lw=1.6, ls="--", label="formal-GIC best fit")
        axes[0, 1].set(ylabel=r"$s^2\xi_0(s)\ [(h^{-1}{\rm Mpc})^2]$")
        axes[0, 1].legend(frameon=False, fontsize=9)
        axes[1, 1].axhline(0.0, color="0.5", lw=0.8)
        axes[1, 1].plot(s, (xi["mean"] - xi["prediction"]) / xi_sigma_mean, "o-", color=COLORS["xi0"], ms=4, lw=0.8)
        axes[1, 1].set(xlabel=r"$s\ [h^{-1}{\rm Mpc}]$", ylabel=r"residual / $\sigma_{\rm mean}$")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        pk_flat = np.asarray(pk_chain, dtype="f8").reshape(-1, 4)
        xi_flat = np.asarray(xi["chain"], dtype="f8").reshape(-1, 3)
        figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
        for index, name in enumerate(("fNL", "b1", "sigma_s")):
            limits = plot_range(pk_flat[:, index], xi_flat[:, index], parameter=name)
            bins = np.linspace(*limits, 80)
            axes[index].hist(pk_flat[:, index], bins=bins, density=True, histtype="step", lw=1.8, color=COLORS["pk0"], label=r"$P_0(k)$")
            axes[index].hist(xi_flat[:, index], bins=bins, density=True, histtype="step", lw=1.8, color=COLORS["xi0"], label=r"$\xi_0(s)$")
            if name == "fNL":
                axes[index].axvline(0.0, color="0.5", lw=0.8, ls="--")
            axes[index].set(xlabel=LABELS[name], ylabel="posterior density")
        axes[0].legend(frameon=False)
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)
    temporary.replace(path)


def contour_context_text(pk: dict[str, Any]) -> str:
    """Return the geometry and fit-range annotation for the active payload."""
    geometry_contract = pk["metadata"].get("geometry_contract")
    zmin, zmax = {
        "wide_lrgall": (0.4, 1.1),
        "boxsafe_lrgall": (0.4, 0.8),
    }.get(geometry_contract, (0.6, 0.8))
    return (
        "AbacusSummit halo lightcone\n"
        "Redshift space\n"
        rf"${zmin:.1f} < z_{{\rm obs}} < {zmax:.1f}$" "\n"
        r"Fiducial $f_{\rm NL}=0$" "\n"
        rf"$k_{{\min,\max}}^{{\rm fit}}={float(pk['kmin_fit_observed']):.4f},\ 0.10\ h\,{{\rm Mpc}}^{{-1}}$" "\n"
        r"$s_{\min,\max}^{\rm fit}=50,\ 350\ h^{-1}{\rm Mpc}$"
    )


def make_contour_plot(
    path: Path,
    *,
    pk_chain: np.ndarray,
    xi_chain: np.ndarray,
    pk_posterior: dict[str, Any],
    xi_posterior: dict[str, Any],
    annotation_text: str | None = None,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pk = np.asarray(pk_chain, dtype="f8").reshape(-1, 4)[:, :3]
    xi = np.asarray(xi_chain, dtype="f8").reshape(-1, 3)
    names = ("fNL", "b1", "sigma_s")
    ranges = {name: plot_range(pk[:, i], xi[:, i], parameter=name) for i, name in enumerate(names)}
    figure, axes = plt.subplots(3, 3, figsize=(8.7, 8.2))
    contour_audit: dict[str, Any] = {}
    for irow, yname in enumerate(names):
        for icol, xname in enumerate(names):
            axis = axes[irow, icol]
            if icol > irow:
                axis.set_axis_off()
                continue
            if icol == irow:
                bins = np.linspace(*ranges[xname], 90)
                axis.hist(pk[:, icol], bins=bins, density=True, histtype="step", lw=1.9, color=COLORS["pk0"])
                axis.hist(xi[:, icol], bins=bins, density=True, histtype="step", lw=1.9, color=COLORS["xi0"])
                axis.set_xlim(*ranges[xname])
                axis.set_yticks([])
                if xname == "fNL":
                    axis.axvline(0.0, color="0.5", lw=0.8, ls="--")
            else:
                key = f"{xname}_vs_{yname}"
                contour_audit[key] = {
                    "xi0": draw_contour(axis, xi[:, icol], xi[:, irow], color=COLORS["xi0"], xlim=ranges[xname], ylim=ranges[yname], zorder=1),
                    "pk0": draw_contour(axis, pk[:, icol], pk[:, irow], color=COLORS["pk0"], xlim=ranges[xname], ylim=ranges[yname], zorder=3),
                }
                axis.set(xlim=ranges[xname], ylim=ranges[yname])
                if xname == "fNL":
                    axis.axvline(0.0, color="0.5", lw=0.8, ls="--", zorder=0)
                if yname == "fNL":
                    axis.axhline(0.0, color="0.5", lw=0.8, ls="--", zorder=0)
            if irow < 2:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel(LABELS[xname])
            if icol == 0 and irow > 0:
                axis.set_ylabel(LABELS[yname])
            elif icol > 0 and irow != icol:
                axis.tick_params(labelleft=False)
    handles = [
        plt.Line2D([], [], color=COLORS["pk0"], lw=2.0),
        plt.Line2D([], [], color=COLORS["xi0"], lw=2.0),
    ]
    labels = [
        rf"$P_0(k):\ f_{{\rm NL}}={posterior_text(pk_posterior['fNL'])}$",
        rf"$\xi_0(s):\ f_{{\rm NL}}={posterior_text(xi_posterior['fNL'])}$",
    ]
    # The first upper-triangle cell sits directly above the second diagonal
    # panel, so anchoring the legend to that cell keeps the triangle balanced.
    axes[0, 1].legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.04, 0.22),
        frameon=False,
        fontsize=13.0,
    )
    if annotation_text is not None:
        axes[0, 2].text(
            0.50,
            0.70,
            annotation_text,
            transform=axes[0, 2].transAxes,
            ha="center",
            va="center",
            fontsize=11.0,
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "white",
                "edgecolor": "0.45",
                "linewidth": 0.9,
            },
        )
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.96, wspace=0.0, hspace=0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.pdf")
    figure.savefig(temporary, format="pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    temporary.replace(path)
    return {
        "credible_contours": [0.68, 0.95],
        "plot_ranges": {name: list(value) for name, value in ranges.items()},
        "density_thresholds": contour_audit,
        "sample_counts": {"pk0": int(pk.shape[0]), "xi0": int(xi.shape[0])},
        "legend_anchor": {"axes": [0, 1], "location": "center left", "bbox_to_anchor": [0.04, 0.22]},
        "annotation_text": annotation_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--pk-payload", type=Path, default=DEFAULT_PK_PAYLOAD)
    parser.add_argument("--xi-prefix", type=Path, default=DEFAULT_XI_PREFIX)
    parser.add_argument("--sigma-grid-step", type=float, default=0.05)
    parser.add_argument("--nmu", type=int, default=96)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=430550)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--fit-pdf", type=Path, default=DEFAULT_FIT_PDF)
    parser.add_argument("--contour-pdf", type=Path, default=DEFAULT_CONTOUR_PDF)
    parser.add_argument(
        "--contour-json",
        type=Path,
        default=None,
        help="Optional audit path; by default it is stored beside --contour-pdf.",
    )
    args = parser.parse_args()
    if int(args.burnin) >= int(args.nsteps):
        raise ValueError("--burnin must be smaller than --nsteps")
    output_npz, output_json = args.output_prefix.with_suffix(".npz"), args.output_prefix.with_suffix(".json")
    contour_json = (
        args.contour_pdf.with_suffix(".json")
        if args.contour_json is None
        else args.contour_json
    )
    outputs = (output_npz, output_json, args.fit_pdf, args.contour_pdf, contour_json)
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"immutable lightcone comparison output exists: {[str(path) for path in outputs if path.exists()]}")

    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))
    pk = load_pk(args.pk_payload)
    xi = load_xi(args.xi_prefix)
    model = WindowConvolvedPk0Model(
        args.pk_payload, sigma_step=float(args.sigma_grid_step), nmu=int(args.nmu)
    )
    validation = model.validate()
    if validation["status"] != "pass":
        raise RuntimeError(f"window-convolved P0 surrogate failed: {validation}")
    covariance = np.asarray(pk["covariance_single_realization"], dtype="f8")
    pk_mean = np.asarray(pk["pk_mean"], dtype="f8")
    nominal = fit_map(model, pk_mean, covariance)
    mcmc, chain, logp = run_mcmc(
        model,
        pk_mean,
        covariance,
        nominal,
        nwalkers=int(args.nwalkers),
        nsteps=int(args.nsteps),
        burnin=int(args.burnin),
        seed=int(args.seed),
    )
    prediction = np.asarray(nominal["prediction"], dtype="f8")
    empirical_covariance = np.asarray(pk["pk_scatter_covariance"], dtype="f8")
    analytic_sigma = np.sqrt(np.diag(covariance))
    empirical_sigma = np.sqrt(np.diag(empirical_covariance))
    covariance_diagnostic = {
        "sample_std_over_jaxpower_sigma": (empirical_sigma / analytic_sigma).tolist(),
        "sample_std_over_jaxpower_sigma_median": float(np.median(empirical_sigma / analytic_sigma)),
        "covariance_relative_frobenius": float(np.linalg.norm(empirical_covariance - covariance) / np.linalg.norm(covariance)),
    }
    comparison = {}
    for name in ("fNL", "b1", "sigma_s"):
        p0_row, xi_row = mcmc["posterior"][name], xi["posterior"][name]
        denominator = np.hypot(float(p0_row["sigma68"]), float(xi_row["sigma68"]))
        comparison[name] = {
            "p0_median": float(p0_row["q50"]),
            "p0_sigma68": float(p0_row["sigma68"]),
            "xi0_median": float(xi_row["q50"]),
            "xi0_sigma68": float(xi_row["sigma68"]),
            "center_difference_p0_minus_xi0": float(p0_row["q50"] - xi_row["q50"]),
            "difference_over_independent_quadrature_sigma_diagnostic": float((p0_row["q50"] - xi_row["q50"]) / denominator),
        }

    make_fit_plot(args.fit_pdf, pk=pk, pk_prediction=prediction, pk_chain=chain, xi=xi)
    contour_audit = make_contour_plot(
        args.contour_pdf,
        pk_chain=chain,
        xi_chain=xi["chain"],
        pk_posterior=mcmc["posterior"],
        xi_posterior=xi["posterior"],
        annotation_text=contour_context_text(pk),
    )
    atomic_savez(
        output_npz,
        phases=np.asarray(PHASES),
        k=np.asarray(pk["k_obs"], dtype="f8"),
        k_edges=np.asarray(pk["k_edges"], dtype="f8"),
        pk0_rsd_by_phase=np.asarray(pk["pk_stack"], dtype="f8"),
        pk0_rsd_mean=pk_mean,
        pk0_model_map=prediction,
        pk0_covariance_single=covariance,
        pk0_covariance_mean=covariance / len(PHASES),
        pk0_empirical_covariance=empirical_covariance,
        pk0_chain_by_step=chain,
        pk0_log_probability_by_step=logp,
        s=np.asarray(xi["s"], dtype="f8"),
        xi0_rsd_mean=np.asarray(xi["mean"], dtype="f8"),
        xi0_model_map=np.asarray(xi["prediction"], dtype="f8"),
        xi0_covariance_single=np.asarray(xi["covariance"], dtype="f8"),
        xi0_chain_flat=np.asarray(xi["chain"], dtype="f8"),
    )
    contour_metadata = {
        "task": "task43_plot_rsd_lightcone_pk0_vs_xi0_contours",
        "status": "pass",
        "scope": "RSD lightcone P0(kmax=0.10) vs formal-GIC xi0(smin=50), observed ell=0 only",
        **contour_audit,
        "output_pdf": str(args.contour_pdf),
        "output_pdf_sha256": sha256_file(args.contour_pdf),
        "comparison_npz": str(output_npz),
        "comparison_npz_sha256": sha256_file(output_npz),
    }
    atomic_write_json(contour_json, contour_metadata)
    chains_pass = bool(all(mcmc["gates"].values()) and all(xi["convergence"].values()))
    validation_pass = bool(
        chains_pass
        and float(nominal["pte_mean_covariance"]) > 0.05
        and float(xi["mean_goodness"]["pte"]) > 0.05
    )
    payload = {
        "task": "task43_fit_rsd_lightcone_pk0_vs_xi0_smin50",
        "status": "complete" if chains_pass else "mcmc_diagnostic_failed",
        "science_validation": "pass" if validation_pass else "validation_failed",
        "scope": {
            "geometry": {
                "wide_lrgall": "positive-octant radial-LOS lightcone, 0.4<zobs<1.1",
                "boxsafe_lrgall": "positive-octant radial-LOS lightcone, 0.4<zobs<0.8",
            }.get(
                pk["metadata"].get("geometry_contract"),
                "positive-octant radial-LOS lightcone, 0.6<zobs<0.8",
            ),
            "observed_multipoles": [0],
            "window_input_multipoles": [0, 2, 4],
            "explicitly_excluded": ["observed ell=2", "smin scan"],
            "pk0": (
                f"{int(np.asarray(pk['k_obs']).size)} Task-4.3-policy bins; "
                f"observed kmin={float(pk['kmin_fit_observed']):.9g}, kmax=0.10 h/Mpc"
            ),
            "xi0": "formal-GIC, s bin edges 50..350 Mpc/h, smin=50 only",
        },
        "cpu_affinity": cpus,
        "nphase": len(PHASES),
        "phases": list(PHASES),
        "pk_payload": str(args.pk_payload),
        "pk_payload_sha256": pk["sha256"],
        "model": {
            "name": "Task-4.3 continuous ell024 window convolution of Kaiser x squared-Lorentzian-FoG PNG",
            "free_parameters": list(PARAMETERS),
            "p_fixed": P_FIXED,
            "sn0_scale": SN0_SCALE,
            "f_growth": model.f_growth,
            "theory_cache": str(model.theory_cache),
            "theory_cache_sha256": sha256_file(model.theory_cache),
            "observed_fit_kmin": float(pk["kmin_fit_observed"]),
            "window_theory_kmin": model.theory_kmin,
            "kmin_separation_gate": bool(float(pk["kmin_fit_observed"]) > model.theory_kmin),
            "window_forward_model": "W_geom[P0,P2,P4]",
            "integral_constraint_policy": "Task 4.3 geometry-only P(k) base; no standalone GIC and no radial-RIC operator",
            "surrogate_validation": validation,
            "cosmology_meta": model.cosmology_meta,
        },
        "covariance": {
            "definition": "Task 4.3 jaxpower Gaussian survey-window P0 covariance with RSD ell024 fiducial",
            "quoted_posterior": "single-lightcone covariance; never divided by 25",
            "mean_fit_quality": "Cmean=Csingle/25",
            "x25_scatter_diagnostic": covariance_diagnostic,
        },
        "pk0": {"nominal": nominal, "mcmc": mcmc},
        "xi0": {
            "source_json": str(xi["json_path"]),
            "source_json_sha256": xi["json_sha256"],
            "source_npz": str(xi["npz_path"]),
            "source_npz_sha256": xi["npz_sha256"],
            "posterior": xi["posterior"],
            "convergence": xi["convergence"],
            "mean_goodness": xi["mean_goodness"],
            "source_status": xi["source_status"],
            "extraction_policy": "only formal-GIC ell=0, smin=50 is loaded",
        },
        "comparison": comparison,
        "gates": {
            "mcmc_all": chains_pass,
            "pk0_mean_pte_above_0p05": bool(float(nominal["pte_mean_covariance"]) > 0.05),
            "xi0_mean_pte_above_0p05": bool(float(xi["mean_goodness"]["pte"]) > 0.05),
        },
        "output_npz": str(output_npz),
        "output_npz_sha256": sha256_file(output_npz),
        "output_fit_pdf": str(args.fit_pdf),
        "output_fit_pdf_sha256": sha256_file(args.fit_pdf),
        "output_contour_pdf": str(args.contour_pdf),
        "output_contour_pdf_sha256": sha256_file(args.contour_pdf),
        "elapsed_sec": float(time.perf_counter() - started),
    }
    atomic_write_json(output_json, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "science_validation": payload["science_validation"],
                "pk0_posterior": mcmc["posterior"],
                "output": str(output_json),
            },
            sort_keys=True,
        )
    )
    if not chains_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
