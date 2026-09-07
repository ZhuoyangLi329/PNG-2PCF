#!/usr/bin/env python3
"""
Overlay "window" 2PCF modeling curves for multiple masscuts on a single figure.

Design goals:
- Recompute the model xi0(r) (window only; baseline not shown) from stored best-fit parameters
  in each tag's `model_validation_normexp/bestfit_params.json`.
- Keep FFTLog k-integration range consistent with mission7 defaults (kint_min=1e-4, kint_max=20).
- Save the combined plot under agent/mission7_log/ (caller decides output path).

Run in an environment that can import: desilike, cosmoprimo, scipy, matplotlib.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def build_log_taper_window(k_array: np.ndarray, kmin: float, kmax: float, frac: float) -> np.ndarray:
    """Cosine taper in log-k to reduce FFTLog ringing at the boundaries."""
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


def build_ir_param_window_normexp(k_array: np.ndarray, k_fund: float, x_power: float) -> np.ndarray:
    """
    Normalized exp_power IR window:
      - k < k_f:  W(k) = [1 - exp(-(k/k_f)^x)] / [1 - exp(-1)]
      - k >= k_f: W(k) = 1
    """
    w = np.ones_like(k_array, dtype=np.float64)
    m = k_array < k_fund
    if np.any(m):
        ratio = np.clip(k_array[m] / max(k_fund, 1e-30), 0.0, None)
        norm = 1.0 - np.exp(-1.0)
        w[m] = (1.0 - np.exp(-(ratio**x_power))) / max(norm, 1e-30)
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
    k_fund: float,
    x_power: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute xi0(r) via FFTLog (scipy.fft.fht) with optional IR window."""
    from scipy.fft import fht, fhtoffset

    window_taper = build_log_taper_window(k_grid, kmin, kmax, taper_frac)
    if use_param_ir_window:
        window_ir = build_ir_param_window_normexp(k_grid, k_fund=k_fund, x_power=x_power)
    else:
        window_ir = np.ones_like(k_grid, dtype=np.float64)

    p0_eff = p0_grid * window_taper * window_ir

    dln = float(np.log(k_grid[1] / k_grid[0]))
    offset = float(fhtoffset(dln, mu=float(mu), initial=0.0, bias=float(bias)))

    a_in = (k_grid**1.5) * p0_eff
    a_out = fht(a_in, dln=dln, mu=float(mu), offset=offset, bias=float(bias))

    n = k_grid.size
    j = np.arange(n)
    j_center = (n - 1) / 2.0
    ln_kc = 0.5 * (np.log(k_grid[0]) + np.log(k_grid[-1]))
    r_grid = np.exp((-ln_kc + offset) + (j - j_center) * dln)

    coef = np.sqrt(np.pi / 2.0) / (2.0 * np.pi**2)
    xi0 = coef * a_out / np.maximum(r_grid**1.5, 1e-300)
    return r_grid, xi0


def _discover_tags(masscut_scan_dir: Path, subdir: str) -> List[str]:
    tags: List[str] = []
    for p in sorted(masscut_scan_dir.glob("*/" + subdir + "/bestfit_params.json")):
        # masscut_scan/<tag>/<subdir>/bestfit_params.json
        try:
            tag = p.parents[1].name
        except Exception:
            continue
        tags.append(tag)
    # unique, stable order
    seen = set()
    out: List[str] = []
    for t in tags:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _load_bestfit_params(path: Path) -> Dict[str, float]:
    with path.open() as f:
        d = json.load(f)
    # ensure plain floats
    return {k: float(v) for k, v in d.items()}


def _compute_xi_window_on_rgrid(
    *,
    bestfit_params: Dict[str, float],
    r_data: np.ndarray,
    box_size: float,
    kint_min: float,
    kint_max: float,
    fftlog_n: int,
    fftlog_padding: float,
    fftlog_mu: float,
    fftlog_bias: float,
    edge_taper_frac: float,
    unit_z: float,
) -> np.ndarray:
    from cosmoprimo import Cosmology
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles

    cosmo_unit = Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )

    k_fund = 2.0 * np.pi / float(box_size)
    x_power = float(4.0 * (box_size / 1000.0))

    k_grid = np.geomspace(kint_min / fftlog_padding, kint_max * fftlog_padding, fftlog_n)

    template = FixedPowerSpectrumTemplate(z=float(unit_z), fiducial=cosmo_unit)
    theory = PNGTracerPowerSpectrumMultipoles(k=k_grid, template=template, mode="b-p")

    # Keep parameter fixed/free states consistent with the fit script:
    # p, sn0 fixed; sigmas free (but its best-fit value is stored and passed below).
    theory.init.params["p"].update(fixed=True, value=float(bestfit_params.get("p", 1.2)))
    theory.init.params["sn0"].update(fixed=True, value=float(bestfit_params.get("sn0", 0.0)))
    theory.init.params["sigmas"].update(fixed=False, value=float(bestfit_params.get("sigmas", 0.0)))

    # desilike uses fnl_loc, b1, sigmas, etc.
    theory(**bestfit_params)
    p0 = np.asarray(theory.power[0], dtype=np.float64)

    r_grid, xi_grid = xi0_from_p0_fftlog(
        k_grid=k_grid,
        p0_grid=p0,
        kmin=float(kint_min),
        kmax=float(kint_max),
        mu=float(fftlog_mu),
        bias=float(fftlog_bias),
        taper_frac=float(edge_taper_frac),
        use_param_ir_window=True,
        k_fund=float(k_fund),
        x_power=float(x_power),
    )

    order = np.argsort(r_grid)
    xi_on_data = np.interp(r_data, r_grid[order], xi_grid[order])
    return xi_on_data


def _read_pcf_mean_std(pcf_dir: Path, rid_min: int, rid_max: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import re

    def _rid(name: str) -> int:
        m = re.search(r"N([0-9]+)", name)
        return int(m.group(1)) if m else -1

    files = sorted(pcf_dir.glob("pcf_rsd_N*.dat"), key=lambda p: _rid(p.name))
    files = [p for p in files if rid_min <= _rid(p.name) <= rid_max]
    if not files:
        raise FileNotFoundError(f"No pcf files found in {pcf_dir} for rid=[{rid_min},{rid_max}]")

    r_grid = None
    xis: List[np.ndarray] = []
    for fp in files:
        arr = np.loadtxt(fp, comments="#")
        if r_grid is None:
            r_grid = arr[:, 0]
        xis.append(arr[:, 3])

    stack = np.asarray(xis, dtype=np.float64)
    xi_mean = np.mean(stack, axis=0)
    xi_std = np.std(stack, axis=0, ddof=1)
    assert r_grid is not None
    return np.asarray(r_grid, dtype=np.float64), xi_mean, xi_std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm")
    ap.add_argument("--subdir", default="model_validation_normexp", help="per-tag model output subdir")
    ap.add_argument("--tags", default="", help="comma-separated tags; empty=auto-discover")
    ap.add_argument("--out", required=True, help="output png path")

    # Data range controls
    ap.add_argument("--realization-min", type=int, default=1)
    ap.add_argument("--realization-max", type=int, default=50)

    # FFTLog/model config (keep aligned with mission7 defaults unless you know what you're doing)
    ap.add_argument("--box-size", type=float, default=1000.0)
    ap.add_argument("--kint-min", type=float, default=1e-4)
    ap.add_argument("--kint-max", type=float, default=20.0)
    ap.add_argument("--fftlog-n", type=int, default=4096)
    ap.add_argument("--fftlog-padding", type=float, default=4.0)
    ap.add_argument("--fftlog-mu", type=float, default=0.5)
    ap.add_argument("--fftlog-bias", type=float, default=0.0)
    ap.add_argument("--edge-taper-frac", type=float, default=0.06)
    ap.add_argument("--unit-z", type=float, default=1.0)

    ap.add_argument("--plot-data", action="store_true", help="also plot measured mean r^2 xi with errors (faint)")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    masscut_scan_dir = project_root / "masscut_scan"

    if args.tags.strip():
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    else:
        tags = _discover_tags(masscut_scan_dir, args.subdir)

    if not tags:
        raise SystemExit(f"No tags found under {masscut_scan_dir} with subdir={args.subdir}")

    series: List[Dict[str, object]] = []
    for tag in tags:
        bestfit_path = masscut_scan_dir / tag / args.subdir / "bestfit_params.json"
        pcf_dir = masscut_scan_dir / tag / "pcf"
        if not bestfit_path.exists():
            print(f"[SKIP] missing bestfit_params.json: {bestfit_path}")
            continue
        if not pcf_dir.is_dir():
            print(f"[SKIP] missing pcf dir: {pcf_dir}")
            continue

        bestfit = _load_bestfit_params(bestfit_path)
        r, xi_mean, xi_std = _read_pcf_mean_std(pcf_dir, args.realization_min, args.realization_max)
        xi_model = _compute_xi_window_on_rgrid(
            bestfit_params=bestfit,
            r_data=r,
            box_size=args.box_size,
            kint_min=args.kint_min,
            kint_max=args.kint_max,
            fftlog_n=args.fftlog_n,
            fftlog_padding=args.fftlog_padding,
            fftlog_mu=args.fftlog_mu,
            fftlog_bias=args.fftlog_bias,
            edge_taper_frac=args.edge_taper_frac,
            unit_z=args.unit_z,
        )

        series.append(
            {
                "tag": tag,
                "r": r,
                "xi_model": xi_model,
                "xi_mean": xi_mean,
                "xi_std": xi_std,
                "bestfit": bestfit,
            }
        )

    if not series:
        raise SystemExit("No usable tags after filtering (missing bestfit/pcf).")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["grid.linestyle"] = "--"

    fig, ax = plt.subplots(1, 1, figsize=(9.4, 6.2))

    # Stable color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(series))))

    for i, s in enumerate(series):
        tag = str(s["tag"])
        r = np.asarray(s["r"], dtype=float)
        xi_model = np.asarray(s["xi_model"], dtype=float)
        xi_mean = np.asarray(s["xi_mean"], dtype=float)
        xi_std = np.asarray(s["xi_std"], dtype=float)
        bestfit = s["bestfit"]  # type: ignore[assignment]
        fnl = float(bestfit.get("fnl_loc", np.nan))
        b1 = float(bestfit.get("b1", np.nan))

        c = colors[i % len(colors)]
        ax.plot(r, r**2 * xi_model, "-", lw=2.2, color=c, label=f"{tag} (b1={b1:.3f}, fnl={fnl:.1f})")

        if args.plot_data:
            ax.errorbar(
                r,
                r**2 * xi_mean,
                yerr=r**2 * xi_std,
                fmt="o",
                ms=2.4,
                lw=0.8,
                capsize=1.6,
                color=c,
                alpha=0.25,
            )

    ax.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$r^2\xi_0(r)$ (model window only)")
    ax.set_title("Mission7: 2PCF model window curves overlay (normalized exp_power)")
    ax.legend(fontsize=8.6, ncol=1, frameon=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out}")


if __name__ == "__main__":
    main()
