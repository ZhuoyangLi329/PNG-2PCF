#!/usr/bin/env python3
"""Plot the 25-phase mean raw-box 2PCF and power spectrum for EZmock tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs/task43_outputs/ezmock_rawbox_z0p725_mmin1p4e13/summary"
    / "task43_ezmock_rawbox_z0p725_mmin1p4e13_x25.npz"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "plots/task43/ezmock_rawbox"
    / "task43_ezmock_rawbox_z0p725_mmin1p4e13_x25_mean_2pcf_pk"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savefig(figure: plt.Figure, path: Path, **kwargs: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    figure.savefig(tmp, **kwargs)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ezmock-input", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--xi-plot-max", type=float, default=550.0)
    parser.add_argument("--comparison-label", type=str, default="Pipeline smoke")
    args = parser.parse_args()

    if not 50.0 < args.xi_plot_max <= 550.0:
        raise ValueError("--xi-plot-max must be in (50, 550]")

    with np.load(args.input, allow_pickle=False) as data:
        phases = np.asarray(data["phases"])
        s = np.asarray(data["s"], dtype="f8")
        xi_all = np.asarray(data["xi0_all"], dtype="f8")
        xi_mean = np.asarray(data["xi0_mean"], dtype="f8")
        xi_std = np.asarray(data["xi0_std"], dtype="f8")
        k = np.asarray(data["k"], dtype="f8")
        valid_k = np.asarray(data["k_valid_mask"], dtype=bool)
        pk_all = np.asarray(data["pk0_all"], dtype="f8")
        pk_mean = np.asarray(data["pk0_mean"], dtype="f8")
        pk_std = np.asarray(data["pk0_std"], dtype="f8")
        ndata = np.asarray(data["ndata"], dtype="i8")
        redshift = np.asarray(data["redshift"], dtype="f8")

    ezmock = None
    if args.ezmock_input is not None:
        with np.load(args.ezmock_input, allow_pickle=False) as data:
            ezmock = {
                "seeds": np.asarray(data["seeds"], dtype="i8"),
                "s": np.asarray(data["s"], dtype="f8"),
                "xi0_all": np.asarray(data["xi0_all"], dtype="f8"),
                "xi0_mean": np.asarray(data["xi0_mean"], dtype="f8"),
                "k": np.asarray(data["k"], dtype="f8"),
                "k_valid_mask": np.asarray(data["k_valid_mask"], dtype=bool),
                "pk0_all": np.asarray(data["pk0_all"], dtype="f8"),
                "pk0_mean": np.asarray(data["pk0_mean"], dtype="f8"),
                "parameter_names": np.asarray(data["parameter_names"]).astype(str),
                "parameter_values": np.asarray(data["parameter_values"], dtype="f8"),
                "fix_amplitude": bool(np.asarray(data["fix_amplitude"]).item()),
                "ngrid": int(np.asarray(data["ngrid"]).item()),
            }

    if phases.size != 25 or xi_all.shape != (25, s.size) or pk_all.shape != (25, k.size):
        raise ValueError(f"unexpected realization shapes: phases={phases.size}, xi={xi_all.shape}, pk={pk_all.shape}")
    if not np.all(np.isfinite(xi_all)) or not np.all(np.isfinite(pk_all[:, valid_k])):
        raise ValueError("non-finite values in plotted bins")
    if ezmock is not None:
        ezmock_nreal = int(ezmock["seeds"].size)
        if ezmock_nreal < 1 or ezmock["xi0_all"].shape != (ezmock_nreal, s.size):
            raise ValueError("invalid EZmock calibration realization arrays")
        if not ezmock["fix_amplitude"]:
            raise ValueError("calibration overlay must have FIX_AMPLITUDE=T")
        if not np.array_equal(ezmock["s"], s) or not np.array_equal(ezmock["k"], k, equal_nan=True):
            raise ValueError("EZmock and Abacus plotting grids differ")
        if not np.all(np.isfinite(ezmock["xi0_all"])):
            raise ValueError("non-finite EZmock xi0 values")
        if not np.all(np.isfinite(ezmock["pk0_all"][:, ezmock["k_valid_mask"]])):
            raise ValueError("non-finite EZmock P0 values")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 10.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)

    axis = axes[0]
    scaled_all = s[None, :] ** 2 * xi_all
    scaled_mean = s**2 * xi_mean
    scaled_std = s**2 * xi_std
    for values in scaled_all:
        axis.plot(s, values, color="0.65", lw=0.55, alpha=0.20, zorder=1)
    axis.fill_between(s, scaled_mean - scaled_std, scaled_mean + scaled_std, color="#2878B5", alpha=0.22, lw=0, label=r"phase scatter ($1\sigma$)", zorder=2)
    axis.plot(s, scaled_mean, color="#145A8D", lw=2.2, label="Abacus x25 mean", zorder=3)
    if ezmock is not None:
        for values in s[None, :] ** 2 * ezmock["xi0_all"]:
            axis.plot(s, values, color="#E18727", lw=0.65, alpha=0.20, zorder=3)
        axis.plot(
            s,
            s**2 * ezmock["xi0_mean"],
            color="#E18727",
            lw=2.2,
            label=f"EZmock x{ezmock_nreal} mean (fixed amplitude)",
            zorder=4,
        )
    axis.axhline(0.0, color="0.25", lw=0.8, ls="--", alpha=0.75)
    axis.set_xlim(50.0, float(args.xi_plot_max))
    axis.set_xlabel(r"$s\ [h^{-1}\,\mathrm{Mpc}]$")
    axis.set_ylabel(r"$s^2\,\xi_0(s)\ [(h^{-1}\,\mathrm{Mpc})^2]$")
    axis.set_title("Periodic-box 2PCF")
    axis.legend(loc="upper right", frameon=False)

    axis = axes[1]
    kval = k[valid_k]
    pk_valid_all = pk_all[:, valid_k]
    pk_valid_mean = pk_mean[valid_k]
    pk_valid_std = pk_std[valid_k]
    for values in pk_valid_all:
        axis.plot(kval, values, color="0.65", lw=0.55, alpha=0.18, zorder=1)
    axis.fill_between(kval, pk_valid_mean - pk_valid_std, pk_valid_mean + pk_valid_std, color="#D9534F", alpha=0.22, lw=0, label=r"phase scatter ($1\sigma$)", zorder=2)
    axis.plot(kval, pk_valid_mean, color="#B52B27", lw=2.2, label="Abacus x25 mean", zorder=3)
    if ezmock is not None:
        ez_valid = valid_k & ezmock["k_valid_mask"]
        for values in ezmock["pk0_all"][:, ez_valid]:
            axis.plot(k[ez_valid], values, color="#2E8B57", lw=0.65, alpha=0.20, zorder=3)
        axis.plot(
            k[ez_valid],
            ezmock["pk0_mean"][ez_valid],
            color="#2E8B57",
            lw=2.2,
            label=f"EZmock x{ezmock_nreal} mean (fixed amplitude)",
            zorder=4,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(3.0e-3, 3.1e-1)
    axis.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    axis.set_ylabel(r"$P_0(k)\ [(h^{-1}\,\mathrm{Mpc})^3]$")
    axis.set_title("Periodic-box power spectrum")
    axis.legend(loc="lower left", frameon=False)

    if ezmock is None:
        title = (
            rf"AbacusSummit base c000 raw boxes: $N_{{\rm phase}}=25$, "
            rf"$\bar z={np.mean(redshift):.3f}$, $M_{{\rm min}}=1.4\times10^{{13}}\,h^{{-1}}M_\odot$"
        )
    else:
        params = dict(zip(ezmock["parameter_names"].tolist(), ezmock["parameter_values"].tolist(), strict=True))
        title = (
            rf"{args.comparison_label}: Abacus x25 vs EZmock x{ezmock_nreal} fixed-amplitude; "
            rf"$\rho_c={params['rho_c']:.3g}$, $\rho_{{\rm exp}}={params['rho_exp']:.3g}$, "
            rf"$b={params['pdf_base']:.3g}$, $\sigma_v={params['sigma_v']:.3g}$, "
            rf"$N_{{\rm grid}}={ezmock['ngrid']}$"
        )
    figure.suptitle(title, fontsize=14)
    output_pdf = args.output_prefix.with_suffix(".pdf")
    output_json = args.output_prefix.with_suffix(".json")
    atomic_savefig(figure, output_pdf, bbox_inches="tight")
    plt.close(figure)

    audit = {
        "task": "task43_plot_ezmock_rawbox_mean_clustering",
        "status": "done",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output_pdf": str(output_pdf),
        "nphase": int(phases.size),
        "ndata_mean": float(np.mean(ndata)),
        "redshift_mean": float(np.mean(redshift)),
        "xi_nbins": int(s.size),
        "pk_valid_nbins": int(np.count_nonzero(valid_k)),
        "xi_plot_max": float(args.xi_plot_max),
        "plot_policy": "thin phase curves, 25-phase mean, and phase-to-phase 1-sigma scatter",
    }
    if ezmock is not None:
        audit.update(
            {
                "ezmock_input": str(args.ezmock_input),
                "ezmock_input_sha256": sha256(args.ezmock_input),
                "ezmock_nreal": int(ezmock["seeds"].size),
                "ezmock_seeds": ezmock["seeds"].tolist(),
                "ezmock_fixed_amplitude": bool(ezmock["fix_amplitude"]),
                "ezmock_ngrid": int(ezmock["ngrid"]),
                "ezmock_parameters": dict(
                    zip(ezmock["parameter_names"].tolist(), ezmock["parameter_values"].tolist(), strict=True)
                ),
                "comparison_label": str(args.comparison_label),
                "plot_policy": f"Abacus x25 mean/scatter with EZmock x{ezmock_nreal} fixed-amplitude mean overlay",
            }
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = output_json.with_name(f".{output_json.name}.{os.getpid()}.tmp")
    tmp_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_json.replace(output_json)
    print(f"[done] {output_pdf}")


if __name__ == "__main__":
    main()
