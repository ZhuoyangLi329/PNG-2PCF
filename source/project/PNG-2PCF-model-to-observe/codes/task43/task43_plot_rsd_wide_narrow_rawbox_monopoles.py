#!/usr/bin/env python3
"""Compare wide/narrow lightcone and rawbox RSD monopoles using x25 means."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from task43_rsd_common import atomic_savez, atomic_write_json, sha256_file


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
RSD_ROOT = PROJECT_ROOT / "outputs/task43_outputs/rsd_validation"
DEFAULT_WIDE_SUMMARY = RSD_ROOT / "lightcone_wide_zobs0p4_1p1/summary/task43_rsd_wide_lightcone_x25_mean_xi0_s30_350_ds10.npz"
DEFAULT_NARROW_SUMMARY = RSD_ROOT / "lightcone/summary/task43_rsd_lightcone_x25_mean_xi02_s30_350_ds10.npz"
DEFAULT_RAWBOX_CLOSURE = RSD_ROOT / "rawbox/closure/task43_rsd_rawbox_x25_fulldiscrete_lorentzian.npz"
DEFAULT_WIDE_COMPARE = RSD_ROOT / "lightcone_wide_zobs0p4_1p1/comparison/task43_rsd_wide_lightcone_x25_pk0_vs_xi0_smin50_l0only_longchain.npz"
DEFAULT_NARROW_COMPARE = RSD_ROOT / "lightcone/comparison/task43_rsd_lightcone_x25_pk0_vs_xi0_smin50_task43realspacewindow_l0only_longchain.npz"
DEFAULT_RAWBOX_COMPARE = RSD_ROOT / "rawbox/comparison/task43_rsd_rawbox_x25_pk0_kmin0p003_vs_xi0_smin50_l0only_longchain.npz"
DEFAULT_PDF = PROJECT_ROOT / "plots/task43/rsd_validation/task43_rsd_wide_vs_narrow_vs_rawbox_monopoles.pdf"
DEFAULT_OUTPUT_PREFIX = RSD_ROOT / "audits/task43_rsd_wide_vs_narrow_vs_rawbox_monopoles"


def load_xi_summary(path: Path, *, expected_ells: tuple[int, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        ells = tuple(int(value) for value in np.asarray(payload["ells"]).ravel())
        if ells != expected_ells:
            raise RuntimeError(f"{path}: multipoles {ells} != {expected_ells}")
        return {
            "s": np.asarray(payload["s"], dtype="f8"),
            "phases": np.asarray(payload["phases"]).astype(str),
            "xi": np.asarray(payload["xi_multipoles_by_phase"], dtype="f8")[:, ells.index(0)],
        }


def load_rawbox(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "s": np.asarray(payload["s"], dtype="f8"),
            "phases": np.asarray(payload["phases"]).astype(str),
            "xi": np.asarray(payload["xi0_rsd"], dtype="f8"),
        }


def load_fit(path: Path, *, rawbox: bool = False) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "k": np.asarray(payload["k"], dtype="f8"),
            "pk": np.asarray(payload["pk0_rsd_mean"], dtype="f8"),
            "pk_model": np.asarray(payload["pk0_model_map"], dtype="f8"),
            "pk_cov_mean": np.asarray(payload["pk0_covariance_mean"], dtype="f8"),
            "s": np.asarray(payload["s"], dtype="f8"),
            "xi": np.asarray(payload["xi0_rsd_mean"], dtype="f8"),
            "xi_model": np.asarray(payload["xi0_model_map"], dtype="f8"),
            "xi_cov_mean": np.asarray(payload["xi0_covariance_single"], dtype="f8") / 25.0,
        }


def roughness(values: np.ndarray, errors: np.ndarray) -> float:
    y = np.asarray(values, dtype="f8")
    sigma = np.asarray(errors, dtype="f8")
    second = y[2:] - 2.0 * y[1:-1] + y[:-2]
    second_sigma = np.sqrt(sigma[2:] ** 2 + 4.0 * sigma[1:-1] ** 2 + sigma[:-2] ** 2)
    return float(np.sqrt(np.mean((second / second_sigma) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-summary", type=Path, default=DEFAULT_WIDE_SUMMARY)
    parser.add_argument("--narrow-summary", type=Path, default=DEFAULT_NARROW_SUMMARY)
    parser.add_argument("--rawbox-closure", type=Path, default=DEFAULT_RAWBOX_CLOSURE)
    parser.add_argument("--wide-compare", type=Path, default=DEFAULT_WIDE_COMPARE)
    parser.add_argument("--narrow-compare", type=Path, default=DEFAULT_NARROW_COMPARE)
    parser.add_argument("--rawbox-compare", type=Path, default=DEFAULT_RAWBOX_COMPARE)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args()
    output_npz, output_json = args.output_prefix.with_suffix(".npz"), args.output_prefix.with_suffix(".json")
    for path in (args.pdf, output_npz, output_json):
        if path.exists():
            raise FileExistsError(f"immutable comparison output exists: {path}")

    xi = {
        "wide": load_xi_summary(args.wide_summary, expected_ells=(0,)),
        "narrow": load_xi_summary(args.narrow_summary, expected_ells=(0, 2)),
        "rawbox": load_rawbox(args.rawbox_closure),
    }
    s = xi["wide"]["s"]
    phases = xi["wide"]["phases"]
    for name, values in xi.items():
        if not np.array_equal(values["s"], s) or not np.array_equal(values["phases"], phases):
            raise RuntimeError(f"{name}: separation or phase grid differs")
        if values["xi"].shape != (25, s.size):
            raise RuntimeError(f"{name}: xi stack shape {values['xi'].shape}")
    fits = {
        "wide": load_fit(args.wide_compare),
        "narrow": load_fit(args.narrow_compare),
        "rawbox": load_fit(args.rawbox_compare, rawbox=True),
    }
    means = {name: np.mean(values["xi"], axis=0) for name, values in xi.items()}
    scatter = {name: np.std(values["xi"], axis=0, ddof=1) for name, values in xi.items()}
    sem = {name: values / np.sqrt(25.0) for name, values in scatter.items()}
    nominal = s >= 50.0

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    colors = {"wide": "#C44E52", "narrow": "#4C72B0", "rawbox": "#2F2F2F"}
    labels = {
        "wide": r"wide lightcone $0.4<z_{\rm obs}<1.1$",
        "narrow": r"narrow lightcone $0.6<z_{\rm obs}<0.8$",
        "rawbox": r"rawbox $z=0.725$",
    }
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.pdf.with_name(f".{args.pdf.name}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary) as pdf:
        figure, axes = plt.subplots(2, 1, figsize=(10.0, 7.3), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})
        scale = s**2
        for name in ("rawbox", "narrow", "wide"):
            axes[0].fill_between(s, scale * (means[name] - sem[name]), scale * (means[name] + sem[name]), color=colors[name], alpha=0.10, linewidth=0)
            axes[0].plot(s, scale * means[name], "o-", ms=3.1, lw=1.25, color=colors[name], label=labels[name])
        axes[0].axhline(0.0, color="0.55", lw=0.8)
        axes[0].axvline(50.0, color="0.55", lw=0.8, ls=":")
        axes[0].set_ylabel(r"$s^2\bar\xi_0(s)\ [(h^{-1}{\rm Mpc})^2]$")
        axes[0].legend(frameon=False, ncol=3, fontsize=8.5)
        axes[0].set_title("Measured RSD monopoles, paired Abacus phases (no theory curves)")
        axes[1].axhline(1.0, color="0.5", lw=0.8)
        axes[1].plot(s, scatter["wide"] / scatter["narrow"], "o-", ms=3.1, lw=1.2, color=colors["wide"], label="wide / narrow")
        axes[1].plot(s, scatter["wide"] / scatter["rawbox"], "o-", ms=3.1, lw=1.2, color=colors["rawbox"], label="wide / rawbox")
        axes[1].axvline(50.0, color="0.55", lw=0.8, ls=":")
        axes[1].set(xlabel=r"$s\ [h^{-1}{\rm Mpc}]$", ylabel="single-phase scatter ratio")
        axes[1].legend(frameon=False)
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(2, 1, figsize=(10.0, 7.3), sharex=False)
        for name in ("rawbox", "narrow", "wide"):
            fit = fits[name]
            pk_sigma = np.sqrt(np.diag(fit["pk_cov_mean"]))
            axes[0].errorbar(
                fit["k"],
                (fit["pk"] - fit["pk_model"]) / pk_sigma,
                yerr=np.ones_like(pk_sigma),
                fmt="o-",
                ms=3.3,
                lw=1.0,
                color=colors[name],
                label=labels[name],
            )
            xi_sigma = np.sqrt(np.diag(fit["xi_cov_mean"]))
            axes[1].plot(fit["s"], (fit["xi"] - fit["xi_model"]) / xi_sigma, "o-", ms=3.3, lw=1.0, color=colors[name], label=labels[name])
        for axis in axes:
            axis.axhline(0.0, color="0.5", lw=0.8)
            axis.axhline(2.0, color="0.7", lw=0.7, ls="--")
            axis.axhline(-2.0, color="0.7", lw=0.7, ls="--")
        axes[0].set(xscale="log", xlabel=r"$k\ [h\,{\rm Mpc}^{-1}]$", ylabel=r"$(P_0-P_0^{\rm model})/\sigma_{\rm mean}$")
        axes[0].legend(frameon=False, ncol=3, fontsize=8.5)
        axes[1].set(xlabel=r"$s\ [h^{-1}{\rm Mpc}]$", ylabel=r"$(\xi_0-\xi_0^{\rm model})/\sigma_{\rm mean}$")
        figure.suptitle("Monopole fit residuals with each geometry's own theory/window/covariance")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)
    temporary.replace(args.pdf)

    wide_narrow_paired = xi["wide"]["xi"] - xi["narrow"]["xi"]
    delta = np.mean(wide_narrow_paired, axis=0)
    delta_sem = np.std(wide_narrow_paired, axis=0, ddof=1) / np.sqrt(25.0)
    delta_z = delta / delta_sem
    atomic_savez(
        output_npz,
        s=s,
        phases=phases,
        xi0_wide_by_phase=xi["wide"]["xi"],
        xi0_narrow_by_phase=xi["narrow"]["xi"],
        xi0_rawbox_by_phase=xi["rawbox"]["xi"],
        xi0_wide_mean=means["wide"],
        xi0_narrow_mean=means["narrow"],
        xi0_rawbox_mean=means["rawbox"],
        xi0_wide_scatter=scatter["wide"],
        xi0_narrow_scatter=scatter["narrow"],
        xi0_rawbox_scatter=scatter["rawbox"],
        wide_minus_narrow_mean=delta,
        wide_minus_narrow_sem=delta_sem,
        wide_minus_narrow_z=delta_z,
    )
    inputs = {
        "wide_summary": args.wide_summary,
        "narrow_summary": args.narrow_summary,
        "rawbox_closure": args.rawbox_closure,
        "wide_compare": args.wide_compare,
        "narrow_compare": args.narrow_compare,
        "rawbox_compare": args.rawbox_compare,
    }
    payload: dict[str, Any] = {
        "task": "task43_plot_rsd_wide_narrow_rawbox_monopoles",
        "status": "pass",
        "scope": "observed monopoles only; wide/narrow lightcone and rawbox x25",
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in inputs.items()},
        "diagnostics_sge50": {
            "wide_over_narrow_scatter_median": float(np.median((scatter["wide"] / scatter["narrow"])[nominal])),
            "wide_over_rawbox_scatter_median": float(np.median((scatter["wide"] / scatter["rawbox"])[nominal])),
            "wide_minus_narrow_paired_z_rms": float(np.sqrt(np.mean(delta_z[nominal] ** 2))),
            "wide_minus_narrow_paired_z_absmax": float(np.max(np.abs(delta_z[nominal]))),
            "s2xi0_mean_roughness_over_sem": {
                name: roughness((s**2 * means[name])[nominal], (s**2 * sem[name])[nominal])
                for name in ("wide", "narrow", "rawbox")
            },
        },
        "output_pdf": str(args.pdf),
        "output_pdf_sha256": sha256_file(args.pdf),
        "output_npz": str(output_npz),
        "output_npz_sha256": sha256_file(output_npz),
    }
    atomic_write_json(output_json, payload)
    print(json.dumps(payload["diagnostics_sge50"] | {"output_pdf": str(args.pdf)}, sort_keys=True))


if __name__ == "__main__":
    main()
