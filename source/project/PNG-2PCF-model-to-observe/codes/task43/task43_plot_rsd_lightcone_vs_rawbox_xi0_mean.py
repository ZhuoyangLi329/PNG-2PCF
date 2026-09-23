#!/usr/bin/env python3
"""Plot the measured x25 RSD xi0 means for lightcone and rawbox.

No theory curve is included.  Uncertainties are empirical standard errors of
the x25 phase means; the difference panel uses phase-paired errors.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from task43_rsd_common import atomic_savez, atomic_write_json, sha256_file


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
DEFAULT_LIGHTCONE = (
    PROJECT_ROOT
    / "outputs/task43_outputs/rsd_validation/lightcone/closure/"
    "task43_rsd_lightcone_x25_jaxpower_rrdeconv_rrnran300k_fulldiscrete_lorentzian.npz"
)
DEFAULT_RAWBOX = (
    PROJECT_ROOT
    / "outputs/task43_outputs/rsd_validation/rawbox/closure/"
    "task43_rsd_rawbox_x25_fulldiscrete_lorentzian.npz"
)
DEFAULT_PDF = (
    PROJECT_ROOT
    / "plots/task43/rsd_validation/"
    "task43_rsd_x25_lightcone_vs_rawbox_xi0_measured_mean.pdf"
)
DEFAULT_OUTPUT_PREFIX = (
    PROJECT_ROOT
    / "outputs/task43_outputs/rsd_validation/audits/"
    "task43_rsd_x25_lightcone_vs_rawbox_xi0_measured_mean"
)


def load_inputs(lightcone: Path, rawbox: Path) -> dict[str, np.ndarray]:
    with np.load(lightcone, allow_pickle=False) as payload:
        s_lc = np.asarray(payload["s"], dtype="f8")
        phases_lc = np.asarray(payload["phases"]).astype(str)
        xi_lc = np.asarray(payload["xi_multipoles_by_phase"], dtype="f8")[:, 0, :]
        stored_lc_mean = np.asarray(payload["xi_multipoles_mean"], dtype="f8")[0]
        ells = np.asarray(payload["ells"], dtype="i8")
    with np.load(rawbox, allow_pickle=False) as payload:
        s_rb = np.asarray(payload["s"], dtype="f8")
        phases_rb = np.asarray(payload["phases"]).astype(str)
        xi_rb = np.asarray(payload["xi0_rsd"], dtype="f8")
        stored_rb_mean = np.asarray(payload["xi0_rsd_mean"], dtype="f8")
    if not np.array_equal(ells, [0, 2]):
        raise RuntimeError(f"unexpected lightcone multipoles {ells}")
    if xi_lc.shape != (25, 32) or xi_rb.shape != (25, 32):
        raise RuntimeError(f"unexpected xi0 stacks lightcone={xi_lc.shape}, rawbox={xi_rb.shape}")
    if not np.array_equal(phases_lc, phases_rb):
        raise RuntimeError("lightcone/rawbox phase labels or ordering differ")
    if not np.array_equal(s_lc, s_rb):
        raise RuntimeError("lightcone/rawbox separation coordinates differ")
    if not np.allclose(np.mean(xi_lc, axis=0), stored_lc_mean, rtol=0.0, atol=1.0e-15):
        raise RuntimeError("stored lightcone mean does not equal the x25 mean")
    if not np.allclose(np.mean(xi_rb, axis=0), stored_rb_mean, rtol=0.0, atol=1.0e-15):
        raise RuntimeError("stored rawbox mean does not equal the x25 mean")
    return {
        "s": s_lc,
        "phases": phases_lc,
        "xi_lightcone": xi_lc,
        "xi_rawbox": xi_rb,
    }


def make_plot(path: Path, arrays: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = arrays["s"]
    lc = arrays["xi_lightcone"]
    rb = arrays["xi_rawbox"]
    lc_mean, rb_mean = np.mean(lc, axis=0), np.mean(rb, axis=0)
    lc_sem = np.std(lc, axis=0, ddof=1) / np.sqrt(lc.shape[0])
    rb_sem = np.std(rb, axis=0, ddof=1) / np.sqrt(rb.shape[0])
    paired = lc - rb
    delta = np.mean(paired, axis=0)
    delta_sem = np.std(paired, axis=0, ddof=1) / np.sqrt(paired.shape[0])
    paired_z = np.divide(delta, delta_sem, out=np.full_like(delta, np.nan), where=delta_sem > 0.0)
    scale = s**2
    colors = {"lightcone": "#C44E52", "rawbox": "#2F2F2F", "difference": "#4C72B0"}

    figure = plt.figure(figsize=(11.4, 7.7))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.55, 1.0), hspace=0.30, wspace=0.27)
    top = figure.add_subplot(grid[0, :])
    difference = figure.add_subplot(grid[1, 0])
    significance = figure.add_subplot(grid[1, 1])

    top.fill_between(s, scale * (rb_mean - rb_sem), scale * (rb_mean + rb_sem), color=colors["rawbox"], alpha=0.13, linewidth=0)
    top.fill_between(s, scale * (lc_mean - lc_sem), scale * (lc_mean + lc_sem), color=colors["lightcone"], alpha=0.16, linewidth=0)
    top.plot(s, scale * rb_mean, "o-", color=colors["rawbox"], ms=3.6, lw=1.45, label="rawbox x25 mean")
    top.plot(s, scale * lc_mean, "o-", color=colors["lightcone"], ms=3.6, lw=1.45, label="lightcone x25 mean")
    top.axhline(0.0, color="0.55", lw=0.8)
    top.axvline(50.0, color="0.55", lw=0.8, ls=":")
    top.set(xlim=(30.0, 350.0), ylabel=r"$s^2\,\bar\xi_0(s)\ [(h^{-1}{\rm Mpc})^2]$")
    top.legend(frameon=False, ncol=2, loc="upper right")
    top.set_title(r"Measured RSD monopole only: paired Abacus phases ph000--ph024 (no theory)")

    difference.fill_between(
        s,
        scale * (delta - delta_sem),
        scale * (delta + delta_sem),
        color=colors["difference"],
        alpha=0.18,
        linewidth=0,
    )
    difference.plot(s, scale * delta, "o-", color=colors["difference"], ms=3.4, lw=1.25)
    difference.axhline(0.0, color="0.45", lw=0.8)
    difference.axvline(50.0, color="0.55", lw=0.8, ls=":")
    difference.set(
        xlim=(30.0, 350.0),
        xlabel=r"$s\ [h^{-1}{\rm Mpc}]$",
        ylabel=r"$s^2(\bar\xi_0^{\rm LC}-\bar\xi_0^{\rm box})$",
    )
    difference.set_title("paired mean difference")

    significance.axhspan(-1.0, 1.0, color="0.7", alpha=0.17, linewidth=0)
    significance.axhline(0.0, color="0.45", lw=0.8)
    significance.axhline(2.0, color="0.6", lw=0.7, ls="--")
    significance.axhline(-2.0, color="0.6", lw=0.7, ls="--")
    significance.axvline(50.0, color="0.55", lw=0.8, ls=":")
    significance.plot(s, paired_z, "o-", color=colors["difference"], ms=3.4, lw=1.25)
    significance.set(
        xlim=(30.0, 350.0),
        xlabel=r"$s\ [h^{-1}{\rm Mpc}]$",
        ylabel=r"$(\bar\xi_0^{\rm LC}-\bar\xi_0^{\rm box})/{\rm SEM}_{\rm paired}$",
    )
    significance.set_title("paired-phase significance")
    figure.text(
        0.5,
        0.012,
        "Bands are empirical standard errors of the x25 means; the lower panels use phase-paired differences.",
        ha="center",
        fontsize=9.5,
        color="0.35",
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.10)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.pdf")
    figure.savefig(temporary, format="pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lightcone", type=Path, default=DEFAULT_LIGHTCONE)
    parser.add_argument("--rawbox", type=Path, default=DEFAULT_RAWBOX)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args()
    output_npz, output_json = args.output_prefix.with_suffix(".npz"), args.output_prefix.with_suffix(".json")
    for path in (args.pdf, output_npz, output_json):
        if path.exists():
            raise FileExistsError(f"immutable comparison output exists: {path}")
    arrays = load_inputs(args.lightcone, args.rawbox)
    s = arrays["s"]
    lc, rb = arrays["xi_lightcone"], arrays["xi_rawbox"]
    lc_mean, rb_mean = np.mean(lc, axis=0), np.mean(rb, axis=0)
    lc_sem = np.std(lc, axis=0, ddof=1) / np.sqrt(lc.shape[0])
    rb_sem = np.std(rb, axis=0, ddof=1) / np.sqrt(rb.shape[0])
    paired = lc - rb
    delta, delta_sem = np.mean(paired, axis=0), np.std(paired, axis=0, ddof=1) / np.sqrt(paired.shape[0])
    paired_z = np.divide(delta, delta_sem, out=np.full_like(delta, np.nan), where=delta_sem > 0.0)
    make_plot(args.pdf, arrays)
    atomic_savez(
        output_npz,
        s=s,
        phases=arrays["phases"],
        xi0_lightcone_by_phase=lc,
        xi0_rawbox_by_phase=rb,
        xi0_lightcone_mean=lc_mean,
        xi0_rawbox_mean=rb_mean,
        xi0_lightcone_sem=lc_sem,
        xi0_rawbox_sem=rb_sem,
        xi0_paired_difference_mean=delta,
        xi0_paired_difference_sem=delta_sem,
        xi0_paired_difference_z=paired_z,
    )
    nominal = s >= 50.0
    tail = s >= 125.0
    payload = {
        "task": "task43_plot_rsd_lightcone_vs_rawbox_xi0_mean",
        "status": "pass",
        "scope": "measured RSD xi0 only, x25 paired phase means, no theory",
        "lightcone_input": str(args.lightcone),
        "lightcone_input_sha256": sha256_file(args.lightcone),
        "rawbox_input": str(args.rawbox),
        "rawbox_input_sha256": sha256_file(args.rawbox),
        "nphase": int(lc.shape[0]),
        "separation_bins": {"centers": s.tolist(), "smin_nominal": 50.0, "smax": 350.0},
        "uncertainty": {
            "curves": "empirical x25 phase scatter / sqrt(25)",
            "difference": "empirical scatter of phase-paired (lightcone - rawbox) / sqrt(25)",
        },
        "diagnostics": {
            "all_bins_paired_z_rms": float(np.sqrt(np.mean(paired_z**2))),
            "all_bins_paired_z_absmax": float(np.max(np.abs(paired_z))),
            "all_bins_paired_z_absmax_s": float(s[np.argmax(np.abs(paired_z))]),
            "sge50_paired_z_rms": float(np.sqrt(np.mean(paired_z[nominal] ** 2))),
            "sge50_paired_z_absmax": float(np.max(np.abs(paired_z[nominal]))),
            "sge50_paired_z_absmax_s": float(s[nominal][np.argmax(np.abs(paired_z[nominal]))]),
            "sge125_paired_z_rms": float(np.sqrt(np.mean(paired_z[tail] ** 2))),
            "sge125_paired_z_absmax": float(np.max(np.abs(paired_z[tail]))),
            "sge125_paired_z_absmax_s": float(s[tail][np.argmax(np.abs(paired_z[tail]))]),
        },
        "output_pdf": str(args.pdf),
        "output_pdf_sha256": sha256_file(args.pdf),
        "output_npz": str(output_npz),
        "output_npz_sha256": sha256_file(output_npz),
    }
    atomic_write_json(output_json, payload)
    print(json.dumps(payload["diagnostics"] | {"output_pdf": str(args.pdf)}, sort_keys=True))


if __name__ == "__main__":
    main()
