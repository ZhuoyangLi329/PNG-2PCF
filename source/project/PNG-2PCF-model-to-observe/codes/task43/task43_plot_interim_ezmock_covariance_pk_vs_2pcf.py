#!/usr/bin/env python3
"""Plot an interim EZmock-covariance P(k) versus 2PCF constraint comparison.

The raw MCMC samples already include the Hartlap correction in each
likelihood precision.  For the displayed contours only, an affine transform
about each posterior median multiplies parameter deviations by the Percival
error factor; this represents ``Cov(theta) -> m1 Cov(theta)`` without moving
the posterior center.  Raw and display samples are both preserved in the
machine audit.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes/task43"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import task43_overlay_pk_window_kmin_contour as base  # noqa: E402
import task43_plot_ric_covariance_pair_longchain as pair  # noqa: E402
import task43_plot_ric_rascalc_pk_vs_2pcf as single  # noqa: E402
from task43_ezmock_covariance_common import atomic_savez, sha256, write_json  # noqa: E402


DEFAULT_NMOCK = 61
DEFAULT_ROOT = PROJECT_ROOT / "outputs/task43_outputs/ezmock_covariance_interim_x61_hartlap_percival_20260721"
OLD_XI_DIR = (
    PROJECT_ROOT
    / "outputs/task43_outputs/ric_singleterm/fits/2pcf_jaxpower_ph000_dchi2_nsub200000_long_mcmc20k"
)
OLD_PK_DIR = (
    PROJECT_ROOT
    / "outputs/task43_outputs/ric_singleterm/fits/pk/"
    "task43_pk_ric_ph000_dchi2_nsub200000_motherbox_long_mcmc50k/"
    "task43_pk_ric_ph000_dchi2_nsub200000_motherbox_long_mcmc50k"
)
OLD_XI_SUMMARY = OLD_XI_DIR / "task43_minimal_closure_mcmc_summary.json"
OLD_XI_SAMPLES = OLD_XI_DIR / "task43_mcmc_radial_singleterm_samples.npz"
OLD_PK_SUMMARY = OLD_PK_DIR / "task43_pk_lightcone_task43_pk_ric_ph000_dchi2_nsub200000_motherbox_long_mcmc50k_fit_summary.json"
OLD_PK_SAMPLES = OLD_PK_DIR / "task43_pk_lightcone_task43_pk_ric_ph000_dchi2_nsub200000_motherbox_long_mcmc50k_fit_samples.npz"
DEFAULT_PLOT = (
    PROJECT_ROOT
    / "plots/task43/ezmock_covariance_interim_x61_hartlap_percival_20260721/"
    "task43_ezmock_x61_hartlap_percival_pk15_vs_2pcf_s50_350_fnl_b1.pdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmock", type=int, default=DEFAULT_NMOCK)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument(
        "--alias-plot",
        type=Path,
        default=None,
        help="Optional second PDF path updated atomically with identical bytes.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_samples(path: Path, names: tuple[str, ...]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        stored_names = [str(item) for item in data["param_names"]] if "param_names" in data.files else list(names)
        samples = np.asarray(data["samples"], dtype="f8")
    return samples[:, [stored_names.index(name) for name in names]]


def percival_transform(samples: np.ndarray, factor: float) -> np.ndarray:
    values = np.asarray(samples, dtype="f8")
    center = np.median(values, axis=0)
    return center + float(factor) * (values - center)


def summary(values: np.ndarray) -> dict[str, float]:
    return single.summarize(np.asarray(values, dtype="f8"))


def compare_constraint(new_values: np.ndarray, old_values: np.ndarray, factor: float) -> dict[str, Any]:
    raw = summary(new_values)
    corrected = summary(percival_transform(new_values[:, None], factor)[:, 0])
    old = summary(old_values)
    old_sigma = 0.5 * (old["err_low"] + old["err_high"])
    new_sigma = 0.5 * (corrected["err_low"] + corrected["err_high"])
    return {
        "old_theory_covariance": old,
        "new_ezmock_raw_hartlap_posterior": raw,
        "new_ezmock_hartlap_plus_percival_reported": corrected,
        "percival_error_factor": float(factor),
        "median_shift_new_minus_old": float(corrected["q50"] - old["q50"]),
        "median_shift_over_old_sigma": float((corrected["q50"] - old["q50"]) / old_sigma),
        "sigma68_ratio_new_corrected_over_old": float(new_sigma / old_sigma),
    }


def main() -> None:
    args = parse_args()
    nmock = int(args.nmock)
    root = Path(args.root)
    freeze_path = root / f"audit/task43_ezmock_interim_x{nmock}_freeze_audit.json"
    xi_dir = root / "fits/2pcf_xi30_radial_ric_long"
    xi_summary_path = xi_dir / "task43_minimal_closure_mcmc_summary.json"
    xi_samples_path = xi_dir / "task43_mcmc_radial_singleterm_samples.npz"
    pk_dir = root / "fits/pk15_radial_ric_long"
    pk_summary_path = pk_dir / "task43_pk_lightcone_pk15_radial_ric_long_fit_summary.json"
    pk_samples_path = pk_dir / "task43_pk_lightcone_pk15_radial_ric_long_fit_samples.npz"
    plot_path = Path(args.plot)
    alias_plot_path = None if args.alias_plot is None else Path(args.alias_plot)
    display_samples_path = root / f"postprocess/task43_ezmock_x{nmock}_percival_display_samples.npz"
    audit_path = root / f"audit/task43_ezmock_x{nmock}_hartlap_percival_pk_vs_2pcf.json"
    required = (
        freeze_path,
        xi_summary_path,
        xi_samples_path,
        pk_summary_path,
        pk_samples_path,
        OLD_XI_SUMMARY,
        OLD_XI_SAMPLES,
        OLD_PK_SUMMARY,
        OLD_PK_SAMPLES,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    freeze = read_json(freeze_path)
    xi_summary = read_json(xi_summary_path)
    pk_summary = read_json(pk_summary_path)
    old_xi_summary = read_json(OLD_XI_SUMMARY)
    old_pk_summary = read_json(OLD_PK_SUMMARY)
    if int(freeze["frozen_nmock"]) != nmock or freeze["fix_amplitude"] is not False:
        raise RuntimeError(f"freeze audit is not the x{nmock} FIX_AMPLITUDE=F snapshot")
    if xi_summary["fit_range"]["nbins"] != 30 or (
        xi_summary["fit_range"]["rmin"], xi_summary["fit_range"]["rmax"]
    ) != (50.0, 350.0):
        raise RuntimeError("2PCF fit range is not 30 bins over s=50--350")
    if pk_summary["data"]["ndata"] != 15 or not np.isclose(pk_summary["data"]["kmax_fit"], 0.1):
        raise RuntimeError("P(k) fit range is not the authoritative 15-bin kmax=0.10 vector")
    xi_corr = xi_summary["covariance"]
    pk_corr = pk_summary["covariance"]
    if not (xi_corr["finite_mock_correction"] and pk_corr["finite_mock_correction"]):
        raise RuntimeError("finite-mock corrections are not active in both likelihoods")
    if (xi_corr["nmock"], xi_corr["ndata"], xi_corr["nparams"]) != (nmock, 30, 2):
        raise RuntimeError("2PCF correction dimensions mismatch")
    if (pk_corr["nmock"], pk_corr["ndata"], pk_corr["nparams"]) != (nmock, 15, 3):
        raise RuntimeError("P(k) correction dimensions mismatch")
    xi_model = single.radial_model(xi_summary)
    old_xi_model = single.radial_model(old_xi_summary)
    operator_names = {
        Path(xi_model["radial_singleterm"]["operator"]["path"]).name,
        Path(pk_summary["config"]["radial_singleterm_ric"]["path"]).name,
        Path(old_xi_model["radial_singleterm"]["operator"]["path"]).name,
        Path(old_pk_summary["config"]["radial_singleterm_ric"]["path"]).name,
    }
    if operator_names != {single.BASE_OPERATOR_NAME}:
        raise RuntimeError(f"radial RIC operators do not match: {operator_names}")

    xi_raw = load_samples(xi_samples_path, ("fnl_loc", "b1"))
    pk_raw_full = load_samples(pk_samples_path, ("fnl_loc", "b1", "sn0"))
    old_xi = load_samples(OLD_XI_SAMPLES, ("fnl_loc", "b1"))
    old_pk_full = load_samples(OLD_PK_SAMPLES, ("fnl_loc", "b1", "sn0"))
    diagnostics = {
        "2pcf": single.chain_diagnostics(xi_raw, nwalkers=int(xi_model["mcmc"]["nwalkers"])),
        "pk": single.chain_diagnostics(pk_raw_full, nwalkers=int(pk_summary["config"]["nwalkers"])),
    }
    if not all(item["pass"] for item in diagnostics.values()):
        raise RuntimeError(f"long-chain convergence gate failed: {diagnostics}")

    xi_factor = float(xi_corr["percival_error_factor"])
    pk_factor = float(pk_corr["percival_error_factor"])
    xi_display = percival_transform(xi_raw, xi_factor)
    pk_display_full = percival_transform(pk_raw_full, pk_factor)
    pk_display = pk_display_full[:, :2]
    atomic_savez(
        display_samples_path,
        xi_param_names=np.asarray(["fnl_loc", "b1"]),
        xi_samples_raw=xi_raw,
        xi_samples_percival_display=xi_display,
        xi_percival_error_factor=np.asarray(xi_factor),
        pk_param_names=np.asarray(["fnl_loc", "b1", "sn0"]),
        pk_samples_raw=pk_raw_full,
        pk_samples_percival_display=pk_display_full,
        pk_percival_error_factor=np.asarray(pk_factor),
    )

    xlim = base.shared_axis_limits([xi_display, pk_display], 0, include_zero=True)
    ylim = base.shared_axis_limits([xi_display, pk_display], 1)
    xi_fnl = summary(xi_display[:, 0])
    pk_fnl = summary(pk_display[:, 0])
    base.apply_publication_style()
    figure, axes, legend_anchor = pair.plot_ppt_corner(
        xi_samples=xi_display,
        pk_samples=pk_display,
        xi_fnl=xi_fnl,
        pk_fnl=pk_fnl,
        pk_summary=pk_summary,
        xlim=xlim,
        ylim=ylim,
        xi_smin=50.0,
        xi_smax=350.0,
    )
    axes[0, 0].text(
        0.965,
        0.955,
        rf"$N_{{\rm mock}}={nmock}$",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        fontsize=12.5,
        color="0.25",
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = plot_path.with_name(f".{plot_path.name}.tmp.pdf")
    try:
        figure.savefig(temporary, format="pdf", bbox_inches="tight", pad_inches=0.08)
        temporary.replace(plot_path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    if alias_plot_path is not None:
        alias_plot_path.parent.mkdir(parents=True, exist_ok=True)
        alias_temporary = alias_plot_path.with_name(f".{alias_plot_path.name}.tmp.pdf")
        try:
            shutil.copyfile(plot_path, alias_temporary)
            alias_temporary.replace(alias_plot_path)
        finally:
            alias_temporary.unlink(missing_ok=True)

    comparison = {
        "2pcf_fnl": compare_constraint(xi_raw[:, 0], old_xi[:, 0], xi_factor),
        "pk_fnl": compare_constraint(pk_raw_full[:, 0], old_pk_full[:, 0], pk_factor),
    }
    audit = {
        "task": "task43_plot_interim_ezmock_covariance_pk_vs_2pcf",
        "status": "pass_interim_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": (
            f"interim x{nmock} result; mathematically valid finite-mock corrections, but especially "
            "the 30-bin xi covariance remains statistically noisy"
        ),
        "frozen_nmock": nmock,
        "data_dimensions": {"2pcf": 30, "pk": 15},
        "fit_parameters": {"2pcf": ["fnl_loc", "b1"], "pk": ["fnl_loc", "b1", "sn0"]},
        "corrections": {"2pcf": xi_corr, "pk": pk_corr},
        "correction_layering": {
            "hartlap": "applied to inverse sample covariance inside each MCMC likelihood",
            "percival": (
                "not applied to data covariance or likelihood; raw posterior errors are multiplied "
                "by sqrt(m1), and only display samples are affinely scaled about their medians"
            ),
        },
        "chain_diagnostics": diagnostics,
        "posterior_display_hartlap_plus_percival": {
            "2pcf": {"fnl_loc": xi_fnl, "b1": summary(xi_display[:, 1])},
            "pk": {
                "fnl_loc": pk_fnl,
                "b1": summary(pk_display_full[:, 1]),
                "sn0": summary(pk_display_full[:, 2]),
            },
        },
        "comparison_to_old_theory_covariance": comparison,
        "inputs": {str(path): sha256(path) for path in required},
        "plot_contract": {
            "format": "pdf_only",
            "percival_corrected_contours": True,
            "shared_axis_ranges": {"fnl_loc": xlim, "b1": ylim},
            "legend_anchor_figure_fraction": list(legend_anchor),
            "same_radial_operator": single.BASE_OPERATOR_NAME,
            "nmock_annotation": nmock,
        },
        "outputs": {
            "pdf": str(plot_path),
            "pdf_alias": None if alias_plot_path is None else str(alias_plot_path),
            "display_samples": str(display_samples_path),
            "audit": str(audit_path),
        },
    }
    write_json(audit_path, audit)
    print(f"[pass] {plot_path}")
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
