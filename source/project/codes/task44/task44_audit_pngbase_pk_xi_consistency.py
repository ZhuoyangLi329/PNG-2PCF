#!/usr/bin/env python3
"""Audit the corrected periodic-box P0--xi0 likelihood contract.

The authoritative P0 selection is the native 49-bin periodic grid.  The old
16-bin lightcone-matched selection is recorded only as superseded provenance.
The xi0 mean is the standard full Hankel transform implemented by the model;
this audit deliberately performs no artificial Fourier-band split or
comparison of arbitrary numerical integration cutoffs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
TASK44_CODE_DIR = PROJECT_ROOT / "codes" / "task44"
if str(TASK44_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(TASK44_CODE_DIR))

from task44_fit_pngbase_pk_xi_consistency import DIAGNOSTIC_ROOT, variant_prefix  # noqa: E402
from task44_plot_pngbase_hodmap_rawbox import load_fit, posterior_row  # noqa: E402
from task44_pngbase_hodmap_rawbox_common import (  # noqa: E402
    BOX_SIZE,
    BOX_VOLUME,
    K_FUND,
    OUTPUT_ROOT,
    PK_LEGACY_LIGHTCONE_MATCHED_EDGES,
    PK_PRIMARY_FIT_EDGES,
    PLOT_ROOT,
    atomic_savez,
    atomic_write_json,
    fit_prefix,
    get_spec,
    legacy_lightcone_matched_fit_prefix,
    set_cpu_affinity,
    sha256_file,
)


SUMMARY_DIR = DIAGNOSTIC_ROOT / "summary"
OUTPUT_JSON = SUMMARY_DIR / "task44_pngbase_pk_xi_consistency_final_audit.json"
OUTPUT_NPZ = SUMMARY_DIR / "task44_pngbase_pk_xi_consistency_final_audit.npz"
OUTPUT_PDF = PLOT_ROOT / "consistency_diagnostics/task44_pngbase_pk_xi_consistency_final_audit.pdf"
ENGINE_AUDIT = OUTPUT_ROOT / "summary/task44_pngbase_hodmap_xi0_three_engine_s30_350_audit.json"

COLORS = {"pk": "#2F2F2F", "xi30": "#DD8452"}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def load_variant(tag: str, kind: str, *, smin: float) -> dict[str, Any]:
    prefix = variant_prefix(tag, kind, smin=smin)
    json_path, npz_path = prefix.with_suffix(".json"), prefix.with_suffix(".npz")
    summary = read_json(json_path)
    if summary.get("output_npz_sha256") != sha256_file(npz_path):
        raise RuntimeError(f"fit hash mismatch: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as handle:
        arrays = {key: np.asarray(handle[key]) for key in handle.files if key != "summary_json"}
    return {"summary": summary, "arrays": arrays, "json_path": json_path, "npz_path": npz_path}


def fit_record(fit: dict[str, Any], parameter: str) -> dict[str, Any]:
    return {
        "posterior": posterior_row(fit, parameter),
        "map": fit["summary"]["map"],
        "status": fit["summary"]["status"],
        "mcmc_gates": fit["summary"]["mcmc"]["gates"],
        "json": str(fit["json_path"]),
        "json_sha256": sha256_file(fit["json_path"]),
        "npz": str(fit["npz_path"]),
        "npz_sha256": sha256_file(fit["npz_path"]),
    }


def engine_max_abs_delta(engine: dict[str, Any]) -> float:
    return float(
        max(
            record["max_abs_xi0"]
            for catalog in engine["catalogs"].values()
            for record in catalog["engines"].values()
        )
    )


def build_catalog(tag: str) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    pk = load_fit(tag, "pk", kmin=0.003)
    pk_fixedp1 = load_fit(tag, "pk", kmin=0.003, fixed_p=1.0)
    xi_free = load_variant(tag, "xi_stochastic_cov", smin=30.0)
    xi_fixedp1 = load_variant(tag, "xi_fixedp1_stochastic_cov", smin=30.0)

    primary_json = fit_prefix(tag, "pk", kmin_edge=0.003).with_suffix(".json")
    edges = np.asarray(pk["arrays"]["coordinate_edges"], dtype="f8")
    source_gates: dict[str, bool] = {}
    for label, fit in (
        ("xi_freep", xi_free),
        ("xi_fixedp1", xi_fixedp1),
    ):
        source = fit["summary"]["residual_stochastic"]
        source_gates[f"{label}_uses_periodic_primary_json"] = Path(
            source["source_primary_pk_json"]
        ) == primary_json
        source_gates[f"{label}_source_hash_matches"] = (
            source["source_primary_pk_json_sha256"] == sha256_file(primary_json)
        )

    gates = {
        "primary_pk_status_pass": pk["summary"].get("status") == "pass",
        "fixedp1_pk_status_pass": pk_fixedp1["summary"].get("status") == "pass",
        "primary_has_exactly_49_bins": edges.shape == (49, 2),
        "primary_edges_match_native_periodic_grid": bool(
            np.allclose(edges, PK_PRIMARY_FIT_EDGES, rtol=0.0, atol=1.0e-14)
        ),
        "primary_bins_are_contiguous": bool(
            np.allclose(edges[1:, 0], edges[:-1, 1], rtol=0.0, atol=1.0e-14)
        ),
        "primary_first_edge_is_0p003": bool(np.isclose(edges[0, 0], 0.003, rtol=0.0, atol=1.0e-14)),
        "primary_last_center_is_0p100": bool(
            np.isclose(np.mean(edges[-1]), 0.100, rtol=0.0, atol=1.0e-14)
        ),
        "all_corrected_xi_fits_pass": bool(
            all(
                fit["summary"].get("status") == "pass"
                and all(fit["summary"]["mcmc"]["gates"].values())
                for fit in (xi_free, xi_fixedp1)
            )
        ),
        **source_gates,
    }

    record = {
        "catalog_baseline_fnl": float(get_spec(tag).fnl),
        "primary_pk_contract": {
            "selection": "all contiguous native periodic-box bins with centres <= 0.100 h/Mpc",
            "nbins": int(edges.shape[0]),
            "first_bin_edges_h_mpc": edges[0].tolist(),
            "last_bin_edges_h_mpc": edges[-1].tolist(),
            "last_bin_center_h_mpc": float(np.mean(edges[-1])),
            "fit": fit_record(pk, "p"),
        },
        "fixedp1_pk": fit_record(pk_fixedp1, "fnl"),
        "xi_stochastic_covariance": fit_record(xi_free, "p"),
        "xi_fixedp1_stochastic_covariance": fit_record(xi_fixedp1, "fnl"),
        "xi_mean_contract": {
            "implementation": "standard full discrete-box Hankel transform of the configured theory template",
            "theory_cache_numerical_upper_bound_h_mpc": 5.0,
            "interpretation": "numerical integration bound only; not a fitted data kmax",
            "artificial_fourier_diagnostics_performed": False,
        },
        "gates": gates,
    }
    arrays = {
        f"{tag}_pk_edges": edges,
        f"{tag}_pk_k": np.asarray(pk["arrays"]["coordinate"], dtype="f8"),
        f"{tag}_pk_data": np.asarray(pk["arrays"]["data"], dtype="f8"),
        f"{tag}_pk_prediction": np.asarray(pk["arrays"]["prediction_map"], dtype="f8"),
        f"{tag}_pk_residual_sigma": np.asarray(pk["arrays"]["residual_sigma"], dtype="f8"),
    }
    plotting = {"pk": pk, "pk_fixedp1": pk_fixedp1, "xi_free": xi_free, "xi_fixedp1": xi_fixedp1}
    return record, arrays, plotting


def asymmetric_error(row: dict[str, float]) -> list[list[float]]:
    return [[row["q50"] - row["q16"]], [row["q84"] - row["q50"]]]


def draw_pdf(summary: dict[str, Any], plotting: dict[str, dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.pdf")
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10.5, "axes.linewidth": 0.9})
    with PdfPages(temporary) as pdf:
        figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex="col")
        for column, tag in enumerate(("c300", "c302")):
            fit = plotting[tag]["pk"]
            k = np.asarray(fit["arrays"]["coordinate"], dtype="f8")
            data = np.asarray(fit["arrays"]["data"], dtype="f8")
            prediction = np.asarray(fit["arrays"]["prediction_map"], dtype="f8")
            sigma = np.sqrt(np.diag(np.asarray(fit["arrays"]["covariance_final"], dtype="f8")))
            axes[0, column].errorbar(k, data, yerr=sigma, fmt=".", ms=4, lw=0.7, color="0.25", label="49 periodic bins")
            axes[0, column].plot(k, prediction, color="#4C72B0", lw=1.4, label="MAP")
            axes[0, column].set_title(f"{tag}: native periodic P0 selection")
            axes[0, column].set_ylabel(r"$P_0(k)$")
            axes[0, column].legend(frameon=False, fontsize=9)
            axes[1, column].axhline(0.0, color="0.55", lw=0.8)
            axes[1, column].plot(k, (data - prediction) / sigma, ".", ms=4, color="#4C72B0")
            axes[1, column].set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
            axes[1, column].set_ylabel(r"$(P_0-P_{\rm MAP})/\sigma$")
            axes[1, column].set_xlim(0.0, 0.105)
        figure.suptitle(
            r"Authoritative $P_0$: $k_{\min}=0.003$, $k_{\max}=0.100\ h\,{\rm Mpc}^{-1}$ (49 bins)",
            fontsize=12.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.45"},
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        pdf.savefig(figure)
        plt.close(figure)

        labels = [r"$P_0(k)$", r"$\xi_0(s)$"]
        y = np.arange(2)[::-1]
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)
        for axis, tag in zip(axes, ("c300", "c302"), strict=True):
            fits = plotting[tag]
            rows = [posterior_row(fits["pk"], "p"), posterior_row(fits["xi_free"], "p")]
            for yy, row, color in zip(y, rows, (COLORS["pk"], COLORS["xi30"]), strict=True):
                axis.errorbar(row["q50"], yy, xerr=asymmetric_error(row), fmt="o", color=color, capsize=3)
            axis.set_yticks(y, labels)
            axis.set_xlabel(r"$p$ at fixed catalog $f_{\rm NL}$")
            axis.set_title(f"{tag}, fixed fNL={get_spec(tag).fnl:g}")
            axis.grid(axis="x", alpha=0.2)
        figure.suptitle(
            r"Correct periodic $P_0$ bins and $\xi_0$ stochastic covariance" "\n"
            r"$P_0$: $k_{\min}=0.003$, $k_{\max}=0.100\ h\,{\rm Mpc}^{-1}$; "
            r"$\xi_0$: $s_{\min}=30$, $s_{\max}=350\ h^{-1}{\rm Mpc}$",
            fontsize=11.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.45"},
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)
        for axis, tag in zip(axes, ("c300", "c302"), strict=True):
            fits = plotting[tag]
            rows = [posterior_row(fits["pk_fixedp1"], "fnl"), posterior_row(fits["xi_fixedp1"], "fnl")]
            for yy, row, color in zip(y, rows, (COLORS["pk"], COLORS["xi30"]), strict=True):
                axis.errorbar(row["q50"], yy, xerr=asymmetric_error(row), fmt="o", color=color, capsize=3)
            axis.axvline(get_spec(tag).fnl, color="0.5", ls="--", lw=1.0)
            axis.set_yticks(y, labels)
            axis.set_xlabel(r"$f_{\rm NL}$ at fixed $p=1$")
            axis.set_title(f"{tag}, baseline fNL={get_spec(tag).fnl:g}")
            axis.grid(axis="x", alpha=0.2)
        figure.suptitle(
            "Fixed-$p=1$ constraints with the corrected data contract\n"
            r"$P_0$: $k_{\min}=0.003$, $k_{\max}=0.100\ h\,{\rm Mpc}^{-1}$; "
            r"$\xi_0$: $s_{\min}=30$, $s_{\max}=350\ h^{-1}{\rm Mpc}$",
            fontsize=11.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.45"},
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
        pdf.savefig(figure)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(10.5, 7.2))
        axis.axis("off")
        lines = [
            "Corrected Task44 likelihood contract",
            "",
            "1. P0 primary: kmin=0.003, kmax=0.100 h/Mpc; 49 contiguous native periodic-box bins.",
            "2. The old lightcone-matched 16-bin selection is superseded provenance, not a result.",
            f"3. FCFC, pycorr, and CuCount agree: max |delta xi0| = {summary['measurement_bridge']['three_engine_max_abs_delta_xi']:.3e}.",
            "4. xi0 primary: smin=30, smax=350 Mpc/h; standard full Hankel transform.",
            "5. k=5 h/Mpc is a numerical integration bound, not a fitted data kmax.",
            "6. No artificial Fourier-band diagnostic is used for physical interpretation.",
            "7. Residual stochastic power affects xi0 covariance, while its mean contact term is zero at s>0.",
        ]
        axis.text(0.04, 0.95, "\n".join(lines), va="top", ha="left", fontsize=13, linespacing=1.55)
        pdf.savefig(figure)
        plt.close(figure)
    temporary.replace(output)
    if output.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError(f"invalid PDF: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    set_cpu_affinity(args.threads)
    targets = (OUTPUT_JSON, OUTPUT_NPZ, OUTPUT_PDF)
    if all(path.is_file() for path in targets) and not args.overwrite:
        print(json.dumps({"status": "skip", "summary": str(OUTPUT_JSON)}, sort_keys=True))
        return
    if any(path.exists() for path in targets) and not args.overwrite:
        raise FileExistsError("partial audit exists; pass --overwrite after inspection")

    engine = read_json(ENGINE_AUDIT)
    catalogs: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    plotting: dict[str, dict[str, Any]] = {}
    for tag in ("c300", "c302"):
        record, these_arrays, these_plotting = build_catalog(tag)
        catalogs[tag] = record
        arrays.update(these_arrays)
        plotting[tag] = these_plotting

    legacy_paths = {
        tag: {
            "fixed_fnl_freep": str(legacy_lightcone_matched_fit_prefix(tag).with_suffix(".json")),
            "fixed_p1_freefnl": str(legacy_lightcone_matched_fit_prefix(tag, fixed_p=1.0).with_suffix(".json")),
        }
        for tag in ("c300", "c302")
    }
    global_gates = {
        "three_pair_engines_pass": engine.get("status") == "pass",
        "periodic_primary_is_49_not_16_bins": bool(PK_PRIMARY_FIT_EDGES.shape == (49, 2) and PK_LEGACY_LIGHTCONE_MATCHED_EDGES.shape == (16, 2)),
        "all_catalog_gates_pass": bool(all(all(record["gates"].values()) for record in catalogs.values())),
        "xi_audit_has_no_artificial_fourier_diagnostic": True,
    }
    summary: dict[str, Any] = {
        "task": "task44_pngbase_pk_xi_consistency_final_audit",
        "status": "pass" if all(global_gates.values()) else "review",
        "geometry": {"boxsize_mpc_h": BOX_SIZE, "volume_mpc_h3": BOX_VOLUME, "kfund_h_mpc": K_FUND, "space": "real", "window_ric_aic": False},
        "authoritative_pk_selection": {
            "selection": "all 49 contiguous native periodic-box bins",
            "bin_width_h_mpc": 0.002,
            "edge_range_h_mpc": [0.003, 0.101],
            "center_range_h_mpc": [0.004, 0.100],
            "legacy_lightcone_matched_16_bin_selection": {"status": "superseded_wrong_primary_selection", "use_in_current_results": False, "paths_retained_for_provenance": legacy_paths},
        },
        "xi_mean_policy": {
            "method": "standard full Hankel transform from the configured theory cache",
            "k5_role": "numerical integration upper bound only",
            "fitted_data_kmax_interpretation": False,
            "artificial_fourier_diagnostics_used": False,
        },
        "measurement_bridge": {"three_engine_audit": str(ENGINE_AUDIT), "three_engine_audit_sha256": sha256_file(ENGINE_AUDIT), "three_engine_max_abs_delta_xi": engine_max_abs_delta(engine), "conclusion": engine.get("conclusion")},
        "catalogs": catalogs,
        "gates": global_gates,
    }
    draw_pdf(summary, plotting, OUTPUT_PDF)
    summary["output_pdf"] = str(OUTPUT_PDF)
    summary["output_pdf_sha256"] = sha256_file(OUTPUT_PDF)
    atomic_savez(OUTPUT_NPZ, **arrays, summary_json=np.asarray(json.dumps(summary, sort_keys=True)))
    summary["output_npz"] = str(OUTPUT_NPZ)
    summary["output_npz_sha256"] = sha256_file(OUTPUT_NPZ)
    atomic_write_json(OUTPUT_JSON, summary)
    print(json.dumps({"status": summary["status"], "gates": global_gates, "summary": str(OUTPUT_JSON), "plot": str(OUTPUT_PDF)}, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
