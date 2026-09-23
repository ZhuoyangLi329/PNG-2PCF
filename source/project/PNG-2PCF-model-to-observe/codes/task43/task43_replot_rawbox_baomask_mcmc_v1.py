#!/usr/bin/env python3
"""Compare frozen P0 and masked/unmasked rawbox xi0 chains without rerunning MCMC."""

from __future__ import annotations

import json
import os

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from task43_rsd_common import PROJECT_ROOT, atomic_write_json, sha256_file
from task43_run_rawbox_baomask_mcmc_v1 import (
    OUT_JSON,
    OUT_NPZ,
    REAL_NAMES,
    REAL_P_JSON,
    REAL_P_NPZ,
    RSD_NAMES,
    RSD_P_JSON,
    RSD_P_NPZ,
    draw_residual_page,
    draw_triangle_page,
)


LEGACY_REAL_XI_NPZ = (
    PROJECT_ROOT
    / "outputs/task43_outputs/rsd_validation/rawbox/realspace_check/fits/smin50/samples.npz"
)
LEGACY_REAL_XI_JSON = (
    PROJECT_ROOT
    / "outputs/task43_outputs/rsd_validation/rawbox/realspace_check/audits/task43_rsd_rawbox_realspace_check.json"
)
OUTPUT = PROJECT_ROOT / "9.11meeting" / "task43_rawbox_real_rsd_pk0_vs_xi0_mask_comparison_v1p4.pdf"


def fnl_summary(samples: np.ndarray, maximum_likelihood: float) -> dict[str, float | str]:
    q16, q50, q84 = np.percentile(np.asarray(samples, dtype="f8")[:, 0], [16.0, 50.0, 84.0])
    return {
        "central_estimator": "joint maximum likelihood",
        "interval_definition": "marginal posterior equal-tail q16--q84",
        "maximum_likelihood": float(maximum_likelihood),
        "q16": float(q16),
        "q50": float(q50),
        "q84": float(q84),
        "minus_1sigma_from_maximum_likelihood": float(maximum_likelihood - q16),
        "plus_1sigma_from_maximum_likelihood": float(q84 - maximum_likelihood),
    }


def main() -> None:
    output_json = OUTPUT.with_suffix(".json")
    if OUTPUT.exists() or output_json.exists():
        raise FileExistsError(f"immutable plot exists: {OUTPUT} / {output_json}")
    summary = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    if summary.get("status") != "pass" or summary.get("output_npz_sha256") != sha256_file(OUT_NPZ):
        raise RuntimeError("BAO-mask MCMC input failed status/hash validation")
    with np.load(OUT_NPZ, allow_pickle=False) as data:
        centers = np.asarray(data["s_centers"], dtype="f8")
        mask = np.asarray(data["primary_mask"], dtype=bool)
        real_chain = np.asarray(data["real_chain_by_step"], dtype="f8")
        rsd_chain = np.asarray(data["rsd_chain_by_step"], dtype="f8")
        real_data = np.asarray(data["xi0_real_mean"], dtype="f8")
        rsd_data = np.asarray(data["xi0_rsd_mean"], dtype="f8")
        real_prediction = np.asarray(data["real_prediction_map"], dtype="f8")
        rsd_prediction = np.asarray(data["rsd_prediction_map"], dtype="f8")
        real_covariance = np.asarray(data["real_covariance_single"], dtype="f8")
        rsd_covariance = np.asarray(data["rsd_covariance_single"], dtype="f8")
    with np.load(REAL_P_NPZ, allow_pickle=False) as data:
        real_p = np.asarray(data["chain_by_step"], dtype="f8").reshape(-1, 2)
    with np.load(LEGACY_REAL_XI_NPZ, allow_pickle=False) as data:
        real_xi_legacy = np.asarray(data["chain_by_step"], dtype="f8").reshape(-1, 2)
    with np.load(RSD_P_NPZ, allow_pickle=False) as data:
        rsd_p = np.asarray(data["pk0_chain_by_step"], dtype="f8").reshape(-1, 4)[:, :3]
        rsd_xi_legacy = np.asarray(data["xi0_chain_by_step"], dtype="f8").reshape(-1, 3)
    real_p_summary = json.loads(REAL_P_JSON.read_text(encoding="utf-8"))
    legacy_real_summary = json.loads(LEGACY_REAL_XI_JSON.read_text(encoding="utf-8"))
    rsd_p_summary = json.loads(RSD_P_JSON.read_text(encoding="utf-8"))
    maximum_likelihood = {
        "real_P0": float(real_p_summary["map"]["fNL"]),
        "real_xi0_no_mask": float(legacy_real_summary["results"]["smin50"]["fNL"]),
        "real_xi0_BAO_mask": float(summary["realspace"]["map"]["theta"]["fNL"]),
        "RSD_P0": float(rsd_p_summary["pk0"]["nominal"]["theta"]["fNL"]),
        "RSD_xi0_no_mask": float(rsd_p_summary["xi0"]["nominal"]["theta"]["fNL"]),
        "RSD_xi0_BAO_mask": float(summary["redshift_space"]["map"]["theta"]["fNL"]),
    }
    for label, samples in (
        ("real P0", real_p),
        ("real xi0 no mask", real_xi_legacy),
        ("real xi0 BAO mask", real_chain.reshape(-1, 2)),
        ("RSD P0", rsd_p),
        ("RSD xi0 no mask", rsd_xi_legacy),
        ("RSD xi0 BAO mask", rsd_chain.reshape(-1, 3)),
    ):
        if not np.all(np.isfinite(samples)):
            raise RuntimeError(f"non-finite samples in {label}")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11.0,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary) as pdf:
        draw_triangle_page(
            pdf,
            real_p,
            real_chain.reshape(-1, 2),
            REAL_NAMES,
            title="Rawbox real space: P0 and xi0 mask comparison",
            legacy_xi_samples=real_xi_legacy,
            fnl_maximum_likelihood={
                "p0": maximum_likelihood["real_P0"],
                "xi0_unmasked": maximum_likelihood["real_xi0_no_mask"],
                "xi0_masked": maximum_likelihood["real_xi0_BAO_mask"],
            },
        )
        draw_triangle_page(
            pdf,
            rsd_p,
            rsd_chain.reshape(-1, 3),
            RSD_NAMES,
            title="Rawbox redshift space: P0 and xi0 mask comparison",
            legacy_xi_samples=rsd_xi_legacy,
            fnl_maximum_likelihood={
                "p0": maximum_likelihood["RSD_P0"],
                "xi0_unmasked": maximum_likelihood["RSD_xi0_no_mask"],
                "xi0_masked": maximum_likelihood["RSD_xi0_BAO_mask"],
            },
        )
        draw_residual_page(
            pdf,
            centers,
            mask,
            real_data,
            rsd_data,
            real_prediction,
            rsd_prediction,
            real_covariance,
            rsd_covariance,
            {"real": summary["realspace"]["map"], "rsd": summary["redshift_space"]["map"]},
        )
    temporary.replace(OUTPUT)
    if OUTPUT.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError("output is not a PDF")
    atomic_write_json(
        output_json,
        {
            "task": "task43_replot_rawbox_baomask_mcmc_v1",
            "status": "pass",
            "layout_revision": "v1p4: joint-ML centers, marginal q16--q84 errors, and legacy unmasked xi0 contours",
            "input_npz": str(OUT_NPZ),
            "input_npz_sha256": sha256_file(OUT_NPZ),
            "input_json": str(OUT_JSON),
            "input_json_sha256": sha256_file(OUT_JSON),
            "legacy_real_xi_npz": str(LEGACY_REAL_XI_NPZ),
            "legacy_real_xi_npz_sha256": sha256_file(LEGACY_REAL_XI_NPZ),
            "legacy_real_xi_json": str(LEGACY_REAL_XI_JSON),
            "legacy_real_xi_json_sha256": sha256_file(LEGACY_REAL_XI_JSON),
            "legacy_rsd_xi_npz": str(RSD_P_NPZ),
            "legacy_rsd_xi_npz_sha256": sha256_file(RSD_P_NPZ),
            "fNL": {
                "real_P0": fnl_summary(real_p, maximum_likelihood["real_P0"]),
                "real_xi0_no_mask": fnl_summary(real_xi_legacy, maximum_likelihood["real_xi0_no_mask"]),
                "real_xi0_BAO_mask": fnl_summary(real_chain.reshape(-1, 2), maximum_likelihood["real_xi0_BAO_mask"]),
                "RSD_P0": fnl_summary(rsd_p, maximum_likelihood["RSD_P0"]),
                "RSD_xi0_no_mask": fnl_summary(rsd_xi_legacy, maximum_likelihood["RSD_xi0_no_mask"]),
                "RSD_xi0_BAO_mask": fnl_summary(rsd_chain.reshape(-1, 3), maximum_likelihood["RSD_xi0_BAO_mask"]),
            },
            "output_pdf": str(OUTPUT),
            "output_pdf_sha256": sha256_file(OUTPUT),
        },
    )
    print(json.dumps({"status": "pass", "output": str(OUTPUT), "sha256": sha256_file(OUTPUT)}))


if __name__ == "__main__":
    main()
