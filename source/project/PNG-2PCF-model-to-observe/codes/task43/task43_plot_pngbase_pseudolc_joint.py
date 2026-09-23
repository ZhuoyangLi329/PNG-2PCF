#!/usr/bin/env python3
"""Render the standard six-page Task4.3 layout for one pngbase mean fit."""

from __future__ import annotations

import argparse

from task43_pngbase_pseudolc_common import COSMOLOGIES, FINAL_ROOT, fit_root, fit_root_png_cov
import task43_plot_lightcone_joint_baomask_v1 as standard_plot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosmology", choices=COSMOLOGIES, required=True)
    parser.add_argument("--covariance-mode", choices=("task43_fnl0", "c302_fnl100"), default="task43_fnl0")
    args = parser.parse_args()
    if args.covariance_mode == "c302_fnl100" and args.cosmology != "c302":
        raise ValueError("c302_fnl100 covariance mode is defined only for c302")
    root = fit_root_png_cov() if args.covariance_mode == "c302_fnl100" else fit_root(args.cosmology)
    standard_plot.DEFAULT_ROOT = root
    standard_plot.AUDIT = root / "task43_pngbase_pseudolc_joint_baomask80_120.json"
    standard_plot.COVARIANCE = root / "task43_pngbase_pseudolc_joint_baomask80_120_covariance.npz"
    covariance_tag = "_cov_fnl100" if args.covariance_mode == "c302_fnl100" else ""
    standard_plot.OUTPUT = FINAL_ROOT / (
        f"task43_pngbase_{args.cosmology}_pseudolc_real_Pxi_rsd_P02xi02_joint{covariance_tag}_"
        "kmax0p08_smin50_baomask80_120.pdf"
    )
    standard_plot.main()


if __name__ == "__main__":
    main()
