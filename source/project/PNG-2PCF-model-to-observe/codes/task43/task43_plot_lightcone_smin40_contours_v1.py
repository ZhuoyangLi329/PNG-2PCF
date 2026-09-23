#!/usr/bin/env python3
"""Write standalone Task 4.3 smin=40 versus smin=50 contour PDFs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from task43_plot_lightcone_smin40_vs50_v1 import (
    NEW_AUDIT,
    NEW_ROOT,
    OLD_AUDIT,
    OLD_ROOT,
    load_chain,
    summary,
    triangle_page,
)
from task43_rsd_common import PROJECT_ROOT, atomic_write_json, sha256_file


OUTPUT_DIR = PROJECT_ROOT / "9.11meeting/task43_lightcone_smin40_contours_v1"
OUTPUTS = {
    "real": OUTPUT_DIR / "task43_lightcone_real_P0_xi0_joint_smin40_vs50_contours_v1.pdf",
    "rsd_monopole": OUTPUT_DIR / "task43_lightcone_rsd_P0_xi0_joint_smin40_vs50_contours_v1.pdf",
    "rsd_multipole": OUTPUT_DIR / "task43_lightcone_rsd_P02_xi02_joint_smin40_vs50_contours_v1.pdf",
}
MANIFEST = OUTPUT_DIR / "manifest.json"


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    old_audit = json.loads(OLD_AUDIT.read_text(encoding="utf-8"))
    new_audit = json.loads(NEW_AUDIT.read_text(encoding="utf-8"))
    for audit, label in ((old_audit, "smin50"), (new_audit, "smin40")):
        if audit.get("status") != "pass" or not all(audit["numerical_gates"].values()):
            raise RuntimeError(f"{label} audit did not pass")
    old = {name: load_chain(OLD_ROOT, name, result) for name, result in old_audit["results"].items()}
    new = {
        name: (old[name] if name in ("real_p0", "rsd_p0", "rsd_p02") else load_chain(NEW_ROOT, name, result))
        for name, result in new_audit["results"].items()
    }
    return old_audit, new_audit, old, new


def build_groups(
    old: dict[str, dict[str, Any]], new: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {
        "real": {
            "title": "Lightcone real space (0.6 < z < 0.8): smin=40 versus 50",
            "parameters": ("fNL", "b1"),
            "variants": (
                ("p", old["real_p0"], r"$P_0$ (unchanged)"),
                ("xi50", old["real_xi0"], r"$\xi_0$, $s_{min}=50$"),
                ("xi40", new["real_xi0"], r"$\xi_0$, $s_{min}=40$"),
                ("joint50", old["real_joint_p0xi0"], r"joint, $s_{min}=50$"),
                ("joint40", new["real_joint_p0xi0"], r"joint, $s_{min}=40$"),
            ),
        },
        "rsd_monopole": {
            "title": "Lightcone RSD monopole (0.4 < zobs < 0.8): smin=40 versus 50",
            "parameters": ("fNL", "b1", "sigma_s"),
            "variants": (
                ("p", old["rsd_p0"], r"$P_0$ (unchanged)"),
                ("xi50", old["rsd_xi0"], r"$\xi_0$, $s_{min}=50$"),
                ("xi40", new["rsd_xi0"], r"$\xi_0$, $s_{min}=40$"),
                ("joint50", old["rsd_joint_p0xi0"], r"joint, $s_{min}=50$"),
                ("joint40", new["rsd_joint_p0xi0"], r"joint, $s_{min}=40$"),
            ),
        },
        "rsd_multipole": {
            "title": "Lightcone RSD multipoles (0.4 < zobs < 0.8): smin=40 versus 50",
            "parameters": ("fNL", "b1", "sigma_s"),
            "variants": (
                ("p", old["rsd_p02"], r"$P_{0,2}$ (unchanged)"),
                ("xi50", old["rsd_xi02"], r"$\xi_{0,2}$, $s_{min}=50$"),
                ("xi40", new["rsd_xi02"], r"$\xi_{0,2}$, $s_{min}=40$"),
                ("joint50", old["rsd_joint_p02xi02"], r"joint, $s_{min}=50$"),
                ("joint40", new["rsd_joint_p02xi02"], r"joint, $s_{min}=40$"),
            ),
        },
    }


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9.6,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def write_pdf(path: Path, group: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary) as pdf:
        triangle_page(
            pdf,
            group["variants"],
            group["parameters"],
            title=group["title"],
        )
    temporary.replace(path)
    if path.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError(f"not a PDF: {path}")


def main() -> None:
    existing = [path for path in (*OUTPUTS.values(), MANIFEST) if path.exists()]
    if existing:
        raise FileExistsError(f"immutable contour output exists: {existing}")
    old_audit, new_audit, old, new = load_inputs()
    groups = build_groups(old, new)
    configure_plotting()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    for name, path in OUTPUTS.items():
        write_pdf(path, groups[name])

    constraints = {
        group_name: {
            key: {
                parameter: summary(chain, parameter)
                for parameter in group["parameters"]
            }
            for key, chain, _label in group["variants"]
        }
        for group_name, group in groups.items()
    }
    outputs = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in OUTPUTS.items()
    }
    atomic_write_json(
        MANIFEST,
        {
            "task": "task43_plot_lightcone_smin40_contours_v1",
            "status": "pass",
            "contour_levels": ["68%", "95%"],
            "center_annotation": "continuous-optimizer maximum likelihood; never posterior median",
            "interval_annotation": "posterior q16 to q84",
            "variants": "unchanged P, xi smin50, xi smin40, joint smin50, joint smin40",
            "old_audit": str(OLD_AUDIT),
            "old_audit_sha256": sha256_file(OLD_AUDIT),
            "new_audit": str(NEW_AUDIT),
            "new_audit_sha256": sha256_file(NEW_AUDIT),
            "covariance_contract": new_audit["covariance_contract"],
            "constraints": constraints,
            "outputs": outputs,
        },
    )
    print(json.dumps({"status": "pass", "output_dir": str(OUTPUT_DIR), "outputs": outputs}, sort_keys=True))


if __name__ == "__main__":
    main()
