#!/usr/bin/env python3
"""Validate and summarize all 25 Task 4.3.2 radial-LOS measurements."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from task43_rsd_common import OUTPUT_ROOT, PHASES, S_EDGES, atomic_savez, atomic_write_json, sha256_file


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def measurement_path(phase: str, manifest_rows: list[dict[str, Any]] | None = None) -> Path:
    if manifest_rows is not None:
        matches = [row for row in manifest_rows if row["phase"] == phase]
        if len(matches) != 1:
            raise ValueError(f"expected one manifest row for {phase}, found {len(matches)}")
        return Path(matches[0]["lightcone_xi_path"])
    return OUTPUT_ROOT / "lightcone" / "xi" / (
        f"task43_rsd_xi0_AbacusSummit_base_c000_{phase}_mmin1p4e13_"
        "zobs0p6_0p8_x25_s30_350_ds10.npz"
    )


def validate_measurement(
    phase: str,
    manifest_rows: list[dict[str, Any]] | None = None,
    expected_ells: tuple[int, ...] = (0, 2),
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    path = measurement_path(phase, manifest_rows)
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing lightcone measurement for {phase}: {path} / {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    if metadata.get("status") != "pass" or metadata.get("output_sha256") != digest:
        raise RuntimeError(f"unvalidated lightcone measurement for {phase}: {path}")
    if metadata.get("los") != "midpoint" or metadata.get("ells") != list(expected_ells):
        raise RuntimeError(f"unexpected LOS/multipoles for {phase}")
    with np.load(path, allow_pickle=False) as source:
        required = {"s", "s_edges", "ells", "xi0", "xi_multipoles", "RR", "zeff", "p0", "phase"}
        if 2 in expected_ells:
            required.add("xi2")
        missing = sorted(required - set(source.files))
        if missing:
            raise KeyError(f"{path} is missing {missing}")
        payload = {key: np.asarray(source[key]) for key in required}
    if not np.array_equal(np.asarray(payload["s_edges"], dtype="f8"), S_EDGES):
        raise RuntimeError(f"separation edges changed for {phase}")
    if tuple(int(value) for value in np.asarray(payload["ells"]).ravel()) != expected_ells:
        raise RuntimeError(f"multipole order changed for {phase}")
    if str(np.asarray(payload["phase"]).item()) != phase:
        raise RuntimeError(f"phase label mismatch in {path}")
    xi = np.asarray(payload["xi_multipoles"], dtype="f8")
    rr = np.asarray(payload["RR"], dtype="f8")
    if xi.shape != (len(expected_ells), S_EDGES.size - 1) or not np.all(np.isfinite(xi)):
        raise RuntimeError(f"invalid xi multipoles for {phase}")
    if rr.shape != (S_EDGES.size - 1,) or not np.all(np.isfinite(rr)) or not np.all(rr > 0.0):
        raise RuntimeError(f"invalid radial RR for {phase}")
    if not np.array_equal(xi[expected_ells.index(0)], np.asarray(payload["xi0"], dtype="f8")):
        raise RuntimeError(f"xi0 alias mismatch for {phase}")
    if 2 in expected_ells and not np.array_equal(
        xi[expected_ells.index(2)], np.asarray(payload["xi2"], dtype="f8")
    ):
        raise RuntimeError(f"xi2 alias mismatch for {phase}")
    return payload, metadata, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--ells", default="0,2", help="Comma-separated observed multipoles; supported: 0 or 0,2")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=OUTPUT_ROOT / "lightcone" / "summary" /
        "task43_rsd_lightcone_x25_mean_xi02_s30_350_ds10",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/plots/task43/rsd_validation") /
        "task43_rsd_lightcone_x25_mean_xi02.pdf",
    )
    args = parser.parse_args()
    ells = tuple(int(value) for value in str(args.ells).split(",") if value.strip())
    if ells not in ((0,), (0, 2)):
        raise ValueError("--ells must be either 0 or 0,2")
    output_npz = args.output_prefix.with_suffix(".npz")
    output_json = args.output_prefix.with_suffix(".json")
    mean_view = args.output_prefix.parent / "task43_rsd_lightcone_x25_mean_fit_view.npz"
    mean_zeff = args.output_prefix.parent / "task43_rsd_lightcone_x25_mean_zeff.npz"
    products = (output_npz, output_json, mean_view, mean_zeff, args.plot)
    if any(path.exists() for path in products):
        raise FileExistsError(f"immutable x25 lightcone summary product exists: {[str(path) for path in products if path.exists()]}")

    manifest_rows = None if args.manifest is None else read_jsonl(args.manifest)
    if manifest_rows is not None and tuple(row["phase"] for row in manifest_rows) != PHASES:
        raise RuntimeError("manifest phase order is not ph000..ph024")
    rows = [validate_measurement(phase, manifest_rows, ells) for phase in PHASES]
    payloads = [row[0] for row in rows]
    metadata_rows = [row[1] for row in rows]
    hashes = [row[2] for row in rows]
    s = np.asarray(payloads[0]["s"], dtype="f8")
    if any(not np.array_equal(s, np.asarray(payload["s"], dtype="f8")) for payload in payloads[1:]):
        raise RuntimeError("radial centers differ across phases")
    xi = np.stack([np.asarray(payload["xi_multipoles"], dtype="f8") for payload in payloads])
    rr = np.stack([np.asarray(payload["RR"], dtype="f8") for payload in payloads])
    zeff = np.asarray([float(np.asarray(payload["zeff"]).item()) for payload in payloads], dtype="f8")
    p0 = np.asarray([float(np.asarray(payload["p0"]).item()) for payload in payloads], dtype="f8")
    if not np.all(p0 == 10000.0):
        raise RuntimeError(f"FKP P0 changed across phases: {p0}")
    mean = np.mean(xi, axis=0)
    scatter = np.std(xi, axis=0, ddof=1)
    vector = np.concatenate([xi[:, iell, :] for iell in range(len(ells))], axis=1)
    scatter_covariance = np.cov(vector, rowvar=False, ddof=1)
    mean_rr = np.mean(rr, axis=0)
    zeff_mean = float(np.mean(zeff))

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    temporary_plot = args.plot.with_name(f".{args.plot.stem}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary_plot) as pdf:
        figure, axes = plt.subplots(len(ells), 1, figsize=(8.4, 3.6 * len(ells) + 0.2), sharex=True)
        axes = np.atleast_1d(axes)
        labels = {0: r"$\xi_0$", 2: r"$\xi_2$"}
        for iell, axis in enumerate(axes):
            scaled = s[None, :] ** 2 * xi[:, iell, :]
            for row in scaled:
                axis.plot(s, row, color="0.72", alpha=0.20, lw=0.55)
            q16, q84 = np.percentile(scaled, [16.0, 84.0], axis=0)
            axis.fill_between(s, q16, q84, color="#4c78a8", alpha=0.22, label="phase 16--84%")
            axis.plot(s, s**2 * mean[iell], color="#b22222", lw=1.6, label="x25 mean")
            axis.axhline(0.0, color="0.35", lw=0.7)
            axis.set_ylabel(rf"$s^2 {labels[ells[iell]]}(s)$")
            axis.legend(frameon=False, fontsize=8)
        axes[-1].set_xlabel(r"$s\,[h^{-1}{\rm Mpc}]$")
        figure.suptitle("Task 4.3.2 radial-LOS fNL=0 lightcone measurements")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)
    embedded = {
        "task": "task43_summarize_rsd_lightcone_x25",
        "status": "pass",
        "classification": "equal-phase mean of 25 independent fNL=0 radial-LOS lightcones",
        "phases": list(PHASES),
        "nphase": len(PHASES),
        "los": "midpoint",
        "ells": list(ells),
        "estimator": "weighted Landy-Szalay with x25 independent split randoms per phase",
        "mean_policy": "unweighted arithmetic mean across phases",
        "covariance_role_warning": "empirical x25 scatter is a diagnostic; quoted posterior uses the validated single-lightcone covariance",
        "input_paths": [str(measurement_path(phase, manifest_rows)) for phase in PHASES],
        "input_sha256": hashes,
        "zeff_mean": zeff_mean,
        "manifest": None if args.manifest is None else str(args.manifest),
        "observed_redshift_open_interval": (
            None
            if manifest_rows is None
            else [float(manifest_rows[0]["zmin_observed"]), float(manifest_rows[0]["zmax_observed"])]
        ),
    }
    summary_payload: dict[str, Any] = {
        "s": s,
        "s_edges": S_EDGES,
        "ells": np.asarray(ells, dtype="i4"),
        "phases": np.asarray(PHASES),
        "xi_multipoles_by_phase": xi,
        "xi0_by_phase": xi[:, ells.index(0)],
        "xi_multipoles_mean": mean,
        "xi0_mean": mean[ells.index(0)],
        "xi_multipoles_scatter_single": scatter,
        "scatter_covariance_x25": scatter_covariance,
        "RR_by_phase": rr,
        "RR_mean": mean_rr,
        "zeff_by_phase": zeff,
        "zeff_mean": np.asarray(zeff_mean, dtype="f8"),
        "p0": np.asarray(10000.0, dtype="f8"),
        "meta_json": np.asarray(json.dumps(embedded, sort_keys=True)),
    }
    if 2 in ells:
        summary_payload["xi2_by_phase"] = xi[:, ells.index(2)]
        summary_payload["xi2_mean"] = mean[ells.index(2)]
        summary_payload["scatter_covariance_x25_xi02"] = scatter_covariance
    atomic_savez(output_npz, **summary_payload)
    fit_meta = {
        **embedded,
        "view_role": f"Task44-compatible multipole mean fit input, ells={list(ells)}",
        "rr_radial_definition": "equal-phase mean of per-phase sum_mu(mean split-random RR_smu)",
    }
    mean_payload: dict[str, Any] = {
        "s": s,
        "s_edges": S_EDGES,
        "ells": np.asarray(ells, dtype="i4"),
        "xi0": mean[ells.index(0)],
        "xi_multipoles": mean,
        "RR": mean_rr,
        "zeff": np.asarray(zeff_mean, dtype="f8"),
        "p0": np.asarray(10000.0, dtype="f8"),
        "phase": np.asarray("x25_mean"),
        "meta_json": np.asarray(json.dumps(fit_meta, sort_keys=True)),
    }
    if 2 in ells:
        mean_payload["xi2"] = mean[ells.index(2)]
    atomic_savez(mean_view, **mean_payload)
    atomic_savez(
        mean_zeff,
        zeff=np.asarray(zeff_mean, dtype="f8"),
        zeff_by_phase=zeff,
        phases=np.asarray(PHASES),
        p0=np.asarray(10000.0, dtype="f8"),
        meta_json=np.asarray(json.dumps(fit_meta, sort_keys=True)),
    )
    temporary_plot.replace(args.plot)
    metadata = {
        **embedded,
        "zeff_by_phase": zeff.tolist(),
        "zeff_range": [float(np.min(zeff)), float(np.max(zeff))],
        "ndata_by_phase": [int(row["ndata"]) for row in metadata_rows],
        "nrandom_by_phase": [int(row["nrandom_total"]) for row in metadata_rows],
        "outputs": {
            "summary_npz": str(output_npz),
            "summary_sha256": sha256_file(output_npz),
            "mean_fit_view": str(mean_view),
            "mean_fit_view_sha256": sha256_file(mean_view),
            "mean_zeff": str(mean_zeff),
            "mean_zeff_sha256": sha256_file(mean_zeff),
            "plot_pdf": str(args.plot),
            "plot_pdf_sha256": sha256_file(args.plot),
        },
    }
    atomic_write_json(output_json, metadata)
    print(json.dumps({"status": "pass", "nphase": len(PHASES), "output": str(output_npz)}, sort_keys=True))


if __name__ == "__main__":
    main()
