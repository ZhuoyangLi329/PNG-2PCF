#!/usr/bin/env python3
"""Measure M>=1e13 occupied-host halo xi0 with three independent engines.

The sample is one unit-weight point per unique HOD host surviving the official
cleaned CompaSO mass cut.  FCFC, pycorr/Corrfunc, and CuCount all use the same
periodic L=2000 Mpc/h geometry, [30,350) Mpc/h bins, and ordered-pair
normalization N(N-1).  Products are isolated from the pre-existing Task44 tree.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task44_pngbase_hodhost_mmin1e13_common import (
    BASE_OUTPUT_ROOT,
    BOX_SIZE,
    BOX_VOLUME,
    CATALOGS,
    HALO_MASS_MIN_HMSUN,
    K_FUND,
    LOG_DIR,
    PLOT_ROOT,
    PROJECT_ROOT,
    S_CENTERS,
    S_EDGES,
    SUMMARY_DIR,
    analytic_rr,
    atomic_savez,
    atomic_write_json,
    ensure_output_dirs,
    get_spec,
    host_ascii_metadata_path,
    host_ascii_path,
    host_catalog_path,
    load_host_catalog,
    set_cpu_affinity,
    sha256_file,
    xi_engine_metadata_path,
    xi_engine_path,
)


FCFC_BINARY = PROJECT_ROOT / "refcode" / "FCFC-main" / "FCFC_2PT_BOX"
FCFC_READER = PROJECT_ROOT / "refcode" / "FCFC-main" / "scripts" / "read_pair_count.py"
ENGINES = ("fcfc", "pycorr", "cucount")
COLORS = {"fcfc": "black", "pycorr": "#4C72B0", "cucount": "#C44E52"}
MARKERS = {"fcfc": None, "pycorr": "o", "cucount": "x"}


def _load_positions(tag: str) -> tuple[np.ndarray, dict[str, Any]]:
    arrays, metadata = load_host_catalog(tag)
    position = np.asarray(arrays["position"], dtype="f8")
    return position, {
        "source": str(host_catalog_path(tag)),
        "source_sha256": metadata["output_sha256"],
        "ndata": int(position.shape[0]),
        "coordinate_min": np.min(position, axis=0).tolist(),
        "coordinate_max": np.max(position, axis=0).tolist(),
        "origin": "positive periodic [0,L)",
    }


def _validated_product(tag: str, engine: str) -> bool:
    output = xi_engine_path(tag, engine)
    metadata_path = xi_engine_metadata_path(tag, engine)
    if not output.is_file() or not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(output)


def save_engine_product(
    tag: str,
    engine: str,
    *,
    dd_raw: np.ndarray,
    dd_norm_total: float,
    dd_normalized: np.ndarray,
    rr: np.ndarray,
    xi0: np.ndarray,
    s_pair_average: np.ndarray,
    measured_edges: np.ndarray,
    elapsed: float,
    engine_metadata: dict[str, Any],
) -> None:
    arrays, host_metadata = load_host_catalog(tag)
    ndata = int(np.asarray(arrays["ndata"]).item())
    expected_rr = analytic_rr()
    dd_raw = np.asarray(dd_raw, dtype="f8")
    dd_normalized = np.asarray(dd_normalized, dtype="f8")
    rr = np.asarray(rr, dtype="f8")
    xi0 = np.asarray(xi0, dtype="f8")
    measured_edges = np.asarray(measured_edges, dtype="f8")
    reconstructed = dd_normalized / expected_rr - 1.0
    contract = {
        "s_edges_bitwise_equal": bool(np.array_equal(measured_edges, S_EDGES)),
        "smax_below_half_box": bool(S_EDGES[-1] < BOX_SIZE / 2.0),
        "dd_raw_finite_nonnegative": bool(np.all(np.isfinite(dd_raw)) and np.all(dd_raw >= 0.0)),
        "dd_normalized_finite_positive": bool(
            np.all(np.isfinite(dd_normalized)) and np.all(dd_normalized > 0.0)
        ),
        "rr_matches_exact_shell_volume": bool(np.allclose(rr, expected_rr, rtol=0.0, atol=2.0e-15)),
        "xi_reconstructed_from_dd_rr": bool(np.allclose(xi0, reconstructed, rtol=0.0, atol=2.0e-13)),
        "normalization_is_ordered_auto_pairs": bool(
            np.isclose(dd_norm_total, ndata * (ndata - 1), rtol=0.0, atol=0.5)
        ),
        "host_catalog_mass_cut_passed": bool(host_metadata["gates"]["selected_mass_cut_exact"]),
    }
    if not all(contract.values()):
        raise RuntimeError(f"{tag}/{engine} measurement contract failed: {contract}")
    output = xi_engine_path(tag, engine)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        output,
        s=np.asarray(S_CENTERS, dtype="f8"),
        s_pair_average=np.asarray(s_pair_average, dtype="f8"),
        s_edges=measured_edges,
        xi0=xi0,
        dd_raw=dd_raw,
        dd_norm_total=np.asarray(dd_norm_total, dtype="f8"),
        dd_normalized=dd_normalized,
        rr_analytic=expected_rr,
        tag=np.asarray(tag),
        engine=np.asarray(engine),
        ndata=np.asarray(ndata, dtype="i8"),
        boxsize=np.asarray(BOX_SIZE, dtype="f8"),
        volume=np.asarray(BOX_VOLUME, dtype="f8"),
        mass_min_hmsun=np.asarray(HALO_MASS_MIN_HMSUN, dtype="f8"),
    )
    metadata = {
        "task": "task44_measure_pngbase_hodhost_mmin1e13_xi",
        "status": "pass",
        "tag": tag,
        "engine": engine,
        "source_host_catalog": str(host_catalog_path(tag)),
        "source_host_catalog_sha256": host_metadata["output_sha256"],
        "output": str(output),
        "output_sha256": sha256_file(output),
        "selection": host_metadata["selection"],
        "ndata": ndata,
        "geometry": {
            "type": "full periodic real-space cube",
            "boxsize_mpc_h": BOX_SIZE,
            "volume_mpc_h3": BOX_VOLUME,
            "kfund_h_mpc": K_FUND,
        },
        "s_edges_mpc_h": S_EDGES.tolist(),
        "estimator": "normalized ordered DD / exact analytic spherical-shell RR - 1",
        "pair_normalization": "ordered auto pairs N*(N-1), no self pairs",
        "contract": contract,
        "engine_metadata": engine_metadata,
        "elapsed_sec": elapsed,
    }
    atomic_write_json(xi_engine_metadata_path(tag, engine), metadata)
    print(f"[done] {tag} {engine} bins={xi0.size} elapsed={elapsed:.2f}s path={output}", flush=True)


def _load_fcfc_pair_count(path: Path) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    spec = importlib.util.spec_from_file_location("task44_host_fcfc_pair_reader", FCFC_READER)
    if spec is None or spec.loader is None:
        raise ImportError(FCFC_READER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.read_pair_count(str(path), verbose=False)
    bin_type, nsbin, nmbin, npbin, edges, _, num1, num2, norm, normalized, raw = result
    checks = {
        "isotropic": int(bin_type) == 0 and nmbin is None and npbin is None,
        "nbins": int(nsbin) == S_EDGES.size - 1,
        "edges": np.allclose(edges, S_EDGES, rtol=0.0, atol=1.0e-13),
        "auto": int(num1) == int(num2),
    }
    if not all(checks.values()):
        raise RuntimeError(f"unexpected FCFC pair file {path}: {checks}")
    return np.asarray(raw, dtype="f8"), np.asarray(normalized, dtype="f8"), float(norm), np.asarray(edges, dtype="f8")


def measure_fcfc(tag: str, *, threads: int, overwrite: bool) -> None:
    if _validated_product(tag, "fcfc") and not overwrite:
        print(f"[skip] validated {xi_engine_path(tag, 'fcfc')}", flush=True)
        return
    if not FCFC_BINARY.is_file() or not os.access(FCFC_BINARY, os.X_OK):
        raise FileNotFoundError(FCFC_BINARY)
    ascii_path = host_ascii_path(tag)
    ascii_metadata = json.loads(host_ascii_metadata_path(tag).read_text(encoding="utf-8"))
    if ascii_metadata.get("output_sha256") != sha256_file(ascii_path):
        raise RuntimeError(f"ASCII hash mismatch: {ascii_path}")
    cpus = set_cpu_affinity(threads)
    work = xi_engine_path(tag, "fcfc").parent / "work"
    work.mkdir(parents=True, exist_ok=True)
    config = work / f"task44_pngbase_{tag}_hodhost_mmin1e13_xi.conf"
    pair = work / f"task44_pngbase_{tag}_hodhost_mmin1e13_DD.bin"
    xi_text_path = work / f"task44_pngbase_{tag}_hodhost_mmin1e13_xi.txt"
    log = LOG_DIR / f"task44_pngbase_{tag}_hodhost_mmin1e13_fcfc.log"
    config.write_text(
        f"""CATALOG = '{ascii_path}'
CATALOG_LABEL = D
CATALOG_TYPE = 0
ASCII_SKIP = 1
ASCII_COMMENT = '#'
ASCII_FORMATTER = '%lf %lf %lf'
POSITION = [$1, $2, $3]
BOX_SIZE = {BOX_SIZE:.17g}
DATA_STRUCT = 0
BINNING_SCHEME = 0
PAIR_COUNT = DD
PAIR_COUNT_FILE = '{pair}'
CF_ESTIMATOR = DD / @@ - 1
CF_OUTPUT_FILE = '{xi_text_path}'
SEP_BIN_MIN = {S_EDGES[0]:.17g}
SEP_BIN_MAX = {S_EDGES[-1]:.17g}
SEP_BIN_SIZE = {S_EDGES[1] - S_EDGES[0]:.17g}
OUTPUT_FORMAT = 0
OVERWRITE = {2 if overwrite else 1}
VERBOSE = T
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(int(threads))
    environment["OMP_DYNAMIC"] = "FALSE"
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(
            [str(FCFC_BINARY), "-c", str(config)],
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    elapsed = time.perf_counter() - started
    table = np.loadtxt(xi_text_path, comments="#", dtype="f8")
    if table.shape != (S_CENTERS.size, 4):
        raise ValueError(f"unexpected FCFC xi table shape: {table.shape}")
    s_pair, lower, upper, xi_text = table.T
    measured = np.column_stack([lower, upper])
    expected = np.column_stack([S_EDGES[:-1], S_EDGES[1:]])
    if not np.allclose(measured, expected, rtol=0.0, atol=1.0e-13):
        raise RuntimeError("FCFC xi bin edges differ from the frozen edges")
    dd_raw, dd_normalized, norm, edges = _load_fcfc_pair_count(pair)
    rr = analytic_rr(edges)
    xi0 = dd_normalized / rr - 1.0
    save_engine_product(
        tag,
        "fcfc",
        dd_raw=dd_raw,
        dd_norm_total=norm,
        dd_normalized=dd_normalized,
        rr=rr,
        xi0=xi0,
        s_pair_average=s_pair,
        measured_edges=edges,
        elapsed=elapsed,
        engine_metadata={
            "name": "FCFC_2PT_BOX",
            "version": "1.0.1",
            "binary": str(FCFC_BINARY),
            "configuration": str(config),
            "pair_count": str(pair),
            "xi_text": str(xi_text_path),
            "xi_text_max_abs_vs_binary_reconstruction": float(np.max(np.abs(xi_text - xi0))),
            "log": str(log),
            "ascii_catalog": str(ascii_path),
            "ascii_catalog_sha256": ascii_metadata["output_sha256"],
            "threads": int(threads),
            "cpu_affinity": cpus,
        },
    )


def measure_pycorr(tag: str, *, threads: int, overwrite: bool) -> None:
    if _validated_product(tag, "pycorr") and not overwrite:
        print(f"[skip] validated {xi_engine_path(tag, 'pycorr')}", flush=True)
        return
    cpus = set_cpu_affinity(threads)
    from pycorr import TwoPointCorrelationFunction, __version__ as pycorr_version
    import Corrfunc

    position, position_metadata = _load_positions(tag)
    started = time.perf_counter()
    correlation = TwoPointCorrelationFunction(
        mode="s",
        edges=S_EDGES,
        data_positions1=position,
        boxsize=BOX_SIZE,
        estimator="natural",
        engine="corrfunc",
        nthreads=int(threads),
        position_type="pos",
    )
    elapsed = time.perf_counter() - started
    dd_raw = np.asarray(correlation.D1D2.wcounts, dtype="f8")
    dd_norm_total = float(correlation.D1D2.wnorm)
    dd_normalized = dd_raw / dd_norm_total
    save_engine_product(
        tag,
        "pycorr",
        dd_raw=dd_raw,
        dd_norm_total=dd_norm_total,
        dd_normalized=dd_normalized,
        rr=np.asarray(correlation.R1R2.normalized_wcounts(), dtype="f8"),
        xi0=np.asarray(correlation.corr, dtype="f8"),
        s_pair_average=np.asarray(correlation.sep, dtype="f8"),
        measured_edges=np.asarray(correlation.D1D2.edges[0], dtype="f8"),
        elapsed=elapsed,
        engine_metadata={
            "name": "pycorr.TwoPointCorrelationFunction",
            "version": str(pycorr_version),
            "counter": "Corrfunc.theory.DD",
            "corrfunc_version": str(Corrfunc.__version__),
            "position": position_metadata,
            "threads": int(threads),
            "cpu_affinity": cpus,
        },
    )


def measure_cucount(tag: str, *, overwrite: bool, require_gpu: bool) -> None:
    if _validated_product(tag, "cucount") and not overwrite:
        print(f"[skip] validated {xi_engine_path(tag, 'cucount')}", flush=True)
        return
    import jax

    jax.config.update("jax_enable_x64", True)
    if require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"CuCount production requires GPU, found {jax.default_backend()}")
    from cucount.jax import BinAttrs, MeshAttrs, Particles, WeightAttrs
    from cucount.types import count2, count2_analytic

    position, position_metadata = _load_positions(tag)
    started = time.perf_counter()
    particles = Particles(position, weights=np.ones(position.shape[0], dtype="f8"), exchange=True)
    battrs = BinAttrs(s=np.asarray(S_EDGES, dtype="f8"))
    mattrs = MeshAttrs(
        particles,
        battrs=battrs,
        boxsize=np.full(3, BOX_SIZE, dtype="f8"),
        boxcenter=np.full(3, BOX_SIZE / 2.0, dtype="f8"),
        periodic=True,
    )
    dd = count2(particles, battrs=battrs, mattrs=mattrs, wattrs=WeightAttrs())["weight"]
    rr_count = count2_analytic(battrs, mattrs=mattrs)
    dd_raw = np.asarray(dd.values("counts"), dtype="f8")
    dd_norm = np.asarray(dd.values("norm"), dtype="f8")
    dd_normalized = np.asarray(dd.value(), dtype="f8")
    rr = np.asarray(rr_count.value(), dtype="f8")
    xi0 = dd_normalized / rr - 1.0
    jax.block_until_ready(xi0)
    elapsed = time.perf_counter() - started
    if not np.allclose(dd_norm, dd_norm[0], rtol=0.0, atol=0.0):
        raise RuntimeError("CuCount DD normalization is not constant across bins")
    save_engine_product(
        tag,
        "cucount",
        dd_raw=dd_raw,
        dd_norm_total=float(dd_norm[0]),
        dd_normalized=dd_normalized,
        rr=rr,
        xi0=xi0,
        s_pair_average=np.asarray(dd.coords("s"), dtype="f8"),
        measured_edges=S_EDGES,
        elapsed=elapsed,
        engine_metadata={
            "name": "cucount.jax.count2",
            "backend": jax.default_backend(),
            "jax_version": str(jax.__version__),
            "devices": [str(device) for device in jax.devices()],
            "position": position_metadata,
            "periodic": True,
        },
    )


def load_engine(tag: str, engine: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not _validated_product(tag, engine):
        raise RuntimeError(f"missing or invalid {tag}/{engine} product")
    metadata = json.loads(xi_engine_metadata_path(tag, engine).read_text(encoding="utf-8"))
    with np.load(xi_engine_path(tag, engine), allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    return metadata, arrays


def _old_galaxy_fcfc_path(tag: str) -> Path:
    return (
        BASE_OUTPUT_ROOT
        / "xi_engine_validation"
        / "fcfc"
        / f"task44_pngbase_{tag}_xi0_s30_350_ds10_fcfc.npz"
    )


def compare_and_plot(*, overwrite: bool) -> None:
    summary_path = SUMMARY_DIR / "task44_pngbase_hodhost_mmin1e13_xi_three_engine_audit.json"
    payload_path = SUMMARY_DIR / "task44_pngbase_hodhost_mmin1e13_xi_three_engine_audit.npz"
    engine_pdf = PLOT_ROOT / "task44_pngbase_hodhost_mmin1e13_xi_three_engine.pdf"
    sample_pdf = PLOT_ROOT / "task44_pngbase_hodhost_mmin1e13_vs_hodlrg_xi.pdf"
    outputs = (summary_path, payload_path, engine_pdf, sample_pdf)
    if all(path.is_file() for path in outputs) and not overwrite:
        print(f"[skip] validated comparison outputs already exist: {summary_path}", flush=True)
        return
    loaded: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    rows: dict[str, Any] = {}
    payload: dict[str, np.ndarray] = {"s": S_CENTERS, "s_edges": S_EDGES}
    all_agreement_gates: list[bool] = []
    for tag in CATALOGS:
        loaded[tag] = {}
        _, reference = load_engine(tag, "fcfc")
        engine_rows: dict[str, Any] = {}
        for engine in ENGINES:
            metadata, current = load_engine(tag, engine)
            loaded[tag][engine] = current
            delta_raw = np.asarray(current["dd_raw"], dtype="f8") - reference["dd_raw"]
            delta_norm = np.asarray(current["dd_normalized"], dtype="f8") - reference["dd_normalized"]
            delta_rr = np.asarray(current["rr_analytic"], dtype="f8") - reference["rr_analytic"]
            delta_xi = np.asarray(current["xi0"], dtype="f8") - reference["xi0"]
            gates = {
                "edges_bitwise_equal": bool(np.array_equal(current["s_edges"], reference["s_edges"])),
                "raw_DD_max_abs_at_most_8_ordered_pairs": float(np.max(np.abs(delta_raw))) <= 8.0,
                "normalized_DD_max_abs_below_1e_11": float(np.max(np.abs(delta_norm))) < 1.0e-11,
                "analytic_RR_max_abs_below_2e_15": float(np.max(np.abs(delta_rr))) < 2.0e-15,
                "xi0_max_abs_below_1e_8": float(np.max(np.abs(delta_xi))) < 1.0e-8,
            }
            all_agreement_gates.extend(gates.values())
            engine_rows[engine] = {
                "metadata": metadata,
                "max_abs_raw_DD_vs_fcfc": float(np.max(np.abs(delta_raw))),
                "max_abs_normalized_DD_vs_fcfc": float(np.max(np.abs(delta_norm))),
                "max_abs_analytic_RR_vs_fcfc": float(np.max(np.abs(delta_rr))),
                "max_abs_xi0_vs_fcfc": float(np.max(np.abs(delta_xi))),
                "rms_xi0_vs_fcfc": float(np.sqrt(np.mean(delta_xi**2))),
                "gates_vs_fcfc": gates,
            }
            payload[f"{tag}_{engine}_xi0"] = np.asarray(current["xi0"], dtype="f8")
            payload[f"{tag}_{engine}_dd_raw"] = np.asarray(current["dd_raw"], dtype="f8")
            payload[f"{tag}_{engine}_delta_xi0_vs_fcfc"] = delta_xi
        rows[tag] = {
            "fixed_baseline_fnl_for_later_pk_fit": get_spec(tag).fnl,
            "ndata": int(np.asarray(reference["ndata"]).item()),
            "engines": engine_rows,
        }

    global_gates = {
        "all_engine_measurement_contracts_pass": bool(
            all(
                all(all(row["engines"][engine]["metadata"]["contract"].values()) for engine in ENGINES)
                for row in rows.values()
            )
        ),
        "all_three_engines_agree_bin_by_bin": bool(all(all_agreement_gates)),
    }
    summary: dict[str, Any] = {
        "task": "task44_pngbase_hodhost_mmin1e13_xi_three_engine_audit",
        "status": "pass" if all(global_gates.values()) else "review",
        "conclusion": (
            "FCFC, pycorr/Corrfunc, and CuCount reproduce the same host-halo xi0 bin by bin"
            if all(global_gates.values())
            else "one or more three-engine agreement gates require review"
        ),
        "sample": "unit-weight unique occupied HOD hosts with official cleaned CompaSO M>=1e13 Msun/h",
        "geometry": {
            "type": "full periodic real-space cube",
            "boxsize_mpc_h": BOX_SIZE,
            "volume_mpc_h3": BOX_VOLUME,
            "s_edges_mpc_h": S_EDGES.tolist(),
        },
        "global_gates": global_gates,
        "catalogs": rows,
    }
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = engine_pdf.with_name(f".{engine_pdf.name}.{os.getpid()}.tmp.pdf")
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for axis, tag in zip(axes, CATALOGS, strict=True):
        for engine in ENGINES:
            kwargs: dict[str, Any] = {"color": COLORS[engine], "label": engine, "lw": 1.25}
            if MARKERS[engine] is not None:
                kwargs.update(marker=MARKERS[engine], ms=4.0, linestyle="none", markerfacecolor="none")
            axis.plot(S_CENTERS, S_CENTERS**2 * loaded[tag][engine]["xi0"], **kwargs)
        axis.set_title(tag, fontsize=14, fontweight="bold")
        axis.set_xlabel(r"$s\ [h^{-1}{\rm Mpc}]$")
        axis.text(
            0.96,
            0.96,
            r"$M_{\rm halo}\geq10^{13}\,h^{-1}M_\odot$" + "\n" + r"$30\leq s<350\,h^{-1}{\rm Mpc}$" + "\nperiodic real space",
            transform=axis.transAxes,
            va="top",
            ha="right",
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
    axes[0].set_ylabel(r"$s^2\xi_0(s)\ [(h^{-1}{\rm Mpc})^2]$")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.005))
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    figure.savefig(temporary)
    plt.close(figure)
    temporary.replace(engine_pdf)

    temporary = sample_pdf.with_name(f".{sample_pdf.name}.{os.getpid()}.tmp.pdf")
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for axis, tag in zip(axes, CATALOGS, strict=True):
        galaxy_path = _old_galaxy_fcfc_path(tag)
        if not galaxy_path.is_file():
            raise FileNotFoundError(galaxy_path)
        with np.load(galaxy_path, allow_pickle=False) as data:
            galaxy_edges = np.asarray(data["s_edges"], dtype="f8")
            galaxy_xi = np.asarray(data["xi0"], dtype="f8")
        if not np.array_equal(galaxy_edges, S_EDGES):
            raise RuntimeError(f"galaxy/halo xi edges differ for {tag}")
        halo_xi = np.asarray(loaded[tag]["fcfc"]["xi0"], dtype="f8")
        axis.plot(S_CENTERS, S_CENTERS**2 * halo_xi, color="#C44E52", lw=1.35, label="host halos")
        axis.plot(S_CENTERS, S_CENTERS**2 * galaxy_xi, color="#4C72B0", lw=1.2, label="HOD LRGs")
        axis.set_title(tag, fontsize=14, fontweight="bold")
        axis.set_xlabel(r"$s\ [h^{-1}{\rm Mpc}]$")
        axis.text(
            0.96,
            0.96,
            r"host: $M_{\rm halo}\geq10^{13}\,h^{-1}M_\odot$" + "\n" + r"$30\leq s<350\,h^{-1}{\rm Mpc}$" + "\nperiodic real space",
            transform=axis.transAxes,
            va="top",
            ha="right",
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
        payload[f"{tag}_hodlrg_fcfc_xi0"] = galaxy_xi
    axes[0].set_ylabel(r"$s^2\xi_0(s)\ [(h^{-1}{\rm Mpc})^2]$")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.005))
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    figure.savefig(temporary)
    plt.close(figure)
    temporary.replace(sample_pdf)

    atomic_savez(payload_path, **payload, summary_json=np.asarray(json.dumps(summary, sort_keys=True)))
    summary["payload_npz"] = str(payload_path)
    summary["payload_npz_sha256"] = sha256_file(payload_path)
    summary["engine_comparison_pdf"] = str(engine_pdf)
    summary["engine_comparison_pdf_sha256"] = sha256_file(engine_pdf)
    summary["halo_vs_hodlrg_pdf"] = str(sample_pdf)
    summary["halo_vs_hodlrg_pdf_sha256"] = sha256_file(sample_pdf)
    summary["pdf_only_gate"] = not any(PLOT_ROOT.rglob("*.png"))
    if not summary["pdf_only_gate"]:
        summary["status"] = "review"
    atomic_write_json(summary_path, summary)
    if summary["status"] != "pass":
        raise RuntimeError(f"three-engine comparison requires review: {summary['global_gates']}")
    print(f"[done] three-engine audit {summary_path}", flush=True)
    print(f"[done] PDFs {engine_pdf} {sample_pdf}", flush=True)


def _parse_tags(values: list[str] | None) -> list[str]:
    return list(CATALOGS) if not values else [get_spec(value).tag for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("fcfc", "pycorr", "cucount", "compare"))
    parser.add_argument("--tag", action="append", choices=tuple(CATALOGS))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--allow-cucount-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    tags = _parse_tags(args.tag)
    if args.mode == "compare":
        compare_and_plot(overwrite=bool(args.overwrite))
        return
    for tag in tags:
        if args.mode == "fcfc":
            measure_fcfc(tag, threads=int(args.threads), overwrite=bool(args.overwrite))
        elif args.mode == "pycorr":
            measure_pycorr(tag, threads=int(args.threads), overwrite=bool(args.overwrite))
        else:
            measure_cucount(tag, overwrite=bool(args.overwrite), require_gpu=not args.allow_cucount_cpu)


if __name__ == "__main__":
    main()
