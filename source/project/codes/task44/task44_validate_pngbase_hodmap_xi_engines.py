#!/usr/bin/env python3
"""Cross-check Task44 periodic-box xi0 with FCFC, pycorr, and CuCount.

All engines use the same positive-coordinate catalogs, L=2000 Mpc/h,
separation bins [30, 40), ..., [340, 350), and the exact periodic analytic
RR shell fraction.  Raw and normalized DD counts are retained so agreement is
tested below the correlation-function estimator level.
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
from matplotlib.backends.backend_pdf import PdfPages

from task44_pngbase_hodmap_rawbox_common import (
    BOX_SIZE,
    BOX_VOLUME,
    CATALOGS,
    FCFC_DIR,
    K_FUND,
    LOG_DIR,
    OUTPUT_ROOT,
    PLOT_ROOT,
    SUMMARY_DIR,
    ascii_catalog_path,
    ascii_metadata_path,
    atomic_savez,
    atomic_write_json,
    ensure_output_dirs,
    get_spec,
    load_positions,
    set_cpu_affinity,
    sha256_file,
)


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
FCFC_BINARY = PROJECT_ROOT / "refcode" / "FCFC-main" / "FCFC_2PT_BOX"
FCFC_READER = PROJECT_ROOT / "refcode" / "FCFC-main" / "scripts" / "read_pair_count.py"
ENGINE_ROOT = OUTPUT_ROOT / "xi_engine_validation"
S_EDGES = np.arange(30.0, 350.0 + 10.0, 10.0, dtype="f8")
S_CENTERS = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
ENGINES = ("fcfc", "pycorr", "cucount")
COLORS = {"fcfc": "black", "pycorr": "#4C72B0", "cucount": "#C44E52"}
MARKERS = {"fcfc": None, "pycorr": "o", "cucount": "x"}


def engine_path(tag: str, engine: str) -> Path:
    return ENGINE_ROOT / engine / f"task44_pngbase_{tag}_xi0_s30_350_ds10_{engine}.npz"


def engine_metadata_path(tag: str, engine: str) -> Path:
    return engine_path(tag, engine).with_suffix(".json")


def analytic_rr(edges: np.ndarray = S_EDGES) -> np.ndarray:
    edges = np.asarray(edges, dtype="f8")
    if float(edges[-1]) >= BOX_SIZE / 2.0:
        raise ValueError("analytic spherical-shell RR requires smax < L/2")
    return 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3) / BOX_VOLUME


def validated_catalog_hash(tag: str) -> str:
    path = SUMMARY_DIR / f"task44_pngbase_{tag}_hodmap_catalog_audit.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    spec = get_spec(tag)
    stat = Path(spec.path).stat()
    if payload.get("status") != "pass" or payload.get("path") != spec.path:
        raise RuntimeError(f"catalog audit does not validate {tag}: {path}")
    if payload.get("size_bytes") != stat.st_size or payload.get("mtime_ns") != stat.st_mtime_ns:
        raise RuntimeError(f"catalog changed since its SHA256 audit: {spec.path}")
    return str(payload["sha256"])


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
    output = engine_path(tag, engine)
    metadata_path = engine_metadata_path(tag, engine)
    expected_rr = analytic_rr()
    measured_edges = np.asarray(measured_edges, dtype="f8")
    reconstructed = np.asarray(dd_normalized, dtype="f8") / expected_rr - 1.0
    contract = {
        "s_edges_exact": bool(np.array_equal(measured_edges, S_EDGES)),
        "smax_below_half_box": bool(S_EDGES[-1] < BOX_SIZE / 2.0),
        "dd_raw_finite_nonnegative": bool(np.all(np.isfinite(dd_raw)) and np.all(np.asarray(dd_raw) >= 0.0)),
        "dd_normalized_finite_positive": bool(
            np.all(np.isfinite(dd_normalized)) and np.all(np.asarray(dd_normalized) > 0.0)
        ),
        "rr_matches_exact_shell_volume": bool(np.allclose(rr, expected_rr, rtol=0.0, atol=2.0e-15)),
        "xi_reconstructed_from_dd_rr": bool(np.allclose(xi0, reconstructed, rtol=0.0, atol=2.0e-13)),
        "normalization_is_ordered_auto_pairs": bool(
            np.isclose(dd_norm_total, get_spec(tag).expected_ngal * (get_spec(tag).expected_ngal - 1), rtol=0.0, atol=0.5)
        ),
    }
    if not all(contract.values()):
        raise RuntimeError(f"{tag}/{engine} measurement contract failed: {contract}")
    atomic_savez(
        output,
        s=np.asarray(S_CENTERS, dtype="f8"),
        s_pair_average=np.asarray(s_pair_average, dtype="f8"),
        s_edges=measured_edges,
        xi0=np.asarray(xi0, dtype="f8"),
        dd_raw=np.asarray(dd_raw, dtype="f8"),
        dd_norm_total=np.asarray(dd_norm_total, dtype="f8"),
        dd_normalized=np.asarray(dd_normalized, dtype="f8"),
        rr_analytic=np.asarray(expected_rr, dtype="f8"),
        tag=np.asarray(tag),
        engine=np.asarray(engine),
        ndata=np.asarray(get_spec(tag).expected_ngal, dtype="i8"),
        boxsize=np.asarray(BOX_SIZE, dtype="f8"),
        volume=np.asarray(BOX_VOLUME, dtype="f8"),
        kfund=np.asarray(K_FUND, dtype="f8"),
    )
    metadata = {
        "task": "task44_validate_pngbase_hodmap_xi_engines",
        "status": "pass",
        "tag": tag,
        "engine": engine,
        "source_hdf5": get_spec(tag).path,
        "source_hdf5_sha256": validated_catalog_hash(tag),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "geometry": "full periodic real-space cube",
        "boxsize_mpc_h": BOX_SIZE,
        "volume_mpc_h3": BOX_VOLUME,
        "kfund_h_mpc": K_FUND,
        "s_edges_mpc_h": S_EDGES.tolist(),
        "estimator": "DD_normalized / exact_analytic_RR - 1",
        "analytic_rr": "4*pi/3*(s_high^3-s_low^3)/L^3; valid because smax<L/2",
        "pair_normalization": "ordered auto pairs N*(N-1)",
        "contract": contract,
        "engine_metadata": engine_metadata,
        "elapsed_sec": elapsed,
    }
    atomic_write_json(metadata_path, metadata)
    print(f"[done] {tag} {engine} bins={xi0.size} elapsed={elapsed:.2f}s path={output}", flush=True)


def load_fcfc_pair_count(path: Path) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    spec = importlib.util.spec_from_file_location("task44_fcfc_pair_reader", FCFC_READER)
    if spec is None or spec.loader is None:
        raise ImportError(FCFC_READER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.read_pair_count(str(path), verbose=False)
    bin_type, nsbin, nmbin, npbin, edges, pi_edges, num1, num2, norm, normalized, raw = result
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
    output = engine_path(tag, "fcfc")
    metadata_path = engine_metadata_path(tag, "fcfc")
    if output.is_file() and metadata_path.is_file() and not overwrite:
        print(f"[skip] {output}", flush=True)
        return
    if not FCFC_BINARY.is_file() or not os.access(FCFC_BINARY, os.X_OK):
        raise FileNotFoundError(FCFC_BINARY)
    ascii_path = ascii_catalog_path(tag)
    ascii_meta_path = ascii_metadata_path(tag)
    if not ascii_path.is_file() or not ascii_meta_path.is_file():
        raise FileNotFoundError(f"missing validated ASCII bridge: {ascii_path}")
    ascii_meta = json.loads(ascii_meta_path.read_text(encoding="utf-8"))
    if ascii_meta.get("output_sha256") != sha256_file(ascii_path):
        raise RuntimeError(f"ASCII hash mismatch: {ascii_path}")
    cpus = set_cpu_affinity(threads)
    work = ENGINE_ROOT / "fcfc_work"
    work.mkdir(parents=True, exist_ok=True)
    config = work / f"task44_pngbase_{tag}_periodic_xi0_s30_350_ds10.conf"
    pair = work / f"task44_pngbase_{tag}_periodic_DD_s30_350_ds10.bin"
    text = work / f"task44_pngbase_{tag}_periodic_xi0_s30_350_ds10.txt"
    log = LOG_DIR / f"task44_pngbase_{tag}_periodic_xi0_s30_350_ds10_fcfc_validation.log"
    config_text = f"""CATALOG = '{ascii_path}'
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
CF_OUTPUT_FILE = '{text}'
SEP_BIN_MIN = {S_EDGES[0]:.17g}
SEP_BIN_MAX = {S_EDGES[-1]:.17g}
SEP_BIN_SIZE = {S_EDGES[1] - S_EDGES[0]:.17g}
OUTPUT_FORMAT = 0
OVERWRITE = {2 if overwrite else 1}
VERBOSE = T
"""
    config.write_text(config_text, encoding="utf-8")
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
    table = np.loadtxt(text, comments="#", dtype="f8")
    if table.shape != (S_EDGES.size - 1, 4):
        raise ValueError(f"unexpected FCFC xi table shape: {table.shape}")
    s, lower, upper, xi0_text = table.T
    if not np.allclose(np.column_stack([lower, upper]), np.column_stack([S_EDGES[:-1], S_EDGES[1:]]), rtol=0.0, atol=1.0e-13):
        raise RuntimeError("FCFC xi bin edges differ from the frozen edges")
    dd_raw, dd_normalized, norm, pair_edges = load_fcfc_pair_count(pair)
    rr_exact = analytic_rr(pair_edges)
    xi0 = dd_normalized / rr_exact - 1.0
    text_roundoff = float(np.max(np.abs(np.asarray(xi0_text, dtype="f8") - xi0)))
    save_engine_product(
        tag,
        "fcfc",
        dd_raw=dd_raw,
        dd_norm_total=norm,
        dd_normalized=dd_normalized,
        rr=rr_exact,
        xi0=np.asarray(xi0, dtype="f8"),
        s_pair_average=np.asarray(s, dtype="f8"),
        measured_edges=pair_edges,
        elapsed=elapsed,
        engine_metadata={
            "name": "FCFC_2PT_BOX",
            "version": "1.0.1",
            "binary": str(FCFC_BINARY),
            "configuration": str(config),
            "pair_count": str(pair),
            "xi_text": str(text),
            "xi_saved_from": "raw binary DD / exact analytic RR - 1",
            "fcfc_text_xi_max_abs_vs_raw_reconstruction": text_roundoff,
            "log": str(log),
            "ascii_catalog": str(ascii_path),
            "ascii_catalog_sha256": ascii_meta["output_sha256"],
            "threads": int(threads),
            "cpu_affinity": cpus,
        },
    )


def measure_pycorr(tag: str, *, threads: int, overwrite: bool) -> None:
    output = engine_path(tag, "pycorr")
    metadata_path = engine_metadata_path(tag, "pycorr")
    if output.is_file() and metadata_path.is_file() and not overwrite:
        print(f"[skip] {output}", flush=True)
        return
    cpus = set_cpu_affinity(threads)
    from pycorr import TwoPointCorrelationFunction, __version__ as pycorr_version
    import Corrfunc

    started = time.perf_counter()
    position, position_meta = load_positions(tag, origin="positive")
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
    rr = np.asarray(correlation.R1R2.normalized_wcounts(), dtype="f8")
    xi0 = np.asarray(correlation.corr, dtype="f8")
    s_pair_average = np.asarray(correlation.sep, dtype="f8")
    save_engine_product(
        tag,
        "pycorr",
        dd_raw=dd_raw,
        dd_norm_total=dd_norm_total,
        dd_normalized=dd_normalized,
        rr=rr,
        xi0=xi0,
        s_pair_average=s_pair_average,
        measured_edges=np.asarray(correlation.D1D2.edges[0], dtype="f8"),
        elapsed=elapsed,
        engine_metadata={
            "name": "pycorr.TwoPointCorrelationFunction",
            "version": str(pycorr_version),
            "counter": "Corrfunc.theory.DD",
            "corrfunc_version": str(Corrfunc.__version__),
            "position": position_meta,
            "threads": int(threads),
            "cpu_affinity": cpus,
            "bin_convention": "low edge inclusive, high edge exclusive",
        },
    )


def measure_cucount(tag: str, *, overwrite: bool, require_gpu: bool) -> None:
    output = engine_path(tag, "cucount")
    metadata_path = engine_metadata_path(tag, "cucount")
    if output.is_file() and metadata_path.is_file() and not overwrite:
        print(f"[skip] {output}", flush=True)
        return
    import jax

    jax.config.update("jax_enable_x64", True)
    if require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"CuCount production requires a GPU backend, found {jax.default_backend()}")
    from cucount.jax import BinAttrs, MeshAttrs, Particles, WeightAttrs
    from cucount.types import count2, count2_analytic

    started = time.perf_counter()
    position, position_meta = load_positions(tag, origin="positive")
    particles = Particles(position, weights=np.ones(position.shape[0], dtype="f8"), exchange=True)
    battrs = BinAttrs(s=np.asarray(S_EDGES, dtype="f8"))
    mattrs = MeshAttrs(
        particles,
        battrs=battrs,
        boxsize=np.full(3, BOX_SIZE, dtype="f8"),
        boxcenter=np.full(3, BOX_SIZE / 2.0, dtype="f8"),
        periodic=True,
    )
    wattrs = WeightAttrs()
    dd = count2(particles, battrs=battrs, mattrs=mattrs, wattrs=wattrs)["weight"]
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
        measured_edges=np.asarray(S_EDGES, dtype="f8"),
        elapsed=elapsed,
        engine_metadata={
            "name": "cucount.jax.count2",
            "backend": jax.default_backend(),
            "jax_version": str(jax.__version__),
            "devices": [str(device) for device in jax.devices()],
            "position": position_meta,
            "periodic": True,
            "ordered_auto_pair_normalization": True,
        },
    )


def load_engine(tag: str, engine: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = engine_path(tag, engine)
    metadata_path = engine_metadata_path(tag, engine)
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing {tag}/{engine}: {path} / {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "pass" or metadata.get("output_sha256") != sha256_file(path):
        raise RuntimeError(f"invalid metadata/hash for {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    return metadata, arrays


def compare_and_plot(*, overwrite: bool) -> None:
    summary_path = SUMMARY_DIR / "task44_pngbase_hodmap_xi0_three_engine_s30_350_audit.json"
    payload_path = SUMMARY_DIR / "task44_pngbase_hodmap_xi0_three_engine_s30_350_audit.npz"
    pdf_path = PLOT_ROOT / "task44_pngbase_hodmap_xi0_three_engine_s30_350_comparison.pdf"
    if all(path.is_file() for path in (summary_path, payload_path, pdf_path)) and not overwrite:
        print(f"[skip] {summary_path}", flush=True)
        return
    rows: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    all_gates: list[bool] = []
    loaded: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for tag in CATALOGS:
        loaded[tag] = {}
        reference_meta, reference = load_engine(tag, "fcfc")
        loaded[tag]["fcfc"] = reference
        engine_rows: dict[str, Any] = {}
        for engine in ENGINES:
            metadata, current = load_engine(tag, engine)
            loaded[tag][engine] = current
            delta_raw = np.asarray(current["dd_raw"], dtype="f8") - np.asarray(reference["dd_raw"], dtype="f8")
            delta_norm = np.asarray(current["dd_normalized"], dtype="f8") - np.asarray(reference["dd_normalized"], dtype="f8")
            delta_rr = np.asarray(current["rr_analytic"], dtype="f8") - np.asarray(reference["rr_analytic"], dtype="f8")
            delta_xi = np.asarray(current["xi0"], dtype="f8") - np.asarray(reference["xi0"], dtype="f8")
            gates = {
                "edges_bitwise_equal": bool(np.array_equal(current["s_edges"], reference["s_edges"])),
                # pycorr/CuCount agree exactly with one another; FCFC moves at most
                # one physical pair (two ordered counts) across an exact bin edge.
                "raw_DD_max_abs_at_most_2_ordered_pairs": float(np.max(np.abs(delta_raw))) <= 2.0,
                "normalized_DD_max_abs_below_2e_13": float(np.max(np.abs(delta_norm))) < 2.0e-13,
                "analytic_RR_max_abs_below_2e_15": float(np.max(np.abs(delta_rr))) < 2.0e-15,
                "xi0_max_abs_below_2e_10": float(np.max(np.abs(delta_xi))) < 2.0e-10,
            }
            all_gates.extend(gates.values())
            engine_rows[engine] = {
                "metadata": metadata,
                "max_abs_raw_DD_pair": float(np.max(np.abs(delta_raw))),
                "max_abs_normalized_DD": float(np.max(np.abs(delta_norm))),
                "max_abs_analytic_RR": float(np.max(np.abs(delta_rr))),
                "max_abs_xi0": float(np.max(np.abs(delta_xi))),
                "rms_xi0": float(np.sqrt(np.mean(delta_xi**2))),
                "gates_vs_fcfc": gates,
            }
            arrays[f"{tag}_{engine}_xi0"] = np.asarray(current["xi0"], dtype="f8")
            arrays[f"{tag}_{engine}_dd_raw"] = np.asarray(current["dd_raw"], dtype="f8")
            arrays[f"{tag}_{engine}_dd_normalized"] = np.asarray(current["dd_normalized"], dtype="f8")
            arrays[f"{tag}_{engine}_rr_analytic"] = np.asarray(current["rr_analytic"], dtype="f8")
            arrays[f"{tag}_{engine}_delta_xi0_vs_fcfc"] = np.asarray(delta_xi, dtype="f8")
        rows[tag] = {
            "fixed_fnl": get_spec(tag).fnl,
            "engines": engine_rows,
            "oscillatory_curve_reproduced_by_all_engines": bool(
                all(engine_rows[engine]["gates_vs_fcfc"]["xi0_max_abs_below_2e_10"] for engine in ENGINES)
            ),
        }
    global_gates = {
        "all_engine_measurement_contracts_pass": bool(
            all(
                all(
                    all(row["engines"][engine]["metadata"]["contract"].values())
                    for engine in ENGINES
                )
                for row in rows.values()
            )
        ),
        "all_three_engines_agree_bin_by_bin": bool(all(all_gates)),
        "same_curve_reproduced_for_both_catalogs": bool(
            all(row["oscillatory_curve_reproduced_by_all_engines"] for row in rows.values())
        ),
    }
    summary = {
        "task": "task44_pngbase_hodmap_xi0_three_engine_s30_350_audit",
        "status": "pass" if all(global_gates.values()) else "review",
        "conclusion": (
            "FCFC, pycorr/Corrfunc, and CuCount reproduce the same bin-by-bin xi0; the oscillatory curve is not an FCFC implementation artifact"
            if all(global_gates.values())
            else "the three pair-count engines do not yet pass the frozen agreement thresholds"
        ),
        "geometry": {
            "type": "full periodic real-space cube",
            "boxsize_mpc_h": BOX_SIZE,
            "volume_mpc_h3": BOX_VOLUME,
            "kfund_h_mpc": K_FUND,
            "s_edges_mpc_h": S_EDGES.tolist(),
        },
        "estimator": "normalized ordered DD / exact analytic spherical-shell RR - 1",
        "global_gates": global_gates,
        "catalogs": rows,
    }
    arrays["s"] = np.asarray(S_CENTERS, dtype="f8")
    arrays["s_edges"] = np.asarray(S_EDGES, dtype="f8")
    atomic_savez(payload_path, **arrays, summary_json=np.asarray(json.dumps(summary, sort_keys=True)))
    summary["payload_npz"] = str(payload_path)
    summary["payload_npz_sha256"] = sha256_file(payload_path)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pdf_path.with_name(f".{pdf_path.name}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary) as pdf:
        for tag in CATALOGS:
            figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True, gridspec_kw={"height_ratios": [2.3, 1.0]})
            s = S_CENTERS
            for engine in ENGINES:
                xi = loaded[tag][engine]["xi0"]
                kwargs: dict[str, Any] = {"color": COLORS[engine], "label": engine, "lw": 1.2}
                if MARKERS[engine] is not None:
                    kwargs.update(marker=MARKERS[engine], ms=4.0, linestyle="none", markerfacecolor="none")
                axes[0].plot(s, s**2 * xi, **kwargs)
            axes[0].set_ylabel(r"$s^2\xi_0(s)\ [(h^{-1}{\rm Mpc})^2]$")
            axes[0].legend(frameon=False, ncol=3)
            axes[0].set_title(rf"{tag}, fixed baseline $f_{{\rm NL}}={get_spec(tag).fnl:g}$; identical catalog and periodic geometry")
            reference = loaded[tag]["fcfc"]["xi0"]
            for engine in ("pycorr", "cucount"):
                delta = loaded[tag][engine]["xi0"] - reference
                axes[1].plot(s, delta, marker=MARKERS[engine], ms=4.0, lw=0.9, color=COLORS[engine], label=f"{engine} - FCFC")
            axes[1].axhline(0.0, color="0.5", lw=0.8)
            axes[1].set_xlabel(r"$s\ [h^{-1}{\rm Mpc}]$")
            axes[1].set_ylabel(r"$\Delta\xi_0$")
            axes[1].legend(frameon=False)
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
        for axis, tag in zip(axes, CATALOGS, strict=True):
            rr = loaded[tag]["fcfc"]["rr_analytic"]
            for engine in ENGINES:
                ratio = loaded[tag][engine]["dd_normalized"] / rr
                axis.plot(S_CENTERS, ratio - 1.0, color=COLORS[engine], lw=1.1, label=engine)
            axis.axhline(0.0, color="0.5", lw=0.8)
            axis.set_title(f"{tag}: normalized DD / analytic RR - 1")
            axis.set_xlabel(r"$s\ [h^{-1}{\rm Mpc}]$")
        axes[0].set_ylabel(r"$\xi_0(s)$")
        axes[1].legend(frameon=False)
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)
    temporary.replace(pdf_path)
    summary["plot_pdf"] = str(pdf_path)
    summary["plot_pdf_sha256"] = sha256_file(pdf_path)
    atomic_write_json(summary_path, summary)
    print(f"[done] three-engine audit {summary_path} status={summary['status']}", flush=True)
    print(f"[done] PDF {pdf_path}", flush=True)


def parse_tags(values: list[str] | None) -> list[str]:
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
    ENGINE_ROOT.mkdir(parents=True, exist_ok=True)
    tags = parse_tags(args.tag)
    if args.mode == "compare":
        compare_and_plot(overwrite=bool(args.overwrite))
        return
    for tag in tags:
        if args.mode == "fcfc":
            measure_fcfc(tag, threads=int(args.threads), overwrite=bool(args.overwrite))
        elif args.mode == "pycorr":
            measure_pycorr(tag, threads=int(args.threads), overwrite=bool(args.overwrite))
        elif args.mode == "cucount":
            measure_cucount(tag, overwrite=bool(args.overwrite), require_gpu=not bool(args.allow_cucount_cpu))


if __name__ == "__main__":
    main()
