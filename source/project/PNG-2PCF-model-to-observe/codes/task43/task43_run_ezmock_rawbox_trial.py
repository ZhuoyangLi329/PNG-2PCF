#!/usr/bin/env python3
"""Run one five-realization fixed-amplitude EZmock calibration trial."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from task43_rawbox_ezmock_common import atomic_savez, set_cpu_affinity, write_json


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/task43_outputs/ezmock_calibration_rawbox_z0p725_mmin1p4e13"
)
TARGET = (
    PROJECT_ROOT
    / "outputs/task43_outputs/ezmock_rawbox_z0p725_mmin1p4e13/summary"
    / "task43_ezmock_rawbox_z0p725_mmin1p4e13_x25.npz"
)
LINEAR_PK = (
    OUTPUT_ROOT
    / "linear_pk/abacus_c000_linear_matter_pk_z0p725_desilike_cosmoprimo.dat"
)
# 旧 notebook 的历史输入曾被保存到 plots/outputs；当前布局缺失时只读回退。
LEGACY_OUTPUT_ROOT = PROJECT_ROOT / "plots/outputs/task43_outputs"
if not TARGET.exists():
    TARGET = (
        LEGACY_OUTPUT_ROOT
        / "ezmock_rawbox_z0p725_mmin1p4e13/summary"
        / "task43_ezmock_rawbox_z0p725_mmin1p4e13_x25.npz"
    )
if not LINEAR_PK.exists():
    LINEAR_PK = (
        LEGACY_OUTPUT_ROOT
        / "ezmock_calibration_rawbox_z0p725_mmin1p4e13/linear_pk"
        / "abacus_c000_linear_matter_pk_z0p725_desilike_cosmoprimo.dat"
    )
EZMOCK_BINARY = PROJECT_ROOT / "EZmock/EZmock-1.0.0/EZmock"
FCFC_BINARY = PROJECT_ROOT / "refcode/FCFC-main/FCFC_2PT_BOX"
PK_SCRIPT = PROJECT_ROOT / "codes/task43/task43_measure_ezmock_trial_pk_jaxpower.py"
BOX_SIZE = 2000.0
REDSHIFT = 0.725
DEFAULT_NREAL = 5
OMEGA_M_NON_NEUTRINO = 0.3137721026737606
OMEGA_NU = 0.0014197664745152646


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="trial001_c1p00_e0p60_b0p22_v200")
    parser.add_argument("--rho-c", type=float, default=1.0)
    parser.add_argument("--rho-exp", type=float, default=0.6)
    parser.add_argument("--pdf-base", type=float, default=0.22)
    parser.add_argument("--sigma-v", type=float, default=200.0)
    parser.add_argument("--ngrid", type=int, default=256)
    parser.add_argument("--ntracer", type=int, default=None)
    parser.add_argument("--meshsize", type=int, default=400)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=430100)
    parser.add_argument("--nreal", type=int, default=DEFAULT_NREAL)
    parser.add_argument("--rand-generator", type=int, choices=(0, 1), default=1)
    parser.add_argument("--pk-interp-log", choices=("T", "F"), default="T")
    parser.add_argument("--invert-phase", choices=("T", "F"), default="F")
    parser.add_argument("--bao-enhance", type=float, default=0.0)
    parser.add_argument("--attach-particle", choices=("T", "F"), default="F")
    parser.add_argument(
        "--classification",
        choices=("pipeline_smoke", "manual_calibration"),
        default="pipeline_smoke",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.label):
        raise ValueError("--label contains unsafe characters")
    if args.rho_c < 0.0 or args.rho_exp <= 0.0:
        raise ValueError("invalid rho_c or rho_exp")
    if not 0.0 < args.pdf_base < 1.0 or args.sigma_v < 0.0:
        raise ValueError("invalid pdf_base or sigma_v")
    if args.ngrid <= 1 or args.ngrid % 2:
        raise ValueError("--ngrid must be an even integer larger than one")
    if not 1 <= args.threads <= 8:
        raise ValueError("--threads must be in [1, 8]")
    if not 1 <= args.nreal <= 25:
        raise ValueError("--nreal must be in [1, 25]")
    if args.ntracer is not None and not 1 <= args.ntracer <= args.ngrid**3:
        raise ValueError("--ntracer must be in [1, NUM_GRID^3]")
    if not TARGET.exists() or not LINEAR_PK.exists():
        raise FileNotFoundError(f"missing target or linear P(k): {TARGET}, {LINEAR_PK}")
    for path in (EZMOCK_BINARY, FCFC_BINARY, PK_SCRIPT):
        if not path.exists():
            raise FileNotFoundError(path)


def run_logged(command: list[str], log_path: Path, *, env: dict[str, str]) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
    return time.perf_counter() - started


def count_catalog_rows(path: Path) -> int:
    """Count non-comment EZmock ASCII rows without loading the full catalog."""
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            stripped = line.lstrip()
            if stripped and not stripped.startswith(b"#"):
                count += 1
    if count <= 0:
        raise RuntimeError(f"empty EZmock catalog: {path}")
    return count


def ezmock_config(
    args: argparse.Namespace,
    *,
    seed: int,
    ntracer: int,
    catalog: Path,
) -> str:
    return f"""BOX_SIZE = {BOX_SIZE:.17g}
NUM_GRID = {int(args.ngrid)}
NUM_TRACER = {int(ntracer)}
LINEAR_PK = '{LINEAR_PK}'
REDSHIFT_PK = {REDSHIFT:.17g}
PK_INTERP_LOG = {args.pk_interp_log}
RAND_GENERATOR = {int(args.rand_generator)}
RAND_SEED = {int(seed)}
FIX_AMPLITUDE = T
INVERT_PHASE = {args.invert_phase}
OMEGA_M = {OMEGA_M_NON_NEUTRINO:.17g}
OMEGA_NU = {OMEGA_NU:.17g}
DE_EOS_W = -1
REDSHIFT = {REDSHIFT:.17g}
BAO_ENHANCE = {float(args.bao_enhance):.17g}
RHO_CRITICAL = {float(args.rho_c):.17g}
RHO_EXP = {float(args.rho_exp):.17g}
PDF_BASE = {float(args.pdf_base):.17g}
SIGMA_VELOCITY = {float(args.sigma_v):.17g}
ATTACH_PARTICLE = {args.attach_particle}
OUTPUT = '{catalog}'
OUTPUT_FORMAT = 0
OUTPUT_HEADER = T
OVERWRITE = {2 if args.overwrite else 0}
VERBOSE = T
"""


def fcfc_config(
    *,
    catalog: Path,
    pair_path: Path,
    output_text: Path,
    overwrite: bool,
) -> str:
    return f"""CATALOG = '{catalog}'
CATALOG_LABEL = D
CATALOG_TYPE = 0
ASCII_SKIP = 0
ASCII_COMMENT = '#'
ASCII_FORMATTER = '%lf %lf %lf %lf %lf %lf'
POSITION = [$1, $2, $3]
BOX_SIZE = {BOX_SIZE:.17g}
DATA_STRUCT = 0
BINNING_SCHEME = 0
PAIR_COUNT = DD
PAIR_COUNT_FILE = '{pair_path}'
CF_ESTIMATOR = DD / @@ - 1
CF_OUTPUT_FILE = '{output_text}'
SEP_BIN_MIN = 50
SEP_BIN_MAX = 550
SEP_BIN_SIZE = 10
OVERWRITE = {2 if overwrite else 1}
VERBOSE = T
"""


def load_target_ntracer() -> int:
    with np.load(TARGET, allow_pickle=False) as data:
        ndata = np.asarray(data["ndata"], dtype="i8")
    if ndata.shape != (25,) or np.any(ndata <= 0):
        raise ValueError("unexpected Abacus target ndata array")
    return int(np.rint(np.mean(ndata)))


def save_xi(
    *,
    text_path: Path,
    output: Path,
    seed: int,
    ntracer: int,
    elapsed: float,
    config_path: Path,
    pair_path: Path,
    catalog: Path,
    threads: int,
    cpus: list[int],
) -> None:
    table = np.loadtxt(text_path, comments="#", dtype="f8")
    if table.shape != (50, 4):
        raise ValueError(f"unexpected FCFC output shape: {table.shape}")
    s, s_lower, s_upper, xi0 = table.T
    if not np.all(np.isfinite(xi0)):
        raise ValueError("FCFC returned non-finite xi0")
    s_edges = np.concatenate([s_lower[:1], s_upper])
    atomic_savez(
        output,
        s=s,
        s_edges=s_edges,
        xi0=xi0,
        seed=np.asarray(seed, dtype="i8"),
        ndata=np.asarray(ntracer, dtype="i8"),
        nbar=np.asarray(ntracer / BOX_SIZE**3, dtype="f8"),
        boxsize=np.asarray(BOX_SIZE, dtype="f8"),
    )
    write_json(
        output.with_suffix(".json"),
        {
            "task": "task43_run_ezmock_rawbox_trial.fcfc",
            "status": "done",
            "seed": int(seed),
            "catalog": str(catalog),
            "output": str(output),
            "configuration": str(config_path),
            "pair_count": str(pair_path),
            "engine": "FCFC_2PT_BOX v1.0.1 OpenMP",
            "estimator": "DD / analytic_RR - 1",
            "boxsize": BOX_SIZE,
            "smin": 50.0,
            "smax": 550.0,
            "ds": 10.0,
            "threads_requested": int(threads),
            "cpu_affinity": cpus,
            "elapsed_sec": float(elapsed),
        },
    )


def summarize(
    args: argparse.Namespace,
    *,
    trial_root: Path,
    seeds: list[int],
    ntracer: int,
    runtime_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    xi_rows, pk_rows, nmodes_rows, ndata_rows = [], [], [], []
    s = s_edges = k = k_edges = None
    for seed in seeds:
        xi_path = trial_root / f"xi_fcfc/xi0_seed{seed}_s50_550_ds10_fcfc.npz"
        pk_path = trial_root / f"pk_jaxpower/pk0_seed{seed}_mesh{args.meshsize}.npz"
        with np.load(xi_path, allow_pickle=False) as data:
            this_s = np.asarray(data["s"], dtype="f8")
            this_s_edges = np.asarray(data["s_edges"], dtype="f8")
            if s is None:
                s, s_edges = this_s, this_s_edges
            elif not np.array_equal(s, this_s) or not np.array_equal(s_edges, this_s_edges):
                raise ValueError("inconsistent FCFC grids")
            xi_rows.append(np.asarray(data["xi0"], dtype="f8"))
        with np.load(pk_path, allow_pickle=False) as data:
            this_k = np.asarray(data["k"], dtype="f8")
            this_k_edges = np.asarray(data["k_edges"], dtype="f8")
            if k is None:
                k, k_edges = this_k, this_k_edges
            elif not np.array_equal(k, this_k, equal_nan=True) or not np.array_equal(k_edges, this_k_edges, equal_nan=True):
                raise ValueError("inconsistent jaxpower grids")
            pk_rows.append(np.asarray(data["pk0"], dtype="f8"))
            nmodes_rows.append(np.asarray(data["nmodes"], dtype="f8"))
            ndata_rows.append(int(np.asarray(data["ndata"]).item()))

    xi_all = np.vstack(xi_rows)
    pk_all = np.vstack(pk_rows)
    nmodes_all = np.vstack(nmodes_rows)
    valid_k = np.all(nmodes_all > 0.0, axis=0) & np.all(np.isfinite(pk_all), axis=0)
    if xi_all.shape[0] != args.nreal or np.count_nonzero(valid_k) == 0:
        raise RuntimeError(f"trial does not contain exactly {args.nreal} complete realizations")

    with np.load(TARGET, allow_pickle=False) as target:
        target_s = np.asarray(target["s"], dtype="f8")
        target_xi = np.asarray(target["xi0_mean"], dtype="f8")
        target_xi_std = np.asarray(target["xi0_std"], dtype="f8")
        target_k = np.asarray(target["k"], dtype="f8")
        target_pk = np.asarray(target["pk0_mean"], dtype="f8")
    if not np.array_equal(s, target_s) or not np.array_equal(k, target_k, equal_nan=True):
        raise ValueError("EZmock and Abacus target grids differ")

    xi_mean = np.mean(xi_all, axis=0)
    pk_mean = np.full(pk_all.shape[1], np.nan, dtype="f8")
    pk_mean[valid_k] = np.mean(pk_all[:, valid_k], axis=0)
    xi_mask = (s >= 50.0) & (s <= 200.0) & (target_xi_std > 0.0)
    pk_mask = valid_k & (k >= 0.01) & (k <= 0.10) & np.isfinite(target_pk) & (target_pk > 0.0)
    metrics = {
        "xi_delta_over_abacus_single_box_scatter_rms_s50_200": float(
            np.sqrt(np.mean(((xi_mean[xi_mask] - target_xi[xi_mask]) / target_xi_std[xi_mask]) ** 2))
        ),
        "xi_absolute_delta_rms_s50_200": float(np.sqrt(np.mean((xi_mean[xi_mask] - target_xi[xi_mask]) ** 2))),
        "pk_fractional_delta_rms_k0p01_0p10": float(
            np.sqrt(np.mean(((pk_mean[pk_mask] - target_pk[pk_mask]) / target_pk[pk_mask]) ** 2))
        ),
        "pk_fractional_delta_mean_k0p01_0p10": float(
            np.mean((pk_mean[pk_mask] - target_pk[pk_mask]) / target_pk[pk_mask])
        ),
    }

    summary_dir = trial_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    output_npz = summary_dir / f"{args.label}_x{args.nreal}_fixedamp_mean_2pcf_pk.npz"
    output_json = output_npz.with_suffix(".json")
    atomic_savez(
        output_npz,
        seeds=np.asarray(seeds, dtype="i8"),
        s=s,
        s_edges=s_edges,
        xi0_all=xi_all,
        xi0_mean=xi_mean,
        xi0_std=(np.std(xi_all, axis=0, ddof=1) if args.nreal > 1 else np.zeros(xi_all.shape[1], dtype="f8")),
        k=k,
        k_edges=k_edges,
        k_valid_mask=valid_k,
        pk0_all=pk_all,
        nmodes_all=nmodes_all,
        pk0_mean=pk_mean,
        pk0_std=np.where(
            valid_k,
            np.std(pk_all, axis=0, ddof=1) if args.nreal > 1 else np.zeros(pk_all.shape[1], dtype="f8"),
            np.nan,
        ),
        ndata=np.asarray(ndata_rows, dtype="i8"),
        nbar=np.asarray(ndata_rows, dtype="f8") / BOX_SIZE**3,
        parameter_names=np.asarray(["rho_c", "rho_exp", "pdf_base", "sigma_v"]),
        parameter_values=np.asarray([args.rho_c, args.rho_exp, args.pdf_base, args.sigma_v], dtype="f8"),
        fix_amplitude=np.asarray(True),
        ngrid=np.asarray(args.ngrid, dtype="i8"),
        meshsize=np.asarray(args.meshsize, dtype="i8"),
        redshift=np.asarray(REDSHIFT, dtype="f8"),
    )
    payload = {
        "task": "task43_run_ezmock_rawbox_trial",
        "status": "done",
        "label": args.label,
        "purpose": (
            f"pipeline smoke test with {args.nreal} fixed-amplitude EZmock realization(s); parameters are not a fitted or physically preferred point"
            if args.classification == "pipeline_smoke"
            else f"manual EZmock parameter calibration with {args.nreal} fixed-amplitude realization(s)"
        ),
        "trial_classification": str(args.classification),
        "fixed_amplitude": True,
        "covariance_requirement": "FIX_AMPLITUDE must be false for production covariance mocks",
        "nreal": int(args.nreal),
        "seeds": seeds,
        "boxsize": BOX_SIZE,
        "redshift": REDSHIFT,
        "ngrid": int(args.ngrid),
        "meshsize": int(args.meshsize),
        "ntracer_requested": int(ntracer),
        "ndata_actual": ndata_rows,
        "parameters": {
            "rho_c": float(args.rho_c),
            "rho_exp": float(args.rho_exp),
            "pdf_base": float(args.pdf_base),
            "sigma_v": float(args.sigma_v),
        },
        "generation_options": {
            "rand_generator": int(args.rand_generator),
            "pk_interp_log": str(args.pk_interp_log),
            "invert_phase": str(args.invert_phase),
            "bao_enhance": float(args.bao_enhance),
            "attach_particle": str(args.attach_particle),
        },
        "linear_pk": str(LINEAR_PK),
        "linear_pk_redshift": REDSHIFT,
        "linear_pk_cosmology": "AbacusSummit_base_c000",
        "target": str(TARGET),
        "output_npz": str(output_npz),
        "runtime_rows": runtime_rows,
        "metrics": metrics,
    }
    write_json(output_json, payload)
    print(f"[done] summary={output_npz}", flush=True)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return output_npz, output_json


def main() -> None:
    args = parse_args()
    validate_args(args)
    cpus = set_cpu_affinity(args.threads)
    ntracer = load_target_ntracer() if args.ntracer is None else int(args.ntracer)
    seeds = [int(args.seed_base) + index for index in range(1, args.nreal + 1)]
    trial_root = OUTPUT_ROOT / args.label
    for dirname in ("catalogs", "ezmock_configs", "fcfc_configs", "fcfc_pairs", "xi_fcfc", "pk_jaxpower", "logs", "summary"):
        (trial_root / dirname).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(args.threads),
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "close",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "JAX_PLATFORMS": "cpu",
            "JAX_PLATFORM_NAME": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    runtime_rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        print(f"[trial] realization {index}/{args.nreal} seed={seed}", flush=True)
        catalog = trial_root / f"catalogs/ezmock_seed{seed}.dat"
        ez_conf = trial_root / f"ezmock_configs/ezmock_seed{seed}.conf"
        ez_log = trial_root / f"logs/ezmock_seed{seed}.log"
        xi_conf = trial_root / f"fcfc_configs/fcfc_seed{seed}_s50_550_ds10.conf"
        pair_path = trial_root / f"fcfc_pairs/DD_seed{seed}_s50_550_ds10.bin"
        xi_text = trial_root / f"xi_fcfc/xi0_seed{seed}_s50_550_ds10_fcfc.txt"
        xi_output = xi_text.with_suffix(".npz")
        fcfc_log = trial_root / f"logs/fcfc_seed{seed}.log"
        pk_output = trial_root / f"pk_jaxpower/pk0_seed{seed}_mesh{args.meshsize}.npz"
        pk_log = trial_root / f"logs/jaxpower_seed{seed}.log"

        ez_conf.write_text(ezmock_config(args, seed=seed, ntracer=ntracer, catalog=catalog), encoding="utf-8")
        if not catalog.exists() or args.overwrite:
            ez_elapsed = run_logged([str(EZMOCK_BINARY), "-c", str(ez_conf)], ez_log, env=env)
        else:
            ez_elapsed = 0.0

        nactual = count_catalog_rows(catalog)
        xi_conf.write_text(
            fcfc_config(catalog=catalog, pair_path=pair_path, output_text=xi_text, overwrite=args.overwrite),
            encoding="utf-8",
        )
        if not xi_output.exists() or args.overwrite:
            fcfc_elapsed = run_logged([str(FCFC_BINARY), "-c", str(xi_conf)], fcfc_log, env=env)
            save_xi(
                text_path=xi_text,
                output=xi_output,
                seed=seed,
                ntracer=nactual,
                elapsed=fcfc_elapsed,
                config_path=xi_conf,
                pair_path=pair_path,
                catalog=catalog,
                threads=args.threads,
                cpus=cpus,
            )
        else:
            fcfc_elapsed = 0.0

        pk_command = [
            sys.executable,
            "-u",
            str(PK_SCRIPT),
            "--catalog",
            str(catalog),
            "--output",
            str(pk_output),
            "--boxsize",
            str(BOX_SIZE),
            "--meshsize",
            str(args.meshsize),
            "--threads",
            str(args.threads),
            "--seed",
            str(seed),
        ]
        if args.overwrite:
            pk_command.append("--overwrite")
        if not pk_output.exists() or args.overwrite:
            pk_elapsed = run_logged(pk_command, pk_log, env=env)
        else:
            pk_elapsed = 0.0

        runtime_rows.append(
            {
                "seed": seed,
                "ndata_actual": int(nactual),
                "ezmock_sec": float(ez_elapsed),
                "fcfc_sec": float(fcfc_elapsed),
                "jaxpower_sec": float(pk_elapsed),
            }
        )

    summarize(args, trial_root=trial_root, seeds=seeds, ntracer=ntracer, runtime_rows=runtime_rows)


if __name__ == "__main__":
    main()
