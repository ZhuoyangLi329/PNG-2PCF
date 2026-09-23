#!/usr/bin/env bash
set -euo pipefail

# Build the fixed 50-bin Task43 jaxpower covariance on a login node.
#
# This is intentionally a serial, CPU-only pipeline:
#   1. ph000 weighted RR(s,mu), s=50..550, ds=10, 20 signed-mu bins;
#   2. raw 50x50 jaxpower covariance with the accepted 50..350 settings;
#   3. ell=0 RR deconvolution with the accepted FKP normalization.
#
# Existing complete products are validated and skipped.  A partial product is
# never overwritten automatically; inspect/remove it deliberately before a
# retry.  No Slurm submission is performed by this script.

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
CODE_DIR="${PROJECT_ROOT}/codes/task43"
SCAN_ROOT="${PROJECT_ROOT}/outputs/task43_outputs/rmax_scan"
COV_DIR="${SCAN_ROOT}/covariance"
SUMMARY_DIR="${PROJECT_ROOT}/outputs/task43_outputs/summary"

MANIFEST="${TASK43_RMAX_MANIFEST:-${SCAN_ROOT}/manifests/task43_rmax_scan_mmin1p4e13_x25.jsonl}"
FKP_SUMMARY="${TASK43_RMAX_FKP_SUMMARY:-${SUMMARY_DIR}/task43_fkp_zeff_mmin1p4e13_x25.npz}"
XI_SUMMARY="${TASK43_RMAX_XI_SUMMARY:-${SCAN_ROOT}/summary/task43_mean_xi_mmin1p4e13_x25_s50_550_ds10_fkpP010000.npz}"
CPUSET="${TASK43_RMAX_CPUSET:-${TASK43_CPUSET:-6-13}}"

RR_BUILDER="${CODE_DIR}/task43_build_rmax_scan_rr_smu.py"
RAW_BUILDER="${CODE_DIR}/task43_diagnose_jaxpower_lightcone_covariance.py"
DECONVOLVER="${CODE_DIR}/task43_apply_rr_smu_deconvolution_to_covariance.py"

RR_PREFIX="${COV_DIR}/task43_rr_smu_ph000_nran100k_seed20260702_s50_550_ds10_nmu20"
RAW_PREFIX="${COV_DIR}/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_550_ds10"
FINAL_PREFIX="${COV_DIR}/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_550_ds10"

OLD_RR="${SUMMARY_DIR}/task43_rr_smu_window_smoke_ph000_nran100k_s50_350_ds10_nmu20_midpoint.npz"
OLD_RAW="${SUMMARY_DIR}/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_350_ds10.npz"
OLD_FINAL="${SUMMARY_DIR}/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_350_ds10.npz"
RR_FKP_NORM="4.89250917556916e-10"

die() {
  echo "[fatal] $*" >&2
  exit 2
}

require_nonempty_file() {
  [[ -f "$1" && -s "$1" ]] || die "missing or empty input: $1"
}

assert_covariance_output() {
  case "$1" in
    "${COV_DIR}"/*) ;;
    *) die "refusing output outside ${COV_DIR}: $1" ;;
  esac
}

# Sets STAGE_STATE to complete or missing; any partial/empty pair is fatal.
inspect_pair() {
  local prefix="$1"
  local npz="${prefix}.npz"
  local json="${prefix}.json"
  if [[ -s "${npz}" && -s "${json}" ]]; then
    STAGE_STATE="complete"
  elif [[ ! -e "${npz}" && ! -e "${json}" ]]; then
    STAGE_STATE="missing"
  else
    die "partial or empty product; inspect manually before retry: ${npz}, ${json}"
  fi
}

for output_prefix in "${RR_PREFIX}" "${RAW_PREFIX}" "${FINAL_PREFIX}"; do
  assert_covariance_output "${output_prefix}"
done

for input_file in \
  "${RR_BUILDER}" \
  "${RAW_BUILDER}" \
  "${DECONVOLVER}" \
  "${MANIFEST}" \
  "${FKP_SUMMARY}" \
  "${XI_SUMMARY}" \
  "${OLD_RR}" \
  "${OLD_RAW}" \
  "${OLD_FINAL}"; do
  require_nonempty_file "${input_file}"
done

ENV_SETUP="/global/common/software/desi/users/adematti/cosmodesi_environment.sh"
require_nonempty_file "${ENV_SETUP}"

cd "${PROJECT_ROOT}"
mkdir -p "${COV_DIR}"

# The environment setup references unset shell variables, so nounset is
# disabled only while sourcing it and restored immediately afterwards.
set +u
source "${ENV_SETUP}" main
set -u

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

CPU_COUNT="$({ python - "${CPUSET}" <<'PY'
import sys

spec = sys.argv[1]
cpus = set()
if not spec or any(char.isspace() for char in spec):
    raise SystemExit("CPU set must be a non-empty taskset list without whitespace")
for item in spec.split(","):
    if not item:
        raise SystemExit(f"invalid empty CPU-set component in {spec!r}")
    if "-" in item:
        fields = item.split("-")
        if len(fields) != 2:
            raise SystemExit(f"invalid CPU range {item!r}")
        lo, hi = (int(value) for value in fields)
        if lo < 0 or hi < lo:
            raise SystemExit(f"invalid CPU range {item!r}")
        cpus.update(range(lo, hi + 1))
    else:
        cpu = int(item)
        if cpu < 0:
            raise SystemExit(f"invalid CPU id {cpu}")
        cpus.add(cpu)
print(len(cpus))
PY
} 2>&1)" || die "invalid TASK43_RMAX_CPUSET=${CPUSET}: ${CPU_COUNT}"
[[ "${CPU_COUNT}" =~ ^[0-9]+$ ]] || die "could not count CPUs in ${CPUSET}: ${CPU_COUNT}"
(( CPU_COUNT >= 1 && CPU_COUNT <= 8 )) || die "login-node CPU set must contain 1..8 CPUs; got ${CPU_COUNT}: ${CPUSET}"
taskset -c "${CPUSET}" true || die "CPU set is not available to this process: ${CPUSET}"

run_python() {
  taskset -c "${CPUSET}" python -u "$@"
}

echo "[config] project=${PROJECT_ROOT}"
echo "[config] cpuset=${CPUSET} nthreads=${CPU_COUNT}"
echo "[config] manifest=${MANIFEST}"
echo "[config] xi_summary=${XI_SUMMARY}"
echo "[config] covariance_dir=${COV_DIR}"

# Validate every upstream coordinate before starting the expensive jaxpower
# stage.  The covariance itself uses ph000, while xi scatter must contain all
# 25 ordered phases.
run_python - "${MANIFEST}" "${FKP_SUMMARY}" "${XI_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

manifest_path, fkp_path, xi_path = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
expected_phases = [f"ph{index:03d}" for index in range(25)]
phases = [str(row.get("phase")) for row in rows]
if phases != expected_phases:
    raise RuntimeError(f"manifest must contain ordered ph000..ph024; got {phases}")
for row in rows:
    for key in ("halo_catalog_path", "random_catalog_path"):
        path = Path(row[key])
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"manifest {row['phase']} {key} is missing/empty: {path}")

with np.load(fkp_path, allow_pickle=False) as data:
    required = {"p0_values", "zeff_random_auto", "nbar", "z_edges", "fkp_weights"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"FKP summary missing keys: {sorted(missing)}")
    p0 = np.asarray(data["p0_values"], dtype="f8")
    matches = np.flatnonzero(np.isclose(p0, 10000.0, rtol=0.0, atol=1.0e-10))
    if matches.size != 1:
        raise RuntimeError(f"FKP summary must contain P0=10000 exactly once; got {p0}")
    zeff = float(np.asarray(data["zeff_random_auto"], dtype="f8")[matches[0]])
    if not np.isclose(zeff, 0.7030110803750067, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"unexpected P0=10000 zeff: {zeff}")
    nbar = np.asarray(data["nbar"], dtype="f8")
    if nbar.shape != (20,) or not np.all(np.isfinite(nbar)) or np.any(nbar <= 0.0):
        raise RuntimeError(f"invalid FKP nbar grid: shape={nbar.shape}")

expected_edges = np.arange(50.0, 551.0, 10.0, dtype="f8")
expected_s = 0.5 * (expected_edges[:-1] + expected_edges[1:])
with np.load(xi_path, allow_pickle=False) as data:
    required = {"s", "s_edges", "xi0", "xi0_all", "phases", "nreal", "p0_all", "zeff"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"xi summary missing keys: {sorted(missing)}")
    if not np.array_equal(np.asarray(data["s_edges"], dtype="f8"), expected_edges):
        raise RuntimeError("xi summary s_edges are not exactly 50..550 with ds=10")
    if not np.array_equal(np.asarray(data["s"], dtype="f8"), expected_s):
        raise RuntimeError("xi summary centers are not exactly 55..545")
    xi = np.asarray(data["xi0_all"], dtype="f8")
    if xi.shape != (25, 50) or not np.all(np.isfinite(xi)):
        raise RuntimeError(f"xi0_all must be finite with shape (25,50); got {xi.shape}")
    if int(np.asarray(data["nreal"]).item()) != 25:
        raise RuntimeError("xi summary nreal must be 25")
    if list(np.asarray(data["phases"]).astype(str)) != expected_phases:
        raise RuntimeError("xi summary phase order is not ph000..ph024")
    if not np.allclose(np.asarray(data["p0_all"], dtype="f8"), 10000.0, rtol=0.0, atol=1.0e-10):
        raise RuntimeError("xi summary does not use P0=10000 for every phase")
print(f"[preflight] 25 phases, xi shape=(25,50), P0=10000, zeff={zeff:.15f}")
PY

validate_rr() {
  run_python - "${RR_PREFIX}.npz" "${RR_PREFIX}.json" "${OLD_RR}" "${RR_FKP_NORM}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

npz_path, json_path, old_path = map(Path, sys.argv[1:4])
norm_constant = float(sys.argv[4])
expected_edges = np.arange(50.0, 551.0, 10.0, dtype="f8")
expected_mu_edges = np.linspace(-1.0, 1.0, 21, dtype="f8")
with np.load(npz_path, allow_pickle=False) as data:
    required = {"rr_value", "rr_counts", "rr_norm", "s_edges", "s_edge_pairs", "mu_edges", "s", "mu", "meta_json"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"RR product missing keys: {sorted(missing)}")
    counts = np.asarray(data["rr_counts"], dtype="f8")
    norm = np.asarray(data["rr_norm"], dtype="f8")
    value = np.asarray(data["rr_value"], dtype="f8")
    if counts.shape != (50, 20) or norm.shape != counts.shape or value.shape != counts.shape:
        raise RuntimeError(f"RR arrays must have shape (50,20); got {counts.shape}, {norm.shape}, {value.shape}")
    if not np.array_equal(np.asarray(data["s_edges"], dtype="f8"), expected_edges):
        raise RuntimeError("RR s_edges are not exactly 50..550, ds=10")
    if not np.array_equal(np.asarray(data["mu_edges"], dtype="f8"), expected_mu_edges):
        raise RuntimeError("RR mu_edges are not exactly 20 signed bins on [-1,1]")
    if not np.array_equal(np.asarray(data["s"], dtype="f8"), 0.5 * (expected_edges[:-1] + expected_edges[1:])):
        raise RuntimeError("RR s centers are inconsistent with edges")
    if not np.array_equal(np.asarray(data["mu"], dtype="f8"), 0.5 * (expected_mu_edges[:-1] + expected_mu_edges[1:])):
        raise RuntimeError("RR mu centers are inconsistent with edges")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0) or np.any(np.sum(counts, axis=1) <= 0.0):
        raise RuntimeError("RR counts are non-finite, negative, or contain an empty s row")
    if not np.all(np.isfinite(norm)) or np.any(norm <= 0.0):
        raise RuntimeError("RR normalization is non-finite or non-positive")
    if not np.allclose(value, counts / norm, rtol=2.0e-15, atol=0.0):
        raise RuntimeError("rr_value != rr_counts / rr_norm")
    embedded = json.loads(str(np.asarray(data["meta_json"]).item()))

meta = json.loads(json_path.read_text(encoding="utf-8"))
for source in (embedded, meta):
    if source.get("status") != "done" or source.get("phase") != "ph000":
        raise RuntimeError("RR metadata status/phase mismatch")
    if int(source.get("n_random_used", -1)) != 100000 or int(source.get("seed", -1)) != 20260702:
        raise RuntimeError("RR metadata random count/seed mismatch")
    if not np.isclose(float(source.get("p0", np.nan)), 10000.0, rtol=0.0, atol=1.0e-10):
        raise RuntimeError("RR metadata P0 mismatch")
    if not 1 <= int(source.get("nthreads", 0)) <= 8:
        raise RuntimeError("RR metadata exceeds the login-node thread limit")
    if not np.isclose(float(source.get("authoritative_rr_fkp_norm", np.nan)), norm_constant, rtol=0.0, atol=1.0e-22):
        raise RuntimeError("RR metadata authoritative FKP normalization mismatch")
    bridge = source.get("bridge_to_350", {})
    if not bridge.get("available") or not bridge.get("s_edges_exact") or not bridge.get("mu_edges_exact") or not bridge.get("allclose"):
        raise RuntimeError(f"RR metadata 50..350 bridge failed: {bridge}")

with np.load(old_path, allow_pickle=False) as old:
    if not np.array_equal(expected_edges[:31], np.asarray(old["s_edges"], dtype="f8")):
        raise RuntimeError("authoritative RR has unexpected s edges")
    if not np.allclose(counts[:30], np.asarray(old["rr_counts"], dtype="f8"), rtol=1.0e-12, atol=1.0e-10):
        raise RuntimeError("RR first 30 bins do not reproduce the authoritative RR counts")
    if not np.array_equal(norm[:30], np.asarray(old["rr_norm"], dtype="f8")):
        raise RuntimeError("RR first 30 bins do not reproduce the authoritative RR normalization")
print(f"[validate:rr] shape={counts.shape} sumw2_norm={norm.flat[0]:.15g} bridge=pass")
PY
}

validate_raw() {
  run_python - "${RAW_PREFIX}.npz" "${RAW_PREFIX}.json" "${OLD_RAW}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

npz_path, json_path, old_path = map(Path, sys.argv[1:])
expected_edges = np.arange(50.0, 551.0, 10.0, dtype="f8")
expected_s = 0.5 * (expected_edges[:-1] + expected_edges[1:])
expected_k = 0.0001 + 0.002 * np.arange(1501, dtype="f8")

def relative_frobenius(left, right):
    denom = np.linalg.norm(right)
    return float(np.linalg.norm(left - right) / denom) if denom else float(np.linalg.norm(left - right))

def require_symmetric_spd(name, matrix):
    matrix = np.asarray(matrix, dtype="f8")
    if matrix.shape != (50, 50) or not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"{name} must be finite with shape (50,50); got {matrix.shape}")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-11, atol=1.0e-15):
        raise RuntimeError(f"{name} is not symmetric")
    eig = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    if eig[0] <= 0.0 or not np.all(np.isfinite(eig)):
        raise RuntimeError(f"{name} is not SPD: min eigenvalue={eig[0]}")
    return eig

with np.load(npz_path, allow_pickle=False) as data:
    required = {
        "s", "s_edges", "k_edges", "covariance_single_realization",
        "corrected_covariance_single_realization", "uncorrected_covariance_single_realization",
        "correlation", "covariance_WW", "covariance_WS", "covariance_SS",
        "covariance_WW_raw", "covariance_WS_raw", "covariance_SS_raw",
        "projection_matrix", "zeff", "p0",
    }
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"raw covariance missing keys: {sorted(missing)}")
    if not np.array_equal(np.asarray(data["s_edges"], dtype="f8"), expected_edges):
        raise RuntimeError("raw covariance s_edges are not exactly 50..550, ds=10")
    if not np.array_equal(np.asarray(data["s"], dtype="f8"), expected_s):
        raise RuntimeError("raw covariance s centers are not exactly 55..545")
    if not np.allclose(np.asarray(data["k_edges"], dtype="f8"), expected_k, rtol=0.0, atol=5.0e-15):
        raise RuntimeError("raw covariance k grid is not k=0.0001..3.0001, dk=0.002")
    projection = np.asarray(data["projection_matrix"], dtype="f8")
    if projection.shape != (50, 1500) or not np.all(np.isfinite(projection)):
        raise RuntimeError(f"projection matrix must be finite (50,1500); got {projection.shape}")
    covariance = np.asarray(data["covariance_single_realization"], dtype="f8")
    eig = require_symmetric_spd("covariance_single_realization", covariance)
    corrected = np.asarray(data["corrected_covariance_single_realization"], dtype="f8")
    if corrected.shape != (50, 50) or not np.all(np.isfinite(corrected)) or not np.allclose(corrected, corrected.T, rtol=1.0e-11, atol=1.0e-15):
        raise RuntimeError("corrected raw covariance is not finite/symmetric (50,50)")
    for key in ("covariance_WW", "covariance_WS", "covariance_SS", "covariance_WW_raw", "covariance_WS_raw", "covariance_SS_raw"):
        value = np.asarray(data[key], dtype="f8")
        if value.shape != (50, 50) or not np.all(np.isfinite(value)) or not np.allclose(value, value.T, rtol=1.0e-11, atol=1.0e-15):
            raise RuntimeError(f"{key} is not finite/symmetric (50,50)")
    corr = np.asarray(data["correlation"], dtype="f8")
    if corr.shape != (50, 50) or not np.all(np.isfinite(corr)) or not np.allclose(np.diag(corr), 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise RuntimeError("raw correlation matrix is invalid")
    if not np.isclose(float(np.asarray(data["p0"]).item()), 10000.0, rtol=0.0, atol=1.0e-10):
        raise RuntimeError("raw covariance P0 mismatch")
    if not np.isclose(float(np.asarray(data["zeff"]).item()), 0.7030110803750067, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("raw covariance zeff mismatch")
    with np.load(old_path, allow_pickle=False) as old:
        if not np.array_equal(expected_edges[:31], np.asarray(old["s_edges"], dtype="f8")):
            raise RuntimeError("authoritative raw covariance has unexpected edges")
        bridges = {
            "corrected": relative_frobenius(corrected[:30, :30], np.asarray(old["corrected_covariance_single_realization"], dtype="f8")),
            "projection": relative_frobenius(projection[:30], np.asarray(old["projection_matrix"], dtype="f8")),
        }
        if any(value > 1.0e-8 for value in bridges.values()):
            raise RuntimeError(f"raw 50..350 bridge failed: {bridges}")

meta = json.loads(json_path.read_text(encoding="utf-8"))
if meta.get("status") != "done" or meta.get("task") != "task43_diagnose_jaxpower_lightcone_covariance":
    raise RuntimeError("raw covariance metadata status/task mismatch")
if meta.get("phase") != "ph000" or int(meta.get("phase_index", -1)) != 0:
    raise RuntimeError("raw covariance metadata phase mismatch")
if not np.isclose(float(meta.get("p0", np.nan)), 10000.0, rtol=0.0, atol=1.0e-10):
    raise RuntimeError("raw covariance metadata P0 mismatch")
for label, expected_count, expected_seed in (("data", 50000, 20260703), ("random", 100000, 20260704)):
    section = meta.get(label, {})
    if int(section.get("n_used", -1)) != expected_count or int(section.get("subsample_seed", -1)) != expected_seed:
        raise RuntimeError(f"raw covariance {label} count/seed mismatch: {section}")
    if section.get("subsample_weight_rescale") is not True:
        raise RuntimeError(f"raw covariance {label} subsample weights were not rescaled")
mesh = meta.get("mesh", {})
if int(mesh.get("meshsize", -1)) != 64 or not np.isclose(float(mesh.get("pad", np.nan)), 400.0):
    raise RuntimeError("raw covariance mesh contract mismatch")
window = meta.get("window", {})
window_edges = np.asarray(window.get("s_edges", []), dtype="f8")
if window.get("basis") != "bessel" or window.get("ells") != [0] or window.get("los") != "local" or window.get("resampler") != "cic" or int(window.get("interlacing", -1)) != 1:
    raise RuntimeError("raw covariance window basis/LOS/mesh assignment mismatch")
if not np.array_equal(window_edges, np.arange(0.0, 3601.0, 2.0, dtype="f8")):
    raise RuntimeError("raw covariance window edges are not 0..3600, ds=2")
interpolation = window.get("interpolation", {})
if interpolation.get("applied") is not True or int(interpolation.get("ncoords", -1)) != 8192 or float(interpolation.get("log10_min", np.nan)) != -2.0 or float(interpolation.get("log10_max", np.nan)) != 8.0:
    raise RuntimeError("raw covariance window interpolation contract mismatch")
k_grid = meta.get("k_grid", {})
if int(k_grid.get("nk", -1)) != 1500 or not np.isclose(float(k_grid.get("kmin", np.nan)), 0.0001) or not np.isclose(float(k_grid.get("kmax", np.nan)), 3.0001) or not np.isclose(float(k_grid.get("dk", np.nan)), 0.002):
    raise RuntimeError("raw covariance k-grid metadata mismatch")
theory = meta.get("theory", {})
for key, expected in (("b1_cov", 2.5), ("fnl_cov", 0.0), ("p_fixed", 1.0), ("sn0_fixed", 0.0)):
    if not np.isclose(float(theory.get(key, np.nan)), expected, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"raw covariance theory {key} mismatch")
if theory.get("cosmology") != "abacus_c000" or meta.get("covariance_flags") != ["smooth", "fftlog"]:
    raise RuntimeError("raw covariance cosmology/flags mismatch")
if int(meta.get("spd", {}).get("n_floored", -1)) != 0:
    raise RuntimeError(f"raw covariance required SPD repair: {meta.get('spd')}")
condition = float(np.linalg.cond(covariance))
if not np.isfinite(condition) or condition >= 1.0e12:
    raise RuntimeError(f"raw covariance is numerically singular: cond={condition}")
print(f"[validate:raw] shape=(50,50) eig_min={eig[0]:.6e} cond={condition:.6g} bridge={bridges}")
PY
}

validate_final() {
  run_python - "${FINAL_PREFIX}.npz" "${FINAL_PREFIX}.json" "${RAW_PREFIX}.npz" "${OLD_FINAL}" "${RR_FKP_NORM}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

npz_path, json_path, raw_path, old_path = map(Path, sys.argv[1:5])
norm_constant = float(sys.argv[5])
expected_edges = np.arange(50.0, 551.0, 10.0, dtype="f8")
expected_s = 0.5 * (expected_edges[:-1] + expected_edges[1:])

def relative_frobenius(left, right):
    denom = np.linalg.norm(right)
    return float(np.linalg.norm(left - right) / denom) if denom else float(np.linalg.norm(left - right))

with np.load(npz_path, allow_pickle=False) as data:
    required = {
        "s", "s_edges", "ells", "ellsin", "rr_window_matrix", "rr_window_inverse",
        "covariance_single_realization", "covariance_WW", "covariance_WS", "covariance_SS",
        "correlation", "meta_json", "source_meta_json", "zeff", "p0",
    }
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"RR-deconvolved covariance missing keys: {sorted(missing)}")
    if not np.array_equal(np.asarray(data["s_edges"], dtype="f8"), expected_edges) or not np.array_equal(np.asarray(data["s"], dtype="f8"), expected_s):
        raise RuntimeError("RR-deconvolved covariance coordinates are not s=50..550, ds=10")
    if not np.array_equal(np.asarray(data["ells"], dtype="i8"), np.array([0], dtype="i8")) or not np.array_equal(np.asarray(data["ellsin"], dtype="i8"), np.array([0], dtype="i8")):
        raise RuntimeError("RR deconvolution must use ell=ellsin=0")
    window = np.asarray(data["rr_window_matrix"], dtype="f8")
    inverse = np.asarray(data["rr_window_inverse"], dtype="f8")
    if window.shape != (50, 50) or inverse.shape != (50, 50) or not np.all(np.isfinite(window)) or not np.all(np.isfinite(inverse)):
        raise RuntimeError("RR window/inverse must be finite (50,50)")
    offdiag = float(np.max(np.abs(window - np.diag(np.diag(window)))))
    if offdiag > 1.0e-14 or np.any(np.diag(window) <= 0.0):
        raise RuntimeError(f"ell=0 RR window is not positive diagonal: offdiag={offdiag}")
    if not np.allclose(window @ inverse, np.eye(50), rtol=1.0e-11, atol=1.0e-12):
        raise RuntimeError("stored RR window inverse is inconsistent")
    covariance = np.asarray(data["covariance_single_realization"], dtype="f8")
    if covariance.shape != (50, 50) or not np.all(np.isfinite(covariance)) or not np.allclose(covariance, covariance.T, rtol=1.0e-11, atol=1.0e-15):
        raise RuntimeError("final covariance is not finite/symmetric (50,50)")
    eig = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    if eig[0] <= 0.0 or not np.all(np.isfinite(eig)):
        raise RuntimeError(f"final covariance is not SPD: min eigenvalue={eig[0]}")
    for key in ("covariance_WW", "covariance_WS", "covariance_SS"):
        value = np.asarray(data[key], dtype="f8")
        if value.shape != (50, 50) or not np.all(np.isfinite(value)) or not np.allclose(value, value.T, rtol=1.0e-11, atol=1.0e-15):
            raise RuntimeError(f"final {key} is not finite/symmetric (50,50)")
    corr = np.asarray(data["correlation"], dtype="f8")
    if corr.shape != (50, 50) or not np.all(np.isfinite(corr)) or not np.allclose(np.diag(corr), 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise RuntimeError("final correlation matrix is invalid")
    embedded = json.loads(str(np.asarray(data["meta_json"]).item()))
    with np.load(raw_path, allow_pickle=False) as raw:
        directly_transformed = inverse @ np.asarray(raw["covariance_single_realization"], dtype="f8") @ inverse.T
        transform_error = relative_frobenius(covariance, directly_transformed)
        if transform_error > 1.0e-10:
            raise RuntimeError(f"final covariance differs from direct RR transform: rel={transform_error}")
    with np.load(old_path, allow_pickle=False) as old:
        bridges = {
            "window": relative_frobenius(window[:30, :30], np.asarray(old["rr_window_matrix"], dtype="f8")),
            "covariance": relative_frobenius(covariance[:30, :30], np.asarray(old["covariance_single_realization"], dtype="f8")),
        }
        if any(value > 1.0e-8 for value in bridges.values()):
            raise RuntimeError(f"final 50..350 bridge failed: {bridges}")

meta = json.loads(json_path.read_text(encoding="utf-8"))
if embedded.get("status") != "done" or meta.get("status") != "done" or meta.get("task") != "task43_apply_rr_smu_deconvolution_to_covariance":
    raise RuntimeError("final covariance metadata status/task mismatch")
rr = meta.get("rr_window", {})
if rr.get("kind") != "RR" or rr.get("ells") != [0] or rr.get("ellsin") != [0] or int(rr.get("resolution", -1)) != 1 or rr.get("shape") != [50, 50]:
    raise RuntimeError(f"final RR-window metadata mismatch: {rr}")
if not np.isclose(float(rr.get("fkp_norm", np.nan)), norm_constant, rtol=0.0, atol=1.0e-22):
    raise RuntimeError("final RR FKP normalization mismatch")
if int(meta.get("spd", {}).get("covariance_single_realization", {}).get("n_floored", -1)) != 0:
    raise RuntimeError(f"final covariance required SPD repair: {meta.get('spd', {}).get('covariance_single_realization')}")
scatter = meta.get("scatter_comparison")
if not isinstance(scatter, dict) or scatter.get("available") is not True or int(scatter.get("nreal", -1)) != 25 or int(scatter.get("nbins", -1)) != 50:
    raise RuntimeError(f"final scatter validation is unavailable/mismatched: {scatter}")
chi2 = float(scatter.get("chi2_mean_per_realization", np.nan))
expected = float(scatter.get("expected_chi2_mean_about_sample_mean", np.nan))
sigma_ratio = float(scatter.get("sample_std_over_cov_sigma_median", np.nan))
if not np.isfinite(chi2) or not np.isfinite(expected) or expected <= 0.0 or not np.isfinite(sigma_ratio) or sigma_ratio <= 0.0:
    raise RuntimeError("final scatter comparison contains invalid diagnostics")
condition = float(np.linalg.cond(covariance))
if not np.isfinite(condition) or condition >= 1.0e12:
    raise RuntimeError(f"final covariance is numerically singular: cond={condition}")
print(
    "[validate:final] shape=(50,50) eig_min={:.6e} cond={:.6g} "
    "RRdiag={:.6g}..{:.6g} bridge={} chi2/expected={:.4g} sigma_ratio_med={:.4g}".format(
        eig[0], condition, np.min(np.diag(window)), np.max(np.diag(window)), bridges,
        chi2 / expected, sigma_ratio,
    )
)
PY
}

inspect_pair "${RR_PREFIX}"
if [[ "${STAGE_STATE}" == "complete" ]]; then
  echo "[skip:rr] complete pair exists; validating before reuse"
else
  echo "[run:rr] building fixed ph000 50x20 RR window"
  run_python "${RR_BUILDER}" \
    --manifest "${MANIFEST}" \
    --fkp-summary "${FKP_SUMMARY}" \
    --output-prefix "${RR_PREFIX}" \
    --phase-index 0 \
    --max-random 100000 \
    --seed 20260702 \
    --p0 10000 \
    --s-min 50 \
    --s-max 550 \
    --s-step 10 \
    --nmu 20 \
    --nthreads "${CPU_COUNT}" \
    --old-rr "${OLD_RR}"
  inspect_pair "${RR_PREFIX}"
  [[ "${STAGE_STATE}" == "complete" ]] || die "RR builder returned without a complete product"
fi
validate_rr

inspect_pair "${RAW_PREFIX}"
if [[ "${STAGE_STATE}" == "complete" ]]; then
  echo "[skip:raw] complete pair exists; validating before reuse"
else
  echo "[run:raw] building raw 50x50 jaxpower covariance"
  run_python "${RAW_BUILDER}" \
    --manifest "${MANIFEST}" \
    --fkp-summary "${FKP_SUMMARY}" \
    --xi-summary "${XI_SUMMARY}" \
    --output-prefix "${RAW_PREFIX}" \
    --phase-index 0 \
    --p0 10000 \
    --max-random 100000 \
    --max-data 50000 \
    --subsample-seed 20260702 \
    --meshsize 64 \
    --mesh-pad 400 \
    --window-s-max 3600 \
    --window-ds 2 \
    --window-basis bessel \
    --window-ells 0 \
    --interpolate-window \
    --interpolate-window-ncoords 8192 \
    --interpolate-window-log10-min -2 \
    --interpolate-window-log10-max 8 \
    --covariance-flags smooth,fftlog \
    --s-edge-min 50 \
    --s-edge-max 550 \
    --s-edge-step 10 \
    --kmin 0.0001 \
    --kmax 3.0001 \
    --dk 0.002 \
    --los local \
    --resampler cic \
    --interlacing 1 \
    --b1-cov 2.5 \
    --fnl-cov 0 \
    --p-fixed 1.0 \
    --sn0-fixed 0 \
    --cosmology abacus_c000 \
    --floor-fraction 1.0e-10
  inspect_pair "${RAW_PREFIX}"
  [[ "${STAGE_STATE}" == "complete" ]] || die "raw builder returned without a complete product"
fi
validate_raw

inspect_pair "${FINAL_PREFIX}"
if [[ "${STAGE_STATE}" == "complete" ]]; then
  echo "[skip:final] complete pair exists; validating before reuse"
else
  echo "[run:final] applying ell=0 RR deconvolution"
  run_python "${DECONVOLVER}" \
    --covariance-path "${RAW_PREFIX}.npz" \
    --rr-smu-path "${RR_PREFIX}.npz" \
    --output-prefix "${FINAL_PREFIX}" \
    --xi-summary "${XI_SUMMARY}" \
    --rr-window-kind RR \
    --rr-fkp-norm "${RR_FKP_NORM}" \
    --ells 0 \
    --resolution 1 \
    --floor-fraction 1.0e-10
  inspect_pair "${FINAL_PREFIX}"
  [[ "${STAGE_STATE}" == "complete" ]] || die "RR deconvolver returned without a complete product"
fi
validate_final

echo "[done] validated fixed covariance: ${FINAL_PREFIX}.npz"
