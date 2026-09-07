#!/usr/bin/env bash
set -euo pipefail

# Task 4.4: fixed-baseline-fNL/free-p analysis of the two z=0.5 HOD-MAP
# periodic real-space LRG boxes.  This is a login-node CPU workflow and is
# deliberately limited to at most eight CPUs.

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
DESI_CLUSTERING_ROOT="/pscratch/sd/l/lzy/desi-clustering"
CODE_DIR="${PROJECT_ROOT}/codes/task44"
TASK43_CODE_DIR="${PROJECT_ROOT}/codes/task43"

: "${TASK44_CPUSET:=6-13}"
: "${TASK44_THREADS:=8}"
: "${TASK44_OVERWRITE:=0}"

cpu_count="$(
  python - "${TASK44_CPUSET}" <<'PY'
import sys

count = 0
for part in sys.argv[1].split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part:
        lower, upper = map(int, part.split("-", 1))
        if upper < lower:
            raise ValueError(f"invalid CPU range: {part}")
        count += upper - lower + 1
    else:
        int(part)
        count += 1
print(count)
PY
)"
if (( cpu_count < 1 || cpu_count > 8 )); then
  echo "[error] TASK44_CPUSET=${TASK44_CPUSET} exposes ${cpu_count} CPUs; use 1--8" >&2
  exit 2
fi
if (( TASK44_THREADS < 1 || TASK44_THREADS > 8 )); then
  echo "[error] TASK44_THREADS=${TASK44_THREADS}; use 1--8" >&2
  exit 2
fi
if [[ "${TASK44_CPU_PINNED:-0}" != "1" ]] && command -v taskset >/dev/null 2>&1; then
  export TASK44_CPU_PINNED=1
  exec taskset -c "${TASK44_CPUSET}" bash "$0" "$@"
fi

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

export PYTHONPATH="${CODE_DIR}:${TASK43_CODE_DIR}:${DESI_CLUSTERING_ROOT}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${TASK44_THREADS} inter_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
MODE="${1:-all}"
OVERWRITE_ARGS=()
if [[ "${TASK44_OVERWRITE}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

measure() {
  python -u "${CODE_DIR}/task44_measure_pngbase_hodmap_rawbox.py" "$@" \
    --threads "${TASK44_THREADS}" "${OVERWRITE_ARGS[@]}"
}

audit() {
  python -u "${CODE_DIR}/task44_measure_pngbase_hodmap_rawbox.py" audit "${OVERWRITE_ARGS[@]}"
}

build_ascii() {
  python -u "${CODE_DIR}/task44_measure_pngbase_hodmap_rawbox.py" build-ascii "${OVERWRITE_ARGS[@]}"
}

pk_primary() {
  measure pk --mesh 400 --origin positive
}

pk_estimator_checks() {
  measure pk --mesh 512 --origin positive
  measure pk --mesh 400 --origin centered
}

xi() {
  python -u "${CODE_DIR}/task44_validate_pngbase_hodmap_xi_engines.py" fcfc \
    --threads "${TASK44_THREADS}" "${OVERWRITE_ARGS[@]}"
}

theory() {
  python -u "${CODE_DIR}/task44_fit_pngbase_hodmap_rawbox.py" theory \
    --threads "${TASK44_THREADS}" --theory-kmax 5 --smin 30 "${OVERWRITE_ARGS[@]}"
}

fit() {
  python -u "${CODE_DIR}/task44_fit_pngbase_hodmap_rawbox.py" fit \
    --probe pk --kmin-edge 0.003 --kmin-edge 0.005 --smin 30 \
    --threads "${TASK44_THREADS}" --theory-kmax 5 \
    --nwalkers 64 --nsteps 10000 --burnin 2000 --seed 20260821 \
    "${OVERWRITE_ARGS[@]}"
}

fit_xi30() {
  python -u "${CODE_DIR}/task44_fit_pngbase_hodmap_rawbox.py" fit \
    --probe xi --smin 30 \
    --threads "${TASK44_THREADS}" --theory-kmax 5 \
    --nwalkers 64 --nsteps 10000 --burnin 2000 --seed 20260821 \
    "${OVERWRITE_ARGS[@]}"
}

fit_xi_fixedp1() {
  python -u "${CODE_DIR}/task44_fit_pngbase_hodmap_rawbox.py" fit \
    --probe xi --smin 30 --fixed-p 1 \
    --threads "${TASK44_THREADS}" --theory-kmax 5 \
    --nwalkers 64 --nsteps 10000 --burnin 2000 --seed 20260821 \
    "${OVERWRITE_ARGS[@]}"
}

fit_pk_fixedp1() {
  python -u "${CODE_DIR}/task44_fit_pngbase_hodmap_rawbox.py" fit \
    --probe pk --kmin-edge 0.003 --fixed-p 1 --smin 30 \
    --threads "${TASK44_THREADS}" --theory-kmax 5 \
    --nwalkers 64 --nsteps 10000 --burnin 2000 --seed 20260821 \
    "${OVERWRITE_ARGS[@]}"
}

fit_consistency_diagnostics() {
  python -u "${CODE_DIR}/task44_fit_pngbase_pk_xi_consistency.py" \
    --variant all --smin 30 --threads "${TASK44_THREADS}" \
    --nwalkers 64 --nsteps 10000 --burnin 2000 --seed 20260821 \
    "${OVERWRITE_ARGS[@]}"
}

audit_consistency_diagnostics() {
  python -u "${CODE_DIR}/task44_audit_pngbase_pk_xi_consistency.py" \
    --threads "${TASK44_THREADS}" "${OVERWRITE_ARGS[@]}"
}

plot() {
  python -u "${CODE_DIR}/task44_plot_pngbase_hodmap_rawbox.py" "${OVERWRITE_ARGS[@]}"
  python -u "${CODE_DIR}/task44_plot_pngbase_hodmap_rawbox_fixedp1.py" "${OVERWRITE_ARGS[@]}"
}

status() {
  python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/task44_outputs/pngbase_hodmap_rawbox_z0p500")
plot_root = Path("plots/task44/pngbase_hodmap_rawbox_z0p500")
audit = root / "summary/task44_pngbase_hodmap_rawbox_fixedfnl_freep_audit.json"
if not audit.is_file():
    raise FileNotFoundError(audit)
payload = json.loads(audit.read_text(encoding="utf-8"))
if payload["status"] != "pass" or not all(payload["global_gates"].values()):
    raise RuntimeError(payload["global_gates"])
engine_audit = root / "summary/task44_pngbase_hodmap_xi0_three_engine_s30_350_audit.json"
engine_payload = json.loads(engine_audit.read_text(encoding="utf-8"))
if engine_payload["status"] != "pass" or not all(engine_payload["global_gates"].values()):
    raise RuntimeError(engine_payload["global_gates"])
fixedp_audit = root / "summary/task44_pngbase_hodmap_rawbox_fixedp1_freefnl_pk_xi_audit.json"
fixedp_payload = json.loads(fixedp_audit.read_text(encoding="utf-8"))
if fixedp_payload["status"] != "pass" or not all(fixedp_payload["global_gates"].values()):
    raise RuntimeError(fixedp_payload["global_gates"])
consistency_audit = root / "consistency_diagnostics/summary/task44_pngbase_pk_xi_consistency_final_audit.json"
consistency_payload = json.loads(consistency_audit.read_text(encoding="utf-8"))
if consistency_payload["status"] != "pass" or not all(consistency_payload["gates"].values()):
    raise RuntimeError(consistency_payload["gates"])
pdfs = sorted(plot_root.glob("*.pdf"))
if len(pdfs) != 4:
    raise RuntimeError(f"expected exactly four final PDFs, found {len(pdfs)}")
pngs = sorted(plot_root.glob("*.png"))
if pngs:
    raise RuntimeError(f"unexpected persistent PNG outputs: {pngs}")
consistency_pdfs = sorted((plot_root / "consistency_diagnostics").glob("*.pdf"))
if len(consistency_pdfs) != 6:
    raise RuntimeError(f"expected exactly six consistency-diagnostic PDFs, found {len(consistency_pdfs)}")
consistency_pngs = sorted((plot_root / "consistency_diagnostics").glob("*.png"))
if consistency_pngs:
    raise RuntimeError(f"unexpected persistent consistency PNG outputs: {consistency_pngs}")
for tag in ("c300", "c302"):
    row = payload["catalogs"][tag]
    corrected = consistency_payload["catalogs"][tag]
    print(
        f"{tag} fixed fNL={row['fixed_fnl']:g}: "
        f"P0 p={row['primary_pk']['p']['q50']:.4f}, "
        f"corrected xi0 smin30 p={corrected['xi_stochastic_covariance']['posterior']['q50']:.4f}, "
        f"shift after omitting lowest-k bin={row['pk_without_lowest_k_bin']['p_shift_over_primary_sigma68']:+.2f} sigma"
    )
print(f"status={payload['status']}; PDFs={len(pdfs)}")
PY
}

case "${MODE}" in
  audit) audit ;;
  build-ascii) build_ascii ;;
  pk) pk_primary ;;
  pk-checks) pk_estimator_checks ;;
  xi) xi ;;
  theory) theory ;;
  fit) fit ;;
  fit-xi30) fit_xi30 ;;
  fit-xi-fixedp1) fit_xi_fixedp1 ;;
  fit-pk-fixedp1) fit_pk_fixedp1 ;;
  fit-consistency) fit_consistency_diagnostics ;;
  audit-consistency) audit_consistency_diagnostics ;;
  consistency)
    fit_consistency_diagnostics
    audit_consistency_diagnostics
    plot
    status
    ;;
  plot) plot ;;
  status) status ;;
  all)
    audit
    build_ascii
    pk_primary
    pk_estimator_checks
    xi
    theory
    fit
    fit_xi30
    fit_xi_fixedp1
    fit_pk_fixedp1
    fit_consistency_diagnostics
    audit_consistency_diagnostics
    plot
    status
    ;;
  *)
    echo "usage: $0 {audit|build-ascii|pk|pk-checks|xi|theory|fit|fit-xi30|fit-xi-fixedp1|fit-pk-fixedp1|fit-consistency|audit-consistency|consistency|plot|status|all}" >&2
    exit 2
    ;;
esac
