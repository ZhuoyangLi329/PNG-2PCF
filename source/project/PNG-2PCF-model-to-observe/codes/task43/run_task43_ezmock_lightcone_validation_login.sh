#!/usr/bin/env bash
set -euo pipefail

# Task43 EZmock fixed-amplitude lightcone validation 的登录节点串行入口。
#
# 执行关系：
#   manifest -> generate -> random -> xi_ez/pk -> xi_ab -> summarize
#
# 资源约束：任一时刻只运行一个 realization；FCFC/EZmock/jaxpower 总占用
# 不超过 8 核。当前 FIX_AMPLITUDE=T 仅用于均值与 pipeline validation，
# 将来的 covariance production 不得复用这个入口。

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
DESI_CLUSTERING_ROOT="/pscratch/sd/l/lzy/desi-clustering"
MANIFEST="${PROJECT_ROOT}/outputs/task43_outputs/ezmock_lightcone_z0p6_0p8_mmin1p4e13_x25_fixedamp_validation/manifests/task43_ezmock_lightcone_fixedamp_x10.jsonl"
TAG="ezmock_fixedamp_x10_mmin1p4e13_x25_fkpP010000"
THREADS="${TASK43_EZLC_THREADS:-8}"
CPUSET="${TASK43_EZLC_CPUSET:-6-13}"

if (( THREADS < 1 || THREADS > 8 )); then
  echo "TASK43_EZLC_THREADS must be in [1,8], got ${THREADS}" >&2
  exit 2
fi

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

export PYTHONPATH="${DESI_CLUSTERING_ROOT}:${PROJECT_ROOT}/codes/task43:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${THREADS} inter_op_parallelism_threads=1"
export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

cd "${PROJECT_ROOT}"
MODE="${1:-status}"
INDEX="${2:-0}"

run_limited() {
  taskset -c "${CPUSET}" "$@"
}

case "${MODE}" in
  manifest)
    run_limited python -u codes/task43/task43_make_ezmock_lightcone_manifest.py
    ;;
  generate)
    export OMP_NUM_THREADS="${THREADS}"
    run_limited python -u codes/task43/task43_build_ezmock_lightcone.py \
      --manifest "${MANIFEST}" --index "${INDEX}" --threads "${THREADS}"
    ;;
  random)
    export OMP_NUM_THREADS=1
    run_limited python -u codes/task43/task43_build_randoms.py \
      --manifest "${MANIFEST}" --index "${INDEX}"
    ;;
  xi_ez)
    export OMP_NUM_THREADS="${THREADS}"
    run_limited python -u codes/task43/task43_measure_lightcone_xi_fcfc.py \
      --sample ezmock --manifest "${MANIFEST}" --index "${INDEX}" --threads "${THREADS}"
    ;;
  xi_ab)
    export OMP_NUM_THREADS="${THREADS}"
    run_limited python -u codes/task43/task43_measure_lightcone_xi_fcfc.py \
      --sample abacus --index "${INDEX}" --threads "${THREADS}"
    ;;
  pk)
    export OMP_NUM_THREADS=1
    WINDOW_ARGS=()
    if [[ "${INDEX}" == "0" ]]; then
      WINDOW_ARGS=(--window-method smooth)
    fi
    run_limited python -u codes/task43/task43_measure_pk_jaxpower.py \
      --manifest "${MANIFEST}" --index "${INDEX}" --tag "${TAG}" --p0 10000 \
      --meshsize 256 --mesh-pad 400 --kmin 0.001 --kmax 0.3001 --dk 0.002 \
      --los local "${WINDOW_ARGS[@]}"
    ;;
  smoke)
    "$0" manifest
    "$0" generate 0
    "$0" random 0
    "$0" xi_ab 0
    "$0" xi_ez 0
    "$0" pk 0
    ;;
  full_ez)
    for i in $(seq 0 9); do
      "$0" generate "${i}"
      "$0" random "${i}"
      "$0" xi_ez "${i}"
      "$0" pk "${i}"
    done
    ;;
  full_abacus_xi)
    for i in $(seq 0 24); do
      "$0" xi_ab "${i}"
    done
    ;;
  summarize)
    export OMP_NUM_THREADS=1
    run_limited python -u codes/task43/task43_summarize_ezmock_lightcone_validation.py --overwrite
    ;;
  status)
    find outputs/task43_outputs/ezmock_lightcone_z0p6_0p8_mmin1p4e13_x25_fixedamp_validation \
      -type f \( -name '*.npz' -o -name '*.json' -o -name '*.pdf' \) | sort
    ;;
  *)
    echo "Usage: $0 {manifest|generate|random|xi_ez|xi_ab|pk|smoke|full_ez|full_abacus_xi|summarize|status} [index]" >&2
    exit 2
    ;;
esac

