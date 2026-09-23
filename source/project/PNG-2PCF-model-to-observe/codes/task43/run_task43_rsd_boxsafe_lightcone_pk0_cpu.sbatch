#!/bin/bash -l
#SBATCH -J t43_bs_pk0
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/boxsafe_pk0_%j.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/boxsafe_pk0_%j.err

# box-safe 0.4<zobs<0.8 P0 测量链：pytest -> pk0 x25(mesh256) -> P0 covariance(mesh128) -> 16-bin summary
# 与 wide 版差别：manifest/tag/contract(boxsafe_lrgall)/fkp 路径；
# 观测 kmin 由 boxsafe 几何体积决定 = 2pi/1464.31 = 0.004290874（<0.005，仍收 16-bin）

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
DESI_CLUSTERING_ROOT="/pscratch/sd/l/lzy/desi-clustering"
BS_ROOT="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/lightcone_boxsafe_zobs0p4_0p8"
MANIFEST="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/manifests/task43_rsd_validation_lightcone_boxsafe_zobs0p4_0p8_x25.jsonl"
MEASURE_TAG="boxsafe_zobs0p4_0p8_x25_fkpP010000"
COV_TAG="boxsafe_zobs0p4_0p8_x25_fkpP010000_fnlcov0_b1cov2p50_sigmas7p50"
MEASURE_DIR="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/lightcone/pk/${MEASURE_TAG}"
COVARIANCE="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/lightcone/pk_covariance/${COV_TAG}/task43_rsd_lightcone_pk0_cov_ph000_mesh128_kmax0p300_dk0p002.npz"
WINDOW_FILE="${MEASURE_DIR}/task43_rsd_lightcone_pk0_ph000_mesh256_kmax0p300_dk0p002.npz"
FKP_PH000="${BS_ROOT}/fkp/task43_rsd_fkp_AbacusSummit_base_c000_ph000_mmin1p4e13_zobs0p4_0p8_dz0p01.npz"
SUMMARY_PREFIX="${BS_ROOT}/pk_summary/task43_rsd_boxsafe_lightcone_pk0_x25_kmin0p004291_kmax0p10_l0only_15bin"

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="${DESI_CLUSTERING_ROOT}:${PROJECT_ROOT}/codes/task43:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=""
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8 inter_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"

python -m pytest -q codes/task43/test_task43_rsd_lightcone_pk0.py codes/task43/test_task43_rsd_wide_lightcone.py
for index in $(seq 0 24); do
  printf -v phase 'ph%03d' "${index}"
  extra=()
  if (( index == 0 )); then extra=(--window-method smooth); fi
  python -u codes/task43/task43_measure_rsd_lightcone_pk0_jaxpower.py \
    --manifest "${MANIFEST}" --phase "${phase}" --tag "${MEASURE_TAG}" \
    --meshsize 256 --mesh-pad 400 --kmin 0.001 --kmax 0.3001 --dk 0.002 "${extra[@]}"
done

python -u codes/task43/task43_make_rsd_lightcone_pk0_covariance_jaxpower.py \
  --manifest "${MANIFEST}" --phase ph000 --tag "${COV_TAG}" \
  --meshsize 128 --mesh-pad 400 --kmin 0.001 --kmax 0.3001 --dk 0.002 \
  --max-data 50000 --max-random 100000 --fit-contract boxsafe_lrgall \
  --b1-cov 2.50 --sigma-s-cov 7.50

python -u codes/task43/task43_summarize_rsd_lightcone_pk0.py \
  --measurement-dir "${MEASURE_DIR}" --window-file "${WINDOW_FILE}" \
  --covariance "${COVARIANCE}" --output-prefix "${SUMMARY_PREFIX}" \
  --geometry-contract boxsafe_lrgall --fkp-summary "${FKP_PH000}" --kmax-fit 0.10

echo "[done] ${SUMMARY_PREFIX}.npz $(date -u +%FT%TZ)"
