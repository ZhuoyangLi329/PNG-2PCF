#!/bin/bash -l
#SBATCH -J t43_rmin_xi_one
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1700M
#SBATCH -t 00:20:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rmin_scan/xi_single_%j.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rmin_scan/xi_single_%j.err

set -euo pipefail

# 执行逻辑大纲：读取一个 manifest row，在 shared GPU 上测量统一的
# 30--350 Mpc/h weighted Landy--Szalay 2PCF，并把线程固定为 1。
PROJECT_DIR="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
CODE_DIR="${PROJECT_DIR}/codes/task43"
: "${TASK43_INDEX:?请把 TASK43_INDEX 设置为 [0,24] 的 manifest row}"
: "${TASK43_MANIFEST:=${PROJECT_DIR}/outputs/task43_outputs/rmin_scan/manifests/task43_rmin_scan_mmin1p4e13_x25.jsonl}"
: "${TASK43_FKP_SUMMARY:=${PROJECT_DIR}/plots/outputs/task43_outputs/summary/task43_fkp_zeff_mmin1p4e13_x25.npz}"

if (( TASK43_INDEX < 0 || TASK43_INDEX > 24 )); then
  echo "[fatal] TASK43_INDEX 必须在 [0,24]，实际为 ${TASK43_INDEX}" >&2
  exit 2
fi

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_NUM_THREADS=1

cd "${PROJECT_DIR}"
python -u "${CODE_DIR}/task43_measure_xi_cucount.py" \
  --manifest "${TASK43_MANIFEST}" \
  --index "${TASK43_INDEX}" \
  --s-edge-min 30 \
  --s-edge-max 350 \
  --s-edge-step 10 \
  --fkp-summary "${TASK43_FKP_SUMMARY}" \
  --p0 10000 \
  --output-tag fkpP010000 \
  --backend jax \
  --nthreads 1 \
  --cosmology DESI
