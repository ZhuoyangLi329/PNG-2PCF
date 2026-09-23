#!/bin/bash -l
#SBATCH -J t43_bs_up
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/boxsafe_upstream_%j.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rsd_validation/boxsafe_upstream_%j.err

# box-safe 0.4<zobs<0.8 上游链：切 catalog(x25) -> random+FKP summary(x25) -> FKP views(x25)
# 复刻 run_task43_rsd_wide_lightcone_upstream_cpu.sbatch 的结构，
# 区别是不再从 ASDF 重建 catalog（用 task43_build_rsd_boxsafe_catalogs.py 从 wide 切）

set -euo pipefail

PROJECT_ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
WIDE_MANIFEST="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/manifests/task43_rsd_validation_lightcone_wide_zobs0p4_1p1_x25.jsonl"
MANIFEST="${PROJECT_ROOT}/outputs/task43_outputs/rsd_validation/manifests/task43_rsd_validation_lightcone_boxsafe_zobs0p4_0p8_x25.jsonl"

mkdir -p "${PROJECT_ROOT}/.task43_home"
export HOME="${PROJECT_ROOT}/.task43_home"
set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
export PYTHONPATH="${PROJECT_ROOT}/codes/task43:${PROJECT_ROOT}/codes/task44:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "${PROJECT_ROOT}"

# 1) 切 25 个 phase 的 catalog 并写全量 manifest（幂等：已验证的 ph000 会跳过）
python -u codes/task43/task43_build_rsd_boxsafe_catalogs.py \
  --manifest "${WIDE_MANIFEST}" --manifest-out "${MANIFEST}"

# 2) 每 phase 重建 random(x25) + FKP nbar summary（幂等：validated 自动跳过）
for index in $(seq 0 24); do
  printf -v phase 'ph%03d' "${index}"
  python -u codes/task43/task43_build_rsd_lightcone_random.py --manifest "${MANIFEST}" --phase "${phase}"
done

# 3) 每 phase 生成 FKP views（data/random 带 WEIGHT_TOTAL 的审记视图，与 wide 线口径一致）
for index in $(seq 0 24); do
  printf -v phase 'ph%03d' "${index}"
  python -u codes/task43/task43_prepare_rsd_lightcone_fkp_views.py --manifest "${MANIFEST}" --phase "${phase}"
done

echo "[upstream] boxsafe x25 upstream chain finished"
