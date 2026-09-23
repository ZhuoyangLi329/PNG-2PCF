#!/bin/bash -l
#SBATCH -J t43_rmin_xi
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1700M
#SBATCH -t 00:20:00
#SBATCH --array=0-24%1
#SBATCH -o /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rmin_scan/xi_array_%A_%a.out
#SBATCH -e /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/logs/task43/rmin_scan/xi_array_%A_%a.err

set -euo pipefail

# array 每个元素只处理一个 phase；%1 保证始终只占用一个 shared GPU。
export TASK43_INDEX="${SLURM_ARRAY_TASK_ID}"
exec /bin/bash /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/task43/run_task43_rmin_scan_xi_gpu_single.sbatch
