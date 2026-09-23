#!/usr/bin/env bash
# 已授权FKP诊断的模型差分；与测量共用同一login、同一6-13 CPU集合，总核数<=8。
set -eo pipefail
cd /pscratch/sd/l/lzy/PNG-2PCF-model-to-observe
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
out=outputs/task43_outputs/rsd_validation/task432_model_repair/fkp_conditional_diagnostics
hostname > "$out/logs/ric_response.hostname"
for phase in ph000 ph012 ph024; do
    attempts=0
    while [[ ! -f "$out/paired/$phase/independent_xi_block1.npz" ]]; do
        attempts=$((attempts + 1))
        if [[ "$attempts" -gt 40 ]]; then
            echo "Missing completed measurements for $phase" >&2
            exit 2
        fi
        sleep 30
    done
    for seed in 4322407 4322447; do
        # 配对差分比单个绝对核容易收敛；首个16k点作加密参照，其余8k两独立scramble。
        nsub=8192
        if [[ "$phase" == ph000 && "$seed" == 4322407 ]]; then nsub=16384; fi
        taskset -c 6-13 python codes/task432/task432_fkp_ric_response.py --phase "$phase" --seed "$seed" --nsub "$nsub"
    done
done
# 测量环境为Python3.12；跨到desilike 3.11时清除前者注入的模块路径。
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 taskset -c 6-13 /global/homes/l/lzy/anaconda3/envs/desilike/bin/python codes/task432/task432_fkp_analyze.py --require-ric
date -u > "$out/logs/ric_response.completed"
