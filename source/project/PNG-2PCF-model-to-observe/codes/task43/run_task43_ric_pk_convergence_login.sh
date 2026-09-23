#!/usr/bin/env bash
# Task4.3 radial single-term RIC 的 P(k) posterior convergence 入口。
#
# 执行大纲：
# 1. 固定当前 15-bin full25 mean payload、jaxpower single-lightcone covariance、
#    free sn0、Abacus c000、kmax=0.10 与 mother-box kth,min=2pi/2000。
# 2. 只切换与 2PCF 共用的 radial operator（dchi 与 random nsub）。
# 3. 每条 convergence 链使用相同 seed、18 walkers、120006 evaluations；正式
#    dchi=2/nsub=200k 主链另有 600012 evaluations，本脚本只做 matched A/B。

set -euo pipefail

ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
PY="/global/homes/l/lzy/anaconda3/envs/desilike/bin/python"
SCRIPT="${ROOT}/codes/task43/task43_fit_pk_lightcone.py"
PAYLOAD="${ROOT}/outputs/task43_outputs/pk_lightcone/summary/task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_payload.npz"
OPDIR="${ROOT}/outputs/task43_outputs/ric_singleterm/operators"
OUTROOT="${ROOT}/outputs/task43_outputs/ric_singleterm/fits/convergence_pk"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_one() {
    local label="$1"
    local operator="$2"
    taskset -c 6-13 "${PY}" "${SCRIPT}" \
        --payload "${PAYLOAD}" \
        --ric-operator "${operator}" \
        --label "${label}" \
        --output-dir "${OUTROOT}" \
        --plot-dir "${ROOT}/plots/task43/ric_singleterm" \
        --target-evals 120006 --nwalkers 18 --burn-fraction 0.10 \
        --seed 20260720 --p-fixed 1.0 --sn0-policy free \
        --cosmology abacus_c000 \
        --window-theory-kmin 0.0031415926535897933 \
        --no-plot --overwrite
}

run_one "dchi2_nsub200000_reference" \
    "${OPDIR}/task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712_L2000.npz"
run_one "dchi1_nsub200000" \
    "${OPDIR}/task43_ric_factorized_operator_ph000_dchi1_nsub200000_sobol2p22_ds2_seed20260712_L2000.npz"
run_one "dchi4_nsub200000" \
    "${OPDIR}/task43_ric_factorized_operator_ph000_dchi4_nsub200000_sobol2p22_ds2_seed20260712_L2000.npz"
run_one "dchi2_nsub10000" \
    "${OPDIR}/task43_ric_factorized_operator_ph000_dchi2_nsub10000_sobol2p22_ds2_seed20260712_L2000.npz"
run_one "dchi2_nsub50000" \
    "${OPDIR}/task43_ric_factorized_operator_ph000_dchi2_nsub50000_sobol2p22_ds2_seed20260712_L2000.npz"
