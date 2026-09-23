#!/usr/bin/env bash
# Task4.3 radial single-term RIC 的 2PCF posterior convergence 入口。
#
# 执行大纲：
# 1. 固定同一个 full25 mean、single-lightcone covariance、L=2000、p=1、
#    shell-average、fixed sn0=0 和 emcee seed。
# 2. 唯一切换 operator：dchi=1/2/4，或 dchi=2 下 nsub=10k/50k/200k。
# 3. 每条链使用 32 walkers x 5000 steps、burnin=1250；结果全部进入独立
#    ric_singleterm/fits/convergence_2pcf 目录，不覆盖历史 no/formal-GIC。

set -euo pipefail

ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
PY="/global/homes/l/lzy/anaconda3/envs/desilike/bin/python"
SCRIPT="${ROOT}/codes/task43/task43_fit_minimal_closure.py"
XI="${ROOT}/outputs/task43_outputs/summary/task43_mean_xi_mmin1p4e13_x25_s50_350_ds10_fkpP010000.npz"
COV="${ROOT}/outputs/task43_outputs/summary/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_350_ds10.npz"
FKP="${ROOT}/outputs/task43_outputs/summary/task43_fkp_zeff_mmin1p4e13_x25.npz"
OPDIR="${ROOT}/outputs/task43_outputs/ric_singleterm/operators"
OUTROOT="${ROOT}/outputs/task43_outputs/ric_singleterm/fits/convergence_2pcf"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_one() {
    local label="$1"
    local operator="$2"
    taskset -c 6-13 "${PY}" "${SCRIPT}" \
        --xi-path "${XI}" \
        --output-dir "${OUTROOT}/${label}" \
        --fit-target mean \
        --covariance-mode npz \
        --covariance-path "${COV}" \
        --covariance-key covariance_single_realization \
        --rmin 50 --rmax 350 \
        --models radial_singleterm \
        --radial-ric-operator "${operator}" \
        --fkp-summary "${FKP}" --p0 10000 \
        --theory-boxsize 2000 --p-fixed 1.0 --png-order full --sn0-fixed 0 \
        --kmax 5 --ndense 60000 --xi-kernel shell-averaged \
        --nwalkers 32 --nsteps 5000 --burnin 1250 --seed 20260720
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
