#!/bin/bash
# Run existing diagnostic stages only; frozen formal inputs are read-only.
set -eo pipefail
PROJECT_ROOT=/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
fit_python=/global/homes/l/lzy/anaconda3/envs/desilike/bin/python
case "${1:-help}" in
  core)
    taskset -c 6-13 "$fit_python" codes/task432/task432_joint_b1_cause_tests.py shot
    taskset -c 6-13 "$fit_python" codes/task432/task432_joint_b1_cause_tests.py ric
    taskset -c 6-13 "$fit_python" codes/task432/task432_joint_b1_common_theory.py
    ;;
  estimator)
    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    export PYTHONPATH="/pscratch/sd/l/lzy/desi-clustering:$PROJECT_ROOT/codes/task43:${PYTHONPATH:-}"
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=""
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=6 inter_op_parallelism_threads=1"
    taskset -c 6-13 python codes/task432/task432_shot_estimator_contact.py
    for phase in ph000 ph012 ph024; do
      taskset -c 6-13 python codes/task43/task43_measure_rsd_lightcone_p02_jaxpower.py --phase "$phase" --tag boxsafe_p02_window_cause_pilot_v1 --window-method smooth
    done
    taskset -c 6-11 python codes/task43/task43_measure_rsd_lightcone_p02_jaxpower.py --phase ph000 --tag boxsafe_p02_window_cause_repeat_v1 --window-method smooth
    ;;
  analyze)
    taskset -c 6-13 "$fit_python" codes/task432/task432_window_cause_analysis.py
    taskset -c 6-13 "$fit_python" codes/task432/task432_cause_report.py
    ;;
  package)
    taskset -c 6-13 "$fit_python" codes/task432/task432_finalize_cause_bundle.py
    ;;
  *) echo "Usage: bash run_cause_tests.sh {core|estimator|analyze|package}" ;;
esac
