#!/bin/bash
set -euo pipefail

PROJECT_DIR="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
CODE_DIR="${PROJECT_DIR}/codes/task43"

: "${TASK43_MANIFEST:=${PROJECT_DIR}/outputs/task43_outputs/manifests/task43_mmin1p4e13_x25.jsonl}"
: "${TASK43_FKP_P0:=10000}"
: "${TASK43_FKP_SUMMARY:=${PROJECT_DIR}/outputs/task43_outputs/summary/task43_fkp_zeff_mmin1p4e13_x25.npz}"
: "${TASK43_OUTPUT_PREFIX:=${PROJECT_DIR}/outputs/task43_outputs/summary/task43_mean_xi_mmin1p4e13_x25_s50_350_ds10_fkpP010000}"
: "${TASK43_OUTPUT_TAG:=fkpP010000}"

cd "${PROJECT_DIR}"

python -u "${CODE_DIR}/task43_summarize_xi.py" \
  --manifest "${TASK43_MANIFEST}" \
  --output-prefix "${TASK43_OUTPUT_PREFIX}" \
  --p0 "${TASK43_FKP_P0}" \
  --output-tag "${TASK43_OUTPUT_TAG}" \
  --fkp-summary "${TASK43_FKP_SUMMARY}" \
  --require-all
