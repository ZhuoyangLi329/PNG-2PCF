#!/usr/bin/env bash
# Validate all 25 extended measurements, bridge their first 30 bins, then
# produce the single common 50-bin mean vector used by every rmax fit.

set -euo pipefail

ROOT="/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"
PY="/global/homes/l/lzy/anaconda3/envs/desilike/bin/python"
MANIFEST="${ROOT}/outputs/task43_outputs/rmax_scan/manifests/task43_rmax_scan_mmin1p4e13_x25.jsonl"
FKP="${ROOT}/outputs/task43_outputs/summary/task43_fkp_zeff_mmin1p4e13_x25.npz"
PREFIX="${ROOT}/outputs/task43_outputs/rmax_scan/summary/task43_mean_xi_mmin1p4e13_x25_s50_550_ds10_fkpP010000"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${ROOT}"
"${PY}" codes/task43/task43_audit_rmax_scan_measurements.py \
  --manifest "${MANIFEST}" \
  --require-all

if [[ -e "${PREFIX}.npz" || -e "${PREFIX}.json" ]]; then
  if [[ ! -e "${PREFIX}.npz" || ! -e "${PREFIX}.json" ]]; then
    echo "[error] incomplete summary pair: ${PREFIX}.{npz,json}" >&2
    exit 2
  fi
  echo "[skip] complete summary pair already exists: ${PREFIX}.{npz,json}"
else
  "${PY}" codes/task43/task43_summarize_xi.py \
    --manifest "${MANIFEST}" \
    --output-prefix "${PREFIX}" \
    --p0 10000 \
    --fkp-summary "${FKP}" \
    --require-all
fi

"${PY}" - "${PREFIX}.npz" <<'PY'
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1])
with np.load(path, allow_pickle=False) as data:
    edges = np.asarray(data["s_edges"], dtype="f8")
    centers = np.asarray(data["s"], dtype="f8")
    xi = np.asarray(data["xi0"], dtype="f8")
    xi_all = np.asarray(data["xi0_all"], dtype="f8")
    nreal = int(np.asarray(data["nreal"]).item())
expected_edges = np.arange(50.0, 560.0, 10.0)
expected_centers = 0.5 * (expected_edges[:-1] + expected_edges[1:])
if nreal != 25 or xi.shape != (50,) or xi_all.shape != (25, 50):
    raise SystemExit(f"invalid summary shapes: nreal={nreal}, xi={xi.shape}, xi_all={xi_all.shape}")
if not np.array_equal(edges, expected_edges) or not np.array_equal(centers, expected_centers):
    raise SystemExit("summary separation coordinates violate the scan contract")
if not np.all(np.isfinite(xi_all)):
    raise SystemExit("summary contains non-finite xi values")
print(f"[gate] summary valid: {path} nreal=25 nbins=50")
PY
