from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from task43_build_rsd_lightcone_random import fkp_path
from task43_build_rsd_formal_gic_window import output_path
from task43_measure_rsd_lightcone_xi import project_multipoles
from task43_rsd_lightcone_pk0_contract import (
    TASK43_WIDE_LRGALL_FIT_EDGES,
    WINDOW_THEORY_KMIN,
    observed_fit_kmin,
)
from task43_summarize_pk_lightcone import apply_fit_bin_policy, fit_mask


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
MANIFEST = PROJECT_ROOT / "outputs/task43_outputs/rsd_validation/manifests/task43_rsd_validation_lightcone_wide_zobs0p4_1p1_x25.jsonl"


def first_row() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8").splitlines()[0])


def test_wide_manifest_and_dynamic_paths() -> None:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 25
    assert [row["phase"] for row in rows] == [f"ph{index:03d}" for index in range(25)]
    assert all((row["zmin_observed"], row["zmax_observed"]) == (0.4, 1.1) for row in rows)
    assert all(len(row["lightcone_source_paths"]) == 11 for row in rows)
    row = rows[0]
    assert fkp_path(row) == Path(row["lightcone_fkp_path"])
    root = Path(row["lightcone_random_path"]).parent.parent / "formal_gic_windows"
    assert "lightcone_wide_zobs0p4_1p1" in str(output_path("ph000", 200000, 430340, root=root))


def test_wide_volume_selects_exactly_one_extra_task43_bin() -> None:
    row = first_row()
    with np.load(row["lightcone_fkp_path"], allow_pickle=False) as payload:
        volume = float(np.sum(np.asarray(payload["volume_shell"], dtype="f8")))
    kmin = observed_fit_kmin(volume)
    assert WINDOW_THEORY_KMIN < kmin < 0.005
    edges = np.column_stack(
        [np.arange(0.001, 0.3001, 0.002), np.arange(0.003, 0.3021, 0.002)]
    )
    centers = np.mean(edges, axis=1)
    stack = np.ones((25, edges.shape[0]), dtype="f8")
    base = fit_mask(centers, edges, stack, kmin_fit=kmin, kmax_fit=0.10)
    selected, _ = apply_fit_bin_policy(
        base, centers, policy="desi_png", stride=2, pivots=[0.01, 0.02], factors=[2, 2]
    )
    assert np.allclose(edges[selected], TASK43_WIDE_LRGALL_FIT_EDGES, rtol=0.0, atol=1.0e-14)


def test_monopole_only_projection_of_constant_xi() -> None:
    mu_edges = np.linspace(-1.0, 1.0, 41)
    xi_smu = np.ones((4, 40), dtype="f8")
    projected = project_multipoles(xi_smu, mu_edges, (0,))
    assert projected.shape == (1, 4)
    assert np.allclose(projected, 1.0, rtol=0.0, atol=1.0e-15)
