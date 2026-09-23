#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task43 EZmock RSD 手动调参 notebook 回放脚本。

逐 cell 复刻 codes/task43/task43_ezmock_rsd_manual_tuning.ipynb 的执行流程：
参数覆盖 -> EZmock 生成 + FCFC/jaxpower 测量 -> Abacus raw-box 目标比较 ->
2x2 对比图。唯一区别是图像改为 savefig 输出 PDF 到 test_figure/，而不是
只在 kernel 中显示；数据仍写入 notebook 自带的运行目录以保持行为一致。

用法（登录节点，限 8 核）:
    OMP_NUM_THREADS=1 taskset -c 6-13 \
        /global/homes/l/lzy/anaconda3/envs/kirisame/bin/python -u \
        codes/task43/task43_ezmock_rsd_manual_tuning_replay_figure.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
CODE_DIR = ROOT / 'codes/task43'
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import task43_ezmock_rsd_manual_tuning as tuning
from task43_ezmock_rsd_manual_tuning import (
    build_ezmock, write_rsd_catalog, measure_fcfc, measure_jaxpower,
    compare_to_targets, load_target, OUTPUT_ROOT,
)

# ---------- cell 1: 与 notebook 完全一致的默认参数 ----------
rho_c = 1.14
rho_exp = 5.0
pdf_base = 0.25
sigma_v = 0.0

box_size = 2000.0
redshift = 0.725
omega_m = 0.3137721026737606
ngrid = 320
ntracer = 1_297_050
fix_amplitude = True
attach_particle = True
rand_generator = 1
invert_phase = False
pk_interp_log = True
bao_enhance = 0.0

rsd_fac_override = None

nreal = 15
seed_base = 433000
threads = 8
pk_mesh_size = 400
xi_smin, xi_smax, xi_ds = 30.0, 350.0, 10.0
mu_bins = 120
pk_kmin, pk_kmax, pk_dk = 0.001, 0.3001, 0.002

target_xi_path = str(ROOT / 'outputs/task43_outputs/rsd_validation/rawbox/summary_x25/task43_rsd_rawbox_closure_x25.npz')
target_pk_dir = str(ROOT / 'outputs/task43_outputs/rsd_validation/rawbox/pk')

# ---------- cell 2: 覆盖后端默认值 ----------
for name, value in {
    'BOX_SIZE': box_size, 'REDSHIFT': redshift, 'OMEGA_M': omega_m,
    'NGRID': ngrid, 'FIX_AMPLITUDE': fix_amplitude,
    'ATTACH_PARTICLE': attach_particle, 'RAND_GENERATOR': rand_generator,
    'INVERT_PHASE': invert_phase, 'PK_INTERP_LOG': pk_interp_log,
    'BAO_ENHANCE': bao_enhance, 'S_MIN': xi_smin, 'S_MAX': xi_smax,
    'DS': xi_ds, 'MU_BINS': mu_bins, 'KMIN': pk_kmin, 'KMAX': pk_kmax,
    'DK': pk_dk, 'PK_MESHSIZE': pk_mesh_size,
}.items():
    setattr(tuning, name, value)
if rsd_fac_override is not None:
    tuning.rsd_factor = lambda: float(rsd_fac_override)

params = (rho_c, rho_exp, pdf_base, sigma_v)
label = f'notebook_rsd_c{rho_c:g}_e{rho_exp:g}_b{pdf_base:g}_v{sigma_v:g}_x{nreal}'
run_root = OUTPUT_ROOT / label
catalog_dir = run_root / 'catalogs'
measurement_dir = run_root / 'measurements'
catalog_dir.mkdir(parents=True, exist_ok=True)
measurement_dir.mkdir(parents=True, exist_ok=True)
print('当前完整设置已加载:', params, 'box=', box_size, 'z=', redshift, 'RSD=', True)
print('输出数据目录:', run_root)
print('本回放脚本把图像保存为 PDF 到 test_figure/；notebook 原版只显示不写盘。')
print('CPU affinity:', sorted(os.sched_getaffinity(0)))


def load_abacus_rawbox_target(xi_path, pk_dir):
    """Load immutable Abacus raw-box x25 xi02 and P02 measurements."""
    xi_path = Path(xi_path)
    pk_dir = Path(pk_dir)
    with np.load(xi_path, allow_pickle=False) as data:
        xi_by_phase = np.asarray(data['xi02_rsd_by_phase'], dtype='f8')
        xi_mean = np.asarray(data['xi02_rsd_mean'], dtype='f8')
        s_target = np.asarray(data['s'], dtype='f8')
        s_edges_target = np.asarray(data['s_edges'], dtype='f8')
        phases = np.asarray(data['phases']).astype(str)
    if xi_by_phase.ndim != 3 or xi_by_phase.shape[1:] != (2, s_target.size):
        raise ValueError(f'Unexpected Abacus xi02 shape: {xi_by_phase.shape}')

    pk_files = sorted(pk_dir.glob(
        'task43_rsd_rawbox_p02_AbacusSummit_base_c000_ph*_mmin1p4e13_mesh400.npz'
    ))
    if len(pk_files) != phases.size:
        raise RuntimeError(f'Expected {phases.size} Abacus P02 files, found {len(pk_files)}')
    pk_records = []
    for path in pk_files:
        with np.load(path, allow_pickle=False) as data:
            pk_records.append({
                'k': np.asarray(data['k'], dtype='f8'),
                'k_edges': np.asarray(data['k_edges'], dtype='f8'),
                'pk0': np.asarray(data['pk0'], dtype='f8'),
                'pk2': np.asarray(data['pk2'], dtype='f8'),
                'phase': str(np.asarray(data['phase']).item()),
            })
    pk_records.sort(key=lambda item: item['phase'])
    pk0_all = np.asarray([item['pk0'] for item in pk_records], dtype='f8')
    pk2_all = np.asarray([item['pk2'] for item in pk_records], dtype='f8')
    k_all = np.asarray([item['k'] for item in pk_records], dtype='f8')
    k_edges_all = np.asarray([item['k_edges'] for item in pk_records], dtype='f8')
    if not np.allclose(k_all, k_all[0][None, :], equal_nan=True):
        raise RuntimeError('Abacus raw-box P02 k coordinates differ between phases')
    if not np.allclose(k_edges_all, k_edges_all[0][None, :, :], equal_nan=True):
        raise RuntimeError('Abacus raw-box P02 k edges differ between phases')
    return {
        'phases': phases,
        's': s_target,
        's_edges': s_edges_target,
        'xi0_all': xi_by_phase[:, 0, :],
        'xi2_all': xi_by_phase[:, 1, :],
        'xi0_mean': xi_mean[0],
        'xi2_mean': xi_mean[1],
        'xi0_std': np.std(xi_by_phase[:, 0, :], axis=0, ddof=1),
        'xi2_std': np.std(xi_by_phase[:, 1, :], axis=0, ddof=1),
        'k': k_all[0],
        'k_edges': k_edges_all[0],
        'pk0_all': pk0_all,
        'pk2_all': pk2_all,
        'pk0_mean': np.mean(pk0_all, axis=0),
        'pk2_mean': np.mean(pk2_all, axis=0),
        'pk0_std': np.std(pk0_all, axis=0, ddof=1),
        'pk2_std': np.std(pk2_all, axis=0, ddof=1),
    }


# ---------- cell 3: 生成 EZmock RSD 并测量 P0/P2、xi0/xi2 ----------
started = time.perf_counter()
xi_rows, pk0_rows, pk2_rows = [], [], []
for i in range(int(nreal)):
    seed = int(seed_base) + i + 1
    catalog = catalog_dir / f'ezmock_rsd_seed{seed}.dat'
    ez = build_ezmock(seed)
    write_rsd_catalog(ez, catalog, params, ntracer)
    xi = measure_fcfc(catalog, measurement_dir, f'fcfc_seed{seed}', threads)
    pk = measure_jaxpower(catalog, seed, measurement_dir)
    xi_rows.append(np.column_stack((xi['xi0'], xi['xi2'])))
    pk0_rows.append(pk['pk0'])
    pk2_rows.append(pk['pk2'])
    print(f'完成 realization {i+1}/{nreal}, seed={seed}', flush=True)

xi_stack = np.asarray(xi_rows, dtype='f8')
pk0_stack = np.asarray(pk0_rows, dtype='f8')
pk2_stack = np.asarray(pk2_rows, dtype='f8')
xi_mean = xi_stack.mean(axis=0)
pk0_mean = pk0_stack.mean(axis=0)
pk2_mean = pk2_stack.mean(axis=0)
s = xi['s']
k0 = pk['k0']
print('xi0/xi2 shape:', xi_stack.shape)
print('P0/P2 shape:', pk0_stack.shape)

# ---------- cell 4: Abacus raw-box target comparison ----------
abacus_target = load_abacus_rawbox_target(target_xi_path, target_pk_dir)
if not np.allclose(s, abacus_target['s']):
    raise RuntimeError(f's-grid mismatch: EZmock={s.shape}, Abacus={abacus_target["s"].shape}')

target_xi = {
    'xi0_rsd_mean': abacus_target['xi0_mean'],
    'xi2_rsd_mean': abacus_target['xi2_mean'],
}
target_pk = {
    'pk0_rsd_mean': abacus_target['pk0_mean'],
    'pk2_rsd_mean': abacus_target['pk2_mean'],
}


def interpolate_rows_to_grid(rows, k_source, k_target):
    """Interpolate EZmock rows to the immutable Abacus k coordinates for RMS only."""
    rows = np.asarray(rows, dtype='f8')
    k_source = np.asarray(k_source, dtype='f8')
    k_target = np.asarray(k_target, dtype='f8')
    result = np.full((rows.shape[0], k_target.size), np.nan, dtype='f8')
    source_k_mask = np.isfinite(k_source)
    target_k_mask = np.isfinite(k_target)
    for index, row in enumerate(rows):
        mask = source_k_mask & np.isfinite(row)
        if np.count_nonzero(mask) < 2:
            continue
        order = np.argsort(k_source[mask])
        result[index, target_k_mask] = np.interp(
            k_target[target_k_mask], k_source[mask][order], row[mask][order],
            left=np.nan, right=np.nan
        )
    return result


# The target values stay untouched.  If the EZmock k-grid differs, only the
# EZmock rows are interpolated onto the target coordinates for a diagnostic RMS.
k_grid_direct_match = np.allclose(k0, abacus_target['k'], equal_nan=True)
pk0_on_target_k = interpolate_rows_to_grid(pk0_stack, k0, abacus_target['k'])
pk2_on_target_k = interpolate_rows_to_grid(pk2_stack, k0, abacus_target['k'])
target_metrics = compare_to_targets(
    xi_stack, pk0_on_target_k, pk2_on_target_k, target_xi, target_pk
)
for metric, value, reference in (
    ('pk0', np.nanmean(pk0_on_target_k, axis=0), abacus_target['pk0_mean']),
    ('pk2', np.nanmean(pk2_on_target_k, axis=0), abacus_target['pk2_mean']),
):
    mask = np.isfinite(value) & np.isfinite(reference)
    target_metrics[f'{metric}_mean_absolute_rms'] = float(
        np.sqrt(np.mean((value[mask] - reference[mask]) ** 2))
    )
print('Abacus raw-box target:', target_xi_path)
print('Abacus raw-box P02 directory:', target_pk_dir)
print('Abacus phases:', abacus_target['phases'])
print('Abacus xi/P shapes:', abacus_target['xi0_all'].shape, abacus_target['pk0_all'].shape)
print('Direct EZmock/Abacus k-grid match:', k_grid_direct_match)
print('RMS uses EZmock interpolated to Abacus k coordinates; Abacus values are unchanged.')
print('EZmock vs Abacus direct RMS:', target_metrics)

# ---------- cell 5: Abacus raw-box mean vs EZmock mean ----------
# Only mean curves are shown so individual realizations do not clutter the plot
# or stretch the automatic y limits.
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

# xi0
axes[0, 0].plot(
    abacus_target['s'], abacus_target['s']**2 * abacus_target['xi0_mean'],
    color='#222222', lw=2.5, label='Abacus raw-box x25 mean'
)
axes[0, 0].plot(
    s, s**2 * xi_mean[:, 0],
    color='#d62728', lw=2.5, ls='--', label=f'EZmock x{nreal} mean'
)
axes[0, 0].axhline(0, color='0.45', lw=0.8, ls=':')
axes[0, 0].set(
    xlabel=r'$s\ [h^{-1}\,Mpc]$', ylabel=r'$s^2\xi_0(s)$',
    title=r'Raw-box RSD $\xi_0$'
)
axes[0, 0].legend(frameon=False)

# xi2
axes[0, 1].plot(
    abacus_target['s'], abacus_target['s']**2 * abacus_target['xi2_mean'],
    color='#222222', lw=2.5, label='Abacus raw-box x25 mean'
)
axes[0, 1].plot(
    s, s**2 * xi_mean[:, 1],
    color='#d62728', lw=2.5, ls='--', label=f'EZmock x{nreal} mean'
)
axes[0, 1].axhline(0, color='0.45', lw=0.8, ls=':')
axes[0, 1].set(
    xlabel=r'$s\ [h^{-1}\,Mpc]$', ylabel=r'$s^2\xi_2(s)$',
    title=r'Raw-box RSD $\xi_2$'
)
axes[0, 1].legend(frameon=False)

# P0; each mean keeps its own k coordinate.
valid_abacus_p0 = np.isfinite(abacus_target['k']) & np.isfinite(abacus_target['pk0_mean'])
valid_ezmock_p0 = np.isfinite(k0) & np.isfinite(pk0_mean)
axes[1, 0].plot(
    abacus_target['k'][valid_abacus_p0], abacus_target['pk0_mean'][valid_abacus_p0],
    color='#222222', lw=2.5, label='Abacus raw-box x25 mean'
)
axes[1, 0].plot(
    k0[valid_ezmock_p0], pk0_mean[valid_ezmock_p0],
    color='#d62728', lw=2.5, ls='--', label=f'EZmock x{nreal} mean'
)
p0_visible = np.concatenate((
    abacus_target['pk0_mean'][valid_abacus_p0],
    pk0_mean[valid_ezmock_p0],
))
p0_visible = p0_visible[np.isfinite(p0_visible) & (p0_visible > 0.0)]
axes[1, 0].set(
    xscale='log', yscale='log', xlabel=r'$k\ [h\,Mpc^{-1}]$',
    ylabel=r'$P_0(k)$', title=f'Raw-box RSD $P_0$ (EZmock mesh={pk_mesh_size})'
)
axes[1, 0].set_ylim(0.85 * p0_visible.min(), 1.15 * p0_visible.max())
axes[1, 0].legend(frameon=False, loc='lower left')

# P2
valid_abacus_p2 = np.isfinite(abacus_target['k']) & np.isfinite(abacus_target['pk2_mean'])
valid_ezmock_p2 = np.isfinite(k0) & np.isfinite(pk2_mean)
axes[1, 1].plot(
    abacus_target['k'][valid_abacus_p2], abacus_target['pk2_mean'][valid_abacus_p2],
    color='#222222', lw=2.5, label='Abacus raw-box x25 mean'
)
axes[1, 1].plot(
    k0[valid_ezmock_p2], pk2_mean[valid_ezmock_p2],
    color='#d62728', lw=2.5, ls='--', label=f'EZmock x{nreal} mean'
)
p2_visible = np.concatenate((
    abacus_target['pk2_mean'][valid_abacus_p2],
    pk2_mean[valid_ezmock_p2],
))
p2_visible = p2_visible[np.isfinite(p2_visible)]
if np.all(p2_visible > 0.0):
    axes[1, 1].set(
        xscale='log', yscale='log', xlabel=r'$k\ [h\,Mpc^{-1}]$',
        ylabel=r'$P_2(k)$', title=f'Raw-box RSD $P_2$ (EZmock mesh={pk_mesh_size})'
    )
    axes[1, 1].set_ylim(0.85 * p2_visible.min(), 1.15 * p2_visible.max())
else:
    # Keep signed P2 visible if a trial parameter set produces both signs.
    p2_lo, p2_hi = p2_visible.min(), p2_visible.max()
    p2_pad = 0.15 * (p2_hi - p2_lo)
    axes[1, 1].set(
        xscale='log', yscale='symlog', xlabel=r'$k\ [h\,Mpc^{-1}]$',
        ylabel=r'$P_2(k)$', title=f'Raw-box RSD $P_2$ (EZmock mesh={pk_mesh_size})'
    )
    axes[1, 1].set_ylim(p2_lo - p2_pad, p2_hi + p2_pad)
axes[1, 1].legend(frameon=False, loc='lower left')

fig.suptitle(
    f'Abacus raw-box vs EZmock RSD means: rho_c={rho_c:g}, '
    f'rho_exp={rho_exp:g}, pdf_base={pdf_base:g}, sigma_v={sigma_v:g}'
)

# notebook 原版这里是 plt.show()；回放按仓库规范只输出 PDF。
out_pdf = ROOT / 'test_figure' / (
    f'task43_ezmock_rsd_manual_tuning_c{rho_c:g}_e{rho_exp:g}'
    f'_b{pdf_base:g}_v{sigma_v:g}_x{nreal}_vs_abacus_x25.pdf'
)
out_pdf.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_pdf)
print('已保存图像:', out_pdf)
print(f'总耗时: {time.perf_counter() - started:.1f} s')
