"""Plot-only audit: exact grids, raw mean checks, no inference or input edits."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])

import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
PROD = ROOT / 'outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50'
BASE = PROD / 'mcmc_preliminary_1000_l2'
ABACUS_XI = ROOT / 'outputs/task43_outputs/rsd_validation/lightcone_boxsafe_zobs0p4_0p8/ell2_increment/task43_rsd_boxsafe_x25_mean_xi02_s30_350_ds10.npz'
REF = ROOT / 'outputs/task43_outputs/rsd_validation/lightcone/kmax0p08_smin50_v1'
OUT = PROD / 'diagnostics_grid_checked'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def edge_indices(source, target):
    result = []
    for pair in target:
        found = np.flatnonzero(np.all(np.isclose(source, pair, rtol=0, atol=1e-12), axis=1))
        assert found.size == 1, (pair, found)
        result.append(found[0])
    return np.asarray(result)


def main():
    with np.load(BASE / 'covariance.npz', allow_pickle=False) as z:
        b = {key: np.asarray(z[key]) for key in z.files}
    assert b['stack'].shape == (1000, 74)
    s = b['s']; mask = (s >= 50) & (s < 200) & ~((s >= 80) & (s < 120))
    kept_s = s[mask]
    # The base vector has already had the BAO mask applied. Map by s values.
    base_s = s[b['xi_mask']]
    local = np.asarray([np.flatnonzero(base_s == value).item() for value in kept_s])
    ids = np.r_[np.arange(22), 22 + local, 48 + local]
    data = b['data_joint'][ids]
    bao_full = (s >= 80) & (s < 120)
    bao_s = s[bao_full]
    with np.load(ABACUS_XI, allow_pickle=False) as abacus:
        bao_data = np.asarray(abacus['xi_multipoles_by_phase']).mean(axis=0)[:, bao_full].reshape(-1)
    expected_stack = b['stack'][:, ids]
    assert kept_s.size == 11 and data.size == 44

    # Independently check the Abacus data against the original full-s summary.
    with np.load(ABACUS_XI, allow_pickle=False) as z:
        assert np.array_equal(s, z['s'])
        phases = np.asarray(z['xi_multipoles_by_phase'])
        ab_mean = phases.mean(axis=0)
        nphase = phases.shape[0]
    raw_xi = np.r_[ab_mean[0, mask], ab_mean[1, mask]]
    np.testing.assert_allclose(data[22:], raw_xi, rtol=1e-12, atol=1e-15)
    with np.load(REF / 'fits/rsd_p02/samples.npz', allow_pickle=False) as z:
        np.testing.assert_allclose(data[:22], z['data'], rtol=0, atol=0)

    rows = sorted((json.loads(line) for line in (PROD / 'manifests/task43_ezmock_rsd_covariance_x1000_fixampF_common50.jsonl').read_text().splitlines() if line.strip()), key=lambda row: row['production_index'])
    assert len(rows) == 1000
    raw_stack = []
    raw_bao = []
    for i, row in enumerate(rows):
        assert int(row['production_index']) == int(b['indices'][i])
        with np.load(row['pk_path'], allow_pickle=False) as p, np.load(row['xi_path'], allow_pickle=False) as x:
            assert int(p['seed']) == int(x['seed']) == int(b['seeds'][i])
            assert np.array_equal(x['s'], s)
            i0 = edge_indices(p['k_edges0'], b['edges'])
            i2 = edge_indices(p['k_edges2'], b['edges'])
            raw_stack.append(np.r_[p['pk0'][i0], p['pk2'][i2][b['p2_keep']], x['xi0'][mask], x['xi2'][mask]])
            raw_bao.append(np.r_[x['xi0'][bao_full], x['xi2'][bao_full]])
    raw_stack = np.asarray(raw_stack)
    bao_mock = np.asarray(raw_bao)
    np.testing.assert_allclose(raw_stack, expected_stack, rtol=0, atol=0)
    mean = raw_stack.mean(axis=0); std = raw_stack.std(axis=0, ddof=1)
    assert np.isfinite(raw_stack).all()

    k0 = b['edges'].mean(axis=1); k2 = k0[b['p2_keep']]
    panels = [('p0', k0, slice(0, 13), r'$P_0(k)$', True),
              ('p2', k2, slice(13, 22), r'$P_2(k)$', True),
              ('xi0', kept_s, slice(22, 33), r'$s^2\xi_0(s)$', False),
              ('xi2', kept_s, slice(33, 44), r'$s^2\xi_2(s)$', False)]
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8))
    fig.subplots_adjust(left=.09, right=.98, top=.86, bottom=.10, hspace=.36, wspace=.24)
    clipped_bands = {}; stats = {}
    for ax, (key, x, sl, ylabel, is_pk) in zip(axes.flat, panels):
        scale = np.ones_like(x) if is_pk else x**2
        y = data[sl] * scale; m = mean[sl] * scale; sd = std[sl] * scale
        if is_pk:
            assert np.all(x > 0) and np.all(y > 0) and np.all(m > 0)
            ax.set_xscale('log'); ax.set_yscale('log')
            groups = [np.arange(len(x))]
            lower = m - sd
            valid = lower > 0
            clipped_bands[key] = np.flatnonzero(~valid).tolist()
            ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
            ax.set_ylabel(ylabel + r' $[(h^{-1}\mathrm{Mpc})^3]$')
            ax.set_title(ylabel + '  (log-log)', loc='left')
            ax.set_ylim(min(y.min(), m.min(), lower[valid].min() if valid.any() else m.min()) * .8, max(y.max(), (m+sd).max()) * 1.35)
            ax.set_xlim(x.min()*.90, x.max()*1.12)
        else:
            groups = np.split(np.arange(len(x)), np.flatnonzero(np.diff(x)>11) + 1)
            lower = m-sd; valid = np.ones_like(x, dtype=bool)
            ax.axhline(0, color='.55', lw=.8)
            ax.axvspan(80, 120, color='.93', zorder=0)
            ax.set(xlabel=r'$s\ [h^{-1}\mathrm{Mpc}]$', ylabel=ylabel, xlim=(48, 202))
            ax.set_title(ylabel, loc='left')
        for j, group in enumerate(groups):
            ax.plot(x[group], m[group], color='#cf6500', lw=1.9, marker='s', ms=3.1,
                    label='EZmock mean (1000)' if j==0 else None)
            ax.fill_between(x[group], lower[group], (m+sd)[group], where=valid[group],
                            color='#cf6500', alpha=.18, linewidth=0,
                            label=r'$\pm1\sigma$ realization scatter' if j==0 else None)
            ax.plot(x[group], y[group], 'o-', color='#25364a', lw=1.15, ms=4.1,
                    label=f'Abacus measurement (mean of {nphase})' if j==0 else None)
        if not is_pk:
            ell_offset = 0 if key == 'xi0' else 4
            by = bao_data[ell_offset:ell_offset+4] * bao_s**2
            bm = bao_mock[:,ell_offset:ell_offset+4].mean(axis=0) * bao_s**2
            bs = bao_mock[:,ell_offset:ell_offset+4].std(axis=0, ddof=1) * bao_s**2
            ax.errorbar(bao_s, by, fmt='s', ms=5.0, mfc='white', mec='#25364a',
                        ecolor='#25364a', capsize=2.5, lw=.9, label='BAO points (not fit)')
            ax.errorbar(bao_s, bm, yerr=bs, fmt='s', ms=4.2, color='#cf6500',
                        ecolor='#cf6500', alpha=.85, capsize=2.0, lw=.8,
                        label='EZmock BAO mean (not fit)')
        ax.grid(which='major', alpha=.20); ax.grid(which='minor', alpha=.08)
        ax.legend(frameon=False, fontsize=8.5, loc='best')
        stats[key] = {'coordinate': x.tolist(), 'abacus': data[sl].tolist(), 'ezmock_mean': mean[sl].tolist(),
                      'ezmock_scatter': std[sl].tolist(), 'mean_minus_abacus': (mean[sl]-data[sl]).tolist()}
    fig.suptitle('Measurement vs EZmock mean - grid-checked comparison', fontsize=16, y=.98)
    fig.text(.5,.925, r'RSD $\ell=0,2$ | 1000 mocks | $50\leq s<200$ | BAO interval $80\leq s<120$ omitted', ha='center', fontsize=11)
    fig.text(.5,.03, 'BAO points are displayed for diagnosis only and are not fitted. Shading is mock-to-mock scatter.', ha='center', fontsize=9)
    OUT.mkdir(exist_ok=True)
    out=OUT/'measurement_vs_ezmockmean_smax200_loglog_gridchecked_with_bao.pdf'
    fig.savefig(out); plt.close(fig)
    audit={'input_hashes': {str(BASE/'covariance.npz'):sha(BASE/'covariance.npz'), str(ABACUS_XI):sha(ABACUS_XI)},
           'nmock':1000, 'nphase_abacus':nphase, 'raw_mock_stack_exact_match':True,
           'abacus_raw_xi_mean_match':True, 'full_s_indices':np.flatnonzero(mask).tolist(),
           'correct_compressed_xi_indices':local.tolist(), 'incorrect_previous_abacus_s':base_s[np.flatnonzero(mask)].tolist(),
           'previous_smax200_xi_and_joint_fits_invalid':True, 'fit_changes':'none',
           'bao_points_displayed_not_fitted':True, 'bao_s':bao_s.tolist(),
           'nonpositive_scatter_lower_bound_log_omitted':clipped_bands, 'panels':stats, 'pdf_sha256':sha(out)}
    out.with_suffix('.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({'pdf':str(out), 'nmock':1000, 'checks':'passed', 'P0_lowest_k_mean_ratio':float(mean[0]/data[0]),
                      'xi0_mean_minus_abacus':(mean[22:33]-data[22:33]).tolist(), 'nonpositive_band_bins':clipped_bands},indent=2))


if __name__=='__main__':
    main()
