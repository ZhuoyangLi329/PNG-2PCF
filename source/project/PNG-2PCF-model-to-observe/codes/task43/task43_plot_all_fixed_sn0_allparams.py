#!/usr/bin/env python3
"""Extend the 9.18meeting triangle to every free parameter of sn0=0 fits.

Read audited P02/joint fixed-sn0 chains and the unchanged xi02 chain. Reuse
the reference colors, 68/95% enclosed-mass contours, smoothing and fonts.
Write one 3x3 corner page per covariance and a full provenance/constraint JSON.
The fixed sn0 is annotated rather than given a fictitious posterior panel.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import numpy as np

from task43_plot_ezmock506_covariance_mcmc_vs_jaxpower import (
    COLORS, VARIANTS, draw_contour, interval_text, plot_range,
)
from task43_rsd_common import PROJECT_ROOT, atomic_write_json, sha256_file

PARENT = PROJECT_ROOT / 'outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60'
OLD = PARENT / 'mcmc_preliminary_900'
FIXED = PARENT / 'mcmc_preliminary_900_fixed_sn0_verified'
OUTDIR = PROJECT_ROOT / '9.22meeting/task43_fixed_sn0'
PARAMETERS = ('fNL', 'b1', 'sigma_s')
LABELS = (r'$f_{\rm NL}$', r'$b_1$', r'$\sigma_s\ [h^{-1}{\rm Mpc}]$')
GATES = ('split_rhat_max_below_1p01', 'postburn_length_min_above_50tau',
         'half_chain_shift_max_below_0p1sigma')
PAGES = (('ezmock900', 'EZmock-900 empirical'), ('jaxpower', 'jaxpower analytic'))


def load_chain(covariance, variant):
    root = OLD if variant == 'xi02' else FIXED
    tag = f'{covariance}_{variant}' + ('' if variant == 'xi02' else '_fixed_sn0')
    path = root / f'chain_{tag}.npz'
    summary_path = root / f'summary_{tag}.json'
    summary = json.loads(summary_path.read_text())
    if not all(summary.get('gates', {}).get(gate) is True for gate in GATES):
        raise ValueError(f'Missing/failed convergence gates: {tag}')
    with np.load(path, allow_pickle=False) as payload:
        raw = np.asarray(payload['chain'], dtype='f8')
        data = np.asarray(payload['data'], dtype='f8')
    if raw.ndim != 3 or raw.shape[-1] != 3 or not np.all(np.isfinite(raw)):
        raise ValueError(f'Invalid three-parameter chain: {tag}')
    names = tuple(summary.get('names', PARAMETERS))
    if names != PARAMETERS or set(summary['posterior']) != set(PARAMETERS):
        raise ValueError(f'Parameter-order mismatch: {tag}')
    if variant != 'xi02' and summary.get('fixed_parameters', {}).get('sn0') != 0.:
        raise ValueError(f'Expected sn0=0: {tag}')
    expected_ndata = {'p02': 22, 'xi02': 52, 'joint': 74}[variant]
    if data.shape != (expected_ndata,):
        raise ValueError(f'Data-shape mismatch: {tag}')
    correction = summary['hartlap_percival']
    if covariance == 'jaxpower' and any(correction[k] != 1. for k in ('hartlap', 'percival_m1')):
        raise ValueError(f'Finite-mock correction applied to analytic covariance: {tag}')
    chain = raw.reshape(-1, 3)
    median = np.median(chain, axis=0)
    reported = np.array([summary['posterior'][p]['q50'] for p in PARAMETERS])
    if not np.allclose(median, reported, rtol=0., atol=1e-10):
        raise ValueError(f'Summary/chain mismatch: {tag}')
    return {'chain': chain, 'data': data, 'summary': summary, 'path': path,
            'shape': list(raw.shape), 'summary_path': summary_path}


def corner_page(pdf, covariance, display, entries, ranges):
    fig, axes = plt.subplots(3, 3, figsize=(9.3, 8.9), squeeze=False)
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if j > i:
                ax.set_axis_off()
                continue
            ax.set_xlim(*ranges[PARAMETERS[j]])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(labelsize=10, labelbottom=(i == 2))
            if i == j:
                bins = np.linspace(*ranges[PARAMETERS[i]], 90)
                for variant, key, _ in VARIANTS:
                    ax.hist(entries[f'{covariance}_{variant}']['chain'][:, i],
                            bins=bins, density=True, histtype='step', lw=1.8,
                            color=COLORS[key])
                ax.set_yticks([])
            else:
                ax.set_ylim(*ranges[PARAMETERS[i]])
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.tick_params(labelleft=(j == 0))
                for n, (variant, key, _) in enumerate(VARIANTS):
                    chain = entries[f'{covariance}_{variant}']['chain']
                    draw_contour(ax, chain[:, j], chain[:, i], color=COLORS[key],
                                 xlim=ranges[PARAMETERS[j]], ylim=ranges[PARAMETERS[i]],
                                 zorder=2+2*n)
                if j == 0:
                    ax.set_ylabel(LABELS[i], fontsize=13)
            if j == 0:
                ax.axvline(0., color='0.55', lw=.8, ls='--', zorder=0)
            if i == 2:
                ax.set_xlabel(LABELS[j], fontsize=13)
    handles, labels = [], []
    for variant, key, label in VARIANTS:
        entry = entries[f'{covariance}_{variant}']
        handles.append(plt.Line2D([], [], color=COLORS[key], lw=2.))
        labels.append(label+'\n'+rf'$f_{{\rm NL}}={interval_text(entry["chain"][:, 0], entry["summary"]["map_theta"][0])}$')
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.96, .918),
               frameon=False, fontsize=16., labelspacing=.65, handletextpad=.7)
    fig.suptitle(f'Task43 RSD lightcone - {display} covariance', fontsize=14., y=.98)
    fig.text(.5, .944, r'$k_{\max}=0.08\,h\,\mathrm{Mpc}^{-1}$; '
             r'$s_{\min}=50\,h^{-1}\mathrm{Mpc}$; BAO mask 80-120; '
             r'$s_{n0}=0$ (fixed)', ha='center', fontsize=10.5)
    fig.text(.976, .52, '68% / 95% contours\n'+r'Shared $f_{\rm NL}, b_1, \sigma_s$ in joint',
             ha='right', va='center', fontsize=10.5, color='0.35')
    fig.text(.5, .013, 'Legend: ML with raw-chain 68% interval; contours use raw chains, as in 9.18meeting.',
             ha='center', fontsize=8., color='0.4')
    fig.subplots_adjust(left=.09, right=.98, bottom=.082, top=.91, wspace=.08, hspace=.08)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    output = OUTDIR / 'task43_ezmock900_all_variants_fixed_sn0_allparams_9.18style.pdf'
    if output.exists() and not args.force:
        raise FileExistsError(output)
    entries = {f'{c}_{v}': load_chain(c, v) for c, _ in PAGES for v, _, _ in VARIANTS}
    for c, _ in PAGES:
        if not np.array_equal(entries[f'{c}_joint']['data'],
                              np.r_[entries[f'{c}_p02']['data'], entries[f'{c}_xi02']['data']]):
            raise ValueError('Joint and marginal datasets do not match')
    ranges = {}
    for i, p in enumerate(PARAMETERS):
        samples = np.concatenate([e['chain'][:, i] for e in entries.values()])
        ranges[p] = plot_range(samples, samples, parameter=p)
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 10.,
                         'axes.linewidth': 1., 'xtick.direction': 'in',
                         'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
    OUTDIR.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.stem}.{os.getpid()}.tmp.pdf')
    with PdfPages(temporary) as pdf:
        for c, display in PAGES:
            corner_page(pdf, c, display, entries, ranges)
    temporary.replace(output)
    constraints = {}
    for tag, e in entries.items():
        s = e['summary']
        constraints[tag] = {'chain_npz': str(e['path']), 'chain_sha256': sha256_file(e['path']),
                            'summary_json': str(e['summary_path']), 'chain_shape': e['shape'],
                            'gates': s['gates'], 'hartlap_percival': s['hartlap_percival'],
                            'parameters': {p: {'maximum_likelihood': float(s['map_theta'][i]),
                                               'raw_plot_quantiles': np.percentile(e['chain'][:, i], [16, 50, 84]).tolist(),
                                               'reported_posterior': s['posterior'][p]}
                                           for i, p in enumerate(PARAMETERS)}}
    audit = {'task': 'all free-parameter contours for consistent sn0=0 fits', 'status': 'pass',
             'fixed_parameters': {'sn0': 0.}, 'contour_parameters': PARAMETERS,
             'parameter_pairs': [['fNL', 'b1'], ['fNL', 'sigma_s'], ['b1', 'sigma_s']],
             'layout': '3x3 lower triangle; 3 one-dimensional marginals per page',
             'pages': [d for _, d in PAGES], 'style_reference': '9.18meeting/task43_ezmock_covariance_mcmc_vs_jaxpower',
             'contour_probabilities': [.68, .95], 'plot_convention': 'raw chain, original 9.18 style; Percival-corrected summaries separately recorded',
             'shared_axis_ranges': ranges, 'constraints': constraints,
             'data_contract': 'P0 13 + P2 9 + xi0 26 + xi2 26; full P-xi cross; C_single, not divided by 25',
             'output_pdf': str(output), 'output_pdf_sha256': sha256_file(output)}
    atomic_write_json(output.with_suffix('.json'), audit)
    print(json.dumps({'output': str(output), 'sha256': audit['output_pdf_sha256'],
                      'gates': {k: e['summary']['gates'] for k, e in entries.items()}}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
