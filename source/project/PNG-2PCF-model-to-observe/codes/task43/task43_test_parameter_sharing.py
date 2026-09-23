#!/usr/bin/env python3
"""ML-only diagnostics for the P02/xi02 parameter-sharing assumptions."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import task43_diagnose_joint_b1 as q
import task43_rsd_ezmock281_mcmc_compare as r

RUN = r.PROJECT_ROOT / 'outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60/mcmc_preliminary_900'
OUT = RUN / 'diagnostics/joint_b1/task43_parameter_sharing_diagnostic.json'

def main():
    pm, xm, dp, dx, _, _ = r.build_models_and_data()
    data = np.r_[dp, dx]
    def p(f, b, s, n):
        a = pm.evaluate(np.asarray([f, b, s, n]))
        return np.r_[a[:13], a[13:26][r.P2_KEEP]]
    def x(f, b, s):
        a = xm.evaluate(np.asarray([f, b, s]), model='formal_gic', window_key='mean')
        return np.r_[a[0][r.XI_MASK], a[2][r.XI_MASK]]
    base = r.OPTIMIZER_STARTS
    out = {}
    with np.load(RUN/'ezmock900_covariance_and_stack.npz') as d:
        cov = np.asarray(d['covariance'], dtype='f8')

    def run(name, fn, starts, names, lo, hi, cc=cov):
        rr, tt = q.fit(fn, data, cc, starts, names, np.asarray(lo), np.asarray(hi))
        out[name] = rr
        print(name, json.dumps(rr), flush=True)

    run('shared_all', lambda t: np.r_[p(t[0], t[1], t[2], t[3]), x(t[0], t[1], t[2])], base,
        ['fNL','b1','sigma_s','sn0'], r.BOUNDS_LO, r.BOUNDS_HI)
    run('split_sigma', lambda t: np.r_[p(t[0], t[1], t[2], t[4]), x(t[0], t[1], t[3])],
        [np.r_[s[:2], 2.5, 5.5, s[3]] for s in base], ['fNL','b1','sigma_s_P','sigma_s_xi','sn0'],
        np.r_[r.BOUNDS_LO[:2],0.,0.,r.BOUNDS_LO[3]], np.r_[r.BOUNDS_HI[:2],30.,30.,r.BOUNDS_HI[3]])
    run('split_fNL_shared_b1', lambda t: np.r_[p(t[0], t[2], t[3], t[5]), x(t[1], t[2], t[4])],
        [np.r_[s[0], s[0], s[1], 2.5, 5.5, s[3]] for s in base],
        ['fNL_P','fNL_xi','b1','sigma_s_P','sigma_s_xi','sn0'],
        [-500.,-500.,.5,0.,0.,-1.], [500.,500.,5.,30.,30.,1.])
    run('split_b1_shared_fNL', lambda t: np.r_[p(t[0], t[1], t[3], t[5]), x(t[0], t[2], t[4])],
        [np.r_[s[0], s[1], s[1], 2.5, 5.5, s[3]] for s in base],
        ['fNL','b1_P','b1_xi','sigma_s_P','sigma_s_xi','sn0'],
        [-500.,.5,.5,0.,0.,-1.], [500.,5.,5.,30.,30.,1.])
    run('split_fNL_b1', lambda t: np.r_[p(t[0], t[1], t[4], t[6]), x(t[2], t[3], t[5])],
        [np.r_[s[0], s[1], s[0], s[1], 2.5, 5.5, s[3]] for s in base],
        ['fNL_P','b1_P','fNL_xi','b1_xi','sigma_s_P','sigma_s_xi','sn0'],
        [-500.,.5,-500.,.5,0.,0.,-1.], [500.,5.,500.,5.,30.,30.,1.])
    run('fixed_sn0_shared_sigma', lambda t: np.r_[p(t[0], t[1], t[2], 0.), x(t[0], t[1], t[2])],
        [s[:3] for s in base], ['fNL','b1','sigma_s'], r.BOUNDS_LO[:3], r.BOUNDS_HI[:3])
    run('fixed_sn0_split_sigma', lambda t: np.r_[p(t[0], t[1], t[2], 0.), x(t[0], t[1], t[3])],
        [np.r_[s[:2], 2.5, 5.5] for s in base], ['fNL','b1','sigma_s_P','sigma_s_xi'],
        [-500.,.5,0.,0.], [500.,5.,30.,30.])

    cross = cov[:22,22:].copy()
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cc = cov.copy(); cc[:22,22:] = alpha*cross; cc[22:,:22] = alpha*cross.T
        run(f'cross_alpha_{alpha:.2f}', lambda t: np.r_[p(t[0], t[1], t[2], t[3]), x(t[0], t[1], t[2])],
            base, ['fNL','b1','sigma_s','sn0'], r.BOUNDS_LO, r.BOUNDS_HI, cc)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2)+'\n')
    print('OUTPUT', OUT, flush=True)

if __name__ == '__main__': main()
