#!/usr/bin/env python3
"""Validate and reuse the completed analytic baseline without sampling again."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
from contextlib import ExitStack

os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
sys.path.insert(0, '/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/codes/task43')
import task43_rsd_ezmock281_inference as inference
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--commit', action='store_true')
    args=parser.parse_args()
    old=inference.PRODUCTION/'mcmc_preliminary_281_long'
    new=inference.PRODUCTION/'mcmc_preliminary_398'
    with ExitStack() as stack:
        for root in (old,new):
            handle=stack.enter_context((root/'run.lock').open('r'))
            fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
        oa=json.loads((old/'freeze.json').read_text())
        na=json.loads((new/'freeze.json').read_text())
        assert oa['nmock']==281 and na['nmock']==398
        assert oa['source_hashes']==na['source_hashes'], 'reference input hashes differ'
        for root,audit in ((old,oa),(new,na)):
            assert inference.digest(root/'covariance.npz')==audit['payload_sha256']
        with np.load(old/'covariance.npz',allow_pickle=False) as x, np.load(new/'covariance.npz',allow_pickle=False) as y:
            for key in ('covariance_jaxpower','edges','p2_keep','s','xi_mask','k_reference',
                        'data_p02','data_xi02','data_joint','phase_data_p02','phase_data_xi02','phase_data_joint'):
                assert np.array_equal(x[key], y[key]), key
        old_model=json.loads((old/'fits/model_audit.json').read_text())
        new_model=json.loads((new/'fits/model_audit.json').read_text())
        model_equal=old_model==new_model
        if not model_equal:
            differences=[key for key in old_model.keys()|new_model.keys() if old_model.get(key)!=new_model.get(key)]
            raise RuntimeError(f'model audit differs: {differences}')
        results={}
        for family,root,audit in (('ezmock281',new,na),('jaxpower',old,oa)):
            for variant in inference.VARIANTS:
                tag=f'{family}_{variant}'
                folder=root/'fits'/tag
                summary=json.loads((folder/'summary.json').read_text())
                assert summary['chains_converged'],tag
                if family=='ezmock281':
                    assert summary['factors']==na['corrections'][variant]
                else:
                    assert all(summary['factors'][k]==1 for k in ('hartlap','percival_m1','sigma_factor'))
                files=sorted(folder.glob('run*.npz'))
                assert len(files)==4,tag
                for chain in files:
                    info=json.loads(chain.with_suffix('.json').read_text())
                    assert info['chain_sha256']==inference.digest(chain)
                    assert info['config']['freeze_sha256']==audit['payload_sha256']
                    assert all(info['mcmc']['gates'].values())
                results[tag]=summary
        metrics={}
        for family in ('ezmock281','jaxpower'):
            widths={v:results[f'{family}_{v}']['posterior_corrected']['fNL']['sigma68'] for v in inference.VARIANTS}
            metrics[family]=dict(sigma68_fNL=widths,
                joint_gain_over_p_fraction=1-widths['joint']/widths['p02'],
                joint_gain_over_xi_fraction=1-widths['joint']/widths['xi02'],
                joint_gain_over_best_fraction=1-widths['joint']/min(widths['p02'],widths['xi02']))
        provenance=dict(source_root=str(old),source_freeze_sha256=inference.digest(old/'freeze.json'),
            source_payload_sha256=oa['payload_sha256'],source_hashes=oa['source_hashes'],
            source_model_audit_sha256=inference.digest(old/'fits/model_audit.json'),
            analytic_covariance_and_data_exact_match=True,model_audit_exact_match=model_equal,
            policy='Reuse all three completed analytic posterior chains byte-for-byte; no new sampling.',
            copied_file_hashes={str(p.relative_to(old/'fits')):inference.digest(p)
                for v in inference.VARIANTS for p in (old/'fits'/f'jaxpower_{v}').iterdir() if p.is_file()})
        comparison=dict(status='pass',nmock=398,results=results,metrics=metrics,analytic_reuse=provenance)
        print(json.dumps(dict(status='validated',cpu_affinity=sorted(os.sched_getaffinity(0)),
            fNL={k:s['posterior_corrected']['fNL'] for k,s in results.items()},metrics=metrics),indent=2),flush=True)
        if not args.commit:
            return
        for variant in inference.VARIANTS:
            tag=f'jaxpower_{variant}'
            src=old/'fits'/tag
            dest=new/'fits'/tag
            # Retain discarded duplicate outputs recoverably outside the active fits.
            if dest.exists():
                backup=new/'superseded_duplicate_analytic'/tag
                if backup.exists():
                    assert all(inference.digest(dest/p.name)==inference.digest(p) for p in src.iterdir() if p.is_file())
                    continue
                backup.parent.mkdir(exist_ok=True)
                dest.rename(backup)
            shutil.copytree(src,dest)
        for rel,expected in provenance['copied_file_hashes'].items():
            assert inference.digest(new/'fits'/rel)==expected
        inference.write_json(new/'analytic_reuse.json',provenance)
        inference.write_json(new/'fits/comparison.json',comparison)
        inference.plot(new)
        print('Finalized current EZmock 398 and reused analytic baseline.',flush=True)


if __name__=='__main__':
    main()
