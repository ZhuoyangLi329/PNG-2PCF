import os, json
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
PROD=ROOT/'outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50'
RB=ROOT/'outputs/task43_outputs/rsd_validation/rawbox'
pk=sorted((PROD/'pk_jaxpower').glob('ezmock_m*_seed610*_p02_mesh256.npz'))
xi=sorted((PROD/'xi_fcfc').glob('ezmock_m*_seed610*_xi02_s30_350_ds10.npz'))
assert len(pk)==len(xi)==10
with np.load(pk[0],allow_pickle=False) as z: k=np.asarray(z['k'])
p0=[]; p2=[]
for f in pk:
 with np.load(f,allow_pickle=False) as z: p0.append(z['pk0']); p2.append(z['pk2'])
p0=np.asarray(p0); p2=np.asarray(p2)
with np.load(xi[0],allow_pickle=False) as z: s=np.asarray(z['s'])
x0=[]; x2=[]
for f in xi:
 with np.load(f,allow_pickle=False) as z: x0.append(z['xi0']); x2.append(z['xi2'])
x0=np.asarray(x0); x2=np.asarray(x2)
abpk=sorted((RB/'pk').glob('task43_rsd_rawbox_p02_AbacusSummit_base_c000_ph*_mmin1p4e13_mesh400.npz'))
assert len(abpk)==25
ap0=[]; ap2=[]
for f in abpk:
 with np.load(f,allow_pickle=False) as z: ap0.append(z['pk0']); ap2.append(z['pk2'])
ap0=np.asarray(ap0); ap2=np.asarray(ap2)
with np.load(RB/'closure/task43_rsd_rawbox_x25_fulldiscrete_lorentzian.npz',allow_pickle=False) as z: s_ab=np.asarray(z['s']); ax0=np.asarray(z['xi0_rsd_mean']); ax2=np.asarray(z['xi2_rsd_mean'])
assert np.array_equal(s,s_ab)
data={'P0':(k,p0.mean(0),p0.std(0,ddof=1),ap0.mean(0)),'P2':(k,p2.mean(0),p2.std(0,ddof=1),ap2.mean(0)),'xi0':(s,x0.mean(0),x0.std(0,ddof=1),ax0),'xi2':(s,x2.mean(0),x2.std(0,ddof=1),ax2)}
fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
for ax,(name,(xx,mean,std,ab)) in zip(axes.flat,data.items()):
 scale=np.ones_like(xx) if name.startswith('P') else xx**2; ym=mean*scale; ys=std*scale; ya=ab*scale; positive=(xx>0)&(ym>0)&(ya>0) if name.startswith('P') else np.ones_like(xx,dtype=bool)
 if name.startswith('P'): ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$'); ax.set_ylabel(rf'${name}(k)$')
 else: ax.set_xlabel(r'$s\ [h^{-1}\mathrm{Mpc}]$'); ax.set_ylabel(rf'$s^2{name}(s)$')
 xx=xx[positive];ym=ym[positive];ys=ys[positive];ya=ya[positive]
 ax.plot(xx,ym,color='#cf6500',lw=1.8,label='EZmock mean (N=10)'); ax.fill_between(xx,ym-ys,ym+ys,color='#cf6500',alpha=.2,lw=0,label='EZmock ±1σ'); ax.plot(xx,ya,color='#25364a',lw=1.6,marker='o',ms=3.2,label='Abacus rawbox mean (N=25)'); ax.set_title(name); ax.grid(alpha=.18,which='both'); ax.legend(frameon=False,fontsize=8)
fig.suptitle('Rawbox EZmock versus Abacus diagnostic · first available 10 EZmocks',fontsize=14)
fig.text(.5,.01,'Rawbox currently contains 10 completed EZmock P/xi pairs, not 20; no fitting or normalization applied.',ha='center',fontsize=9)
out=PROD/'diagnostics_grid_checked/rawbox_first10_ezmock_vs_abacus.pdf'; out.parent.mkdir(exist_ok=True); fig.savefig(out); plt.close(fig); out.with_suffix('.json').write_text(json.dumps({'n_ezmock':10,'n_abacus':25,'requested':20,'pdf':str(out)},indent=2)); print(out)
