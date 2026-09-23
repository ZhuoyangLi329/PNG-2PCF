import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
MAN=ROOT/'outputs/task43_outputs/ezmock_calibration_rawbox_z0p725_mmin1p4e13/manual_Ng320_Nt1297050_invF_rng1_bao0_attT_c1p14_e5_b0p25_v0'
RB=ROOT/'outputs/task43_outputs/rsd_validation/rawbox'
with np.load(MAN/'pk_jaxpower/pk0_seed439961_mesh400.npz',allow_pickle=False) as z: mk=np.asarray(z['k']); mp=np.asarray(z['pk0'])
with np.load(MAN/'xi_fcfc/xi0_seed439961_s50_550_ds10_fcfc.npz',allow_pickle=False) as z: ms=np.asarray(z['s']); mx=np.asarray(z['xi0'])
pk=sorted((RB/'pk').glob('task43_rsd_rawbox_p02_AbacusSummit_base_c000_ph*_mmin1p4e13_mesh400.npz')); assert len(pk)==25
ap=[]
for f in pk:
 with np.load(f,allow_pickle=False) as z: ap.append(z['pk0'])
ap=np.asarray(ap)
with np.load(pk[0],allow_pickle=False) as z: ak=np.asarray(z['k'])
with np.load(RB/'closure/task43_rsd_rawbox_x25_fulldiscrete_lorentzian.npz',allow_pickle=False) as z: ass=np.asarray(z['s']); ax=np.asarray(z['xi0_rsd_mean'])
fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
axes[0].set(xscale='log',yscale='log',xlabel=r'$k\ [h\,\mathrm{Mpc}^{-1}]$',ylabel=r'$P_0(k)$',title='Notebook manual tuning: $P_0$')
axes[0].plot(ak,ap.mean(0),color='#25364a',lw=2,label='Abacus rawbox mean (N=25)')
axes[0].plot(mk,mp,color='#cf6500',lw=1.8,ls='--',label='Notebook EZmock seed 439961 (FIX_AMPLITUDE=T)')
axes[0].grid(alpha=.18,which='both');axes[0].legend(frameon=False,fontsize=8)
axes[1].set(xlabel=r'$s\ [h^{-1}\mathrm{Mpc}]$',ylabel=r'$s^2\xi_0(s)$',title=r'Notebook manual tuning: $\xi_0$')
axes[1].plot(ass,ass**2*ax,color='#25364a',lw=2,label='Abacus rawbox mean (N=25)')
axes[1].plot(ms,ms**2*mx,color='#cf6500',lw=1.8,ls='--',label='Notebook EZmock seed 439961 (FIX_AMPLITUDE=T)')
axes[1].axhline(0,color='.5',lw=.7);axes[1].grid(alpha=.18);axes[1].legend(frameon=False,fontsize=8)
fig.suptitle('Notebook manual tuning versus Abacus rawbox · c=1.14, e=5, b=0.25, sigma_v=0',fontsize=13)
fig.text(.5,.01,'Only seed 439961 completed in the saved notebook output; seeds 439921/439941 failed before measurement.',ha='center',fontsize=8.5)
out=ROOT/'outputs/task43_outputs/ezmock_calibration_rawbox_z0p725_mmin1p4e13/notebook_manual_seed439961_vs_abacus.pdf';fig.savefig(out);plt.close(fig);print(out)
