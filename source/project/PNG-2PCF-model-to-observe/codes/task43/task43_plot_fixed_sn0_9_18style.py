#!/usr/bin/env python3
"""9.18meeting-style triangle plot for the fixed-sn0 joint test."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from task43_plot_rawbox_joint_baomask_v1 import COLORS
from task43_plot_rsd_rawbox_pk0_vs_xi0_contours import density_levels, plot_range
from task43_rsd_common import PROJECT_ROOT, atomic_write_json, sha256_file

ROOT=PROJECT_ROOT
SHARED=ROOT/'outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60/mcmc_preliminary_900'
FIXED=ROOT/'outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60/mcmc_preliminary_900_fixed_sn0_verified'
OUTDIR=ROOT/'9.22meeting/task43_fixed_sn0'
VARIANTS=(('p02','p',r'$P_0+P_2$'),('xi02','xi',r'$\xi_0+\xi_2$'),('joint','joint','joint'),('fixed','fixed',r'joint ($sn0=0$'))
COL={**COLORS,'fixed':'#9467bd'}

def load(root:Path, tag:str):
    with np.load(root/f'chain_{tag}.npz',allow_pickle=False) as d:
        chain=np.asarray(d['chain'],dtype='f8').reshape(-1,d['chain'].shape[-1])
    summary=json.loads((root/f'summary_{tag}.json').read_text())
    return {'chain':chain,'summary':summary,'path':root/f'chain_{tag}.npz'}

def contour(ax,x,y,color,xlim,ylim,z):
    h,xe,ye=np.histogram2d(x,y,bins=(170,160),range=(xlim,ylim)); h=gaussian_filter(h.astype('f8'),sigma=2.5,mode='nearest')
    l95,l68=density_levels(h); xc=.5*(xe[:-1]+xe[1:]); yc=.5*(ye[:-1]+ye[1:])
    ax.contourf(xc,yc,h.T,levels=[l95,l68,float(h.max())*1.001],colors=[to_rgba(color,.10),to_rgba(color,.22)],antialiased=True,zorder=z)
    ax.contour(xc,yc,h.T,levels=[l95,l68],colors=color,linewidths=[1.2,1.8],zorder=z+1)

def interval(samples, ml):
    q16,q84=np.percentile(samples,[16,84]); return rf'{ml:.2f}_{{-{ml-q16:.2f}}}^{{+{q84-ml:.2f}}}'

def page(pdf,cov,display,chains,ranges):
    fig,ax=plt.subplots(2,2,figsize=(6.8,6.0)); ax[0,1].set_axis_off()
    for v,c,_ in VARIANTS:
        entry=chains[f'{cov}_{v}']; ax[0,0].hist(entry['chain'][:,0],bins=np.linspace(*ranges['fNL'],90),density=True,histtype='step',lw=1.8,color=COL[c]); ax[1,1].hist(entry['chain'][:,1],bins=np.linspace(*ranges['b1'],90),density=True,histtype='step',lw=1.8,color=COL[c])
    ax[0,0].set_xlim(*ranges['fNL']); ax[0,0].set_yticks([]); ax[0,0].tick_params(labelbottom=False); ax[0,0].axvline(0,color='.55',lw=.8,ls='--')
    for z,(v,c,_) in enumerate(VARIANTS,2):
        e=chains[f'{cov}_{v}']; contour(ax[1,0],e['chain'][:,0],e['chain'][:,1],COL[c],ranges['fNL'],ranges['b1'],z*2)
    ax[1,0].set(xlim=ranges['fNL'],ylim=ranges['b1'],xlabel=r'$f_{\rm NL}$',ylabel=r'$b_1$'); ax[1,0].axvline(0,color='.55',lw=.8,ls='--'); ax[1,1].set_xlim(*ranges['b1']); ax[1,1].set_yticks([]); ax[1,1].set_xlabel(r'$b_1$')
    handles=[]; labels=[]
    for v,c,label in VARIANTS:
        e=chains[f'{cov}_{v}']; handles.append(plt.Line2D([],[],color=COL[c],lw=2)); labels.append(label+'\n'+rf'$f_{{\rm NL}}={interval(e["chain"][:,0],e["summary"]["map_theta"][0])}$')
    fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.99,.95),frameon=False,fontsize=15.0,labelspacing=.55,handletextpad=.7)
    fig.suptitle('Task43 lightcone RSD (box-safe 0.4 < z_obs < 0.8; kmax=0.08, smin=50): P02, BAO-masked xi02, and joint — '+display+' covariance',fontsize=10.5,y=.985)
    fig.subplots_adjust(left=.13,right=.97,bottom=.10,top=.93,wspace=.08,hspace=.08); pdf.savefig(fig,bbox_inches='tight',pad_inches=.08); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--force',action='store_true'); a=ap.parse_args(); OUTDIR.mkdir(parents=True,exist_ok=True)
    ez='ezmock900'; pages=((ez,'EZmock-900 empirical'),('jaxpower','jaxpower analytic'))
    chains={}
    for cov,_ in pages:
        chains[f'{cov}_p02']=load(SHARED,f'{cov}_p02'); chains[f'{cov}_xi02']=load(SHARED,f'{cov}_xi02'); chains[f'{cov}_joint']=load(SHARED,f'{cov}_joint'); chains[f'{cov}_fixed']=load(FIXED,f'{cov}_joint_fixed_sn0')
    diagnostic=SHARED/'diagnostics/joint_b1/task43_joint_b1_diagnostic.json'
    if diagnostic.is_file():
        payload=json.loads(diagnostic.read_text())
        for cov,_ in pages:
            theta=payload['covariances'][cov]['baseline']['joint']['theta']
            chains[f'{cov}_joint']['summary']['map_theta']=[float(theta['fNL']),float(theta['b1']),float(theta['sigma_s']),float(theta['sn0'])]
    fn=np.concatenate([e['chain'][:,0] for e in chains.values()]); b=np.concatenate([e['chain'][:,1] for e in chains.values()]); ranges={'fNL':plot_range(fn,fn,parameter='fNL'),'b1':plot_range(b,b,parameter='b1')}
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans','Liberation Sans'],'pdf.fonttype':42,'ps.fonttype':42,'font.size':10,'axes.linewidth':1,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
    out=OUTDIR/'task43_ezmock900_joint_fixed_sn0_9.18style.pdf'; outj=out.with_suffix('.json')
    if out.exists() and not a.force: raise FileExistsError(out)
    tmp=out.with_name('.'+out.name+'.tmp.pdf')
    with PdfPages(tmp) as pdf:
        for cov,d in pages: page(pdf,cov,d,chains,ranges)
    tmp.replace(out)
    info={'task':'task43_fixed_sn0_9_18style','status':'pass','style_reference':'9.18meeting/task43_ezmock_covariance_mcmc_vs_jaxpower','fixed_parameter':'sn0=0','pages':[d for _,d in pages],'variants':[v for v,_,_ in VARIANTS],'chains':{k:str(v['path']) for k,v in chains.items()},'sha256':sha256_file(out),'output_pdf':str(out)}
    atomic_write_json(outj,info); print(json.dumps(info,sort_keys=True))
if __name__=='__main__': main()
