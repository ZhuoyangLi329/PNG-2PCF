#!/usr/bin/env python3
"""EZmock900 monopole-only ownership-aware corner plot."""
from __future__ import annotations
import json, os
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.ndimage import gaussian_filter
from task43_plot_rsd_rawbox_pk0_vs_xi0_contours import density_levels, plot_range
from task43_plot_rawbox_joint_baomask_v1 import COLORS
from task43_rsd_common import PROJECT_ROOT, atomic_write_json, sha256_file

ROOT=PROJECT_ROOT; R=ROOT/'outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60/mcmc_preliminary_900_l0only_verified'; W=ROOT/'outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60/mcmc_preliminary_900_l0only_sn0wide_verified'; OUTDIR=ROOT/'9.22meeting/task43_l0only'
PARAMS=('fNL','b1','sigma_s','sn0'); LABEL={'fNL':r'$f_{\rm NL}$','b1':r'$b_1$','sigma_s':r'$\sigma_s\ [h^{-1}{\rm Mpc}]$','sn0':r'$s_{n0}$'}; VAR=(('p0_only','P0 only',COLORS['p']),('xi0_only',r'$\xi$0 only',COLORS['xi']),('joint_p0_xi0','joint $sn0\in[-1,1]$',COLORS['joint']),('joint_wide','joint $sn0\in[-3,3]$','#9467bd'))
def load(v):
    root= W if v=='joint_wide' else R; tag='joint_p0_xi0_sn0wide' if v=='joint_wide' else v
    with np.load(root/f'chain_ezmock900_{tag}.npz',allow_pickle=False) as d: c=np.asarray(d['chain'],dtype='f8').reshape(-1,d['chain'].shape[-1])
    s=json.loads((root/f'summary_ezmock900_{tag}.json').read_text()); names=tuple(s.get('names',('fNL','b1','sigma_s','sn0'))); return {'chain':c,'summary':s,'names':names,'path':root/f'chain_ezmock900_{tag}.npz'}
def val(e,p):
    return None if p not in e['names'] else e['chain'][:,e['names'].index(p)]
def draw(ax,x,y,color,xlim,ylim,z):
    h,xe,ye=np.histogram2d(x,y,bins=(150,150),range=(xlim,ylim)); h=gaussian_filter(h.astype('f8'),2.4,mode='nearest'); a,b=density_levels(h); xc=.5*(xe[:-1]+xe[1:]); yc=.5*(ye[:-1]+ye[1:]); ax.contourf(xc,yc,h.T,levels=[a,b,float(h.max())*1.001],colors=[to_rgba(color,.09),to_rgba(color,.20)],zorder=z); ax.contour(xc,yc,h.T,levels=[a,b],colors=color,linewidths=[1.,1.6],zorder=z+1)
def main():
    e={v:load(v) for v,_,_ in VAR}; ranges={}
    for p in PARAMS:
        a=np.concatenate([val(x,p) for x in e.values() if val(x,p) is not None]); ranges[p]=plot_range(a,np.array([]),parameter='sigma_s' if p=='sigma_s' else p)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans','Liberation Sans'],'pdf.fonttype':42,'ps.fonttype':42,'font.size':9.2,'axes.linewidth':1.,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True}); OUTDIR.mkdir(parents=True,exist_ok=True); out=OUTDIR/'task43_ezmock900_l0only_sn0wide_allparams_9.18style.pdf'; tmp=out.with_name('.'+out.stem+'.tmp.pdf')
    with PdfPages(tmp) as pdf:
        fig,ax=plt.subplots(4,4,figsize=(10.5,9.7),squeeze=False)
        for i,p in enumerate(PARAMS):
            for j in range(4):
                a=ax[i,j]
                if j>i: a.set_axis_off(); continue
                a.set_xlim(*ranges[PARAMS[j]]); a.xaxis.set_major_locator(MaxNLocator(nbins=4)); a.tick_params(labelsize=8)
                if i==j:
                    for v,_,color in VAR:
                        x=val(e[v],p)
                        if x is not None: a.hist(x,bins=np.linspace(*ranges[p],60),density=True,histtype='step',lw=1.5,color=color)
                    a.set_yticks([])
                else:
                    a.set_ylim(*ranges[p]); a.yaxis.set_major_locator(MaxNLocator(nbins=4))
                    for n,(v,_,color) in enumerate(VAR):
                        x,y=val(e[v],PARAMS[j]),val(e[v],p)
                        if x is not None and y is not None: draw(a,x,y,color,ranges[PARAMS[j]],ranges[p],2+2*n)
                    if j==0: a.set_ylabel(LABEL[p],fontsize=10)
                if i==3: a.set_xlabel(LABEL[PARAMS[j]],fontsize=10)
                else: a.set_xticklabels([])
                if j>0 and i!=j: a.set_yticklabels([])
        ax[0,0].axvline(0,color='.5',lw=.7,ls='--'); handles=[]; labels=[]
        for v,label,color in VAR:
            s=e[v]['summary']; q=s['posterior']['fNL']; ml=float(s['map_theta'][0]); handles.append(plt.Line2D([],[],color=color,lw=2)); labels.append(label+'\n'+rf'$f_{{\rm NL}}={ml:.2f}_{{-{ml-q["q16"]:.2f}}}^{{+{q["q84"]-ml:.2f}}}$')
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,.995),ncol=2,frameon=False,fontsize=10.5,handletextpad=.5,columnspacing=1.); fig.suptitle('Task43 RSD lightcone: EZmock-900; monopole only ($\\ell=0$); free $s_{n0}$',fontsize=12,y=1.02); fig.text(.985,.005,'P0+\\xi0 only; violet curve uses the widened $sn0$ prior',ha='right',fontsize=8.3,color='.35'); fig.subplots_adjust(left=.08,right=.985,bottom=.065,top=.90,wspace=.06,hspace=.06); pdf.savefig(fig,bbox_inches='tight',pad_inches=.08); plt.close(fig)
    tmp.replace(out); audit={'task':'task43_ezmock900_l0only_sn0wide_allparams','status':'pass','multipoles':['ell0'],'excluded_multipoles':['ell2'],'free_parameters':PARAMS,'sn0_prior_comparison':{'standard':[-1.,1.],'wide':[-3.,3.]},'ownership':{'p0_only':['fNL','b1','sigma_s','sn0'],'xi0_only':['fNL','b1','sigma_s'],'joint_standard':['fNL','b1','sigma_s','sn0'],'joint_wide':['fNL','b1','sigma_s','sn0']},'chains':{v:str(e[v]['path']) for v,_,_ in VAR},'output_pdf':str(out),'output_pdf_sha256':sha256_file(out)}; atomic_write_json(out.with_suffix('.json'),audit); print(json.dumps(audit,sort_keys=True))
if __name__=='__main__': main()
