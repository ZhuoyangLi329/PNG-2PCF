#!/usr/bin/env python3
"""All-free-parameter corner for the EZmock fixed-sn0 split-sigma test."""
from __future__ import annotations
import json, os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.ndimage import gaussian_filter
from task43_plot_rsd_rawbox_pk0_vs_xi0_contours import density_levels, plot_range
from task43_plot_rawbox_joint_baomask_v1 import COLORS
from task43_rsd_common import PROJECT_ROOT, atomic_write_json, sha256_file

PARENT=PROJECT_ROOT/'outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60'
OLD=PARENT/'mcmc_preliminary_900'; FIX=PARENT/'mcmc_preliminary_900_fixed_sn0_verified'; SPLIT=PARENT/'mcmc_preliminary_900_split_sigma_fixed_sn0_verified'; OUTDIR=PROJECT_ROOT/'9.22meeting/task43_fixed_sn0'
PARAMS=('fNL','b1','sigma_s_P','sigma_s_xi'); LABELS={'fNL':r'$f_{\rm NL}$','b1':r'$b_1$','sigma_s_P':r'$\sigma_{s,P}\ [h^{-1}{\rm Mpc}]$','sigma_s_xi':r'$\sigma_{s,\xi}\ [h^{-1}{\rm Mpc}]$'}; VAR=(('p02',r'P02 ($\sigma_{s,P}$)',COLORS['p']),('xi02',r'xi02 ($\sigma_{s,\xi}$)',COLORS['xi']),('split','joint split','#9467bd'))

def load(root,tag,names):
    with np.load(root/f'chain_{tag}.npz',allow_pickle=False) as d: c=np.asarray(d['chain'],dtype='f8').reshape(-1,d['chain'].shape[-1])
    s=json.loads((root/f'summary_{tag}.json').read_text()); return {'chain':c,'summary':s,'path':root/f'chain_{tag}.npz','names':names}

def entries():
    p=load(FIX,'ezmock900_p02_fixed_sn0',('fNL','b1','sigma_s_P'))
    x=load(OLD,'ezmock900_xi02',('fNL','b1','sigma_s_xi'))
    z=load(SPLIT,'ezmock900_joint_split_sigma_fixed_sn0',('fNL','b1','sigma_s_P','sigma_s_xi'))
    return {'p02':p,'xi02':x,'split':z}

def vals(e,p):
    if p not in e['names']: return None
    return e['chain'][:,e['names'].index(p)]

def draw(ax,x,y,color,xlim,ylim,z):
    h,xe,ye=np.histogram2d(x,y,bins=(150,150),range=(xlim,ylim)); h=gaussian_filter(h.astype('f8'),2.4,mode='nearest'); a,b=density_levels(h); xc=.5*(xe[:-1]+xe[1:]); yc=.5*(ye[:-1]+ye[1:]); ax.contourf(xc,yc,h.T,levels=[a,b,float(h.max())*1.001],colors=[to_rgba(color,.09),to_rgba(color,.20)],zorder=z); ax.contour(xc,yc,h.T,levels=[a,b],colors=color,linewidths=[1.0,1.6],zorder=z+1)

def interval(e,p):
    s=e['summary']; i=e['names'].index(p); q=s['posterior'][p]; ml=float(s['map_theta'][i]); return rf'{ml:.2f}_{{-{ml-q["q16"]:.2f}}}^{{+{q["q84"]-ml:.2f}}}'

def main():
    es=entries(); ranges={p:plot_range(np.concatenate([vals(e,p) for e in es.values() if vals(e,p) is not None]),np.array([]),parameter='sigma_s' if p.startswith('sigma') else p) for p in PARAMS}
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans','Liberation Sans'],'pdf.fonttype':42,'ps.fonttype':42,'font.size':9.2,'axes.linewidth':1.,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
    OUTDIR.mkdir(parents=True,exist_ok=True); out=OUTDIR/'task43_ezmock900_split_sigma_fixed_sn0_allparams_strict_9.18style.pdf'; tmp=out.with_name('.'+out.stem+'.tmp.pdf')
    with PdfPages(tmp) as pdf:
        fig,ax=plt.subplots(4,4,figsize=(10.5,9.7),squeeze=False)
        for i,p in enumerate(PARAMS):
            for j in range(4):
                a=ax[i,j]
                if j>i: a.set_axis_off(); continue
                a.set_xlim(*ranges[PARAMS[j]]); a.xaxis.set_major_locator(MaxNLocator(nbins=4)); a.tick_params(labelsize=8)
                if i==j:
                    bins=np.linspace(*ranges[p],60)
                    for v,label,color in VAR:
                        x=vals(es[v],p)
                        if x is not None: a.hist(x,bins=bins,density=True,histtype='step',lw=1.5,color=color)
                    a.set_yticks([])
                else:
                    a.set_ylim(*ranges[p]); a.yaxis.set_major_locator(MaxNLocator(nbins=4))
                    for n,(v,label,color) in enumerate(VAR):
                        x,y=vals(es[v],PARAMS[j]),vals(es[v],p)
                        if x is not None and y is not None: draw(a,x,y,color,ranges[PARAMS[j]],ranges[p],2+2*n)
                    if j==0: a.set_ylabel(LABELS[p],fontsize=10)
                if i==3: a.set_xlabel(LABELS[PARAMS[j]],fontsize=10)
                else: a.set_xticklabels([])
                if j>0 and i!=j: a.set_yticklabels([])
            if p=='fNL': ax[i,i].axvline(0,color='.5',lw=.7,ls='--')
        handles=[]; labels=[]
        for v,label,color in VAR:
            e=es[v]; handles.append(plt.Line2D([],[],color=color,lw=2)); labels.append(label+'\n'+rf'$f_{{\rm NL}}={interval(e,"fNL")}$')
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,.995),ncol=2,frameon=False,fontsize=10.5,handletextpad=.5,columnspacing=1.)
        fig.suptitle('Task43 RSD lightcone: EZmock-900; fixed $s_{n0}=0$; separate $\\sigma_s$',fontsize=12,y=1.02); fig.text(.985,.005,'P02 constrains $\\sigma_{s,P}$; $\\xi$ constrains $\\sigma_{s,\\xi}$; joint has both',ha='right',fontsize=8.3,color='.35'); fig.subplots_adjust(left=.08,right=.985,bottom=.065,top=.90,wspace=.06,hspace=.06); pdf.savefig(fig,bbox_inches='tight',pad_inches=.08); plt.close(fig)
    tmp.replace(out)
    audit={'task':'task43_ezmock900_split_sigma_fixed_sn0_allparams_strict','status':'pass','style_reference':'9.18meeting/task43_ezmock_covariance_mcmc_vs_jaxpower','fixed_parameters':{'sn0':0.},'shared_parameters':['fNL','b1'],'separate_parameters':['sigma_s_P','sigma_s_xi'],'contour_parameters':PARAMS,'variants':['P02 owns sigma_s_P','xi02 owns sigma_s_xi','joint split owns both'],'contour_probabilities':[.68,.95],'parameter_ownership':{'p02':['fNL','b1','sigma_s_P'],'xi02':['fNL','b1','sigma_s_xi'],'joint_split':list(PARAMS)},'chains':{v:str(es[v]['path']) for v,_,_ in VAR},'output_pdf':str(out),'output_pdf_sha256':sha256_file(out)}; atomic_write_json(out.with_suffix('.json'),audit); print(json.dumps(audit,sort_keys=True))

if __name__=='__main__': main()
