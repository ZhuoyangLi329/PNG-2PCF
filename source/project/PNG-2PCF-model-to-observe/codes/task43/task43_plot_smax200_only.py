import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

root=Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/mcmc_smax200_1000")
fits=root/"fits"; variants=[("p02","P0+P2","#2763a3"),("xi02","xi0+xi2","#329457"),("joint","Joint","#bf2f3b")]
sig={}; fig,ax=plt.subplots(figsize=(8.4,6.4),constrained_layout=True)
for v,label,color in variants:
    sm=json.loads((fits/f"ezmock281_{v}"/"summary.json").read_text())
    f=sm["posterior_corrected"]["fNL"]; sig[v]=f["sigma68"]; pieces=[]
    for file in sorted((fits/("ezmock281_"+v)).glob("run*.npz")):
        with np.load(file,allow_pickle=False) as z: pieces.append(np.asarray(z["chain"]).reshape(-1,len(sm["parameter_names"]))[:,:2])
    pts=np.vstack(pieces); ctr=np.median(pts,axis=0); pts=ctr+(pts-ctr)*sm["factors"]["sigma_factor"]
    lim=np.percentile(pts,[.05,99.95],axis=0); h,xe,ye=np.histogram2d(pts[:,0],pts[:,1],bins=120,range=lim.T.tolist()); d=gaussian_filter(h,1.2); order=np.sort(d.ravel())[::-1]; cum=np.cumsum(order)/order.sum(); levels=np.sort([order[np.searchsorted(cum,q)] for q in (.6827,.9545)]); x=(xe[1:]+xe[:-1])/2; y=(ye[1:]+ye[:-1])/2
    ax.contour(x,y,d.T,levels=levels,colors=color,linewidths=[1.2,2.0]); q50=f["q50"]; plus=f["q84"]-q50; minus=q50-f["q16"]; txt=f"{label} ($f_{{\\rm NL}}={q50:+.1f}^{{+{plus:.1f}}}_{{-{minus:.1f}}}$)"; ax.plot([],[],color=color,label=txt)
gp=1-sig["joint"]/sig["p02"]; gx=1-sig["joint"]/sig["xi02"]
ax.set(xlabel=r"$f_{\rm NL}$",ylabel=r"$b_1$",title="EZmock RSD lightcone · $s_{\max}=200\ h^{-1}\mathrm{Mpc}$"); ax.legend(frameon=False,fontsize=10); ax.grid(alpha=.18); ax.text(.02,.02,f"1000 realizations · Hartlap + Percival\nJoint reduction: {100*gp:.1f}% vs P0+P2; {100*gx:.1f}% vs xi0+xi2",transform=ax.transAxes,va="bottom",ha="left",fontsize=9,bbox=dict(facecolor="white",alpha=.8,edgecolor="none"))
out=root/"contours_ezmock1000_smax200_only.pdf"; fig.savefig(out); plt.close(fig); print(out)
