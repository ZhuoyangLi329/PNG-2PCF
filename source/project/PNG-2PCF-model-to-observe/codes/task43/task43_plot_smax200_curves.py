import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT=Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe"); sys.path.insert(0,str(PROJECT/"codes/task43"))
from task43_run_lightcone_joint_baomask_v1 import load_rsd_specs

root=PROJECT/"outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/mcmc_smax200_1000"; fits=root/"fits"
with np.load(root/"covariance.npz",allow_pickle=False) as z: payload={k:np.asarray(z[k]) for k in z.files}
specs,_,_=load_rsd_specs(smin=50.,pk_kmax=.08); smap={s.name:s for s in specs}
joint=json.loads((fits/"ezmock281_joint/summary.json").read_text()); th=np.asarray([joint["map"]["theta"][n] for n in ("fNL","b1","sigma_s","sn0")])
pred_p=smap["rsd_p02"].evaluate(th); xiidx=np.r_[np.flatnonzero(payload["xi_mask"]),26+np.flatnonzero(payload["xi_mask"])]
pred_x=smap["rsd_xi02"].evaluate(th[:3])[xiidx]
data_p=payload["data_p02"]; data_x=payload["data_xi02"]; stack=payload["stack"]
edges=np.asarray(json.loads((PROJECT/"outputs/task43_outputs/rsd_validation/lightcone/kmax0p08_smin50_v1/task43_lightcone_standard_joint_baomask80_120_v1.json").read_text())["rsd"]["pk_fit_edges_h_mpc"])
k=0.5*(edges[:,0]+edges[:,1]); k2=k[payload["p2_keep"]]; s=payload["s"][payload["xi_mask"]]
arrays={"P0":(k,data_p[:13],pred_p[:13],stack[:,:13]),"P2":(k2,data_p[13:],pred_p[13:],stack[:,13:22]),"xi0":(s,data_x[:11],pred_x[:11],stack[:,22:33]),"xi2":(s,data_x[11:],pred_x[11:],stack[:,33:44])}
plt.rcParams.update({"font.size":10.5,"font.family":"sans-serif","pdf.fonttype":42,"xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True})
fig,ax=plt.subplots(2,4,figsize=(16,7.2),sharex="col",constrained_layout=True)
colors={"P0":"#2763a3","P2":"#bf2f3b","xi0":"#329457","xi2":"#8c5aa8"}
for col,name in enumerate(("P0","P2","xi0","xi2")):
 x,d,p,st=arrays[name]; scale=x*x if name.startswith("xi") else np.ones_like(x); mean=st.mean(axis=0); std=st.std(axis=0,ddof=1); ydata=scale*d; ypred=scale*p; ym=scale*mean; ys=scale*std; c=colors[name]
 top,bottom=ax[:,col]
 top.errorbar(x,ydata,fmt="o",ms=4,color="0.2",capsize=2,lw=.8,label="measured"); top.plot(x,ypred,color=c,lw=1.8,label="joint best-fit"); top.set_title(name,fontsize=12); top.legend(frameon=False,fontsize=8)
 bottom.errorbar(x,ydata,fmt="o",ms=4,color="0.2",capsize=2,lw=.8,label="measured"); bottom.plot(x,ym,color="#D55E00",lw=1.8,label="EZmock mean (N=1000)"); bottom.fill_between(x,ym-ys,ym+ys,color="#D55E00",alpha=.18,lw=0,label=r"$\pm1\sigma$ mock scatter"); bottom.legend(frameon=False,fontsize=8)
 if name.startswith("P"):
  top.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$"); bottom.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$"); top.set_ylabel(r"$P_\ell(k)$"); bottom.set_ylabel(r"$P_\ell(k)$")
 else:
  top.set_xlabel(r"$s\ [h^{-1}\mathrm{Mpc}]$"); bottom.set_xlabel(r"$s\ [h^{-1}\mathrm{Mpc}]$"); top.set_ylabel(r"$s^2\xi_\ell(s)$"); bottom.set_ylabel(r"$s^2\xi_\ell(s)$")
 for a in (top,bottom): a.grid(alpha=.18); a.axhline(0,color="0.6",lw=.6)
fig.suptitle(r"Task43 RSD lightcone: $s_{\max}=200\ h^{-1}\mathrm{Mpc}$ · $l=0,2$ curve checks",fontsize=13)
fig.text(.5,.005,"Top: joint best-fit versus measured mean · Bottom: measured mean versus 1000-EZmock mean with mock scatter",ha="center",fontsize=10)
out=root/"bestfit_vs_measurement_vs_ezmockmean_smax200_l02.pdf"; fig.savefig(out); plt.close(fig); print(out)
