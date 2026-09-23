import json
from pathlib import Path
import numpy as np
import task43_diagnose_joint_b1 as q
import task43_rsd_ezmock281_mcmc_compare as r

def main():
    pm,xm,dp,dx,_,_=r.build_models_and_data()
    def p(t):
        a=pm.evaluate(t); return np.r_[a[:13],a[13:26][r.P2_KEEP]]
    def x(t):
        a=xm.evaluate(np.asarray(t)[:3],model='formal_gic',window_key='mean'); return np.r_[a[0][r.XI_MASK],a[2][r.XI_MASK]]
    def j(t): return np.r_[p(t),x(t)]
    root=Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60/mcmc_preliminary_900')
    with np.load(root/'ezmock900_covariance_and_stack.npz') as d:
      stack=d['stack']; cov=d['covariance']; mean=stack.mean(axis=0); print('stack',stack.shape)
    print('EZ mean p',q.fit(p,mean[:22],cov[:22,:22],r.OPTIMIZER_STARTS, r.JOINT_PARAMS, r.BOUNDS_LO,r.BOUNDS_HI)[0])
    xstarts = [np.asarray(s)[:3] for s in r.OPTIMIZER_STARTS]
    print('EZ mean x',q.fit(x,mean[22:],cov[22:,22:],xstarts, r.JOINT_PARAMS[:3],r.BOUNDS_LO[:3],r.BOUNDS_HI[:3])[0])
    print('EZ mean j',q.fit(j,mean,cov,r.OPTIMIZER_STARTS,r.JOINT_PARAMS,r.BOUNDS_LO,r.BOUNDS_HI)[0])
    cov0 = cov.copy(); cov0[:22,22:] = 0.; cov0[22:,:22] = 0.
    print('EZ mean j_zero_cross',q.fit(j,mean,cov0,r.OPTIMIZER_STARTS,r.JOINT_PARAMS,r.BOUNDS_LO,r.BOUNDS_HI)[0])
    def js(t):
      f,b,sp,sx,sn = np.asarray(t)
      return np.r_[p([f,b,sp,sn]), x([f,b,sx])]
    starts5 = [np.r_[np.asarray(s)[:2], 2.5, 5.5, np.asarray(s)[3]] for s in r.OPTIMIZER_STARTS]
    names5 = ['fNL','b1','sigma_s_P','sigma_s_xi','sn0']; lo5=np.r_[r.BOUNDS_LO[:2],0.,0.,r.BOUNDS_LO[3]]; hi5=np.r_[r.BOUNDS_HI[:2],15.,15.,r.BOUNDS_HI[3]]
    print('EZ mean j_split_sigma',q.fit(js,mean,cov,starts5,names5,lo5,hi5)[0])
    print('EZ mean j_split_sigma_zero_cross',q.fit(js,mean,cov0,starts5,names5,lo5,hi5)[0])
if __name__=='__main__': main()
