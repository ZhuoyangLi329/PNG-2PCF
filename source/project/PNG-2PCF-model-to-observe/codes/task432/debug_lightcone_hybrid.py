#!/usr/bin/env python3
"""Debug lightcone hybrid P/xi covariance and model-vector ordering."""
from __future__ import annotations

import numpy as np

import task432_lightcone_png_velocileptors as m
from task43_run_lightcone_joint_baomask_v1 import load_rsd_specs, ScaledGaussianMetric
from task432_linear_gsm_rawbox import precision_from_covariance


def main() -> None:
    specs, meta, arrays = load_rsd_specs(smin=50.0, pk_kmax=0.08)
    by_name = {spec.name: spec for spec in specs}
    p = by_name["rsd_p02"]
    x = by_name["rsd_xi02"]
    j = by_name["rsd_joint_p02xi02"]
    with np.load(m.RSD_P_PAYLOAD, allow_pickle=False) as payload:
        zeff = float(np.asarray(payload["zeff"]).item())
    cache = m.build_cache(zeff=zeff, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000")
    with np.load(cache, allow_pickle=False) as payload:
        k, pk, alpha = payload["k_eff"], payload["pk_dd"], payload["alpha"]
        f = float(np.asarray(payload["f_growth"]).item())
    with np.load(m.RSD_X_SUMMARY, allow_pickle=False) as payload:
        s = np.asarray(payload["s"], dtype="f8")
    mask = m.rawbox_xi_primary_mask(s)
    xmodel = m.LightconePNGVelocileptors(k, pk, alpha, f, s)
    theta = np.asarray([-5.598810614359554, 2.4367990445864867, 3.329315916568756, 0.3583007188526892])
    candidates = {
        "hybrid_joint": theta,
        "standard_p_map": np.asarray([-1.5749954196890252, 2.3637428985007065, 1.0404527745539127, 0.12274886793805889]),
        "standard_joint_map": np.asarray([5.7153271579170974, 2.337524782372792, 0.0, 0.16844404717378175]),
    }
    for cname, ctheta in candidates.items():
        pcheck = p.evaluate(ctheta)
        pdiff = p.data - pcheck
        pmetric = ScaledGaussianMetric(p.covariance)
        print(cname, "pchi", pmetric.chi2(pdiff), "p_norm", float(np.linalg.norm(pdiff)))
    joint_metric = ScaledGaussianMetric(j.covariance)
    for cname, ctheta in candidates.items():
        xcheck = xmodel.evaluate(fnl=float(ctheta[0]), b1=float(ctheta[1]), nint=600)
        xv = np.concatenate([xcheck[0][mask], xcheck[2][mask]])
        jv = np.concatenate([p.evaluate(ctheta), xv])
        print(cname, "joint_chi", joint_metric.chi2(j.data - jv), "xchi", ScaledGaussianMetric(x.covariance).chi2(x.data - xv))
    pvec = p.evaluate(theta)
    xvals = xmodel.evaluate(fnl=theta[0], b1=theta[1], nint=600)
    xvec = np.concatenate([xvals[0][mask], xvals[2][mask]])
    jvec = np.concatenate([pvec, xvec])
    print("shapes", p.data.shape, x.data.shape, j.data.shape, pvec.shape, xvec.shape, jvec.shape)
    for name, spec, vec in (("p", p, pvec), ("x", x, xvec), ("joint", j, jvec)):
        metric = ScaledGaussianMetric(spec.covariance)
        precision = precision_from_covariance(spec.covariance)
        diff = np.asarray(spec.data) - vec
        print(name, "metric_chi2", metric.chi2(diff), "precision_chi2", float(diff @ precision @ diff), "norm", float(np.linalg.norm(diff)))
    print("joint residual blocks", np.linalg.norm(j.data[:p.data.size]-pvec), np.linalg.norm(j.data[p.data.size:]-xvec))


if __name__ == "__main__":
    main()
