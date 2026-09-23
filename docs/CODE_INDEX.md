# Code index

## Task43 baseline and measurement

- `source/project/PNG-2PCF-model-to-observe/codes/task43/task43_run_lightcone_joint_baomask_v1.py`: frozen 9.18 data/covariance contract.
- `task43_rsd_model.py`: FullDiscrete Kaiser × Lorentzian-FoG xi operator.
- `task43_fit_rsd_lightcone_x25.py`: xi0/xi2 fit and formal GIC.
- `task43_rsd_joint_p02xi02_fit.py`: joint P02+xi02 assembly.
- `task43_rsd_boxsafe_p02_increment.py`: P02 window model.
- `task43_build_rsd_formal_gic_window.py`: RR/GIC window construction.
- catalog/random/measurement scripts: `task43_build_rsd_lightcone_catalog.py`, `task43_build_rsd_lightcone_random.py`, `task43_measure_rsd_lightcone_*`.

## Task432 new model and RIC

- `task432_lightcone_png_velocileptors.py`: direct lightcone hybrid GSM + minimal PNG response.
- `task432_png_velocileptors_gsm.py`: rawbox hybrid PNG/GSM diagnostic.
- `task432_hybrid_gic.py`: hybrid scalar GIC.
- `task432_hybrid_*_0918_contract.py`: reproducible 9.18-contract fits.
- `task432_full_ric_geometry.py`: radial projection geometry and kernels.
- `task432_full_ric_backend.py`: fixed eBOSS counting backend bridge.
- `task432_full_ric_response.py`: P/xi response matrices.
- `task432_full_ric_compile.py`, `task432_full_ric_fit.py`, `task432_full_ric_postflight.py`: full RIC engine and audits.
- `task432_full_ric_reference.py`: small-catalog algebra and Poisson reference.
- `task432_interim_findings.md`: model diagnostics and limitations.

All active Task432 source snapshots are mirrored under the same relative directory. Large binary caches and chains are intentionally excluded.
