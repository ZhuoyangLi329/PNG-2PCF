# Task43: fNL=0 AbacusSummit Halo Lightcone Minimal Closure

## Goal

Build the first 4.3 minimal closure test.  This is not a DESI-like analysis yet.
It only checks whether a halo lightcone, matching random catalog,
Landy-Szalay 2PCF measurement, and first window/GIC correction can recover a
Gaussian `fNL=0` baseline without generating a spurious large-scale bias.

## Fixed Scope

- Simulation: `/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/halo_light_cones/AbacusSummit_base_c000_ph000..ph024`.
- Realizations: 25 phases, `ph000` through `ph024`.
- Redshift bin: `0.6 < z < 0.8`.
- Shells to read first: `z0.575`, `z0.650`, `z0.725`, `z0.800`; final selection is by `redshift_interp`.
- Tracer: halos only, no HOD, no galaxy mocks, no DESI fiber assignment.
- Selection: fixed number density, implemented as top halos by `N_interp` per realization.
- First space: real space using `pos_interp`; keep `vel_interp` or `v_L2com` only as metadata for later RSD.
- 2PCF: monopole only, `s = 50..350 Mpc/h`, using cucount and Landy-Szalay.
- Production random multiplier: `25x` the halo catalog size. Smaller random
  multipliers are allowed only for smoke tests and must not be used for the
  closure fit.
- Model: Task4.1/4.2 `BinAvgFit + FullDiscrete` at pair-weighted `z_eff`.
- Covariance for `fNL` constraints: jaxpower Gaussian survey-window
  covariance from the same weighted lightcone random catalog. This is the only
  active Task43 lightcone covariance method.

## Output Layout

```text
outputs/task43_outputs/
  manifests/
  halo_catalogs/
  randoms/
  xi_cucount/
  jaxpower_covariance/
  fits/
  summary/
plots/task43/
codes/logs/task43/
```

Keep the directory names stable once scripts exist.

## Script Plan

1. `task43_make_manifest.py`
   - Write one JSONL row per phase.
   - Record phase, input shell paths, `zmin/zmax`, target number density, and output paths.

2. `task43_build_halo_catalog.py`
   - Read `lc_halo_info.asdf` with the cosmodesi/abacusutils environment.
   - Load `pos_interp`, `redshift_interp`, `N_interp`, and optional velocity fields.
   - Select `0.6 < redshift_interp < 0.8`.
   - Apply fixed-number-density selection by sorting `N_interp`.
   - Save columns: `RA, DEC, Z, X, Y, Zcart, WEIGHT, halo_mass_proxy, realization`.
   - Save metadata: input files, selected count, threshold `N_interp`, sky bounds, radial bounds.

3. `task43_build_randoms.py`
   - Use the same lightcone footprint and redshift window as the data.
   - First version: uniform angular points inside the measured halo lightcone footprint and uniform comoving-volume radial sampling over `0.6 < z < 0.8`.
   - Save randoms with the same coordinate columns and `WEIGHT=1`.
   - Record random multiplier and seed.

4. `task43_measure_xi_cucount.py`
   - Reuse the Task23 cucount/Landy-Szalay pattern.
   - Output `DD, DR, RR, xi0, s_edges, z_eff, ndata, nrandom`.
   - Cache `RR` for identical random catalogs.

5. `task43_make_jaxpower_lightcone_covariance.py`
   - Use the Task43 weighted random catalog/FKP field to estimate the survey
     covariance window.
   - Produce Gaussian window covariance for the same xi0 bins used by the fit.
   - Save `covariance_single_realization`, `covariance_of_mean`,
     `covariance`, `correlation`, and full JSON metadata.
   - Do not replace this with any non-jaxpower covariance.

6. `task43_fit_minimal_closure.py`
   - Fit `b1` and later `fNL`; keep `sn0` fixed in the first smoke and allow a free-`sn0` variant.
   - Compare no-GIC and GIC-corrected models.
   - Write one machine-readable summary with `fNL` mean/pull, residual slope, chi2, and gate status.

## Smoke Order

1. Run `ph000` only with a small random multiplier.
2. Confirm halo catalog columns, random footprint, and finite `xi0`.
3. Run 3 phases, still with cheap randoms, and check that residuals are not dominated by an obvious coordinate or normalization bug.
4. Run all 25 phases with the production `25x` random multiplier.
5. Run jaxpower covariance and the first `fNL` closure fit.

## Validation Gates

- Catalog gate: all 25 halo catalogs exist, have finite coordinates, and record their `N_interp` threshold.
- Random gate: randoms use the same angular mask and radial range as data; metadata records the sampling law.
- 2PCF gate: every phase has finite `DD, DR, RR, xi0`, and `RR > 0` in all fit bins.
- Model gate: `mean(xi_data - xi_model)` has no systematic large-scale slope across the fit range.
- Inference gate: fitted `fNL` mean is consistent with 0 using jaxpower
  Gaussian survey-window covariance.
- Robustness gate: changing `rmin`, halo number density, and random multiplier does not qualitatively change the conclusion.

If any gate fails, the output status must be `validation_failed`, not a successful lightcone/window/GIC validation.
