# Task43 Run Log

## 2026-07-02 Active Lightcone Covariance Convention

- Active Task43 lightcone covariance method is fixed to jaxpower Gaussian
  survey-window covariance. This is the only covariance method to use for
  current lightcone fNL closure tests.
- Previous non-jaxpower lightcone covariance attempts were removed from this
  active runlog and must not be quoted as Task43 current results.
- The active sample is fixed `M_min=1.4e13 h^-1 Msun`, `0.6 < z < 0.8`,
  real-space AbacusSummit halo lightcone, phases `ph000..ph024`, with `25x`
  matching randoms and `P0=10000` FKP weights.
- Login-node work should be used whenever practical and must stay at or below
  12 active CPU cores. The 2PCF step currently requires GPU cucount; the
  wrapper requests the machine-required GPU allocation but constrains the
  actual process to 12 CPU cores.
- The separate rawbox branch is maintained independently and must not be
  moved, deleted, overwritten, or cancelled by this lightcone work.

## 2026-07-02 Current Inputs And 2PCF

- Validated active manifest:
  `outputs/task43_outputs/manifests/task43_mmin1p4e13_x25.jsonl`.
- Input audit:
  `outputs/task43_outputs/summary/task43_mmin1p4e13_input_audit.json`.
  It verifies 25 phases, real space, `0.6<z<0.8`, `random_multiplier=25`,
  fixed mass threshold metadata, `N_interp>=6638`, actual mass threshold
  `1.4000083132767432e13 h^-1 Msun`, halo counts `328019..334531`, and
  random counts `8200475..8363275`.
- FKP/effective-redshift summary:
  `outputs/task43_outputs/summary/task43_fkp_zeff_mmin1p4e13_x25.npz`
  and `.json`.
  For `P0=10000`: `zeff_random_auto=0.7030110803750067`,
  `zeff_data_auto=0.7030205482753431`,
  `zeff_data_random_cross=0.7030157272236283`,
  `nbar_min=1.5732625249942459e-4`,
  `nbar_mean=1.7481889481189985e-4`, and
  `nbar_max=1.9248529546483687e-4`.
- Weighted 2PCF method:
  `cucount_jax + Landy-Szalay`, with `WEIGHT_TOTAL = WEIGHT * WEIGHT_FKP`
  for data and randoms. Measurement bins are `s_edges=50..350, ds=10`
  with centers `55..345`.
- 2PCF output pattern:
  `outputs/task43_outputs/xi_cucount/xi0_*mmin1p4e13*x25_fkpP010000.npz`.
- Current production status at the latest poll in this runlog: 24/25 weighted
  xi files exist. The active single-phase dependency chain is continuing, with
  remaining dependency jobs still owned by the 2PCF production chain.

## 2026-07-02 Jaxpower Covariance Smoke

- Added active lightcone covariance script:
  `codes/task43/task43_make_jaxpower_lightcone_covariance.py`.
- The script reads the current `M_min=1.4e13` lightcone manifest and FKP
  summary, builds a random-catalog FKP survey window with
  `jaxpower.compute_fkp2_covariance_window`, evaluates
  `Mesh2SpectrumPoles` using `P_h(k)+1/nbar`, applies
  `jaxpower.compute_spectrum2_covariance`, and projects to xi0 with
  `jaxpower.cov2.matrix_project_to_correlation`.
- Current smoke covariance:
  `outputs/task43_outputs/summary/jaxpower_covariance_lightcone_mmin1p4e13_ph000_smoke_mesh32_nran20k_s80_350_ds30_for_partial16.npz`
  with metadata JSON at the same prefix.
- Smoke settings:
  `ph000` halo/random only, `P0=10000`, `zeff=0.7030110803750067`,
  `meshsize=32`, `max_data=10000`, `max_random=20000`,
  `window_s=0..800, ds=40`, `k=0.005..0.505, dk=0.025`, and xi bins
  `s_edges=80..350, ds=30`.
- `covariance_single_realization` diagnostics:
  shape `(9, 9)`, `sqrt(diag)=2.275e-03..9.258e-03`, minimum eigenvalue
  `1.159e-07`, condition number `937.6`, and no SPD floor applied
  (`n_floored=0`).
- The saved `covariance` and `covariance_of_mean` keys in this smoke file are
  scaled by the file metadata `nreal=25`. For partial-realization early tests
  and for the current Task4.3 fNL convention, the fit must explicitly read
  `covariance_single_realization`; do not reinterpret the quoted constraints
  as covariance-of-mean constraints.
- This is a Gaussian survey-window covariance smoke. It is not a full
  connected-trispectrum or full configuration-space LS covariance.

## 2026-07-02 Early Jaxpower-Covariance Fit

- Temporary partial xi data vector:
  `outputs/task43_outputs/summary/task43_xi_mmin1p4e13_x25_fkpP010000_partial16_earlydiag_s80_350_ds30.npz`.
  This is pinned to the subset manifest
  `outputs/task43_outputs/manifests/task43_mmin1p4e13_x25_partial16_earlydiag.jsonl`
  (`ph000..ph015`) so the early fit remains reproducible while the remaining
  phases continue to run.
- Fit target: `all-realizations`, `nreal=16`, 9 bins with centers `95..335`,
  fixed `p=1.1`, fixed `sn0=0`.
- Covariance for both fits is the jaxpower smoke file above with
  `covariance_key=covariance_single_realization`.
- `no_gic` output:
  `outputs/task43_outputs/fits/partial16_mmin1p4e13_x25_fkpP010000_s80_350_ds30_jaxpowercov_mesh32_nran20k_no_gic_earlydiag/`.
  Result: `fNL = -16.94 -7.70 +6.72`,
  `b1 = 2.465 -0.102 +0.101`, `chi2_map_total=81.142`.
- `formal_gic` output:
  `outputs/task43_outputs/fits/partial16_mmin1p4e13_x25_fkpP010000_s80_350_ds30_jaxpowercov_mesh32_nran20k_formalgic_w2nsub10k_earlydiag/`.
  This uses a separate RR pair-kernel W2 smoke with `nsub=10000`:
  `outputs/task43_outputs/summary/task43_formal_gic_window_mmin1p4e13_x25_ph000_fkpP010000_nsub10000_seed20260703_earlydiag.npz`.
  Result: `fNL = -13.29 -8.10 +6.60`,
  `b1 = 2.452 -0.097 +0.102`, `chi2_map_total=81.137`,
  `sigma_W^2_map=1.696e-05`.
- These early results are smoke diagnostics only. The final result requires
  all 25 2PCF phases and a higher-stat jaxpower covariance/window test.

## 2026-07-02 Partial18 Jaxpower-Covariance Fit

- Current completed data vector:
  `outputs/task43_outputs/summary/task43_xi_mmin1p4e13_x25_fkpP010000_partial18_earlytest_s80_350_ds30.npz`.
  It uses subset manifest
  `outputs/task43_outputs/manifests/task43_mmin1p4e13_x25_partial18_earlytest.jsonl`
  (`ph000..ph017`), `nreal=18`, and contains no fit-covariance keys.
- Strongest current jaxpower covariance test:
  `outputs/task43_outputs/summary/jaxpower_covariance_lightcone_mmin1p4e13_ph000_medium_mesh64_nran1M_ndata200k_win3600_k001_801_s80_350_ds30.npz`.
  Settings: `meshsize=64`, `max_random=1000000`, `max_data=200000`,
  `window_s=0..3600, ds=100`, `k=0.001..0.801, dk=0.02`.
  Diagnostics: `sqrt(diag)=1.500e-04..2.113e-03`, minimum eigenvalue
  `7.102e-10`, condition number `1.032e4`.
- Random convergence check: relative to the same mesh/window/k run with
  `max_random=200000`, the nran1M diagonal sigma ratio is
  `0.829..0.911`. The covariance is therefore still an early Gaussian-window
  diagnostic, not a final converged covariance.
- Partial18 fNL fit with this nran1M covariance:
  `outputs/task43_outputs/fits/partial18_mmin1p4e13_x25_fkpP010000_s80_350_ds30_jaxpowercov_mesh64_nran1M_no_gic_formalgic_w2nsub10k_earlytest/`.
  Fit target is `all-realizations`, covariance key is
  `covariance_single_realization`.
  - `no_gic`: `fNL = -9.27 -1.42 +1.53`,
    `b1 = 2.371 -0.0049 +0.0052`, `chi2_map_total=11684.46`.
  - `formal_gic`: `fNL = -2.08 -1.57 +1.69`,
    `b1 = 2.348 -0.0053 +0.0051`, `chi2_map_total=11627.20`,
    `sigma_W^2_map=1.822e-05`.
- Interpretation: the partial18 fit runs successfully with the required
  jaxpower covariance key, but this result must not be quoted as a physics
  constraint. The extremely small `fNL` error and very large chi2 indicate that
  the current Gaussian-window covariance/model combination is not yet an
  acceptable closure covariance.

## 2026-07-02 Lightcone Jaxpower Covariance Debug

- External sanity scale from the validated 2Gpc rawbox branch: the rawbox
  `smin=80`, jaxpower covariance, free-`sn0` fit gives
  `fNL=-3.69 -18.74 +17.11`
  (`outputs/task43_outputs/rawbox_z0p725_mmin1p3e13/summary/task43_rawbox_abacusc000_p1p0_free_sn0_jaxpower_cov_smin_scan_fnl_b1.csv`).
  The current lightcone shell has `volume_shell sum=1.9081699227e9 (Mpc/h)^3`,
  roughly one quarter of the 2Gpc rawbox volume, so a lightcone `fNL` error of
  `~1-2` is unphysical and marks the partial18 nran1M result as invalid.
- The partial18 nran1M fit did read the intended covariance key:
  `covariance_single_realization`; in the covariance file
  `diag(covariance_single_realization)/diag(covariance)=25`. The failure is not
  a mean-vs-single covariance key mixup.
- The nran1M lightcone covariance has a pathological correlation structure:
  `sqrt(diag)=1.500e-04..2.113e-03`,
  correlation off-diagonal max `0.9916`, correlation condition number
  `4.25e3`, covariance condition number `1.03e4`. Projecting the 18-realization
  scatter onto its eigenmodes gives variance ratios of about
  `384, 169, 96` in the three smallest covariance modes, explaining
  `chi2~1.16e4`.
- By contrast, the 18-realization measured scatter has
  `sqrt(diag)=2.984e-04..1.064e-03`, correlation condition number `32.6`; the
  rawbox jaxpower covariance at the same `smin=80` has correlation condition
  number `12.7`.
- Fine-window and padding diagnostics were run on login-node smoke settings:
  - `debug_mesh32_nran20k_win800_ds2_k005_505_s80_350_ds30`: replacing
    `window_ds=40` with `ds=2` lowers the maximum off-diagonal correlation from
    `0.822` to `0.698`, but the correlation condition number remains high
    (`~523`).
  - `debug_mesh32_nran20k_win800_ds2_k005_505_norescale_s80_350_ds30` is
    identical to the rescaled-weight run at relative Frobenius difference
    `1.4e-15`; subsample weight rescaling is not the dominant cause.
  - `debug_mesh64_nran20k_pad400_win350_ds2_k005_505_s80_350_ds30` and
    `debug_mesh64_nran20k_pad800_win800_ds2_k005_505_s80_350_ds30` show that
    larger zero padding changes the diagonal scale but still leaves correlation
    condition numbers of `~4e2`.
- Component decomposition for the `mesh64/nran20k/pad400/window350/ds2` smoke
  using `compute_spectrum2_covariance(..., return_type="list")`: diagonal
  fractions are roughly `WW=5-10%`, `WS=21-38%`, `SS=54-73%`; however the `WW`
  component alone is extremely low-rank (`corr_cond~4.1e4`). The current
  failure is therefore a lightcone window-covariance structure problem, not a
  simple MCMC or covariance-key problem.

## 2026-07-02 Lightcone Jaxpower Covariance Fix

- Added diagnostic script:
  `codes/task43/task43_diagnose_jaxpower_lightcone_covariance.py`.
  It records FKP weights, alpha, analytic small-separation window limits,
  jaxpower `WW/WS/SW/SS` window summaries, `WW/WS/SS` covariance components,
  and optional comparison to the realization scatter.
- Found and patched a data-subsampling bug in
  `codes/task43/task43_make_jaxpower_lightcone_covariance.py`: jaxpower's
  `S` shot-window terms use data-weight products, so thinning the data catalog
  biases `WS` by `N_data_total/N_data_used` and `SS` by this factor squared.
  The script now computes `WW/WS/SS` separately and applies the inverse factors
  before summing. A 10k-data corrected run agrees with a full-data diagnostic
  at `~8e-4` relative Frobenius difference.
- The remaining large covariance pathology was traced to the `k` projection
  grid, not to the covariance key or random count alone. The invalid early
  grid used `kmax<=0.8` and `dk~0.02`; this truncates the high-k/shot-like
  contribution needed for configuration-space covariance and creates an
  artificially low-rank correlation matrix.
- Login-node convergence diagnostics, all with `mesh64`, `window_s=0..350`,
  `window_ds=1`, corrected `WS/SS`, `s=80..350, ds=30`:
  - `k=1e-4..3, dk=0.010`: `corr_cond=141`, max offdiag `0.922`.
  - `k=1e-4..3, dk=0.005`: `corr_cond=13.8`, max offdiag `0.576`.
  - `k=1e-4..3, dk=0.002`: `corr_cond=13.6`, max offdiag `0.569`.
  A separate `basis=bessel + interpolate_window_function + flags=smooth,fftlog`
  test at `kmax=0.505` still had `corr_cond=8.76e3`, so FFTLog/window
  interpolation alone was not the fix.
- The active fit-compatible covariance-fix file is:
  `outputs/task43_outputs/summary/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_mesh64_nran100k_ndata50k_pad400_win350_ds1_k0001_3000_dk002_s80_350_ds30.npz`.
  It uses `max_random=100000`, `max_data=50000`, `P0=10000`, `kmin=1e-4`,
  `kmax=3.0001`, `dk=0.002`, and `covariance_single_realization`.
  Diagnostics: `sqrt(diag)=1.542e-04..8.474e-04`, covariance condition number
  `92.27`, correlation condition number `12.53`, max offdiag `0.560`, and
  positive SPD floor-free eigenvalues.
- Partial18 covariance-fix closure fit, first joint-likelihood diagnostic:
  `outputs/task43_outputs/fits/partial18_mmin1p4e13_x25_fkpP010000_s80_350_ds30_jaxpowercov_kmax3_dk002_mesh64_nran100k_ndata50k_no_gic_formalgic_w2nsub10k_covfix/`.
  Fit target is `all-realizations`, `nreal=18`, covariance key is
  `covariance_single_realization`.
  - `no_gic`: `fNL = -11.55 -7.95 +7.73`,
    `b1 = 2.330 -0.047 +0.046`, `chi2_map_total=504.71`.
  - `formal_gic`: `fNL = -7.30 -8.28 +7.96`,
    `b1 = 2.317 -0.047 +0.048`, `chi2_map_total=505.25`,
    `sigma_W^2_map=1.648e-05`.
- This all-realizations fit is not the reporting convention requested for the
  lightcone closure result. It treats the 18 realizations as 18 independent
  observed vectors in a joint likelihood and therefore tightens the posterior
  by roughly `sqrt(18)`.
- Current reporting convention: use the multi-realization mean xi as the
  observed data vector, but use the single-realization covariance for the
  likelihood. This is `fit-target=mean` with explicit
  `covariance-key=covariance_single_realization`.
- Partial18 mean-vector + single-covariance fit:
  `outputs/task43_outputs/fits/partial18_mean_mmin1p4e13_x25_fkpP010000_s80_350_ds30_jaxpowercov_singlecov_kmax3_dk002_mesh64_nran100k_ndata50k_no_gic_formalgic_w2nsub10k_covfix/`.
  Fit target is `mean`, `nreal=18` only enters the xi averaging, and the
  covariance key is still `covariance_single_realization`.
  - `no_gic`: `fNL = -27.39 -59.25 +38.46`,
    `b1 = 2.287 -0.240 +0.208`, `chi2_map_total=1.75`.
  - `formal_gic`: `fNL = -22.77 -59.91 +38.08`,
    `b1 = 2.303 -0.248 +0.199`, `chi2_map_total=1.78`,
    `sigma_W^2_map=1.636e-05`.
- Interpretation: the jaxpower covariance problem that caused `fNL` errors of
  `~1-2` is fixed for the current partial18 smoke. The current quotable
  diagnostic is the mean-vector + single-covariance fit above. It is still an
  early 18-realization Gaussian-window diagnostic, not the final
  25-realization result and not a RascalC/full connected covariance. All old
  lightcone jaxpower fits using `kmax<=0.8`, and the all-realizations
  joint-likelihood fit above, must not be quoted as the current limit.

## 2026-07-02 Lightcone L=2000, p=1.0, PNG-order diagnostic

- The AbacusSummit lightcone theory normalization must use the base-box
  `L=2000 Mpc/h` fundamental mode. Added `--theory-boxsize` to
  `codes/task43/task43_fit_minimal_closure.py`; all current lightcone fits
  below use `boxsize=2000` and `kfund=0.003141592653589793`.
- The p=1.0 full-PNG fit is the required science model and includes the
  `fNL^2 b_phi^2 alpha^2 P` term. It produced a visibly left-skewed `fNL`
  marginal in the current lightcone setup:
  `outputs/task43_outputs/fits/partial18_mean_mmin1p4e13_x25_fkpP010000_s80_350_ds30_jaxpowercov_singlecov_kmax3_dk002_mesh64_nran100k_ndata50k_theoryL2000_p1p0_no_gic_formalgic_w2nsub10k_covfix/`.
  With posterior-median reporting:
  - `no_gic`: `fNL = -20.10 -51.46 +34.29`.
  - `formal_gic`: `fNL = -18.97 -48.06 +35.17`,
    `sigma_W^2_map=1.596e-05`.
- A deterministic `fNL,b1` grid check showed that this skew is not an emcee
  artifact. The full-PNG grid gives `fNL = -21.41 -49.80 +34.06`, matching the
  MCMC marginal. A temporary linear-response diagnostic, with the quadratic
  `fNL^2` term removed, gives an almost Gaussian posterior:
  `fNL = -9.14 -30.78 +32.44`, `skew ~= 0.04`. This is **not** the adopted
  science model; it is only a diagnostic showing which model term participates
  in the lightcone skew. The diagnostic grid is saved in
  `outputs/task43_outputs/summary/task43_lightcone_p1p0_fixedsn0_no_gic_gridposterior_diagnostic.npz`.
- Added `--png-order {full,linear}` to `task43_fit_minimal_closure.py` for
  diagnostics only. The default and required Task43 science model remains
  `full`. The following linear-PNG run should not be quoted as the lightcone
  constraint unless the rawbox and Task4 comparisons are deliberately redone in
  the same linearized model.
- Diagnostic-only p=1.0, L=2000, linear-PNG, mean-vector + single-realization
  covariance fit:
  `outputs/task43_outputs/fits/partial18_mean_mmin1p4e13_x25_fkpP010000_s80_350_ds30_jaxpowercov_singlecov_kmax3_dk002_mesh64_nran100k_ndata50k_theoryL2000_p1p0_linear_no_gic_formalgic_w2nsub10k_covfix/`.
  It uses `fit-target=mean`, `covariance-key=covariance_single_realization`,
  `p_fixed=1.0`, `sn0=0` fixed, `zeff=0.7030110804`, and
  `png_order=linear`.
  - `no_gic`: `fNL = -9.45 -29.79 +31.39`,
    `b1 = 2.2738 -0.2471 +0.2013`, `chi2_map_total=1.745`.
  - `formal_gic`: `fNL = -3.42 -35.98 +36.21`,
    `b1 = 2.2389 -0.2324 +0.2054`, `chi2_map_total=1.774`,
    `sigma_W^2_map=1.595e-05`.
- Diagnostic-only rawbox-style triangle plot and machine-readable summary:
  `plots/task43/task43_partial18_mean_singlecov_L2000_p1p0_linear_fnl_b1_contour.pdf`
  and
  `plots/task43/task43_partial18_mean_singlecov_L2000_p1p0_linear_fnl_b1_contour.json`.
  The plot follows the rawbox convention: central values are posterior
  medians, 68% intervals are `q16/q84`, and MAP values are recorded only as
  diagnostics in the JSON.

## 2026-07-02 Full-PNG tail diagnostic against rawbox/Task4.1

- Corrected model interpretation: rawbox is **not** a linear-PNG control.  The
  rawbox/Task43 no-RSD model uses
  `P_h(k) = [b1 + fNL * 2 delta_c (b1-p) * alpha(k)]^2 P_dd(k) + sn0`, hence
  includes the `fNL^2 b_phi^2 alpha^2 P_dd` term.  Evidence:
  `codes/task4/task41_rawbox_norsd_fnl100_profiler.py` lines 209-211.  The
  Task45 Quijote active 4.1 path also uses desilike
  `PNGTracerPowerSpectrumMultipoles(mode="b-p")`, whose installed source in the
  project `desilike` environment forms
  `(bX + f mu^2) * (bY + f mu^2)` after adding `bfnl_loc * alpha` to `b1`.
- Added diagnostic script:
  `codes/task43/task43_diagnose_full_png_tail.py`.  It must be run in the
  project environment, e.g.
  `/global/homes/l/lzy/anaconda3/envs/desilike/bin/python`, not the bare base
  shell.  It is read-only with respect to existing fits and writes:
  `outputs/task43_outputs/summary/task43_full_png_tail_diagnostic_lightcone_vs_rawbox.json`,
  `outputs/task43_outputs/summary/task43_full_png_tail_diagnostic_lightcone_vs_rawbox.npz`,
  and
  `plots/task43/task43_full_png_tail_diagnostic_lightcone_vs_rawbox.pdf`.
  This script was later removed in the 2026-07-03 Task43 cleanup because it
  hard-coded obsolete partial18 inputs; do not use it as a current entrypoint.
- The diagnostic compares the current lightcone full-PNG fixed-sn0 grid against
  the rawbox full-PNG free-sn0 control using the same real-space full-PNG basis
  construction and `covariance_single_realization`.
  - Lightcone full PNG, fixed `sn0=0`: profile best
    `fNL=-10.5, b1=2.3325, chi2=1.7447`; profile `Delta chi2=1` interval
    `[-46.0, 18.0]`, profile `Delta chi2=2.30` interval `[-71.5, 32.5]`;
    marginal grid quantile `fNL=-21.41 -50.17 +34.23`.
  - Rawbox full PNG, profiled free `sn0 in [-1,1]`: profile best
    `fNL=-3.5, b1=2.300, sn0=1.0, chi2=3.2258`; profile `Delta chi2=1`
    interval `[-20.5, 12.5]`, profile `Delta chi2=2.30` interval
    `[-30.5, 21.0]`; marginal grid quantile
    `fNL=-5.10 -18.13 +17.08`, consistent with the rawbox MCMC
    `fNL=-3.69 -18.74 +17.11`.
- Current conclusion: the lightcone tail is **not** explained by rawbox missing
  the `fNL^2` term.  Both sides include full PNG.  The current lightcone
  likelihood itself has a longer negative-`fNL` profile shoulder under the
  present Gaussian window covariance/minimal real-space setup; marginalization
  preserves this asymmetry.  The next debugging target is therefore the
  lightcone covariance/data-vector/theory-projection geometry, not replacing the
  science model with a linearized PNG model.
- Additional covariance-variant diagnostics:
  `outputs/task43_outputs/summary/task43_full_png_tail_covariance_variant_diagnostic.json`,
  `outputs/task43_outputs/summary/task43_full_png_tail_covariance_swap_extreme_diagnostic.json`,
  and
  `outputs/task43_outputs/summary/task43_full_png_tail_rawbox_cov_scaled_to_lightcone_sigma.json`.
  These tests keep the full-PNG model fixed.
  - Replacing the lightcone covariance correlation matrix with the rawbox
    correlation matrix while keeping the lightcone diagonal barely changes the
    lightcone result: `fNL=-21.71 -50.80 +34.68`.
  - Diagonalizing the lightcone covariance shortens the tail but leaves an
    asymmetric broad result: `fNL=-20.22 -36.36 +26.05`.
  - Using the unscaled rawbox covariance on the lightcone data gives a narrow,
    nearly rawbox-like result: `fNL=-13.35 -18.59 +17.06`.
  - Using the unscaled lightcone covariance on the rawbox data gives a broad,
    left-skewed result: `fNL=-13.53 -45.87 +33.64`.
  - Scaling the rawbox covariance by the mean lightcone/rawbox sigma ratio
    `1.733` gives `fNL=-9.00 -36.80 +31.47`.
- Interpretation update: the current full-PNG left tail is driven mainly by the
  much weaker single-realization covariance amplitude / information content of
  the present lightcone setup, not by rawbox lacking the quadratic term and not
  primarily by the off-diagonal correlation pattern.  This still does **not**
  make the current jaxpower Gaussian window covariance a final covariance; it
  only identifies why the present full-PNG posterior shape differs from the
  rawbox control.
- Synthetic full-PNG closure diagnostic:
  `outputs/task43_outputs/summary/task43_full_png_synthetic_tail_direction_diagnostic.json`.
  This uses noiseless synthetic data vectors generated from the same full-PNG
  basis and then refits them with the current covariance metrics.
  - With the current lightcone covariance, a noiseless `fNL=0, b1=2.33` vector
    already gives `fNL=-8.02 -42.58 +31.67`; `fNL=-50` gives
    `fNL=-67.30 -56.72 +44.40`; `fNL=+50` gives
    `fNL=50.27 -24.61 +27.74`.
  - With the rawbox covariance, a noiseless `fNL=0, b1=2.30` vector gives
    `fNL=-1.75 -18.34 +17.20`, while `fNL=+50` gives
    `fNL=49.85 -14.30 +15.11`.
  - This shows that the tail direction depends on the full-PNG likelihood
    geometry and on where the preferred/true `fNL` sits.  Positive-`fNL`
    Quijote tags are therefore not expected to show the same left tail as the
    current `fNL=0` Abacus lightcone sanity test.
- Active Task4.1 UltraNest summaries are consistent with this:
  - Quijote `fid` (`fNL=0`) already has left-side wider posteriors, e.g.
    fixed-sn0 `xi_r80_350`: `fNL=-14.38 -89.63 +68.76`; fixed-sn0
    `xi_r100_350`: `fNL=-20.33 -92.02 +54.63`.
  - Quijote `LCp50` and `LCp100` are centered at positive `fNL`, and their
    broader side is often the positive side, e.g. fixed-sn0 `LCp50 xi_r80_350`:
    `fNL=40.16 -68.01 +107.30`; fixed-sn0 `LCp100 xi_r80_350`:
    `fNL=102.70 -87.42 +127.45`.
  - Hence the Task4.1 comparison does not support dropping the full-PNG
    quadratic term; it supports keeping full PNG and reporting/profile-checking
    the non-Gaussian posterior shape explicitly.

## 2026-07-02 Full25 `s=50..350, ds=10` lightcone fit test

- Created the full 25-realization FKP-weighted mean xi summary from existing
  `cucount` measurements:
  `outputs/task43_outputs/summary/task43_mean_xi_mmin1p4e13_x25_s50_350_ds10_fkpP010000.npz`.
  The summary covers all phases `ph000..ph024`, has 30 bins with centers
  `55..345`, and uses `zeff=0.7030110803750067` from
  `task43_fkp_zeff_mmin1p4e13_x25.npz`.
- Built the matched jaxpower Gaussian survey-window xi0 covariance:
  `outputs/task43_outputs/summary/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_mesh64_nran100k_ndata50k_pad400_win350_ds1_k0001_3000_dk002_p1p0_s50_350_ds10.npz`.
  It uses `meshsize=64`, `max_random=100000`, `max_data=50000`,
  `window_s_max=350`, `window_ds=1`, `kmin=1e-4`, `kmax=3.0001`,
  `dk=0.002`, `b1_cov=2.5`, `fnl_cov=0`, `p_fixed=1.0`, and
  `P0=10000`. The fit must read `covariance_single_realization`; the
  `covariance` and `covariance_of_mean` keys are builder byproducts divided
  by 25 and are not the Task4.3 fNL quote convention.
- Covariance diagnostics for the single-realization matrix: shape `30x30`,
  sigma range `1.609e-04..1.505e-03`, min eigenvalue `2.254e-09`,
  condition number `2701.6`, and no SPD floor was applied. This is still only
  a jaxpower Gaussian survey-window covariance; it is not a full connected
  trispectrum or RascalC/LS covariance.
- Ran the full-PNG, `L=2000`, `p=1.0`, fixed-`sn0=0` mean-vector fit with
  `covariance-key=covariance_single_realization`:
  `outputs/task43_outputs/fits/full25_mean_mmin1p4e13_x25_fkpP010000_s50_350_ds10_jaxpowercov_singlecov_kmax3_dk002_mesh64_nran100k_ndata50k_theoryL2000_p1p0_no_gic_formalgic_w2nsub10k_covfix/`.
  - `no_gic`: MAP `fNL=54.62`, `b1=2.3928`, `chi2=23.78`;
    posterior `fNL=55.14 -10.88 +11.80`,
    `b1=2.3936 -0.0640 +0.0602`.
  - `formal_gic`: MAP `fNL=59.24`, `b1=2.3864`, `chi2=22.59`;
    posterior `fNL=59.24 -11.31 +12.34`,
    `b1=2.3784 -0.0613 +0.0632`;
    `sigma_W^2_map=3.825e-05`.
- Wrote the PDF-only fNL-b1 contour and JSON summary:
  `plots/task43/task43_full25_mean_s50_350_ds10_singlecov_L2000_p1p0_fnl_b1_contour.pdf`
  and
  `plots/task43/task43_full25_mean_s50_350_ds10_singlecov_L2000_p1p0_fnl_b1_contour.json`.
  The plot marks MAP points with numeric labels and quotes posterior
  `q50/q16/q84` intervals separately.

## 2026-07-02 `s=50..350, ds=10` covariance failure diagnosis

- The `s50_350_ds10` fit above must **not** be quoted as a valid Task43
  constraint.  Its small `fNL` errors are caused by a failed jaxpower
  covariance diagnostic, not by improved statistical power.  The machine
  readable diagnostic is:
  `outputs/task43_outputs/summary/task43_s50_ds10_jaxpower_covariance_failure_diagnostic.json`.
- Direct realization-scatter whitening check for the same 25 phases and the
  same `covariance_single_realization`:
  - expected mean chi2 about the empirical mean:
    `(Nreal-1)/Nreal * Nbin = 28.8`;
  - full covariance gives mean chi2 `147.12` with range `58.03..454.81`;
  - diagonal-only jaxpower covariance still gives mean chi2 `68.59`;
  - median `sample_std / jaxpower_sigma = 1.45`, with range `1.01..2.32`;
  - therefore both the diagonal amplitude and the covariance eigen-directions
    are wrong, with the full whitening failure at `147.12/28.8 = 5.11`.
- A small login-node smoke comparison tested whether the finite window support
  is part of the problem:
  - histogram/smooth `window_s_max=350`, `dk=0.005`, `nran=20k`,
    `ndata=10k` gives scatter mean chi2 `168.48`;
  - `basis=bessel + interpolate_window_function + flags=smooth,fftlog`,
    `window_s_max=3600`, `window_ds=2` gives scatter mean chi2 `113.49`.
  The full-window FFTLog path improves the correlation/eigenstructure but does
  not by itself fix the covariance normalization.
- Structural issue identified from DESI's reference usage in
  `clustering_statistics/correlation2_tools.py`: the cut-sky xi projection
  should form `projector = matrix_project_to_correlation(...)` and then apply
  `np.linalg.solve(compute_RR2_window(RR, ...), projector)` to deconvolve the
  RR/LS pair window.  The current Task43 covariance scripts explicitly use
  `window_deconvolution=None`, so they project a windowed-P covariance to xi,
  not a validated Landy-Szalay xi covariance.
- Current `cucount` xi files only store 1D `RR(s)` counts.  The strict DESI
  RR deconvolution path requires `RR(s, mu)` pair counts, so the next
  covariance repair step is to remeasure or additionally save random-random
  pair counts with a mu dimension for the lightcone random catalog, then build
  the RR deconvolution matrix before projecting `C[P]` to `C[xi]`.
- Bias-localization diagnostic with the same failed covariance but different
  `rmin` shows that the positive `fNL` shift is not caused only by the
  `50<s<80` bins:
  - `rmin=80`: no-GIC posterior `fNL=59.17 -17.69 +21.06`, MAP `58.39`;
  - `rmin=100`: no-GIC posterior `fNL=59.80 -17.61 +21.82`, MAP `59.81`;
  - `rmin=120`: posterior becomes broad/unstable,
    `fNL=-47.40 -98.23 +57.89`, optimizer MAP near zero.
  This points to tension around the `80..120` Mpc/h part of the data vector
  combined with the flawed covariance weighting.

## 2026-07-02 RR-deconvolved jaxpower covariance baseline for `s=50..350, ds=10`

- Built a higher-stat full-window FFTLog/bessel jaxpower Gaussian survey-window
  covariance baseline on the login node:
  `outputs/task43_outputs/summary/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_350_ds10.npz`.
  It uses the current `M_min=1.4e13` lightcone random, `meshsize=64`,
  `max_random=100000`, `max_data=50000`, `window_s_max=3600`,
  `window_ds=2`, `window_basis=bessel`, `smooth,fftlog`, `kmin=1e-4`,
  `kmax=3.0001`, `dk=0.002`, `p_fixed=1.0`, and `P0=10000`.
- Applied the RR(s,mu) LS-window deconvolution with
  `codes/task43/task43_apply_rr_smu_deconvolution_to_covariance.py` and
  `outputs/task43_outputs/summary/task43_rr_smu_window_smoke_ph000_nran100k_s50_350_ds10_nmu20_midpoint.npz`.
  The deconvolved covariance is:
  `outputs/task43_outputs/summary/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_350_ds10.npz`.
- Single-realization scatter check against the 25 measured lightcone xi vectors:
  expected mean chi2 about the empirical mean is `28.8`, while this
  deconvolved covariance gives mean chi2 `39.50` with range `23.45..66.75`.
  The median `sample_std / covariance_sigma` is `1.08`, with range
  `0.73..1.33`.  This is the Task43 baseline jaxpower Gaussian window
  covariance.  It replaces the failed direct projection (`147.12/28.8`) and
  is the basic covariance convention for the current lightcone fNL fits.
- Ran the full-PNG, `L=2000`, `p=1.0`, fixed `sn0=0`, `s=50..350`,
  `ds=10` mean-vector fit with `covariance-key=covariance_single_realization`:
  `outputs/task43_outputs/fits/full25_mean_s50_350_ds10_besselfftlog_rrdeconv_fitcov_L2000_p1p0/`.
  - `no_gic`: MAP `fNL=11.97`, `b1=2.4100`, `chi2=9.12`;
    posterior `fNL=10.38 -27.76 +28.34`,
    `b1=2.3989 -0.1049 +0.1039`.
  - `formal_gic`: MAP `fNL=15.64`, `b1=2.4060`, `chi2=8.97`;
    posterior `fNL=15.86 -27.23 +26.81`,
    `b1=2.4004 -0.1074 +0.1052`;
    `sigma_W^2_map=2.304e-05`.
- Wrote a reusable PDF-only getdist-style plotting script:
  `codes/task43/task43_plot_lightcone_fnl_b1_contour.py`.
  The final contour plot and JSON metadata are:
  `plots/task43/task43_full25_mean_s50_350_ds10_besselfftlog_rrdeconv_singlecov_L2000_p1p0_fnl_b1_contour.pdf`
  and
  `plots/task43/task43_full25_mean_s50_350_ds10_besselfftlog_rrdeconv_singlecov_L2000_p1p0_fnl_b1_contour.json`.

## 2026-07-02 MCMC rerun contours for `smin=50` and `smin=60`

- Reran MCMC for two fit ranges using the same RR-deconvolved jaxpower
  single-realization covariance and the same theory setup: `L=2000`,
  `p=1.0`, fixed `sn0=0`, `P0=10000`, `zeff=0.7030110803750067`,
  `models=no_gic,formal_gic`.
- `smin=50`, `s=50..350`, `ds=10`, 30 bins:
  `outputs/task43_outputs/fits/full25_mean_s50_350_ds10_besselfftlog_rrdeconv_fitcov_L2000_p1p0_mcmc_rerun_contours/`.
  - `no_gic`: MAP `fNL=12.11`, `b1=2.4123`, `chi2=9.12`;
    posterior `fNL=9.99 -27.65 +27.17`.
  - `formal_gic`: MAP `fNL=15.68`, `b1=2.4073`, `chi2=8.97`;
    posterior `fNL=12.38 -28.07 +28.14`.
  - PDF contour:
    `plots/task43/task43_full25_mean_smin50_s50_350_ds10_besselfftlog_rrdeconv_singlecov_L2000_p1p0_fnl_b1_contour.pdf`.
- `smin=60`, `s=60..350`, `ds=10`, 29 bins:
  `outputs/task43_outputs/fits/full25_mean_s60_350_ds10_besselfftlog_rrdeconv_fitcov_L2000_p1p0_mcmc_rerun_contours/`.
  - `no_gic`: MAP `fNL=11.59`, `b1=2.3370`, `chi2=8.11`;
    posterior `fNL=7.05 -30.14 +30.21`.
  - `formal_gic`: MAP `fNL=15.14`, `b1=2.3334`, `chi2=7.97`;
    posterior `fNL=12.11 -32.09 +32.07`.
  - PDF contour:
    `plots/task43/task43_full25_mean_smin60_s60_350_ds10_besselfftlog_rrdeconv_singlecov_L2000_p1p0_fnl_b1_contour.pdf`.
- `smin=80`, `s=80..350`, `ds=10`, 27 bins, same fNL0 lightcone mean
  sample and same baseline covariance:
  `outputs/task43_outputs/fits/full25_mean_s80_350_ds10_besselfftlog_rrdeconv_fitcov_L2000_p1p0_mcmc_rerun_contours/`.
  - `no_gic`: MAP `fNL=3.65`, `b1=2.2209`, `chi2=4.61`;
    posterior `fNL=-6.44 -47.95 +37.17`.
  - `formal_gic`: MAP `fNL=6.74`, `b1=2.2202`, `chi2=4.52`;
    posterior `fNL=-0.89 -48.26 +39.59`;
    `sigma_W^2_map=1.737e-05`.
  - PDF contour:
    `plots/task43/task43_full25_mean_smin80_s80_350_ds10_besselfftlog_rrdeconv_singlecov_L2000_p1p0_fnl_b1_contour.pdf`.

## 2026-07-02 shell-averaged xi-kernel lightcone check

- Checked the latest rawbox shell-averaged result table:
  `outputs/task43_outputs/rawbox_z0p725_mmin1p3e13/summary/task43_rawbox_abacusc000_p1p0_free_sn0_s50_350_ds10_shellavg_jaxpower_cov_smin506080_fnl_b1.csv`.
  The rawbox summaries explicitly record `xi_kernel=shell-averaged`.
- Reran the lightcone `smin=50,60,80`, `smax=350`, `ds=10` fits with only the
  theory-space xi kernel changed to `--xi-kernel shell-averaged`.  The input
  data remain the fNL0 25-realization lightcone mean, and the covariance remains
  the baseline jaxpower Gaussian survey-window covariance with
  `covariance-key=covariance_single_realization`.  Other theory settings are
  unchanged: `L=2000`, `p=1.0`, fixed `sn0=0`, `P0=10000`,
  `zeff=0.7030110803750067`, and `models=no_gic,formal_gic`.
- `smin=50`, 30 bins, fit directory:
  `outputs/task43_outputs/fits/full25_mean_s50_350_ds10_shellavg_besselfftlog_rrdeconv_fitcov_L2000_p1p0_mcmc/`.
  - `no_gic`: MAP `fNL=6.36`, `b1=2.4678`, `chi2=5.86`;
    posterior `fNL=3.56 -27.53 +25.92`.
  - `formal_gic`: MAP `fNL=9.29`, `b1=2.4631`, `chi2=5.76`;
    posterior `fNL=7.42 -28.03 +26.67`;
    `sigma_W^2_map=2.227e-05`.
  - PDF contour:
    `plots/task43/task43_full25_mean_smin50_s50_350_ds10_shellavg_besselfftlog_rrdeconv_singlecov_L2000_p1p0_fnl_b1_contour.pdf`.
- `smin=60`, 29 bins, fit directory:
  `outputs/task43_outputs/fits/full25_mean_s60_350_ds10_shellavg_besselfftlog_rrdeconv_fitcov_L2000_p1p0_mcmc/`.
  - `no_gic`: MAP `fNL=6.19`, `b1=2.4234`, `chi2=5.54`;
    posterior `fNL=2.92 -30.94 +29.14`.
  - `formal_gic`: MAP `fNL=9.04`, `b1=2.4189`, `chi2=5.44`;
    posterior `fNL=9.85 -29.43 +27.48`;
    `sigma_W^2_map=2.137e-05`.
  - PDF contour:
    `plots/task43/task43_full25_mean_smin60_s60_350_ds10_shellavg_besselfftlog_rrdeconv_singlecov_L2000_p1p0_fnl_b1_contour.pdf`.
- `smin=80`, 27 bins, fit directory:
  `outputs/task43_outputs/fits/full25_mean_s80_350_ds10_shellavg_besselfftlog_rrdeconv_fitcov_L2000_p1p0_mcmc/`.
  - `no_gic`: MAP `fNL=0.97`, `b1=2.3209`, `chi2=3.50`;
    posterior `fNL=-6.96 -44.71 +33.86`.
  - `formal_gic`: MAP `fNL=3.72`, `b1=2.3218`, `chi2=3.42`;
    posterior `fNL=-0.47 -41.83 +33.37`;
    `sigma_W^2_map=1.832e-05`.
  - PDF contour:
    `plots/task43/task43_full25_mean_smin80_s80_350_ds10_shellavg_besselfftlog_rrdeconv_singlecov_L2000_p1p0_fnl_b1_contour.pdf`.
- The plot metadata for all three shell-averaged runs records
  `truth_fnl=null`, `xi_kernel=shell-averaged`, and
  `covariance.key=covariance_single_realization`, so these are fNL0 lightcone
  fits with the single-realization covariance, not synthetic nonzero-fNL tests.
- Compared with the previous center-kernel `smin=80` fit
  (`no_gic: fNL=-6.44 -47.95 +37.17`; `formal_gic:
  fNL=-0.89 -48.26 +39.59`), the shell-averaged kernel changes the central
  values only mildly and gives a slightly tighter posterior for this fit range.

## 2026-07-03 covariance convention clarification

- Task4.3 lightcone fNL results use the single-covariance convention. The
  observed data vector is the 25-phase mean xi to suppress realization noise
  in the closure/data-vector diagnostic, but the likelihood covariance key is
  always `covariance_single_realization`, representing one lightcone survey.
  Do not divide by `Nreal`.
- `covariance_single_realization/Nreal` would correspond to a different
  question: constraints from an average of `Nreal` independent surveys/phases,
  with errors shrinking by roughly `sqrt(Nreal)`. It is not the convention used
  for the current Task4.3 fNL quotes.
- Current `singlecov` plot/fit directories are therefore correctly labeled as
  single-covariance fits. Future agents should not "fix" them to
  covariance-of-mean fits.
- If a covariance NPZ contains `covariance` or `covariance_of_mean` keys that
  are divided by 25, treat them as builder byproducts or separate diagnostics,
  not as the Task4.3 fNL quote key.

## 2026-07-03 Task4.3 audit notes

- Rechecked the current authoritative full25 shell-averaged fit summaries:
  they use `fit_target=mean`,
  `covariance.key=covariance_single_realization`, `L=2000`, `p_fixed=1.0`,
  `png_order=full`, fixed `sn0=0`, `xi_kernel=shell-averaged`,
  `P0=10000`, and `models=no_gic,formal_gic`.
- Rechecked the measurement chain: the manifest has 25 phases
  `ph000..ph024`, `0.6<z<0.8`, real-space `mmin1p4e13`,
  `random_multiplier=25`, matching halo/random paths, and
  `random_radial_policy=data_redshift_resample`. The xi summary has
  `nreal=25`, 30 bins `s=55..345`, finite `xi0_all`, positive `RR_all`,
  and `zeff=0.7030110803750067` from the all-random-auto FKP summary.
- Rechecked formal GIC: the active shell-averaged fit summaries were produced
  from
  `task43_formal_gic_window_mmin1p4e13_x25_ph000_fkpP010000_L2000_nsub10000_seed20260703_earlydiag.npz`.
  This is parameter-dependent `sigma_W^2(theta)`, not `ic_constant`, but the
  old earlydiag cache was later removed in the 2026-07-03 cleanup.  The required
  clean `10k/50k/200k` W2 convergence products were not present at this audit
  point; they were later completed in the section below.
- Rechecked covariance: the active covariance is
  `jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_350_ds10.npz`.
  It is SPD and has `sigma=3.29e-4..1.67e-3`, but its JSON still warns that it
  is a diagnostic covariance repair requiring validation. Scatter comparison:
  `chi2_mean_per_realization=39.50` vs expected `28.8`, and
  `sample_std_over_cov_sigma_median=1.084`.
- Code guard added in `task43_fit_minimal_closure.py`: `--covariance-key auto`
  now prefers `covariance_single_realization` for all fit targets, and warns if
  `covariance` or `covariance_of_mean` is selected. This protects future mean
  fits from accidentally using a divided covariance key.

## 2026-07-03 L2000 formal-GIC W2 convergence launch

- Cleaned the active Task43 setup before launching convergence:
  `task43_config.BOX_SIZE` is now `2000.0`, and
  `task43_build_formal_gic_window.py` accepts/records `--theory-boxsize`.
  Older login-node wrappers and stale partial/debug artifacts were removed; the
  cleanup manifest is
  `old_doc_codes/deletion_manifests/delete_manifest_20260703T072256Z_task43_cleanup.json`.
- Added the convergence driver
  `codes/task43/run_task43_formalgic_w2_convergence_L2000.sbatch`.  It builds
  clean L2000 W2 caches for `nsub=10000,50000,200000`, then reruns
  `formal_gic` fits for `smin=50,60,80` with the current full25 mean xi,
  current RR-deconvolved jaxpower covariance, `covariance_single_realization`,
  shell-averaged xi kernel, `p_fixed=1.0`, full PNG, and fixed `sn0=0`.
- Added summarizer `codes/task43/task43_summarize_w2_convergence.py`, which will
  write
  `outputs/task43_outputs/summary/task43_formal_gic_w2_convergence_L2000_seed20260703.json`
  and `.csv`.  It compares W2 against the `200k` cache and compares the new
  posteriors against the existing shell-averaged baseline.
- A small local W2 smoke using `nsub=128`, `ndense=2000`, and
  `--theory-boxsize 2000` completed successfully.  Its metadata reported
  `n_random_total=8254350`, `coverage=1.0`, and `theory_boxsize=2000.0`.
- Submitted the formal convergence job as Slurm job `55421431` with output logs
  `codes/logs/task43/w2conv_L2000_55421431.{out,err}`.  At this log entry the
  job was still `PENDING` with reason `Priority`; the completed result is
  recorded in the next section.

## 2026-07-03 L2000 formal-GIC W2 convergence completed

- Slurm job `55421431` completed successfully with exit code `0:0`, elapsed
  `00:06:56`, and batch-step MaxRSS `1904304K`.  The only stderr messages were
  repeated JAX CUDA plugin warnings on a CPU node; they did not stop the run.
- W2 caches written and metadata patched to record `theory_boxsize=2000.0`:
  `outputs/task43_outputs/summary/task43_formal_gic_window_mmin1p4e13_x25_ph000_fkpP010000_L2000_nsub10000_seed20260703.npz`,
  `..._nsub50000_seed20260703.npz`, and
  `..._nsub200000_seed20260703.npz`.
- Summary products:
  `outputs/task43_outputs/summary/task43_formal_gic_w2_convergence_L2000_seed20260703.json`,
  `outputs/task43_outputs/summary/task43_formal_gic_w2_convergence_L2000_seed20260703.csv`,
  and
  `plots/task43/task43_formal_gic_w2_convergence_L2000_seed20260703.pdf`.
- W2 absolute differences relative to `nsub=200000`:
  `nsub=10000`: RMS `1.919e-05`, max `5.650e-04`;
  `nsub=50000`: RMS `1.955e-05`, max `7.033e-04`.  Relative W2 differences are
  not a good scalar diagnostic because W2 crosses near zero.
- Formal-GIC `nsub=200000` reference fits relative to the previous
  shell-averaged baseline:
  - `smin=50`: `fNL=6.06 -29.13 +26.71`, `Delta q50=-1.36`
    (`-0.050 sigma_old`), `sigma_W2_map=2.245e-05`.
  - `smin=60`: `fNL=7.94 -29.29 +28.74`, `Delta q50=-1.92`
    (`-0.067 sigma_old`), `sigma_W2_map=2.158e-05`.
  - `smin=80`: `fNL=-0.72 -45.18 +34.86`, `Delta q50=-0.25`
    (`-0.007 sigma_old`), `sigma_W2_map=1.851e-05`.
- Conclusion: the clean `10k/50k/200k` formal-GIC W2 convergence check does not
  materially change the Task4.3 fNL constraints.  The maximum `nsub=200000`
  center shift is only `0.067 sigma_old`; W2 random subsampling is not the
  limiting systematic for the current minimal lightcone closure result.
