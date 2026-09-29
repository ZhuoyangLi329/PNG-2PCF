# Task432 velocileptors/GSM full-RIC repair plan

## Objective

Replace only the xi0/xi2 forward model with the trusted velocileptors/GSM implementation, while preserving the frozen P-side model, data vector, and covariance. Add the same full radial integral-constraint operator used by the formal hybrid analysis.

## Stage 1: operator implementation

1. Reuse the frozen Task432 geometry kernels and Response projection code.
2. Evaluate velocileptors xi0(r) and xi2(r) on the full internal radial grid required by the RIC kernel, not only the 52 fitted bins.
3. Project the full radial xi multipoles through the midpoint LS response, including cross and auto terms and the production RR normalization.
4. Add the consistent fixed data/random Poisson terms and the existing stochastic sn0 response.
5. Preserve the P0/P2 full-RIC branch and the original 74-vector ordering.

## Stage 2: validation gates

- Direct-vs-emulator xi prediction error below the existing C_single quadratic-form thresholds.
- RIC kernel convergence under radial-grid, quadrature, and independent random-pool refinement.
- Zero-PNG and fixed-bias xi-only closure with finite residuals and no boundary-driven solution.
- Compare raw xi-only chi2, b1, fNL, and residuals against the old FullDiscrete/hybrid xi model.
- No formal joint chain until all xi-only gates pass.

## Stage 3: joint sensitivity

Run short MAP/smoke fits first, then one diagnostic joint chain using the unchanged EZmock1000 C_single. Report P-only, xi-only, joint b1 and the full-cross versus block-diagonal sensitivity. Do not replace the formal result yet.

## Stage 4: covariance matching

In parallel, quantify the mismatch between the halo phase random policy (`data_observed_redshift_resample`) and the EZmock common-50 covariance random policy. Build a halo-matched paired covariance/cross-block test before promoting any new joint constraint.

## Current formal status

All products in this folder are diagnostic until the full-RIC velocileptors operator, xi-only closure, and covariance-policy audit pass.
