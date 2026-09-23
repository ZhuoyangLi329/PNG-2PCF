# Task43 jaxpower-only 2PCF `smax` scan goal

## Objective

Test whether extending the fitted 2PCF upper separation edge from 350 to
400/450/500/550 Mpc/h improves the local-PNG constraint under one fixed,
matched Task43 baseline.  The deliverable is a relative information test under
the current RR-deconvolved jaxpower diagnostic covariance and radial
single-term RIC model; it is not promoted to a science-ready constraint.

## Frozen experiment contract

- Data: AbacusSummit `base_c000`, `ph000..ph024`, `0.6 < z < 0.8`, real
  space, `Mmin=1.4e13 h^-1 Msun`, with each phase's 25x matching random.
- Measurement: `cucount.jax`, weighted Landy-Szalay,
  `WEIGHT_TOTAL=WEIGHT*WEIGHT_FKP`, `P0=10000`.
- Common vector: one measurement with edges `50..550` and `ds=10` Mpc/h;
  centers are `55..545` and the vector has 50 bins.
- Nested fits: `smax` always denotes an upper **edge**.  The five crops contain
  30/35/40/45/50 bins and end at centers 345/395/445/495/545 Mpc/h.
- Covariance: one 50x50 `covariance_single_realization` from jaxpower, with
  every fit using its leading nested subblock.  Neither covariance-of-mean nor
  the rank-deficient 25-phase sample covariance enters the likelihood.
- jaxpower settings: ph000, 50k data, 100k random, subsample seed 20260702,
  mesh64, pad400, window support `0..3600` with `ds=2`, bessel basis,
  interpolation on 8192 log coordinates, `smooth,fftlog`,
  `k=0.0001..3.0001`, `dk=0.002`, `fNL_cov=0`, `b1_cov=2.5`, `p=1`,
  `sn0=0`, Abacus c000 cosmology.
- RR deconvolution: ph000 100k random, seed 20260702, signed-mu `nmu=20`,
  accepted Task43 ordered-pair normalization and
  `rr_fkp_norm=4.89250917556916e-10`, `kind=RR`, `ell=0`, resolution 1.
- Mean model: L=2000 Mpc/h mother-box FullDiscrete modes, full PNG including
  the quadratic term, `p=1`, shell-averaged xi kernel, fixed `sn0=0`, and one
  factorized radial `IC^(rad,rad)` auto term.  Density-RIC cross terms remain
  outside this test.
- MCMC: five matched chains, 64 walkers x 20,000 steps, burn-in 5,000, seed
  20260720, common priors and common inputs except for the nested upper edge.

## Execution stages and hard gates

1. Build an isolated manifest that references the controlled July-2026
   archive in place; never copy, move, or relink the large catalogs.
2. Run ph000 on one shared GPU.  Require exact 50-bin coordinates, finite
   xi/DD/DR/RR, positive radial RR, correct catalog counts/provenance, and a
   first-30-bin bridge with `max|delta xi|/sigma_old < 0.05`.
3. Run all 25 phases with GPU array concurrency `%1`; validate every phase and
   produce the one common 25-phase mean vector.
4. Build the 50-bin RR window and jaxpower covariance on the login node with at
   most eight CPUs.  Require finite symmetric SPD matrices, no eigenvalue
   flooring, an invertible positive ell=0 RR window, and numerical reproduction
   of the accepted 30-bin covariance block.
5. Reuse the ph000/dchi2/200k/Sobol factorized radial kernel and compile one
   uniquely named 50-bin operator.  Require positive target normalizations,
   finite bases, FullDiscrete-grid agreement, and a first-30-row bridge.
6. Run the five chains.  For every free parameter require post-burn
   `length/tau > 100` and split-median shift `< 0.05 sigma`.
7. Recompute old and new posteriors with common 68.27% quantiles.  Before
   interpreting the ladder, require xi/covariance/operator bridges plus a new
   350 posterior shift `<0.1 sigma_old` and width ratio within 5%.
8. For every adjacent added five-bin block, evaluate the Schur-complement
   residual and compare `25*chi2_cond` with chi-square(5); flag `PTE < 0.01`.
9. Classify improvement only when the 68% half-width shrinks by at least 5%,
   the fNL median moves by less than `0.2 sigma_350`, and all added-tail
   residual gates pass.
10. Write a machine-readable JSON/CSV audit and one PDF-only two-panel ladder;
    update `agent/task.md` and perform a final file/provenance hygiene audit.

## Resource and output isolation

- GPU pair counts: `desi_g`, shared A100, one GPU, 32 allocated CPUs required by
  queue policy but numerical libraries/pair-count threading fixed to one;
  array concurrency at most one.
- All RR, covariance, operator, fitting, and postprocessing work is CPU-only on
  the login node with at most eight CPUs and is never submitted as a CPU Slurm
  job.
- Outputs live only below `outputs/task43_outputs/rmax_scan/`, the core figure
  below `plots/task43/rmax_scan/`, and logs below
  `codes/logs/task43/rmax_scan/`.  Existing Task43 results are immutable bridge
  references and are never overwritten.

## Interpretation boundary

Even if the numerical improvement gate passes, the result must be phrased as a
relative change under the fixed jaxpower Gaussian diagnostic covariance and
single radial-RIC auto term.  The covariance omits connected/non-Gaussian and
full Landy-Szalay terms, the theory and covariance have different low-k support
conventions, and the 350--550 padding/window-angular/RIC-tail convergence has
not been promoted by this single scan.
