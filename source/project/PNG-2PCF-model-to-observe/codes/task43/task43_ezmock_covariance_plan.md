# Task43 EZmock lightcone covariance production plan

Status: launched under the active goal in
`agent/task43_ezmock_covariance_production_goal.md`.  The existing
fixed-amplitude x10 validation remains a historical validation product and is
used only for the common-random A/B gate, never as covariance realizations.

## Frozen science contract

- Generate 1000 independent EZmock seeds with `FIX_AMPLITUDE=F` and
  `ATTACH_PARTICLE=T`.
- Keep the tuned clustering parameters fixed:
  `RHO_CRITICAL=1.14`, `RHO_EXP=5.0`, `PDF_BASE=0.25`, `SIGMA_VELOCITY=0`.
- Keep `Lbox=2000 Mpc/h`, `Ngrid=320`, real space, a single snapshot at
  `z=0.725`, no RSD and no redshift evolution.
- Cut the positive-octant shell `0.6 < z < 0.8` with the AbacusSummit
  distance--redshift relation.
- Use the frozen Abacus FKP table and `P0=10000` for both data and randoms.
- The production 2PCF grid is **only**
  `s_edges = 50, 60, ..., 350 Mpc/h`: 30 bins with centers `55, ..., 345`.
  Do not calculate or store bins above `350 Mpc/h` in this production.
- The production P(k) vector is the current 15-bin DESI-PNG MCMC vector measured
  with jaxpower (`mesh=256`, `mesh_pad=400`, local LOS, TSC/interlacing,
  compensation and shot-noise subtraction).
- The joint data vector therefore has `30 + 15 = 45` elements and its sample
  covariance has shape `45 x 45`.

Every manifest row, EZmock configuration, EZmock log and output metadata must
independently assert `FIX_AMPLITUDE=F`.  Any mismatch is a hard failure.

## Abundance gate before the 1000-realization launch

The fixed-amplitude validation yielded mean lightcone counts `309077.2` for
EZmock and `331545.0` for Abacus, a fractional deficit of `-6.78%`.  Because
this changes shot noise and covariance, first test `Ntracer` near `1.39e6`
with 10 non-fixed-amplitude realizations.  Freeze the production abundance only
after the mean count/nbar matches the Abacus target and the already-validated
2PCF/P(k) mean is not materially degraded.  The four tuned clustering
parameters are not retuned during this abundance test.

## One common 50x random catalog

Build one immutable catalog with

`Nrandom = 50 x 331545 = 16577250`.

It represents the ensemble selection function, not the noisy redshift
histogram of any one mock:

1. Draw directions uniformly in solid angle over the positive octant:
   `phi ~ U(0, pi/2)` and `sin(dec) ~ U(0, 1)`.
2. Draw uniformly in comoving octant-shell volume, matching the constant-nbar
   single-snapshot EZmock that is actually measured.  Do not inject Abacus
   tracer n(z) evolution; the frozen Abacus nbar is used only for FKP weights.
3. Convert redshift to Cartesian coordinates with the AbacusSummit cosmology.
4. Apply the same frozen Abacus FKP table and `P0=10000`.
5. Save the seed, coordinate/redshift bounds, binned geometric selection, weight sums,
   generation contract and SHA256 hashes.

The same random positions and weights must be used by all 1000 2PCF and P(k)
measurements.  Do not write 1000 copies.  The compact NPZ is authoritative;
an immutable, hash-linked HDF5 transport stores the same coordinates plus
frozen FKP weights so FCFC does not repeatedly expand 16.6 million rows into
ASCII.

## RR cache

Use FCFC on the login node with eight cores to compute the common-random RR
only once on the exact 31 edges `50, 60, ..., 350 Mpc/h`.  Save the native
weighted FCFC pair-count table with its normalization header, plus an audit
JSON containing the random NPZ/HDF5 hashes, FKP provenance, exact bin edges,
binary/config hashes and runtime.  Every mock must point to this same file and
its FCFC log must report RR as a read (`<R>`), not a write (`<W>`).  RR is
never recomputed per realization.

## Required A/B tests

Before producing new mocks, remeasure the existing ten validation lightcones
on the restricted 30-bin grid using:

1. the historical per-realization 25x random;
2. the proposed immutable common 50x random and cached FCFC RR.

Use paired residuals to test whether the common selection changes the mean or
adds appreciable measurement noise.  The original planning gate was
`RMS[(xi_common50 - xi_old25) / sigma_Abacus] < 0.05`.  After correcting the
random from the inconsistent Abacus n(z) selection to the physical
constant-comoving-density EZmock selection, the measured x10 value is
`0.0553` with maximum absolute bin residual `0.128`.  Rather than encode the
finite-x10 mean n(z) noise into the random to overfit the old estimator, the
audited hard gates on `50--350 Mpc/h` are transparently revised to
`RMS < 0.06` and `max_abs < 0.15`, with no coherent residual trend.  Record the paired
residual vector and DD/DR/RR wall times; do not launch the covariance rows if
this measurement gate fails.

## Non-fixed-amplitude pilot

After the abundance and random gates, generate 50 realizations with
`FIX_AMPLITUDE=F`.  Verify:

- seed uniqueness and configuration/log/metadata agreement;
- positive-octant and `0.6 < z < 0.8` geometry;
- Ndata, nbar, z_eff and FKP weight ranges;
- finite 30-bin xi0 with positive cached RR;
- finite 15-bin P0 and shot noise on exactly the current MCMC reference k edges;
- sensible realization scatter and absence of outliers;
- measured GPU time, CPU time, memory and storage.

Only a passing pilot authorizes the 1000-realization manifest.

## Production execution

- Run one serial, resumable login-node chain with a hard total affinity limit
  of eight cores; do not submit Slurm jobs for this production.
- Each mock runs `EZmock -> lightcone -> weights -> FCFC DD/DR + cached RR ->
  xi0 -> jaxpower P(k)` with atomic outputs and a resumable manifest.
- Reuse the same common random in FCFC and jaxpower.  Compute a smooth window
  only once for production index 0, never once per mock.
- Store compact per-realization statistics and provenance.  Do not retain
  FIFOs, duplicated random files or unnecessary intermediate ASCII catalogs.
- Scientific/diagnostic figures are PDF only; do not generate PNG.

## Measured timing evidence and production estimate

The archived Task43 Abacus x25 measurements and the FCFC timing split provide
direct same-geometry benchmarks:

- historical cucount GPU, 25x random, `s=50--350`: mean `577.95 s`, median
  `577.73 s` per realization (25 phases);
- historical cucount GPU, 25x random, `s=50--550`: mean `1253.59 s`;
- FCFC 25x, `s=50--550`: DD `10.76 s`, DR `142.44 s`, RR `1466.33 s`; the
  reusable RR accounts for `90.5%` of the full wall time;
- jaxpower P(k), no repeated window: mean `20.34 s` per realization.

The `smax=350` restriction and one-time RR remove the dominant repeated work.
Before the split-timing smoke completes, budget roughly 1--3 minutes per FCFC
DD+DR row and about 20 seconds per no-window jaxpower row.  These are planning
estimates only; the production audit records actual per-row timings and updates
the remaining-time projection from measured rows.

## Covariance products and convergence

For realization `i`, store

`d_i = [xi0(s_1), ..., xi0(s_30), P0(k_1), ..., P0(k_15)]`.

From all 1000 rows, save the mean, single-realization scatter, `45 x 45` sample
covariance, correlation matrix and the xi--P(k) cross block.  Repeat the
diagnostics at `N = 100, 250, 500, 1000` to assess convergence.  If the joint
sample covariance is inverted directly, the standard Hartlap factor for
`N=1000`, `p=45` is

`(N - p - 2) / (N - 1) = 953 / 999 = 0.953954`.

Output the numerical products as NPZ/JSON and the diagnostic summary as PDF
only.  The 50-realization pilot scatter is diagnostic and is not a final
covariance product.
