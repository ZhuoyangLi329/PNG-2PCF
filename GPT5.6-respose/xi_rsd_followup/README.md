# Xi-only RSD follow-up: frozen P(k), linear checks, and Gaussian streaming

The full Chinese review is `README_zh.md` in the repository directory:
https://github.com/ZhuoyangLi329/PNG-2PCF/tree/main/GPT5.6-respose/xi_rsd_followup

Source review pinned to `e6f217b3e4890a22ac866b0a840ac9d6ae00056a`.
The production P(k) model, its settings and shared caches are NOT modified.
The old recommendation to upgrade the P(k) fitting model is not the execution
plan for this follow-up.

## What the two choices mean

A correct linear configuration-space RSD implementation is equivalent to
Kaiser for identical mode support. The existing code's apply-Kaiser-then-
transform ordering is not itself an error. A full Gaussian streaming integral
with leading-order moments adds selected higher-order mapping effects. It is
not a different linear theory and is not a full nonlinear/PNG prediction.

Keeping the production P fit unchanged permits testing an independent xi
model. Requiring, in addition, that xi be the exact transform of the same
fully specified P(k,mu) at every k removes that freedom: only numerically
equivalent rewrites are then possible. Compare the alternative models on
their supported scales; do not silently demand equality of unlike nuisance
parameters or replace the frozen P pipeline.

## Files

- `xi_rsd_reference.py`: independent xi-side linear mode moments, signed
  Gaussian pair mapping, old FoG's exact LOS kernel, and final shell projection.
- `test_xi_rsd_reference.py`: 11 synthetic tests, including GSM's linear limit
  and negative controls for two incorrect streaming implementations.
- `validation.json`: results actually produced by the test command and source
  SHA256 values. Tests overwrite only this result file in their own directory.
- `README_zh.md` (repository): full derivation, source audit, PNG and periodic-box
  requirements, five experiments, and execution-agent handoff.

Run from this directory with Python, NumPy and SciPy:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python test_xi_rsd_reference.py
```

## Important boundaries

No original rawbox data, catalog measurements, fits or MCMC were run here.
The GSM integrator uses an infinite-LOS integral with explicit numerical
truncation; its returned quadrature error does not include tail, periodic
image, moment-interpolation or physical-model errors. Production periodic
images and finite-mu FCFC projection remain separate required checks.

Input pair variances are CENTRAL variances; the transverse variance is ONE
component. All velocities must first be converted to displacements in the
same length unit as r. Stream `1+xi_real`, not already redshift-space xi. Do
not apply Kaiser or the old FoG again. The real-space angle is y/r, not the
observed s_parallel/s. Never renormalize the position-dependent kernel along
the integration path. The old Lorentzian extra kernel has variance
`2*sigma_s**2`; that is not the total GSM pair variance.

The first science check should be measured-real-space-input GSM closure on
paired catalogs, with extra variance initially zero. This isolates the
mapping approximation but is not an independent theory validation. Only
then replace the measured moments with theoretical moments and validate
held-out phases, PNG response, and parameter coverage.
