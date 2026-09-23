# Task 4.3.2 interim findings

## Current contract

- `50 <= s < 350 Mpc/h`.
- Remove `80 <= s < 120 Mpc/h`, i.e. centers `85, 95, 105, 115 Mpc/h`.
- Box-safe lightcone `0.4 < z_obs < 0.8`.
- Lightcone P2 active cut `k >= 0.015 h/Mpc`.
- Curves are arithmetic means of 25 phases; likelihood and error bars use `C_single` without `/25`.
- `sn0` is P-side only; the xi contact term is fixed to zero.

## Model audit

The Task43 Kaiser x squared-Lorentzian-FoG algebra and the shell-averaged `j2`
kernel were checked without changing production code.

- `j2` fixed 24-point shell quadrature vs analytic primitive:
  relative L2 `1.06e-14`, max absolute difference `1.23e-14`.
- Continuous-angle Gauss integration converges well at `nmu=96/128`.
  At `nmu=64` and representative `sigma_s=8`, the relative L2 difference
  against `nmu=192` is about `1e-6` for ell=2, too small to explain the
  several-Mpc/h sigma discrepancy by itself.
- Rawbox P2 uses exact parent-mode angular values, while current xi FullDiscrete
  uses continuous angular shell moments.  The exact-lattice minus continuous
  low-k xi2 operator difference reaches about `7.5%` in relative L2 for
  `k_switch=0.005--0.01 h/Mpc`; xi0 is about `1e-3` or smaller.

The finite-lattice effect is therefore a required operator audit, but the first
matched fit shows that it is not sufficient by itself to move the joint result.

## Rawbox BAO-masked split-sigma diagnostic

Long chains: `64 walkers x 12000 steps`, burn-in `2000`, fixed diagnostic
covariance, P02 16 bins and xi0/xi2 26 bins each.

| fit | `b1` q50 | `sigma_s` / `sigma_s_P` | `sigma_s_xi` |
|---|---:|---:|---:|
| P marginal | `2.5763` | `0.99` | -- |
| xi marginal | `2.6073` | -- | `4.10` |
| shared joint | `2.5764` | `0.98` | -- |
| split joint | `2.5764` | `1.01` | `4.25` |

The split model clearly separates the effective P and xi velocity nuisance,
but joint `b1` being close to the P-side constraint is not itself a problem.
The relevant diagnostic is whether joint `b1` lies outside the interval spanned
by the P and xi marginal results, or undergoes a large conditional shift away
from both marginals.

## Box-safe lightcone BAO-masked split-sigma diagnostic

Long chains: `64 walkers x 12000 steps`, burn-in `2000`, P0=15 bins, P2=11
bins, xi0/xi2=26 bins each, same frozen `C_single` family and cross block.

| fit | `b1` q50 | `sigma_s` / `sigma_s_P` | `sigma_s_xi` |
|---|---:|---:|---:|
| P marginal | `2.3684` | `1.16` | -- |
| xi marginal | `2.4080` | -- | `5.14` |
| shared joint | `2.3668` | `0.99` | -- |
| split joint | `2.3636` | `1.09` | `3.67` |

Allowing separate sigma parameters changes joint `b1` by only about `0.003`.
In this matched active contract the joint value is essentially on the P side,
which is acceptable because P is the tighter probe; this result does not by
itself indicate a joint-b1 failure.

## Current interpretation

1. The P/xi sigma discrepancy is real within the current diagnostic model.
2. Joint b1 must be judged against the convex interval between the P and xi
   marginals, not by whether it follows P or xi.
3. Under the current matched BAO-masked contract, joint b1 is close to P and is
   not itself anomalous.
4. The older 9.11 meeting contract must be reproduced exactly before its apparent
   joint-b1 displacement can be called a real discrepancy.
5. The next model work should focus on the xi measurement/operator contract:
   exact finite-lattice angular projection for rawbox, FCFC `(s,mu)` projection,
   higher multipoles in the lightcone xi forward model, and scale-dependent or
   GSM pairwise velocity structure.
6. The split-sigma results must remain diagnostic-only until xi shape closure is
   re-tested with the repaired operator/model.

## Existing GSM diagnostics

The repository already contains same-catalog empirical-moment GSM diagnostics
under `outputs/task43_outputs/rsd_validation/rawbox/xi_rsd_v2/`. They are oracle
tests with no fitted independent theory parameters, so they are not a production
replacement yet.

- Refined empirical GSM still has large no-fit closure chi2 and is explicitly
  marked as same-catalog/oracle scope.
- The density/moment swap test shows that measured versus leading-theory pair
  velocity moments materially changes the xi0/xi2 residuals.
- This supports a later scale-dependent pair-velocity/GSM branch, but the next
  minimal repair remains the exact measurement/operator audit and matched
  split-sigma comparison.

## Joint-b1 convex-interval check

The joint-b1 criterion must be evaluated within one identical data/covariance
contract. It is acceptable for joint to follow the tighter P-side constraint;
the failure signature is joint lying appreciably outside the interval spanned by
the P and xi marginal centers.

- The 9.11 legacy meeting contract has q50 values approximately
  `P02=2.414`, `xi02=2.408`, `joint=2.354`; the joint is below both marginal
  centers and is therefore a genuine old-contract anomaly.
- The current promoted BAO-masked active contract gives approximately
  `P02=2.368`, `xi02=2.408`, `joint=2.364`; this is essentially P-side and
  differs from P by only about `0.005`, so it is not the same anomaly.
- The old 9.11 plot uses a different P2/bin/covariance contract from the current
  active 15+11-bin diagnostic, so its joint-b1 displacement must be reproduced
  under that exact old contract before attributing it to the xi RSD model.

## Rawbox scale-dependent FoG diagnostic

The two-band xi model uses `k_split=0.08 h/Mpc` and independent
`sigma_xi_low/high`, while the P-side model is unchanged.

- Long-chain joint MAP chi2: `2.9858`.
- Shared/split-sigma rawbox joint MAP chi2: about `3.09/2.9875` in the same
  diagnostic family.
- Scale-dependent joint `b1` median: `2.5745`, still following the P-side
  `b1=2.5762` rather than moving toward xi-only `b1=2.6111`.
- The xi-side posterior prefers broad, poorly separated low/high effective
  dispersions, so the two-band extension is not a meaningful closure repair.

This closes the Phase-4 scale-dependent-FoG test: a constant/shared sigma is
not the only issue, but this simple k-band extension does not solve the xi
shape or joint-b1 behavior either.

## Large-scale linear-input GSM diagnostic

The new independent implementation is
`codes/task432/task432_linear_gsm_rawbox.py`.  It follows the pair-conserving
GSM of Scoccimarro (2004) and Reid & White (2011), while using leading linear
real-space/velocity moments rather than CLPT or EFT.  The P-side remains the
current Task43 low-k model.  The xi-side therefore does not use the frozen
FullDiscrete redshift-space cache at k=3--5 h/Mpc.

The model has two variants:

- pure linear GSM: no new xi-side fitted parameter;
- eBOSS-style linear GSM + one `sigma_FOG`, added as an extra LOS pairwise
  variance in (Mpc/h)^2.  This is a phenomenological configuration-space
  velocity width, not an EFT counterterm.

The first implementation exposed and fixed one angular-projection bug: the
monopole requires the explicit factor `1/2` multiplying the integral over
mu in [-1, 1].  The corrected code was syntax-checked locally and rerun on
NERSC.

### k_max_GSM scan

The full quadrature scan used `shell_order=4`, `angular_order=12`,
`stream_order=8`, `zmax=8`, and tested values beyond the initial proposal:

| k_max_GSM [h/Mpc] | pure linear GSM joint MAP chi2 | + sigma_FOG joint MAP chi2 | joint b1 (+ sigma_FOG) | sigma_FOG [Mpc/h] |
|---:|---:|---:|---:|---:|
| 0.15 | 42.17 | 42.17 | 2.5366 | 0.00 |
| 0.25 | 51.85 | 11.66 | 2.5626 | 16.68 |
| 0.50 | 49.76 | 5.68 | 2.5717 | 16.92 |
| 0.75 | 47.67 | 6.43 | 2.5710 | 16.71 |
| 1.00 | 48.57 | 6.26 | 2.5712 | 16.74 |
| 2.00 | 47.18 | 6.31 | 2.5713 | 16.61 |

Interpretation:

1. Pure linear GSM without a dispersion width is not sufficient for the
   current xi0/xi2 closure, even though the fitted range starts at
   50 Mpc/h.
2. Adding one eBOSS-style `sigma_FOG` changes the joint MAP chi2 from about
   50 to about 5.7 at k_max=0.5 h/Mpc, while keeping joint b1 essentially on
   the P-side value.
3. The cutoff is not converged at 0.15 h/Mpc and is still changing strongly
   at 0.25 h/Mpc, but it is stable from roughly 0.5 through 2.0 h/Mpc.
4. The preferred `sigma_FOG` is about 16.7 Mpc/h in the current displacement
   convention.  It should not be identified with the old xi-side constant
   `sigma_s_xi` without matching definitions.
5. The eBOSS-style one-parameter GSM can reduce the P+xi joint chi2 to about
   5.7, but its xi-only chi2 remains about 67, so the apparent joint improvement
   is not evidence that the xi model is correct.  It remains diagnostic-only.

### Streaming-integral convergence

At k_max_GSM=0.5 h/Mpc with the one-parameter `sigma_FOG` branch:

| streaming setting | joint MAP chi2 |
|---|---:|
| zmax=6 | 5.675570 |
| zmax=8 | 5.675568 |
| zmax=10 | 5.675568 |
| stream quadrature order=8 | 5.675568 |
| stream quadrature order=12 | 5.675347 |

The streaming line-of-sight integral is therefore converged at the displayed
precision for the adopted large-scale contract.  The remaining discrepancy
with the current FullDiscrete baseline is model/input physics, not an
integration-tail artifact.

### Current linear-GSM conclusion

For the present `s >= 50 Mpc/h` rawbox analysis, the current full linear-input
GSM is not yet a valid replacement for the xi RSD model.  The controlled next
route is to validate a first-order streaming baseline and then use CLPT-GS
moments before applying the full Gaussian mapping, without adding EFT at first.

```text
shared:  fNL, b1
P-side:  current sigma_s_P, sn0
xi-side: first-order streaming baseline or CLPT-GS moments + GSM
```

The current full linear-GSM plus sigma_FOG branch remains a diagnostic model
until its xi-only residuals and Kaiser-limit closure pass; its fitted
sigma_FOG should not be interpreted as a physical velocity constraint.

## Kaiser-limit audit result

The audit products are:

- `outputs/task43_outputs/rsd_validation/task432_model_repair/gsm_kaiser_limit_audit/task432_gsm_kaiser_limit_audit.json`
- `plots/task43/rsd_validation/task432_model_repair/task432_gsm_kaiser_limit_audit.pdf`

At k_max=0.5 h/Mpc, first-order streaming differs from direct Kaiser by about
10% in xi0 and 5.8% in xi2 on the promoted mask.  The full Gaussian GSM
differs from first-order streaming by about 4.6% in xi0 but about 107% in xi2.
At k_max=1.0 h/Mpc the first-order xi0/xi2 differences are about 4.1%/5.3%,
while the full-GSM xi2 correction is still about 105% relative to first order.

Changing the finite-difference step from h=0.02 to 0.10 Mpc/h changes the
reported first-order differences below the displayed precision.  The large
full-GSM xi2 discrepancy is therefore not a derivative-step or missing
projection-factor artifact.  It is caused by applying a full Gaussian
resummation to leading linear velocity moments that have not been upgraded to
consistent nonlinear/CLPT moments.

## Velocileptors GSM cross-check

The missing package was installed into the user desilike environment as
`velocileptors-3.1` with `pyfftw-0.15.1`; the environment needs two compatibility
aliases (`scipy.signal.tukey` and `np.trapezoid`) for this older NumPy/SciPy
combination.

Using the same rawbox xi0/xi2 data, promoted mask and xi covariance, with
`kmax=0.5 h/Mpc`, `b1_L=b1_E-1`, b2/bs/b3 fixed to zero, all EFT-like alpha
parameters zero and s2fog zero, the trusted velocileptors GSM gives:

| model | b1 | xi-only MAP chi2 |
|---|---:|---:|
| velocileptors GSM, b1 free | `2.6393` | `3.935` |
| velocileptors GSM, b1 fixed 2.55 | `2.5500` | `6.884` |
| custom first-order streaming | `2.3499` | `41.214` |
| custom full linear GSM + sigma_FOG | `2.3937` | `67.36` |

At s=55 Mpc/h, velocileptors with b1=2.55 gives approximately
`xi0=0.02123` and `xi2=-0.03071`, compared with measured means
`xi0=0.02270` and `xi2=-0.03104`.  Its residuals stay at a few 1e-4 across
the promoted xi data vector.

This resolves the main diagnostic question: the GSM real-to-redshift mapping
is viable, while the custom linear radial-moment provider/convention is not.
The next implementation should use the trusted CLPT/GSM moment machinery as
the xi RSD baseline and add PNG scale-dependent bias consistently, rather than
refitting the current custom sigma_FOG branch.

Cross-check products:

- `plots/task43/rsd_validation/task432_model_repair/task432_velocileptors_gsm_crosscheck.pdf`
- `outputs/task43_outputs/rsd_validation/task432_model_repair/velocileptors_gsm_crosscheck/velocileptors_gsm_crosscheck/task432_velocileptors_gsm_crosscheck.json`

## Hybrid PNG velocileptors GSM

A minimal PNG response was added on top of the trusted velocileptors GSM
baseline, without EFT, b2/bs/b3 or s2fog.  The response adds the current
scale-dependent PNG bias to the real-space tracer P response and the leading
density--velocity cross response; the velocity--velocity baseline remains the
Gaussian CLPT/GSM prediction.

The xi-only MAP is:

```text
fNL = 4.78
b1  = 2.616
chi2 = 4.16
```

The xi-only hybrid posterior gives approximately:

```text
fNL = -1.42 +11.96/-12.97
b1  = 2.640 +0.058/-0.058
```

The hybrid `b1` is compatible with the original FullDiscrete xi02 result
`b1=2.607+0.057/-0.058` and is far closer than the custom linear-GSM value
`b1≈2.394`.  The corresponding 2PCF-only contour PDF is:

- `9.22meeting/task432_2pcf_rsd_comparison/task432_2pcf_original_vs_png_velocileptors_gsm_9.22style.pdf`

This is still a hybrid PNG-response diagnostic, not a complete CLPT-PNG bias
operator basis, but it passes the present rawbox xi-only closure much better
than the custom linear-moment GSM.

## Machine products

- `outputs/task43_outputs/rsd_validation/task432_model_repair/audits/task432_baseline_contract.json`
- `outputs/task43_outputs/rsd_validation/task432_model_repair/audits/task432_rsd_model_audit.json`
- `outputs/task43_outputs/rsd_validation/task432_model_repair/rawbox_baomask_split_sigma_longchain/audits/task432_split_sigma_summary.json`
- `outputs/task43_outputs/rsd_validation/task432_model_repair/lightcone_baomask_split_sigma_longchain/audits/task432_lightcone_split_sigma_summary.json`
- `plots/task43/rsd_validation/task432_model_repair/task432_rsd_model_audit.pdf`
- `codes/task432/task432_linear_gsm_rawbox.py`
- `outputs/task43_outputs/rsd_validation/task432_model_repair/rawbox_linear_gsm_kmax_scan/audits/task432_linear_gsm_kmax_scan.json`
- `outputs/task43_outputs/rsd_validation/task432_model_repair/rawbox_linear_gsm_fog_kmax_scan/audits/task432_linear_gsm_kmax_scan_fog.json`
- `plots/task43/rsd_validation/task432_model_repair/task432_linear_gsm_kmax_scan.pdf`
- `plots/task43/rsd_validation/task432_model_repair/task432_linear_gsm_kmax_scan_fog.pdf`

## Original-vs-GSM contour comparison

The contour PDF is:

- `plots/task43/rsd_validation/task432_model_repair/task432_original_vs_linear_gsm_contours.pdf`

It overlays the original 64-walker split-sigma FullDiscrete joint chain with a
new `k_max_GSM=0.5 h/Mpc` linear-GSM + `sigma_FOG` joint chain under the same
promoted BAO-mask and fixed joint covariance.

The preliminary posterior summaries are:

| parameter | original split-sigma | linear GSM + sigma_FOG |
|---|---:|---:|
| `fNL` | `0.743 +11.11/-11.85` | `0.515 +11.35/-10.95` |
| `b1` | `2.5764 +0.0453/-0.0455` | `2.5712 +0.0459/-0.0443` |
| `sigma_s_P` | `1.006 +0.818/-0.693` | `1.055 +0.833/-0.719` |
| xi-side width | `sigma_s_xi=4.252 +2.966/-2.844` | `sigma_FOG=16.971 +1.144/-1.097` |
| `sn0` | `0.1505 +0.0924/-0.0925` | `0.1637 +0.0900/-0.0945` |

The shared `fNL`, `b1`, P-side sigma and `sn0` contours are nearly unchanged.
The xi-side width values are not numerically comparable because
`sigma_s_xi` is a high-k Fourier FoG parameter whereas `sigma_FOG` is an
extra configuration-space LOS pair-variance width.

The new contour chain used 16 walkers, 900 steps and 200 burn-in steps, with
post-burn length/tau between about 16 and 21, so this PDF is a stable
diagnostic comparison but not a replacement for the original long-chain
precision.

The meeting-style all-parameter corner is now written to:

- `9.22meeting/task432_linear_gsm_vs_original/task432_original_vs_linear_gsm_allparams_9.22style.pdf`

It follows the existing 9.18/9.22 convention: a lower-triangle 5x5 corner,
raw-chain diagonal histograms, filled and line 68%/95% contours, inward ticks,
shared axis ranges, and all free parameters `fNL`, `b1`, `sigma_s_P`, the
xi-side width, and `sn0`.

In that figure, the xi-side display column is explicitly labeled
`sigma_xi/GSM`: the original chain supplies `sigma_s_xi`, while the new chain
supplies `sigma_FOG`.  They are displayed in the same column for visual
comparison but are not claimed to be the same physical parameter.

## 2PCF-only contour comparison

The correct 2PCF-only comparison is now in:

- `9.22meeting/task432_2pcf_rsd_comparison/task432_2pcf_original_vs_gsm_9.22style.pdf`

This PDF contains separate full corners for:

1. original FullDiscrete `xi0+xi2` marginal with `fNL`, `b1`, `sigma_s_xi`;
2. linear GSM `xi0+xi2` marginal with `fNL`, `b1`, `sigma_FOG`;
3. an overlay only for the shared `fNL` and `b1` parameters.

The 2PCF-only posterior summaries are:

| parameter | original xi02 | linear GSM xi02 |
|---|---:|---:|
| `fNL` | `3.81 +12.14/-12.43` | `5.19 +15.79/-15.26` |
| `b1` | `2.6073 +0.0573/-0.0578` | `2.3937 +0.0576/-0.0603` |
| xi-side width | `sigma_s_xi=4.10 +1.07/-1.55` | `sigma_FOG=15.16 +0.31/-0.32` |

Thus the GSM replacement changes the xi-only `b1` constraint substantially,
by about `-0.214`, while the fNL shift is modest compared with its broad
uncertainty.  This is the relevant model-to-model 2PCF comparison; the nearly
unchanged b1 seen in the earlier joint plot was caused by the P-side constraint
and should not be used as the 2PCF-only comparison.

During contour preparation I separately checked the angular projection.  The
xi2 projection already contains the correct `(5/2)` factor for the full
mu=-1..1 integral; the only actual normalization fix was the xi0 `(1/2)`
factor.  The remaining linear-GSM xi2 difference is therefore a velocity
moment/streaming-model closure issue, not a second missing projection factor.

## Kaiser-limit audit result

The new audit products are:

- `outputs/task43_outputs/rsd_validation/task432_model_repair/gsm_kaiser_limit_audit/task432_gsm_kaiser_limit_audit.json`
- `plots/task43/rsd_validation/task432_model_repair/task432_gsm_kaiser_limit_audit.pdf`

At k_max=0.5 h/Mpc, first-order streaming differs from direct Kaiser by about
10% in xi0 and 5.8% in xi2 on the promoted mask, while the full Gaussian GSM
differs from first-order streaming by about 4.6% in xi0 but 107% in xi2.  The
finite-difference step scan h=0.02--0.10 Mpc/h is stable, so the large xi2
effect is a genuine inconsistency of inserting leading linear moments into the
full Gaussian exponent, not a derivative-step artifact.

## Velocileptors GSM cross-check

The package `velocileptors-3.1` with `pyfftw-0.15.1` was installed in the user
desilike environment, with compatibility aliases for the current NumPy/SciPy
versions.  Using the same rawbox xi0/xi2 data, promoted mask and xi covariance,
with EFT-like parameters and s2fog fixed to zero, the trusted GSM gives:

| model | b1 | xi-only MAP chi2 |
|---|---:|---:|
| velocileptors GSM, b1 free | `2.6393` | `3.935` |
| velocileptors GSM, b1 fixed 2.55 | `2.5500` | `6.884` |
| custom first-order streaming | `2.3499` | `41.214` |
| custom full linear GSM + sigma_FOG | `2.3937` | `67.36` |

This shows that the GSM mapping is viable and the custom linear moment provider
was the failed component.  The cross-check PDF is
`plots/task43/rsd_validation/task432_model_repair/task432_velocileptors_gsm_crosscheck.pdf`.

## Hybrid PNG velocileptors GSM

A minimal PNG response was added to the trusted velocileptors GSM baseline,
without EFT, b2/bs/b3 or s2fog.  The rawbox xi-only MAP is:

```text
fNL = 4.78
b1  = 2.616
chi2 = 4.16
```

The hybrid xi-only posterior gives approximately:

```text
fNL = -1.42 +11.96/-12.97
b1  = 2.640 +0.058/-0.058
```

The hybrid b1 is compatible with the original FullDiscrete xi02 result and is
far closer than the custom linear-GSM result.  The rawbox hybrid contour is
`9.22meeting/task432_2pcf_rsd_comparison/task432_2pcf_original_vs_png_velocileptors_gsm_9.22style.pdf`.

## Standard RSD halo lightcone mock

The hybrid model was then applied to the standard Task43 RSD halo lightcone
contract: box-safe 0.4 < z_obs < 0.8, P0=13 bins below kmax=0.08 h/Mpc,
P2=9 bins with k>=0.015 h/Mpc, xi0/xi2=26 bins each, and the same promoted
50--350 Mpc/h range with 80--120 Mpc/h removed.

Original FullDiscrete lightcone results were approximately:

```text
xi-only:  b1 = 2.408 +0.101/-0.103, fNL = -3.92 +27.39/-28.63, MAP chi2 = 5.00
joint:   b1 = 2.341 +0.046/-0.046, fNL = 2.12 +22.70/-25.18, MAP chi2 = 12.54
```

The hybrid PNG velocileptors/GSM results are approximately:

```text
xi-only:  b1 = 2.426 +0.093/-0.099, fNL = -6.18 +25.02/-28.81
joint:   b1 = 2.389 +0.064/-0.054, fNL = -12.56 +22.66/-25.06
```

The hybrid xi-only b1 is close to the original lightcone xi-only constraint,
while the joint b1 shifts upward by about 0.05.  The lightcone comparison PDF
is `9.22meeting/task432_lightcone_png_comparison/task432_lightcone_original_vs_png_gsm_9.22style.pdf`.
