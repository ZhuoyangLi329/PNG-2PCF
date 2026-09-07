# Mission 11: k-rebinning validation on 1Gpc fastPM + Quijote

## Summary

| Dataset                      |  fnl_fit |  t_base |   t_ER |   t_CR |  |D/s|_B | |D/s|_ER | |D/s|_CR |     rel_ER |     rel_CR |  x_ER |   x_CR |
|------------------------------|----------|---------|--------|--------|----------|----------|----------|------------|------------|-------|--------|
| 1Gpc fastPM fnl100           |    101.2 |     7.1 |   0.39 |  0.014 |   0.2265 |   0.2295 |   0.2295 |   2.11e-01 |   1.97e-01 |    19x |    514x |
| 1Gpc fastPM fnl0             |      5.3 |     6.9 |   0.42 |  0.014 |   0.2458 |   0.2488 |   0.2489 |   6.06e-03 |   6.25e-03 |    16x |    512x |
| Quijote fnl=0                |      0.4 |     9.5 |   0.52 |  0.018 |   0.1068 |   0.1066 |   0.1063 |   6.66e-03 |   6.82e-03 |    18x |    521x |
| Quijote fnl=20               |     19.5 |     9.2 |   0.44 |  0.018 |   0.1048 |   0.1044 |   0.1042 |   7.25e-03 |   7.43e-03 |    21x |    519x |
| Quijote fnl=30               |     29.1 |     9.0 |   0.37 |  0.017 |   0.1032 |   0.1029 |   0.1027 |   7.68e-03 |   7.88e-03 |    24x |    543x |
| Quijote fnl=50               |     48.3 |     8.8 |   0.42 |  0.017 |   0.0997 |   0.0994 |   0.0992 |   9.33e-03 |   9.53e-03 |    21x |    530x |
| Quijote fnl=75               |     72.3 |     8.8 |   0.46 |  0.018 |   0.0946 |   0.0944 |   0.0943 |   1.23e-02 |   1.27e-02 |    19x |    483x |
| Quijote fnl=100              |     96.4 |     9.4 |   0.49 |  0.017 |   0.0889 |   0.0889 |   0.0888 |   3.40e-02 |   3.59e-02 |    19x |    541x |

## Notes

- All boxes L=1Gpc, kmax=15, shells=4749436
- CachedRebin precompute time: 0.30s (one-time cost)
- Covariance: fastPM uses fnl0 fastPM (50 files); Quijote uses fnl=0 fid (500 files)
- rel_ER/rel_CR = mean|Dxi/xi_baseline|, measures numerical accuracy vs exact discrete sum
- |D/s| = mean|Delta/sigma| vs measured 2PCF data, measures physical agreement
