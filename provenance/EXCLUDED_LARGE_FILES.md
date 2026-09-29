# Reproducibility and exclusions

This mirror is a review package, not a runnable clone of the full NERSC project.

Included: active source snapshots, meeting PDFs/JSON/Markdown, compact audits, fit summaries, and literature notes.

Excluded deliberately: raw lightcone/halo/random catalogs, MCMC `chain.h5`, temporary logs, and production caches/chains too large for the selected review snapshot. Compact covariance/geometry arrays and selected formal `samples.npz` are included where explicitly indexed; this is not a blanket exclusion of every NPZ. Omitted products remain on NERSC and are referenced by the original provenance metadata.

The exclusion is intentional to keep the GitHub review package small and to avoid uploading large data products. The original snapshot file list and byte sizes are in `provenance/uploaded_files.tsv`; the 2026-09-29 increment is recorded separately in `provenance/task432_20260929_sources.json`.

The 2026-09-29 increment includes the compact coarse/local velocileptors full-RIC grids and the 9.30 diagnostic-chain NPZ. Per-fNL intermediate rows and the compile log are excluded because the consolidated grids and manifests are included. This selection does not imply that the repository contains every production cache or environment required by the hard-coded NERSC entry points.
