# Reproducibility and exclusions

This mirror is a review package, not a runnable clone of the full NERSC project.

Included: active source snapshots, meeting PDFs/JSON/Markdown, compact audits, fit summaries, and literature notes.

Excluded deliberately: raw lightcone/halo/random catalogs, full covariance NPZ/HDF5 caches, MCMC `chain.h5`, large `samples.npz`, temporary logs, and intermediate binary kernels. These remain on NERSC in the source project and are referenced by the original provenance metadata.

The exclusion is intentional to keep the private GitHub review package small and to avoid uploading large data products. The exact mirrored file list and byte sizes are in `provenance/uploaded_files.tsv`.
