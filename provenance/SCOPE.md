# Scope and provenance

This replacement commit is intentionally a new review snapshot. It does not erase Git history or the other existing branches. It removes the old main-tree review files from the current branch and replaces them with a Task 4.3 lightcone/RSD package. `replaced_tree.json` records the old main-tree entries.

The source project remains on NERSC at `/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe`. The mirror includes source hashes, compact result arrays, final PDFs, meeting JSON/Markdown, and explicit exclusion notes. The mirror is not a full runnable clone because raw catalogs, huge covariance/geometry caches, and HDF5 chains remain on NERSC.

## Incremental update, 2026-09-29

The 9.30 pipeline update adds to this existing review tree; it does not perform another replacement. Seven new NERSC source scripts and selected small products are copied byte-for-byte, recorded in `task432_20260929_sources.json`. Existing 9.18/9.22 scientific snapshots and results remain unchanged. Updated reader documentation fixes old bin counts, stale README links, and the distinction between early hybrid and full-RIC sn0 parameters. The new offline verifier checks archive integrity only.
