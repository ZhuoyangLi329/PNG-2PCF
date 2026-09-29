# Review-data array inventory

The NPZ files here are compact, explicitly selected derivatives of NERSC products. Every file is described by `ARRAY_INDEX.json` with array names, shapes, and dtypes. Posterior excerpts are deterministic subsets of the original flattened samples and are supplied for rapid plotting/review only; they are not independent chains and must not replace the original convergence audits.

The full 9.22 formal_mean4 post-burn `samples.npz` files are retained under the mirrored project path because they are below the GitHub size limit and are useful for checking the reported marginals. Original `chain.h5`, raw catalogs, and very large caches are excluded.

The 9.30 diagnostic arrays are preserved at their original mirrored project paths, not added as posterior excerpts here. Their explicit array/role inventory is `provenance/task432_20260929_array_index.json`; the short xi/joint chains and inherited P excerpt must not be treated as independent, uniformly corrected formal posteriors.
