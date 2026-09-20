#!/usr/bin/env python3
"""
Generate a small synthetic expression matrix for fast Compass integration tests.

Design (see TESTING_PLAN.md, section 7):
  - Gene expression counts per cell are drawn from a negative-binomial
    distribution: counts[g, c] ~ NB(mu_g * s_c, phi), where mu_g is a
    gene-specific baseline expression level and s_c is a cell-specific
    size factor. This mimics the over-dispersion of real scRNA-seq counts.
  - Gene names are drawn from the gene-protein associations of the bundled
    RECON2.2 model, so every gene in the matrix maps to at least one
    reaction (the association step finds no orphan genes).
  - The matrix is CPM-normalized (Compass's expected input format) and
    written as a TSV in the same format as
    compass/Resources/Test-Data/tsv_format/expression.tsv.
  - Everything is seeded, so the committed file is reproducible:
      python scripts/generate_test_data.py --out tests/data/synthetic_expression.tsv

Expected downstream properties:
  - The number of cells is small (default 24), so a full
    --reaction-range-limited run finishes in a reasonable time.
  - With --selected-reactions-for-each-cell the run is near-instant.
"""
import argparse
import os

import numpy as np
import pandas as pd

# Baseline expression levels: log-uniform between ~0.1 and ~100 "units".
# On the order of magnitudes seen in CPM-normalized scRNA-seq.
MU_LOG_LO, MU_LOG_HI = -2.3, 4.6
# Cell size factors: log-normal-ish spread around 1.
S_LOG_SIGMA = 0.5
# Negative-binomial dispersion: smaller phi -> more over-dispersion.
PHI = 1.0


def sample_counts(n_genes, n_cells, genes, seed):
    """
    Draw a CPM-normalized expression matrix.

    Returns a DataFrame with genes as index and cells as columns.
    """
    rng = np.random.default_rng(seed)
    mu = 10.0 ** rng.uniform(MU_LOG_LO, MU_LOG_HI, n_genes)
    s = np.exp(rng.normal(0.0, S_LOG_SIGMA, n_cells))

    # Counts: NB via Gamma-Poisson mixture (mean mu_g*s_c, disp PHI)
    cell_means = np.outer(mu, s)  # (n_genes, n_cells)
    p = 1.0 / (1.0 + cell_means * PHI)
    k = cell_means / PHI
    counts = rng.negative_binomial(k, p)
    counts = counts.astype(float)
    assert counts.shape == (n_genes, n_cells)

    # Make ~30% of entries zero (dropout), preserving sparsity of scRNA-seq
    dropout_mask = rng.random(size=(n_genes, n_cells)) < 0.3
    counts[dropout_mask] = 0.0

    # CPM normalization (Compass convention: per-cell total 1e6)
    cpm = counts / counts.sum(axis=0) * 1e6
    cpm[~np.isfinite(cpm)] = 0.0

    cell_names = [f"SYNTH_CELL_{i:03d}" for i in range(n_cells)]
    return pd.DataFrame(cpm, index=list(genes), columns=cell_names)


def get_model_genes(model_name="RECON2.2"):
    """Collect the set of gene names associated with model reactions."""
    from compass.models import load_metabolic_model

    model = load_metabolic_model(model_name)
    genes = set()
    for r in model.reactions.values():
        if r.gene_associations is not None:
            genes.update(r.gene_associations.list_genes())
    genes.discard("")
    return sorted(genes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cells", type=int, default=24)
    parser.add_argument("--n-genes", type=int, default=600,
                        help="How many of the model's associated genes to include "
                             "(default 600; 0 = all)")
    parser.add_argument("--model", default="RECON2.2",
                        help="Model to draw gene names from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output TSV path")
    args = parser.parse_args()

    all_genes = get_model_genes(args.model)
    rng = np.random.default_rng(args.seed + 1)  # different stream from counts
    if 0 < args.n_genes < len(all_genes):
        idx = rng.choice(len(all_genes), args.n_genes, replace=False)
        genes = [all_genes[i] for i in sorted(idx)]
    else:
        genes = all_genes

    df = sample_counts(len(genes), args.n_cells, genes, seed=args.seed)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out, sep="\t")

    print(f"Wrote {args.out}: {df.shape[0]} genes x {df.shape[1]} cells "
          f"(seed={args.seed}, {args.n_cells} cells requested)")
    print(f"Non-zero entries: {(df > 0).sum().sum()} / {df.size} "
          f"({100.0 * (df > 0).sum().sum() / df.size:.1f}%)")


if __name__ == "__main__":
    main()
