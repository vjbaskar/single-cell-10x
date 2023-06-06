#!/usr/bin/env python3
"""Generate UMAP from integrated data."""

import argparse
from pathlib import Path

import scanpy as sc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--umap-png", required=True)
    p.add_argument("--batch-key", default="batch")
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--leiden-resolution", type=float, default=0.5)
    return p.parse_args()


def main():
    """Compute neighbors, leiden and UMAP."""
    args = parse_args()
    adata = sc.read_h5ad(args.input)

    if "X_scvi" in adata.obsm:
        sc.pp.neighbors(adata, use_rep="X_scvi", n_neighbors=args.n_neighbors)
    else:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata, n_neighbors=args.n_neighbors)

    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=args.leiden_resolution)

    colors = ["leiden"]
    if args.batch_key in adata.obs.columns:
        colors.append(args.batch_key)
    fig = sc.pl.umap(adata, color=colors, show=False, return_fig=True)
    Path(args.umap_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.umap_png, dpi=200, bbox_inches="tight")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output)


if __name__ == "__main__":
    main()
