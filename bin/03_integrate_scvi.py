#!/usr/bin/env python3
"""Integrate with SCVI."""

import argparse
from pathlib import Path

import scanpy as sc
import scvi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--batch-key", default="batch")
    p.add_argument("--n-latent", type=int, default=30)
    p.add_argument("--max-epochs", type=int, default=200)
    return p.parse_args()


def main():
    """Fit SCVI and save latent embedding."""
    args = parse_args()
    adata = sc.read_h5ad(args.input)
    if args.batch_key not in adata.obs.columns:
        adata.obs[args.batch_key] = "batch_1"

    scvi.model.SCVI.setup_anndata(adata, batch_key=args.batch_key)
    model = scvi.model.SCVI(adata, n_latent=args.n_latent)
    model.train(max_epochs=args.max_epochs)
    adata.obsm["X_scvi"] = model.get_latent_representation()

    model_dir_path = Path(args.model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    model.save(model_dir_path, overwrite=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output)


if __name__ == "__main__":
    main()
