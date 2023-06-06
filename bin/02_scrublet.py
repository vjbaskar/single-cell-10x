#!/usr/bin/env python3
"""Run scrublet on filtered data."""

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
import scrublet as scr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--output-annotated", required=True)
    p.add_argument("--expected-doublet-rate", type=float, default=0.06)
    return p.parse_args()


def main():
    """Score and remove predicted doublets."""
    args = parse_args()
    adata = sc.read_h5ad(args.input)
    counts_matrix = adata.X
    if not isinstance(counts_matrix, np.ndarray):
        counts_matrix = counts_matrix.tocsr()

    scrub = scr.Scrublet(counts_matrix, expected_doublet_rate=args.expected_doublet_rate)
    doublet_scores, predicted_doublets = scrub.scrub_doublets()
    adata.obs["doublet_score"] = doublet_scores
    adata.obs["predicted_doublet"] = predicted_doublets.astype(bool)

    Path(args.output_annotated).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output_annotated)
    singlets = adata[~adata.obs["predicted_doublet"]].copy()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    singlets.write_h5ad(args.output)


if __name__ == "__main__":
    main()
