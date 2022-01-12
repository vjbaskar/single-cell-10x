#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--qc-summary", required=True)
    p.add_argument("--min-genes", type=int, default=200)
    p.add_argument("--min-cells", type=int, default=3)
    p.add_argument("--max-mito-pct", type=float, default=20.0)
    return p.parse_args()


def main():
    args = parse_args()
    adata = sc.read_h5ad(args.input)
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    n_obs_before = adata.n_obs
    n_vars_before = adata.n_vars
    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    adata = adata[adata.obs["pct_counts_mt"] <= args.max_mito_pct].copy()

    summary = pd.DataFrame([{
        "cells_before": n_obs_before,
        "genes_before": n_vars_before,
        "cells_after": adata.n_obs,
        "genes_after": adata.n_vars,
        "min_genes": args.min_genes,
        "min_cells": args.min_cells,
        "max_mito_pct": args.max_mito_pct,
    }])
    Path(args.qc_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.qc_summary, index=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output)


if __name__ == "__main__":
    main()
