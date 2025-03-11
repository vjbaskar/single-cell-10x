#!/usr/bin/env python3
"""Preprocess cells by simple QC filters."""

from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
import typer

app = typer.Typer()

@app.command()
def main(
    input: str = typer.Option(..., "--input", "-i"),
    output: str = typer.Option(..., "--output", "-o"),
    qc_summary: str = typer.Option(..., "--qc-summary", "-q"),
    min_genes: int = typer.Option(200),
    min_cells: int = typer.Option(3),
    max_mito_pct: float = typer.Option(20.0),
):
    """Run preprocessing on one AnnData."""
    meta = pd.read_csv(input)
    if "sample_name" not in meta.columns or "starsolo" not in meta.columns:
        raise ValueError("metadata needs sample_name and starsolo")

    adata_list = []
    rows = []
    obs_cols = [c for c in meta.columns if c != "starsolo"]
    for rec in meta.itertuples(index=False):
        sample_name = getattr(rec, "sample_name")
        starsolo_path = getattr(rec, "starsolo")
        adata = sc.read_10x_mtx(starsolo_path, var_names="gene_symbols", make_unique=True, cache=False)
        adata.obs_names = [sample_name + "_" + bc for bc in adata.obs_names]
        for col in obs_cols:
            adata.obs[col] = getattr(rec, col)
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
        n_obs_before = adata.n_obs
        n_vars_before = adata.n_vars
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        adata = adata[adata.obs["pct_counts_mt"] <= max_mito_pct].copy()
        rows.append({
            "sample_name": sample_name,
            "cells_before": n_obs_before,
            "genes_before": n_vars_before,
            "cells_after": adata.n_obs,
            "genes_after": adata.n_vars,
            "min_genes": min_genes,
            "min_cells": min_cells,
            "max_mito_pct": max_mito_pct,
        })
        adata_list.append(adata)

    if len(adata_list) == 0:
        raise ValueError("no samples in metadata")

    adata = ad.concat(
        adata_list,
        join="outer",
        merge="same",
        label="batch",
        keys=[str(x) for x in meta["sample_name"].tolist()],
        fill_value=0,
    )
    summary = pd.DataFrame(rows).sort_values("sample_name")
    Path(qc_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(qc_summary, index=False)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output)


if __name__ == "__main__":
    app()
