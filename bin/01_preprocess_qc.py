#!/usr/bin/env python3
"""Preprocess cells by simple QC filters."""

from pathlib import Path

import pandas as pd
import scanpy as sc
import typer

app = typer.Typer()

@app.command()
def main(
    input: str = typer.Option(...),
    output: str = typer.Option(...),
    qc_summary: str = typer.Option(...),
    min_genes: int = typer.Option(200),
    min_cells: int = typer.Option(3),
    max_mito_pct: float = typer.Option(20.0),
):
    """Run preprocessing on one AnnData."""
    adata = sc.read_h5ad(input)
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    n_obs_before = adata.n_obs
    n_vars_before = adata.n_vars
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata = adata[adata.obs["pct_counts_mt"] <= max_mito_pct].copy()

    summary = pd.DataFrame([{
        "cells_before": n_obs_before,
        "genes_before": n_vars_before,
        "cells_after": adata.n_obs,
        "genes_after": adata.n_vars,
        "min_genes": min_genes,
        "min_cells": min_cells,
        "max_mito_pct": max_mito_pct,
    }])
    Path(qc_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(qc_summary, index=False)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output)


if __name__ == "__main__":
    app()
