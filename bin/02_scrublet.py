#!/usr/bin/env python3
"""Run scrublet on filtered data."""

from pathlib import Path

import numpy as np
import scanpy as sc
import scrublet as scr
import typer

app = typer.Typer()

@app.command()
def main(
    input: str = typer.Option(..., "--input", "-i"),
    output: str = typer.Option(..., "--output", "-o"),
    output_annotated: str = typer.Option(..., "--output-annotated", "-a"),
    expected_doublet_rate: float = typer.Option(0.06),
):
    """Score and remove predicted doublets."""
    adata = sc.read_h5ad(input)
    counts_matrix = adata.X
    if not isinstance(counts_matrix, np.ndarray):
        counts_matrix = counts_matrix.tocsr()

    scrub = scr.Scrublet(counts_matrix, expected_doublet_rate=expected_doublet_rate)
    doublet_scores, predicted_doublets = scrub.scrub_doublets()
    adata.obs["doublet_score"] = doublet_scores
    adata.obs["predicted_doublet"] = predicted_doublets.astype(bool)
    adata.obs["predicted_doublet"] = adata.obs["predicted_doublet"].astype("category")

    Path(output_annotated).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_annotated)
    singlets = adata[~adata.obs["predicted_doublet"]].copy()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    singlets.write_h5ad(output)


if __name__ == "__main__":
    app()
