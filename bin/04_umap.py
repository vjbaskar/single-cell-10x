#!/usr/bin/env python3
"""Generate UMAP from integrated data."""

from pathlib import Path

import scanpy as sc
import typer

app = typer.Typer()

@app.command()
def main(
    input: str = typer.Option(..., "--input", "-i"),
    output: str = typer.Option(..., "--output", "-o"),
    umap_png: str = typer.Option(..., "--umap-png", "-p"),
    batch_key: str = typer.Option("batch"),
    n_neighbors: int = typer.Option(15),
    leiden_resolution: float = typer.Option(0.5),
):
    """Compute neighbors, leiden and UMAP."""
    adata = sc.read_h5ad(input)

    if "X_scvi" in adata.obsm:
        sc.pp.neighbors(adata, use_rep="X_scvi", n_neighbors=n_neighbors)
    else:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)

    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=leiden_resolution)

    colors = ["leiden"]
    if batch_key in adata.obs.columns:
        colors.append(batch_key)
    fig = sc.pl.umap(adata, color=colors, show=False, return_fig=True)
    Path(umap_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(umap_png, dpi=200, bbox_inches="tight")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output)


if __name__ == "__main__":
    app()
