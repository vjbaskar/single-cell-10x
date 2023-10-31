#!/usr/bin/env python3
"""Integrate with SCVI."""

from pathlib import Path

import scanpy as sc
import scvi
import typer

app = typer.Typer()

@app.command()
def main(
    input: str = typer.Option(...),
    output: str = typer.Option(...),
    model_dir: str = typer.Option(...),
    batch_key: str = typer.Option("batch"),
    n_latent: int = typer.Option(30),
    max_epochs: int = typer.Option(200),
):
    """Fit SCVI and save latent embedding."""
    adata = sc.read_h5ad(input)
    if batch_key not in adata.obs.columns:
        adata.obs[batch_key] = "batch_1"

    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
    model = scvi.model.SCVI(adata, n_latent=n_latent)
    model.train(max_epochs=max_epochs)
    adata.obsm["X_scvi"] = model.get_latent_representation()

    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    model.save(model_dir_path, overwrite=True)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output)


if __name__ == "__main__":
    app()
