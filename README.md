# nf-scanpy

simple nextflow pipeline for scanpy workflow for single cell analysis

## steps

1. preprocess qc
2. doublet detection
3. scvi integration
4. umap

## run

```bash
nextflow run main.nf --input_h5ad data/input.h5ad --outdir results
```

## notes
- uses scanpy - install from environment.yml
- uses scvi-tools
- uses scanpy_plus
