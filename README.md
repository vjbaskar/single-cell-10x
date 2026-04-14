# nf-scanpy

simple nextflow + scanpy workflow

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

- uses scvi-tools
