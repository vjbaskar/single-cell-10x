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
## env
This uses the conda env `environment.yml` file. It is a simple conda environment for standard workflow and uses a custom set of functions from [scanpy_plus](https://github.com/haniffalab/scanpy_plus). 

Note that this uses `sc.pp.scrublet` for doublet detection as default and therefore not compatible with older versions of `scanpy (<1.10)`. Prior to this it was housed in `sc.ext.pp.scrublet`

```bash
conda env create -n environment.yml 
```

## help

```bash
nextflow run main.nf --help
```
## notes
- uses scanpy - install from environment.yml
- uses scvi-tools
- uses scanpy_plus
