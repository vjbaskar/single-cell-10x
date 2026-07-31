# single-cell-10x pipeline help

## Usage

`nextflow run main.nf --metadata_csv <metadata.csv> --outdir <output_dir>`

## Options

- `--metadata_csv <metadata.csv>`: Path to the metadata CSV file
- `--outdir <output_dir>`: Path to the output directory
- `--min_genes <min_genes>`: Minimum number of genes per cell
- `--min_cells <min_cells>`: Minimum number of cells per gene
- `--max_mito_pct <max_mito_pct>`: Maximum percentage of mitochondrial genes
- `--expected_doublet_rate <expected_doublet_rate>`: Expected doublet rate
- `--batch_key <batch_key>`: Key for the batch variable
- `--n_latent <n_latent>`: Number of latent dimensions
- `--max_epochs <max_epochs>`: Maximum number of epochs
- `--n_neighbors <n_neighbors>`: Number of neighbors
- `--leiden_resolution <leiden_resolution>`: Leiden resolution
- `--help`: Show this help message
