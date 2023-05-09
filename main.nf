#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.input_h5ad = params.input_h5ad ?: "data/input.h5ad"
params.outdir = params.outdir ?: "results"
params.min_genes = params.min_genes ?: 200
params.min_cells = params.min_cells ?: 3
params.max_mito_pct = params.max_mito_pct ?: 20.0
params.expected_doublet_rate = params.expected_doublet_rate ?: 0.06
params.batch_key = params.batch_key ?: "batch"
params.n_latent = params.n_latent ?: 30
params.max_epochs = params.max_epochs ?: 200
params.n_neighbors = params.n_neighbors ?: 15
params.leiden_resolution = params.leiden_resolution ?: 0.5

process PREPROCESS_QC {
    tag "preprocess_qc"
    publishDir "${params.outdir}/01_preprocess", mode: "copy"

    input:
    path input_h5ad

    output:
    path "preprocessed.h5ad", emit: preprocessed_h5ad
    path "qc_summary.csv", emit: qc_summary

    script:
    """
    python ${projectDir}/bin/01_preprocess_qc.py \
      --input ${input_h5ad} \
      --output preprocessed.h5ad \
      --qc-summary qc_summary.csv \
      --min-genes ${params.min_genes} \
      --min-cells ${params.min_cells} \
      --max-mito-pct ${params.max_mito_pct}
    """
}

process DOUBLET_SCRUBLET {
    tag "doublet_scrublet"
    publishDir "${params.outdir}/02_doublet", mode: "copy"

    input:
    path preprocessed_h5ad

    output:
    path "singlets.h5ad", emit: singlets_h5ad
    path "scrublet_annotated.h5ad", emit: scrublet_annotated_h5ad

    script:
    """
    python ${projectDir}/bin/02_scrublet.py \
      --input ${preprocessed_h5ad} \
      --output singlets.h5ad \
      --output-annotated scrublet_annotated.h5ad \
      --expected-doublet-rate ${params.expected_doublet_rate}
    """
}

process INTEGRATE_SCVI {
    tag "integrate_scvi"
    publishDir "${params.outdir}/03_integration", mode: "copy"

    input:
    path singlets_h5ad

    output:
    path "integrated_scvi.h5ad", emit: integrated_h5ad
    path "scvi_model", emit: scvi_model_dir

    script:
    """
    python ${projectDir}/bin/03_integrate_scvi.py \
      --input ${singlets_h5ad} \
      --output integrated_scvi.h5ad \
      --model-dir scvi_model \
      --batch-key ${params.batch_key} \
      --n-latent ${params.n_latent} \
      --max-epochs ${params.max_epochs}
    """
}

process UMAP_VIS {
    tag "umap_vis"
    publishDir "${params.outdir}/04_umap", mode: "copy"

    input:
    path integrated_h5ad

    output:
    path "umap.h5ad", emit: umap_h5ad
    path "umap.png", emit: umap_png

    script:
    """
    python ${projectDir}/bin/04_umap.py \
      --input ${integrated_h5ad} \
      --output umap.h5ad \
      --umap-png umap.png \
      --batch-key ${params.batch_key} \
      --n-neighbors ${params.n_neighbors} \
      --leiden-resolution ${params.leiden_resolution}
    """
}

workflow {
    ch_input = Channel.fromPath(params.input_h5ad, checkIfExists: true)
    preprocessed = PREPROCESS_QC(ch_input)
    singlets = DOUBLET_SCRUBLET(preprocessed.preprocessed_h5ad)
    integrated = INTEGRATE_SCVI(singlets.singlets_h5ad)
    UMAP_VIS(integrated.integrated_h5ad)
}
