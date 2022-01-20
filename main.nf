#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.input_h5ad = params.input_h5ad ?: "data/input.h5ad"
params.outdir = params.outdir ?: "results"
params.min_genes = params.min_genes ?: 200
params.min_cells = params.min_cells ?: 3
params.max_mito_pct = params.max_mito_pct ?: 20.0

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

workflow {
    ch_input = Channel.fromPath(params.input_h5ad, checkIfExists: true)
    PREPROCESS_QC(ch_input)
}
