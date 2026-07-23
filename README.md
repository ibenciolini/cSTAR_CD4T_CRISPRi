# cSTAR_CD4T_CRISPRi

Reconstruction of condition-specific gene regulatory networks in primary human CD4+ T cells using cSTAR (Cell State Transition Assessment and Regulation) applied to a genome-scale CRISPRi Perturb-seq dataset.

## Overview

This repository contains the analysis pipeline for a project applying cSTAR to the GWCD4i dataset (Zhu et al. 2025), which profiles primary human CD4+ T cells from four donors at rest and at two time points after stimulation (Stim8hr, Stim48hr), using genome-scale CRISPRi.

The pipeline computes a dynamic phenotype descriptor (DPD) score for every perturbed gene, selects a core set of 200 genes per condition, and reconstructs the underlying gene regulatory network at two resolutions:

- Pseudobulk, using cstarpy's MRA, RP and FPTU functions on donor-aggregated data.
- Single-cell, using the same core genes, with ROLS and FPTU run on Sonic (UCD HPC).

Results are cross-validated against KEGG, STRING and TRRUST, and compared between the pseudobulk and single-cell networks.

## Repository structure

```
cSTAR_CD4T_CRISPRi/
├── README.md
├── data/
│ └── GETDATA.md # how to obtain the required data files
└── scripts/
├── 0_DataPrep.ipynb # NTC cell extraction (one-time, per donor/condition)
├── 1_DPDScoring.ipynb # DPD scoring, pseudobulk
├── 2_CoreGenes.ipynb # core gene selection, R matrix construction
├── 3_NetworkPseudobulk.ipynb # pseudobulk network inference (MRA + RP + FPTU)
├── 4_NetworkSingleCell.ipynb # single-cell edge calling and external validation
├── 5_Clustering.ipynb # hierarchical clustering and GO enrichment
└── sonic/
    ├── 1_prep_stv.py # STV/DPD prep for single-cell
    ├── 2_rols.py # single-cell network inference (ROLS)
    ├── 3_fptu.py # single-cell FPTU extension
    ├── submit_1_prep_stv.sh # SLURM submission scripts
    ├── submit_2_rols.sh
    └── submit_3_fptu.sh
```

## Data

The dataset is not included in this repository due to its size. See `data/GETDATA.md` for instructions on how to obtain it.

## Running the pipeline

The notebooks in `scripts/` are numbered in the order they are meant to be run. Each notebook must be run once per condition (`Rest`, `Stim8hr`, `Stim48hr`), set at the top of the notebook.

1. `0_DataPrep.ipynb`: run once per donor and condition to extract non-targeting control (NTC) cells from the raw per-donor files.
2. `1_DPDScoring.ipynb`: computes the stimulation vector and DPD scores for every perturbed gene.
3. `2_CoreGenes.ipynb`: selects the 200 core genes per condition and builds the square response matrix `R` used for network inference.
4. `3_NetworkPseudobulk.ipynb`: pseudobulk network inference.
5. Single-cell network inference runs on Sonic (UCD HPC), not locally: submit `scripts/sonic/submit_1_prep_stv.sh`, then `submit_2_rols.sh`, then `submit_3_fptu.sh`, one condition at a time. Outputs are then transferred back for use in Notebooks 4 and 5.
6. `4_NetworkSingleCell.ipynb`: single-cell edge calling, KEGG/STRING/TRRUST validation, and comparison against the pseudobulk network.
7. `5_Clustering.ipynb`: hierarchical clustering and GO enrichment.

## Requirements

- Python 3.12
- `cstarpy` (network inference)
- Standard scientific Python stack: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `networkx`
- `scanpy`, `anndata` for single-cell data handling
- `gprofiler-official` for GO enrichment
- `jax` (single-cell network inference on Sonic)

## Author

Ilaria Benciolini, Systems Biology Ireland, University College Dublin.
Supervised by Dr. Oleksii Rukhlenko and Dr. Vadim Zhernovkov.
