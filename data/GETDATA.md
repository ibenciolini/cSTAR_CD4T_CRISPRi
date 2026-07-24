# Obtaining the data

This project uses the GWCD4i dataset (Zhu et al. 2025, bioRxiv) and three external reference databases for network validation. None of these files are included in this repository due to their size (the per-donor single-cell files are 130-160 GB each).

## GWCD4i Perturb-seq dataset

Zhu, R., Dann, E., Yan, J., Reyes Retana, J., Goto, R., Guitche, R.C., Petersen, L.K., Ota, M., Pritchard, J.K., Marson, A. (2025). Genome-scale perturb-seq in primary human CD4+ T cells maps context-specific regulators of T cell programs and human immune traits. bioRxiv. https://doi.org/10.64898/2025.12.23.696273

Dataset page (Chan Zuckerberg Initiative Virtual Cells Platform): https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq

Files needed for this pipeline:

- `D{1,2,3,4}_{Rest,Stim8hr,Stim48hr}.assigned_guide.h5ad`: one file per donor and condition, raw single-cell counts with guide assignment. Used by `0_DataPrep.ipynb`, `2_CoreGenes.ipynb` (UMAP), and the Sonic scripts.
- `GWCD4i.DE_stats.h5ad`: genome-wide pseudobulk differential expression results. Used by `1_DPDScoring.ipynb`, `2_CoreGenes.ipynb`, and NB4's KEGG comparison.

How to access:

- VCP CLI
  
  ```bash
  pip install vcp-cli
  vcp data search "Primary Human CD4+ T Cell Perturb-seq" --exact
  ```
  
  Then follow the CLI prompts to download the specific files listed above. Full usage guide: https://chanzuckerberg.github.io/vcp-cli/usage/data.html

Place all downloaded files in `data/`.

## External validation databases

Used in `4_NetworkSingleCell.ipynb` for edge validation against known interactions.

### KEGG, T cell receptor signalling pathway (hsa04660)

```bash
curl -o hsa04660.xml https://rest.kegg.jp/get/hsa04660/kgml
curl -o hsa04310.xml https://rest.kegg.jp/get/hsa04310/kgml
curl -o hsa04120.xml https://rest.kegg.jp/get/hsa04120/kgml
```

Pathway page: https://www.kegg.jp/entry/hsa04660

### STRING v12.0

Download page: https://string-db.org/cgi/download?species_text=Homo+sapiens

Files needed (species 9606, Homo sapiens):

- `9606.protein.links.v12.0.txt.gz`
- `9606.protein.info.v12.0.txt.gz`

### TRRUST v2

```bash
curl -o trrust_rawdata.human.tsv https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv
```

Database page: https://www.grnpedia.org/trrust/

Note: if the direct download fails, the same data can be retrieved through OmnipathR:

```r
library(OmnipathR)
trrust <- import_tf_target_interactions(resources = "TRRUST", organism = 9606)
```

Place all three files in `data/`.
